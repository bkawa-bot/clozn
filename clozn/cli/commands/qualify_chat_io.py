"""Live, exact-model qualification for the native structured-chat path."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from clozn._io import atomic_write_json
from clozn.artifacts.contracts import (
    CHAT_IO_ARTIFACT_TYPE, CHAT_IO_ARTIFACT_VERSION, CHAT_IO_EVIDENCE_SCHEMA,
    CHAT_IO_JSON_SCHEMA_SUBSET_ID, CHAT_IO_PIPELINE, CHAT_IO_QUALIFICATION_SUITE_ID,
    CONTRACT_VERSION, gguf_identity, validate_chat_io_profile,
)
from clozn.cli.commands.models import _flags_for, resolve_model
from clozn.cli.engine_process import _free_port, spawn_engine
from clozn.runs.identity import template_fingerprint
from clozn.server.structured_io import (
    NATIVE_WORKER_PIPELINE, QUALIFICATIONS_ENV, QUALIFICATION_SCHEMA, normalize_contract,
    validate_native_message, validate_qualification_registry,
)

_TOOL = {"type": "function", "function": {
    "name": "lookup_weather", "description": "Look up weather for one city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"], "additionalProperties": False},
}}
_STRICT_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "temperature_c": {"type": "integer"}},
    "required": ["city", "temperature_c"], "additionalProperties": False,
}


def _engine_client_class():
    from clozn.cli.commands.quant_check import _import_engine_client
    return _import_engine_client()


def _worker_pipeline(health: Mapping[str, Any]) -> dict[str, str]:
    native = health.get("native_chat_io")
    if not isinstance(native, Mapping):
        raise ValueError("worker health did not report native_chat_io")
    for flag in ("available", "atomic", "buffered"):
        if native.get(flag) is not True:
            raise ValueError(f"worker native_chat_io.{flag} is not true")
    actual = {name: native.get(name) for name in NATIVE_WORKER_PIPELINE}
    if actual != NATIVE_WORKER_PIPELINE:
        raise ValueError(f"worker native pipeline mismatch: expected {NATIVE_WORKER_PIPELINE!r}, got {actual!r}")
    return dict(NATIVE_WORKER_PIPELINE)


def _checked_message(response: Mapping[str, Any], *, model_sha256: str,
                     pipeline: Mapping[str, str], contract: Mapping[str, Any]) -> dict[str, Any]:
    chat_io = response.get("chat_io")
    if not isinstance(chat_io, Mapping):
        raise ValueError("native completion omitted chat_io")
    if chat_io.get("model_sha256") != model_sha256:
        raise ValueError("native completion model SHA-256 changed during qualification")
    if chat_io.get("pipeline") != pipeline:
        raise ValueError("native completion pipeline changed during qualification")
    if chat_io.get("parse_error") is not None:
        raise ValueError(f"native parser failed: {chat_io['parse_error']!r}")
    return validate_native_message(chat_io.get("message"), contract)


def run_qualification(client, *, model_sha256: str, template_sha256: str,
                      max_tokens: int = 128) -> tuple[dict[str, str], dict[str, int]]:
    """Run the fixed battery, raising on the first failure."""
    health = client.health()
    if health.get("model_sha256") != model_sha256:
        raise ValueError(f"attached worker model SHA-256 is {health.get('model_sha256')!r}, expected {model_sha256!r}")
    pipeline = _worker_pipeline(health)
    actual_template = template_fingerprint(client.apply_template)
    if actual_template != template_sha256:
        raise ValueError(f"worker template fingerprint is {actual_template!r}, expected {template_sha256!r}")

    tool_contract = normalize_contract({"tools": [_TOOL], "tool_choice": "auto",
                                        "parallel_tool_calls": False})
    first_messages = [
        {"role": "system", "content": "Use tools when the user requests live data."},
        {"role": "user", "content": "Call lookup_weather for Paris. Do not answer directly."},
    ]
    first = client.complete_chat(
        first_messages,
        tools=[_TOOL], tool_choice="auto", parallel_tool_calls=False,
        max_tokens=max_tokens, temperature=0.0, seed=1701)
    first_checked = _checked_message(first, model_sha256=model_sha256,
                                     pipeline=pipeline, contract=tool_contract)
    if first_checked.get("kind") != "tool_call" or first_checked.get("name") != "lookup_weather":
        raise ValueError("tool-call case did not produce lookup_weather")
    native_message = dict(first["chat_io"]["message"])
    native_message["tool_calls"] = [dict(native_message["tool_calls"][0])]
    call_id = native_message["tool_calls"][0].get("id") or "call_clozn_qualification"
    native_message["tool_calls"][0]["id"] = call_id
    continuation = client.complete_chat([
        *first_messages, native_message,
        {"role": "tool", "tool_call_id": call_id,
         "content": '{"city":"Paris","temperature_c":21}'},
    ], tools=[_TOOL], tool_choice="auto", parallel_tool_calls=False,
       max_tokens=max_tokens, temperature=0.0, seed=1702)
    continued = _checked_message(continuation, model_sha256=model_sha256,
                                 pipeline=pipeline, contract=tool_contract)
    if continued.get("kind") != "message":
        raise ValueError("tool-result continuation called a tool instead of answering")

    object_contract = normalize_contract({"response_format": {"type": "json_object"}})
    json_object = client.complete_chat(
        [{"role": "user", "content": "Return one JSON object with key ok and boolean value true."}],
        json_schema={"type": "object"}, max_tokens=max_tokens, temperature=0.0, seed=1703)
    object_checked = _checked_message(json_object, model_sha256=model_sha256,
                                      pipeline=pipeline, contract=object_contract)
    if object_checked.get("kind") != "json_object":
        raise ValueError("json_object case did not produce an object")

    schema_contract = normalize_contract({"response_format": {"type": "json_schema", "json_schema": {
        "name": "weather", "strict": True, "schema": _STRICT_SCHEMA}}})
    strict = client.complete_chat(
        [{"role": "user", "content": "Return Paris with temperature_c 21."}],
        json_schema=_STRICT_SCHEMA, max_tokens=max_tokens, temperature=0.0, seed=1704)
    strict_checked = _checked_message(strict, model_sha256=model_sha256,
                                      pipeline=pipeline, contract=schema_contract)
    if strict_checked.get("kind") != "json_schema":
        raise ValueError("strict json_schema case did not validate")
    return ({"tool_call": first_checked["kind"], "tool_result_continuation": continued["kind"],
             "json_object": object_checked["kind"], "json_schema": strict_checked["kind"]},
            {"pipeline": 1, "tools": 2, "json_object": 1, "json_schema": 1})


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def _prospective_registry(path: Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            registry = validate_qualification_registry(json.load(handle))
    else:
        registry = {"schema_version": QUALIFICATION_SCHEMA, "entries": []}
    key = (entry["model_sha256"], entry["template_fingerprint"])
    entries = [item for item in registry["entries"]
               if (item["model_sha256"], item["template_fingerprint"]) != key]
    entries.append(dict(entry))
    return validate_qualification_registry({"schema_version": QUALIFICATION_SCHEMA, "entries": entries})


def emit_qualification(*, identity: Mapping[str, Any], fingerprint: str,
                       cases: Mapping[str, Any], passed: Mapping[str, int], artifact_dir: Path,
                       registry_path: Path, source_id: str) -> dict[str, Any]:
    """Validate output in a sibling temp directory before atomically activating it."""
    if artifact_dir.exists():
        raise ValueError(f"qualification artifact already exists: {artifact_dir}")
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{artifact_dir.name}-", dir=artifact_dir.parent))
    try:
        features = ["tools", "json_object", "json_schema"]
        subsets = {"tool_parameters": CHAT_IO_JSON_SCHEMA_SUBSET_ID,
                   "json_schema": CHAT_IO_JSON_SCHEMA_SUBSET_ID}
        evidence = {
            "schema_version": CHAT_IO_EVIDENCE_SCHEMA, "suite_id": CHAT_IO_QUALIFICATION_SUITE_ID,
            "model_sha256": identity["sha256"], "template_fingerprint": fingerprint,
            "pipeline": dict(CHAT_IO_PIPELINE), "features": features, "schema_subsets": subsets,
            "results": {name: {"passed": count, "failed": 0} for name, count in passed.items()},
            "cases": dict(cases),
        }
        payload = _json_bytes(evidence)
        evidence_name = "qualification-evidence.json"
        (temp / evidence_name).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        manifest = {
            "contract_version": CONTRACT_VERSION, "artifact_type": CHAT_IO_ARTIFACT_TYPE,
            "artifact_version": CHAT_IO_ARTIFACT_VERSION,
            "model": {"source_id": source_id, "architecture": identity["architecture"],
                      "hidden_size": identity["hidden_size"], "layer_count": identity["layer_count"],
                      "vocab_size": identity["vocab_size"], "tokenizer_sha256": identity["tokenizer_sha256"],
                      "compatible_gguf_sha256": [identity["sha256"]]},
            "profile": {"template_fingerprint": fingerprint, "pipeline": dict(CHAT_IO_PIPELINE),
                        "features": features, "schema_subsets": subsets,
                        "evidence": {"path": evidence_name, "sha256": digest}},
            "files": {evidence_name: {"bytes": len(payload), "sha256": digest}},
        }
        atomic_write_json(str(temp / "manifest.json"), manifest, ensure_ascii=False, indent=2, sort_keys=True)
        checked = validate_chat_io_profile(manifest, identity, fingerprint, temp)
        registry = _prospective_registry(registry_path, checked["registry_entry"])
        os.replace(temp, artifact_dir)
        atomic_write_json(str(registry_path), registry, ensure_ascii=False, indent=2, sort_keys=True)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return {"artifact": str(artifact_dir), "registry": str(registry_path),
            "model_sha256": identity["sha256"], "template_fingerprint": fingerprint,
            "features": features, "cases": dict(cases)}


def add_subparser(sub):
    parser = sub.add_parser("qualify-chat-io",
                            help="live-qualify one exact GGUF/template for native structured output")
    parser.add_argument("gguf", help="exact local GGUF to qualify")
    parser.add_argument("--attach", action="store_true",
                        help="attach to an already-running private worker on --port")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--registry", default=None)
    parser.add_argument("--source-id", default=None)
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(fn=cmd_qualify_chat_io)
    return parser


def cmd_qualify_chat_io(args):
    from clozn.cli import main as ctx
    model = resolve_model(args.gguf)
    identity = gguf_identity(model)
    port = args.port or _free_port()
    if args.attach and not args.port:
        raise ctx.CloznError("--attach requires an explicit --port")
    if args.max_tokens < 1:
        raise ctx.CloznError("--max-tokens must be positive")
    proc = None
    try:
        if not args.attach:
            proc, _health, _gpu = spawn_engine(model, port, _flags_for(model), prefer_gpu=not args.cpu)
        client = _engine_client_class()(port=port, timeout=180.0)
        fingerprint = template_fingerprint(client.apply_template)
        if not fingerprint:
            raise ValueError("worker could not render the canonical chat template")
        cases, passed = run_qualification(client, model_sha256=identity["sha256"],
                                          template_sha256=fingerprint, max_tokens=args.max_tokens)
        artifact_root = Path(os.environ.get("CLOZN_ARTIFACTS_DIR") or Path(ctx.HOME) / "artifacts")
        artifact_dir = Path(args.artifact_dir) if args.artifact_dir else (
            artifact_root / "chat_io" / identity["sha256"] / fingerprint)
        registry_path = Path(args.registry or os.environ.get(QUALIFICATIONS_ENV)
                             or (Path(ctx.HOME) / "chat_io_qualifications.json"))
        report = emit_qualification(identity=identity, fingerprint=fingerprint, cases=cases,
                                    passed=passed, artifact_dir=artifact_dir,
                                    registry_path=registry_path,
                                    source_id=args.source_id or Path(model).stem)
    except ctx.CloznError:
        raise
    except Exception as exc:
        raise ctx.CloznError(f"chat-I/O qualification failed; nothing installed: {exc}") from None
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(f"qualified chat I/O for {report['model_sha256']}")
        print(f"artifact: {report['artifact']}")
        print(f"registry: {report['registry']}")
    return 0
