"""Portable exports for ``clozn.experiment.result.v0`` artifacts.

The module is dependency-free and network-free.  It produces local review artifacts; it never
publishes to Hugging Face or invokes an external evaluation runner.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from typing import Any

from clozn import __version__
from clozn.experiments import suite as experiment_suite

EEE_SCHEMA = "0.2.2"
EEE_INSTANCE_SCHEMA = "instance_level_eval_0.2.2"
RELATIONSHIPS = frozenset({"first_party", "third_party", "collaborative", "other"})
ADAPTERS = frozenset({"promptfoo", "inspect", "lighteval"})


class EvalExportError(ValueError):
    """An experiment cannot be represented without inventing required information."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_jsonl(records: list[dict]) -> str:
    return "".join(_canonical(record) + "\n" for record in records)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(value: Any) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip()).strip("-._")
    return slug[:100] or "evaluation"


def _created_epoch(result: dict, supplied: str | int | float | None) -> str:
    if supplied is not None:
        try:
            value = float(supplied)
        except (TypeError, ValueError):
            raise EvalExportError("retrieved_timestamp must be a Unix epoch number") from None
        if value < 0:
            raise EvalExportError("retrieved_timestamp must be non-negative")
        return str(int(value)) if value.is_integer() else str(value)
    raw = result.get("created_at")
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return str(int(parsed.timestamp()))
    except (TypeError, ValueError, OverflowError):
        raise EvalExportError("experiment created_at is not an ISO timestamp; pass retrieved_timestamp") from None


def _case_map(result: dict) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for suite_name in ("target", "guard"):
        for case in result["manifest"]["suites"][suite_name]["cases"]:
            out[(suite_name, case["name"])] = case
    return out


def _messages(case: dict) -> list[dict]:
    if isinstance(case.get("messages"), list):
        return [{"role": str(m["role"]), "content": str(m["content"])} for m in case["messages"]]
    return [{"role": "user", "content": str(case.get("prompt") or "")}]


def _prompt_text(case: dict) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in _messages(case))


def _references(case: dict) -> list[str]:
    expect = case.get("expect") or {}
    refs: list[str] = []
    for key in ("contains", "not_contains", "matches", "finish_reason"):
        if key not in expect:
            continue
        values = expect[key] if isinstance(expect[key], list) else [expect[key]]
        refs.extend(f"{key}: {value}" for value in values)
    return refs


def normalize_case_records(source: dict | list[dict]) -> list[dict]:
    """Return deterministic case/cell records used by all adapters.

    A raw list is accepted for programmatic adapter use, but an experiment artifact is validated
    through the same complete-matrix validator used by CI before any records are returned.
    """
    if isinstance(source, list):
        if any(not isinstance(row, dict) for row in source):
            raise EvalExportError("normalized case records must be JSON objects")
        return [dict(row) for row in source]
    try:
        result = experiment_suite.validate_result(source)
    except experiment_suite.ManifestError as exc:
        raise EvalExportError(str(exc)) from exc
    cases = _case_map(result)
    records = []
    for cell in sorted(result["cells"], key=lambda c: (c["variant"], c["suite"], c["case"], c["seed"])):
        case = cases[(cell["suite"], cell["case"])]
        scored = cell.get("status") in {"pass", "fail"}
        run = cell.get("run") if isinstance(cell.get("run"), dict) else {}
        records.append({
            "case_id": f"{cell['suite']}/{cell['case']}",
            "suite": cell["suite"], "case": cell["case"],
            "messages": _messages(case), "prompt": _prompt_text(case),
            "references": _references(case), "expect": case.get("expect") or {},
            "variant": cell["variant"], "variant_kind": cell.get("variant_kind"), "seed": cell["seed"],
            "status": cell.get("status"), "score": (1.0 if cell.get("status") == "pass" else 0.0) if scored else None,
            "response": cell.get("response"), "run_id": cell.get("run_id"),
            "model": run.get("model"), "identity": run.get("identity") or run.get("captured_identity"),
            "assertions": cell.get("assertions") or [], "receipts": cell.get("receipts"),
            "error": cell.get("error"), "run": run, "raw_cell": cell,
        })
    return records


def _model_for_variant(result: dict, variant: dict, records: list[dict], model_ids: dict[str, str]) -> tuple[str, str]:
    override = model_ids.get(variant["name"])
    observed = sorted({str(row["model"]) for row in records if row.get("model")})
    declared = variant.get("model") or result["manifest"].get("defaults", {}).get("model")
    if override:
        return override, override.rsplit("/", 1)[-1]
    if len(observed) > 1:
        raise EvalExportError(
            f"variant {variant['name']!r} captured multiple model identities; pass --model-id {variant['name']}=OWNER/REPO"
        )
    model_id = observed[0] if observed else declared
    if not isinstance(model_id, str) or not model_id.strip():
        raise EvalExportError(f"variant {variant['name']!r} has no captured model identity")
    return model_id.strip(), model_id.strip().rsplit("/", 1)[-1]


def _generation_args(result: dict, variant: dict) -> dict:
    source = {**result["manifest"].get("defaults", {}), **variant}
    allowed = ("temperature", "top_p", "top_k", "max_tokens")
    args = {key: source[key] for key in allowed if source.get(key) is not None}
    if variant.get("system_prompt"):
        args["prompt_template"] = variant["system_prompt"]
    return args


def _identity_details(records: list[dict]) -> dict[str, str]:
    details = {}
    for key in ("model_sha256", "model_size_bytes", "template_fingerprint", "engine_build", "clozn_version"):
        values = {str(row["identity"][key]) for row in records
                  if isinstance(row.get("identity"), dict) and row["identity"].get(key) is not None}
        if len(values) == 1:
            details[key] = values.pop()
    return details


def _source_data(result: dict, suite_name: str, hf_benchmark: dict | None, count: int) -> dict:
    if hf_benchmark:
        dataset_id = hf_benchmark.get("dataset_id")
        if not isinstance(dataset_id, str) or "/" not in dataset_id:
            raise EvalExportError("hf_benchmark.dataset_id must be an OWNER/DATASET identifier")
        return {"dataset_name": result["name"], "source_type": "hf_dataset",
                "hf_repo": dataset_id, "samples_number": count,
                "additional_details": {"clozn_suite": suite_name}}
    return {"dataset_name": result["name"], "source_type": "other",
            "additional_details": {"clozn_suite": suite_name}}


def _token_usage(run: dict) -> dict | None:
    usage = run.get("usage") if isinstance(run.get("usage"), dict) else {}
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if output_tokens is None:
        trace = run.get("trace")
        steps = trace.get("steps") if isinstance(trace, dict) else trace
        if isinstance(steps, list):
            output_tokens = len(steps)
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    return {"input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens}


def _instance(row: dict, evaluation_id: str, evaluation_result_id: str, model_id: str) -> dict:
    response = "" if row.get("response") is None else str(row["response"])
    record = {
        "schema_version": EEE_INSTANCE_SCHEMA,
        "evaluation_id": evaluation_id, "model_id": model_id,
        "evaluation_name": row["suite"], "evaluation_result_id": evaluation_result_id,
        "sample_id": f"{row['case']}@seed-{row['seed']}",
        "sample_hash": _sha256_text(_canonical({"input": row["prompt"], "reference": row["references"]})),
        "interaction_type": "single_turn",
        "input": {"raw": row["prompt"], "formatted": None, "reference": row["references"], "choices": None},
        "output": {"raw": [response], "reasoning_trace": None}, "messages": None,
        "answer_attribution": [{"turn_idx": 0, "source": "output.raw", "extracted_value": response,
                                "extraction_method": "custom", "is_terminal": True}],
        "evaluation": {"score": row["score"], "is_correct": row["score"] == 1.0},
        "error": None,
        "metadata": {"clozn_suite": row["suite"], "clozn_case": row["case"],
                     "clozn_variant": row["variant"], "clozn_seed": str(row["seed"]),
                     "clozn_run_id": str(row.get("run_id") or "not recorded"),
                     "clozn_assertions": _canonical(row.get("assertions") or [])},
    }
    usage = _token_usage(row.get("run") or {})
    if usage is not None:
        record["token_usage"] = usage
    duration = (row.get("run") or {}).get("duration_ms")
    if isinstance(duration, (int, float)) and duration >= 0:
        record["performance"] = {"latency_ms": duration}
    return record


def export_community_bundle(
    source: dict,
    *,
    source_organization: str,
    evaluator_relationship: str,
    model_ids: dict[str, str] | None = None,
    hf_benchmark: dict | None = None,
    retrieved_timestamp: str | int | float | None = None,
) -> dict:
    """Create EEE 0.2.2 aggregates/instances and optional HF Community Eval rows."""
    if not isinstance(source_organization, str) or not source_organization.strip():
        raise EvalExportError("source organization is required for EEE provenance")
    if evaluator_relationship not in RELATIONSHIPS:
        raise EvalExportError(f"evaluator relationship must be one of {sorted(RELATIONSHIPS)}")
    try:
        result = experiment_suite.validate_result(source)
    except experiment_suite.ManifestError as exc:
        raise EvalExportError(str(exc)) from exc
    timestamp = _created_epoch(result, retrieved_timestamp)
    all_rows = normalize_case_records(result)
    model_ids = model_ids or {}
    aggregates, instances, hf_exports, issues = [], {}, [], []
    for variant in result["manifest"]["variants"]:
        variant_name = variant["name"]
        rows = [row for row in all_rows if row["variant"] == variant_name]
        model_id, model_name = _model_for_variant(result, variant, rows, model_ids)
        evaluation_id = f"clozn/{result['experiment_id']}/{_slug(variant_name)}/{timestamp}"
        instance_rows: list[dict] = []
        metrics = []
        hf_rows = []
        for suite_name in ("target", "guard"):
            suite_rows = [row for row in rows if row["suite"] == suite_name]
            scored = [row for row in suite_rows if row["score"] is not None]
            excluded = len(suite_rows) - len(scored)
            if not scored:
                issues.append({"level": "warning", "code": "no_scored_cells", "variant": variant_name,
                               "suite": suite_name, "message": "EEE metric omitted: no pass/fail cells"})
                continue
            score = sum(row["score"] for row in scored) / len(scored)
            result_id = f"{result['experiment_id']}/{variant_name}/{suite_name}/pass_rate"
            metrics.append({
                "evaluation_result_id": result_id, "evaluation_name": suite_name,
                "source_data": _source_data(result, suite_name, hf_benchmark, len(scored)),
                "evaluation_timestamp": timestamp,
                "metric_config": {"evaluation_description": f"Clozn {suite_name} assertion pass rate",
                                  "metric_id": "clozn.assertion_pass_rate", "metric_name": "Assertion pass rate",
                                  "metric_kind": "pass_rate", "metric_unit": "proportion",
                                  "lower_is_better": False, "score_type": "continuous",
                                  "min_score": 0.0, "max_score": 1.0,
                                  "additional_details": {"excluded_unscored_or_error": str(excluded)}},
                "score_details": {"score": score, "details": {"passed": str(sum(r['score'] for r in scored)),
                                                                "scored": str(len(scored))}},
                "generation_config": {"generation_args": _generation_args(result, variant)},
            })
            instance_rows.extend(_instance(row, evaluation_id, result_id, model_id) for row in scored)
            if hf_benchmark:
                task_id = hf_benchmark.get(f"{suite_name}_task_id")
                if not isinstance(task_id, str) or not task_id.strip():
                    raise EvalExportError(f"hf_benchmark.{suite_name}_task_id is required")
                hf_rows.append({"dataset": {"id": hf_benchmark["dataset_id"], "task_id": task_id.strip()},
                                "value": score,
                                "date": datetime.fromtimestamp(float(timestamp), tz=timezone.utc).date().isoformat(),
                                "notes": f"Clozn {result['experiment_id']} {variant_name}; {len(scored)} scored cells"})
        instance_text = canonical_jsonl(instance_rows)
        filename = f"{_slug(result['experiment_id'])}-{_slug(variant_name)}.jsonl"
        aggregate = {
            "schema_version": EEE_SCHEMA, "evaluation_id": evaluation_id,
            "evaluation_timestamp": timestamp, "retrieved_timestamp": timestamp,
            "source_metadata": {"source_name": "Clozn", "source_type": "evaluation_run",
                                "source_organization_name": source_organization.strip(),
                                "evaluator_relationship": evaluator_relationship,
                                "additional_details": {"clozn_experiment_id": result["experiment_id"],
                                                       "manifest_sha256": result["manifest_sha256"]}},
            "model_info": {"name": model_name, "id": model_id,
                           "inference_platform": "local",
                           "inference_engine": {"name": "Clozn", "version": __version__},
                           "additional_details": _identity_details(rows)},
            "eval_library": {"name": "Clozn", "version": __version__},
            "evaluation_results": metrics,
            "detailed_evaluation_results": {"format": "jsonl", "file_path": filename,
                                            "hash_algorithm": "sha256", "checksum": _sha256_text(instance_text),
                                            "total_rows": len(instance_rows)},
        }
        aggregates.append({"variant": variant_name, "model_id": model_id,
                           "filename": filename.removesuffix(".jsonl") + ".json", "record": aggregate})
        instances[filename] = instance_rows
        if hf_rows:
            if "/" not in model_id:
                issues.append({"level": "error", "code": "hf_model_repo_required", "variant": variant_name,
                               "message": "HF Community Eval preview omitted: model id is not OWNER/REPO"})
            else:
                hf_exports.append({"variant": variant_name, "model_id": model_id,
                                   "filename": f".eval_results/clozn-{_slug(result['experiment_id'])}.yaml",
                                   "records": hf_rows})
    bundle = {
        "schema_version": "clozn.eval_export.v1", "format": "eee-community-bundle",
        "source": {"experiment_id": result["experiment_id"], "manifest_sha256": result["manifest_sha256"]},
        "aggregates": aggregates, "instances": instances, "hf_community": hf_exports,
        "issues": issues,
        "evidence": [{"coordinate": {key: cell.get(key) for key in ("suite", "case", "variant", "seed")},
                      "cell": cell} for cell in result["cells"]],
    }
    return validate_community_bundle(bundle)


def validate_community_bundle(bundle: dict) -> dict:
    """Validate Clozn's cross-file linkage and hashes without an optional JSON-schema dependency."""
    if not isinstance(bundle, dict) or bundle.get("schema_version") != "clozn.eval_export.v1":
        raise EvalExportError("community bundle schema_version is invalid")
    instances = bundle.get("instances")
    if not isinstance(instances, dict):
        raise EvalExportError("community bundle instances must be an object")
    for item in bundle.get("aggregates") or []:
        if not isinstance(item, dict) or not isinstance(item.get("record"), dict):
            raise EvalExportError("community bundle aggregate entry is invalid")
        aggregate = item["record"]
        if aggregate.get("schema_version") != EEE_SCHEMA:
            raise EvalExportError("EEE aggregate schema_version is invalid")
        detail = aggregate.get("detailed_evaluation_results") or {}
        filename = detail.get("file_path")
        rows = instances.get(filename)
        if not isinstance(rows, list):
            raise EvalExportError(f"EEE aggregate references missing instance file {filename!r}")
        text = canonical_jsonl(rows)
        if detail.get("checksum") != _sha256_text(text) or detail.get("total_rows") != len(rows):
            raise EvalExportError(f"EEE instance checksum/count mismatch for {filename!r}")
        if any(row.get("evaluation_id") != aggregate.get("evaluation_id") for row in rows):
            raise EvalExportError(f"EEE instance linkage mismatch for {filename!r}")
        result_ids = {row.get("evaluation_result_id") for row in aggregate.get("evaluation_results") or []}
        if any(row.get("evaluation_result_id") not in result_ids for row in rows):
            raise EvalExportError(f"EEE metric linkage mismatch for {filename!r}")
    return bundle


def _issue(code: str, message: str, **fields) -> dict:
    return {"level": "warning", "code": code, **fields, "message": message}


def _values(value) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def export_promptfoo(source: dict | list[dict], *, provider: str | None = None) -> dict:
    records = normalize_case_records(source)
    unique: dict[str, dict] = {}
    for row in records:
        unique.setdefault(row["case_id"], row)
    tests, issues = [], []
    for row in unique.values():
        assertions = []
        expect = row.get("expect") or {}
        for value in _values(expect.get("contains")):
            assertions.append({"type": "contains", "value": value})
        for value in _values(expect.get("not_contains")):
            assertions.append({"type": "not-contains", "value": value})
        if expect.get("matches") is not None:
            assertions.append({"type": "regex", "value": expect["matches"]})
        if expect.get("finish_reason") is not None:
            issues.append(_issue("unsupported_finish_reason", "Promptfoo assertion omitted; add runner-specific metadata",
                                 case_id=row["case_id"], field="finish_reason"))
        tests.append({"description": row["case_id"], "vars": {"prompt": row["prompt"]}, "assert": assertions,
                      "metadata": {"clozn_suite": row["suite"], "clozn_expect": row["expect"]}})
    config = {"description": "Exported from a Clozn experiment", "prompts": ["{{prompt}}"], "tests": tests}
    if provider:
        config["providers"] = [provider]
    else:
        issues.append(_issue("provider_required", "Set a Promptfoo provider before running this config", field="provider"))
    return {"adapter": "promptfoo", "format": "promptfoo-config-json.v1", "config": config, "issues": issues}


def export_inspect(source: dict | list[dict]) -> dict:
    records = normalize_case_records(source)
    unique: dict[str, dict] = {}
    for row in records:
        unique.setdefault(row["case_id"], row)
    samples, issues = [], []
    for row in unique.values():
        refs = row["references"]
        target = refs[0] if len(refs) == 1 and refs[0].startswith("contains: ") else None
        if target:
            target = target.removeprefix("contains: ")
        else:
            issues.append(_issue("custom_scorer_required", "Inspect sample retains Clozn criteria in metadata",
                                 case_id=row["case_id"], field="target"))
        sample = {"id": row["case_id"], "input": row["messages"],
                  "metadata": {"clozn_suite": row["suite"], "clozn_expect": row["expect"]}}
        if target is not None:
            sample["target"] = target
        samples.append(sample)
    return {"adapter": "inspect", "format": "inspect-samples-json.v1", "records": samples,
            "issues": issues, "runner_note": "Load these as inspect_ai.dataset.Sample objects and choose an explicit scorer."}


def export_lighteval(source: dict | list[dict], *, dataset_uri: str | None = None) -> dict:
    records = normalize_case_records(source)
    unique: dict[str, dict] = {}
    for row in records:
        unique.setdefault(row["case_id"], row)
    rows = [{"id": row["case_id"], "query": row["prompt"], "references": row["references"],
             "clozn_suite": row["suite"], "clozn_expect": row["expect"]} for row in unique.values()]
    issues = [_issue("prompt_function_required", "LightEval custom tasks require Python prompt/scoring functions; rows are data only")]
    if not dataset_uri:
        issues.append(_issue("dataset_uri_required", "Publish or point LightEval at the exported rows before running", field="dataset_uri"))
    task = {"name": "clozn_export", "suite": ["community"], "dataset_path": dataset_uri,
            "available_splits": ["test"], "evaluation_splits": ["test"]}
    return {"adapter": "lighteval", "format": "lighteval-dataset-json.v1", "records": rows,
            "task_spec": task, "issues": issues}


def export_adapter(source: dict | list[dict], target: str, **options) -> dict:
    if target == "promptfoo":
        return export_promptfoo(source, provider=options.get("provider"))
    if target == "inspect":
        return export_inspect(source)
    if target == "lighteval":
        return export_lighteval(source, dataset_uri=options.get("dataset_uri"))
    raise EvalExportError(f"unknown eval adapter {target!r}; choose one of {sorted(ADAPTERS)}")
