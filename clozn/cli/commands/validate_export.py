"""commands.validate_export -- `clozn validate-export` (Phase-1 §4.5, "Deployment-equivalence check v0",
PRODUCT_ROADMAP.md: "garbled-GGUF export bugs are a documented, recurring developer disaster"): did the
export pipeline (HF trainer checkpoint -> GGUF, or GGUF -> re-quantized/re-packaged GGUF) preserve the
things that decide whether the deployed model behaves like the one that was actually trained?

THE DISASTER THIS GATES: a fine-tuner exports a checkpoint to GGUF (llama.cpp convert script, a merge
tool, a third-party quantizer) and the export silently gets the chat template wrong, drops/relabels
BOS or EOS, or drifts the tokenizer vocabulary relative to what the model was actually trained/evaluated
against. None of this shows up as a crash -- the GGUF loads and generates fluent-looking text, just
wrong text, and the deployed behavior quietly diverges from what was validated. `clozn diff-model` (see
that module's docstring) already answers "did the WEIGHTS change" for a base-vs-fine-tune pair sharing one
file format; this module answers the orthogonal question -- "did the EXPORT preserve identity" -- for a
model that has been carried across the trainer-to-runtime boundary.

v0 SCOPE (this module): GGUF-side only. Both files being compared are already GGUF, read via
`clozn.artifacts.contracts.gguf_identity` (tokenizer_sha256, chat_template_sha256, vocab_size,
architecture/hidden_size/layer_count, whole-file sha256 -- all already-shipped, exact, model-agnostic
identity fields; this module's job is almost entirely COMPARING two calls to that one function, plus a
few raw-metadata reads `gguf_identity` doesn't itself expose). A true HF-checkpoint-vs-GGUF comparison
(reading the ORIGINAL trainer output's tokenizer_config.json / chat_template.jinja / config.json and
comparing THOSE against the exported GGUF) needs a Torch/transformers-capable environment to load the HF
side -- out of scope for this stdlib-only CLI; that is the v1 direction (an `--hf-checkpoint <dir>` mode
that shells out to a small torch-side reader, or accepts a pre-extracted JSON sidecar of the HF tokenizer
config so this module itself stays torch-free).

TWO MODES, one subcommand, disambiguated by argument count:

  1. TWO-FILE: `clozn validate-export <expected.gguf> <exported.gguf>` -- a static metadata equivalence
     report, NO engine boot needed. Compares, field by field: tokenizer_sha256, chat_template_sha256,
     vocab_size, architecture, hidden_size, layer_count (`compare_identities`). Quantization is compared
     too but purely informational -- a quant difference is an EXPECTED, common reason to run this exact
     command (e.g. validating a re-quantized export against its fp16 source) and never fails the gate on
     its own. Exit 0 if every non-quantization field matches; exit 1 if any of them mismatches. Optional
     `--known-answers N`: also boots both GGUFs (`run_known_answers_check`, mirrors `clozn diff-model`'s
     LIVE two-engine path exactly) and runs `diff_model.run_diff_model`'s reference-anchored ladder as a
     behavioral spot check -- reported alongside the static report, but per this task's own framing
     ("optional... spot check, reporting the verdict") it is INFORMATIONAL ONLY and does not itself change
     the exit code; the static metadata comparison is the one thing this gate actually enforces. Stated
     explicitly in the text report, not left implicit.

  2. SINGLE-FILE: `clozn validate-export <exported.gguf>` -- a sanity report on one export in isolation
     (no reference to compare against): does it carry a chat template at all, does that template contain
     any of the family's expected role-marker strings (heuristic; see TEMPLATE_MARKERS below), are BOS/EOS
     token ids present in metadata, is vocab_size a sane positive integer -- plus the file hash + full
     identity block, printed for the record regardless of whether anything is wrong. Every failed check is
     a WARNING, never silent: default exit is 0 (warnings are reported, not gated); `--strict` makes any
     warning exit 1. The measured caveat this mode exists to catch (see `clozn diff_model.py`'s own
     module docstring, "A BASE model driven through a chat template often emits degenerate text"): an
     export that lost its chat_template metadata will still load and generate, just badly, the moment
     something drives it through chat scaffolding -- this is the cheapest possible gate against exactly
     that, checkable before ever booting an engine.

TEMPLATE_MARKERS is a small heuristic table of known chat-format role-marker substrings, keyed by the same
"tmpl" vocabulary `clozn.cli.commands.models.KNOWN` already uses to pick engine launch flags (llama3,
mistral, gemma) plus "chatml" for the common case of a recognized chat model with no explicit "tmpl"
override (Qwen and friends -- ChatML's <|im_start|>/<|im_end|> by construction of those templates).
`_expected_tmpl_key` resolves a GGUF's expected family the exact same way `_flags_for` already does (by
filename fragment) -- no new per-model table invented. An export whose filename doesn't match any KNOWN
fragment falls back to a GENERIC check across every family's markers (a weaker, explicitly labeled signal:
"could this template plausibly be A chat template at all", not "is it the RIGHT one").

HONESTY (this task's own requirement, restated in the printed footer of every report,
`_HONESTY_FOOTER`): this is a STATIC metadata gate, plus (two-file, opt-in) a per-token BEHAVIORAL SPOT
CHECK over a small sample. Neither proves semantic equivalence between two checkpoints -- only that these
specific, cheap-to-check signals matched (or didn't) on these exact files. Unavailable metadata is never
silently treated as a pass: `compare_field` reports UNKNOWN (both sides missing the same field) as its own
status, distinct from MATCH/MISMATCH.

TEST COVERAGE (tests/test_validate_export.py, model-free throughout): `compare_field`/`compare_identities`
against hand-built fake identity dicts (mirrors tests/test_artifact_contracts.py's own fixture style, no
real GGUF needed); `check_chat_template_presence`/`check_template_markers`/`check_bos_eos_ids`/
`check_vocab_size_sane`/`_expected_tmpl_key`/`single_file_findings` against hand-built metadata dicts;
`format_two_file_report`/`format_single_file_report` against canned report dicts; `cmd_validate_export`'s
exit-code contract by monkeypatching `contracts.gguf_identity` and `_read_raw_metadata` (mirrors
tests/test_artifact_contracts.py's `monkeypatch.setattr(contracts, "gguf_header_from_path", ...)` pattern)
against throwaway files that only need to exist, never be real GGUFs; `add_subparser`'s argparse wiring.

DEFERRED (by design, same discipline as `clozn diff-model`/`clozn quant-check`/`clozn ci`'s own diff
check): `run_known_answers_check` -- the real two-engine boot behind `--known-answers` -- needs a free GPU
and two engine processes, so it is never invoked by this module's own test suite; `cmd_validate_export`'s
own tests monkeypatch `run_known_answers_check` itself wherever `--known-answers` is exercised, exactly how
tests/test_ci_check.py monkeypatches `ci_check.run_diff_check`. Once a GPU is free:

    clozn validate-export base-fp16.gguf exported-q4.gguf --known-answers 8
    clozn validate-export exported.gguf --strict

`add_subparser` builds its OWN `validate-export` subparser and calls `.set_defaults(fn=cmd_validate_export)`
itself; registered in clozn/cli/main.py alongside the other subcommands (registration only -- no other
change to main.py's dispatch).
"""
from __future__ import annotations

import json
import os
import sys

from clozn.artifacts import contracts
from clozn.cli import formatting as fmt
from clozn.cli.fit_planner import gguf_header_from_path
from clozn.cli.commands.models import resolve_model, _flags_for
from clozn.cli.engine_process import _free_port, spawn_engine
import clozn.cli.commands.diff_model as dm
import clozn.cli.commands.quant_check as qc

# ------------------------------------------------------------------------------------------- two-file mode

# The identity fields whose disagreement is an actual export bug -- everything `gguf_identity` reports
# about tokenizer contract and activation shape EXCEPT quantization (see module docstring: a quant
# difference is the common, expected reason to run this command and is reported separately, never folded
# into this list).
_IDENTITY_FIELDS = ("tokenizer_sha256", "chat_template_sha256", "vocab_size",
                    "architecture", "hidden_size", "layer_count")


def compare_field(name: str, expected, exported) -> dict:
    """One field's three-way verdict -- MATCH / MISMATCH / UNKNOWN, never silently skipped (this task's own
    honesty requirement): UNKNOWN only when BOTH sides are missing the value (nothing to compare at all);
    MATCH when both sides carry the SAME known value; MISMATCH otherwise -- including the case where only
    one side has a value, which is itself exactly the kind of drift this gate exists to catch (e.g. the
    export lost its chat_template entirely -- exported is None, expected is a real hash -- reads as a
    MISMATCH, not an UNKNOWN, since one side DID resolve a value and it disagrees with the other)."""
    if expected is None and exported is None:
        status = "UNKNOWN"
    elif expected == exported:
        status = "MATCH"
    else:
        status = "MISMATCH"
    return {"field": name, "status": status, "expected": expected, "exported": exported}


def compare_identities(expected_identity: dict, exported_identity: dict) -> dict:
    """Pure comparison of two `contracts.gguf_identity(...)`-shaped dicts (no I/O -- testable on hand-built
    fixtures, exactly like tests/test_artifact_contracts.py's own `_header`/`model` fixtures). Returns:
        {"fields": [compare_field(...) for each of _IDENTITY_FIELDS],
         "mismatched_fields": [name, ...],
         "quantization": {"expected": ..., "exported": ..., "same": bool},   # informational, never gates
         "ok": bool}   # True iff every field in _IDENTITY_FIELDS is MATCH or UNKNOWN (no MISMATCH)
    "ok" is this module's whole exit-code contract for two-file mode: quantization never participates."""
    fields = [compare_field(name, expected_identity.get(name), exported_identity.get(name))
             for name in _IDENTITY_FIELDS]
    mismatched = [f["field"] for f in fields if f["status"] == "MISMATCH"]
    q_expected, q_exported = expected_identity.get("quantization"), exported_identity.get("quantization")
    return {
        "fields": fields,
        "mismatched_fields": mismatched,
        "quantization": {"expected": q_expected, "exported": q_exported, "same": q_expected == q_exported},
        "ok": not mismatched,
    }


# --------------------------------------------------------------------------- two-file mode: known-answers

def run_known_answers_check(expected_path: str, exported_path: str, n: int, *, cpu: bool = False) -> dict:
    """THE LIVE PATH behind `--known-answers N` -- DEFERRED, same discipline as `diff_model.cmd_diff_model`
    / `quant_check.cmd_quant_check` / `ci_check.run_diff_check`: boots BOTH GGUFs via `spawn_engine` (the
    SAME boot path `clozn run`/`clozn diff-model` use) on two free ports, then delegates entirely to
    `diff_model.run_diff_model`'s reference-anchored ladder (generate under the expected/reference file,
    teacher-force under both) -- exactly the behavioral question this spot check asks: "what would the
    exported file change about the expected file's behavior". Always tears down both engines, even on
    error. Written and reachable, but needs a free GPU and two engine processes, so it is NEVER invoked by
    this module's own test suite (`cmd_validate_export`'s tests monkeypatch this whole function instead,
    exactly how tests/test_ci_check.py monkeypatches `ci_check.run_diff_check`). Returns:
        {"tokenizer_compat": {...}, "template_match": bool, "agg": {...}, "verdict": {...}}
    """
    EngineClient = qc._import_engine_client()
    port_a, port_b = _free_port(), _free_port()
    prefer_gpu = not cpu

    proc_a = proc_b = None
    try:
        proc_a, _health_a, _gpu_a = spawn_engine(expected_path, port_a, _flags_for(expected_path),
                                                 prefer_gpu=prefer_gpu)
        proc_b, _health_b, _gpu_b = spawn_engine(exported_path, port_b, _flags_for(exported_path),
                                                 prefer_gpu=prefer_gpu)
        eng_a, eng_b = EngineClient(port=port_a), EngineClient(port=port_b)
        import argparse
        diff_args = argparse.Namespace(runs=n, from_log=False, topk=8, max_tokens=200,
                                       both=False, own_templates=False)
        result = dm.run_diff_model(eng_a, eng_b, diff_args, label_a="expected", label_b="exported")
    finally:
        for proc in (proc_a, proc_b):
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()

    ref = result["reference_anchored"]
    return {"tokenizer_compat": result["tokenizer_compat"], "template_match": result["template_match"],
            "agg": ref["agg"], "verdict": ref["verdict"]}


# ------------------------------------------------------------------------------------------ single-file mode

# Known chat-format role-marker substrings, keyed by the SAME "tmpl" vocabulary
# clozn.cli.commands.models.KNOWN already uses ("chatml" is this module's own label for the common
# recognized-chat-model-with-no-explicit-tmpl case -- see module docstring). A heuristic, deliberately
# small and named as such: presence proves the template COULD be a real chat template of that family;
# absence is a WARNING, not a claim the template is definitely wrong (a legitimately different but valid
# custom template is possible and this v0 gate cannot rule that out -- see _HONESTY_FOOTER).
TEMPLATE_MARKERS = {
    "chatml": ("<|im_start|>", "<|im_end|>"),
    "llama3": ("<|start_header_id|>", "<|eot_id|>"),
    "mistral": ("[INST]", "[/INST]"),
    "gemma": ("<start_of_turn>", "<end_of_turn>"),
}


def _expected_tmpl_key(path: str) -> str | None:
    """Best-effort expected chat-template FAMILY for a GGUF, from the same `_flags_for(path)`/`KNOWN` table
    `clozn run`/`clozn serve` already use to pick engine launch flags -- no new per-model table invented
    here. Returns the "tmpl" flag when KNOWN names one explicitly (llama3/mistral/gemma), "chatml" for any
    other recognized-or-guessed chat model (KNOWN's chat:True entries with no explicit "tmpl", plus
    `_flags_for`'s own fallback guess for an unrecognized filename containing "instruct"/"chat"), or None
    when nothing suggests this is a chat model at all (in which case `check_template_markers` falls back to
    a GENERIC check across every known family instead of refusing to check anything)."""
    flags = _flags_for(path)
    if not flags.get("chat"):
        return None
    return flags.get("tmpl", "chatml")


def _read_raw_metadata(path: str) -> dict:
    """Direct GGUF header metadata read -- needed for single-file mode's chat_template STRING inspection
    (the marker-string heuristic) and BOS/EOS token id presence, neither of which survives into
    `contracts.gguf_identity`'s digest-only fields (chat_template_sha256 is a HASH; you cannot grep a hash
    for a marker substring). Uses the exact same header reader `contracts.gguf_identity` itself calls
    under the hood (`clozn.cli.fit_planner.gguf_header_from_path`) rather than adding a new field to
    `contracts.gguf_identity` -- that module is out of scope for this task (another workstream owns it) and
    deliberately narrow by its own docstring. Raises the same `ValueError`/`NeedMoreBytes` that function
    raises on a malformed/truncated GGUF; `cmd_validate_export` turns that into a clean `CloznError`."""
    header = gguf_header_from_path(path)
    return header.get("metadata") or {}


def check_chat_template_presence(chat_template) -> dict:
    """WARN (never a hard failure at this layer -- `--strict` decides whether a warning gates) when no
    non-empty chat_template string is present at all. Cites the measured failure mode by name: see
    `clozn/cli/commands/diff_model.py`'s module docstring, "A BASE model driven through a chat template
    often emits degenerate text" -- an export missing this metadata will still load and generate fine
    right up until something (an OpenAI-shaped client, `clozn run --chat`, Open WebUI) drives it through
    chat scaffolding, at which point it degrades silently."""
    has_template = isinstance(chat_template, str) and chat_template.strip() != ""
    if has_template:
        return {"check": "chat_template_present", "status": "OK",
                "detail": "chat_template metadata is present"}
    return {"check": "chat_template_present", "status": "WARN",
            "detail": ("no tokenizer.chat_template metadata found. A template-less GGUF driven through "
                       "chat scaffolding (system/user/assistant roles) emits degenerate text -- measured "
                       "caveat, see clozn diff-model's module docstring. This export may genuinely be a "
                       "base (non-chat) checkpoint, or the export step silently dropped the template; "
                       "this check cannot tell the two apart, only that the metadata is missing.")}


def check_template_markers(chat_template, expected_tmpl_key: str | None) -> dict:
    """Heuristic: does the template contain any marker string of its EXPECTED family (`expected_tmpl_key`,
    from `_expected_tmpl_key`), or -- when the family couldn't be determined from the filename -- any
    marker of ANY known family (a strictly weaker, explicitly labeled GENERIC signal). UNKNOWN (not
    WARN/OK) when there is no template at all to inspect -- `check_chat_template_presence` already reports
    that absence; this check has nothing to grep and would be double-counting the same fact under a
    different name if it also warned."""
    if not (isinstance(chat_template, str) and chat_template.strip()):
        return {"check": "template_role_markers", "status": "UNKNOWN",
                "detail": "no chat_template to inspect (see chat_template_present)"}
    if expected_tmpl_key and expected_tmpl_key in TEMPLATE_MARKERS:
        markers = TEMPLATE_MARKERS[expected_tmpl_key]
        found = [m for m in markers if m in chat_template]
        if found:
            return {"check": "template_role_markers", "status": "OK",
                    "detail": f"found expected {expected_tmpl_key} marker(s): {found}"}
        return {"check": "template_role_markers", "status": "WARN",
                "detail": (f"template present, but none of the expected {expected_tmpl_key} role markers "
                          f"{list(markers)} were found -- possible wrong-template export (this filename "
                          f"matched a known {expected_tmpl_key}-family model)")}
    all_markers = [m for markers in TEMPLATE_MARKERS.values() for m in markers]
    found = [m for m in all_markers if m in chat_template]
    if found:
        return {"check": "template_role_markers", "status": "OK",
                "detail": f"filename didn't match a known family; GENERIC check found marker(s): {found}"}
    return {"check": "template_role_markers", "status": "WARN",
            "detail": ("filename didn't match a known family, and no known chat-format marker string was "
                      "found anywhere in the template -- cannot confirm this template renders real chat "
                      "turns at all (weak signal: a legitimate, unrecognized custom template would also "
                      "warn here)")}


def check_bos_eos_ids(metadata: dict) -> dict:
    """WARN if either tokenizer.ggml.bos_token_id or tokenizer.ggml.eos_token_id is absent from metadata --
    a documented garbled-export failure mode (wrong sequence boundaries silently change what "the model is
    done" or "this is the start of a turn" means to the runtime)."""
    bos = metadata.get("tokenizer.ggml.bos_token_id")
    eos = metadata.get("tokenizer.ggml.eos_token_id")
    missing = [name for name, val in (("BOS", bos), ("EOS", eos)) if val is None]
    if not missing:
        return {"check": "bos_eos_ids_present", "status": "OK",
                "detail": f"bos_token_id={bos}, eos_token_id={eos}"}
    return {"check": "bos_eos_ids_present", "status": "WARN",
            "detail": (f"missing {'/'.join(missing)} token id metadata "
                      f"(tokenizer.ggml.bos_token_id={bos!r}, tokenizer.ggml.eos_token_id={eos!r}) -- a "
                      "wrong or omitted BOS/EOS id is a documented garbled-export failure mode")}


def check_vocab_size_sane(vocab_size) -> dict:
    """WARN unless vocab_size is a real positive integer -- the cheapest possible sanity floor on the
    tokenizer metadata actually having been read/exported at all."""
    if isinstance(vocab_size, int) and not isinstance(vocab_size, bool) and vocab_size > 0:
        return {"check": "vocab_size_sane", "status": "OK", "detail": f"vocab_size={vocab_size}"}
    return {"check": "vocab_size_sane", "status": "WARN",
            "detail": (f"vocab_size is {vocab_size!r} -- not a sane positive integer; tokenizer metadata "
                      "may be missing, truncated, or corrupt in this export")}


def single_file_findings(path: str, identity: dict, metadata: dict) -> list:
    """The whole single-file checklist (module docstring's mode 2), in the exact order the roadmap item
    lists them: chat-template presence, expected role markers, BOS/EOS ids, vocab_size sanity. Pure given
    `identity` (a `contracts.gguf_identity(path)`-shaped dict) and `metadata` (a raw GGUF metadata dict,
    from `_read_raw_metadata`) -- no I/O of its own, testable on hand-built dicts."""
    chat_template = metadata.get("tokenizer.chat_template")
    return [
        check_chat_template_presence(chat_template),
        check_template_markers(chat_template, _expected_tmpl_key(path)),
        check_bos_eos_ids(metadata),
        check_vocab_size_sane(identity.get("vocab_size")),
    ]


# ------------------------------------------------------------------------------------------------ rendering

_HONESTY_FOOTER = (
    "validate-export is a STATIC metadata gate (tokenizer/template/vocab/architecture identity read "
    "straight off the GGUF header), plus -- with --known-answers -- an optional per-token BEHAVIORAL SPOT "
    "CHECK over a small sample. Neither proves semantic equivalence between two checkpoints: it only "
    "confirms (or catches drift in) these specific, cheap-to-check signals on these exact files."
)


def _fmt_identity_block(identity: dict) -> list:
    return [
        f"  architecture={identity.get('architecture')}  hidden_size={identity.get('hidden_size')}  "
        f"layer_count={identity.get('layer_count')}  vocab_size={identity.get('vocab_size')}",
        f"  quantization={identity.get('quantization')}  file_size={identity.get('file_size')} bytes",
        f"  tokenizer_sha256={identity.get('tokenizer_sha256')}",
        f"  chat_template_sha256={identity.get('chat_template_sha256')}",
        f"  sha256={identity.get('sha256')}",
    ]


def format_two_file_report(report: dict) -> str:
    """Pure JSON(two-file report dict)->text render -- no I/O, testable on a canned dict. `report` is the
    shape `cmd_validate_export` builds: {"mode": "two-file", "expected_path", "exported_path",
    "expected_identity", "exported_identity", "comparison" (compare_identities' result),
    "known_answers" (run_known_answers_check's result, or None)}."""
    lines = [f"validate-export: {report.get('expected_path')} (expected) vs "
            f"{report.get('exported_path')} (exported)", ""]
    comparison = report.get("comparison") or {}
    for f in comparison.get("fields", []):
        lines.append(f"  [{f['status']:<8}] {f['field']}: expected={f['expected']!r}  "
                     f"exported={f['exported']!r}")
    q = comparison.get("quantization") or {}
    same = "same" if q.get("same") else "DIFFERS (reported, not a failure)"
    lines.append(f"  [INFO    ] quantization: expected={q.get('expected')!r}  exported={q.get('exported')!r}"
                f"  ({same})")
    lines.append("")
    lines.append("expected identity (for the record):")
    lines.extend(_fmt_identity_block(report.get("expected_identity") or {}))
    lines.append("exported identity (for the record):")
    lines.extend(_fmt_identity_block(report.get("exported_identity") or {}))
    lines.append("")
    if comparison.get("ok"):
        lines.append(f"{fmt.BOLD}PASS{fmt.RST} -- tokenizer/template/vocab/architecture identity matches")
    else:
        lines.append(f"{fmt.BOLD}FAIL{fmt.RST} -- mismatch in: {', '.join(comparison.get('mismatched_fields', []))}")

    ka = report.get("known_answers")
    if ka is not None:
        lines.append("")
        lines.append("known-answers behavioral spot check (informational -- does not change the exit code):")
        compat = ka.get("tokenizer_compat") or {}
        lines.append(f"  tokenizer preflight: {'compatible' if compat.get('compatible') else 'INCOMPATIBLE'}")
        lines.append(f"  chat templates match under both engines: {ka.get('template_match')}")
        lines.append(qc.format_ladder(ka.get("agg", {})))
        lines.append("")
        lines.append(dm.format_verdict(ka.get("verdict", {})))
    lines.append("")
    lines.append(_HONESTY_FOOTER)
    return "\n".join(lines)


def format_single_file_report(report: dict) -> str:
    """Pure JSON(single-file report dict)->text render. `report` is the shape `cmd_validate_export`
    builds: {"mode": "single-file", "path", "identity", "findings" (single_file_findings' result)}."""
    identity = report.get("identity") or {}
    findings = report.get("findings") or []
    lines = [f"validate-export (single-file): {report.get('path')}", "", "identity (for the record):"]
    lines.extend(_fmt_identity_block(identity))
    lines.append("")
    lines.append("checks:")
    n_warn = 0
    for f in findings:
        if f["status"] == "WARN":
            n_warn += 1
        lines.append(f"  [{f['status']:<7}] {f['check']}: {f['detail']}")
    lines.append("")
    lines.append(f"{n_warn} warning(s)." if n_warn else "no warnings.")
    lines.append("")
    lines.append(_HONESTY_FOOTER)
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ the CLI

def add_subparser(sub):
    """Registers `clozn validate-export` on an argparse subparsers object -- same pattern as
    commands.diff_model/quant_check/ci_check.add_subparser: this module's own tests can build a throwaway
    parser and exercise --help/defaults/flag-parsing without touching main.py, and it documents the exact
    main.py registration edit as real, testable code. NOT called automatically -- clozn/cli/main.py owns
    build_parser()/dispatch."""
    pv = sub.add_parser("validate-export", help="deployment-equivalence check v0 (Phase-1 §4.5): compare "
                        "an exported GGUF's tokenizer/template/vocab/architecture identity against a "
                        "known-good reference GGUF (two-file mode), or sanity-check one exported GGUF on "
                        "its own (single-file mode, omit the second argument) -- static metadata only "
                        "unless --known-answers asks for the optional live behavioral spot check")
    pv.add_argument("gguf_a", help="two-file mode: the EXPECTED/reference GGUF. single-file mode (gguf_b "
                    "omitted): the exported GGUF to sanity-check on its own. A known short name, local "
                    "GGUF path, or fuzzy filename fragment -- resolved like `clozn run`'s model arg")
    pv.add_argument("gguf_b", nargs="?", default=None, help="the EXPORTED GGUF under test -- omit for "
                    "single-file mode")
    pv.add_argument("--known-answers", type=int, default=0, metavar="N", dest="known_answers",
                    help="two-file mode only: also boot both GGUFs and run diff-model's reference-anchored "
                         "ladder over N runs as a behavioral spot check (LIVE -- needs a free GPU and two "
                         "engine processes); reported alongside the static report but does NOT change the "
                         "exit code. Default 0 skips it entirely and stays purely static/no-GPU")
    pv.add_argument("--strict", action="store_true", help="single-file mode: exit 1 if any check produced "
                    "a warning (e.g. missing chat template), instead of the default exit 0 with warnings "
                    "printed. No effect in two-file mode (which already exits 1 on any mismatch)")
    pv.add_argument("--cpu", action="store_true", help="force the CPU build for --known-answers' two engines")
    pv.add_argument("--json", action="store_true",
                    help="print the raw report as JSON instead of the text summary")
    pv.set_defaults(fn=cmd_validate_export)
    return pv


def cmd_validate_export(args):
    """`clozn validate-export <expected.gguf> [exported.gguf] [--known-answers N] [--strict]` -- see
    add_subparser for the full flag list and the module docstring for the exact schema/exit-code contract
    of each mode. Two-file mode (`args.gguf_b` given): exit 0 if `compare_identities(...)["ok"]`, else 1.
    Single-file mode (`args.gguf_b` is None): exit 0 unless `--strict` and at least one WARN finding, in
    which case 1. Any GGUF that can't even be read raises a clean `CloznError` (main.py's standard exit 1),
    same as every other `resolve_model`-based command in this codebase."""
    from clozn.cli import main as ctx

    path_a = resolve_model(args.gguf_a)

    if args.gguf_b:
        path_b = resolve_model(args.gguf_b)
        try:
            identity_a = contracts.gguf_identity(path_a)
            identity_b = contracts.gguf_identity(path_b)
        except Exception as e:
            raise ctx.CloznError(f"couldn't read GGUF identity: {e}")

        comparison = compare_identities(identity_a, identity_b)
        known_answers = None
        if args.known_answers:
            print(f"{fmt.DIM}- known-answers spot check: booting both engines for {args.known_answers} "
                 f"run(s)...{fmt.RST}", file=sys.stderr)
            known_answers = run_known_answers_check(path_a, path_b, args.known_answers, cpu=args.cpu)

        report = {"mode": "two-file", "expected_path": path_a, "exported_path": path_b,
                 "expected_identity": identity_a, "exported_identity": identity_b,
                 "comparison": comparison, "known_answers": known_answers}
        if args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print(format_two_file_report(report))
        return 0 if comparison["ok"] else 1

    if args.known_answers:
        raise ctx.CloznError("--known-answers needs a second GGUF to compare against (two-file mode only): "
                             "clozn validate-export <expected.gguf> <exported.gguf> --known-answers N")

    try:
        identity = contracts.gguf_identity(path_a)
        metadata = _read_raw_metadata(path_a)
    except Exception as e:
        raise ctx.CloznError(f"couldn't read GGUF {path_a!r}: {e}")

    findings = single_file_findings(path_a, identity, metadata)
    report = {"mode": "single-file", "path": path_a, "identity": identity, "findings": findings}
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_single_file_report(report))

    any_warn = any(f["status"] == "WARN" for f in findings)
    return 1 if (any_warn and args.strict) else 0
