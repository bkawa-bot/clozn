"""commands.ci_check -- `clozn ci baseline` / `clozn ci check` (Phase-1 §4.4, "Headless CI gate", PRODUCT_
ROADMAP.md: "'CI' isn't CI until a pipeline can fail on it"): a DETERMINISTIC gate over the EXISTING
primitives -- `clozn test-model` (clozn.eval.golden), the tiny-test harness (clozn.testkit), and
`clozn diff-model` (clozn.cli.commands.diff_model) -- with no new orchestration layer of its own. Every
number this module measures comes from calling one of those three, unmodified; this module's own job is
recording a BASELINE (the budgets a later run must clear) and then CHECKING a model against it, printing a
compact report, writing a machine-readable one, and returning one of exactly four exit codes.

WHY NOT `clozn.testkit.ci` -- that module (clozn/testkit/ci.py) is a DIFFERENT, pre-existing thing: a
live suite orchestrator (Client/run_case/run_suite/diff_suites) that calls a running gateway's /v1/chat/
completions itself and diffs two of its OWN SuiteResults. This module never generates anything and never
owns an HTTP client -- it is a thin, budgeted comparison wrapper around three ALREADY-shipped commands'
own outputs, per this task's "no new orchestration" scope. The two modules may converge later (a `clozn
ci` case type backed by testkit.ci is a natural v1 extension) but v0 deliberately does not reach for it.

======================================================================================== BASELINE SCHEMA
`clozn ci baseline <out.json> <model> [options]` writes:

    {
      "schema_version": 1,
      "created_at": "<ISO-8601 UTC>",
      "pin_model": <bool>,                        -- see IDENTITY POLICY below
      "identity": {                                -- clozn.runs.identity.runtime_identity(model_path=...)
          "model_path": "...", "model_sha256": "...", "model_size_bytes": N,
          "clozn_version": "...", "captured_at": "..."   -- (template_fingerprint/engine_build omitted:
                                                              no live engine is booted just to baseline)
      },
      "checks": {
        "golden": {
          "enabled": <bool>, "which": "all",        -- clozn test-model's --set
          "min_pass_rate": <float>,                 -- budget; default = THIS baseline run's own pass_rate
          "measured": {"n": N, "n_correct": N, "pass_rate": <float|None>}
        },
        "tiny": {
          "enabled": <bool>,                        -- true iff --tiny was given at least once
          "files": [
            {"path": "...", "baseline_status": "pass|fail|skip|error",
             "baseline_counts": {...}, "baseline_passing_tests": ["<test name>", ...],
             "error": <str|None>}                    -- set if the file itself failed to load/run
          ]
        },
        "diff": {
          "enabled": <bool>,                         -- true iff --reference was given
          "reference": "<gguf path>", "runs": <int>,  -- the diff-model ladder size used for this baseline
          "max_argmax_flips_total": <int>,            -- budget; default = THIS run's measured flip count
          "max_mean_abs_delta_nats": <float>,         -- budget; default = THIS run's measured mean delta
          "measured": {"total_tokens": N, "total_flipped": N,
                       "mean_abs_delta_nats_all_mean": <float|None>, "verdict": "<classify_verdict label>"}
        }
      }
    }

A check with "enabled": false was never measured and `clozn ci check` never attempts it -- there is no
"disabled but has budgets anyway" state.

======================================================================================== IDENTITY POLICY
Every baseline is measured against SOME model; `identity.model_sha256` (cached, see clozn/runs/identity.py)
is that model's fingerprint. `pin_model` (default False, `--pin-model` at baseline time) decides what a
sha MISMATCH means at check time:

  * pin_model false (the common case: "gate a NEW model against an OLD baseline") -- a different sha is
    EXPECTED and never refused; the budgets still apply (that is the whole point of the gate).
  * pin_model true -- the baseline is declaring "this artifact is only valid for THIS exact model file".
    `clozn ci check` REFUSES (raises `CIIdentityRefusal`, exit code 3, before running any of the three
    checks) unless `--allow-model-change` is passed. A refusal also fires if either side's sha256 could
    not be established at all (a hashing failure is treated as "cannot prove a match", never silently
    waved through as one) -- this is the one case `--allow-model-change` exists to override deliberately.

======================================================================================= EXIT CODE CONTRACT
    0 -- every enabled check passed its budget
    1 -- at least one enabled check ran and violated its budget (this INCLUDES a check that could not
         run at all -- "FAILED-with-reason", per this task's honesty requirement, never a silent skip --
         and the degenerate case of a baseline with NO enabled checks, which fails rather than vacuously
         passing: a gate that checks nothing is a misconfiguration, not a clean bill of health)
    2 -- execution error: the baseline file couldn't be loaded/parsed, or the model argument couldn't even
         be resolved to a file -- i.e. the gate never got far enough to evaluate a single check
    3 -- identity-policy refusal (see above) -- checked BEFORE any check runs, since it's a precondition
         on the whole gate, not a budget any individual check enforces

Deterministic by construction: every number above comes from a stored fixture (golden fixture / tiny-test
run-store records / diff-model's teacher-forced ladder), never from wall-clock timing -- there is
deliberately NO latency/throughput budget in v0 (those are measured-machine-dependent and flaky across
CI runners; a perf gate belongs in a dedicated benchmark job, not this correctness gate).

============================================================================================ TEST COVERAGE
Model-free throughout (mirrors tests/test_diff_model.py's own discipline): `identity_policy_check` is pure
and tested directly; `run_golden_check` is tested against FIXTURE `clozn.eval.golden.run_and_grade`/
`.engine_health` outputs (mirrors tests/test_cli_test_model.py's own monkeypatch target); `run_tiny_check`
is tested against a REAL tiny-test spec file + an isolated run store (mirrors tests/test_testkit_cli.py's
`iso`/`_make_run` fixtures) -- no monkeypatching needed, since the tiny-test harness's static checks are
already model-free by design; `run_diff_check` is the one LIVE, GPU-needing primitive (boots two engines,
exactly `cmd_diff_model`'s own deferred path) and is therefore never invoked by this module's own tests --
`build_baseline`/`run_gate`'s handling of the "diff" check is instead exercised by monkeypatching
`ci_check.run_diff_check` itself, exactly how tests/test_diff_model.py monkeypatches `dm.run_direction`
(a function ITS module defines that wraps further primitives). `add_subparser`'s argparse wiring is
exercised on a throwaway parser and via `build_parser()`.
"""
from __future__ import annotations

import json
import os
import sys
import types
from datetime import datetime, timezone

from clozn.cli import formatting as fmt

_DEFAULT_URL = "http://127.0.0.1:8080"
SCHEMA_VERSION = 1


class CIIdentityRefusal(Exception):
    """`clozn ci check` exit 3 -- the baseline is pinned (`pin_model: true`) and the current model's
    identity does not provably match it, and `--allow-model-change` wasn't passed. Raised by `run_gate`
    BEFORE any of the three checks run (see module docstring's EXIT CODE CONTRACT) and caught inside
    `cmd_ci_check` itself -- it must never reach `clozn.cli.main`'s generic `CloznError` handler, which
    only knows how to return 1, not this module's own 4-way exit-code contract."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =================================================================================== primitive runners ===
# Each of these wraps ONE existing, already-tested primitive and nothing else. They are called by bare
# name from build_baseline/run_gate below (never `ci_check.run_x(...)`), so `monkeypatch.setattr(ci_check,
# "run_golden_check", fake)` redirects every call site -- exactly the pattern tests/test_diff_model.py
# uses for `dm.run_direction`.

def run_golden_check(url: str, which: str) -> dict:
    """LIVE (needs a running Clozn gateway at `url` -- same precondition as `clozn test-model`, see
    commands.test_model's module docstring). Calls `clozn.eval.golden.run_and_grade` + `.engine_health`
    UNMODIFIED and reduces their output to what a budget check needs. Returns:
        {"n": int, "n_correct": int, "pass_rate": float|None (None iff n == 0 -- no probes graded),
         "wrong": [{"q", "gold", "reply"}, ...], "model": str|None, "model_sha256": str|None}
    Never catches its own errors -- a gateway that's down should raise here; `_check_golden` below is
    where that becomes a FAILED-with-reason check result instead of an uncaught exception."""
    from clozn.eval import golden

    rows = golden.run_and_grade(url, which)
    health = golden.engine_health(url)
    n = len(rows)
    n_correct = sum(1 for r in rows if r.get("correct"))
    pass_rate = round(n_correct / n, 6) if n else None
    wrong = [{"q": r.get("q"), "gold": r.get("gold"), "reply": r.get("reply")}
             for r in rows if not r.get("correct")]
    return {"n": n, "n_correct": n_correct, "pass_rate": pass_rate, "wrong": wrong,
            "model": health.get("model"), "model_sha256": health.get("model_sha256")}


def run_tiny_check(spec_path: str) -> dict:
    """Model-free (no gateway needed for the STATIC assertions this gate uses -- see
    clozn/testkit/runner.py's module docstring; this never passes `sub`/`fetch_receipt`, so any `leans_on`
    causal assertion is honestly SKIPPED, matching the harness's own default `--live`-less behavior, never
    silently passed). Loads the JSON spec at `spec_path` and evaluates it with `clozn.testkit.run_suite`
    UNMODIFIED -- exactly what `clozn test <file>` itself does. Returns, on a loadable spec:
        {"file": spec_path, "status": <suite status>, "counts": {...}, "error": None,
         "by_test": {<test name>: <status>}, "suite": <raw run_suite() result>}
    or, if the file itself couldn't be read/parsed/shaped (mirrors commands.test._load_test_spec's own
    three failure modes):
        {"file": spec_path, "status": "error", "error": <str>}
    Never raises."""
    from clozn import testkit

    try:
        with open(spec_path, encoding="utf-8") as f:
            spec = json.load(f)
    except OSError as e:
        return {"file": spec_path, "status": "error", "error": f"could not read {spec_path}: {e}"}
    except json.JSONDecodeError as e:
        return {"file": spec_path, "status": "error", "error": f"{spec_path} is not valid JSON: {e}"}
    if not isinstance(spec, dict) or not isinstance(spec.get("tests"), list) or not spec["tests"]:
        return {"file": spec_path, "status": "error",
                "error": f"{spec_path}: spec must be a JSON object with a non-empty 'tests' list"}

    suite = testkit.run_suite(spec)
    by_test = {t.get("name"): t.get("status") for t in (suite.get("tests") or [])}
    return {"file": spec_path, "status": suite.get("status"), "counts": suite.get("counts", {}),
            "error": None, "by_test": by_test, "suite": suite}


def run_diff_check(reference: str, candidate: str, *, runs: int = 8, cpu: bool = False) -> dict:
    """THE LIVE PATH -- DEFERRED, same discipline as `quant_check.cmd_quant_check`/`diff_model.
    cmd_diff_model`: boots the reference and candidate GGUFs on two engine processes (`clozn.cli.
    engine_process.spawn_engine`, the SAME boot path `clozn run`/`clozn diff-model` use) and delegates to
    `clozn.cli.commands.diff_model.run_diff_model` (unmodified) for ONLY the reference-anchored ladder --
    the gate's budgets ask "did the candidate change/forget the reference's behavior", which is exactly
    that direction; `--both`'s reverse candidate-anchored ladder is a `clozn diff-model` feature this gate
    does not need. Always tears down both engines, even on error. Written and reachable, but needs a free
    GPU and two engine processes, so it is NEVER invoked by this module's own test suite (see module
    docstring's TEST COVERAGE section) -- `build_baseline`/`run_gate`'s own tests monkeypatch this whole
    function instead. Returns:
        {"total_tokens": int, "total_flipped": int, "mean_abs_delta_nats_all_mean": float|None,
         "verdict": str, "top_flips": [...], "caveat": str|None, "topk_note": str|None,
         "label_a": str, "label_b": str}
    """
    import clozn.cli.commands.diff_model as dm
    import clozn.cli.commands.quant_check as qc
    from clozn.cli.commands.models import resolve_model, _flags_for
    from clozn.cli.engine_process import _free_port, spawn_engine

    EngineClient = qc._import_engine_client()
    model_a = resolve_model(reference)
    model_b = resolve_model(candidate)
    label_a = os.path.splitext(os.path.basename(model_a))[0]
    label_b = os.path.splitext(os.path.basename(model_b))[0]
    port_a = _free_port()
    port_b = _free_port()
    prefer_gpu = not cpu

    proc_a = proc_b = None
    try:
        proc_a, _h_a, _g_a = spawn_engine(model_a, port_a, _flags_for(model_a), prefer_gpu=prefer_gpu)
        proc_b, _h_b, _g_b = spawn_engine(model_b, port_b, _flags_for(model_b), prefer_gpu=prefer_gpu)
        eng_a = EngineClient(port=port_a)
        eng_b = EngineClient(port=port_b)
        diff_args = types.SimpleNamespace(runs=runs, from_log=False, topk=8, max_tokens=200,
                                          both=False, own_templates=False)
        result = dm.run_diff_model(eng_a, eng_b, diff_args, label_a=label_a, label_b=label_b)
    finally:
        for proc in (proc_a, proc_b):
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()

    ref = result["reference_anchored"]
    agg, verdict = ref["agg"], ref["verdict"]
    return {"total_tokens": agg.get("total_tokens", 0), "total_flipped": agg.get("total_flipped", 0),
            "mean_abs_delta_nats_all_mean": verdict.get("mean_abs_delta_nats_all_mean"),
            "verdict": verdict.get("verdict"), "top_flips": agg.get("top_flips", []),
            "caveat": agg.get("caveat"), "topk_note": agg.get("topk_note"),
            "label_a": label_a, "label_b": label_b}


# ========================================================================================= identity =====

def _current_identity(model_path: str) -> dict:
    from clozn.runs import identity
    return identity.runtime_identity(model_path=model_path)


def identity_policy_check(baseline_identity: dict, pin_model: bool, current_identity: dict,
                          allow_model_change: bool) -> dict:
    """Pure (no I/O of its own). Decides whether `clozn ci check` may proceed -- see module docstring's
    IDENTITY POLICY section for the exact rule. Returns:
        {"pin_model": bool, "allow_model_change": bool, "baseline_sha256": str|None,
         "current_sha256": str|None, "match": bool, "ok": bool, "reason": str|None}
    "match" is True only when BOTH shas are known and equal -- an unknown sha on either side is never
    silently treated as a match. "ok" is False only when pin_model is true, match is False, and
    allow_model_change is false; "reason" is populated exactly then, and distinguishes an actual mismatch
    from an unverifiable one (missing sha on either side) in its wording."""
    baseline_sha = (baseline_identity or {}).get("model_sha256")
    current_sha = (current_identity or {}).get("model_sha256")
    match = baseline_sha is not None and current_sha is not None and baseline_sha == current_sha
    out = {"pin_model": bool(pin_model), "allow_model_change": bool(allow_model_change),
          "baseline_sha256": baseline_sha, "current_sha256": current_sha, "match": match,
          "ok": True, "reason": None}
    if pin_model and not match and not allow_model_change:
        if baseline_sha is None or current_sha is None:
            reason = (
                "baseline is pinned to a specific model (pin_model: true) but the model identity could "
                f"not be verified -- missing sha256 on the {'baseline' if baseline_sha is None else 'current model'} "
                "side. Pass --allow-model-change to proceed anyway."
            )
        else:
            reason = (
                "baseline is pinned to a specific model (pin_model: true) and the current model's sha256 "
                f"({current_sha}) does not match the baseline's ({baseline_sha}). Pass --allow-model-change "
                "if gating a NEW model against this baseline is intentional -- that is often the whole "
                "point of this gate."
            )
        out["ok"] = False
        out["reason"] = reason
    return out


# ======================================================================================= baseline build =

def build_baseline(*, model_path: str, url: str = _DEFAULT_URL, which: str = "all",
                   golden_enabled: bool = True, min_pass_rate: float | None = None,
                   tiny_files: list[str] | None = None, reference: str | None = None,
                   diff_runs: int = 8, cpu: bool = False,
                   max_argmax_flips_total: int | None = None,
                   max_mean_abs_delta_nats: float | None = None,
                   pin_model: bool = False) -> dict:
    """Model-free ORCHESTRATION (given the three runner functions above, which tests monkeypatch): measure
    each enabled v0 check ONCE against `model_path` and assemble the baseline artifact (module docstring's
    BASELINE SCHEMA). Never talks to a gateway/engine/disk itself beyond `run_golden_check`/
    `run_tiny_check`/`run_diff_check` and computing `model_path`'s identity."""
    tiny_files = list(tiny_files or [])
    identity = _current_identity(model_path)

    checks: dict = {}

    if golden_enabled:
        g = run_golden_check(url, which)
        measured_rate = g.get("pass_rate")
        checks["golden"] = {
            "enabled": True, "which": which,
            "min_pass_rate": min_pass_rate if min_pass_rate is not None else measured_rate,
            "measured": {"n": g["n"], "n_correct": g["n_correct"], "pass_rate": measured_rate},
        }
    else:
        checks["golden"] = {"enabled": False}

    file_records = []
    for path in tiny_files:
        res = run_tiny_check(path)
        by_test = res.get("by_test") or {}
        passing = sorted(name for name, status in by_test.items() if status == "pass")
        file_records.append({
            "path": path, "baseline_status": res.get("status"),
            "baseline_counts": res.get("counts", {}), "baseline_passing_tests": passing,
            "error": res.get("error"),
        })
    checks["tiny"] = {"enabled": bool(tiny_files), "files": file_records}

    if reference:
        d = run_diff_check(reference, model_path, runs=diff_runs, cpu=cpu)
        measured_flips = d.get("total_flipped", 0)
        measured_delta = d.get("mean_abs_delta_nats_all_mean")
        checks["diff"] = {
            "enabled": True, "reference": reference, "runs": diff_runs,
            "max_argmax_flips_total": (max_argmax_flips_total if max_argmax_flips_total is not None
                                       else measured_flips),
            "max_mean_abs_delta_nats": (max_mean_abs_delta_nats if max_mean_abs_delta_nats is not None
                                        else measured_delta),
            "measured": {"total_tokens": d.get("total_tokens", 0), "total_flipped": measured_flips,
                        "mean_abs_delta_nats_all_mean": measured_delta, "verdict": d.get("verdict")},
        }
    else:
        checks["diff"] = {"enabled": False}

    return {"schema_version": SCHEMA_VERSION, "created_at": _now_iso(), "pin_model": bool(pin_model),
            "identity": identity, "checks": checks}


# ============================================================================================ per-check =
# Each takes the baseline's own recorded config for that check (plus whatever else it needs) and returns
# a report-shaped dict: {"ran": bool, "passed": bool, "budget": {...}, "observed": {...}|None,
# "reason": str|None, "worst_offenders": [...]}. A check that raised is caught HERE -- never propagated --
# and turned into "ran": False, "passed": False, "reason": "<what broke>": the "FAILED-with-reason, never
# silently skipped" rule this task requires.

def _check_golden(cfg: dict, url: str) -> dict:
    which = cfg.get("which", "all")
    budget = cfg.get("min_pass_rate")
    try:
        result = run_golden_check(url, which)
    except Exception as e:
        return {"ran": False, "passed": False, "budget": {"min_pass_rate": budget, "which": which},
                "observed": None, "reason": f"golden check could not run: {type(e).__name__}: {e}",
                "worst_offenders": []}
    pass_rate = result.get("pass_rate")
    observed = {"n": result.get("n", 0), "n_correct": result.get("n_correct", 0), "pass_rate": pass_rate}
    if pass_rate is None:
        return {"ran": True, "passed": False, "budget": {"min_pass_rate": budget, "which": which},
                "observed": observed, "reason": "no probes were graded (n=0) -- nothing to gate on",
                "worst_offenders": []}
    passed = budget is None or pass_rate >= budget
    reason = None if passed else f"pass_rate {pass_rate} < budget min_pass_rate {budget} (n={observed['n']})"
    return {"ran": True, "passed": passed, "budget": {"min_pass_rate": budget, "which": which},
            "observed": observed, "reason": reason, "worst_offenders": result.get("wrong", [])[:20]}


def _check_tiny(cfg: dict) -> dict:
    files_cfg = cfg.get("files") or []
    per_file, all_regressions = [], []
    any_broken = False
    for fc in files_cfg:
        path = fc.get("path")
        baseline_passing = list(fc.get("baseline_passing_tests") or [])
        try:
            res = run_tiny_check(path)
        except Exception as e:
            res = {"status": "error", "error": f"tiny check could not run: {type(e).__name__}: {e}"}
        if res.get("error") is not None or "by_test" not in res:
            any_broken = True
            per_file.append({"path": path, "ran": False, "reason": res.get("error") or "unknown error",
                             "n_baseline_passing": len(baseline_passing)})
            for name in baseline_passing:
                all_regressions.append({"file": path, "test": name, "run_id": None,
                                        "reason": res.get("error") or "the tiny-test file could not run"})
            continue

        by_test = res.get("by_test") or {}
        by_test_run = {t.get("name"): t.get("run_id") for t in (res["suite"].get("tests") or [])}
        file_regressions = []
        for name in baseline_passing:
            now_status = by_test.get(name)
            if now_status != "pass":
                file_regressions.append({
                    "file": path, "test": name, "run_id": by_test_run.get(name),
                    "reason": f"was pass at baseline, now "
                              f"{now_status if name in by_test else 'MISSING (test renamed/removed)'}",
                })
        all_regressions.extend(file_regressions)
        per_file.append({"path": path, "ran": True, "status": res.get("status"),
                         "counts": res.get("counts", {}), "n_baseline_passing": len(baseline_passing),
                         "n_regressed": len(file_regressions)})

    passed = not any_broken and not all_regressions
    reason = None
    if any_broken and all_regressions:
        reason = "one or more tiny-test files could not run, and baseline-passing tests regressed"
    elif any_broken:
        reason = "one or more tiny-test files could not run"
    elif all_regressions:
        reason = f"{len(all_regressions)} previously-passing tiny test(s) no longer pass"
    return {"ran": True, "passed": passed,
            "budget": {"rule": "every test that passed at baseline must still pass"},
            "observed": {"files": per_file}, "reason": reason, "worst_offenders": all_regressions[:20]}


def _check_diff(cfg: dict, model_path: str, *, cpu: bool = False) -> dict:
    reference = cfg.get("reference")
    runs = cfg.get("runs", 8)
    max_flips = cfg.get("max_argmax_flips_total")
    max_delta = cfg.get("max_mean_abs_delta_nats")
    budget = {"max_argmax_flips_total": max_flips, "max_mean_abs_delta_nats": max_delta}
    try:
        result = run_diff_check(reference, model_path, runs=runs, cpu=cpu)
    except Exception as e:
        return {"ran": False, "passed": False, "budget": budget, "observed": None,
                "reason": f"diff check could not run: {type(e).__name__}: {e}", "worst_offenders": []}

    flips = result.get("total_flipped", 0)
    delta = result.get("mean_abs_delta_nats_all_mean")
    flips_ok = max_flips is None or flips <= max_flips
    delta_ok = max_delta is None or delta is None or delta <= max_delta
    passed = flips_ok and delta_ok
    reasons = []
    if not flips_ok:
        reasons.append(f"total_flipped {flips} > budget max_argmax_flips_total {max_flips}")
    if not delta_ok:
        reasons.append(f"mean_abs_delta_nats_all_mean {delta} > budget max_mean_abs_delta_nats {max_delta}")
    observed = {"total_tokens": result.get("total_tokens", 0), "total_flipped": flips,
               "mean_abs_delta_nats_all_mean": delta, "verdict": result.get("verdict")}
    return {"ran": True, "passed": passed, "budget": budget, "observed": observed,
            "reason": "; ".join(reasons) or None, "worst_offenders": result.get("top_flips", [])[:20],
            "caveat": result.get("caveat"), "topk_note": result.get("topk_note")}


# ================================================================================================ gate ==

def run_gate(*, baseline: dict, model_path: str, url: str = _DEFAULT_URL,
            allow_model_change: bool = False, cpu: bool = False) -> dict:
    """Model-free ORCHESTRATION of `clozn ci check` (given the runner functions above). Raises
    `CIIdentityRefusal` when the identity policy refuses -- checked BEFORE any of the three checks run,
    since it is a precondition on the whole gate (module docstring's IDENTITY POLICY). Every other
    problem is folded into that ONE check's own FAILED-with-reason result (never raised), so the report's
    "overall" is "fail" (exit 1), not an exception -- exit 2 is reserved for `cmd_ci_check`'s own baseline-
    loading/model-resolution failures, which happen before this function is ever called. Returns:
        {"identity": {...current...}, "identity_policy": {...}, "checks": {"golden": {...}?, "tiny": {...}?,
         "diff": {...}?}, "overall": "pass"|"fail", "reason": str|None, "generated_at": "..."}
    """
    current = _current_identity(model_path)
    pin_model = bool(baseline.get("pin_model"))
    policy = identity_policy_check(baseline.get("identity") or {}, pin_model, current, allow_model_change)
    if not policy["ok"]:
        raise CIIdentityRefusal(policy["reason"])

    checks_cfg = baseline.get("checks") or {}
    report_checks: dict = {}
    any_fail = False

    g_cfg = checks_cfg.get("golden") or {}
    if g_cfg.get("enabled"):
        report_checks["golden"] = _check_golden(g_cfg, url)
        any_fail = any_fail or not report_checks["golden"]["passed"]

    t_cfg = checks_cfg.get("tiny") or {}
    if t_cfg.get("enabled"):
        report_checks["tiny"] = _check_tiny(t_cfg)
        any_fail = any_fail or not report_checks["tiny"]["passed"]

    d_cfg = checks_cfg.get("diff") or {}
    if d_cfg.get("enabled"):
        report_checks["diff"] = _check_diff(d_cfg, model_path, cpu=cpu)
        any_fail = any_fail or not report_checks["diff"]["passed"]

    reason = None
    if not report_checks:
        any_fail = True
        reason = "baseline declares no enabled checks -- nothing to gate on (misconfigured baseline)"
    elif any_fail:
        failed = [name for name, c in report_checks.items() if not c["passed"]]
        reason = f"budget violated: {', '.join(failed)}"

    return {"identity": current, "identity_policy": policy, "checks": report_checks,
            "overall": "fail" if any_fail else "pass", "reason": reason, "generated_at": _now_iso()}


# ================================================================================================ CLI ==

def _short(sha):
    return (sha[:12] + "...") if isinstance(sha, str) and len(sha) > 12 else sha


def format_ci_report(report: dict) -> str:
    """Pure JSON(`run_gate` result, + exit_code) -> text render -- no I/O, testable on a canned dict.
    Every observed number is printed next to its own scope (n / total_tokens / budget) rather than bare,
    per this task's honesty requirement; a check's caveat (diff-model's, riding through verbatim) prints
    alongside it, never dropped."""
    lines = [f"clozn ci check -- overall: {report.get('overall', '?').upper()}"]
    if report.get("reason"):
        lines.append(f"  {report['reason']}")
    pol = report.get("identity_policy") or {}
    lines.append(f"  identity: pin_model={pol.get('pin_model')}  match={pol.get('match')}  "
                f"baseline_sha256={_short(pol.get('baseline_sha256'))}  "
                f"current_sha256={_short(pol.get('current_sha256'))}")
    for name, c in (report.get("checks") or {}).items():
        mark = "PASS" if c.get("passed") else "FAIL"
        lines.append(f"\n  [{mark}] {name}")
        if c.get("reason"):
            lines.append(f"        {c['reason']}")
        if c.get("observed") is not None:
            lines.append(f"        observed: {c['observed']}")
        if c.get("budget") is not None:
            lines.append(f"        budget:   {c['budget']}")
        for off in (c.get("worst_offenders") or [])[:5]:
            lines.append(f"        offender: {off}")
        if c.get("caveat"):
            lines.append(f"        {fmt.DIM}{c['caveat']}{fmt.RST}")
        if c.get("topk_note"):
            lines.append(f"        {fmt.DIM}{c['topk_note']}{fmt.RST}")
    return "\n".join(lines)


def add_subparser(sub):
    """Registers `clozn ci` (with its `baseline`/`check` sub-subcommands) on an argparse subparsers
    object -- same pattern as commands.diff_model/quant_check.add_subparser: this module's own tests can
    build a throwaway parser and exercise argparse wiring without touching main.py, and it documents the
    exact main.py registration edit as real, testable code."""
    pci = sub.add_parser("ci", help="headless CI gate (Phase-1 §4.4): a deterministic pass/fail over the "
                         "EXISTING test-model/tiny-test/diff-model primitives, budgeted against a "
                         "recorded baseline")
    ci_sub = pci.add_subparsers(dest="ci_cmd")
    pci.set_defaults(fn=_cmd_ci_no_subcommand)

    pb = ci_sub.add_parser("baseline", help="measure the v0 checks (golden/tiny/[diff]) against a model "
                          "ONCE and freeze the budgets a later `clozn ci check` enforces")
    pb.add_argument("out", help="path to write the baseline JSON artifact to")
    pb.add_argument("model", help="the model to measure (known short name / local GGUF path / fuzzy "
                    "fragment, resolved like `clozn run`'s model arg)")
    pb.add_argument("--url", default=_DEFAULT_URL, help="Clozn gateway base URL for the golden "
                    "(test-model) check (default :8080)")
    pb.add_argument("--set", dest="which", default="all",
                    choices=["easy", "hard", "arith", "extended", "both", "all"],
                    help="test-model probe set to measure (default: all)")
    pb.add_argument("--no-golden", action="store_true", help="skip the golden (test-model) check entirely")
    pb.add_argument("--min-pass-rate", type=float, default=None,
                    help="golden budget override (default: THIS run's own measured pass rate)")
    pb.add_argument("--tiny", action="append", default=[], metavar="FILE",
                    help="a tiny-test spec file (`clozn test`'s JSON format) to include in the gate -- "
                         "repeatable; every test passing NOW becomes a must-still-pass budget")
    pb.add_argument("--reference", default=None, help="a reference GGUF for the diff-model check "
                    "(optional -- omit to skip it entirely). LIVE: boots two engines, needs a free GPU")
    pb.add_argument("--diff-runs", type=int, default=8, dest="diff_runs",
                    help="--runs passed to the diff-model ladder for the diff check (default 8)")
    pb.add_argument("--max-argmax-flips-total", type=int, default=None,
                    help="diff budget override (default: THIS run's own measured flip count)")
    pb.add_argument("--max-mean-abs-delta-nats", type=float, default=None,
                    help="diff budget override (default: THIS run's own measured mean |delta_nats|)")
    pb.add_argument("--pin-model", action="store_true",
                    help="record this exact model's sha256 as a hard requirement -- a later `clozn ci "
                         "check` against a DIFFERENT model's sha refuses (exit 3) unless "
                         "--allow-model-change is passed. Default off: the common case is gating a NEW "
                         "model against an OLD baseline")
    pb.add_argument("--cpu", action="store_true", help="force the CPU build for the diff check's two engines")
    pb.set_defaults(fn=cmd_ci_baseline)

    pc = ci_sub.add_parser("check", help="run the checks a baseline declares against a model and gate on "
                          "the recorded budgets -- exit 0 pass / 1 budget violated / 2 execution error / "
                          "3 identity-policy refusal")
    pc.add_argument("--baseline", required=True, metavar="FILE",
                    help="baseline JSON written by `clozn ci baseline`")
    pc.add_argument("model", help="the model under test (resolved like `clozn run`'s model arg)")
    pc.add_argument("--url", default=_DEFAULT_URL, help="Clozn gateway base URL for the golden check "
                    "(default :8080)")
    pc.add_argument("--allow-model-change", action="store_true", dest="allow_model_change",
                    help="proceed even if the baseline is pinned (pin_model: true) and this model's "
                         "sha256 doesn't match it")
    pc.add_argument("--report", default=None, metavar="OUT.json",
                    help="also write the machine-readable report to this path")
    pc.add_argument("--json", action="store_true", help="print the report as JSON instead of the text summary")
    pc.add_argument("--cpu", action="store_true", help="force the CPU build for the diff check's two engines")
    pc.set_defaults(fn=cmd_ci_check)

    return pci


def _cmd_ci_no_subcommand(_args):
    print("clozn ci: use `clozn ci baseline <out.json> <model>` or "
          "`clozn ci check --baseline <file> <model>`")
    return 2


def cmd_ci_baseline(args):
    """`clozn ci baseline <out.json> <model> [options]` -- see add_subparser for the full flag list and
    the module docstring for the exact schema written. Always exits 0 (a baseline MEASURES; it never
    gates), unless the model can't be resolved at all (a typed `CloznError`, which `clozn.cli.main`
    prints and maps to exit 1 -- consistent with every other `resolve_model` caller in this codebase)."""
    from clozn._io import atomic_write_json
    from clozn.cli.commands.models import resolve_model

    model_path = resolve_model(args.model)
    tiny_files = [os.path.abspath(p) for p in (args.tiny or [])]

    baseline = build_baseline(
        model_path=model_path, url=args.url, which=args.which, golden_enabled=not args.no_golden,
        min_pass_rate=args.min_pass_rate, tiny_files=tiny_files, reference=args.reference,
        diff_runs=args.diff_runs, cpu=args.cpu, max_argmax_flips_total=args.max_argmax_flips_total,
        max_mean_abs_delta_nats=args.max_mean_abs_delta_nats, pin_model=args.pin_model,
    )
    atomic_write_json(args.out, baseline, indent=2)

    print(f"clozn ci baseline: wrote {args.out}")
    g = baseline["checks"]["golden"]
    if g.get("enabled"):
        print(f"  golden: n={g['measured']['n']}  pass_rate={g['measured']['pass_rate']}  "
              f"budget min_pass_rate={g['min_pass_rate']}")
    t = baseline["checks"]["tiny"]
    if t.get("enabled"):
        total_passing = sum(len(f.get("baseline_passing_tests") or []) for f in t["files"])
        print(f"  tiny: {len(t['files'])} file(s), {total_passing} baseline-passing test(s) to hold")
    d = baseline["checks"]["diff"]
    if d.get("enabled"):
        print(f"  diff: reference={d['reference']}  "
              f"budget max_argmax_flips_total={d['max_argmax_flips_total']}  "
              f"max_mean_abs_delta_nats={d['max_mean_abs_delta_nats']}")
    return 0


def cmd_ci_check(args):
    """`clozn ci check --baseline <file> <model> [options]` -- see add_subparser for the full flag list
    and the module docstring for the exit-code contract. Deliberately handles its OWN error paths (rather
    than letting `resolve_model`'s `CloznError` or a bad baseline file bubble up to `clozn.cli.main`,
    which only knows how to turn any `CloznError` into exit 1) so all four documented exit codes are
    actually reachable."""
    from clozn._io import atomic_write_json
    from clozn.cli.commands.models import resolve_model

    try:
        with open(args.baseline, encoding="utf-8") as f:
            baseline = json.load(f)
    except OSError as e:
        print(f"{fmt.BOLD}clozn ci check:{fmt.RST} could not read baseline {args.baseline!r}: {e}",
              file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"{fmt.BOLD}clozn ci check:{fmt.RST} baseline {args.baseline!r} is not valid JSON: {e}",
              file=sys.stderr)
        return 2
    if not isinstance(baseline, dict) or "checks" not in baseline:
        print(f"{fmt.BOLD}clozn ci check:{fmt.RST} {args.baseline!r} is not a `clozn ci baseline` "
              "artifact (missing 'checks')", file=sys.stderr)
        return 2

    try:
        model_path = resolve_model(args.model)
    except Exception as e:
        print(f"{fmt.BOLD}clozn ci check:{fmt.RST} could not resolve model {args.model!r}: {e}",
              file=sys.stderr)
        return 2

    try:
        report = run_gate(baseline=baseline, model_path=model_path, url=args.url,
                          allow_model_change=args.allow_model_change, cpu=args.cpu)
    except CIIdentityRefusal as e:
        print(f"{fmt.BOLD}clozn ci check: REFUSED{fmt.RST} -- {e}", file=sys.stderr)
        if args.report:
            atomic_write_json(args.report, {"refused": True, "reason": str(e)}, indent=2)
        return 3
    except Exception as e:
        print(f"{fmt.BOLD}clozn ci check:{fmt.RST} execution error: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2

    exit_code = 0 if report["overall"] == "pass" else 1
    report["baseline_path"] = args.baseline
    report["model"] = model_path
    report["exit_code"] = exit_code

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(format_ci_report(report))
    if args.report:
        atomic_write_json(args.report, report, indent=2, default=str)
    return exit_code
