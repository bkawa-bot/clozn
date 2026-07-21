"""commands.eval -- `clozn eval [--set easy|hard|arith|both|all|extended]`: outcome-grounded calibration on
a live endpoint.

The thin CLI shell around clozn.eval.bench: run the built-in factual probe set through a RUNNING Clozn
gateway, grade each answer against gold (eval.outcome), read each reply's answer-span confidence from its
logged run trace, and print Brier / ECE-vs-truth / a risk-coverage curve for selective generation. This
is the TRUTH tier actuary.py flags as missing -- calibration against correctness, not the acceptance proxy.

Needs a running Clozn gateway (default http://127.0.0.1:8080); it reads per-token confidence from the run
journal, which the OpenAI wire format omits. Small built-in n -- a directional demonstration that the
model's own confidence separates its right answers from its wrong ones, not a benchmark score.

Registration in clozn/cli/main.py mirrors quant-check exactly: import `cmd_eval, add_subparser as
_add_eval` alongside the other commands.* imports, and call `_add_eval(sub)` in build_parser() before
`return p`.
"""
from __future__ import annotations

import json
import math
import unicodedata


_PROBE_SETS = ("easy", "hard", "arith", "both", "all", "extended")
_SCORE_AGGREGATES = ("min", "mean")


def add_subparser(sub):
    """Register `clozn eval` on an argparse subparsers object (own function so its wiring is testable
    without dispatching; mirrors commands.quant_check.add_subparser)."""
    pe = sub.add_parser("eval", help="outcome-grounded calibration: run a factual probe set through a live "
                        "gateway and score Brier/ECE/risk-coverage against TRUTH (needs a running endpoint)")
    pe.add_argument("--url", default="http://127.0.0.1:8080", help="Clozn gateway base URL (default :8080)")
    pe.add_argument("--set", dest="which", default="arith",
                    choices=_PROBE_SETS,
                    help="which built-in probe set (default: arith -- programmatic, guaranteed golds, "
                         "graded errors; 'easy'/'hard' are curated factual sets; 'extended' is the v2 set "
                         "-- reasoning puzzles, common misconceptions, and trick questions; 'all' folds "
                         "every set together)")
    pe.add_argument("--score", default="min", choices=["min", "mean"],
                    help="answer-span aggregate used as the abstention signal (default: min token conf)")
    pe.add_argument("--target-error", type=float, default=0.05, dest="target_error",
                    help="the selective-generation error budget the recommended policy is tuned to (0.05)")
    pe.add_argument("--json", action="store_true", help="print the raw calibration report as JSON")
    pe.add_argument("--save", action="store_true", help="persist this report so the studio can serve it "
                    "as the TRUTH-tier curve at GET /journal/calibration (beside the proxy at /journal/actuary)")
    pe.add_argument("--task", default="general",
                    help="task label for a model/task-specific calibration profile (default: general)")
    pe.add_argument("--wizard", action="store_true",
                    help="guided calibration: confirm task, probes, score aggregate, error budget, and saving")
    pe.add_argument("--list-profiles", action="store_true",
                    help="list installed model/task calibration profiles without running live probes")
    pe.set_defaults(fn=cmd_eval)
    return pe


def _task_label(value) -> str:
    raw = str(value or "")
    if any(unicodedata.category(char) == "Cc" for char in raw):
        raise ValueError("task must be a non-empty label of at most 80 characters without control characters")
    task = " ".join(raw.strip().split()).lower()
    if not task or len(task) > 80:
        raise ValueError("task must be a non-empty label of at most 80 characters without control characters")
    return task


def _target_error(value) -> float:
    try:
        target = float(value)
    except (TypeError, ValueError):
        raise ValueError("target error must be a number from 0 to 1") from None
    if not math.isfinite(target) or target < 0 or target > 1:
        raise ValueError("target error must be a finite number from 0 to 1")
    return target


def _read_answer(prompt: str) -> str | None:
    """One wizard read. Redirected/closed stdin means "accept safe defaults", never a crash."""
    try:
        return input(prompt).strip()
    except (EOFError, OSError, StopIteration):
        return None


def _choose(prompt: str, default, parse, choices: tuple[str, ...] | None = None):
    """Prompt at most twice, then retain the already-validated/default value."""
    for _ in range(2):
        raw = _read_answer(prompt)
        if raw is None or raw == "":
            return default
        try:
            value = parse(raw)
            if choices is not None and value not in choices:
                raise ValueError(f"choose one of {', '.join(choices)}")
            return value
        except ValueError as exc:
            print(f"  {exc}; press Enter to keep {default!s}")
    return default


def _choose_save(default: bool) -> bool:
    raw = _read_answer(f"Save this model/task profile? [{'Y/n' if default else 'y/N'}]: ")
    if raw is None or raw == "":
        return default
    answer = raw.lower()
    if answer in {"y", "yes"}:
        return True
    if answer in {"n", "no"}:
        return False
    print(f"  unrecognized answer; using {'yes' if default else 'no'}")
    return default


def _wizard(args) -> tuple[str, str, str, float, bool]:
    task = _choose(f"Task label [{args.task}]: ", args.task, _task_label)
    which = _choose(f"Probe set ({'/'.join(_PROBE_SETS)}) [{args.which}]: ",
                    args.which, str, _PROBE_SETS)
    score = _choose(f"Token score aggregate ({'/'.join(_SCORE_AGGREGATES)}) [{args.score}]: ",
                    args.score, str, _SCORE_AGGREGATES)
    target = _choose(f"Target answered-error [{args.target_error}]: ",
                     args.target_error, _target_error)
    save = _choose_save(bool(getattr(args, "save", False)))
    # Validate defaults too: EOF must never let a malformed --task/--target-error reach model work.
    task, target = _task_label(task), _target_error(target)
    print(f"\nCalibration plan: task={task}  set={which}  score={score}  "
          f"target_error={target:g}  save={'yes' if save else 'no'}")
    return task, which, score, target, save


def _saved_paths(saved) -> str:
    if isinstance(saved, dict):
        profile = (saved.get("profile_path") or saved.get("path") or saved.get("profile")
                   or saved.get("task_path"))
        active = saved.get("active_path") or saved.get("legacy_path") or saved.get("active")
        if profile and active and str(profile) != str(active):
            return f"{profile}  (active report -> {active})"
        if profile or active:
            return f"{profile or active}  (active report updated)"
        if saved.get("model") and saved.get("task"):
            return f"{saved['model']} / {saved['task']}  (active report updated)"
        return "model/task profile saved (active report updated)"
    return f"{saved}  (active report updated)"


def _print_wizard_scope(out: dict, task: str, which: str, score: str,
                        target_error: float, recommendation: dict) -> None:
    summary = recommendation.get("summary") or {}
    model = out.get("model") or "unavailable"
    print(f"\nCalibration scope: {out.get('n', 0)} gradeable sample(s), "
          f"{out.get('unmatched', 0)} unmatched; model={model}  task={task}  probes={which}.")
    print(f"  Policy tradeoff @ target_error={target_error:g}: {score}-token aggregate; "
          f"answer={summary.get('n_answer', 0)}, ask={summary.get('n_ask', 0)}, "
          f"abstain={summary.get('n_abstain', 0)}, correct_withheld={summary.get('correct_withheld', 0)}.")
    print("  Distribution limit: this built-in probe mix may not represent the task's live inputs; "
          "treat a small sample as directional evidence.")
    print("  Caveat: confidence comes from recorded token probabilities; it is not a live fact-check "
          "and does not verify a new answer's claims.")


def _list_saved_profiles(*, json_output: bool = False) -> int:
    from clozn.eval import store as eval_store

    profiles = eval_store.list_profiles()
    if json_output:
        print(json.dumps({"profiles": profiles}, indent=2, default=str))
        return 0
    if not profiles:
        print("No model/task calibration profiles installed. Run `clozn eval --wizard` to fit one.")
        return 0
    print("Installed model/task calibration profiles (newest first):")
    for profile in profiles:
        policy_summary = profile.get("policy") or {}
        print(f"  {profile.get('model') or '?'}  task={profile.get('task') or '?'}  "
              f"probes={profile.get('set') or '?'}  n={profile.get('n', '?')}  "
              f"score={profile.get('score') or '?'}  answer_at={policy_summary.get('answer_at', '?')}  "
              f"ask_at={policy_summary.get('ask_at', '—')}")
    print("Thresholds are outcome-grounded for the named model/task distribution; they are not a live fact-check.")
    return 0


def cmd_eval(args):
    """`clozn eval [--set ...] [--score ...] [--json]` -- LIVE: needs a running Clozn gateway (see module
    docstring). Delegates to clozn.eval.bench; prints the human report, or the raw report JSON with --json."""
    if getattr(args, "list_profiles", False):
        return _list_saved_profiles(json_output=bool(getattr(args, "json", False)))
    try:
        if getattr(args, "wizard", False):
            task, which, score, te, save = _wizard(args)
        else:
            task, which, score = _task_label(getattr(args, "task", "general")), args.which, args.score
            te, save = _target_error(getattr(args, "target_error", 0.05)), bool(getattr(args, "save", False))
    except ValueError as exc:
        print(f"clozn eval: {exc}")
        return 2

    from clozn.eval import bench
    from clozn.eval import policy

    out = bench.bench(args.url, which, score)
    rec = policy.recommend(out.get("pairs", []), target_error=te)
    if getattr(args, "json", False):
        print(json.dumps({"model": out.get("model"), "task": task, "set": which, "score": score,
                          "target_error": te, "n": out["n"], "unmatched": out["unmatched"],
                          "report": out["report"], "policy": rec, "rows": out["rows"]},
                         indent=2, default=str))
    else:
        bench._print(out, which, score, te)
    if getattr(args, "wizard", False):
        _print_wizard_scope(out, task, which, score, te, rec)
    if save:
        from clozn.eval import store as eval_store
        if not isinstance(out.get("model"), str) or not out["model"].strip():
            print("\n  calibration was not saved: no active model identity was captured by the probe run")
            return 1
        payload = {"set": which, "score": score, "target_error": te, "model": out.get("model"),
                   "task": task,
                   "n": out["n"], "unmatched": out["unmatched"], "report": out["report"],
                   "policy": rec, "rows": out["rows"]}
        try:
            saved = eval_store.save_profile(payload, task=task)
        except (OSError, ValueError) as exc:
            print(f"\n  calibration was not saved: {exc}")
            return 1
        print(f"\n  saved TRUTH-tier profile -> {_saved_paths(saved)}"
              "  (served at GET /journal/calibration)")
    return 0
