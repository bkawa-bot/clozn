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


def add_subparser(sub):
    """Register `clozn eval` on an argparse subparsers object (own function so its wiring is testable
    without dispatching; mirrors commands.quant_check.add_subparser)."""
    pe = sub.add_parser("eval", help="outcome-grounded calibration: run a factual probe set through a live "
                        "gateway and score Brier/ECE/risk-coverage against TRUTH (needs a running endpoint)")
    pe.add_argument("--url", default="http://127.0.0.1:8080", help="Clozn gateway base URL (default :8080)")
    pe.add_argument("--set", dest="which", default="arith",
                    choices=["easy", "hard", "arith", "both", "all", "extended"],
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
    pe.set_defaults(fn=cmd_eval)
    return pe


def cmd_eval(args):
    """`clozn eval [--set ...] [--score ...] [--json]` -- LIVE: needs a running Clozn gateway (see module
    docstring). Delegates to clozn.eval.bench; prints the human report, or the raw report JSON with --json."""
    from clozn.eval import bench

    from clozn.eval import policy

    out = bench.bench(args.url, args.which, args.score)
    te = getattr(args, "target_error", 0.05)
    rec = policy.recommend(out.get("pairs", []), target_error=te)
    if getattr(args, "json", False):
        print(json.dumps({"n": out["n"], "unmatched": out["unmatched"], "report": out["report"],
                          "policy": rec, "rows": out["rows"]}, indent=2, default=str))
    else:
        bench._print(out, args.which, args.score, te)
    if getattr(args, "save", False):
        from clozn.eval import store as eval_store
        payload = {"set": args.which, "score": args.score, "target_error": te, "model": out.get("model"),
                   "n": out["n"], "unmatched": out["unmatched"], "report": out["report"],
                   "policy": rec, "rows": out["rows"]}
        path = eval_store.save(payload)
        print(f"\n  saved TRUTH-tier report -> {path}  (served at GET /journal/calibration)")
    return 0
