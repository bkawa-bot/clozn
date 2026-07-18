"""python -m clozn.eval.bench [--url URL] [--set easy|hard|arith|both|all|extended] [--score mean|min]

Run the factual probe set through a LIVE clozn endpoint and print outcome-grounded calibration (Brier /
ECE-vs-truth / risk-coverage for selective generation). This is the reproducible version of the harness in
eval.calibration: it needs a running Clozn gateway because the per-item score is the model's answer-span
confidence, read from each probe's logged run trace (the OpenAI wire format omits per-token probabilities).

Honesty: the built-in sets are SMALL (tens of items). Treat the numbers as a directional demonstration
that the model's own confidence does (or doesn't) separate its right answers from its wrong ones -- not a
benchmark score. Wide error bars; report n alongside every figure.
"""
from __future__ import annotations

import argparse

from clozn.eval import calibration as cal, outcome, policy, probes
from clozn.runs import actuary


def _answer_confidences(trace: dict) -> list[float]:
    """Per-token confidence over the reply's CONTENT tokens (empty/whitespace pieces dropped -- they're
    structural, not answer content)."""
    toks = trace.get("tokens") or []
    conf = trace.get("confidence") or []
    return [conf[i] for i in range(min(len(toks), len(conf))) if (toks[i] or "").strip()]


def _newest_runs_by_prompt() -> dict:
    """Map prompt_summary -> its newest organic run (the probe we just issued)."""
    by_q: dict = {}
    for r in actuary.organic(actuary.load_runs()):
        q = (r.get("prompt_summary") or "").strip()
        if q and (q not in by_q or (r.get("created_ts") or 0) > (by_q[q].get("created_ts") or 0)):
            by_q[q] = r
    return by_q


def bench(base_url: str, which: str = "hard", score: str = "min") -> dict:
    """Run the chosen probe set live, pair each reply to its logged run trace, and return
    {rows, report, n, unmatched}. `score` picks the answer-span aggregate: 'min' (weakest token -- the
    standard selective-generation signal) or 'mean'."""
    pset = {"easy": probes.PROBES, "hard": probes.HARD_PROBES, "arith": probes.ARITH_PROBES,
            "both": probes.PROBES + probes.HARD_PROBES,
            "extended": probes.EXTENDED_PROBES,
            "all": probes.PROBES + probes.HARD_PROBES + probes.ARITH_PROBES + probes.EXTENDED_PROBES}[which]
    results = probes.run_probes(base_url, pset)
    by_q = _newest_runs_by_prompt()
    rows, pairs, unmatched, model = [], [], 0, None
    for p in results:
        run = by_q.get(p["q"].strip())
        if not run:
            unmatched += 1
            continue
        model = model or run.get("model")           # provenance: which model these numbers describe
        reply = run.get("response") or ""
        correct = outcome.grade(reply, p["gold"], p["kind"], aliases=p.get("aliases", []))
        confs = _answer_confidences(run.get("trace") or {})
        if correct is None or not confs:
            continue
        s = (min(confs) if score == "min" else sum(confs) / len(confs))
        pairs.append((s, correct))
        rows.append({"q": p["q"], "reply": reply, "gold": p["gold"], "correct": bool(correct), "score": round(s, 3)})
    return {"rows": rows, "report": cal.report(pairs), "pairs": pairs, "n": len(pairs),
            "unmatched": unmatched, "model": model}


def _print(out: dict, which: str, score: str, target_error: float = 0.05) -> None:
    rep = out["report"]
    print(f"\nclozn.eval.bench — set={which}  score={score}-token confidence  n={out['n']}"
          f"  (unmatched={out['unmatched']})")
    if not rep.get("available"):
        print("  no gradeable items (is the Clozn gateway running and logging traces?)")
        return
    for r in sorted(out["rows"], key=lambda r: r["score"]):     # least-confident first — the abstain tail
        tag = "OK  " if r["correct"] else "MISS"
        print(f"  {r['score']:.2f} {tag}  {r['reply'][:22]:22} (gold {r['gold'][:18]})  {r['q'][:44]}")
    print(f"\n  base_error={rep['base_error']}  brier={rep['brier']}  ece={rep['ece']}  aurc={rep['aurc']}")
    temp = rep.get("temperature_scaling") or {}
    if temp.get("available"):
        print(f"  temperature T={temp['temperature']:.3f}  nll {temp['nll_before']:.3f} -> {temp['nll_after']:.3f}"
              f"  ece {temp['ece_before']:.3f} -> {temp['ece_after']:.3f}")
    else:
        print(f"  temperature unavailable: {temp.get('reason', 'no fitted transform')}")
    for cov in (50, 70, 90):
        s = rep["selective"][cov]
        red = s["error_reduction_vs_full"]
        print(f"  selective @{cov:>2}% coverage: error={s['error_at_coverage']:.3f}"
              f"  (full={s['full_coverage_error']:.3f}, reduction={red})")
    rec = policy.recommend(out.get("pairs", []), target_error=target_error)
    ps = rec["summary"]
    print(f"\n  policy @ target_error={target_error}:  answer_at={rec['answer_at']}  ask_at={rec['ask_at']}"
          f"  (achievable={rec['achievable']})")
    print(f"    answer {ps['n_answer']} / ask {ps['n_ask']} / abstain {ps['n_abstain']}"
          f"   coverage={ps['coverage']}  answered_error={ps['answered_error']}")
    print(f"    caught {ps['errors_caught']} wrong answers; withheld {ps['correct_withheld']} correct (the cost)")
    print("  note: small n — a directional demonstration, not a benchmark score.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Outcome-grounded calibration on a live clozn endpoint.")
    ap.add_argument("--url", default="http://127.0.0.1:8080", help="Clozn gateway base URL")
    ap.add_argument("--set", dest="which", default="arith",
                    choices=["easy", "hard", "arith", "both", "all", "extended"])
    ap.add_argument("--score", default="min", choices=["min", "mean"])
    ap.add_argument("--target-error", type=float, default=0.05, dest="target_error",
                    help="the selective-generation error budget the policy is tuned to (default 0.05)")
    args = ap.parse_args(argv)
    out = bench(args.url, args.which, args.score)
    _print(out, args.which, args.score, args.target_error)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
