"""python -m clozn.eval.bench [--url URL] [--set easy|hard|both] [--score mean|min]

Run the factual probe set through a LIVE clozn endpoint and print outcome-grounded calibration (Brier /
ECE-vs-truth / risk-coverage for selective generation). This is the reproducible version of the harness in
eval.calibration: it needs a running clozn studio because the per-item score is the model's answer-span
confidence, read from each probe's logged run trace (the OpenAI wire format omits per-token probabilities).

Honesty: the built-in sets are SMALL (tens of items). Treat the numbers as a directional demonstration
that the model's own confidence does (or doesn't) separate its right answers from its wrong ones -- not a
benchmark score. Wide error bars; report n alongside every figure.
"""
from __future__ import annotations

import argparse

from clozn.eval import calibration as cal, outcome, probes
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
    pset = {"easy": probes.PROBES, "hard": probes.HARD_PROBES,
            "both": probes.PROBES + probes.HARD_PROBES}[which]
    results = probes.run_probes(base_url, pset)
    by_q = _newest_runs_by_prompt()
    rows, pairs, unmatched = [], [], 0
    for p in results:
        run = by_q.get(p["q"].strip())
        if not run:
            unmatched += 1
            continue
        reply = run.get("response") or ""
        correct = outcome.grade(reply, p["gold"], p["kind"], aliases=p.get("aliases", []))
        confs = _answer_confidences(run.get("trace") or {})
        if correct is None or not confs:
            continue
        s = (min(confs) if score == "min" else sum(confs) / len(confs))
        pairs.append((s, correct))
        rows.append({"q": p["q"], "reply": reply, "gold": p["gold"], "correct": bool(correct), "score": round(s, 3)})
    return {"rows": rows, "report": cal.report(pairs), "n": len(pairs), "unmatched": unmatched}


def _print(out: dict, which: str, score: str) -> None:
    rep = out["report"]
    print(f"\nclozn.eval.bench — set={which}  score={score}-token confidence  n={out['n']}"
          f"  (unmatched={out['unmatched']})")
    if not rep.get("available"):
        print("  no gradeable items (is the studio running and logging traces?)")
        return
    for r in sorted(out["rows"], key=lambda r: r["score"]):     # least-confident first — the abstain tail
        tag = "OK  " if r["correct"] else "MISS"
        print(f"  {r['score']:.2f} {tag}  {r['reply'][:22]:22} (gold {r['gold'][:18]})  {r['q'][:44]}")
    print(f"\n  base_error={rep['base_error']}  brier={rep['brier']}  ece={rep['ece']}  aurc={rep['aurc']}")
    for cov in (50, 70, 90):
        s = rep["selective"][cov]
        red = s["error_reduction_vs_full"]
        print(f"  selective @{cov:>2}% coverage: error={s['error_at_coverage']:.3f}"
              f"  (full={s['full_coverage_error']:.3f}, reduction={red})")
    print("  note: small n — a directional demonstration, not a benchmark score.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Outcome-grounded calibration on a live clozn endpoint.")
    ap.add_argument("--url", default="http://127.0.0.1:8090", help="clozn studio base URL")
    ap.add_argument("--set", dest="which", default="hard", choices=["easy", "hard", "both"])
    ap.add_argument("--score", default="min", choices=["min", "mean"])
    args = ap.parse_args(argv)
    out = bench(args.url, args.which, args.score)
    _print(out, args.which, args.score)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
