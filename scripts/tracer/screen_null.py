"""screen_null.py -- the genuine screen-null the tracer's remaining-list has owed since 2026-07-20.

The concern: the S0 screen nominates sites, S1 confirms them. But does the WHOLE pipeline find
structure that is SPECIFIC to the token being traced, or would it manufacture a comparable
"circuit" for any plausible token? The earlier null (junk extra concepts) tested nothing --
diluting a target with unrelated concepts never displaced the target's own signal. This is the
decisive version: trace the token the model ACTUALLY produced (should PASS with real nodes) vs a
plausible-but-WRONG token the model did NOT produce (Paris vs Berlin for "capital of France is").

If the tracer is honest, the null token -- one the model assigns low probability and is not
computing toward -- should FAIL to find a clean circuit: NO_CAUSAL_NODES, or FAILED_CONTROLS, or
at minimum far fewer strong nodes than the true token, and a largely DIFFERENT node set. If it
returns a comparable PASS with comparable strong nodes for a token the model never output, the
screen (and the trace) is finding generic activity, not the computation behind the answer -- and
that would invalidate the screen.

Runs on ANY GGUF via the ablate screen (no sidecar). Writes runs/experiments/screen_null_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from clozn.analysis import tracer  # noqa: E402


def post(engine, path, body, timeout=600):
    req = urllib.request.Request(engine.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


# (prompt, the WRONG-but-plausible null continuation). The true continuation is the model's own
# greedy answer, discovered at runtime -- we never hand the tracer the token we want.
CASES = [
    ("The capital of France is", " Berlin"),
    ("The capital of France is", " London"),
    ("The largest planet in our solar system is", " Mars"),
    ("Water freezes at a temperature of 0 degrees", " Fahrenheit"),
    ("The chemical symbol for gold is", " Ag"),        # Ag is silver -- plausible, wrong
    ("The opposite of hot is", " warm"),               # plausible, not the greedy answer
    ("Two plus two equals", " five"),
    ("The first President of the United States was George", " Bush"),
]


def node_set(r):
    return {(n["layer"], n["pos"]) for n in r.get("nodes", [])}


def strong_count(r):
    return sum(1 for n in r.get("nodes", []) if n.get("strength") == "strong")


def trace_token(engine, prompt, continuation):
    """Trace continuation-token 0 of `continuation` on this prompt (ablate screen)."""
    r = tracer.trace(prompt, continuation, 0, engine_url=engine,
                     budget=tracer.TraceBudget(max_candidates=12, max_edges=3,
                                               ablate_screen_arms=48),
                     seed=0, screen_mode="ablate")
    return r


def run(engine, tag):
    results = []
    for prompt, null_cont in CASES:
        # the token the model ACTUALLY produces
        gen = post(engine, "/v1/completions", {"prompt": prompt, "max_tokens": 2, "temperature": 0})
        true_cont = gen["choices"][0]["text"]
        # baseline probabilities of both, so we can report how wrong the null actually is
        b_true = post(engine, "/score", {"prompt": prompt, "continuation": true_cont, "topk": 0})
        b_null = post(engine, "/score", {"prompt": prompt, "continuation": null_cont, "topk": 0})
        lp_true = float(b_true["tokens"][0]["logprob"])
        lp_null = float(b_null["tokens"][0]["logprob"])

        rt = trace_token(engine, prompt, true_cont)
        rn = trace_token(engine, prompt, null_cont)
        if not rt.get("ok") or not rn.get("ok"):
            results.append({"prompt": prompt, "true": true_cont, "null": null_cont,
                            "blocked": rt.get("blocked") or rn.get("blocked")})
            continue
        st, sn = node_set(rt), node_set(rn)
        overlap = len(st & sn) / max(1, len(st | sn))
        row = {
            "prompt": prompt, "true_token": true_cont.strip(), "null_token": null_cont.strip(),
            "true_logprob": round(lp_true, 3), "null_logprob": round(lp_null, 3),
            "true_verdict": rt["controls"]["verdict"], "null_verdict": rn["controls"]["verdict"],
            "true_strong": strong_count(rt), "null_strong": strong_count(rn),
            "true_nodes": len(rt["nodes"]), "null_nodes": len(rn["nodes"]),
            "node_jaccard": round(overlap, 3),
        }
        results.append(row)
        print(f"{prompt[:34]!r:<36} true={row['true_token']!r}({lp_true:.1f}) "
              f"null={row['null_token']!r}({lp_null:.1f}) | "
              f"verdict {row['true_verdict']}/{row['null_verdict']} | "
              f"strong {row['true_strong']}/{row['null_strong']} | jaccard {overlap:.2f}")

    graded = [r for r in results if "true_verdict" in r]
    # The screen DISCRIMINATES on a case when the null is meaningfully weaker: fewer strong nodes
    # OR a non-PASS verdict OR low node overlap. Report the aggregate honestly.
    true_strong = [r["true_strong"] for r in graded]
    null_strong = [r["null_strong"] for r in graded]
    null_nonpass = sum(1 for r in graded if r["null_verdict"] != "PASS")
    weaker = sum(1 for r in graded if r["null_strong"] < r["true_strong"])
    summary = {
        "tag": tag, "n": len(graded),
        "mean_true_strong": round(sum(true_strong) / len(true_strong), 2) if graded else None,
        "mean_null_strong": round(sum(null_strong) / len(null_strong), 2) if graded else None,
        "null_nonpass_verdicts": f"{null_nonpass}/{len(graded)}",
        "null_weaker_than_true": f"{weaker}/{len(graded)}",
        "mean_node_jaccard": round(sum(r["node_jaccard"] for r in graded) / len(graded), 3) if graded else None,
        "reading": ("screen DISCRIMINATES: null tokens yield weaker/different structure"
                    if graded and (weaker + null_nonpass) >= len(graded)
                    else "MIXED/FAILS: null tokens produce comparable structure -- screen may be "
                         "finding generic activity, not answer-specific computation"),
    }
    out = os.path.join(REPO, "runs", "experiments", f"screen_null_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=2)
    print("\n=== screen-null ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"wrote {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://127.0.0.1:8091")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    run(args.engine, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
