"""Circuit-tracer validation battery: many prompts, several categories, aggregate scorecard.

Design choices that keep this honest:
  * The continuation is whatever the model ACTUALLY generates greedily (we don't hand it the
    answer we want) -- so every trace targets a token the model really produced.
  * Categories deliberately include cases we expect to be hard or to have NO clean circuit
    (distributed reasoning, arithmetic, low-margin/uncertain). A battery where everything passes
    is a broken battery.
  * The headline metric is the S4 predicted-vs-observed scorecard SUMMED over prompts: the graph
    published per-node flip predictions from (delta_full, margin) before any generation ran.
"""
import json
import sys
import time
import urllib.request

sys.path.insert(0, r"C:\Users\brigi\src\clozn")
from clozn.analysis import tracer  # noqa: E402

ENGINE = "http://127.0.0.1:8080"
JL = r"C:\Users\brigi\.clozn\artifacts\jlens\qwen3.5-9b-v1"

# (category, prompt, extra concepts, which continuation token index to trace)
CASES = [
    ("factual_easy", "The capital of France is", "France,capital", 0),
    ("factual_easy", "The largest planet in our solar system is", "planet,Jupiter", 0),
    ("factual_easy", "Water freezes at a temperature of", "water,ice", 0),
    ("factual_distractor",
     "Kyoto was the capital of Japan for over a thousand years and remains its cultural heart. "
     "However, since the Meiji era the government and the Imperial Palace have been located "
     "elsewhere. The modern capital of Japan is", "Japan,Kyoto", 0),
    ("factual_distractor",
     "Sydney is the largest city in Australia and the one most tourists visit first. "
     "The seat of the federal parliament, however, is in the purpose-built city of", "Australia,Sydney", 0),
    ("factual_distractor",
     "Istanbul is Turkey's biggest city and its economic centre. The capital, however, is", "Turkey,Istanbul", 0),
    ("in_context",
     "In this story, the wizard's name is Zorblax. The knight's name is Pellinore. "
     "When the dragon attacked, the wizard cast a spell. The name of the wizard is", "wizard,Zorblax", 0),
    ("in_context",
     "Alice put the key in the blue box. Bob put the coin in the red box. "
     "Later, Alice went to retrieve her key, opening the box coloured", "key,blue", 0),
    ("syntactic",
     "The keys to the cabinet in the hallway upstairs", "keys,are", 0),
    ("syntactic",
     "Neither the manager nor the employees who work the night shift", "employees,were", 0),
    ("arithmetic", "Compute step by step. 17 plus 26 equals", "sum,addition", 0),
    ("arithmetic", "The product of 12 and 12 is", "product,multiplication", 0),
    ("later_position", "The capital of France is Paris, which is located on the river", "Seine,river", 0),
    ("later_position", "The chemical symbol for gold is", "gold,symbol", 1),
    ("low_margin", "The best programming language for beginners is probably", "programming,language", 0),
    ("low_margin", "My favourite colour is", "colour", 0),
]


def post(path, body, timeout=300):
    req = urllib.request.Request(ENGINE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def greedy_continuation(prompt, n=3):
    """What the model ACTUALLY says next (greedy), as text."""
    r = post("/v1/completions", {"prompt": prompt, "max_tokens": n, "temperature": 0})
    return r["choices"][0]["text"]


results = []
t_start = time.time()
for i, (cat, prompt, concepts, tgt_idx) in enumerate(CASES, 1):
    cont = greedy_continuation(prompt, n=max(2, tgt_idx + 2))
    if not cont.strip():
        print(f"[{i:2}/{len(CASES)}] {cat}: EMPTY continuation, skipped")
        continue
    t0 = time.time()
    r = tracer.trace(prompt, cont, tgt_idx, engine_url=ENGINE, jlens_dir=JL,
                     budget=tracer.TraceBudget(max_candidates=20,
                                               extra_concepts=[c for c in concepts.split(",") if c]),
                     seed=0)
    wall = time.time() - t0
    if not r.get("ok"):
        print(f"[{i:2}/{len(CASES)}] {cat}: BLOCKED {r.get('blocked')}")
        results.append({"cat": cat, "prompt": prompt, "blocked": r.get("blocked")})
        continue
    sc = r["prediction_scorecard"]
    acct = r["accounting"]
    tgt = r["target"]
    gen = sc.get("generation_tier") or {}
    claimed_edges = [e for e in r.get("edges", []) if e["claimed"]]
    solid = [n for n in r["nodes"] if not n.get("marginal")]
    strong = [n for n in r["nodes"] if n.get("strength") == "strong"]
    weak = [n for n in r["nodes"] if n.get("strength") == "weak"]
    legs = [n["legibility"] for n in strong if n["legibility"] is not None]
    row = {
        "cat": cat, "prompt": prompt[:60], "target": tgt["piece"],
        "margin": tgt["margin"], "verdict": r["controls"]["verdict"],
        "survivors": acct["survivors"], "solid": len(solid),
        "strong": len(strong), "weak": len(weak),
        "delta_total": acct["delta_total"], "sum_solo": acct["sum_solo"],
        "interaction_gap": acct["interaction_gap"],
        "edges_claimed": len(claimed_edges), "edges_total": len(r.get("edges", [])),
        "correct": sc["correct_predictions"], "wrong": sc["wrong_predictions"],
        "diverged_early": sc["diverged_early"],
        "pred_flips": sc["predicted_flips"], "obs_flips": sc["observed_flips"],
        "baseline_greedy": gen.get("baseline_greedy"),
        "median_legibility": (sorted(legs)[len(legs)//2] if legs else None),
        "wall": wall,
    }
    results.append(row)
    print(f"[{i:2}/{len(CASES)}] {cat:20} {tgt['piece']!r:12} "
          f"margin {tgt['margin'] if tgt['margin'] is None else round(tgt['margin'],2)!s:>6} "
          f"| {r['controls']['verdict']:15} | strong {len(strong):2} weak {len(weak):2} | "
          f"edges {len(claimed_edges)}/{len(r.get('edges',[]))} | "
          f"S4 {sc['correct_predictions']}ok/{sc['wrong_predictions']}wrong"
          f"{' /'+str(sc['diverged_early'])+'early' if sc['diverged_early'] else ''} "
          f"| {wall:.1f}s")

print("\n" + "=" * 100)
ok = [r for r in results if "blocked" not in r]
tot_correct = sum(r["correct"] for r in ok)
tot_wrong = sum(r["wrong"] for r in ok)
tot_early = sum(r["diverged_early"] for r in ok)
tot_pred = sum(r["pred_flips"] for r in ok)
tot_obs = sum(r["obs_flips"] or 0 for r in ok)
print(f"TRACES: {len(ok)}/{len(CASES)} ok, {len(results)-len(ok)} blocked, "
      f"total wall {time.time()-t_start:.0f}s")
print(f"VERDICTS: " + ", ".join(f"{v}={sum(1 for r in ok if r['verdict']==v)}"
                                for v in ["PASS", "NO_CAUSAL_NODES", "FAILED_CONTROLS"]))
print(f"S4 SCORECARD (aggregate): {tot_correct} correct, {tot_wrong} wrong, {tot_early} diverged-early "
      f"| flips predicted {tot_pred}, observed {tot_obs}")
den = tot_correct + tot_wrong
print(f"  prediction accuracy: {tot_correct}/{den} = {(tot_correct/den*100 if den else 0):.1f}%")
print(f"EDGES: {sum(r['edges_claimed'] for r in ok)} claimed / {sum(r['edges_total'] for r in ok)} tested")
ngap = [r for r in ok if r["interaction_gap"] is not None and r["sum_solo"]]
if ngap:
    ratios = sorted(r["interaction_gap"] / r["sum_solo"] for r in ngap if r["sum_solo"])
    print(f"INTERACTION GAP / sum_solo: median {ratios[len(ratios)//2]:+.0%} "
          f"(range {ratios[0]:+.0%} .. {ratios[-1]:+.0%})")
legs = [r["median_legibility"] for r in ok if r["median_legibility"] is not None]
if legs:
    legs = sorted(legs)
    print(f"MEDIAN LEGIBILITY per trace: median {legs[len(legs)//2]:.0%} "
          f"(range {legs[0]:.0%} .. {legs[-1]:.0%})")
print("\nBY CATEGORY:")
for cat in dict.fromkeys(c[0] for c in CASES):
    rs = [r for r in ok if r["cat"] == cat]
    if not rs:
        continue
    c = sum(r["correct"] for r in rs); w = sum(r["wrong"] for r in rs)
    print(f"  {cat:20} n={len(rs)}  verdicts={{{', '.join(sorted({r['verdict'] for r in rs}))}}}  "
          f"strong={sum(r['strong'] for r in rs):3} weak={sum(r['weak'] for r in rs):3}  "
          f"S4 {c}ok/{w}wrong")
print(f"\nNODE TIERS overall: strong={sum(r['strong'] for r in ok)}, weak={sum(r['weak'] for r in ok)}, "
      f"marginal={sum(r['survivors']-r['strong']-r['weak'] for r in ok)} "
      f"(of {sum(r['survivors'] for r in ok)} that beat the median-based noise floor)")

with open(r"C:\Users\brigi\AppData\Local\Temp\claude\C--Users-brigi-src-clozn\d351b6fa-f0ca-40d4-9b7a-377886b898e2\scratchpad\battery_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nresults -> battery_results.json")
