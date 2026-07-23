"""molecules.py -- the "minimal jointly-sufficient set" program (2026-07-23).

The cycle's deepest result is that individually-nameable units carry almost no causal mass on
these models: single positions ~0 (vs a +7.6 greedy SPAN), single SAE features ~0.5%, single
sites overshoot when summed (-60/-82% interaction gaps), single heads no >=2x separation. The
atom is wrong; the molecule is real. This operationalizes the molecule directly.

Redefine "circuit" as the SMALLEST COALITION OF SITES whose JOINT ablation reproduces most of the
answer's causal effect AND beats a random coalition of the same size. Method:
  1. Trace to get candidate (layer, pos) sites and the full joint delta (all survivors ablated).
  2. GREEDY set construction: start empty; at each step add the site whose addition most increases
     the coalition's joint delta; record marginal gains. This is the set analogue of the greedy
     SPAN search that beat single positions ~70x.
  3. MINIMAL SUFFICIENT SET: the smallest prefix of that greedy order reaching TARGET (default 80%)
     of the full-survivor joint delta.
  4. MATCHED CONTROL: random-k coalitions (same k, same layers-pool) -- the molecule is real only
     if the found k-set's joint delta clears the random-k crowd (report the separation ratio).

Contrastive (--contrast auto) makes every delta answer-SELECTIVE (the screen-null fix). Absolute
mode is the default -- directly comparable to the existing interaction-gap receipts.

Writes runs/experiments/molecules_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from clozn.analysis import tracer  # noqa: E402


# The molecule (a multi-site coalition) can only appear where the answer genuinely depends on
# DISTINCT EARLIER positions -- KV retrieval, induction, multi-hop. Simple factual completion is
# dominated by the readout position (the last row, where prediction happens, always matters), so
# it has no molecule by construction. These prompts route the answer through specific prompt tokens.
CASES = [
    ("The box is blue. The lamp is red. The cup is green. The color of the box is", None),
    ("The box is blue. The lamp is red. The cup is green. The color of the lamp is", None),
    ("Anna has the key. Ben has the map. Carl has the torch. The person with the map is", None),
    ("The wizard Zorblax cast a spell. Everyone cheered for the wizard", None),
    ("Paris is in France. Tokyo is in Japan. Cairo is in Egypt. Tokyo is in", None),
    ("The red door leads to the vault. The blue door leads to the exit. The vault is behind the", None),
]


def _capture_mean_rows(engine, prompt, cont_ids, layers, n_p):
    cap = tracer._score(engine, prompt, cont_ids, capture={"layers": layers,
                                                           "positions": list(range(n_p))})
    caps = cap["captured"]
    mean_rows = {}
    for L in layers:
        rows = caps[str(L)]
        M = np.stack([np.asarray(rows[str(p)], np.float32) for p in range(n_p) if str(p) in rows])
        mean_rows[L] = M.mean(0)
    return mean_rows


def run(engine, tag, target_frac, contrast, exclude_final):
    results = []
    for prompt, _ in CASES:
        # the model's own greedy answer token
        gen = tracer._post(engine, "/v1/completions",
                           {"prompt": prompt, "max_tokens": 2, "temperature": 0})
        true_cont = gen["choices"][0]["text"]
        tr = tracer.trace(prompt, true_cont, 0, engine_url=engine, screen_mode="ablate",
                          contrast=("auto" if contrast else None),
                          budget=tracer.TraceBudget(max_candidates=16, ablate_screen_arms=48))
        if not tr.get("ok"):
            results.append({"prompt": prompt, "blocked": tr.get("blocked")})
            continue
        base = tracer._score(engine, prompt, true_cont, topk=2)
        n_p = int(base["n_prompt"])
        # Drop the trivial readout positions (the last `exclude_final` prompt rows): the row that
        # predicts the answer always matters, so including it hands the greedy a size-1 answer.
        # The molecule question is whether the EARLIER positions form a coalition.
        cand = [c for c in tr["all_candidates"] if int(c["pos"]) < n_p - exclude_final]
        if not cand:
            results.append({"prompt": prompt, "note": "no candidates after excluding readout"})
            continue
        layers = sorted({int(c["layer"]) for c in cand})
        y_id = int(base["tokens"][0]["id"])
        cont_ids = [y_id]
        base_lp = float(base["tokens"][0]["logprob"])
        others = [t for t in (base["tokens"][0].get("topk") or []) if int(t["id"]) != y_id]
        foil_id = int(others[0]["id"]) if (contrast and others) else None
        cont_ids_foil = [foil_id] if foil_id is not None else None
        base_lp_foil = (float(tracer._score(engine, prompt, cont_ids_foil)["tokens"][0]["logprob"])
                        if cont_ids_foil else 0.0)
        mean_rows = _capture_mean_rows(engine, prompt, cont_ids, layers, n_p)

        def joint_delta(sites):
            """Contrastive-or-absolute joint delta of ablating this set of (layer,pos) sites."""
            if not sites:
                return 0.0
            specs = tracer.group_joint_writes([{"layer": L, "pos": p} for L, p in sites], mean_rows)
            r_ = tracer._score(engine, prompt, cont_ids, write=specs)
            d_y = base_lp - float(r_["tokens"][0]["logprob"])
            if cont_ids_foil is None:
                return d_y
            rf = tracer._score(engine, prompt, cont_ids_foil, write=specs)
            return d_y - (base_lp_foil - float(rf["tokens"][0]["logprob"]))

        pool = [(int(c["layer"]), int(c["pos"])) for c in cand]
        full = joint_delta(pool)                       # all candidates ablated at once
        # ---- greedy coalition construction: grow until marginal |delta| gain dies ----
        chosen, gains, remaining, traj = [], [], list(pool), []
        cur = 0.0
        while remaining:
            best, best_d = None, None
            for s in remaining:
                d = joint_delta(chosen + [s])
                if best_d is None or abs(d) > abs(best_d):
                    best, best_d = s, d
            gain = abs(best_d) - abs(cur)
            if gain <= 1e-4 and len(chosen) >= 1:      # coalition no longer grows
                break
            chosen.append(best); gains.append(round(gain, 4)); remaining.remove(best)
            cur = best_d; traj.append(cur)
        peak = cur                                     # the greedy MAXIMUM (not `full`: contrastive
        # ablation is non-monotonic -- adding foil-supporting sites REDUCES the selective gap).
        # ---- minimal sufficient set: smallest prefix reaching target_frac of the greedy peak ----
        k = next((i + 1 for i, v in enumerate(traj) if abs(v) >= abs(target_frac * peak)), len(traj))
        min_set = chosen[:k]
        min_set_delta = traj[k - 1] if traj else 0.0
        # ---- matched random-k control: random POSITIONS at the SAME layers (not the candidate
        # pool, which is all-strong). This is the honest control -- does a same-size set of
        # arbitrary sites at these layers reach the coalition's delta? ----
        rng = np.random.default_rng(0)
        chosen_keys = set(min_set)
        cand_layers = sorted({L for L, _ in pool})
        ctl = []
        for _ in range(8):
            picks = []
            for _t in range(200):
                if len(picks) >= k:
                    break
                s = (cand_layers[int(rng.integers(len(cand_layers)))], int(rng.integers(n_p)))
                if s not in chosen_keys and s not in picks:
                    picks.append(s)
            ctl.append(abs(joint_delta(picks)))
        ctl_max = max(ctl) if ctl else 0.0
        sep = abs(min_set_delta) / ctl_max if ctl_max > 1e-9 else None
        # sum of the chosen sites' SOLO deltas (the atom story) vs the coalition (the molecule)
        solo_sum = sum(joint_delta([s]) for s in min_set)
        row = {
            "prompt": prompt, "answer": true_cont.strip(), "scoring": "contrastive" if contrast else "absolute",
            "pool_size": len(pool), "full_joint_delta": round(full, 4),
            "min_set_size": k, "min_set_delta": round(min_set_delta, 4),
            "frac_of_full": round(abs(min_set_delta) / abs(full), 3) if abs(full) > 1e-9 else None,
            "sum_of_solos": round(solo_sum, 4),
            "coalition_vs_solosum": (round(abs(min_set_delta) / abs(solo_sum), 2)
                                     if abs(solo_sum) > 1e-9 else None),
            "random_k_max": round(ctl_max, 4), "separation_vs_random_k": round(sep, 2) if sep else None,
            "marginal_gains": gains,
        }
        results.append(row)
        print(f"{prompt[:32]!r:<34} ans={row['answer']!r:<10} k={k}/{len(pool)} "
              f"delta {min_set_delta:+.2f} ({row['frac_of_full']} of full) "
              f"vs random-k {sep:.2f}x  solo-sum {solo_sum:+.2f}")

    graded = [r for r in results if "min_set_size" in r]
    summary = {
        "tag": tag, "scoring": "contrastive" if contrast else "absolute", "target_frac": target_frac,
        "exclude_final": exclude_final,
        "n": len(graded),
        "mean_min_set_size": round(np.mean([r["min_set_size"] for r in graded]), 2) if graded else None,
        "mean_pool_size": round(np.mean([r["pool_size"] for r in graded]), 2) if graded else None,
        "mean_separation_vs_random_k": round(np.mean([r["separation_vs_random_k"] for r in graded
                                                      if r["separation_vs_random_k"]]), 2) if graded else None,
        "reading": None,
    }
    if graded:
        seps = [r["separation_vs_random_k"] for r in graded if r["separation_vs_random_k"]]
        beats = sum(1 for s in seps if s >= 2.0)
        summary["reading"] = (
            f"minimal sufficient coalition beats matched random-k >=2x in {beats}/{len(seps)} cases "
            f"(mean {summary['mean_separation_vs_random_k']}x); the molecule is a real, controllable "
            f"unit where the atom was not" if beats >= len(seps) * 0.6 else
            f"WEAK: coalition beats random-k >=2x in only {beats}/{len(seps)} -- the found set is not "
            f"clearly better than a same-size random set; report honestly")
    out = os.path.join(REPO, "runs", "experiments", f"molecules_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=2)
    print("\n=== molecules (minimal jointly-sufficient set) ===")
    for kk, vv in summary.items():
        print(f"  {kk}: {vv}")
    print(f"wrote {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://127.0.0.1:8091")
    ap.add_argument("--tag", default="qwen2.5-7b")
    ap.add_argument("--target-frac", type=float, default=0.8)
    ap.add_argument("--contrast", action="store_true", help="answer-selective (contrastive) deltas")
    ap.add_argument("--exclude-final", type=int, default=1, help="drop the last N prompt (readout) positions from the pool")
    args = ap.parse_args()
    run(args.engine, args.tag, args.target_frac, args.contrast, args.exclude_final)
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------------------------
# VERDICT (2026-07-23, measured on Qwen2.5-7B): the molecule fails its control too.
#
# Three batteries (runs/experiments/molecules_qwen2.5-7b{,-abs,-kv}.json):
#   1. simple factual, contrastive : min-set = 1. A single readout-position site carries the whole
#      answer-SELECTIVITY (frac_of_full > 1: adding category-scaffolding sites REDUCES the
#      selective gap -- contrastive ablation is non-monotonic). Selection is localized where
#      category is distributed -- a real, small finding, but not a coalition.
#   2. simple factual, absolute : min-set = 1, separation 1.43x. The last position dominates (it
#      IS the prediction row); barely beats random late positions. No molecule -- expected.
#   3. DISTRIBUTED (KV / induction / multi-hop), readout EXCLUDED, absolute : mean min-set 1.67,
#      separation 0.93x vs the STRONGEST of 8 random same-size, same-layer coalitions. The
#      greedily-OPTIMIZED coalition does NOT beat a random coalition. 0/6 clear the >=2x bar.
#
# Reading: at the (layer, position) granularity under RESIDUAL mean-ablation, there is no
# privileged small coalition -- targeted small sets are indistinguishable from random small sets.
# The distributed-function thesis now holds at a FIFTH level: positions (~0 solo), SAE features
# (~0.5%), heads (no >=2x), site sum-of-solos (-60/-82% gap), and now position COALITIONS
# (<=1x random). The atom is not the unit; neither is the residual-ablation molecule.
#
# The one thing that DID beat its controls by 100x+ -- the greedy SPAN in provenance -- is a
# DIFFERENT instrument: attention SEVERANCE of CONTIGUOUS INPUT tokens, not residual mean-ablation
# of an arbitrary position set. That is the honest methodological seam: localized structure shows
# up under the sharp input-severance instrument, and dissolves under the blunt residual-ablation
# one. Which is itself a finding about what "a circuit" can mean on these models -- and where to
# look for it (the input edges, not the residual sites). Open: sharper coalition instruments
# (attention-edge coalitions, feature coalitions), larger k, other layers. Any revival starts from
# a coalition that beats its control -- this one does not.
