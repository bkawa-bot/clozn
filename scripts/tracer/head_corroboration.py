"""head_corroboration.py -- the head-units corroboration test (HEAD_UNITS_DESIGN.md section 3).

Two INDEPENDENT per-Q-head interventions at the same (layer, site):
  * OUTPUT ablation -- head_write replaces head h's kqv_out slice with its mean slice (severs
    what the head CONTRIBUTES).
  * EDGE knockout   -- attn_knockout with head=h zeroes head h's attention row into the site
    (severs what the head READS; renormalized).
If "head h matters here" is a real mechanism claim, the two measurements should agree on WHICH
heads matter. Agreement = the strongest head-level claim this stack can make; disagreement is
itself a finding (a head can matter via its value path but not its routing, or vice versa).

Also runs the design's RANDOM-HEAD control: the ablation deltas at the strongest site vs the
same head indices ablated at a random non-candidate site -- head claims must separate from
"any head, anywhere".

Needs --no-flash-attn (the knockout half). Writes runs/experiments/head_corroboration_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)


def post(engine, path, body, timeout=900):
    req = urllib.request.Request(engine.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            m = (i + j) / 2.0
            for k in range(i, j + 1):
                r[order[k]] = m
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


CASES = [
    ("factual", "The capital of France is", " Paris", 21),
    ("induction", "The wizard Zorblax cast a spell. Everyone cheered for the wizard", " Z", 14),
    ("kv", "The box is blue. The lamp is red. The cup is green. The color of the box is", " blue", 14),
]


def run_case(engine, category, prompt, continuation, site_L, rng_site):
    base = post(engine, "/score", {"prompt": prompt, "continuation": continuation, "topk": 0})
    base_lp = float(base["tokens"][0]["logprob"])
    cont_ids = [int(base["tokens"][0]["id"])]
    n_p = int(base["n_prompt"])
    site_p = n_p - 1

    cap = post(engine, "/score", {"prompt": prompt, "continuation_ids": cont_ids, "topk": 0,
                                  "head_capture": {"layers": [site_L],
                                                   "positions": list(range(n_p)), "rows": True}})
    dims = cap["head_dims"]
    n_head, d_h = int(dims["n_head"]), int(dims["d_head"])
    rows = cap["head_rows"][str(site_L)]

    mean_slices = []
    for h in range(n_head):
        m = [0.0] * d_h
        for pos_row in rows.values():
            s = pos_row[h * d_h:(h + 1) * d_h]
            for i, x in enumerate(s):
                m[i] += x
        mean_slices.append([x / len(rows) for x in m])

    # arm A: OUTPUT ablation per head at the site
    ablate = []
    for h in range(n_head):
        r = post(engine, "/score", {"prompt": prompt, "continuation_ids": cont_ids, "topk": 0,
                                    "head_write": {"layer": site_L, "head": h,
                                                   "positions": [site_p],
                                                   "values": mean_slices[h]}})
        ablate.append(base_lp - float(r["tokens"][0]["logprob"]))

    # arm B: EDGE knockout per head into the site (all keys, renormalized)
    knock = []
    keys = list(range(site_p))
    for h in range(n_head):
        r = post(engine, "/score", {"prompt": prompt, "continuation_ids": cont_ids, "topk": 0,
                                    "attn_knockout": {"layer": site_L, "head": h,
                                                      "queries": [site_p], "keys": keys,
                                                      "renormalize": True}})
        knock.append(base_lp - float(r["tokens"][0]["logprob"]))

    # RANDOM-HEAD site control: same ablation values applied at a random non-final site
    ctl = []
    for h in range(n_head):
        r = post(engine, "/score", {"prompt": prompt, "continuation_ids": cont_ids, "topk": 0,
                                    "head_write": {"layer": site_L, "head": h,
                                                   "positions": [rng_site % max(1, site_p)],
                                                   "values": mean_slices[h]}})
        ctl.append(base_lp - float(r["tokens"][0]["logprob"]))

    abs_a, abs_k = [abs(x) for x in ablate], [abs(x) for x in knock]
    top_a = sorted(range(n_head), key=lambda h: -abs_a[h])[:3]
    top_k = sorted(range(n_head), key=lambda h: -abs_k[h])[:3]
    max_ctl = max(abs(x) for x in ctl)
    seps = sorted((abs_a[h] / max_ctl if max_ctl > 1e-9 else None) for h in top_a)
    return {
        "category": category, "layer": site_L, "site": site_p, "n_head": n_head,
        "rho_ablate_vs_knockout": round(spearman(abs_a, abs_k), 4),
        "top3_ablate": top_a, "top3_knockout": top_k,
        "top3_overlap": len(set(top_a) & set(top_k)),
        "top_head_sep_vs_random_site": (round(max(abs_a) / max_ctl, 2) if max_ctl > 1e-9 else None),
        "ablate": [round(x, 4) for x in ablate],
        "knockout": [round(x, 4) for x in knock],
        "control_max_abs": round(max_ctl, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="http://127.0.0.1:8091")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    health = json.loads(urllib.request.urlopen(args.engine + "/health", timeout=10).read())
    if not health.get("capabilities", {}).get("attn_knockout"):
        print("needs --no-flash-attn (the knockout half)", file=sys.stderr)
        return 2
    results = []
    for i, (cat, prompt, cont, L) in enumerate(CASES):
        r = run_case(args.engine, cat, prompt, cont, L, rng_site=1 + i)
        results.append(r)
        print(f"[{cat:<10}] L{r['layer']}@{r['site']}: rho(ablate,knockout)={r['rho_ablate_vs_knockout']:<7} "
              f"top3 overlap {r['top3_overlap']}/3  top-head vs random-site {r['top_head_sep_vs_random_site']}x")
        print(f"    ablate top3 {r['top3_ablate']}  knockout top3 {r['top3_knockout']}")
    out = os.path.join(REPO, "runs", "experiments", f"head_corroboration_{args.tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"results": results}, open(out, "w"), indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
