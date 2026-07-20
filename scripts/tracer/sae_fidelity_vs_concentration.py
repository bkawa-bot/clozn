"""Does the v3 finding generalize? Across prompts and positions, measure:

  A. CAUSAL FIDELITY of the SAE:  1 - delta(substitute)/delta(mean_ablate)
     "how much of this site's causal content survives replacing h with the SAE's reconstruction"
     ~1.0 => the dictionary spans what matters here.
  B. CONCENTRATION: (sum of top-k single-feature ablation deltas) / delta(mean_ablate)
     ~1.0 => a few features carry the site (a sparse circuit).  ~0 => distributed across many.

A near 1 with B near 0 is the interesting combination: the SAE captures the causal content, but
it is spread across dozens of features -- i.e. there is no sparse circuit AT THIS SITE, which is
a claim about the model, not a failure of the tool.

Also reports explained variance so the (variance vs function) dissociation is visible.
"""
import json
import time
import urllib.request

import numpy as np

ENGINE = "http://127.0.0.1:8080"
SAE = r"C:\Users\brigi\.clozn\sae\andyrdt_l15"
LAYER, D_IN, D_SAE = 15, 3584, 131072
TOPK_FEATS = 12
OUT = r"C:\Users\brigi\AppData\Local\Temp\claude\C--Users-brigi-src-clozn\d351b6fa-f0ca-40d4-9b7a-377886b898e2\scratchpad\sae_generalize.json"

CASES = [
    ("induction", "In this story, the wizard's name is Zorblax and the knight's name is Pellinore. "
                  "When the dragon attacked the village, the wizard raised his staff and cast a spell. "
                  "The name of the wizard who cast the spell is"),
    ("factual", "The capital of France is"),
    ("factual2", "The largest planet in our solar system is"),
    ("distractor", "Kyoto was the capital of Japan for over a thousand years. Since the Meiji era, "
                   "however, the government has been located elsewhere. The modern capital of Japan is"),
    ("syntactic", "The keys to the cabinet in the hallway upstairs"),
    ("arithmetic", "The product of 12 and 12 is"),
]

def post(path, body, timeout=600):
    req = urllib.request.Request(ENGINE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

W_enc = np.memmap(f"{SAE}/w_enc_t.f16.bin", dtype=np.float16, mode="r", shape=(D_SAE, D_IN))
W_dec = np.memmap(f"{SAE}/w_dec.f16.bin", dtype=np.float16, mode="r", shape=(D_SAE, D_IN))
b_enc = np.fromfile(f"{SAE}/b_enc.f16.bin", dtype=np.float16).astype(np.float32)
b_dec = np.fromfile(f"{SAE}/b_dec.f16.bin", dtype=np.float16).astype(np.float32)
thr = np.fromfile(f"{SAE}/threshold.f32.bin", dtype=np.float32)

rows = []
for name, prompt in CASES:
    cont = post("/v1/completions", {"prompt": prompt, "max_tokens": 2,
                                    "temperature": 0})["choices"][0]["text"]
    base = post("/score", {"prompt": prompt, "continuation": cont, "topk": 0})
    n_p = base["n_prompt"]
    base_lp = float(base["tokens"][0]["logprob"])
    cont_ids = [int(base["tokens"][0]["id"])]
    tgt_piece = base["tokens"][0]["piece"]

    H = np.zeros((n_p, D_IN), dtype=np.float32)
    for s in range(0, n_p, 48):
        chunk = list(range(s, min(s + 48, n_p)))
        r = post("/score", {"prompt": prompt, "continuation_ids": cont_ids,
                            "capture": {"layers": [LAYER], "positions": chunk}})
        cap = (r.get("captured") or {}).get(str(LAYER)) or {}
        for p in chunk:
            H[p] = np.asarray(cap[str(p)], dtype=np.float32)
    mean_row = H.mean(axis=0)

    def arm(pos, values):
        v = np.asarray(values, dtype=np.float32).tolist()
        r = post("/score", {"prompt": prompt, "continuation_ids": cont_ids,
                            "write": {"layer": LAYER, "positions": [pos], "values": v}})
        return base_lp - float(r["tokens"][0]["logprob"])

    PRE = np.zeros((n_p, D_SAE), dtype=np.float32)
    for f0 in range(0, D_SAE, 16384):
        PRE[:, f0:f0 + 16384] = H @ np.asarray(W_enc[f0:f0 + 16384], dtype=np.float32).T
    PRE += b_enc[None, :]
    ACT = np.where(PRE > thr[None, :], PRE, 0.0)

    # the decisive site = the position with the largest mean-ablation effect, EXCLUDING position 0
    # (the attention sink: norm ~200x typical, the SAE is wildly out of distribution there -- see
    # notes; including it would measure a pathology, not the circuit).
    scan = sorted(((arm(p, mean_row), p) for p in range(1, n_p)), key=lambda x: -abs(x[0]))
    d_mean, p_star = scan[0]

    h = H[p_star]
    idx = np.nonzero(ACT[p_star])[0]
    D = np.asarray(W_dec[idx], dtype=np.float32)
    h_hat = b_dec + ACT[p_star, idx] @ D
    ev = float(1.0 - np.linalg.norm(h - h_hat) ** 2 / (np.linalg.norm(h) ** 2 + 1e-9))
    d_sub = arm(p_star, h_hat)
    fidelity = 1.0 - (d_sub / d_mean) if abs(d_mean) > 1e-9 else float("nan")

    order = idx[np.argsort(-ACT[p_star, idx])][:TOPK_FEATS]
    feat_deltas = [arm(p_star, h - ACT[p_star, f] * np.asarray(W_dec[f], dtype=np.float32))
                   for f in order]
    conc_top1 = max(feat_deltas, key=abs) / d_mean if abs(d_mean) > 1e-9 else float("nan")
    conc_topk = sum(feat_deltas) / d_mean if abs(d_mean) > 1e-9 else float("nan")

    rows.append({"case": name, "target": tgt_piece, "pos": int(p_star), "n_active": int(idx.size),
                 "explained_var": ev, "delta_mean": d_mean, "delta_sub": d_sub,
                 "causal_fidelity": fidelity, "conc_top1": conc_top1, "conc_topk": conc_topk,
                 "norm_h": float(np.linalg.norm(h))})
    print(f"{name:12} tgt {tgt_piece!r:10} pos {p_star:>3} | n_act {idx.size:>4} "
          f"EV {ev:>6.2f} | d_mean {d_mean:>+7.3f} d_sub {d_sub:>+7.3f} "
          f"| fidelity {fidelity:>6.1%} | top1 {conc_top1:>6.1%} top{TOPK_FEATS} {conc_topk:>6.1%}")

print("\n" + "=" * 104)
fid = [r["causal_fidelity"] for r in rows]
ev = [r["explained_var"] for r in rows]
c1 = [r["conc_top1"] for r in rows]
ck = [r["conc_topk"] for r in rows]
na = [r["n_active"] for r in rows]
print(f"CAUSAL FIDELITY of the SAE reconstruction: median {np.median(fid):.1%} "
      f"(range {min(fid):.1%}..{max(fid):.1%})")
print(f"EXPLAINED VARIANCE at those sites:         median {np.median(ev):.1%} "
      f"(range {min(ev):.1%}..{max(ev):.1%})")
print(f"CONCENTRATION top-1 feature:               median {np.median(c1):.1%}")
print(f"CONCENTRATION top-{TOPK_FEATS} features:              median {np.median(ck):.1%}")
print(f"ACTIVE FEATURES at the decisive site:      median {int(np.median(na))}")
print("\nRead: high fidelity + low concentration => the dictionary SPANS the causal content, but no")
print("small set of features carries it -- distributed representation, not a sparse circuit.")
json.dump(rows, open(OUT, "w"), indent=2)
print(f"-> {OUT}")
