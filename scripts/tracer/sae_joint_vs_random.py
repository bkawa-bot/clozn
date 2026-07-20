"""The control that could overturn 'distributed, not a circuit'.

sum-of-singles is NOT joint ablation. A feature-level circuit could exist as a SET whose joint
removal matters even though no member matters alone (super-additivity). The interaction gaps
measured earlier (-60% median) prove singles and joints diverge badly, so this must be tested
directly before concluding anything.

Arms at the decisive site, all canonical (h - sum a_f d_dec[f] over the chosen set):
  * top-k active features jointly, k = 1, 2, 4, 8, 16, 32, all
  * RANDOM-k active features (matched k) -- the control: if top-k and random-k behave the same,
    'top' carries no special information and the site is simply distributed
  * b_dec only (every feature removed) -- the ceiling
"""
import json
import urllib.request

import numpy as np

ENGINE = "http://127.0.0.1:8080"
SAE = r"C:\Users\brigi\.clozn\sae\andyrdt_l15"
LAYER, D_IN, D_SAE = 15, 3584, 131072
OUT = r"C:\Users\brigi\AppData\Local\Temp\claude\C--Users-brigi-src-clozn\d351b6fa-f0ca-40d4-9b7a-377886b898e2\scratchpad\sae_joint.json"

CASES = [
    ("induction", "In this story, the wizard's name is Zorblax and the knight's name is Pellinore. "
                  "When the dragon attacked the village, the wizard raised his staff and cast a spell. "
                  "The name of the wizard who cast the spell is"),
    ("factual", "The capital of France is"),
    ("distractor", "Kyoto was the capital of Japan for over a thousand years. Since the Meiji era, "
                   "however, the government has been located elsewhere. The modern capital of Japan is"),
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
rng = np.random.default_rng(0)
allrows = []

for name, prompt in CASES:
    cont = post("/v1/completions", {"prompt": prompt, "max_tokens": 2,
                                    "temperature": 0})["choices"][0]["text"]
    base = post("/score", {"prompt": prompt, "continuation": cont, "topk": 0})
    n_p, base_lp = base["n_prompt"], float(base["tokens"][0]["logprob"])
    cont_ids = [int(base["tokens"][0]["id"])]
    tgt = base["tokens"][0]["piece"]

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

    scan = sorted(((arm(p, mean_row), p) for p in range(1, n_p)), key=lambda x: -abs(x[0]))
    d_mean, p = scan[0]
    h = H[p]
    idx = np.nonzero(ACT[p])[0]
    order = idx[np.argsort(-ACT[p, idx])]

    def ablate_set(fs):
        if len(fs) == 0:
            return h
        D = np.asarray(W_dec[np.asarray(fs)], dtype=np.float32)
        return h - ACT[p, np.asarray(fs)] @ D

    print(f"\n{name}: target {tgt!r} at pos {p}, {idx.size} active features, "
          f"mean-ablation {d_mean:+.3f}")
    print(f"  {'k':>5} {'top-k joint':>13} {'(% of site)':>12} {'random-k':>10} {'(% of site)':>12}")
    row = {"case": name, "pos": int(p), "n_active": int(idx.size), "delta_mean": d_mean,
           "target": tgt, "ks": []}
    for k in [1, 2, 4, 8, 16, 32, len(order)]:
        if k > len(order):
            continue
        d_top = arm(p, ablate_set(order[:k]))
        rnd = rng.choice(idx, size=min(k, idx.size), replace=False)
        d_rnd = arm(p, ablate_set(rnd))
        print(f"  {k:>5} {d_top:>+13.4f} {100*d_top/d_mean:>11.1f}% "
              f"{d_rnd:>+10.4f} {100*d_rnd/d_mean:>11.1f}%")
        row["ks"].append({"k": int(k), "delta_top": d_top, "delta_random": d_rnd,
                          "pct_top": 100 * d_top / d_mean, "pct_rnd": 100 * d_rnd / d_mean})
    d_bdec = arm(p, b_dec)
    print(f"  b_dec only (all features removed): {d_bdec:+.4f} "
          f"({100*d_bdec/d_mean:.1f}% of the site's mass)")
    row["delta_bdec"] = d_bdec
    row["pct_bdec"] = 100 * d_bdec / d_mean
    allrows.append(row)

print("\n" + "=" * 88)
print("If top-k tracks random-k, 'which' features you remove doesn't matter -- only how many.")
print("That is the signature of a distributed code, and rules out a sparse feature circuit here.")
json.dump(allrows, open(OUT, "w"), indent=2)
print(f"-> {OUT}")
