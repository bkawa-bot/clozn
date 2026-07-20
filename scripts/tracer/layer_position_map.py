"""Where does causal mass actually live? A (layer x position) mean-ablation map.

Run BEFORE any feature-level cross-position work, because the only SAE on disk is fixed at
layer 15 -- if the source position's causal mass lives at a different depth, the dictionary
structurally cannot see the circuit and that is a fact about the artifact, not the model.

For each layer in LAYERS and each prompt position, replace h(layer, pos) with that layer's mean
row and measure the teacher-forced logprob drop on the target token. ~n_layers x n_pos arms.
"""
import json
import time
import urllib.request

import numpy as np

ENGINE = "http://127.0.0.1:8080"
LAYERS = [2, 8, 15, 18, 21, 25]
OUT = r"C:\Users\brigi\AppData\Local\Temp\claude\C--Users-brigi-src-clozn\d351b6fa-f0ca-40d4-9b7a-377886b898e2\scratchpad\layer_pos_map.json"

PROMPT = ("In this story, the wizard's name is Zorblax and the knight's name is Pellinore. "
          "When the dragon attacked the village, the wizard raised his staff and cast a spell. "
          "The name of the wizard who cast the spell is")
CONT = " Zorblax"

def post(path, body, timeout=600):
    req = urllib.request.Request(ENGINE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

base = post("/score", {"prompt": PROMPT, "continuation": CONT, "topk": 0})
n_p, base_lp = base["n_prompt"], float(base["tokens"][0]["logprob"])
cont_ids = [int(base["tokens"][0]["id"])]
toks = post("/harvest", {"text": PROMPT, "layer": 15})["tokens"]
print(f"target {base['tokens'][0]['piece']!r} logprob {base_lp:.4f} | {n_p} positions x {len(LAYERS)} layers")

# capture every layer's rows in one pass per layer-chunk
H = {}
for L in LAYERS:
    Hl = np.zeros((n_p, 3584), dtype=np.float32)
    for s in range(0, n_p, 48):
        chunk = list(range(s, min(s + 48, n_p)))
        r = post("/score", {"prompt": PROMPT, "continuation_ids": cont_ids,
                            "capture": {"layers": [L], "positions": chunk}})
        cap = (r.get("captured") or {}).get(str(L)) or {}
        for p in chunk:
            Hl[p] = np.asarray(cap[str(p)], dtype=np.float32)
    H[L] = Hl

t0 = time.time()
M = np.zeros((len(LAYERS), n_p), dtype=np.float32)
for li, L in enumerate(LAYERS):
    mean_row = H[L].mean(axis=0).tolist()
    for p in range(n_p):
        r = post("/score", {"prompt": PROMPT, "continuation_ids": cont_ids,
                            "write": {"layer": L, "positions": [p], "values": mean_row}})
        M[li, p] = base_lp - float(r["tokens"][0]["logprob"])
print(f"{len(LAYERS)*n_p} arms in {time.time()-t0:.1f}s\n")

# interesting positions: the source, the competitor, the final, and whatever else ranks
SRC = next(p for p, t in enumerate(toks) if t.strip() == "Z")
COMP = next((p for p, t in enumerate(toks) if "Pell" in t), None)
FINAL = n_p - 1
named = {SRC: "SOURCE 'Z'", COMP: "COMPETITOR 'Pell'", FINAL: "FINAL", 0: "sink 'In'"}

print(f"{'layer':>6} | " + " ".join(f"{L:>9}" for L in ['SOURCE', 'COMPET', 'FINAL', 'sink']) +
      " | top non-final position")
print("-" * 92)
for li, L in enumerate(LAYERS):
    row = M[li]
    order = np.argsort(-np.abs(row))
    top_nonfinal = next(p for p in order if p != FINAL)
    print(f"{L:>6} | {row[SRC]:>9.3f} {row[COMP]:>9.3f} {row[FINAL]:>9.3f} {row[0]:>9.3f}"
          f" | pos {top_nonfinal:>3} {toks[top_nonfinal]!r:12} {row[top_nonfinal]:+.3f}")

print(f"\nSOURCE position ({SRC}, {toks[SRC]!r}) causal mass by layer:")
for li, L in enumerate(LAYERS):
    row = M[li]
    rank = int(np.argsort(-np.abs(row)).tolist().index(SRC)) + 1
    med = float(np.median(np.abs(row)))
    print(f"  L{L:>2}: delta {row[SRC]:+8.4f}  rank {rank:>2}/{n_p}  "
          f"({abs(row[SRC])/max(med,1e-9):>6.1f}x the median position)")

best = max(LAYERS, key=lambda L: abs(M[LAYERS.index(L), SRC]))
print(f"\n--> the source position peaks at layer {best}. SAE is at layer 15.")
json.dump({"layers": LAYERS, "matrix": M.tolist(), "tokens": toks, "src": int(SRC),
           "comp": int(COMP) if COMP else None, "final": int(FINAL), "base_lp": base_lp},
          open(OUT, "w"), indent=2)
print(f"-> {OUT}")
