"""Find a config with SPARSE, COHERENT features. Sweep L1 hard at the converged config, report
both fire-rate and top-token coherence so we can pick honestly."""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.discover import TinySAE, standardize  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs")
d = np.load(os.path.join(RUNS, "qwen_engine_acts.npz"), allow_pickle=True)
X = d["X"]
pieces = list(d["pieces"])
Xs, _, _ = standardize(X)
norm = [str(p).strip().lower() for p in pieces]


def coherence(C, topn=20):
    F = C.shape[1]
    coh = np.zeros(F)
    for j in range(F):
        order = np.argsort(C[:, j])[::-1][:topn]
        toks = [norm[i] for i in order if norm[i] != ""]
        if toks:
            coh[j] = collections.Counter(toks).most_common(1)[0][1] / len(toks)
    return coh


print("-- hard L1 sweep (epochs=80, lr=3e-2, batch=512), coherence of live features --")
print("   live = fire in [0.002, 0.15] (stricter sparsity bar)")
for l1 in (0.1, 0.3, 0.6, 1.0, 2.0):
    sae = TinySAE(Xs.shape[1], m=512, l1=l1, seed=0).fit(Xs, batch_size=512, epochs=80, lr=3e-2)
    C = sae.codes(Xs)
    fire = (C > 1e-6).mean(0)
    for hi in (0.15, 0.4):
        live_mask = (fire >= 0.002) & (fire <= hi)
        nlive = int(live_mask.sum())
        if nlive:
            coh = coherence(C)
            mc = float(coh[live_mask].mean())
        else:
            mc = 0.0
        print(f"  l1={l1:<5} max_fire={hi}  live={nlive:<4} mean_fire={fire.mean()*100:4.1f}% "
              f"coherence(live)={mc*100:3.0f}% mse={sae.recon_error(Xs):.3f}")
