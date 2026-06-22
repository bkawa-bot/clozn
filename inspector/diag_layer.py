"""Is the SAE=PCA collapse a SCALE effect or a LAYER effect? The engine taps layer 2 (early,
lexical). Here we tap the SAME Qwen-0.5B at a MIDDLE layer (12) via the HF hook (no engine crash)
on the SAME WikiText corpus, and run the SAME coherence metric + SAE config. If mid-layer features
are more semantic / SAE pulls ahead, the layer-2 result is a tap-location artifact, not a verdict
on SAEs at scale.
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.discover import TinySAE, standardize  # noqa: E402
from clozn.sources.hf_transformer import TransformerActs  # noqa: E402

LAYER = int(sys.argv[1]) if len(sys.argv) > 1 else 12
NTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 6000


def coherence(scores, pieces, topn=20):
    norm = [str(p).strip().lower() for p in pieces]
    F = scores.shape[1]
    coh = np.zeros(F)
    modal = [""] * F
    tops = [[] for _ in range(F)]
    for j in range(F):
        order = np.argsort(scores[:, j])[::-1][:topn]
        toks = [norm[i] for i in order if norm[i] != ""]
        tops[j] = [pieces[i] for i in order]
        if toks:
            tok, c = collections.Counter(toks).most_common(1)[0]
            modal[j], coh[j] = tok, c / len(toks)
    return coh, modal, tops


def main():
    print(f"HF tap: Qwen2.5-0.5B layer {LAYER}, ~{NTOK} WikiText tokens (same metric as engine run)")
    acts = TransformerActs("Qwen/Qwen2.5-0.5B", layer=LAYER)
    X, toks = acts.collect(n_tokens=NTOK, ctx=64, batch=32)
    acts.close()
    print(f"  collected {X.shape[0]} x {X.shape[1]} on {acts.device}")
    Xs, _, _ = standardize(X)

    K = 64
    _, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    pca_coh, _, _ = coherence(Xs @ Vt[:K].T, toks)
    print(f"  PCA top-{K} mean coherence = {pca_coh.mean()*100:.0f}%")

    for l1 in (0.3, 0.6, 1.0):
        sae = TinySAE(Xs.shape[1], m=512, l1=l1, seed=0).fit(Xs, batch_size=512, epochs=80, lr=3e-2)
        C = sae.codes(Xs)
        fire = (C > 1e-6).mean(0)
        live = (fire >= 0.002) & (fire <= 0.4)
        coh, modal, tops = coherence(C, toks)
        mc = float(coh[live].mean()) if live.any() else 0.0
        print(f"  [SAE l1={l1}] live={int(live.sum()):<4} fire={fire.mean()*100:4.1f}% "
              f"coherence={mc*100:.0f}% mse={sae.recon_error(Xs):.3f}")
        if l1 == 0.6:
            ranked = sorted(np.where(live)[0], key=lambda j: -coh[j])[:10]
            print("    top features:")
            for j in ranked:
                print(f"      f{j:<4} {coh[j]*100:3.0f}% fire={fire[j]*100:4.1f}% "
                      f"{' '.join(repr(t.strip()) for t in tops[j][:8])}")


if __name__ == "__main__":
    main()
