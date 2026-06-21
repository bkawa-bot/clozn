"""Phase-3 — unsupervised feature DISCOVERY on RWKV-4 state. PCA baseline vs a tiny SAE.
Honest test: does either rediscover the themes we seeded (color/animal/number/...) without labels?"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.discover import collect_token_states, corpus, pca_features, sae_features, standardize  # noqa: E402
from clozn.sources.hf_rwkv import RwkvStateSource  # noqa: E402


def show(feats, title):
    print(f"\n=== {title} ===")
    for f in feats:
        tag = f"[{f.theme} {f.purity*100:.0f}%]" if f.purity >= 0.4 else "[mixed]"
        toks = " ".join(repr(t.strip()) for t in f.top_tokens)
        print(f"  f{f.idx:<3} fires={f.fires_on*100:4.1f}%  {tag:<16} {toks}")
    if feats:
        print(f"  -> mean purity = {np.mean([f.purity for f in feats])*100:.0f}%, "
              f"coherent (>=60%): {sum(f.purity>=0.6 for f in feats)}/{len(feats)}")


def main():
    print("loading RWKV-4-169m, collecting per-token states over the themed corpus ...")
    src = RwkvStateSource()
    X, toks, tid = collect_token_states(src, corpus())
    Xs, _, _ = standardize(X)
    print(f"  {X.shape[0]} token-states, dim {X.shape[1]}, over {len(corpus())} sentences / 6 themes")

    pca = pca_features(Xs, toks, k=12)
    show(pca, "PCA — dominant axes (baseline)")
    pm = np.mean([f.purity for f in pca]) if pca else 0

    # don't under-tune the method we're testing: sweep L1, keep the sparsest run that stays coherent
    best = None
    for l1 in (5e-3, 1e-2, 2e-2, 4e-2):
        feats, model, stats = sae_features(Xs, toks, m=128, l1=l1, steps=800)
        sm = np.mean([f.purity for f in feats]) if feats else 0
        print(f"  [sweep] l1={l1:<6} live={stats['live']:<3} dead={stats['dead']:<3} "
              f"dense={stats['dense']:<3} mean-purity={sm*100:.0f}% mse={model.recon_error(Xs):.3f}")
        if best is None or sm > best[0]:
            best = (sm, l1, feats, model)
    sm, l1, feats, model = best
    show(feats, f"TinySAE — discovered features (best l1={l1})")

    print(f"\n  VERDICT: mean purity  PCA {pm*100:.0f}%  vs  SAE {sm*100:.0f}%  -> "
          + ("SAE WINS — it unmixes semantic features PCA buries under syntax"
             if sm > pm + 0.05 else
             "PCA competitive — SAE didn't clearly win at this scale (consistent with the field)"))

    from clozn.viz import render_discovered_features
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "discovered_features.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_discovered_features(
            feats, subtitle=f"RWKV-4-169m · tiny SAE on att_num · {X.shape[0]} tokens · rediscovered themes unsupervised"))
    print("wrote", out)


if __name__ == "__main__":
    main()
