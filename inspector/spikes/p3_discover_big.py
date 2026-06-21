"""
Phase-3 — feature discovery on a BIGGER model. Same pipeline, same themed corpus, larger RWKV-4,
so any change in the discovered features is a scale effect, not a method change.

Usage:  python spikes/p3_discover_big.py [hf_model_name]
        default RWKV/rwkv-4-1b5-pile (1.5B, ~9x the 169m baseline).
Baseline to compare against (169m, same pipeline): mean purity 65%, 7/12 coherent
(color/number/emotion/animal/place).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.discover import collect_token_states, corpus, pca_features, sae_features, standardize  # noqa: E402
from clozn.sources.hf_rwkv import RwkvStateSource  # noqa: E402
from clozn.viz import render_discovered_features  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "RWKV/rwkv-4-1b5-pile"
SHORT = NAME.split("/")[-1].replace("rwkv-4-", "").replace("-pile", "")


def show(feats, title):
    print(f"\n=== {title} ===")
    for f in feats:
        tag = f"[{f.theme} {f.purity*100:.0f}%]" if f.purity >= 0.6 else "[mixed]"
        print(f"  f{f.idx:<3} fires={f.fires_on*100:4.1f}%  {tag:<16} "
              f"{' '.join(repr(t.strip()) for t in f.top_tokens[:6])}")


def main():
    print(f"loading {NAME} (this is the big download/load) ...")
    src = RwkvStateSource(name=NAME)
    X, toks, _ = collect_token_states(src, corpus())
    Xs, _, _ = standardize(X)
    print(f"  {X.shape[0]} token-states, hidden dim {X.shape[1]} (169m was 768)")

    pca = pca_features(Xs, toks, k=12)
    pm = np.mean([f.purity for f in pca])

    best = None
    for l1 in (2e-2, 4e-2, 8e-2):
        feats, model, stats = sae_features(Xs, toks, m=128, l1=l1, steps=800)
        mp = np.mean([f.purity for f in feats]) if feats else 0.0
        coh = sum(f.purity >= 0.6 for f in feats)
        print(f"  [sweep] l1={l1:<5} coherent={coh:<2} mean-purity={mp*100:.0f}% mse={model.recon_error(Xs):.3f}")
        if best is None or mp > best[0]:
            best = (mp, l1, feats)
    mp, l1, feats = best
    show(feats, f"{SHORT} — discovered features (best l1={l1})")

    coh = sum(f.purity >= 0.6 for f in feats)
    print(f"\n  {SHORT}: SAE mean-purity {mp*100:.0f}% ({coh}/12 coherent) · PCA baseline {pm*100:.0f}%")
    print(f"  169m baseline (same pipeline): SAE 65% (7/12) · PCA 12%")
    print(f"  -> scale effect: {'+' if mp > 0.65 else ''}{(mp-0.65)*100:.0f} pts mean-purity vs 169m")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs",
                       f"discovered_{SHORT}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_discovered_features(
            feats, title=f"Clozn · Discovered Features ({SHORT})",
            subtitle=f"{NAME} · tiny SAE on att_num · {X.shape[0]} tokens · vs 169m baseline"))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
