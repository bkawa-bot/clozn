"""
Phase-4 — REAL feature discovery: a real transformer (Qwen2.5-0.5B), a real corpus (WikiText,
streamed), a real SAE (minibatch-trained on collected residual-stream activations). No planted
themes — whatever the model actually uses. This is the pipeline we'll scale to Qwen-7B / Dream.

Usage: python spikes/p4_qwen_discover.py [model] [layer] [n_tokens]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.discover import TinySAE, describe_sae, standardize  # noqa: E402
from clozn.sources.hf_transformer import TransformerActs       # noqa: E402
from clozn.viz import render_discovered_features               # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 12
NTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 50000
SHORT = NAME.split("/")[-1]


def main():
    print(f"loading {NAME}, collecting ~{NTOK} residual-stream activations at layer {LAYER} ...")
    acts = TransformerActs(NAME, layer=LAYER)
    X, toks = acts.collect(n_tokens=NTOK, ctx=64, batch=32)
    acts.close()
    print(f"  device={acts.device}  collected {X.shape[0]} token-activations, hidden {X.shape[1]}")
    Xs, _, _ = standardize(X)

    best = None
    # NB our L1 is averaged over N*m, so it needs to be ~O(1) (not tiny) to bite on standardized acts.
    for l1 in (1.0, 2.0):
        sae = TinySAE(Xs.shape[1], m=512, l1=l1, seed=0).fit(Xs, batch_size=4096, epochs=10, lr=4e-3)
        feats = describe_sae(sae, Xs, toks, keep=30, topn=10)
        live = ((sae.codes(Xs) > 1e-6).mean(0))
        nlive = int(((live >= 0.002) & (live <= 0.4)).sum())
        print(f"\n  [l1={l1}] live features={nlive}, recon mse={sae.recon_error(Xs):.3f}")
        for f in feats[:12]:
            print(f"    f{f.idx:<4} fires={f.fires_on*100:4.1f}%  {' '.join(repr(t.strip()) for t in f.top_tokens[:8])}")
        if best is None or nlive > best[0]:
            best = (nlive, l1, feats)

    nlive, l1, feats = best
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs",
                       f"discovered_real_{SHORT}_L{LAYER}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_discovered_features(
            feats, title=f"Clozn · Real Discovery ({SHORT})",
            subtitle=f"{NAME} layer {LAYER} · {X.shape[0]} WikiText tokens · SAE l1={l1} · un-seeded"))
    print(f"\nwrote {out}  (rendered l1={l1})")


if __name__ == "__main__":
    main()
