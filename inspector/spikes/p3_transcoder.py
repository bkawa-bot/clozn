"""
Phase-3 — a TRANSCODER on RWKV's channel-mix block, head-to-head vs an SAE on the same site.

The field moved from SAEs to transcoders because, on transformers, transcoder features come out
more interpretable. Does that hold on RWKV? We hook block L's channel-mix (RWKV's MLP), capture
its (input, output) per token, and train two sparse dictionaries on the SAME encoder input:
  - SAE        : reconstruct the INPUT      (decode(encode(x)) ≈ x)
  - TRANSCODER : predict the block's OUTPUT (decode(encode(x)) ≈ y)
Then we compare how thematically coherent each one's discovered features are. Honest test, both
L1-tuned the same way; we report whichever wins.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.discover import collect_block_io, corpus, sae_features, standardize  # noqa: E402
from clozn.sources.hf_rwkv import RwkvStateSource          # noqa: E402
from clozn.viz import render_discovered_features            # noqa: E402

LAYERS = (3, 6, 9)


def best_run(Xs, toks, Y):
    best = None
    for l1 in (4e-2, 8e-2):
        feats, _, _ = sae_features(Xs, toks, m=128, l1=l1, steps=600, Y=Y)
        mp = np.mean([f.purity for f in feats]) if feats else 0.0
        if best is None or mp > best[0]:
            best = (mp, sum(f.purity >= 0.6 for f in feats), feats)
    return best


def main():
    print("loading RWKV-4-169m; SAE vs transcoder on channel-mix, swept across layers ...")
    src = RwkvStateSource()
    print(f"\n  {'layer':<7}{'SAE purity':<14}{'TRANSCODER purity':<20}winner")
    overall = []
    for L in LAYERS:
        Xin, Yout, toks, _ = collect_block_io(src, corpus(), layer=L)
        Xs, _, _ = standardize(Xin)
        Ys, _, _ = standardize(Yout)
        s_mp, s_coh, _ = best_run(Xs, toks, None)
        t_mp, t_coh, t_feats = best_run(Xs, toks, Ys)
        win = "transcoder" if t_mp > s_mp + 0.03 else ("SAE" if s_mp > t_mp + 0.03 else "~tie")
        print(f"  L{L:<6}{s_mp*100:>4.0f}% ({s_coh})      {t_mp*100:>4.0f}% ({t_coh})            {win}")
        overall.append((L, s_mp, t_mp, t_feats))

    # detail + render the layer where the transcoder did best
    L, s_mp, t_mp, t_feats = max(overall, key=lambda r: r[2])
    print(f"\n=== top transcoder features @ best layer L{L} ===")
    for f in t_feats[:10]:
        tag = f"[{f.theme} {f.purity*100:.0f}%]" if f.purity >= 0.6 else "[mixed]"
        print(f"  f{f.idx:<3} {tag:<16} {' '.join(repr(t.strip()) for t in f.top_tokens[:6])}")

    s_best = max(r[1] for r in overall); t_best = max(r[2] for r in overall)
    print(f"\n  OVERALL: best SAE {s_best*100:.0f}%  vs  best TRANSCODER {t_best*100:.0f}%  -> "
          + ("transcoder wins (more interpretable, as the field finds at scale)"
             if t_best > s_best + 0.03 else
             "SAE still competitive at this scale — honest: the transcoder edge needs scale/depth, "
             "not a 169M model with ~440 tokens (matches the field's 'rigor over hype' turn)"))

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "transcoder_features.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_discovered_features(
            t_feats, title="Clozn · Transcoder Features",
            subtitle=f"RWKV-4-169m · channel-mix L{L} input→output · discovered unsupervised"))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
