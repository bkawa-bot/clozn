"""
Phase-3 — the concept atlas: a "what's readable" map of the RWKV-4 hidden state.

Probes one recurrent state (att_num) for several linguistic features at once and reports which
are linearly decodable. Sentiment is also causally verified (patched-and-measured); the
grammatical features are decoded only and labelled as such — we never claim the model *uses* a
direction we haven't tested. Honest answer to "what can we see in the hidden state?"
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from clozn.atlas import concept_atlas               # noqa: E402
from clozn.sources.hf_rwkv import RwkvStateSource    # noqa: E402
from clozn.viz import render_concept_atlas           # noqa: E402


def main():
    print("loading RWKV-4-169m and probing the hidden state for several concepts ...")
    src = RwkvStateSource()
    cards = concept_atlas(src)

    print("\n=== CONCEPT ATLAS — what's linearly readable from att_num (chance = 50%) ===")
    for c in sorted(cards, key=lambda c: -c.decodability):
        if c.causal is True:
            tag = f"causal ✓ (Δ={c.delta:+.3f})"
        elif c.causal is False:
            tag = "decoded, not causal"
        else:
            tag = "decoded, causality untested"
        print(f"  {c.name:<22} {c.decodability*100:5.1f}%   [{c.pos_label} vs {c.neg_label}]   {tag}")

    runs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
    os.makedirs(runs, exist_ok=True)
    out = os.path.join(runs, "concept_atlas.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_concept_atlas(sorted(cards, key=lambda c: -c.decodability),
                                     subtitle="RWKV-4-169m · att_num (layer-mean) · 6-fold held-out"))
    print("\nwrote", out)
    print("the atlas maps what the recurrent state exposes — readable, with the causal claim gated.")


if __name__ == "__main__":
    main()
