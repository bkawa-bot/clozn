"""
Phase-3 — the full modern loop, unsupervised: DISCOVER a feature, then STEER it and VERIFY it's causal.

1. A tiny SAE discovers features on RWKV state (no labels).
2. We take a coherent one's *decoder direction* as a steering vector.
3. We add it to the live state and re-measure the model's own output (ops.verify_causal) — does
   pushing the discovered "color" feature make the model emit more color words?

This is the SOTA pattern (features → test causally by steering) run end-to-end on our own substrate.
Honest: 169M model, steering is brittle — we report the dose-response whatever it shows.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.discover import collect_token_states, corpus, sae_features, standardize, THEME_WORDS  # noqa: E402
from clozn.ops import verify_causal                  # noqa: E402
from clozn.sources.hf_rwkv import RwkvStateSource     # noqa: E402

# a neutral prime ending just before an adjective/word slot, per theme (prime, suffix)
PRIMES = {
    "color":   ("The wall was painted a", " very"),
    "animal":  ("At the farm we saw a", " single"),
    "food":    ("For dinner we ate some", " really"),
    "place":   ("Last summer we traveled to a", " quiet"),
    "emotion": ("When the news came in I felt", " so"),
    "number":  ("I counted them and there were", " about"),
}


def main():
    print("loading RWKV-4-169m, discovering features ...")
    src = RwkvStateSource()
    X, toks, _ = collect_token_states(src, corpus())
    Xs, mu, sd = standardize(X)
    feats, sae, _ = sae_features(Xs, toks, m=128, l1=4e-2, steps=800)

    # pick the most coherent discovered feature whose theme we have a prime for
    feat = next((f for f in feats if f.purity >= 0.6 and f.theme in PRIMES), None)
    if feat is None:
        print("no steerable coherent feature discovered this run"); return
    theme = feat.theme
    print(f"  steering discovered feature f{feat.idx} -> theme '{theme}' "
          f"(purity {feat.purity*100:.0f}%, top: {[t.strip() for t in feat.top_tokens[:5]]})")

    word_ids = np.array([src.tok.encode(' ' + w)[0] for w in THEME_WORDS[theme]
                         if len(src.tok.encode(' ' + w)) == 1])
    direction = sae.decoder_direction(feat.idx, sd)              # (768,) raw-state units
    typ = float(np.mean([np.linalg.norm(r) for r in X]))
    steer = direction * typ * 0.1                                # 1 alpha = 10% of state magnitude
    prime, suffix = PRIMES[theme]
    suffix_ids = src.encode(suffix)

    def intervene(alpha):
        def f(state):
            s = {k: v.copy() for k, v in state.items()}
            s["att_num"][0] = s["att_num"][0] + alpha * steer[:, None]   # broadcast over 12 layers
            return s
        return f

    def behavior(source):
        snap = source.get_state()
        for tid in suffix_ids:
            source.step(tid)
        probs = source._last_logits.softmax(-1)[0].detach().cpu().numpy()
        score = float(probs[word_ids].sum())                    # P(theme words) at the next slot
        source.set_state(snap)
        return score

    print(f"\n=== STEER the discovered '{theme}' feature, measure P({theme} words) ===")
    print(f"  prime={prime!r} + suffix={suffix!r};  readout = P of {len(word_ids)} {theme} words")
    src.reset(); src.feed(prime)
    res = verify_causal(src, intervene(3.0), behavior)
    print(f"  baseline P = {res['baseline']:.4f}   +steer P = {res['intervened']:.4f}   "
          f"Δ = {res['delta']:+.4f}   causal={res['causal']}")

    print("\n  dose-response (steer the discovered feature up/down):")
    alphas = [-4, -2, 0, 2, 4, 6]
    scores = []
    for a in alphas:
        src.reset(); src.feed(prime)
        src.set_state(intervene(a)(src.get_state()))
        scores.append(behavior(src))
    lo, hi = min(scores), max(scores)
    for a, sc in zip(alphas, scores):
        n = int(round((sc - lo) / (hi - lo + 1e-9) * 36))
        print(f"    a={a:+3d}  P({theme})={sc:.4f}  {'#'*n}")
    up = scores[-1] > scores[0]
    print(f"\n  -> steering the DISCOVERED feature {'raises' if up else 'does not raise'} P({theme}) "
          f"({'causal ✓' if up and (hi-lo) > 1e-3 else 'weak/brittle at 169M — honest'})")
    print("discover → steer → verify: the full modern loop, unsupervised, on our substrate.")


if __name__ == "__main__":
    main()
