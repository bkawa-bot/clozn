"""
Phase-1 M3 — Probe + Verify on a REAL recurrent model (RWKV-4 via transformers).

Thin driver over clozn.probes.probe_and_verify (the shared M3 method). Two questions, kept
strictly separate (the honesty guardrail):

  (A) DECODABILITY — is a sentiment direction linearly *readable* from the recurrent memory?
  (B) CAUSALITY    — does the model actually *use* it? (diff-in-means steer + verify_causal,
                     swept over alpha; a monotonic dose-response is the real evidence.)

Whatever the numbers say, we report them — 169M model, steering is brittle by our own guardrail.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from clozn.probes import probe_and_verify             # noqa: E402
from clozn.sources.hf_rwkv import RwkvStateSource     # noqa: E402
from clozn.viz import render_probe_panel              # noqa: E402


def main():
    print("loading RWKV-4-169m (cached) ...")
    src = RwkvStateSource()
    r = probe_and_verify(src, name="sentiment")

    print("\n=== (A) DECODABILITY — is sentiment linearly readable from att_num? ===")
    print(f"  6-fold held-out accuracy = {r.decodability*100:.1f}%   (chance = 50%)")
    print("  -> sentiment is linearly decodable from the recurrent state."
          if r.decodability > 0.7 else "  -> weak/no linear decodability at this size.")

    print("\n=== (B) CAUSALITY — does the model USE that direction? ===")
    print(f"  baseline balance        = {r.verify['baseline']:.4f}")
    print(f"  +steer                  = {r.verify['intervened']:.4f}")
    print(f"  delta                   = {r.verify['delta']:+.4f}   causal={r.verify['causal']}")

    print("\n=== dose-response: sweep steering alpha (1 unit = 10% state magnitude) ===")
    lo, hi = min(r.scores), max(r.scores)
    for a, sc in zip(r.alphas, r.scores):
        n = int(round((sc - lo) / (hi - lo + 1e-9) * 40))
        print(f"  a={a:+5.1f}  balance={sc:.4f}  {'·'*n}")
    print(f"\n  monotonic with +sentiment steering: {r.monotonic}")
    print("  -> the recurrent memory direction is CAUSAL for sentiment output."
          if r.causal and (hi - lo) > 1e-3 else
          "  -> decodable but weakly/non-causal at this scale (honest result; steering is brittle).")

    runs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
    os.makedirs(runs, exist_ok=True)
    out = os.path.join(runs, "probe_verify.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_probe_panel(r.alphas, r.scores, r.decodability, r.verify,
                                   concept="sentiment", subtitle="RWKV-4-169m · att_num · prime 'I think it was'"))
    print("\nwrote", out)
    print("M3 ✓  probe + verify primitive works on real recurrent state.")


if __name__ == "__main__":
    main()
