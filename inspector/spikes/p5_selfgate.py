"""
Self-gate for p5_transcoder_scale: BEFORE the big run, prove
  (1) a tiny TWO-LAYER /harvest gives token-aligned [n,896] matrices at two DIFFERENT layers, and
  (2) a tiny transcoder (input L -> sparse code -> reconstruct output L') trains: MSE drops, features live.

Run from inspector/ with the cloze venv python. Launches the engine itself (PATH must already include
the engine bin + CUDA v13.3 bin\\x64 if you run the server separately; this script sets PATH for the
child it spawns). Reuses p4_big_sae's server management + harvest_text, and discover.TinySAE as the
transcoder (Y != X). Exits non-zero if either gate fails.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.discover import TinySAE, standardize  # noqa: E402
import spikes.p4_big_sae as big  # noqa: E402  (reuse start_server / kill_server / harvest_text)

L_IN = 2     # input layer (the engine's default tap; l_out-2)
L_OUT = 6    # output layer a few residual steps downstream (l_out-6); both valid mid-layer taps (1..23)


def harvest_two_layers(text: str):
    """Harvest the SAME text at L_IN and L_OUT. The forward is deterministic, so the two matrices are
    token-aligned (row r is the same token in both). Returns (Xin, Xout, pieces)."""
    a_in, p_in, lyr_in = big.harvest_text(text, layer=L_IN)
    a_out, p_out, lyr_out = big.harvest_text(text, layer=L_OUT)
    assert p_in == p_out, "token pieces differ between the two harvests — forward not deterministic?"
    assert a_in.shape == a_out.shape, f"shape mismatch {a_in.shape} vs {a_out.shape}"
    assert lyr_in == L_IN and lyr_out == L_OUT, f"layers came back as {lyr_in},{lyr_out}"
    return a_in, a_out, p_in


def main() -> int:
    print("=== SELF-GATE: two-layer harvest + tiny transcoder ===", flush=True)
    big.start_server()
    try:
        text = ("The mitochondria is the powerhouse of the cell. Paris is the capital of France, "
                "and the river Seine flows through it. Seven plus eight equals fifteen.")
        Xin, Xout, pieces = harvest_two_layers(text)
        print(f"  GATE 1 ok: L_in={L_IN} L_out={L_OUT}  Xin={Xin.shape}  Xout={Xout.shape}  "
              f"n_tokens={len(pieces)}", flush=True)
        # The two layers must actually DIFFER (else the transcoder is trivial identity) but be aligned.
        diff = float(np.mean(np.abs(Xin - Xout)))
        cos = float((Xin * Xout).sum() / (np.linalg.norm(Xin) * np.linalg.norm(Xout) + 1e-9))
        print(f"  layers differ: mean|Xin-Xout|={diff:.3f}, global cos={cos:.3f}  "
              f"(want diff>0 so the in->out map is non-trivial)", flush=True)
        print(f"  first 8 token pieces: {pieces[:8]}", flush=True)
    finally:
        big.kill_server()

    if diff < 1e-3:
        print("  GATE 1 FAIL: the two layers are identical — pick a different L_OUT.", flush=True)
        return 1

    # GATE 2: a tiny transcoder. Harvest is only ~30 tokens; to actually exercise training, gather a
    # few short sentences. But the gate's POINT is "does the machinery train" — so reuse the cached big
    # L=2 matrix as INPUT and a cheap linear-ish target won't test in->out. Instead, do a small REAL
    # two-layer harvest over a handful of sentences and train a tiny transcoder on it.
    big.start_server()
    try:
        sents = [
            "The dog chased the cat across the green yard while the children laughed loudly.",
            "Paris and Rome are large European cities full of old museums and busy streets.",
            "She counted one two three four five six seven eight nine and finally ten apples.",
            "The chocolate cake was warm and sweet and everyone wanted a second large slice.",
            "A wide river flows quietly through the dark forest under the tall ancient trees.",
            "He felt happy and excited about the long summer journey across the open desert.",
        ]
        Xin_list, Xout_list, pieces_all = [], [], []
        for s in sents:
            ai, ao, pc = harvest_two_layers(s)
            Xin_list.append(ai); Xout_list.append(ao); pieces_all.extend(pc)
        Xin = np.concatenate(Xin_list, 0)
        Xout = np.concatenate(Xout_list, 0)
    finally:
        big.kill_server()

    print(f"\n  GATE 2 corpus: {Xin.shape[0]} tokens x {Xin.shape[1]} dims (two layers, aligned)",
          flush=True)
    # Standardize input and output independently (transcoder reconstructs the OUTPUT distribution).
    Xin_s, _, _ = standardize(Xin)
    Xout_s, _, sd_out = standardize(Xout)
    d = Xin_s.shape[1]

    # Baseline: variance of the (standardized) target — a transcoder must beat MSE ~= 1.0 (predicting
    # the mean) to be doing anything. Also report the trivial "copy input" MSE (in->out are different
    # layers, so copying the standardized input is NOT free).
    var_target = float(np.mean(Xout_s ** 2))                      # ~1.0 by construction (standardized)
    copy_mse = float(np.mean((Xin_s - Xout_s) ** 2))
    print(f"  target variance (standardized) = {var_target:.3f}; copy-input MSE = {copy_mse:.3f}",
          flush=True)

    # Tiny transcoder: TinySAE with Y=Xout_s (in->out), small m, a few hundred steps.
    m = 4 * d  # 4x expansion (tiny)
    tc = TinySAE(d, m, l1=1e-3, seed=0).fit(Xin_s, steps=400, lr=3e-3, Y=Xout_s)
    mse = tc.recon_error(Xin_s, Y=Xout_s)
    codes = tc.codes(Xin_s)
    fire = (codes > 1e-6).mean(0)
    n_live = int(((fire >= 0.002) & (fire <= 0.5)).sum())
    print(f"  tiny transcoder: m={m} (4x), 400 steps -> MSE={mse:.3f} "
          f"(target var {var_target:.3f}), live feats={n_live}/{m}, mean fire={fire.mean()*100:.1f}%",
          flush=True)

    ok = (mse < 0.9 * var_target) and (n_live > 0)
    if ok:
        print(f"\n  GATE 2 ok: transcoder MSE ({mse:.3f}) << target var ({var_target:.3f}) and "
              f"{n_live} live features. Machinery trains.", flush=True)
        print("\n=== SELF-GATE PASSED — clear to run the big transcoder. ===", flush=True)
        return 0
    print(f"\n  GATE 2 FAIL: MSE={mse:.3f} vs var={var_target:.3f}, live={n_live}.", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
