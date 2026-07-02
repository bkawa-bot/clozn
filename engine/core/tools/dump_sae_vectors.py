"""dump_sae_vectors.py -- torch-side parity vectors for the engine's SaeEncoder (tests/test_sae_encoder.cpp).

Computes the JumpReLU encode EXACTLY as research/sae7b.py GpuSAE.encode does (fp16 GEMM on the GPU,
fp16 bias add, fp32 widen, relu * (hp > threshold)) on a small deterministic input batch, and dumps:

    x.f32.bin            [rows, d_in]   the input activations (what the engine tap would hand over)
    ref_gated.f32.bin    [rows, d_sae]  the full gated feature matrix (the strongest receipt: the
                                        C++ side diffs ALL 131k features, not just the selected k)
    ref_topk_idx.i32.bin [rows, k]      top-k per row by kernels/sae_topk/reference.py semantics
    ref_topk_val.f32.bin [rows, k]        (indices ascending, tie -> lower index, zero-padded)
    manifest.txt                        key/value pairs the C++ test parses

Model-free by design (the "stronger, cheaper receipt"): no 7B forward is needed for NUMERICS parity.
Realistic sparsity still matters, though -- pure Gaussian rows barely clear the learned thresholds --
so half the rows are built ON the SAE's own manifold: b_dec + a few W_dec feature directions at
typical activation magnitudes (which the encoder then re-detects), plus Gaussian rows as the
random-direction stress case, plus optional REAL residual rows via --acts <npy> ([n, d_in] fp32,
e.g. harvested from the HF model or the engine's /harvest).

    python dump_sae_vectors.py [--pt ~/hf_models/andyrdt_l15_sae.pt] [--out ~/.clozn/sae/andyrdt_l15/vectors]
                               [--k 32] [--acts real_acts.npy]
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "kernels", "sae_topk"))
from reference import sae_topk as reference_topk  # noqa: E402  (the pinned numpy oracle)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def build_inputs(d, rng):
    """8 deterministic rows: 4 on-manifold (b_dec + W_dec directions at real magnitudes), 4 Gaussian."""
    d_in = int(d["d_in"])
    b_dec = d["b_dec"].float().numpy()
    W_dec = d["W_dec"]  # [d_sae, d_in] fp16 cpu
    rows = []
    # On-manifold rows: a handful of feature directions each, activation values in the 3..25 range
    # brain_readout sees on real text, plus mild noise so nothing is exactly axis-aligned.
    for j, (feats, vals) in enumerate([
        ([123, 4567, 89012], [8.0, 15.0, 5.0]),
        ([31337, 100000, 42, 77777], [22.0, 3.5, 11.0, 6.0]),
        ([2048, 65536, 130000, 999, 55555, 12345], [4.0, 9.0, 18.0, 2.5, 7.0, 12.0]),
        ([500], [25.0]),
    ]):
        x = b_dec.copy()
        for f, v in zip(feats, vals):
            x += v * W_dec[f].float().numpy()
        x += 0.05 * rng.standard_normal(d_in).astype(np.float32)
        rows.append(x)
    # Gaussian rows at residual-ish L2 norms (random-direction stress; mostly sub-threshold).
    for norm in (20.0, 45.0, 80.0, 120.0):
        g = rng.standard_normal(d_in).astype(np.float32)
        rows.append(g * (norm / np.linalg.norm(g)))
    return np.stack(rows).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", default=os.path.expanduser("~/hf_models/andyrdt_l15_sae.pt"))
    ap.add_argument("--out", default=os.path.expanduser("~/.clozn/sae/andyrdt_l15/vectors"))
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--acts", default=None, help="optional [n, d_in] fp32 .npy of REAL residual rows to append")
    args = ap.parse_args()

    d = torch.load(args.pt, map_location="cpu")
    x = build_inputs(d, np.random.default_rng(20260702))
    if args.acts:
        real = np.load(args.acts).astype(np.float32)[:8]
        assert real.shape[1] == int(d["d_in"]), f"--acts d_in {real.shape[1]} != {int(d['d_in'])}"
        x = np.concatenate([x, real], axis=0)
    rows = x.shape[0]

    # The oracle, verbatim from research/sae7b.py GpuSAE.encode (fp16 all the way to the widen).
    W_enc = d["W_enc"].to(DEV)
    b_enc = d["b_enc"].to(DEV)
    b_dec = d["b_dec"].to(DEV)
    threshold = d["threshold"].to(DEV)
    with torch.no_grad():
        xt = torch.tensor(x, device=DEV)
        hp = ((xt.half() - b_dec) @ W_enc + b_enc).float()
        gated = (torch.relu(hp) * (hp > threshold).float()).cpu().numpy().astype(np.float32)

    ref = reference_topk(gated.astype(np.float64), args.k, relu=True)

    os.makedirs(args.out, exist_ok=True)
    x.tofile(os.path.join(args.out, "x.f32.bin"))
    gated.tofile(os.path.join(args.out, "ref_gated.f32.bin"))
    ref.indices.astype(np.int32).tofile(os.path.join(args.out, "ref_topk_idx.i32.bin"))
    ref.values.astype(np.float32).tofile(os.path.join(args.out, "ref_topk_val.f32.bin"))
    with open(os.path.join(args.out, "manifest.txt"), "w") as f:
        f.write(f"rows {rows}\nd_in {int(d['d_in'])}\nd_sae {int(d['d_sae'])}\nk {args.k}\n"
                "x x.f32.bin\nref_gated ref_gated.f32.bin\n"
                "ref_idx ref_topk_idx.i32.bin\nref_val ref_topk_val.f32.bin\n")

    nnz = (gated > 0).sum(1)
    print(f"dumped {rows} rows -> {args.out}  (device={DEV})")
    print(f"  nnz/row: {nnz.tolist()}")
    best = ref.values.argmax(1)  # indices are ascending per row; find the max-VALUE slot
    print(f"  top-1 per row: {[(int(ref.indices[r, c]), round(float(ref.values[r, c]), 2)) for r, c in enumerate(best)]}")


if __name__ == "__main__":
    main()
