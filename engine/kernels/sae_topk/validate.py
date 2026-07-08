"""Validate the CUDA sae_topk kernel against the numpy reference oracle (ROADMAP 3.3).

Generates fixed SAE pre-activation matrices and, for a sweep of (k, relu) cases, runs both
reference.sae_topk and the compiled CUDA kernel (sae_validate.exe) on the SAME bytes and
diffs them: selected feature indices must match EXACTLY (per row, ascending); values within
a float32-vs-float64 tolerance (the kernel reduces in float32, the reference in float64).

    python validate.py <path-to-sae_validate.exe>

Mirrors ../confidence_select/validate.py. The index match is the load-bearing assertion
(it is what a sparse code's reconstruction depends on); the value match is the eps check.
"""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from reference import sae_topk  # noqa: E402

TMP = Path(tempfile.gettempdir())
IN_BIN = TMP / "_sae_validate_pre.bin"
OUT_BIN = TMP / "_sae_validate_out.bin"

# (label, rows, n_features, seed, k, relu)
# A spread of shapes: small/large feature dims, small/large k, relu on and off, plus a
# row block with deliberate ties (constructed below for the "ties" case).
COMBOS = [
    ("relu  k=8   nF=512  ", 6, 512, 0, 8, True),
    ("relu  k=16  nF=4096 ", 8, 4096, 1, 16, True),
    ("relu  k=32  nF=16384", 4, 16384, 2, 32, True),
    ("relu  k=1   nF=512  ", 6, 512, 3, 1, True),
    ("signed k=8  nF=512  ", 6, 512, 4, 8, False),
    ("relu  k=64  nF=2048 ", 5, 2048, 5, 64, False),
    ("k>nF  k=20  nF=8    ", 4, 8, 6, 20, True),  # k exceeds n_features -> clamp+pad
]


def make_pre_acts(rows, n_features, seed):
    rng = np.random.default_rng(seed)
    # Standard-normal pre-acts, then shift so ~half are negative (exercises the ReLU gate).
    return (rng.standard_normal((rows, n_features)) - 0.1).astype(np.float32)


def run_one(exe, pre, k, relu):
    ROWS, NFEAT = pre.shape
    IN_BIN.write_bytes(struct.pack("<ii", ROWS, NFEAT) + pre.tobytes())

    ref = sae_topk(pre.astype(np.float64), k, relu=relu)

    subprocess.run([exe, str(IN_BIN), str(OUT_BIN), str(k), str(1 if relu else 0)],
                   check=True, stdout=subprocess.DEVNULL)
    data = OUT_BIN.read_bytes()
    n = ROWS * k
    idx = np.frombuffer(data[: 4 * n], dtype=np.int32).reshape(ROWS, k).astype(np.int64)
    val = np.frombuffer(data[4 * n : 8 * n], dtype=np.float32).reshape(ROWS, k)

    k_eff = ref.k_eff
    # Indices: exact match on the meaningful columns [:k_eff], per row.
    idx_ok = np.array_equal(ref.indices[:, :k_eff], idx[:, :k_eff])
    # Values: within float32-vs-float64 epsilon on the meaningful columns.
    dval = float(np.abs(ref.values[:, :k_eff] - val[:, :k_eff]).max()) if k_eff > 0 else 0.0
    return idx_ok, dval, idx_ok and dval < 1e-3


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python validate.py <path-to-sae_validate.exe>")
        return 1
    exe = sys.argv[1]

    print("CUDA sae_topk vs numpy reference  (per-row top-k over the feature dim)\n")
    all_ok = True
    for label, rows, nfeat, seed, k, relu in COMBOS:
        pre = make_pre_acts(rows, nfeat, seed)
        idx_ok, dval, ok = run_one(exe, pre, k, relu)
        all_ok &= ok
        print(f"  {label}  indices={'ok' if idx_ok else 'X'}  "
              f"val|d|={dval:.1e}  -> {'PASS' if ok else 'FAIL'}")

    # Explicit tie case: many equal values in a row -> tie rule (lower index wins) must match.
    tie = np.zeros((3, 64), dtype=np.float32)
    tie[0, :] = 1.0  # every feature equal: top-k must be the k LOWEST indices
    tie[1, 10:20] = 5.0  # a tied plateau of 10 features
    tie[2, ::2] = 2.0  # every even feature tied
    idx_ok, dval, ok = run_one(exe, tie, 8, True)
    all_ok &= ok
    print(f"  {'ties  k=8   nF=64   ':>20}  indices={'ok' if idx_ok else 'X'}  "
          f"val|d|={dval:.1e}  -> {'PASS' if ok else 'FAIL'}")

    IN_BIN.unlink(missing_ok=True)
    OUT_BIN.unlink(missing_ok=True)
    print("\nRESULT:", "ALL PASS -- CUDA sae_topk matches the reference" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
