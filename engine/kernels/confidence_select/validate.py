"""Validate the CUDA confidence_select kernel against the numpy reference oracle.

Generates one fixed set of logits and, for every *deterministic* path (greedy /
temperature=0, no top_p), runs both reference.confidence_select and the compiled CUDA
kernel (validate.exe) on the SAME bytes and diffs them: token picks and selected indices
must match exactly; confidences within a float32-vs-float64 tolerance (the kernel
softmaxes in float32, the reference in float64).

    python validate.py <path-to-validate.exe>

Covers the three confidence variants (max_prob / margin / neg_entropy) x top-k and
threshold selection. NOT covered (documented as scaffold, not asserted): the sampled
path uses curand and cannot bit-match numpy's RNG, and top_p is still a TODO stub.
"""

import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from reference import confidence_select  # noqa: E402

N, V, SEED = 6, 256, 0
TMP = Path(tempfile.gettempdir())
IN_BIN = TMP / "_cs_validate_logits.bin"
OUT_BIN = TMP / "_cs_validate_out.bin"

# (label, conf_kind int, conf str, mode int, k_commit, tau, min_commit)
COMBOS = [
    ("max_prob   / top-k(2)   ", 0, "max_prob", 0, 2, 0.0, 1),
    ("margin     / top-k(2)   ", 1, "margin", 0, 2, 0.0, 1),
    ("neg_entropy/ top-k(2)   ", 2, "neg_entropy", 0, 2, 0.0, 1),
    ("max_prob   / thresh(.05)", 0, "max_prob", 1, 0, 0.05, 1),
    ("margin     / thresh(.05)", 1, "margin", 1, 0, 0.05, 1),
    ("neg_entropy/ thresh(.05)", 2, "neg_entropy", 1, 0, 0.05, 1),
]


def run_one(exe, logits, conf_kind, conf, mode, k_commit, tau, min_commit):
    if mode == 0:  # top-k
        ref = confidence_select(logits.astype(np.float64), temperature=0.0, k_commit=k_commit, confidence=conf)
    else:  # threshold
        ref = confidence_select(logits.astype(np.float64), temperature=0.0, tau=tau, min_commit=min_commit, confidence=conf)
    subprocess.run([exe, str(IN_BIN), str(OUT_BIN), str(conf_kind), str(mode),
                    str(k_commit), str(tau), str(min_commit)], check=True,
                   stdout=subprocess.DEVNULL)
    data = OUT_BIN.read_bytes()
    tok = np.frombuffer(data[: 4 * N], dtype=np.int32)
    cf = np.frombuffer(data[4 * N : 8 * N], dtype=np.float32)
    nsel = struct.unpack("<i", data[8 * N : 8 * N + 4])[0]
    sel = np.frombuffer(data[8 * N + 4 : 8 * N + 4 + 4 * nsel], dtype=np.int32).tolist()
    tok_ok = np.array_equal(ref.token_ids, tok)
    sel_ok = list(ref.selected) == sel
    dconf = float(np.abs(ref.confidences - cf).max())
    return tok_ok, sel_ok, dconf, tok_ok and sel_ok and dconf < 1e-3


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python validate.py <path-to-validate.exe>")
        return 1
    exe = sys.argv[1]
    rng = np.random.default_rng(SEED)
    logits = rng.standard_normal((N, V)).astype(np.float32)
    IN_BIN.write_bytes(struct.pack("<ii", N, V) + logits.tobytes())

    print(f"CUDA confidence_select vs numpy reference  (N={N}, V={V}, seed={SEED}, greedy)\n")
    all_ok = True
    for label, conf_kind, conf, mode, k, tau, mc in COMBOS:
        tok_ok, sel_ok, dconf, ok = run_one(exe, logits, conf_kind, conf, mode, k, tau, mc)
        all_ok &= ok
        print(f"  {label}  picks={'ok' if tok_ok else 'X'}  "
              f"selected={'ok' if sel_ok else 'X'}  conf|d|={dconf:.1e}  -> {'PASS' if ok else 'FAIL'}")

    IN_BIN.unlink(missing_ok=True)
    OUT_BIN.unlink(missing_ok=True)
    print("\nRESULT:", "ALL PASS -- CUDA deterministic paths match the reference" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
