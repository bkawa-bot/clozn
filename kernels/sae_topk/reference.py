"""SAE-sparsify top-k kernel — CPU numpy reference oracle (Clozn ROADMAP 3.3).

The on-device primitive for SAE / transcoder inference at scale. Given an SAE
*pre-activation* matrix ``[rows x n_features]`` (one row per token position, one
column per dictionary feature), keep for each row only its top-k features
(indices + values) and zero the rest — the SAE's **sparse code**.

This is the confidence-select top-k (``../confidence_select/reference.py``)
**repointed at the feature dimension** (ARCHITECTURE.md: "interp kernels =
repoint the confidence-select top-k at the feature dimension"). Same tie rule,
same "exact picks / epsilon values" parity contract, same numpy-only,
self-contained style so a CUDA port can be validated the same way.

Where it sits in SAE inference (the seam this op fills):

    activations [rows x d_model]
        --( encoder matmul  W_enc^T · acts + b_enc, then ReLU )-->
    pre_acts    [rows x n_features]          # dense, all features "lit"
        --( THIS kernel: per-row top-k over the FEATURE dim )-->
    sparse code [rows x k]  (indices + values)   # only k features survive
        --( decoder matmul  sum_j val_j · W_dec[:, idx_j] + b_dec )-->
    reconstruction [rows x d_model]

The encoder/decoder are dense cuBLAS GEMMs; the genuinely sparse, kernel-shaped
middle step is this top-k. ``n_features`` is large (4k-130k for real
dictionaries) and ``k`` is small (~16-128) — the regime the confidence-select
block-per-row structure already targets, just over features instead of vocab.

Semantics (pinned; the CUDA kernel must reproduce these exactly):

* **Top-k over features, per row.** For each row keep the ``min(k, n_features)``
  largest pre-activation values; everything else is implicitly zero.
* **Tie rule — toward the LOWER feature index.** Identical to the confidence
  kernel's argmax / ``_select_topk`` (``np.argsort(-x, kind="stable")`` keeps the
  original, lower index among equal values). Deterministic and device-matchable.
* **Selected indices returned ASCENDING** per row (mirrors
  ``confidence_select._select_topk``'s ``sorted(...)``), with the value array
  aligned to those indices. A reconstruction is order-independent, but a fixed
  order makes the parity check exact.
* **ReLU gate (optional, default ON).** A real SAE applies ReLU before the
  top-k, so negative pre-activations are not features. With ``relu=True`` a
  selected slot whose value is ``<= 0`` is emitted as a zero value (and rows with
  fewer than ``k`` positive features still emit ``k`` indices — the surplus carry
  value ``0.0`` — so the output is a fixed ``[rows x k]`` shape, the shape a GPU
  kernel writes). Set ``relu=False`` for a pure top-k over signed values.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SaeTopKResult:
    """The sparse code for a block of rows.

    Attributes:
        indices: shape ``(rows, k)`` int64; the selected feature indices per row,
            ascending. (When ``k > n_features`` only ``n_features`` columns are
            meaningful; see ``k_eff``.)
        values: shape ``(rows, k)`` float64; the pre-activation value at each
            selected index, aligned with ``indices``. Zeroed entries (ReLU-gated
            non-positive picks) are ``0.0``.
        k_eff: ``min(k, n_features)`` — the number of meaningful columns. Columns
            ``>= k_eff`` (only possible when ``k > n_features``) are padding.
    """

    indices: IntArray
    values: FloatArray
    k_eff: int

    def to_dense(self, n_features: int) -> FloatArray:
        """Scatter the sparse code back to a dense ``[rows, n_features]`` matrix.

        The exact input the decoder GEMM would otherwise consume densely — handy
        for tests and for a reference reconstruction. Implicitly-zero features
        stay zero.
        """
        rows = self.indices.shape[0]
        dense = np.zeros((rows, n_features), dtype=np.float64)
        for r in range(rows):
            for c in range(self.k_eff):
                dense[r, self.indices[r, c]] = self.values[r, c]
        return dense


def sae_topk(
    pre_acts: NDArray[np.floating],
    k: int,
    *,
    relu: bool = True,
) -> SaeTopKResult:
    """Per-row top-k over the feature dimension — the SAE sparse code (ROADMAP 3.3).

    Args:
        pre_acts: ``[rows, n_features]`` SAE pre-activations (post encoder matmul,
            pre-ReLU). Read, never mutated.
        k: features to keep per row. ``min(k, n_features)`` are meaningful. Must
            be >= 1.
        relu: when True (default, the real-SAE path) the top-k is taken over the
            ReLU'd values: a selected feature whose pre-activation is ``<= 0``
            contributes a ``0.0`` value (so it adds nothing to the reconstruction),
            while its index slot is still filled (fixed ``[rows, k]`` output). When
            False, the top-k is over the raw signed values.

    Returns:
        SaeTopKResult with per-row ascending feature indices and aligned values.

    Tie rule: equal values resolve toward the LOWER feature index (stable argsort
    on the negated values), matching the confidence-select kernel exactly.
    """
    pre_acts = np.asarray(pre_acts)
    if pre_acts.ndim != 2:
        raise ValueError(
            f"pre_acts must be 2-D [rows, n_features], got shape {pre_acts.shape}"
        )
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    rows, n_features = pre_acts.shape
    k_eff = min(k, n_features)

    # Selection runs over the ReLU'd values when gating: ReLU only ever lowers a
    # value, so it can change *which* features rank in the top-k (a strongly
    # negative feature is never selected over a positive one). The reference and
    # the kernel must therefore rank the SAME quantity. We keep the raw values
    # around to emit (a gated pick reports 0.0, not the negative raw value).
    raw = pre_acts.astype(np.float64)
    ranked = np.maximum(raw, 0.0) if relu else raw

    out_idx = np.empty((rows, k), dtype=np.int64)
    out_val = np.zeros((rows, k), dtype=np.float64)

    for r in range(rows):
        # Stable descending sort on the negated ranked values keeps the LOWER
        # feature index among ties — the confidence-select convention exactly.
        order = np.argsort(-ranked[r], kind="stable")
        top = order[:k_eff]
        top_sorted = np.sort(top)  # selected indices ascending (mirrors _select_topk)
        out_idx[r, :k_eff] = top_sorted
        # Emit the RANKED value (ReLU-gated when relu=True): a non-positive pick
        # reports 0.0 so the decoder sees no contribution from a "dead" feature.
        out_val[r, :k_eff] = ranked[r, top_sorted]
        # Pad columns (only when k > n_features) repeat the last index with a 0
        # value so a fixed-width output is well-defined; they carry no mass.
        if k_eff < k:
            out_idx[r, k_eff:] = top_sorted[-1] if k_eff > 0 else 0
            out_val[r, k_eff:] = 0.0

    return SaeTopKResult(indices=out_idx, values=out_val, k_eff=k_eff)
