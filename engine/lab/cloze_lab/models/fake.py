"""FakeAdapter — the lab's oracle model: deterministic, torch-free, contract-strict.

Logits and K/V are seeded hashes, not learned values, but the *contract* is
faithful. Each position's drawer carries a **context fingerprint**: a stable
hash of the position and the tokens it can see through ``attn_mask`` at the
moment the drawer was built. A position's logits are then a function of the
fingerprints of its visible neighbors' drawers. Two consequences this buys, both
needed to test the cache (DESIGN §5.5) without a checkpoint:

* **Tier C drift is real.** When a neighbor's token changes, a still-unchanged
  position's *true* K/V should shift (bidirectional attention let it see the
  neighbor). The delta cache reuses that position's stale drawer anyway — and
  here the stale drawer carries an old fingerprint, so logits reading it diverge
  from a full recompute. Exactly the approximation §5.5 describes.
* **Tier A/B stay exact.** A prompt or frozen-block position only ever sees other
  frozen positions (the one-way law), so its fingerprint never changes — reusing
  its drawer is exact, no drift.

Mask bugs, cache bookkeeping bugs, and the off-vs-delta divergence are therefore
all observable in seeded tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from cloze_lab.models.base import (
    BoolArray,
    Family,
    FloatArray,
    ForwardResult,
    IntArray,
    KVState,
    ModelConfig,
    check_attn_mask,
    check_board,
    check_indices,
)

# Domain-separation tags so logits and K/V fingerprints never share an RNG stream.
_LOGITS_TAG = 1
_KV_TAG = 2

# Real dLLM adapters put ~0 mass on the mask sentinel (the model is trained to
# fill, never to emit MASK). The fake mirrors that by pinning the mask logit far
# below any sampled value, so it is never argmax'd and contributes ~0 probability
# — otherwise "committing" MASK would leave a slot masked and stall the loop.
_MASK_SUPPRESS = -1e30


@dataclass(frozen=True, slots=True, eq=False)
class _Drawer:
    built_as: int  # the token this position's K/V was computed under (the built-as label)
    fp: int  # context fingerprint: hash(position, visible-neighbor tokens) at build time
    vec: FloatArray  # the K/V vector attention would read (derived from fp)


@dataclass(frozen=True, slots=True, eq=False)
class FakeKV:
    """Concrete drawers for FakeAdapter; only the fake and its tests look inside."""

    entries: dict[int, _Drawer]  # position -> drawer

    @property
    def seq_len(self) -> int:
        return len(self.entries)


class FakeAdapter:
    """Implements ModelAdapter without torch. See module docstring for semantics."""

    def __init__(self, seed: int = 0, vocab_size: int = 64, kv_dim: int = 8) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative (it feeds numpy SeedSequence)")
        if vocab_size < 8:
            raise ValueError("vocab_size too small to leave room for special tokens")
        self._seed = seed
        self._kv_dim = kv_dim
        self._config = ModelConfig(
            family=Family.FAKE,
            vocab_size=vocab_size,
            mask_token_id=vocab_size - 1,
            eos_token_id=1,
            default_steps=8,
            block_length=8,
        )

    @property
    def config(self) -> ModelConfig:
        return self._config

    def forward(
        self,
        board: IntArray,
        attn_mask: BoolArray,
        *,
        kv: KVState | None = None,
        recompute_kv: Sequence[int] | None = None,
        logits_for: Sequence[int] | None = None,
    ) -> ForwardResult:
        board = check_board(board, self._config.vocab_size)
        n = board.shape[0]
        attn_mask = check_attn_mask(attn_mask, n)

        if kv is None:
            if recompute_kv is not None:
                raise ValueError("recompute_kv given without kv: nothing to reuse")
            old: dict[int, _Drawer] = {}
        else:
            if not isinstance(kv, FakeKV):
                raise TypeError(f"FakeAdapter got foreign KVState {type(kv).__name__}")
            old = kv.entries
            if not set(old) <= set(range(n)):
                raise ValueError("kv covers positions outside the current board")

        fresh = set(range(n)) if recompute_kv is None else set(
            check_indices("recompute_kv", recompute_kv, n)
        )
        missing = [p for p in range(n) if p not in fresh and p not in old]
        if missing:
            raise ValueError(f"positions {missing} neither recomputed nor present in kv")

        # A fresh drawer captures the position's CURRENT visible context; a reused
        # drawer keeps its old fingerprint (possibly stale w.r.t. a changed neighbor).
        entries: dict[int, _Drawer] = {}
        for p in range(n):
            if p in fresh:
                fp = self._fingerprint(p, board, attn_mask[p])
                entries[p] = _Drawer(built_as=int(board[p]), fp=fp, vec=self._vec(fp))
            else:
                entries[p] = old[p]

        want = list(range(n)) if logits_for is None else check_indices(
            "logits_for", logits_for, n
        )
        vocab = self._config.vocab_size
        logits = np.empty((len(want), vocab), dtype=np.float32)
        for row, p in enumerate(want):
            logits[row] = self._logits_at(p, attn_mask[p], entries)
        return ForwardResult(logits=logits, kv=FakeKV(entries))

    def encode(self, text: str) -> list[int]:
        usable = self._config.vocab_size - 3  # keep 0, eos, and mask out of reach
        return [2 + b % usable for b in text.encode("utf-8")]

    def decode(self, ids: Sequence[int]) -> str:
        mask = self._config.mask_token_id
        return "".join("░" if int(i) == mask else f"<{int(i)}>" for i in ids)

    def _fingerprint(self, p: int, board: IntArray, row_mask: BoolArray) -> int:
        """Stable hash of (position, the tokens it can see now) — the drawer's context."""
        entropy = [self._seed, _KV_TAG, p]
        for q in np.flatnonzero(row_mask):
            entropy += [int(q), int(board[q])]
        return int(np.random.SeedSequence(entropy).generate_state(1, dtype=np.uint32)[0])

    def _vec(self, fp: int) -> FloatArray:
        return np.random.default_rng([fp]).standard_normal(self._kv_dim).astype(np.float32)

    def _logits_at(self, p: int, row_mask: BoolArray, drawers: dict[int, _Drawer]) -> FloatArray:
        # Attention reads drawers: logits depend on the fingerprints of visible
        # neighbors, so a stale neighbor drawer (old context) shifts these logits.
        entropy = [self._seed, _LOGITS_TAG, p]
        for q in np.flatnonzero(row_mask):
            entropy += [int(q), drawers[int(q)].fp]
        logits = np.random.default_rng(entropy).standard_normal(
            self._config.vocab_size
        ).astype(np.float32)
        logits[self._config.mask_token_id] = _MASK_SUPPRESS  # never emit the mask sentinel
        return logits
