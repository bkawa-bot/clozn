"""ModelAdapter interface — THE seam between pure scheduler logic and model checkpoints.

DESIGN invariants this file encodes:

1. *ModelAdapter seam.* Everything under ``scheduler/`` is pure logic against this
   interface: board tokens + attention mask + cached KV in -> logits + new KV out.
   torch/transformers imports are allowed only under ``cloze_lab/models/``; this
   module itself depends on numpy alone, so scheduler code may import these types.
4. *Scheduler writes tokens; model writes KV.* The adapter never mutates the board;
   the scheduler never inspects or fabricates a ``KVState`` — it shuttles the handle
   back in and says which positions to rebuild (``recompute_kv``).

``ModelConfig`` mirrors the dLLM GGUF metadata of DESIGN.md §4.2 one-for-one, so the
future C++ runtime consumes the same fields this lab does.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float32]


class Family(StrEnum):
    """Model family (GGUF: ``diffusion.family``); keys EOS rules and adapter quirks."""

    DREAM = "dream"
    LLADA = "llada"
    LLADA2_MOE = "llada2_moe"
    RND1 = "rnd1"
    FAKE = "fake"  # lab-only test adapter; never a real GGUF value


class AttnKind(StrEnum):
    """Attention structure (GGUF: ``diffusion.attn``)."""

    BIDIRECTIONAL = "bidirectional"
    BLOCK_CAUSAL = "block_causal"


class Schedule(StrEnum):
    """Recommended default unmask policy (GGUF: ``diffusion.schedule``, DESIGN §5.2)."""

    CONFIDENCE_TOPK = "confidence_topk"
    THRESHOLD = "threshold"
    ENTROPY = "entropy"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Per-model diffusion metadata, one field per DESIGN.md §4.2 GGUF key."""

    family: Family
    vocab_size: int
    mask_token_id: int
    eos_token_id: int | None = None
    default_steps: int = 64
    block_length: int = 32  # 0 = whole-sequence mode (DESIGN §5.4)
    schedule: Schedule = Schedule.CONFIDENCE_TOPK
    attn: AttnKind = AttnKind.BIDIRECTIONAL

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError(f"mask_token_id {self.mask_token_id} outside vocab")
        if self.eos_token_id is not None:
            if not 0 <= self.eos_token_id < self.vocab_size:
                raise ValueError(f"eos_token_id {self.eos_token_id} outside vocab")
            if self.eos_token_id == self.mask_token_id:
                raise ValueError("eos_token_id must differ from mask_token_id")
        if self.default_steps <= 0:
            raise ValueError(f"default_steps must be positive, got {self.default_steps}")
        if self.block_length < 0:
            raise ValueError(f"block_length must be >= 0, got {self.block_length}")


@dataclass(frozen=True, slots=True)
class LoadConfig:
    """How to load a checkpoint; consumed by the HF adapters, opaque to the scheduler."""

    model_id: str  # HF repo id or local path
    device: str = "cpu"  # "cpu" | "cuda" | "mps"
    dtype: str = "float32"  # torch dtype name as a string, so this module stays torch-free
    revision: str | None = None
    trust_remote_code: bool = True  # Dream/LLaDA repos ship custom modeling code


@runtime_checkable
class KVState(Protocol):
    """Opaque per-position drawers (K/V), owned entirely by the adapter that made them.

    The scheduler may hold a KVState and hand it back, never look inside; the
    ``cached_token`` built-as labels that reconcile board and drawers live in the
    scheduler's cache manager (DESIGN §5.5), not here.
    """

    @property
    def seq_len(self) -> int:
        """Number of positions these drawers cover."""
        ...


@dataclass(frozen=True, slots=True)
class ForwardResult:
    """One pass: logits for the requested positions plus complete drawers for the board."""

    logits: FloatArray  # [len(logits_for) or seq, vocab] float32, row-aligned to the request
    kv: KVState  # drawers for every board position (reused entries carried over verbatim)


def check_board(board: IntArray, vocab_size: int) -> IntArray:
    """Validate a board (1-D, integer, in-vocab) and return it as int64."""
    board = np.asarray(board)
    if board.ndim != 1:
        raise ValueError(f"board must be 1-D, got shape {board.shape}")
    if not np.issubdtype(board.dtype, np.integer):
        raise ValueError(f"board must be integer-typed, got {board.dtype}")
    if board.size and (board.min() < 0 or board.max() >= vocab_size):
        raise ValueError("board contains token ids outside vocab")
    return board.astype(np.int64)


def check_attn_mask(attn_mask: BoolArray, n: int) -> BoolArray:
    """Validate an attention mask (bool, [n, n]) and return it as an ndarray."""
    attn_mask = np.asarray(attn_mask)
    if attn_mask.dtype != np.bool_:
        raise ValueError(f"attn_mask must be bool, got {attn_mask.dtype}")
    if attn_mask.shape != (n, n):
        raise ValueError(f"attn_mask shape {attn_mask.shape} != ({n}, {n})")
    return attn_mask


def check_indices(name: str, idxs: Sequence[int], n: int) -> list[int]:
    """Validate a position list (in-board, sorted, unique) and return it as ints."""
    out = [int(i) for i in idxs]
    if any(not 0 <= i < n for i in out):
        raise ValueError(f"{name} contains positions outside the board")
    if out != sorted(set(out)):
        raise ValueError(f"{name} must be sorted and unique")
    return out


@runtime_checkable
class ModelAdapter(Protocol):
    """The seam. One sequence at a time, no batch dimension in the lab.

    Contract for ``forward`` (invariant 4): the adapter reads the board, never writes
    it; it computes drawers only for the positions it is told to, and the returned
    ``ForwardResult.kv`` must cover every board position. Reused drawers contribute
    the K/V of the token they were *built as* — when the board changed underneath
    them, the logits lawfully reflect the stale value. That drift is the Tier C
    approximation (DESIGN §5.5); bounding it is the scheduler's job, not the model's.
    """

    @property
    def config(self) -> ModelConfig:
        """Diffusion metadata for the loaded checkpoint."""
        ...

    def forward(
        self,
        board: IntArray,
        attn_mask: BoolArray,
        *,
        kv: KVState | None = None,
        recompute_kv: Sequence[int] | None = None,
        logits_for: Sequence[int] | None = None,
    ) -> ForwardResult:
        """Run one pass over the board.

        Args:
            board: [seq] token ids, masks included as ``mask_token_id``. Never mutated.
            attn_mask: [seq, seq] bool; ``attn_mask[q, k]`` True means position q may
                attend to position k. The scheduler builds it (the one-way law lives
                there, not here).
            kv: drawers from a previous pass over this board, or None for a cold start.
            recompute_kv: sorted, unique positions whose drawers must be rebuilt from
                the current board. None means all of them (cold start / full refresh).
                Every position *not* listed must already exist in ``kv``; passing
                recompute_kv without kv is an error.
            logits_for: sorted, unique positions whose logits are wanted (typically
                the masked ones). None means all positions.

        Returns:
            ForwardResult with float32 logits rows aligned to ``logits_for`` order and
            a complete KVState for the board.
        """
        ...

    def encode(self, text: str) -> list[int]:
        """Tokenize text to ids (tokenizers live with checkpoints, behind the seam)."""
        ...

    def decode(self, ids: Sequence[int]) -> str:
        """Detokenize ids to text; must accept any id in vocab, including the mask."""
        ...
