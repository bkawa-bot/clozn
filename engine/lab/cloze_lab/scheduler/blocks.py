"""Block manager and masks (DESIGN.md §5.4): semi-autoregressive block diffusion.

Generate the output left->right in blocks of length L. Within the active block,
diffusion runs over [prompt + frozen blocks ‖ active block]; finalized blocks
freeze. The attention is **block-causal** — the *one-way law* (DESIGN): a
position attends to its own block and every earlier one, never forward. That is
what keeps frozen-block K/V exact across later blocks (the basis for the Tier B
cache, §5.5): a frozen block's attention pattern — and so its K/V — is identical
whether or not later blocks exist.

``block_len = 0`` is whole-sequence mode: a single block, fully bidirectional
(prompt and output mutually attend), best for infill, no inter-block structure.
This module is pure logic; numpy is used only to build the boolean mask.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cloze_lab.models.base import BoolArray


@dataclass(frozen=True, slots=True)
class Block:
    """One output block: board positions [start, end)."""

    index: int
    start: int
    end: int

    @property
    def span(self) -> tuple[int, int]:
        return (self.start, self.end)


@dataclass(frozen=True, slots=True)
class BlockPlan:
    """Left-to-right blocks over the output region [prompt_len, prompt_len+max_new).

    ``block_len = 0`` => one whole-sequence block. ``block_len > 0`` => semi-AR
    blocks of that length (the last may be shorter).
    """

    prompt_len: int
    max_new: int
    block_len: int

    def __post_init__(self) -> None:
        if self.prompt_len < 1:
            raise ValueError(f"prompt_len must be >= 1, got {self.prompt_len}")
        if self.max_new < 1:
            raise ValueError(f"max_new must be >= 1, got {self.max_new}")
        if self.block_len < 0:
            raise ValueError(f"block_len must be >= 0 (0 = whole-sequence), got {self.block_len}")

    @property
    def whole_sequence(self) -> bool:
        return self.block_len == 0

    def blocks(self) -> list[Block]:
        start, end = self.prompt_len, self.prompt_len + self.max_new
        if self.block_len == 0:
            return [Block(index=0, start=start, end=end)]
        out: list[Block] = []
        pos = start
        while pos < end:
            out.append(Block(index=len(out), start=pos, end=min(pos + self.block_len, end)))
            pos += self.block_len
        return out


def block_id(pos: int, prompt_len: int, block_len: int) -> int:
    """-1 for prompt positions; 0, 1, 2, ... for successive output blocks."""
    if pos < prompt_len:
        return -1
    return (pos - prompt_len) // block_len


def attention_mask(working_len: int, prompt_len: int, block_len: int) -> BoolArray:
    """[working_len, working_len] bool mask; ``M[q, k]`` True means q may attend to k.

    whole-sequence (block_len=0): fully bidirectional (all True). Block mode:
    ``M[q, k] = block_id(k) <= block_id(q)`` — the prompt attends only to itself,
    each block attends to the prompt and all earlier blocks plus bidirectionally
    within itself, and nothing attends forward (the one-way law).
    """
    if block_len == 0:
        return np.ones((working_len, working_len), dtype=bool)
    ids = np.array(
        [block_id(p, prompt_len, block_len) for p in range(working_len)], dtype=np.int64
    )
    return ids[None, :] <= ids[:, None]  # M[q, k] = ids[k] <= ids[q]
