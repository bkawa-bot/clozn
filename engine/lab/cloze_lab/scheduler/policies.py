"""Unmask/commit policies (DESIGN.md §5.2): which candidates get inked this pass.

Policies are pure logic — no numpy, no torch: candidates in, selection out.
``confidence_topk`` is build-order step 3; ``threshold`` lands with the adaptive
stepper (step 7, the natural pairing — DESIGN §5.2/§5.3); ``entropy`` and
``remask_lowconf`` follow.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class Candidate:
    """One masked position's sampled proposal: token_id at pos, with confidence."""

    pos: int
    token_id: int
    confidence: float


@dataclass(frozen=True, slots=True)
class StepContext:
    """What a policy may consult when selecting — DESIGN §5.2's (step, block_state).

    Carries the step index and, under fixed(T) stepping, the total step budget;
    block-state fields join here when the block manager lands (§5.4).
    """

    step: int
    steps_total: int | None = None

    def __post_init__(self) -> None:
        if self.step < 0:
            raise ValueError(f"step must be >= 0, got {self.step}")
        if self.steps_total is not None and not self.step < self.steps_total:
            raise ValueError(
                f"step {self.step} outside the fixed budget of {self.steps_total} steps"
            )

    @property
    def steps_remaining(self) -> int | None:
        """Steps left including this one; None when no fixed budget is set."""
        return None if self.steps_total is None else self.steps_total - self.step


@dataclass(frozen=True, slots=True)
class Selection:
    """The policy's verdict for one pass, both tuples pos-ascending.

    ``commit``: candidates to ink. ``revise``: already-committed positions to
    re-mask — always empty until the remask_lowconf policy exists (§5.2).
    """

    commit: tuple[Candidate, ...]
    revise: tuple[Candidate, ...] = ()


@runtime_checkable
class UnmaskPolicy(Protocol):
    """DESIGN §5.2 interface: select(candidates, step/block context) -> selection.

    Contract: ``candidates`` are this pass's masked positions (unique pos, any
    order); the returned ``commit`` is a subset of them, pos-ascending; selection
    is deterministic — same inputs, same picks (golden tests rely on it).
    """

    def select(self, candidates: Sequence[Candidate], ctx: StepContext) -> Selection: ...


def _check_unique(candidates: Sequence[Candidate]) -> None:
    positions = [c.pos for c in candidates]
    if len(set(positions)) != len(positions):
        raise ValueError("candidates contain duplicate positions")


def _by_confidence(candidates: Sequence[Candidate]) -> list[Candidate]:
    """Most confident first; confidence ties break toward the lower position so
    picks are exact on every platform."""
    return sorted(candidates, key=lambda c: (-c.confidence, c.pos))


def _commit(selected: Sequence[Candidate]) -> Selection:
    return Selection(commit=tuple(sorted(selected, key=lambda c: c.pos)))


@dataclass(frozen=True, slots=True)
class ConfidenceTopK:
    """``confidence_topk`` (§5.2 default): ink the k most confident candidates.

    k=None (default) inks an even quota, ceil(n_masked / steps_remaining): never
    zero while masks remain, and exactly drains the board within a fixed(T)
    budget — the natural k-ramp. A fixed integer k inks min(k, n_masked) per
    pass and leaves termination to the caller's rails. Confidence ties break
    toward the lower position so picks are exact on every platform.
    """

    k: int | None = None

    def __post_init__(self) -> None:
        if self.k is not None and self.k < 1:
            raise ValueError(f"k must be >= 1 (or None for quota mode), got {self.k}")

    def select(self, candidates: Sequence[Candidate], ctx: StepContext) -> Selection:
        _check_unique(candidates)
        if not candidates:
            return Selection(commit=())

        if self.k is None:
            if ctx.steps_remaining is None:
                raise ValueError("quota mode (k=None) requires ctx.steps_total")
            take = ceil(len(candidates) / ctx.steps_remaining)
        else:
            take = self.k
        take = min(take, len(candidates))

        return _commit(_by_confidence(candidates)[:take])


@dataclass(frozen=True, slots=True)
class Threshold:
    """``threshold(tau)`` (§5.2): ink every candidate with confidence >= tau.

    The natural partner for the adaptive stepper (§5.3): commit whatever has
    cleared the bar this pass and let the stepper finish the block once the board
    drains. ``min_commit`` is the min-one-commit progress rail — if fewer than
    ``min_commit`` candidates clear tau, ink the most confident ones anyway, so a
    pass never stalls by committing nothing (which, since the model is a pure
    function of the unchanged board, would repeat forever up to T_max). Unlike
    quota mode, this is step-budget-free, so it composes with adaptive stepping
    (which supplies no steps_total).
    """

    tau: float
    min_commit: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau <= 1.0:
            raise ValueError(f"tau must be in [0, 1], got {self.tau}")
        if self.min_commit < 1:
            raise ValueError(f"min_commit must be >= 1 (0 permits stalls), got {self.min_commit}")

    def select(self, candidates: Sequence[Candidate], ctx: StepContext) -> Selection:
        _check_unique(candidates)
        if not candidates:
            return Selection(commit=())

        above = [c for c in candidates if c.confidence >= self.tau]
        if len(above) >= self.min_commit:
            return _commit(above)
        # progress rail: nothing (or too little) cleared tau — force the top few.
        return _commit(_by_confidence(candidates)[: self.min_commit])


@runtime_checkable
class RevisionPolicy(Protocol):
    """§5.2 revision interface: pick already-committed positions to RE-MASK.

    Kept separate from ``UnmaskPolicy`` on purpose — a reviser is an orthogonal
    opt-in, so the commit path (and every golden fixture pinned to it) is untouched
    when no reviser is wired in. The scheduler only ever offers ``committed``
    positions from the *active* block, so frozen blocks (Tier A/B) are never
    retracted (DESIGN §6.1). ``revision_counts`` is the per-position lifetime count,
    so a reviser can enforce a hard cap and guarantee termination.
    """

    def revisions(
        self,
        committed: Sequence[Candidate],
        ctx: StepContext,
        revision_counts: Mapping[int, int],
    ) -> tuple[Candidate, ...]: ...


@dataclass(frozen=True, slots=True)
class RemaskLowConf:
    """``remask_lowconf`` (§5.2) — the "token revision" feature ("the model changes its mind").

    Each step, re-mask any already-committed active-block token whose *recomputed*
    confidence has fallen below ``tau_revise`` (the board shifted under it, so the
    model now doubts its earlier pick), freeing it to be re-predicted. Capped at
    ``max_revisions`` per position so the block always terminates. The returned
    candidates carry the model's current (low) pick — ``revisions`` is informational
    (``id``/``conf`` for the tokens_revised event); the scheduler writes the mask.
    """

    tau_revise: float
    max_revisions: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau_revise <= 1.0:
            raise ValueError(f"tau_revise must be in [0, 1], got {self.tau_revise}")
        if self.max_revisions < 1:
            raise ValueError(f"max_revisions must be >= 1, got {self.max_revisions}")

    def revisions(
        self,
        committed: Sequence[Candidate],
        ctx: StepContext,
        revision_counts: Mapping[int, int],
    ) -> tuple[Candidate, ...]:
        _check_unique(committed)
        picked = [
            c
            for c in committed
            if c.confidence < self.tau_revise
            and revision_counts.get(c.pos, 0) < self.max_revisions
        ]
        return tuple(sorted(picked, key=lambda c: c.pos))
