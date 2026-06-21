"""Step control (DESIGN.md §5.3): how many passes a block runs.

A step controller owns exactly one decision the pass loop used to make inline —
"after this pass, run another?" — plus the per-pass ``StepContext`` handed to the
policy. It is pure logic (no numpy/torch) and does no per-pass work; forward,
sample, and commit stay in ``generate.py`` (invariant 4). The block manager
(§5.4) will run a controller *per block*, so a controller holds no cross-block
state — every field is set at construction.

Two rails keep adaptive stepping from stalling, enforced at two layers by design:

- **T_max** (the hard cap) is the loop bound: ``generate.py`` iterates
  ``range(controller.steps_cap)``, so even a misbehaving ``should_continue`` that
  always returns True cannot hang — termination is structural, exactly as today's
  ``range(config.steps)`` guarantees it.
- **min-one-commit** (anti-stall) lives in the *policy* (``Threshold.min_commit``),
  the only place that can force a commit. ``ConfidenceTopK`` already never inks
  zero while masks remain, so this rail is unreachable on the fixed(T) fixtures
  and cannot perturb them.

``FixedStepper`` reproduces the old loop byte-for-byte: it supplies
``steps_total = steps`` so quota mode keeps its budget, and stops only when the
board drains (the ``range(steps)`` bound supplies the T-step ceiling). Adaptive
stopping is not a different stop *rule* — it is the same "stop when drained"
detected early, because a threshold policy drains the board as soon as every
remaining slot clears tau (DESIGN §5.3's "stop when all positions clear tau"),
well before ``t_max``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from cloze_lab.scheduler.policies import StepContext


@dataclass(frozen=True, slots=True)
class StepOutcome:
    """The board-delta of one completed pass, fed back to the controller."""

    step: int  # 0-based index of the pass just run
    n_committed: int  # tokens inked this pass
    n_masked_after: int  # masked positions remaining after the commit


@runtime_checkable
class StepController(Protocol):
    @property
    def steps_cap(self) -> int:
        """Hard upper bound on passes for one block (the loop's ``range`` bound)."""
        ...

    def context(self, step: int) -> StepContext:
        """The StepContext handed to ``policy.select`` for pass ``step``."""
        ...

    def should_continue(self, outcome: StepOutcome) -> bool:
        """After a pass: True to run another, False to finalize the block."""
        ...


@dataclass(frozen=True, slots=True)
class FixedStepper:
    """fixed(T): exactly ``steps`` passes (minus the board-drained early exit).

    The default, behaviorally identical to the original
    ``for step in range(config.steps)`` loop — including ``steps_total = steps``
    so ConfidenceTopK quota mode keeps its exact per-pass budget.
    """

    steps: int

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {self.steps}")

    @property
    def steps_cap(self) -> int:
        return self.steps

    def context(self, step: int) -> StepContext:
        return StepContext(step=step, steps_total=self.steps)

    def should_continue(self, outcome: StepOutcome) -> bool:
        return outcome.n_masked_after > 0


@dataclass(frozen=True, slots=True)
class AdaptiveStepper:
    """adaptive(T_max): run until the board drains or ``t_max`` passes (§5.3).

    Supplies no step budget (``steps_total = None``), so it composes with the
    threshold policy and *cannot* be paired with quota mode (which raises,
    correctly — quota's even drain is undefined without a known budget). The
    early stop is data-driven: a threshold(tau) policy drains the board as soon
    as every remaining slot clears tau, so the loop breaks well before ``t_max``.
    """

    t_max: int

    def __post_init__(self) -> None:
        if self.t_max < 1:
            raise ValueError(f"t_max must be >= 1, got {self.t_max}")

    @property
    def steps_cap(self) -> int:
        return self.t_max

    def context(self, step: int) -> StepContext:
        return StepContext(step=step, steps_total=None)

    def should_continue(self, outcome: StepOutcome) -> bool:
        return outcome.n_masked_after > 0
