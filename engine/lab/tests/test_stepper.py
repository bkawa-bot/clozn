"""Tests for the step controllers (DESIGN §5.3)."""

import pytest

from cloze_lab.scheduler.policies import StepContext
from cloze_lab.scheduler.stepper import (
    AdaptiveStepper,
    FixedStepper,
    StepController,
    StepOutcome,
)


def outcome(remaining: int, step: int = 0, committed: int = 1) -> StepOutcome:
    return StepOutcome(step=step, n_committed=committed, n_masked_after=remaining)


class TestFixedStepper:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(FixedStepper(4), StepController)

    def test_cap_and_context(self) -> None:
        s = FixedStepper(4)
        assert s.steps_cap == 4
        ctx = s.context(1)
        assert ctx == StepContext(step=1, steps_total=4)  # quota mode keeps its budget

    def test_continue_until_drained(self) -> None:
        s = FixedStepper(4)
        assert s.should_continue(outcome(remaining=3)) is True
        assert s.should_continue(outcome(remaining=0)) is False

    @pytest.mark.parametrize("bad", [0, -1])
    def test_rejects_nonpositive_steps(self, bad: int) -> None:
        with pytest.raises(ValueError, match="steps must be"):
            FixedStepper(bad)


class TestAdaptiveStepper:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(AdaptiveStepper(8), StepController)

    def test_cap_and_context(self) -> None:
        s = AdaptiveStepper(8)
        assert s.steps_cap == 8
        # No fixed budget under adaptive -> steps_total None (quota mode would raise).
        assert s.context(3) == StepContext(step=3, steps_total=None)

    def test_continue_until_drained(self) -> None:
        s = AdaptiveStepper(8)
        assert s.should_continue(outcome(remaining=2)) is True
        assert s.should_continue(outcome(remaining=0)) is False

    @pytest.mark.parametrize("bad", [0, -3])
    def test_rejects_nonpositive_tmax(self, bad: int) -> None:
        with pytest.raises(ValueError, match="t_max must be"):
            AdaptiveStepper(bad)


def test_fixed_context_matches_legacy_inline_construction() -> None:
    # The load-bearing compatibility line: FixedStepper(T).context(step) must equal
    # the StepContext the pre-stepper loop built inline (StepContext(step, steps_total=T)).
    for steps in (1, 3, 8):
        s = FixedStepper(steps)
        for step in range(steps):
            assert s.context(step) == StepContext(step=step, steps_total=steps)
