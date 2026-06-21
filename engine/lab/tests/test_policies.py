"""Tests for the unmask-policy interface and ConfidenceTopK (DESIGN §5.2)."""

import pytest

from cloze_lab.scheduler.policies import (
    Candidate,
    ConfidenceTopK,
    Selection,
    StepContext,
    Threshold,
    UnmaskPolicy,
)


def cands(*pairs: tuple[int, float]) -> list[Candidate]:
    return [Candidate(pos=p, token_id=100 + p, confidence=c) for p, c in pairs]


class TestStepContext:
    def test_steps_remaining(self) -> None:
        assert StepContext(step=1, steps_total=4).steps_remaining == 3
        assert StepContext(step=3, steps_total=4).steps_remaining == 1
        assert StepContext(step=5).steps_remaining is None

    @pytest.mark.parametrize("kwargs", [{"step": -1}, {"step": 4, "steps_total": 4}])
    def test_invalid(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            StepContext(**kwargs)


class TestConfidenceTopK:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(ConfidenceTopK(), UnmaskPolicy)

    def test_fixed_k_picks_most_confident_pos_ascending(self) -> None:
        cs = cands((0, 0.2), (1, 0.9), (2, 0.5), (3, 0.8))
        sel = ConfidenceTopK(k=2).select(cs, StepContext(step=0))
        assert [c.pos for c in sel.commit] == [1, 3]  # pos order, not rank order
        assert sel.revise == ()

    def test_ties_break_toward_lower_pos(self) -> None:
        cs = cands((4, 0.5), (2, 0.5), (7, 0.5))
        sel = ConfidenceTopK(k=2).select(cs, StepContext(step=0))
        assert [c.pos for c in sel.commit] == [2, 4]

    def test_k_exceeding_candidates_takes_all(self) -> None:
        cs = cands((0, 0.1), (1, 0.2))
        sel = ConfidenceTopK(k=10).select(cs, StepContext(step=0))
        assert [c.pos for c in sel.commit] == [0, 1]

    def test_empty_candidates(self) -> None:
        assert ConfidenceTopK(k=3).select([], StepContext(step=0)) == Selection(commit=())

    def test_deterministic(self) -> None:
        cs = cands((0, 0.3), (1, 0.3), (2, 0.9))
        a = ConfidenceTopK(k=2).select(cs, StepContext(step=0))
        b = ConfidenceTopK(k=2).select(list(reversed(cs)), StepContext(step=0))
        assert a == b

    def test_duplicate_positions_rejected(self) -> None:
        cs = cands((1, 0.3), (1, 0.5))
        with pytest.raises(ValueError, match="duplicate"):
            ConfidenceTopK(k=1).select(cs, StepContext(step=0))

    @pytest.mark.parametrize("bad_k", [0, -3])
    def test_bad_k_rejected(self, bad_k: int) -> None:
        with pytest.raises(ValueError, match="k must be"):
            ConfidenceTopK(k=bad_k)

    def test_quota_mode_requires_budget(self) -> None:
        with pytest.raises(ValueError, match="steps_total"):
            ConfidenceTopK().select(cands((0, 0.5)), StepContext(step=0))


class TestQuotaDrainsExactly:
    @pytest.mark.parametrize(("n_masked", "steps_total"), [(10, 4), (5, 5), (7, 3), (1, 1), (16, 8), (3, 7)])
    def test_fixed_budget_empties_board_on_last_step(self, n_masked: int, steps_total: int) -> None:
        policy = ConfidenceTopK()
        remaining = cands(*[(p, (p * 37 % 11) / 11) for p in range(n_masked)])
        inked_per_step = []
        for step in range(steps_total):
            sel = policy.select(remaining, StepContext(step=step, steps_total=steps_total))
            inked = {c.pos for c in sel.commit}
            assert sel.commit, "quota must never ink zero while masks remain"
            remaining = [c for c in remaining if c.pos not in inked]
            inked_per_step.append(len(inked))
            if not remaining:
                break
        assert not remaining
        # the ramp never exceeds an even share by more than rounding
        assert max(inked_per_step) <= ceil_div(n_masked, steps_total) + 1


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


class TestThreshold:
    ctx = StepContext(step=0)  # threshold ignores the context (step-budget-free)

    def test_satisfies_protocol(self) -> None:
        assert isinstance(Threshold(0.5), UnmaskPolicy)

    def test_commits_everything_at_or_above_tau(self) -> None:
        cs = cands((0, 0.4), (1, 0.9), (2, 0.6), (3, 0.5))
        sel = Threshold(0.6).select(cs, self.ctx)
        assert [c.pos for c in sel.commit] == [1, 2]  # 0.9 and 0.6 clear; 0.5/0.4 don't

    def test_boundary_is_inclusive(self) -> None:
        sel = Threshold(0.5).select(cands((0, 0.5)), self.ctx)
        assert [c.pos for c in sel.commit] == [0]

    def test_min_one_commit_rail_forces_top_when_none_clear(self) -> None:
        cs = cands((0, 0.1), (1, 0.3), (2, 0.2))
        sel = Threshold(0.9).select(cs, self.ctx)  # nothing clears 0.9
        assert [c.pos for c in sel.commit] == [1]  # forced: the single most confident

    def test_min_commit_greater_than_one(self) -> None:
        cs = cands((0, 0.1), (1, 0.3), (2, 0.2), (3, 0.05))
        sel = Threshold(0.9, min_commit=2).select(cs, self.ctx)
        assert [c.pos for c in sel.commit] == [1, 2]  # top 2 by confidence, pos-ascending

    def test_above_tau_beats_min_commit(self) -> None:
        # when more than min_commit clear tau, commit ALL of them, not just min_commit
        cs = cands((0, 0.95), (1, 0.95), (2, 0.95))
        sel = Threshold(0.9, min_commit=1).select(cs, self.ctx)
        assert [c.pos for c in sel.commit] == [0, 1, 2]

    def test_commit_is_pos_ascending(self) -> None:
        cs = cands((5, 0.9), (1, 0.9), (3, 0.9))
        sel = Threshold(0.5).select(cs, self.ctx)
        assert [c.pos for c in sel.commit] == [1, 3, 5]

    def test_empty_candidates(self) -> None:
        assert Threshold(0.5).select([], self.ctx) == Selection(commit=())

    def test_duplicate_positions_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            Threshold(0.5).select(cands((1, 0.9), (1, 0.8)), self.ctx)

    @pytest.mark.parametrize("bad_tau", [-0.1, 1.1])
    def test_tau_must_be_in_unit_interval(self, bad_tau: float) -> None:
        with pytest.raises(ValueError, match="tau must be"):
            Threshold(bad_tau)

    @pytest.mark.parametrize("bad_min", [0, -1])
    def test_min_commit_must_be_positive(self, bad_min: int) -> None:
        with pytest.raises(ValueError, match="min_commit"):
            Threshold(0.5, min_commit=bad_min)
