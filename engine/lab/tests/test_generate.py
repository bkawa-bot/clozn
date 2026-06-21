"""End-to-end tests of the pass loop against the FakeAdapter (fixed/adaptive, blocks)."""

import itertools
from pathlib import Path

import numpy as np
import pytest

from cloze_lab.generate import (
    GenerateConfig,
    GenerateResult,
    generate,
    sample_candidates,
    truncate_at_eos,
)
from cloze_lab.golden import GoldenCase, build_adapter, build_policy
from cloze_lab.models.fake import FakeAdapter
from cloze_lab.scheduler.cache import CacheConfig
from cloze_lab.scheduler.events import GenFinished, StepStats, TokensCommitted
from cloze_lab.scheduler.policies import ConfidenceTopK, Threshold
from cloze_lab.scheduler.stepper import AdaptiveStepper, FixedStepper

GOLDEN_DIR = Path(__file__).parent / "golden"


def ticking_clock():
    counter = itertools.count()
    return lambda: float(next(counter))


@pytest.fixture
def fake() -> FakeAdapter:
    return FakeAdapter(seed=7)


@pytest.fixture
def prompt(fake: FakeAdapter) -> list[int]:
    return fake.encode("hi")


class TestSampleCandidates:
    def test_greedy_argmax_and_confidence(self) -> None:
        logits = np.array([[0.0, 2.0, 1.0]], dtype=np.float32)
        (c,) = sample_candidates(logits, [5])
        assert c.pos == 5 and c.token_id == 1
        probs = np.exp(logits[0] - logits[0].max())
        assert c.confidence == pytest.approx(probs[1] / probs.sum())

    def test_sampling_is_seed_deterministic(self) -> None:
        logits = np.random.default_rng(0).normal(size=(4, 16)).astype(np.float32)
        a = sample_candidates(logits, range(4), temperature=1.0, rng=np.random.default_rng(3))
        b = sample_candidates(logits, range(4), temperature=1.0, rng=np.random.default_rng(3))
        assert a == b

    def test_errors(self) -> None:
        logits = np.zeros((2, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="rows"):
            sample_candidates(logits, [0])
        with pytest.raises(ValueError, match="rng"):
            sample_candidates(logits, [0, 1], temperature=1.0)
        with pytest.raises(ValueError, match="temperature"):
            sample_candidates(logits, [0, 1], temperature=-1.0)


class TestTruncateAtEos:
    def test_cases(self) -> None:
        assert truncate_at_eos([5, 6, 7], eos_token_id=None) == [5, 6, 7]
        assert truncate_at_eos([5, 6, 7], eos_token_id=9) == [5, 6, 7]
        assert truncate_at_eos([5, 9, 7], eos_token_id=9) == [5]
        assert truncate_at_eos([9, 6], eos_token_id=9) == []


class TestGenerateLoop:
    def test_drains_board_and_emits_canonical_sequence(self, fake, prompt) -> None:
        result = generate(fake, prompt, GenerateConfig(max_new=5, steps=3))
        assert isinstance(result, GenerateResult)
        assert not (result.board == fake.config.mask_token_id).any()
        assert np.array_equal(result.board[: len(prompt)], prompt)  # prompt untouched
        names = [type(e).__name__ for e in result.events]
        assert names == (
            ["GenStarted", "BlockStarted"]
            + ["TokensCommitted", "StepStats"] * 3
            + ["BlockFinalized", "GenFinished"]
        )
        committed = sum(len(e.items) for e in result.events if isinstance(e, TokensCommitted))
        assert committed == 5
        finished = result.events[-1]
        assert finished.reason in ("eos", "length")
        assert finished.steps_total == 3

    def test_remaining_decreases_to_zero(self, fake, prompt) -> None:
        result = generate(fake, prompt, GenerateConfig(max_new=6, steps=4))
        remaining = [e.remaining for e in result.events if isinstance(e, StepStats)]
        assert remaining == sorted(remaining, reverse=True)
        assert remaining[-1] == 0

    def test_deterministic_with_injected_clock(self, fake, prompt) -> None:
        cfg = GenerateConfig(max_new=5, steps=3, temperature=1.0, seed=11)
        a = generate(fake, prompt, cfg, clock=ticking_clock())
        b = generate(fake, prompt, cfg, clock=ticking_clock())
        assert a.events == b.events
        assert np.array_equal(a.board, b.board)

    def test_greedy_ignores_seed_sampling_does_not(self, fake, prompt) -> None:
        g1 = generate(fake, prompt, GenerateConfig(max_new=5, steps=3, seed=1))
        g2 = generate(fake, prompt, GenerateConfig(max_new=5, steps=3, seed=2))
        assert np.array_equal(g1.board, g2.board)
        s1 = generate(fake, prompt, GenerateConfig(max_new=5, steps=3, temperature=1.0, seed=1))
        s2 = generate(fake, prompt, GenerateConfig(max_new=5, steps=3, temperature=1.0, seed=2))
        assert not np.array_equal(s1.board, s2.board)

    def test_steps_exhausted_leaves_visible_holes(self, fake, prompt) -> None:
        result = generate(
            fake, prompt, GenerateConfig(max_new=5, steps=2), policy=ConfidenceTopK(k=1)
        )
        finished = result.events[-1]
        assert isinstance(finished, GenFinished)
        assert finished.reason == "steps_exhausted"
        assert int((result.board == fake.config.mask_token_id).sum()) == 3
        assert "░" in result.text

    def test_on_event_streams_everything(self, fake, prompt) -> None:
        seen = []
        result = generate(
            fake, prompt, GenerateConfig(max_new=4, steps=2), on_event=seen.append
        )
        assert tuple(seen) == result.events

    def test_step_ms_uses_injected_clock(self, fake, prompt) -> None:
        result = generate(fake, prompt, GenerateConfig(max_new=4, steps=2), clock=ticking_clock())
        for stats in (e for e in result.events if isinstance(e, StepStats)):
            assert stats.ms == 1000.0

    def test_validation(self, fake, prompt) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            generate(fake, [], GenerateConfig(max_new=4, steps=2))
        with pytest.raises(ValueError, match="max_new"):
            GenerateConfig(max_new=0, steps=2)
        with pytest.raises(ValueError, match="steps"):
            GenerateConfig(max_new=4, steps=0)


class TestAdaptiveStepping:
    def test_explicit_fixed_stepper_matches_default(self, fake, prompt) -> None:
        cfg = GenerateConfig(max_new=6, steps=3)
        default = generate(fake, prompt, cfg, clock=ticking_clock())
        explicit = generate(fake, prompt, cfg, stepper=FixedStepper(cfg.steps), clock=ticking_clock())
        assert default.events == explicit.events  # the default IS FixedStepper(config.steps)
        assert np.array_equal(default.board, explicit.board)

    def test_threshold_drains_early_below_t_max(self, fake, prompt) -> None:
        # The adaptive speed win: the board drains in fewer passes than t_max.
        result = generate(
            fake, prompt, GenerateConfig(max_new=8, steps=12),
            policy=Threshold(tau=0.10), stepper=AdaptiveStepper(t_max=12),
        )
        finished = result.events[-1]
        assert isinstance(finished, GenFinished)
        assert finished.reason == "length"
        assert finished.steps_total < 12  # stopped early
        assert not (result.board == fake.config.mask_token_id).any()

    def test_min_one_commit_drains_when_tau_unreachable(self, fake, prompt) -> None:
        # tau=0.99 is never met; the rail forces one commit per pass, draining 1/step.
        result = generate(
            fake, prompt, GenerateConfig(max_new=5, steps=8),
            policy=Threshold(tau=0.99), stepper=AdaptiveStepper(t_max=8),
        )
        committed = [e.committed for e in result.events if isinstance(e, StepStats)]
        assert committed == [1, 1, 1, 1, 1]
        assert result.events[-1].reason == "length"

    def test_t_max_rail_caps_and_leaves_holes(self, fake, prompt) -> None:
        result = generate(
            fake, prompt, GenerateConfig(max_new=8, steps=3),
            policy=Threshold(tau=0.99), stepper=AdaptiveStepper(t_max=3),
        )
        finished = result.events[-1]
        assert finished.reason == "steps_exhausted"
        assert finished.steps_total == 3
        assert int((result.board == fake.config.mask_token_id).sum()) == 5

    def test_adaptive_rejects_quota_mode(self, fake, prompt) -> None:
        # Quota (k=None) needs a fixed budget; adaptive supplies none -> clear error.
        with pytest.raises(ValueError, match="quota mode"):
            generate(
                fake, prompt, GenerateConfig(max_new=6, steps=8),
                policy=ConfidenceTopK(k=None), stepper=AdaptiveStepper(t_max=8),
            )

    def test_adaptive_with_fixed_k_works(self, fake, prompt) -> None:
        # Fixed integer k needs no budget, so it composes with adaptive stepping.
        result = generate(
            fake, prompt, GenerateConfig(max_new=6, steps=8),
            policy=ConfidenceTopK(k=2), stepper=AdaptiveStepper(t_max=8),
        )
        assert not (result.board == fake.config.mask_token_id).any()
        assert result.events[-1].reason in ("eos", "length")

    def test_never_commits_mask_token(self, fake, prompt) -> None:
        # The FakeAdapter must not emit the mask sentinel (would stall the loop).
        result = generate(fake, prompt, GenerateConfig(max_new=8, steps=4))
        committed_ids = [
            item.id
            for e in result.events
            if isinstance(e, TokensCommitted)
            for item in e.items
        ]
        assert fake.config.mask_token_id not in committed_ids


class TestBlockMode:
    def test_blocks_emit_one_block_pair_each(self, fake) -> None:
        from cloze_lab.scheduler.events import BlockFinalized, BlockStarted

        prompt = fake.encode("hello cloze")  # drains both blocks without an early EOS
        result = generate(fake, prompt, GenerateConfig(max_new=8, steps=4, block_len=4))
        starts = [e.block for e in result.events if isinstance(e, BlockStarted)]
        finals = [e.block for e in result.events if isinstance(e, BlockFinalized)]
        assert starts == [0, 1] and finals == [0, 1]  # two blocks, left to right

    def test_blocks_left_to_right_then_drained(self, fake) -> None:
        prompt = fake.encode("hello cloze")
        result = generate(fake, prompt, GenerateConfig(max_new=8, steps=4, block_len=4))
        assert not (result.board == fake.config.mask_token_id).any()
        assert result.events[-1].reason == "length"
        # block 0 finalizes entirely before block 1 commits anything
        order = [
            (e.block, e.items)
            for e in result.events
            if isinstance(e, TokensCommitted)
        ]
        first_block1 = next(i for i, (b, _) in enumerate(order) if b == 1)
        assert all(b == 0 for b, _ in order[:first_block1])

    def test_global_t_is_monotonic_across_blocks(self, fake, prompt) -> None:
        result = generate(fake, prompt, GenerateConfig(max_new=8, steps=4, block_len=4))
        ts = [e.t for e in result.events if isinstance(e, StepStats)]
        assert ts == sorted(ts) and len(set(ts)) == len(ts)  # strictly increasing

    def test_per_block_step_resets(self, fake, prompt) -> None:
        result = generate(fake, prompt, GenerateConfig(max_new=8, steps=4, block_len=4))
        steps_by_block: dict[int, list[int]] = {}
        for e in result.events:
            if isinstance(e, StepStats):
                steps_by_block.setdefault(e.block, []).append(e.step)
        for block, steps in steps_by_block.items():
            assert steps[0] == 0  # each block's step index restarts at 0

    def test_eos_in_early_block_skips_later_blocks(self) -> None:
        # Reuse the golden's seed/config: EOS in an early block => finished_early,
        # so later blocks never run and trailing positions stay masked.
        case = GoldenCase.read(GOLDEN_DIR / "fake_blocks_eos_early.json")
        adapter = build_adapter(case.model)
        result = generate(
            adapter, case.prompt_ids, GenerateConfig(**case.config),
            policy=build_policy(case.policy),
        )
        assert result.events[-1].reason == "eos"
        p = len(case.prompt_ids)
        assert (result.board[p:] == adapter.config.mask_token_id).any()

    def test_whole_sequence_default_equals_block_len_zero(self, fake, prompt) -> None:
        cfg_default = GenerateConfig(max_new=6, steps=3)
        cfg_explicit = GenerateConfig(max_new=6, steps=3, block_len=0)
        a = generate(fake, prompt, cfg_default, clock=ticking_clock())
        b = generate(fake, prompt, cfg_explicit, clock=ticking_clock())
        assert a.events == b.events


class TestCacheIntegration:
    def test_default_cache_is_off(self, fake, prompt) -> None:
        cfg = GenerateConfig(max_new=8, steps=5)
        default = generate(fake, prompt, cfg, clock=ticking_clock())
        off = generate(fake, prompt, cfg, cache=CacheConfig(mode="off"), clock=ticking_clock())
        assert default.events == off.events

    @pytest.mark.parametrize("block_len", [0, 4])
    def test_delta_diverges_from_off(self, fake, prompt, block_len: int) -> None:
        cfg = GenerateConfig(max_new=12, steps=8, block_len=block_len)
        off = generate(fake, prompt, cfg, cache=CacheConfig(mode="off"))
        delta = generate(fake, prompt, cfg, cache=CacheConfig(mode="delta", full_refresh_every=4))
        assert not np.array_equal(off.board, delta.board)  # Tier C drift

    @pytest.mark.parametrize("block_len", [0, 4])
    def test_full_refresh_every_one_is_exact(self, fake, prompt, block_len: int) -> None:
        # A full refresh every step recomputes everything, so delta == off exactly.
        cfg = GenerateConfig(max_new=12, steps=8, block_len=block_len)
        off = generate(fake, prompt, cfg, cache=CacheConfig(mode="off"))
        exact = generate(fake, prompt, cfg, cache=CacheConfig(mode="delta", full_refresh_every=1))
        assert np.array_equal(off.board, exact.board)

    def test_cache_hit_reported_in_step_stats(self, fake, prompt) -> None:
        cfg = GenerateConfig(max_new=10, steps=6)
        delta = generate(fake, prompt, cfg, cache=CacheConfig(mode="delta", full_refresh_every=4))
        hits = [e.cache_hit for e in delta.events if isinstance(e, StepStats)]
        assert hits[0] == 0.0  # cold start recomputes all
        assert max(hits) > 0.0  # later passes reuse drawers
        off = generate(fake, prompt, cfg, cache=CacheConfig(mode="off"))
        assert all(e.cache_hit == 0.0 for e in off.events if isinstance(e, StepStats))

    def test_tier_b_free_freeze_is_exact_in_block_mode(self, fake, prompt) -> None:
        # Free freeze keeps frozen blocks exact: a block-mode delta run that fully
        # refreshes the active block every step equals the exact off run.
        cfg = GenerateConfig(max_new=12, steps=6, block_len=4)
        off = generate(fake, prompt, cfg, cache=CacheConfig(mode="off"))
        exact = generate(fake, prompt, cfg, cache=CacheConfig(mode="delta", full_refresh_every=1))
        assert np.array_equal(off.board, exact.board)

    def test_block_delta_reuses_frozen_blocks(self, fake, prompt) -> None:
        # Later block-mode passes reach high cache-hit because frozen blocks (the
        # growing prefix) are reused, never recomputed.
        cfg = GenerateConfig(max_new=12, steps=6, block_len=4)
        delta = generate(fake, prompt, cfg, cache=CacheConfig(mode="delta", full_refresh_every=4))
        hits = [e.cache_hit for e in delta.events if isinstance(e, StepStats)]
        assert max(hits) > 0.5  # the frozen prefix dominates reuse in later blocks
