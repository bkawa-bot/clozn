"""Tests for the bench harness (DESIGN §8): speed stats, divergence, markdown A/B."""

import itertools

import pytest

from cloze_lab.bench.divergence import divergence
from cloze_lab.bench.report import markdown_table, run_ab
from cloze_lab.bench.speed import speed_stats
from cloze_lab.generate import GenerateConfig, generate
from cloze_lab.models.fake import FakeAdapter
from cloze_lab.scheduler.cache import CacheConfig


def ticking_clock():
    counter = itertools.count()
    return lambda: float(next(counter))


@pytest.fixture
def fake() -> FakeAdapter:
    return FakeAdapter(seed=7)


@pytest.fixture
def prompt(fake: FakeAdapter) -> list[int]:
    return fake.encode("hello cloze")


class TestSpeedStats:
    def test_off_does_full_work(self, fake, prompt) -> None:
        cfg = GenerateConfig(max_new=12, steps=8)
        s = speed_stats(generate(fake, prompt, cfg, cache=CacheConfig(mode="off")))
        assert s.forwards == 8
        assert s.mean_cache_hit == 0.0
        assert s.recompute_fraction == 1.0
        assert s.new_tokens == 12
        assert s.steps_per_token == pytest.approx(8 / 12)

    def test_delta_reuses_work(self, fake, prompt) -> None:
        cfg = GenerateConfig(max_new=12, steps=8)
        s = speed_stats(
            generate(fake, prompt, cfg, cache=CacheConfig(mode="delta", full_refresh_every=4))
        )
        assert s.mean_cache_hit > 0.0  # drawers reused
        assert s.recompute_fraction < 1.0


class TestDivergence:
    def test_off_vs_off_is_exact(self, fake, prompt) -> None:
        cfg = GenerateConfig(max_new=12, steps=8)
        a = generate(fake, prompt, cfg, cache=CacheConfig(mode="off"))
        b = generate(fake, prompt, cfg, cache=CacheConfig(mode="off"))
        d = divergence(a, b)
        assert d.exact and d.token_match == 1.0 and d.text_match
        assert d.mean_conf_delta == 0.0

    def test_delta_diverges(self, fake, prompt) -> None:
        cfg = GenerateConfig(max_new=12, steps=8)
        off = generate(fake, prompt, cfg, cache=CacheConfig(mode="off"))
        delta = generate(fake, prompt, cfg, cache=CacheConfig(mode="delta", full_refresh_every=4))
        d = divergence(off, delta)
        assert not d.exact
        assert 0.0 <= d.token_match < 1.0
        assert d.n_positions == 12

    def test_full_refresh_one_is_exact(self, fake, prompt) -> None:
        cfg = GenerateConfig(max_new=12, steps=8)
        off = generate(fake, prompt, cfg, cache=CacheConfig(mode="off"))
        exact = generate(fake, prompt, cfg, cache=CacheConfig(mode="delta", full_refresh_every=1))
        assert divergence(off, exact).exact

    def test_mismatched_lengths_rejected(self, fake, prompt) -> None:
        a = generate(fake, prompt, GenerateConfig(max_new=8, steps=4))
        b = generate(fake, prompt, GenerateConfig(max_new=6, steps=4))
        with pytest.raises(ValueError, match="output lengths"):
            divergence(a, b)


class TestReport:
    def test_run_ab_baseline_then_variants(self, fake, prompt) -> None:
        cfg = GenerateConfig(max_new=12, steps=8)
        rows = run_ab(
            fake, prompt, cfg,
            [CacheConfig(mode="off"), CacheConfig(mode="delta", full_refresh_every=4)],
            clock=ticking_clock(),
        )
        assert rows[0].divergence is None  # baseline
        assert rows[1].divergence is not None and not rows[1].divergence.exact

    def test_run_ab_requires_a_baseline(self, fake, prompt) -> None:
        with pytest.raises(ValueError, match="baseline"):
            run_ab(fake, prompt, GenerateConfig(max_new=4, steps=2), [])

    def test_markdown_table_is_ascii_and_well_formed(self, fake, prompt) -> None:
        cfg = GenerateConfig(max_new=12, steps=8)
        rows = run_ab(
            fake, prompt, cfg,
            [CacheConfig(mode="off"), CacheConfig(mode="delta", full_refresh_every=1)],
            clock=ticking_clock(),
        )
        table = markdown_table(rows, title="demo")
        table.encode("ascii")  # must not raise (cross-platform stdout)
        lines = [ln for ln in table.splitlines() if ln.startswith("|")]
        assert len(lines) == 4  # header, separator, baseline row, variant row
        assert "token-match" in lines[0]  # the honesty column is present
        assert "baseline" in table  # off row marked as the baseline
        assert "100.0%" in table  # refresh=1 variant is exact vs off
