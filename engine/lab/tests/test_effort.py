"""Effort presets (DESIGN §5.3): the fast/balanced/quality speed-quality knob."""

import pytest

from cloze_lab.effort import EFFORT_LEVELS, EFFORT_PRESETS, resolve_effort


def test_all_levels_present() -> None:
    assert set(EFFORT_PRESETS) == set(EFFORT_LEVELS) == {"fast", "balanced", "quality"}


def test_effort_orders_steps_fast_to_quality() -> None:
    # more effort => more denoise passes.
    assert EFFORT_PRESETS["fast"].steps < EFFORT_PRESETS["balanced"].steps < EFFORT_PRESETS["quality"].steps


def test_caches_are_real_model_safe() -> None:
    # quality = full recompute; fast/balanced = EXACT prefix reuse (refresh=1), the only
    # delta the Dream-family adapters support. No approximate Tier C (refresh>1) here.
    assert EFFORT_PRESETS["quality"].cache.mode == "off"
    for level in ("fast", "balanced"):
        c = EFFORT_PRESETS[level].cache
        assert c.mode == "delta" and c.full_refresh_every == 1
    # block mode is required for prefix reuse to actually kick in.
    assert all(EFFORT_PRESETS[lvl].block_len > 0 for lvl in ("fast", "balanced", "quality"))


def test_resolve_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown effort"):
        resolve_effort("turbo")
