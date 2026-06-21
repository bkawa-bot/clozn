"""Infill / editing (DESIGN: native dLLM fill-in-the-middle) — fill a masked gap between a
prefix and a suffix under full bidirectional attention. Torch-free via FakeAdapter."""

import pytest

from cloze_lab.generate import GenerateConfig, infill
from cloze_lab.models.fake import FakeAdapter
from cloze_lab.scheduler.events import GenFinished, GenStarted, TokensCommitted, TokensRevised
from cloze_lab.scheduler.policies import RemaskLowConf


def test_fills_gap_and_preserves_both_sides() -> None:
    adapter = FakeAdapter(seed=7)
    prefix = adapter.encode("def add")
    suffix = adapter.encode("return x")
    gap = 5
    result = infill(adapter, prefix, suffix, gap, GenerateConfig(max_new=gap, steps=8))

    board = result.board.tolist()
    mask = adapter.config.mask_token_id
    lo, hi = len(prefix), len(prefix) + gap
    assert board[:lo] == prefix  # prefix verbatim
    assert board[hi:] == suffix  # suffix verbatim — the fixed right-context
    assert mask not in board[lo:hi]  # gap fully filled
    assert result.events[-1].reason == "length"
    assert result.events[-1].new_tokens == gap


def test_validates_inputs() -> None:
    adapter = FakeAdapter(seed=1)
    with pytest.raises(ValueError, match="gap must be"):
        infill(adapter, adapter.encode("a"), adapter.encode("b"), 0, GenerateConfig(max_new=1, steps=2))
    with pytest.raises(ValueError, match="prefix or a suffix"):
        infill(adapter, [], [], 4, GenerateConfig(max_new=4, steps=2))


def test_emits_event_stream() -> None:
    adapter = FakeAdapter(seed=3)
    result = infill(adapter, adapter.encode("x"), adapter.encode("y"), 4, GenerateConfig(max_new=4, steps=6))
    assert isinstance(result.events[0], GenStarted)
    assert isinstance(result.events[-1], GenFinished)
    assert any(isinstance(e, TokensCommitted) for e in result.events)


def test_one_sided_context_is_allowed() -> None:
    adapter = FakeAdapter(seed=2)
    only_prefix = infill(adapter, adapter.encode("hello"), [], 3, GenerateConfig(max_new=3, steps=6))
    only_suffix = infill(adapter, [], adapter.encode("world"), 3, GenerateConfig(max_new=3, steps=6))
    assert only_prefix.events[-1].new_tokens == 3
    assert only_suffix.events[-1].new_tokens == 3


def test_infill_supports_revisions() -> None:
    adapter = FakeAdapter(seed=7)
    result = infill(
        adapter, adapter.encode("a"), adapter.encode("b"), 4,
        GenerateConfig(max_new=4, steps=12), reviser=RemaskLowConf(tau_revise=1.0, max_revisions=1),
    )
    assert any(isinstance(e, TokensRevised) for e in result.events)
    assert result.events[-1].reason in ("length", "steps_exhausted")  # terminates either way
