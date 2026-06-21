"""Renderer tests: the TUI consumes only events, headless via rich's recorder."""

import pytest
from rich.console import Console

from cloze_lab.generate import GenerateConfig, generate
from cloze_lab.models.fake import FakeAdapter
from cloze_lab.render.tui import BoardRenderer, _conf_style, live_generate
from cloze_lab.scheduler.events import BlockStarted, ReviseItem, StepStats, TokensRevised


def export(renderer: BoardRenderer) -> str:
    console = Console(record=True, width=120, force_terminal=True)
    console.print(renderer.render())
    return console.export_text()


@pytest.fixture
def fake() -> FakeAdapter:
    return FakeAdapter(seed=7)


@pytest.fixture
def prompt(fake: FakeAdapter) -> list[int]:
    return fake.encode("hi")


def test_masks_resolve_into_text(fake, prompt) -> None:
    events = generate(fake, prompt, GenerateConfig(max_new=5, steps=3)).events
    renderer = BoardRenderer(fake.decode, prompt)
    renderer.on_event(events[0])  # gen_started
    renderer.on_event(events[1])  # block_started
    first = export(renderer)
    assert first.count("░") == 5
    assert fake.decode(prompt) in first
    for event in events[2:]:
        renderer.on_event(event)
    final = export(renderer)
    assert "░" not in final
    assert "tok/s" in final  # gen_finished footer


def test_footer_shows_progress_mid_run(fake, prompt) -> None:
    events = generate(fake, prompt, GenerateConfig(max_new=4, steps=2)).events
    renderer = BoardRenderer(fake.decode, prompt)
    for event in events:
        renderer.on_event(event)
        if isinstance(event, StepStats):
            break
    out = export(renderer)
    assert "step 1" in out and "masked" in out


def test_revision_event_renders(fake, prompt) -> None:
    renderer = BoardRenderer(fake.decode, prompt)
    start = len(prompt)
    renderer.on_event(BlockStarted(t=0, block=0, span=(start, start + 2)))
    renderer.on_event(
        TokensRevised(t=1, block=0, items=(ReviseItem(pos=start, old=3, id=5, conf=0.4),))
    )
    out = export(renderer)
    assert "<5>" in out
    assert out.count("░") == 1  # the untouched slot still masked


def test_conf_style_ramps_and_flashes() -> None:
    assert _conf_style(0.0, fresh=False) != _conf_style(1.0, fresh=False)
    assert "bold" in _conf_style(0.5, fresh=True)
    assert "bold" not in _conf_style(0.5, fresh=False)


def test_live_generate_matches_plain_generate(fake, prompt) -> None:
    console = Console(record=True, width=120, force_terminal=True)
    config = GenerateConfig(max_new=5, steps=3)
    live_result = live_generate(fake, prompt, config, console=console)
    direct = generate(fake, prompt, config)
    assert live_result.text == direct.text
    assert "tok/s" in console.export_text()  # final frame left on screen
