"""Terminal denoise visualization (rich): masks render as ░ resolving into text.

NOTE: the canonical, demo-grade front-end is now the **browser viz** served by
``cloze-server`` at ``GET /`` (core/serve/viz_html.hpp) — it renders real
whitespace, per-token cells, confidence-as-text-color, paced playback, and the
revision view. This TUI is kept as a minimal, zero-dependency reference consumer
of the same event spine (it prints the raw board, so an instruct chat template or
special/EOS tokens will show verbatim — use raw prompts and ``skip_special_tokens``
if you want it tidy).

Strictly a consumer of the §5.1 event stream (DESIGN invariant 2) plus a
detokenizer for turning ids into glyphs — the same data a DSP client will get.
Committed tokens are brightness-tinted by confidence and flash bold on the pass
they land (§6.2 client-rendering guidance); revisions flash red.

Demo without a checkpoint:  python -m cloze_lab.render.tui
The real thing (CPU bf16):  python -m cloze_lab.render.tui --model dream
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from cloze_lab.generate import GenerateConfig, GenerateResult, generate
from cloze_lab.models.base import ModelAdapter
from cloze_lab.scheduler.cache import CacheConfig
from cloze_lab.scheduler.events import (
    BlockStarted,
    Event,
    GenFinished,
    StepStats,
    TokensCommitted,
    TokensRevised,
)
from cloze_lab.scheduler.policies import RevisionPolicy, UnmaskPolicy

Decode = Callable[[Sequence[int]], str]

_MASK_STYLE = "grey30"
_PROMPT_STYLE = "grey66"


def _conf_style(conf: float, fresh: bool) -> str:
    """Confidence-tinted brightness, bold on the pass the token landed."""
    v = 140 + round(115 * min(max(conf, 0.0), 1.0))
    return f"rgb({v},{v},{v})" + (" bold" if fresh else "")


@dataclass(slots=True)
class _Slot:
    token_id: int
    conf: float
    t: int
    revised: bool = False


class BoardRenderer:
    """Folds §5.1 events into a live picture of the board."""

    def __init__(
        self,
        decode: Decode,
        prompt_ids: Sequence[int] | None = None,
        mask_glyph: str = "░",
    ) -> None:
        self._decode = decode
        self._prompt_text = decode(list(prompt_ids)) if prompt_ids else ""
        self._mask_glyph = mask_glyph
        self._span: tuple[int, int] | None = None
        self._slots: dict[int, _Slot] = {}
        self._latest_t = -1
        self._stats: StepStats | None = None
        self._finished: GenFinished | None = None

    def on_event(self, event: Event) -> None:
        if isinstance(event, BlockStarted):
            start, end = event.span
            if self._span is not None:  # later blocks extend the picture
                start, end = min(start, self._span[0]), max(end, self._span[1])
            self._span = (start, end)
        elif isinstance(event, TokensCommitted):
            self._latest_t = event.t
            for item in event.items:
                self._slots[item.pos] = _Slot(item.id, item.conf, event.t)
        elif isinstance(event, TokensRevised):
            self._latest_t = event.t
            for item in event.items:
                self._slots[item.pos] = _Slot(item.id, item.conf, event.t, revised=True)
        elif isinstance(event, StepStats):
            self._stats = event
        elif isinstance(event, GenFinished):
            self._finished = event
        # gen_started / block_finalized carry nothing the picture needs yet

    def render(self) -> Panel:
        board = Text(overflow="fold")
        if self._prompt_text:
            board.append(self._prompt_text, style=_PROMPT_STYLE)
        if self._span is not None:
            for pos in range(self._span[0], self._span[1]):
                slot = self._slots.get(pos)
                if slot is None:
                    board.append(self._mask_glyph, style=_MASK_STYLE)
                    continue
                fresh = slot.t == self._latest_t
                style = "red bold" if slot.revised and fresh else _conf_style(slot.conf, fresh)
                board.append(self._decode([slot.token_id]) or "·", style=style)

        footer = Text(style="dim")
        if self._finished is not None:
            f = self._finished
            footer.append(
                f"{f.reason} · {f.new_tokens} tok · {f.wall_ms / 1000:.2f} s · "
                f"{f.tok_per_s:.1f} tok/s · {f.steps_total} steps"
            )
        elif self._stats is not None:
            s = self._stats
            footer.append(f"step {s.step + 1} · +{s.committed} · {s.remaining} masked · {s.ms:.0f} ms")
        else:
            footer.append("waiting…")
        return Panel(Group(board, footer), title="cloze lab — denoise", border_style="grey42")


def live_generate(
    adapter: ModelAdapter,
    prompt_ids: Sequence[int],
    config: GenerateConfig,
    *,
    policy: UnmaskPolicy | None = None,
    reviser: RevisionPolicy | None = None,
    cache: CacheConfig | None = None,
    console: Console | None = None,
    delay: float = 0.0,
) -> GenerateResult:
    """Run the pass loop while rendering every event as it happens.

    Pass a ``reviser`` (e.g. ``RemaskLowConf``) to watch the model change its mind —
    revised slots flash red as their committed tokens are retracted and re-predicted.
    ``cache`` threads an effort preset's KV-reuse setting through to the loop.
    """
    renderer = BoardRenderer(adapter.decode, prompt_ids)
    console = console or Console()
    with Live(renderer.render(), console=console, refresh_per_second=24) as live:

        def on_event(event: Event) -> None:
            renderer.on_event(event)
            live.update(renderer.render())
            if delay and isinstance(event, StepStats):
                time.sleep(delay)

        return generate(
            adapter, prompt_ids, config, policy=policy, reviser=reviser, cache=cache, on_event=on_event
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cloze-lab-tui", description="Watch a diffusion LM denoise.")
    parser.add_argument("--model", choices=("fake", "dream", "dcoder"), default="fake")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new", type=int, default=16)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay", type=float, default=0.15, help="pause after each pass (seconds)")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    # Shared adapter builder (fake | dream | dcoder); imported lazily to avoid a cycle.
    from cloze_lab.cli import build_adapter

    adapter = build_adapter(args.model, device=args.device)
    config = GenerateConfig(
        max_new=args.max_new, steps=args.steps, temperature=args.temperature, seed=args.seed
    )
    live_generate(adapter, adapter.encode(args.prompt), config, delay=args.delay)


if __name__ == "__main__":
    main()
