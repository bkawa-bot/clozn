"""Effort presets — the daily-driver speed/quality knob (DESIGN §5.3 effort presets).

One name bundles the levers that trade compute for quality. The primary lever is the
number of denoise passes per block (``steps``): fewer steps commit more tokens per
forward (faster, but more aggressive parallel commits), more steps refine more
sequentially (higher quality). The unmask policy stays ``ConfidenceTopK`` quota across
all three (deterministic, drains the board within the budget).

* **fast** — few steps; exact prefix-cache reuse in block mode (Tier A/B, the frozen
  prefix is reused verbatim — a real speedup at no quality cost). Lowest latency.
* **balanced** — moderate steps; same exact prefix reuse.
* **quality** — most steps; full recompute every pass (cache off), the no-shortcuts
  baseline (avoids even the rare fp near-tie flips that reuse can introduce under
  quantization).

All three are real-model safe: the Dream-family adapters support exact contiguous-suffix
reuse (``full_refresh_every=1``) and full recompute, but NOT approximate Tier C reuse
(``full_refresh_every>1``), which only the FakeAdapter expresses. Exposed as
``cloze run --effort {fast,balanced,quality}``.
"""

from __future__ import annotations

from dataclasses import dataclass

from cloze_lab.scheduler.cache import CacheConfig

EFFORT_LEVELS = ("fast", "balanced", "quality")


@dataclass(frozen=True, slots=True)
class EffortPreset:
    """A (steps, block_len, cache) bundle for one effort level."""

    steps: int
    block_len: int
    cache: CacheConfig


# fast/balanced use delta(full_refresh_every=1) = EXACT prefix reuse (block mode), which
# the real Dream-family adapters support; quality uses off (full recompute). Approximate
# Tier C reuse (refresh>1) is deliberately not used here — only the FakeAdapter expresses it.
EFFORT_PRESETS: dict[str, EffortPreset] = {
    "fast": EffortPreset(steps=4, block_len=8, cache=CacheConfig(mode="delta", full_refresh_every=1)),
    "balanced": EffortPreset(steps=8, block_len=8, cache=CacheConfig(mode="delta", full_refresh_every=1)),
    "quality": EffortPreset(steps=16, block_len=8, cache=CacheConfig(mode="off")),
}


def resolve_effort(name: str) -> EffortPreset:
    """Look up a preset by name, with a helpful error on a bad level."""
    try:
        return EFFORT_PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown effort {name!r}; choose one of {EFFORT_LEVELS}") from None
