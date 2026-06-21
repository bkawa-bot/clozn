"""Speed harness (DESIGN.md §8): per-run timing/work stats from the event stream.

A consumer of the §5.1 events (invariant 2) — it never reruns the model. The
work proxy (mean cache-hit, recompute fraction) is deterministic; wall-clock
tok/s reflects whatever clock the run used, so pin it only with an injected clock.
"""

from __future__ import annotations

from dataclasses import dataclass

from cloze_lab.generate import GenerateResult
from cloze_lab.scheduler.events import GenFinished, StepStats


@dataclass(frozen=True, slots=True)
class SpeedStats:
    forwards: int  # number of forward passes (StepStats events)
    mean_cache_hit: float  # average fraction of positions reused per pass
    recompute_fraction: float  # 1 - mean_cache_hit: the K/V work actually done
    new_tokens: int
    steps_total: int
    steps_per_token: float  # forward passes per committed token (lower is faster)
    wall_ms: float
    tok_per_s: float


def speed_stats(result: GenerateResult) -> SpeedStats:
    steps = [e for e in result.events if isinstance(e, StepStats)]
    finished = next(e for e in result.events if isinstance(e, GenFinished))
    forwards = len(steps)
    mean_hit = sum(e.cache_hit for e in steps) / forwards if forwards else 0.0
    new_tokens = finished.new_tokens
    spt = finished.steps_total / new_tokens if new_tokens else float("inf")
    return SpeedStats(
        forwards=forwards,
        mean_cache_hit=mean_hit,
        recompute_fraction=1.0 - mean_hit,
        new_tokens=new_tokens,
        steps_total=finished.steps_total,
        steps_per_token=spt,
        wall_ms=finished.wall_ms,
        tok_per_s=finished.tok_per_s,
    )
