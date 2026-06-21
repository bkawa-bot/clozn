"""Divergence harness (DESIGN.md §8 / §5.5): how far an approximate run strays.

The honesty column (invariant 5): pair a fast variant (e.g. cache=delta) against
an exact baseline (cache=off) over identical inputs and report exact-match % and a
confidence delta, so every speed claim ships with the quality it cost. A consumer
of the event stream — it compares two finished runs, never reruns the model.
"""

from __future__ import annotations

from dataclasses import dataclass

from cloze_lab.generate import GenerateResult
from cloze_lab.scheduler.events import GenStarted, TokensCommitted


def _prompt_len(result: GenerateResult) -> int:
    started = next(e for e in result.events if isinstance(e, GenStarted))
    return started.prompt_tokens


def _mean_confidence(result: GenerateResult) -> float:
    confs = [
        item.conf
        for e in result.events
        if isinstance(e, TokensCommitted)
        for item in e.items
    ]
    return sum(confs) / len(confs) if confs else 0.0


@dataclass(frozen=True, slots=True)
class DivergenceStats:
    n_positions: int  # output positions compared
    token_match: float  # fraction identical (1.0 = exact)
    text_match: bool  # decoded outputs identical
    mean_conf_baseline: float
    mean_conf_variant: float

    @property
    def exact(self) -> bool:
        return self.token_match == 1.0

    @property
    def mean_conf_delta(self) -> float:
        return self.mean_conf_variant - self.mean_conf_baseline


def divergence(baseline: GenerateResult, variant: GenerateResult) -> DivergenceStats:
    """Compare a ``variant`` run against the exact ``baseline`` over the output region."""
    p = _prompt_len(baseline)
    if _prompt_len(variant) != p:
        raise ValueError("baseline and variant have different prompt lengths")
    base_out = [int(x) for x in baseline.board[p:]]
    var_out = [int(x) for x in variant.board[p:]]
    if len(base_out) != len(var_out):
        raise ValueError("baseline and variant have different output lengths")
    n = len(base_out)
    matches = sum(1 for a, b in zip(base_out, var_out) if a == b)
    return DivergenceStats(
        n_positions=n,
        token_match=matches / n if n else 1.0,
        text_match=baseline.text == variant.text,
        mean_conf_baseline=_mean_confidence(baseline),
        mean_conf_variant=_mean_confidence(variant),
    )
