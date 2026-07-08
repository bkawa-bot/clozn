"""Typed generation events (DESIGN.md §5.1) — the event-sourced spine.

The scheduler emits these; TUI, benchmarks, logs, and the future server are
consumers only (DESIGN invariant 2). Field names are the §5.1 wire keys
verbatim (pos/id/conf, old, span, ...), so JSONL logs and future DSP frames
are these dataclasses serialized with no mapping layer.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Union


@dataclass(frozen=True, slots=True)
class CommitItem:
    """One inked token, as it appears in tokens_committed items."""

    pos: int
    id: int
    conf: float


@dataclass(frozen=True, slots=True)
class ReviseItem:
    """One re-masked token (remask_lowconf, §5.2); unused until revisions exist."""

    pos: int
    old: int
    id: int
    conf: float


@dataclass(frozen=True, slots=True)
class WorkspaceReadoutItem:
    """One named latent-workspace readout score."""

    label: str
    score: float


@dataclass(frozen=True, slots=True)
class GenStarted:
    t: int
    prompt_tokens: int
    block_len: int  # 0 = whole-sequence mode (§5.4)
    max_new: int


@dataclass(frozen=True, slots=True)
class BlockStarted:
    t: int
    block: int
    span: tuple[int, int]  # [start, end) board positions


@dataclass(frozen=True, slots=True)
class TokensCommitted:
    t: int
    block: int
    items: tuple[CommitItem, ...]


@dataclass(frozen=True, slots=True)
class TokensRevised:
    t: int
    block: int
    items: tuple[ReviseItem, ...]


@dataclass(frozen=True, slots=True)
class StepStats:
    t: int
    block: int
    step: int
    committed: int
    remaining: int
    ms: float
    cache_hit: float  # 0.0 until the cache tiers exist (§5.5)


@dataclass(frozen=True, slots=True)
class BlockFinalized:
    t: int
    block: int
    text: str
    steps_used: int


@dataclass(frozen=True, slots=True)
class GenFinished:
    t: int
    reason: str  # "eos" | "length" | "steps_exhausted"
    new_tokens: int
    wall_ms: float
    steps_total: int
    tok_per_s: float


@dataclass(frozen=True, slots=True)
class WorkspaceReadout:
    """Latent workspace readout for one token/layer position.

    Placeholder providers can emit this today; real adapters can later fill the
    same payload from logit lens, Jacobian Lens, SAE probes, or linear probes.
    """

    t: int
    run_id: str
    token_index: int
    token_text: str
    layer: int
    position: int
    top_readouts: tuple[WorkspaceReadoutItem, ...]
    entropy: float
    provider: str


Event = Union[
    GenStarted,
    BlockStarted,
    TokensCommitted,
    TokensRevised,
    StepStats,
    BlockFinalized,
    GenFinished,
    WorkspaceReadout,
]

_TYPE_NAMES: dict[type, str] = {
    GenStarted: "gen_started",
    BlockStarted: "block_started",
    TokensCommitted: "tokens_committed",
    TokensRevised: "tokens_revised",
    StepStats: "step_stats",
    BlockFinalized: "block_finalized",
    GenFinished: "gen_finished",
    WorkspaceReadout: "workspace_readout",
}


def event_to_dict(event: Event) -> dict:
    """§5.1 wire form: {"t": ..., "type": ..., **payload}."""
    d = asdict(event)
    return {"t": d.pop("t"), "type": _TYPE_NAMES[type(event)], **d}


def to_jsonl_line(event: Event) -> str:
    return json.dumps(event_to_dict(event), ensure_ascii=False)


def write_jsonl(events: Iterable[Event], path: str | Path) -> None:
    """Flight-recorder log: one event per line, replayable (§5.1)."""
    with open(path, "w", encoding="utf-8") as f:
        for event in events:
            f.write(to_jsonl_line(event) + "\n")
