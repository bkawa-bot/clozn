"""KV cache manager (DESIGN.md §5.5): decide which positions to recompute each pass.

Pure scheduler logic — it never fabricates K/V (invariant 4); it only chooses
``recompute_kv`` and threads the adapter's ``KVState`` plus the ``cached_token``
built-as labels that reconcile board and drawers. Three tiers, by region:

* **Tier A (prompt) / Tier B (frozen blocks)** — exact and *free* under block-causal
  attention (``frozen_prefix=True``). When a block finalizes, its context is
  permanently frozen (the one-way law), so its K/V is recomputed exactly **once** —
  folded into the next block's first forward, the moment the block's tokens are
  final — and then never again. The ``_frozen_until`` boundary tracks how far this
  exact, no-recompute region extends.
* **Tier C (active block)** — approximate. Each step recomputes the positions whose
  token changed (newly committed) and reuses the rest. Reused positions whose
  *context* shifted (a neighbor committed) carry stale K/V — the drift. A periodic
  full refresh (``full_refresh_every``) and a churn trigger (``refresh_fraction``)
  bound it. ``mode="off"`` recomputes everything every pass (exact, the baseline
  the divergence bench measures against).

Whole-sequence mode (``frozen_prefix=False``) has no frozen region — the prompt
attends to the output, so nothing's context is frozen (DESIGN: "no inter-block
caching"); a full refresh there is simply a recompute of everything.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from cloze_lab.models.base import KVState

_MODES = ("off", "delta")


@dataclass(frozen=True, slots=True)
class CacheConfig:
    """Exactness knobs, exposed never hidden (DESIGN invariant 5)."""

    mode: str = "off"  # "off" = exact every pass | "delta" = Tier C reuse
    full_refresh_every: int = 4  # force a full active-block refresh every N block-steps
    refresh_fraction: float = 0.5  # if >this fraction of the active block changed, full refresh

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError(f"mode must be one of {_MODES}, got {self.mode!r}")
        if self.full_refresh_every < 1:
            raise ValueError(f"full_refresh_every must be >= 1, got {self.full_refresh_every}")
        if not 0.0 <= self.refresh_fraction <= 1.0:
            raise ValueError(f"refresh_fraction must be in [0, 1], got {self.refresh_fraction}")


@dataclass(frozen=True, slots=True)
class ForwardPlan:
    """The cache's decision for one forward."""

    kv: KVState | None  # drawers to reuse from; None = cold start (recompute all)
    recompute_kv: list[int] | None  # positions to rebuild; None = all
    cache_hit: float  # fraction of working positions reused (0.0 when recomputing all)


@dataclass(slots=True)
class CacheManager:
    """Threads K/V and cached_token across the passes of one generation (§5.5)."""

    config: CacheConfig
    _kv: KVState | None = field(default=None, init=False)
    _cached_token: dict[int, int] = field(default_factory=dict, init=False)
    _frozen_until: int = field(default=0, init=False)  # [0, _frozen_until) is frozen-exact

    def plan(
        self,
        board: Sequence[int],
        active: tuple[int, int],
        block_step: int,
        *,
        frozen_prefix: bool = False,
    ) -> ForwardPlan:
        """Decide reuse for the forward over ``board`` whose active block is ``active``.

        Between refreshes only the positions whose token changed (the newly
        committed ones) are rebuilt; everything else is reused, so a reused
        position whose *context* shifted carries stale K/V — the Tier C drift. A
        full refresh is an exact step over the non-frozen region, which bounds the
        drift and makes ``full_refresh_every=1`` identical to ``off``.

        Under ``frozen_prefix`` (block-causal attention), [0, _frozen_until) is
        frozen-exact and never recomputed. A block's first forward is always
        ``block_step == 0`` — a full refresh — which recomputes the just-finalized
        block (still un-frozen, its tokens now final) exactly once; ``observe`` then
        advances the boundary, freezing it. So no separate freeze pass is needed.
        """
        n = len(board)
        # off (the single load-bearing guard: never reuse, so adapters without KV
        # support stay happy), or cold start (no drawers yet): recompute all, no kv.
        if self.config.mode == "off" or self._kv is None:
            return ForwardPlan(kv=None, recompute_kv=None, cache_hit=0.0)

        lo, hi = active
        frozen = set(range(self._frozen_until)) if frozen_prefix else set()
        active_positions = set(range(lo, hi))
        new = {p for p in range(n) if p not in self._cached_token} - frozen
        changed = {p for p in self._cached_token if int(board[p]) != self._cached_token[p]} - frozen
        churn = len(changed & active_positions) / len(active_positions) if active_positions else 0.0
        full = block_step % self.config.full_refresh_every == 0 or churn > self.config.refresh_fraction

        # Frozen positions are excluded from BOTH branches — that exclusion is the
        # free freeze (a full refresh recomputes everything *not* frozen).
        recompute = (set(range(n)) - frozen) if full else (new | changed)
        ordered = sorted(recompute)
        cache_hit = 1.0 - len(ordered) / n if n else 0.0
        return ForwardPlan(kv=self._kv, recompute_kv=ordered, cache_hit=cache_hit)

    def observe(
        self,
        board: Sequence[int],
        new_kv: KVState,
        recomputed: list[int] | None,
        *,
        active: tuple[int, int] | None = None,
        frozen_prefix: bool = False,
    ) -> None:
        """Record the forward's drawers, refresh cached_token, and advance the freeze
        boundary: under ``frozen_prefix`` the region up to ``active.start`` was just
        recomputed exactly, so it becomes frozen-exact.

        (Off mode also records here, but ``plan`` ignores the stored state and
        always full-recomputes — so the off guard lives in one place, ``plan``.)
        """
        self._kv = new_kv
        if recomputed is None:
            self._cached_token = {p: int(board[p]) for p in range(len(board))}
        else:
            for p in recomputed:
                self._cached_token[p] = int(board[p])
        if frozen_prefix and active is not None:
            self._frozen_until = active[0]
