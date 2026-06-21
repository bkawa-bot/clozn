"""Tests for the KV cache manager (DESIGN §5.5)."""

import pytest

from cloze_lab.scheduler.cache import CacheConfig, CacheManager


class _StubKV:
    """Minimal KVState — the manager only stores/passes it, never inspects it."""

    def __init__(self, n: int) -> None:
        self.seq_len = n


def mgr(mode: str = "delta", **kw) -> CacheManager:
    return CacheManager(config=CacheConfig(mode=mode, **kw))


class TestCacheConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"mode": "bogus"},
            {"full_refresh_every": 0},
            {"refresh_fraction": -0.1},
            {"refresh_fraction": 1.1},
        ],
    )
    def test_validation(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            CacheConfig(**kwargs)

    def test_defaults(self) -> None:
        c = CacheConfig()
        assert c.mode == "off" and c.full_refresh_every == 4 and c.refresh_fraction == 0.5


class TestOffMode:
    def test_always_recomputes_all_and_threads_no_kv(self) -> None:
        m = mgr(mode="off")
        board = [10, 11, 63, 63]
        for step in range(3):
            plan = m.plan(board, active=(2, 4), block_step=step)
            assert plan.recompute_kv is None  # all
            assert plan.kv is None  # never reuse
            assert plan.cache_hit == 0.0
            m.observe(board, _StubKV(4), None)

    def test_never_reuses_even_after_state_is_populated(self) -> None:
        # The off guard lives in plan(): even with drawers + cached_token recorded,
        # off must still full-recompute and thread no kv (this is what stops an
        # adapter without KV support from being handed a reuse plan).
        m = mgr(mode="off")
        m.observe([10, 11, 63, 63], _StubKV(4), None)  # populate _kv and cached_token
        plan = m.plan([10, 11, 30, 63], active=(2, 4), block_step=1)
        assert plan.kv is None and plan.recompute_kv is None


class TestDeltaMode:
    def test_cold_start_recomputes_all(self) -> None:
        m = mgr()
        plan = m.plan([10, 11, 63, 63], active=(2, 4), block_step=0)
        assert plan.recompute_kv is None and plan.kv is None

    def test_only_changed_recomputed_between_refreshes(self) -> None:
        m = mgr(full_refresh_every=4)
        b0 = [10, 11, 63, 63]
        m.observe(b0, _StubKV(4), None)  # cold start: cached_token = all of b0
        # commit position 2 (63 -> 30); block_step 1 (not a full-refresh step)
        b1 = [10, 11, 30, 63]
        plan = m.plan(b1, active=(2, 4), block_step=1)
        assert plan.recompute_kv == [2]  # only the changed position
        assert plan.kv is not None  # reuses prior drawers
        assert plan.cache_hit == pytest.approx(1 - 1 / 4)

    def test_full_refresh_recomputes_everything(self) -> None:
        m = mgr(full_refresh_every=4)
        b0 = [10, 11, 63, 63]
        m.observe(b0, _StubKV(4), None)
        b1 = [10, 11, 30, 63]
        plan = m.plan(b1, active=(2, 4), block_step=4)  # 4 % 4 == 0 -> full refresh
        assert plan.recompute_kv == [0, 1, 2, 3]
        assert plan.cache_hit == 0.0

    def test_refresh_fraction_triggers_full(self) -> None:
        m = mgr(full_refresh_every=99, refresh_fraction=0.5)
        b0 = [10, 11, 63, 63]
        m.observe(b0, _StubKV(4), None)
        # both active positions (2,3) changed -> churn 1.0 > 0.5 -> full refresh
        b1 = [10, 11, 30, 31]
        plan = m.plan(b1, active=(2, 4), block_step=1)
        assert plan.recompute_kv == [0, 1, 2, 3]

    def test_new_positions_always_recomputed(self) -> None:
        # A grown board (next block) introduces positions absent from cached_token.
        m = mgr(full_refresh_every=99)
        m.observe([10, 11], _StubKV(2), None)  # only positions 0,1 known
        plan = m.plan([10, 11, 63, 63], active=(2, 4), block_step=1)
        assert 2 in plan.recompute_kv and 3 in plan.recompute_kv  # new positions

    def test_observe_refreshes_cached_token_only_for_recomputed(self) -> None:
        m = mgr(full_refresh_every=99)
        m.observe([10, 11, 63, 63], _StubKV(4), None)
        b1 = [10, 11, 30, 40]  # both changed, but we only recompute position 2
        m.observe(b1, _StubKV(4), [2])
        # position 3 still tracked as stale (63), so it shows as changed next plan
        plan = m.plan(b1, active=(2, 4), block_step=1)
        assert plan.recompute_kv == [3]


class TestFreeFreeze:
    """Tier B free freeze under block-causal attention (frozen_prefix=True)."""

    def _step(self, m: CacheManager, board: list, active: tuple, step: int):
        plan = m.plan(board, active, step, frozen_prefix=True)
        m.observe(board, _StubKV(len(board)), plan.recompute_kv, active=active, frozen_prefix=True)
        return plan

    def test_cold_start_freezes_the_prompt(self) -> None:
        m = mgr(full_refresh_every=2)
        self._step(m, [10, 11, 63, 63, 63, 63], active=(2, 4), step=0)  # cold start
        # block-0 step 1: the prompt [0,2) is now frozen and must not be recomputed
        plan = m.plan([10, 11, 30, 63, 63, 63], active=(2, 4), block_step=1, frozen_prefix=True)
        assert not (set(range(0, 2)) & set(plan.recompute_kv))

    def test_block_transition_recomputes_finalized_block_once(self) -> None:
        m = mgr(full_refresh_every=2)
        # block0 = [2,4): cold start + drain
        self._step(m, [10, 11, 63, 63, 63, 63], active=(2, 4), step=0)
        # block1 = [4,6): first forward must recompute the just-finalized block0 [2,4)
        # (freeze it exactly) AND the new active block1 [4,6)
        plan = m.plan([10, 11, 20, 21, 63, 63], active=(4, 6), block_step=0, frozen_prefix=True)
        assert {2, 3} <= set(plan.recompute_kv)  # block0 frozen exactly, once
        assert {4, 5} <= set(plan.recompute_kv)  # new active block

    def test_frozen_blocks_never_recomputed_again(self) -> None:
        m = mgr(full_refresh_every=2)
        self._step(m, [10, 11, 63, 63, 63, 63], active=(2, 4), step=0)  # block0
        self._step(m, [10, 11, 20, 21, 63, 63], active=(4, 6), step=0)  # block1 first (freezes block0)
        # block1 step 1: prompt + block0 [0,4) are frozen — never recomputed
        plan = m.plan([10, 11, 20, 21, 30, 63], active=(4, 6), block_step=1, frozen_prefix=True)
        assert not (set(range(0, 4)) & set(plan.recompute_kv))
        assert all(p >= 4 for p in plan.recompute_kv)

    def test_frozen_excluded_even_on_a_full_refresh(self) -> None:
        # The free freeze itself: a full-refresh step recomputes everything that can
        # drift, but NEVER the frozen prefix (that is the whole point — frozen blocks
        # cost zero recompute, even on exact steps).
        m = mgr(full_refresh_every=2)
        self._step(m, [10, 11, 63, 63, 63, 63], active=(2, 4), step=0)  # block0
        self._step(m, [10, 11, 20, 21, 63, 63], active=(4, 6), step=0)  # block1 first; freezes block0
        # block_step 2 => full refresh (2 % 2 == 0), prefix [0,4) still frozen
        plan = m.plan([10, 11, 20, 21, 30, 31], active=(4, 6), block_step=2, frozen_prefix=True)
        assert not (set(range(0, 4)) & set(plan.recompute_kv))  # frozen prefix excluded
        assert {4, 5} <= set(plan.recompute_kv)  # active block fully refreshed

    def test_whole_sequence_has_no_frozen_region(self) -> None:
        # frozen_prefix=False (the default) keeps the pre-free-freeze behavior.
        m = mgr(full_refresh_every=99)
        m.observe([10, 11, 63, 63], _StubKV(4), None)  # no frozen_prefix => boundary stays 0
        plan = m.plan([10, 11, 30, 63], active=(2, 4), block_step=1)
        assert plan.recompute_kv == [2]  # only the changed position; nothing frozen
