"""Contract tests for the ModelAdapter seam, exercised through FakeAdapter."""

import numpy as np
import pytest

from cloze_lab.models.base import (
    Family,
    ForwardResult,
    KVState,
    ModelAdapter,
    ModelConfig,
)
from cloze_lab.models.fake import FakeAdapter, FakeKV


def full_mask(n: int) -> np.ndarray:
    return np.ones((n, n), dtype=bool)


@pytest.fixture
def fake() -> FakeAdapter:
    return FakeAdapter(seed=7)


class TestModelConfig:
    def test_valid(self) -> None:
        cfg = ModelConfig(family=Family.DREAM, vocab_size=100, mask_token_id=99)
        assert cfg.block_length == 32

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"vocab_size": 0, "mask_token_id": 0},
            {"vocab_size": 10, "mask_token_id": 10},
            {"vocab_size": 10, "mask_token_id": 9, "eos_token_id": 10},
            {"vocab_size": 10, "mask_token_id": 9, "eos_token_id": 9},
            {"vocab_size": 10, "mask_token_id": 9, "default_steps": 0},
            {"vocab_size": 10, "mask_token_id": 9, "block_length": -1},
        ],
    )
    def test_invalid(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            ModelConfig(family=Family.DREAM, **kwargs)


class TestProtocolConformance:
    def test_fake_is_model_adapter(self, fake: FakeAdapter) -> None:
        assert isinstance(fake, ModelAdapter)

    def test_forward_returns_kv_state(self, fake: FakeAdapter) -> None:
        result = fake.forward(np.array([2, 3, 4]), full_mask(3))
        assert isinstance(result, ForwardResult)
        assert isinstance(result.kv, KVState)
        assert result.kv.seq_len == 3


class TestForwardShapes:
    def test_full(self, fake: FakeAdapter) -> None:
        result = fake.forward(np.array([2, 3, 4, 5]), full_mask(4))
        assert result.logits.shape == (4, fake.config.vocab_size)
        assert result.logits.dtype == np.float32

    def test_logits_for_subset_aligns_with_full(self, fake: FakeAdapter) -> None:
        board = np.array([2, 3, 4, 5, 6])
        full = fake.forward(board, full_mask(5))
        sub = fake.forward(board, full_mask(5), logits_for=[1, 3])
        assert sub.logits.shape == (2, fake.config.vocab_size)
        assert np.array_equal(sub.logits[0], full.logits[1])
        assert np.array_equal(sub.logits[1], full.logits[3])

    def test_empty_logits_for(self, fake: FakeAdapter) -> None:
        result = fake.forward(np.array([2, 3]), full_mask(2), logits_for=[])
        assert result.logits.shape == (0, fake.config.vocab_size)


class TestDeterminism:
    def test_same_seed_same_output(self) -> None:
        board = np.array([2, 9, 4])
        a = FakeAdapter(seed=3).forward(board, full_mask(3))
        b = FakeAdapter(seed=3).forward(board, full_mask(3))
        assert np.array_equal(a.logits, b.logits)
        assert isinstance(a.kv, FakeKV) and isinstance(b.kv, FakeKV)
        for p in range(3):
            assert np.array_equal(a.kv.entries[p].vec, b.kv.entries[p].vec)

    def test_different_seed_different_output(self) -> None:
        board = np.array([2, 9, 4])
        a = FakeAdapter(seed=3).forward(board, full_mask(3))
        b = FakeAdapter(seed=4).forward(board, full_mask(3))
        assert not np.array_equal(a.logits, b.logits)

    def test_adapter_is_stateless_across_calls(self, fake: FakeAdapter) -> None:
        board = np.array([2, 9, 4])
        first = fake.forward(board, full_mask(3))
        second = fake.forward(board, full_mask(3))
        assert np.array_equal(first.logits, second.logits)


class TestMaskVisibility:
    def test_change_propagates_only_through_visibility(self, fake: FakeAdapter) -> None:
        # Causal mask: position q sees {0..q}. logits[p] read the fingerprints of
        # p's visible neighbors, so a change at the LAST position reaches only its
        # own logits (nothing earlier sees it).
        causal = np.tril(np.ones((4, 4), dtype=bool))
        la = fake.forward(np.array([3, 4, 5, 6]), causal).logits
        lb = fake.forward(np.array([3, 4, 5, 9]), causal).logits  # change position 3
        assert np.array_equal(la[0], lb[0])
        assert np.array_equal(la[1], lb[1])
        assert np.array_equal(la[2], lb[2])
        assert not np.array_equal(la[3], lb[3])  # position 3 sees itself

    def test_change_to_shared_neighbor_reaches_all_who_see_it(self, fake: FakeAdapter) -> None:
        causal = np.tril(np.ones((4, 4), dtype=bool))
        la = fake.forward(np.array([3, 4, 5, 6]), causal).logits
        lb = fake.forward(np.array([3, 9, 5, 6]), causal).logits  # change position 1
        assert np.array_equal(la[0], lb[0])  # position 0 cannot see 1
        for p in (1, 2, 3):
            assert not np.array_equal(la[p], lb[p])  # all see position 1

    def test_mask_itself_changes_logits(self, fake: FakeAdapter) -> None:
        board = np.array([3, 4, 5])
        narrow = full_mask(3)
        narrow[0, 2] = False  # position 0 no longer sees 2; changes 0's fingerprint
        la = fake.forward(board, full_mask(3)).logits
        lb = fake.forward(board, narrow).logits
        assert not np.array_equal(la[0], lb[0])


class TestKVContract:
    def test_full_refresh_rebuilds_all_drawers(self, fake: FakeAdapter) -> None:
        board1 = np.array([5, 6, 7])
        board2 = np.array([5, 9, 7])
        first = fake.forward(board1, full_mask(3))
        second = fake.forward(board2, full_mask(3), kv=first.kv)  # recompute_kv=None
        assert isinstance(second.kv, FakeKV)
        assert [second.kv.entries[p].built_as for p in range(3)] == [5, 9, 7]

    def test_reused_drawer_keeps_built_as_label(self, fake: FakeAdapter) -> None:
        board1 = np.array([5, 6, 7, 8])
        board2 = np.array([5, 9, 7, 8])  # position 1 changed
        first = fake.forward(board1, full_mask(4))
        stale = fake.forward(board2, full_mask(4), kv=first.kv, recompute_kv=[0, 2, 3])
        assert isinstance(first.kv, FakeKV) and isinstance(stale.kv, FakeKV)
        drawer = stale.kv.entries[1]
        assert drawer.built_as == 6  # still the token it was built under
        assert np.array_equal(drawer.vec, first.kv.entries[1].vec)

    def test_stale_drawer_drifts_from_full_recompute(self, fake: FakeAdapter) -> None:
        # Tier C drift: reuse position 1's drawer after its token changed; the stale
        # drawer carries position 1's OLD context fingerprint, so logits reading it
        # diverge from an honest full recompute over the new board.
        board1 = np.array([5, 6, 7, 8])
        board2 = np.array([5, 9, 7, 8])  # position 1 changed
        first = fake.forward(board1, full_mask(4))
        stale = fake.forward(board2, full_mask(4), kv=first.kv, recompute_kv=[0, 2, 3])
        fresh = fake.forward(board2, full_mask(4))
        assert not np.array_equal(stale.logits, fresh.logits)  # the drift
        assert stale.kv.entries[1].built_as == 6  # the stale drawer's built-as label

    def test_frozen_context_reuse_is_exact(self, fake: FakeAdapter) -> None:
        # Tier A/B exactness: under block-causal attention a frozen position never
        # sees a later change, so its fingerprint is invariant and reusing its
        # drawer is exact (no drift).
        from cloze_lab.scheduler.blocks import attention_mask

        m = attention_mask(4, prompt_len=1, block_len=1)  # pos0 prompt, 1/2/3 blocks
        board1 = np.array([5, 6, 7, 8])
        board2 = np.array([5, 6, 9, 8])  # change block-1 (pos2); frozen pos0,1 unaffected
        k1 = fake.forward(board1, m).kv.entries
        k2 = fake.forward(board2, m).kv.entries
        assert k1[0].fp == k2[0].fp  # prompt (Tier A)
        assert k1[1].fp == k2[1].fp  # frozen block (Tier B)
        assert k1[2].fp != k2[2].fp  # the changed block itself

    def test_prefix_kv_extends_to_longer_board(self, fake: FakeAdapter) -> None:
        prefix = fake.forward(np.array([5, 6]), full_mask(2))
        grown = fake.forward(
            np.array([5, 6, 7, 8]), full_mask(4), kv=prefix.kv, recompute_kv=[2, 3]
        )
        assert isinstance(grown.kv, FakeKV)
        assert [grown.kv.entries[p].built_as for p in range(4)] == [5, 6, 7, 8]


class TestValidation:
    def test_recompute_without_kv(self, fake: FakeAdapter) -> None:
        with pytest.raises(ValueError, match="without kv"):
            fake.forward(np.array([2, 3]), full_mask(2), recompute_kv=[0])

    def test_missing_drawer(self, fake: FakeAdapter) -> None:
        prev = fake.forward(np.array([5, 6, 7]), full_mask(3))
        with pytest.raises(ValueError, match="neither recomputed nor present"):
            fake.forward(np.array([5, 6, 7, 8]), full_mask(4), kv=prev.kv, recompute_kv=[0])

    def test_kv_larger_than_board(self, fake: FakeAdapter) -> None:
        prev = fake.forward(np.array([5, 6, 7]), full_mask(3))
        with pytest.raises(ValueError, match="outside the current board"):
            fake.forward(np.array([5, 6]), full_mask(2), kv=prev.kv)

    @pytest.mark.parametrize("bad", [[3], [-1], [1, 0], [1, 1]])
    def test_bad_indices(self, fake: FakeAdapter, bad: list[int]) -> None:
        board = np.array([2, 3, 4])
        with pytest.raises(ValueError):
            fake.forward(board, full_mask(3), logits_for=bad)

    def test_bad_board_and_mask(self, fake: FakeAdapter) -> None:
        with pytest.raises(ValueError, match="1-D"):
            fake.forward(np.zeros((2, 2), dtype=np.int64), full_mask(2))
        with pytest.raises(ValueError, match="integer"):
            fake.forward(np.array([0.5, 1.5]), full_mask(2))
        with pytest.raises(ValueError, match="outside vocab"):
            fake.forward(np.array([2, 64]), full_mask(2))
        with pytest.raises(ValueError, match="bool"):
            fake.forward(np.array([2, 3]), np.ones((2, 2), dtype=np.int64))
        with pytest.raises(ValueError, match="shape"):
            fake.forward(np.array([2, 3]), full_mask(3))


class TestTokenizer:
    def test_encode_deterministic_and_in_vocab(self, fake: FakeAdapter) -> None:
        ids = fake.encode("hello cloze")
        assert ids == fake.encode("hello cloze")
        assert all(0 <= i < fake.config.vocab_size for i in ids)
        assert fake.config.mask_token_id not in ids
        assert fake.config.eos_token_id not in ids

    def test_decode_renders_masks(self, fake: FakeAdapter) -> None:
        text = fake.decode([fake.config.mask_token_id, 5])
        assert "░" in text and "<5>" in text
