"""Tests for the block manager and the one-way law (DESIGN §5.4)."""

import numpy as np
import pytest

from cloze_lab.generate import GenerateConfig, generate
from cloze_lab.models.fake import FakeAdapter
from cloze_lab.scheduler.blocks import Block, BlockPlan, attention_mask, block_id


class TestBlockPlan:
    def test_whole_sequence_is_one_block(self) -> None:
        plan = BlockPlan(prompt_len=2, max_new=8, block_len=0)
        assert plan.whole_sequence
        assert plan.blocks() == [Block(index=0, start=2, end=10)]

    def test_even_blocks(self) -> None:
        plan = BlockPlan(prompt_len=2, max_new=8, block_len=4)
        assert [b.span for b in plan.blocks()] == [(2, 6), (6, 10)]

    def test_partial_last_block(self) -> None:
        plan = BlockPlan(prompt_len=2, max_new=8, block_len=3)
        assert [b.span for b in plan.blocks()] == [(2, 5), (5, 8), (8, 10)]

    def test_block_len_exceeding_max_new_is_one_block(self) -> None:
        plan = BlockPlan(prompt_len=2, max_new=4, block_len=99)
        assert [b.span for b in plan.blocks()] == [(2, 6)]

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"prompt_len": 0, "max_new": 4, "block_len": 2},
            {"prompt_len": 2, "max_new": 0, "block_len": 2},
            {"prompt_len": 2, "max_new": 4, "block_len": -1},
        ],
    )
    def test_validation(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            BlockPlan(**kwargs)


class TestBlockId:
    def test_prompt_is_minus_one(self) -> None:
        assert block_id(0, prompt_len=2, block_len=3) == -1
        assert block_id(1, prompt_len=2, block_len=3) == -1

    def test_output_blocks_count_from_zero(self) -> None:
        # prompt_len=2, block_len=3: positions 2,3,4 -> block 0; 5,6,7 -> block 1
        assert [block_id(p, 2, 3) for p in range(2, 8)] == [0, 0, 0, 1, 1, 1]


class TestAttentionMask:
    def test_whole_sequence_is_all_true(self) -> None:
        m = attention_mask(working_len=6, prompt_len=2, block_len=0)
        assert m.shape == (6, 6)
        assert m.all()

    def test_block_causal_equals_block_id_rule(self) -> None:
        # prompt {0,1}=id -1, block0 {2,3}=id 0, block1 {4,5}=id 1
        m = attention_mask(working_len=6, prompt_len=2, block_len=2)
        ids = [block_id(p, 2, 2) for p in range(6)]
        expected = np.array([[ids[k] <= ids[q] for k in range(6)] for q in range(6)])
        assert np.array_equal(m, expected)

    def test_one_way_law_no_forward_attention(self) -> None:
        m = attention_mask(working_len=6, prompt_len=2, block_len=2)
        # prompt attends only to prompt, never forward to output
        assert not m[0, 2:].any()
        # block 0 (pos 2,3) sees prompt + itself, NOT block 1 (pos 4,5)
        assert m[2, 0] and m[2, 3] and not m[2, 4] and not m[2, 5]
        # block 1 (pos 4,5) sees everything earlier + itself
        assert m[4, :].all()

    def test_intra_block_is_bidirectional(self) -> None:
        m = attention_mask(working_len=6, prompt_len=2, block_len=2)
        assert m[2, 3] and m[3, 2]  # block 0 internal
        assert m[4, 5] and m[5, 4]  # block 1 internal


class TestOneWayLawProperty:
    """The spec-defining property: a frozen block is computed independently of any
    blocks that come after it (block-causal attention forbids forward attention)."""

    def test_first_block_tokens_independent_of_later_blocks(self) -> None:
        prompt = FakeAdapter(seed=7).encode("hello cloze")
        L = 4

        def first_block(max_new: int) -> np.ndarray:
            adapter = FakeAdapter(seed=7)
            result = generate(
                adapter, prompt, GenerateConfig(max_new=max_new, steps=L, block_len=L)
            )
            p = len(prompt)
            return result.board[p : p + L].copy()

        one = first_block(L)  # block 0 alone
        two = first_block(2 * L)  # block 0 + a second block
        three = first_block(3 * L)  # block 0 + two more blocks
        assert np.array_equal(one, two)
        assert np.array_equal(one, three)
