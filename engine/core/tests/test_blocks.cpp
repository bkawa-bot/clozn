// test_blocks.cpp — checks for the C++ blocks port, mirroring test_blocks.py.
#include "cloze/blocks.hpp"

#include <cassert>
#include <cstdio>

using namespace cloze;

int main() {
    // whole-sequence: one block spanning the whole output region.
    {
        auto bs = BlockPlan{5, 8, 0}.blocks();
        assert(bs.size() == 1);
        assert(bs[0].start == 5 && bs[0].end == 13);
    }
    // block mode, even division: 8 over block_len 4 => [5,9), [9,13).
    {
        auto bs = BlockPlan{5, 8, 4}.blocks();
        assert(bs.size() == 2);
        assert(bs[0].start == 5 && bs[0].end == 9);
        assert(bs[1].start == 9 && bs[1].end == 13);
    }
    // partial last block: 8 over block_len 3 => 3, 3, 2.
    {
        auto bs = BlockPlan{5, 8, 3}.blocks();
        assert(bs.size() == 3);
        assert(bs[2].start == 11 && bs[2].end == 13);
    }
    // block_id: -1 for prompt, 0/1/... for output blocks.
    {
        assert(block_id(4, 5, 4) == -1);
        assert(block_id(5, 5, 4) == 0);
        assert(block_id(9, 5, 4) == 1);
    }
    // the one-way law: prompt(0..4) + one block(5..8), working_len = 9.
    {
        auto m = attention_mask(9, /*prompt_len=*/5, /*block_len=*/4);
        assert(m.at(5, 0) == true);   // block sees the prompt
        assert(m.at(5, 8) == true);   // bidirectional within the block
        assert(m.at(0, 5) == false);  // prompt does NOT see the block (no forward attention)
        assert(m.at(8, 5) == true);   // within block, both directions
    }
    // whole-sequence mask: fully bidirectional.
    {
        auto m = attention_mask(6, 5, 0);
        assert(m.at(0, 5) == true && m.at(5, 0) == true);
    }
    std::printf("test_blocks: all assertions passed\n");
    return 0;
}
