// cloze/blocks.hpp — block manager + masks (DESIGN §5.4), C++ port of
// lab/cloze_lab/scheduler/blocks.py. Semi-autoregressive block diffusion: generate the
// output left->right in blocks; attention is block-causal (the one-way law) — a position
// attends to its own block and every earlier one, never forward, which is what keeps
// frozen-block K/V exact (the basis for the Tier B cache).
#pragma once

#include <vector>

namespace cloze {

// One output block: board positions [start, end).
struct Block {
    int index;
    int start;
    int end;
};

// Left-to-right blocks over the output region [prompt_len, prompt_len + max_new).
// block_len == 0 => one whole-sequence block; > 0 => semi-AR blocks (last may be shorter).
struct BlockPlan {
    int prompt_len;
    int max_new;
    int block_len;

    bool whole_sequence() const { return block_len == 0; }
    std::vector<Block> blocks() const;  // throws on invalid dimensions
};

// -1 for prompt positions; 0, 1, 2, ... for successive output blocks.
int block_id(int pos, int prompt_len, int block_len);

// [n, n] row-major boolean mask; at(q, k) true means query q may attend to key k.
struct Mask {
    int n = 0;
    std::vector<unsigned char> data;  // n*n, 1 = attend
    bool at(int q, int k) const { return data[static_cast<size_t>(q) * n + k] != 0; }
};

// whole-sequence (block_len=0): fully bidirectional. Block mode: M[q,k] = block_id(k) <=
// block_id(q) — the prompt attends only to itself, each block sees the prompt + earlier
// blocks + bidirectionally within itself, nothing forward (the one-way law).
Mask attention_mask(int working_len, int prompt_len, int block_len);

}  // namespace cloze
