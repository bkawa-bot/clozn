// blocks.cpp — implementation of cloze/blocks.hpp, mirroring blocks.py.
#include "cloze/blocks.hpp"

#include <algorithm>
#include <stdexcept>

namespace cloze {

std::vector<Block> BlockPlan::blocks() const {
    if (prompt_len < 1) throw std::invalid_argument("prompt_len must be >= 1");
    if (max_new < 1) throw std::invalid_argument("max_new must be >= 1");
    if (block_len < 0) throw std::invalid_argument("block_len must be >= 0 (0 = whole-sequence)");

    const int start = prompt_len;
    const int end = prompt_len + max_new;
    std::vector<Block> out;
    if (block_len == 0) {
        out.push_back(Block{0, start, end});
        return out;
    }
    for (int pos = start; pos < end; pos += block_len) {
        const int b_end = (pos + block_len < end) ? pos + block_len : end;
        out.push_back(Block{static_cast<int>(out.size()), pos, b_end});
    }
    return out;
}

int block_id(int pos, int prompt_len, int block_len) {
    if (pos < prompt_len) return -1;
    return (pos - prompt_len) / block_len;
}

Mask attention_mask(int working_len, int prompt_len, int block_len) {
    Mask m;
    m.n = working_len;
    m.data.assign(static_cast<size_t>(working_len) * working_len, 0);
    if (block_len == 0) {
        // fully bidirectional: all attend.
        std::fill(m.data.begin(), m.data.end(), static_cast<unsigned char>(1));
        return m;
    }
    std::vector<int> ids(working_len);
    for (int p = 0; p < working_len; ++p) ids[p] = block_id(p, prompt_len, block_len);
    // M[q, k] = ids[k] <= ids[q]  (the one-way law).
    for (int q = 0; q < working_len; ++q)
        for (int k = 0; k < working_len; ++k)
            m.data[static_cast<size_t>(q) * working_len + k] = (ids[k] <= ids[q]) ? 1 : 0;
    return m;
}

}  // namespace cloze
