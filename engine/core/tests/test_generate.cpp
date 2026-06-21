// test_generate.cpp — control-flow checks for the C++ generate pass loop, driven by a
// deterministic in-test FakeAdapter (no ggml, no model). This runs in the backend-free CI
// build: it locks the loop's wiring of policies + stepper + blocks + cache + sample that the
// ggml driver also exercises, but without needing a GGUF or llama.cpp.
//
// It does NOT reproduce the Python fake goldens — those depend on numpy's RNG, which isn't
// bit-portable to C++ (the same reason sample_candidates is greedy-only here). Instead the
// FakeAdapter is position-deterministic so the picks are predictable from first principles:
// position p always argmaxes to token f(p) = p % (vocab-1), peaked so every committed slot is
// near-equally confident; confidence ties break toward the lower position (confidence_topk),
// so quota fills strictly left-to-right and the final board is exactly f(p) per slot.
#include "cloze/generate.hpp"
#include "cloze/model.hpp"
#include "fake_adapter.hpp"  // the deterministic, model-free adapter

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace cloze;

namespace {
const std::vector<int> PROMPT = {1, 2, 3, 4, 5};  // p = 5; first slot is position 5
}  // namespace

int main() {
    // (a) whole-sequence quota fill: positions 5..8 -> f = {5,6,7,8}, drains in 4 steps.
    {
        FakeAdapter fake(16, /*eos=*/-1, /*eos_at=*/-1);
        GenerateConfig cfg{/*max_new=*/4, /*steps=*/4, /*block_len=*/0, /*topk=*/-1};
        GenerateResult r = generate(fake, PROMPT, cfg);
        assert((r.generated == std::vector<int>{5, 6, 7, 8}));
        assert(r.reason == "length");
        assert(r.new_tokens == 4);
    }

    // (b) block mode and delta KV reuse produce the IDENTICAL board (the adapter is
    // position-deterministic, so only the loop's block/cache wiring is under test).
    {
        FakeAdapter fake(16, -1, -1);
        GenerateConfig cfg{4, 4, /*block_len=*/2, -1};  // blocks [5,7), [7,9)
        GenerateResult whole = generate(fake, PROMPT, GenerateConfig{4, 4, 0, -1});
        GenerateResult blocks = generate(fake, PROMPT, cfg);
        CacheConfig delta;
        delta.mode = "delta";
        delta.full_refresh_every = 1;
        GenerateResult reuse = generate(fake, PROMPT, cfg, delta);
        assert(blocks.board == whole.board);
        assert(reuse.board == whole.board);
        assert(blocks.reason == "length" && reuse.reason == "length");
    }

    // (c) EOS at position 6 finishes the run: output truncates before it, reason "eos".
    {
        FakeAdapter fake(16, /*eos=*/2, /*eos_at=*/6);
        GenerateConfig cfg{4, 4, 0, -1};
        GenerateResult r = generate(fake, PROMPT, cfg);
        // step order (lowest pos first): pos5->5, pos6->EOS(2), pos7->7, pos8->8.
        assert((r.generated == std::vector<int>{5}));  // only pos5 survives truncation
        assert(r.reason == "eos");
        assert(r.new_tokens == 1);
    }

    // (d) steps exhausted: fixed k=1 over 2 steps leaves 2 of 4 slots masked.
    {
        FakeAdapter fake(16, -1, -1);
        GenerateConfig cfg{/*max_new=*/4, /*steps=*/2, /*block_len=*/0, /*topk=*/1};
        GenerateResult r = generate(fake, PROMPT, cfg);
        const int mask = fake.config().mask_token_id;
        int holes = 0;
        for (size_t i = 5; i < r.board.size(); ++i)
            if (r.board[i] == mask) ++holes;
        assert(holes == 2);
        assert(r.board[5] == 5 && r.board[6] == 6);  // first two filled left-to-right
        assert(r.reason == "steps_exhausted");
    }

    std::printf("test_generate: all assertions passed\n");
    return 0;
}
