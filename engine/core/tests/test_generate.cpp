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
#include "cloze/sample.hpp"
#include "fake_adapter.hpp"  // the deterministic, model-free adapter

#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace cloze;

namespace {
const std::vector<int> PROMPT = {1, 2, 3, 4, 5};  // p = 5; first slot is position 5

class OnlyTokenConstraint final : public TokenConstraint {
public:
    explicit OnlyTokenConstraint(int allowed) : allowed_(allowed) {}

    void apply(std::vector<float>& logits) override {
        ++apply_calls;
        for (size_t i = 0; i < logits.size(); ++i) {
            if (static_cast<int>(i) != allowed_) {
                logits[i] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    void accept(int token) override { accepted.push_back(token); }

    int apply_calls = 0;
    std::vector<int> accepted;

private:
    int allowed_;
};
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

    // (e) AR constraints run before selection and atomically accept the token
    // that was actually committed. The unconstrained argmax is token 0; the
    // constraint forces token 2 and observes exactly that token once.
    {
        ForwardResult fwd;
        fwd.logits = {9.0f, 3.0f, 1.0f};
        fwd.n_requested = 1;
        fwd.vocab = 3;
        OnlyTokenConstraint constraint(/*allowed=*/2);
        SampleOpts opts;
        opts.constraint = &constraint;
        const Candidate candidate = sample_committed_candidate(fwd, /*position=*/7, opts);
        assert(candidate.pos == 7);
        assert(candidate.token_id == 2);
        assert(constraint.apply_calls == 1);
        assert((constraint.accepted == std::vector<int>{2}));
    }

    // A stateful constraint is intentionally one-row/AR-only. This prevents a
    // caller from accidentally applying one grammar state across diffusion slots.
    {
        ForwardResult fwd;
        fwd.logits = {1.0f, 0.0f, 0.0f, 1.0f};
        fwd.n_requested = 2;
        fwd.vocab = 2;
        OnlyTokenConstraint constraint(/*allowed=*/0);
        SampleOpts opts;
        opts.constraint = &constraint;
        bool rejected = false;
        try {
            (void)sample_candidates(fwd, {0, 1}, opts);
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }

    // A constraint that leaves no legal token fails closed; it must never fall
    // through to token 0 because a softmax over all -inf logits is undefined.
    {
        ForwardResult fwd;
        fwd.logits = {3.0f, 2.0f};
        fwd.n_requested = 1;
        fwd.vocab = 2;
        OnlyTokenConstraint constraint(/*allowed outside vocab=*/9);
        SampleOpts opts;
        opts.constraint = &constraint;
        bool rejected = false;
        try {
            (void)sample_committed_candidate(fwd, /*position=*/0, opts);
        } catch (const std::runtime_error&) {
            rejected = true;
        }
        assert(rejected);
        assert(constraint.accepted.empty());
    }

    // Lazy tool grammar gating follows token sequences, not decoded substring guesses: it becomes
    // active only after the complete start tag, releases after the complete end tag, and re-arms
    // for a later reasoning block in the same response.
    {
        ReasoningBlockGate gate({10, 11}, {20, 21});
        assert(!gate.active());
        gate.accept(10);
        assert(!gate.active());
        gate.accept(11);
        assert(gate.active());
        gate.accept(99);
        gate.accept(20);
        assert(gate.active());
        gate.accept(21);
        assert(!gate.active());
        gate.accept(10);
        gate.accept(11);
        assert(gate.active());
        gate.accept(20);
        gate.accept(21);
        assert(!gate.active());
    }

    // A half-specified reasoning boundary cannot safely gate a lazy grammar.
    {
        bool rejected = false;
        try {
            (void)ReasoningBlockGate({}, {20});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }

    std::printf("test_generate: all assertions passed\n");
    return 0;
}
