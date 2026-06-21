// test_revision.cpp — the remask_lowconf revision path (DESIGN §5.2), the "model changes its mind"
// feature, ported from lab/tests/test_revision.py::TestRevisionEndToEnd. Backend-free: the
// position-deterministic FakeAdapter peaks every committed slot to ~0.995 confidence, so
// tau_revise=1.0 re-masks every committed token once (cap=1) — a hard stress test that the
// per-position cap still guarantees termination under maximally aggressive revision. Also pins the
// opt-in contract: with revision off, the loop is byte-identical to the committed goldens.
#include "cloze/generate.hpp"
#include "cloze/model.hpp"
#include "fake_adapter.hpp"

#include <cassert>
#include <cstdio>
#include <map>
#include <variant>
#include <vector>

using namespace cloze;

namespace {
const std::vector<int> PROMPT = {1, 2, 3, 4, 5};  // p = 5; generated region is positions 5..8
}  // namespace

int main() {
    // (a) revisions fire, respect the per-position cap, stay in the generated region, terminate.
    {
        FakeAdapter fake(16, /*eos=*/-1, /*eos_at=*/-1);
        const int mask = fake.config().mask_token_id;
        const int p = static_cast<int>(PROMPT.size());
        GenerateConfig cfg{/*max_new=*/4, /*steps=*/20, /*block_len=*/0, /*topk=*/-1};
        // tau_revise=1.0 re-masks every committed token once (cap=1): maximal churn, must still drain.
        ReviseConfig revise{/*enabled=*/true, /*tau_revise=*/1.0, /*max_revisions=*/1};
        GenerateResult r = generate(fake, PROMPT, cfg, CacheConfig{}, nullptr, {}, revise);

        int n_revised_events = 0;
        std::map<int, int> per_pos;
        for (const Event& e : r.events) {
            if (const auto* tr = std::get_if<TokensRevised>(&e)) {
                ++n_revised_events;
                for (const ReviseItem& it : tr->items) {
                    assert(it.old != mask);            // a real, previously-committed token was retracted
                    assert(it.conf < 1.0);             // below tau_revise
                    assert(p <= it.pos && it.pos < p + cfg.max_new);  // generated region only
                    per_pos[it.pos] += 1;
                }
            }
        }
        assert(n_revised_events > 0);                  // revision actually happened
        for (const auto& kv : per_pos) assert(kv.second <= revise.max_revisions);  // cap honored
        assert(std::holds_alternative<GenFinished>(r.events.back()));  // it terminated
        assert(r.reason == "length");                  // and fully drained despite the churn
        assert((r.generated == std::vector<int>{5, 6, 7, 8}));  // same final board as the no-revise run
    }

    // (b) opt-in: revision off => no tokens_revised events and the byte-identical committed board.
    {
        FakeAdapter fake(16, -1, -1);
        GenerateConfig cfg{/*max_new=*/4, /*steps=*/8, /*block_len=*/0, /*topk=*/-1};
        GenerateResult r = generate(fake, PROMPT, cfg);  // default ReviseConfig: disabled
        for (const Event& e : r.events) assert(!std::holds_alternative<TokensRevised>(e));
        assert((r.generated == std::vector<int>{5, 6, 7, 8}));
    }

    // (c) infill revises too — the gap's low-confidence fills can be reconsidered with both sides.
    {
        FakeAdapter fake(16, -1, -1);
        const std::vector<int> prefix = {1, 2};
        const std::vector<int> suffix = {9, 10};
        GenerateConfig cfg{/*max_new=*/0, /*steps=*/20, /*block_len=*/0, /*topk=*/-1};
        ReviseConfig revise{true, 1.0, 1};
        GenerateResult r = infill(fake, prefix, suffix, /*gap=*/3, cfg, nullptr, {}, revise);
        int n_revised = 0;
        for (const Event& e : r.events)
            if (std::holds_alternative<TokensRevised>(e)) ++n_revised;
        assert(n_revised > 0);
        assert(std::holds_alternative<GenFinished>(r.events.back()));
        assert(r.reason == "length");  // gap fully filled
    }

    // (d) denoise() fills scattered holes and leaves fixed context untouched — the "revise this
    // selection" primitive. board has masks at positions 2, 4, 5; everything else is fixed (7).
    {
        FakeAdapter fake(16, -1, -1);
        const int mask = fake.config().mask_token_id;  // 15
        std::vector<int> board = {7, 7, mask, 7, mask, mask, 7};
        GenerateConfig cfg{/*max_new=*/0, /*steps=*/8, /*block_len=*/0, /*topk=*/-1};
        GenerateResult r = denoise(fake, board, cfg);
        // FakeAdapter argmaxes position p to f(p) = p % 15, so the holes resolve to their own index.
        assert((r.board == std::vector<int>{7, 7, 2, 7, 4, 5, 7}));  // fixed slots untouched, holes filled
        assert(r.reason == "length");
        assert(r.new_tokens == 3);
        assert(std::holds_alternative<GenFinished>(r.events.back()));
    }

    // (e) denoise() with the remask reviser still fills and terminates (holes can be reconsidered).
    {
        FakeAdapter fake(16, -1, -1);
        const int mask = fake.config().mask_token_id;
        std::vector<int> board = {1, mask, mask, 4};
        GenerateConfig cfg{/*max_new=*/0, /*steps=*/20, /*block_len=*/0, /*topk=*/-1};
        GenerateResult r = denoise(fake, board, cfg, nullptr, {}, ReviseConfig{true, 1.0, 1});
        assert((r.board == std::vector<int>{1, 1, 2, 4}));  // pos1->f(1)=1, pos2->f(2)=2
        assert(r.reason == "length");
        int n_revised = 0;
        for (const Event& e : r.events)
            if (std::holds_alternative<TokensRevised>(e)) ++n_revised;
        assert(n_revised > 0);
    }

    std::printf("test_revision: all assertions passed\n");
    return 0;
}
