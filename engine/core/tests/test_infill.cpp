// test_infill.cpp — native dLLM infilling (a capability AR models structurally lack), backend-free.
// Mirrors lab/tests/test_infill.py: fill a masked gap between a prefix and suffix under full
// bidirectional attention; both fixed sides are preserved verbatim and the gap is fully filled.
#include "cloze/events.hpp"
#include "cloze/generate.hpp"
#include "fake_adapter.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

using namespace cloze;

#define CHECK(cond) do { if (!(cond)) { std::fprintf(stderr, "CHECK failed: %s (line %d)\n", #cond, __LINE__); return 3; } } while (0)

namespace {
template <class T>
bool any_of(const std::vector<Event>& evs) {
    for (const Event& e : evs) if (std::holds_alternative<T>(e)) return true;
    return false;
}
}  // namespace

int main() {
    const int MASK = 15;  // FakeAdapter(vocab=16) mask id

    // Fills the gap, preserves both sides; reason "length", new_tokens == gap.
    {
        FakeAdapter fake(16, /*eos=*/-1, /*eos_at=*/-1);
        const std::vector<int> prefix = {1, 2, 3};
        const std::vector<int> suffix = {7, 8};
        const int gap = 5;
        GenerateConfig cfg{/*max_new=*/gap, /*steps=*/8, /*block_len=*/0, /*topk=*/-1};
        GenerateResult r = infill(fake, prefix, suffix, gap, cfg);

        const int lo = (int)prefix.size(), hi = lo + gap;
        CHECK((std::vector<int>(r.board.begin(), r.board.begin() + lo) == prefix));         // prefix verbatim
        CHECK((std::vector<int>(r.board.begin() + hi, r.board.end()) == suffix));           // suffix verbatim
        for (int i = lo; i < hi; ++i) CHECK(r.board[i] != MASK);                             // gap fully filled
        CHECK(r.reason == "length");
        CHECK(r.new_tokens == gap);
        CHECK((int)r.generated.size() == gap);
        CHECK(std::holds_alternative<GenStarted>(r.events.front()));
        CHECK(std::holds_alternative<GenFinished>(r.events.back()));
        CHECK(any_of<TokensCommitted>(r.events));
    }

    // One-sided context is allowed (prefix-only and suffix-only — suffix-only fills from pos 0).
    {
        FakeAdapter fake(16, -1, -1);
        GenerateConfig cfg{3, 6, 0, -1};
        GenerateResult pre = infill(fake, {5, 6}, {}, 3, cfg);
        GenerateResult suf = infill(fake, {}, {5, 6}, 3, cfg);
        CHECK(pre.new_tokens == 3 && suf.new_tokens == 3);
        for (int v : pre.generated) CHECK(v != MASK);
        for (int v : suf.generated) CHECK(v != MASK);
    }

    // Input validation.
    {
        FakeAdapter fake(16, -1, -1);
        GenerateConfig cfg{4, 2, 0, -1};
        bool threw = false;
        try { infill(fake, {1}, {2}, 0, cfg); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
        threw = false;
        try { infill(fake, {}, {}, 4, cfg); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    }

    std::printf("test_infill: all assertions passed\n");
    return 0;
}
