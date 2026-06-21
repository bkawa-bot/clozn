// test_policies.cpp — framework-free checks for the C++ policy port, mirroring the
// assertions in lab/tests/test_policies.py. Exits non-zero on any failure.
#include "cloze/policies.hpp"

#include <cassert>
#include <cstdio>
#include <vector>

using namespace cloze;

static std::vector<int> positions(const Selection& s) {
    std::vector<int> p;
    for (const auto& c : s.commit) p.push_back(c.pos);
    return p;
}

int main() {
    // quota ramp with UNEVEN division: 7 masks / 3 steps => ceil(7/3) = 3 at step 0
    // (the ceil->floor bug the lab's fake_quota_uneven golden guards against).
    {
        std::vector<Candidate> c;
        for (int i = 0; i < 7; ++i) c.push_back({i, i, 0.50 + 0.01 * i});
        auto sel = confidence_topk(c, StepContext{0, 3}, /*k=*/-1);
        assert(sel.commit.size() == 3);
    }

    // fixed k=2: ink the two most confident, returned pos-ascending.
    {
        std::vector<Candidate> c = {{0, 0, 0.1}, {1, 1, 0.9}, {2, 2, 0.5}};
        auto sel = confidence_topk(c, StepContext{0, -1}, /*k=*/2);
        assert((positions(sel) == std::vector<int>{1, 2}));
    }

    // confidence ties break toward the lower position.
    {
        std::vector<Candidate> c = {{0, 0, 0.5}, {1, 1, 0.5}, {2, 2, 0.5}};
        auto sel = confidence_topk(c, StepContext{0, -1}, /*k=*/1);
        assert((positions(sel) == std::vector<int>{0}));
    }

    // threshold: commit those clearing tau...
    {
        std::vector<Candidate> c = {{0, 0, 0.2}, {1, 1, 0.8}, {2, 2, 0.1}};
        auto sel = threshold(c, /*tau=*/0.5, /*min_commit=*/1);
        assert((positions(sel) == std::vector<int>{1}));
    }
    // ...and the min-one-commit rail when none clear tau (forces the top-1).
    {
        std::vector<Candidate> c = {{0, 0, 0.2}, {1, 1, 0.8}, {2, 2, 0.1}};
        auto sel = threshold(c, /*tau=*/0.99, /*min_commit=*/1);
        assert((positions(sel) == std::vector<int>{1}));
    }

    std::printf("test_policies: all assertions passed\n");
    return 0;
}
