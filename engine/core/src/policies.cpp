// policies.cpp — implementation of cloze/policies.hpp. Mirrors
// lab/cloze_lab/scheduler/policies.py statement-for-statement.
#include "cloze/policies.hpp"

#include <algorithm>
#include <stdexcept>

namespace cloze {

namespace {

// Most confident first; confidence ties break toward the lower position (the
// Python `sorted(key=(-conf, pos))`), so picks are exact on every platform.
std::vector<Candidate> by_confidence(std::vector<Candidate> c) {
    std::sort(c.begin(), c.end(), [](const Candidate& a, const Candidate& b) {
        if (a.confidence != b.confidence) return a.confidence > b.confidence;
        return a.pos < b.pos;
    });
    return c;
}

// The committed selection, pos-ascending (Python `_commit`).
Selection commit(std::vector<Candidate> selected) {
    std::sort(selected.begin(), selected.end(),
              [](const Candidate& a, const Candidate& b) { return a.pos < b.pos; });
    return Selection{std::move(selected), {}};
}

}  // namespace

Selection confidence_topk(const std::vector<Candidate>& candidates, const StepContext& ctx, int k) {
    if (candidates.empty()) return Selection{};
    const int n = static_cast<int>(candidates.size());

    int take;
    if (k < 0) {  // quota mode
        const int rem = ctx.steps_remaining();
        if (rem < 0) throw std::invalid_argument("quota mode (k<0) requires ctx.steps_total");
        take = (n + rem - 1) / rem;  // ceil(n / steps_remaining)
    } else {
        take = k;
    }
    if (take > n) take = n;

    auto ranked = by_confidence(candidates);
    ranked.resize(static_cast<size_t>(take));
    return commit(std::move(ranked));
}

std::vector<Candidate> remask_lowconf(const std::vector<Candidate>& committed, double tau_revise,
                                      int max_revisions, const std::map<int, int>& revision_counts) {
    std::vector<Candidate> picked;
    for (const Candidate& c : committed) {
        auto it = revision_counts.find(c.pos);
        const int count = (it == revision_counts.end()) ? 0 : it->second;
        if (c.confidence < tau_revise && count < max_revisions) picked.push_back(c);
    }
    std::sort(picked.begin(), picked.end(),
              [](const Candidate& a, const Candidate& b) { return a.pos < b.pos; });
    return picked;
}

Selection threshold(const std::vector<Candidate>& candidates, double tau, int min_commit) {
    if (candidates.empty()) return Selection{};

    std::vector<Candidate> above;
    for (const auto& c : candidates)
        if (c.confidence >= tau) above.push_back(c);

    if (static_cast<int>(above.size()) >= min_commit) return commit(std::move(above));

    // progress rail: nothing (or too little) cleared tau — force the top few.
    auto ranked = by_confidence(candidates);
    if (static_cast<int>(ranked.size()) > min_commit) ranked.resize(static_cast<size_t>(min_commit));
    return commit(std::move(ranked));
}

}  // namespace cloze
