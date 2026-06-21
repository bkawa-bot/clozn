#include "cloze/selector.hpp"

#include "cloze/sample.hpp"

namespace cloze {

Selection CpuCommitSelector::select(const ForwardResult& fwd, const std::vector<int>& want,
                                    const StepContext& ctx, int topk) const {
    // The two lab steps, fused behind this seam: per-position (token, confidence), then the
    // top-k selection. The kernel selector replaces exactly this with one device-side call.
    const std::vector<Candidate> cands = sample_candidates(fwd, want);
    return confidence_topk(cands, ctx, topk);
}

}  // namespace cloze
