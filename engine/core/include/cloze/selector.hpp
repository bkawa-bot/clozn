// cloze/selector.hpp — the fused sample+confidence+select seam (DESIGN §4.3). Each pass the
// generate loop hands the masked-position logits to a CommitSelector, which returns the set
// of (pos, token, confidence) to ink this step. This is the exact seam the confidence-select
// kernel plugs into: sample + confidence + selection are one fused step, so the GPU version
// can do them device-side and ship only ~2*n_masked back instead of the full logits.
//
// Two implementations share this interface:
//   - CpuCommitSelector (here): sample_candidates + confidence_topk — backend-free, the exact
//     path the goldens pin. The default.
//   - KernelCommitSelector (core/, CLOZE_BUILD_CUDA): wraps cloze::confidence_select so the
//     fusion runs on-device. It MUST produce identical picks to the CPU selector — parity is
//     the integration's correctness gate (DESIGN invariant 3).
#pragma once

#include <vector>

#include "cloze/model.hpp"
#include "cloze/policies.hpp"

namespace cloze {

class CommitSelector {
public:
    virtual ~CommitSelector() = default;

    // fwd.logits rows align 1:1 with `want` (the masked board positions, ascending). ctx +
    // topk resolve the quota/fixed k exactly as confidence_topk does (topk < 0 => quota
    // ceil(n/steps_remaining); topk >= 1 => fixed). Returns the commit set, pos-ascending.
    virtual Selection select(const ForwardResult& fwd, const std::vector<int>& want,
                             const StepContext& ctx, int topk) const = 0;
};

// The default, backend-free selector: greedy argmax + float64 softmax confidence, then top-k.
class CpuCommitSelector : public CommitSelector {
public:
    Selection select(const ForwardResult& fwd, const std::vector<int>& want,
                     const StepContext& ctx, int topk) const override;
};

}  // namespace cloze
