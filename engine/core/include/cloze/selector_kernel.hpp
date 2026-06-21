// cloze/selector_kernel.hpp — a CommitSelector backed by the confidence-select CUDA kernel
// (DESIGN §4.3). It runs sample + confidence + top-k device-side via cloze::confidence_select
// (the greedy / MaxProb / TopK path validated against reference.py on sm_120), so the §4.3
// fusion happens on the GPU instead of host-side sample_candidates + confidence_topk.
//
// The interface carries NO CUDA types — only the .cu implementation includes cuda_runtime and
// the kernel header — so this stays includable from plain C++ TUs. Picks MUST match
// CpuCommitSelector (parity is the integration gate, DESIGN invariant 3); confidences are
// float32 here vs float64 on the CPU path, so they agree only within epsilon.
//
// Note (honesty): this slice hands the kernel HOST logits and uploads them, so it does not yet
// capture the §4.3 transfer win (that needs llama's device-resident logits — a follow-up). It
// establishes the seam + parity; the kernel's transfer saving is measured standalone by cs_bench.
#pragma once

#include <vector>

#include "cloze/selector.hpp"

namespace cloze {

class KernelCommitSelector : public CommitSelector {
public:
    Selection select(const ForwardResult& fwd, const std::vector<int>& want,
                     const StepContext& ctx, int topk) const override;
};

}  // namespace cloze
