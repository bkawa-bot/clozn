// cloze/policies.hpp — unmask/commit policies (DESIGN §5.2), C++ port of
// lab/cloze_lab/scheduler/policies.py. Pure logic: no ggml, no model dependency —
// the same framework-free property the Python scheduler has (DESIGN invariant 1),
// which is what makes this port a translation rather than a rewrite.
//
// Confidences are double (Python float == C double; the golden fixtures store float64).
#pragma once

#include <map>
#include <vector>

namespace cloze {

// One masked position's sampled proposal: token_id at pos, with confidence.
struct Candidate {
    int pos;
    int token_id;
    double confidence;
};

// What a policy may consult — DESIGN §5.2's (step, block_state). steps_total < 0
// means "no fixed budget" (the adaptive stepper supplies none).
struct StepContext {
    int step;
    int steps_total = -1;

    // Steps left including this one; -1 when no fixed budget is set.
    int steps_remaining() const { return steps_total < 0 ? -1 : steps_total - step; }
};

// A policy's verdict for one pass, both vectors pos-ascending. `revise` is empty
// until the remask_lowconf revision policy is ported (§5.2).
struct Selection {
    std::vector<Candidate> commit;
    std::vector<Candidate> revise;
};

// confidence_topk (§5.2 default): ink the k most confident candidates.
//   k < 0  => quota mode: ceil(n_masked / steps_remaining) (requires ctx.steps_total >= 0).
//   k >= 1 => fixed k: ink min(k, n_masked) per pass.
// Confidence ties break toward the LOWER position so picks are exact on every platform.
Selection confidence_topk(const std::vector<Candidate>& candidates, const StepContext& ctx, int k);

// threshold(tau) (§5.2): ink every candidate with confidence >= tau. If fewer than
// min_commit clear tau, ink the min_commit most confident anyway (the min-one-commit
// progress rail, so a pass never stalls).
Selection threshold(const std::vector<Candidate>& candidates, double tau, int min_commit);

// remask_lowconf (§5.2) — "the model changes its mind". Given the recomputed candidates for
// already-COMMITTED active-block positions, return the ones to RE-MASK: confidence < tau_revise
// AND under the per-position max_revisions cap (which guarantees termination). Returned
// pos-ascending; the candidates carry the model's current (low) pick for the tokens_revised event.
std::vector<Candidate> remask_lowconf(const std::vector<Candidate>& committed, double tau_revise,
                                      int max_revisions, const std::map<int, int>& revision_counts);

}  // namespace cloze
