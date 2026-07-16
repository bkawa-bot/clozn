// cloze/sample.hpp — the §4.3 confidence-select CPU reference, C++ side. Mirrors
// lab/cloze_lab/generate.py::sample_candidates: per requested position, a token and its
// float64 softmax probability (the commit confidence).
//
// Default = greedy (temperature 0, no penalty): argmax token + its raw softmax prob, bit-for-bit
// the path the golden fixtures pin. Optional, opt-in knobs (SampleOpts) add a repetition penalty
// and temperature sampling. When temperature > 0 the draw uses a C++ rng and is therefore NOT
// bit-reproducible against the lab's numpy Generator — that's fine: only the greedy path is the
// cross-runtime oracle (DESIGN invariant 3); stochastic output is stochastic by definition.
// Confidence is computed in double (float64) so the greedy picks/confidences line up with the
// Python oracle within epsilon.
#pragma once

#include <random>
#include <vector>

#include "cloze/model.hpp"     // ForwardResult
#include "cloze/policies.hpp"  // Candidate

namespace cloze {

// Optional sampling controls. A default-constructed SampleOpts reproduces the greedy,
// penalty-free path exactly, so passing none changes nothing.
struct SampleOpts {
    double temperature = 0.0;       // 0 = greedy argmax; > 0 = draw from softmax(logits / T)
    double rep_penalty = 1.0;       // 1.0 = off; > 1 downweights tokens already on `board`
    int    top_k = 0;               // 0 (or >= vocab) = off; > 0 keeps only the k highest-prob tokens
    double top_p = 1.0;             // 1.0 = off; (0,1) keeps the smallest nucleus with cumulative prob >= top_p
    const std::vector<int>* board = nullptr;  // tokens already on the board (for rep_penalty); null = none
    int mask_token = -1;            // excluded from the penalty set
    std::mt19937_64* rng = nullptr; // required when temperature > 0 (deterministic given its seed)
};

// fwd.logits is row-major [n_requested, vocab]; positions[r] is the board position of row r (so the
// returned Candidate.pos is a board index, not a row index). Throws if the position count disagrees
// with the logits row count, or if temperature > 0 without an rng.
std::vector<Candidate> sample_candidates(const ForwardResult& fwd,
                                         const std::vector<int>& positions,
                                         const SampleOpts& opts = {});

}  // namespace cloze
