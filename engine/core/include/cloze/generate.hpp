// cloze/generate.hpp — the pass loop (DESIGN §5, build-order step 4), C++ port of
// lab/cloze_lab/generate.py::generate. Wires adapter + policy + stepper + blocks into the
// whole-sequence fixed(T) denoiser. Backend-free: it speaks only to the abstract
// ModelAdapter (DESIGN invariant 1), so it links into the scheduler/runtime libs with
// no ggml dependency and can be driven by any adapter (the ggml one, or a fake).
//
// Slice 1 scope mirrors the lab's step 4: full board, full recompute every pass (cache
// off), greedy commit, no revision, no KV reuse. Blocks/cache/adaptive-rails/revision are
// later slices that must keep this loop's golden picks intact.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "cloze/cache.hpp"
#include "cloze/events.hpp"
#include "cloze/model.hpp"
#include "cloze/probe.hpp"
#include "cloze/selector.hpp"
#include "cloze/stepper.hpp"

namespace cloze {

struct GenerateConfig {
    int max_new;        // masked slots to denoise after the prompt
    int steps;          // fixed(T): passes per block
    int block_len = 0;  // 0 = whole-sequence; > 0 = semi-AR blocks (slice 2+)
    int topk = -1;      // ConfidenceTopK k: < 0 = per-pass quota, >= 1 = fixed-k
};

// Revision (§5.2 remask_lowconf, "the model changes its mind"): each pass, re-mask any
// already-committed active-block token whose RECOMPUTED confidence fell below tau_revise, freeing
// it to be re-predicted next pass; capped at max_revisions per position (guarantees termination).
// Opt-in: when disabled (default), the commit path is byte-identical to before (goldens untouched)
// and the pluggable CommitSelector is used; when enabled, the loop uses the CPU sample+select path
// so it can also score the committed positions, and emits tokens_revised events.
struct ReviseConfig {
    bool enabled = false;
    double tau_revise = 0.5;
    int max_revisions = 1;
};

// Sampling controls (opt-in; defaults reproduce the greedy goldens exactly, bit-for-bit, and keep
// the pluggable CommitSelector seam). temperature 0 = greedy argmax; > 0 draws from
// softmax(logits / T) with a per-generation rng seeded by `seed` (deterministic within this runtime,
// not bit-matched to the lab's numpy draw). rep_penalty > 1 downweights tokens already on the board
// (CTRL/HF convention) to curb greedy repetition loops. When either is non-default the loop takes the
// CPU sample path (the kernel selector is greedy-only), same as the reviser.
struct SampleConfig {
    double temperature = 0.0;
    double rep_penalty = 1.0;
    uint64_t seed = 0;
};

struct GenerateResult {
    std::vector<int> board;      // full final board (prompt + generated slots)
    std::vector<int> generated;  // generated ids, truncated at EOS if present
    std::string text;            // decode(generated)
    std::string reason;          // "eos" | "steps_exhausted" | "length"
    int new_tokens = 0;          // == generated.size()
    int steps_total = 0;         // total passes run across all blocks
    std::vector<Event> events;   // the §5.1 event stream for this run (invariant 2; replayable)
};

// Denoise config.max_new masked slots after prompt_ids. Uses Stepper::fixed(config.steps)
// and confidence_topk(config.topk); the prompt must be non-empty (the Dream-family shift
// reads row p-1, so position 0 is never a generation target). Throws on invalid config.
//
// `cache` controls K/V reuse (DESIGN §5.5); default off = full recompute every pass (exact).
// In block mode (block_len > 0), CacheConfig{mode="delta", full_refresh_every=1} gives exact
// Tier A/B reuse under the one-way law — the frozen prefix is reused, the active block
// recomputed every pass. The picks are identical to cache off; only the work differs.
//
// `selector` is the fused sample+confidence+select step (§4.3); nullptr => the default
// backend-free CpuCommitSelector. Pass the CUDA KernelCommitSelector to run that fusion
// device-side — it produces identical picks, so the goldens are unaffected.
//
// `on_event` (optional) is streamed the §5.1 events live as they're emitted (for a live TUI /
// server); the same events are also collected on GenerateResult.events. Emission is pure
// observation — it never touches the board, so the goldens are unaffected (invariant 2).
GenerateResult generate(ModelAdapter& adapter,
                        const std::vector<int>& prompt_ids,
                        const GenerateConfig& config,
                        const CacheConfig& cache = CacheConfig{},
                        const CommitSelector* selector = nullptr,
                        const std::function<void(const Event&)>& on_event = {},
                        const ReviseConfig& revise = {},
                        const SampleConfig& sample = {},
                        const ConceptProbes* probes = nullptr);

// Fill `gap` masked slots BETWEEN a prefix and a suffix — native dLLM infilling — a capability autoregressive
// models structurally lack. The board is prefix + [MASK]*gap + suffix
// and the masked middle is denoised under FULL bidirectional attention, so every filled slot sees
// the fixed right-context (suffix) as well as the left. Whole-sequence + full recompute (exact);
// correctness, not cache reuse, is the point. Needs a non-empty prefix OR suffix; gap >= 1.
// config.steps is the step budget (config.max_new/block_len are ignored — the gap is the region).
// Returns the whole prefix+fill+suffix board; `text`/`generated` are just the filled gap.
GenerateResult infill(ModelAdapter& adapter,
                      const std::vector<int>& prefix_ids,
                      const std::vector<int>& suffix_ids,
                      int gap,
                      const GenerateConfig& config,
                      const CommitSelector* selector = nullptr,
                      const std::function<void(const Event&)>& on_event = {},
                      const ReviseConfig& revise = {},
                      const SampleConfig& sample = {},
                      const ConceptProbes* probes = nullptr);

// Denoise a board that ALREADY carries MASK tokens at arbitrary positions — the generalized
// cloze operation that powers "revise this selection". Full bidirectional attention over the
// whole board: every masked position is a fill target, every non-masked position is fixed context
// seen from both sides, and all holes are predicted in parallel each pass (so they see each other
// as they resolve). infill is the single-contiguous-gap special case of this; here the holes may
// be many and scattered (the user highlighted several spans). Whole-sequence, full recompute every
// pass (exact) — correctness, not cache reuse, is the point. config.steps is the budget;
// config.max_new / config.block_len are ignored (the holes already in `board_in` define the region).
// Needs at least one MASK in board_in. Returns the whole filled board on `board`/`generated`, and
// `text` = decode(board). The remask_lowconf `revise` knob composes here too (a filled hole whose
// confidence falls can be reconsidered, with both sides in view).
GenerateResult denoise(ModelAdapter& adapter,
                       const std::vector<int>& board_in,
                       const GenerateConfig& config,
                       const CommitSelector* selector = nullptr,
                       const std::function<void(const Event&)>& on_event = {},
                       const ReviseConfig& revise = {},
                       const SampleConfig& sample = {},
                       const ConceptProbes* probes = nullptr);

}  // namespace cloze
