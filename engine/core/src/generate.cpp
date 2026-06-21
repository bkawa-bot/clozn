#include "cloze/generate.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <utility>

#include "cloze/blocks.hpp"
#include "cloze/cache.hpp"
#include "cloze/policies.hpp"
#include "cloze/sample.hpp"
#include "cloze/selector.hpp"
#include "cloze/whitebox.hpp"  // features_from / lens_from (shared with the AR loop)

namespace cloze {

namespace {

// Generated ids up to (excluding) the first EOS; the whole span if none / eos disabled.
// Mirrors lab generate.py::truncate_at_eos.
std::vector<int> truncate_at_eos(const std::vector<int>& span, int eos) {
    if (eos < 0) return span;
    auto it = std::find(span.begin(), span.end(), eos);
    return std::vector<int>(span.begin(), it);
}

// One pass's verdict: what was inked (tokens_committed) and what was re-masked (tokens_revised).
struct StepSelection {
    std::vector<Candidate> committed;
    std::vector<ReviseItem> revised;  // empty unless revising
};

// Commit (and, when revising, re-mask) over the model output for one pass; mutates `board` in place.
//
// Disabled (default): exactly sel_impl.select over the masked positions — byte-identical to the
// pre-revision loop, so the golden picks are untouched and the pluggable CommitSelector (CPU or the
// CUDA kernel) is honored.
//
// Enabled (§5.2 remask_lowconf): `want` carries the masked positions AND the active block's
// already-committed positions, so the CPU sample path scores both. The masked subset is inked by
// confidence_topk; the committed subset is offered to remask_lowconf, and any it returns (recomputed
// confidence < tau_revise, still under the per-position cap) is re-masked — freeing it to be
// re-predicted next pass. The kernel selector is bypassed here (the reviser needs host logits for the
// committed rows), matching the generate.hpp contract.
StepSelection commit_and_revise(const ForwardResult& fwd, const std::vector<int>& want,
                                const std::vector<int>& masked, const StepContext& ctx, int topk,
                                const ReviseConfig& revise, const SampleConfig& sample,
                                const CommitSelector& sel_impl, std::vector<int>& board, int mask,
                                std::map<int, int>& revision_counts, std::mt19937_64& rng) {
    StepSelection out;
    const bool sampling = sample.temperature > 0.0 || sample.rep_penalty != 1.0;

    // Greedy, no penalty, no revision => the pluggable selector seam (CPU or the CUDA kernel) —
    // byte-identical to the goldens.
    if (!revise.enabled && !sampling) {
        Selection sel = sel_impl.select(fwd, want, ctx, topk);
        for (const Candidate& c : sel.commit) board[c.pos] = c.token_id;
        out.committed = std::move(sel.commit);
        return out;
    }

    // CPU sample path (revision and/or temperature/penalty active): score every requested row with
    // the sampling opts, then split masked (commit candidates) from already-committed (revision
    // candidates). The repetition penalty sees the whole current board.
    SampleOpts opts;
    opts.temperature = sample.temperature;
    opts.rep_penalty = sample.rep_penalty;
    opts.board = &board;
    opts.mask_token = mask;
    opts.rng = &rng;
    const std::vector<Candidate> cands = sample_candidates(fwd, want, opts);
    const std::set<int> masked_set(masked.begin(), masked.end());
    std::vector<Candidate> masked_cands, committed_cands;
    for (const Candidate& c : cands) {
        if (masked_set.count(c.pos)) masked_cands.push_back(c);
        else committed_cands.push_back(c);
    }

    Selection sel = confidence_topk(masked_cands, ctx, topk);
    for (const Candidate& c : sel.commit) board[c.pos] = c.token_id;
    out.committed = std::move(sel.commit);

    if (revise.enabled) {
        const std::vector<Candidate> to_revise =
            remask_lowconf(committed_cands, revise.tau_revise, revise.max_revisions, revision_counts);
        for (const Candidate& c : to_revise) {
            out.revised.push_back(ReviseItem{c.pos, board[c.pos], c.token_id, c.confidence});
            board[c.pos] = mask;  // re-masked: looks "changed" to the cache, so it recomputes next pass
            revision_counts[c.pos] += 1;
        }
    }
    return out;
}

// features_from / lens_from now live in cloze/whitebox.hpp (backend-free, shared with generate_ar).

}  // namespace

GenerateResult generate(ModelAdapter& adapter,
                        const std::vector<int>& prompt_ids,
                        const GenerateConfig& config,
                        const CacheConfig& cache,
                        const CommitSelector* selector,
                        const std::function<void(const Event&)>& on_event,
                        const ReviseConfig& revise,
                        const SampleConfig& sample,
                        const ConceptProbes* probes) {
    if (prompt_ids.empty()) throw std::invalid_argument("prompt_ids must be non-empty");
    if (config.max_new < 1) throw std::invalid_argument("max_new must be >= 1");
    if (config.steps < 1) throw std::invalid_argument("steps must be >= 1");
    if (config.block_len < 0) throw std::invalid_argument("block_len must be >= 0");

    const ModelConfig& mcfg = adapter.config();
    const int mask = mcfg.mask_token_id;
    const int eos = mcfg.eos_token_id;
    const Stepper stepper = Stepper::fixed(config.steps);
    const CpuCommitSelector default_selector;
    const CommitSelector& sel_impl = selector ? *selector : default_selector;
    CacheManager cache_mgr(cache);
    // Block-causal attention => the prefix is exactly frozen (Tier A/B); whole-sequence has
    // no exact frozen prefix (the prompt sees the changing active tokens bidirectionally).
    const bool frozen_prefix = config.block_len > 0;

    const int p = static_cast<int>(prompt_ids.size());
    const int n = p + config.max_new;

    std::vector<int> board = prompt_ids;
    board.resize(n, mask);  // append max_new mask slots

    BlockPlan plan{p, config.max_new, config.block_len};
    const std::vector<Block> block_list = plan.blocks();

    std::vector<Event> events;
    auto emit = [&](Event e) {
        if (on_event) on_event(e);
        events.push_back(std::move(e));
    };
    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point a) {
        return std::chrono::duration<double, std::milli>(clock::now() - a).count();
    };
    const clock::time_point t_start = clock::now();

    int t = 0;          // global pass counter, monotonic across blocks
    int last_t = 0;     // t of the last completed pass (for finalize events)
    int gen_end = p;    // exclusive end of the region actually generated
    std::map<int, int> revision_counts;  // per-position lifetime re-masks (reviser cap)
    std::mt19937_64 rng(sample.seed);     // sampling rng; unused on the greedy default path

    emit(GenStarted{0, p, config.block_len, config.max_new});

    auto count_masked = [&](int lo, int hi) {
        int c = 0;
        for (int i = lo; i < hi; ++i)
            if (board[i] == mask) ++c;
        return c;
    };
    auto has_eos = [&](int lo, int hi) {
        if (eos < 0) return false;
        for (int i = lo; i < hi; ++i)
            if (board[i] == eos) return true;
        return false;
    };

    for (const Block& block : block_list) {
        const int working_len = block.end;
        const Mask attn = attention_mask(working_len, p, config.block_len);

        emit(BlockStarted{t, block.index, {block.start, block.end}});

        const std::pair<int, int> active{block.start, block.end};
        int block_steps = 0;
        for (int step = 0; step < stepper.steps_cap(); ++step) {
            std::vector<int> masked;  // masked positions in this block, ascending
            for (int i = block.start; i < block.end; ++i)
                if (board[i] == mask) masked.push_back(i);
            if (masked.empty()) break;  // block is full — stop (revision only runs while masks remain)

            // When revising, also request logits for the active block's already-committed tokens so
            // the reviser can score them. Committed positions are always within the active block, so
            // frozen Tier A/B blocks are never offered for revision (DESIGN §6.1).
            std::vector<int> want = masked;
            if (revise.enabled) {
                for (int i = block.start; i < block.end; ++i)
                    if (board[i] != mask) want.push_back(i);
                std::sort(want.begin(), want.end());
            }

            const clock::time_point t0 = clock::now();
            const std::vector<int> view(board.begin(), board.begin() + working_len);
            // §5.5 cache: plan which positions to recompute (off => all, cold KV), forward,
            // then record the drawers + advance the frozen boundary.
            ForwardPlan plan = cache_mgr.plan(view, active, step, frozen_prefix);
            ForwardResult fwd = adapter.forward(view, attn, plan.kv, plan.recompute_kv, want);
            cache_mgr.observe(view, fwd.kv, plan.recompute_kv, active, frozen_prefix);

            // §4.3 fused sample+confidence+select behind the selector seam (CPU by default, the CUDA
            // confidence-select kernel when one is passed); when revising, also re-masks low-conf
            // committed tokens (§5.2), which the selector seam doesn't cover.
            const StepContext ctx = stepper.context(step);
            StepSelection sel = commit_and_revise(fwd, want, masked, ctx, config.topk, revise,
                                                  sample, sel_impl, board, mask, revision_counts, rng);

            block_steps = step + 1;
            const double ms = ms_since(t0);
            const int remaining = count_masked(block.start, block.end);

            if (!sel.revised.empty()) emit(TokensRevised{t, block.index, std::move(sel.revised)});
            std::vector<CommitItem> items;
            items.reserve(sel.committed.size());
            for (const Candidate& c : sel.committed) items.push_back({c.pos, c.token_id, c.confidence});
            emit(TokensCommitted{t, block.index, std::move(items)});
            emit(StepStats{t, block.index, step, static_cast<int>(sel.committed.size()), remaining,
                           ms, plan.cache_hit});
            if (auto sf = features_from(fwd, t, block.index, probes)) emit(*sf);
            if (!fwd.activations.empty())
                if (auto sl = lens_from(fwd, want, static_cast<int>(masked.size()), t, block.index, 5)) emit(*sl);
            last_t = t;
            ++t;
            if (!stepper.should_continue(StepOutcome{step,
                                                     static_cast<int>(sel.committed.size()),
                                                     remaining})) {
                break;
            }
        }

        gen_end = block.end;
        std::vector<int> block_span(board.begin() + block.start, board.begin() + block.end);
        const std::string block_text = adapter.decode(truncate_at_eos(block_span, eos));
        emit(BlockFinalized{last_t, block.index, block_text, block_steps});

        // EOS commit finishes the run; later blocks are intentionally left ungenerated.
        if (has_eos(p, gen_end) && &block != &block_list.back()) break;
    }

    std::vector<int> gen_span(board.begin() + p, board.begin() + gen_end);
    std::vector<int> kept = truncate_at_eos(gen_span, eos);

    GenerateResult result;
    result.board = board;
    result.new_tokens = static_cast<int>(kept.size());
    result.steps_total = t;
    if (has_eos(p, gen_end)) {
        result.reason = "eos";  // explicit stop wins over leftover holes
    } else if (count_masked(p, gen_end) > 0) {
        result.reason = "steps_exhausted";  // step budget left holes, no EOS
    } else {
        result.reason = "length";
    }
    result.text = adapter.decode(kept);
    result.generated = std::move(kept);

    const double wall_ms = ms_since(t_start);
    const double tok_per_s = wall_ms > 0 ? result.new_tokens * 1000.0 / wall_ms : 0.0;
    emit(GenFinished{last_t, result.reason, result.new_tokens, wall_ms, t, tok_per_s});
    result.events = std::move(events);
    return result;
}

GenerateResult infill(ModelAdapter& adapter,
                      const std::vector<int>& prefix_ids,
                      const std::vector<int>& suffix_ids,
                      int gap,
                      const GenerateConfig& config,
                      const CommitSelector* selector,
                      const std::function<void(const Event&)>& on_event,
                      const ReviseConfig& revise,
                      const SampleConfig& sample,
                      const ConceptProbes* probes) {
    if (gap < 1) throw std::invalid_argument("gap must be >= 1");
    if (prefix_ids.empty() && suffix_ids.empty())
        throw std::invalid_argument("infill needs a prefix or a suffix for context");
    if (config.steps < 1) throw std::invalid_argument("steps must be >= 1");

    const ModelConfig& mcfg = adapter.config();
    const int mask = mcfg.mask_token_id;
    const Stepper stepper = Stepper::fixed(config.steps);
    const CpuCommitSelector default_selector;
    const CommitSelector& sel_impl = selector ? *selector : default_selector;

    const int lo = static_cast<int>(prefix_ids.size());
    const int hi = lo + gap;
    std::vector<int> board = prefix_ids;
    board.insert(board.end(), gap, mask);
    board.insert(board.end(), suffix_ids.begin(), suffix_ids.end());
    const int n = static_cast<int>(board.size());

    // Full bidirectional over the whole board: the gap sees both fixed sides (block_len 0).
    const Mask attn = attention_mask(n, lo, 0);

    std::vector<Event> events;
    auto emit = [&](Event e) { if (on_event) on_event(e); events.push_back(std::move(e)); };
    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point a) {
        return std::chrono::duration<double, std::milli>(clock::now() - a).count();
    };
    const clock::time_point t_start = clock::now();

    int t = 0, last_t = 0, steps_used = 0;
    std::map<int, int> revision_counts;  // per-position lifetime re-masks (reviser cap)
    std::mt19937_64 rng(sample.seed);     // sampling rng; unused on the greedy default path
    emit(GenStarted{0, lo, 0, gap});
    emit(BlockStarted{0, 0, {lo, hi}});

    auto count_masked = [&]() {
        int c = 0;
        for (int i = lo; i < hi; ++i) if (board[i] == mask) ++c;
        return c;
    };

    for (int step = 0; step < stepper.steps_cap(); ++step) {
        std::vector<int> masked;
        for (int i = lo; i < hi; ++i) if (board[i] == mask) masked.push_back(i);
        if (masked.empty()) break;

        // When revising, also score the gap's already-committed slots so the model can reconsider
        // a low-confidence fill with both fixed sides in view.
        std::vector<int> want = masked;
        if (revise.enabled) {
            for (int i = lo; i < hi; ++i) if (board[i] != mask) want.push_back(i);
            std::sort(want.begin(), want.end());
        }

        const clock::time_point t0 = clock::now();
        // Cache off, full recompute every pass — exact (correctness over reuse for a one-shot fill).
        ForwardResult fwd = adapter.forward(board, attn, /*kv=*/nullptr, /*recompute_kv=*/std::nullopt, want);
        const StepContext ctx = stepper.context(step);
        StepSelection sel = commit_and_revise(fwd, want, masked, ctx, config.topk, revise,
                                              sample, sel_impl, board, mask, revision_counts, rng);

        steps_used = step + 1;
        const double ms = ms_since(t0);
        const int remaining = count_masked();
        if (!sel.revised.empty()) emit(TokensRevised{t, 0, std::move(sel.revised)});
        std::vector<CommitItem> items;
        items.reserve(sel.committed.size());
        for (const Candidate& c : sel.committed) items.push_back({c.pos, c.token_id, c.confidence});
        emit(TokensCommitted{t, 0, std::move(items)});
        emit(StepStats{t, 0, step, static_cast<int>(sel.committed.size()), remaining, ms, 0.0});
        if (auto sf = features_from(fwd, t, 0, probes)) emit(*sf);
        if (!fwd.activations.empty())
            if (auto sl = lens_from(fwd, want, static_cast<int>(masked.size()), t, 0, 5)) emit(*sl);
        last_t = t;
        ++t;
        if (!stepper.should_continue(StepOutcome{step, static_cast<int>(sel.committed.size()),
                                                 remaining}))
            break;
    }

    std::vector<int> fill(board.begin() + lo, board.begin() + hi);
    const std::string fill_text = adapter.decode(fill);
    emit(BlockFinalized{last_t, 0, fill_text, steps_used});

    const int remaining = count_masked();
    GenerateResult result;
    result.board = board;
    result.generated = std::move(fill);
    result.text = fill_text;
    result.new_tokens = gap - remaining;
    result.steps_total = t;
    result.reason = remaining > 0 ? "steps_exhausted" : "length";

    const double wall_ms = ms_since(t_start);
    const double tok_per_s = wall_ms > 0 ? result.new_tokens * 1000.0 / wall_ms : 0.0;
    emit(GenFinished{last_t, result.reason, result.new_tokens, wall_ms, t, tok_per_s});
    result.events = std::move(events);
    return result;
}

GenerateResult denoise(ModelAdapter& adapter,
                       const std::vector<int>& board_in,
                       const GenerateConfig& config,
                       const CommitSelector* selector,
                       const std::function<void(const Event&)>& on_event,
                       const ReviseConfig& revise,
                       const SampleConfig& sample,
                       const ConceptProbes* probes) {
    if (config.steps < 1) throw std::invalid_argument("steps must be >= 1");

    const ModelConfig& mcfg = adapter.config();
    const int mask = mcfg.mask_token_id;
    const Stepper stepper = Stepper::fixed(config.steps);
    const CpuCommitSelector default_selector;
    const CommitSelector& sel_impl = selector ? *selector : default_selector;

    std::vector<int> board = board_in;
    const int n = static_cast<int>(board.size());
    if (n < 1) throw std::invalid_argument("board must be non-empty");
    int n_holes = 0;
    for (int id : board)
        if (id == mask) ++n_holes;
    if (n_holes < 1) throw std::invalid_argument("board has no masked positions to fill");
    // The originally-masked positions (the holes to fill). Revision is confined to THESE — the fixed
    // context (everything the caller did NOT mask) is never offered for re-masking, so a selection
    // revise can't chew into the surrounding text.
    std::set<int> holes;
    for (int i = 0; i < n; ++i)
        if (board[i] == mask) holes.insert(i);

    // Full bidirectional over the whole board (active_start = 0 => the ggml adapter recomputes the
    // whole board exactly; no frozen prefix). Every hole sees all fixed context AND the other holes.
    const Mask attn = attention_mask(n, 0, 0);

    std::vector<Event> events;
    auto emit = [&](Event e) { if (on_event) on_event(e); events.push_back(std::move(e)); };
    using clock = std::chrono::steady_clock;
    auto ms_since = [](clock::time_point a) {
        return std::chrono::duration<double, std::milli>(clock::now() - a).count();
    };
    const clock::time_point t_start = clock::now();

    int t = 0, last_t = 0, steps_used = 0;
    std::map<int, int> revision_counts;  // per-position lifetime re-masks (reviser cap)
    std::mt19937_64 rng(sample.seed);     // sampling rng; unused on the greedy default path
    // The "region" is the whole board; prompt_tokens here = the fixed (non-masked) context count.
    emit(GenStarted{0, n - n_holes, 0, n_holes});
    emit(BlockStarted{0, 0, {0, n}});

    auto count_masked = [&]() {
        int c = 0;
        for (int i = 0; i < n; ++i)
            if (board[i] == mask) ++c;
        return c;
    };

    for (int step = 0; step < stepper.steps_cap(); ++step) {
        std::vector<int> masked;
        for (int i = 0; i < n; ++i)
            if (board[i] == mask) masked.push_back(i);
        if (masked.empty()) break;

        std::vector<int> want = masked;
        if (revise.enabled) {
            for (int i : holes)
                if (board[i] != mask) want.push_back(i);  // already-filled HOLES only (never fixed context)
            std::sort(want.begin(), want.end());
        }

        const clock::time_point t0 = clock::now();
        // Cache off, full recompute every pass — exact (correctness over reuse for a one-shot fill).
        ForwardResult fwd = adapter.forward(board, attn, /*kv=*/nullptr, /*recompute_kv=*/std::nullopt, want);
        const StepContext ctx = stepper.context(step);
        StepSelection sel = commit_and_revise(fwd, want, masked, ctx, config.topk, revise,
                                              sample, sel_impl, board, mask, revision_counts, rng);

        steps_used = step + 1;
        const double ms = ms_since(t0);
        const int remaining = count_masked();
        if (!sel.revised.empty()) emit(TokensRevised{t, 0, std::move(sel.revised)});
        std::vector<CommitItem> items;
        items.reserve(sel.committed.size());
        for (const Candidate& c : sel.committed) items.push_back({c.pos, c.token_id, c.confidence});
        emit(TokensCommitted{t, 0, std::move(items)});
        emit(StepStats{t, 0, step, static_cast<int>(sel.committed.size()), remaining, ms, 0.0});
        if (auto sf = features_from(fwd, t, 0, probes)) emit(*sf);
        if (!fwd.activations.empty())
            if (auto sl = lens_from(fwd, want, static_cast<int>(masked.size()), t, 0, 5)) emit(*sl);
        last_t = t;
        ++t;
        if (!stepper.should_continue(StepOutcome{step, static_cast<int>(sel.committed.size()),
                                                 remaining}))
            break;
    }

    const std::string text = adapter.decode(board);  // the whole filled board
    emit(BlockFinalized{last_t, 0, text, steps_used});

    const int remaining = count_masked();
    GenerateResult result;
    result.board = board;
    result.generated = board;  // the full board; the caller knows which positions were holes
    result.text = text;
    result.new_tokens = n_holes - remaining;
    result.steps_total = t;
    result.reason = remaining > 0 ? "steps_exhausted" : "length";

    const double wall_ms = ms_since(t_start);
    const double tok_per_s = wall_ms > 0 ? result.new_tokens * 1000.0 / wall_ms : 0.0;
    emit(GenFinished{last_t, result.reason, result.new_tokens, wall_ms, t, tok_per_s});
    result.events = std::move(events);
    return result;
}

}  // namespace cloze
