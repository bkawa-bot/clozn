#include "cloze/generate_ar.hpp"

#include <chrono>
#include <random>
#include <stdexcept>
#include <utility>

#include "cloze/sample.hpp"
#include "cloze/whitebox.hpp"  // features_from / lens_from (shared with the diffusion loop)

namespace cloze {

GenerateResult generate_ar(GgmlAdapter& adapter,
                           const std::vector<int>& prompt_ids,
                           const GenerateConfig& config,
                           const std::function<void(const Event&)>& on_event,
                           const SampleConfig& sample,
                           const ConceptProbes* read_probes,
                           const std::vector<float>* prefix_embd,
                           int prefix_rows,
                           const std::vector<int>* reference) {
    if (prompt_ids.empty()) throw std::invalid_argument("prompt_ids must be non-empty");
    if (config.max_new < 1) throw std::invalid_argument("max_new must be >= 1");

    // Early-stop-on-divergence is armed only with a non-empty reference. This is a pure termination
    // condition -- nothing about sampling, batching, or the KV path changes -- so the generated prefix
    // is bit-identical to what a full (reference-less) generation would produce up to the stop point.
    const bool ref_active = (reference != nullptr && !reference->empty());
    const int ref_len = ref_active ? static_cast<int>(reference->size()) : 0;
    bool diverged = false;
    int diverged_at = -1;

    const ModelConfig& mcfg = adapter.config();
    const int eos = mcfg.eos_token_id;

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

    const int p = static_cast<int>(prompt_ids.size());
    emit(GenStarted{0, p, 0, config.max_new});  // block_len 0: AR has no blocks

    // Causal attention + a fresh KV cache. A steering control vector set by the caller persists
    // across decodes (it's on the context, independent of the attention mode).
    adapter.set_causal(true);

    // Optional: splice a PyTorch-trained soft prefix in ahead of the prompt (fills KV [0, prefix_rows))
    // so a memory learned on the HF model shapes this ggml generation. The prompt then decodes at n_past=base.
    int base = 0;
    if (prefix_embd != nullptr && prefix_rows > 0) {
        adapter.ar_forward_embd(*prefix_embd, prefix_rows, 0);
        base = prefix_rows;
    }

    std::mt19937_64 rng(sample.seed);
    std::vector<int> seq = prompt_ids;  // full running sequence (for the repetition penalty)
    SampleOpts sopts;
    sopts.temperature = sample.temperature;
    sopts.rep_penalty = sample.rep_penalty;
    sopts.top_k = sample.top_k;
    sopts.top_p = sample.top_p;
    sopts.mask_token = mcfg.mask_token_id;  // -1 on an AR model; harmless
    if (sample.rep_penalty != 1.0) sopts.board = &seq;
    if (sample.temperature > 0.0) sopts.rng = &rng;

    std::vector<int> generated;
    generated.reserve(config.max_new);
    std::string reason = "length";

    // Hard context window: absolute positions [0, n_ctx) fit the KV cache. The prompt (+ any soft prefix)
    // must fit it; a prompt that exceeds it is a client error surfaced cleanly (the handler turns it into a
    // 400), never a silent truncation and never an uncaught 500.
    const int n_ctx = adapter.n_ctx();
    if (base + p > n_ctx)
        throw std::invalid_argument("prompt exceeds context window (n_ctx): reduce the prompt or raise --ctx");

    // Prefill: the last prompt row's logits are the distribution for the first generated token.
    ForwardResult fwd = adapter.ar_forward(prompt_ids, base);
    int n_past = base + p;
    int t = 0;

    for (int k = 0; k < config.max_new; ++k) {
        // Stop GRACEFULLY at the context limit: a token committed at position n_past must be decoded back
        // into the KV (ar_forward at n_past), which needs n_past < n_ctx. When there's no room left, this is
        // a "length" stop (finish_reason "length", a clean 200) -- NOT an ar_forward "exceeds n_ctx" throw,
        // which the non-streaming handler would surface as an empty 500. THE root fix for the long-generation
        // 500: a generation that reaches the context window stops, it never overflows.
        if (n_past >= n_ctx) { reason = "length"; break; }
        const std::vector<int> want = {n_past};  // the position about to be generated
        const std::vector<Candidate> cand = sample_candidates(fwd, want, sopts);
        if (cand.empty()) break;
        const int tok = cand[0].token_id;
        const double conf = cand[0].confidence;

        // The commit + the logit-lens (what this slot is considering = the distribution we sampled).
        emit(TokensCommitted{t, 0, {CommitItem{n_past, tok, conf}}});
        if (auto sl = lens_from(fwd, want, 1, t, 0, 5)) emit(*sl);

        generated.push_back(tok);
        seq.push_back(tok);
        const bool is_eos = (eos >= 0 && tok == eos);

        // Early-stop on divergence from the reference (prove-all ablated arms): the just-committed token
        // is already in `generated`, so the partial reply is a bit-exact PREFIX of the full reply. We stop
        // BEFORE feeding this token back in -- the remaining ~max_new-k decodes are exactly the work we're
        // here to save. `tok != reference[k]` (or running past the reference's length) is the divergence.
        // `is_eos` is folded in: an early EOS that the reference didn't have is itself a divergence at k.
        if (ref_active && (k >= ref_len || tok != (*reference)[k])) {
            diverged = true;
            diverged_at = k;
            ++t;  // count this committed step (matches the non-diverged ++t below)
            break;
        }

        // Feed the token back in: this decode (a) advances the KV and (b) yields the hidden state at
        // the token we just generated — its concept read. So StepFeatures is labeled with the
        // GENERATED token's own position (act_rows = {n_past}), the intuitive "this token is X".
        fwd = adapter.ar_forward({tok}, n_past);
        if (auto sf = features_from(fwd, t, 0, read_probes)) emit(*sf);
        if (auto sa = activations_from(fwd, t, 0)) emit(*sa);  // raw state (heavy; on-demand)

        ++n_past;
        ++t;
        if (is_eos) { reason = "eos"; break; }
    }

    GenerateResult result;
    std::vector<int> kept = generated;
    if (reason == "eos" && !kept.empty()) kept.pop_back();  // drop the trailing EOS for the text
    result.new_tokens = static_cast<int>(kept.size());
    result.generated = kept;
    result.text = adapter.decode(kept);
    result.steps_total = t;
    result.reason = reason;
    result.ref_active = ref_active;    // divergence checking was armed (a reference was supplied)
    result.diverged = diverged;        // true => stopped early at diverged_at; false + ref_active => matched fully
    result.diverged_at = diverged_at;  // generation index of the first divergent token (-1 if none)
    result.board = prompt_ids;
    result.board.insert(result.board.end(), generated.begin(), generated.end());

    const double wall_ms = ms_since(t_start);
    const double tok_per_s = wall_ms > 0 ? result.new_tokens * 1000.0 / wall_ms : 0.0;
    emit(GenFinished{t > 0 ? t - 1 : 0, result.reason, result.new_tokens, wall_ms, t, tok_per_s});
    result.events = std::move(events);
    return result;
}

}  // namespace cloze
