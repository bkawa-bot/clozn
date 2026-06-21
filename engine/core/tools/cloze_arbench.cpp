// cloze_arbench.cpp — the honest head-to-head: autoregressive vs diffusion DECODE THROUGHPUT on
// the SAME weights + GPU + llama.cpp backend, so only the decode paradigm differs.
//
//   AR mode:        causal_attn ON, greedy, KV cache, one token per llama_decode (standard LLM).
//   diffusion mode: causal_attn OFF, Cloze denoise (commit many tokens per forward pass).
//
// HONESTY (read before quoting any number): in AR mode this model produces GARBAGE — Dream is a
// masked-diffusion model, not next-token-trained — so AR-mode *output* is meaningless. But AR-mode
// *throughput* (tok/s) is representative: a 7B-Q8_0 AR decode costs the same matmuls regardless of
// what the weights were trained for, so this is a fair speed proxy for "any 7B-Q8_0 AR model on
// this GPU." Diffusion-mode output is the real Cloze path. We measure SPEED here, not quality; the
// quality column is the prompts' actual text (printed) + the separate cloze-bench divergence sweep.
//
//   cloze-arbench <model.gguf> [--gpu-layers N] [--new N] [--mask-token ID] [--ctx N] [--prompt "..."]
#include "cloze/generate.hpp"
#include "cloze/model_ggml.hpp"
#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace cloze;
using clk = std::chrono::steady_clock;

namespace {
void quiet_log(ggml_log_level lvl, const char* t, void*) {
    if (lvl == GGML_LOG_LEVEL_ERROR) std::fputs(t, stderr);
}
double ms_since(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Standard autoregressive greedy decode of `n_new` tokens over a CAUSAL context (one token per
// llama_decode, KV cache). Returns the MEDIAN generation tok/s over `reps` runs after a discarded
// warmup (prefill excluded — pure decode rate). KV is cleared + re-prefilled per run.
double ar_tok_per_s(llama_model* model, const std::vector<int>& prompt, int n_new, int n_ctx, int reps) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx; cp.n_batch = n_ctx; cp.n_ubatch = n_ctx;  // causal_attn defaults ON
    llama_context* ctx = llama_init_from_model(model, cp);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_batch b = llama_batch_init(n_ctx, 0, 1);
    auto add = [&](int tok, int pos, bool want_logits) {
        b.token[b.n_tokens] = tok; b.pos[b.n_tokens] = pos;
        b.n_seq_id[b.n_tokens] = 1; b.seq_id[b.n_tokens][0] = 0;
        b.logits[b.n_tokens] = want_logits; ++b.n_tokens;
    };
    auto one = [&]() -> double {
        llama_memory_clear(llama_get_memory(ctx), true);
        b.n_tokens = 0;
        for (int i = 0; i < (int)prompt.size(); ++i) add(prompt[i], i, i == (int)prompt.size() - 1);
        llama_decode(ctx, b);  // prefill (not timed)
        int pos = (int)prompt.size();
        const auto t0 = clk::now();
        for (int k = 0; k < n_new; ++k) {
            const float* logits = llama_get_logits_ith(ctx, -1);
            int argmax = 0; float best = logits[0];
            for (int v = 1; v < n_vocab; ++v) if (logits[v] > best) { best = logits[v]; argmax = v; }
            b.n_tokens = 0; add(argmax, pos++, true);
            if (llama_decode(ctx, b) != 0) break;
        }
        const double ms = ms_since(t0);
        return ms > 0 ? n_new * 1000.0 / ms : 0.0;
    };
    one();  // warmup (discarded) — absorb CUDA init / graph reserve
    std::vector<double> runs;
    for (int r = 0; r < reps; ++r) runs.push_back(one());
    llama_batch_free(b);
    llama_free(ctx);
    return median(runs);
}

// Fraction of reference positions whose token matches (intra-diffusion quality vs the clean run).
double token_match(const std::vector<int>& ref, const std::vector<int>& q) {
    if (ref.empty()) return 0.0;
    int m = 0; const size_t n = std::min(ref.size(), q.size());
    for (size_t i = 0; i < n; ++i) if (ref[i] == q[i]) ++m;
    return (double)m / ref.size();
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> [--gpu-layers N] [--new N] "
                                          "[--mask-token ID] [--ctx N] [--prompt \"...\"]\n", argv[0]); return 1; }
    const std::string path = argv[1];
    // EOS disabled (unreachable id) on purpose: this is a THROUGHPUT bench — every config must
    // decode the full n_new tokens, not stop early (esp. instruct models that emit EOS on a raw
    // prompt). Quality is measured separately (cloze-bench / the CLI), never here.
    int gpu_layers = 0, n_new = 64, mask = 151665, eos = 2147483647, n_ctx = 512;
    std::string prompt = "def quicksort(arr):";
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i]; auto nx = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--gpu-layers") gpu_layers = std::atoi(nx());
        else if (a == "--new") n_new = std::atoi(nx());
        else if (a == "--mask-token") mask = std::atoi(nx());
        else if (a == "--ctx") n_ctx = std::atoi(nx());
        else if (a == "--prompt") prompt = nx();
    }

    const int reps = 5;  // median of 5 timed runs (after a discarded warmup) per config
    llama_log_set(quiet_log, nullptr);
    auto model = std::make_shared<GgmlModel>(path, mask, eos, gpu_layers);
    const std::vector<int> prompt_ids = model->encode(prompt);

    std::fprintf(stderr, "[arbench] %s, n_new=%d, prompt=%d tok, median of %d (warmup discarded)\n",
                 gpu_layers > 0 ? "GPU" : "CPU", n_new, (int)prompt_ids.size(), reps);

    // AR baseline (strong: causal + KV cache; throughput is a fair proxy for any same-arch AR
    // model — but AR-mode TEXT is garbage for a diffusion model, so no quality is claimed from it).
    const double ar = ar_tok_per_s(model->handle(), prompt_ids, n_new, n_ctx, reps);

    // Diffusion (real Cloze path), warmup + median, capturing generated tokens for the quality
    // column. The QUALITY axis is steps/token: fewer steps = more tokens/pass = faster, lower
    // quality. We anchor quality intra-diffusion: token-match vs the cleanest run (whole-seq, the
    // most passes) — there is NO AR quality anchor here (AR-mode text is garbage).
    struct Row { std::string tag; bool block; double tps, spt; std::vector<int> toks; };
    auto run_diff = [&](int block_len, const char* cache_mode, int steps, const char* tag) -> Row {
        GgmlAdapter adapter(model, n_ctx);
        GenerateConfig cfg; cfg.max_new = n_new; cfg.steps = steps; cfg.block_len = block_len; cfg.topk = -1;
        CacheConfig cache; cache.mode = cache_mode; if (std::string(cache_mode) == "delta") cache.full_refresh_every = 1;
        auto one = [&]() { const auto t0 = clk::now(); GenerateResult r = generate(adapter, prompt_ids, cfg, cache);
                           const double ms = ms_since(t0); return std::make_pair(ms > 0 ? r.new_tokens * 1000.0 / ms : 0.0, r); };
        one();  // warmup (discarded)
        std::vector<double> tps; GenerateResult last;
        for (int r = 0; r < reps; ++r) { auto pr = one(); tps.push_back(pr.first); last = pr.second; }
        const double spt = last.new_tokens > 0 ? (double)last.steps_total / last.new_tokens : 0.0;
        return Row{tag, block_len > 0, median(tps), spt, last.generated};
    };

    std::vector<Row> rows;
    for (int steps : {n_new, n_new / 2, n_new / 4}) {
        if (steps < 1) continue;
        char tag[40]; std::snprintf(tag, sizeof(tag), "diffusion whole (s<=%d)", steps);
        rows.push_back(run_diff(0, "off", steps, tag));
    }
    for (int steps : {16, 8, 4}) {  // semi-AR blocks of 16 + exact Tier A/B reuse — the main efficiency lever
        char tag[44]; std::snprintf(tag, sizeof(tag), "diffusion blk16+reuse (s<=%d)", steps);
        rows.push_back(run_diff(16, "delta", steps, tag));
    }
    // Per-mode quality anchor: each row's cleanest (1.0 steps/tok) run of the SAME mode (comparing a
    // block run to the whole-seq anchor would just show mode difference, not quality loss).
    auto anchor = [&](bool block) -> const std::vector<int>& {
        for (const Row& r : rows) if (r.block == block) return r.toks;  // first = most passes = cleanest
        return rows.front().toks;
    };

    std::printf("\n  %-26s %8s %9s %16s\n", "mode", "tok/s", "steps/tok", "match-vs-clean*");
    std::printf("  %-26s %8.1f %9.2f %16s\n", "autoregressive (causal)", ar, 1.00, "n/a (garbage)");
    for (const Row& r : rows)
        std::printf("  %-26s %8.1f %9.2f %15.0f%%\n", r.tag.c_str(), r.tps, r.spt,
                    token_match(anchor(r.block), r.toks) * 100.0);

    // Quality-matched: AR vs the fastest clean (1.0 steps/tok) diffusion config.
    double best_clean = 0.0;
    for (const Row& r : rows) if (r.spt >= 0.99 && r.tps > best_clean) best_clean = r.tps;
    std::printf("\n  * match-vs-clean = token agreement with the SAME mode's 1.0-steps/tok run.\n");
    std::printf("  MATCHED-QUALITY (clean, 1.0 steps/tok): AR %.1f vs best clean diffusion %.1f tok/s"
                " -> AR %.2fx faster\n", ar, best_clean, best_clean > 0 ? ar / best_clean : 0.0);
    std::printf("  Diffusion only exceeds AR below 1.0 steps/tok, trading quality (see column).\n"
                "  Scope: batch-1, greedy, no speculative decoding; AR-mode text is a speed proxy.\n"
                "  The favorable direction grows with model size/bandwidth (7B closer than 0.5B).\n");
    return 0;
}
