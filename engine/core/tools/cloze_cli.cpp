// cloze_cli.cpp — the native runtime's command-line entry point. Loads a GGUF diffusion LM and
// denoises a prompt through the C++ scheduler (the same GgmlAdapter + generate loop the goldens
// pin), printing the completion plus an honest stats line. This is the "it actually runs" artifact
// — the daily-driver shape behind the tests.
//
//   cloze <model.gguf> "<prompt>" [options]
// options:
//   --max-new N      masked slots to denoise after the prompt   (default 32)
//   --steps T        fixed(T) passes per block                  (default 8)
//   --block-len L    0 = whole-sequence; L>0 = semi-AR blocks    (default 0)
//   --topk K         commit width: <0 quota, >=1 fixed-k         (default -1 = quota)
//   --mask-token ID  the checkpoint's mask token id             (default 151665, open-dCoder <M>)
//   --eos ID         eos token id; <0 = take the model's own    (default -1)
//   --gpu-layers N   layers to offload to the GPU               (default 0 = CPU)
//   --cache MODE     off | delta (delta = exact Tier A/B reuse in block mode)  (default off)
//   --ctx N          context size                                (default 4096)
//   --stream         show a live per-step denoise progress line (consumes the §5.1 events)
//   --log FILE       write the §5.1 event stream as JSONL (the replayable flight recorder)
//   --suffix "<txt>" infill mode: fill the gap BETWEEN the prompt (prefix) and this suffix
//   --gap N          infill gap size (masked slots between prefix and suffix)  (default 8)
#include "cloze/events.hpp"
#include "cloze/generate.hpp"
#include "cloze/model_ggml.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <variant>
#include <vector>

using namespace cloze;

namespace {

// Quiet by default: drop llama/ggml INFO+DEBUG chatter so the CLI prints just the completion +
// stats. WARN/ERROR still surface. --verbose restores the full log.
void quiet_log(ggml_log_level level, const char* text, void* /*user_data*/) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN)
        std::fputs(text, stderr);
}

[[noreturn]] void usage(const char* argv0) {
    std::fprintf(stderr,
        "usage: %s <model.gguf> \"<prompt>\" [--max-new N] [--steps T] [--block-len L]\n"
        "          [--topk K] [--mask-token ID] [--eos ID] [--gpu-layers N] [--cache off|delta]\n"
        "          [--ctx N]\n", argv0);
    std::exit(1);
}

int int_arg(const char* v, const char* argv0) {
    char* end = nullptr;
    const long x = std::strtol(v, &end, 10);
    if (end == v || *end) usage(argv0);
    return static_cast<int>(x);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) usage(argv[0]);
    const std::string model_path = argv[1];
    const std::string prompt = argv[2];

    GenerateConfig cfg;
    cfg.max_new = 32;
    cfg.steps = 8;
    cfg.block_len = 0;
    cfg.topk = -1;
    int mask_token = 151665, eos = -1, gpu_layers = 0, n_ctx = 4096;
    std::string cache_mode = "off";
    bool verbose = false, stream = false, infill_mode = false;
    std::string log_path, suffix_text;
    int gap = 8;

    for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) usage(argv[0]); return argv[++i]; };
        if (a == "--verbose")          verbose = true;
        else if (a == "--stream")      stream = true;
        else if (a == "--log")         log_path = next();
        else if (a == "--suffix")      { suffix_text = next(); infill_mode = true; }
        else if (a == "--gap")         { gap = int_arg(next(), argv[0]); infill_mode = true; }
        else if (a == "--max-new")     cfg.max_new = int_arg(next(), argv[0]);
        else if (a == "--steps")       cfg.steps = int_arg(next(), argv[0]);
        else if (a == "--block-len")   cfg.block_len = int_arg(next(), argv[0]);
        else if (a == "--topk")        cfg.topk = int_arg(next(), argv[0]);
        else if (a == "--mask-token")  mask_token = int_arg(next(), argv[0]);
        else if (a == "--eos")         eos = int_arg(next(), argv[0]);
        else if (a == "--gpu-layers")  gpu_layers = int_arg(next(), argv[0]);
        else if (a == "--ctx")         n_ctx = int_arg(next(), argv[0]);
        else if (a == "--cache")       cache_mode = next();
        else { std::fprintf(stderr, "unknown option: %s\n", a.c_str()); usage(argv[0]); }
    }
    if (cache_mode != "off" && cache_mode != "delta") usage(argv[0]);

    if (!verbose) llama_log_set(quiet_log, nullptr);  // before load, so the loader is quiet too
    GgmlAdapter adapter(model_path, mask_token, eos, n_ctx, gpu_layers);
    const std::vector<int> prompt_ids = adapter.encode(prompt);
    if (prompt_ids.empty()) { std::fprintf(stderr, "prompt tokenized to nothing\n"); return 1; }

    CacheConfig cache;
    cache.mode = cache_mode;
    if (cache_mode == "delta") cache.full_refresh_every = 1;  // exact Tier A/B reuse

    // --stream: a live consumer of the §5.1 event spine — overwrite a per-step progress line as
    // the board denoises (commits per pass, slots remaining). The CLI is a consumer only.
    std::function<void(const Event&)> on_event;
    if (stream) {
        on_event = [](const Event& e) {
            if (const auto* ss = std::get_if<StepStats>(&e)) {
                std::fprintf(stderr, "\r[denoise] block %d  step %d  +%d committed  %d remaining   ",
                             ss->block, ss->step, ss->committed, ss->remaining);
                std::fflush(stderr);
            } else if (std::holds_alternative<GenFinished>(e)) {
                std::fprintf(stderr, "\r\033[K");  // clear the progress line before the result
            }
        };
    }

    const auto t0 = std::chrono::steady_clock::now();
    GenerateResult r;
    if (infill_mode) {
        // Native infilling: fill the gap between the prompt (prefix) and --suffix, under full
        // bidirectional attention — a capability autoregressive models structurally lack.
        if (gap < 1) { std::fprintf(stderr, "--gap must be >= 1\n"); return 1; }
        const std::vector<int> suffix_ids = adapter.encode(suffix_text);
        r = infill(adapter, prompt_ids, suffix_ids, gap, cfg, nullptr, on_event);
    } else {
        r = generate(adapter, prompt_ids, cfg, cache, nullptr, on_event);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!log_path.empty() && !write_jsonl(r.events, log_path))
        std::fprintf(stderr, "warning: could not write event log to %s\n", log_path.c_str());

    // The completion. Infill shows the fill in context: prefix + [fill] + suffix.
    if (infill_mode)
        std::printf("%s%s%s\n", prompt.c_str(), r.text.c_str(), suffix_text.c_str());
    else
        std::printf("%s%s\n", prompt.c_str(), r.text.c_str());
    std::fflush(stdout);  // flush before the stderr stats so the ordering is stable

    // Honest stats: tokens, the pass count (steps/token < 1 is the whole dLLM premise), wall time,
    // and the structural forward work (decode-token count) the cache reuse reduces.
    const double steps_per_tok = r.new_tokens > 0 ? (double)r.steps_total / r.new_tokens : 0.0;
    const double tok_per_s = ms > 0 ? r.new_tokens * 1000.0 / ms : 0.0;
    std::fprintf(stderr,
        "\n[cloze] %d tokens | %d passes (%.2f steps/token) | reason=%s | %.1f ms | %.1f tok/s\n"
        "        decoded %lld token-positions (forward work)%s%s\n",
        r.new_tokens, r.steps_total, steps_per_tok, r.reason.c_str(), ms, tok_per_s,
        adapter.decoded_tokens(),
        infill_mode ? " | infill (bidirectional)"
                    : (cfg.block_len > 0 ? " | semi-AR blocks" : " | whole-sequence"),
        (!infill_mode && cache_mode == "delta") ? " | KV reuse on" : "");
    return 0;
}
