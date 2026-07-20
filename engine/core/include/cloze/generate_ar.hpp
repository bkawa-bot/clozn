// cloze/generate_ar.hpp — autoregressive generation with the white-box read+steer harness.
//
// Unlike generate()/infill()/denoise() (the backend-free diffusion scheduler validated against the
// lab goldens), this is a thin left-to-right decode loop that lives in the ggml layer: it drives
// GgmlAdapter::ar_forward directly (incremental causal KV decode), so it can't be backend-free.
// It exists because the interpretability layer is model-AGNOSTIC — the activation tap, concept
// probes, logit-lens, and steering all sit on a ForwardResult, not on the denoiser — so the entire
// llama.cpp AR model zoo (Llama/Qwen/Mistral/Gemma/...) gets the same white-box treatment as a dLLM.
//
// It emits the SAME §5.1 events as the diffusion loops — GenStarted, TokensCommitted (one item per
// token), StepFeatures + StepLens per token, GenFinished — so every consumer (CLI, server SSE, viz)
// works unchanged. The honest asymmetry: AR has no token-revision / parallel-pre-commit / infill
// views (those are uniquely diffusion); AR gives the standard per-token read.
#pragma once

#include <functional>
#include <vector>

#include "cloze/events.hpp"
#include "cloze/generate.hpp"    // GenerateConfig, SampleConfig, GenerateResult
#include "cloze/model_ggml.hpp"  // GgmlAdapter
#include "cloze/probe.hpp"       // ConceptProbes

namespace cloze {

struct BranchResult {
    std::vector<int> generated;
    std::string text;
    std::string reason;  // "eos" | "length"
    int new_tokens = 0;
};

// Greedy by default (SampleConfig: temperature 0). Stops at config.max_new tokens or EOS
// (config.steps / block_len / topk are ignored — AR commits exactly one token per pass).
// `read_probes` (optional) supplies the concept directions for the per-token StepFeatures; calibrate
// them in CAUSAL mode (activations differ from the bidirectional diffusion tap). Steering is applied
// by the caller via adapter.set_steer(...) before this call, exactly as on the diffusion paths.
GenerateResult generate_ar(GgmlAdapter& adapter,
                           const std::vector<int>& prompt_ids,
                           const GenerateConfig& config,
                           const std::function<void(const Event&)>& on_event = {},
                           const SampleConfig& sample = {},
                           const ConceptProbes* read_probes = nullptr,
                           // Optional PyTorch-trained soft prefix: prefix_rows x n_embd raw embeddings spliced
                           // in ahead of the prompt (via ar_forward_embd) before decoding, so a memory learned
                           // on the HF model rides into this ggml generation. nullptr/0 = no prefix (default).
                           const std::vector<float>* prefix_embd = nullptr,
                           int prefix_rows = 0,
                           // Optional early-stop reference (prove-all ablated arms): the baseline reply's
                           // committed token ids. Generation halts at the first generated token that differs
                           // from reference[k] -- yielding a bit-exact PREFIX of the full reply plus a
                           // diverged/diverged_at verdict. Sampling/batching are untouched: this is ONLY a
                           // termination check, so greedy determinism holds (the reply is what full generation
                           // would produce, truncated). nullptr/empty = no reference (full generation, default).
                           const std::vector<int>* reference = nullptr);

// Batched multi-sequence branching: prefill a shared prompt once, then decode N independent
// continuations in parallel using a single llama_decode per step. Each branch gets its own
// KV sequence (via branch_kv) and its own RNG (base_sample.seed + branch_index). Greedy
// branches from the same prompt produce identical output (the correctness bar). Returns one
// BranchResult per branch. Cleans up branch sequences before returning.
std::vector<BranchResult> generate_ar_branched(
    GgmlAdapter& adapter,
    const std::vector<int>& prompt_ids,
    int n_branches,
    int max_tokens,
    const SampleConfig& base_sample = {});

}  // namespace cloze
