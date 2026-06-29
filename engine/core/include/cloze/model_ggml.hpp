// cloze/model_ggml.hpp — the L0 ggml/llama.cpp ModelAdapter. This is the ONE place a model
// backend lives on the C++ side (DESIGN invariant 1, the C++ analogue of torch living
// only under lab/cloze_lab/models/). The scheduler/runtime never include this header.
//
// Diffusion forward (slice 1): load a GGUF, decode the board bidirectionally
// (llama_set_causal_attn false), apply the Dream-family shift — logits for board position p
// are read from output row p-1 (the adapter owns the shift).
//
// KV reuse under the one-way law (slice 2): llama.cpp's high-level API exposes only causal
// on/off, never an arbitrary attention mask — so the block-causal one-way law cannot be a
// mask. Instead we get it from DECODE ORDER. Each segment (prompt, then each block) is
// decoded while later positions don't yet exist, so its frozen K/V never attends forward —
// exactly Tier A/B exactness. The active block is re-decoded every step (Tier C = full
// active recompute, exact). The shifted head's source row for a block's first slot
// (position active_start-1, which is frozen and not re-emitted) is captured as a "boundary
// row" when that prefix block is frozen. This reproduces the lab's block-causal goldens
// without a custom ggml mask. The active-block start is read from the mask:
// active_start = min{q : mask(q, n-1)} (0 when whole-sequence / fully bidirectional).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "cloze/model.hpp"
#include "llama.h"

namespace cloze {

// The opaque KV handle threaded by the scheduler (invariant 4). The llama context holds one
// stateful KV cache, so this is just a marker of how many board positions it now covers;
// the adapter keeps the real bookkeeping (frozen boundary + boundary row) internally. A
// non-null kv passed back into forward() is the "reuse" signal (vs null = cold start).
struct GgmlKV : KVState {
    int n;
    explicit GgmlKV(int n_) : n(n_) {}
    int seq_len() const override { return n; }
};

// The loaded model weights + vocab + ModelConfig, shareable across contexts. llama_model is
// read-only after load, so N contexts can share ONE GgmlModel — essential for serving
// concurrency: the weights (the bulk of VRAM) are loaded once; only the per-context KV/compute
// buffers replicate. Hold it by shared_ptr and hand it to GgmlAdapter(model, n_ctx, ...).
class GgmlModel {
public:
    // mask_token_id is checkpoint-specific (open-dCoder <M> = 151665). eos_token_id < 0 => take
    // the model's own EOS from the vocab. n_gpu_layers > 0 offloads weights to the GPU.
    GgmlModel(const std::string& model_path, int mask_token_id,
              int eos_token_id = -1, int n_gpu_layers = 0);
    ~GgmlModel();
    GgmlModel(const GgmlModel&) = delete;
    GgmlModel& operator=(const GgmlModel&) = delete;

    const ModelConfig& config() const { return cfg_; }
    llama_model* handle() const { return model_; }
    const llama_vocab* vocab() const { return vocab_; }
    // Diffusion head convention (GGUF `diffusion.shift_logits`): true = Dream-family SHIFTED head
    // (logits for position p come from row p-1); false = LLaDA-family IN-PLACE head (row p). Defaults
    // true when the key is absent (e.g. a Dream GGUF converted as Qwen2) — matches llama's diffusion-cli.
    bool shift_logits() const { return shift_logits_; }

    // Tokenization lives on the model (vocab-only, no context) so it's usable without — and
    // concurrently with — any context (a server tokenizes here, then acquires a context to run).
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

private:
    llama_model* model_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    ModelConfig cfg_{};
    bool shift_logits_ = true;  // see shift_logits(): default to the Dream-family shifted head
};

class GgmlAdapter : public ModelAdapter {
public:
    // Standalone: load a fresh model + create a context over it (the original API).
    // device_logits_passthrough: when set AND the active-block logits land in a device buffer
    // AND no frozen boundary row is needed this pass, forward() returns the device-resident
    // logits tensor (DESIGN §4.3 zero-copy) and SKIPS the host D2H copy; otherwise it falls back
    // to the host path. Requires a GGML_CUDA llama + n_gpu_layers offload to actually trigger.
    GgmlAdapter(const std::string& model_path, int mask_token_id,
                int eos_token_id = -1, int n_ctx = 4096,
                int n_gpu_layers = 0, bool device_logits_passthrough = false);
    // Shared model: create a private context over an already-loaded GgmlModel (for a server pool
    // of concurrent contexts that share one copy of the weights).
    explicit GgmlAdapter(std::shared_ptr<GgmlModel> model, int n_ctx = 4096,
                         bool device_logits_passthrough = false);
    ~GgmlAdapter() override;

    GgmlAdapter(const GgmlAdapter&) = delete;
    GgmlAdapter& operator=(const GgmlAdapter&) = delete;

    const ModelConfig& config() const override { return cfg_; }

    ForwardResult forward(const std::vector<int>& board,
                          const Mask& mask,
                          const std::shared_ptr<KVState>& kv,
                          const std::optional<std::vector<int>>& recompute_kv,
                          const std::vector<int>& logits_for) override;

    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& ids) const override;

    // Total token-positions pushed through llama_decode since the last reset. The
    // hardware-independent measure of forward work: with cache off it grows by the full
    // board each pass; with Tier A/B reuse it grows only by the active block (+ one-time
    // freezes). The documented work metric (DESIGN invariant 5) — a wall-clock number ships
    // alongside it, never instead of it.
    long long decoded_tokens() const { return decoded_tokens_; }
    void reset_decoded_tokens() { decoded_tokens_ = 0; }

    // Toggle the §4.3 zero-copy device-logits path at runtime (vs the constructor default).
    void set_device_passthrough(bool on) { device_passthrough_ = on; }
    // White-box activation tap (Tier 2): when on, forward() fills ForwardResult.activations with the
    // per-position hidden state (llama_get_embeddings_ith) for the active block and takes the host
    // path (no device-logits passthrough). Default off => empty activations, zero overhead, the 8
    // scheduler goldens untouched. The white-box projection (concept probes) is a separate consumer.
    void set_emit_activations(bool on);
    bool emit_activations() const { return emit_activations_; }
    int tap_layer() const { return tap_layer_; }  // residual layer the white-box tap reads (0 = final via embeddings)
    void set_tap_layer(int il);  // change which layer the cb_eval callback captures (0 = final via embeddings)
    // White-box WRITE side (Tier 2): activation steering via a llama control vector. `data` is the
    // n_embd*n_layer buffer (layer-1-indexed) applied to the residual stream over [il_start, il_end]
    // (inclusive, 1-based); clear_steer() removes it. The causal counterpart to the read tap — push a
    // concept direction in and watch the denoiser's commits move. NOT a golden path (off by default).
    void set_steer(const std::vector<float>& data, int il_start, int il_end);
    void clear_steer();
    // White-box STATE-WRITE (GAP #1, task #43): overwrite the residual at board `positions` with
    // `values` ([positions.size()*n_embd]) at the l_out-<il> mid layer (same naming as the read tap),
    // applied on EVERY subsequent forward via the SAME eval callback as the tap (ggml_backend_tensor_set,
    // no llama patch) until clear_write(). The causal inverse of the tap: read state out
    // (ForwardResult.activations) -> edit -> write_state -> observe the next forward. Returns true if armed.
    bool write_state(int il, const std::vector<int>& positions,
                     const std::vector<float>& values) override;
    void clear_write();
    int n_layer() const { return n_layer_; }

    // --- Autoregressive (causal) decode: the white-box read/steer harness over a standard
    // left-to-right LLM (any llama.cpp AR GGUF — Llama/Qwen/Mistral/...), as opposed to the
    // diffusion denoiser. The interpretability primitives (tap, probes, steering) are identical;
    // only the GENERATION paradigm differs. set_causal(true) flips the context to causal attention
    // and clears the KV (KV computed under the other attn mode is invalid); ar_forward decodes the
    // next tokens incrementally (KV-cached) and returns the LAST token's next-token logits + its
    // hidden state (tap-aware) as a ForwardResult the white-box helpers consume. No head shift
    // (the AR head is in-place: row i predicts token i+1). Not a golden path (diffusion is the oracle).
    void set_causal(bool on);
    bool causal() const { return causal_; }
    ForwardResult ar_forward(const std::vector<int>& tokens, int n_past);

    // Like ar_forward, but the inputs are RAW embeddings (n_rows x n_embd, row-major) spliced in at
    // [n_past, n_past+n_rows) via the llama_batch.embd path (the same one multimodal models use to inject
    // image vectors) instead of token ids. The bridge that lets a PyTorch-trained soft prefix ride into
    // the ggml runtime: train-on-HF, serve-on-llama.cpp. Returns the last row's next-token logits.
    ForwardResult ar_forward_embd(const std::vector<float>& embd, int n_rows, int n_past);

    // Forward-HARVEST (the §3.1 "activation harvesting at scale" path): one causal forward over a
    // whole text and ALL its per-token residuals at the tap layer — NOT just the last row like
    // ar_forward. Decodes `tokens` at absolute positions [0, n) under causal attention with the tap
    // on, then returns the full [n x n_embd] tap_buf_ (row r = the residual the model computed for
    // token r, having read tokens 0..r). One forward per text => natural-text activations, cheaply,
    // and no sustained-generation streaming (which crashed §3.6). Returns activations + act_rows
    // (= [0, n)) + n_embd in a ForwardResult; logits are left empty (harvest wants the state, not a
    // distribution). The caller must have set_causal(true) + set_emit_activations(true) + a mid tap
    // (tap_layer_ > 0); on the final-layer fallback (tap_layer_ == 0) it returns the per-token
    // embeddings instead (one D2H per token — slower but still correct). Not a golden path.
    ForwardResult harvest(const std::vector<int>& tokens);
    // Number of forwards that actually took the device path (host D2H skipped) — lets a test
    // prove the zero-copy path was exercised rather than silently falling back to host.
    long long device_forwards() const { return device_forwards_; }
    void reset_device_forwards() { device_forwards_ = 0; }

    // Logits floats that crossed PCIe device->host (llama's decode-time D2H). The §4.3 metric:
    // the zero-copy path drives this toward zero (only the kernel's ~2*n_masked outputs cross),
    // vs n_outputs*vocab per pass on the host path. Structural data movement, not wall-clock.
    long long logits_d2h_floats() const { return logits_d2h_floats_; }
    void reset_logits_d2h_floats() { logits_d2h_floats_ = 0; }

private:
    // Decode board[from, to) at absolute positions [from, to), reusing whatever KV currently
    // covers [0, from). decode_only just runs llama_decode (no logits extract — leaves them on
    // the backend for the zero-copy path). decode_segment additionally returns the HOST logits
    // (triggering the D2H copy): row j == position from+j, valid until the next decode.
    void decode_only(const std::vector<int>& board, int from, int to);
    const float* decode_segment(const std::vector<int>& board, int from, int to);

    // Decode [from, to) and freeze it: capture the boundary row (logits for position to-1,
    // the shifted-head source for the next block's first slot) and advance frozen_end_.
    void freeze_segment(const std::vector<int>& board, int from, int to);

    // active_start = min{q : mask(q, n-1)}; 0 when the mask is fully bidirectional.
    static int active_start_from_mask(const Mask& mask, int n);

    void evict_from(int pos);  // drop KV for positions [pos, inf)

    void init_context(int n_ctx);  // create + configure the llama context over model_

    std::shared_ptr<GgmlModel> model_owner_;  // keeps the (possibly shared) model alive
    llama_model* model_ = nullptr;            // == model_owner_->handle()
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    ModelConfig cfg_{};
    int n_ctx_ = 0;
    bool device_passthrough_ = false;
    bool device_confirmed_ = false;  // device residency proven by a prior probe (then safe to skip D2H)
    bool emit_activations_ = false;  // white-box tap: fill ForwardResult.activations (forces host path)
    bool causal_ = false;            // attention mode: false = diffusion (bidirectional), true = AR (causal)
    int n_embd_ = 0;                 // hidden size, cached from the model for the activation tap
    int n_layer_ = 0;                // layer count, cached for control-vector (steering) sizing
    int tap_layer_ = 0;              // residual layer to tap (~2/3 depth); 0 => final layer via embeddings
    std::string tap_name_;           // "l_out-<tap_layer_>": the residual tensor the eval callback captures
    std::vector<float> tap_buf_;     // last-decode residual at tap_layer_ [rows*n_embd], filled by eval_cb
    int tap_rows_ = 0;               // token rows in tap_buf_ (= the last decode segment's length)
    // White-box state-WRITE (GAP #1): the mirror of the tap — eval_cb overwrites these positions' rows
    // at write_name_ during decode (ggml_backend_tensor_set), applied until clear_write().
    bool have_write_ = false;
    int write_layer_ = 0;
    std::string write_name_;             // "l_out-<write_layer_>"
    std::vector<int> write_positions_;   // board positions to overwrite
    std::vector<float> write_buf_;       // [write_positions_.size()*n_embd] new residual rows
    int write_from_ = 0;                 // current decode segment's `from` (board position -> tensor row)
    // ggml scheduler eval callback: grabs the mid-layer residual during llama_decode (no source patch).
    static bool eval_cb_thunk(struct ggml_tensor* t, bool ask, void* user_data);
    bool eval_cb(struct ggml_tensor* t, bool ask);

    // KV reuse state. [0, frozen_end_) is laid into the llama cache and frozen-exact.
    int frozen_end_ = 0;
    std::vector<float> boundary_row_;  // logits for position frozen_end_-1 (empty if none)
    long long decoded_tokens_ = 0;     // token-positions sent through llama_decode
    long long device_forwards_ = 0;    // forwards that took the zero-copy device path
    long long logits_d2h_floats_ = 0;  // logits floats copied device->host (the D2H the skip avoids)
};

}  // namespace cloze
