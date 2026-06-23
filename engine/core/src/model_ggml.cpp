#include "cloze/model_ggml.hpp"

#include <cstring>
#include <stdexcept>
#include <utility>

#include "ggml-backend.h"  // ggml_backend_buffer_is_host (device-residency guard)

namespace cloze {

namespace {
// llama_backend_init/free are process-global and refcount-free; init once on first
// adapter, never free (cheap, and freeing while another adapter lives would crash).
bool g_backend_inited = false;
}  // namespace

GgmlModel::GgmlModel(const std::string& model_path, int mask_token_id,
                     int eos_token_id, int n_gpu_layers) {
    if (!g_backend_inited) {
        llama_backend_init();
        g_backend_inited = true;
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;  // > 0 offloads weights to GPU so logits can be device-resident
    model_ = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model_) throw std::runtime_error("failed to load model: " + model_path);

    vocab_ = llama_model_get_vocab(model_);
    cfg_.vocab_size = llama_vocab_n_tokens(vocab_);
    cfg_.mask_token_id = mask_token_id;
    if (eos_token_id >= 0) {
        cfg_.eos_token_id = eos_token_id;
    } else {
        const llama_token eos = llama_vocab_eos(vocab_);
        cfg_.eos_token_id = eos < 0 ? -1 : static_cast<int>(eos);  // -1 == none
    }
    // Head convention from GGUF metadata (same key + default as llama's diffusion-cli): "true"/absent
    // => Dream-family shifted head, "false" => LLaDA-family in-place head.
    char shift_buf[16];
    if (llama_model_meta_val_str(model_, "diffusion.shift_logits", shift_buf, sizeof(shift_buf)) >= 0) {
        shift_logits_ = (std::strcmp(shift_buf, "true") == 0);
    }
}

GgmlModel::~GgmlModel() {
    if (model_) llama_model_free(model_);
}

void GgmlAdapter::init_context(int n_ctx) {
    n_ctx_ = n_ctx;
    model_ = model_owner_->handle();
    vocab_ = model_owner_->vocab();
    cfg_ = model_owner_->config();
    n_embd_ = llama_model_n_embd(model_);    // hidden size for the white-box activation tap
    n_layer_ = llama_model_n_layer(model_);  // layer count for control-vector steering
    tap_layer_ = n_layer_ > 3 ? 2 : 0;  // layer 2: best per-token probe separation (sweep-validated)
    tap_name_ = "l_out-" + std::to_string(tap_layer_);   // per-layer residual name (llama-context.cpp "%s-%d")

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx_;
    cp.n_batch = n_ctx_;
    cp.n_ubatch = n_ctx_;  // single ubatch: a whole segment decodes in one pass
    cp.cb_eval = &GgmlAdapter::eval_cb_thunk;  // observe the mid-layer residual (white-box tap, no patch)
    cp.cb_eval_user_data = this;
    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) throw std::runtime_error("failed to create llama context");
    llama_set_causal_attn(ctx_, false);  // the diffusion forward: fully bidirectional
}

GgmlAdapter::GgmlAdapter(const std::string& model_path, int mask_token_id,
                         int eos_token_id, int n_ctx,
                         int n_gpu_layers, bool device_logits_passthrough)
    : model_owner_(std::make_shared<GgmlModel>(model_path, mask_token_id, eos_token_id, n_gpu_layers)),
      device_passthrough_(device_logits_passthrough) {
    init_context(n_ctx);
}

GgmlAdapter::GgmlAdapter(std::shared_ptr<GgmlModel> model, int n_ctx, bool device_logits_passthrough)
    : model_owner_(std::move(model)), device_passthrough_(device_logits_passthrough) {
    if (!model_owner_) throw std::invalid_argument("GgmlAdapter: null GgmlModel");
    init_context(n_ctx);
}

GgmlAdapter::~GgmlAdapter() {
    if (ctx_) llama_free(ctx_);  // the model is freed by GgmlModel when the last adapter releases it
    // backend intentionally left initialized (see note above).
}

void GgmlAdapter::evict_from(int pos) {
    // Drop KV for positions [pos, inf) on the single sequence (seq 0).
    llama_memory_seq_rm(llama_get_memory(ctx_), 0, pos, -1);
}

void GgmlAdapter::set_emit_activations(bool on) {
    emit_activations_ = on;
    // Final-layer embeddings stay on as the fallback (cheap); the mid-layer tap_layer_ path is captured
    // by eval_cb during decode. Additive: per-token hidden states alongside logits (pooling NONE).
    llama_set_embeddings(ctx_, on);
}

void GgmlAdapter::set_tap_layer(int il) {
    tap_layer_ = (il > 0 && il < n_layer_) ? il : 0;
    tap_name_ = tap_layer_ > 0 ? ("l_out-" + std::to_string(tap_layer_)) : "";
}

bool GgmlAdapter::eval_cb_thunk(struct ggml_tensor* t, bool ask, void* user_data) {
    return static_cast<GgmlAdapter*>(user_data)->eval_cb(t, ask);
}

bool GgmlAdapter::eval_cb(struct ggml_tensor* t, bool ask) {
    const char* nm = ggml_get_name(t);
    const bool read = emit_activations_ && tap_layer_ > 0 && tap_name_ == nm;   // the read tap
    const bool write = have_write_ && write_layer_ > 0 && write_name_ == nm;    // the state-WRITE
    if (!read && !write) return false;                        // not a residual tensor we read or write
    if (ask) return true;                                     // yes — hand me its data after it computes
    const int ne0 = static_cast<int>(t->ne[0]);  // n_embd
    const int ne1 = static_cast<int>(t->ne[1]);  // token rows (the decode segment)
    if (read) {
        tap_rows_ = ne1;
        tap_buf_.resize(static_cast<size_t>(ne0) * ne1);
        ggml_backend_tensor_get(t, tap_buf_.data(), 0, tap_buf_.size() * sizeof(float));
    }
    // WRITE side (GAP #1): overwrite each marked position's row (row = board position - this segment's
    // `from`), AFTER the read (so the tap reports the PRE-edit state) and before downstream layers consume
    // t — the activation-patch propagates forward. Rows outside [0, ne1) (e.g. a frozen-prefix decode) are
    // skipped, so the write lands only on the active block's decode.
    if (write && ne0 == n_embd_ &&
        write_buf_.size() == write_positions_.size() * static_cast<size_t>(n_embd_)) {
        for (size_t i = 0; i < write_positions_.size(); ++i) {
            const int row = write_positions_[i] - write_from_;
            if (row >= 0 && row < ne1) {
                ggml_backend_tensor_set(t, write_buf_.data() + i * static_cast<size_t>(n_embd_),
                                        static_cast<size_t>(row) * n_embd_ * sizeof(float),
                                        static_cast<size_t>(n_embd_) * sizeof(float));
            }
        }
    }
    return true;
}

void GgmlAdapter::set_steer(const std::vector<float>& data, int il_start, int il_end) {
    // Apply a control vector to the residual stream (the white-box WRITE). data is n_embd*n_layer,
    // layer-1-indexed; only [il_start, il_end] are applied. Empty data clears.
    llama_set_adapter_cvec(ctx_, data.empty() ? nullptr : data.data(),
                           data.size(), n_embd_, il_start, il_end);
}

void GgmlAdapter::clear_steer() {
    llama_set_adapter_cvec(ctx_, nullptr, 0, n_embd_, 0, n_layer_);
}

bool GgmlAdapter::write_state(int il, const std::vector<int>& positions,
                              const std::vector<float>& values) {
    if (il <= 0 || il >= n_layer_) return false;   // 0 = final (no l_out name); writable mids are [1, n_layer)
    if (n_embd_ <= 0) return false;
    if (values.size() != positions.size() * static_cast<size_t>(n_embd_)) return false;
    write_layer_ = il;
    write_name_ = "l_out-" + std::to_string(il);
    write_positions_ = positions;
    write_buf_ = values;
    have_write_ = true;
    return true;
}

void GgmlAdapter::clear_write() {
    have_write_ = false;
    write_positions_.clear();
    write_buf_.clear();
}

void GgmlAdapter::set_causal(bool on) {
    causal_ = on;
    llama_set_causal_attn(ctx_, on);
    // The attention mode changed, so any KV laid down under the other mode is now invalid
    // (a token's K/V depends on what it was allowed to attend to). Reset to a clean cache.
    llama_memory_clear(llama_get_memory(ctx_), true);
    frozen_end_ = 0;
    boundary_row_.clear();
}

ForwardResult GgmlAdapter::ar_forward(const std::vector<int>& tokens, int n_past) {
    const int len = static_cast<int>(tokens.size());
    if (len <= 0) throw std::invalid_argument("ar_forward: empty tokens");
    if (n_past < 0) throw std::invalid_argument("ar_forward: n_past < 0");
    if (n_past + len > n_ctx_) throw std::invalid_argument("ar_forward: exceeds n_ctx");
    write_from_ = n_past;   // board position -> tensor row mapping for the white-box state-WRITE (eval_cb)

    // Incremental causal decode: place `tokens` at absolute positions [n_past, n_past+len),
    // reusing whatever KV already covers [0, n_past). Only the LAST row is an output (the
    // next-token distribution we sample); the mid-layer tap (eval_cb) captures every row anyway.
    decoded_tokens_ += len;
    llama_batch batch = llama_batch_init(len, 0, 1);
    batch.n_tokens = len;
    for (int i = 0; i < len; ++i) {
        batch.token[i] = static_cast<llama_token>(tokens[i]);
        batch.pos[i] = n_past + i;          // absolute position: RoPE + KV slot
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == len - 1) ? 1 : 0;
    }
    const int rc = llama_decode(ctx_, batch);
    llama_batch_free(batch);
    if (rc != 0) throw std::runtime_error("ar_forward: llama_decode failed");

    const int vocab = cfg_.vocab_size;
    ForwardResult out;
    out.n_requested = 1;
    out.vocab = vocab;
    out.kv = std::make_shared<GgmlKV>(n_past + len);

    // Next-token logits for the last position. In-place AR head — NO Dream-family shift: row i's
    // logits already predict token i+1 (standard causal LM), unlike the diffusion forward.
    const float* logits = llama_get_logits_ith(ctx_, -1);
    if (logits) out.logits.assign(logits, logits + vocab);

    // White-box activation tap (Tier 2): the hidden state at the last decoded position — "the
    // model's state having just read this token". One row; act_rows = the absolute position.
    if (emit_activations_ && n_embd_ > 0) {
        out.n_embd = n_embd_;
        out.act_rows = {n_past + len - 1};
        out.activations.assign(static_cast<size_t>(n_embd_), 0.0f);
        if (tap_layer_ > 0 && tap_rows_ == len &&
            tap_buf_.size() == static_cast<size_t>(len) * n_embd_) {
            // mid-layer residual l_out-<tap_layer_>: last row = last token
            std::memcpy(out.activations.data(),
                        tap_buf_.data() + static_cast<size_t>(len - 1) * n_embd_,
                        static_cast<size_t>(n_embd_) * sizeof(float));
        } else {
            // final-layer fallback: the single output row (logits set only on the last token) is index 0
            const float* e = llama_get_embeddings_ith(ctx_, 0);
            if (e) std::memcpy(out.activations.data(), e, static_cast<size_t>(n_embd_) * sizeof(float));
        }
    }
    return out;
}

ForwardResult GgmlAdapter::harvest(const std::vector<int>& tokens) {
    const int len = static_cast<int>(tokens.size());
    if (len <= 0) throw std::invalid_argument("harvest: empty tokens");
    if (len > n_ctx_) throw std::invalid_argument("harvest: exceeds n_ctx");

    // One causal forward over the whole text from a clean cache: positions [0, len). We need ALL
    // rows' logits=1 so the tap captures every row (decode_only sets logits=1 everywhere). Reset the
    // KV first so this text's K/V never sees a prior text's positions (each /harvest is independent).
    llama_memory_clear(llama_get_memory(ctx_), true);
    frozen_end_ = 0;
    boundary_row_.clear();
    decode_only(tokens, 0, len);   // logits=1 at every position; tap_buf_ fills with all `len` rows

    const int vocab = cfg_.vocab_size;
    ForwardResult out;
    out.n_requested = 0;           // harvest returns state, not a distribution (logits left empty)
    out.vocab = vocab;
    out.kv = std::make_shared<GgmlKV>(len);

    if (n_embd_ <= 0) return out;  // model has no hidden size? nothing to harvest
    out.n_embd = n_embd_;
    out.act_rows.resize(len);
    for (int r = 0; r < len; ++r) out.act_rows[r] = r;  // row r = token r's residual

    if (emit_activations_ && tap_layer_ > 0 && tap_rows_ == len &&
        tap_buf_.size() == static_cast<size_t>(len) * n_embd_) {
        out.activations = tap_buf_;  // mid-layer residual l_out-<tap_layer_>, all `len` rows in order
    } else {
        // Final-layer fallback (tap_layer_ == 0, or the cb didn't fire): pull every row's embedding.
        // decode_only set logits=1 for all positions, so output index r == position r.
        out.activations.assign(static_cast<size_t>(len) * n_embd_, 0.0f);
        for (int r = 0; r < len; ++r) {
            const float* e = llama_get_embeddings_ith(ctx_, r);
            if (e) std::memcpy(out.activations.data() + static_cast<size_t>(r) * n_embd_, e,
                               static_cast<size_t>(n_embd_) * sizeof(float));
        }
    }
    return out;
}

void GgmlAdapter::decode_only(const std::vector<int>& board, int from, int to) {
    write_from_ = from;   // board position -> tensor row mapping for the white-box state-WRITE (eval_cb)
    const int len = to - from;
    if (len <= 0) throw std::invalid_argument("empty decode segment");
    decoded_tokens_ += len;
    llama_batch batch = llama_batch_init(len, 0, 1);
    batch.n_tokens = len;
    for (int i = 0; i < len; ++i) {
        batch.token[i] = static_cast<llama_token>(board[from + i]);
        batch.pos[i] = from + i;            // absolute position: RoPE + KV slot
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;                // logits at every position of the segment
    }
    const int rc = llama_decode(ctx_, batch);
    llama_batch_free(batch);
    if (rc != 0) throw std::runtime_error("llama_decode failed");
}

const float* GgmlAdapter::decode_segment(const std::vector<int>& board, int from, int to) {
    decode_only(board, from, to);
    return llama_get_logits(ctx_);  // host copy; row j == position from+j, valid until next decode
}

void GgmlAdapter::freeze_segment(const std::vector<int>& board, int from, int to) {
    // Decode [from, to) reusing the frozen [0, from); its K/V never sees forward (nothing
    // beyond `to` is in the cache yet), so it is frozen-exact under the one-way law.
    evict_from(from);
    const float* rows = decode_segment(board, from, to);
    // Boundary row = logits for position to-1 (the shifted-head source for position `to`).
    const int vocab = cfg_.vocab_size;
    const float* src = rows + static_cast<size_t>((to - 1) - from) * vocab;
    boundary_row_.assign(src, src + vocab);
    frozen_end_ = to;
}

int GgmlAdapter::active_start_from_mask(const Mask& mask, int n) {
    // The active (last) block = {q : block_id(q) == max} = {q : mask(q, n-1) == 1}, since
    // mask(q, k) = block_id(k) <= block_id(q) and block_id(n-1) is the max. Its start is the
    // smallest such q. A fully-bidirectional (all-ones) mask => 0 (whole-sequence).
    if (mask.n != n) throw std::invalid_argument("mask size != board size");
    for (int q = 0; q < n; ++q)
        if (mask.at(q, n - 1)) return q;
    return n;  // no position attends the last key — degenerate; treated as all-frozen
}

ForwardResult GgmlAdapter::forward(const std::vector<int>& board,
                                   const Mask& mask,
                                   const std::shared_ptr<KVState>& kv,
                                   const std::optional<std::vector<int>>& recompute_kv,
                                   const std::vector<int>& logits_for) {
    const int n = static_cast<int>(board.size());
    if (n == 0) throw std::invalid_argument("empty board");
    if (n > n_ctx_) throw std::invalid_argument("board exceeds n_ctx");

    // Everything before the active block is frozen-exact; the active block is recomputed.
    const int active_start = active_start_from_mask(mask, n);
    const bool reuse = (kv != nullptr);

    // recompute_kv, when given, must be a contiguous suffix [s, n) and its start must not
    // precede the active block (we never recompute a frozen block's interior — that scattered
    // Tier C path isn't expressible incrementally; the lab raises the same way).
    if (recompute_kv.has_value()) {
        const auto& r = *recompute_kv;
        if (!r.empty()) {
            const int s = r.front();
            for (int i = 0; i < static_cast<int>(r.size()); ++i)
                if (r[i] != s + i || r.back() != n - 1)
                    throw std::runtime_error("GgmlAdapter: recompute_kv must be a contiguous "
                                             "suffix [s, n) (Tier A/B prefix reuse)");
        }
    }

    if (!reuse) {
        // Cold start: rebuild the frozen prefix from scratch.
        llama_memory_clear(llama_get_memory(ctx_), true);
        frozen_end_ = 0;
        boundary_row_.clear();
    }
    // Lay down + freeze the just-finalized block(s) so [0, active_start) is frozen-exact.
    // The gap is always exactly one block (prompt, or the block that just finalized), so a
    // single segment decode suffices and its boundary row feeds the active block's first slot.
    if (frozen_end_ < active_start) {
        freeze_segment(board, frozen_end_, active_start);
    } else if (frozen_end_ > active_start) {
        // Active block moved backward — only happens if a caller reuses across an
        // incompatible board; rebuild from cold to stay exact.
        llama_memory_clear(llama_get_memory(ctx_), true);
        frozen_end_ = 0;
        boundary_row_.clear();
        if (active_start > 0) freeze_segment(board, 0, active_start);
    }

    // Decode the active block [active_start, n), reusing the frozen prefix [0, active_start).
    evict_from(active_start);
    const int vocab = cfg_.vocab_size;

    ForwardResult out;
    out.n_requested = static_cast<int>(logits_for.size());
    out.vocab = vocab;
    out.kv = std::make_shared<GgmlKV>(n);

    if (active_start >= n) return out;  // no active rows (degenerate); logits_for must be empty

    // The Dream-family shift: logits for position m come from source row m-1 (position 0 uses its
    // own row 0 as filler, matching the lab's max(m-1, 0) — needed for suffix-only infill). A row
    // is in the active decode iff its source row >= active_start; source == active_start-1 is the
    // frozen boundary row, served from the host boundary_row_ (one row, at most one per pass).
    // Decide the device path BEFORE decoding so we can suppress llama's decode-time D2H when we
    // will read on-device — but only once residency is proven (the first probe pass never skips).
    const bool shift = model_owner_->shift_logits();  // GGUF diffusion.shift_logits: Dream=true, LLaDA=false (in-place)
    auto src_of = [shift](int m) { return shift ? (m >= 1 ? m - 1 : 0) : m; };
    bool boundary_needed = false;
    for (int m : logits_for) {
        if (m < 0 || m >= n) throw std::invalid_argument("logits_for position out of range");
        if (src_of(m) < active_start) boundary_needed = true;
    }
    // A device path that needs the boundary row requires it to have been captured (frozen prefix).
    // The white-box tap reads HOST embeddings, so it always takes the host path (never the zero-copy
    // device-logits passthrough), which keeps logits + activations both on the host this pass.
    const bool want_device = device_passthrough_ && !emit_activations_ &&
                             (!boundary_needed || !boundary_row_.empty());
    const bool skip_d2h = want_device && device_confirmed_;

    if (skip_d2h) llama_set_skip_raw_logits(ctx_, true);  // CLOZE PATCH: no full-vocab D2H this decode
    decode_only(board, active_start, n);
    if (skip_d2h) llama_set_skip_raw_logits(ctx_, false);
    // When not skipped, llama copied n_outputs*vocab logits floats to host during decode.
    if (!skip_d2h) logits_d2h_floats_ += static_cast<long long>(n - active_start) * vocab;

    // White-box activation tap (Tier 2): pull the per-position hidden state for the active block.
    // Embeddings were enabled in set_emit_activations(); decode_only set logits=1 for every active
    // position, so output index j == active row j == board position active_start+j (batch order).
    if (emit_activations_ && n_embd_ > 0) {
        const int n_active = n - active_start;
        out.n_embd = n_embd_;
        out.act_rows.resize(n_active);
        for (int j = 0; j < n_active; ++j) out.act_rows[j] = active_start + j;
        if (tap_layer_ > 0 && tap_rows_ == n_active &&
            tap_buf_.size() == static_cast<size_t>(n_active) * n_embd_) {
            out.activations = tap_buf_;  // mid-layer residual l_out-<tap_layer_>, row j = position active_start+j
        } else {
            // final-layer fallback via the embeddings API (output index j == active row j == position+j)
            out.activations.assign(static_cast<size_t>(n_active) * n_embd_, 0.0f);
            for (int j = 0; j < n_active; ++j) {
                const float* e = llama_get_embeddings_ith(ctx_, j);
                if (e) std::memcpy(out.activations.data() + static_cast<size_t>(j) * n_embd_, e,
                                   static_cast<size_t>(n_embd_) * sizeof(float));
            }
        }
    }

    // Zero-copy device path (DESIGN §4.3): hand back the device tensor + per-position source rows
    // (a boundary row, if any, as the single host row in out.boundary_row); the host `logits` stays
    // empty. (Pairs with KernelCommitSelector; a host selector needs the host path.)
    if (want_device) {
        ggml_tensor* t = llama_get_logits_tensor(ctx_);  // synchronizes; raw graph output (batch order)
        if (t && t->buffer && !ggml_backend_buffer_is_host(t->buffer)) {
            device_confirmed_ = true;  // proven device-resident -> future passes may skip the D2H
            out.device_resident = true;
            out.device_logits = static_cast<const float*>(t->data);
            out.device_n_rows = static_cast<int>(t->ne[1]);
            out.device_src_rows.reserve(out.n_requested);
            for (int m : logits_for) {
                const int src = src_of(m);
                if (src >= active_start) {
                    out.device_src_rows.push_back(src - active_start);  // in the device tensor
                } else {  // the frozen boundary row (host)
                    out.device_src_rows.push_back(-1);
                    out.boundary_row = boundary_row_;  // one row; selector H2D's it into its slot
                }
            }
            ++device_forwards_;
            return out;  // host `logits` left empty: the full-vocab D2H is skipped
        }
        if (skip_d2h)
            throw std::runtime_error("GgmlAdapter: skipped the logits D2H but logits are not "
                                     "device-resident (residency changed after confirmation)");
        // else: first probe on a host-resident build — fall through to the host path (D2H ran).
    }

    // Host path (fallback / non-CUDA build): pull the host logits, apply the shift.
    const float* rows = llama_get_logits(ctx_);  // row j == position active_start+j
    out.logits.resize(static_cast<size_t>(out.n_requested) * vocab);
    for (int r = 0; r < out.n_requested; ++r) {
        const int m = logits_for[r];
        const int src = src_of(m);
        const float* row;
        if (src >= active_start) {
            row = rows + static_cast<size_t>(src - active_start) * vocab;
        } else if (src == active_start - 1 && !boundary_row_.empty()) {
            row = boundary_row_.data();
        } else {
            throw std::runtime_error("GgmlAdapter: shifted-head source row is frozen and "
                                     "uncaptured (logits_for not at the active block front)");
        }
        std::memcpy(out.logits.data() + static_cast<size_t>(r) * vocab, row,
                    static_cast<size_t>(vocab) * sizeof(float));
    }
    return out;
}

std::vector<int> GgmlModel::encode(const std::string& text) const {
    // No BOS, parse special tokens — matches the lab's raw tok.encode for Qwen2 and the
    // forward test. Two-pass: probe the needed size, then tokenize.
    int need = -llama_tokenize(vocab_, text.c_str(), static_cast<int>(text.size()),
                               nullptr, 0, /*add_special=*/false, /*parse_special=*/true);
    if (need <= 0) return {};
    std::vector<llama_token> toks(need);
    int n = llama_tokenize(vocab_, text.c_str(), static_cast<int>(text.size()), toks.data(),
                           need, /*add_special=*/false, /*parse_special=*/true);
    if (n < 0) throw std::runtime_error("tokenize failed");
    return std::vector<int>(toks.begin(), toks.begin() + n);
}

std::string GgmlModel::decode(const std::vector<int>& ids) const {
    std::string out;
    char piece[512];
    for (int id : ids) {
        int np = llama_token_to_piece(vocab_, static_cast<llama_token>(id), piece,
                                      sizeof(piece), /*lstrip=*/0, /*special=*/false);
        if (np < 0) np = 0;
        out.append(piece, static_cast<size_t>(np));
    }
    return out;
}

std::vector<int> GgmlAdapter::encode(const std::string& text) const {
    return model_owner_->encode(text);
}

std::string GgmlAdapter::decode(const std::vector<int>& ids) const {
    return model_owner_->decode(ids);
}

}  // namespace cloze
