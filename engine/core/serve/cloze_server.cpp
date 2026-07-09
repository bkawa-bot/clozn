// cloze_server.cpp — the L2 serving layer: an HTTP server over the cloze runtime. Loads one GGUF
// diffusion LM and serves completions + infill, with SSE streaming that emits the §5.1 event spine
// directly (the native streaming protocol the events were designed for). Uses the single-header
// cpp-httplib + nlohmann/json that llama.cpp already vendors — no new dependencies.
//
//   cloze-server <model.gguf> [--port N] [--host H] [--gpu-layers N] [--mask-token ID] [--eos ID] [--ctx N]
//
// Endpoints:
//   GET  /health           -> {"status":"ok","model":...}
//   POST /v1/completions    {prompt, max_tokens, steps, block_len, topk, cache, stream}
//                          -> OpenAI-ish {choices:[{text,finish_reason}], usage}; stream=true => SSE
//   POST /v1/infill         {prefix, suffix, gap, steps, topk, stream} -> fill-in-the-middle
//
// The GgmlAdapter wraps one stateful llama context, so a mutex serializes generation; HTTP I/O is
// concurrent, generation is one-at-a-time (correct for a single context). cpp-httplib is the split
// build (httplib.cpp compiled into this target by CMake supplies the out-of-line definitions).
#include "httplib.h"
#include "nlohmann/json.hpp"

#include "cloze/events.hpp"
#include "cloze/generate.hpp"
#include "cloze/generate_ar.hpp"
#include "cloze/model_ggml.hpp"
#ifdef CLOZE_SAE
#include "cloze/sae.hpp"  // on-device SAE feature readout (--sae; built with CLOZE_BUILD_SAE)
#endif
#include "viz_html.hpp"

#include "ggml.h"       // J-lens: standalone CPU ggml graph (J_l @ h -> rms_norm -> head)
#include "ggml-cpu.h"   // ggml_graph_compute_with_ctx
#include "gguf.h"       // read the GGUF's own output_norm.weight + output.weight for the J-lens head

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using json = nlohmann::json;
using namespace cloze;

namespace {

std::atomic<uint64_t> g_req_counter{0};
std::string make_id(const char* prefix) {
    return std::string(prefix) + std::to_string(g_req_counter.fetch_add(1));
}

// dLLM reason -> OpenAI finish_reason.
const char* finish_reason(const std::string& reason) {
    return reason == "eos" ? "stop" : "length";  // length | steps_exhausted -> "length"
}

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

GenerateConfig config_from(const json& body) {
    GenerateConfig cfg;
    int mn = body.value("max_tokens", 32); cfg.max_new = mn < 1 ? 1 : mn;   // clamp >= 1: the generators
    int st = body.value("steps", 8);       cfg.steps   = st < 1 ? 1 : st;   // throw on < 1, and on the
    //                                          STREAMING path an uncaught throw silently abort()s the engine.
    cfg.block_len = body.value("block_len", 0);
    cfg.topk = body.value("topk", -1);
    return cfg;
}

CacheConfig cache_from(const json& body) {
    CacheConfig cache;
    cache.mode = body.value("cache", std::string("off"));
    if (cache.mode == "delta") cache.full_refresh_every = 1;
    return cache;
}

// §5.2 revision: opt-in "the model changes its mind". Off by default => the commit path is unchanged.
ReviseConfig revise_from(const json& body) {
    ReviseConfig revise;
    revise.enabled = body.value("revise", false);
    revise.tau_revise = body.value("tau_revise", 0.5);
    revise.max_revisions = body.value("max_revisions", 1);
    return revise;
}

// Sampling: opt-in temperature + repetition penalty. Defaults (T=0, penalty=1) keep greedy decoding,
// so omitting them is byte-identical to before.
SampleConfig sample_from(const json& body) {
    SampleConfig sample;
    sample.temperature = body.value("temperature", 0.0);
    sample.rep_penalty = body.value("rep_penalty", 1.0);
    sample.seed = body.value("seed", static_cast<uint64_t>(0));
    return sample;
}

// SSE data payload for one event, enriched for the browser viz (which has no tokenizer):
//   gen_started      += prompt_pieces[] (decoded prompt tokens, so each shows as its own cell)
//   tokens_committed += piece per token (the committed text)
// All other events pass through as the plain §5.1 wire form. Presentation only — the core events
// stay {prompt_tokens,...} / {pos,id,conf}.
// A decoded token piece can be a PARTIAL multi-byte UTF-8 sequence: byte-fallback tokens split a character
// (e.g. "½", an em-dash, an emoji) across token boundaries, so ONE token's piece may end mid-codepoint.
// nlohmann's default (strict) serializer THROWS on the incomplete bytes -- and on the streaming path that
// aborted the whole generation mid-reply (truncated text, and the trailing gen_finished/finish_reason frames
// never sent). Serialize with `replace` so an invalid byte becomes U+FFFD and a split char never kills the
// stream: the reply completes and the final frames arrive. (A split char shows as U+FFFD until the model's
// next multi-byte token; a full buffering fix is a separate refinement.)
static std::string dump_json(const json& j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

std::string sse_data(const Event& e, const GgmlModel& model, const std::vector<int>& prompt_ids,
                     const std::vector<int>& suffix_ids) {
    if (const auto* gs = std::get_if<GenStarted>(&e)) {
        json pieces = json::array();
        for (int id : prompt_ids) pieces.push_back(model.decode({id}));
        json sfx = json::array();
        for (int id : suffix_ids) sfx.push_back(model.decode({id}));  // infill: the fixed right-context
        return dump_json(json{{"t", gs->t}, {"type", "gen_started"}, {"prompt_tokens", gs->prompt_tokens},
                    {"block_len", gs->block_len}, {"max_new", gs->max_new},
                    {"prompt_pieces", pieces}, {"suffix_pieces", sfx}});
    }
    if (const auto* tc = std::get_if<TokensCommitted>(&e)) {
        json items = json::array();
        for (const auto& it : tc->items)
            items.push_back({{"pos", it.pos}, {"id", it.id}, {"conf", it.conf},
                             {"piece", model.decode({it.id})}});
        return dump_json(json{{"t", tc->t}, {"type", "tokens_committed"}, {"block", tc->block},
                    {"items", items}});
    }
    if (const auto* sl = std::get_if<StepLens>(&e)) {
        json pieces = json::array();
        for (int id : sl->ids) pieces.push_back(model.decode({id}));     // decode candidates (viz has no tokenizer)
        return dump_json(json{{"t", sl->t}, {"type", "step_lens"}, {"block", sl->block}, {"k", sl->k},
                    {"positions", sl->positions}, {"ids", sl->ids}, {"probs", sl->probs},
                    {"pieces", pieces}});
    }
    return to_jsonl_line(e);
}

// Map a list of [start, end) UTF-8 BYTE ranges in `text` onto a token board with those tokens
// MASKED — the "revise this selection" lowering. A token is masked when its byte span overlaps any
// requested range, so a partial selection expands to whole tokens (you revise whole tokens, never
// sub-token bytes). Each token's byte span is its detokenized piece length, summed left to right;
// this reconstructs the text exactly for the byte-level Qwen2/Dream BPE vocab (special tokens, which
// detok to empty, are the only drift source and don't occur in ordinary pasted text).
// `grow` adds that many EXTRA mask slots after each selected run, giving the model headroom to
// rewrite a span into a different length: it fills what it needs and pads the rest (an EOS/empty
// piece renders blank), so a K-token selection can become anywhere from short to K+grow tokens.
std::vector<int> masked_board_from_spans(const GgmlModel& model, const std::string& text,
                                         const std::vector<std::pair<int, int>>& byte_spans,
                                         int mask_token, int grow) {
    const std::vector<int> toks = model.encode(text);
    std::vector<int> board;
    board.reserve(toks.size() + static_cast<size_t>(grow > 0 ? grow * 4 : 0));
    int cum = 0;
    bool prev_selected = false;
    for (size_t i = 0; i < toks.size(); ++i) {
        const int start = cum;
        cum += static_cast<int>(model.decode({toks[i]}).size());
        const int end = cum;  // token i spans bytes [start, end)
        bool selected = false;
        for (const auto& sp : byte_spans)
            if (start < sp.second && end > sp.first) { selected = true; break; }
        if (!selected && prev_selected)
            for (int g = 0; g < grow; ++g) board.push_back(mask_token);  // headroom after a selected run
        board.push_back(selected ? mask_token : toks[i]);
        prev_selected = selected;
    }
    if (prev_selected)
        for (int g = 0; g < grow; ++g) board.push_back(mask_token);  // selection ran to the end
    return board;
}

// SSE payload for the revise stream. gen_started is enriched with `layout`: one entry per board
// position ({pos, masked, piece}) so the browser can render the full interleaved board — fixed
// tokens as text, masked tokens as empty slots — without a tokenizer. Other events match sse_data.
std::string sse_data_revise(const Event& e, const GgmlModel& model, const std::vector<int>& board,
                            int mask_token) {
    if (const auto* gs = std::get_if<GenStarted>(&e)) {
        json layout = json::array();
        for (size_t pos = 0; pos < board.size(); ++pos) {
            const bool masked = board[pos] == mask_token;
            layout.push_back({{"pos", static_cast<int>(pos)}, {"masked", masked},
                              {"piece", masked ? std::string() : model.decode({board[pos]})}});
        }
        return json{{"t", gs->t}, {"type", "gen_started"}, {"prompt_tokens", gs->prompt_tokens},
                    {"block_len", gs->block_len}, {"max_new", gs->max_new}, {"layout", layout}}.dump();
    }
    if (const auto* tc = std::get_if<TokensCommitted>(&e)) {
        json items = json::array();
        for (const auto& it : tc->items)
            items.push_back({{"pos", it.pos}, {"id", it.id}, {"conf", it.conf},
                             {"piece", model.decode({it.id})}});
        return dump_json(json{{"t", tc->t}, {"type", "tokens_committed"}, {"block", tc->block},
                    {"items", items}});
    }
    if (const auto* sl = std::get_if<StepLens>(&e)) {
        json pieces = json::array();
        for (int id : sl->ids) pieces.push_back(model.decode({id}));     // decode candidates (viz has no tokenizer)
        return dump_json(json{{"t", sl->t}, {"type", "step_lens"}, {"block", sl->block}, {"k", sl->k},
                    {"positions", sl->positions}, {"ids", sl->ids}, {"probs", sl->probs},
                    {"pieces", pieces}});
    }
    return to_jsonl_line(e);
}

// ============================ state-stream protocol (phase 1.2) ============================
// The inspector-facing wire form (protocol/SPEC.md). The engine's §5.1 events FOLD into canonical
// StateStep frames: {step, token, state, readouts, meta}. One forward pass's events (which all share
// the same `t`) collapse into one StateStep; gen_started/finished become control frames. Gated by the
// request flag protocol:true — without it, the legacy sse_data(...) frames stream unchanged, so the
// existing viz keeps working. See SPEC.md "Engine §5.1 event -> StateStep mapping" + "The wire".

// Base64 (standard alphabet, padded) of raw bytes — for tensor `data` on the wire.
std::string base64_encode(const uint8_t* data, size_t len) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 3 <= len; i += 3) {
        const uint32_t n = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | data[i + 2];
        out.push_back(tbl[(n >> 18) & 63]); out.push_back(tbl[(n >> 12) & 63]);
        out.push_back(tbl[(n >> 6) & 63]);  out.push_back(tbl[n & 63]);
    }
    if (i < len) {  // 1 or 2 trailing bytes
        uint32_t n = uint32_t(data[i]) << 16;
        const bool two = (i + 1 < len);
        if (two) n |= uint32_t(data[i + 1]) << 8;
        out.push_back(tbl[(n >> 18) & 63]);
        out.push_back(tbl[(n >> 12) & 63]);
        out.push_back(two ? tbl[(n >> 6) & 63] : '=');
        out.push_back('=');
    }
    return out;
}

// A tensor on the wire: {dtype, shape, data} where data = base64 of the little-endian raw bytes
// (SPEC.md "Tensors on the wire"). We tap float32 activations; x86/CUDA are little-endian, so the
// in-memory floats ARE the little-endian bytes — a straight reinterpret, no byte-swizzling.
json tensor_json_f32(const std::vector<float>& values, std::vector<int> shape) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
    return json{{"dtype", "float32"}, {"shape", std::move(shape)},
                {"data", base64_encode(bytes, values.size() * sizeof(float))}};
}

// Folds the §5.1 event stream into StateStep frames and writes them to an SSE sink. One StateStep
// per forward pass (events sharing a `t`); gen_started/finished + block start/finalize fold into the
// running `meta`. `state_full` controls whether the heavy raw-activation tensor rides each frame
// (state="full") or is omitted (the light default). The builder is stateful across the run: it
// accumulates the current pass, and FLUSHES it (emits one frame) when the next pass begins (a new
// tokens_committed at a higher t) or at gen_finished.
class StateStepBuilder {
public:
    StateStepBuilder(const GgmlModel& model, const char* substrate, bool state_full,
                     std::function<void(const std::string&)> write)
        : model_(model), substrate_(substrate), state_full_(state_full), write_(std::move(write)) {}

    void on_event(const Event& e) {
        if (const auto* gs = std::get_if<GenStarted>(&e)) {
            // Control frame: stream begin. prompt/total counts in meta.
            json meta{{"kind", "begin"}, {"substrate", substrate_}, {"block_len", gs->block_len},
                      {"prompt_tokens", gs->prompt_tokens}, {"max_new", gs->max_new}};
            emit_control(gs->t, meta);
            return;
        }
        if (const auto* bs = std::get_if<BlockStarted>(&e)) {
            block_ = bs->block;
            span_ = json::array({bs->span.first, bs->span.second});
            return;
        }
        if (const auto* tr = std::get_if<TokensRevised>(&e)) {
            // A revision is its own StateStep (diffusion-only). Flush any open commit first so the
            // revise frame is distinct, then emit the revise frame immediately.
            flush();
            json items = json::array();
            for (const auto& it : tr->items)
                items.push_back({{"pos", it.pos}, {"old", it.old}, {"id", it.id}, {"conf", it.conf},
                                 {"piece", model_.decode({it.id})}});
            json meta = base_meta();
            meta["kind"] = "revise";
            json frame{{"step", tr->t}, {"token", nullptr}, {"state", nullptr},
                       {"readouts", json::array()}, {"meta", meta}, {"revised", items}};
            write_frame(frame);
            return;
        }
        if (const auto* tc = std::get_if<TokensCommitted>(&e)) {
            flush();             // a new pass begins -> close out the previous StateStep
            open_ = true;
            step_ = tc->t;
            // token = the committed id(s) for this pass (+ pieces for the tokenizer-free consumer).
            token_ = json::array();
            json confs = json::array();
            for (const auto& it : tc->items) {
                token_.push_back({{"pos", it.pos}, {"id", it.id}, {"piece", model_.decode({it.id})}});
                confs.push_back(it.conf);
            }
            commit_confs_ = std::move(confs);
            return;
        }
        if (const auto* ss = std::get_if<StepStats>(&e)) {
            ensure_open(ss->t);
            stats_ = json{{"committed", ss->committed}, {"remaining", ss->remaining},
                          {"step", ss->step}, {"ms", ss->ms}, {"cache_hit", ss->cache_hit}};
            return;
        }
        if (const auto* sf = std::get_if<StepFeatures>(&e)) {
            ensure_open(sf->t);
            // One Readout per concept. value = the per-slot scores for that concept (position-major
            // sliced into a per-concept vector). Honesty invariant: confidence travels; causal_verified
            // is null (probes are correlational until patched — SPEC's wire honesty rule).
            const int K = static_cast<int>(sf->features.size());
            const int rows = K > 0 ? static_cast<int>(sf->scores.size()) / K : 0;
            for (int k = 0; k < K; ++k) {
                json per_slot = json::array();
                double maxabs = 0.0;
                for (int r = 0; r < rows; ++r) {
                    const float v = sf->scores[static_cast<size_t>(r) * K + k];
                    per_slot.push_back(v);
                    if (std::fabs(v) > maxabs) maxabs = std::fabs(v);
                }
                readouts_.push_back({{"name", sf->features[k]},
                                     {"value", {{"positions", sf->positions}, {"scores", per_slot}}},
                                     {"confidence", maxabs}, {"causal_verified", nullptr}});
            }
            return;
        }
        if (const auto* sl = std::get_if<StepLens>(&e)) {
            ensure_open(sl->t);
            // The logit-lens readout: top-k candidates+probs per requested slot. value carries the
            // decoded pieces too (the consumer has no tokenizer). confidence = the top-1 prob seen.
            json per_pos = json::array();
            double top_conf = 0.0;
            const int k = sl->k;
            for (size_t r = 0; r < sl->positions.size(); ++r) {
                json cand = json::array();
                for (int j = 0; j < k; ++j) {
                    const size_t idx = r * static_cast<size_t>(k) + j;
                    if (idx >= sl->ids.size()) break;
                    const float prob = sl->probs[idx];
                    if (j == 0 && prob > top_conf) top_conf = prob;
                    cand.push_back({{"id", sl->ids[idx]}, {"prob", prob},
                                    {"piece", model_.decode({sl->ids[idx]})}});
                }
                per_pos.push_back({{"pos", sl->positions[r]}, {"candidates", cand}});
            }
            readouts_.push_back({{"name", "logit-lens"}, {"value", per_pos},
                                 {"confidence", top_conf}, {"causal_verified", nullptr}});
            return;
        }
        if (const auto* sa = std::get_if<StepActivations>(&e)) {
            ensure_open(sa->t);
            // The heavy state: raw per-position hidden state. Held; attached to the frame on flush()
            // only when state="full" (the light frame omits it). Encoded {dtype,shape,data}.
            if (state_full_) {
                state_ = json::object();
                state_["positions"] = sa->positions;
                state_["hidden"] = tensor_json_f32(
                    sa->values, {static_cast<int>(sa->positions.size()), sa->n_embd});
            }
            return;
        }
        if (const auto* gf = std::get_if<GenFinished>(&e)) {
            flush();  // close the last pass before the end-of-stream control frame
            json meta{{"kind", "end"}, {"substrate", substrate_}, {"reason", gf->reason},
                      {"new_tokens", gf->new_tokens}, {"wall_ms", gf->wall_ms},
                      {"steps_total", gf->steps_total}, {"tok_per_s", gf->tok_per_s}};
            emit_control(gf->t, meta);
            return;
        }
    }

    // Flush any pass still open (defensive; gen_finished normally does this).
    void finish() { flush(); }

private:
    json base_meta() const {
        json m{{"substrate", substrate_}, {"block", block_}};
        if (!span_.is_null()) m["span"] = span_;
        return m;
    }
    void ensure_open(int t) {
        if (!open_) { open_ = true; step_ = t; }
    }
    void emit_control(int step, const json& meta) {
        json frame{{"step", step}, {"token", nullptr}, {"state", nullptr},
                   {"readouts", json::array()}, {"meta", meta}};
        write_frame(frame);
    }
    void write_frame(const json& frame) { write_("data: " + dump_json(frame) + "\n\n"); }

    // Emit the accumulated pass as one StateStep, then reset for the next pass.
    void flush() {
        if (!open_) return;
        json meta = base_meta();
        meta["kind"] = "step";
        if (!stats_.is_null()) meta.update(stats_);
        if (!commit_confs_.is_null()) meta["confidence"] = commit_confs_;
        json frame{{"step", step_},
                   {"token", token_.is_null() ? json(json::array()) : token_},
                   {"state", state_.is_null() ? json(nullptr) : state_},
                   {"readouts", readouts_},
                   {"meta", meta}};
        write_frame(frame);
        // reset pass accumulators
        open_ = false;
        token_ = json();
        state_ = json();
        stats_ = json();
        commit_confs_ = json();
        readouts_ = json::array();
    }

    const GgmlModel& model_;
    const char* substrate_;
    bool state_full_;
    std::function<void(const std::string&)> write_;

    // run-scoped
    int block_ = 0;
    json span_ = json();

    // pass-scoped
    bool open_ = false;
    int step_ = 0;
    json token_ = json();
    json state_ = json();
    json stats_ = json();
    json commit_confs_ = json();
    json readouts_ = json::array();
};

// Per-position board layout for the white-box SNAPSHOT: {pos,id,masked,piece}. Lets a client save
// the exact board state from any response and POST it back to /v1/board to restore/branch.
json board_layout_json(const GgmlModel& model, const std::vector<int>& board, int mask_token) {
    json layout = json::array();
    for (size_t pos = 0; pos < board.size(); ++pos) {
        const bool masked = board[pos] == mask_token;
        layout.push_back({{"pos", static_cast<int>(pos)}, {"id", board[pos]}, {"masked", masked},
                          {"piece", masked ? std::string() : model.decode({board[pos]})}});
    }
    return layout;
}

// ---- on-device SAE feature readout (--sae <dir>; ROADMAP 3.3 wired into the server) ----
// When active, every featureful request taps the residual at the SAE's OWN layer, encodes the
// tapped rows on the GPU (cloze/sae.hpp: JumpReLU GEMV + the sae_topk kernel) and rides each pass's
// top-k features onto the stream as a SECOND StepFeatures event whose names are raw feature indices
// ("sae:<id>") — the same positions-x-features wire shape the concept probes already use, so
// StateStepBuilder / the inspector parse it unchanged. The Neuronpedia id -> label mapping stays
// host/Python side (research/np_labels_l15.json via brain_readout.py), by design: the engine ships
// indices, never a 131k-entry string table. The holder exists in every build so the run lambdas
// capture it uniformly; without CLOZE_BUILD_SAE it is permanently off and --sae refuses at startup.
struct SaeServe {
#ifdef CLOZE_SAE
    SaeEncoder enc;   // the device-resident encoder weights (loaded once at startup)
#endif
    bool on = false;  // loaded + n_embd == d_in verified; readouts ride featureful requests
    int layer = 0;    // the SAE's residual layer — the read tap moves here when on
    int k = 16;       // features kept per position (--sae-k)
};

#ifdef CLOZE_SAE
// One pass's SAE readout: encode the raw-activation event's rows (chunked so the encoder workspace
// stays bounded at ~16 MB regardless of diffusion block size) and fold every row's top-k into one
// StepFeatures over the UNION of lit features (score 0 where a feature missed a row's top-k).
// nullopt when nothing lit / dims mismatch / CUDA failure — the pass just has no SAE readout.
std::optional<StepFeatures> sae_features_from(const StepActivations& sa, SaeEncoder& enc, int k) {
    const int rows = static_cast<int>(sa.positions.size());
    if (rows <= 0 || sa.n_embd != enc.d_in()) return std::nullopt;
    constexpr int kRowChunk = 32;  // 0.5 MB/row of device workspace; 32 caps it while covering blocks
    std::vector<int32_t> idx;
    std::vector<float> val;
    idx.reserve(static_cast<size_t>(rows) * k);
    val.reserve(static_cast<size_t>(rows) * k);
    for (int r0 = 0; r0 < rows; r0 += kRowChunk) {
        const int rn = rows - r0 < kRowChunk ? rows - r0 : kRowChunk;
        std::vector<int32_t> ci;
        std::vector<float> cv;
        if (!enc.encode_topk(sa.values.data() + static_cast<size_t>(r0) * sa.n_embd, rn, k, ci, cv))
            return std::nullopt;
        idx.insert(idx.end(), ci.begin(), ci.end());
        val.insert(val.end(), cv.begin(), cv.end());
    }
    // Union of live features (value > 0; zero-valued slots are the top-k pad), ascending ids.
    std::vector<int32_t> feats;
    for (size_t i = 0; i < idx.size(); ++i)
        if (val[i] > 0.0f) feats.push_back(idx[i]);
    std::sort(feats.begin(), feats.end());
    feats.erase(std::unique(feats.begin(), feats.end()), feats.end());
    if (feats.empty()) return std::nullopt;

    StepFeatures sf;
    sf.t = sa.t;
    sf.block = sa.block;
    sf.positions = sa.positions;
    sf.features.reserve(feats.size());
    for (int32_t f : feats) sf.features.push_back("sae:" + std::to_string(f));
    sf.scores.assign(static_cast<size_t>(rows) * feats.size(), 0.0f);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < k; ++j) {
            const size_t i = static_cast<size_t>(r) * k + j;
            if (val[i] <= 0.0f) continue;
            const size_t c = std::lower_bound(feats.begin(), feats.end(), idx[i]) - feats.begin();
            sf.scores[static_cast<size_t>(r) * feats.size() + c] = val[i];
        }
    return sf;
}
#endif  // CLOZE_SAE

// Wrap a run's event sink so each StepActivations (emitted whenever the white-box tap is on) also
// yields its SAE readout, BEFORE the activations event itself — both land in the same StateStep.
// Identity when inactive (or in a CLOZE_SAE-less build), so the plain path is byte-identical.
std::function<void(const Event&)> with_sae_readout(const std::function<void(const Event&)>& on_event,
                                                   SaeServe& sae, bool active) {
#ifdef CLOZE_SAE
    if (active && sae.on && on_event) {
        SaeEncoder* enc = &sae.enc;
        const int k = sae.k;
        return [on_event, enc, k](const Event& e) {
            if (const auto* sa = std::get_if<StepActivations>(&e))
                if (auto sf = sae_features_from(*sa, *enc, k)) on_event(Event(*sf));
            on_event(e);
        };
    }
#else
    (void)sae;
    (void)active;
#endif
    return on_event;
}

// ---- white-box concept probe calibration (Tier 2) ----
// Categorize a decoded token piece: "number" (has a digit), "punct" (no alphanumerics), "word"
// (all letters), or nullptr (mixed / unknown). Mirrors lab p4_dream_probe.category. ASCII-only
// (English calibration text); non-ASCII bytes just fall through to nullptr, which is fine here.
const char* token_category(const std::string& piece) {
    std::string s, low;
    for (unsigned char c : piece)
        if (!std::isspace(c)) { s.push_back(static_cast<char>(c)); low.push_back(static_cast<char>(std::tolower(c))); }
    if (s.empty()) return nullptr;
    bool any_digit = false, any_alnum = false, all_alpha = true;
    for (unsigned char c : s) {
        if (std::isdigit(c)) any_digit = true;
        if (std::isalnum(c)) any_alnum = true;
        if (!std::isalpha(c)) all_alpha = false;
    }
    if (any_digit) return "number";
    if (!any_alnum) return "punct";
    if (all_alpha) {
        // closed-class function words vs open-class content words — the syntactic-role split.
        static const std::set<std::string> kFunction = {
            "the","a","an","of","and","to","in","is","it","that","this","for","on","with","as","at",
            "by","or","but","not","are","was","were","be","been","being","i","you","he","she","they",
            "we","his","her","its","their","our","my","your","from","up","out","if","then","than","so",
            "no","do","does","did","have","has","had","will","would","can","could","should","may","me"
        };
        return kFunction.count(low) ? "function" : "content";
    }
    return nullptr;
}

// Build training-free diff-in-means category probes in the model's OWN mid-layer activation
// space: run a small labeled corpus through the adapter (which must have emit_activations on),
// standardize, and for each category take (its mean - the pooled mean of the other categories),
// unit-normalized. Self-contained — no lab->core probe transfer (the activation spaces differ).
//
// Two families of probes are built in one pass:
//   (1) PER-TOKEN: punct/number/function/content — each token is categorized individually.
//   (2) CONTRASTIVE (sentence-level): code/question — all tokens in a positive sentence contribute
//       to the positive class, all tokens in a negative sentence to the negative class. This
//       captures sentence-level concepts that don't reduce to per-token labels.
ConceptProbes calibrate_concept_probes(GgmlAdapter& ad, const GgmlModel& model) {
    // --- per-token calibration corpus (existing) ---
    static const std::vector<std::string> corpus = {
        "The quick brown fox jumps over the lazy dog near the old stone bridge.",
        "In 2024 the company sold 15 boats, 320 bikes, and 7 cars to 4 buyers.",
        "She said, \"Hello!\" Then he asked: why now? Nobody really knew.",
        "Pi is about 3.14159 and the speed of light is 299792458 meters per second.",
        "Prices fell 12 percent in March, rose 8 percent in April, and held in May.",
        "Wait, stop -- listen carefully; the answer matters more than the question.",
    };
    static const std::vector<std::string> cats = {"punct", "number", "function", "content"};

    // --- contrastive calibration pairs: {positive_sentences, negative_sentences, name} ---
    struct ContrastivePair {
        std::vector<std::string> pos;
        std::vector<std::string> neg;
        std::string name;
    };
    static const std::vector<ContrastivePair> contrastive = {
        {   // code vs prose
            {"def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
             "for i in range(10): result = data[i] * weights[i] + bias",
             "int main(int argc, char** argv) { return process(argc, argv); }"},
            {"The sun set behind the mountains as the birds flew south for the winter.",
             "She walked to the park and sat on the bench near the fountain.",
             "The report concluded that revenue grew steadily throughout the quarter."},
            "code"
        },
        {   // question vs statement
            {"What is the meaning of life and why does it matter?",
             "How many people live in Tokyo and what languages do they speak?",
             "Why did the experiment fail and what should we change next time?"},
            {"The meaning of life is a deeply personal question with many answers.",
             "About fourteen million people live in Tokyo and most speak Japanese.",
             "The experiment failed because the sample size was too small."},
            "question"
        },
    };

    int n_embd = 0;
    long long N = 0;
    std::vector<double> sum, sumsq;
    std::vector<std::vector<double>> cat_sum(cats.size());
    std::vector<long long> cat_cnt(cats.size(), 0);

    // Contrastive accumulators — sentence-level: all tokens in a pos sentence -> pos, neg -> neg.
    std::vector<std::vector<double>> con_pos_sum(contrastive.size());
    std::vector<std::vector<double>> con_neg_sum(contrastive.size());
    std::vector<long long> con_pos_cnt(contrastive.size(), 0);
    std::vector<long long> con_neg_cnt(contrastive.size(), 0);

    auto run_sentence = [&](const std::string& text) -> std::pair<std::vector<float>, size_t> {
        const std::vector<int> toks = model.encode(text);
        if (toks.size() < 2) return {{}, 0};
        const Mask m = attention_mask(static_cast<int>(toks.size()), 0, 0);
        const ForwardResult fwd = ad.forward(toks, m, nullptr, std::nullopt, {});
        if (fwd.activations.empty() || fwd.n_embd <= 0) return {{}, 0};
        if (n_embd == 0) {
            n_embd = fwd.n_embd;
            sum.assign(n_embd, 0.0);
            sumsq.assign(n_embd, 0.0);
            for (auto& cs : cat_sum) cs.assign(n_embd, 0.0);
            for (auto& cs : con_pos_sum) cs.assign(n_embd, 0.0);
            for (auto& cs : con_neg_sum) cs.assign(n_embd, 0.0);
        }
        return {fwd.activations, fwd.act_rows.size()};
    };

    // Pass 1: per-token corpus — accumulate per-category means + global stats.
    for (const std::string& text : corpus) {
        const std::vector<int> toks = model.encode(text);
        if (toks.size() < 2) continue;
        const Mask m = attention_mask(static_cast<int>(toks.size()), 0, 0);
        const ForwardResult fwd = ad.forward(toks, m, nullptr, std::nullopt, {});
        if (fwd.activations.empty() || fwd.n_embd <= 0) continue;
        if (n_embd == 0) {
            n_embd = fwd.n_embd;
            sum.assign(n_embd, 0.0);
            sumsq.assign(n_embd, 0.0);
            for (auto& cs : cat_sum) cs.assign(n_embd, 0.0);
            for (auto& cs : con_pos_sum) cs.assign(n_embd, 0.0);
            for (auto& cs : con_neg_sum) cs.assign(n_embd, 0.0);
        }
        for (size_t r = 0; r < fwd.act_rows.size(); ++r) {
            const float* h = fwd.activations.data() + r * static_cast<size_t>(n_embd);
            for (int i = 0; i < n_embd; ++i) { sum[i] += h[i]; sumsq[i] += static_cast<double>(h[i]) * h[i]; }
            ++N;
            const char* c = token_category(model.decode({toks[fwd.act_rows[r]]}));
            if (!c) continue;
            for (size_t ci = 0; ci < cats.size(); ++ci)
                if (cats[ci] == c) {
                    for (int i = 0; i < n_embd; ++i) cat_sum[ci][i] += h[i];
                    ++cat_cnt[ci];
                    break;
                }
        }
    }

    // Pass 2: contrastive corpus — accumulate sentence-level pos/neg means (all tokens contribute).
    for (size_t ci = 0; ci < contrastive.size(); ++ci) {
        for (const std::string& text : contrastive[ci].pos) {
            const std::vector<int> toks = model.encode(text);
            if (toks.size() < 2) continue;
            const Mask m = attention_mask(static_cast<int>(toks.size()), 0, 0);
            const ForwardResult fwd = ad.forward(toks, m, nullptr, std::nullopt, {});
            if (fwd.activations.empty()) continue;
            for (size_t r = 0; r < fwd.act_rows.size(); ++r) {
                const float* h = fwd.activations.data() + r * static_cast<size_t>(n_embd);
                for (int i = 0; i < n_embd; ++i) { sum[i] += h[i]; sumsq[i] += static_cast<double>(h[i]) * h[i]; }
                ++N;
                for (int i = 0; i < n_embd; ++i) con_pos_sum[ci][i] += h[i];
                ++con_pos_cnt[ci];
            }
        }
        for (const std::string& text : contrastive[ci].neg) {
            const std::vector<int> toks = model.encode(text);
            if (toks.size() < 2) continue;
            const Mask m = attention_mask(static_cast<int>(toks.size()), 0, 0);
            const ForwardResult fwd = ad.forward(toks, m, nullptr, std::nullopt, {});
            if (fwd.activations.empty()) continue;
            for (size_t r = 0; r < fwd.act_rows.size(); ++r) {
                const float* h = fwd.activations.data() + r * static_cast<size_t>(n_embd);
                for (int i = 0; i < n_embd; ++i) { sum[i] += h[i]; sumsq[i] += static_cast<double>(h[i]) * h[i]; }
                ++N;
                for (int i = 0; i < n_embd; ++i) con_neg_sum[ci][i] += h[i];
                ++con_neg_cnt[ci];
            }
        }
    }

    ConceptProbes p;
    if (n_embd == 0 || N == 0) return p;
    p.n_embd = n_embd;
    p.mean.resize(n_embd);
    p.inv_std.resize(n_embd);
    for (int i = 0; i < n_embd; ++i) {
        const double mu = sum[i] / static_cast<double>(N);
        const double var = sumsq[i] / static_cast<double>(N) - mu * mu;
        p.mean[i] = static_cast<float>(mu);
        p.inv_std[i] = static_cast<float>(1.0 / std::sqrt(var > 1e-12 ? var : 1e-12));
    }

    // Per-token probes: diff-in-means (each category vs pooled rest).
    for (size_t ci = 0; ci < cats.size(); ++ci) {
        if (cat_cnt[ci] < 3) continue;
        long long neg_cnt = 0;
        std::vector<double> neg_sum(n_embd, 0.0);
        for (size_t cj = 0; cj < cats.size(); ++cj)
            if (cj != ci && cat_cnt[cj] > 0) {
                neg_cnt += cat_cnt[cj];
                for (int i = 0; i < n_embd; ++i) neg_sum[i] += cat_sum[cj][i];
            }
        if (neg_cnt == 0) continue;
        std::vector<float> dir(n_embd);
        double norm = 0.0;
        for (int i = 0; i < n_embd; ++i) {
            const double pos_std = (cat_sum[ci][i] / static_cast<double>(cat_cnt[ci]) - p.mean[i]) * p.inv_std[i];
            const double neg_std = (neg_sum[i] / static_cast<double>(neg_cnt) - p.mean[i]) * p.inv_std[i];
            const double d = pos_std - neg_std;
            dir[i] = static_cast<float>(d);
            norm += d * d;
        }
        norm = std::sqrt(norm > 1e-12 ? norm : 1e-12);
        for (int i = 0; i < n_embd; ++i) dir[i] = static_cast<float>(dir[i] / norm);
        p.names.push_back(cats[ci]);
        p.dirs.insert(p.dirs.end(), dir.begin(), dir.end());
    }

    // Contrastive probes: sentence-level diff-in-means (pos vs neg class).
    for (size_t ci = 0; ci < contrastive.size(); ++ci) {
        if (con_pos_cnt[ci] < 3 || con_neg_cnt[ci] < 3) continue;
        std::vector<float> dir(n_embd);
        double norm = 0.0;
        for (int i = 0; i < n_embd; ++i) {
            const double pos_std = (con_pos_sum[ci][i] / static_cast<double>(con_pos_cnt[ci]) - p.mean[i]) * p.inv_std[i];
            const double neg_std = (con_neg_sum[ci][i] / static_cast<double>(con_neg_cnt[ci]) - p.mean[i]) * p.inv_std[i];
            const double d = pos_std - neg_std;
            dir[i] = static_cast<float>(d);
            norm += d * d;
        }
        norm = std::sqrt(norm > 1e-12 ? norm : 1e-12);
        for (int i = 0; i < n_embd; ++i) dir[i] = static_cast<float>(dir[i] / norm);
        p.names.push_back(contrastive[ci].name);
        p.dirs.insert(p.dirs.end(), dir.begin(), dir.end());
    }

    return p;
}

// Build a llama control-vector buffer (n_embd*n_layer, layer-1-indexed) that ADDS coef * a concept's
// raw steering direction at every layer in [lo, hi] (other layers zero). Empty if the concept isn't
// found. NOTE: llama only has cvec tensors for layers 1..n_layer-1 (llama-adapter.cpp — no tensor for
// the last layer), so callers must keep hi <= n_layer-1. Pairs with GgmlAdapter::set_steer.
std::vector<float> build_steer_cvec(const ConceptProbes& p, const std::string& concept,
                                    double coef, int lo, int hi, int n_layer) {
    int idx = -1;
    for (int i = 0; i < p.size(); ++i) if (p.names[i] == concept) { idx = i; break; }
    if (idx < 0 || p.n_embd <= 0 || n_layer <= 0) return {};
    const std::vector<float> v = p.steer_vector(idx);
    std::vector<float> data(static_cast<size_t>(p.n_embd) * n_layer, 0.0f);
    for (int L = lo; L <= hi; ++L) {
        if (L < 1 || L >= n_layer) continue;  // valid cvec layers are 1..n_layer-1
        float* slice = data.data() + static_cast<size_t>(L - 1) * p.n_embd;
        for (int i = 0; i < p.n_embd; ++i) slice[i] = static_cast<float>(coef * v[i]);
    }
    return data;
}

// ---- J-lens readout (JLENS_ENGINE_PLAN.md J2) --------------------------------------------------------
// lens_l(h) = output.weight @ rmsnorm(J_l @ h): a fitted Jacobian J_l [d_model x d_model] f16 transports a
// layer-l residual into the FINAL-layer basis, then the model's OWN final RMSNorm + output head unembed it
// -> per-position top-k "disposed to say" tokens (Anthropic's J-lens, transferred to the GGUF engine in
// J0/J1). The J_l matrices are fitted sidecars (config dir, out of git, treated like model weights); the
// final_norm + head are read straight from the loaded GGUF (so the head is the model's own quantized q6_K
// lm_head -> expect drift vs the fp32 numpy oracle, tolerance not bitwise). The apply is a small standalone
// CPU ggml graph (the 7B stays on the GPU); weights are loaded once at startup, mirroring the SAE sidecar.
struct JlensServe {
    bool on = false;
    std::string err;                            // last load failure, human-readable
    int d_model = 0, vocab = 0;
    int default_layer = 0;                      // manifest engine_default_tap_layer (the /jlens default)
    float eps = 1e-6f;                          // final RMSNorm epsilon (read from the GGUF; Qwen2.5 = 1e-6)
    struct ggml_context* wctx = nullptr;        // persistent: output_norm, output.weight, and each J_l
    struct ggml_tensor* out_norm = nullptr;     // F32 [d_model]
    struct ggml_tensor* out_head = nullptr;     // the GGUF's own lm_head, quantized [d_model, vocab]
    std::map<int, struct ggml_tensor*> Jl;      // layer -> F16 [d_model, d_model] (ggml layout Jl(k,m)=J[m,k])
    std::mutex mtx;                             // a readout builds a transient graph; serialize (on-demand)

    JlensServe() = default;
    JlensServe(const JlensServe&) = delete;
    JlensServe& operator=(const JlensServe&) = delete;
    ~JlensServe() { if (wctx) ggml_free(wctx); }

    std::vector<int> layers() const {
        std::vector<int> v;
        for (const auto& kv : Jl) v.push_back(kv.first);
        return v;
    }
    bool has(int layer) const { return Jl.count(layer) > 0; }

    // Read `sz` bytes at absolute file offset `off` into `dst` (64-bit seek: the head lives past 2 GB).
    static bool read_at(std::ifstream& f, void* dst, size_t off, size_t sz) {
        f.clear();
        f.seekg(static_cast<std::streamoff>(off), std::ios::beg);
        f.read(static_cast<char*>(dst), static_cast<std::streamsize>(sz));
        return static_cast<size_t>(f.gcount()) == sz;
    }

    // Load the J_l sidecars from `dir` (+ manifest.json) and the GGUF's OWN final_norm/head from
    // `model_path`. n_embd/n_vocab come from the loaded model. Returns false + sets err on any failure.
    bool load(const std::string& dir, const std::string& model_path, int n_embd, int n_vocab) {
        d_model = n_embd;
        vocab = n_vocab;
        std::vector<int> want_layers;
        {   // manifest: which layers + the default tap + a d_model sanity check.
            std::ifstream mf(dir + "/manifest.json", std::ios::binary);
            if (!mf) { err = "no manifest.json in " + dir; return false; }
            json m;
            try { mf >> m; } catch (...) { err = "manifest.json parse failed"; return false; }
            if (m.contains("d_model") && m["d_model"].is_number() && m["d_model"].get<int>() != d_model) {
                err = "manifest d_model " + std::to_string(m["d_model"].get<int>()) +
                      " != model n_embd " + std::to_string(d_model);
                return false;
            }
            if (m.contains("layers") && m["layers"].is_array())
                for (const auto& l : m["layers"]) if (l.is_number_integer()) want_layers.push_back(l.get<int>());
            default_layer = m.value("engine_default_tap_layer", want_layers.empty() ? 0 : want_layers.front());
        }
        if (want_layers.empty()) { err = "manifest lists no layers"; return false; }

        // Read output_norm.weight + output.weight straight from the GGUF (no_alloc meta -> targeted file read;
        // never loads the whole 4.7 GB file). The head is the model's OWN quantized lm_head.
        struct ggml_context* meta = nullptr;
        struct gguf_init_params gp; gp.no_alloc = true; gp.ctx = &meta;
        struct gguf_context* gg = gguf_init_from_file(model_path.c_str(), gp);
        if (!gg || !meta) { err = "gguf_init_from_file failed (head read)"; if (gg) gguf_free(gg); return false; }
        {   // eps from the model's own metadata (matches the oracle's 1e-6 for Qwen2.5); default keeps 1e-6.
            int64_t k = gguf_find_key(gg, "qwen2.attention.layer_norm_rms_epsilon");
            if (k >= 0) eps = gguf_get_val_f32(gg, k);
        }
        // Untied head is "output.weight"; a tied-embedding model has none -> fall back to token_embd.weight.
        std::string head_name = ggml_get_tensor(meta, "output.weight") ? "output.weight" : "token_embd.weight";
        struct ggml_tensor* mo = ggml_get_tensor(meta, head_name.c_str());
        struct ggml_tensor* mn = ggml_get_tensor(meta, "output_norm.weight");
        if (!mo || !mn) { err = "GGUF missing output(.weight)/output_norm.weight"; ggml_free(meta); gguf_free(gg); return false; }
        const int64_t oid = gguf_find_tensor(gg, head_name.c_str());
        const int64_t nid = gguf_find_tensor(gg, "output_norm.weight");
        const size_t data_off = gguf_get_data_offset(gg);
        const size_t o_off = data_off + gguf_get_tensor_offset(gg, oid), o_sz = gguf_get_tensor_size(gg, oid);
        const size_t n_off = data_off + gguf_get_tensor_offset(gg, nid), n_sz = gguf_get_tensor_size(gg, nid);
        const enum ggml_type o_type = mo->type;  const int64_t o_ne0 = mo->ne[0], o_ne1 = mo->ne[1];
        const enum ggml_type n_type = mn->type;  const int64_t n_ne0 = mn->ne[0];
        ggml_free(meta); gguf_free(gg);
        if (o_ne0 != d_model || o_ne1 != vocab) { err = "head shape mismatch vs model"; return false; }

        // Persistent weight context sized to hold the head + norm + every J_l.
        const size_t j_bytes = static_cast<size_t>(d_model) * d_model * sizeof(uint16_t);   // f16
        const size_t mem = o_sz + n_sz + j_bytes * want_layers.size()
                         + (want_layers.size() + 4) * ggml_tensor_overhead() + (1ULL << 20);
        struct ggml_init_params ip; ip.mem_size = mem; ip.mem_buffer = nullptr; ip.no_alloc = false;
        wctx = ggml_init(ip);
        if (!wctx) { err = "ggml_init(jlens weights) failed"; return false; }

        std::ifstream mfile(model_path, std::ios::binary);
        if (!mfile) { err = "cannot reopen model file for head read"; return false; }
        out_head = ggml_new_tensor_2d(wctx, o_type, o_ne0, o_ne1);
        out_norm = ggml_new_tensor_1d(wctx, n_type, n_ne0);
        if (!read_at(mfile, out_head->data, o_off, o_sz) || !read_at(mfile, out_norm->data, n_off, n_sz)) {
            err = "short read on the GGUF head/norm"; return false;
        }
        for (int L : want_layers) {
            const std::string path = dir + "/J_layer" + std::to_string(L) + ".f16";
            std::ifstream jf(path, std::ios::binary);
            if (!jf) { err = "missing sidecar " + path; return false; }
            struct ggml_tensor* J = ggml_new_tensor_2d(wctx, GGML_TYPE_F16, d_model, d_model);
            jf.read(static_cast<char*>(J->data), static_cast<std::streamsize>(j_bytes));
            if (static_cast<size_t>(jf.gcount()) != j_bytes) { err = "short read on " + path; return false; }
            Jl[L] = J;
        }
        on = true;
        return true;
    }

    // Per-position top-k of lens_l(h). h = [n_tokens * d_model] host f32 (harvest layout, position-major).
    // Fills out[pos] = up to `topk` (token_id, logit) descending. Serialized (transient graph per call).
    bool readout(const float* h, int n_tokens, int layer, int topk,
                 std::vector<std::vector<std::pair<int, float>>>& out, std::string& e) {
        if (!on) { e = "jlens not loaded"; return false; }
        auto it = Jl.find(layer);
        if (it == Jl.end()) { e = "no J-lens sidecar for layer " + std::to_string(layer); return false; }
        if (n_tokens <= 0) { e = "no tokens"; return false; }
        struct ggml_tensor* J = it->second;
        std::lock_guard<std::mutex> lk(mtx);

        // Transient CPU graph context: leaves (h) + intermediates (hJ, normed, logits) + work buffer.
        const size_t tens = static_cast<size_t>((3 * d_model) + vocab) * n_tokens * sizeof(float);
        const size_t mem = tens + tens / 2 + ggml_graph_overhead() + 32 * ggml_tensor_overhead() + (128ULL << 20);
        struct ggml_init_params ip; ip.mem_size = mem; ip.mem_buffer = nullptr; ip.no_alloc = false;
        struct ggml_context* ctx = ggml_init(ip);
        if (!ctx) { e = "ggml_init(jlens graph) failed"; return false; }

        struct ggml_tensor* h_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_tokens);   // h_t(k,n)=h[n][k]
        std::memcpy(h_t->data, h, static_cast<size_t>(d_model) * n_tokens * sizeof(float));
        struct ggml_tensor* hJ = ggml_mul_mat(ctx, J, h_t);           // [d_model, n]: x[n][m]=sum_k J[m,k]*h[n][k]
        struct ggml_tensor* nm = ggml_rms_norm(ctx, hJ, eps);         // final RMSNorm over the feature dim
        nm = ggml_mul(ctx, nm, out_norm);                             // * output_norm.weight (broadcast over n)
        struct ggml_tensor* logits = ggml_mul_mat(ctx, out_head, nm); // [vocab, n]: the model's own head

        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, logits);
        int nthreads = static_cast<int>(std::thread::hardware_concurrency());
        if (nthreads < 1) nthreads = 4;
        if (nthreads > 16) nthreads = 16;
        const enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, nthreads);
        if (st != GGML_STATUS_SUCCESS) { ggml_free(ctx); e = "jlens graph compute failed"; return false; }

        const float* Lg = static_cast<const float*>(logits->data);   // column p = Lg + p*vocab
        const int k = std::min(topk, vocab);
        out.assign(static_cast<size_t>(n_tokens), {});
        std::vector<int> idx(static_cast<size_t>(vocab));
        for (int p = 0; p < n_tokens; ++p) {
            const float* row = Lg + static_cast<size_t>(p) * vocab;
            for (int v = 0; v < vocab; ++v) idx[static_cast<size_t>(v)] = v;
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [&](int a, int b) { return row[a] > row[b]; });
            auto& dst = out[static_cast<size_t>(p)];
            dst.reserve(static_cast<size_t>(k));
            for (int j = 0; j < k; ++j) dst.emplace_back(idx[static_cast<size_t>(j)], row[idx[static_cast<size_t>(j)]]);
        }
        ggml_free(ctx);
        return true;
    }
};

// A pool of contexts over ONE shared model. Each request acquires a free context (blocking until
// one is available), runs on it, and releases it — so N workers serve N requests concurrently
// while the weights (the bulk of VRAM) are loaded once. RAII Lease guarantees release on any exit.
class ContextPool {
public:
    ContextPool(std::shared_ptr<GgmlModel> model, int workers, int n_ctx) {
        for (int i = 0; i < workers; ++i) {
            adapters_.push_back(std::make_unique<GgmlAdapter>(model, n_ctx));
            free_.push(adapters_.back().get());
        }
    }
    class Lease {
    public:
        Lease(ContextPool& p, GgmlAdapter* a) : pool_(p), adapter_(a) {}
        ~Lease() { pool_.release(adapter_); }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        GgmlAdapter& operator*() const { return *adapter_; }
    private:
        ContextPool& pool_;
        GgmlAdapter* adapter_;
    };
    Lease acquire() {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [&] { return !free_.empty(); });
        GgmlAdapter* a = free_.front();
        free_.pop();
        return Lease(*this, a);
    }
    int size() const { return static_cast<int>(adapters_.size()); }

private:
    void release(GgmlAdapter* a) {
        { std::lock_guard<std::mutex> lk(mtx_); free_.push(a); }
        cv_.notify_one();
    }
    std::vector<std::unique_ptr<GgmlAdapter>> adapters_;
    std::queue<GgmlAdapter*> free_;
    std::mutex mtx_;
    std::condition_variable cv_;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [--port N] [--host H] [--gpu-layers N] "
                             "[--mask-token ID] [--eos ID] [--ctx N] [--workers N] "
                             "[--sae <dir>] [--sae-k N] [--jlens <dir>]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    int port = 8080, gpu_layers = 0, mask_token = 151665, eos = -1, n_ctx = 4096, workers = 1;
    std::string host = "127.0.0.1";
    std::string sae_dir;  // --sae: exported SAE weight dir (tools/export_sae_weights.py); off by default
    int sae_k = 16;
    std::string jlens_dir;  // --jlens: J-lens sidecar dir; else CLOZN_JLENS_DIR; else ~/.clozn/jlens (off if empty)
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--port") port = std::atoi(next());
        else if (a == "--host") host = next();
        else if (a == "--gpu-layers") gpu_layers = std::atoi(next());
        else if (a == "--mask-token") mask_token = std::atoi(next());
        else if (a == "--eos") eos = std::atoi(next());
        else if (a == "--ctx") n_ctx = std::atoi(next());
        else if (a == "--workers") workers = std::atoi(next());
        else if (a == "--sae") sae_dir = next();
        else if (a == "--sae-k") sae_k = std::atoi(next());
        else if (a == "--jlens") jlens_dir = next();
    }
    if (workers < 1) workers = 1;
    if (sae_k < 1) sae_k = 1;
    if (jlens_dir.empty()) {                     // --jlens > $CLOZN_JLENS_DIR > ~/.clozn/jlens (the sidecar home)
        if (const char* e = std::getenv("CLOZN_JLENS_DIR")) jlens_dir = e;
        else if (const char* h = std::getenv("USERPROFILE")) jlens_dir = std::string(h) + "/.clozn/jlens";
        else if (const char* h2 = std::getenv("HOME")) jlens_dir = std::string(h2) + "/.clozn/jlens";
    }

    llama_log_set(quiet_log, nullptr);
    // One copy of the weights, N contexts over it — concurrent requests, one model in (V)RAM.
    auto model = std::make_shared<GgmlModel>(model_path, mask_token, eos, gpu_layers);
    ContextPool pool(model, workers, n_ctx);

    // --sae: load the on-device SAE encoder BEFORE probe calibration (the read tap must move to the
    // SAE's layer so the probes calibrate in the space they'll project). Refusals are hard errors —
    // a server that silently dropped the requested readout would be lying about what it emits.
    SaeServe sae_serve;
    sae_serve.k = sae_k;
    if (!sae_dir.empty()) {
#ifdef CLOZE_SAE
        if (!sae_serve.enc.load(sae_dir)) {
            std::fprintf(stderr, "[cloze-server] --sae load failed: %s\n", sae_serve.enc.error().c_str());
            return 1;
        }
        sae_serve.layer = sae_serve.enc.layer();
        sae_serve.on = true;  // n_embd-vs-d_in verified against the model below (calibration block)
        std::fprintf(stderr, "[cloze-server] SAE ready: %d features, d_in %d, layer %d, k %d (%.0f MB device)\n",
                     sae_serve.enc.d_sae(), sae_serve.enc.d_in(), sae_serve.layer, sae_serve.k,
                     sae_serve.enc.device_bytes() / 1e6);
#else
        std::fprintf(stderr, "[cloze-server] --sae requires a server built with -DCLOZE_BUILD_SAE=ON "
                             "(this binary has no CUDA SAE encoder)\n");
        return 1;
#endif
    }

    // Mode follows the MODEL: a diffusion dLLM (LLaDA/Dream) carries a mask token in its GGUF; a
    // standard autoregressive LLM (Llama/Qwen/Mistral/...) does not. AR mode serves /v1/completions
    // via the causal generate_ar loop (the same white-box reads + steering, different generation
    // paradigm); the diffusion-only endpoints (infill/revise/board) return 400. The interpretability
    // is model-agnostic — only the decode differs. `--ar` forces it for an AR model converted with a
    // stray mask id.
    bool force_ar = false;
    for (int i = 2; i < argc; ++i) if (std::string(argv[i]) == "--ar") force_ar = true;
    const bool ar_mode = force_ar || llama_vocab_mask(model->vocab()) < 0;

    // White-box Tier 2: build concept probes once, in the model's own activation space (a temp
    // context with the tap on; ~a few CPU forwards). Passed to generate when a request asks for
    // features. If calibration yields nothing, those requests fall back to the raw "norm" feature.
    // Two probe sets, decoupling READ from WRITE (the read/write tap tension the layer sweep
    // exposed): a diff-in-means direction only steers at the layer it was calibrated in (the
    // residual basis rotates across depth), but per-token concepts read SHARPEST at an early
    // layer (sweep: layer 2 ~18% above 2/3-depth). So concept_probes reads at the adapter's
    // default early tap (display); steer_probes recalibrates at mid-depth (2/3) for an effective
    // control vector. ~2x the startup forwards (a few CPU passes) and one extra K*n_embd buffer.
    ConceptProbes concept_probes;  // READ: sharp early-layer tap, drives the white-box display
    ConceptProbes steer_probes;    // WRITE: mid-depth tap, drives the steering control vector
    {
        std::fprintf(stderr, "[cloze-server] calibrating white-box concept probes (%s)...\n",
                     ar_mode ? "causal/AR" : "bidirectional/diffusion");
        GgmlAdapter cal(model, 512);
        // AR models read activations under CAUSAL attention; calibrate the probe directions in the
        // same regime they'll be projected/steered in, or the diff-in-means directions won't transfer
        // (a token's hidden state differs bidirectional vs causal). forward() under causal mode returns
        // the per-token causal hidden states the AR tap also sees.
        if (ar_mode) cal.set_causal(true);
        cal.set_emit_activations(true);
#ifdef CLOZE_SAE
        if (sae_serve.on) {
            // The SAE only reads the residual space it was trained on: n_embd must equal d_in (the
            // cached andyrdt L15 SAE is Qwen2.5-7B-Instruct's 3584 — a Llama-1B GGUF can't serve it).
            if (cal.n_embd() != sae_serve.enc.d_in()) {
                std::fprintf(stderr, "[cloze-server] --sae mismatch: model n_embd %d != SAE d_in %d "
                                     "(this SAE targets Qwen2.5-7B-Instruct layer %d residuals)\n",
                             cal.n_embd(), sae_serve.enc.d_in(), sae_serve.layer);
                return 1;
            }
            // Featureful requests will tap at the SAE's layer, so the concept probes must calibrate
            // THERE (a diff-in-means direction only reads in the space it was calibrated in). Costs
            // the early-tap sharpness; keeping the display honest matters more than the sharpness.
            cal.set_tap_layer(sae_serve.layer);
            std::fprintf(stderr, "[cloze-server] --sae active: read tap + concept probes move to layer %d\n",
                         sae_serve.layer);
        }
#endif
        const int read_tap = cal.tap_layer();                   // default early tap (SAE layer when --sae)
        concept_probes = calibrate_concept_probes(cal, *model);
        const int steer_tap = cal.n_layer() * 2 / 3;
        cal.set_tap_layer(steer_tap);
        steer_probes = calibrate_concept_probes(cal, *model);    // mid-depth (steer-effective)
        std::fprintf(stderr, "[cloze-server] %d concept probe(s) ready:", concept_probes.size());
        for (const auto& nm : concept_probes.names) std::fprintf(stderr, " %s", nm.c_str());
        std::fprintf(stderr, " (read tap %d, steer tap %d)\n", read_tap, steer_tap);
    }

    // J-lens (JLENS_ENGINE_PLAN.md J2): load the fitted J_l sidecars + the GGUF's own final_norm/head once,
    // so POST /jlens can read each position's "disposed to say" tokens. Off (route 400s) if the dir is
    // absent/incomplete -- a research feature, never required for chat/harvest/score to serve.
    JlensServe jlens;
    if (!jlens_dir.empty()) {
        int jl_n_embd = 0;
        { ContextPool::Lease lease = pool.acquire(); jl_n_embd = (*lease).n_embd(); }
        if (jlens.load(jlens_dir, model_path, jl_n_embd, model->config().vocab_size)) {
            std::fprintf(stderr, "[cloze-server] J-lens ready: d_model %d, vocab %d, eps %.1e, layers",
                         jlens.d_model, jlens.vocab, jlens.eps);
            for (int L : jlens.layers()) std::fprintf(stderr, " %d", L);
            std::fprintf(stderr, " (default %d) from %s\n", jlens.default_layer, jlens_dir.c_str());
        } else {
            std::fprintf(stderr, "[cloze-server] J-lens off: %s\n", jlens.err.c_str());
        }
    }

    httplib::Server svr;

    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        json h{{"status", "ok"}, {"model", model_path},
               {"mode", ar_mode ? "autoregressive" : "diffusion"},
               {"n_ctx", n_ctx},                              // configured context window (repro metadata)
               {"gpu_layers", gpu_layers},                    // layers offloaded to the GPU (0 => CPU-resident)
               {"device", gpu_layers > 0 ? "cuda" : "cpu"}};  // CUDA build; device follows the offload setting
#ifdef CLOZE_SAE
        if (sae_serve.on)
            h["sae"] = {{"d_sae", sae_serve.enc.d_sae()}, {"layer", sae_serve.layer}, {"k", sae_serve.k}};
#endif
        if (jlens.on)
            h["jlens"] = {{"layers", jlens.layers()}, {"default_layer", jlens.default_layer},
                          {"d_model", jlens.d_model}, {"vocab", jlens.vocab}};
        res.set_content(h.dump(), "application/json");
    });

    // The real-time denoise visualization (a pure consumer of the SSE event stream).
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(VIZ_HTML, "text/html; charset=utf-8");
    });

    // Shared body of completions + infill: build the runner, then either stream the events as SSE
    // or run once and return JSON.
    auto handle = [&](const httplib::Request& req, httplib::Response& res, bool is_infill) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        if (ar_mode && is_infill) {  // infill (fill-in-the-middle) is structurally diffusion-only
            res.status = 400;
            res.set_content(json{{"error", "infill requires a diffusion model; this is autoregressive"}}.dump(),
                            "application/json");
            return;
        }
        const GenerateConfig cfg = config_from(body);
        const CacheConfig cache = cache_from(body);
        const ReviseConfig revise = revise_from(body);
        const SampleConfig sample = sample_from(body);
        const bool stream = body.value("stream", false);
        // Phase 1.2 state-stream protocol: protocol:true reshapes the SSE frames to StateStep; state:"full"
        // rides the heavy raw-activation tensor on each frame (the light frame omits it). "full" implies
        // the activation tap (the raw state only exists when the tap is on), so it forces features on.
        const bool protocol = body.value("protocol", false);
        const bool state_full = body.value("state", std::string("light")) == std::string("full");
        const bool features = body.value("features", false) || state_full;  // white-box: emit per-slot activations
        // White-box WRITE: steer:{concept, coef, layer?} pushes a concept direction into the residual
        // stream during the denoise (a control vector). coef 0 / no object => no steering.
        std::string steer_concept; double steer_coef = 0.0; int steer_layer = 0;
        if (body.contains("steer") && body["steer"].is_object()) {
            steer_concept = body["steer"].value("concept", std::string());
            steer_coef = body["steer"].value("coef", 0.0);
            steer_layer = body["steer"].value("layer", 0);
        }

        // Encode inputs via the model's vocab (no context needed — fully concurrent).
        std::vector<int> prompt_ids, suffix_ids;
        int gap = 0;
        if (is_infill) {
            prompt_ids = model->encode(body.value("prefix", std::string()));
            suffix_ids = model->encode(body.value("suffix", std::string()));
            gap = body.value("gap", 8);
        } else {
            prompt_ids = model->encode(body.value("prompt", std::string()));
        }
        if (prompt_ids.empty() && !(is_infill && !suffix_ids.empty())) {
            res.status = 400;
            res.set_content(json{{"error", "empty prompt"}}.dump(), "application/json");
            return;
        }

        // Optional PyTorch-trained soft prefix (the train-on-HF / serve-on-llama.cpp bridge): a flat
        // prefix_rows x n_embd float array spliced in ahead of the prompt before decoding. AR-only.
        std::vector<float> prefix_embd;
        const int prefix_rows = body.value("prefix_rows", 0);
        if (prefix_rows > 0 && body.contains("prefix_embd") && body["prefix_embd"].is_array()) {
            prefix_embd = body["prefix_embd"].get<std::vector<float>>();   // AR: ar_forward_embd; diffusion: set_diffusion_prefix
        }
        // Optional RAW tone direction (the studio's engine tone dials): an n_embd control vector applied
        // via set_steer during THIS generation -- so memory (prefix) + tone steer ride together. AR-only.
        std::vector<float> steer_vec;
        if (ar_mode && body.contains("steer_vec") && body["steer_vec"].is_array()) {
            steer_vec = body["steer_vec"].get<std::vector<float>>();
        }

        // One call into the runtime on a POOLED context (acquire blocks until one is free, so N
        // workers run N requests concurrently; the Lease releases it on any exit).
        auto run = [&pool, &concept_probes, &steer_probes, &sae_serve, prompt_ids, suffix_ids, gap, cfg, cache, revise, sample, is_infill, ar_mode, features, steer_concept, steer_coef, steer_layer, prefix_embd, prefix_rows, steer_vec](
                       const std::function<void(const Event&)>& on_event) {
            ContextPool::Lease lease = pool.acquire();
            (*lease).set_emit_activations(features);  // white-box tap on for this request (off by default)
            // --sae: read at the SAE's own layer and ride each pass's top-k features on the stream.
            const bool sae_on = features && sae_serve.on;
            const int default_tap = (*lease).tap_layer();
            if (sae_on) (*lease).set_tap_layer(sae_serve.layer);
            const std::function<void(const Event&)> ev = with_sae_readout(on_event, sae_serve, sae_on);
            const ConceptProbes* probes = (features && concept_probes.ready()) ? &concept_probes : nullptr;
            const bool steering = !steer_concept.empty() && steer_coef != 0.0 && steer_probes.ready();
            const bool raw_steer = !steer_vec.empty();   // a raw tone direction (studio engine dials)
            if (steering || raw_steer) {
                const int nl = (*lease).n_layer();
                int lo, hi;
                if (steer_layer >= 1) { lo = hi = (steer_layer < nl ? steer_layer : nl - 1); }
                else { const int tl = nl * 2 / 3;  // steer at mid-depth, where steer_probes is calibrated
                       lo = (tl - 2 > 1 ? tl - 2 : 1); hi = (tl + 2 < nl ? tl + 2 : nl - 1); }
                if (lo < 1) lo = 1;
                if (steering) {
                    (*lease).set_steer(build_steer_cvec(steer_probes, steer_concept, steer_coef, lo, hi, nl), lo, hi);
                } else {                          // RAW tone direction: build the same cvec layout straight from it
                    const int ne = static_cast<int>(steer_vec.size());
                    std::vector<float> cvec(static_cast<size_t>(ne) * nl, 0.0f);
                    const double c = steer_coef != 0.0 ? steer_coef : 1.0;
                    for (int L = lo; L <= hi; ++L) {
                        if (L < 1 || L >= nl) continue;
                        float* slice = cvec.data() + static_cast<size_t>(L - 1) * ne;
                        for (int i = 0; i < ne; ++i) slice[i] = static_cast<float>(c * steer_vec[i]);
                    }
                    (*lease).set_steer(cvec, lo, hi);
                }
            }
            // AR model => the causal left-to-right loop (same white-box reads/steering, no scheduler).
            // Diffusion => the denoiser (whole-sequence generate, or infill between prefix/suffix).
            // Diffusion: a soft prefix rides in as a frozen block via set_diffusion_prefix (AR uses the
            // ar_forward_embd arg instead). Either way it's the HF-trained memory, injected into the GGUF.
            const bool diff_prefix = !ar_mode && !prefix_embd.empty() && prefix_rows > 0;
            if (diff_prefix) (*lease).set_diffusion_prefix(prefix_embd, prefix_rows);
            // Restore the pooled context to a clean state (steer/prefix/tap/emit off) on EVERY exit path.
            // Critical: a generator can throw (n_ctx exceeded, llama_decode failure, ...). On the STREAMING
            // path that throw escapes into cpp-httplib's worker thread with no handler -> std::terminate() ->
            // abort(): a silent hard crash (no trace on a Windows Release build). So clean up + rethrow -- the
            // streaming provider below catches the rethrow and emits a clean error frame; the non-streaming
            // caller is already inside httplib's routing try/catch, so it degrades to a 500. Either way the
            // pooled context goes back clean, never dirty.
            auto cleanup = [&]() {
                if (diff_prefix) (*lease).clear_diffusion_prefix();
                if (steering || raw_steer) (*lease).clear_steer();
                if (sae_on) (*lease).set_tap_layer(default_tap);
                (*lease).set_emit_activations(false);  // reset before returning the pooled context
            };
            try {
                GenerateResult r = ar_mode
                       ? generate_ar(*lease, prompt_ids, cfg, ev, sample, probes,
                                     prefix_embd.empty() ? nullptr : &prefix_embd, prefix_rows)
                       : (is_infill
                            ? infill(*lease, prompt_ids, suffix_ids, gap, cfg, nullptr, ev, revise, sample, probes)
                            : generate(*lease, prompt_ids, cfg, cache, nullptr, ev, revise, sample, probes));
                cleanup();
                return r;
            } catch (...) {
                cleanup();
                throw;
            }
        };
        const std::string id = make_id(is_infill ? "infill-" : "cmpl-");
        const char* object = is_infill ? "infill" : "text_completion";

        if (stream) {
            const char* substrate = ar_mode ? "autoregressive" : "diffusion";
            // SSE: each §5.1 event becomes a `data: <json>\n\n` frame — the native streaming wire.
            res.set_chunked_content_provider(
                "text/event-stream",
                [run, id, object, model, prompt_ids, suffix_ids, protocol, state_full, substrate, mask_token]
                (size_t, httplib::DataSink& sink) {
                    auto write = [&](const std::string& s) { sink.write(s.data(), s.size()); };
                    try {                                 // a generator throw here would otherwise escape into
                                                          // httplib's worker thread -> abort(); catch it below.
                    if (protocol) {
                        // State-stream protocol: fold the §5.1 events into StateStep frames.
                        StateStepBuilder builder(*model, substrate, state_full, write);
                        auto on_event = [&](const Event& e) { builder.on_event(e); };
                        GenerateResult r = run(on_event);
                        builder.finish();
                        // Final summary frame: the canonical snapshot (board + text + layout) the
                        // consumer restores via /v1/board (meta.kind="final").
                        json final_frame = {{"kind", "final"}, {"id", id}, {"object", object},
                                            {"text", r.text}, {"finish_reason", finish_reason(r.reason)},
                                            {"board", r.board},
                                            {"layout", board_layout_json(*model, r.board, mask_token)}};
                        write("data: " + dump_json(final_frame) + "\n\n");
                        write("data: [DONE]\n\n");
                        sink.done();
                        return true;
                    }
                    auto on_event = [&](const Event& e) {
                        write("data: " + sse_data(e, *model, prompt_ids, suffix_ids) + "\n\n");
                    };
                    GenerateResult r = run(on_event);
                    // A final OpenAI-style frame carrying the assembled text, then [DONE].
                    json final_frame = {{"id", id}, {"object", object},
                                        {"choices", json::array({{{"text", r.text}, {"index", 0},
                                                     {"finish_reason", finish_reason(r.reason)}}})}};
                    write("data: " + dump_json(final_frame) + "\n\n");
                    write("data: [DONE]\n\n");
                    sink.done();
                    return true;
                    } catch (const std::exception& e) {
                        // The generator threw (n_ctx exceeded, decode failure, ...). Emit a clean error frame
                        // and close the stream gracefully -- run() already restored the pooled context.
                        json err = {{"error", std::string("generation failed: ") + e.what()}};
                        write("data: " + err.dump() + "\n\n");
                        write("data: [DONE]\n\n");
                        sink.done();
                        return true;
                    } catch (...) {
                        write("data: {\"error\":\"generation failed\"}\n\n");
                        write("data: [DONE]\n\n");
                        sink.done();
                        return true;
                    }
                });
            return;
        }

        // Non-streaming: run() can throw on a genuinely exceptional decode state (a prompt that exceeds
        // n_ctx, a llama_decode failure). Catch it here so it becomes a clean 400 JSON error, NEVER an
        // uncaught throw that cpp-httplib turns into an empty-body 500 (the streaming path above has the
        // same guard). A generation that merely reaches the context window is NOT exceptional -- it stops
        // gracefully with finish_reason "length" inside generate_ar; this catch is for the real failures.
        try {
            GenerateResult r = run({});
            json resp = {
                {"id", id}, {"object", object},
                {"choices", json::array({{{"text", r.text}, {"index", 0},
                             {"finish_reason", finish_reason(r.reason)}}})},
                {"board", r.board},  // white-box SNAPSHOT: the full final board (save + POST to /v1/board)
                {"layout", board_layout_json(*model, r.board, mask_token)},
                {"usage", {{"completion_tokens", r.new_tokens}, {"steps_total", r.steps_total}}},
            };
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(dump_json(json{{"error", std::string("generation failed: ") + e.what()}}),
                            "application/json");
        }
    };

    // POST /harvest — the §3.1 "activation harvesting at scale" READ endpoint. Body {text, layer?}:
    // tokenize `text`, run ONE causal forward over all its tokens with the white-box tap on, and
    // return every token's residual at the tap layer (NOT a generation — no sampling, no streaming).
    // Response: {tokens:[piece,...], layer:N, activations:{dtype:"float32", shape:[n_tokens,n_embd],
    // data: base64-LE}} — the matrix a discovery harness (SAE/PCA) trains on. `layer` (optional)
    // overrides the adapter's default tap (else the calibrated early tap, layer 2 for Qwen-0.5B);
    // an out-of-range value falls back to the final layer. One forward per text => efficient, and it
    // captures NATURAL-text activations (every input token), sidestepping the sustained-generation
    // crash the generation-based harvest hit. Works in either mode (it forces causal locally); the
    // pooled context is restored (tap default + emit off) before release.
    svr.Post("/harvest", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        const std::string text = body.value("text", std::string());
        if (text.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "empty text"}}.dump(), "application/json");
            return;
        }
        const std::vector<int> tokens = model->encode(text);
        if (tokens.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "text tokenized to zero tokens"}}.dump(), "application/json");
            return;
        }
        // One forward decodes all tokens in a single ubatch (n_ubatch == n_ctx), so a passage longer
        // than the context can't be harvested in one pass — reject it cleanly (400) BEFORE acquiring a
        // context, so an over-length passage is a skippable client error, never a 500. The harvester
        // chunks its corpus under this; the explicit guard keeps the contract honest at the edge.
        if (static_cast<int>(tokens.size()) > n_ctx) {
            res.status = 400;
            res.set_content(json{{"error", "text too long for one forward"},
                                 {"n_tokens", static_cast<int>(tokens.size())},
                                 {"n_ctx", n_ctx}}.dump(), "application/json");
            return;
        }
        const bool has_layer = body.contains("layer") && body["layer"].is_number();
        const int req_layer = has_layer ? body["layer"].get<int>() : -1;

        try {
            ForwardResult fwd;
            int used_layer = 0;
            {
                ContextPool::Lease lease = pool.acquire();
                GgmlAdapter& ad = *lease;
                const int default_tap = ad.tap_layer();
                ad.set_causal(true);            // harvest under causal attention (also clears the KV)
                ad.set_emit_activations(true);  // white-box tap on for this call
                if (has_layer) ad.set_tap_layer(req_layer);  // 0 / out-of-range => final-layer fallback
                used_layer = ad.tap_layer();
                try {
                    fwd = ad.harvest(tokens);
                    ad.set_emit_activations(false);   // restore the pooled context for the next request
                    ad.set_tap_layer(default_tap);
                } catch (...) {
                    ad.set_emit_activations(false);   // restore even on failure, then rethrow
                    ad.set_tap_layer(default_tap);
                    throw;
                }
            }

            json pieces = json::array();
            for (int id : tokens) pieces.push_back(model->decode({id}));
            json resp = {
                {"tokens", pieces},
                {"layer", used_layer},
                {"n_tokens", static_cast<int>(tokens.size())},
                {"n_embd", fwd.n_embd},
                {"activations", tensor_json_f32(
                    fwd.activations,
                    {static_cast<int>(tokens.size()), fwd.n_embd})},
            };
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // POST /jlens — the J-lens readout (JLENS_ENGINE_PLAN.md J2). Body {text, layer?, topk?=5}: tap the
    // residual `h` at `layer` (the /harvest machinery), transport it into the final-layer basis with the
    // fitted J_l, then unembed through the model's OWN final RMSNorm + output head -> each position's top-k
    // "disposed to say" tokens. NOT generation, NO sampling: a deterministic linear read. Response
    // {layer, n_tokens, tokens:[piece...], readouts:[[{id,piece,score}...k]...n]} (score = the raw lens
    // logit). Guards mirror /harvest: missing sidecar -> 400 + available layers; empty/over-n_ctx -> 400;
    // malformed JSON -> 400; the pooled context's tap/emit flags are restored on EVERY exit.
    svr.Post("/jlens", [&](const httplib::Request& req, httplib::Response& res) {
        if (!jlens.on) {
            res.status = 400;
            res.set_content(json{{"error", "J-lens not loaded (start with --jlens <dir> or set CLOZN_JLENS_DIR)"}}.dump(),
                            "application/json");
            return;
        }
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        const std::string text = body.value("text", std::string());
        if (text.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "empty text"}}.dump(), "application/json");
            return;
        }
        int topk = body.value("topk", 5);
        if (topk < 1) topk = 1;
        const int layer = body.value("layer", jlens.default_layer);
        if (!jlens.has(layer)) {
            res.status = 400;
            res.set_content(json{{"error", "no J-lens sidecar for that layer"},
                                 {"layer", layer}, {"available", jlens.layers()}}.dump(), "application/json");
            return;
        }
        const std::vector<int> tokens = model->encode(text);
        if (tokens.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "text tokenized to zero tokens"}}.dump(), "application/json");
            return;
        }
        if (static_cast<int>(tokens.size()) > n_ctx) {
            res.status = 400;
            res.set_content(json{{"error", "text too long for one forward"},
                                 {"n_tokens", static_cast<int>(tokens.size())}, {"n_ctx", n_ctx}}.dump(),
                            "application/json");
            return;
        }
        try {
            ForwardResult fwd;
            {   // tap `h` at `layer` via the harvest path; restore the pooled context's tap/emit on all exits.
                ContextPool::Lease lease = pool.acquire();
                GgmlAdapter& ad = *lease;
                const int default_tap = ad.tap_layer();
                ad.set_causal(true);            // harvest under causal attention (also clears the KV)
                ad.set_emit_activations(true);
                ad.set_tap_layer(layer);
                try {
                    fwd = ad.harvest(tokens);
                    ad.set_emit_activations(false);
                    ad.set_tap_layer(default_tap);
                } catch (...) {
                    ad.set_emit_activations(false);
                    ad.set_tap_layer(default_tap);
                    throw;
                }
            }
            if (fwd.n_embd != jlens.d_model ||
                fwd.activations.size() != static_cast<size_t>(tokens.size()) * jlens.d_model) {
                res.status = 400;
                res.set_content(json{{"error", "harvest returned no/mismatched activations for this layer"}}.dump(),
                                "application/json");
                return;
            }
            std::vector<std::vector<std::pair<int, float>>> tk;
            std::string e;
            if (!jlens.readout(fwd.activations.data(), static_cast<int>(tokens.size()), layer, topk, tk, e)) {
                res.status = 400;
                res.set_content(json{{"error", e}}.dump(), "application/json");
                return;
            }
            json pieces = json::array();
            for (int id : tokens) pieces.push_back(model->decode({id}));
            json readouts = json::array();
            for (const auto& row : tk) {
                json rr = json::array();
                for (const auto& pr : row)
                    rr.push_back({{"id", pr.first}, {"piece", model->decode({pr.first})}, {"score", pr.second}});
                readouts.push_back(rr);
            }
            json resp = {{"layer", layer}, {"n_tokens", static_cast<int>(tokens.size())},
                         {"tokens", pieces}, {"readouts", readouts}};
            // dump_json (replace handler): token/readout pieces can be a PARTIAL multi-byte UTF-8 sequence
            // (byte-fallback tokens split a codepoint) -- strict dump() would throw on those; replace emits
            // U+FFFD instead, same as the streaming path. Determinism holds (identical inputs -> identical bytes).
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // POST /harvest/layers — the per-layer activation SUMMARY: one causal forward, and every layer's
    // residual is reduced to the L2 norm of each token's hidden state (GgmlAdapter::layer_summary). Returns
    // the depth x position "MRI" map ([n_layer][n_tokens] norms) + a per-layer mean, in ONE forward — the
    // cheap cross-depth view /harvest (single-layer, full tensor) can't give without n_layer separate calls.
    svr.Post("/harvest/layers", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        const std::string text = body.value("text", std::string());
        if (text.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "empty text"}}.dump(), "application/json");
            return;
        }
        const std::vector<int> tokens = model->encode(text);
        if (tokens.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "text tokenized to zero tokens"}}.dump(), "application/json");
            return;
        }
        if (static_cast<int>(tokens.size()) > n_ctx) {
            res.status = 400;
            res.set_content(json{{"error", "text too long for one forward"},
                                 {"n_tokens", static_cast<int>(tokens.size())},
                                 {"n_ctx", n_ctx}}.dump(), "application/json");
            return;
        }
        try {
            LayerSummary ls;
            {
                ContextPool::Lease lease = pool.acquire();
                GgmlAdapter& ad = *lease;
                ad.set_causal(true);              // summary under causal attention (also clears the KV)
                ls = ad.layer_summary(tokens);    // one forward, all layers (the method restores its own flag)
            }
            json pieces = json::array();
            for (int id : tokens) pieces.push_back(model->decode({id}));
            json norms = json::array();
            json layer_mean = json::array();
            for (const std::vector<float>& layer : ls.norms) {
                json row = json::array();
                double sum = 0.0;
                for (float v : layer) { row.push_back(v); sum += v; }
                norms.push_back(row);
                layer_mean.push_back(layer.empty() ? 0.0 : sum / static_cast<double>(layer.size()));
            }
            json resp = {
                {"tokens", pieces},
                {"n_tokens", ls.n_tokens},
                {"n_layer", ls.n_layer},
                {"norms", norms},           // [n_layer][n_tokens]: |residual| per token per layer
                {"layer_mean", layer_mean}, // [n_layer]: mean token norm at each layer
            };
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // POST /state — GAP #1 (task #43): the WRITE-and-observe inverse of /harvest, over HTTP. Body
    // {text, layer, positions:[int], values:[float] = positions.size()*n_embd (the EDITED residual rows a
    // client read via /harvest and changed)}: run a baseline causal forward, OVERWRITE those positions'
    // residual at `layer` (GgmlAdapter::write_state — the patch-free eval-callback activation patch), run
    // again, then clear — and report how the model's next-token prediction moved. /harvest (read) + this
    // (write) close the read->edit->write->observe loop on the LIVE model over HTTP. Leak-free: the write
    // is cleared before the pooled context is released.
    svr.Post("/state", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("text") || !body.contains("layer") ||
            !body.contains("positions") || !body.contains("values")) {
            res.status = 400;
            res.set_content(json{{"error", "need {text, layer, positions:[int], values:[float]}"}}.dump(),
                            "application/json");
            return;
        }
        const std::string text = body["text"].get<std::string>();
        const int layer = body["layer"].get<int>();
        const std::vector<int> positions = body["positions"].get<std::vector<int>>();
        const std::vector<float> values = body["values"].get<std::vector<float>>();
        const std::vector<int> tokens = model->encode(text);
        if (tokens.empty() || static_cast<int>(tokens.size()) > n_ctx) {
            res.status = 400;
            res.set_content(json{{"error", "text empty or too long for one forward"}}.dump(),
                            "application/json");
            return;
        }
        auto top3 = [&](const std::vector<float>& lg, int vocab) {
            json arr = json::array();
            if (lg.empty() || vocab <= 0) return arr;
            float mx = lg[0];
            for (int v = 1; v < vocab; ++v) mx = std::max(mx, lg[v]);
            double Z = 0.0;
            for (int v = 0; v < vocab; ++v) Z += std::exp(double(lg[v]) - double(mx));
            std::vector<int> idx(static_cast<size_t>(vocab));
            for (int v = 0; v < vocab; ++v) idx[static_cast<size_t>(v)] = v;
            const int k = std::min(3, vocab);
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [&](int a, int b) { return lg[a] > lg[b]; });
            for (int j = 0; j < k; ++j) {
                const int v = idx[static_cast<size_t>(j)];
                arr.push_back({{"token", model->decode({v})},
                               {"prob", std::exp(double(lg[v]) - double(mx)) / Z}});
            }
            return arr;
        };
        try {
            std::vector<float> base_lg, edit_lg;
            int vocab = 0;
            bool applied = false;
            {
                ContextPool::Lease lease = pool.acquire();
                GgmlAdapter& ad = *lease;
                try {
                    ad.set_causal(true);                       // causal forward (also clears the KV)
                    ForwardResult b = ad.ar_forward(tokens, 0);
                    base_lg = b.logits; vocab = b.vocab;
                    applied = ad.write_state(layer, positions, values);
                    if (applied) {
                        ad.set_causal(true);                   // fresh KV, with the write now armed
                        ForwardResult e = ad.ar_forward(tokens, 0);
                        edit_lg = e.logits;
                    }
                    ad.clear_write();                          // leak-free: never sticks on the pooled ctx
                } catch (...) {
                    ad.clear_write();
                    throw;
                }
            }
            double moved = 0.0;
            if (applied && base_lg.size() == edit_lg.size()) {
                for (size_t i = 0; i < base_lg.size(); ++i) {
                    const double d = double(base_lg[i]) - double(edit_lg[i]);
                    moved += d * d;
                }
                moved = std::sqrt(moved);
            }
            json resp = {
                {"applied", applied}, {"layer", layer},
                {"n_positions", static_cast<int>(positions.size())},
                {"n_values", static_cast<int>(values.size())},
                {"moved_l2", moved},
                {"baseline_top", top3(base_lg, vocab)},
                {"edited_top", applied ? top3(edit_lg, vocab) : json::array()},
            };
            if (!applied)
                resp["error"] = "write_state rejected (layer must be in [1, n_layer); "
                                "values.size must equal positions.size * n_embd)";
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // POST /apply_template — render chat messages into a prompt string using THE MODEL'S OWN embedded
    // chat template (from the GGUF's tokenizer.chat_template metadata), so every model gets its correct
    // format instead of a hardcoded ChatML. This is the seam that makes clozn model-agnostic: the Python
    // substrate sends {messages:[{role,content}]}, the engine templates per-model (Qwen -> ChatML
    // <|im_start|>, Llama-3 -> <|start_header_id|>, Gemma -> <start_of_turn>, ...) via
    // llama_chat_apply_template (which detects the template family from the jinja source). No context/KV
    // and no sampling: pure model-metadata + string work, so NO pool lease is taken (nothing to leak,
    // fully concurrent with generation). No-embedded-template is surfaced as a clean 400, never silently
    // mis-formatted. Body: {messages:[{role,content}], add_assistant?:bool=true} -> {prompt, template_source}.
    svr.Post("/apply_template", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
            res.status = 400;
            res.set_content(json{{"error", "'messages' must be a non-empty array of {role, content}"}}.dump(),
                            "application/json");
            return;
        }
        // The model's OWN embedded chat template (nullptr name => the default tokenizer.chat_template).
        const char* tmpl = llama_model_chat_template(model->handle(), /*name=*/nullptr);
        if (tmpl == nullptr) {
            res.status = 400;
            res.set_content(json{{"error", "model has no embedded chat template; cannot format messages "
                                           "per-model (re-convert the GGUF with its tokenizer.chat_template, "
                                           "or send a pre-rendered prompt to /v1/completions)"}}.dump(),
                            "application/json");
            return;
        }
        // Own the role/content strings for the lifetime of the call: llama_chat_message holds raw
        // const char* into these, so the backing std::strings must outlive the apply call.
        std::vector<std::string> roles, contents;
        roles.reserve(body["messages"].size());
        contents.reserve(body["messages"].size());
        for (const auto& m : body["messages"]) {
            if (!m.is_object()) {
                res.status = 400;
                res.set_content(json{{"error", "each message must be an object with role + content"}}.dump(),
                                "application/json");
                return;
            }
            roles.push_back(m.value("role", std::string()));
            contents.push_back(m.value("content", std::string()));
        }
        const bool add_ass = body.value("add_assistant", true);
        try {
            std::vector<llama_chat_message> chat;
            chat.reserve(roles.size());
            size_t total = 0;
            for (size_t i = 0; i < roles.size(); ++i) {
                chat.push_back(llama_chat_message{roles[i].c_str(), contents[i].c_str()});
                total += roles[i].size() + contents[i].size();
            }
            // Recommended alloc is 2 * total chars; grow-and-retry if the template expands past it
            // (llama_chat_apply_template returns the TRUE formatted length even when the buffer is short).
            std::vector<char> buf(2 * total + 1024);
            int32_t n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_ass,
                                                  buf.data(), static_cast<int32_t>(buf.size()));
            if (n > static_cast<int32_t>(buf.size())) {
                buf.resize(static_cast<size_t>(n));
                n = llama_chat_apply_template(tmpl, chat.data(), chat.size(), add_ass,
                                              buf.data(), static_cast<int32_t>(buf.size()));
            }
            if (n < 0) {
                res.status = 400;
                res.set_content(json{{"error", "chat template application failed (template family not "
                                               "supported by llama_chat_apply_template)"}}.dump(),
                                "application/json");
                return;
            }
            std::string prompt(buf.data(), static_cast<size_t>(n));
            // llama.cpp's chat-template formatters emit only the turn structure, NOT the leading BOS --
            // HF templates rely on the tokenizer to prepend it. But clozn tokenizes this prompt
            // downstream with add_special=false, so a model whose tokenizer prepends BOS (Llama-3: yes;
            // Qwen2.5: no) would otherwise lose it. Prepend the BOS as its special-token PIECE when the
            // model wants one and it's not already there; the downstream parse_special tokenization folds
            // it back into the single BOS id. Guarded on add_bos, so non-BOS models (Qwen ChatML) stay
            // byte-identical to before this route existed.
            const llama_vocab* vocab = model->vocab();
            bool bos_prepended = false;
            if (llama_vocab_get_add_bos(vocab)) {
                const llama_token bos = llama_vocab_bos(vocab);
                if (bos >= 0) {
                    char pbuf[64];
                    const int pn = llama_token_to_piece(vocab, bos, pbuf, sizeof(pbuf), 0, /*special=*/true);
                    if (pn > 0) {
                        const std::string bos_piece(pbuf, static_cast<size_t>(pn));
                        if (prompt.compare(0, bos_piece.size(), bos_piece) != 0) {
                            prompt = bos_piece + prompt;
                            bos_prepended = true;
                        }
                    }
                }
            }
            json resp = {{"prompt", prompt}, {"template_source", "model"},
                         {"bos_prepended", bos_prepended}};
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // POST /score — teacher-forced per-token logprob of a continuation, AR-only (the
    // reproduce-and-prove foundation). No sampling: ONE causal decode of prompt++continuation
    // (GgmlAdapter::ar_forward_score) reads back, for each continuation token, what the model
    // actually thought of the token it was FORCED to see next -- the log-softmax probability the
    // model assigned to A[i] at the position that predicts it. Logits at absolute sequence index j
    // predict token j+1, so A[i] (absolute index n_p+i) reads from the log-softmax of logits at
    // n_p+i-1: logits_for = {n_p-1 .. n_p+n_a-2} (n_a rows; no logits needed for earlier prompt
    // positions). Continuation ids are the PRIMARY form (exact -- from a stored trace); a raw
    // `continuation` string is a fallback that retokenizes independently and can drift at the
    // prompt/continuation BPE boundary -- flagged `boundary_approximate` in the response.
    // Request: {prompt|prompt_ids, continuation_ids|continuation, topk?, steer?:{concept?,coef,
    // layer}, steer_vec?}. Response: {n_prompt, n_cont, tokens:[{id,piece,logprob,topk?}], sum_logprob}.
    svr.Post("/score", [&](const httplib::Request& req, httplib::Response& res) {
        if (!ar_mode) {  // the AR next-token factorization; diffusion has no equivalent one-shot read
            res.status = 400;
            res.set_content(json{{"error", "score requires an autoregressive model"}}.dump(),
                            "application/json");
            return;
        }
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }

        const int vocab_n = model->config().vocab_size;
        auto ids_in_range = [&](const std::vector<int>& ids) {
            for (int id : ids) if (id < 0 || id >= vocab_n) return false;
            return true;
        };

        // Prompt: token ids take precedence (exact -- what a stored trace/generation actually saw);
        // else tokenize `prompt` with the SAME model->encode /v1/completions uses, so n_p matches.
        std::vector<int> prompt_ids;
        if (body.contains("prompt_ids") && body["prompt_ids"].is_array()) {
            for (const auto& v : body["prompt_ids"]) prompt_ids.push_back(v.get<int>());
        } else {
            prompt_ids = model->encode(body.value("prompt", std::string()));
        }
        if (prompt_ids.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "empty prompt"}}.dump(), "application/json");
            return;
        }
        if (!ids_in_range(prompt_ids)) {
            res.status = 400;
            res.set_content(json{{"error", "prompt_ids out of vocab range"}}.dump(), "application/json");
            return;
        }

        // Continuation: token ids are PRIMARY; text is a documented-approximate fallback (BPE
        // boundary merges mean tokenizing prompt+continuation separately can differ from tokenizing
        // their concatenation).
        std::vector<int> cont_ids;
        bool boundary_approximate = false;
        if (body.contains("continuation_ids") && body["continuation_ids"].is_array()) {
            for (const auto& v : body["continuation_ids"]) cont_ids.push_back(v.get<int>());
        } else if (body.contains("continuation")) {
            cont_ids = model->encode(body.value("continuation", std::string()));
            boundary_approximate = true;
        }
        if (cont_ids.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "empty continuation (need continuation_ids or continuation)"}}.dump(),
                            "application/json");
            return;
        }
        if (!ids_in_range(cont_ids)) {
            res.status = 400;
            res.set_content(json{{"error", "continuation_ids out of vocab range"}}.dump(), "application/json");
            return;
        }

        const int n_p = static_cast<int>(prompt_ids.size());
        const int n_a = static_cast<int>(cont_ids.size());
        if (n_p + n_a > n_ctx) {
            res.status = 400;
            res.set_content(json{{"error", "prompt + continuation exceeds n_ctx"},
                                 {"n_prompt", n_p}, {"n_cont", n_a}, {"n_ctx", n_ctx}}.dump(),
                            "application/json");
            return;
        }

        const int topk = body.value("topk", 0);

        // Same steer parsing/lease discipline as /v1/completions: a NAMED concept (steer_probes) or
        // a RAW direction (steer_vec -- how the studio's tone dials arrive) as a control vector over
        // [lo,hi] layers. Cleared on every exit (including a throw) so a scored request never leaks
        // a steered lease back into the pool.
        std::string steer_concept; double steer_coef = 0.0; int steer_layer = 0;
        if (body.contains("steer") && body["steer"].is_object()) {
            steer_concept = body["steer"].value("concept", std::string());
            steer_coef = body["steer"].value("coef", 0.0);
            steer_layer = body["steer"].value("layer", 0);
        }
        std::vector<float> steer_vec;
        if (body.contains("steer_vec") && body["steer_vec"].is_array()) {
            steer_vec = body["steer_vec"].get<std::vector<float>>();
        }

        std::vector<int> tokens = prompt_ids;
        tokens.insert(tokens.end(), cont_ids.begin(), cont_ids.end());
        std::vector<int> logits_for(static_cast<size_t>(n_a));
        for (int i = 0; i < n_a; ++i) logits_for[static_cast<size_t>(i)] = n_p - 1 + i;

        try {
            ForwardResult fwd;
            {
                ContextPool::Lease lease = pool.acquire();
                GgmlAdapter& ad = *lease;
                const bool steering = !steer_concept.empty() && steer_coef != 0.0 && steer_probes.ready();
                const bool raw_steer = !steer_vec.empty();
                auto cleanup = [&]() { if (steering || raw_steer) ad.clear_steer(); };
                try {
                    ad.set_causal(true);  // the AR forward needs causal attention (also => a clean KV)
                    if (steering || raw_steer) {
                        const int nl = ad.n_layer();
                        int lo, hi;
                        if (steer_layer >= 1) { lo = hi = (steer_layer < nl ? steer_layer : nl - 1); }
                        else { const int tl = nl * 2 / 3;
                               lo = (tl - 2 > 1 ? tl - 2 : 1); hi = (tl + 2 < nl ? tl + 2 : nl - 1); }
                        if (lo < 1) lo = 1;
                        if (steering) {
                            ad.set_steer(build_steer_cvec(steer_probes, steer_concept, steer_coef, lo, hi, nl), lo, hi);
                        } else {
                            const int ne = static_cast<int>(steer_vec.size());
                            std::vector<float> cvec(static_cast<size_t>(ne) * nl, 0.0f);
                            const double c = steer_coef != 0.0 ? steer_coef : 1.0;
                            for (int L = lo; L <= hi; ++L) {
                                if (L < 1 || L >= nl) continue;
                                float* slice = cvec.data() + static_cast<size_t>(L - 1) * ne;
                                for (int i = 0; i < ne; ++i) slice[i] = static_cast<float>(c * steer_vec[i]);
                            }
                            ad.set_steer(cvec, lo, hi);
                        }
                    }
                    fwd = ad.ar_forward_score(tokens, logits_for);
                    cleanup();
                } catch (...) {
                    cleanup();
                    throw;
                }
            }

            const int vocab = fwd.vocab;
            json tok_json = json::array();
            double sum_logprob = 0.0;
            for (int r = 0; r < n_a; ++r) {
                const float* row = fwd.row(r);
                // log-softmax over the vocab (max-subtract + logsumexp, float32).
                float mx = row[0];
                for (int t = 1; t < vocab; ++t) if (row[t] > mx) mx = row[t];
                float sumexp = 0.0f;
                for (int t = 0; t < vocab; ++t) sumexp += std::exp(row[t] - mx);
                const float logZ = mx + std::log(sumexp);

                const int actual = cont_ids[static_cast<size_t>(r)];
                const float logprob = row[actual] - logZ;
                sum_logprob += logprob;

                json item{{"id", actual}, {"piece", model->decode({actual})}, {"logprob", logprob}};
                if (topk > 0) {
                    std::vector<int> idx(static_cast<size_t>(vocab));
                    for (int t = 0; t < vocab; ++t) idx[static_cast<size_t>(t)] = t;
                    const int k = std::min(topk, vocab);
                    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                                      [&](int a, int b) { return row[a] > row[b]; });
                    json tk = json::array();
                    for (int j = 0; j < k; ++j) {
                        const int t = idx[static_cast<size_t>(j)];
                        tk.push_back({{"id", t}, {"piece", model->decode({t})}, {"logprob", row[t] - logZ}});
                    }
                    item["topk"] = tk;
                }
                tok_json.push_back(item);
            }

            json resp = {
                {"n_prompt", n_p}, {"n_cont", n_a},
                {"tokens", tok_json}, {"sum_logprob", sum_logprob},
            };
            if (boundary_approximate) resp["boundary_approximate"] = true;
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    svr.Post("/v1/completions",
             [&](const httplib::Request& req, httplib::Response& res) { handle(req, res, false); });
    svr.Post("/v1/infill",
             [&](const httplib::Request& req, httplib::Response& res) { handle(req, res, true); });

    // Revise a selection: {text, spans:[{start,end} byte offsets], steps, topk, revise, tau_revise}.
    // The highlighted token spans are re-masked and re-predicted in place under full bidirectional
    // attention (the generalized cloze op, denoise()). SSE streams the §5.1 spine with a board layout.
    svr.Post("/v1/revise", [&](const httplib::Request& req, httplib::Response& res) {
        if (ar_mode) {  // in-place re-mask + re-predict needs bidirectional attention (diffusion-only)
            res.status = 400;
            res.set_content(json{{"error", "revise requires a diffusion model; this is autoregressive"}}.dump(),
                            "application/json");
            return;
        }
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        const std::string text = body.value("text", std::string());
        if (text.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "empty text"}}.dump(), "application/json");
            return;
        }
        std::vector<std::pair<int, int>> spans;
        if (body.contains("spans") && body["spans"].is_array())
            for (const auto& s : body["spans"]) {
                const int a = s.value("start", 0), b = s.value("end", 0);
                if (b > a) spans.emplace_back(a, b);
            }
        if (spans.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "no spans selected to revise"}}.dump(), "application/json");
            return;
        }
        int grow = body.value("grow", 0);
        if (grow < 0) grow = 0;
        std::vector<int> board = masked_board_from_spans(*model, text, spans, mask_token, grow);
        int holes = 0;
        for (int id : board)
            if (id == mask_token) ++holes;
        if (holes == 0) {
            res.status = 400;
            res.set_content(json{{"error", "selection covered no whole token"}}.dump(),
                            "application/json");
            return;
        }
        const GenerateConfig cfg = config_from(body);
        const ReviseConfig revise = revise_from(body);
        const SampleConfig sample = sample_from(body);
        const bool stream = body.value("stream", false);
        const bool features = body.value("features", false);
        std::string steer_concept; double steer_coef = 0.0; int steer_layer = 0;
        if (body.contains("steer") && body["steer"].is_object()) {
            steer_concept = body["steer"].value("concept", std::string());
            steer_coef = body["steer"].value("coef", 0.0);
            steer_layer = body["steer"].value("layer", 0);
        }

        auto run = [&pool, &concept_probes, &steer_probes, &sae_serve, board, cfg, revise, sample, features, steer_concept, steer_coef, steer_layer](const std::function<void(const Event&)>& on_event) {
            ContextPool::Lease lease = pool.acquire();
            (*lease).set_emit_activations(features);
            const bool sae_on = features && sae_serve.on;
            const int default_tap = (*lease).tap_layer();
            if (sae_on) (*lease).set_tap_layer(sae_serve.layer);
            const std::function<void(const Event&)> ev = with_sae_readout(on_event, sae_serve, sae_on);
            const ConceptProbes* probes = (features && concept_probes.ready()) ? &concept_probes : nullptr;
            const bool steering = !steer_concept.empty() && steer_coef != 0.0 && steer_probes.ready();
            if (steering) {
                const int nl = (*lease).n_layer();
                int lo, hi;
                if (steer_layer >= 1) { lo = hi = (steer_layer < nl ? steer_layer : nl - 1); }
                else { const int tl = nl * 2 / 3;  // steer at mid-depth, where steer_probes is calibrated
                       lo = (tl - 2 > 1 ? tl - 2 : 1); hi = (tl + 2 < nl ? tl + 2 : nl - 1); }
                if (lo < 1) lo = 1;
                (*lease).set_steer(build_steer_cvec(steer_probes, steer_concept, steer_coef, lo, hi, nl), lo, hi);
            }
            auto r = denoise(*lease, board, cfg, nullptr, ev, revise, sample, probes);
            if (steering) (*lease).clear_steer();
            if (sae_on) (*lease).set_tap_layer(default_tap);
            (*lease).set_emit_activations(false);
            return r;
        };
        const std::string id = make_id("revise-");

        if (stream) {
            res.set_chunked_content_provider(
                "text/event-stream",
                [run, id, model, board, mask_token](size_t, httplib::DataSink& sink) {
                    auto write = [&](const std::string& s) { sink.write(s.data(), s.size()); };
                    try {                                 // a generator throw here would otherwise escape into
                                                          // httplib's worker thread -> abort(); catch it below.
                    auto on_event = [&](const Event& e) {
                        write("data: " + sse_data_revise(e, *model, board, mask_token) + "\n\n");
                    };
                    GenerateResult r = run(on_event);
                    json final_frame = {{"id", id}, {"object", "revise"},
                                        {"choices", json::array({{{"text", r.text}, {"index", 0},
                                                     {"finish_reason", finish_reason(r.reason)}}})}};
                    write("data: " + dump_json(final_frame) + "\n\n");
                    write("data: [DONE]\n\n");
                    sink.done();
                    return true;
                    } catch (const std::exception& e) {
                        // The generator threw (n_ctx exceeded, decode failure, ...). Emit a clean error frame
                        // and close the stream gracefully, mirroring /v1/completions' streaming guard.
                        json err = {{"error", std::string("generation failed: ") + e.what()}};
                        write("data: " + err.dump() + "\n\n");
                        write("data: [DONE]\n\n");
                        sink.done();
                        return true;
                    } catch (...) {
                        write("data: {\"error\":\"generation failed\"}\n\n");
                        write("data: [DONE]\n\n");
                        sink.done();
                        return true;
                    }
                });
            return;
        }

        GenerateResult r = run({});
        json resp = {
            {"id", id}, {"object", "revise"},
            {"choices", json::array({{{"text", r.text}, {"index", 0},
                         {"finish_reason", finish_reason(r.reason)}}})},
            {"usage", {{"completion_tokens", r.new_tokens}, {"steps_total", r.steps_total}}},
        };
        res.set_content(dump_json(resp), "application/json");
    });

    // /v1/board — restore/branch a board SNAPSHOT. Accepts a raw board (token ids; mask_token marks
    // holes), denoises it under full bidirectional attention, returns the resolved board. The
    // white-box "restore" verb: snapshot a board from any response, edit positions (set a slot to the
    // mask id to re-open it), POST it here to re-resolve. The raw-board generalization of /v1/revise
    // (which derives the masked board from text + byte spans). {board:[ids], steps, topk, block_len,
    // revise, tau_revise, temperature, seed, stream}.
    svr.Post("/v1/board", [&](const httplib::Request& req, httplib::Response& res) {
        if (ar_mode) {  // restoring/denoising a masked board needs bidirectional attention (diffusion-only)
            res.status = 400;
            res.set_content(json{{"error", "board restore requires a diffusion model; this is autoregressive"}}.dump(),
                            "application/json");
            return;
        }
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        if (!body.contains("board") || !body["board"].is_array() || body["board"].empty()) {
            res.status = 400;
            res.set_content(json{{"error", "'board' must be a non-empty array of token ids"}}.dump(),
                            "application/json");
            return;
        }
        std::vector<int> board;
        board.reserve(body["board"].size());
        for (const auto& v : body["board"]) board.push_back(v.get<int>());
        int holes = 0;
        for (int tid : board)
            if (tid == mask_token) ++holes;
        if (holes == 0) {
            res.status = 400;
            res.set_content(json{{"error", "board has no MASK positions to resolve"}}.dump(),
                            "application/json");
            return;
        }
        const GenerateConfig cfg = config_from(body);
        const ReviseConfig revise = revise_from(body);
        const SampleConfig sample = sample_from(body);
        const bool stream = body.value("stream", false);
        const bool features = body.value("features", false);
        std::string steer_concept; double steer_coef = 0.0; int steer_layer = 0;
        if (body.contains("steer") && body["steer"].is_object()) {
            steer_concept = body["steer"].value("concept", std::string());
            steer_coef = body["steer"].value("coef", 0.0);
            steer_layer = body["steer"].value("layer", 0);
        }

        auto run = [&pool, &concept_probes, &steer_probes, &sae_serve, board, cfg, revise, sample, features, steer_concept, steer_coef, steer_layer](const std::function<void(const Event&)>& on_event) {
            ContextPool::Lease lease = pool.acquire();
            (*lease).set_emit_activations(features);
            const bool sae_on = features && sae_serve.on;
            const int default_tap = (*lease).tap_layer();
            if (sae_on) (*lease).set_tap_layer(sae_serve.layer);
            const std::function<void(const Event&)> ev = with_sae_readout(on_event, sae_serve, sae_on);
            const ConceptProbes* probes = (features && concept_probes.ready()) ? &concept_probes : nullptr;
            const bool steering = !steer_concept.empty() && steer_coef != 0.0 && steer_probes.ready();
            if (steering) {
                const int nl = (*lease).n_layer();
                int lo, hi;
                if (steer_layer >= 1) { lo = hi = (steer_layer < nl ? steer_layer : nl - 1); }
                else { const int tl = nl * 2 / 3;  // steer at mid-depth, where steer_probes is calibrated
                       lo = (tl - 2 > 1 ? tl - 2 : 1); hi = (tl + 2 < nl ? tl + 2 : nl - 1); }
                if (lo < 1) lo = 1;
                (*lease).set_steer(build_steer_cvec(steer_probes, steer_concept, steer_coef, lo, hi, nl), lo, hi);
            }
            auto r = denoise(*lease, board, cfg, nullptr, ev, revise, sample, probes);
            if (steering) (*lease).clear_steer();
            if (sae_on) (*lease).set_tap_layer(default_tap);
            (*lease).set_emit_activations(false);
            return r;
        };
        const std::string id = make_id("board-");

        if (stream) {
            res.set_chunked_content_provider(
                "text/event-stream",
                [run, id, model, board, mask_token](size_t, httplib::DataSink& sink) {
                    auto write = [&](const std::string& s) { sink.write(s.data(), s.size()); };
                    try {                                 // a generator throw here would otherwise escape into
                                                          // httplib's worker thread -> abort(); catch it below.
                    auto on_event = [&](const Event& e) {
                        write("data: " + sse_data_revise(e, *model, board, mask_token) + "\n\n");
                    };
                    GenerateResult r = run(on_event);
                    json final_frame = {{"id", id}, {"object", "board"}, {"board", r.board},
                                        {"layout", board_layout_json(*model, r.board, mask_token)},
                                        {"choices", json::array({{{"text", r.text}, {"index", 0},
                                                     {"finish_reason", finish_reason(r.reason)}}})}};
                    write("data: " + dump_json(final_frame) + "\n\n");
                    write("data: [DONE]\n\n");
                    sink.done();
                    return true;
                    } catch (const std::exception& e) {
                        // The generator threw (n_ctx exceeded, decode failure, ...). Emit a clean error frame
                        // and close the stream gracefully, mirroring /v1/completions' streaming guard.
                        json err = {{"error", std::string("generation failed: ") + e.what()}};
                        write("data: " + err.dump() + "\n\n");
                        write("data: [DONE]\n\n");
                        sink.done();
                        return true;
                    } catch (...) {
                        write("data: {\"error\":\"generation failed\"}\n\n");
                        write("data: [DONE]\n\n");
                        sink.done();
                        return true;
                    }
                });
            return;
        }

        GenerateResult r = run({});
        json resp = {
            {"id", id}, {"object", "board"}, {"board", r.board},
            {"layout", board_layout_json(*model, r.board, mask_token)},
            {"choices", json::array({{{"text", r.text}, {"index", 0},
                         {"finish_reason", finish_reason(r.reason)}}})},
            {"usage", {{"completion_tokens", r.new_tokens}, {"steps_total", r.steps_total}}},
        };
        res.set_content(dump_json(resp), "application/json");
    });

    // POST /intervene — the state-stream protocol's WRITE channel (SPEC.md "Intervene"). Accepts an
    // Intervention {kind, target, vector?, coef} and applies it, then runs a generation so the effect
    // "shows up in the next steps" (the engine is stateless-per-request, so an intervention is realized
    // ON a concrete generation rather than parked on a context). kind:"steer" wraps the existing
    // control-vector path: either a NAMED concept (target.concept -> build_steer_cvec over the
    // calibrated steer_probes) or a RAW direction (vector:[n_embd] -> a control vector built directly).
    // target may carry the usual generation params (prompt, max_tokens, steps, ...) + stream/protocol/
    // state; the response is the steered result (board+text), or the StateStep SSE stream when stream:true.
    // (edit/restore are a board op: snapshot = the `board` in any response, restore = POST /v1/board.)
    svr.Post("/intervene", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        const std::string kind = body.value("kind", std::string("steer"));
        if (kind != "steer") {
            // edit/restore/patch are realized through /v1/board (snapshot+restore), not here.
            res.status = 400;
            res.set_content(json{{"error", "/intervene currently supports kind:'steer' only; "
                                  "use /v1/board for edit/restore (snapshot+restore)"},
                                 {"kind", kind}}.dump(), "application/json");
            return;
        }
        const json target = body.value("target", json::object());
        const double coef = body.value("coef", target.value("coef", 1.0));
        const std::string concept = target.value("concept", body.value("concept", std::string()));
        const int req_layer = target.value("layer", body.value("layer", 0));
        // Raw direction (optional): an explicit [n_embd] steering vector the inspector supplies.
        std::vector<float> raw_vec;
        if (body.contains("vector") && body["vector"].is_array())
            for (const auto& v : body["vector"]) raw_vec.push_back(v.get<float>());
        if (concept.empty() && raw_vec.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "steer needs target.concept or a raw 'vector'"}}.dump(),
                            "application/json");
            return;
        }
        if (!concept.empty() && !steer_probes.ready()) {
            res.status = 400;
            res.set_content(json{{"error", "no concept probes calibrated; pass a raw 'vector' instead"}}.dump(),
                            "application/json");
            return;
        }
        // Resolve a concept name against the calibrated set up-front (clear 404-style error).
        if (!concept.empty()) {
            bool found = false;
            for (const auto& nm : steer_probes.names) if (nm == concept) { found = true; break; }
            if (!found) {
                res.status = 400;
                res.set_content(json{{"error", "unknown concept"}, {"concept", concept},
                                     {"available", steer_probes.names}}.dump(), "application/json");
                return;
            }
        }

        // The generation the intervention is applied to. Reuses the completions request shape; the
        // generation params (prompt, max_tokens, steps, ...) may sit at top level OR nested under
        // `target` — merge them (target wins) so both spellings work.
        json gen = body;
        if (target.is_object())
            for (auto it = target.begin(); it != target.end(); ++it) gen[it.key()] = it.value();
        const GenerateConfig cfg = config_from(gen);
        const ReviseConfig revise = revise_from(gen);
        const SampleConfig sample = sample_from(gen);
        const bool stream = body.value("stream", false);
        const bool protocol = body.value("protocol", false);
        const bool state_full = body.value("state", std::string("light")) == std::string("full");
        const bool features = body.value("features", false) || state_full;
        std::vector<int> prompt_ids = model->encode(target.value("prompt", body.value("prompt", std::string())));
        if (prompt_ids.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "intervene needs a target.prompt to apply the steer to"}}.dump(),
                            "application/json");
            return;
        }

        // The runner: acquire a context, BUILD + SET the control vector (the wrapped set_steer path),
        // generate, then clear. The layer window matches the per-request steer path (mid-depth, where
        // steer_probes is calibrated) unless target.layer pins one.
        // Written inside run() below, read by the streaming content-provider AFTER this handler returns
        // (set_chunked_content_provider only registers the callback; it fires later, from httplib's worker
        // thread, once this handler's stack frame is long gone). A plain handler-local `json` captured by
        // reference here would dangle the moment the handler returns -- shared_ptr keeps it alive for both
        // lambdas (same fix shape as raw_vec below, which only needs to be read, so a by-value copy suffices).
        auto applied_layers = std::make_shared<json>(json::array());
        auto run = [&pool, &steer_probes, &concept_probes, &sae_serve, raw_vec, concept, coef, req_layer, prompt_ids,
                    cfg, revise, sample, features, ar_mode, applied_layers](
                       const std::function<void(const Event&)>& on_event) {
            ContextPool::Lease lease = pool.acquire();
            (*lease).set_emit_activations(features);
            const bool sae_on = features && sae_serve.on;
            const int default_tap = (*lease).tap_layer();
            if (sae_on) (*lease).set_tap_layer(sae_serve.layer);
            const std::function<void(const Event&)> ev = with_sae_readout(on_event, sae_serve, sae_on);
            const ConceptProbes* probes = (features && concept_probes.ready()) ? &concept_probes : nullptr;
            const int nl = (*lease).n_layer();
            int lo, hi;
            if (req_layer >= 1) { lo = hi = (req_layer < nl ? req_layer : nl - 1); }
            else { const int tl = nl * 2 / 3; lo = (tl - 2 > 1 ? tl - 2 : 1); hi = (tl + 2 < nl ? tl + 2 : nl - 1); }
            if (lo < 1) lo = 1;
            // Build the n_embd*n_layer control-vector buffer: from the named concept (build_steer_cvec)
            // or straight from the raw direction (same layout, applied over [lo,hi]).
            std::vector<float> cvec;
            if (!concept.empty()) {
                cvec = build_steer_cvec(steer_probes, concept, coef, lo, hi, nl);
            } else {
                const int ne = steer_probes.n_embd > 0 ? steer_probes.n_embd : static_cast<int>(raw_vec.size());
                cvec.assign(static_cast<size_t>(ne) * nl, 0.0f);
                const int m = ne < static_cast<int>(raw_vec.size()) ? ne : static_cast<int>(raw_vec.size());
                for (int L = lo; L <= hi; ++L) {
                    if (L < 1 || L >= nl) continue;
                    float* slice = cvec.data() + static_cast<size_t>(L - 1) * ne;
                    for (int i = 0; i < m; ++i) slice[i] = static_cast<float>(coef * raw_vec[i]);
                }
            }
            *applied_layers = json::array({lo, hi});
            (*lease).set_steer(cvec, lo, hi);
            auto r = ar_mode ? generate_ar(*lease, prompt_ids, cfg, ev, sample, probes)
                             : generate(*lease, prompt_ids, cfg, CacheConfig{}, nullptr, ev, revise, sample, probes);
            (*lease).clear_steer();
            if (sae_on) (*lease).set_tap_layer(default_tap);
            (*lease).set_emit_activations(false);
            return r;
        };
        const std::string id = make_id("interv-");
        const char* substrate = ar_mode ? "autoregressive" : "diffusion";

        if (stream) {
            res.set_chunked_content_provider(
                "text/event-stream",
                [run, id, model, protocol, state_full, substrate, mask_token, applied_layers, kind, concept, coef]
                (size_t, httplib::DataSink& sink) {
                    auto write = [&](const std::string& s) { sink.write(s.data(), s.size()); };
                    try {                                 // a generator throw here would otherwise escape into
                                                          // httplib's worker thread -> abort(); catch it below.
                    GenerateResult r;
                    if (protocol) {
                        StateStepBuilder builder(*model, substrate, state_full, write);
                        r = run([&](const Event& e) { builder.on_event(e); });
                        builder.finish();
                    } else {
                        r = run([&](const Event& e) {
                            write("data: " + to_jsonl_line(e) + "\n\n");
                        });
                    }
                    json final_frame = {{"kind", "final"}, {"id", id}, {"object", "intervene"},
                                        {"applied", true},
                                        {"intervention", {{"kind", kind}, {"concept", concept}, {"coef", coef},
                                                          {"layers", *applied_layers}}},
                                        {"text", r.text}, {"finish_reason", finish_reason(r.reason)},
                                        {"board", r.board},
                                        {"layout", board_layout_json(*model, r.board, mask_token)}};
                    write("data: " + dump_json(final_frame) + "\n\n");
                    write("data: [DONE]\n\n");
                    sink.done();
                    return true;
                    } catch (const std::exception& e) {
                        // The generator threw (n_ctx exceeded, decode failure, ...). Emit a clean error frame
                        // and close the stream gracefully, mirroring /v1/completions' streaming guard.
                        json err = {{"error", std::string("generation failed: ") + e.what()}};
                        write("data: " + err.dump() + "\n\n");
                        write("data: [DONE]\n\n");
                        sink.done();
                        return true;
                    } catch (...) {
                        write("data: {\"error\":\"generation failed\"}\n\n");
                        write("data: [DONE]\n\n");
                        sink.done();
                        return true;
                    }
                });
            return;
        }

        GenerateResult r = run({});
        json resp = {
            {"id", id}, {"object", "intervene"}, {"applied", true},
            {"intervention", {{"kind", kind}, {"concept", concept}, {"coef", coef}, {"layers", *applied_layers}}},
            {"choices", json::array({{{"text", r.text}, {"index", 0},
                         {"finish_reason", finish_reason(r.reason)}}})},
            {"board", r.board},
            {"layout", board_layout_json(*model, r.board, mask_token)},
            {"usage", {{"completion_tokens", r.new_tokens}, {"steps_total", r.steps_total}}},
        };
        res.set_content(dump_json(resp), "application/json");
    });

    std::fprintf(stderr, "[cloze-server] %s %s, %d worker(s) on http://%s:%d  (model: %s)\n",
                 ar_mode ? "autoregressive" : "diffusion",
                 gpu_layers > 0 ? "GPU" : "CPU", pool.size(), host.c_str(), port, model_path.c_str());
    if (!svr.listen(host, port)) {
        std::fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
