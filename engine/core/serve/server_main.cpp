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

#include "server_shared.hpp"   // helpers + white-box state structs (was the anonymous namespace)
#include "server_context.hpp"  // ServerContext + register_*_routes declarations

using json = nlohmann::json;
using namespace cloze;

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

    // Phase 12.4: the shared state the relocated route families read (instead of [&]-capturing
    // main locals). Reference members alias these locals; the model rides as a shared_ptr.
    ServerContext ctx{model, pool, concept_probes, steer_probes, sae_serve, jlens,
                       n_ctx, mask_token, ar_mode, gpu_layers, model_path};

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
        // Optional early-stop reference (prove-all ablated arms, AR-only): the baseline reply's committed
        // token ids. generate_ar halts at the first generated token that differs from this list -- so the
        // ablated arm decodes only up to the point the answer provably changed, not the full ~max_tokens.
        // Purely a termination check; greedy output up to the stop is bit-identical to full generation.
        std::vector<int> reference_tokens;
        if (ar_mode && body.contains("reference_tokens") && body["reference_tokens"].is_array()) {
            reference_tokens = body["reference_tokens"].get<std::vector<int>>();
        }

        // One call into the runtime on a POOLED context (acquire blocks until one is free, so N
        // workers run N requests concurrently; the Lease releases it on any exit).
        auto run = [&pool, &concept_probes, &steer_probes, &sae_serve, prompt_ids, suffix_ids, gap, cfg, cache, revise, sample, is_infill, ar_mode, features, steer_concept, steer_coef, steer_layer, prefix_embd, prefix_rows, steer_vec, reference_tokens](
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
                                     prefix_embd.empty() ? nullptr : &prefix_embd, prefix_rows,
                                     reference_tokens.empty() ? nullptr : &reference_tokens)
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
                    if (r.ref_active) {  // early-stop-on-divergence verdict (prove-all ablated arms)
                        final_frame["diverged"] = r.diverged;
                        final_frame["diverged_at"] = r.diverged_at;
                    }
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
            if (r.ref_active) {  // early-stop-on-divergence verdict (prove-all ablated arms)
                resp["diverged"] = r.diverged;
                resp["diverged_at"] = r.diverged_at;
            }
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(dump_json(json{{"error", std::string("generation failed: ") + e.what()}}),
                            "application/json");
        }
    };

    register_whitebox_routes(svr, ctx);

    register_jlens_routes(svr, ctx);


    register_state_routes(svr, ctx);



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


    std::fprintf(stderr, "[cloze-server] %s %s, %d worker(s) on http://%s:%d  (model: %s)\n",
                 ar_mode ? "autoregressive" : "diffusion",
                 gpu_layers > 0 ? "GPU" : "CPU", pool.size(), host.c_str(), port, model_path.c_str());
    if (!svr.listen(host, port)) {
        std::fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
