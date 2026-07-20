// serve/routes_whitebox.cpp -- the white-box read routes: /harvest, /harvest/layers, /score, /apply_template (Phase 12.4 split of the serve monolith).
// Moved VERBATIM from the monolith; shared state is read through ServerContext -- the register
// fn re-binds local aliases so each handler body is byte-identical to the original.
#include "httplib.h"

#include "server_context.hpp"

namespace cloze {

void register_whitebox_routes(httplib::Server& svr, ServerContext& ctx) {
    auto& model = ctx.model;
    auto& pool = ctx.pool;
    auto& concept_probes = ctx.concept_probes;
    auto& steer_probes = ctx.steer_probes;
    auto& sae_serve = ctx.sae_serve;
    auto& jlens = ctx.jlens;
    const int& n_ctx = ctx.n_ctx;
    const int& mask_token = ctx.mask_token;
    const bool& ar_mode = ctx.ar_mode;
    (void)model; (void)pool; (void)concept_probes; (void)steer_probes; (void)sae_serve;
    (void)jlens; (void)n_ctx; (void)mask_token; (void)ar_mode;

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
        const int n_total = n_p + n_a;

        // Circuit-tracer slice 1 (notes/CIRCUIT_TRACER_DESIGN.md): teacher-forced score UNDER an
        // intervention. write:{layer, positions:[absolute seq pos], values:[positions*n_embd rows]}
        // overwrites those residual rows at `layer` during this ONE forward (GgmlAdapter::write_state
        // — the /state activation patch, leak-free clear on every exit). capture:{layers, positions}
        // returns those positions' residual rows from the multi-layer capture plane, read off the
        // SAME forward — under the patch, if one is armed (at the write layer itself the captured row
        // is the PRE-edit state, same convention as the read tap). Together they are the tracer's two
        // primitives: "score with node ablated" and "patch A, capture at B" in one call each.
        // `write` may be ONE spec {layer, positions, values} or an ARRAY of them — the array form is
        // the tracer's joint arm (all candidate nodes ablated simultaneously, across layers, in one
        // forward — GgmlAdapter::add_write_state per spec).
        struct WriteReq { int layer; std::vector<int> positions; std::vector<float> values; };
        std::vector<WriteReq> write_reqs;
        if (body.contains("write")) {
            json specs = json::array();
            if (body["write"].is_object()) specs.push_back(body["write"]);
            else if (body["write"].is_array()) specs = body["write"];
            for (const json& wb : specs) {
                WriteReq w{};
                w.layer = wb.is_object() ? wb.value("layer", 0) : 0;
                if (wb.is_object() && wb.contains("positions") && wb["positions"].is_array())
                    w.positions = wb["positions"].get<std::vector<int>>();
                if (wb.is_object() && wb.contains("values") && wb["values"].is_array())
                    w.values = wb["values"].get<std::vector<float>>();
                if (w.layer < 1 || w.positions.empty() || w.values.empty()) {
                    res.status = 400;
                    res.set_content(json{{"error",
                        "each write spec needs {layer >= 1, positions:[int], values:[float] = positions*n_embd}"}}.dump(),
                        "application/json");
                    return;
                }
                for (int p : w.positions) {
                    if (p < 0 || p >= n_total) {
                        res.status = 400;
                        res.set_content(json{{"error", "write position out of range"}, {"position", p},
                                             {"n_tokens", n_total}}.dump(), "application/json");
                        return;
                    }
                }
                write_reqs.push_back(std::move(w));
            }
            if (write_reqs.empty()) {
                res.status = 400;
                res.set_content(json{{"error", "write must be an object or non-empty array of specs"}}.dump(),
                                "application/json");
                return;
            }
        }
        // attn_knockout: [{layer, head?, queries:[int], keys:[int], renormalize?}] — sever
        // "query position reads key position" at a layer/head. This is the cross-position
        // primitive residual patching could not provide (notes/CIRCUIT_TRACER_DESIGN.md §5f):
        // cutting the EDGE dodges both the re-supply problem and the unpatchable last layer.
        // Needs the server started with --no-flash-attn (else the softmax is fused and never
        // materializes) — refused cleanly rather than silently ignored.
        std::vector<GgmlAdapter::AttnKnockout> knockouts;
        if (body.contains("attn_knockout")) {
            json ks = json::array();
            if (body["attn_knockout"].is_object()) ks.push_back(body["attn_knockout"]);
            else if (body["attn_knockout"].is_array()) ks = body["attn_knockout"];
            for (const json& kb : ks) {
                GgmlAdapter::AttnKnockout k;
                if (!kb.is_object()) continue;
                k.layer = kb.value("layer", -1);
                k.head = kb.value("head", -1);
                k.renormalize = kb.value("renormalize", false);
                if (kb.contains("queries") && kb["queries"].is_array())
                    k.queries = kb["queries"].get<std::vector<int>>();
                if (kb.contains("keys") && kb["keys"].is_array())
                    k.keys = kb["keys"].get<std::vector<int>>();
                if (k.layer < 0 || k.queries.empty() || k.keys.empty()) {
                    res.status = 400;
                    res.set_content(json{{"error", "each attn_knockout needs {layer >= 0, "
                                                   "queries:[int], keys:[int]}"}}.dump(),
                                    "application/json");
                    return;
                }
                knockouts.push_back(std::move(k));
            }
            if (knockouts.empty()) {
                res.status = 400;
                res.set_content(json{{"error", "attn_knockout must be an object or non-empty array"}}.dump(),
                                "application/json");
                return;
            }
        }

        std::vector<int> capture_layers, capture_positions;
        if (body.contains("capture") && body["capture"].is_object()) {
            const json& cb = body["capture"];
            if (cb.contains("layers") && cb["layers"].is_array())
                capture_layers = cb["layers"].get<std::vector<int>>();
            if (cb.contains("positions") && cb["positions"].is_array())
                capture_positions = cb["positions"].get<std::vector<int>>();
            if (capture_layers.empty() || capture_positions.empty()) {
                res.status = 400;
                res.set_content(json{{"error", "capture needs {layers:[int], positions:[int]}"}}.dump(),
                                "application/json");
                return;
            }
            for (int p : capture_positions) {
                if (p < 0 || p >= n_total) {
                    res.status = 400;
                    res.set_content(json{{"error", "capture position out of range"}, {"position", p},
                                         {"n_tokens", n_total}}.dump(), "application/json");
                    return;
                }
            }
        }

        try {
            ForwardResult fwd;
            CaptureFrame cap_frame;             // filled synchronously by the capture sink (if armed)
            std::vector<int> cap_layers_armed;  // layers that survived set_capture_layers validation
            {
                ContextPool::Lease lease = pool.acquire();
                GgmlAdapter& ad = *lease;
                const bool steering = !steer_concept.empty() && steer_coef != 0.0 && steer_probes.ready();
                const bool raw_steer = !steer_vec.empty();
                const bool writing = !write_reqs.empty();
                const bool capturing = !capture_layers.empty();
                const bool knocking = !knockouts.empty();
                auto cleanup = [&]() {
                    if (steering || raw_steer) ad.clear_steer();
                    if (writing) ad.clear_write();
                    if (capturing) { ad.set_capture_sink({}); ad.set_capture_layers({}); }
                    if (knocking) ad.clear_attn_knockouts();
                };
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
                    if (writing) {
                        ad.clear_write();
                        for (const WriteReq& w : write_reqs) {
                            if (!ad.add_write_state(w.layer, w.positions, w.values)) {
                                throw std::invalid_argument(
                                    "write rejected: layer must be in [1, n_layer) and values.size must "
                                    "equal positions.size * n_embd (n_embd " + std::to_string(ad.n_embd()) +
                                    ", layer " + std::to_string(w.layer) + ")");
                            }
                        }
                    }
                    if (knocking) {
                        // Refuse rather than silently no-op: with flash attention the softmax is
                        // fused and the weights we would zero never exist as a tensor.
                        if (!ad.knockout_available())
                            throw std::invalid_argument(
                                "attn_knockout requires the server to be started with --no-flash-attn "
                                "(flash attention fuses the softmax, so the attention weights never "
                                "materialize and the knockout would be silently ignored)");
                        for (const auto& k : knockouts) {
                            if (k.layer >= ad.n_layer())
                                throw std::invalid_argument("attn_knockout layer out of range [0, n_layer)");
                            if (k.head >= ad.n_head())
                                throw std::invalid_argument("attn_knockout head out of range [0, n_head)");
                        }
                        ad.set_attn_knockouts(knockouts);
                    }
                    if (capturing) {
                        ad.set_capture_layers(capture_layers);
                        cap_layers_armed = ad.capture_layers();  // invalid layers were dropped
                        if (cap_layers_armed.empty())
                            throw std::invalid_argument("capture layers all out of range (1..n_layer-1)");
                        ad.set_capture_sink([&cap_frame](CaptureFrame&& f) { cap_frame = std::move(f); });
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
            if (!write_reqs.empty()) {
                resp["write_applied"] = true;
                resp["n_writes"] = static_cast<int>(write_reqs.size());
            }
            if (!knockouts.empty()) {
                resp["knockout_applied"] = true;
                resp["n_knockouts"] = static_cast<int>(knockouts.size());
            }
            if (!capture_layers.empty()) {
                // captured: {"<layer>": {"<pos>": [n_embd floats], ...}, ...} — the residual rows the
                // tracer keeps for computing edited values (mean-/directional-ablate) client-side.
                json cap_json = json::object();
                for (const auto& lv : cap_frame.layers) {
                    json rows = json::object();
                    for (int p : capture_positions) {
                        if (p < 0 || p >= cap_frame.rows) continue;
                        const float* row = lv.second.data() + static_cast<size_t>(p) * cap_frame.n_embd;
                        rows[std::to_string(p)] = std::vector<float>(row, row + cap_frame.n_embd);
                    }
                    cap_json[std::to_string(lv.first)] = std::move(rows);
                }
                // NO SILENT EMPTIES. A layer can pass the [1, n_layer) range check and still yield
                // nothing. The known case is the LAST layer (n_layer-1): llama.cpp applies the
                // inp_out_ids optimization there, materializing ONLY the rows that produce logits
                // (one row for a single-target /score) instead of all positions -- so `l_out-<last>`
                // fires but is the wrong shape for a whole-sequence capture, and fire_capture drops
                // it. Returning {} would read as "captured, all zeros" to a caller. Report the gap;
                // a request where NOTHING landed is an outright 400.
                //
                // This also BOUNDS cross-position path patching: the last layer cannot be held, so a
                // source whose influence is re-imported by final-layer attention cannot be measured
                // by patching a destination column (measured: 0% routed at every held depth for a
                // late-layer source -- notes/CIRCUIT_TRACER_DESIGN.md 5f).
                json missing = json::array();
                for (int L : cap_layers_armed)
                    if (!cap_json.contains(std::to_string(L))) missing.push_back(L);
                if (cap_json.empty()) {
                    res.status = 400;
                    res.set_content(json{{"error", "capture produced no rows for any requested layer; "
                                                   "the last layer (n_layer-1) materializes only the "
                                                   "logit rows (inp_out_ids), so whole-sequence capture "
                                                   "needs layer <= n_layer-2"},
                                         {"requested", capture_layers},
                                         {"armed", cap_layers_armed}}.dump(), "application/json");
                    return;
                }
                resp["captured"] = std::move(cap_json);
                resp["n_embd"] = cap_frame.n_embd;
                if (!missing.empty()) resp["capture_missing"] = std::move(missing);
            }
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
    // <|im_start|>, Llama-3 -> <|start_header_id|>, Gemma -> <start_of_turn>, ...) via the
    // pinned backend's full Jinja renderer. No context/KV
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
        if (!ctx.chat_templates.available()) {
            res.status = 400;
            res.set_content(json{{"error", "model has no embedded chat template; cannot format messages "
                                           "per-model (re-convert the GGUF with its tokenizer.chat_template, "
                                           "or send a pre-rendered prompt to /v1/completions)"}}.dump(),
                            "application/json");
            return;
        }
        std::vector<std::pair<std::string, std::string>> messages;
        messages.reserve(body["messages"].size());
        for (const auto& m : body["messages"]) {
            if (!m.is_object()) {
                res.status = 400;
                res.set_content(json{{"error", "each message must be an object with role + content"}}.dump(),
                                "application/json");
                return;
            }
            const std::string role = m.value("role", std::string());
            if (role.empty() || !m.contains("content") || !m["content"].is_string()) {
                res.status = 400;
                res.set_content(json{{"error", "each message must have a non-empty role and string content"}}.dump(),
                                "application/json");
                return;
            }
            messages.emplace_back(role, m["content"].get<std::string>());
        }
        const bool add_assistant = body.value("add_assistant", true);
        try {
            const std::string prompt = ctx.chat_templates.apply(messages, add_assistant);
            json resp = {{"prompt", prompt}, {"template_source", "model"},
                         {"renderer", "jinja"}};
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });
}

}  // namespace cloze
