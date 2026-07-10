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
}

}  // namespace cloze
