"""Engine-backed chat + steering + model routing, and the two studio chat surfaces that log a run
(POST /say for the HF/qwen memory model, POST /denoise for the Dream diffusion window). `/engine/*` here
covers WRITE/generation calls (as opposed to the pure readouts in routes/readouts.py): observe (edit a
residual, watch the prediction move), the tone dials applied via the engine (axes + A/B check), and THE
HYBRID engine chat (GGUF generation with the HF-trained memory prefix/prompt-block injected). `/say` and
`/denoise` dispatch to whichever substrate is active (SUB.handle) and additionally log the run -- unlike
the fully generic SUB.handle(path, body) fallback (still in clozn.server.app's do_POST), these two shape
a specific response AND capture a trace/memory record for the Run Inspector. Mechanical extraction of the
matching branches out of do_POST; behavior unchanged. -> engine chat / substrate calls / model routing.

GET/POST /sampling/mode (REPRODUCE_AND_PROVE_PLAN S5): the interactive-chat sampling settings -- on/off +
temperature/top_p/top_k/repeat_penalty -- read by EngineSubstrate.chat/chat_stream on every turn. Mirrors
/timetravel/mode's shape (GET the live config; POST any subset of keys, get the resolved config + whether
anything changed back). Turning this off (or leaving it on -- either way) never touches the receipt/
replay/forced-scoring stack: those always pass sample=False ("greedy": True), which short-circuits before
this setting is even read (see EngineSubstrate.chat / _resolve_sampling).
"""
import time

from clozn.server import app as ctx


def try_get(h, p):
    if p == "/sampling/mode":   # S5: the live interactive-chat sampling settings (never mutates)
        h._json(200, ctx._sampling_settings())
        return True
    return False


def try_post(h, p, body):
    if p == "/sampling/mode":   # S5: adjust/toggle interactive-chat sampling (on/off + the 4 params)
        import clozn.memory.mode as memory_mode
        changed = False
        if "sampling" in body:
            memory_mode.set_setting("sampling", bool(body.get("sampling")))
            changed = True
        for key in ("sample_temperature", "sample_top_p", "sample_repeat_penalty"):
            if key in body:
                try:
                    memory_mode.set_setting(key, float(body[key]))
                    changed = True
                except (TypeError, ValueError):
                    h._json(400, {"error": f"{key} must be a number"})
                    return True
        if "sample_top_k" in body:
            try:
                memory_mode.set_setting("sample_top_k", int(body["sample_top_k"]))
                changed = True
            except (TypeError, ValueError):
                h._json(400, {"error": "sample_top_k must be an integer"})
                return True
        out = ctx._sampling_settings()
        out["changed"] = changed
        h._json(200, out)
        return True
    if p == "/engine/observe":   # WRITE a scaled residual back at one token, OBSERVE how the prediction moves
        try:
            pos = int(body.get("position", 0))
            scale = float(body.get("scale", 4.0))

            def tf(a):
                a = a.copy()
                if 0 <= pos < a.shape[0]:
                    a[pos] = a[pos] * scale
                return a

            hv, obs = ctx.ENGINE.edit_and_observe(str(body.get("text", ""))[:300], transform=tf, positions=[pos])
            h._json(200, {"summary": obs.summary(), "shifted": obs.shifted(),
                         "moved_l2": obs.moved_l2, "baseline_top": obs.baseline_top,
                         "edited_top": obs.edited_top, "tokens": hv.tokens,
                         "position": pos, "scale": scale})
        except Exception as e:
            h._json(502, {"error": f"engine: {e}"})
        return True
    if p == "/engine/steer/axes":   # the tone dials, but they apply on the GGUF via the engine
        from clozn.behavior.steering.axes import AXES
        es = ctx._engine_steer()
        h._json(200, {"axes": [{"name": k, "poles": AXES[k]["poles"]} for k in AXES],
                     "ready": bool(es and es.ready), "engine": bool(ctx.ENGINE_QWEN)})
        return True
    if p == "/engine/steer/check":   # A/B one dial on the engine GGUF: baseline vs steered generation
        es = ctx._engine_steer()
        if es is None:
            h._json(502, {"error": "no engine configured (set CLOZN_ENGINE_QWEN_PORT)"})
            return True
        try:
            prompt = str(body.get("prompt", "Tell me about the city at night."))[:300]
            axis, val = str(body.get("axis", "warm")), float(body.get("value", 1.0))
            mx = int(body.get("max_tokens", 60))
            base = es.generate(prompt, strength={}, max_new=mx)            # no dial = the baseline
            stee = es.generate(prompt, strength={axis: val}, max_new=mx)
            h._json(200, {"prompt": prompt, "axis": axis, "value": val,
                         "baseline": base.strip(), "steered": stee.strip()})
        except Exception as e:
            h._json(502, {"error": f"engine-steer: {e}"})
        return True
    if p == "/engine/chat":   # THE HYBRID: chat on the GGUF via the engine, with the HF-trained memory injected
        if ctx.ENGINE_QWEN is None:
            h._json(502, {"error": "no engine configured"})
            return True
        msgs = body.get("messages", [])
        t0 = time.time()
        memout = {}
        try:
            mx = int(body.get("max_tokens", 220))
            kw = {}
            mem = getattr(ctx.SUB, "memory", None) if ctx.SUB else None
            if ctx._memory_mode() == "prompt":
                # PROMPT MODE on the engine: the cards ride as the system block INSIDE the chat
                # template (compiled straight from the card store -- no HF model needed at all),
                # and the trained prefix is NOT injected. Strength maps to on/off; the topic gate
                # omits the block off-topic, exactly as on the HF path. This also means a FRESH
                # install (no trained prefix) finally gets memory over the engine.
                ms = float(getattr(mem, "memory_strength", 1.0)) if mem is not None \
                    else ctx._disk_memory()[1]
                block, applied, gate = ctx._prompt_block_for(mem, ctx._last_user(msgs), strength=ms)
                assembled = ctx._inject_block(msgs, block)
                memout.update(mode="prompt", applied=applied, gate=gate, strength=ms,
                              prompt_block=block, assembled_messages=assembled)
                prompt = ctx._engine_tmpl(ctx.ENGINE_QWEN, assembled)   # the loaded GGUF's own template, not Qwen ChatML
            else:
                prompt = ctx._engine_tmpl(ctx.ENGINE_QWEN, msgs)        # (the internalized-prefix path is Qwen-trained,
                #                                                  but the CHAT TEMPLATE is still the model's own)
                # MEMORY: the live HF prefix if a qwen substrate is loaded, else the SAVED prefix from
                # disk -- so engine-chat works with NO HF model resident (the pure-engine substrate).
                if mem is not None and getattr(mem, "prefix", None) is not None:
                    prefix = mem.prefix.detach().float().cpu()
                    ms = float(getattr(mem, "memory_strength", 1.0))
                else:
                    prefix, ms = ctx._disk_memory()
                # TOPIC RELEVANCE gate on the injection strength (mirror the HF chat's gate="auto"):
                # scale ms by how on-topic the last user turn is vs the active rules. Only when a LIVE
                # memory with rules is present (the qwen substrate) -- the pure-engine/disk path has no
                # rule texts to gate against, so it degrades to no-gating (rel==1.0), the prior
                # behavior. Defensive: any failure leaves ms unscaled.
                try:
                    if mem is not None and getattr(mem, "rules", None):
                        ms = ms * float(mem._gate(ctx._last_user(msgs)))
                except Exception:
                    pass
                if prefix is not None:             # inject the trained soft prefix (dial x relevance)
                    kw = {"prefix_embd": (prefix * ms).flatten().tolist(),
                          "prefix_rows": int(prefix.shape[0])}
            # backlog #5: record the EXACT rendered chat-template string the model saw (both memory
            # modes render one). _log_run reads memout["final_prompt"] -> the run record's final_prompt.
            memout["final_prompt"] = prompt
            # TONE: live dial values if a substrate is up, else the saved values from disk
            st = getattr(getattr(ctx.SUB, "steer", None), "strength", None) if ctx.SUB else None
            if not st:
                st = ctx._disk_dials()
            if st and any(st.values()):
                es = ctx._engine_steer()
                sv = es.steer_vector(st) if es is not None else None
                if sv:
                    kw["steer_vec"] = sv
                    kw["steer"] = {"coef": 1.0, "layer": es.layer}
            ctx._apply_anchored_memory(kw, memout, ctx._last_user(msgs))
            # Generate + capture a per-token trace alongside (B3). Reply is byte-identical to the
            # plain complete(); the trace feeds the Run Inspector timeline. steps=[] (diffusion, or a
            # stream hiccup) -> runlog stores a clean empty trace.
            reply_raw, steps, finish, _divinfo = ctx._engine_complete_traced(ctx.ENGINE_QWEN, prompt, mx, kw)
            reply = reply_raw.strip()
            # Pass the raw step list; runlog.record normalizes it -> {tokens, confidence, alternatives}.
            h._log_run("engine_chat", msgs, reply, "clozn-qwen (engine)", t0, trace=steps,
                       mem_out=memout, finish_reason=finish)
            # "memory" == did memory actually ride this reply (block in prompt mode, prefix otherwise)
            h._json(200, {"reply": reply,
                         "memory": bool(memout.get("applied")) or bool(kw.get("prefix_embd")),
                         "tone": bool(kw.get("steer_vec")), "via": "engine (GGUF)"})
        except Exception as e:
            h._log_run("engine_chat", msgs, "", "clozn-qwen (engine)", t0, error=str(e), mem_out=memout)
            h._json(502, {"error": f"engine-chat: {e}"})
        return True
    if p == "/say":   # studio chat (qwen memory model) -> capture it as a run
        if not (ctx.SUB and getattr(ctx.SUB, "handle", None)):
            h._json(409, {"error": f"'{p}' isn't served by the '{ctx.SUBNAME}' substrate"})
            return True
        msg = str(body.get("message", ""))
        t0 = time.time()
        # HF studio chat: capture a per-token trace (B3) + the per-turn memory record. We hand
        # SUB.handle collectors via body["_trace_out"] / body["_mem_out"] (server-side only, never
        # echoed); QwenSubstrate's /say fills them through say()/_say_prompt -> _generate's
        # pass-through recorder. Reply text is byte-identical with or without them.
        trace_steps = []
        body["_trace_out"] = trace_steps
        memout = {}
        body["_mem_out"] = memout
        try:
            r = ctx.SUB.handle(p, body)
        except Exception as e:
            h._log_run("studio_chat", [{"role": "user", "content": msg}], "",
                      "clozn-qwen", t0, error=str(e), mem_out=memout)
            h._json(500, {"error": f"{type(e).__name__}: {e}"})
            return True
        if r is None:
            h._json(409, {"error": f"'{p}' isn't served by the '{ctx.SUBNAME}' substrate",
                         "need": "qwen", "active": ctx.SUBNAME})
            return True
        # runlog.record normalizes the raw step list -> {tokens, confidence, alternatives}; a diffusion
        # substrate (or any path that filled nothing) yields [] -> a clean empty trace.
        h._log_run("studio_chat", [{"role": "user", "content": msg}],
                  str(r.get("reply", "")), "clozn-qwen", t0, trace=trace_steps, mem_out=memout)
        h._json(200, r)
        return True
    if p == "/denoise":   # Dream diffusion window -> capture it as a run
        if not (ctx.SUB and getattr(ctx.SUB, "handle", None)):
            h._json(409, {"error": f"'{p}' isn't served by the '{ctx.SUBNAME}' substrate",
                         "need": "dream", "active": ctx.SUBNAME})
            return True
        prompt = str(body.get("prompt", ""))
        t0 = time.time()
        try:
            r = ctx.SUB.handle(p, body)
        except Exception as e:
            h._log_run("denoise", [{"role": "user", "content": prompt}], "",
                      "clozn-dream", t0, error=str(e))
            h._json(500, {"error": f"{type(e).__name__}: {e}"})
            return True
        if r is None:
            h._json(409, {"error": f"'{p}' isn't served by the '{ctx.SUBNAME}' substrate",
                         "need": "dream", "active": ctx.SUBNAME})
            return True
        h._log_run("denoise", [{"role": "user", "content": prompt}],
                  str(r.get("final_text", "")), "clozn-dream", t0)
        h._json(200, r)
        return True
    return False
