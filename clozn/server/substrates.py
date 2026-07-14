"""Model adapters shared by the product runtime and offline lab code.

``EngineSubstrate`` is the only product-serving adapter.  ``QwenSubstrate`` and
``DreamSubstrate`` remain as PyTorch lab implementations for training/calibration work;
the product gateway has no loader or switch route that can activate them.  The app module remains the seam:
mutable server state (SUB/SUBNAME/ENGINE*/SLOTS/...) and the helpers routes and tests patch live there,
and this module reads them through `ctx` (late-bound, so a monkeypatch on the app module is always
seen). app re-exports every public name here, so `from clozn.server import app as cs; cs.EngineSubstrate`
keeps working unchanged.
"""
from __future__ import annotations

import json
import os
import sys
import threading

from clozn.server.config import REPO_ROOT, DEMO                        # noqa: F401
from clozn.server import app as ctx   # the seam: live server state + patchable helpers (see docstring)

class Substrate:
    """Shared studio surface for any substrate: the /memory/* trait cards and the /steer/* tone dials, on
    whatever model the subclass loads. A subclass sets self.steer, self._mem (a memory object exposing
    .rules / .prefix / .consolidate(rules) / .reset()), self._pers_steer, self._steer_ready/_steer_info,
    and defines _gen(prompt) -- a one-shot generate used by the /steer/check A/B (AR generate vs denoise).
    So memory + dials are written ONCE and work identically on Qwen and Dream."""

    def _memory(self, path, body):
        """Card-backed memory (D2 + E1). Cards carry the metadata + review status; m.rules stays == the
        ACTIVE-card texts and drives the prefix via consolidate(). Status changes go through _mem_sync_rules,
        which only retrains when the active set actually moved -- so pending/no-op edits never touch the prefix."""
        import clozn.memory.cards as memory_cards
        m = self._mem
        self._ensure_cards_migrated()           # one-time seed of legacy rules -> active cards (no retrain)

        if path == "/memory/cards":             # OBJECTS now (not bare strings) -- the review layer
            return {"cards": memory_cards.list_cards(), "has_prefix": m.prefix is not None,
                    "mode": ctx._memory_mode(),     # the UI adapts its copy / hides retrain chrome on this
                    "retraining": ctx._retrain_status_mode()}   # fold the in-flight signal in (one reload sees it)

        if path == "/memory/retrain-status":    # the poll target: is a background consolidate() running?
            return ctx._retrain_status_mode()       # prompt mode: never ({active:false, mode:"prompt"})

        if path == "/memory/add":               # propose a card as PENDING -> does NOT affect the prefix
            text = str(body.get("text", "")).strip()
            if not text:
                return {"ok": False, "reason": "empty trait"}
            card = memory_cards.create(text, status="pending", kind="preference",
                                       risk=ctx._risk_of(text), source_run_id=body.get("source_run_id"),
                                       evidence=str(body.get("evidence", "")))
            if not card:
                return {"ok": False, "reason": "could not create card"}
            # If this is really a STYLE preference, surface the tone DIAL that delivers it (the trained
            # prefix carries topical prefs well but style ones weakly). Card is still created + pending;
            # this only SUGGESTS the better mechanism -- null when the text isn't a style match.
            return {**card, "dial_suggestion": ctx._dial_suggestion(text)}

        if path == "/memory/remove":            # delete by id -> if it was active, rebuild from the rest
            cid = str(body.get("id", "")).strip()
            if not cid:                          # (index removed -- ids are the stable handle now)
                return {"ok": False, "reason": "need a card id"}
            was_active = (memory_cards.get(cid) or {}).get("status") == "active"
            ok = memory_cards.delete(cid)
            if not ok:
                return {"ok": False, "reason": "no such card"}
            # delete is synchronous+fast; the retrain (only if an ACTIVE card left the set) is backgrounded.
            resync = ctx._start_retrain(m, "remove", cid) if was_active else {"retraining": False}
            return {"ok": True, "removed": cid, "resync": resync}

        if path in ("/memory/approve", "/memory/reject", "/memory/disable", "/memory/enable"):
            return self._card_status(path.rsplit("/", 1)[1], str(body.get("id", "")).strip())

        if path == "/memory/edit":              # change a card's text; if active, retrain on the new text
            cid = str(body.get("id", "")).strip()
            new_text = str(body.get("text", "")).strip()
            if not (cid and new_text):
                return {"ok": False, "reason": "need id and text"}
            card = memory_cards.update(cid, text=new_text, risk=ctx._risk_of(new_text))
            if card is None:
                return {"ok": False, "reason": "no such card"}
            if card.get("status") == "active":   # editing an active card's text retrains -> in the background
                card = {**card, "resync": ctx._start_retrain(m, "edit", cid)}
            return card

        if path == "/memory/strength":          # the memory dial. Internalized: scales how hard the prefix
            # bites (0 = off, >1 = stronger). PROMPT mode: on/off only -- 0 never injects the block, any
            # >0 injects when the topic gate lets it in (nothing scales continuously; the UI hint says so).
            if "value" in body and hasattr(m, "memory_strength"):
                m.memory_strength = max(0.0, min(2.0, float(body["value"])))
                if hasattr(m, "save"):
                    try:
                        m.save()                             # persists inside the .pt (needs a prefix)
                    except Exception:
                        pass
                try:                                         # mirror to settings so the dial survives a
                    import clozn.memory.mode as memory_mode                       # restart in prompt mode (no .pt to carry it)
                    memory_mode.set_setting("memory_strength", float(m.memory_strength))
                except Exception:
                    pass
            return {"strength": float(getattr(m, "memory_strength", 1.0)), "has_prefix": m.prefix is not None,
                    "mode": ctx._memory_mode()}

        if path == "/memory/gatecheck":         # DEBUG (live calibration): the topic-relevance gate for a prompt
            # Exposes both raw signals + the final gate + per-rule cosines for the active rules, so the bands
            # (lo_t/hi_t/lo_o/hi_o) can be tuned against real on/off-topic prompts. Fully guarded: on any
            # failure (or no embedder) it reports the no-gating baseline (gate 1.0) rather than raising.
            prompt = str(body.get("prompt", ""))
            rules = list(getattr(m, "rules", []) or [])
            try:
                from clozn.memory.topic_gate import get_gate
                dbg = get_gate().debug(prompt, rules)
            except Exception as e:
                dbg = {"gate": 1.0, "topic": 0.0, "openness": 0.0, "relevance": {},
                       "ok": False, "error": f"{type(e).__name__}: {e}"}
            # `gate` here is the RAW topic gate (relevance only); the applied scale is memory_strength x
            # gate (internalized) or include-iff gate >= ctx.PROMPT_GATE_MIN (prompt mode) -- mode says which.
            return {"prompt": prompt, "rules": rules, "mode": ctx._memory_mode(),
                    "strength": float(getattr(m, "memory_strength", 1.0)), **dbg}
        return None

    # ---- E1 review lifecycle: a status change rebuilds m.rules from the active set, retrains iff it moved -
    def _card_status(self, action, cid):
        """approve->active, reject->rejected, disable->disabled, enable->active. The STATUS flip (fast) is
        synchronous; the RETRAIN it may trigger (rebuild the prefix from active_texts) is backgrounded so
        the response returns immediately. The card keeps its FINAL status; a separate _RETRAIN flag carries
        the in-flight signal. ctx._start_retrain no-ops when the active set didn't actually move (prefix safe).

        PROVENANCE GATE: 'approve' is refused for a card that CLAIMS a run (source_run_id
        set) but carries no quoted_span to back that claim up -- memory_cards.is_provenance_claim_unbacked.
        This is never auto-approvable; the reviewer sees why via the reason string (the Memory page also
        flags it so this should rarely even be attempted). reject/disable/enable are NOT gated -- you must
        always be able to discard or de-activate a card regardless of its provenance."""
        import clozn.memory.cards as memory_cards
        if not cid:
            return {"ok": False, "reason": "need a card id"}
        if action == "approve":
            existing = memory_cards.get(cid)
            if existing is not None and memory_cards.is_provenance_claim_unbacked(existing):
                return {"ok": False, "reason": "no provenance -- this card cites a run but has no quoted "
                                                "span backing it up, so it can't be approved"}
        target = {"approve": "active", "reject": "rejected",
                  "disable": "disabled", "enable": "active"}[action]
        card = memory_cards.set_status(cid, target)
        if card is None:
            return {"ok": False, "reason": "no such card"}
        resync = ctx._start_retrain(self._mem, action, cid)  # retrains on a thread iff the active set changed
        return {**card, "resync": resync}

    def _ensure_cards_migrated(self):
        """Seed the card store from this substrate's legacy rule-strings exactly once per process."""
        if getattr(self, "_cards_migrated", False):
            return
        ctx._mem_migrate(self._mem)
        self._cards_migrated = True

    def _ensure_steer(self):
        """Compute the axis vectors once, race-safe (double-checked lock). Two dial calls racing on first
        use could otherwise both run compute() on the shared model at once and corrupt it (IndexError)."""
        if not self._steer_ready:
            with self._steer_lock:
                if not self._steer_ready:
                    self._steer_info = self.steer.compute()
                    self._steer_ready = True

    def _steer(self, path, body):
        from clozn.behavior.steering.axes import AXES
        if path == "/steer/axes":
            calib = ctx._dial_calibration()   # {} when uncalibrated/offline -- ctx._with_calibration no-ops per axis
            axes = [ctx._with_calibration(
                        {"name": k, "poles": AXES[k]["poles"], "value": self.steer.strength.get(k, 0.0),
                         "max": AXES[k].get("max", 1.5)}, calib.get(k))
                    for k in AXES]
            lib_names = ctx._library_dial_names()   # shipped-library custom dials -- NOT user-made, never "yours"
            for k, v in getattr(self.steer, "custom", {}).items():   # user-defined + shipped-library dials
                axis = {"name": k, "poles": v["poles"], "value": self.steer.strength.get(k, 0.0), "max": v["max"]}
                if k in lib_names:
                    axis["library"] = True     # shipped, curated dial -- distinct from a user's own custom
                else:
                    axis["custom"] = True      # unchanged: a genuine user-made dial ("yours" + deletable)
                axes.append(ctx._with_calibration(axis, calib.get(k)))
            return {"axes": axes, "ready": self._steer_ready, "substrate": self.name}
        self._ensure_steer()                    # compute the axis vectors once on first real use (race-safe)
        if path == "/steer/compute":
            return {"ready": True, **self._steer_info}
        if path == "/steer/set":
            self.steer.set(str(body["name"]), float(body.get("value", 0.0)))
            self.steer.save_state(self._pers_steer)
            return {"active": self.steer.active()}
        if path == "/steer/check":              # A/B one dial: baseline vs steered (subclass _gen)
            prompt = str(body.get("prompt", ""))[:300]
            base = self._gen(prompt)
            self.steer.clear()
            self.steer.set(str(body["name"]), float(body.get("value", 1.0)))
            self.steer.engage()
            try:
                steered = self._gen(prompt)
            finally:
                self.steer.disengage()
                self.steer.clear()
            return {"prompt": prompt, "axis": body.get("name"), "value": body.get("value", 1.0),
                    "baseline": base, "steered": steered}
        if path == "/steer/custom":             # USER-DEFINED dial: compute mean(+pole)-mean(-pole) live
            if not hasattr(self.steer, "add_custom"):
                return {"error": "custom dials are not supported on this substrate yet"}
            name = str(body.get("name", "")).strip()[:24]
            pos, neg = str(body.get("pos", "")).strip(), str(body.get("neg", "")).strip()
            if not (name and pos and neg):
                return {"error": "need a name and both poles (pos, neg)"}
            info = self.steer.add_custom(name, pos, neg, float(body.get("max", 0.5)))
            self.steer.save_custom(ctx._pers(f"studio_custom_{self.name}.json"))
            return {"name": name, "max": info["max"], "custom": list(self.steer.custom)}
        if path == "/steer/custom_delete":
            if hasattr(self.steer, "remove_custom"):
                self.steer.remove_custom(str(body.get("name", "")))
                self.steer.save_custom(ctx._pers(f"studio_custom_{self.name}.json"))
                self.steer.save_state(self._pers_steer)
            return {"custom": list(getattr(self.steer, "custom", {}))}
        if path == "/steer/concept/set":         # Tier-1 #1: any-concept dial (dir(c)) -- ZERO calibration
            import clozn.behavior.steering.concept_dir as concept_dir
            cs = ctx._engine_concept_steer()
            if cs is None:
                return {"error": "concept dials need the product model worker (CLOZN_ENGINE_PORT)"}
            concept = str(body.get("concept", "")).strip()
            if not concept:
                return {"error": "need a concept word"}
            strength = float(body.get("strength", concept_dir.DEFAULT_STRENGTH))
            result = cs.steer_toward(concept, strength)
            result["active"] = cs.active()
            return result
        if path == "/steer/concept/check":       # A/B: baseline vs dir(concept)-steered (mirrors /steer/check)
            import clozn.behavior.steering.concept_dir as concept_dir
            cs = ctx._engine_concept_steer()
            if cs is None:
                return {"error": "concept dials need the product model worker (CLOZN_ENGINE_PORT)"}
            concept = str(body.get("concept", "")).strip()
            if not concept:
                return {"error": "need a concept word"}
            strength = float(body.get("strength", concept_dir.DEFAULT_STRENGTH))
            prompt = str(body.get("prompt", ""))[:300]
            max_new = int(body.get("max_new", 90))
            base = concept_dir._text_of(cs.ec.complete(prompt, max_tokens=max_new))
            built = cs.steer_toward(concept, strength)
            if not built.get("ok"):
                return {"prompt": prompt, "concept": concept, "strength": strength,
                        "baseline": base, "steered": None,
                        "blocked": built.get("blocked"), "note": built.get("note")}
            steered = concept_dir._text_of(cs.ec.intervene(
                prompt, vector=built["vector"], coef=built["coef"], layer=built["layer"], max_tokens=max_new))
            return {"prompt": prompt, "concept": concept, "strength": strength, "layer": built["layer"],
                    "token_id": built["token_id"], "coef": built["coef"],
                    "baseline": base, "steered": steered, "note": built.get("note")}
        return None


class QwenSubstrate(Substrate):
    """One Qwen-7B + SAE behind the concept readout AND the memory + tone dials."""
    name = "qwen"

    def __init__(self):
        from clozn.readouts.brain import BrainReadout
        from clozn.readouts.sae7b import GpuSAE, load7b
        from clozn.substrates.self_teach import SelfTeach
        from clozn.behavior.steering.hf_adapter import SteeringControl
        sae = GpuSAE()
        tok, model = load7b()
        self.brain = BrainReadout(model, tok, sae, DEMO, os.path.join(REPO_ROOT, "research"))
        self.memory = SelfTeach("Qwen/Qwen2.5-7B-Instruct", model=model, tok=tok,   # shares the model
                                persist_path=ctx._pers("studio_memory.pt"))
        self.steer = SteeringControl(model, tok)            # tone dials on the same model
        self._mem = self.memory
        self._steer_ready, self._steer_info, self._steer_lock = False, {}, threading.Lock()
        self._pers_steer = ctx._pers("studio_personality.json")
        self.steer.load_state(self._pers_steer)             # restore the personality dials across restarts
        self.steer.load_custom(ctx._pers(f"studio_custom_{self.name}.json"))    # + any user-defined dials
        # + the shipped dial library (research/deploy_dial_library.py), if it has ever been deployed on
        # this install -- a no-op (load_custom returns immediately) until that script runs. Loaded SECOND
        # so a rare --force-deployed name collision (a user custom + a library dial sharing a name)
        # resolves to the library's direction on every subsequent boot, matching what --force just did live.
        self.steer.load_custom(ctx._pers("studio_library.json"))
        if ctx._memory_mode() == "prompt":
            # PROMPT MODE boots from the CARD STORE (the prefix isn't applied): adopt the active-card
            # texts as m.rules right away so /state + runlog bookkeeping don't lag until the first
            # /memory call. sync_cards never touches the prefix; it also runs the one-time migration.
            self.memory.sync_cards()
            self._cards_migrated = True
            try:                                            # in prompt mode the strength dial persists in
                import clozn.memory.mode as memory_mode                          # settings (the .pt needs a prefix to save; a
                s = memory_mode.get_setting("memory_strength")            # fresh install has none)
                if s is not None:
                    self.memory.memory_strength = max(0.0, min(2.0, float(s)))
            except Exception:
                pass

    def handle(self, path, body):
        if path == "/think":
            return self.brain.think(str(body.get("text", ""))[:500], str(body.get("sid", "default")))
        if path == "/concepts":                 # read what fired inside (no generation) -> annotate a reply
            return self.brain.concepts_only(str(body.get("text", ""))[:500])
        if path == "/say":
            with ctx._TRAIN_LOCK:                    # studio chat touches the shared model -> wait out a retrain
                # body["_trace_out"] / body["_mem_out"] (optional, server-side only): collectors the handler
                # passes for the Run Inspector trace + the per-turn memory record; never echoed to the client.
                if ctx._memory_mode() == "prompt":
                    return {"reply": self._say_prompt(body["message"], body.get("max_new", 200),
                                                      trace_out=body.get("_trace_out"),
                                                      mem_out=body.get("_mem_out"))}
                return {"reply": self.memory.say(body["message"], body.get("max_new", 200),
                                                 trace_out=body.get("_trace_out"))}
        if path == "/consolidate":               # a manual retrain -> the same shared-model lock as card retrains
            with ctx._TRAIN_LOCK:
                return self.memory.consolidate(body.get("rules"), body.get("steps", 120), body.get("lr", 0.012),
                                               body.get("n_probe", 8), body.get("max_norm", 14.0))
        if path == "/whatlearned":
            if ctx._memory_mode() == "prompt":
                return self._whatlearned_prompt()
            return {"report": self.memory.what_learned(), "mode": "internalized"}
        if path == "/check":                     # generates on the shared model -> wait out a retrain
            with ctx._TRAIN_LOCK:
                if ctx._memory_mode() == "prompt":
                    return self._check_prompt(body["prompt"], body.get("max_new", 200))
                return self.memory.check(body["prompt"], body.get("max_new", 200))
        if path == "/reset":
            with ctx._TRAIN_LOCK:                     # mutates the prefix/model state -> don't race a retrain
                self.brain.reset(str(body.get("sid", "default")))
                return self.memory.reset(body.get("keep_prefix", False))
        if path.startswith("/memory/"):
            return self._memory(path, body)
        if path.startswith("/steer/"):
            return self._steer(path, body)
        return None

    # ---- prompt-mode twins of the SelfTeach conversation endpoints ---------------------------------
    # Same surfaces, same shapes -- but memory rides as the gated system block and the model runs
    # prefix-free (use_prefix=False). SelfTeach itself is untouched: it stays the internalized-mode
    # engine and the research instrument (the self-audit experiments REQUIRE a non-text memory).

    def _say_prompt(self, message, max_new=200, trace_out=None, mem_out=None):
        """One /say turn in prompt mode: history grows exactly as SelfTeach.say, but the memory is the
        compiled block (topic-gated on THIS user turn). Runs under the caller's ctx._TRAIN_LOCK; takes
        m.lock like say() so concurrent history appends can't interleave."""
        m = self.memory
        with m.lock:
            m.history.append({"role": "user", "content": message})
            block, applied, gate = ctx._prompt_block_for(m, message)
            assembled = ctx._inject_block(m.history, block)
            if mem_out is not None:
                mem_out.update(mode="prompt", applied=applied, gate=gate,
                               prompt_block=block, assembled_messages=assembled)
            reply = m._generate(assembled, use_prefix=False, max_new=max_new,
                                sample=True, trace_out=trace_out)
            m.history.append({"role": "assistant", "content": reply})
            return reply

    def _whatlearned_prompt(self):
        """Prompt-mode /whatlearned: ask from a fresh context WITH the block injected, ungated (the
        self-view shows the full memory, mirroring what_learned's apply_gate=False). Honesty: in this
        mode the model is READING its cards out of context, not introspecting a trained prefix -- the
        `mode` field is there so the UI labels it as reading, not self-knowledge."""
        m = self.memory
        cards = ctx._prompt_mem_cards(m)
        if not cards:
            return {"report": "(no active memory cards yet -- add or approve one on the Memory page)",
                    "mode": "prompt"}
        import clozn.memory.mode as memory_mode
        block = memory_mode.compile_prompt_block([c["text"] for c in cards])
        ask = ("What have you picked up about me so far -- my interests, anything I seem to care about, "
               "and how I like you to respond? List what you know, one item per line.")   # == what_learned's
        with m.lock:
            report = m._generate([{"role": "system", "content": block}, {"role": "user", "content": ask}],
                                 use_prefix=False, max_new=200, sample=False)
        return {"report": report, "mode": "prompt"}

    def _check_prompt(self, prompt, max_new=200):
        """Prompt-mode /check, mirroring check()'s response shape: baseline vs block-in-context. The
        block is binary per turn, so `ungated` == block always in, and `gated` == what a real chat does
        (identical when the gate lets it in, the plain baseline when the topic gates it out -- greedy
        decode makes the reuse exact, no second generation needed)."""
        m = self.memory
        with m.lock:
            msgs = [{"role": "user", "content": prompt}]
            base = m._generate(msgs, use_prefix=False, max_new=max_new, sample=False)
            cards = ctx._prompt_mem_cards(m)
            if not cards:
                return {"prompt": prompt, "gate": None, "baseline": base, "mode": "prompt",
                        "ungated": "(no active memory cards)", "gated": "(no active memory cards)"}
            texts = [c["text"] for c in cards]
            import clozn.memory.mode as memory_mode
            block = memory_mode.compile_prompt_block(texts)
            g = round(ctx._prompt_gate(prompt, texts), 3)
            ungated = m._generate(ctx._inject_block(msgs, block), use_prefix=False, max_new=max_new, sample=False)
            gated = ungated if g >= ctx.PROMPT_GATE_MIN else base
            return {"prompt": prompt, "gate": g, "baseline": base, "ungated": ungated, "gated": gated,
                    "mode": "prompt"}

    def _gen(self, prompt):                     # AR generate for the /steer/check A/B
        return self.steer.generate(prompt, 90)

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        """One stateless chat completion with the WHOLE tunable self applied: the memory (as the trained
        prefix in internalized mode, as the topic-gated system block in prompt mode) AND the active
        tone-steering sliders, both on the shared model. This is what the OpenAI-compatible endpoint
        serves -- normal chat on the surface, legible and tunable underneath. Serializes behind an
        in-flight memory retrain (ctx._TRAIN_LOCK) so a reply can't race the shared model+gradients
        mid-consolidate -- it waits, briefly, rather than corrupting.
        trace_out (optional list): filled with the per-token trace for the Run Inspector; reply unchanged.
        mem_out (optional dict): prompt mode fills {mode, applied, gate} -- what memory ACTUALLY rode
        this turn -- so the run log records per-turn application, not just the active set."""
        with ctx._TRAIN_LOCK:                        # wait out any background retrain, then hold for this reply
            self._last_generation_meta = ctx._qwen_generation_meta(max_new, sample=sample, stream=False)
            self._last_finish_reason = None
            if self.steer.strength:             # persisted personality -> ensure vectors are ready (race-safe)
                self._ensure_steer()
            self.steer.engage()
            try:
                if ctx._memory_mode() == "prompt":
                    # PROMPT MODE: the cards ride as the system block (omitted when the topic gate says
                    # this turn is off-memory); the model runs prefix-free. The block wording is the
                    # exact distillation target the prefix trains toward, so behavior stays comparable.
                    block, applied, gate = ctx._prompt_block_for(self.memory, ctx._last_user(messages))
                    assembled = ctx._inject_block(messages, block)
                    if mem_out is not None:
                        mem_out.update(mode="prompt", applied=applied, gate=gate,
                                       prompt_block=block, assembled_messages=assembled)
                    reply = self.memory._generate(assembled, use_prefix=False,
                                                  max_new=max_new, sample=sample, trace_out=trace_out)
                    self._last_finish_reason = getattr(self.memory, "_last_finish_reason", None)
                    return reply
                # gate="auto" -> _generate scales the memory prefix by memory_strength x TOPIC RELEVANCE, so
                # the OpenAI /v1/chat path gets the same on-topic gating as /say (fixes the always-on
                # over-bleed). memory_strength 0 still zeroes it; a missing embedder falls back to no-gating.
                reply = self.memory._generate(messages, use_prefix=True, max_new=max_new, sample=sample,
                                              gate="auto", trace_out=trace_out)
                self._last_finish_reason = getattr(self.memory, "_last_finish_reason", None)
                return reply
            finally:
                self.steer.disengage()

    def last_finish_reason(self):
        return getattr(self, "_last_finish_reason", None)

    def run_meta(self):
        return dict(getattr(self, "_last_generation_meta", None) or
                    ctx._qwen_generation_meta(sample=True))

    def last_stream_trace(self):
        """The per-token trace captured during the most recent chat_stream (raw step list, or []). The SSE
        handler reads this AFTER the generator is exhausted to log it -- streaming yields text, not tokens,
        so the trace is assembled from the recorder's rows + the generated ids, not from the chunks."""
        return list(getattr(self, "_last_stream_trace", []) or [])

    def chat_stream(self, messages, max_new=256, mem_out=None):
        """Streaming chat: yields text chunks as the AR model generates -- memory + tone steering
        applied -- via a TextIteratorStreamer with generate() in a thread. Local AR is slow, so this is
        the big UX win the diffusion side doesn't need (diffusion is trace-based, not left-to-right).
        mem_out: as in chat() -- prompt mode records what memory actually rode this turn.

        Per-token trace (B3): a pure pass-through RecordingLogitsProcessor rides along and the generated
        ids are captured from generate()'s return -> after the stream ends we assemble the trace into
        self._last_stream_trace for the SSE handler to log. Pass-through means the streamed chunks are
        byte-identical to before; the whole capture is wrapped so any failure just leaves the trace empty."""
        import threading
        import torch
        from transformers import TextIteratorStreamer
        ctx._TRAIN_LOCK.acquire()                    # serialize behind an in-flight retrain (released in finally)
        self._last_generation_meta = ctx._qwen_generation_meta(max_new, sample=True, stream=True)
        self._last_finish_reason = None
        if self.steer.strength:
            self._ensure_steer()
        m = self.memory
        if ctx._memory_mode() == "prompt":
            # PROMPT MODE: the gated system block replaces the prefix concat below -- it simply becomes
            # part of the chat template, so the streaming mechanics are untouched.
            block, applied, gate = ctx._prompt_block_for(m, ctx._last_user(messages))
            assembled = ctx._inject_block(messages, block)
            if mem_out is not None:
                mem_out.update(mode="prompt", applied=applied, gate=gate,
                               prompt_block=block, assembled_messages=assembled)
            e = m._embed(m._chat_ids(assembled))
        else:
            e = m._embed(m._chat_ids(messages))
            if m.prefix is not None:                        # prepend the consolidated memory prefix, scaled by
                # memory_strength x TOPIC RELEVANCE (same on-topic gating as _generate's gate="auto"; this path
                # inlines generate() for streaming so it can't call _generate, but the scale must match). A
                # missing embedder makes rel==1.0 (no gating); memory_strength 0 zeroes the prefix entirely.
                last_user = next((mm["content"] for mm in reversed(messages) if mm.get("role") == "user"), "")
                g = m.memory_strength * m._gate(last_user)
                e = torch.cat([(g * m.prefix.detach()).to(e.dtype)[None], e], 1)
        att = torch.ones(e.shape[:2], device=e.device, dtype=torch.long)
        streamer = TextIteratorStreamer(m.tok, skip_prompt=False, skip_special_tokens=True)
        kw = dict(inputs_embeds=e, attention_mask=att, max_new_tokens=max_new, do_sample=True,
                  temperature=0.7, top_p=0.9, repetition_penalty=1.3, no_repeat_ngram_size=3,
                  pad_token_id=m.eos or 0, streamer=streamer)            # trim steering-induced loops
        self._last_stream_trace = []                        # reset; filled after the stream if capture succeeds
        recorder = None
        try:                                                # observe-only trace capture (never affects output)
            from clozn.substrates.qwen import RecordingLogitsProcessor
            from transformers import LogitsProcessorList
            recorder = RecordingLogitsProcessor()
            kw["logits_processor"] = LogitsProcessorList([recorder])
        except Exception:
            recorder = None
        gen_out = {}                                        # holder so the thread can hand back generate()'s ids

        def _gen():
            with torch.no_grad():
                out = m.model.generate(**kw)
                try:
                    gen_out["ids"] = [int(t) for t in out[0].tolist()]   # inputs_embeds -> generated ids only
                except Exception:
                    pass

        self.steer.engage()                                 # tone dials apply during the streamed generation
        th = threading.Thread(target=_gen, daemon=True)
        th.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            th.join()
            raw_ids = list(gen_out.get("ids", []) or [])
            try:
                from clozn.substrates.qwen import finish_reason_from_generated_ids
                self._last_finish_reason = finish_reason_from_generated_ids(raw_ids, m.eos, max_new)
            except Exception:
                self._last_finish_reason = None
            if recorder is not None:                        # assemble the trace from rows + emitted ids
                try:
                    from clozn.substrates.qwen import steps_from_records
                    gen_ids = list(raw_ids)
                    while gen_ids and gen_ids[-1] == (m.eos or -1):
                        gen_ids.pop()
                    self._last_stream_trace = steps_from_records(recorder.records, gen_ids, m.tok)
                except Exception:
                    self._last_stream_trace = []
            self.steer.disengage()
            ctx._TRAIN_LOCK.release()                           # done streaming -> let a queued retrain proceed

    def state(self):
        return self.memory.state()


class DreamSubstrate(Substrate):
    """Dream-7B diffusion: the denoise window, plus the SAME trait-card memory and tone dials as Qwen."""
    name = "dream"

    def __init__(self):
        from cloze_lab.cli import build_adapter
        from clozn.substrates.denoise import trace_for
        from clozn.behavior.steering.dream_adapter import DreamSteering
        from clozn.substrates.dream_memory import DreamMemory
        self.adapter = build_adapter("dream", device="cuda", quant="nf4")
        self._trace = trace_for
        self.steer = DreamSteering(self.adapter)            # tone dials on the diffusion model
        self._steer_ready, self._steer_info, self._steer_lock = False, {}, threading.Lock()
        self._pers_steer = ctx._pers("studio_dream_personality.json")
        self.steer.load_state(self._pers_steer)
        self.dmem = DreamMemory(self.adapter,               # diffusion-native memory (trained soft prefix)
                                persist_path=ctx._pers("studio_dream_memory.pt"))
        self._mem = self.dmem

    def handle(self, path, body):
        if path == "/denoise":
            prompt = str(body.get("prompt", ""))[:300]
            with ctx._TRAIN_LOCK:                              # wait out a background retrain (it moves dmem.prefix)
                self.steer.engage()                        # active dials steer every denoising pass
                try:
                    ad = self.adapter
                    # In PROMPT mode the trained dream prefix is NOT applied: cards may have been edited
                    # instantly (no consolidate), so the prefix can be stale vs the cards -- and denoise
                    # is a raw completion window with no system slot for the block. Memory simply doesn't
                    # ride here in prompt mode (honest omission beats a stale injection).
                    if self.dmem.prefix is not None and ctx._memory_mode() != "prompt":
                        from clozn.substrates.dream_memory import PrefixAdapter   # memory present -> inject into the REAL scheduler
                        ad = PrefixAdapter(self.adapter, self.dmem.prefix.detach())
                    return self._trace(ad, prompt)         # the cloze_lab scheduler (+ the steering hook)
                finally:
                    self.steer.disengage()
        if path.startswith("/memory/"):
            return self._memory(path, body)
        if path.startswith("/steer/"):
            return self._steer(path, body)
        return None

    def _gen(self, prompt):                                # base denoise final text for the /steer/check A/B
        return self._trace(self.adapter, str(prompt)[:200])["final_text"]

    def state(self):
        return {"dials": self.steer.active(), "cards": self.dmem.rules}


class _EngineMemory:
    """Thin prompt-mode memory for the engine substrate: the CARD STORE *is* the memory. No model, no
    learned prefix (the soft-prefix TTT is a lab experiment now, not shipped in the engine product -- see
    RUNTIME_SPLIT.md). Exposes exactly the surface the base Substrate._memory handler, ctx._prompt_block_for,
    and the receipts/replay stack read: .rules (active-card texts), .prefix (always None), .memory_strength,
    ._exclude_card_ids (replay sets this for per-card receipts), .consolidate/.reset (no-ops -- prompt mode
    never trains), .state(), .lock."""

    def __init__(self):
        self.prefix = None
        self._exclude_card_ids = None
        self.lock = threading.Lock()
        try:
            import clozn.memory.mode as memory_mode                    # 0.35 == the shipped product default (commit f3e9f60, the
            self.memory_strength = float(memory_mode.get_setting("memory_strength", 0.35))   # over-bleed fix);
        except Exception:                          # matches SelfTeach.__init__ so a fresh engine-first boot
            self.memory_strength = 0.35            # doesn't diverge. (Prompt mode treats it as on/off anyway.)

    @property
    def rules(self):
        import clozn.memory.cards as memory_cards
        return [c["text"] for c in (memory_cards.list_cards() or []) if c.get("status") == "active"]

    @rules.setter
    def rules(self, _value):
        # The card store IS the memory here, so `rules` is derived and has nothing to set. The shared
        # _mem_sync_rules() assigns m.rules for the soft-prefix (SelfTeach) backend; make that a harmless
        # no-op on the engine substrate instead of an AttributeError -- otherwise every approve/reject/
        # disable/enable/remove crashed AFTER already mutating the store (scrappy error toast, action
        # silently succeeded). The store stays the single source of truth.
        pass

    def consolidate(self, rules):
        return {"ok": True, "mode": "prompt"}      # prompt mode never trains a prefix

    def reset(self):
        pass

    def state(self):
        import clozn.memory.cards as memory_cards
        return {"mode": "prompt", "has_prefix": False,
                "cards": len(memory_cards.list_cards() or []), "rules": self.rules}


class EngineSubstrate(Substrate):
    """PURE-ENGINE substrate: chat + prompt-mode memory + tone dials on the C++ GGUF runtime, NO PyTorch
    model resident. THIS is the class that brings the whole torch-free Server tier -- /v1/chat/completions,
    replay, receipts, explain, narrate, counterfactual -- onto the fast engine, because every one of those
    routes through SUB.chat(). Memory is prompt-mode only (the card store as a topic-gated system block);
    dials apply via EngineSteer's steer_vec. See RUNTIME_SPLIT.md (the keystone)."""

    name = "engine"

    def __init__(self):
        if ctx.ENGINE is None:
            raise RuntimeError("engine substrate needs the supervised GGUF worker (set CLOZN_ENGINE_PORT)")
        self.engine = ctx.ENGINE
        self.steer = ctx._engine_steer()            # an EngineSteer on the GGUF (tone dials via steer_vec)
        if self.steer is not None:               # metadata-only: the shipped library's names/poles/max, so
            try:                                  # they show up in /steer/axes immediately (their direction
                self.steer.load_library(ctx._pers("studio_library.json"))   # vectors are computed lazily by compute())
                self.steer.load_custom(ctx._pers(f"studio_custom_{self.name}.json"))  # + the user's own custom dials
            except Exception:
                pass
        self._mem = _EngineMemory()
        self.memory = self._mem                 # the studio reads SUB.memory in a few places
        self._pers_steer = ctx._pers("studio_personality.json")
        self._steer_ready = False
        self._steer_info = {}
        self._steer_lock = threading.Lock()
        self.brain = None                       # no SAE/brain on the pure-engine substrate (concepts 409 cleanly)
        # T0.2: reflect the ACTUALLY-LOADED GGUF, not a hardcoded Qwen assumption. Derive the family from
        # the engine's /health model file (best-effort -- never blocks boot if the engine isn't up yet)
        # and pin the tone-dial steer tap to THIS model's mid-depth: Qwen-7B -> 14 (unchanged), Llama-3.2-1B
        # -> 8, an unrecognized GGUF keeps EngineSteer's generic default. run_meta() re-derives this lazily
        # too, so the run record is correct even when the engine comes up after the substrate.
        self.model_family = None
        self.model_id = None
        try:
            h = self.engine.health() if (self.engine and hasattr(self.engine, "health")) else {}
            self.model_family, _info = _engine_model_info((h or {}).get("model", ""))
            self.model_id = _info["model_id"]
            if self.steer is not None and _info["steer_layer"] is not None:
                self.steer.layer = _info["steer_layer"]
        except Exception:
            pass
        if self.steer is not None:              # restore persisted dial values (shared personality.json)
            try:
                self.steer.load_state(self._pers_steer)
            except Exception:
                pass

    def _gen(self, prompt):                     # one-shot generate for the /steer/check A/B (base _steer)
        if self.steer is not None:
            return self.steer.generate(prompt, max_new=90)
        from clozn.behavior.steering.engine_adapter import EngineSteer
        return EngineSteer._text(self.engine.complete(prompt, max_tokens=90))

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None,
             reference_tokens=None, apply_anchored=False):
        """One stateless chat completion on the engine with memory (prompt-mode card block) + tone dials
        applied. Mirrors QwenSubstrate.chat's contract EXACTLY (same signature, same trace_out/mem_out
        fill) so the receipts/replay stack is backend-agnostic.

        `sample`: the caller's request to sample (True) or force greedy (False). REPRODUCE_AND_PROVE_PLAN
        S5: `sample=True` (the default, and what interactive chat -- openai.py's /v1/chat/completions --
        passes) resolves via ctx._resolve_sampling against the persisted "sampling" setting (default ON,
        Ollama/llama.cpp's canonical temperature=0.8/top_p=0.9/top_k=40/repeat_penalty=1.1, a FRESH seed
        every turn); the setting off degrades to greedy, byte-identical to pre-S5 behavior.
        `sample=False` ALWAYS decodes greedy (temperature 0) regardless of the setting -- this is what the
        receipt/replay/forced-scoring stack relies on: it forces the STORED token ids over this generation
        (replay.py passes `sample=False` for every `{"greedy": True}` change spec, which every receipt
        path uses), so a sampled interactive run's receipts are computed exactly like a greedy run's.

        `reference_tokens` (optional): the baseline reply's committed token ids. When present, the engine
        EARLY-STOPS this generation at the first token that differs from the reference (prove-all ablated
        arms) -- so the reply is a bit-exact PREFIX of what full generation would produce, plus a divergence
        verdict stashed for last_divergence(). This is a pure termination check -- decode/sampling (greedy
        or not) are otherwise untouched, so a diverged reply is still a bit-exact prefix either way."""
        # `apply_anchored` is explicit so live OpenAI chat can use X7 anchored memory, while receipts/replay
        # keep the pre-existing deterministic baseline unless they intentionally opt in.
        samp = ctx._resolve_sampling(sample)
        self._last_generation_meta = ctx._engine_generation_meta(max_new, stream=False, sample=samp)
        self._last_diverged = None
        self._last_diverged_at = None
        # MEMORY: the active cards as a topic-gated system block (omitted off-topic / when strength 0).
        block, applied, gate = ctx._prompt_block_for(self.memory, ctx._last_user(messages))
        assembled = ctx._inject_block(messages, block)
        prompt = ctx._engine_tmpl(self.engine, assembled)   # per-model template (the loaded GGUF's own), not Qwen ChatML
        if mem_out is not None:
            # final_prompt = the EXACT rendered string the model saw (backlog #5); assembled_messages is its
            # pre-template form. Both recorded so the run is inspectable at either level.
            mem_out.update(mode="prompt", applied=applied, gate=gate,
                           prompt_block=block, assembled_messages=assembled, final_prompt=prompt)
        # TONE: dials from self.steer.strength (replay toggles this in place), falling back to disk.
        kw = {}
        st = (getattr(self.steer, "strength", None) if self.steer is not None else None) or ctx._disk_dials()
        if self.steer is not None and st and any(st.values()):
            sv = self.steer.steer_vector(st)
            if sv:
                kw["steer_vec"] = sv
                kw["steer"] = {"coef": 1.0, "layer": self.steer.layer}
        comp = ctx._apply_anchored_memory(kw, mem_out, ctx._last_user(messages)) if apply_anchored else None
        if reference_tokens:                                # prove-all early-stop: halt when the answer changes
            kw["reference_tokens"] = [int(t) for t in reference_tokens if t is not None]
        reply_raw, steps, finish, divinfo = ctx._engine_complete_traced(self.engine, prompt, max_new, kw,
                                                                    sample=samp)
        self._last_finish_reason = finish                   # stash for last_finish_reason() (the log path)
        self._last_diverged, self._last_diverged_at = divinfo  # stash for last_divergence()
        if comp is not None:                                 # LOOP GUARD: only when anchored memory was
            reply_raw, steps, finish = ctx._anchored_loop_guard(  # ACTUALLY injected this turn (comp is not
                self.engine, prompt, max_new, kw, samp, comp, reply_raw, steps, finish, mem_out)
            self._last_finish_reason = finish                # None) -- see ctx._anchored_loop_guard's docstring
        if trace_out is not None:
            trace_out.extend(steps)
        return reply_raw.strip()

    def last_divergence(self):
        """The early-stop verdict from the most recent chat(): (diverged, diverged_at). (None, None) when
        the last chat carried no reference_tokens. Read by replay to record whether an ablated arm's reply
        was truncated at the point it provably changed."""
        return (getattr(self, "_last_diverged", None), getattr(self, "_last_diverged_at", None))

    def score_tokens(self, messages, continuation_ids=None, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        """Teacher-forced per-token logprob of a continuation under EXPLICIT (block,
        steer_strengths) conditions -- the seam the forced-scoring
        stack (rederive.py, forced receipts) builds on. Assembles the prompt EXACTLY like chat()
        (ctx._inject_block + ctx._engine_tmpl -- the loaded model's own chat template) and the steer_vec EXACTLY
        like chat() (self.steer.steer_vector),
        but from the CALLER's `block`/`steer_strengths` -- NEVER from live self.memory/self.steer.strength
        -- so a with/without arm is reconstructed purely from a run record (memory ablation = recompile
        the block without a card; dial ablation = zero a strength and recompute) rather than from
        whatever the live substrate happens to be doing right now. That's what makes receipt arms
        reconstructable: two calls with different explicit `block`/`steer_strengths`, same messages
        and continuation_ids, are directly comparable. No sampling anywhere; deterministic.

        `block`: a prompt-mode memory block string (or None to omit it), e.g. run.memory.prompt_block.
        `steer_strengths`: a {dial_name: strength} dict (or None for no steer), e.g. run.behavior.dials.
        `continuation_ids`: the PRIMARY continuation form (token ids, e.g. from a stored trace) --
        takes precedence over `continuation` when both are given (mirrors EngineClient.score).
        `continuation`: a TEXT fallback (S3's rederive.py, for a run whose trace lacks per-token ids) --
        the engine retokenizes it independently of the prompt, which can drift at the prompt/
        continuation BPE boundary (flagged `boundary_approximate` by /score itself; see
        REPRODUCE_AND_PROVE_PLAN.md's tokenization-boundary caveat).
        `steer_vec`: an explicit RAW steer direction, ADDED on top of whatever `steer_strengths`
        produces (or used alone if `steer_strengths` is falsy) -- the S3 null-floor control needs a
        direction with no named dial behind it ("a random vector of equal norm at the same layer").

        Returns [{"id", "piece", "logprob"}, ...] (+ "topk" per token when topk>0), one entry per
        continuation token, in the SAME order as continuation_ids (or the engine's own retokenization
        of `continuation` text).
        """
        assembled = ctx._inject_block(messages, block)
        prompt = ctx._engine_tmpl(self.engine, assembled)   # per-model template (the loaded GGUF's own), not Qwen ChatML
        kw = {}
        sv = None
        if self.steer is not None and steer_strengths and any(steer_strengths.values()):
            sv = self.steer.steer_vector(steer_strengths)
        if steer_vec is not None:
            sv = [a + b for a, b in zip(sv, steer_vec)] if sv else list(steer_vec)
        if sv:
            kw["steer_vec"] = sv
            # self.steer.layer is model-aware (pinned per-family in __init__); with no steer built, pass
            # layer 0 so the ENGINE picks its own calibrated mid-depth band -- not a hardcoded Qwen 14.
            kw["steer"] = {"coef": 1.0, "layer": self.steer.layer if self.steer is not None else 0}
        if continuation_ids is not None:
            kw["continuation_ids"] = [int(t) for t in continuation_ids]
        elif continuation is not None:
            kw["continuation"] = str(continuation)
        r = self.engine.score(prompt=prompt, topk=int(topk), **kw)
        return r.get("tokens", [])

    def jlens(self, text, layer=None, topk=5):
        """Proxy the engine's /jlens for the Run Inspector J-lens panel -- mirrors score_tokens' /score
        proxy. Returns a NORMALIZED dict (never raises): the engine's {layer, n_tokens, tokens, readouts}
        plus available_layers (from /health's jlens.layers). Graceful absence: if the engine was started
        WITHOUT --jlens (no jlens block in /health), returns {available:False, reason:...} so the panel
        shows a clean 'lens not loaded' instead of an error. An unknown layer surfaces the engine's 400
        body (the available layers) cleanly rather than throwing."""
        try:
            h = self.engine.health() if (self.engine and hasattr(self.engine, "health")) else {}
            jl = (h or {}).get("jlens") or {}
            avail = [int(x) for x in (jl.get("layers") or [])]
        except Exception:
            avail = []
        if not avail:
            return {"available": False, "reason": "the engine was started without --jlens"}
        try:
            r = self.engine.jlens(text, layer=layer, topk=int(topk))
        except ctx.EngineError as e:
            # e.g. an unknown layer -> the engine's 400 {error, available}. Surface it cleanly (the panel
            # can offer the loaded layers); available_layers already comes from /health above.
            return {"available": True, "error": str(e), "available_layers": avail,
                    "layer": layer, "n_tokens": 0, "tokens": [], "readouts": []}
        return {"available": True, "layer": r.get("layer"), "available_layers": avail,
                "n_tokens": int(r.get("n_tokens", 0) or 0),
                "tokens": r.get("tokens", []), "readouts": r.get("readouts", [])}

    def last_stream_trace(self):
        """The per-token trace captured during the most recent chat_stream (raw step list, or []) --
        same contract as QwenSubstrate.last_stream_trace: the SSE handler reads this AFTER the generator
        is exhausted, to log the run's Run Inspector timeline."""
        return list(getattr(self, "_last_stream_trace", []) or [])

    def last_finish_reason(self):
        """The stop cause ("stop"|"length"|...) from the most recent chat()/chat_stream, or None. Same
        stash-and-read contract as last_stream_trace: the handler reads it AFTER generation, so the run
        logs WHY the engine stopped instead of a hard-coded 'stop'."""
        return getattr(self, "_last_finish_reason", None)

    def run_meta(self):
        """Reproducibility metadata -- WHAT produced a run -- for the run record. Fetched once from
        /health (model file -> quant, engine mode) and cached; the STATIC baseline here is the honest
        greedy default (temperature 0) -- the ACTUAL regime a specific reply used (greedy, or S5 sampled
        with its params + seed) rides in _last_generation_meta, filled by chat()/chat_stream() and merged
        in below, so a call made before any generation still reports the honest baseline rather than a
        guess. Health-derived fields are omitted when unavailable rather than guessed. Never raises:
        metadata never breaks a run."""
        health_meta = getattr(self, "_run_meta", None)
        if health_meta is None:
            health_meta = {}
            try:
                h = self.engine.health() if (self.engine and hasattr(self.engine, "health")) else {}
                mp = str((h or {}).get("model", ""))
                if mp:
                    health_meta["model_file"] = mp.replace("\\", "/").rsplit("/", 1)[-1]
                    q = _quant_from_name(health_meta["model_file"])
                    if q:
                        health_meta["quant"] = q
                    # T0.2: which model actually produced this run (derived from the loaded GGUF, not a
                    # hardcoded id). family is the registry key; model_id the friendly HF name when known.
                    fam, info = _engine_model_info(mp)
                    if fam:
                        health_meta["family"] = fam
                    if info.get("model_id"):
                        health_meta["model_id"] = info["model_id"]
                if (h or {}).get("mode"):
                    health_meta["mode"] = h["mode"]
                for k in ("n_ctx", "device", "gpu_layers"):
                    v = (h or {}).get(k)
                    if v is not None:
                        health_meta[k] = v
            except Exception:
                pass
            self._run_meta = dict(health_meta)
        meta = ctx._engine_generation_meta()
        meta.update(dict(health_meta))
        meta.update(getattr(self, "_last_generation_meta", None) or {})
        return dict(meta)

    def chat_stream(self, messages, max_new=256, mem_out=None, lens=None, on_frame=None):
        """Streaming twin of chat(): the SAME memory-block + tone-dial + anchored-memory construction
        (kept in lockstep -- see chat()'s comments; do not let this drift from that logic), but opens the engine's
        /v1/completions with stream:True (mirrors _engine_complete_traced's request) and yields text as
        the engine commits it, instead of waiting on one blocking call. This is what makes /v1/chat/
        completions's SSE branch (_sse_chat, gated on `getattr(SUB, "chat_stream", None)`) fire on the
        pure-engine substrate too -- before this existed, a streaming request here silently fell through
        to one non-streamed chat() reply. mem_out: as in chat() -- prompt mode records what memory
        actually rode this turn.

        F1 LIVE LENS: lens = a dict {layer?, topk?, every?} (or {} for engine defaults) rides to the
        engine as body["lens"]; the engine then interleaves `jlens_live` frames (the J-lens
        "disposed to say" readout for each committed token, computed mid-generation) with the token
        frames. Each one is handed to `on_frame(obj)` as it arrives -- a side-channel, because this
        generator's yields are text pieces and must stay that way for every existing consumer. A
        failing on_frame is dropped (never kills generation).

        Per-token trace (mirrors QwenSubstrate.chat_stream's B3 contract): every parsed SSE frame is
        collected, then folded into self._last_stream_trace via runlog.accumulate_ar_events once the
        stream ends -- normal completion OR an early GeneratorExit (the consumer stopped early) -- so a
        partial stream still logs whatever trace it managed. Wrapped so any parse hiccup just leaves it
        [], same as the non-streaming path's fallback.

        SAMPLING (S5): chat_stream has no per-call `sample` arg -- unlike chat(), no receipt/replay path
        ever drives this method (only the live SSE studio chat does), so it is always ELIGIBLE to sample;
        ctx._resolve_sampling(True) alone decides, against the persisted "sampling" setting (default ON --
        temperature/top_p/top_k/repeat_penalty/a fresh seed) exactly like chat()'s sample=True path. The
        setting off degrades to greedy, byte-identical to pre-S5 behavior."""
        import urllib.error
        import urllib.request
        import clozn.runs.store as runlog
        samp = ctx._resolve_sampling(True)
        self._last_generation_meta = ctx._engine_generation_meta(max_new, stream=True, sample=samp)
        # MEMORY + TONE: built EXACTLY as chat() builds them.
        block, applied, gate = ctx._prompt_block_for(self.memory, ctx._last_user(messages))
        assembled = ctx._inject_block(messages, block)
        prompt = ctx._engine_tmpl(self.engine, assembled)   # per-model template (the loaded GGUF's own), not Qwen ChatML
        if mem_out is not None:
            # final_prompt = the EXACT rendered string the model saw (backlog #5); kept in lockstep with chat().
            mem_out.update(mode="prompt", applied=applied, gate=gate,
                           prompt_block=block, assembled_messages=assembled, final_prompt=prompt)
        kw = {}
        st = (getattr(self.steer, "strength", None) if self.steer is not None else None) or ctx._disk_dials()
        if self.steer is not None and st and any(st.values()):
            sv = self.steer.steer_vector(st)
            if sv:
                kw["steer_vec"] = sv
                kw["steer"] = {"coef": 1.0, "layer": self.steer.layer}
        # F6 ANCHORED MEMORY (X7): active bags compose into ONE gated steer_vec at L21 and ride live chat.
        ctx._apply_anchored_memory(kw, mem_out, ctx._last_user(messages))
        body = dict(kw); body["prompt"] = prompt; body["max_tokens"] = int(max_new)
        if samp and samp.get("on"):     # S5: real sampling -- temperature/rep_penalty/seed from settings
            body["temperature"] = float(samp["temperature"])
            body["rep_penalty"] = float(samp["repeat_penalty"])
            body["seed"] = int(samp["seed"])
            # top_p/top_k are NOT sent -- this engine build's sampler has no nucleus/top-k truncation to
            # receive them (see ctx._resolve_sampling's docstring); they still ride _last_generation_meta.
        else:
            body["temperature"] = 0.0; body["rep_penalty"] = 1.0; body["seed"] = 0
        body["stream"] = True
        if lens is not None:                # F1 live lens: opt-in passthrough (engine validates layer etc.)
            body["lens"] = lens if isinstance(lens, dict) else True
        req = urllib.request.Request(self.engine.base + "/v1/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        self._last_stream_trace = []        # reset; reassembled in `finally` below (empty on any hiccup)
        frames = []
        try:
            resp = urllib.request.urlopen(req, timeout=getattr(self.engine, "timeout", 600))
        except urllib.error.HTTPError as he:
            # surface the engine's own error text (e.g. a bad lens layer's 400) instead of a bare code
            try:
                detail = json.loads(he.read()).get("error") or str(he)
            except Exception:
                detail = str(he)
            raise RuntimeError(f"engine: {detail}")
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                frames.append(obj)
                if obj.get("type") == "jlens_live":     # F1: side-channel to the SSE relay, never yielded
                    if on_frame is not None:
                        try:
                            on_frame(obj)
                        except Exception:
                            on_frame = None             # a dead callback must never kill generation
                    continue
                if obj.get("type") == "tokens_committed":
                    for it in obj.get("items") or []:
                        piece = it.get("piece", "")
                        if piece:
                            yield piece
        finally:
            # ALWAYS release the engine connection -- whether the stream ran to [DONE] or the consumer
            # stopped early (this `finally` also runs when the caller .close()s us mid-stream, via a
            # GeneratorExit at the `yield` above); guarded so a close() hiccup can never mask a
            # propagating GeneratorExit -- it must reach the caller, never be swallowed here. (The
            # engine-side crash-on-disconnect is a separate C++-side task; this just closes cleanly.)
            try:
                resp.close()
            except Exception:
                pass
            try:
                self._last_stream_trace = runlog.accumulate_ar_events(frames)
            except Exception:
                self._last_stream_trace = []
            try:
                self._last_finish_reason = runlog.finish_reason_from_frames(frames)
            except Exception:
                self._last_finish_reason = None
            # LOOP GUARD, streaming twin: the engine sets the anchored
            # steer at generation-START (body["steer_vec"] above) and every piece is yielded to the
            # caller live over SSE -- by the time this `finally` runs, the client has ALREADY received
            # the whole reply. There is no seamless mid-stream re-injection here (unlike chat()'s
            # auto-retry-at-half-strength): the honest thing this path can do is DETECT the degeneracy
            # after the fact and FLAG the run -- never fake a retry/self-heal capability streaming
            # doesn't have. Only checked when anchored memory actually rode this turn (mem_out["anchored"]
            # set, not anchored_skipped/absent) -- a degeneracy safety, never a claim the memory "worked".
            if mem_out is not None and mem_out.get("anchored"):
                try:
                    from clozn.memory import anchored as _anchored_lg
                    pieces = [str(s.get("piece", "")) for s in (self._last_stream_trace or [])]
                    if _anchored_lg.detect_loop(pieces):
                        mem_out["anchored_loop_guard"] = {
                            "fired": True, "action": "flagged-only", "resolved": False,
                            "note": ("streaming reply already reached the client -- detected after the "
                                     "fact, no mid-stream retry is possible on this path")}
                except Exception:
                    pass

    def handle(self, path, body):
        r = self._memory(path, body)
        if r is not None:
            return r
        return self._steer(path, body)

    def state(self):
        import clozn.memory.cards as memory_cards
        return {"dials": dict(getattr(self.steer, "strength", {}) or {}),
                "cards": len(memory_cards.list_cards() or [])}


def _quant_from_name(name):
    """Pull the GGUF quant tag (Q4_K_M, Q8_0, IQ4_XS, F16, ...) out of a model filename, or None. GGUF
    files name their quantization in the basename, so this is the one bit of repro metadata we can read
    for free (no engine change) off /health's model path."""
    import re
    m = re.search(r"(IQ\d+[A-Z0-9_]*|Q\d+(?:_[A-Z0-9]+)+|Q\d+|BF16|F16|F32)", str(name), re.IGNORECASE)
    return m.group(1).upper() if m else None


# --- engine model registry (T0.2) ---------------------------------------------------------------------
# The engine substrate reflects the ACTUALLY-LOADED GGUF, not a hardcoded "Qwen2.5-7B" id/assumption.
# The ONE Qwen-specific assumption the engine substrate carried was the tone-dial steer TAP LAYER
# (mid-depth: 14 for Qwen-7B's 28 layers). This tiny registry keys that -- plus a friendly model_id for
# run_meta -- off the loaded model's family (derived from its /health filename), with a sensible default
# for any unrecognized GGUF (steer_layer None => don't pin a layer; let the engine use its OWN per-model
# calibrated mid-depth steer band). Everything else the engine already calibrates per-model server-side
# (the C++ concept/steer probe taps at startup, and the chat template via /apply_template). This is NOT a
# framework -- it is the minimal table that removes the last hardcoded-Qwen coupling from the engine path.
_ENGINE_MODELS = {
    "qwen2.5-7b":   {"model_id": "Qwen/Qwen2.5-7B-Instruct",         "steer_layer": 14},  # 28L -> mid 14 (unchanged)
    "qwen2.5-0.5b": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct",       "steer_layer": 12},  # 24L -> mid 12
    "llama-3.2-1b": {"model_id": "meta-llama/Llama-3.2-1B-Instruct", "steer_layer": 8},   # 16L -> mid 8
    "llama-3.2-3b": {"model_id": "meta-llama/Llama-3.2-3B-Instruct", "steer_layer": 14},  # 28L -> mid 14
}
_ENGINE_MODEL_DEFAULT = {"model_id": None, "steer_layer": None}  # unknown GGUF: nothing pinned; engine picks


def _model_family_from_name(name):
    """Coarse model family key ('qwen2.5-7b', 'llama-3.2-1b', ...) from a GGUF filename, or None -- the
    engine substrate looks up per-model assumptions in _ENGINE_MODELS by this key instead of hardcoding
    Qwen's. Same free derive-off-/health-filename trick as _quant_from_name (no engine change needed)."""
    import re
    s = str(name or "").lower()
    m = re.search(r"qwen[._]?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)b", s)
    if m:
        return f"qwen{m.group(1)}-{m.group(2)}b"
    m = re.search(r"llama[._-]?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)b", s)
    if m:
        return f"llama-{m.group(1)}-{m.group(2)}b"
    return None


def _engine_model_info(name):
    """(family, {model_id, steer_layer}) for the loaded GGUF -- the engine substrate's per-model
    assumptions -- or (None, the default with nothing pinned) for an unrecognized model."""
    fam = _model_family_from_name(name)
    return fam, dict(_ENGINE_MODELS.get(fam, _ENGINE_MODEL_DEFAULT))


def _engine_complete_traced(engine, prompt, max_tokens, kw, sample=None):
    """Generate on the engine and ALSO capture a per-token trace (issue B3), returning
    (reply, steps, finish, divinfo).

    The engine's non-streaming /v1/completions carries only the final text -- no per-token confidence. To
    populate the Run Inspector timeline we ask the SAME request with stream:True and fold its per-token
    `tokens_committed`/`step_lens` frames into steps via runlog.accumulate_ar_events. Generation defaults to
    greedy (temperature 0) so the reassembled text is identical to the blocking call; passing a
    ctx._resolve_sampling() dict as `sample` (S5) switches BOTH the streaming attempt and the fallback below to
    the same temperature/rep_penalty/seed, so the two paths always agree. Either way we only capture
    ALONGSIDE; the client still receives the same single JSON reply (this streams engine<->server, never to
    the client). Any streaming hiccup falls back to the plain complete() so a run is never lost -- just
    without a trace. (AR GGUFs only; a diffusion engine commits out of reading order and emits no such
    per-token stream.)

    `divinfo` is (diverged, diverged_at) when the request carried `reference_tokens` (prove-all early-stop):
    diverged True/False is the engine's verdict, else (None, None). The reply on a diverged run is a
    BIT-EXACT PREFIX of the full generation (the engine only adds a stop check -- decode is unchanged).

    `sample`: None (or a falsy dict) -- greedy, temperature=0/rep_penalty=1/seed=0, byte-identical to
    pre-S5 behavior. A ctx._resolve_sampling() dict -- temperature/repeat_penalty/seed ride the request as
    the engine's own SampleConfig keys (`temperature`, `rep_penalty`, `seed`); top_p/top_k are NOT sent
    (the engine's sampler -- engine/core/src/sample.cpp -- has no nucleus/top-k truncation to receive
    them)."""
    on = bool(sample and sample.get("on"))
    temperature = float(sample["temperature"]) if on else 0.0
    rep_penalty = float(sample["repeat_penalty"]) if on else 1.0
    seed = int(sample["seed"]) if on else 0
    import urllib.request
    body = dict(kw); body["prompt"] = prompt; body["max_tokens"] = int(max_tokens)
    body["temperature"] = temperature; body["rep_penalty"] = rep_penalty; body["seed"] = seed
    body["stream"] = True
    try:
        req = urllib.request.Request(engine.base + "/v1/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        frames, text = [], ""
        diverged, diverged_at = None, None
        with urllib.request.urlopen(req, timeout=getattr(engine, "timeout", 600)) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                frames.append(obj)
                ch = obj.get("choices")                     # the final OpenAI-style frame carries the full text
                if ch and isinstance(ch, list) and ch[0].get("text"):
                    text = ch[0]["text"]
                if "diverged" in obj:                       # early-stop verdict rides the final frame
                    diverged = obj.get("diverged")
                    diverged_at = obj.get("diverged_at")
        import clozn.runs.store as runlog
        steps = runlog.accumulate_ar_events(frames)
        finish = runlog.finish_reason_from_frames(frames)   # the engine's real stop cause (else None)
        if not text:                                        # no final frame text -> reassemble from the pieces
            text = "".join(s.get("piece", "") for s in steps)
        if steps or text:
            return text, steps, finish, (diverged, diverged_at)
    except Exception:
        pass
    # Fallback: the original blocking path, reply preserved, trace simply empty. The non-streaming
    # /v1/completions carries the same `diverged`/`diverged_at` when a reference was sent. Same
    # temperature/rep_penalty/seed as the streaming attempt above -- the fallback must never silently
    # decode under a DIFFERENT regime than the one recorded in the run's meta.
    r = engine.complete(prompt, max_tokens=max_tokens, temperature=temperature, rep_penalty=rep_penalty,
                        seed=seed, **kw)
    ch = r.get("choices") if isinstance(r, dict) else None
    finish = ch[0].get("finish_reason") if (ch and isinstance(ch[0], dict)) else None
    divinfo = (r.get("diverged"), r.get("diverged_at")) if isinstance(r, dict) else (None, None)
    return (ch[0].get("text", "") if ch else str(r)), [], finish, divinfo
