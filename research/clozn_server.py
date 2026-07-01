"""clozn_server.py -- the UNIFIED instrument. One port, one model, the whole white-box surface.

  substrate 'qwen' (default): ONE Qwen-7B serves BOTH the brain (/think -- concepts the model engages)
                              AND the memory (/say /consolidate /check /whatlearned) -- they share the
                              single loaded model, so the instrument's brain and memory tabs are both live.
  substrate 'dream':          Dream-7B serves /denoise (the diffusion window).

Only one 7B fits the GPU, so switching substrates re-execs the process with the other one (a clean GPU);
the instrument shows the active substrate and offers the switch. Serves the instrument + every window
from inspector/demo, so the iframes' fetches all land here.

    cloze .venv python research/clozn_server.py --port 8090
"""
import argparse
import json
import os
import sys
import threading
import time

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "engine", "lab"))   # so the dream substrate can import cloze_lab
DEMO = os.path.join(HERE, "..", "inspector", "demo")

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer   # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "engine", "client"))     # the engine white-box SDK
import numpy as np                                                   # noqa: E402
try:
    from cloze_engine import EngineClient
    ENGINE = EngineClient(port=int(os.environ.get("CLOZN_ENGINE_PORT", "8091")))            # the live C++ runtime
    ENGINE_QWEN = EngineClient(port=int(os.environ.get("CLOZN_ENGINE_QWEN_PORT", "8092")))  # a Qwen GGUF engine -> concepts
except Exception:
    ENGINE = ENGINE_QWEN = None

CLOZN_DIR = os.path.join(os.path.expanduser("~"), ".clozn")   # studio memory + personality persist here


def _pers(name):
    return os.path.join(CLOZN_DIR, name)


ENGINE_STEER = None        # lazy EngineSteer on the Qwen GGUF engine -- tone dials on the C++ runtime, any GGUF


def _engine_steer():
    global ENGINE_STEER
    if ENGINE_STEER is None and ENGINE_QWEN is not None:
        from steering import EngineSteer
        ENGINE_STEER = EngineSteer(ENGINE_QWEN)
    return ENGINE_STEER


def _qwen_tmpl(messages):
    """Render chat messages into Qwen's chat-template STRING for the engine's raw /v1/completions -- the
    same template the HF memory prefix was trained against, so the injected prefix lands in the right context."""
    sysmsg = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    for m in messages:
        if m.get("role") == "system" and m.get("content"):
            sysmsg = m["content"]
    s = f"<|im_start|>system\n{sysmsg}<|im_end|>\n"
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            s += f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n"
    return s + "<|im_start|>assistant\n"


def _disk_memory():
    """The trained memory prefix + strength, read from disk -- so engine-chat needs NO HF model resident.
    The prefix is just saved vectors; only TRAINING a new one needs PyTorch's gradients."""
    import torch
    path = _pers("studio_memory.pt")
    if not os.path.isfile(path):
        return None, 1.0
    try:
        d = torch.load(path, map_location="cpu")
        pre = d.get("prefix")
        return (pre.float() if pre is not None else None), float(d.get("memory_strength", 1.0))
    except Exception:
        return None, 1.0


def _disk_dials():
    """The saved tone-dial values (personality.json IS the strength dict) -- no HF model needed."""
    path = _pers("studio_personality.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return {k: float(v) for k, v in json.load(f).items()}
    except Exception:
        return {}


# ------- memory cards <-> the working prefix (D2 + E1) --------------------------------------------
# The cards (research/memory_cards.py) are the metadata + review layer; the trained soft-prefix is
# UNCHANGED. The contract that keeps the prefix safe: m.rules is ALWAYS the texts of the ACTIVE cards,
# and the prefix is built from m.rules via m.consolidate(rules) exactly as before. So a card's STATUS
# decides what's in m.rules, which drives the prefix. We only ever retrain when the active set actually
# changes (a no-op transition -- e.g. approving a card whose text is already active -- never touches it).

_SUSPICIOUS = ("ignore ", "disregard ", "system prompt", "you are now", "forget ", "override",
               "jailbreak", "developer mode", "instead of", "from now on you", "pretend ")


def _risk_of(text: str) -> str:
    """Cheap heuristic: flag instruction-like / prompt-injection-ish memory text as 'suspicious' so the
    reviewer sees it. A memory is meant to be a fact/preference ABOUT the user, not a command to the model."""
    t = (text or "").lower()
    return "suspicious" if any(s in t for s in _SUSPICIOUS) else "low"


def _mem_migrate(m):
    """Seed the card store from a memory object's legacy rule-strings, ONCE. migrate_from_rules is a
    no-op when the store already has cards, and it creates them as ACTIVE -- the prefix is already trained
    on these exact rules, so we do NOT re-consolidate here. Returns the cards created (or [])."""
    import memory_cards
    try:
        return memory_cards.migrate_from_rules(list(getattr(m, "rules", []) or []))
    except Exception:
        return []


def _runs_for_card(card_id):
    """Best-effort: the run summaries whose memory.cards_applied names this card (by id OR by text).
    cards_applied currently records the active rule TEXTS (see _log_run), so we match on text primarily
    and on id as a forward-compatible fallback. Returns [] when the card / runs are gone (never raises)."""
    import memory_cards
    import runlog
    try:
        card = memory_cards.get(card_id)
        text = (card or {}).get("text", "")
        needles = {n for n in (card_id, text) if n}
        if not needles:
            return []
        out = []
        for r in runlog.list_runs(500):
            applied = ((r.get("memory") or {}).get("cards_applied")) or []
            applied = [str(a) for a in applied]
            if needles & set(applied):
                out.append(r)
        return out
    except Exception:
        return []


def _mem_sync_rules(m, reconsolidate=True):
    """Make m.rules == the active-card texts, then rebuild the prefix ONLY if the active set changed.

    This is the one place the prefix can move. If the active texts are identical to what m.rules already
    holds, we leave the prefix completely untouched (the expensive, working artifact is preserved). When
    the set changed and reconsolidate is on, we retrain from the active texts (SLOW -- expected on
    approve/reject/disable/edit). If the active set became EMPTY (e.g. the last card was disabled), we
    reset() so the now-unused prefix stops biting -- reset() is zero-arg on both memory backends."""
    import memory_cards
    new_rules = memory_cards.active_texts()
    changed = list(getattr(m, "rules", []) or []) != list(new_rules)
    m.rules = list(new_rules)
    result = None
    if changed and reconsolidate:
        if new_rules:
            result = m.consolidate(list(new_rules))
        else:                                    # nothing active anymore -> drop the prefix entirely
            try:
                result = m.reset()
            except Exception:
                pass
            m.rules = []                          # reset() may clear rules; keep them in sync
    return {"changed": changed, "rules": list(new_rules), "consolidate": result}


ARGS = None
SUB = None         # the active substrate object
SUBNAME = "qwen"

# ------- async retrain: one background retrain at a time, chats serialize behind it -----------------
# Mutating a memory card retrains the soft-prefix via consolidate() -- ~4-5 min on the 4-bit 7B. We must
# NOT block the HTTP handler for that. So the card STATUS flip (fast) stays synchronous and the RETRAIN
# runs on a daemon thread. Two module-level guards (a process singleton, like the model itself):
#   _TRAIN_LOCK  -- held for the WHOLE consolidate(); the chat/generate paths acquire+release it so a
#                   reply can't race the shared model+gradients mid-retrain (they queue, they don't error).
#   _RETRAIN     -- the in-flight signal the UI polls: {active, card_id, action, started_at, error}.
# _RETRAIN_META guards reads/writes of the _RETRAIN dict (a tiny critical section, distinct from the long
# _TRAIN_LOCK). Mirrors the _ensure_steer double-checked-lock: we don't launch a 2nd retrain while one runs.
_TRAIN_LOCK = threading.RLock()
_RETRAIN_META = threading.Lock()
_RETRAIN = {"active": False, "card_id": None, "action": None, "started_at": None, "error": None}


def _retrain_status():
    """A snapshot of the in-flight retrain signal (copy -- never hand out the live dict)."""
    with _RETRAIN_META:
        return dict(_RETRAIN)


def _retrain_in_flight():
    with _RETRAIN_META:
        return bool(_RETRAIN["active"])


def _join_retrain(timeout=None):
    """Block until no retrain is in flight (acquire+release _TRAIN_LOCK). Used by tests to await the
    background consolidate deterministically, and available for a graceful shutdown. Returns True once
    the lock was momentarily held with nothing active; False on timeout."""
    if not _TRAIN_LOCK.acquire(timeout=timeout if timeout is not None else -1):
        return False
    try:
        return not _retrain_in_flight()
    finally:
        _TRAIN_LOCK.release()


def _start_retrain(m, action, card_id):
    """Launch _mem_sync_rules(m) -- the SLOW consolidate() -- on a daemon thread and return immediately.

    Returns {retraining: True} once the thread is running, or {retraining: False} if there's nothing to do
    (the active set didn't move -- checked synchronously first, so a no-op transition never spins a thread)
    or a retrain is already in flight (we refuse to stack them, like _ensure_steer refuses a double compute).
    The worker holds _TRAIN_LOCK for the whole consolidate so chats serialize behind it, and clears _RETRAIN
    on finish (success OR error) so the UI's poll always terminates."""
    import memory_cards
    # cheap synchronous pre-check: would the active set actually change? if not, do NOT spawn a thread.
    if list(getattr(m, "rules", []) or []) == list(memory_cards.active_texts()):
        return {"retraining": False, "changed": False}
    with _RETRAIN_META:
        if _RETRAIN["active"]:                        # a retrain is already running -> don't stack a second
            return {"retraining": True, "busy": True, "queued": False}
        _RETRAIN.update(active=True, card_id=card_id, action=action,
                        started_at=time.time(), error=None)

    def _work():
        err = None
        try:
            with _TRAIN_LOCK:                         # hold across consolidate() -> chats wait, never race
                _mem_sync_rules(m, reconsolidate=True)
        except Exception as e:                        # a failed retrain must still clear the flag
            err = f"{type(e).__name__}: {e}"
        finally:
            with _RETRAIN_META:
                _RETRAIN.update(active=False, error=err)

    threading.Thread(target=_work, daemon=True).start()
    return {"retraining": True, "action": action, "card_id": card_id}


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
        import memory_cards
        m = self._mem
        self._ensure_cards_migrated()           # one-time seed of legacy rules -> active cards (no retrain)

        if path == "/memory/cards":             # OBJECTS now (not bare strings) -- the review layer
            return {"cards": memory_cards.list_cards(), "has_prefix": m.prefix is not None,
                    "retraining": _retrain_status()}   # fold the in-flight signal in (one reload sees it)

        if path == "/memory/retrain-status":    # the poll target: is a background consolidate() running?
            return _retrain_status()

        if path == "/memory/add":               # propose a card as PENDING -> does NOT affect the prefix
            text = str(body.get("text", "")).strip()
            if not text:
                return {"ok": False, "reason": "empty trait"}
            card = memory_cards.create(text, status="pending", kind="preference",
                                       risk=_risk_of(text), source_run_id=body.get("source_run_id"),
                                       evidence=str(body.get("evidence", "")))
            return card or {"ok": False, "reason": "could not create card"}

        if path == "/memory/remove":            # delete by id -> if it was active, rebuild from the rest
            cid = str(body.get("id", "")).strip()
            if not cid:                          # (index removed -- ids are the stable handle now)
                return {"ok": False, "reason": "need a card id"}
            was_active = (memory_cards.get(cid) or {}).get("status") == "active"
            ok = memory_cards.delete(cid)
            if not ok:
                return {"ok": False, "reason": "no such card"}
            # delete is synchronous+fast; the retrain (only if an ACTIVE card left the set) is backgrounded.
            resync = _start_retrain(m, "remove", cid) if was_active else {"retraining": False}
            return {"ok": True, "removed": cid, "resync": resync}

        if path in ("/memory/approve", "/memory/reject", "/memory/disable", "/memory/enable"):
            return self._card_status(path.rsplit("/", 1)[1], str(body.get("id", "")).strip())

        if path == "/memory/edit":              # change a card's text; if active, retrain on the new text
            cid = str(body.get("id", "")).strip()
            new_text = str(body.get("text", "")).strip()
            if not (cid and new_text):
                return {"ok": False, "reason": "need id and text"}
            card = memory_cards.update(cid, text=new_text, risk=_risk_of(new_text))
            if card is None:
                return {"ok": False, "reason": "no such card"}
            if card.get("status") == "active":   # editing an active card's text retrains -> in the background
                card = {**card, "resync": _start_retrain(m, "edit", cid)}
            return card

        if path == "/memory/strength":          # the memory dial: how hard the prefix bites (0 = off, >1 = stronger)
            if "value" in body and hasattr(m, "memory_strength"):
                m.memory_strength = max(0.0, min(2.0, float(body["value"])))
                if hasattr(m, "save"):
                    try:
                        m.save()
                    except Exception:
                        pass
            return {"strength": float(getattr(m, "memory_strength", 1.0)), "has_prefix": m.prefix is not None}
        return None

    # ---- E1 review lifecycle: a status change rebuilds m.rules from the active set, retrains iff it moved -
    def _card_status(self, action, cid):
        """approve->active, reject->rejected, disable->disabled, enable->active. The STATUS flip (fast) is
        synchronous; the RETRAIN it may trigger (rebuild the prefix from active_texts) is backgrounded so
        the response returns immediately. The card keeps its FINAL status; a separate _RETRAIN flag carries
        the in-flight signal. _start_retrain no-ops when the active set didn't actually move (prefix safe)."""
        import memory_cards
        if not cid:
            return {"ok": False, "reason": "need a card id"}
        target = {"approve": "active", "reject": "rejected",
                  "disable": "disabled", "enable": "active"}[action]
        card = memory_cards.set_status(cid, target)
        if card is None:
            return {"ok": False, "reason": "no such card"}
        resync = _start_retrain(self._mem, action, cid)  # retrains on a thread iff the active set changed
        return {**card, "resync": resync}

    def _ensure_cards_migrated(self):
        """Seed the card store from this substrate's legacy rule-strings exactly once per process."""
        if getattr(self, "_cards_migrated", False):
            return
        _mem_migrate(self._mem)
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
        from steering import AXES
        if path == "/steer/axes":
            axes = [{"name": k, "poles": AXES[k]["poles"], "value": self.steer.strength.get(k, 0.0),
                     "max": AXES[k].get("max", 1.5)} for k in AXES]
            for k, v in getattr(self.steer, "custom", {}).items():   # user-defined dials alongside the built-ins
                axes.append({"name": k, "poles": v["poles"], "value": self.steer.strength.get(k, 0.0),
                             "max": v["max"], "custom": True})
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
            self.steer.save_custom(_pers(f"studio_custom_{self.name}.json"))
            return {"name": name, "max": info["max"], "custom": list(self.steer.custom)}
        if path == "/steer/custom_delete":
            if hasattr(self.steer, "remove_custom"):
                self.steer.remove_custom(str(body.get("name", "")))
                self.steer.save_custom(_pers(f"studio_custom_{self.name}.json"))
                self.steer.save_state(self._pers_steer)
            return {"custom": list(getattr(self.steer, "custom", {}))}
        return None


class QwenSubstrate(Substrate):
    """One Qwen-7B + SAE behind the concept readout AND the memory + tone dials."""
    name = "qwen"

    def __init__(self):
        from brain_readout import BrainReadout
        from sae7b import GpuSAE, load7b
        from self_teach_server import SelfTeach
        from steering import SteeringControl
        sae = GpuSAE()
        tok, model = load7b()
        self.brain = BrainReadout(model, tok, sae, DEMO, HERE)
        self.memory = SelfTeach("Qwen/Qwen2.5-7B-Instruct", model=model, tok=tok,   # shares the model
                                persist_path=_pers("studio_memory.pt"))
        self.steer = SteeringControl(model, tok)            # tone dials on the same model
        self._mem = self.memory
        self._steer_ready, self._steer_info, self._steer_lock = False, {}, threading.Lock()
        self._pers_steer = _pers("studio_personality.json")
        self.steer.load_state(self._pers_steer)             # restore the personality dials across restarts
        self.steer.load_custom(_pers(f"studio_custom_{self.name}.json"))    # + any user-defined dials

    def handle(self, path, body):
        if path == "/think":
            return self.brain.think(str(body.get("text", ""))[:500], str(body.get("sid", "default")))
        if path == "/concepts":                 # read what fired inside (no generation) -> annotate a reply
            return self.brain.concepts_only(str(body.get("text", ""))[:500])
        if path == "/say":
            with _TRAIN_LOCK:                    # studio chat touches the shared model -> wait out a retrain
                return {"reply": self.memory.say(body["message"], body.get("max_new", 200))}
        if path == "/consolidate":               # a manual retrain -> the same shared-model lock as card retrains
            with _TRAIN_LOCK:
                return self.memory.consolidate(body.get("rules"), body.get("steps", 120), body.get("lr", 0.012),
                                               body.get("n_probe", 8), body.get("max_norm", 14.0))
        if path == "/whatlearned":
            return {"report": self.memory.what_learned()}
        if path == "/check":                     # generates on the shared model -> wait out a retrain
            with _TRAIN_LOCK:
                return self.memory.check(body["prompt"], body.get("max_new", 200))
        if path == "/reset":
            with _TRAIN_LOCK:                     # mutates the prefix/model state -> don't race a retrain
                self.brain.reset(str(body.get("sid", "default")))
                return self.memory.reset(body.get("keep_prefix", False))
        if path.startswith("/memory/"):
            return self._memory(path, body)
        if path.startswith("/steer/"):
            return self._steer(path, body)
        return None

    def _gen(self, prompt):                     # AR generate for the /steer/check A/B
        return self.steer.generate(prompt, 90)

    def chat(self, messages, max_new=256, sample=True):
        """One stateless chat completion with the WHOLE tunable self applied: the consolidated memory
        prefix (learned + added traits) AND the active tone-steering sliders, both on the shared model.
        This is what the OpenAI-compatible endpoint serves -- normal chat on the surface, legible and
        tunable underneath. Serializes behind an in-flight memory retrain (_TRAIN_LOCK) so a reply can't
        race the shared model+gradients mid-consolidate -- it waits, briefly, rather than corrupting."""
        with _TRAIN_LOCK:                        # wait out any background retrain, then hold for this reply
            if self.steer.strength:             # persisted personality -> ensure vectors are ready (race-safe)
                self._ensure_steer()
            self.steer.engage()
            try:
                return self.memory._generate(messages, use_prefix=True, max_new=max_new, sample=sample,
                                             gate=self.memory.memory_strength)
            finally:
                self.steer.disengage()

    def chat_stream(self, messages, max_new=256):
        """Streaming chat: yields text chunks as the AR model generates -- memory prefix + tone steering
        applied -- via a TextIteratorStreamer with generate() in a thread. Local AR is slow, so this is
        the big UX win the diffusion side doesn't need (diffusion is trace-based, not left-to-right)."""
        import threading
        import torch
        from transformers import TextIteratorStreamer
        _TRAIN_LOCK.acquire()                    # serialize behind an in-flight retrain (released in finally)
        if self.steer.strength:
            self._ensure_steer()
        m = self.memory
        e = m._embed(m._chat_ids(messages))
        if m.prefix is not None:                            # prepend the consolidated memory prefix (scaled by the dial)
            e = torch.cat([(m.memory_strength * m.prefix.detach()).to(e.dtype)[None], e], 1)
        att = torch.ones(e.shape[:2], device=e.device, dtype=torch.long)
        streamer = TextIteratorStreamer(m.tok, skip_prompt=False, skip_special_tokens=True)
        kw = dict(inputs_embeds=e, attention_mask=att, max_new_tokens=max_new, do_sample=True,
                  temperature=0.7, top_p=0.9, repetition_penalty=1.3, no_repeat_ngram_size=3,
                  pad_token_id=m.eos or 0, streamer=streamer)            # trim steering-induced loops

        def _gen():
            with torch.no_grad():
                m.model.generate(**kw)

        self.steer.engage()                                 # tone dials apply during the streamed generation
        th = threading.Thread(target=_gen, daemon=True)
        th.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            th.join()
            self.steer.disengage()
            _TRAIN_LOCK.release()                           # done streaming -> let a queued retrain proceed

    def state(self):
        return self.memory.state()


class DreamSubstrate(Substrate):
    """Dream-7B diffusion: the denoise window, plus the SAME trait-card memory and tone dials as Qwen."""
    name = "dream"

    def __init__(self):
        from cloze_lab.cli import build_adapter
        from denoise_server import trace_for
        from steering import DreamSteering
        from dream_memory import DreamMemory
        self.adapter = build_adapter("dream", device="cuda", quant="nf4")
        self._trace = trace_for
        self.steer = DreamSteering(self.adapter)            # tone dials on the diffusion model
        self._steer_ready, self._steer_info, self._steer_lock = False, {}, threading.Lock()
        self._pers_steer = _pers("studio_dream_personality.json")
        self.steer.load_state(self._pers_steer)
        self.dmem = DreamMemory(self.adapter,               # diffusion-native memory (trained soft prefix)
                                persist_path=_pers("studio_dream_memory.pt"))
        self._mem = self.dmem

    def handle(self, path, body):
        if path == "/denoise":
            prompt = str(body.get("prompt", ""))[:300]
            with _TRAIN_LOCK:                              # wait out a background retrain (it moves dmem.prefix)
                self.steer.engage()                        # active dials steer every denoising pass
                try:
                    ad = self.adapter
                    if self.dmem.prefix is not None:       # memory present -> inject the prefix into the REAL scheduler
                        from dream_memory import PrefixAdapter
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


def load_substrate(name):
    if name == "engine":
        return None        # pure-engine: NO HF model -- serve the GGUF via the C++ engine + the saved prefix from disk
    return QwenSubstrate() if name == "qwen" else DreamSubstrate()


def switch_substrate(name):
    """Re-exec the whole process with the new substrate -> a clean GPU (the only honest way; one 7B fits)."""
    py = sys.executable
    os.execv(py, [py, os.path.abspath(__file__), "--substrate", name, "--port", str(ARGS.port),
                  "--host", ARGS.host])


def _engine_complete_traced(engine, prompt, max_tokens, kw):
    """Generate on the engine and ALSO capture a per-token trace (issue B3), returning (reply, steps).

    The engine's non-streaming /v1/completions carries only the final text -- no per-token confidence. To
    populate the Run Inspector timeline we ask the SAME request with stream:True and fold its per-token
    `tokens_committed`/`step_lens` frames into steps via runlog.accumulate_ar_events. Generation is greedy
    (temperature 0), so the reassembled text is identical to the blocking call -- we only capture ALONGSIDE;
    the client still receives the same single JSON reply (this streams engine<->server, never to the client).
    Any streaming hiccup falls back to the plain complete() so a run is never lost -- just without a trace.
    (AR GGUFs only; a diffusion engine commits out of reading order and emits no such per-token stream.)
    """
    import urllib.request
    body = dict(kw); body["prompt"] = prompt; body["max_tokens"] = int(max_tokens)
    body["temperature"] = 0.0; body["stream"] = True
    try:
        req = urllib.request.Request(engine.base + "/v1/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        frames, text = [], ""
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
        import runlog
        steps = runlog.accumulate_ar_events(frames)
        if not text:                                        # no final frame text -> reassemble from the pieces
            text = "".join(s.get("piece", "") for s in steps)
        if steps or text:
            return text, steps
    except Exception:
        pass
    # Fallback: the original blocking path, reply preserved, trace simply empty.
    r = engine.complete(prompt, max_tokens=max_tokens, temperature=0.0, **kw)
    ch = r.get("choices") if isinstance(r, dict) else None
    return (ch[0].get("text", "") if ch else str(r)), []


def make_handler():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            b = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _json(self, code, o):
            self._send(code, json.dumps(o), "application/json")

        def _html(self, name):
            self._send(200, open(os.path.join(DEMO, name), encoding="utf-8").read(), "text/html; charset=utf-8")

        def _sse_chat(self, messages, max_new, model):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            def chunk(delta, finish=None):
                o = {"id": "chatcmpl-clozn", "object": "chat.completion.chunk", "model": model,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(("data: " + json.dumps(o) + "\n\n").encode("utf-8"))
                self.wfile.flush()

            # HF chat stream (QwenSubstrate.chat_stream): pieces only, no per-token confidence/alternatives.
            # We deliberately DON'T pass `trace` here -> the run's trace stays an empty {}. Capturing one would
            # mean output_scores on the transformers streaming path -- a separate, riskier issue, out of B3.
            t0 = time.time(); acc = []
            try:
                chunk({"role": "assistant"})
                for piece in SUB.chat_stream(messages, max_new):
                    acc.append(piece); chunk({"content": piece})
                chunk({}, finish="stop")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self._log_run("openai_api", messages, "".join(acc), model, t0)
            except Exception as e:
                self._log_run("openai_api", messages, "".join(acc), model, t0, error=str(e))
                try:
                    self.wfile.write(("data: " + json.dumps({"error": str(e)}) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass

        def _client(self, ua):
            ua = (ua or "").lower()
            for k, v in (("open-webui", "Open WebUI"), ("openwebui", "Open WebUI"), ("cursor", "Cursor"),
                         ("vscode", "VS Code"), ("python-requests", "script"), ("httpx", "script"),
                         ("openai-python", "script"), ("curl", "curl"), ("mozilla", "browser")):
                if k in ua:
                    return v
            return ua[:24] or "unknown"

        def _log_run(self, source, messages, response, model, started, error=None, trace=None):
            """Persist this interaction as an inspectable run (never let logging break the request)."""
            try:
                import runlog
                # cards_applied == the ACTIVE-card texts. Post-D2, SUB._mem.rules is kept in sync with the
                # active cards (see _mem_sync_rules), so reading .rules still reports exactly what shaped the
                # reply. Reading SUB.memory would miss the dream cards -- use _mem (self.memory on qwen,
                # self.dmem on dream). Only ACTIVE cards feed the prefix, so only those count as applied.
                mem = getattr(SUB, "_mem", None) if SUB else None
                if mem is not None:
                    cards = getattr(mem, "rules", None) or getattr(mem, "cards", None) or []
                    memd = {"cards_applied": list(cards),
                            "strength": float(getattr(mem, "memory_strength", 1.0)),
                            "has_prefix": getattr(mem, "prefix", None) is not None,
                            "proposed_cards": []}
                    if cards:                                    # record that the active cards influenced a run
                        try:
                            import memory_cards
                            for c in memory_cards.list_cards(status="active"):
                                memory_cards.bump_usage(c["id"])
                        except Exception:
                            pass
                else:
                    memd = {}
                # only meaningfully-nonzero dials (|v| >= 0.05); steer.active() drops exact-zeros but a
                # slider nudged to a hair (e.g. 0.02) still slips through and would clutter the record.
                dials = SUB.steer.active() if (SUB and hasattr(SUB, "steer")) else {}
                dials = {k: v for k, v in dials.items() if abs(float(v)) >= 0.05}
                runlog.record(source=source, client=self._client(self.headers.get("User-Agent", "")),
                              model=str(model), substrate=SUBNAME, messages=messages, response=response,
                              memory=memd, behavior={"active_dials": dials}, started=started, error=error,
                              trace=trace)
            except Exception:
                pass

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/", "/index.html", "/instrument.html"):
                return self._html("instrument.html")
            if p == "/substrate":
                return self._json(200, {"active": SUBNAME, "available": ["qwen", "dream"]})
            if p == "/v1/models":            # OpenAI-compatible model list (so OAI clients connect)
                return self._json(200, {"object": "list", "data": [
                    {"id": "clozn-qwen", "object": "model", "owned_by": "clozn"}]})
            if p == "/engine/health":
                try:
                    return self._json(200, {"engine": ENGINE.health()})
                except Exception as e:
                    return self._json(502, {"error": f"engine unreachable: {e}"})
            if p == "/state":
                return self._json(200, {"substrate": SUBNAME, **(SUB.state() if SUB else {})})
            if p.startswith("/memory/") and p.endswith("/runs"):   # E1: which runs used this card
                cid = p[len("/memory/"):-len("/runs")]
                return self._json(200, {"card_id": cid, "runs": _runs_for_card(cid)})
            if p == "/runs":                 # the Run Log -- every interaction, newest first (the Studio Runs page)
                import runlog
                return self._json(200, {"runs": runlog.list_runs(80)})
            if p.startswith("/runs/"):
                import runlog
                r = runlog.get_run(p.split("/runs/", 1)[1])
                return self._json(200, r) if r else self._json(404, {"error": "run not found"})
            if p.endswith((".html", ".css", ".js")):
                fn = os.path.normpath(os.path.join(DEMO, p.lstrip("/")))   # serve subdirs (pages/) too, safely
                if fn.startswith(os.path.normpath(DEMO)) and os.path.isfile(fn):
                    ct = ("text/html" if p.endswith(".html") else
                          "text/css" if p.endswith(".css") else "application/javascript")
                    return self._send(200, open(fn, encoding="utf-8").read(), ct + "; charset=utf-8")
            self._json(404, {"error": "GET " + p})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            p = self.path.split("?")[0].rstrip("/") or "/"
            if p == "/substrate":
                name = str(body.get("name", "qwen"))
                if name == SUBNAME:
                    return self._json(200, {"active": SUBNAME, "switched": False})
                if name not in ("qwen", "dream"):
                    return self._json(400, {"error": "unknown substrate"})
                self._json(200, {"active": name, "switched": True, "note": "reloading -- poll /substrate"})
                threading.Thread(target=lambda: (time.sleep(0.4), switch_substrate(name)), daemon=True).start()
                return
            if p.startswith("/runs/") and p.endswith("/replay"):   # F1: re-run a past run under changed state -> a child run
                rid = p[len("/runs/"):-len("/replay")]
                import runlog
                run = runlog.get_run(rid)
                if run is None:
                    return self._json(404, {"error": "run not found"})
                if not (SUB and getattr(SUB, "chat", None)):   # replay generates -> needs the qwen (chat) substrate
                    return self._json(503, {"error": "replay needs the qwen substrate"})
                changes = body.get("changes_applied", body.get("changes")) or {}
                try:
                    import replay
                    child = replay.replay(run, changes, SUB)
                except Exception as e:
                    return self._json(500, {"error": f"replay failed: {type(e).__name__}: {e}"})
                if child is None:
                    return self._json(500, {"error": "replay failed"})
                return self._json(200, child)
            if p.startswith("/runs/") and p.endswith("/propose-memory"):   # E2: propose a pending card from a past run
                rid = p[len("/runs/"):-len("/propose-memory")]
                import runlog
                run = runlog.get_run(rid)
                if run is None:
                    return self._json(200, {"ok": False, "reason": "no such run"})
                # only a substrate whose memory exposes propose_memory qualifies (QwenSubstrate). Dream's
                # memory has no such method -> the proposal is simply not offered there.
                mem = getattr(SUB, "memory", None) if SUB else None
                if mem is None or not hasattr(mem, "propose_memory"):
                    return self._json(200, {"ok": False, "reason": "proposal not available for this substrate"})
                import memory_cards
                # Neutralize tone steering during the extraction so the dials don't color the read -- snapshot
                # SUB.steer.strength, zero it, and RESTORE in a finally (mirror replay.py; never persist this).
                steer = getattr(SUB, "steer", None)
                saved_strength = dict(getattr(steer, "strength", {}) or {}) if steer is not None else None
                try:
                    if steer is not None:
                        try:
                            steer.strength = {}             # all dials neutral for the duration of the read
                        except Exception:
                            pass
                    text = mem.propose_memory(run["messages"], run.get("response"))
                except Exception as e:                      # propose_memory is defensive, but never crash the handler
                    return self._json(200, {"proposed": False, "reason": f"proposal failed: {type(e).__name__}"})
                finally:
                    if steer is not None and saved_strength is not None:
                        try:
                            steer.strength = dict(saved_strength)   # restore EXACTLY (temp neutralization only)
                        except Exception:
                            pass
                if text is None:
                    return self._json(200, {"proposed": False,
                                            "reason": "no durable preference found in this run"})
                card = memory_cards.create(text, status="pending", kind="preference",
                                           risk=_risk_of(text), source_run_id=rid,
                                           evidence=f"proposed from run {rid}")
                if not card:
                    return self._json(200, {"proposed": False, "reason": "could not create card"})
                return self._json(200, {"proposed": True, "card": card})
            if p == "/engine/harvest":   # READ the real C++ runtime's activations (any substrate; the engine is separate)
                try:
                    h = ENGINE.harvest(str(body.get("text", ""))[:300])
                    norms = np.linalg.norm(h.activations, axis=1)
                    return self._json(200, {"tokens": h.tokens, "layer": int(h.layer), "n_embd": h.n_embd,
                                            "norms": [round(float(x), 3) for x in norms]})
                except Exception as e:
                    return self._json(502, {"error": f"engine: {e}"})
            if p == "/engine/observe":   # WRITE a scaled residual back at one token, OBSERVE how the prediction moves
                try:
                    pos = int(body.get("position", 0))
                    scale = float(body.get("scale", 4.0))

                    def tf(a):
                        a = a.copy()
                        if 0 <= pos < a.shape[0]:
                            a[pos] = a[pos] * scale
                        return a

                    h, obs = ENGINE.edit_and_observe(str(body.get("text", ""))[:300], transform=tf, positions=[pos])
                    return self._json(200, {"summary": obs.summary(), "shifted": obs.shifted(),
                                            "moved_l2": obs.moved_l2, "baseline_top": obs.baseline_top,
                                            "edited_top": obs.edited_top, "tokens": h.tokens,
                                            "position": pos, "scale": scale})
                except Exception as e:
                    return self._json(502, {"error": f"engine: {e}"})
            if p == "/engine/concepts":   # the brain's concepts, but read from the Qwen GGUF engine (harvest L15 + SAE)
                try:
                    if not (SUB and getattr(SUB, "brain", None)):
                        return self._json(409, {"error": "concepts need the qwen substrate (it holds the SAE)"})
                    return self._json(200, SUB.brain.concepts_from_engine(
                        str(body.get("text", ""))[:300], ENGINE_QWEN, int(body.get("layer", 15))))
                except Exception as e:
                    return self._json(502, {"error": f"engine-qwen: {e}"})
            if p == "/engine/steer/axes":   # the tone dials, but they apply on the GGUF via the engine
                from steering import AXES
                es = _engine_steer()
                return self._json(200, {"axes": [{"name": k, "poles": AXES[k]["poles"]} for k in AXES],
                                        "ready": bool(es and es.ready), "engine": bool(ENGINE_QWEN)})
            if p == "/engine/steer/check":   # A/B one dial on the engine GGUF: baseline vs steered generation
                es = _engine_steer()
                if es is None:
                    return self._json(502, {"error": "no engine configured (set CLOZN_ENGINE_QWEN_PORT)"})
                try:
                    prompt = str(body.get("prompt", "Tell me about the city at night."))[:300]
                    axis, val = str(body.get("axis", "warm")), float(body.get("value", 1.0))
                    mx = int(body.get("max_tokens", 60))
                    base = es.generate(prompt, strength={}, max_new=mx)            # no dial = the baseline
                    stee = es.generate(prompt, strength={axis: val}, max_new=mx)
                    return self._json(200, {"prompt": prompt, "axis": axis, "value": val,
                                            "baseline": base.strip(), "steered": stee.strip()})
                except Exception as e:
                    return self._json(502, {"error": f"engine-steer: {e}"})
            if p == "/engine/chat":   # THE HYBRID: chat on the GGUF via the engine, with the HF-trained memory injected
                if ENGINE_QWEN is None:
                    return self._json(502, {"error": "no engine configured"})
                msgs = body.get("messages", [])
                t0 = time.time()
                try:
                    prompt = _qwen_tmpl(msgs)
                    mx = int(body.get("max_tokens", 220))
                    kw = {}
                    # MEMORY: the live HF prefix if a qwen substrate is loaded, else the SAVED prefix from disk
                    # -- so engine-chat works with NO HF model resident (the pure-engine substrate).
                    mem = getattr(SUB, "memory", None) if SUB else None
                    if mem is not None and getattr(mem, "prefix", None) is not None:
                        prefix = mem.prefix.detach().float().cpu()
                        ms = float(getattr(mem, "memory_strength", 1.0))
                    else:
                        prefix, ms = _disk_memory()
                    if prefix is not None:                         # inject the trained soft prefix (scaled by the dial)
                        kw = {"prefix_embd": (prefix * ms).flatten().tolist(), "prefix_rows": int(prefix.shape[0])}
                    # TONE: live dial values if a substrate is up, else the saved values from disk
                    st = getattr(getattr(SUB, "steer", None), "strength", None) if SUB else None
                    if not st:
                        st = _disk_dials()
                    if st and any(st.values()):
                        es = _engine_steer()
                        sv = es.steer_vector(st) if es is not None else None
                        if sv:
                            kw["steer_vec"] = sv
                            kw["steer"] = {"coef": 1.0, "layer": es.layer}
                    # Generate + capture a per-token trace alongside (B3). Reply is byte-identical to the
                    # plain complete(); the trace feeds the Run Inspector timeline. steps=[] (diffusion, or a
                    # stream hiccup) -> runlog stores a clean empty trace.
                    reply_raw, steps = _engine_complete_traced(ENGINE_QWEN, prompt, mx, kw)
                    reply = reply_raw.strip()
                    # Pass the raw step list; runlog.record normalizes it -> {tokens, confidence, alternatives}.
                    self._log_run("engine_chat", msgs, reply, "clozn-qwen (engine)", t0, trace=steps)
                    return self._json(200, {"reply": reply, "memory": bool(kw.get("prefix_embd")),
                                            "tone": bool(kw.get("steer_vec")), "via": "engine (GGUF)"})
                except Exception as e:
                    self._log_run("engine_chat", msgs, "", "clozn-qwen (engine)", t0, error=str(e))
                    return self._json(502, {"error": f"engine-chat: {e}"})
            if p == "/v1/chat/completions":   # OpenAI-compatible: chat with memory prefix + tone steering applied
                if not (SUB and getattr(SUB, "chat", None)):
                    return self._json(503, {"error": "chat needs the qwen substrate"})
                msgs, mx = body.get("messages", []), int(body.get("max_tokens", 256))
                if body.get("stream") and getattr(SUB, "chat_stream", None):
                    return self._sse_chat(msgs, mx, str(body.get("model", "clozn-qwen")))
                t0 = time.time()
                reply = SUB.chat(msgs, mx, float(body.get("temperature", 0.7)) > 0)
                self._log_run("openai_api", msgs, reply, body.get("model", "clozn-qwen"), t0)
                return self._json(200, {"id": "chatcmpl-clozn", "object": "chat.completion",
                                        "created": int(time.time()), "model": body.get("model", "clozn-qwen"),
                                        "choices": [{"index": 0, "finish_reason": "stop",
                                                     "message": {"role": "assistant", "content": reply}}],
                                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
            if p == "/say":   # studio chat (qwen memory model) -> capture it as a run
                if not (SUB and getattr(SUB, "handle", None)):
                    return self._json(409, {"error": f"'{p}' isn't served by the '{SUBNAME}' substrate"})
                msg = str(body.get("message", ""))
                t0 = time.time()
                try:
                    r = SUB.handle(p, body)
                except Exception as e:
                    self._log_run("studio_chat", [{"role": "user", "content": msg}], "",
                                  "clozn-qwen", t0, error=str(e))
                    return self._json(500, {"error": f"{type(e).__name__}: {e}"})
                if r is None:
                    return self._json(409, {"error": f"'{p}' isn't served by the '{SUBNAME}' substrate",
                                            "need": "qwen", "active": SUBNAME})
                # HF studio chat (QwenSubstrate): no `trace` -> the run's trace stays an empty {}. This path
                # generates via transformers with no native per-token confidence/alternatives; capturing one
                # (output_scores on the streaming path) is a separate, riskier issue and out of scope (B3).
                self._log_run("studio_chat", [{"role": "user", "content": msg}],
                              str(r.get("reply", "")), "clozn-qwen", t0)
                return self._json(200, r)
            if p == "/denoise":   # Dream diffusion window -> capture it as a run
                if not (SUB and getattr(SUB, "handle", None)):
                    return self._json(409, {"error": f"'{p}' isn't served by the '{SUBNAME}' substrate",
                                            "need": "dream", "active": SUBNAME})
                prompt = str(body.get("prompt", ""))
                t0 = time.time()
                try:
                    r = SUB.handle(p, body)
                except Exception as e:
                    self._log_run("denoise", [{"role": "user", "content": prompt}], "",
                                  "clozn-dream", t0, error=str(e))
                    return self._json(500, {"error": f"{type(e).__name__}: {e}"})
                if r is None:
                    return self._json(409, {"error": f"'{p}' isn't served by the '{SUBNAME}' substrate",
                                            "need": "dream", "active": SUBNAME})
                self._log_run("denoise", [{"role": "user", "content": prompt}],
                              str(r.get("final_text", "")), "clozn-dream", t0)
                return self._json(200, r)
            try:
                r = SUB.handle(p, body) if SUB else None
                if r is None:
                    return self._json(409, {"error": f"'{p}' isn't served by the '{SUBNAME}' substrate",
                                            "need": "dream" if p == "/denoise" else "qwen", "active": SUBNAME})
                self._json(200, r)
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return H


def main():
    global ARGS, SUB, SUBNAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--substrate", default="qwen", choices=("qwen", "dream", "engine"))
    ARGS = ap.parse_args()
    SUBNAME = ARGS.substrate
    print(f"clozn server: loading '{SUBNAME}' substrate ...", flush=True)
    SUB = load_substrate(SUBNAME)
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), make_handler())
    print(f"\n  CLOZN instrument -> http://{ARGS.host}:{ARGS.port}/   (substrate: {SUBNAME})\n", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
