"""clozn.server.facts_store -- the FACTS tier: the slot-memory store wired to the studio.

SlotBox (the verbatim cue->answer store INSIDE the model, with the surprise gate) plus the fact-mining
regex + miner. Extracted from clozn.server.app; app keeps the process-wide state owners (SLOTS/SNAPSHOTS
and their _slots_box/_snap_store constructors, which rebind app globals) and re-exports this module's
names, so `cs.SlotBox` / `ctx._mine_fact` keep resolving on the app module unchanged.
"""
from __future__ import annotations

import os
import threading
import time

from clozn.server import app as ctx   # the seam: live server state + patchable helpers (see docstring)

# ------- the FACTS tier: slot-memory store wired to the studio ----------------------------------------
# slotmem_qwen.SlotMem is the explicit, editable, honest-about-ignorance fact store (centered-key
# addressing, surprise-gated writes, confidence-gate abstention -- proven 0.95 flat to N=200). SlotBox
# is the thin studio wiring: it lazily builds ONE SlotMem SHARING the substrate's Qwen-7B (SUB.memory
# .model -- no second model, per the item spec), keeps a PER-PROFILE store (~/.clozn/profiles/<name>
# .slots.pt), and gates every operation behind memory_facts (default OFF -- the latency rule: a slot
# read is an extra forward, kept off the 7B hot path until measured; when on, we log slot_ms honestly).
#
# v1 CONTRACT (deliberately conservative -- protect the shipped chat path): a slot READ produces a
# RECEIPT (hit / gate value / abstention / the answer the store would inject) that the Facts panel
# shows and the runlog records; it does NOT alter the chat reply, so the 7B generation stays
# byte-identical whether facts are on or off. Actually STEERING the reply with the injected value is
# the next rung (documented seam). Auto-WRITE does mutate the store: it runs the surprise gate on a
# candidate (cue -> answer) mined from the turn, so the gate visibly refuses what the model already
# knows (the Titans write policy, the load-bearing provable part).

class SlotBox:
    """Owns the studio's live SlotMem + its per-profile persistence. Built lazily on first use so a
    fresh install with facts OFF never pays for it. Every public method is a no-op / empty receipt when
    `memory_facts` is off or no shareable model is loaded -- the caller stays oblivious to both."""

    def __init__(self, mem_provider):
        # mem_provider() -> the substrate memory object (SUB.memory) whose .model/.tok we SHARE, or None.
        self._mem_provider = mem_provider
        self._slots = None                 # the SlotMem (None until built)
        self._profile = None               # the profile name whose store is currently resident
        self._lock = threading.Lock()      # serialize build + store mutations (the model is shared)

    # ---- lifecycle --------------------------------------------------------------------------------
    def _shared_model(self):
        try:
            m = self._mem_provider()
        except Exception:
            m = None
        model = getattr(m, "model", None)
        tok = getattr(m, "tok", None)
        return (model, tok) if (model is not None and tok is not None) else (None, None)

    def _build(self):
        """Build the SlotMem on the shared backbone + load the active profile's store. Returns the
        SlotMem or None (no model yet, or slotmem import/HF unavailable). Holds _lock."""
        if self._slots is not None:
            return self._slots
        model, tok = self._shared_model()
        if model is None:
            return None
        try:
            import clozn.memory.facts_mode as facts_mode
            import clozn.memory.slotmem_qwen.store as slotmem_qwen
            self._slots = slotmem_qwen.SlotMem.from_shared(model, tok, facts_mode.LAYER)
        except Exception as e:
            print(f"[facts] could not build slot store: {type(e).__name__}: {e}", flush=True)
            self._slots = None
            return None
        self._load_active()               # bring the current profile's saved facts in
        return self._slots

    def _active_profile(self):
        try:
            return ctx._active_profile_name()
        except Exception:
            return None

    def _load_active(self):
        """(Re)load the store for the currently-active profile into self._slots. Silent on a missing
        file (a profile with no facts yet is empty, not an error); a layer mismatch is logged + skipped."""
        if self._slots is None:
            return
        import clozn.memory.facts_mode as facts_mode
        prof = self._active_profile()
        path = facts_mode.store_path(prof)
        self._slots.entries = []
        self._profile = prof
        if os.path.isfile(path):
            try:
                self._slots.load(path)
            except Exception as e:
                print(f"[facts] skipped loading {path}: {type(e).__name__}: {e}", flush=True)

    def _save_active(self):
        if self._slots is None:
            return
        import clozn.memory.facts_mode as facts_mode
        try:
            self._slots.save(facts_mode.store_path(self._profile))
        except Exception as e:
            print(f"[facts] save failed: {type(e).__name__}: {e}", flush=True)

    def _ensure_profile(self):
        """If the active profile changed since we last loaded, swap the resident store to it (per-profile
        isolation: one persona's facts must never read another's). Cheap string compare; loads only on a
        real change."""
        if self._slots is None:
            return
        if self._active_profile() != self._profile:
            self._load_active()

    def on_profile_switch(self):
        """Called by a profile switch: reload the new profile's store if the box is already live. When
        facts are off / not built yet, nothing to do (the store loads lazily on first use)."""
        with self._lock:
            if self._slots is not None:
                self._load_active()

    # ---- reads / writes (all gated by memory_facts) ----------------------------------------------
    def status(self):
        """{enabled, layer, profile, count} -- the Facts panel header. Never builds the model just to
        answer (count is 0 until the store is actually resident)."""
        import clozn.memory.facts_mode as facts_mode
        n = len(self._slots.entries) if self._slots is not None else 0
        return {"enabled": facts_mode.enabled(), "layer": facts_mode.LAYER,
                "profile": self._profile or self._active_profile() or "default", "count": n}

    def list_entries(self):
        """[{cue, answer, label}] for the resident store, [] when off / unbuilt. Read-only, no model
        forward -- safe to call on every Facts-panel load."""
        import clozn.memory.facts_mode as facts_mode
        if not facts_mode.enabled():
            return []
        with self._lock:
            if self._build() is None:
                return []
            self._ensure_profile()
            return [{"cue": e["cue"], "answer": e["answer"], "label": e["label"]}
                    for e in self._slots.entries]

    def add(self, cue: str, answer: str, gate: bool = True):
        """Explicit fact write with the SURPRISE GATE on (the refusal is the receipt: a fact the model
        already knows is SKIPPED, not stored). Persists on a real write. {ok, written, surprise, reason?}."""
        import clozn.memory.facts_mode as facts_mode
        cue, answer = str(cue or "").strip(), str(answer or "")
        if not cue or not answer.strip():
            return {"ok": False, "reason": "need a cue and an answer"}
        if not facts_mode.enabled():
            return {"ok": False, "reason": "the facts tier is off (enable it first)"}
        with self._lock:
            if self._build() is None:
                return {"ok": False, "reason": "no model loaded to hold the fact store"}
            self._ensure_profile()
            with ctx._TRAIN_LOCK:              # the store write runs forwards on the shared model
                r = self._slots.write(cue, answer, gate=gate)
                if r.get("written"):
                    self._slots.calibrate_gate()
            if r.get("written"):
                self._save_active()
                return {"ok": True, "written": True, "surprise": r.get("surprise")}
            return {"ok": True, "written": False, "surprise": r.get("surprise"),
                    "reason": "the model already knows this (surprise below the write gate) -- not stored"}

    def delete(self, cue: str | None = None, index=None):
        """Surgical per-entry removal (the slotmem contract: the victim drops, every other entry stays
        bit-identical). Match by exact cue, else by index. Persists. {ok, removed, remaining}."""
        import clozn.memory.facts_mode as facts_mode
        if not facts_mode.enabled():
            return {"ok": False, "reason": "the facts tier is off"}
        with self._lock:
            if self._build() is None:
                return {"ok": False, "reason": "no fact store loaded"}
            self._ensure_profile()
            ents = self._slots.entries
            victim = None
            if cue is not None and str(cue).strip():
                victim = next((k for k, e in enumerate(ents) if e["cue"] == str(cue)), None)
            elif index is not None:
                try:
                    idx = int(index)
                    victim = idx if 0 <= idx < len(ents) else None
                except (TypeError, ValueError):
                    victim = None
            if victim is None:
                return {"ok": False, "reason": "no matching fact"}
            removed = ents.pop(victim)["cue"]
            self._slots.calibrate_gate()  # the abstain floor is derived from the store -> recompute
            self._save_active()
            return {"ok": True, "removed": removed, "remaining": len(ents)}

    def read_receipt(self, query: str):
        """The honest read RECEIPT for a query: which entry the store WOULD fire (or that it abstains),
        the key similarity, the abstain floor, the answer it would inject, and the measured slot_ms. Does
        NOT alter any chat reply (v1). {enabled, hit, abstained, sim, gate_floor, cue, answer, slot_ms}."""
        import clozn.memory.facts_mode as facts_mode
        if not facts_mode.enabled():
            return {"enabled": False}
        query = str(query or "").strip()
        with self._lock:
            if self._build() is None or not self._slots.entries:
                return {"enabled": True, "hit": None, "abstained": True, "empty": True,
                        "count": 0, "slot_ms": 0.0}
            self._ensure_profile()
            t0 = time.time()
            with ctx._TRAIN_LOCK:             # the read is a forward on the shared model
                r = self._slots.read(query, gated=True)
            slot_ms = round((time.time() - t0) * 1000.0, 1)
            hit, abst = r.get("hit"), r.get("abstained", False)
            out = {"enabled": True, "hit": hit, "abstained": bool(abst),
                   "sim": (round(float(r["sim"]), 4) if r.get("sim") is not None else None),
                   "gate_floor": (round(float(self._slots.gate_floor), 4)
                                  if self._slots.gate_floor is not None else None),
                   "count": len(self._slots.entries), "slot_ms": slot_ms}
            if hit is not None and not abst:
                e = self._slots.entries[hit]
                out["cue"], out["answer"] = e["cue"], e["answer"]
            return out

    def auto_write(self, messages, reply):
        """Surprise-gated auto-write FROM CONVERSATION: mine a single declarative (cue -> answer) from the
        last user turn and write it under the gate, so the gate refuses what the model already knows. A
        no-op (returns None) when off, when nothing mineable is found, or when the model isn't loaded.
        Best-effort + defensive -- it must never break a chat turn. Returns the write receipt when it
        actually attempted a write (for the runlog), else None."""
        import clozn.memory.facts_mode as facts_mode
        if not facts_mode.enabled():
            return None
        cand = _mine_fact(ctx._last_user(messages))
        if cand is None:
            return None
        cue, answer = cand
        try:
            with self._lock:
                if self._build() is None:
                    return None
                self._ensure_profile()
                with ctx._TRAIN_LOCK:
                    r = self._slots.write(cue, answer, gate=True)
                    if r.get("written"):
                        self._slots.calibrate_gate()
                if r.get("written"):
                    self._save_active()
                return {"cue": cue, "answer": answer, **r}
        except Exception as e:
            print(f"[facts] auto-write skipped: {type(e).__name__}: {e}", flush=True)
            return None


# One process-wide SlotBox, bound to whatever substrate is live (its _mem_provider reads SUB fresh, so a
# substrate swap is picked up automatically). None until the first substrate boots.

import re as _re

_FACT_RE = _re.compile(
    r"\b((?:my|our|the|his|her|their)\b[\w '\-]{1,40}?)\s+(?:is|are|was|were)\s+(?:called\s+|named\s+)?"
    r"([A-Za-z0-9][\w '\-]{0,40}?)\s*[.!?]?$",
    _re.IGNORECASE)


def _mine_fact(text: str):
    """One (cue, answer) from a short declarative user turn, or None. cue is the statement's subject
    rendered as a completion prompt ("My dog's name is" -> answer " Biscuit"); answer carries the leading
    space the store's value schedule expects. None when the turn is a question, too long, or not a clean
    "<subject> is <value>"."""
    t = str(text or "").strip()
    if not t or "?" in t or len(t) > 120 or len(t.split()) > 20:
        return None
    m = _FACT_RE.search(t)
    if not m:
        return None
    subj, val = m.group(1).strip(), m.group(2).strip()
    if not subj or not val or len(val) < 2:
        return None
    # rebuild the cue as the model would be prompted to COMPLETE it, preserving the copula the user used.
    copula = _re.search(r"\b(is|are|was|were)\b", t[m.start():], _re.IGNORECASE)
    verb = copula.group(1).lower() if copula else "is"
    cue = f"{subj} {verb}"
    return cue, " " + val


