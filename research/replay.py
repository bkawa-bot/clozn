"""replay.py -- the Replay & Compare engine (roadmap issue F1).

Re-run a stored run under a modified state (memory off, behavior neutral, a nudged/overridden dial, ...)
and persist the result as a CHILD run (parent_run_id set), so a replay is itself an inspectable run the
Studio can diff against its parent.

The load-bearing rule here: the LIVE studio must never be left mutated. A replay only *temporarily* changes
the substrate to generate one reply, then restores it exactly in a finally. It applies changes by writing
the same dials/strength the normal chat path reads:

  * memory  -- sub.memory.memory_strength (the /v1/chat path passes gate=memory_strength; 0 => prefix off)
  * behavior -- sub.steer.strength (sub.chat() engages the hook, which reads .strength)

so a replay is exactly "chat, but with these knobs different for one turn". The temporary dials are NEVER
persisted (no save_state during replay) -- that would silently rewrite the user's personality.

Stdlib + the sibling `runlog` only; the substrate is passed in (the live SUB), never imported, so this
module is unit-testable against a fake substrate with no model.
"""
from __future__ import annotations

import time

import runlog

NUDGE_STEP = 0.5            # a "nudge" bumps one dial this far toward its + pole (then set() caps it per-axis)


def _snapshot_strength(steer) -> dict:
    """A shallow copy of the dial dict we can restore verbatim later (values are floats)."""
    try:
        return dict(getattr(steer, "strength", {}) or {})
    except Exception:
        return {}


def _apply_changes(changes: dict, sub) -> dict:
    """Mutate the live substrate's knobs per `changes`, in place. Returns a small dict of notes (e.g. the
    still-string-based memory-card toggles that are best-effort no-ops for now). Never raises."""
    notes: dict = {}
    steer = getattr(sub, "steer", None)
    mem = getattr(sub, "memory", None) or getattr(sub, "_mem", None)

    # --- memory ---
    if changes.get("memory_off"):
        if mem is not None and hasattr(mem, "memory_strength"):
            mem.memory_strength = 0.0                       # suppress the prefix for this one generation

    # the memory-card system is still string-based (rules list, no per-card ids); honor these as best-effort
    # notes for now rather than silently pretending they took effect.
    if changes.get("disabled_memory_ids"):
        notes["disabled_memory_ids"] = ("not applied: memory cards are string-based (no per-card ids yet); "
                                        "use memory_off to suppress the whole prefix")
    if changes.get("edited_memory"):
        notes["edited_memory"] = ("not applied: memory-card editing is not wired yet; "
                                  "use memory_off to compare with/without memory")

    # --- behavior / tone dials ---
    if steer is not None:
        if changes.get("behavior_off"):
            steer.clear()                                   # neutral: drop every dial for this turn

        overrides = changes.get("behavior_overrides")
        if isinstance(overrides, dict):
            for name, val in overrides.items():
                try:
                    steer.set(str(name), float(val))        # set() caps to the axis's per-axis max
                except Exception:
                    pass

        nudge = changes.get("nudge")
        if nudge:
            try:
                cur = float(getattr(steer, "strength", {}).get(str(nudge), 0.0))
                steer.set(str(nudge), cur + NUDGE_STEP)     # bump toward the + pole; set() caps it
            except Exception:
                pass

    return notes


def _effective_dials(sub) -> dict:
    """The dials actually in force after applying the changes (what shaped the child reply)."""
    steer = getattr(sub, "steer", None)
    if steer is None:
        return {}
    try:
        if hasattr(steer, "active"):
            return dict(steer.active())
        return {k: v for k, v in _snapshot_strength(steer).items() if v}
    except Exception:
        return {}


def replay(run: dict, changes: dict, sub) -> dict | None:
    """Re-run `run` under `changes` on the live substrate `sub`; record the result as a child run and return
    it. Returns None on any failure (a replay must never raise into the request handler).

    `run`      -- a run dict from runlog.get_run(id) (needs at least "id" and "messages").
    `changes`  -- the change spec (see module docstring): memory_off / behavior_off / nudge /
                  behavior_overrides / disabled_memory_ids / edited_memory / plain. {} == a plain re-roll.
    `sub`      -- the live substrate (SUB); must expose .chat(messages, max_new=, sample=). Its
                  .memory.memory_strength and .steer.strength are snapshotted and restored around generation.
    """
    try:
        if not run or not isinstance(run, dict):
            return None
        messages = run.get("messages") or []
        chat = getattr(sub, "chat", None)
        if not callable(chat):
            return None
        changes = changes or {}

        steer = getattr(sub, "steer", None)
        mem = getattr(sub, "memory", None) or getattr(sub, "_mem", None)

        # snapshot the exact live state so we can restore it verbatim (never leave the studio mutated)
        saved_strength = _snapshot_strength(steer)
        saved_mem = getattr(mem, "memory_strength", None) if mem is not None else None

        t0 = time.time()
        notes = _apply_changes(changes, sub)
        eff_dials = _effective_dials(sub)
        try:
            reply = chat(messages, max_new=256, sample=True)
        finally:
            # restore EXACTLY -- and never persist the temporary dials (no save_state here).
            if steer is not None:
                try:
                    steer.strength = dict(saved_strength)
                except Exception:
                    pass
            if mem is not None and saved_mem is not None:
                try:
                    mem.memory_strength = saved_mem
                except Exception:
                    pass

        reply = reply if isinstance(reply, str) else str(reply)

        # child memory summary: what memory looked like *for this replay* (strength reflects memory_off)
        cards = []
        if mem is not None:
            cards = list(getattr(mem, "rules", None) or getattr(mem, "cards", None) or [])
        memd = {
            "cards_applied": [] if changes.get("memory_off") else cards,
            "strength": 0.0 if changes.get("memory_off") else float(saved_mem if saved_mem is not None else 1.0),
            "has_prefix": (getattr(mem, "prefix", None) is not None) if mem is not None else False,
            "proposed_cards": [],
        }
        if notes:
            memd["notes"] = notes

        rid = runlog.record(
            source="replay",
            client="studio",
            model=run.get("model"),
            substrate=run.get("substrate"),
            messages=messages,
            response=reply,
            memory=memd,
            behavior={"active_dials": eff_dials},
            parent_run_id=run.get("id"),
            changes_applied=changes,
            started=t0,
        )
        if rid is None:
            return None
        child = runlog.get_run(rid)
        return child if child is not None else {"id": rid, "response": reply, "parent_run_id": run.get("id")}
    except Exception:
        return None
