"""replay.py -- the Replay & Compare engine (roadmap issue F1).

Re-run a stored run under a modified state (memory off, behavior neutral, a nudged/overridden dial, ...)
and persist the result as a CHILD run (parent_run_id set), so a replay is itself an inspectable run the
Studio can diff against its parent.

The load-bearing rule here: the LIVE studio must never be left mutated. A replay only *temporarily* changes
the substrate to generate one reply, then restores it exactly in a finally. It applies changes by writing
the same dials/strength the normal chat path reads:

  * memory  -- sub.memory.memory_strength (0 => prefix off / prompt block never injected), and -- in
               prompt memory mode -- sub.memory._exclude_card_ids, which the prompt-block compiler skips:
               `disabled_memory_ids` is REAL per-card ablation there (the receipts win the mode buys).
               In internalized mode a single card can't leave the fused prefix without a retrain, so
               `disabled_memory_ids` stays an honest "not applied" note.
  * behavior -- sub.steer.strength (sub.chat() engages the hook, which reads .strength)

so a replay is exactly "chat, but with these knobs different for one turn". The temporary dials are NEVER
persisted (no save_state during replay) -- that would silently rewrite the user's personality.

Stdlib + the siblings `runlog` / `memory_mode` / `memory_cards` (all stdlib-only themselves); the
substrate is passed in (the live SUB), never imported, so this module is unit-testable against a fake
substrate with no model.
"""
from __future__ import annotations

import time

import clozn.runs.store as runlog

NUDGE_STEP = 0.5            # a "nudge" bumps one dial this far toward its + pole (then set() caps it per-axis)


def _inject_prompt_instructions(messages: list[dict], instructions) -> list[dict]:
    """Add request-local system instructions without changing delivered messages.

    This mirrors prompt-memory assembly: caller system context remains first and the
    Clozn-owned block is appended to it, otherwise a system message is prepended.
    The journal still records ``messages`` unchanged; only assembled/final prompt
    evidence contains this intervention.
    """
    blocks = [str(value).strip() for value in (instructions or []) if str(value).strip()]
    if not blocks:
        return [dict(message) for message in messages]
    block = "\n\n".join(blocks)
    copied = [dict(message) for message in messages]
    for message in copied:
        if message.get("role") == "system":
            message["content"] = (str(message.get("content") or "") + "\n\n" + block).strip()
            return copied
    return [{"role": "system", "content": block}] + copied


def _mode() -> str:
    """The active memory mode; any hiccup resolves to "internalized" (the long-standing behavior),
    mirroring clozn_server._memory_mode."""
    try:
        import clozn.memory.mode as memory_mode
        return memory_mode.get_mode()
    except Exception:
        return "internalized"


def _snapshot_strength(steer) -> dict:
    """A shallow copy of the dial dict we can restore verbatim later (values are floats)."""
    try:
        return dict(getattr(steer, "strength", {}) or {})
    except Exception:
        return {}


def _apply_changes(changes: dict, sub, mode: str) -> dict:
    """Mutate the live substrate's knobs per `changes`, in place. Returns a small dict of notes for the
    parts that CAN'T take effect in the active memory mode (honest, never silently pretended). Never
    raises."""
    notes: dict = {}
    steer = getattr(sub, "steer", None)
    mem = getattr(sub, "memory", None) or getattr(sub, "_mem", None)

    # --- memory ---
    if changes.get("memory_off"):
        if mem is not None and hasattr(mem, "memory_strength"):
            mem.memory_strength = 0.0                       # prefix suppressed / block never injected

    if changes.get("disabled_memory_ids"):
        if mode == "prompt" and mem is not None:
            # REAL per-card ablation: the prompt-block compiler (clozn_server._prompt_block_for) skips
            # these ids, so the one replay generation runs on the block minus exactly those cards.
            # Instant -- no retrain. The attribute is snapshotted + restored by replay() below.
            mem._exclude_card_ids = [str(i) for i in changes["disabled_memory_ids"]]
        else:
            # internalized: the cards are FUSED into one trained prefix -- a single card can't be
            # ablated without a retrain. Say so rather than silently pretending.
            notes["disabled_memory_ids"] = ("not applied: in internalized memory mode the cards are fused "
                                            "into one trained prefix (per-card ablation needs a retrain); "
                                            "use memory_off, or switch memory mode to 'prompt'")
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


def _memory_scope_for_run(run: dict):
    """Reconstruct only exact scopes captured on the parent run.

    A User-Agent client association remains useful for sidecar lookup, but must never activate app-scoped
    memory during a replay. Project association is always explicit because it has no fallback source.
    """
    from clozn.memory.scope import MemoryScope
    app_key = run.get("client_key") if run.get("client_key_source") == "header" else None
    try:
        return MemoryScope(app_key=app_key, project_key=run.get("project_key"))
    except Exception:
        return MemoryScope()


def replay(run: dict, changes: dict, sub, reference_tokens=None, *,
           prompt_instructions=None, max_new: int | None = None) -> dict | None:
    """Re-run `run` under `changes` on the live substrate `sub`; record the result as a child run and return
    it. Returns None on any failure (a replay must never raise into the request handler).

    `run`      -- a run dict from runlog.get_run(id) (needs at least "id" and "messages").
    `changes`  -- the change spec (see module docstring): memory_off / behavior_off / nudge /
                  behavior_overrides / disabled_memory_ids / edited_memory / plain. {} == a plain re-roll.
    `sub`      -- the live substrate (SUB); must expose .chat(messages, max_new=, sample=). Its
                  .memory.memory_strength and .steer.strength are snapshotted and restored around generation.
    `reference_tokens` -- optional baseline reply token ids (prove-all early-stop): when the substrate's
                  chat() supports it, this ablated arm's generation HALTS at the first token that differs
                  from the baseline, so the child's `response` is a bit-exact prefix of the full reply and
                  the child carries `diverged`/`diverged_at`. A substrate whose chat() lacks the kwarg (torch
                  QwenSubstrate, test fakes) simply generates fully -- correctness is preserved because the
                  receipt layer falls back to the string compare when `diverged` is absent. The returned
                  child ALSO always carries `generated_ids` (the committed token ids, tier-independent), so a
                  baseline replay can hand its own tokens to the ablated arms even at a trace-dropping tier."""
    try:
        if not run or not isinstance(run, dict):
            return None
        from clozn.runs.think_tags import sanitize_messages
        messages = sanitize_messages(run.get("messages") or [])
        generation_messages = _inject_prompt_instructions(messages, prompt_instructions)
        chat = getattr(sub, "chat", None)
        if not callable(chat):
            return None
        changes = changes or {}
        mode = _mode()
        run_scope = _memory_scope_for_run(run)

        steer = getattr(sub, "steer", None)
        mem = getattr(sub, "memory", None) or getattr(sub, "_mem", None)

        # snapshot the exact live state so we can restore it verbatim (never leave the studio mutated)
        saved_strength = _snapshot_strength(steer)
        saved_mem = getattr(mem, "memory_strength", None) if mem is not None else None
        # normal state carries NO _exclude_card_ids at all -- remember whether it existed so restore can
        # delete it again (leaving it behind would silently ablate every future chat)
        had_exclude = mem is not None and hasattr(mem, "_exclude_card_ids")
        saved_exclude = getattr(mem, "_exclude_card_ids", None) if had_exclude else None

        t0 = time.time()
        notes = _apply_changes(changes, sub, mode)
        eff_dials = _effective_dials(sub)
        trace_steps: list = []          # per-token trace of the replay reply (B3) -- the baseline-vs-replay
        replay_memout: dict = {}        # exact post-change assembled/rendered prompt for the child receipt
        #                                 token diff needs it; replay previously never passed trace_out.
        try:
            # greedy:true (the receipts path) decodes deterministically, so the original-vs-replayed
            # difference is attributable to the CHANGE, not to sampling dice. Default stays sampled.
            # Capture the per-token trace when chat supports it (the real substrates do); fall back for a
            # chat that predates trace_out -- replay's sub contract is just (messages, max_new=, sample=).
            sampled = not bool(changes.get("greedy"))
            # Build the call kwargs and drop any the substrate's chat() doesn't accept (torch QwenSubstrate
            # / test fakes predate trace_out and/or reference_tokens). Progressive-degrade on the exact
            # unknown kwarg named in the TypeError, so the reply is never lost -- just less instrumented.
            budget = int(max_new) if isinstance(max_new, int) and max_new > 0 else 256
            call_kw = {"max_new": budget, "sample": sampled, "trace_out": trace_steps,
                       "mem_out": replay_memout, "memory_scope": run_scope}
            if reference_tokens:
                call_kw["reference_tokens"] = reference_tokens
            while True:
                try:
                    reply = chat(generation_messages, **call_kw)
                    break
                except TypeError as e:
                    msg = str(e)
                    dropped = next((k for k in ("reference_tokens", "trace_out", "mem_out", "memory_scope")
                                    if k in call_kw and k in msg), None)
                    if dropped is None:
                        raise                                # a real TypeError from inside chat, not a kwarg
                    del call_kw[dropped]
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
            if mem is not None:
                try:
                    if had_exclude:
                        mem._exclude_card_ids = saved_exclude
                    elif hasattr(mem, "_exclude_card_ids"):
                        del mem._exclude_card_ids
                except Exception:
                    pass

        reply = reply if isinstance(reply, str) else str(reply)

        # Capture the committed token ids NOW, from the in-memory trace -- BEFORE the capture-tier logic
        # below may drop trace_steps to []. A baseline replay hands these to its ablated arms as the
        # early-stop reference, and they must survive even at a trace-dropping tier.
        generated_ids = [int(s["id"]) for s in (trace_steps or [])
                         if isinstance(s, dict) and s.get("id") is not None]
        # The early-stop verdict (prove-all ablated arms): (diverged, diverged_at) or (None, None).
        diverged = diverged_at = None
        if hasattr(sub, "last_divergence"):
            try:
                diverged, diverged_at = sub.last_divergence()
            except Exception:
                diverged = diverged_at = None

        # the replay's own stop cause + repro metadata (engine substrate) -- the SAME fields a live run
        # carries, read after generation (the finally above doesn't touch these stashes). Per-substrate
        # best-effort: a substrate without them (e.g. an HF stub) simply records None / {}.
        finish = sub.last_finish_reason() if hasattr(sub, "last_finish_reason") else None
        meta = None
        try:
            if hasattr(sub, "run_meta"):
                meta = sub.run_meta() or None
        except Exception:
            meta = None
        # capture tier: record it, and drop the trace at light -- the same record policy as the live path.
        try:
            from clozn.runs import capture_mode
            _tier = capture_mode.tier()
            meta = {**(meta or {}), "capture_tier": _tier}
            if not capture_mode.captures_trace(_tier):
                trace_steps = []
        except Exception:
            pass

        # child memory summary: what memory looked like *for this replay* (strength reflects memory_off).
        # In prompt mode the summary is card-store-based and honors the per-card ablation: cards_applied
        # is the ELIGIBLE set (active minus the disabled ids) + applied_ids for the per-card receipt UI.
        # (Eligible, not per-turn-gated: replay can't see inside sub.chat; same convention as live
        # internalized runs, which record the whole active set.)
        excluded = changes.get("disabled_memory_ids") or []
        if mode == "prompt":
            eligible = None
            try:
                import clozn.memory.mode as memory_mode
                eligible = memory_mode.active_cards(excluded, request_scope=run_scope)
            except Exception:
                pass
            if eligible is None:                             # store unavailable -> id-less rules fallback
                eligible = [{"id": None, "text": t} for t in (getattr(mem, "rules", None) or []) if t]
            cards = [c.get("text", "") for c in eligible]
            ids = [c.get("id") for c in eligible]
        else:
            cards = list(getattr(mem, "rules", None) or getattr(mem, "cards", None) or []) if mem is not None else []
            ids = None
        memd = {
            "cards_applied": [] if changes.get("memory_off") else cards,
            "strength": 0.0 if changes.get("memory_off") else float(saved_mem if saved_mem is not None else 1.0),
            "has_prefix": (getattr(mem, "prefix", None) is not None) if mem is not None else False,
            "mode": mode,
            "proposed_cards": [],
        }
        if ids is not None:
            memd["applied_ids"] = [] if changes.get("memory_off") else ids
            scope_kinds = [c.get("scope_kind") for c in eligible]
            if any(kind in {"global", "app", "project"} for kind in scope_kinds):
                memd["applied_scope_kinds"] = (
                    [] if changes.get("memory_off") else
                    [kind if kind in {"global", "app", "project"} else "global"
                     for kind in scope_kinds]
                )
        if notes:
            memd["notes"] = notes

        meta = {**(meta or {}), "max_tokens": budget}
        identity = None
        try:
            if hasattr(sub, "identity_meta"):
                identity = sub.identity_meta() or None
        except Exception:
            identity = None
        rid = runlog.record(
            source="replay",
            client=run.get("client") or "studio",
            model=run.get("model"),
            substrate=run.get("substrate"),
            messages=messages,
            response=reply,
            memory=memd,
            behavior={"active_dials": eff_dials},
            trace=trace_steps,
            finish_reason=finish,
            meta=meta,
            parent_run_id=run.get("id"),
            changes_applied=changes,
            started=t0,
            assembled_messages=replay_memout.get("assembled_messages"),
            final_prompt=replay_memout.get("final_prompt"),
            identity=identity,
            session_key=run.get("session_key"),
            client_key=run.get("client_key"),
            client_key_source=run.get("client_key_source"),
            project_key=run.get("project_key"),
        )
        if rid is None:
            return None
        child = runlog.get_run(rid)
        if child is None:
            child = {"id": rid, "response": reply, "parent_run_id": run.get("id")}
        # Attach the early-stop bookkeeping to the returned child (not persisted in the run record's core
        # fields -- these are for the receipt orchestrator that called replay). `generated_ids` is the
        # committed-token reference a baseline hands to its ablated arms; `diverged`/`diverged_at` let the
        # receipt read the verdict without re-deriving it.
        child["generated_ids"] = generated_ids
        if diverged is not None:
            child["diverged"] = diverged
            child["diverged_at"] = diverged_at
        return child
    except Exception:
        return None
