"""Memory-mode + trait-card surfaces that live directly in the HTTP dispatch (as opposed to the per-action
`/memory/cards`, `/memory/add`, `/memory/approve`, ... family, which is substrate-polymorphic domain
dispatch handled by `Substrate._memory` in clozn.server.app and reached through the generic
`SUB.handle(path, body)` fallback -- not a per-path HTTP route, so it stays there). This module covers:
which mechanism carries the cards (GET/POST /memory/mode, works on ANY substrate), which runs used a
given card (GET /memory/<id>/runs), and proposing a pending card from a past run
(POST /runs/<id>/propose-memory). Mechanical extraction; behavior unchanged. -> clozn.memory.
"""
from clozn.server import app as ctx


def try_get(h, p):
    if p == "/memory/mode":          # which mechanism carries the cards (works on ANY substrate)
        import clozn.memory.mode as memory_mode
        h._json(200, {"mode": ctx._memory_mode(), "modes": list(memory_mode.MODES)})
        return True
    if p.startswith("/memory/") and p.endswith("/runs"):   # E1: which runs used this card
        cid = p[len("/memory/"):-len("/runs")]
        h._json(200, {"card_id": cid, "runs": ctx._runs_for_card(cid)})
        return True
    return False


def try_post(h, p, body):
    if p == "/memory/mode":   # swap the memory mechanism (persisted; takes effect immediately)
        import clozn.memory.mode as memory_mode
        mode = str(body.get("mode", "")).strip().lower()
        if mode not in memory_mode.MODES:
            h._json(400, {"error": f"unknown mode (want one of {list(memory_mode.MODES)})"})
            return True
        if not memory_mode.set_mode(mode):
            h._json(200, {"ok": False, "reason": "could not persist the mode setting"})
            return True
        out = {"ok": True, "mode": mode}
        # Toggling BACK to internalized: prompt-mode card edits never consolidated, so the trained
        # prefix can be STALE relative to the cards. If the active set differs from what the
        # current prefix embodies (_trained_rules), kick the normal background retrain so chats
        # don't serve a personality the cards no longer describe. Cheap guards: nothing to do when
        # there's no live memory, or when cards and prefix are both empty.
        if mode == "internalized" and ctx.SUB is not None and getattr(ctx.SUB, "_mem", None) is not None:
            m = ctx.SUB._mem
            try:
                import clozn.memory.cards as memory_cards
                active = memory_cards.active_texts()
                trained = list(getattr(m, "_trained_rules", []) or [])
                if set(active) != set(trained) and (active or getattr(m, "prefix", None) is not None):
                    out["resync"] = ctx._start_retrain(m, "mode-switch", None, force=True)
            except Exception:
                pass
        h._json(200, out)
        return True
    if p.startswith("/runs/") and p.endswith("/propose-memory"):   # E2: propose a pending card from a past run
        rid = p[len("/runs/"):-len("/propose-memory")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(200, {"ok": False, "reason": "no such run"})
            return True
        # only a substrate whose memory exposes propose_memory qualifies (QwenSubstrate). Dream's
        # memory has no such method -> the proposal is simply not offered there.
        mem = getattr(ctx.SUB, "memory", None) if ctx.SUB else None
        if mem is None or not hasattr(mem, "propose_memory"):
            h._json(200, {"ok": False, "reason": "proposal not available for this substrate"})
            return True
        import clozn.memory.cards as memory_cards
        # Neutralize tone steering during the extraction so the dials don't color the read -- snapshot
        # SUB.steer.strength, zero it, and RESTORE in a finally (mirror replay.py; never persist this).
        steer = getattr(ctx.SUB, "steer", None)
        saved_strength = dict(getattr(steer, "strength", {}) or {}) if steer is not None else None
        try:
            if steer is not None:
                try:
                    steer.strength = {}             # all dials neutral for the duration of the read
                except Exception:
                    pass
            text = mem.propose_memory(run["messages"], run.get("response"))
        except Exception as e:                      # propose_memory is defensive, but never crash the handler
            h._json(200, {"proposed": False, "reason": f"proposal failed: {type(e).__name__}"})
            return True
        finally:
            if steer is not None and saved_strength is not None:
                try:
                    steer.strength = dict(saved_strength)   # restore EXACTLY (temp neutralization only)
                except Exception:
                    pass
        if text is None:
            h._json(200, {"proposed": False, "reason": "no durable preference found in this run"})
            return True
        # PROVENANCE (the OBEY defense): the model just synthesized `text` as a
        # third-person summary of the conversation -- it can be a plausible-sounding hallucination
        # (a measured failure mode) or a faithfully-mined injected instruction. Cite
        # the actual user words it was drawn from so a reviewer (and has_provenance()) can check the
        # claim, not just read the model's word for it.
        turn, span = ctx._provenance_of(run["messages"])
        card = memory_cards.create(text, status="pending", kind="preference",
                                   risk=ctx._risk_of(text), source_run_id=rid,
                                   source_turn=turn, quoted_span=span,
                                   evidence=f"proposed from run {rid}")
        if not card:
            h._json(200, {"proposed": False, "reason": "could not create card"})
            return True
        # Route a STYLE preference to the tone dial that delivers it (see /memory/add). The card
        # is still created + pending; dial_suggestion is null for a topical/non-style proposal.
        h._json(200, {"proposed": True, "card": card, "dial_suggestion": ctx._dial_suggestion(text)})
        return True
    return False
