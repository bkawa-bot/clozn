"""Propose-and-review: fold accumulated feedback signals into pending dial proposals
(POST /preferences), and approve (persist the dial) or dismiss one (POST /preferences/resolve). Approve
is the ONLY place a dial actually gets set from this surface. Mechanical extraction of the matching
`if p == "/preferences..."` branches out of clozn.server.app's do_POST; behavior unchanged.
-> clozn.behavior (feedback + preferences).
"""
import time

from clozn.server import app as ctx


def try_post(h, p, body):
    if p == "/preferences":   # propose-and-review: fold the feedback pattern into proposals + return
        from clozn.behavior import feedback, preferences   # the PENDING ones (a (dial,direction) lean that crossed the
        sigs = feedback.list_signals()  # threshold). refresh() creates/updates; it NEVER sets a dial.
        wd = body.get("window_days")
        if isinstance(wd, (int, float)):
            cut = time.time() - float(wd) * 86400
            sigs = [s for s in sigs if float(s.get("ts", 0)) >= cut]
        pending = preferences.refresh(
            sigs, threshold=int(body.get("threshold", preferences.DEFAULT_THRESHOLD)))
        h._json(200, {"pending": pending})
        return True
    if p == "/preferences/resolve":   # APPROVE (persist the dial) or DISMISS a proposal -- the review
        from clozn.behavior import preferences            # half of propose-and-review. Approve is the ONLY place a
        pr = preferences.resolve(str(body.get("id", "")), str(body.get("action", "")))  # dial is set.
        if pr is None:
            h._json(400, {"error": "unknown proposal id, or action not in {approve,dismiss}"})
            return True
        applied = None
        if pr["status"] == "approved" and ctx.SUB is not None and getattr(ctx.SUB, "steer", None) is not None:
            try:                       # persist the dial exactly like the F2 save-fix does (steer.set
                ctx.SUB.steer.set(pr["dial"], float(pr["suggested_value"]))   # caps per-axis)
                if hasattr(ctx.SUB.steer, "save_state") and getattr(ctx.SUB, "_pers_steer", None):
                    ctx.SUB.steer.save_state(ctx.SUB._pers_steer)
                applied = {"dial": pr["dial"],
                           "value": float(ctx.SUB.steer.strength.get(pr["dial"], pr["suggested_value"]))}
            except Exception as e:
                applied = {"error": f"{type(e).__name__}: {e}"}
        h._json(200, {"ok": True, "proposal": pr, "applied": applied})
        return True
    return False
