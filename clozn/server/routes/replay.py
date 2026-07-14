"""Re-running a past run under changed state: POST /runs/<id>/replay (F1: apply changes_applied and
regenerate a child run) and POST /runs/<id>/counterfactual (M3: one dial re-gen). Both need a chat-capable
substrate since they regenerate. Mechanical extraction of the matching `if p.startswith("/runs/") and
p.endswith(...)` branches out of clozn.server.app's do_POST; behavior unchanged. -> clozn.replay.
"""
from clozn.server import app as ctx


def try_post(h, p, body):
    if p.startswith("/runs/") and p.endswith("/replay"):   # F1: re-run a past run under changed state -> a child run
        rid = p[len("/runs/"):-len("/replay")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        if not (ctx.SUB and getattr(ctx.SUB, "chat", None)):   # replay regenerates through the product model
            h._json(503, {"error": "replay requires a ready product model worker"})
            return True
        changes = body.get("changes_applied", body.get("changes")) or {}
        try:
            from clozn import replay
            child = replay.replay(run, changes, ctx.SUB)
        except Exception as e:
            h._json(500, {"error": f"replay failed: {type(e).__name__}: {e}"})
            return True
        if child is None:
            h._json(500, {"error": "replay failed"})
            return True
        h._json(200, child)
        return True
    if p.startswith("/runs/") and p.endswith("/counterfactual"):   # M3: one counterfactual dial re-gen
        rid = p[len("/runs/"):-len("/counterfactual")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        if not (ctx.SUB and getattr(ctx.SUB, "chat", None)):   # both arms regenerate through the product model
            h._json(503, {"error": "counterfactual requires a ready product model worker"})
            return True
        overrides = body.get("behavior_overrides")
        if not isinstance(overrides, dict) or not overrides:
            h._json(400, {"error": "need a behavior_overrides dict: {dial_name: value, ...}"})
            return True
        import clozn.replay.counterfactual as counterfactual
        try:
            out = counterfactual.counterfactual(run, overrides, ctx.SUB)
        except Exception as e:
            h._json(500, {"error": f"counterfactual failed: {type(e).__name__}: {e}"})
            return True
        if out is None:
            h._json(500, {"error": "counterfactual failed (bad overrides, or the replay "
                         "could not be generated)"})
            return True
        h._json(200, out)
        return True
    return False
