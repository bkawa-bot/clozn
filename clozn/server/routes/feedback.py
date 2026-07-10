"""Preference-signal capture: log a directional signal (e.g. a Run Inspector "Too verbose" click) tied to
the run that prompted it (POST /feedback), and the rollup a learning step reads -- per-(dial,direction)
counts + the last run driving each (POST /feedback/summary). Records only; changes nothing
(agency-agnostic). Mechanical extraction of the matching `if p == "/feedback..."` branches out of
clozn.server.app's do_POST; behavior unchanged. -> clozn.behavior.feedback.
"""


def try_post(h, p, body):
    if p == "/feedback":   # preference-signal CAPTURE (the plumbing) -- log a directional signal
        # (e.g. a Run Inspector "Too verbose" click) tied to the run that prompted it, so a later
        # accumulate-and-propose step can mine "you keep asking for concise". Records only; changes
        # nothing (agency-agnostic), and never fails the user's action over a feedback write.
        from clozn.behavior import feedback
        sig = feedback.record(body.get("run_id"), str(body.get("kind", "quick_repair")),
                              dial=body.get("dial"), direction=body.get("direction"),
                              meta=body.get("meta"))
        h._json(200, {"ok": True, "signal": sig})
        return True
    if p == "/feedback/summary":   # the rollup a learning step reads: per-(dial,direction) counts +
        from clozn.behavior import feedback            # the last run driving each, over an optional recent window (days)
        wd = body.get("window_days")
        ws = float(wd) * 86400 if isinstance(wd, (int, float)) else None
        h._json(200, feedback.summary(window_seconds=ws))
        return True
    return False
