"""The actuarial journal over HTTP: the full ActuaryReport (GET /journal/actuary) and calibrated trust
spans for one run (POST /runs/<id>/trust_spans) -- run confidence mapped through the user's OWN
journal-derived acceptance curve (clozn.runs.actuary + clozn.runs.calibrated_trust).

Everything served here inherits actuary.py's honesty stance verbatim: "trusted" is a behavioral PROXY
(the run was kept -- not errored/truncated/test-failed/re-rolled), never a verified-correctness figure.
The dataclass `note` fields ride the wire unchanged, and trust_spans carries calibrated_trust.NOTE at
the top level, so no consumer can read these numbers without the proxy language attached. A journal
with no scored organic runs answers available:false -- absence over an invented curve.

NOT YET REGISTERED. To enable, in clozn/server/app.py add
    from clozn.server.routes import journal as _journal_routes
and append `_journal_routes` to _GET_ROUTES (anywhere before _runs_fallback_routes, which would
otherwise swallow nothing here -- /journal/* doesn't collide -- but keep the convention) and to
_POST_ROUTES (before any generic /runs/ fallback; today's _POST_ROUTES has none, so order is free).

The report is cached in-process for 60s (recomputing over a few hundred journal files is cheap but not
free on every poll); every response states its age as "computed_ago_s" so a consumer knows how stale
the curve it was served is.
"""
import threading
import time

_CACHE_TTL_S = 60.0
_LOCK = threading.Lock()
_REPORT = None          # the cached actuary.ActuaryReport
_COMPUTED_AT = 0.0      # time.time() when _REPORT was computed


def _report():
    """The (possibly cached) ActuaryReport + its age in seconds. Recomputes when older than
    _CACHE_TTL_S. Serialized under _LOCK: the server is threaded, and two concurrent recomputes over
    the same journal files would be wasted work (the read itself is safe, just not free)."""
    global _REPORT, _COMPUTED_AT
    with _LOCK:
        now = time.time()
        if _REPORT is None or (now - _COMPUTED_AT) >= _CACHE_TTL_S:
            from clozn.runs import actuary
            _REPORT = actuary.load_and_analyze()
            _COMPUTED_AT = now
        return _REPORT, max(0.0, time.time() - _COMPUTED_AT)


def try_get(h, p):
    if p == "/journal/actuary":   # the full actuarial report -- calibration/drift/failure model, all proxy-labelled
        import dataclasses
        report, age = _report()
        out = dataclasses.asdict(report)      # nested dataclasses -> plain dicts; every `note` rides verbatim
        out["computed_ago_s"] = round(age, 1)
        h._json(200, out)
        return True
    return False


def try_post(h, p, body):
    if p.startswith("/runs/") and p.endswith("/trust_spans"):   # confidence spans + the journal's acceptance curve
        rid = p[len("/runs/"):-len("/trust_spans")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if not run:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.runs import calibrated_trust, confidence_spans
        report, age = _report()
        cal = report.calibration
        if not calibrated_trust.has_curve(cal):
            # 200, not an error: an unscored journal is a clean, expected state. available:false beats
            # mapping through a curve that does not exist.
            h._json(200, {"available": False, "run_id": rid,
                          "reason": "the journal has no scored organic runs yet -- there is no "
                                    "acceptance curve to map this run's confidence through, and this "
                                    "endpoint will not invent one"})
            return True
        sp = confidence_spans.spans(run)                        # REUSE the existing segmentation, unchanged
        h._json(200, {"available": True, "run_id": rid,
                      "spans": calibrated_trust.attach(sp, cal),
                      "summary": confidence_spans.summarize(sp),
                      "n_scored": cal.n_scored,                 # how many journal runs the whole curve rests on
                      "computed_ago_s": round(age, 1),
                      "note": calibrated_trust.NOTE})
        return True
    return False
