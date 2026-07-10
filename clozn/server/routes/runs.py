"""The Run Log: list runs, fetch one, and its structural GET-only views (timeline / lineage / family /
confidence spans) -- all zero-generation reads over clozn.runs.store. Mechanical extraction of the
matching `if p == ...` / `if p.startswith("/runs/") and p.endswith(...)` branches out of
clozn.server.app's do_GET; behavior unchanged.

NOTE on the generic "/runs/<id>" fallback: it is deliberately NOT here yet. GET /runs/<id>/export
(receipts.py) is still checked in app.py's legacy dispatch until that family is split out, and a generic
prefix match here would shadow it (both start with "/runs/"). The fallback moves into this module only
once every more-specific /runs/<id>/<suffix> GET has its own registered family ahead of this one in the
dispatch order -- see app.py's `_GET_ROUTES` comment.
"""


def try_get(h, p):
    if p == "/runs":                 # the Run Log -- every interaction, newest first (the Studio Runs page)
        import clozn.runs.store as runlog
        h._json(200, {"runs": runlog.list_runs(80)})
        return True
    if p.startswith("/runs/") and p.endswith("/timeline"):   # ordered RunEvent list -- zero generation
        import clozn.runs.store as runlog
        rid = p[len("/runs/"):-len("/timeline")]
        run = runlog.get_run(rid)
        if not run:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.runs import timeline as run_timeline
        h._json(200, {"run_id": rid, "events": run_timeline.timeline(run)})
        return True
    if p.startswith("/runs/") and p.endswith("/lineage"):   # branch/replay ancestry + child tree
        import clozn.runs.store as runlog
        rid = p[len("/runs/"):-len("/lineage")]
        out = runlog.lineage(rid)
        if not out:
            h._json(404, {"error": "run not found"})
            return True
        h._json(200, out)
        return True
    if p.startswith("/runs/") and p.endswith("/family"):   # the WHOLE branch family as GET /runs-shaped
        # summaries -- the full lineage past the /runs 80-window, so the client's buildLineageFromRuns
        # builds the complete tree instead of the recent-runs slice. (Distinct from /lineage, which
        # returns the server-built ancestors/children/tree object.)
        import clozn.runs.store as runlog
        rid = p[len("/runs/"):-len("/family")]
        fam = runlog.lineage_family(rid)
        if fam is None:
            h._json(404, {"error": "run not found"})
            return True
        h._json(200, {"runs": fam})
        return True
    if p.startswith("/runs/") and p.endswith("/spans"):   # confidence spans -- the shape of the reply's certainty
        import clozn.runs.store as runlog
        rid = p[len("/runs/"):-len("/spans")]
        run = runlog.get_run(rid)
        if not run:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.runs import confidence_spans
        sp = confidence_spans.spans(run)
        h._json(200, {"run_id": rid, "spans": sp, "summary": confidence_spans.summarize(sp)})
        return True
    return False


def try_get_fallback(h, p):
    """The generic GET /runs/<id> catch-all -- registered LAST (see app.py's _GET_ROUTES), after every
    more-specific /runs/<id>/<suffix> family has had first refusal."""
    if p.startswith("/runs/"):
        import clozn.runs.store as runlog
        r = runlog.get_run(p.split("/runs/", 1)[1])
        h._json(200, r) if r else h._json(404, {"error": "run not found"})
        return True
    return False
