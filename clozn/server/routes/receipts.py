"""The receipts surface: a run's downloadable bundle (GET /runs/<id>/export), the free M1 explain read,
the M2 leave-one-out/redundancy receipts, the S3 teacher-forced rederivation, the J3 J-lens feed (general
+ per-run), and the M4 accountable-self narration. All read a past run and either reshape free signals
or regenerate arms to prove a causal claim about it. Mechanical extraction of the matching
`if p.startswith("/runs/") and p.endswith(...)` / `if p == "/jlens"` branches out of clozn.server.app's
do_GET/do_POST; behavior unchanged. -> clozn.receipts (+ the app's own J-lens envelope helpers).
"""
from clozn.server import app as ctx


def try_get(h, p):
    if p.startswith("/runs/") and p.endswith("/export"):   # one-call download: run + M1 explain + trace
        import clozn.runs.store as runlog
        import clozn.receipts.explain as explain
        import clozn.receipts.bundle as receipt_bundle
        from urllib.parse import urlparse, parse_qs
        rid = p[len("/runs/"):-len("/export")]
        run = runlog.get_run(rid)
        if not run:
            h._json(404, {"error": "run not found"})
            return True
        try:
            xr = explain.explain(run)                # M1: pure read/reshape, no generation
        except Exception:
            xr = None
        bundle = receipt_bundle.build(run, explain=xr)
        fmt = (parse_qs(urlparse(h.path).query).get("format") or ["json"])[0]
        if fmt == "md":
            h._send(200, receipt_bundle.to_markdown(bundle), "text/markdown; charset=utf-8",
                   extra_headers={"Content-Disposition": 'attachment; filename="' + rid + '.md"'})
            return True
        h._json(200, bundle,
               extra_headers={"Content-Disposition": 'attachment; filename="' + rid + '.json"'})
        return True
    return False


def try_post(h, p, body):
    if p.startswith("/runs/") and p.endswith("/explain"):   # M1: assemble the FREE signals -- zero generation
        rid = p[len("/runs/"):-len("/explain")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        import clozn.receipts.explain as explain
        h._json(200, explain.explain(run))
        return True
    if p.startswith("/runs/") and p.endswith("/receipts"):   # M2: prove-all -- leave-one-out + redundancy guard
        rid = p[len("/runs/"):-len("/receipts")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        mode = str(body.get("mode") or "regen")
        if mode not in ("regen", "forced", "both"):
            h._json(400, {"error": "mode must be one of regen|forced|both"})
            return True
        # regen/both regenerate both arms -> needs the qwen substrate; forced-only never
        # generates (S3: teacher-forced /score on the engine substrate) -- no chat needed.
        if mode in ("regen", "both") and not (ctx.SUB and getattr(ctx.SUB, "chat", None)):
            h._json(503, {"error": "receipts need the qwen substrate"})
            return True
        from clozn import receipts
        try:
            h._json(200, receipts.prove_all(run, ctx.SUB, mode=mode))
        except Exception as e:
            h._json(500, {"error": f"receipts failed: {type(e).__name__}: {e}"})
        return True
    if p.startswith("/runs/") and p.endswith("/receipt"):   # M2: one rigorous both-arms-greedy causal receipt
        rid = p[len("/runs/"):-len("/receipt")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        mode = str(body.get("mode") or "regen")
        if mode not in ("regen", "forced", "both"):
            h._json(400, {"error": "mode must be one of regen|forced|both"})
            return True
        if mode in ("regen", "both") and not (ctx.SUB and getattr(ctx.SUB, "chat", None)):
            h._json(503, {"error": "receipt needs the qwen substrate"})
            return True
        influence = body.get("influence")
        if not isinstance(influence, dict) or not influence:
            h._json(400, {"error": "need an influence spec: one of "
                         "{card_id|dial|memory_off|behavior_off}"})
            return True
        from clozn import receipts
        try:
            out = receipts.receipt(run, influence, ctx.SUB, mode=mode)
        except Exception as e:
            h._json(500, {"error": f"receipt failed: {type(e).__name__}: {e}"})
            return True
        if out is None:
            h._json(500, {"error": "receipt failed (bad influence spec, or the replay "
                         "could not be generated)"})
            return True
        h._json(200, out)
        return True
    if p.startswith("/runs/") and p.endswith("/rederive"):   # S3: deterministic teacher-forced re-derivation
        rid = p[len("/runs/"):-len("/rederive")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        if not (ctx.SUB and getattr(ctx.SUB, "score_tokens", None)):
            h._json(503, {"error": "rederive needs the engine substrate (score_tokens)"})
            return True
        import clozn.receipts.rederive as rederive
        try:
            out = rederive.rederive(run, ctx.SUB)
        except Exception as e:
            h._json(500, {"error": f"rederive failed: {type(e).__name__}: {e}"})
            return True
        if out is None:
            h._json(500, {"error": "rederive failed (no continuation to score, or the "
                         "engine score call failed)"})
            return True
        h._json(200, out)
        return True
    if p == "/jlens":   # J3: general J-lens passthrough (brain/lab page) -- per-token "disposed to say"
        text = str(body.get("text", ""))
        if not text:
            h._json(400, {"error": "need a 'text' to read"})
            return True
        layer = body.get("layer")
        topk = int(body.get("topk", 5) or 5)
        want_protocol = bool(body.get("protocol", False))
        if not (ctx.SUB and getattr(ctx.SUB, "jlens", None)):   # non-engine substrate -> clean 200 absence
            h._json(200, {"available": False, "run_id": None,
                         "reason": "the active substrate has no J-lens (needs the engine substrate)"})
            return True
        res = ctx.SUB.jlens(text, layer=layer, topk=topk)
        h._json(200, ctx._jlens_envelope(res, run_id=None, text_source="input", want_protocol=want_protocol))
        return True
    if p.startswith("/runs/") and p.endswith("/jlens"):   # J3: the Run Inspector J-lens feed for a run
        rid = p[len("/runs/"):-len("/jlens")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        if not (ctx.SUB and getattr(ctx.SUB, "jlens", None)):
            h._json(200, {"available": False, "run_id": rid,
                         "reason": "the active substrate has no J-lens (needs the engine substrate)"})
            return True
        text, text_source = ctx._jlens_run_text(run)          # prefer the stored response; note the source
        if not text:
            h._json(200, {"available": False, "run_id": rid,
                         "reason": "this run has no response/text to read"})
            return True
        layer = body.get("layer")
        topk = int(body.get("topk", 5) or 5)
        want_protocol = bool(body.get("protocol", False))
        res = ctx.SUB.jlens(text, layer=layer, topk=topk)
        h._json(200, ctx._jlens_envelope(res, run_id=rid, text_source=text_source, want_protocol=want_protocol))
        return True
    if p.startswith("/runs/") and p.endswith("/narrate"):   # M4: accountable-self narration + confabulation-diff
        rid = p[len("/runs/"):-len("/narrate")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        if not (ctx.SUB and getattr(ctx.SUB, "chat", None)):   # constrained + unconstrained BOTH generate -> needs qwen
            h._json(503, {"error": "narration needs the qwen substrate"})
            return True
        import clozn.receipts.narrate as narrate
        matcher = narrate.lexical_default            # the weak keyword proxy -- the LABELED fallback judge
        try:
            import clozn.receipts.semantic_matcher as semantic_matcher                  # the real, INDEPENDENT cross-encoder judge, if present
            if semantic_matcher.available():
                matcher = semantic_matcher.nli_support_matcher
        except Exception:
            pass
        try:
            # returns the receipt-constrained narration + confabulation flags; the raw unconstrained
            # "why" is NEVER a field in the result (narrate.py's structural trap guard). narrate()'s
            # own `note` states which matcher ran, so the response is self-describing about its honesty.
            out = narrate.narrate(run, ctx.SUB, support_matcher=matcher)
        except Exception as e:
            h._json(500, {"error": f"narrate failed: {type(e).__name__}: {e}"})
            return True
        h._json(200, out)
        return True
    return False
