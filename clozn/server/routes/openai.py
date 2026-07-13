"""OpenAI-compatible surface: GET /v1/models (so OAI clients can connect) and POST
/v1/chat/completions -- chat with the memory prefix/prompt-block + tone steering applied, streaming via
sse.py when requested, plus the M5 run_id bridge and the opt-in FRONTIER confidence-spans field.
Mechanical extraction of the matching branches out of clozn.server.app's do_GET/do_POST; behavior
unchanged.
"""
import time

from clozn.server import app as ctx


def try_get(h, p):
    if p == "/v1/models":            # OpenAI-compatible model list (so OAI clients connect)
        h._json(200, {"object": "list", "data": [
            {"id": "clozn-qwen", "object": "model", "owned_by": "clozn"}]})
        return True
    return False


def try_post(h, p, body):
    if p != "/v1/chat/completions":   # OpenAI-compatible: chat with memory prefix + tone steering applied
        return False
    if not (ctx.SUB and getattr(ctx.SUB, "chat", None)):
        h._json(503, {"error": "chat needs the qwen substrate"})
        return True
    msgs, mx = body.get("messages", []), int(body.get("max_tokens", 256))
    if body.get("stream") and getattr(ctx.SUB, "chat_stream", None):
        from clozn.server import sse
        # F1 live lens: clozn_lens {layer?, topk?, every?} (or true) is a clozn extension -- absent for
        # standard OpenAI clients, so their streams stay byte-identical (same opt-in rule as clozn_trust).
        sse.sse_chat(h, msgs, mx, str(body.get("model", "clozn-qwen")), lens=body.get("clozn_lens"))
        return True
    t0 = time.time()
    trace_steps = []                            # HF non-stream: capture a per-token trace (B3)
    memout = {}                                 # prompt mode: what memory actually rode this turn
    chat_kw = {"trace_out": trace_steps, "mem_out": memout}
    if isinstance(ctx.SUB, ctx.EngineSubstrate):
        chat_kw["apply_anchored"] = True
    reply = ctx.SUB.chat(msgs, mx, float(body.get("temperature", 0.7)) > 0, **chat_kw)
    fr = ctx.SUB.last_finish_reason() if hasattr(ctx.SUB, "last_finish_reason") else None
    openai_fr = ctx._openai_finish_reason(fr)
    # runlog.record normalizes the raw step list -> {tokens, confidence, alternatives}.
    rid = h._log_run("openai_api", msgs, reply, body.get("model", "clozn-qwen"), t0,
                     trace=trace_steps, mem_out=memout, finish_reason=fr,
                     finish_reason_fallback=None if fr else openai_fr)
    resp = {"id": "chatcmpl-clozn", "object": "chat.completion",
           "created": int(time.time()), "model": body.get("model", "clozn-qwen"),
           "choices": [{"index": 0, "finish_reason": openai_fr,
                        "message": {"role": "assistant", "content": reply}}],
           "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    # M5 any-client run_id bridge (EXPLAIN_THIS_ANSWER_SPEC.md): surface the id two ways so a
    # companion `clozn explain <run_id>` can inspect THIS reply from any OpenAI-compatible client
    # -- an additive top-level field (spec-compliant clients ignore unknown fields) and a response
    # header (for clients that only expose headers). Clean omission when logging failed (rid is
    # None) -- never emit a literal "null"/"None".
    extra_headers = {"X-Clozn-Run-Id": rid} if rid else None
    if rid:
        resp["clozn_run_id"] = rid
    # FRONTIER §1.1 "trust as an API field": when the caller OPTS IN (clozn_trust:true -- default
    # OFF, so a standard OpenAI response stays byte-unchanged / fully compatible), attach
    # claim-level confidence spans over the reply. Built by the SAME producer as
    # GET /runs/<id>/spans (confidence_spans.spans over the normalized token trace), from THIS
    # turn's trace -- so an agent can branch on per-claim confidence inline, without a second call.
    # HONESTY (FRONTIER §6 ledger): these are RAW, UNCALIBRATED model probabilities. clozn_spans_note
    # says so verbatim -- self-confidence != correctness; NO calibration is done here, and nothing
    # implies confidence == correctness.
    if body.get("clozn_trust"):
        try:
            from clozn.runs import confidence_spans
            import clozn.runs.store as _runlog
            _run_for_spans = {"trace": _runlog.steps_to_trace(trace_steps)}
            resp["clozn_spans"] = confidence_spans.spans(_run_for_spans)
            resp["clozn_spans_note"] = ("uncalibrated raw token confidence -- "
                                        "self-confidence != correctness")
        except Exception:
            pass                          # trust is additive: a spans hiccup never breaks the reply
    h._json(200, resp, extra_headers=extra_headers)
    return True
