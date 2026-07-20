"""Generation transport at the public/private boundary.

The private C++ worker emits Clozn state events alongside its completion stream.  The
public gateway exposes those frames only on ``/api/clozn/generate``.  Standard OpenAI
``/v1/completions`` responses are normalized here so ordinary clients never receive
engine-internal event types.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import time
import urllib.error
import urllib.request

from clozn.server import app as ctx
from clozn.server.http_policy import send_cors_headers


@dataclass
class InstrumentedChatResult:
    """One compatibility request after it has crossed Clozn's instrumented substrate.

    Compatibility routes own their wire formats, but they must not each grow a second
    inference path.  This result is the shared seam: memory assembly, steering, trace
    capture, finish-reason capture, and run journaling have already happened by the
    time a route turns it into OpenAI- or Ollama-shaped JSON.
    """

    reply: str
    trace_steps: list
    memory: dict
    finish_reason: str | None
    public_finish_reason: str
    run_id: str | None


def instrumented_chat(handler, messages: list, *, model: str, max_tokens: int = 256,
                      sample=True, source: str, extra_meta: dict | None = None) -> InstrumentedChatResult:
    """Run chat through the active substrate and persist the resulting evidence.

    This is deliberately below every compatibility serializer.  The active substrate
    is where Clozn applies prompt-card memory, tone steering, anchored memory, and the
    traced engine call; ``handler._log_run`` is where that evidence becomes a receipt.
    A route calling ``ENGINE.complete`` directly skips all of those layers.

    Generation errors are journaled before being re-raised so callers can preserve
    their protocol-specific error envelope without losing the failed experiment.
    """
    sub = ctx.active_sub(handler)
    if not (sub and getattr(sub, "chat", None)):
        raise RuntimeError("model worker unavailable")

    started = time.time()
    trace_steps = []
    memout = {}
    chat_kw = {"trace_out": trace_steps, "mem_out": memout}
    if isinstance(sub, ctx.EngineSubstrate):
        # Live compatibility traffic gets the same anchored-memory behavior as the
        # OpenAI route. Receipt/replay callers remain explicitly deterministic.
        chat_kw["apply_anchored"] = True
    try:
        reply = sub.chat(messages, int(max_tokens), sample, **chat_kw)
    except Exception as exc:
        handler._log_run(source, messages, "", model, started, error=str(exc),
                         mem_out=memout, extra_meta=extra_meta)
        raise

    finish = sub.last_finish_reason() if hasattr(sub, "last_finish_reason") else None
    public_finish = ctx._openai_finish_reason(finish)
    rid = handler._log_run(source, messages, reply, model, started,
                           trace=trace_steps, mem_out=memout, finish_reason=finish,
                           finish_reason_fallback=None if finish else public_finish,
                           extra_meta=extra_meta)
    return InstrumentedChatResult(
        reply=str(reply), trace_steps=trace_steps, memory=memout,
        finish_reason=finish, public_finish_reason=public_finish, run_id=rid,
    )


def model_id() -> str:
    try:
        model = str((ctx.ENGINE.health() or {}).get("model") or "clozn-local")
        name = os.path.basename(model).removesuffix(".gguf")
        return name or "clozn-local"
    except Exception:
        return "clozn-local"


_POLICY_NOTES = {
    "ask": ("confidence on this reply falls in the calibrated 'ask' band -- consider a "
            "clarifying follow-up rather than treating it as a confident answer (from "
            "clozn eval's selective-generation policy, not a live fact-check)"),
    "abstain": ("confidence on this reply falls in the calibrated 'abstain' band -- this answer is "
                "likely wrong; treat it with significant skepticism (from clozn eval's "
                "selective-generation policy, not a live fact-check)"),
}


def policy_signal(trace_steps, model: str | None) -> dict | None:
    """The selective-generation policy's verdict for one just-completed /v1/chat/completions reply, or
    None when there is nothing honest to say -- no calibration saved yet (`clozn eval --save`), the saved
    calibration doesn't match this model or carries no usable score aggregate, or this reply's confidence
    is in the 'answer' band. Reuses clozn.eval.policy.classify_run, which mirrors
    clozn.runs.calibrated_trust.attach_truth's provenance rules (exact model match, a fitted score
    aggregate) so this can never fabricate a verdict the saved report can't back up.

    Signals BOTH the 'ask' and 'abstain' bands -- the calibration backlog item #10 ("a retrieval/clarify
    action wired to the policy's ask band", plus its abstain follow-on: when confidence is low enough that
    the model is likely wrong, say so explicitly rather than staying silent). It is a metadata field the
    caller attaches to the response (or an SSE side-frame), never a change to the generated text; the
    caller decides what, if anything, to do with it -- the 'ask' note suggests a clarifying follow-up, the
    'abstain' note is a stronger warning that the reply is likely wrong. `trace_steps` is the RAW
    per-token step list (chat()'s trace_out, or chat_stream's last_stream_trace()) -- normalized here via
    clozn.runs.store.steps_to_trace, the same shape a stored run's trace carries. Never raises."""
    try:
        from clozn.eval import policy as eval_policy, store as eval_store
        import clozn.runs.store as runlog
        saved = eval_store.load()
        if not saved:
            return None
        trace = runlog.steps_to_trace(trace_steps)
        if not trace:
            return None
        verdict = eval_policy.classify_run(trace, saved, model=model)
        band = verdict.get("band")
        if not verdict.get("available") or band not in ("ask", "abstain"):
            return None
        return {
            "band": band,
            "score": verdict["score"],
            "score_aggregate": verdict["score_aggregate"],
            "answer_at": verdict["answer_at"],
            "ask_at": verdict["ask_at"],
            "note": _POLICY_NOTES[band],
        }
    except Exception:
        return None


# Backward-compat alias: earlier callers imported this name back when only the 'ask' band was wired.
ask_band_signal = policy_signal


def _request(body: dict):
    if ctx.ENGINE is None:
        raise RuntimeError("model worker unavailable")
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        ctx.ENGINE.base + "/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=getattr(ctx.ENGINE, "timeout", 600))


def _error(handler, exc: Exception) -> None:
    status = int(getattr(exc, "code", 502) or 502)
    detail = str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        try:
            payload = json.loads(exc.read().decode("utf-8", "replace"))
            detail = str(payload.get("error") or detail)
        except Exception:
            pass
    handler._json(status, {"error": {"message": detail, "type": "upstream_error", "code": status}})


def native_completion(handler, body: dict) -> None:
    """Transparent Clozn event stream used by the CLI and Studio instrumentation."""
    try:
        response = _request(body)
    except Exception as exc:
        _error(handler, exc)
        return
    try:
        if body.get("stream"):
            handler.send_response(getattr(response, "status", 200))
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Cache-Control", "no-cache")
            send_cors_headers(handler)
            handler.end_headers()
            try:
                for line in response:
                    handler.wfile.write(line)
                    handler.wfile.flush()
            except Exception as exc:
                frame = {"error": {"message": str(exc), "type": "upstream_error"}}
                try:
                    handler.wfile.write(("data: " + json.dumps(frame) + "\n\n").encode("utf-8"))
                    handler.wfile.write(b"data: [DONE]\n\n")
                    handler.wfile.flush()
                except Exception:
                    pass
            return
        raw = response.read()
        handler._send(getattr(response, "status", 200), raw, "application/json")
    finally:
        response.close()


def _finish_reason(frame: dict) -> str | None:
    reason = frame.get("reason")
    if reason == "eos":
        return "stop"
    if reason:
        return "length"
    return None


def _completion_chunk(text: str, model: str, finish_reason=None) -> dict:
    return {
        "id": "cmpl-clozn",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": finish_reason}],
    }


def openai_completion(handler, body: dict) -> None:
    """Strict OpenAI text-completion view over the worker's richer event protocol."""
    from clozn.server.openai_compat import CompatibilityError, normalize_completion_request
    try:
        body = normalize_completion_request(body)
    except CompatibilityError as exc:
        handler._json(400, {"error": {"message": str(exc), "type": "invalid_request_error",
                                      "param": exc.param, "code": exc.code}})
        return
    model = str(body.get("model") or model_id())
    try:
        response = _request(body)
    except Exception as exc:
        _error(handler, exc)
        return
    try:
        if body.get("stream"):
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Cache-Control", "no-cache")
            send_cors_headers(handler)
            handler.end_headers()
            finish = None
            emitted_text = False
            try:
                for raw in response:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        frame = json.loads(payload)
                    except Exception:
                        continue
                    if frame.get("type") == "tokens_committed":
                        for item in frame.get("items") or []:
                            chunk = _completion_chunk(str(item.get("piece") or ""), model)
                            handler.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode("utf-8"))
                            handler.wfile.flush()
                            emitted_text = True
                    elif frame.get("type") == "gen_finished":
                        finish = _finish_reason(frame)
                    elif frame.get("object") == "text_completion":
                        choice = (frame.get("choices") or [{}])[0]
                        # AR streams already emitted token pieces; diffusion streams generally expose
                        # their assembled board only in this final standard-shaped frame.
                        if not emitted_text and choice.get("text"):
                            chunk = _completion_chunk(str(choice["text"]), model)
                            handler.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode("utf-8"))
                            handler.wfile.flush()
                            emitted_text = True
                        finish = choice.get("finish_reason") or finish
            except Exception as exc:
                error = {"error": {"message": str(exc), "type": "upstream_error"}}
                try:
                    handler.wfile.write(("data: " + json.dumps(error) + "\n\n").encode("utf-8"))
                    handler.wfile.write(b"data: [DONE]\n\n")
                    handler.wfile.flush()
                except Exception:
                    pass
                return
            final = _completion_chunk("", model, finish or "stop")
            handler.wfile.write(("data: " + json.dumps(final) + "\n\n").encode("utf-8"))
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()
            return

        upstream = json.loads(response.read().decode("utf-8"))
        choice = (upstream.get("choices") or [{}])[0]
        normalized = {
            "id": str(upstream.get("id") or "cmpl-clozn"),
            "object": "text_completion",
            "created": int(upstream.get("created") or time.time()),
            "model": model,
            "choices": [{
                "text": str(choice.get("text") or ""),
                "index": 0,
                "logprobs": choice.get("logprobs"),
                "finish_reason": choice.get("finish_reason") or "stop",
            }],
        }
        if upstream.get("usage") is not None:
            normalized["usage"] = upstream["usage"]
        handler._json(200, normalized)
    except Exception as exc:
        _error(handler, exc)
    finally:
        response.close()
