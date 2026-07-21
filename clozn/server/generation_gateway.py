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
import secrets
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from clozn.server import app as ctx
from clozn.server.http_policy import send_cors_headers
from clozn.runs.context_receipt import warnings_for


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
    warnings: list[dict]
    reasoning: dict
    structured: Any = None


def request_memory_scope(handler):
    """Resolve exact request-local memory scope; User-Agent never activates app memory."""
    from clozn.memory.scope import MemoryScope
    from clozn.runs.association import request_explicit_client, request_project
    headers = getattr(handler, "headers", None)
    return MemoryScope(app_key=request_explicit_client(headers), project_key=request_project(headers))


def apply_corrective_policy(handler, messages: list) -> tuple[list, dict | None]:
    """Apply active profile/session response policies to a copied request.

    Callers retain the original messages for journaling, so the context receipt
    never claims the client supplied Clozn's instruction. The returned evidence is
    safe to place in run metadata.
    """
    from clozn.behavior import corrective_retries
    from clozn.runs.association import request_session
    session_key = request_session(getattr(handler, "headers", None))
    profile_name = ctx._active_profile_name()
    presets = corrective_retries.effective_presets(
        session_key=session_key, profile_name=profile_name,
    )
    return (
        corrective_retries.inject(messages, presets),
        corrective_retries.evidence(
            presets, session_key=session_key, profile_name=profile_name,
        ),
    )


def instrumented_chat(handler, messages: list, *, model: str, max_tokens: int = 256,
                      sample=True, source: str, extra_meta: dict | None = None,
                      journal_messages: list | None = None,
                      output_processor: Callable[[str, Any, str | None], Any] | None = None,
                      native_structured: Mapping[str, Any] | None = None,
                      ) -> InstrumentedChatResult:
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
    if native_structured is not None and not callable(getattr(sub, "_complete_chat_native", None)):
        raise RuntimeError("active model worker does not expose atomic native structured chat")

    started = time.time()
    trace_steps = []
    memout = {}
    logged_messages = journal_messages if journal_messages is not None else messages
    chat_kw = {"trace_out": trace_steps, "mem_out": memout}
    if isinstance(sub, ctx.EngineSubstrate):
        # Live compatibility traffic gets the same anchored-memory behavior as the
        # OpenAI route. Receipt/replay callers remain explicitly deterministic.
        chat_kw["apply_anchored"] = True
        chat_kw["memory_scope"] = request_memory_scope(handler)
    native_result = None
    try:
        if native_structured is not None:
            contract = dict(native_structured)
            native_result = sub._complete_chat_native(
                messages,
                tools=contract.get("tools") or None,
                tool_choice=contract.get("tool_choice", "auto"),
                json_schema=contract.get("json_schema"),
                parallel_tool_calls=False,
                max_new=int(max_tokens), sample=sample,
                trace_out=trace_steps, mem_out=memout,
                # Structured output cannot safely run the ordinary anchored retry/loop policy.
                apply_anchored=False,
                enable_thinking=True,
                reasoning_format="none",
                memory_scope=request_memory_scope(handler),
            )
            reply = native_result["raw_model_output"]
        else:
            reply = sub.chat(messages, int(max_tokens), sample, **chat_kw)
    except Exception as exc:
        handler._log_run(source, logged_messages, "", model, started, error=str(exc),
                         mem_out=memout, extra_meta=extra_meta)
        raise

    raw_reply = str(reply)
    from clozn.runs.think_tags import prompt_opens_think, sanitize_reply, sanitize_steps
    implicit_think = prompt_opens_think(memout.get("final_prompt"))
    think = sanitize_reply(raw_reply, implicit_open=implicit_think)
    public_steps = trace_steps
    reasoning_steps = []
    if think.stripped:
        public_steps, reasoning_steps, _ = sanitize_steps(trace_steps, implicit_open=implicit_think)
    finish = sub.last_finish_reason() if hasattr(sub, "last_finish_reason") else None
    public_finish = ctx._openai_finish_reason(finish)
    structured = None
    evidence = None
    if output_processor is not None:
        try:
            processor_value = native_result if native_result is not None else think.public_text
            structured = output_processor(raw_reply, processor_value, finish)
            evidence = (structured.get("evidence") if isinstance(structured, Mapping)
                        else getattr(structured, "evidence", None))
        except Exception as exc:
            failure_meta = dict(extra_meta or {})
            evidence = getattr(exc, "evidence", None)
            rid = handler._log_run(
                source, logged_messages, raw_reply, model, started,
                error=f"{getattr(exc, 'code', 'structured_output_error')}: {exc}",
                trace=trace_steps, mem_out=memout, finish_reason=finish,
                finish_reason_fallback=None if finish else public_finish,
                extra_meta=failure_meta,
                output_contract=evidence if isinstance(evidence, dict) else None,
            )
            try:
                exc.run_id = rid
            except Exception:
                pass
            if rid is None:
                from clozn.server.structured_io import StructuredIOError
                persistence = StructuredIOError(
                    "structured output failed and its evidence could not be durably journaled",
                    code="journal_persistence_failed", param=None,
                    evidence={"cause_code": getattr(exc, "code", type(exc).__name__),
                              "output_contract": evidence if isinstance(evidence, dict) else {}},
                )
                persistence.run_id = None
                raise persistence from exc
            raise
    success_meta = dict(extra_meta or {})
    rid = handler._log_run(source, logged_messages, raw_reply, model, started,
                           trace=trace_steps, mem_out=memout, finish_reason=finish,
                           finish_reason_fallback=None if finish else public_finish,
                           extra_meta=success_meta,
                           output_contract=evidence if isinstance(evidence, dict) else None)
    if output_processor is not None and rid is None:
        from clozn.server.structured_io import StructuredIOError
        persistence = StructuredIOError(
            "structured output was validated but its evidence could not be durably journaled",
            code="journal_persistence_failed", param=None,
            evidence={"output_contract": evidence if isinstance(evidence, dict) else {}},
        )
        persistence.run_id = None
        raise persistence
    return InstrumentedChatResult(
        reply=think.public_text, trace_steps=public_steps, memory=memout,
        finish_reason=finish, public_finish_reason=public_finish, run_id=rid,
        warnings=warnings_for(finish, {"max_tokens": int(max_tokens)}),
        reasoning=think.journal(reasoning_steps=reasoning_steps),
        structured=structured,
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


def policy_signal(trace_steps, model: str | None, task: str | None = None) -> dict | None:
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
        load_profile = getattr(eval_store, "load_profile", None)
        if callable(load_profile):
            # The indexed store owns selection semantics: explicit task is exact;
            # omitted task is the newest profile for the exact model.  A miss is
            # final -- never borrow another task's thresholds.
            saved = load_profile(model, task)
        else:
            saved = eval_store.load()
            if task is not None:
                legacy_task = (saved.get("task") or saved.get("set")) if isinstance(saved, dict) else None
                if legacy_task != task:
                    return None
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
            "calibration_task": (saved.get("task") or saved.get("set") or task),
            "calibration_model": saved.get("model"),
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


def _completion_messages(prompt: str) -> list[dict[str, str]]:
    """Represent a legacy text prompt at the shared, message-based substrate seam.

    The public route remains OpenAI's legacy ``prompt -> text`` API.  Inside Clozn,
    however, using the same user-turn representation as every other compatibility
    entrance is what makes memory assembly, steering, rendered-prompt capture, token
    traces, and journaling impossible to bypass.
    """
    return [{"role": "user", "content": prompt}]


def _completion_sample(body: dict):
    """Translate the validated legacy sampling vocabulary to Substrate.chat's contract."""
    sample = {key: body[key] for key in ("temperature", "top_p", "top_k", "seed")
              if key in body}
    if "rep_penalty" in body:
        sample["repeat_penalty"] = body["rep_penalty"]
    # No explicit fields means Clozn's configured interactive sampling, just like
    # /v1/chat/completions.  A dict containing temperature=0 remains explicit and
    # resolves to greedy in EngineSubstrate.
    return sample or True


def _stream_completion(handler, messages: list, *, model: str, max_tokens: int,
                       sample, journal_messages=None, corrective_evidence=None) -> None:
    """Instrumented OpenAI legacy-completion SSE stream.

    Generation crosses ``Substrate.chat_stream`` and the completed/failed/abandoned
    turn is journaled exactly once.  The serializer intentionally emits only standard
    ``text_completion`` chunks: native worker frames and Clozn trace frames never leak
    onto a compatibility endpoint.

    A streaming run id cannot be sent as an HTTP header because the header is committed
    before generation, while the journal creates the id after generation.  Nor is a
    proprietary trailing chunk injected into this strict legacy wire shape.  The run is
    still persisted and can be associated through the server-side latest-run/session
    side channel when that Phase-2 facility lands.
    """
    sub = ctx.active_sub(handler)
    started = time.time()
    memout: dict = {}
    acc: list[str] = []
    stream_id = "cmpl-" + secrets.token_hex(8)
    created = int(time.time())
    logged_messages = journal_messages if journal_messages is not None else messages
    extra_meta = {"compatibility_api": "openai", "openai_operation": "completion",
                  "corrective_policy": corrective_evidence}

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    send_cors_headers(handler)
    handler.end_headers()

    def write_chunk(text: str, finish_reason=None, *, run_id=None, warnings=None) -> None:
        chunk = {
            "id": stream_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"text": text, "index": 0, "logprobs": None,
                         "finish_reason": finish_reason}],
        }
        if run_id:
            chunk["clozn_run_id"] = run_id
        if warnings:
            chunk["clozn_warnings"] = list(warnings)
        handler.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode("utf-8"))
        handler.wfile.flush()

    gen = None
    disconnect_error = None
    think_stream = None
    try:
        if getattr(sub, "chat_stream", None):
            import inspect
            try:
                params = inspect.signature(sub.chat_stream).parameters
            except Exception:
                params = {}
            stream_kw = {"mem_out": memout}
            if "sample" in params:
                stream_kw["sample"] = sample
            if "memory_scope" in params:
                stream_kw["memory_scope"] = request_memory_scope(handler)
            gen = sub.chat_stream(messages, max_tokens, **stream_kw)
            for piece in gen:
                raw_text = str(piece)
                acc.append(raw_text)
                if think_stream is None:
                    from clozn.runs.think_tags import ThinkTagStream, prompt_opens_think
                    think_stream = ThinkTagStream(
                        implicit_open=prompt_opens_think(memout.get("final_prompt"))
                    )
                text = think_stream.feed(raw_text)
                if not text:
                    continue
                try:
                    write_chunk(text)
                except OSError as exc:
                    disconnect_error = exc
                    break
        else:
            # A custom/lab substrate may implement only chat().  Keep the request on
            # the instrumented seam and emit its completed text as one standard chunk.
            try:
                generated = instrumented_chat(
                    handler, messages, model=model, max_tokens=max_tokens, sample=sample,
                    source="openai_api",
                    extra_meta={**extra_meta, "requested_stream": True},
                    journal_messages=logged_messages,
                )
            except Exception as exc:
                # instrumented_chat already journaled the failure exactly once.
                error = {"error": {"message": str(exc), "type": "upstream_error"}}
                try:
                    handler.wfile.write(("data: " + json.dumps(error) + "\n\n").encode("utf-8"))
                    handler.wfile.write(b"data: [DONE]\n\n")
                    handler.wfile.flush()
                except OSError:
                    pass
                return
            acc.append(generated.reply)
            try:
                write_chunk(generated.reply)
                write_chunk("", generated.public_finish_reason, run_id=generated.run_id,
                            warnings=generated.warnings)
                handler.wfile.write(b"data: [DONE]\n\n")
                handler.wfile.flush()
            except OSError:
                pass
            return

        if disconnect_error is not None:
            req_ctx = getattr(sub, "_request", None)
            if req_ctx is not None and hasattr(req_ctx, "cancel"):
                req_ctx.cancel()
            handler._log_run(
                "openai_api", logged_messages, "".join(acc), model, started,
                error=f"client disconnected mid-stream: {disconnect_error}", mem_out=memout,
                extra_meta={**extra_meta, "stream_failure": "client_disconnected"},
            )
            return

        if think_stream is not None:
            tail, _think = think_stream.finish()
            if tail:
                try:
                    write_chunk(tail)
                except OSError as exc:
                    disconnect_error = exc
        if disconnect_error is not None:
            req_ctx = getattr(sub, "_request", None)
            if req_ctx is not None and hasattr(req_ctx, "cancel"):
                req_ctx.cancel()
            handler._log_run(
                "openai_api", logged_messages, "".join(acc), model, started,
                error=f"client disconnected mid-stream: {disconnect_error}", mem_out=memout,
                extra_meta={**extra_meta, "stream_failure": "client_disconnected"},
            )
            return

        finish = sub.last_finish_reason() if hasattr(sub, "last_finish_reason") else None
        public_finish = ctx._openai_finish_reason(finish)
        trace = sub.last_stream_trace() if hasattr(sub, "last_stream_trace") else None
        run_id = handler._log_run(
            "openai_api", logged_messages, "".join(acc), model, started, trace=trace,
            mem_out=memout, finish_reason=finish,
            finish_reason_fallback=None if finish else public_finish,
            extra_meta=extra_meta,
        )
        try:
            write_chunk("", public_finish, run_id=run_id,
                        warnings=warnings_for(finish, {"max_tokens": int(max_tokens)}))
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()
        except OSError:
            # The model completed and the durable run above is already accurate; a
            # disconnect while delivering the terminal marker must not create a
            # second contradictory failure run.
            return
    except Exception as exc:
        handler._log_run(
            "openai_api", logged_messages, "".join(acc), model, started, error=str(exc),
            mem_out=memout,
            extra_meta={**extra_meta, "stream_failure": "worker_disconnected"},
        )
        try:
            error = {"error": {"message": str(exc), "type": "upstream_error"}}
            handler.wfile.write(("data: " + json.dumps(error) + "\n\n").encode("utf-8"))
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()
        except OSError:
            pass
    finally:
        if gen is not None and hasattr(gen, "close"):
            try:
                gen.close()
            except Exception:
                pass


def openai_completion(handler, body: dict) -> None:
    """Strict OpenAI text-completion view over Clozn's instrumented substrate."""
    from clozn.server.openai_compat import CompatibilityError, normalize_completion_request
    try:
        body = normalize_completion_request(body)
    except CompatibilityError as exc:
        handler._json(400, {"error": {"message": str(exc), "type": "invalid_request_error",
                                      "param": exc.param, "code": exc.code}})
        return
    model = str(body.get("model") or model_id())
    journal_messages = _completion_messages(body["prompt"])
    messages, corrective_evidence = apply_corrective_policy(handler, journal_messages)
    max_tokens = int(body.get("max_tokens", 256))
    sample = _completion_sample(body)
    sub = ctx.active_sub(handler)
    if not (sub and getattr(sub, "chat", None)):
        handler._json(503, {"error": {"message": "model worker unavailable",
                                      "type": "service_unavailable"}})
        return
    if body.get("stream"):
        _stream_completion(handler, messages, model=model, max_tokens=max_tokens, sample=sample,
                           journal_messages=journal_messages,
                           corrective_evidence=corrective_evidence)
        return
    try:
        generated = instrumented_chat(
            handler, messages, model=model, max_tokens=max_tokens, sample=sample,
            source="openai_api",
            extra_meta={"compatibility_api": "openai", "openai_operation": "completion",
                        "corrective_policy": corrective_evidence},
            journal_messages=journal_messages,
        )
    except Exception as exc:
        _error(handler, exc)
        return
    response = {
        "id": "cmpl-" + secrets.token_hex(8),
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "text": generated.reply,
            "index": 0,
            "logprobs": None,
            "finish_reason": generated.public_finish_reason,
        }],
    }
    if generated.run_id:
        response["clozn_run_id"] = generated.run_id
    if generated.warnings:
        response["clozn_warnings"] = generated.warnings
    headers = {"X-Clozn-Run-Id": generated.run_id} if generated.run_id else {}
    if generated.warnings:
        headers["X-Clozn-Warning"] = "output-truncated"
    handler._json(200, response, extra_headers=headers)
