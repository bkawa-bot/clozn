"""Generation transport at the public/private boundary.

The private C++ worker emits Clozn state events alongside its completion stream.  The
public gateway exposes those frames only on ``/api/clozn/generate``.  Standard OpenAI
``/v1/completions`` responses are normalized here so ordinary clients never receive
engine-internal event types.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from clozn.server import app as ctx
from clozn.server.http_policy import send_cors_headers


def model_id() -> str:
    try:
        model = str((ctx.ENGINE.health() or {}).get("model") or "clozn-local")
        name = os.path.basename(model).removesuffix(".gguf")
        return name or "clozn-local"
    except Exception:
        return "clozn-local"


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
            "usage": upstream.get("usage") or {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
            },
        }
        handler._json(200, normalized)
    except Exception as exc:
        _error(handler, exc)
    finally:
        response.close()
