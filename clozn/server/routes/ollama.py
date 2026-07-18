"""Thin Ollama API compatibility shim -- NOT a reimplementation of Ollama.

Covers just enough of Ollama's HTTP surface (`GET /api/tags`, `GET /api/version`,
`POST /api/generate`, `POST /api/chat`) that a client built against the Ollama wire
protocol (e.g. the `ollama` Python/JS client libraries, or tools that hardcode
`http://localhost:11434`) can point at clozn's engine instead and get back
Ollama-shaped JSON. There is no model pull/push, no multi-model registry, no
embeddings endpoint, and (for now) no streaming -- `stream: true` on `/api/generate`
or `/api/chat` gets a clean 501 rather than a silently-wrong response.

This module is a DROP-IN convenience layered on the same engine the OpenAI-compatible
routes use (routes/openai.py); it is intentionally NOT derived from that code, since
Ollama's and OpenAI's request/response shapes only coincidentally overlap. `GET
/api/version` always answers "0.0.0-clozn" -- a fixed, honest string so a client (or a
human) can tell at a glance this is clozn wearing an Ollama-shaped socket, not real
Ollama.

`clozn.server.app` is imported inside each function body (not at module level) to
avoid circular imports at module load time.
"""
import os
import time


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _engine_params(options: dict) -> dict:
    """Map Ollama's `options` object to the engine's raw /v1/completions field names
    (EngineClient.complete passes **params straight through, unlike the OpenAI routes'
    validated/renamed field set -- see server_shared.hpp's body.value(...) reads)."""
    out = {}
    if options.get("num_predict") is not None:
        try:
            n = int(options["num_predict"])
        except (TypeError, ValueError):
            n = None
        if n is not None and n > 0:
            out["max_tokens"] = n
    for src, dst in (("temperature", "temperature"), ("top_p", "top_p"), ("top_k", "top_k"),
                     ("repeat_penalty", "rep_penalty"), ("seed", "seed")):
        value = options.get(src)
        if value is not None:
            out[dst] = value
    return out


def _choice_text(upstream: dict) -> str:
    choices = upstream.get("choices") or [{}]
    first = choices[0] if choices else {}
    return str((first or {}).get("text") or "")


def try_get(h, p):
    if p == "/api/tags":
        import clozn.server.app as ctx
        models = []
        if ctx.ENGINE is not None:
            try:
                info = ctx.ENGINE.health() or {}
                model_path = str(info.get("model") or "")
                name = os.path.basename(model_path).removesuffix(".gguf") or "clozn-local"
                models.append({"name": name, "model": name, "size": 0, "digest": "",
                               "modified_at": _iso_now()})
            except Exception:
                pass    # no reachable engine -- an empty list is the honest answer, not an error
        h._json(200, {"models": models})
        return True
    if p == "/api/version":
        h._json(200, {"version": "0.0.0-clozn"})   # deliberately not a real Ollama version string
        return True
    return False


def try_post(h, p, body):
    if p == "/api/generate":
        import clozn.server.app as ctx
        if ctx.ENGINE is None:
            h._json(502, {"error": "no engine configured"})
            return True
        if body.get("stream"):
            h._json(501, {"error": "streaming not yet supported for /api/generate"})
            return True
        from clozn.server.generation_gateway import model_id
        model = str(body.get("model") or model_id())
        prompt = str(body.get("prompt", ""))
        params = _engine_params(body.get("options") or {})
        try:
            upstream = ctx.ENGINE.complete(prompt, **params)
        except Exception as e:
            h._json(502, {"error": f"engine: {e}"})
            return True
        h._json(200, {"model": model, "response": _choice_text(upstream), "done": True})
        return True
    if p == "/api/chat":
        import clozn.server.app as ctx
        if ctx.ENGINE is None:
            h._json(502, {"error": "no engine configured"})
            return True
        if body.get("stream"):
            h._json(501, {"error": "streaming not yet supported for /api/chat"})
            return True
        from clozn.server.generation_gateway import model_id
        model = str(body.get("model") or model_id())
        messages = body.get("messages") or []
        params = _engine_params(body.get("options") or {})
        try:
            prompt = ctx._engine_tmpl(ctx.ENGINE, messages)
            upstream = ctx.ENGINE.complete(prompt, **params)
        except Exception as e:
            h._json(502, {"error": f"engine: {e}"})
            return True
        h._json(200, {"model": model,
                      "message": {"role": "assistant", "content": _choice_text(upstream)},
                      "done": True})
        return True
    return False
