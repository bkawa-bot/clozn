"""Thin Ollama API compatibility shim -- NOT a reimplementation of Ollama.

Covers just enough of Ollama's HTTP surface (`GET /api/tags`, `GET /api/version`,
`POST /api/generate`, `POST /api/chat`) that a client built against the Ollama wire
protocol (e.g. the `ollama` Python/JS client libraries, or tools that hardcode
`http://localhost:11434`) can point at clozn's engine instead and get back
Ollama-shaped JSON. There is no model pull/push, no multi-model registry, no
embeddings endpoint, and (for now) no streaming -- `stream: true` on `/api/generate`
or `/api/chat` gets a clean 501 rather than a silently-wrong response.

This module is a DROP-IN convenience layered on the same instrumented substrate the
OpenAI-compatible chat route uses. The wire serializers remain independent because
Ollama's and OpenAI's request/response shapes only coincidentally overlap, but both
cross generation_gateway.instrumented_chat before a response is shaped. `GET
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


def _generation_options(options: dict) -> tuple[int, dict | bool]:
    """Map Ollama decode options to EngineSubstrate.chat's sampling contract."""
    max_tokens = 256
    if options.get("num_predict") is not None:
        try:
            n = int(options["num_predict"])
        except (TypeError, ValueError):
            n = None
        if n is not None and n > 0:
            max_tokens = n
    sample = {}
    for key in ("temperature", "top_p", "top_k", "repeat_penalty", "seed"):
        value = options.get(key)
        if value is not None:
            sample[key] = value
    # No explicit override means "use Clozn's configured interactive sampling",
    # which is the same contract the OpenAI chat route uses.
    return max_tokens, sample or True


def _receipt_fields(result) -> tuple[dict, dict | None]:
    if not result.run_id:
        return {}, None
    return {"clozn_run_id": result.run_id}, {"X-Clozn-Run-Id": result.run_id}


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
        if not (ctx.active_sub(h) and getattr(ctx.active_sub(h), "chat", None)):
            h._json(502, {"error": "no engine configured"})
            return True
        if body.get("stream"):
            h._json(501, {"error": "streaming not yet supported for /api/generate"})
            return True
        from clozn.server.generation_gateway import model_id
        from clozn.server.generation_gateway import instrumented_chat
        model = str(body.get("model") or model_id())
        prompt = str(body.get("prompt", ""))
        max_tokens, sample = _generation_options(body.get("options") or {})
        messages = []
        if body.get("system") is not None:
            messages.append({"role": "system", "content": str(body.get("system") or "")})
        messages.append({"role": "user", "content": prompt})
        try:
            generated = instrumented_chat(
                h, messages, model=model, max_tokens=max_tokens, sample=sample,
                source="ollama_api",
                extra_meta={"compatibility_api": "ollama", "ollama_operation": "generate"},
            )
        except Exception as e:
            h._json(502, {"error": f"engine: {e}"})
            return True
        receipt, headers = _receipt_fields(generated)
        h._json(200, {"model": model, "response": generated.reply, "done": True,
                      **receipt}, extra_headers=headers)
        return True
    if p == "/api/chat":
        import clozn.server.app as ctx
        if not (ctx.active_sub(h) and getattr(ctx.active_sub(h), "chat", None)):
            h._json(502, {"error": "no engine configured"})
            return True
        if body.get("stream"):
            h._json(501, {"error": "streaming not yet supported for /api/chat"})
            return True
        from clozn.server.generation_gateway import model_id
        from clozn.server.generation_gateway import instrumented_chat
        model = str(body.get("model") or model_id())
        messages = body.get("messages") or []
        from clozn.runs.receipt_footer import strip_footers
        messages = strip_footers(messages)
        max_tokens, sample = _generation_options(body.get("options") or {})
        try:
            generated = instrumented_chat(
                h, messages, model=model, max_tokens=max_tokens, sample=sample,
                source="ollama_api",
                extra_meta={"compatibility_api": "ollama", "ollama_operation": "chat"},
            )
        except Exception as e:
            h._json(502, {"error": f"engine: {e}"})
            return True
        receipt, headers = _receipt_fields(generated)
        h._json(200, {"model": model,
                      "message": {"role": "assistant", "content": generated.reply},
                      "done": True, **receipt}, extra_headers=headers)
        return True
    return False
