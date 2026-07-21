"""Drive released Open WebUI through its configured OpenAI-compatible provider."""
from __future__ import annotations

import argparse
import json
import urllib.request

if __package__:
    from .fake_gateway import (
        TOOL_ARGUMENTS,
        TOOL_CALL_ID,
        TOOL_FINAL_ANSWER,
        TOOL_NAME,
        TOOL_RESULT,
        TOOL_SPEC,
    )
else:
    from fake_gateway import (  # type: ignore[no-redef]
        TOOL_ARGUMENTS,
        TOOL_CALL_ID,
        TOOL_FINAL_ANSWER,
        TOOL_NAME,
        TOOL_RESULT,
        TOOL_SPEC,
    )


def _json(url: str, body=None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def _stream(url: str, body: dict) -> str:
    request = urllib.request.Request(
        url, data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    pieces = []
    with urllib.request.urlopen(request, timeout=60) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            event = json.loads(payload)
            choices = event.get("choices") or []
            if choices:
                piece = (choices[0].get("delta") or {}).get("content")
                if piece:
                    pieces.append(piece)
    return "".join(pieces)


def _contains_model(value, model: str) -> bool:
    if isinstance(value, dict):
        if value.get("id") == model or value.get("name") == model or value.get("model") == model:
            return True
        return any(_contains_model(item, model) for item in value.values())
    if isinstance(value, list):
        return any(_contains_model(item, model) for item in value)
    return False


def _tool_exchange(base: str) -> dict:
    """Exercise a complete caller-managed tool continuation through Open WebUI.

    The released Open WebUI server is the proxy under test. The external fake
    gateway validates every forwarded field and supplies deterministic OpenAI
    response objects; this proves transport fidelity, not model qualification.
    """
    user_message = {"role": "user", "content": "What is the weather in Paris?"}
    first = _json(base + "/api/chat/completions", {
        "model": "clozn-local",
        "messages": [user_message],
        "stream": False,
        "tools": [TOOL_SPEC],
    })
    choice = first["choices"][0]
    assert choice["finish_reason"] == "tool_calls", first
    assistant = choice["message"]
    assert assistant.get("role") == "assistant", first
    assert assistant.get("content") is None, first
    calls = assistant.get("tool_calls")
    assert isinstance(calls, list) and len(calls) == 1, first
    call = calls[0]
    assert call.get("id") == TOOL_CALL_ID, first
    assert call.get("type") == "function", first
    assert call.get("function", {}).get("name") == TOOL_NAME, first
    assert json.loads(call["function"]["arguments"]) == TOOL_ARGUMENTS, first

    tool_message = {
        "role": "tool",
        "tool_call_id": TOOL_CALL_ID,
        "name": TOOL_NAME,
        "content": json.dumps(TOOL_RESULT, separators=(",", ":")),
    }
    final = _json(base + "/api/chat/completions", {
        "model": "clozn-local",
        "messages": [user_message, assistant, tool_message],
        "stream": False,
        "tools": [TOOL_SPEC],
    })
    final_choice = final["choices"][0]
    assert final_choice["finish_reason"] == "stop", final
    assert final_choice["message"] == {
        "role": "assistant",
        "content": TOOL_FINAL_ANSWER,
    }, final
    return {"tool_call_id": TOOL_CALL_ID, "final": TOOL_FINAL_ANSWER}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    models = _json(base + "/api/models")
    assert _contains_model(models, "clozn-local"), models
    reply = _json(base + "/api/chat/completions", {
        "model": "clozn-local",
        "messages": [{"role": "user", "content": "Conformance probe"}],
        "stream": False,
    })
    content = reply["choices"][0]["message"]["content"]
    assert content == "External client round trip.", reply
    streamed = _stream(base + "/api/chat/completions", {
        "model": "clozn-local",
        "messages": [{"role": "user", "content": "Streaming conformance probe"}],
        "stream": True,
    })
    assert streamed == "External client round trip.", streamed
    tool_exchange = _tool_exchange(base)
    print(json.dumps({"client": "open-webui", "model": "clozn-local",
                      "nonstream": content, "stream": streamed,
                      "tool_exchange": tool_exchange}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
