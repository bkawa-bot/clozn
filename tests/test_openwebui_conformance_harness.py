"""Model-free checks for the released Open WebUI external-process tool harness."""
from __future__ import annotations

import json

from tests.clients.fake_gateway import (
    TOOL_ARGUMENTS,
    TOOL_CALL_ID,
    TOOL_FINAL_ANSWER,
    TOOL_NAME,
    TOOL_RESULT,
    TOOL_SPEC,
    tool_completion,
)


def _initial_body():
    return {
        "model": "clozn-local",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "stream": False,
        "tools": [TOOL_SPEC],
    }


def test_fake_provider_requires_and_completes_exact_tool_continuation():
    initial = _initial_body()
    status, first = tool_completion(initial)
    assert status == 200
    choice = first["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assistant = choice["message"]
    call = assistant["tool_calls"][0]
    assert call["id"] == TOOL_CALL_ID
    assert json.loads(call["function"]["arguments"]) == TOOL_ARGUMENTS

    tool_result = {
        "role": "tool",
        "tool_call_id": TOOL_CALL_ID,
        "name": TOOL_NAME,
        "content": json.dumps(TOOL_RESULT, separators=(",", ":")),
    }
    status, final = tool_completion({
        **initial,
        "messages": [initial["messages"][0], assistant, tool_result],
    })
    assert status == 200
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["choices"][0]["message"]["content"] == TOOL_FINAL_ANSWER


def test_fake_provider_rejects_reshaped_tool_transport():
    body = _initial_body()
    body["tools"] = []
    status, response = tool_completion(body)
    assert status == 400
    assert response["error"]["code"] == "conformance_probe_mismatch"
    assert "tool definition" in response["error"]["message"]

