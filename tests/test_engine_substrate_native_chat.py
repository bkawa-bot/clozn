"""Model-free tests for EngineSubstrate's private atomic native chat-I/O seam."""
from __future__ import annotations

import json

import pytest

from clozn.server import app as cs


class _AtomicEngine:
    def __init__(self, response=None, error=None):
        self.response = response or _native_response()
        self.error = error
        self.calls = []

    def complete_chat(self, messages, **options):
        self.calls.append({"messages": [dict(message) for message in messages], "options": dict(options)})
        if self.error is not None:
            raise self.error
        return self.response


class _Steer:
    def __init__(self):
        self.strength = {"warm": 0.65}
        self.layer = 14

    def steer_vector(self, strengths):
        assert strengths == self.strength
        return [0.25, -0.5]


def _native_response(*, trace=True, parse_error=None):
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_native",
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"Kyoto"}'},
        }],
    }
    raw = '  <tool_call>{"name":"weather","arguments":{"city":"Kyoto"}}</tool_call>  '
    chat_io = {
        "raw_model_output": raw,
        "rendered_prompt": "<s>[INST] Weather? [/INST]",
        "model_sha256": "a" * 64,
        "message": message,
        "openai_json": json.dumps(message, separators=(",", ":")),
        "format": "ministral-v3",
        "pipeline": {
            "executor_id": "clozn.chat_io.atomic_executor.v1",
            "renderer_id": "clozn.chat_io.llama_common.renderer.v1",
            "grammar_id": "clozn.chat_io.ar_grammar.v1",
            "parser_id": "clozn.chat_io.llama_common.parser.v1",
        },
        "trace": [],
    }
    if trace:
        chat_io["trace"] = [
            {"type": "tokens_committed", "items": [
                {"pos": 18, "id": 71, "piece": "<tool_call>", "conf": 0.91234},
                {"pos": 19, "id": 72, "piece": "{", "conf": 0.80126},
            ]},
            {"type": "step_lens", "positions": [18], "k": 2,
             "ids": [71, 99], "pieces": ["<tool_call>", "<think>"], "probs": [0.91, 0.04]},
        ]
    if parse_error is not None:
        chat_io.pop("message")
        chat_io.pop("openai_json")
        chat_io["parse_error"] = dict(parse_error)
    return {
        "id": "cmpl-native",
        "object": "text_completion",
        "choices": [{"text": raw, "index": 0, "finish_reason": "stop"}],
        "board": [],
        "layout": [],
        "usage": {"prompt_tokens": 18, "completion_tokens": 2, "steps_total": 2},
        "chat_io": chat_io,
    }


def _substrate(engine, steer=None):
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = engine
    sub.steer = steer
    sub.memory = object()
    sub._mem = sub.memory
    return sub


def test_private_native_chat_preserves_clozn_layers_and_real_worker_evidence(monkeypatch):
    block = "Known user context:\n- Prefers concise weather answers."
    applied = [{"id": "mem_weather", "text": "Prefers concise weather answers.",
                "relevance": 0.88}]
    monkeypatch.setattr(cs, "_prompt_block_for", lambda mem, user: (block, applied, 0.88))
    monkeypatch.setattr(cs, "_disk_dials", lambda: pytest.fail("live tone strengths must win"))
    engine = _AtomicEngine()
    sub = _substrate(engine, _Steer())
    messages = [
        {"role": "system", "content": "Client policy"},
        {"role": "user", "content": "Weather in Kyoto?"},
    ]
    tools = [{"type": "function", "function": {"name": "weather"}}]
    sampling = {"temperature": 0.3, "top_p": 0.75, "top_k": 12,
                "repeat_penalty": 1.04, "seed": 987}
    trace_out = []
    mem_out = {}

    result = sub._complete_chat_native(
        messages,
        tools=tools,
        tool_choice="required",
        parallel_tool_calls=False,
        max_new=41,
        sample=sampling,
        trace_out=trace_out,
        mem_out=mem_out,
        enable_thinking=False,
    )

    assert messages[0]["content"] == "Client policy"  # memory assembly never mutates caller input
    assert engine.calls == [{
        "messages": [
            {"role": "system", "content": f"Client policy\n\n{block}"},
            {"role": "user", "content": "Weather in Kyoto?"},
        ],
        "options": {
            "tools": tools,
            "tool_choice": "required",
            "json_schema": None,
            "parallel_tool_calls": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "reasoning_format": "none",
            "max_tokens": 41,
            "steer_vec": [0.25, -0.5],
            "steer": {"coef": 1.0, "layer": 14},
            "temperature": 0.3,
            "rep_penalty": 1.04,
            "top_k": 12,
            "top_p": 0.75,
            "seed": 987,
        },
    }]

    # No stripping or JSON reserialization: this is the worker's actual model output and parsed message.
    assert result["raw_model_output"].startswith("  <tool_call>")
    assert result["raw_model_output"].endswith("  ")
    assert result["message"] == _native_response()["chat_io"]["message"]
    assert result["parse_error"] is None
    assert result["pipeline"] == _native_response()["chat_io"]["pipeline"]
    assert result["rendered_prompt"] == mem_out["final_prompt"]
    assert result["model_sha256"] == "a" * 64
    assert result["finish_reason"] == "stop"
    assert result["usage"] == {"prompt_tokens": 18, "completion_tokens": 2, "steps_total": 2}

    assert [step["piece"] for step in trace_out] == ["<tool_call>", "{"]
    assert trace_out == result["trace"] == sub._request.trace
    assert trace_out[0]["alts"] == [{"token_id": 99, "piece": "<think>", "prob": 0.04}]
    assert sub._request.finish_reason == "stop"
    assert sub._request.prompt_tokens == 18
    assert sub._request.sampling["seed"] == 987
    assert sub._request.generation_meta["max_tokens"] == 41
    assert sub._request.generation_meta["stream"] is False
    assert sub._request.steering_snapshot == {"warm": 0.65}
    assert sub._request.memory_manifest == mem_out
    assert mem_out["prompt_block"] == block
    assert mem_out["assembled_messages"] == engine.calls[0]["messages"]
    assert mem_out["final_prompt"] == "<s>[INST] Weather? [/INST]"


def test_private_native_chat_greedy_and_missing_native_trace_stay_explicitly_empty(monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", lambda mem, user: (None, [], 0.0))
    monkeypatch.setattr(cs, "_disk_dials", lambda: {})
    engine = _AtomicEngine(_native_response(trace=False))
    sub = _substrate(engine)
    trace_out = []

    result = sub._complete_chat_native(
        [{"role": "user", "content": "Weather?"}],
        json_schema={"type": "object"},
        sample=False,
        trace_out=trace_out,
    )

    assert engine.calls[0]["options"] | {} == {
        "tools": None,
        "tool_choice": "auto",
        "json_schema": {"type": "object"},
        "parallel_tool_calls": False,
        "add_generation_prompt": True,
        "enable_thinking": True,
        "reasoning_format": "none",
        "max_tokens": 256,
        "temperature": 0.0,
        "rep_penalty": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "seed": 0,
    }
    assert result["trace"] == trace_out == sub._request.trace == []
    assert sub._request.sampling is None
    assert sub._request.memory_manifest["assembled_messages"] == [
        {"role": "user", "content": "Weather?"}
    ]


def test_private_native_chat_retains_raw_trace_and_usage_when_native_parse_fails(monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", lambda mem, user: (None, [], 0.0))
    monkeypatch.setattr(cs, "_disk_dials", lambda: {})
    parse_error = {"code": "native_parse_failed", "message": "expected a tool close tag"}
    engine = _AtomicEngine(_native_response(parse_error=parse_error))
    sub = _substrate(engine)
    trace_out = []

    result = sub._complete_chat_native(
        [{"role": "user", "content": "Weather?"}],
        json_schema={"type": "object"},
        sample=False,
        trace_out=trace_out,
    )

    assert result["raw_model_output"] == engine.response["chat_io"]["raw_model_output"]
    assert result["parse_error"] == parse_error
    assert result["message"] is None
    assert result["openai_json"] is None
    assert result["usage"] == engine.response["usage"]
    assert result["trace"] == trace_out == sub._request.trace
    assert [step["piece"] for step in trace_out] == ["<tool_call>", "{"]
    assert sub._request.finish_reason == "stop"
    assert sub._request.prompt_tokens == 18


def test_private_native_chat_keeps_request_memory_manifest_when_worker_fails(monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", lambda mem, user: ("MEMORY", [{"id": "m1"}], 0.7))
    monkeypatch.setattr(cs, "_disk_dials", lambda: {})
    sub = _substrate(_AtomicEngine(error=RuntimeError("native parse failed")))
    mem_out = {}

    with pytest.raises(RuntimeError, match="native parse failed"):
        sub._complete_chat_native(
            [{"role": "user", "content": "hello"}],
            json_schema={"type": "object"},
            mem_out=mem_out,
        )

    assert sub._request.memory_manifest == mem_out
    assert sub._request.finish_reason is None
    assert sub._request.prompt_tokens is None
    assert sub._request.trace == []
