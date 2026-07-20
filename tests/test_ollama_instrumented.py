"""End-to-end contracts for the Ollama-shaped entrance to Clozn's runtime.

The important compatibility invariant is not merely that the response looks like
Ollama.  /api/chat and /api/generate must cross the active Substrate.chat seam so
memory, steering, per-token trace capture, finish reasons, and the run journal all
observe the same execution.  These tests drive the real handler without a model or
socket and resolve the returned X-Clozn-Run-Id against an isolated journal.
"""
from __future__ import annotations

import io
import json

import pytest

import clozn.memory.cards as memory_cards
import clozn.memory.mode as memory_mode
import clozn.runs.store as runlog
from clozn.server import app as cs


class FakeSteer:
    strength = {"grounded": 0.4}

    def active(self):
        return dict(self.strength)


class FakeMemory:
    memory_strength = 0.75
    prefix = None
    rules = []


class InstrumentedSub:
    name = "engine"
    brain = None

    def __init__(self, *, fail=False):
        self._mem = self.memory = FakeMemory()
        self.steer = FakeSteer()
        self.fail = fail
        self.calls = []
        self._meta = {"model_id": "fake-qwen", "sampler_mode": "sample",
                      "sampling": "sample", "temperature": 0.8}

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self.calls.append({"messages": [dict(m) for m in messages],
                           "max_new": max_new, "sample": sample})
        self._meta.update(max_tokens=int(max_new), stream=False)
        if mem_out is not None:
            block = "You are a helpful assistant.\n- Prefer grounded answers."
            mem_out.update(
                mode="prompt",
                applied=[{"id": None, "text": "Prefer grounded answers.", "relevance": 0.88}],
                gate=0.73,
                strength=0.75,
                prompt_block=block,
                assembled_messages=[{"role": "system", "content": block}]
                + [dict(m) for m in messages],
                final_prompt="<rendered>grounded prompt</rendered>",
            )
        if self.fail:
            raise RuntimeError("synthetic decode failure")
        if trace_out is not None:
            trace_out.extend([
                {"pos": 0, "token_id": 41, "piece": "Observed", "prob": 0.93,
                 "alts": [{"token_id": 42, "piece": "Maybe", "prob": 0.04}]},
                {"pos": 1, "token_id": 43, "piece": " reply.", "prob": 0.47,
                 "alts": [{"token_id": 44, "piece": " answer.", "prob": 0.39}]},
            ])
        return "Observed reply."

    def last_finish_reason(self):
        return "stop"

    def run_meta(self):
        return dict(self._meta)


def _dispatch(method: str, path: str, body=None):
    raw = json.dumps(body if body is not None else {}).encode("utf-8")
    handler_type = cs.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    handler.headers = {"Content-Length": str(len(raw)), "User-Agent": "ollama-python/0.test"}
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = method
    handler.close_connection = False
    getattr(handler, f"do_{method}")()
    return handler.wfile.getvalue()


def _payload(raw: bytes) -> dict:
    return json.loads(raw.partition(b"\r\n\r\n")[2].decode("utf-8"))


@pytest.fixture
def iso(tmp_path, monkeypatch):
    sub = InstrumentedSub()
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "missing.pt")])
    monkeypatch.setattr(cs, "SUB", sub)
    monkeypatch.setattr(cs, "SUBNAME", "engine")
    return sub


def test_ollama_routes_are_registered_on_the_real_handler(iso):
    out = _payload(_dispatch("GET", "/api/version"))
    assert out == {"version": "0.0.0-clozn"}


def test_ollama_chat_uses_instrumented_substrate_and_returns_resolvable_run_id(iso):
    raw = _dispatch("POST", "/api/chat", {
        "model": "qwen3.5:9b",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
        "options": {"num_predict": 7, "temperature": 0.3, "top_p": 0.8,
                    "top_k": 24, "repeat_penalty": 1.15, "seed": 17},
    })
    out = _payload(raw)
    rid = out["clozn_run_id"]

    assert out == {"model": "qwen3.5:9b",
                   "message": {"role": "assistant", "content": "Observed reply."},
                   "done": True, "clozn_run_id": rid}
    assert f"X-Clozn-Run-Id: {rid}".encode() in raw.partition(b"\r\n\r\n")[0]
    assert iso.calls == [{
        "messages": [{"role": "user", "content": "hello"}],
        "max_new": 7,
        "sample": {"temperature": 0.3, "top_p": 0.8, "top_k": 24,
                   "repeat_penalty": 1.15, "seed": 17},
    }]

    logged = runlog.get_run(rid)
    assert logged["source"] == "ollama_api"
    assert logged["client"] == "ollama-python/0.test"
    assert logged["response"] == "Observed reply."
    assert logged["trace"]["tokens"] == ["Observed", " reply."]
    assert logged["trace"]["confidence"] == [0.93, 0.47]
    assert logged["trace"]["alternatives"][1][0]["piece"] == " answer."
    assert logged["memory"]["cards_applied"] == ["Prefer grounded answers."]
    assert logged["memory"]["gate"] == 0.73
    assert logged["behavior"]["active_dials"] == {"grounded": 0.4}
    assert logged["final_prompt"] == "<rendered>grounded prompt</rendered>"
    assert logged["finish_reason"] == "stop"
    assert logged["meta"]["compatibility_api"] == "ollama"
    assert logged["meta"]["ollama_operation"] == "chat"


def test_ollama_generate_enters_the_same_path_as_a_user_turn(iso):
    out = _payload(_dispatch("POST", "/api/generate", {
        "model": "qwen3.5:9b",
        "system": "Answer tersely.",
        "prompt": "Why is the sky blue?",
        "stream": False,
        "options": {"num_predict": 9, "temperature": 0},
    }))
    rid = out["clozn_run_id"]
    assert out["response"] == "Observed reply."
    assert iso.calls[0] == {
        "messages": [{"role": "system", "content": "Answer tersely."},
                     {"role": "user", "content": "Why is the sky blue?"}],
        "max_new": 9,
        "sample": {"temperature": 0},
    }
    logged = runlog.get_run(rid)
    assert logged["prompt_summary"] == "Why is the sky blue?"
    assert logged["meta"]["ollama_operation"] == "generate"
    assert logged["trace"]["token_ids"] == [41, 43]


def test_ollama_generation_failure_is_still_an_inspectable_run(iso):
    iso.fail = True
    out = _payload(_dispatch("POST", "/api/chat", {
        "model": "qwen3.5:9b", "messages": [{"role": "user", "content": "break"}],
        "stream": False,
    }))
    assert out == {"error": "engine: synthetic decode failure"}
    rows = runlog.list_runs(1)
    assert len(rows) == 1
    logged = runlog.get_run(rows[0]["id"])
    assert logged["source"] == "ollama_api"
    assert logged["error"] == "synthetic decode failure"
    assert logged["meta"]["compatibility_api"] == "ollama"


def test_ollama_streaming_remains_an_explicit_unsupported_operation(iso):
    out = _payload(_dispatch("POST", "/api/chat", {
        "messages": [{"role": "user", "content": "hello"}], "stream": True,
    }))
    assert out == {"error": "streaming not yet supported for /api/chat"}
    assert iso.calls == []
    assert runlog.list_runs(1) == []
