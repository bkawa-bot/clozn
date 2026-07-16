"""test_bridge_server -- the M5 any-client run_id bridge (EXPLAIN_THIS_ANSWER_SPEC.md): a user chatting
through ANY OpenAI-compatible client gets the clozn run_id back with their reply, so a companion
`clozn explain <run_id>` can inspect that exact reply -- without that client needing to know clozn exists.

Unlike M2-M4 (receipts.py / counterfactual.py / narrate.py, each unit-tested standalone with the server test
only proving thin wiring on top), M5 has no separate module: the whole feature IS the wiring -- `_log_run`
returning the run id it already computes, and `_json`/`_send` gaining an optional `extra_headers` param --
so this file is where that logic actually gets exercised end to end.

No model, no GPU: drives the REAL clozn_server do_POST handler (the object.__new__(H) no-socket trick used
by test_counterfactual_server.py / test_narrate_server.py) against an isolated runlog store + memory_cards
store + memory_mode settings, with a FAKE qwen-shaped substrate. Proves: a /v1/chat/completions POST returns
an otherwise-untouched OpenAI chat.completion body that ALSO carries "clozn_run_id"; that id resolves via
runlog.get_run() to the exact run just logged; the raw HTTP response carries an X-Clozn-Run-Id header with
the same value; a logging failure (runlog.record -> None) omits both cleanly rather than emitting a literal
"null"/"None"; the pre-existing 503-no-substrate path and _json's other (no-extra-header) call sites are
untouched; and the streaming path is verified to be left exactly as it was (the run id is deferred there,
by design -- not silently dropped without anyone noticing).
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.server import app as cs   # noqa: E402
import clozn.memory.cards as memory_cards         # noqa: E402
import clozn.memory.mode as memory_mode          # noqa: E402
import clozn.runs.store as runlog                # noqa: E402


# --- a fake, qwen-shaped substrate: /v1/chat/completions' non-stream path calls chat(messages, max_new,
# sample, trace_out=, mem_out=); its stream path (exercised only by the one streaming-sanity test below)
# calls chat_stream(messages, max_new, mem_out=). ---------------------------------------------------------

class FakeSteer:
    def __init__(self, strength=None):
        self.strength = dict(strength or {})

    def active(self):
        return {k: v for k, v in self.strength.items() if v}


class FakeMem:
    def __init__(self, strength=1.0):
        self.memory_strength = float(strength)
        self.rules = []
        self.prefix = None


class FakeSub:
    name = "qwen"

    def __init__(self, finish_reason="stop"):
        self.memory = FakeMem()
        self._mem = self.memory
        self.steer = FakeSteer()
        self.calls = 0
        self.finish_reason = finish_reason
        self._run_meta = {"model_id": "fake-qwen", "sampler_mode": "greedy",
                          "sampling": "greedy", "temperature": 0.0}

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self.calls += 1
        self._run_meta.update(max_tokens=int(max_new), stream=False)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        return "A plain reply."

    def chat_stream(self, messages, max_new=256, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=True)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        for piece in ("Hel", "lo"):
            yield piece

    def last_finish_reason(self):
        return self.finish_reason

    def run_meta(self):
        return dict(self._run_meta)


class NoFinishSub:
    name = "qwen"

    def __init__(self):
        self.memory = FakeMem()
        self._mem = self.memory
        self.steer = FakeSteer()
        self._run_meta = {"model_id": "legacy-fake-qwen", "sampler_mode": "sample",
                          "sampling": "sample", "temperature": 0.7}

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=False)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        return "A plain reply."

    def chat_stream(self, messages, max_new=256, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=True)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        for piece in ("Hel", "lo"):
            yield piece

    def run_meta(self):
        return dict(self._run_meta)


class PromptCaptureSub(FakeSub):
    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=False)
        block = "You are a helpful assistant talking with a returning user.\n- Keep it brief."
        assembled = [{"role": "system", "content": block}] + [dict(m) for m in messages]
        if mem_out is not None:
            mem_out.update(mode="prompt",
                           applied=[{"id": None, "text": "Keep it brief.", "relevance": 0.82}],
                           gate=0.91, prompt_block=block, assembled_messages=assembled)
        return "Brief reply."


class InternalizedSub(FakeSub):
    def __init__(self):
        super().__init__()
        self.memory.prefix = "PREFIX"
        self.memory.rules = ["Keep it brief."]

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=False)
        return "Prefix-shaped reply."


# --- driving the real handler without a socket (mirrors test_counterfactual_server / test_narrate_server) ---

def _dispatch(method, path, body_obj=None):
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"{method} {path} HTTP/1.1", "HTTP/1.1", method
    getattr(h, f"do_{method}")()
    return h.wfile.getvalue()


def _post_raw(path, body_obj=None):
    """The full raw bytes off the (fake) wire -- header block AND body -- for header assertions."""
    return _dispatch("POST", path, body_obj)


def _post(path, body_obj=None):
    """Just the parsed JSON body (matches the other server tests' convention)."""
    _, _, payload = _post_raw(path, body_obj).partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate the run/card/settings stores; SUB starts as a FakeSub (tests that want the 503 path
    override it to None explicitly)."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "no_such.pt")])
    monkeypatch.setattr(cs, "SUB", FakeSub())
    return tmp_path


# ==================================================================================== /v1/chat/completions

def test_chat_completions_needs_the_substrate_503_unchanged(iso, monkeypatch):
    """The pre-existing 503 path (a plain 2-arg self._json call) must be byte-for-byte unchanged: no bridge
    field in the body, no X-Clozn-Run-Id header -- proving _json's extra_headers param is opt-in only."""
    monkeypatch.setattr(cs, "SUB", None)
    raw = _post_raw("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    header_block, _, payload = raw.partition(b"\r\n\r\n")
    assert b"X-Clozn-Run-Id" not in header_block
    assert json.loads(payload.decode("utf-8")) == {"error": {"message": "model worker unavailable", "type": "service_unavailable"}}


def test_chat_completions_carries_clozn_run_id_that_resolves_to_the_logged_run(iso):
    out = _post("/v1/chat/completions", {"model": "clozn-qwen",
                                         "messages": [{"role": "user", "content": "hi there"}]})
    # the OpenAI shape is untouched except for the one additive field
    assert set(out.keys()) == {"id", "object", "created", "model", "choices", "usage", "clozn_run_id"}
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "A plain reply."}
    # the bridge: a real id that resolves, via runlog, to the exact run this reply just produced
    rid = out["clozn_run_id"]
    assert isinstance(rid, str) and rid
    logged = runlog.get_run(rid)
    assert logged is not None
    assert logged["source"] == "openai_api"
    assert logged["finish_reason"] == "stop"
    assert logged["meta"]["finish_reason_source"] == "substrate"
    assert logged["meta"]["sampler_mode"] == "greedy"
    assert logged["meta"]["temperature"] == 0.0
    assert logged["meta"]["max_tokens"] == 256
    assert logged["response"] == "A plain reply."
    assert logged["messages"] == [{"role": "user", "content": "hi there"}]


def test_prompt_mode_logs_the_exact_assembled_messages(iso, monkeypatch):
    memory_mode.set_mode("prompt")
    monkeypatch.setattr(cs, "SUB", PromptCaptureSub())
    out = _post("/v1/chat/completions", {"model": "clozn-qwen",
                                         "messages": [{"role": "user", "content": "hi"}]})
    logged = runlog.get_run(out["clozn_run_id"])

    assert logged["response"] == "Brief reply."
    assert logged["assembled_messages"] == [
        {"role": "system", "content": "You are a helpful assistant talking with a returning user.\n- Keep it brief."},
        {"role": "user", "content": "hi"},
    ]
    assert logged["memory"]["mode"] == "prompt"
    assert logged["memory"]["prompt_block"].endswith("- Keep it brief.")
    assert logged["memory"]["cards_applied"] == ["Keep it brief."]
    assert logged["memory"]["relevance"] == [0.82]


def test_internalized_mode_does_not_fabricate_an_assembled_prompt(iso, monkeypatch):
    monkeypatch.setenv("CLOZN_RUNTIME_KIND", "lab")
    memory_mode.set_mode("internalized")
    monkeypatch.setattr(cs, "SUB", InternalizedSub())
    out = _post("/v1/chat/completions", {"model": "clozn-qwen",
                                         "messages": [{"role": "user", "content": "hi"}]})
    logged = runlog.get_run(out["clozn_run_id"])

    assert logged["response"] == "Prefix-shaped reply."
    assert logged["assembled_messages"] is None
    assert logged["memory"]["mode"] == "internalized"
    assert logged["memory"]["has_prefix"] is True


def test_chat_completions_uses_real_length_finish_reason_end_to_end(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSub(finish_reason="length"))
    out = _post("/v1/chat/completions", {"model": "clozn-qwen",
                                         "messages": [{"role": "user", "content": "hi"}],
                                         "max_tokens": 3})
    assert out["choices"][0]["finish_reason"] == "length"
    logged = runlog.get_run(out["clozn_run_id"])
    assert logged["finish_reason"] == "length"
    assert "truncated" in logged["flags"]
    assert logged["meta"]["finish_reason_source"] == "substrate"
    assert logged["meta"]["max_tokens"] == 3


def test_chat_completions_fallback_finish_reason_is_explicit_not_persisted_as_real(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", NoFinishSub())
    out = _post("/v1/chat/completions", {"model": "clozn-qwen",
                                         "messages": [{"role": "user", "content": "hi"}],
                                         "max_tokens": 4})
    assert out["choices"][0]["finish_reason"] == "stop"
    logged = runlog.get_run(out["clozn_run_id"])
    assert logged["finish_reason"] is None
    assert logged["meta"]["finish_reason_source"] == "fallback"
    assert logged["meta"]["finish_reason_fallback"] == "stop"
    assert logged["meta"]["max_tokens"] == 4


def test_chat_completions_raw_http_response_carries_the_x_clozn_run_id_header(iso):
    raw = _post_raw("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi there"}]})
    header_block, _, payload = raw.partition(b"\r\n\r\n")
    rid = json.loads(payload.decode("utf-8"))["clozn_run_id"]
    assert f"X-Clozn-Run-Id: {rid}".encode("utf-8") in header_block


def test_chat_completions_omits_the_bridge_cleanly_when_logging_fails(iso, monkeypatch):
    """runlog.record failing (returning None, its own documented contract) must not surface a literal
    "null"/"None" anywhere, and must not break the reply itself -- logging must never break the request."""
    monkeypatch.setattr(runlog, "record", lambda **kw: None)
    raw = _post_raw("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    header_block, _, payload = raw.partition(b"\r\n\r\n")
    body = json.loads(payload.decode("utf-8"))
    assert "clozn_run_id" not in body
    assert b"X-Clozn-Run-Id" not in header_block
    assert "null" not in payload.decode("utf-8").lower()
    assert body["choices"][0]["message"]["content"] == "A plain reply."   # the reply itself is unaffected


def test_streaming_path_is_left_unchanged_run_id_deferred_not_dropped(iso):
    """Documents + enforces the deferral decision: streaming still ends in [DONE] with no clozn_run_id or
    X-Clozn-Run-Id anywhere -- this is a deliberate scope fence (see the comment in _sse_chat), not a bug."""
    raw = _post_raw("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": True})
    text = raw.decode("utf-8")
    assert "data: [DONE]" in text
    assert "clozn_run_id" not in text
    assert "X-Clozn-Run-Id" not in text
    # the reply still streamed through untouched
    assert '"content": "Hel"' in text and '"content": "lo"' in text


def test_streaming_path_uses_real_length_finish_reason_and_logs_it(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSub(finish_reason="length"))
    raw = _post_raw("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                             "stream": True, "max_tokens": 2})
    text = raw.decode("utf-8")
    assert '"finish_reason": "length"' in text
    logged = runlog.get_run(runlog.list_runs(1)[0]["id"])
    assert logged["finish_reason"] == "length"
    assert logged["meta"]["finish_reason_source"] == "substrate"
    assert logged["meta"]["max_tokens"] == 2
    assert logged["meta"]["stream"] is True
