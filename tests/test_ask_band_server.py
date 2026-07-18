"""tests/test_ask_band_server.py -- the selective-generation policy's 'ask' and 'abstain' bands wired into
POST /v1/chat/completions as a metadata-only signal (clozn_policy), the calibration backlog item:
"a retrieval/clarify action wired to the policy's ask band" plus its abstain follow-on. Metadata only -- the
generated reply text is never touched; the field is silent (absent) unless a saved, model-matching
calibration actually says 'ask' or 'abstain' for this reply's confidence (clozn.eval.policy.classify_run via
clozn.server.generation_gateway.policy_signal).

Model-free: drives the REAL clozn_server do_POST handler with no socket (object.__new__(H)), isolated
runlog/cards/settings/eval stores, a fake substrate whose chat() fills a per-token trace. Mirrors
test_trust_field_server.py's conventions.
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

from clozn.server import app as cs                # noqa: E402
from clozn.eval import store as eval_store         # noqa: E402
import clozn.memory.cards as memory_cards          # noqa: E402
import clozn.memory.mode as memory_mode            # noqa: E402
import clozn.runs.store as runlog                  # noqa: E402


class FakeSteer:
    def __init__(self):
        self.strength = {}

    def active(self):
        return {}


class FakeMem:
    def __init__(self):
        self.memory_strength = 1.0
        self.rules = []
        self.prefix = None


MODEL = "fake-clozn-model"

# min-confidence 0.6 -> sits inside SAVED's ask band [0.4, 0.8)
ASK_STEPS = [{"piece": "The", "conf": 0.95}, {"piece": " answer", "conf": 0.6}]
# min-confidence 0.9 -> at/above the answer_at threshold
ANSWER_STEPS = [{"piece": "The", "conf": 0.97}, {"piece": " answer", "conf": 0.9}]
# min-confidence 0.1 -> below SAVED's ask_at threshold (abstain territory)
ABSTAIN_STEPS = [{"piece": "The", "conf": 0.3}, {"piece": " answer", "conf": 0.1}]

SAVED = {"model": MODEL, "score": "min", "policy": {"answer_at": 0.8, "ask_at": 0.4}}


class TraceSub:
    """A qwen-shaped substrate whose chat() fills trace_out with a real per-token trace, mirroring
    test_trust_field_server.py's TraceSub."""
    name = "qwen"

    def __init__(self, steps=ASK_STEPS, reply="The answer."):
        self.memory = FakeMem()
        self._mem = self.memory
        self.steer = FakeSteer()
        self._steps = steps
        self._reply = reply
        self._run_meta = {"model_id": MODEL, "sampler_mode": "greedy",
                          "sampling": "greedy", "temperature": 0.0}

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=False)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        if trace_out is not None:
            trace_out.extend([dict(s) for s in self._steps])
        return self._reply

    def last_finish_reason(self):
        return "stop"

    def run_meta(self):
        return dict(self._run_meta)


def _dispatch(path, body_obj):
    raw = json.dumps(body_obj).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    return h.wfile.getvalue()


def _post(path, body_obj):
    _, _, payload = _dispatch(path, body_obj).partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "no_such.pt")])
    monkeypatch.setattr(eval_store, "_PATH", str(tmp_path / "eval_report.json"))
    monkeypatch.setattr(cs, "SUB", TraceSub())
    return tmp_path


def _body(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "what's the answer?"}], **extra}


# ================================================================= the ask band attaches clozn_policy

def test_ask_band_attaches_clozn_policy(iso):
    eval_store.save(dict(SAVED))
    out = _post("/v1/chat/completions", _body())
    assert out["clozn_policy"] == {
        "band": "ask", "score": 0.6, "score_aggregate": "min", "answer_at": 0.8, "ask_at": 0.4,
        "note": out["clozn_policy"]["note"],
    }
    assert "ask" in out["clozn_policy"]["note"]
    # metadata only -- the reply text itself is untouched
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "The answer."}


# ================================================================= the abstain band attaches clozn_policy

def test_abstain_band_attaches_clozn_policy(iso, monkeypatch):
    eval_store.save(dict(SAVED))
    monkeypatch.setattr(cs, "SUB", TraceSub(steps=ABSTAIN_STEPS))
    out = _post("/v1/chat/completions", _body())
    assert out["clozn_policy"] == {
        "band": "abstain", "score": 0.1, "score_aggregate": "min", "answer_at": 0.8, "ask_at": 0.4,
        "note": out["clozn_policy"]["note"],
    }
    assert "abstain" in out["clozn_policy"]["note"] and "likely wrong" in out["clozn_policy"]["note"]
    # metadata only -- the reply text itself is untouched
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "The answer."}


# ================================================================= graceful degradation

def test_no_metadata_when_no_calibration_saved(iso):
    out = _post("/v1/chat/completions", _body())
    assert "clozn_policy" not in out
    # unchanged OpenAI shape (+ the pre-existing clozn_run_id bridge field) -- byte-compatible by default
    assert set(out.keys()) == {"id", "object", "created", "model", "choices", "clozn_run_id"}


def test_no_metadata_when_confidence_is_in_the_answer_band(iso, monkeypatch):
    eval_store.save(dict(SAVED))
    monkeypatch.setattr(cs, "SUB", TraceSub(steps=ANSWER_STEPS))
    out = _post("/v1/chat/completions", _body())
    assert "clozn_policy" not in out


def test_no_metadata_on_model_mismatch(iso):
    eval_store.save({**SAVED, "model": "a-totally-different-model"})
    out = _post("/v1/chat/completions", _body())
    assert "clozn_policy" not in out


def test_no_metadata_when_saved_report_carries_no_policy(iso):
    eval_store.save({"model": MODEL, "score": "min"})
    out = _post("/v1/chat/completions", _body())
    assert "clozn_policy" not in out
