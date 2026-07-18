"""tests/test_ask_band_signal.py -- generation_gateway.ask_band_signal: the route-layer glue that turns a
just-completed reply's raw per-token trace + the saved calibration (clozn eval --save) into the metadata-
only 'ask band' signal (calibration backlog: "a retrieval/clarify action wired to the policy's ask band").
Both /v1/chat/completions (openai.py, non-streaming) and the SSE stream (sse.py) call this SAME function,
so covering it here covers the one piece of logic both routes share.

Pure-ish: the only I/O is clozn.eval.store.load(), isolated per test via monkeypatching eval_store._PATH
(mirrors tests/test_journal_calibration.py's convention) -- never touches the real ~/.clozn.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clozn.eval import store as eval_store                     # noqa: E402
from clozn.server import generation_gateway as gw              # noqa: E402


MODEL = "fake-clozn-model"
SAVED = {"model": MODEL, "score": "min", "policy": {"answer_at": 0.8, "ask_at": 0.4}}

ASK_STEPS = [{"piece": "The", "conf": 0.95}, {"piece": " answer", "conf": 0.6}]     # min 0.6 -> ask
ANSWER_STEPS = [{"piece": "The", "conf": 0.97}, {"piece": " answer", "conf": 0.9}]  # min 0.9 -> answer
ABSTAIN_STEPS = [{"piece": "The", "conf": 0.2}, {"piece": " answer", "conf": 0.1}]  # min 0.1 -> abstain


def _save(tmp_path, monkeypatch, payload):
    path = str(tmp_path / "eval_report.json")
    monkeypatch.setattr(eval_store, "_PATH", path)
    eval_store.save(dict(payload), path)


def test_ask_band_returns_the_signal(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, SAVED)
    out = gw.policy_signal(ASK_STEPS, MODEL)
    assert out["band"] == "ask"
    assert out["score"] == 0.6
    assert out["answer_at"] == 0.8 and out["ask_at"] == 0.4
    assert out["score_aggregate"] == "min"
    assert "note" in out and "ask" in out["note"]


def test_abstain_band_returns_a_stronger_signal(tmp_path, monkeypatch):
    """Below the ask band, the model is likely wrong -- the note must say so, more strongly than the ask
    note, and must still carry the same score/threshold fields."""
    _save(tmp_path, monkeypatch, SAVED)
    out = gw.policy_signal(ABSTAIN_STEPS, MODEL)
    assert out["band"] == "abstain"
    assert out["score"] == 0.1
    assert out["answer_at"] == 0.8 and out["ask_at"] == 0.4
    assert out["score_aggregate"] == "min"
    assert "note" in out and "abstain" in out["note"] and "likely wrong" in out["note"]


def test_answer_band_returns_none(tmp_path, monkeypatch):
    """A confident answer says nothing this endpoint needs to add."""
    _save(tmp_path, monkeypatch, SAVED)
    assert gw.policy_signal(ANSWER_STEPS, MODEL) is None


def test_no_calibration_saved_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_store, "_PATH", str(tmp_path / "nope.json"))
    assert gw.policy_signal(ASK_STEPS, MODEL) is None


def test_model_mismatch_returns_none(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, SAVED)
    assert gw.policy_signal(ASK_STEPS, "a-completely-different-model") is None


def test_no_trace_returns_none(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, SAVED)
    assert gw.policy_signal([], MODEL) is None
    assert gw.policy_signal(None, MODEL) is None


def test_never_raises_on_malformed_saved_report(tmp_path, monkeypatch):
    path = str(tmp_path / "eval_report.json")
    monkeypatch.setattr(eval_store, "_PATH", path)
    eval_store.save({"policy": "not-a-dict"}, path)
    assert gw.policy_signal(ASK_STEPS, MODEL) is None


def test_ask_band_signal_alias_still_works(tmp_path, monkeypatch):
    """Backward compat: callers that imported the old name before the abstain band was wired in still work."""
    assert gw.ask_band_signal is gw.policy_signal
    _save(tmp_path, monkeypatch, SAVED)
    assert gw.ask_band_signal(ASK_STEPS, MODEL)["band"] == "ask"
