"""Tests for the hard-signals detector (signals.py) -- pure over synthetic runs."""
from __future__ import annotations

from clozn.runs import signals


def _run(reply="A fine reply.", **kw):
    r = {"source": "chat", "response": reply, "finish_reason": "stop"}
    r.update(kw)
    return r


def test_clean_run_has_no_hard_signals():
    assert signals.hard_signals(_run("The capital of France is Paris.")) == []


def test_error_and_truncation():
    assert "errored" in " ".join(signals.hard_signals(_run(error="boom")))
    assert any("cut off" in s for s in signals.hard_signals(_run(finish_reason="length")))


def test_empty_reply():
    assert any("empty" in s for s in signals.hard_signals(_run("   ")))


def test_repetition_loop():
    looped = " ".join(["the cake"] * 6)
    assert any("repeating" in s for s in signals.hard_signals(_run(looped)))


def test_bad_json_block_flagged_valid_not():
    bad = "Here you go:\n```json\n{ \"a\": 1, }\n```"
    assert any("JSON" in s for s in signals.hard_signals(_run(bad)))
    good = "Here you go:\n```json\n{ \"a\": 1 }\n```"
    assert not any("JSON" in s for s in signals.hard_signals(_run(good)))


def test_bad_json_with_trailing_content_is_caught():
    # valid JSON prefix but junk before the close fence -> the whole block isn't valid JSON -> flag
    blk = "```json\n{\"a\": 1}\n// a trailing comment\n```"
    assert any("JSON" in s for s in signals.hard_signals(_run(blk)))
    # a non-JSON fenced code block (python) is never JSON-checked
    py = "```python\nprint('hi')\n```"
    assert not any("JSON" in s for s in signals.hard_signals(_run(py)))


def test_is_organic_skips_machine_traffic():
    assert signals.is_organic(_run()) is True
    assert signals.is_organic(_run(source="replay")) is False
    assert signals.is_organic(_run(parent_run_id="run_x")) is False


def test_never_raises_on_junk():
    for junk in (None, {}, "str", {"response": 42}):
        assert signals.hard_signals(junk) == []
