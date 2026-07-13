"""Tests for the server-side push (push.py) -- synchronous dispatch + a recording notifier, no OS."""
from __future__ import annotations

from clozn.watch.push import push_if_alerting
from clozn.watch.notify import RecordingNotifier


def _run(rid, confs, **kw):
    r = {"id": rid, "source": "chat", "finish_reason": "stop", "prompt_summary": "q",
         "trace": {"tokens": [f"t{i}" for i in range(len(confs))], "confidence": list(confs)}}
    r.update(kw)
    return r


def test_clean_run_pushes_nothing():
    note = RecordingNotifier()
    al = push_if_alerting(_run("a", [0.95, 0.93]), "http://h/r/a", notifier=note, async_dispatch=False)
    assert al is None and note.sent == []


def test_truncated_run_pushes_one_toast_with_link():
    note = RecordingNotifier()
    al = push_if_alerting(_run("a", [0.9, 0.9], finish_reason="length"), "http://h:8090/r/a",
                          notifier=note, async_dispatch=False)
    assert al is not None and al.severity == "high"
    assert len(note.sent) == 1
    title, body, url = note.sent[0]
    assert url == "http://h:8090/r/a" and "clozn" in title


def test_push_never_raises_on_junk():
    note = RecordingNotifier()
    assert push_if_alerting(None, "http://h/r/x", notifier=note, async_dispatch=False) is None
    assert note.sent == []
