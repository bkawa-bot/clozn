"""test_confidence_spans_server -- GET /runs/<id>/spans, the endpoint wiring for confidence spans.

Zero generation (confidence_spans.spans() is a pure reshape of what's already logged), so like /timeline
and /export this route needs no substrate at all -- SUB can stay None. Drives the REAL clozn_server do_GET
handler with the no-socket object.__new__(H) trick (mirrors test_run_timeline_server.py's `_get()` harness).
confidence_spans.py itself is unit-tested exhaustively against fixture dicts in test_confidence_spans.py;
this file only proves the thin endpoint wiring: the route matches, a missing run is a clean 404, and a real
run's spans + summary come back over HTTP.
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

import clozn_server as cs   # noqa: E402
import runlog                # noqa: E402


def _get(path):
    """Drive do_GET without a socket; return (raw header block, raw body bytes) (mirrors test_run_timeline_server.py)."""
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(b"")
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": "0", "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"GET {path} HTTP/1.1", "HTTP/1.1", "GET"
    h.do_GET()
    head, _, body = h.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode("latin-1"), body


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate the run log; SUB stays None -- the endpoint must not need a substrate."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", None)
    return tmp_path


def test_spans_missing_run_is_a_clean_404(iso):
    head, body = _get("/runs/run_nope/spans")
    assert "404" in head
    assert json.loads(body) == {"error": "run not found"}


def test_spans_needs_no_substrate_at_all(iso):
    """The whole point of a pure reshape: it must work with SUB is None (unlike /replay, /branch)."""
    rid = runlog.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey",
                        trace={"tokens": ["hey"], "confidence": [0.9]})
    assert cs.SUB is None
    head, body = _get(f"/runs/{rid}/spans")
    assert "200" in head
    data = json.loads(body)
    assert data["run_id"] == rid
    assert "error" not in data
    assert data["spans"] == [{"start": 0, "end": 0, "text": "hey", "band": "strong", "mean_conf": 0.9,
                              "min_conf": 0.9, "n_tokens": 1, "hesitations": 0}]
    assert data["summary"] == "Confident throughout."


def test_spans_happy_path_returns_spans_and_summary_over_http(iso):
    tokens = ["The", " sky", " is", " blue", ".", " Maybe", " it", " will", " rain", " later", "."]
    confidence = [0.95, 0.90, 0.92, 0.88, 0.99, 0.45, 0.30, 0.55, 0.60, 0.85, 0.97]
    rid = runlog.record(
        source="engine_chat", model="clozn-qwen",
        messages=[{"role": "user", "content": "weather?"}],
        response="The sky is blue. Maybe it will rain later.",
        trace={"tokens": tokens, "confidence": confidence},
    )
    head, body = _get(f"/runs/{rid}/spans")
    assert "200" in head
    data = json.loads(body)
    assert data["run_id"] == rid
    bands = [(s["start"], s["end"], s["band"]) for s in data["spans"]]
    assert bands == [(0, 4, "strong"), (5, 6, "shaky"), (7, 8, "okay"), (9, 10, "strong")]
    # text round-tripped verbatim over the wire
    assert "".join(s["text"] for s in data["spans"]) == "".join(tokens)
    assert data["summary"] == "Mostly steady, but 1 shaky span (weakest in the middle)."


def test_spans_run_with_no_trace_returns_an_empty_list_and_empty_summary(iso):
    rid = runlog.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey there")
    head, body = _get(f"/runs/{rid}/spans")
    assert "200" in head
    data = json.loads(body)
    assert data["spans"] == []
    assert data["summary"] == ""
