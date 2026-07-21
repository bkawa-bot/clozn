"""Focused HTTP wiring tests for GET /runs/<id>/diagnosis."""
from __future__ import annotations

import io
import json

import pytest

from clozn.server import app as cs
import clozn.runs.store as runlog


def _get(path):
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(b"")
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": "0", "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"GET {path} HTTP/1.1", "HTTP/1.1", "GET"
    h.do_GET()
    head, _, body = h.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode("latin-1"), json.loads(body)


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", None)


def test_diagnosis_missing_run_is_a_clean_404(iso):
    head, body = _get("/runs/run_nope/diagnosis")
    assert "404" in head
    assert body == {"error": "run not found"}


def test_diagnosis_is_zero_generation_and_receives_related_runs(iso):
    association = "client_" + "a" * 24
    target = runlog.record(
        source="openai_api", client="pytest", client_key=association,
        messages=[{"role": "user", "content": "Explain gravity"}],
        response="Mass attracts mass.", finish_reason="stop",
        started=1000.0, ended=1000.4,
    )
    related = runlog.record(
        source="openai_api", client="pytest", client_key=association,
        messages=[{"role": "user", "content": "Give a title"}], response="Gravity",
        finish_reason="stop", started=1001.0, ended=1001.1,
    )

    assert cs.SUB is None
    head, body = _get(f"/runs/{target}/diagnosis")

    assert "200" in head
    assert body["schema"] == "clozn.run_diagnosis.v1"
    assert body["run_id"] == target
    auxiliary = body["client_auxiliary_calls"]
    assert auxiliary["status"] == "observed"
    related_evidence = next(item for item in auxiliary["evidence"]
                            if item["path"] == "related_runs")
    assert related_evidence["value"][0]["run_id"] == related
