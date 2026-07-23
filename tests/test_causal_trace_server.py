"""HTTP-level tests for POST /runs/<id>/causal-trace. Drives the real do_POST handler with the
no-socket object.__new__(H) trick used across this suite; tracer.trace is monkeypatched so these
never touch a live engine. Mirrors tests/test_provenance_server.py."""
from __future__ import annotations

import io
import json

import pytest

import clozn.analysis.tracer as tracer
from clozn.server import app as cs
import clozn.runs.store as runlog


def _post(path, body=None):
    raw = json.dumps(body if body is not None else {}).encode("utf-8")
    handler_type = cs.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    handler.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    handler.requestline = f"POST {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = "POST"
    handler.do_POST()
    head, _, payload = handler.wfile.getvalue().partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(payload.decode("utf-8"))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", None)
    monkeypatch.setattr(cs, "ENGINE", None)
    return tmp_path


def _seed(*, final_prompt="EXACT RENDERED PROMPT", response="Paris"):
    return runlog.record(
        source="openai_api", messages=[{"role": "user", "content": "the capital?"}],
        response=response, final_prompt=final_prompt,
    )


_RECEIPT = {"ok": True, "target": {"pos": 0, "piece": " Paris"},
            "controls": {"verdict": "PASS"}, "nodes": [],
            "config": {"scoring": "contrastive", "screen_mode": "mean_ablation"}}
_BLOCKED = {"ok": False, "blocked": "capture returned no rows"}


def test_run_not_found_is_404(isolated):
    status, out = _post("/runs/missing/causal-trace")
    assert status == 404 and out == {"error": "run not found"}


def test_missing_final_prompt_is_200_ok_false(isolated):
    rid = _seed(final_prompt=None)
    status, out = _post(f"/runs/{rid}/causal-trace")
    assert status == 200 and out["ok"] is False and "final_prompt" in out["blocked"]


def test_missing_response_is_200_ok_false(isolated):
    rid = _seed(response="")
    status, out = _post(f"/runs/{rid}/causal-trace")
    assert status == 200 and out["ok"] is False and "response" in out["blocked"]


def test_negative_position_is_400(isolated):
    rid = _seed()
    status, out = _post(f"/runs/{rid}/causal-trace", {"position": -1})
    assert status == 400 and "position" in out["error"]


def test_bad_screen_mode_is_400(isolated):
    rid = _seed()
    status, out = _post(f"/runs/{rid}/causal-trace", {"screen_mode": "nonsense"})
    assert status == 400 and "screen_mode" in out["error"]


def test_bad_contrast_type_is_400(isolated):
    rid = _seed()
    status, out = _post(f"/runs/{rid}/causal-trace", {"contrast": [1, 2]})
    assert status == 400 and "contrast" in out["error"]


def test_dispatch_passes_position_and_contrast_defaults(isolated, monkeypatch):
    seen = {}

    def fake_trace(prompt, continuation, target_idx, **kw):
        seen.update(prompt=prompt, continuation=continuation, target_idx=target_idx, **kw)
        return _RECEIPT

    monkeypatch.setattr(tracer, "trace", fake_trace)
    rid = _seed(final_prompt="RENDERED", response="Paris")
    status, out = _post(f"/runs/{rid}/causal-trace", {"position": 3})
    assert status == 200 and out["ok"] is True
    assert seen["prompt"] == "RENDERED" and seen["continuation"] == "Paris"
    assert seen["target_idx"] == 3
    assert seen["contrast"] == "auto"          # answer-selective default
    assert seen["screen_mode"] == "ablate"      # any-GGUF default


def test_explicit_null_contrast_disables_it(isolated, monkeypatch):
    seen = {}
    monkeypatch.setattr(tracer, "trace",
                        lambda *a, **k: seen.update(k) or _RECEIPT)
    rid = _seed()
    _post(f"/runs/{rid}/causal-trace", {"contrast": None})
    assert seen["contrast"] is None


def test_blocked_dict_passes_through_200(isolated, monkeypatch):
    monkeypatch.setattr(tracer, "trace", lambda *a, **k: _BLOCKED)
    rid = _seed()
    status, out = _post(f"/runs/{rid}/causal-trace")
    assert status == 200 and out["ok"] is False and "blocked" in out
