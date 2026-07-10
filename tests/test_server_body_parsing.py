"""test_server_body_parsing -- do_POST's body-parsing guard (bugs #4/#5).

Every POST route in clozn.server.app does `body.get(...)` with no isinstance check, and until now
do_POST() itself had no try/except around `json.loads(...)`. Two failure modes reached every single POST
route unguarded:

  #4 a body that isn't even valid JSON (dropped comma, truncated bytes, garbage) raised an uncaught
     json.JSONDecodeError inside do_POST, before any route ever ran.
  #5 a body that IS valid JSON but not an object (`[1,2,3]`, `"hi"`, `42`, `null`) sailed straight
     through to a route's `body.get(...)` and blew up with AttributeError/whatever the non-dict lacks.

Both are now caught centrally in do_POST with a clean 400 JSON response. Drives the REAL do_POST handler
with no socket (object.__new__(H)), mirroring test_jlens_server.py / test_receipts_server.py.
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from clozn.server import app as cs   # noqa: E402
import clozn.runs.store as runlog     # noqa: E402


def _post_raw(path, raw: bytes):
    """Like the other server tests' `_post`, but takes raw bytes so a genuinely malformed (invalid) JSON
    body can be sent -- json.dumps() can only ever produce valid JSON, so the malformed-body repro (#4)
    needs to bypass it and write bytes straight into rfile."""
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    head, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode("latin-1"), json.loads(payload.decode("utf-8"))


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", None)
    return tmp_path


# --------------------------------------------------------------------------------- #4: malformed JSON body

def test_malformed_json_body_is_a_clean_400_not_a_traceback(iso):
    """Repro: a body that isn't valid JSON at all (trailing comma) used to raise an uncaught
    json.JSONDecodeError inside do_POST itself, before any route ever ran."""
    head, out = _post_raw("/jlens", b'{"text": "hi", }')
    assert "400" in head
    assert "error" in out


def test_garbage_bytes_body_is_a_clean_400(iso):
    head, out = _post_raw("/runs/run_x/explain", b"not json at all")
    assert "400" in head
    assert "error" in out


def test_truncated_json_body_is_a_clean_400(iso):
    head, out = _post_raw("/runs/run_x/receipt", b'{"influence": {"dial": "warm"')   # cut off mid-object
    assert "400" in head
    assert "error" in out


def test_empty_body_still_falls_back_to_an_empty_object(iso):
    """The pre-existing `or b"{}"` fallback for a zero-length body (e.g. a POST with no Content-Length)
    must keep working: this is a valid, empty JSON object, not a parse failure -- the 400 below comes
    from /jlens's OWN validation ("need a 'text'"), not from the new parse guard."""
    head, out = _post_raw("/jlens", b"")
    assert "400" in head
    assert out == {"error": "need a 'text' to read"}


# ----------------------------------------------------------------------------- #5: valid JSON, non-dict body

def test_list_json_body_is_a_clean_400_not_an_attributeerror(iso):
    """Repro: valid JSON, but not an object -- every route does body.get(...) unguarded, so this used to
    raise AttributeError: 'list' object has no attribute 'get' deep inside whichever route matched."""
    head, out = _post_raw("/jlens", b"[1, 2, 3]")
    assert "400" in head
    assert "error" in out


def test_string_json_body_is_a_clean_400(iso):
    head, out = _post_raw("/runs/run_x/receipt", b'"just a string"')
    assert "400" in head
    assert "error" in out


def test_number_json_body_is_a_clean_400(iso):
    head, out = _post_raw("/jlens", b"42")
    assert "400" in head
    assert "error" in out


def test_null_json_body_is_a_clean_400(iso):
    head, out = _post_raw("/jlens", b"null")
    assert "400" in head
    assert "error" in out


# --------------------------------------------------------------------------- sanity: the happy path survives

def test_a_normal_dict_body_still_reaches_its_route(iso):
    """Guardrail: the new checks must not regress the ordinary case -- a legit dict body still reaches
    /jlens's own logic (here: its own honest 400 for missing text, proving dispatch happened at all)."""
    head, out = _post_raw("/jlens", json.dumps({"text": ""}).encode("utf-8"))
    assert "400" in head
    assert out == {"error": "need a 'text' to read"}
