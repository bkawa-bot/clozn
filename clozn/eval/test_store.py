"""Tests for the eval report store (eval.store) -- round-trip persistence, framework-free."""
from __future__ import annotations

from clozn.eval import store


def test_save_load_round_trip(tmp_path):
    path = str(tmp_path / "eval_report.json")
    payload = {"set": "arith", "report": {"available": True, "ece": 0.1}, "n": 60}
    written = store.save(payload, path)
    assert written == path
    loaded = store.load(path)
    assert loaded["set"] == "arith" and loaded["n"] == 60
    assert "saved_ts" in loaded                                   # stamped on save


def test_save_preserves_an_explicit_saved_ts(tmp_path):
    path = str(tmp_path / "r.json")
    store.save({"saved_ts": 123.0, "x": 1}, path)
    assert store.load(path)["saved_ts"] == 123.0


def test_load_missing_or_junk_returns_none(tmp_path):
    assert store.load(str(tmp_path / "nope.json")) is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert store.load(str(bad)) is None
