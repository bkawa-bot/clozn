"""Tests for the eval report store (eval.store) -- round-trip persistence, framework-free."""
from __future__ import annotations

import json

import pytest

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


def _payload(model="model-a", saved_ts=100.0, marker="first"):
    return {
        "model": model,
        "saved_ts": saved_ts,
        "set": "arith",
        "score": "min",
        "policy": {"answer_at": 0.8, "ask_at": 0.5},
        "marker": marker,
    }


def test_save_profile_stores_provenance_and_activates_legacy_report(tmp_path):
    path = str(tmp_path / "eval_report.json")
    profile = store.save_profile(_payload(), "  Code   Generation ", path)

    assert profile["schema"] == "clozn.calibration_profile.v1"
    assert profile["model"] == "model-a"
    assert profile["task"] == "code generation"
    assert profile["provenance"]["model_match"] == "exact"
    assert profile["provenance"]["task_match"] == "exact_normalized_task"
    assert "not a per-answer correctness probability" in profile["provenance"]["claim_limit"]
    assert store.load(path) == profile
    assert store.load_profile("model-a", "code generation", path) == profile
    assert not (tmp_path / "calibration_profiles.json.tmp").exists()


def test_registry_is_derived_from_monkeypatched_legacy_path(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_PATH", str(tmp_path / "isolated" / "eval_report.json"))
    store.save_profile(_payload(), "chat")

    assert (tmp_path / "isolated" / "eval_report.json").is_file()
    registry = tmp_path / "isolated" / "calibration_profiles.json"
    assert registry.is_file()
    raw = json.loads(registry.read_text(encoding="utf-8"))
    assert raw["schema"] == "clozn.calibration_profile_registry.v1"
    assert len(raw["profiles"]) == 1


def test_same_model_task_replaces_only_that_pair(tmp_path):
    path = str(tmp_path / "eval_report.json")
    store.save_profile(_payload(saved_ts=10, marker="old-chat"), "chat", path)
    store.save_profile(_payload(saved_ts=20, marker="arith"), "arith", path)
    store.save_profile(_payload(model="model-b", saved_ts=30, marker="other-model"), "chat", path)
    replacement = store.save_profile(_payload(saved_ts=40, marker="new-chat"), " CHAT ", path)

    profiles = store.list_profiles(path)
    assert len(profiles) == 3
    assert store.load_profile("model-a", "chat", path) == replacement
    assert store.load_profile("model-a", "arith", path)["marker"] == "arith"
    assert store.load_profile("model-b", "chat", path)["marker"] == "other-model"
    assert [p["marker"] for p in profiles] == ["new-chat", "other-model", "arith"]


def test_task_omitted_selects_newest_for_exact_model_and_explicit_task_never_falls_back(tmp_path):
    path = str(tmp_path / "eval_report.json")
    store.save_profile(_payload(saved_ts=100, marker="chat"), "chat", path)
    store.save_profile(_payload(saved_ts=300, marker="newest"), "retrieval qa", path)
    store.save_profile(_payload(model="MODEL-A", saved_ts=400, marker="case-other"), "chat", path)

    assert store.load_profile("model-a", path=path)["marker"] == "newest"
    assert store.load_profile("model-a", " Retrieval   QA ", path)["marker"] == "newest"
    assert store.load_profile("model-a", "missing task", path) is None
    assert store.load_profile("MODEL-A", path=path)["marker"] == "case-other"
    assert store.load_profile("Model-A", path=path) is None


def test_profile_validation_is_stricter_without_changing_legacy_save(tmp_path):
    path = str(tmp_path / "eval_report.json")
    # Legacy save/load continues to preserve arbitrary payloads exactly apart from
    # its longstanding saved_ts stamp behavior.
    store.save({"model": None, "x": object().__class__.__name__, "saved_ts": "legacy"}, path)
    assert store.load(path) == {"model": None, "x": "object", "saved_ts": "legacy"}

    with pytest.raises(ValueError, match="dictionary"):
        store.save_profile([], "chat", path)
    with pytest.raises(ValueError, match="model"):
        store.save_profile({"model": ""}, "chat", path)
    with pytest.raises(ValueError, match="task"):
        store.save_profile(_payload(), "   ", path)
    with pytest.raises(ValueError, match="control characters"):
        store.save_profile(_payload(), "chat\nretrieval", path)
    with pytest.raises(ValueError, match="at most 80"):
        store.save_profile(_payload(), "x" * 81, path)
    with pytest.raises(ValueError, match="saved_ts"):
        store.save_profile(_payload(saved_ts=float("nan")), "chat", path)
    with pytest.raises(ValueError, match="JSON serializable"):
        store.save_profile({**_payload(), "bad": object()}, "chat", path)


def test_corrupt_registry_never_raises_and_can_be_recovered(tmp_path):
    path = str(tmp_path / "eval_report.json")
    registry = tmp_path / "calibration_profiles.json"
    registry.write_text("{not json", encoding="utf-8")

    assert store.list_profiles(path) == []
    assert store.load_profile("model-a", path=path) is None
    recovered = store.save_profile(_payload(marker="recovered"), "chat", path)
    assert store.list_profiles(path) == [recovered]

    registry.write_text(json.dumps({"schema": "wrong", "profiles": "also wrong"}), encoding="utf-8")
    assert store.list_profiles(path) == []
    # The active legacy report remains a safe compatibility fallback, but still
    # cannot be borrowed by another explicit task.
    assert store.load_profile("model-a", "chat", path) == recovered
    assert store.load_profile("model-a", "another task", path) is None


def test_corrupt_entries_are_ignored_without_hiding_valid_profiles(tmp_path):
    path = str(tmp_path / "eval_report.json")
    valid = store.save_profile(_payload(), "chat", path)
    registry = tmp_path / "calibration_profiles.json"
    raw = json.loads(registry.read_text(encoding="utf-8"))
    raw["profiles"].extend([
        None,
        {"schema": "clozn.calibration_profile.v1", "model": "model-a", "task": ""},
        {"schema": "clozn.calibration_profile.v1", "model": "model-a", "task": "chat",
         "saved_ts": "yesterday"},
        {"schema": "clozn.calibration_profile.v1", "model": "model-a", "task": "chat",
         "provenance": {}},
    ])
    registry.write_text(json.dumps(raw), encoding="utf-8")

    assert store.list_profiles(path) == [valid]
    assert store.load_profile("model-a", "chat", path) == valid
