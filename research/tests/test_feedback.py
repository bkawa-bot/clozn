"""Model-free tests for feedback.py -- the preference-signal capture store. No server, no model; a temp
file stands in for ~/.clozn/feedback.json (monkeypatch the module's _PATH)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import feedback  # noqa: E402


def test_record_fields_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "_PATH", str(tmp_path / "feedback.json"))
    s = feedback.record("run_1", "quick_repair", dial="concise", direction=1,
                        meta={"complaint": "verbose"}, _now=1000.0)
    assert s["run_id"] == "run_1" and s["dial"] == "concise" and s["direction"] == 1
    assert s["kind"] == "quick_repair" and s["meta"] == {"complaint": "verbose"}
    assert s["ts"] == 1000.0 and isinstance(s["id"], str)

    feedback.record("run_2", "quick_repair", dial="concise", direction=1, _now=1001.0)
    feedback.record("run_3", "quick_repair", dial="warm", direction=1, _now=1002.0)
    feedback.record("run_4", "thumb", _now=1003.0)   # directionless -> counted under `other`

    summ = feedback.summary()
    assert summ["total"] == 4 and summ["other"] == 1
    top = summ["by_dial"][0]                          # ranked by count desc
    assert top["dial"] == "concise" and top["count"] == 2
    assert top["last_run_id"] == "run_2"             # the most-recent run driving the concise signal


def test_list_order_filter_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "_PATH", str(tmp_path / "fb.json"))
    for i, r in enumerate(("a", "b", "c")):
        feedback.record(r, "quick_repair", dial="concise", direction=1, _now=100.0 + i)
    assert feedback.list_signals()[0]["run_id"] == "c"          # newest first
    assert [x["run_id"] for x in feedback.list_signals(run_id="b")] == ["b"]
    assert len(feedback.list_signals(limit=2)) == 2


def test_window_filters_old_signals(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "_PATH", str(tmp_path / "fb.json"))
    feedback.record("old", "quick_repair", dial="concise", direction=1, _now=100.0)
    feedback.record("new", "quick_repair", dial="concise", direction=1, _now=100_000.0)
    summ = feedback.summary(window_seconds=86400, _now=100_050.0)   # 1-day window -> only the recent one
    assert summ["total"] == 1


def test_missing_and_corrupt_file_never_raise(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "_PATH", str(tmp_path / "nope.json"))
    assert feedback.list_signals() == [] and feedback.summary()["total"] == 0
    bad = tmp_path / "bad.json"
    bad.write_text("not json{", encoding="utf-8")
    monkeypatch.setattr(feedback, "_PATH", str(bad))
    assert feedback.list_signals() == []                        # corrupt -> [], no raise
    s = feedback.record("r", "quick_repair", dial="warm", direction=1)   # recovers by overwriting
    assert s["dial"] == "warm"


def test_blank_kind_and_bad_direction_coerced(tmp_path, monkeypatch):
    monkeypatch.setattr(feedback, "_PATH", str(tmp_path / "fb.json"))
    s = feedback.record("r", "   ", dial="concise", direction="nope")
    assert s["kind"] == "unknown" and s["direction"] is None     # never drop a signal over a bad field
