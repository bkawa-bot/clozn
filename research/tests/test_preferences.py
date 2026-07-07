"""Model-free tests for preferences.py -- the propose-and-review consumer. A temp file stands in for
~/.clozn/preferences.json; signals are plain dicts (no feedback store, no server, no model)."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import preferences  # noqa: E402


def _sigs(dial, direction, n, start=0):
    return [{"run_id": f"r{start + i}", "dial": dial, "direction": direction, "ts": 100.0 + i}
            for i in range(n)]


def test_proposes_only_at_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(preferences, "_PATH", str(tmp_path / "pref.json"))
    assert preferences.refresh(_sigs("concise", 1, 2), threshold=3, _now=1.0) == []   # below -> nothing
    pend = preferences.refresh(_sigs("concise", 1, 3), threshold=3, lean=0.5, _now=2.0)
    assert len(pend) == 1
    p = pend[0]
    assert p["dial"] == "concise" and p["direction"] == 1 and p["suggested_value"] == 0.5
    assert p["status"] == "pending" and p["count"] == 3 and "concise" in p["label"]
    assert p["evidence"] == ["r0", "r1", "r2"]        # the runs that drove it -- the proposal's receipts


def test_pending_refreshes_not_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(preferences, "_PATH", str(tmp_path / "p.json"))
    preferences.refresh(_sigs("concise", 1, 3), threshold=3, _now=1.0)
    pend = preferences.refresh(_sigs("concise", 1, 5), threshold=3, _now=2.0)
    assert len(pend) == 1 and pend[0]["count"] == 5    # same proposal, updated count -- never a duplicate
    assert len(preferences.list_proposals()) == 1


def test_approve_marks_done_and_never_reproposes(tmp_path, monkeypatch):
    monkeypatch.setattr(preferences, "_PATH", str(tmp_path / "p.json"))
    pend = preferences.refresh(_sigs("warm", 1, 3), threshold=3, _now=1.0)
    pr = preferences.resolve(pend[0]["id"], "approve", _now=2.0)
    assert pr["status"] == "approved" and pr["resolved_ts"] == 2.0
    assert preferences.list_proposals(status="pending") == []
    assert preferences.refresh(_sigs("warm", 1, 9), threshold=3, _now=3.0) == []   # applied -> stays quiet


def test_dismiss_sticks_until_a_fresh_burst(tmp_path, monkeypatch):
    monkeypatch.setattr(preferences, "_PATH", str(tmp_path / "p.json"))
    pend = preferences.refresh(_sigs("concise", 1, 3), threshold=3, _now=1.0)
    preferences.resolve(pend[0]["id"], "dismiss", _now=2.0)
    assert preferences.list_proposals(status="dismissed")[0]["dismissed_at_count"] == 3
    assert preferences.refresh(_sigs("concise", 1, 5), threshold=3, _now=3.0) == []   # 5 < 3+3 -> respected
    pend2 = preferences.refresh(_sigs("concise", 1, 6), threshold=3, _now=4.0)        # 6 >= 6 -> resurfaces
    assert len(pend2) == 1 and pend2[0]["status"] == "pending"


def test_resolve_unknown_or_bad_action_is_none(tmp_path, monkeypatch):
    monkeypatch.setattr(preferences, "_PATH", str(tmp_path / "p.json"))
    assert preferences.resolve("nope", "approve") is None
    pend = preferences.refresh(_sigs("warm", 1, 3), threshold=3, _now=1.0)
    assert preferences.resolve(pend[0]["id"], "banana") is None    # bad action -> no-op, no raise
