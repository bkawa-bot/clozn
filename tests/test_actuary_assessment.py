"""Past-only failure assessment and its journal route; model-free, GPU-free."""
from types import SimpleNamespace

from clozn.runs import actuary
from clozn.server.routes import journal


def _run(rid, ts, confs, *, bad=False, source="chat", parent=None, ent=None):
    return {
        "id": rid,
        "created_ts": ts,
        "source": source,
        "parent_run_id": parent,
        "finish_reason": "stop",
        "flags": ["low-confidence"] if bad else [],
        "trace": {"confidence": confs, "entropy": ent or [0.2] * len(confs)},
    }


def _history():
    good = [_run(f"g{i}", i, [.92, .88], ent=[.15, .2]) for i in range(6)]
    bad = [_run(f"b{i}", 20 + i, [.24, .18], bad=True, ent=[2.1, 1.9]) for i in range(6)]
    return good + bad


def test_bad_shaped_run_warns_against_sufficient_past_only_evidence():
    current = _run("current", 100, [.21, .19], ent=[2.0, 2.2])
    future = _run("future", 200, [.2, .2], bad=True)
    derived = _run("derived", 50, [.2], bad=True, source="replay", parent="b0")
    out = actuary.assess_failure(current, _history() + [current, future, derived])

    assert out["available"] is True
    assert out["warning_eligible"] is True
    assert out["warning"] is True
    assert out["score"] >= out["threshold"] == 0.65
    assert out["n_good"] == out["n_bad"] == 6
    assert out["n_past_organic"] == 12
    assert out["temporal_cutoff"] is True
    assert out["drivers"] and out["drivers"][0]["feature"]
    assert "NOT a correctness predictor" in out["note"]


def test_good_shaped_run_is_not_flagged_by_same_model():
    out = actuary.assess_failure(_run("current", 100, [.94, .9], ent=[.1, .2]), _history())
    assert out["available"] and out["warning_eligible"]
    assert out["warning"] is False
    assert out["score"] < out["threshold"]


def test_small_reference_set_is_visible_but_cannot_warn():
    history = _history()[:2] + _history()[6:8]
    out = actuary.assess_failure(_run("current", 100, [.2, .2]), history)
    assert out["available"] is True
    assert out["weak_evidence"] is True
    assert out["warning_eligible"] is False
    assert out["warning"] is False


def test_untrained_reference_set_reports_absence_instead_of_ambiguous_half():
    out = actuary.assess_failure(_run("current", 100, [.2]), [_run("only-good", 1, [.9])])
    assert out["available"] is False
    assert out["score"] is None
    assert out["warning"] is False


class _Handler:
    def __init__(self):
        self.status = None
        self.body = None

    def _json(self, status, body):
        self.status, self.body = status, body


def test_run_actuary_route_scores_the_record(monkeypatch):
    current = _run("current", 100, [.2, .2], ent=[2.0, 2.1])
    monkeypatch.setattr("clozn.runs.store.get_run", lambda rid: current if rid == "current" else None)
    monkeypatch.setattr(actuary, "load_runs", lambda: _history() + [current])
    h = _Handler()

    assert journal.try_post(h, "/runs/current/actuary", {}) is True
    assert h.status == 200
    assert h.body["run_id"] == "current"
    assert h.body["warning"] is True


def test_run_actuary_route_404s_for_missing_record(monkeypatch):
    monkeypatch.setattr("clozn.runs.store.get_run", lambda rid: None)
    h = _Handler()
    assert journal.try_post(h, "/runs/missing/actuary", {}) is True
    assert h.status == 404 and h.body == {"error": "run not found"}
