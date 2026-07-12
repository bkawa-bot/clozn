"""Tests for the actuarial journal (actuary.py) — pure functions over synthetic run records, no fs/engine."""
from __future__ import annotations

from clozn.runs import actuary


def _run(rid, ts, confs, *, error=None, finish="stop", flags=None, parent=None,
         tiny=None, source="chat", prompt="what is the capital of france", ent=None):
    return {
        "id": rid, "created_ts": ts, "created_at": "t", "source": source,
        "prompt_summary": prompt, "finish_reason": finish, "error": error,
        "flags": flags or [], "parent_run_id": parent, "tiny_tests": tiny or [],
        "trace": {"confidence": confs, **({"entropy": ent} if ent else {})},
    }


# ---- outcome proxy --------------------------------------------------------

def test_is_bad_proxies():
    assert actuary.is_bad(_run("a", 1, [.9], error="boom"))
    assert actuary.is_bad(_run("b", 1, [.9], finish="length"))
    assert actuary.is_bad(_run("c", 1, [.9], flags=["low-confidence"]))
    assert actuary.is_bad(_run("d", 1, [.9], tiny=[{"name": "x", "pass": False}]))
    assert not actuary.is_bad(_run("e", 1, [.9]))


def test_superseded_is_bad():
    runs = [_run("parent", 1, [.9]), _run("child", 2, [.9], parent="parent")]
    sup = actuary._superseded_ids(runs)
    assert actuary.is_bad(runs[0], sup)          # parent was re-rolled
    assert not actuary.is_bad(runs[1], sup)


# ---- calibration ----------------------------------------------------------

def test_calibration_separates_confident_good_from_confident_bad():
    runs = []
    # high-confidence trusted runs
    for i in range(10):
        runs.append(_run(f"g{i}", i, [0.95, 0.9]))
    # high-confidence but BAD runs (errored) -> the bin should show a trusted_rate < 1 and a positive gap
    for i in range(10):
        runs.append(_run(f"b{i}", 100 + i, [0.95, 0.9], error="x"))
    cal = actuary.calibration(runs, n_bins=10)
    assert cal.n_scored == 20
    top = [b for b in cal.bins if b.n and b.hi > 0.85][0]
    assert top.trusted_rate is not None and 0.4 <= top.trusted_rate <= 0.6   # ~half were bad
    assert top.gap > 0.2                                                     # over-confident (proxy)
    assert cal.ece_proxy is not None and cal.ece_proxy > 0.1


def test_calibration_skips_traceless_runs():
    runs = [_run("a", 1, []), _run("b", 2, [0.8])]
    cal = actuary.calibration(runs)
    assert cal.n_runs == 2 and cal.n_scored == 1


# ---- drift ----------------------------------------------------------------

def test_drift_flags_a_confidence_drop_in_a_class():
    runs = []
    for i in range(6):                                    # OLD: high conf
        runs.append(_run(f"o{i}", i, [0.95, 0.92]))
    for i in range(6):                                    # NEW: much lower conf, same prompt-class
        runs.append(_run(f"n{i}", 1000 + i, [0.55, 0.5]))
    alarms = actuary.drift(runs, min_class_n=4, band=0.08)
    assert len(alarms) == 1
    a = alarms[0]
    assert a.delta is not None and a.delta < -0.2
    assert a.severity == "alarm"


def test_drift_ignores_small_classes():
    runs = [_run(f"x{i}", i, [0.9], prompt=f"unique prompt number {i} here") for i in range(6)]
    assert actuary.drift(runs, min_class_n=4) == []       # every class is n=1


def test_drift_no_alarm_when_stable():
    runs = [_run(f"s{i}", i, [0.9, 0.88]) for i in range(12)]
    assert actuary.drift(runs, min_class_n=4, band=0.08) == []


# ---- failure signature ----------------------------------------------------

def test_failure_model_learns_and_scores():
    runs = []
    for i in range(15):                                   # good: high conf, no dips, not truncated
        runs.append(_run(f"g{i}", i, [0.95, 0.93, 0.9]))
    for i in range(15):                                   # bad: low conf + a deep dip + truncated
        runs.append(_run(f"b{i}", 100 + i, [0.3, 0.2, 0.35], finish="length"))
    m = actuary.fit_failure_model(runs)
    assert m.n_good == 15 and m.n_bad == 15
    assert m.weights                                       # learned some separators
    good_like = _run("q", 1, [0.96, 0.94])
    bad_like = _run("r", 1, [0.25, 0.2], finish="length")
    assert actuary.failure_score(good_like, m) < 0.5
    assert actuary.failure_score(bad_like, m) > 0.5


def test_failure_score_untrained_is_neutral():
    m = actuary.FailureModel()
    assert actuary.failure_score(_run("a", 1, [0.5]), m) == 0.5


# ---- top-level ------------------------------------------------------------

def test_organic_filter_drops_machine_traffic():
    runs = [
        _run("u1", 1, [0.9], source="chat"),
        _run("u2", 2, [0.9], source="openai_api"),
        _run("arm", 3, [0.9], source="replay"),
        _run("branch", 4, [0.9], source="branch"),
        _run("child", 5, [0.9], source="chat", parent="u1"),   # derived arm despite chat label
        _run("unknown", 6, [0.9], source="some_new_surface"),  # fail-open: kept
    ]
    org = actuary.organic(runs)
    ids = {r["id"] for r in org}
    assert ids == {"u1", "u2", "unknown"}


def test_analyze_defaults_to_organic():
    runs = [_run(f"u{i}", i, [0.9, 0.85], source="chat") for i in range(20)]
    runs += [_run(f"arm{i}", 100 + i, [0.3], source="replay", error="x") for i in range(40)]
    rep = actuary.analyze(runs)
    assert rep.n_total == 60 and rep.n_organic == 20 and rep.n_runs == 20   # arms excluded
    rep_all = actuary.analyze(runs, organic_only=False)
    assert rep_all.n_runs == 60
    txt = actuary.render(rep)
    assert "organic runs" in txt and "CALIBRATION" in txt and "FAILURE MODEL" in txt


def test_empty_journal_is_safe():
    rep = actuary.analyze([])
    assert rep.n_runs == 0
    assert actuary.render(rep)                              # renders without crashing
