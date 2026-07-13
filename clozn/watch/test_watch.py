"""Tests for the ambient-alert watcher (clozn.watch) -- pure, no server, no OS toast."""
from __future__ import annotations

from clozn.watch import alerts, watcher
from clozn.watch.notify import RecordingNotifier


def _run(rid, confs=None, *, source="chat", parent=None, error=None, finish="stop",
         tiny=None, prompt="what is the capital of france"):
    r = {"id": rid, "source": source, "parent_run_id": parent, "error": error,
         "finish_reason": finish, "prompt_summary": prompt, "tiny_tests": tiny or [], "flags": []}
    if confs is not None:
        r["trace"] = {"tokens": [f"t{i}" for i in range(len(confs))], "confidence": list(confs)}
    return r


# ---- should_alert decisions ----

def test_clean_confident_run_is_silent():
    assert alerts.should_alert(_run("a", [0.95, 0.93, 0.9])) is None


def test_error_is_high():
    al = alerts.should_alert(_run("a", error="boom"))
    assert al and al.severity == "high" and al.reason == "error"


def test_truncated_is_high():
    al = alerts.should_alert(_run("a", [0.9, 0.9], finish="length"))
    assert al and al.reason == "truncated"


def test_failed_tiny_test_is_high():
    al = alerts.should_alert(_run("a", [0.9], tiny=[{"name": "has Paris", "pass": False}]))
    assert al and al.reason == "tiny_test_failed" and "has Paris" in al.headline


def test_low_mean_conf_is_medium():
    al = alerts.should_alert(_run("a", [0.4, 0.45, 0.5, 0.42]))
    assert al and al.severity == "medium" and al.reason == "low_mean_conf"


def test_single_very_uncertain_token_is_medium():
    al = alerts.should_alert(_run("a", [0.95, 0.95, 0.2, 0.95]))   # mean high, one deep dip
    assert al and al.reason == "shaky_span"


def test_machine_traffic_is_skipped():
    assert alerts.should_alert(_run("a", [0.2, 0.2], source="replay")) is None      # a probe
    assert alerts.should_alert(_run("a", [0.2, 0.2], parent="run_parent")) is None  # a derived arm


def test_no_trace_no_confidence_alarm():
    # no trace at all -> no confidence signal; only the hard signals (error/truncate) can fire
    assert alerts.should_alert(_run("a", None)) is None
    assert alerts.should_alert(_run("a", None, error="x")).reason == "error"


def test_junk_never_raises():
    for junk in (None, {}, {"id": None}, {"id": "x", "trace": "nope"}):
        assert alerts.should_alert(junk) is None


# ---- the watcher (fake client + recording notifier) ----

class FakeClient:
    def __init__(self, summaries, records):
        self.summaries = summaries          # list of run summary dicts (newest first, like GET /runs)
        self.records = records              # {id: full record}
        self.fetches = []

    def list_runs(self, limit=50):
        return list(self.summaries)

    def get_run(self, rid):
        self.fetches.append(rid)
        return self.records.get(rid)


def test_prime_baselines_history_without_alerting():
    recs = {"r1": _run("r1", [0.2]), "r2": _run("r2", [0.2])}
    c = FakeClient([{"id": "r2"}, {"id": "r1"}], recs)
    st = watcher.WatchState()
    n = watcher.prime(c, st)
    assert n == 2 and st.seen == {"r1", "r2"}
    # a poll right after prime fires nothing (all already seen)
    assert watcher.run_once(c, RecordingNotifier(), st, "http://h") == []


def test_new_sketchy_run_fires_once_with_link():
    recs = {"r1": _run("r1", [0.9])}
    c = FakeClient([{"id": "r1"}], recs)
    st = watcher.WatchState(); watcher.prime(c, st)                  # baseline r1
    # a new shaky run lands
    recs["r2"] = _run("r2", [0.2, 0.3, 0.25], prompt="tricky q")
    c.summaries = [{"id": "r2", "source": "chat"}, {"id": "r1"}]
    note = RecordingNotifier()
    fired = watcher.run_once(c, note, st, "http://h:8090")
    assert len(fired) == 1 and fired[0].run_id == "r2"
    assert len(note.sent) == 1
    _title, _body, url = note.sent[0]
    assert url == "http://h:8090/r/r2"
    # a second poll does NOT re-alert the same run
    assert watcher.run_once(c, note, st, "http://h:8090") == []


def test_machine_run_skipped_without_a_full_fetch():
    c = FakeClient([{"id": "r9", "source": "replay"}], {})
    st = watcher.WatchState()
    watcher.run_once(c, RecordingNotifier(), st, "http://h")
    assert c.fetches == []                                           # never fetched the replay arm's record
    assert "r9" in st.seen                                           # but marked seen so it's not reconsidered
