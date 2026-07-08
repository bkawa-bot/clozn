"""Schema-level, no-model tests for research/runlog.py (roadmap issue I3).

Covers record -> list_runs -> get_run, asserts the run schema fields exist, that the cheap UI flags are
computed (a run carrying a memory card gets 'memory'; a low-confidence trace gets 'low-confidence'; etc.),
and that pruning keeps <= KEEP runs. The store is isolated by pointing runlog.RUNS_DIR at a pytest tmp dir
at runtime -- RUNS_DIR is a module global, so we don't need to (and per I3 must not) edit runlog.py.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # research/ on path
import runlog  # noqa: E402


@pytest.fixture
def store(tmp_path):
    """Redirect the run store to a temp dir for the duration of one test."""
    original = runlog.RUNS_DIR
    runlog.RUNS_DIR = str(tmp_path / "runs")
    try:
        yield runlog
    finally:
        runlog.RUNS_DIR = original


def test_record_returns_id_and_persists(store):
    rid = store.record(source="cli", client="clozn-cli", model="qwen", substrate="QwenSubstrate",
                       messages=[{"role": "user", "content": "hi there"}], response="hello")
    assert rid is not None
    assert rid.startswith("run_")
    rec = store.get_run(rid)
    assert rec is not None
    assert rec["id"] == rid


def test_record_schema_fields(store):
    rid = store.record(source="studio_chat",
                       messages=[{"role": "user", "content": "what is 2+2?"}], response="4")
    rec = store.get_run(rid)
    for k in ("id", "created_at", "created_ts", "source", "client", "model", "substrate",
              "prompt_summary", "response_summary", "messages", "response", "memory", "behavior",
              "assembled_messages", "trace", "timing", "parent_run_id", "changes_applied", "error", "flags"):
        assert k in rec, f"missing schema field {k}"
    assert rec["source"] == "studio_chat"
    assert rec["prompt_summary"] == "what is 2+2?"        # last user message summarized
    assert rec["response_summary"] == "4"
    assert set(("started_at", "ended_at", "duration_ms")).issubset(rec["timing"])


def test_record_persists_assembled_messages_when_provided(store):
    assembled = [{"role": "system", "content": "MEMORY BLOCK"},
                 {"role": "user", "content": "what is 2+2?"}]
    rid = store.record(source="studio_chat",
                       messages=[{"role": "user", "content": "what is 2+2?"}],
                       response="4", assembled_messages=assembled)
    assert store.get_run(rid)["assembled_messages"] == assembled


def test_prompt_summary_uses_last_user_message(store):
    rid = store.record(source="cli", messages=[
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ], response="ok")
    assert store.get_run(rid)["prompt_summary"] == "second"


def test_flag_memory(store):
    rid = store.record(source="studio_chat", messages=[{"role": "user", "content": "q"}],
                       response="a", memory={"cards_applied": ["mem_1"]})
    assert "memory" in store.get_run(rid)["flags"]


def test_flag_pending_memory(store):
    rid = store.record(source="studio_chat", messages=[{"role": "user", "content": "q"}],
                       response="a", memory={"proposed_cards": ["mem_2"]})
    assert "pending-memory" in store.get_run(rid)["flags"]


def test_flag_steered(store):
    rid = store.record(source="studio_chat", messages=[{"role": "user", "content": "q"}],
                       response="a", behavior={"active_dials": {"concise": 0.4}})
    assert "steered" in store.get_run(rid)["flags"]


def test_flag_low_confidence(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a",
                       trace={"tokens": ["x", "y"], "confidence": [0.9, 0.12]})
    assert "low-confidence" in store.get_run(rid)["flags"]


def test_flag_replayed_and_error(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a",
                       parent_run_id="run_parent", error="boom")
    flags = store.get_run(rid)["flags"]
    assert "replayed" in flags
    assert "error" in flags


def test_no_spurious_flags(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="short answer",
                       trace={"tokens": ["a"], "confidence": [0.95]})
    assert store.get_run(rid)["flags"] == []


def test_list_runs_newest_first(store):
    # ids embed a ms timestamp; pass increasing `started` so ordering is deterministic
    r1 = store.record(source="cli", messages=[{"role": "user", "content": "one"}], response="1",
                      started=1000.0, ended=1000.1)
    r2 = store.record(source="cli", messages=[{"role": "user", "content": "two"}], response="2",
                      started=2000.0, ended=2000.1)
    r3 = store.record(source="cli", messages=[{"role": "user", "content": "three"}], response="3",
                      started=3000.0, ended=3000.1)
    ids = [r["id"] for r in store.list_runs()]
    assert ids == [r3, r2, r1]                            # newest first


def test_list_runs_returns_summary_fields_only(store):
    store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="yo")
    rows = store.list_runs()
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) == set(runlog.SUMMARY_FIELDS)
    # the heavy fields are intentionally NOT in the summary
    assert "messages" not in row
    assert "trace" not in row


def test_list_runs_limit(store):
    for i in range(5):
        store.record(source="cli", messages=[{"role": "user", "content": str(i)}], response=str(i),
                     started=1000.0 + i, ended=1000.0 + i)
    assert len(store.list_runs(limit=3)) == 3


def test_get_run_missing(store):
    assert store.get_run("run_does_not_exist") is None


def test_pruning_keeps_at_most_KEEP(store, monkeypatch):
    monkeypatch.setattr(runlog, "KEEP", 3)               # shrink the cap so the test is cheap
    ids = []
    for i in range(6):
        ids.append(store.record(source="cli", messages=[{"role": "user", "content": str(i)}],
                                response=str(i), started=1000.0 + i, ended=1000.0 + i))
    remaining = {r["id"] for r in store.list_runs(limit=100)}
    assert len(remaining) <= 3
    # the 3 most recent survived; the oldest were pruned
    assert remaining == set(ids[-3:])
    assert store.get_run(ids[0]) is None
