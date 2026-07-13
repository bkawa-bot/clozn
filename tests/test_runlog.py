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
import clozn.runs.store as runlog  # noqa: E402


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
              "assembled_messages", "final_prompt", "trace", "timing", "parent_run_id",
              "changes_applied", "error", "flags"):
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


def test_record_persists_final_prompt_when_provided(store):
    """backlog #5: the EXACT rendered chat-template string the model saw is stored on the run record."""
    rendered = ("<|im_start|>system\nMEMORY BLOCK<|im_end|>\n"
                "<|im_start|>user\nwhat is 2+2?<|im_end|>\n<|im_start|>assistant\n")
    rid = store.record(source="engine_chat",
                       messages=[{"role": "user", "content": "what is 2+2?"}],
                       response="4", final_prompt=rendered)
    assert store.get_run(rid)["final_prompt"] == rendered


def test_record_final_prompt_defaults_to_none(store):
    """No rendered string in hand (e.g. a torch substrate) -> the field is present but None; consumers
    then fall back to assembled_messages. Present-but-None, never a KeyError."""
    rid = store.record(source="studio_chat",
                       messages=[{"role": "user", "content": "hi"}], response="hey")
    rec = store.get_run(rid)
    assert "final_prompt" in rec
    assert rec["final_prompt"] is None


def test_legacy_run_without_final_prompt_loads_fine(store):
    """A run persisted BEFORE this field existed must still load; a reader uses run.get('final_prompt')."""
    import json
    os.makedirs(store.RUNS_DIR, exist_ok=True)
    legacy = {"id": "run_legacy_000000", "source": "engine_chat",
              "messages": [{"role": "user", "content": "hi"}], "response": "hey",
              "assembled_messages": [{"role": "user", "content": "hi"}]}   # NOTE: no "final_prompt" key
    with open(os.path.join(store.RUNS_DIR, "run_legacy_000000.json"), "w", encoding="utf-8") as f:
        json.dump(legacy, f)
    rec = store.get_run("run_legacy_000000")
    assert rec is not None                                  # loads without crashing
    assert rec.get("final_prompt") is None                 # absent -> None via .get(), the documented fallback
    assert rec["assembled_messages"] == [{"role": "user", "content": "hi"}]


def test_log_run_forwards_final_prompt_to_the_record(store, monkeypatch):
    """The handler glue (backlog #5): _log_run reads mem_out['final_prompt'] and persists it as
    run.final_prompt. Drives the REAL do_POST handler object with no socket + SUB=None (no engine, no
    model) -- purely the forwarding logic, mirroring test_rederive_server.py's object.__new__(H) trick."""
    import time
    from clozn.server import app as cs
    monkeypatch.setattr(cs, "SUB", None)                   # no substrate -> dials {}, run_meta skipped
    h = object.__new__(cs.make_handler())
    h.headers = {"User-Agent": "pytest"}
    rendered = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    rid = h._log_run("engine_chat", [{"role": "user", "content": "hi"}], "hey", "clozn-engine", time.time(),
                     mem_out={"mode": "prompt", "applied": [], "gate": 0.0,
                              "assembled_messages": [{"role": "user", "content": "hi"}],
                              "final_prompt": rendered})
    assert rid is not None
    assert store.get_run(rid)["final_prompt"] == rendered


def test_log_run_persists_anchored_memory_manifest(store, monkeypatch):
    """Anchored bags are logged as memory that rode the turn even when no prompt card block was injected."""
    import time
    from clozn.server import app as cs
    monkeypatch.setattr(cs, "SUB", None)
    h = object.__new__(cs.make_handler())
    h.headers = {"User-Agent": "pytest"}

    rid = h._log_run("openai_api", [{"role": "user", "content": "tea?"}], "tea", "clozn-engine",
                     time.time(),
                     mem_out={"mode": "prompt", "applied": [], "gate": 0.0,
                              "anchored": [{"card_id": "mem_tea", "gate": 0.5,
                                            "alpha_top3": [{"token": "tea", "alpha": 0.7}]}],
                              "anchored_layer": 21, "anchored_s_total": 0.25})

    rec = store.get_run(rid)
    assert rec["memory"]["anchored"][0]["card_id"] == "mem_tea"
    assert rec["memory"]["anchored_layer"] == 21
    assert rec["memory"]["anchored_s_total"] == pytest.approx(0.25)
    assert "memory" in rec["flags"]
    assert "anchored-memory" in rec["flags"]


def test_log_run_persists_anchored_loop_guard_retried(store, monkeypatch):
    """The loop guard's honest self-healing record (X7_PRODUCT_DESIGN.md section 5) rides the run record
    exactly like anchored/anchored_layer/anchored_s_total do -- and turns into the visible "memory-retried"
    run flag."""
    import time
    from clozn.server import app as cs
    monkeypatch.setattr(cs, "SUB", None)
    h = object.__new__(cs.make_handler())
    h.headers = {"User-Agent": "pytest"}

    rid = h._log_run("openai_api", [{"role": "user", "content": "tea?"}], "a clean retried reply",
                     "clozn-engine", time.time(),
                     mem_out={"mode": "prompt", "applied": [], "gate": 0.0,
                              "anchored": [{"card_id": "mem_tea", "gate": 1.0,
                                            "alpha_top3": [{"token": "tea", "alpha": 0.7}]}],
                              "anchored_layer": 21, "anchored_s_total": 0.25,
                              "anchored_loop_guard": {"fired": True, "action": "retried@s/2",
                                                      "resolved": True}})

    rec = store.get_run(rid)
    assert rec["memory"]["anchored_loop_guard"] == {"fired": True, "action": "retried@s/2",
                                                     "resolved": True}
    assert rec["memory"]["anchored_s_total"] == pytest.approx(0.25)          # the HALVED value, honestly
    assert "memory-retried" in rec["flags"]
    assert "memory-loop-guard" not in rec["flags"]
    assert "anchored-memory" in rec["flags"]           # still true: anchored memory DID ride this turn


def test_log_run_persists_anchored_loop_guard_disabled(store, monkeypatch):
    """Still looped at half strength -> the substrate zeroed the anchored steer entirely; the run is
    flagged "memory-loop-guard", never "memory-retried" (the retry did not resolve it)."""
    import time
    from clozn.server import app as cs
    monkeypatch.setattr(cs, "SUB", None)
    h = object.__new__(cs.make_handler())
    h.headers = {"User-Agent": "pytest"}

    rid = h._log_run("openai_api", [{"role": "user", "content": "tea?"}], "finally clean",
                     "clozn-engine", time.time(),
                     mem_out={"mode": "prompt", "applied": [], "gate": 0.0,
                              "anchored": [{"card_id": "mem_tea", "gate": 1.0, "alpha_top3": []}],
                              "anchored_layer": 21, "anchored_s_total": 0.0,
                              "anchored_loop_guard": {"fired": True, "action": "disabled",
                                                      "resolved": True}})

    rec = store.get_run(rid)
    assert rec["memory"]["anchored_loop_guard"]["action"] == "disabled"
    assert "memory-loop-guard" in rec["flags"]
    assert "memory-retried" not in rec["flags"]


def test_flag_memory_retried(store):
    rid = store.record(source="openai_api", messages=[{"role": "user", "content": "q"}], response="a",
                       memory={"anchored": [{"card_id": "c"}],
                               "anchored_loop_guard": {"fired": True, "action": "retried@s/2",
                                                       "resolved": True}})
    flags = store.get_run(rid)["flags"]
    assert "memory-retried" in flags
    assert "memory-loop-guard" not in flags


def test_flag_memory_loop_guard(store):
    rid = store.record(source="openai_api", messages=[{"role": "user", "content": "q"}], response="a",
                       memory={"anchored": [{"card_id": "c"}],
                               "anchored_loop_guard": {"fired": True, "action": "disabled",
                                                       "resolved": True}})
    flags = store.get_run(rid)["flags"]
    assert "memory-loop-guard" in flags
    assert "memory-retried" not in flags


def test_flag_memory_loop_guard_streaming_flagged_only(store):
    """The streaming twin's detect-and-flag-only outcome (action != "retried@s/2") also reads as
    "memory-loop-guard" -- the flag names the OUTCOME (was it cleanly retried, or not), not the mechanism."""
    rid = store.record(source="openai_api", messages=[{"role": "user", "content": "q"}], response="a",
                       memory={"anchored": [{"card_id": "c"}],
                               "anchored_loop_guard": {"fired": True, "action": "flagged-only",
                                                       "resolved": False}})
    assert "memory-loop-guard" in store.get_run(rid)["flags"]


def test_no_loop_guard_flag_when_the_guard_never_fired(store):
    rid = store.record(source="openai_api", messages=[{"role": "user", "content": "q"}], response="a",
                       memory={"cards_applied": ["x"]})
    flags = store.get_run(rid)["flags"]
    assert "memory-retried" not in flags
    assert "memory-loop-guard" not in flags


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


# ------------------------------------------------------------------ include_replays (receipt-journal spam)
# A `/runs/<id>/receipts` "prove-all" persists one child run per arm (baseline + one per fired influence +
# one per redundancy-guard pair -- clozn.replay.replay.replay(), source="replay") so each is itself an
# inspectable, diffable run. Real usage surfaced this as noise: one "prove-all" click produced ~7
# near-duplicate entries in a run-history listing. include_replays=False lets a browsing view opt out
# without losing the underlying data (still fully readable via get_run()).

def test_list_runs_include_replays_true_by_default(store):
    store.record(source="cli", messages=[{"role": "user", "content": "real"}], response="a", started=1000.0)
    store.record(source="replay", messages=[{"role": "user", "content": "real"}], response="b", started=2000.0)
    assert len(store.list_runs()) == 2


def test_list_runs_include_replays_false_drops_replay_sourced_entries(store):
    real = store.record(source="cli", messages=[{"role": "user", "content": "real"}], response="a",
                        started=1000.0)
    store.record(source="replay", messages=[{"role": "user", "content": "real"}], response="b",
                 started=2000.0, parent_run_id=real)
    rows = store.list_runs(include_replays=False)
    assert [r["id"] for r in rows] == [real]


def test_list_runs_include_replays_false_still_honors_limit(store):
    for i in range(5):
        store.record(source="cli", messages=[{"role": "user", "content": str(i)}], response=str(i),
                     started=1000.0 + i, ended=1000.0 + i)
        store.record(source="replay", messages=[{"role": "user", "content": str(i)}], response=str(i),
                     started=1000.5 + i, ended=1000.5 + i)
    rows = store.list_runs(limit=3, include_replays=False)
    assert len(rows) == 3
    assert all(r.get("source") != "replay" for r in
              [store.get_run(r["id"]) for r in rows])


def test_list_runs_include_replays_false_replay_still_readable_by_id(store):
    """The filter hides replay children from a listing view -- it must never make them unreachable."""
    real = store.record(source="cli", messages=[{"role": "user", "content": "real"}], response="a",
                        started=1000.0)
    replay_id = store.record(source="replay", messages=[{"role": "user", "content": "real"}], response="b",
                             started=2000.0, parent_run_id=real)
    store.list_runs(include_replays=False)   # must not affect what's on disk
    assert store.get_run(replay_id) is not None
    assert store.get_run(replay_id)["source"] == "replay"


def test_get_run_missing(store):
    assert store.get_run("run_does_not_exist") is None


# ---------------------------------------------------------------------------------- path traversal (security)
# GET /runs/<rid>/... hands `rid` to get_run() straight from the URL path -- a hostile `rid` like
# "../config" or an absolute path must never let a read escape RUNS_DIR onto some other readable .json
# (e.g. ~/.clozn/config.json). None of these attempts should raise either -- get_run() never raises,
# an unsafe id is just treated like "not found".

def test_get_run_rejects_dotdot_traversal(store, tmp_path):
    secret = tmp_path / "config.json"
    secret.write_text('{"api_key": "super-secret"}', encoding="utf-8")
    os.makedirs(store.RUNS_DIR, exist_ok=True)
    assert store.get_run("../config") is None
    assert store.get_run("..\\config") is None


def test_get_run_rejects_absolute_path(store, tmp_path):
    secret = tmp_path / "config.json"
    secret.write_text('{"api_key": "super-secret"}', encoding="utf-8")
    abs_no_ext = str(secret)[:-len(".json")]           # get_run appends ".json" itself
    assert store.get_run(abs_no_ext) is None
    assert store.get_run("/etc/passwd") is None


def test_get_run_rejects_non_string_or_empty_id(store):
    """The degenerate-run-record case (#3): a `{}` on disk summarizes to id=None; get_run(None) must
    return None cleanly, never raise (the old code did `None + ".json"` -> TypeError)."""
    assert store.get_run(None) is None
    assert store.get_run("") is None
    assert store.get_run(123) is None


def test_update_tiny_tests_rejects_traversal_and_writes_nothing_outside_runs_dir(store, tmp_path):
    """`clozn test --attach` reaches attachments.update_tiny_tests(rid, ...) with a CLI-supplied rid --
    same guard, same contract (False on a bad id), and this is the WRITE side: confirm nothing lands
    outside RUNS_DIR at all, not just that the return value is honest."""
    target = tmp_path / "config.json"
    target.write_text('{"api_key": "super-secret"}', encoding="utf-8")
    os.makedirs(store.RUNS_DIR, exist_ok=True)

    assert store.update_tiny_tests("../config", [{"a": 1}]) is False
    assert store.update_tiny_tests("..\\config", [{"a": 1}]) is False
    assert target.read_text(encoding="utf-8") == '{"api_key": "super-secret"}'   # untouched
    assert not os.path.isfile(os.path.join(store.RUNS_DIR, "config.json"))       # nothing spilled inside either


# -------------------------------------------------------------- round-2 pressure test #1 (HIGH): atomic writes
# record() and update_tiny_tests() are the other two user-data JSON writers sharing the same defect memory
# cards / settings had: open(path, "w") then json.dump(obj, f) truncates the target BEFORE a non-
# serializable value can raise. Both already swallow exceptions and return None/False (never raise out to
# the caller) -- the fix is that a bad write must never destroy/corrupt an existing run record, and must
# never leave a stray truncated file behind for a run that never really completed.

def test_record_bad_meta_fails_cleanly_and_leaves_other_runs_untouched(store):
    good_rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="ok")
    assert store.get_run(good_rid) is not None
    before = set(os.listdir(store.RUNS_DIR))

    bad_rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="ok",
                           meta={"bad": {1, 2, 3}})          # a set -- json can't serialize it
    assert bad_rid is None                                    # documented contract: None on failure

    after = set(os.listdir(store.RUNS_DIR))
    assert after == before                                    # no stray/truncated file left behind
    assert store.get_run(good_rid)["response"] == "ok"        # the earlier good run is untouched


def test_update_tiny_tests_bad_payload_fails_cleanly_and_leaves_the_run_record_untouched(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="ok")
    assert store.update_tiny_tests(rid, [{"name": "t1", "passed": True}]) is True
    assert store.get_run(rid)["tiny_tests"] == [{"name": "t1", "passed": True}]

    # a nested set inside the list -- update_tiny_tests coerces the outer value with list(), but a
    # non-serializable value nested inside an entry still reaches json.dumps
    ok = store.update_tiny_tests(rid, [{"name": "t2", "detail": {1, 2, 3}}])
    assert ok is False

    assert store.get_run(rid)["tiny_tests"] == [{"name": "t1", "passed": True}]   # prior attachment survives


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
