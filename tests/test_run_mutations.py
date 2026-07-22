"""Focused transactional run-journal privacy and retention tests."""
from __future__ import annotations

from contextlib import closing
import json
import os

import pytest

from clozn.runs import mutations
from clozn.runs import store


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _record(*, prompt="private prompt", response="private response", trace=None, started=None):
    return store.record(
        source="openai_api", client="private-client-label",
        client_key="client_opaque", session_key="session_opaque", project_key="project_opaque",
        model="model", substrate="engine",
        messages=[{"role": "user", "content": prompt}],
        assembled_messages=[{"role": "system", "content": "private memory"},
                            {"role": "user", "content": prompt}],
        final_prompt="rendered private prompt", response=response,
        reasoning={"blocks": [{"text": "private reasoning"}]},
        memory={"cards_applied": ["private preference"]},
        behavior={"tool": {"arguments": "private tool argument"}},
        trace=trace or {"tokens": ["private", " token"], "confidence": [0.9, 0.8]},
        meta={"max_tokens": 32, "private_extension": "private meta"},
        identity={"model_path": "C:/Users/private-name/models/model.gguf",
                  "model_sha256": "a" * 64},
        output_contract={"raw_model_output": "private tool output",
                         "request": {"tools": [{"description": "private tool"}]}},
        started=started, ended=started,
    )


def _raw(run_id):
    with closing(store._connect()) as db:
        row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return row, json.loads(row["payload_json"])


def test_redact_replaces_content_with_tombstone_and_cleans_trace(isolated):
    run_id = _record()
    source = store.get_run(run_id)
    source["influence_map"] = {
        "schema": "clozn.context_answer_influence.v1",
        "prompt_spans": [{"text": "private influence context"}],
        "answer_spans": [{"text": "private influence answer"}],
    }
    assert store.replace_run(source)
    _, raw = _raw(run_id)
    digest = raw["trace_ref"]["sha256"]
    path = store._blob_path(digest)
    # The influence map is potentially-large derived evidence and goes through the SAME blob
    # machinery as trace (store._pack), keyed by influence_map_ref -- it must never be lost
    # (regular GC path) AND must be scrubbed immediately on redaction (privacy path), exactly like trace.
    influence_digest = raw["influence_map_ref"]["sha256"]
    influence_path = store._blob_path(influence_digest)
    assert os.path.isfile(influence_path)
    with open(influence_path, encoding="utf-8") as handle:
        assert "private influence" in handle.read()

    result = mutations.redact_run(run_id)

    assert result["trace_cleanup"] == {"status": "deleted", "sha256": digest}
    assert not os.path.exists(path)
    assert result["influence_map_cleanup"] == {"status": "deleted", "sha256": influence_digest}
    assert not os.path.exists(influence_path)
    redacted = store.get_run(run_id)
    assert redacted["redaction"]["schema"] == mutations.REDACTION_SCHEMA
    assert redacted["redaction"]["influence_evidence_removed"] is True
    assert redacted["flags"] == ["redacted"]
    assert redacted["messages"] == [] and redacted["response"] == ""
    assert redacted["reasoning"] == {} and redacted["output_contract"] == {}
    assert redacted["trace"] == {}
    assert "model_path" not in redacted["identity"]
    assert "influence_map" not in redacted
    encoded = json.dumps(redacted)
    for secret in ("private prompt", "private response", "private reasoning", "private tool",
                   "private memory", "private preference", "client_opaque", "session_opaque",
                   "project_opaque", "private meta", "private-name"):
        assert secret not in encoded
    assert "private influence" not in encoded
    row, _ = _raw(run_id)
    assert row["client_key"] is None and row["session_key"] is None
    assert row["prompt_summary"] == "" and row["response_summary"] == ""


def test_redaction_is_idempotent(isolated):
    run_id = _record()
    first = mutations.redact_run(run_id)
    second = mutations.redact_run(run_id)
    assert first["redaction"] == second["redaction"]
    assert second["already_redacted"] is True
    assert second["trace_cleanup"]["status"] == "not_applicable"


def test_shared_trace_survives_until_last_reference_is_removed(isolated):
    trace = {"tokens": ["same"], "confidence": [0.5]}
    first = _record(prompt="one", trace=trace, started=1.0)
    second = _record(prompt="two", trace=trace, started=2.0)
    _, raw = _raw(first)
    path = store._blob_path(raw["trace_ref"]["sha256"])

    assert mutations.redact_run(first)["trace_cleanup"]["status"] == "retained_shared"
    assert os.path.isfile(path)
    assert mutations.delete_run(second)["trace_cleanup"]["status"] == "deleted"
    assert not os.path.exists(path)


def test_delete_truly_removes_row_and_unique_trace(isolated):
    run_id = _record()
    _, raw = _raw(run_id)
    path = store._blob_path(raw["trace_ref"]["sha256"])
    assert mutations.delete_run(run_id)["ok"] is True
    assert store.get_run(run_id) is None
    assert run_id not in {row["id"] for row in store.list_runs()}
    assert not os.path.exists(path)


def test_invalid_and_missing_exact_ids_are_clean(isolated):
    with pytest.raises(mutations.MutationError, match="exact valid run ID"):
        mutations.redact_run("../escape")
    with pytest.raises(mutations.MutationError, match="exact valid run ID"):
        mutations.delete_run("")
    with pytest.raises(mutations.MutationError, match="not found"):
        mutations.redact_run("run_missing")
    with pytest.raises(mutations.MutationError, match="not found"):
        mutations.delete_run("run_missing")


def test_redaction_database_failure_rolls_back_and_keeps_blob(isolated):
    run_id = _record()
    before = store.get_run(run_id)
    _, raw = _raw(run_id)
    path = store._blob_path(raw["trace_ref"]["sha256"])
    with closing(store._connect()) as db, db:
        db.execute("CREATE TRIGGER reject_redaction BEFORE UPDATE ON runs "
                   "BEGIN SELECT RAISE(ABORT, 'blocked'); END")
    with pytest.raises(mutations.MutationError, match="blocked"):
        mutations.redact_run(run_id)
    assert store.get_run(run_id) == before
    assert os.path.isfile(path)


def test_delete_database_failure_rolls_back_and_keeps_blob(isolated):
    run_id = _record()
    _, raw = _raw(run_id)
    path = store._blob_path(raw["trace_ref"]["sha256"])
    with closing(store._connect()) as db, db:
        db.execute("CREATE TRIGGER reject_delete BEFORE DELETE ON runs "
                   "BEGIN SELECT RAISE(ABORT, 'blocked-delete'); END")
    with pytest.raises(mutations.MutationError, match="blocked-delete"):
        mutations.delete_run(run_id)
    assert store.get_run(run_id) is not None
    assert os.path.isfile(path)


def test_prune_dry_run_plans_exact_oldest_rows_without_mutation(isolated):
    ids = [_record(prompt=str(index), started=float(index),
                   trace={"tokens": [str(index)], "confidence": [0.5]})
           for index in range(1, 5)]
    result = mutations.prune_to(2)
    assert result["dry_run"] is True
    assert result["delete_count"] == 2 and result["deleted_count"] == 0
    assert [entry["run_id"] for entry in result["runs"]] == ids[:2]
    assert {run["id"] for run in store.iter_runs()} == set(ids)
    assert len(result["orphaned_trace_digests"]) == 2


def test_prune_live_removes_rows_but_defers_blob_cleanup(isolated):
    ids = [_record(prompt=str(index), started=float(index),
                   trace={"tokens": [str(index)], "confidence": [0.5]})
           for index in range(1, 4)]
    result = mutations.prune_to(1, dry_run=False)
    assert result["deleted_count"] == 2
    assert [entry["run_id"] for entry in result["runs"]] == ids[:2]
    assert [run["id"] for run in store.iter_runs()] == [ids[2]]
    assert result["blob_cleanup"] == "deferred_to_gc"
    for digest in result["orphaned_trace_digests"]:
        assert os.path.isfile(store._blob_path(digest))


def test_prune_live_failure_rolls_back_all_candidate_rows(isolated):
    ids = [_record(prompt=str(index), started=float(index)) for index in range(1, 4)]
    with closing(store._connect()) as db, db:
        # Run IDs are generated from store._SAFE_RID; interpolation here is test-only and
        # avoids SQLite's prohibition on bound parameters inside CREATE TRIGGER.
        db.execute("CREATE TRIGGER reject_partial_retention BEFORE DELETE ON runs "
                   f"WHEN OLD.id = '{ids[1]}' "
                   "BEGIN SELECT RAISE(ABORT, 'blocked-retention'); END")
    with pytest.raises(mutations.MutationError, match="blocked-retention"):
        mutations.prune_to(0, dry_run=False)
    assert {run["id"] for run in store.iter_runs()} == set(ids)


def test_retention_kept_shared_digest_is_not_reported_orphan(isolated):
    shared = {"tokens": ["same"], "confidence": [0.5]}
    old = _record(prompt="old", trace=shared, started=1.0)
    kept = _record(prompt="new", trace=shared, started=2.0)
    result = mutations.prune_to(1)
    assert [entry["run_id"] for entry in result["runs"]] == [old]
    assert result["orphaned_trace_digests"] == []
    assert kept in {run["id"] for run in store.iter_runs()}


@pytest.mark.parametrize("keep", [-1, True, 1.5, "1"])
def test_retention_rejects_invalid_keep(isolated, keep):
    with pytest.raises(mutations.MutationError):
        mutations.prune_to(keep)
