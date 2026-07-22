"""Contract tests for clozn/runs/gc.py -- blob garbage collection (BACKLOG §2).

Covers, per the backlog item's explicit contract:
  - dry-run vs. live: dry_run=True (the default) never touches disk; dry_run=False deletes exactly the
    unreferenced set.
  - referenced-blob protection: a blob any run row still points to is NEVER deleted, dry-run or live.
  - path containment: a candidate whose resolved path would land outside the blob root is refused, not
    deleted -- proven directly against the containment check gc.py runs immediately before every unlink.
  - malformed on-disk entries (wrong filename shape) are reported separately and never touched.
  - the TOCTOU guard: collect() re-reads the referenced set immediately before deleting, so a run that
    lands between planning and deleting is never destroyed out from under itself.
"""
from __future__ import annotations

from contextlib import closing
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clozn.runs.gc as gc            # noqa: E402
import clozn.runs.store as store      # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Redirect the run store (DB + blobs) to a temp dir for the duration of one test -- same pattern
    tests/test_runlog.py uses (RUNS_DIR is a module global gc.py reads indirectly through store)."""
    monkeypatch.setattr(store, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _write_orphan_blob(digest: str, content: bytes = b'{"orphan": true}') -> str:
    """Drop a blob file directly on disk with no run row ever referencing it -- simulates what a pruned or
    replaced run leaves behind."""
    path = store._blob_path(digest)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(content)
    return path


_FAKE_DIGEST_A = "a" * 64
_FAKE_DIGEST_B = "b" * 64


# ================================================================================================== plan()

def test_plan_keeps_referenced_and_flags_orphan_for_deletion(isolated):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey",
                       trace={"tokens": ["a"], "confidence": [0.9]})
    assert store.get_run(rid) is not None                # sanity: the run persisted
    orphan_path = _write_orphan_blob(_FAKE_DIGEST_A)

    result = gc.plan()

    kept_digests = {e["digest"] for e in result["keep"]}
    delete_digests = {e["digest"] for e in result["delete"]}
    assert _FAKE_DIGEST_A in delete_digests
    assert _FAKE_DIGEST_A not in kept_digests
    assert result["referenced_count"] >= 1
    assert os.path.isfile(orphan_path)                 # plan() NEVER touches disk


def test_plan_never_deletes_anything(isolated):
    _write_orphan_blob(_FAKE_DIGEST_A)
    before = set(os.listdir(os.path.join(store._blob_root(), _FAKE_DIGEST_A[:2])))
    gc.plan()
    gc.plan()
    after = set(os.listdir(os.path.join(store._blob_root(), _FAKE_DIGEST_A[:2])))
    assert before == after


def test_plan_on_empty_store_is_empty(isolated):
    result = gc.plan()
    assert result["keep"] == []
    assert result["delete"] == []
    assert result["referenced_count"] == 0


# =========================================================================================== dry-run vs. live

def test_collect_dry_run_lists_but_does_not_delete(isolated):
    orphan_path = _write_orphan_blob(_FAKE_DIGEST_A)

    result = gc.collect(dry_run=True)

    assert result["dry_run"] is True
    assert result["deleted"] == []
    assert result["failed"] == []
    assert any(e["digest"] == _FAKE_DIGEST_A for e in result["delete"])
    assert os.path.isfile(orphan_path)                  # still on disk -- dry run touched nothing


def test_collect_live_deletes_exactly_the_unreferenced_set(isolated):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey",
                       trace={"tokens": ["a"], "confidence": [0.9]})
    referenced_digest = store.get_run(rid)["trace"]
    assert "unavailable" not in referenced_digest       # the real trace blob loaded clean
    # find the referenced digest on disk directly (via the run's own payload, not guesswork)
    with closing(store._connect()) as db:
        row = db.execute("SELECT payload_json FROM runs WHERE id = ?", (rid,)).fetchone()
    referenced_sha = json.loads(row["payload_json"])["trace_ref"]["sha256"]
    referenced_path = store._blob_path(referenced_sha)
    assert os.path.isfile(referenced_path)

    orphan_path = _write_orphan_blob(_FAKE_DIGEST_A)

    result = gc.collect(dry_run=False)

    assert result["dry_run"] is False
    assert {e["digest"] for e in result["deleted"]} == {_FAKE_DIGEST_A}
    assert result["failed"] == []
    assert not os.path.isfile(orphan_path)              # the orphan is gone
    assert os.path.isfile(referenced_path)               # the referenced blob is untouched


def test_collect_live_with_nothing_to_delete_is_a_safe_no_op(isolated):
    store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey",
                trace={"tokens": ["a"], "confidence": [0.9]})
    result = gc.collect(dry_run=False)
    assert result["deleted"] == []
    assert result["failed"] == []
    assert len(result["keep"]) == 1


# ------------------------------------------------------- Phase 3.7: influence-map blobs, same namespace

def test_referenced_influence_map_blob_survives_gc(isolated):
    """The persisted context<->answer influence-map artifact shares the trace blob's content-addressed
    namespace (store._pack's influence_map_ref) -- GC must protect it exactly like a trace blob as long
    as a run still references it."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey")
    run = store.get_run(rid)
    run["influence_map"] = {"schema": "clozn.context_answer_influence.v1", "matrix": [[0.5]]}
    assert store.replace_run(run)
    with closing(store._connect()) as db:
        row = db.execute("SELECT payload_json FROM runs WHERE id=?", (rid,)).fetchone()
    influence_digest = json.loads(row["payload_json"])["influence_map_ref"]["sha256"]
    influence_path = store._blob_path(influence_digest)
    assert os.path.isfile(influence_path)

    result = gc.collect(dry_run=False)

    assert influence_digest not in {e["digest"] for e in result["deleted"]}
    assert os.path.isfile(influence_path)


def test_orphaned_influence_map_blob_is_collected_once_unreferenced(isolated):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey")
    run = store.get_run(rid)
    run["influence_map"] = {"schema": "clozn.context_answer_influence.v1", "matrix": [[0.5]]}
    assert store.replace_run(run)
    with closing(store._connect()) as db:
        row = db.execute("SELECT payload_json FROM runs WHERE id=?", (rid,)).fetchone()
    influence_digest = json.loads(row["payload_json"])["influence_map_ref"]["sha256"]
    influence_path = store._blob_path(influence_digest)

    # Replace the run again WITHOUT an influence_map -- the old blob is now orphaned, same as a trace
    # blob left behind by a replaced run.
    run2 = store.get_run(rid)
    run2.pop("influence_map", None)
    assert store.replace_run(run2)
    assert os.path.isfile(influence_path)                # untouched until GC actually runs

    result = gc.collect(dry_run=False)

    assert influence_digest in {e["digest"] for e in result["deleted"]}
    assert not os.path.isfile(influence_path)


# ======================================================================================= referenced-blob protection

def test_referenced_blob_is_never_deleted_even_when_content_addressed_dedup_shares_it(isolated):
    """Two runs whose traces serialize to byte-identical JSON share ONE blob (content-addressed dedup) --
    GC must still keep it as long as ANY run references it, not just the first/last."""
    r1 = store.record(source="cli", messages=[{"role": "user", "content": "a"}], response="x",
                      trace={"tokens": ["t"], "confidence": [0.5]}, started=1.0, ended=1.0)
    r2 = store.record(source="cli", messages=[{"role": "user", "content": "b"}], response="y",
                      trace={"tokens": ["t"], "confidence": [0.5]}, started=2.0, ended=2.0)
    with closing(store._connect()) as db:
        rows = {row["id"]: json.loads(row["payload_json"])["trace_ref"]["sha256"]
                for row in db.execute("SELECT id, payload_json FROM runs")}
    assert rows[r1] == rows[r2]                          # confirmed: sharing one blob

    result = gc.collect(dry_run=False)
    assert result["deleted"] == []                        # referenced twice over -- never orphaned
    assert os.path.isfile(store._blob_path(rows[r1]))


def test_pruned_run_orphans_its_blob_and_gc_then_removes_it(isolated):
    """The realistic path an orphan comes from: store._prune() drops old rows past KEEP, and the blob they
    alone referenced is left behind on disk until GC cleans it up."""
    monkey_keep = store.KEEP
    try:
        store.KEEP = 1
        r1 = store.record(source="cli", messages=[{"role": "user", "content": "old"}], response="1",
                          trace={"tokens": ["old"], "confidence": [0.5]}, started=1.0, ended=1.0)
        with closing(store._connect()) as db:
            old_digest = json.loads(
                db.execute("SELECT payload_json FROM runs WHERE id = ?", (r1,)).fetchone()["payload_json"]
            )["trace_ref"]["sha256"]
        old_path = store._blob_path(old_digest)
        assert os.path.isfile(old_path)

        store.record(source="cli", messages=[{"role": "user", "content": "new"}], response="2",
                    trace={"tokens": ["new"], "confidence": [0.6]}, started=2.0, ended=2.0)
        # KEEP=1 -> the prune inside the second record() call evicted r1's row already
        assert store.get_run(r1) is None
        assert os.path.isfile(old_path)                    # the blob itself is untouched by pruning

        result = gc.collect(dry_run=False)
        assert old_digest in {e["digest"] for e in result["deleted"]}
        assert not os.path.isfile(old_path)
    finally:
        store.KEEP = monkey_keep


# ======================================================================================================= malformed

def test_malformed_filenames_are_reported_separately_and_never_touched(isolated):
    root = store._blob_root()
    bogus_dir = os.path.join(root, "zz")
    os.makedirs(bogus_dir, exist_ok=True)
    bogus_path = os.path.join(bogus_dir, "not-a-real-digest.json")
    with open(bogus_path, "w", encoding="utf-8") as handle:
        handle.write("{}")

    result = gc.collect(dry_run=False)

    assert any(e["path"] == bogus_path for e in result["malformed"])
    assert all(e["path"] != bogus_path for e in result["delete"])
    assert all(e["path"] != bogus_path for e in result["deleted"])
    assert os.path.isfile(bogus_path)                      # never deleted, never even considered


def test_dir_prefix_mismatch_is_treated_as_malformed(isolated):
    """A file sitting under the WRONG 2-hex prefix directory for its own filename (shouldn't happen from
    normal writes, but defense-in-depth: never trust the directory name alone)."""
    root = store._blob_root()
    wrong_dir = os.path.join(root, "00")                    # digest starts with 'a', not '00'
    os.makedirs(wrong_dir, exist_ok=True)
    path = os.path.join(wrong_dir, _FAKE_DIGEST_A + ".json")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("{}")

    result = gc.collect(dry_run=False)

    assert any(e["path"] == path for e in result["malformed"])
    assert os.path.isfile(path)


# ================================================================================================ path containment

def test_is_contained_true_for_a_real_blob_path(isolated):
    root = store._blob_root()
    path = _write_orphan_blob(_FAKE_DIGEST_A)
    assert gc._is_contained(root, path) is True


def test_is_contained_false_for_a_path_outside_root(tmp_path):
    root = str(tmp_path / "blobs" / "sha256")
    os.makedirs(root, exist_ok=True)
    outside = tmp_path / "elsewhere" / "secret.json"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("{}", encoding="utf-8")
    assert gc._is_contained(root, str(outside)) is False


def test_is_contained_false_for_dotdot_traversal(tmp_path):
    root = str(tmp_path / "blobs" / "sha256")
    os.makedirs(root, exist_ok=True)
    (tmp_path / "config.json").write_text('{"api_key": "secret"}', encoding="utf-8")
    traversal = os.path.join(root, "..", "..", "config.json")
    assert gc._is_contained(root, traversal) is False


def test_digest_from_path_rejects_traversal_in_the_relative_path(isolated):
    """Even if something upstream ever handed gc a path with `..` components, _digest_from_path's strict
    <2-hex>/<64-hex>.json regex refuses to extract a digest from it -- it becomes `malformed`, never a
    deletion candidate. Belt-and-suspenders alongside _is_contained's own independent check."""
    root = store._blob_root()
    evil = os.path.join(root, "..", "..", "config.json")
    assert gc._digest_from_path(root, evil) is None


def test_collect_never_deletes_a_path_outside_the_declared_blob_root(tmp_path, monkeypatch):
    """A forged blob_root pointed somewhere that doesn't contain the files glob() found (contrived, but
    proves the containment check is load-bearing, not just decorative): collect() must refuse, not delete."""
    monkeypatch.setattr(store, "RUNS_DIR", str(tmp_path / "runs"))
    store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey")
    real_root = store._blob_root()
    _write_orphan_blob(_FAKE_DIGEST_A)

    # sabotage containment for every candidate by pointing _is_contained's root elsewhere
    monkeypatch.setattr(gc, "_is_contained", lambda root, path: False)
    result = gc.collect(dry_run=False)
    assert result["deleted"] == []
    assert all(e["error"].startswith("refused") for e in result["failed"])
    assert os.path.isfile(store._blob_path(_FAKE_DIGEST_A))   # still there -- refused, not deleted


# ========================================================================================================= TOCTOU

def test_collect_protects_a_blob_referenced_between_plan_and_delete(isolated, monkeypatch):
    """Simulates a run landing in the (tiny) window between collect()'s internal plan() and its delete
    loop: the SECOND _referenced_digests() call (the live re-check) must be the one that decides, not the
    first (the plan) -- so a blob that becomes referenced in that window is protected, never deleted."""
    orphan_path = _write_orphan_blob(_FAKE_DIGEST_A)
    calls = {"n": 0}
    real_referenced = gc._referenced_digests

    def _flaky(db):
        calls["n"] += 1
        if calls["n"] == 1:
            return set()                        # plan() sees it as unreferenced
        return {_FAKE_DIGEST_A}                  # the re-check sees a run landed referencing it

    monkeypatch.setattr(gc, "_referenced_digests", _flaky)
    result = gc.collect(dry_run=False)

    assert result["deleted"] == []
    assert os.path.isfile(orphan_path)
    assert calls["n"] == 2                        # confirms both the plan-time and delete-time reads ran
