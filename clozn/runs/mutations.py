"""Transactional privacy and retention mutations for the local run journal.

SQLite is authoritative. A database mutation commits before trace cleanup; cleanup
then rechecks all references under a SQLite writer lock. The safe failure mode is an
orphan for existing GC, never a surviving row whose evidence was deleted.
"""
from __future__ import annotations

from contextlib import closing
import json
import os
import re
import sqlite3
import time
from typing import Any

from . import store

REDACTION_SCHEMA = "clozn.run_redaction.v1"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_REMOVED_FIELDS = (
    "assembled_messages", "behavior", "changes_applied", "client_key",
    "client_key_source", "context_receipt", "error", "final_prompt", "memory",
    "influence_map", "messages", "meta", "output_contract", "project_key", "prompt_summary",
    "reasoning", "response", "response_summary", "session_key", "tiny_tests",
    "trace", "warnings",
)


class MutationError(RuntimeError):
    """A journal mutation could not be completed atomically."""


def _require_run_id(run_id: Any) -> str:
    if not store._valid_rid(run_id):
        raise MutationError("run_id must be an exact valid run ID")
    return run_id


def _blob_digest(payload: Any, key: str) -> str | None:
    ref = payload.get(key) if isinstance(payload, dict) else None
    digest = ref.get("sha256") if isinstance(ref, dict) else None
    return digest if isinstance(digest, str) and _DIGEST_RE.fullmatch(digest) else None


def _trace_digest(payload: Any) -> str | None:
    return _blob_digest(payload, "trace_ref")


def _influence_map_digest(payload: Any) -> str | None:
    return _blob_digest(payload, "influence_map_ref")


def _begin(db: sqlite3.Connection) -> None:
    db.isolation_level = None
    db.execute("BEGIN IMMEDIATE")


def _end(db: sqlite3.Connection, error: BaseException | None) -> None:
    try:
        db.execute("ROLLBACK" if error is not None else "COMMIT")
    except sqlite3.Error:
        if error is None:
            raise


def _load_payload(raw: str) -> dict | None:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _tombstone(row: sqlite3.Row, payload: dict | None) -> dict:
    payload = payload or {}
    receipt = {
        "schema": REDACTION_SCHEMA,
        "status": "redacted",
        "redacted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "removed_fields": list(_REMOVED_FIELDS),
        "trace_evidence_removed": _trace_digest(payload) is not None,
        "influence_evidence_removed": _influence_map_digest(payload) is not None,
        "source_payload_readable": bool(payload),
    }
    identity = payload.get("identity")
    identity = identity if isinstance(identity, dict) else {}
    safe_identity = {
        key: identity[key]
        for key in ("model_sha256", "model_size_bytes", "template_fingerprint",
                    "engine_build", "clozn_version", "captured_at")
        if key in identity
    }
    return {
        "id": row["id"], "created_at": row["created_at"],
        "created_ts": float(row["created_ts"]), "recorded_ts": float(row["recorded_ts"]),
        "source": row["source"], "client": row["client"],
        "client_key": None, "client_key_source": None, "session_key": None,
        "project_key": None, "model": row["model"], "substrate": row["substrate"],
        "prompt_summary": "", "response_summary": "", "messages": [], "response": "",
        "reasoning": {}, "assembled_messages": None, "final_prompt": None,
        "memory": {}, "behavior": {}, "trace_ref": {},
        "timing": {"duration_ms": int(row["duration_ms"] or 0)},
        "parent_run_id": row["parent_run_id"], "changes_applied": None, "error": None,
        "finish_reason": row["finish_reason"], "meta": {},
        "identity": safe_identity,
        "output_contract": {}, "context_receipt": {}, "warnings": [],
        "flags": ["redacted"], "redaction": receipt,
    }


def _cleanup_blob(digest: str | None) -> dict[str, Any]:
    """Synchronously remove one content-addressed blob (trace or influence-map) IF nothing else still
    references it -- the immediate counterpart to the deferred, explicitly-requested `clozn migrate --gc`
    sweep. Generic over which ref key produced the digest; `gc._referenced_digests` already scans both."""
    if digest is None:
        return {"status": "not_applicable", "sha256": None}
    path, root = store._blob_path(digest), store._blob_root()
    try:
        from . import gc
        with closing(store._connect()) as db:
            _begin(db)
            error = None
            try:
                if digest in gc._referenced_digests(db):
                    result = {"status": "retained_shared", "sha256": digest}
                elif not os.path.exists(path):
                    result = {"status": "already_missing", "sha256": digest}
                elif not gc._is_contained(root, path):
                    result = {"status": "failed", "sha256": digest,
                              "error": "refused: path escapes blob root"}
                else:
                    try:
                        os.remove(path)
                    except OSError as exc:
                        result = {"status": "failed", "sha256": digest,
                                  "error": f"{type(exc).__name__}: {exc}"}
                    else:
                        result = {"status": "deleted", "sha256": digest}
            except BaseException as exc:
                error = exc
                raise
            finally:
                _end(db, error)
        return result
    except Exception as exc:
        return {"status": "failed", "sha256": digest,
                "error": f"{type(exc).__name__}: {exc}"}


def redact_run(run_id: str) -> dict[str, Any]:
    """Replace exactly one run with a content-free audit tombstone."""
    run_id = _require_run_id(run_id)
    store._ensure()
    digest = None
    influence_digest = None
    try:
        with closing(store._connect()) as db:
            _begin(db)
            error = None
            try:
                row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
                if row is None:
                    raise MutationError(f"run {run_id!r} was not found")
                else:
                    payload = _load_payload(row["payload_json"])
                    existing = payload.get("redaction") if payload else None
                    if isinstance(existing, dict) and existing.get("status") == "redacted":
                        result = {"ok": True, "action": "redact", "run_id": run_id,
                                  "already_redacted": True, "redaction": dict(existing)}
                    else:
                        digest = _trace_digest(payload)
                        influence_digest = _influence_map_digest(payload)
                        redacted = _tombstone(row, payload)
                        encoded = json.dumps(redacted, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":"), allow_nan=False)
                        db.execute(
                            "UPDATE runs SET client_key=NULL, session_key=NULL, error=NULL, "
                            "prompt_summary='', response_summary='', payload_json=? WHERE id=?",
                            (encoded, run_id),
                        )
                        result = {"ok": True, "action": "redact", "run_id": run_id,
                                  "already_redacted": False, "redaction": redacted["redaction"]}
            except BaseException as exc:
                error = exc
                raise
            finally:
                _end(db, error)
    except MutationError:
        raise
    except Exception as exc:
        raise MutationError(
            f"could not redact run {run_id!r}: {type(exc).__name__}: {exc}") from None
    not_applicable = {"status": "not_applicable", "sha256": None}
    applicable = result.get("ok") and not result.get("already_redacted")
    result["trace_cleanup"] = _cleanup_blob(digest) if applicable else not_applicable
    result["influence_map_cleanup"] = _cleanup_blob(influence_digest) if applicable else not_applicable
    return result


def delete_run(run_id: str) -> dict[str, Any]:
    """Delete exactly one run row; no tombstone survives."""
    run_id = _require_run_id(run_id)
    store._ensure()
    digest = None
    influence_digest = None
    try:
        with closing(store._connect()) as db:
            _begin(db)
            error = None
            try:
                row = db.execute("SELECT payload_json FROM runs WHERE id=?", (run_id,)).fetchone()
                if row is None:
                    raise MutationError(f"run {run_id!r} was not found")
                else:
                    payload = _load_payload(row["payload_json"])
                    digest = _trace_digest(payload)
                    influence_digest = _influence_map_digest(payload)
                    db.execute("DELETE FROM runs WHERE id=?", (run_id,))
                    result = {"ok": True, "action": "delete", "run_id": run_id}
            except BaseException as exc:
                error = exc
                raise
            finally:
                _end(db, error)
    except MutationError:
        raise
    except Exception as exc:
        raise MutationError(
            f"could not delete run {run_id!r}: {type(exc).__name__}: {exc}") from None
    not_applicable = {"status": "not_applicable", "sha256": None}
    result["trace_cleanup"] = _cleanup_blob(digest) if result.get("ok") else not_applicable
    result["influence_map_cleanup"] = _cleanup_blob(influence_digest) if result.get("ok") else not_applicable
    return result


def _retention_plan(db: sqlite3.Connection, keep: int) -> dict[str, Any]:
    rows = db.execute(
        "SELECT id, recorded_ts, payload_json FROM runs "
        "ORDER BY recorded_ts DESC, id DESC"
    ).fetchall()
    retained, removed = rows[:keep], rows[keep:]

    def digest_of(row) -> str | None:
        return _trace_digest(_load_payload(row["payload_json"]))

    retained_digests = {digest for row in retained if (digest := digest_of(row)) is not None}
    entries, removed_digests = [], set()
    for row in reversed(removed):  # report exact deletion order oldest-first
        digest = digest_of(row)
        if digest is not None:
            removed_digests.add(digest)
        entry = {"run_id": row["id"], "recorded_ts": float(row["recorded_ts"])}
        if digest is not None:
            entry["trace_sha256"] = digest
        entries.append(entry)
    return {
        "keep": keep, "total_count": len(rows), "kept_count": len(retained),
        "delete_count": len(removed), "runs": entries,
        "orphaned_trace_digests": sorted(removed_digests - retained_digests),
    }


def prune_to(keep: int, *, dry_run: bool = True) -> dict[str, Any]:
    """Plan or apply exact oldest-row retention; blob removal stays with existing GC."""
    if not isinstance(keep, int) or isinstance(keep, bool) or keep < 0:
        raise MutationError("keep must be a non-negative integer")
    if not isinstance(dry_run, bool):
        raise MutationError("dry_run must be a boolean")
    store._ensure()
    try:
        with closing(store._connect()) as db:
            if dry_run:
                plan = _retention_plan(db, keep)
            else:
                _begin(db)
                error = None
                try:
                    plan = _retention_plan(db, keep)
                    db.executemany("DELETE FROM runs WHERE id=?",
                                   [(entry["run_id"],) for entry in plan["runs"]])
                except BaseException as exc:
                    error = exc
                    raise
                finally:
                    _end(db, error)
    except Exception as exc:
        raise MutationError(f"could not apply retention: {type(exc).__name__}: {exc}") from None
    return {
        "ok": True, "action": "retention", "dry_run": dry_run, **plan,
        "run_ids": [entry["run_id"] for entry in plan["runs"]],
        "deleted_count": 0 if dry_run else plan["delete_count"],
        "blob_cleanup": "deferred_to_gc",
    }


__all__ = [
    "MutationError", "REDACTION_SCHEMA", "delete_run", "prune_to", "redact_run",
]
