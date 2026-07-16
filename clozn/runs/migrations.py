"""Real, transactional schema migrations for clozn/runs/store.py's SQLite database (BACKLOG §2).

Replaces the old `_ensure()` "CREATE TABLE IF NOT EXISTS + upsert a stamp" approach. That approach worked
by accident (every DDL statement was individually idempotent) but had three real gaps:
  - no audit trail -- a bare integer said "we're at version 1" but nothing recorded WHICH steps actually
    ran, or when;
  - no failure semantics -- `executescript` runs several statements back to back with no defined recovery
    if one of them ever fails partway (SQLite CAN roll back DDL, but only if the driver is told to use a
    real transaction, which the old code never did);
  - no dry-run -- `clozn` mutated the on-disk DB the instant anything opened it, with no way to preview.

Design
------
Each migration is a small, ordered (version, description, apply(db)) step. `migrate()` applies every
PENDING step in order, each inside its own explicit transaction: the step's DDL/DML and the ledger row
that marks it applied land in the SAME COMMIT, or neither lands at all (ROLLBACK propagates the original
exception to the caller). A failure at step N therefore leaves the DB at EXACTLY version N-1 -- fully
usable, never half-migrated -- and a subsequent `migrate()` call retries from N.

The ledger deliberately reuses the pre-existing `schema_meta(key, value)` table (rather than adding a new
`schema_migrations` table) so that a fresh, fully-migrated DB's `sqlite_master` schema (its CREATE
TABLE/INDEX text) is BYTE-IDENTICAL to what the old `_ensure()` produced -- the migration bookkeeping is
extra ROWS in an existing table, not an extra table. This is asserted directly in
tests/test_runs_migrations.py by diffing `sqlite_master` dumps. Per-migration rows are keyed
`migration:<version>` (JSON value: description + applied_at); the coarse `schema_version` key is kept in
sync too since it predates this module and nothing else in the repo reads the per-migration rows.

Python's sqlite3 module does NOT auto-open a transaction before DDL in its default ("") isolation mode --
only before INSERT/UPDATE/DELETE -- so a naive `db.executescript(...)` between two explicit
BEGIN/COMMIT calls silently runs outside any transaction and can't be rolled back. `migrate()` works
around this by switching the connection to `isolation_level = None` (manual/autocommit mode) for its own
duration and issuing `BEGIN IMMEDIATE` / `COMMIT` / `ROLLBACK` itself -- SQLite the engine fully supports
transactional DDL once the driver gets out of its way.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

_MIGRATION_KEY_RE = re.compile(r"^migration:(\d+)$")


@dataclass(frozen=True)
class Migration:
    version: int
    description: str
    apply: Callable[[sqlite3.Connection], None]


def _migration_0001_initial_schema(db: sqlite3.Connection) -> None:
    """The baseline schema: byte-for-byte what the old `_ensure()` created in one `executescript` call --
    but issued as individual `execute()` calls here, NOT `executescript()`. `executescript()` implicitly
    COMMITs any already-open transaction before it runs (Python sqlite3 docs: "If there is a pending
    transaction, an implicit COMMIT statement is executed first") -- inside `migrate()`'s explicit `BEGIN
    IMMEDIATE ... COMMIT` wrapper that silently ends OUR transaction partway through, so a later step's
    failure could no longer roll this one back. Individual `execute()` calls have no such side effect.
    Kept as ONE migration (not split further) because there is nothing partial about it to test -- the
    mid-migration-failure contract is proven generically in tests/test_runs_migrations.py against a
    throwaway fabricated migration list, not by intentionally breaking this real one."""
    db.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            id TEXT PRIMARY KEY,
            created_ts REAL NOT NULL,
            created_at TEXT NOT NULL,
            source TEXT NOT NULL,
            client TEXT NOT NULL,
            model TEXT NOT NULL,
            substrate TEXT NOT NULL,
            parent_run_id TEXT,
            finish_reason TEXT,
            error TEXT,
            prompt_summary TEXT NOT NULL,
            response_summary TEXT NOT NULL,
            duration_ms INTEGER NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    db.execute("CREATE INDEX IF NOT EXISTS runs_created_idx ON runs(created_ts DESC, id DESC)")
    db.execute("CREATE INDEX IF NOT EXISTS runs_source_idx ON runs(source, created_ts DESC)")
    db.execute("CREATE INDEX IF NOT EXISTS runs_parent_idx ON runs(parent_run_id, created_ts ASC)")
    db.execute("CREATE INDEX IF NOT EXISTS runs_model_idx ON runs(model, created_ts DESC)")


# The shipped, ordered migration set. Append-only: once released, a migration's `apply` must never be
# edited (a DB that already applied it would silently diverge from one that applies the edited version) --
# ship a NEW migration with a higher version instead.
MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "initial schema: schema_meta + runs + indexes", _migration_0001_initial_schema),
)

TARGET_VERSION = max(m.version for m in MIGRATIONS)


def _ensure_ledger_table(db: sqlite3.Connection) -> None:
    """Bootstrap the ledger table itself, outside any migration transaction. Safe to call unconditionally
    on both a brand-new DB file (creates it) and an existing legacy one (already has this exact table from
    the old `_ensure()` -- a no-op). This is NOT "migration 0": it never needs rolling back, because
    creating an empty key/value table has no partial state to roll back TO."""
    db.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")


def current_version(db: sqlite3.Connection) -> int:
    """The highest migration version whose ledger row is present. 0 for a brand-new DB (including one that
    doesn't even have the schema_meta table yet -- e.g. an in-memory DB nobody has touched)."""
    _ensure_ledger_table(db)
    rows = db.execute("SELECT key FROM schema_meta WHERE key LIKE 'migration:%'").fetchall()
    versions = []
    for row in rows:
        m = _MIGRATION_KEY_RE.match(row[0])
        if m:
            versions.append(int(m.group(1)))
    return max(versions, default=0)


def pending(db: sqlite3.Connection, migrations: Sequence[Migration] = MIGRATIONS) -> list[Migration]:
    """Migrations not yet applied to `db`, in ascending version order."""
    applied = current_version(db)
    return sorted((m for m in migrations if m.version > applied), key=lambda m: m.version)


def migrate(db: sqlite3.Connection, migrations: Sequence[Migration] = MIGRATIONS) -> list[int]:
    """Apply every pending migration to `db`, each in its own transaction. Returns the versions actually
    applied (empty list if already current). Raises on the first failing step WITHOUT applying any step
    after it -- the caller decides whether that's fatal (the CLI) or should degrade quietly (store._ensure,
    which already tolerated an unusable DB before this module existed)."""
    _ensure_ledger_table(db)
    applied: list[int] = []
    prior_isolation = db.isolation_level
    db.isolation_level = None      # manual transaction control -- see module docstring for why this
                                    # matters: default mode never auto-BEGINs around DDL, so without this
                                    # a mid-step failure would leave whatever DDL already ran committed.
    try:
        for m in pending(db, migrations):
            db.execute("BEGIN IMMEDIATE")
            try:
                m.apply(db)
                stamp = json.dumps({"description": m.description,
                                     "applied_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                db.execute("INSERT INTO schema_meta(key, value) VALUES(?, ?)", (f"migration:{m.version}", stamp))
                db.execute(
                    "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(m.version),),
                )
            except BaseException:
                db.execute("ROLLBACK")
                raise
            else:
                db.execute("COMMIT")
            applied.append(m.version)
    finally:
        db.isolation_level = prior_isolation
    return applied


def status(db: sqlite3.Connection, migrations: Sequence[Migration] = MIGRATIONS) -> dict:
    """A doctor-style snapshot for `clozn migrate` / `clozn migrate --dry-run`: current version, target
    version, and the ordered list of steps that would run. Read-only -- never mutates `db`."""
    current = current_version(db)
    target = max((m.version for m in migrations), default=0)
    todo = pending(db, migrations)
    return {
        "current_version": current,
        "target_version": target,
        "up_to_date": current >= target,
        "pending": [{"version": m.version, "description": m.description} for m in todo],
    }
