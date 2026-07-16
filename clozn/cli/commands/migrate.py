"""commands.migrate -- two related but distinct things live here:

  * `clozn migrate-runs <dir>` (legacy, unchanged): one-shot import of the pre-SQLite `run_*.json` journal
    directory into the store. Nothing below touches this.
  * `clozn migrate` (BACKLOG §2, new): the run STORE's own SQLite schema -- reports current vs. target
    schema version, applies pending transactional migrations (clozn.runs.migrations), and previews with
    `--dry-run`; `--gc` switches to blob garbage collection (clozn.runs.gc) instead, deleting blob files no
    run row references (dry-run by default via the same `--dry-run` flag).

`add_subparser`'s wiring mirrors commands.eval/commands.quant_check exactly: import `add_subparser as
_add_migrate` in clozn/cli/main.py and call `_add_migrate(sub)` in build_parser() before `return p`.
"""
from __future__ import annotations

from contextlib import closing
import json
import os


def cmd_migrate_runs(args):
    from clozn.cli import main as ctx
    from clozn.runs import store

    source = os.path.abspath(os.path.expanduser(args.path or os.path.join(ctx.HOME, "runs")))
    result = store.import_json_dir(source)
    print(
        f"run migration: {result['imported']} imported, {result['skipped']} already present, "
        f"{result['invalid']} invalid ({result['found']} JSON files found in {source})"
    )


# ============================================================================================ `clozn migrate`
# Schema migrations + blob GC for the run store's SQLite DB. Kept in this file (not a new module) since it
# is, semantically, the same family of "reconcile local on-disk state" command as migrate-runs above --
# just for the store's schema instead of a legacy JSON journal.

def add_subparser(sub):
    """Register `clozn migrate` on an argparse subparsers object (own function so its wiring is testable
    without dispatching; mirrors commands.eval.add_subparser / commands.quant_check.add_subparser)."""
    pm = sub.add_parser(
        "migrate",
        help="run-store schema migrations + blob GC (BACKLOG §2) -- reports current vs. target schema "
             "version and applies pending migrations; --gc instead garbage-collects unreferenced trace "
             "blobs. Not to be confused with `migrate-runs` (the legacy JSON-journal importer).",
    )
    pm.add_argument("--dry-run", action="store_true", dest="dry_run",
                    help="report/preview only -- schema: print the pending-migration plan without applying "
                         "it; --gc: list blobs that WOULD be deleted without deleting anything")
    pm.add_argument("--gc", action="store_true",
                    help="garbage-collect blob files no run row references, instead of running schema "
                         "migrations")
    pm.add_argument("--json", action="store_true", help="print a machine-readable report instead of text")
    pm.set_defaults(fn=cmd_migrate)
    return pm


def cmd_migrate(args):
    """`clozn migrate [--dry-run] [--gc] [--json]` -- dispatches to the schema-migration report/apply path
    or, with --gc, the blob-GC path. Model-free and local-only: touches nothing but ~/.clozn/runs."""
    if getattr(args, "gc", False):
        return _cmd_gc(args)
    return _cmd_schema(args)


def _cmd_schema(args):
    from clozn.cli.main import CloznError
    from clozn.runs import migrations, store

    # store._connect() only makedirs()'s RUNS_DIR itself; the blob root is a subdirectory _ensure()
    # normally creates too -- reproduce that here so a from-scratch `clozn migrate` (before any run has
    # ever been recorded) doesn't leave the blob dir missing.
    os.makedirs(store.RUNS_DIR, exist_ok=True)
    os.makedirs(store._blob_root(), exist_ok=True)
    as_json = bool(getattr(args, "json", False))
    dry_run = bool(getattr(args, "dry_run", False))

    with closing(store._connect()) as db:
        report = migrations.status(db)          # read-only: never applies anything by itself
        applied: list[int] = []
        error: str | None = None
        if not dry_run and not report["up_to_date"]:
            try:
                applied = migrations.migrate(db)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"

    # One combined report, printed exactly once -- so --json output is a single parseable object whether
    # or not anything was actually applied.
    out = dict(report, dry_run=dry_run, applied=applied)
    if as_json:
        print(json.dumps(out, indent=2))
    else:
        print(f"schema version: {report['current_version']} -> target {report['target_version']}"
              f"{'  (up to date)' if report['up_to_date'] else ''}")
        for step in report["pending"]:
            tag = "would apply" if dry_run else ("applied" if step["version"] in applied else "pending")
            print(f"  {tag}: migration {step['version']} -- {step['description']}")
        if dry_run and report["pending"]:
            print(f"(dry run -- {len(report['pending'])} migration(s) NOT applied)")
        elif applied:
            print(f"applied {len(applied)} migration(s): {applied}")

    if error:
        raise CloznError(f"migration failed: {error} (DB left at the last successfully-applied version -- "
                         f"each migration step is its own transaction, see clozn/runs/migrations.py; safe "
                         f"to re-run `clozn migrate` once the underlying issue is fixed)")
    return 0


def _cmd_gc(args):
    from clozn.runs import gc

    result = gc.collect(dry_run=bool(getattr(args, "dry_run", False)))
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2))
        return 0
    verb = "would delete" if result["dry_run"] else "deleted"
    print(f"blob GC: {result['total_blobs']} blob(s) on disk under {result['blob_root']}, "
          f"{result['referenced_count']} referenced digest(s) in the DB")
    print(f"  keep:   {len(result['keep'])}")
    print(f"  {verb}: {len(result['delete'])}")
    for entry in result["delete"]:
        print(f"    - {entry['digest']}  ({entry['bytes']} bytes)  {entry['path']}")
    if result["malformed"]:
        print(f"  malformed (left alone, not a valid blob path): {len(result['malformed'])}")
    if not result["dry_run"]:
        if result["failed"]:
            print(f"  FAILED to delete {len(result['failed'])}:")
            for entry in result["failed"]:
                print(f"    - {entry['digest']}  {entry['path']}  ({entry['error']})")
        print(f"  actually deleted: {len(result['deleted'])}")
    elif result["delete"]:
        print("(dry run -- nothing deleted; re-run without --dry-run to actually delete)")
    return 0
