"""Explicit migrations for beta-era local data."""
from __future__ import annotations

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
