"""CLI-layer tests for `clozn migrate` (BACKLOG §2): argparse wiring + cmd_migrate's dispatch to the
schema-report/apply path and the --gc blob-GC path. clozn/runs/test_runs_migrations.py and
tests/test_runs_gc.py already cover the underlying engines exhaustively; this file only proves the CLI
shell wires flags through correctly and prints/raises what it should.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clozn.cli.commands.migrate as mig  # noqa: E402
import clozn.runs.store as store          # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _build_parser():
    p = argparse.ArgumentParser(prog="clozn")
    sub = p.add_subparsers(dest="cmd")
    mig.add_subparser(sub)
    return p


# ================================================================================================ add_subparser

def test_add_subparser_defaults():
    p = _build_parser()
    args = p.parse_args(["migrate"])
    assert args.cmd == "migrate"
    assert args.dry_run is False
    assert args.gc is False
    assert args.json is False
    assert args.fn is mig.cmd_migrate


def test_add_subparser_parses_all_flags():
    p = _build_parser()
    args = p.parse_args(["migrate", "--dry-run", "--gc", "--json"])
    assert args.dry_run is True
    assert args.gc is True
    assert args.json is True


# ============================================================================================ schema path (no --gc)

def test_cmd_migrate_reports_pending_on_a_fresh_store(isolated, capsys):
    args = argparse.Namespace(dry_run=False, gc=False, json=True)
    rc = mig.cmd_migrate(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    # a fresh store has already been migrated as a SIDE EFFECT of this very call (apply mode, not dry-run)
    assert out["current_version"] == 0                  # the report reflects the version BEFORE this call
    assert out["target_version"] >= 1
    assert out["applied"] == [1, 2]


def test_cmd_migrate_dry_run_does_not_touch_the_db(isolated, capsys):
    args = argparse.Namespace(dry_run=True, gc=False, json=True)
    rc = mig.cmd_migrate(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["applied"] == []
    assert out["dry_run"] is True
    assert out["up_to_date"] is False

    # confirm nothing was actually applied to disk: current_version is still 0 on a fresh connect
    from clozn.runs import migrations
    from contextlib import closing
    with closing(store._connect()) as db:
        assert migrations.current_version(db) == 0


def test_cmd_migrate_apply_then_rerun_is_up_to_date(isolated, capsys):
    args = argparse.Namespace(dry_run=False, gc=False, json=True)
    mig.cmd_migrate(args)
    capsys.readouterr()
    rc = mig.cmd_migrate(args)                            # second call: nothing pending
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["current_version"] == out["target_version"]
    assert out["up_to_date"] is True
    assert out["applied"] == []


def test_cmd_migrate_text_output_mentions_version(isolated, capsys):
    args = argparse.Namespace(dry_run=False, gc=False, json=False)
    mig.cmd_migrate(args)
    text = capsys.readouterr().out
    assert "schema version:" in text
    assert "applied 2 migration" in text


def test_cmd_migrate_surfaces_failure_as_cloznerror(isolated, monkeypatch):
    from clozn.cli.main import CloznError
    from clozn.runs import migrations

    def _broken_migrate(db, migrations_list=migrations.MIGRATIONS):
        raise RuntimeError("disk full")

    monkeypatch.setattr(migrations, "migrate", _broken_migrate)
    args = argparse.Namespace(dry_run=False, gc=False, json=False)
    with pytest.raises(CloznError, match="disk full"):
        mig.cmd_migrate(args)


# =========================================================================================================== --gc

def test_cmd_migrate_gc_dry_run_lists_without_deleting(isolated, capsys):
    store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey",
                trace={"tokens": ["a"], "confidence": [0.9]})
    orphan_digest = "c" * 64
    from clozn.runs import store as _store
    orphan_path = _store._blob_path(orphan_digest)
    os.makedirs(os.path.dirname(orphan_path), exist_ok=True)
    with open(orphan_path, "w", encoding="utf-8") as handle:
        handle.write("{}")

    args = argparse.Namespace(dry_run=True, gc=True, json=True)
    rc = mig.cmd_migrate(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is True
    assert out["deleted"] == []
    assert any(e["digest"] == orphan_digest for e in out["delete"])
    assert os.path.isfile(orphan_path)                   # dry run -- still there


def test_cmd_migrate_gc_live_deletes_and_reports(isolated, capsys):
    store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey",
                trace={"tokens": ["a"], "confidence": [0.9]})
    orphan_digest = "d" * 64
    from clozn.runs import store as _store
    orphan_path = _store._blob_path(orphan_digest)
    os.makedirs(os.path.dirname(orphan_path), exist_ok=True)
    with open(orphan_path, "w", encoding="utf-8") as handle:
        handle.write("{}")

    args = argparse.Namespace(dry_run=False, gc=True, json=True)
    rc = mig.cmd_migrate(args)
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["dry_run"] is False
    assert {e["digest"] for e in out["deleted"]} == {orphan_digest}
    assert not os.path.isfile(orphan_path)


def test_cmd_migrate_gc_text_output_reports_kept_and_deleted(isolated, capsys):
    store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey",
                trace={"tokens": ["a"], "confidence": [0.9]})
    args = argparse.Namespace(dry_run=False, gc=True, json=False)
    rc = mig.cmd_migrate(args)
    assert rc == 0
    text = capsys.readouterr().out
    assert "blob GC" in text
    assert "keep:" in text
    assert "actually deleted: 0" in text
