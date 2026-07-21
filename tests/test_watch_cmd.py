from __future__ import annotations

import json

import clozn.runs.store as runlog
from clozn.cli import main as cli


def test_watch_once_prints_latest_local_run_as_ndjson(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    first = runlog.record(source="api", model="a", messages=[{"role": "user", "content": "one"}],
                          response="first")
    latest = runlog.record(source="api", model="b", messages=[{"role": "user", "content": "two"}],
                           response="second")
    assert cli.main(["watch", "--once", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["id"] == latest
    assert out["id"] != first


def test_watch_since_once_drains_in_insertion_order(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    first = runlog.record(source="api", messages=[{"role": "user", "content": "one"}], response="1")
    second = runlog.record(source="api", messages=[{"role": "user", "content": "two"}], response="2")
    third = runlog.record(source="api", messages=[{"role": "user", "content": "three"}], response="3")
    assert cli.main(["watch", "--since", first, "--once", "--json"]) == 0
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [row["id"] for row in rows] == [second, third]


def test_watch_rejects_unknown_since_id(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    assert cli.main(["watch", "--since", "run_missing", "--once"]) == 1
    assert "run not found" in capsys.readouterr().err
