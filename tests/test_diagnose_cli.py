"""Model-free CLI coverage for ``clozn diagnose``."""
from __future__ import annotations

import json

import pytest

from clozn.cli import main as cli
import clozn.runs.store as runlog


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))


def _record(*, source="openai_api", started=1000.0, session_key=None, model="m"):
    return runlog.record(
        source=source, model=model, session_key=session_key,
        messages=[{"role": "user", "content": "hello"}], response="world",
        trace=[{"piece": "world", "dt_ms": 4.0}],
        finish_reason="stop", meta={"prompt_tokens": 2, "max_tokens": 8, "n_ctx": 32},
        started=started, ended=started + 0.01,
    )


def test_parser_supports_last_filters_and_exact_run_id():
    last = cli.build_parser().parse_args([
        "diagnose", "last", "--session", "tab-a", "--client-id", "app-a",
        "--client", "script", "--model", "m", "--include-derived", "--json",
    ])
    assert last.target == "last" and last.session == "tab-a" and last.client_id == "app-a"
    assert last.client == "script" and last.model == "m"
    assert last.include_derived is True and last.json is True
    exact = cli.build_parser().parse_args(["diagnose", "run_x"])
    assert exact.target == "run_x" and exact.fn.__name__ == "cmd_diagnose"


def test_last_uses_latest_matching_run_and_excludes_derived(isolated, capsys):
    wanted = _record(started=1000.0, model="wanted")
    _record(started=1001.0, model="other")
    runlog.record(source="replay", model="wanted", messages=[], response="derived",
                  parent_run_id=wanted, started=1002.0)

    args = cli.build_parser().parse_args(["diagnose", "last", "--model", "wanted", "--json"])
    assert cli.main(["diagnose", "last", "--model", "wanted", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["run_id"] == wanted
    assert args.include_derived is False


def test_exact_run_renders_evidence_only_sections(isolated, capsys):
    rid = _record()
    assert cli.main(["diagnose", rid]) == 0
    out = capsys.readouterr().out
    assert f"diagnosis - {rid}" in out
    assert "WHY SLOW" in out and "WHY CUT OFF" in out and "CLIENT AUXILIARY CALLS" in out
    assert "Per-token evidence covers all" in out


def test_filters_are_rejected_for_exact_run(isolated, capsys):
    rid = _record()
    assert cli.main(["diagnose", rid, "--model", "m"]) == 1
    assert "filters apply only" in capsys.readouterr().err
