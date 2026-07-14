"""Model-free tests for `clozn trace` over the canonical SQLite run journal."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)                                  # repo root (clozn/cli.py lives here)
sys.path.insert(0, REPO)

import clozn.cli.main as cli  # noqa: E402
import clozn.cli.formatting as fmt  # noqa: E402
import clozn.runs.store as runlog  # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cli, "HOME", str(tmp_path / ".clozn"))     # HOME is owned by clozn.cli.main
    # The color globals live in clozn.cli.formatting -- trace_io._render_trace reads fmt.DIM/BOLD/RST live.
    monkeypatch.setattr(fmt, "COLOR", False)
    monkeypatch.setattr(fmt, "DIM", "")
    monkeypatch.setattr(fmt, "BOLD", "")
    monkeypatch.setattr(fmt, "RST", "")
    return tmp_path


def _args(*, list=False):
    return SimpleNamespace(list=list)


def _write_legacy(home: Path, *, rid="legacy-1", prompt="old prompt"):
    trace_dir = home / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / f"{rid}.json").write_text(json.dumps({
        "meta": {"id": rid, "model": "old-model", "prompt": prompt, "backend": "CPU", "n": 1},
        "steps": [{"piece": "Old", "conf": 0.2, "alts": [{"piece": " older", "prob": 0.19}]}],
    }), encoding="utf-8")


def test_trace_defaults_to_shared_runlog_even_when_legacy_cache_exists(isolated, capsys):
    _write_legacy(Path(cli.HOME), prompt="legacy prompt")
    runlog.record(source="studio_chat", client="studio", model="new-model", substrate="engine",
                  messages=[{"role": "user", "content": "new prompt"}], response="New answer",
                  trace={"tokens": ["New", " answer"], "confidence": [0.92, 0.42],
                         "alternatives": [[], [{"piece": " option", "prob": 0.31}]]},
                  started=2000.0, ended=2000.1)

    cli.cmd_trace(_args())

    out = capsys.readouterr().out
    assert "new prompt" in out
    assert "New answer" in out
    assert "almost: option 0.31" in out
    assert "legacy prompt" not in out


def test_trace_skips_internal_replay_runs_when_picking_the_last_run(isolated, capsys):
    """Regression (receipt-journal spam): a `/runs/<id>/receipts` prove-all persists an internal
    leave-one-out re-generation as its own run (source="replay", clozn.replay.replay.replay()) -- it
    must not masquerade as "the last run" a bare `clozn trace` shows."""
    runlog.record(source="cli", client="cli", model="new-model", substrate="engine",
                  messages=[{"role": "user", "content": "the real prompt"}], response="Real answer",
                  trace={"tokens": ["Real", " answer"], "confidence": [0.9, 0.9]},
                  started=1000.0, ended=1000.1)
    runlog.record(source="replay", client="studio", model="new-model", substrate="engine",
                  messages=[{"role": "user", "content": "the real prompt"}], response="Ablated answer",
                  trace={"tokens": ["Ablated"], "confidence": [0.9]},
                  started=2000.0, ended=2000.1)   # newer, but internal -- must be skipped

    cli.cmd_trace(_args())

    out = capsys.readouterr().out
    assert "the real prompt" in out
    assert "Ablated answer" not in out


def test_trace_list_skips_internal_replay_runs(isolated, capsys):
    runlog.record(source="cli", messages=[{"role": "user", "content": "a real prompt"}], response="hey",
                 started=1000.0, ended=1000.1)
    runlog.record(source="replay", messages=[{"role": "user", "content": "a real prompt"}], response="hey2",
                 started=2000.0, ended=2000.1)

    cli.cmd_trace(_args(list=True))

    out = capsys.readouterr().out
    assert "a real prompt" in out
    assert out.count("a real prompt") == 1     # the replay-sourced duplicate is not listed


def test_trace_shows_token_id_logprob_and_topk_entropy_with_honest_labels(isolated, capsys):
    """Backlog #2: `clozn trace` shows the v2 per-token fields, compactly and honestly labeled -- 'logp'
    for the derived logprob, 'H@k(approx)' (never bare 'H', which is reserved for the true full-softmax
    entropy the HF/Qwen path can compute) for the engine path's top-k entropy approximation."""
    runlog.record(source="engine_chat", client="studio", model="new-model", substrate="engine",
                  messages=[{"role": "user", "content": "new prompt"}], response="New answer",
                  trace={"tokens": ["New", " answer"], "confidence": [0.92, 0.42],
                         "token_ids": [11, 22], "topk_entropy": [None, 0.87],
                         "alternatives": [[], [{"piece": " option", "prob": 0.31}]]},
                  started=2000.0, ended=2000.1)

    cli.cmd_trace(_args())

    out = capsys.readouterr().out
    assert "id 22" in out
    assert "logp " in out
    assert "H@k(approx) 0.870" in out
    assert " H " not in out and "H 0." not in out       # never the true-entropy label for a top-k value


# ----------------------------------------------------------- stray legacy JSON is never an implicit data source

def _write_bare_run(runs_dir: Path, name="run_bare"):
    os.makedirs(runs_dir, exist_ok=True)
    (runs_dir / f"{name}.json").write_text("{}", encoding="utf-8")


def test_trace_ignores_a_bare_legacy_json_file(isolated, capsys):
    _write_bare_run(Path(runlog.RUNS_DIR))
    cli.cmd_trace(_args())
    assert "no runs yet" in capsys.readouterr().out


def test_trace_list_ignores_a_bare_legacy_json_file(isolated, capsys):
    _write_bare_run(Path(runlog.RUNS_DIR))
    cli.cmd_trace(_args(list=True))
    out = capsys.readouterr().out
    assert "no runs yet" in out


def test_trace_list_ignores_bare_json_alongside_a_sqlite_run(isolated, capsys):
    _write_bare_run(Path(runlog.RUNS_DIR), name="run_aaaa_bare")
    runlog.record(source="cli", messages=[{"role": "user", "content": "a real prompt"}], response="hey",
                 started=2000.0, ended=2000.1)

    cli.cmd_trace(_args(list=True))

    out = capsys.readouterr().out
    assert "WHEN" in out
    assert "a real prompt" in out
