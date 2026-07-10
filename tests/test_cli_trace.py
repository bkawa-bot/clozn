"""Model-free tests for `clozn trace` source selection.

The default trace command must read the same ~/.clozn/runs journal as Studio. The old
~/.clozn/traces cache remains available through --legacy-cache and must not be pruned by new writes.
"""
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


def _args(*, list=False, legacy_cache=False):
    return SimpleNamespace(list=list, legacy_cache=legacy_cache)


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


def test_trace_legacy_cache_flag_still_reads_old_trace_files(isolated, capsys):
    _write_legacy(Path(cli.HOME), prompt="legacy prompt")
    runlog.record(source="studio_chat", client="studio", model="new-model",
                  messages=[{"role": "user", "content": "new prompt"}], response="New answer",
                  trace={"tokens": ["New"], "confidence": [0.9]}, started=2000.0, ended=2000.1)

    cli.cmd_trace(_args(legacy_cache=True))

    out = capsys.readouterr().out
    assert "legacy prompt" in out
    assert "almost: older 0.19" in out
    assert "new prompt" not in out


def test_save_trace_does_not_prune_legacy_cache_files(isolated):
    trace_dir = Path(cli.HOME) / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    for i in range(30):
        (trace_dir / f"old-{i:02d}.json").write_text("{}", encoding="utf-8")

    cli._save_trace({"id": "new-trace"}, [])

    assert (trace_dir / "new-trace.json").exists()
    assert len(list(trace_dir.glob("*.json"))) == 31


def test_trace_cache_files_sorts_by_mtime_not_filename(isolated):
    """Regression: a stray non-timestamp filename (e.g. `new-trace.json`, left over from some other
    process writing straight into the traces dir) must not be picked as "the latest trace" just because
    it sorts last lexicographically -- ASCII puts digits before lowercase letters, so `new-trace.json`
    sorts after every real `<unix-ts>-<pid>.json` name regardless of when it was actually written.
    `clozn branch` relies on `_trace_cache_files()[-1]` being the most RECENTLY WRITTEN file."""
    import clozn.cli.trace_io as trace_io

    trace_dir = Path(cli.HOME) / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)

    old_real = trace_dir / "1700000000-001.json"
    old_real.write_text(json.dumps({"meta": {"id": "1700000000-001"}, "steps": [{"piece": "old"}]}),
                         encoding="utf-8")
    os.utime(old_real, (1_700_000_000, 1_700_000_000))

    stray = trace_dir / "new-trace.json"                # written in between, but sorts last by NAME
    stray.write_text(json.dumps({"meta": {"id": "new-trace"}, "steps": []}), encoding="utf-8")
    os.utime(stray, (1_700_000_500, 1_700_000_500))

    latest_real = trace_dir / "1700001000-002.json"      # the actually-latest trace, by write time
    latest_real.write_text(json.dumps({"meta": {"id": "1700001000-002"}, "steps": [{"piece": "new"}]}),
                            encoding="utf-8")
    os.utime(latest_real, (1_700_001_000, 1_700_001_000))

    files = trace_io._trace_cache_files()

    assert files[-1] == str(latest_real)


def test_trace_parser_exposes_legacy_cache_flag():
    args = cli.build_parser().parse_args(["trace", "--legacy-cache"])
    assert args.legacy_cache is True
