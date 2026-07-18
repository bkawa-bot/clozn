"""`clozn test-model` (FRONTIER_BETS §9.3, "the model's own CI") -- argparse wiring, the golden.diff
bucketing, and the CLI shell's --save / regression-detection / exit-code behavior. Model-free throughout:
`clozn.eval.golden.run_and_grade` and `.engine_health` (the only two functions that talk to a live
gateway) are monkeypatched everywhere below, mirroring tests/test_cli_migrate.py's
monkeypatch-the-store-path convention for isolating disk I/O into tmp_path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clozn.cli.commands.test_model as tm  # noqa: E402
from clozn.cli.main import build_parser     # noqa: E402
from clozn.eval import golden               # noqa: E402


def _row(q, gold, reply, correct, kind="exact"):
    return {"q": q, "gold": gold, "kind": kind, "aliases": [], "reply": reply, "correct": correct}


# ================================================================================================ add_subparser

def _subparser_choices(p):
    for a in p._actions:
        if getattr(a, "choices", None) and "test-model" in a.choices:
            return a.choices
    return {}


def test_test_model_is_registered():
    assert "test-model" in _subparser_choices(build_parser())


def test_defaults_and_dispatch():
    ns = build_parser().parse_args(["test-model"])
    assert ns.which == "all"
    assert ns.save is False
    assert ns.json is False
    assert ns.url.endswith(":8080")
    assert ns.fn is tm.cmd_test_model


def test_accepts_set_save_json():
    ns = build_parser().parse_args(["test-model", "--set", "hard", "--save", "--json", "--url", "http://x:1"])
    assert ns.which == "hard" and ns.save is True and ns.json is True and ns.url == "http://x:1"


# ==================================================================================================== golden.diff

def test_diff_regression_new_pass_changed_unchanged():
    golden_rows = [
        _row("q1", "Paris", "Paris", True),      # will regress
        _row("q2", "42", "wrong", False),         # will become a new pass
        _row("q3", "Rome", "It's Rome I think", True),   # same bucket, different text -> changed
        _row("q4", "Tokyo", "Tokyo", True),        # identical -> unchanged
    ]
    current_rows = [
        _row("q1", "Paris", "London", False),
        _row("q2", "42", "42", True),
        _row("q3", "Rome", "Rome", True),
        _row("q4", "Tokyo", "Tokyo", True),
    ]
    out = golden.diff(golden_rows, current_rows)
    assert out["n_regressions"] == 1 and out["regressions"][0]["q"] == "q1"
    assert out["n_new_passes"] == 1 and out["new_passes"][0]["q"] == "q2"
    assert out["n_changed"] == 1 and out["changed"][0]["q"] == "q3"
    assert out["n_unchanged"] == 1 and out["unchanged"][0]["q"] == "q4"
    assert out["n_new"] == 0 and out["n_missing"] == 0


def test_diff_flags_new_and_missing_probes():
    golden_rows = [_row("old-q", "A", "A", True)]
    current_rows = [_row("new-q", "B", "B", True)]
    out = golden.diff(golden_rows, current_rows)
    assert out["n_new"] == 1 and out["new"][0]["q"] == "new-q"
    assert out["n_missing"] == 1 and out["missing"][0]["q"] == "old-q"
    assert out["n_regressions"] == 0


def test_diff_ungradeable_now_counts_as_a_regression():
    # was correct, now None (ungradeable) -- still a regression, not silently dropped
    golden_rows = [_row("q1", "42", "42", True)]
    current_rows = [_row("q1", "42", "", None)]
    out = golden.diff(golden_rows, current_rows)
    assert out["n_regressions"] == 1


# ============================================================================================= golden.save/load

def test_save_and_load_round_trip(tmp_path):
    path = str(tmp_path / "golden.json")
    rows = [_row("q1", "Paris", "Paris", True)]
    health = {"model": "some/model.Q4_K_M.gguf", "model_sha256": "abc123", "device": "cuda"}
    written = golden.save(rows, which="easy", health=health, path=path)
    assert written == path and os.path.isfile(path)

    loaded = golden.load(path)
    assert loaded["which"] == "easy"
    assert loaded["model"] == "some/model.Q4_K_M.gguf"
    assert loaded["model_sha256"] == "abc123"
    assert loaded["rows"] == rows
    assert "saved_ts" in loaded


def test_load_missing_file_returns_none(tmp_path):
    assert golden.load(str(tmp_path / "nope.json")) is None


# ======================================================================================================= cmd_test_model

def _args(which="all", save=False, as_json=False, url="http://127.0.0.1:8080"):
    return argparse.Namespace(which=which, save=save, json=as_json, url=url)


def test_save_creates_fixture_file(tmp_path, monkeypatch, capsys):
    fixture_path = str(tmp_path / "golden.json")
    monkeypatch.setattr(golden, "_PATH", fixture_path)
    monkeypatch.setattr(golden, "run_and_grade", lambda url, which, model="clozn":
                        [_row("What is the capital of France?", "Paris", "Paris", True)])
    monkeypatch.setattr(golden, "engine_health", lambda url, timeout=10.0: {"model": "m.gguf"})

    rc = tm.cmd_test_model(_args(save=True))
    assert rc == 0
    assert os.path.isfile(fixture_path)

    saved = golden.load(fixture_path)
    assert saved["rows"][0]["reply"] == "Paris"
    assert saved["model"] == "m.gguf"
    out = capsys.readouterr().out
    assert "saved 1 probe" in out


def test_diff_no_fixture_exits_2(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(golden, "_PATH", str(tmp_path / "golden.json"))
    monkeypatch.setattr(golden, "run_and_grade", lambda url, which, model="clozn": [])
    monkeypatch.setattr(golden, "engine_health", lambda url, timeout=10.0: {})

    rc = tm.cmd_test_model(_args())
    assert rc == 2
    assert "no golden fixture saved yet" in capsys.readouterr().out


def test_diff_detects_regression_exit_1(tmp_path, monkeypatch, capsys):
    fixture_path = str(tmp_path / "golden.json")
    monkeypatch.setattr(golden, "_PATH", fixture_path)
    golden.save([_row("What is the capital of France?", "Paris", "Paris", True)],
               which="all", health={"model": "m.gguf"}, path=fixture_path)

    # re-run now returns a WRONG answer for the same probe -- quant/dial regression
    monkeypatch.setattr(golden, "run_and_grade", lambda url, which, model="clozn":
                        [_row("What is the capital of France?", "Paris", "London", False)])
    monkeypatch.setattr(golden, "engine_health", lambda url, timeout=10.0: {"model": "m.gguf"})

    rc = tm.cmd_test_model(_args())
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "regressions=1" in out


def test_diff_no_regression_exit_0(tmp_path, monkeypatch, capsys):
    fixture_path = str(tmp_path / "golden.json")
    monkeypatch.setattr(golden, "_PATH", fixture_path)
    golden.save([_row("What is the capital of France?", "Paris", "Paris", True)],
               which="all", health={"model": "m.gguf"}, path=fixture_path)

    monkeypatch.setattr(golden, "run_and_grade", lambda url, which, model="clozn":
                        [_row("What is the capital of France?", "Paris", "Paris", True)])
    monkeypatch.setattr(golden, "engine_health", lambda url, timeout=10.0: {"model": "m.gguf"})

    rc = tm.cmd_test_model(_args())
    assert rc == 0
    assert "PASS" in capsys.readouterr().out


def test_diff_json_output_shape(tmp_path, monkeypatch, capsys):
    fixture_path = str(tmp_path / "golden.json")
    monkeypatch.setattr(golden, "_PATH", fixture_path)
    golden.save([_row("q1", "42", "wrong", False)], which="arith", health={}, path=fixture_path)

    monkeypatch.setattr(golden, "run_and_grade", lambda url, which, model="clozn":
                        [_row("q1", "42", "42", True)])
    monkeypatch.setattr(golden, "engine_health", lambda url, timeout=10.0: {})

    rc = tm.cmd_test_model(_args(as_json=True))
    assert rc == 0    # a new pass, not a regression
    out = json.loads(capsys.readouterr().out)
    assert out["n_new_passes"] == 1
    assert out["fixture_which"] == "arith"
