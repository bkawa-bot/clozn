"""test_cli_branch -- model-free tests for `clozn branch` (clozn.cli.commands.explain.cmd_branch).

Regression coverage for a usability papercut found in hands-on testing: the "original:"/"branch:" summary
lines are meant to be ONE terminal line each (see the command's own prints), but a reply containing a raw
newline (very ordinary -- e.g. a multi-line story) split them into extra, unindented lines that read as
garbled/misaligned output. Separately, the continuation text used to be over-eagerly `.strip()`-ed before
being spliced onto the alternative token, silently eating the leading space token-pieces rely on as their
word separator and running two words together (e.g. "...suddenlyand it lived..."). A third, related bug:
the final print read `alt['piece']` directly instead of the `alt_piece` value already computed a few lines
above (which falls back to `alt.get("text", ...)`), so a trace whose alt dicts use the `"text"` key (a
schema this same function already tolerates everywhere else) crashed with a raw KeyError.

No engine, no model, no GPU: `_find_warm` is monkeypatched to report an (already warm) fake port so
`cmd_branch` never calls `spawn_engine`, and `complete_once` is monkeypatched to return canned continuation
text instead of making an HTTP call.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

import clozn.cli.main as cli  # noqa: E402
import clozn.cli.formatting as fmt  # noqa: E402
import clozn.cli.commands.explain as explain  # noqa: E402
import clozn.runs.store as runlog  # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "HOME", str(tmp_path / ".clozn"))
    monkeypatch.setattr(fmt, "COLOR", False)
    monkeypatch.setattr(fmt, "DIM", "")
    monkeypatch.setattr(fmt, "BOLD", "")
    monkeypatch.setattr(fmt, "RST", "")
    # No real engine: report an already-warm fake port so cmd_branch skips spawn_engine entirely, and
    # stub the one-shot completion instead of hitting the network.
    monkeypatch.setattr(explain, "_find_warm", lambda model: (12345, None))
    monkeypatch.setattr(explain, "resolve_model", lambda arg: "fake-model.gguf")
    monkeypatch.setattr(explain, "_flags_for", lambda model: {"chat": False})
    monkeypatch.setattr(explain, "_friendly", lambda model: "fake-model")
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return Path(cli.HOME)


def _write_trace(home: Path, steps, prompt="hi", rid="t1"):
    runlog.record(source="cli", client="cli", model="fake-model", substrate="engine",
                  messages=[{"role": "user", "content": prompt}],
                  response="".join(s.get("piece", s.get("text", "")) for s in steps),
                  trace=runlog.steps_to_trace(steps))


def _args(*, at=None, pick=0, max=32, cpu=False):
    return SimpleNamespace(at=at, pick=pick, max=max, cpu=cpu)


def test_branch_escapes_embedded_newlines_to_a_single_aligned_line(isolated, monkeypatch, capsys):
    monkeypatch.setattr(explain, "complete_once",
                        lambda port, prefix, max_tokens: " and it lived\nhappily ever after.")
    _write_trace(isolated, [
        {"piece": "Once upon a time", "prob": 0.9, "alts": []},
        {"piece": ",\nthere", "prob": 0.4, "alts": [{"piece": ", suddenly", "prob": 0.3}]},
        {"piece": " was a cat.", "prob": 0.95, "alts": []},
    ], prompt="Tell me a short story about a cat.")

    explain.cmd_branch(_args())

    out = capsys.readouterr().out
    lines = out.splitlines()
    original_line = next(l for l in lines if l.strip().startswith("original:"))
    branch_line = next(l for l in lines if l.strip().startswith("branch:"))
    # the raw newline must never split these into extra lines -- it shows up as a literal '\n' escape
    assert "\\n" in original_line
    assert "there was a cat." in original_line          # still present, just on the SAME line
    assert "\\n" in branch_line
    assert "happily ever after." in branch_line


def test_branch_does_not_merge_words_across_the_alt_and_continuation(isolated, monkeypatch, capsys):
    """Regression: the continuation used to be over-eagerly `.strip()`-ed, eating the leading space that
    separates the alt token from the model's continuation and running two words together."""
    monkeypatch.setattr(explain, "complete_once",
                        lambda port, prefix, max_tokens: " and it lived happily ever after.")
    _write_trace(isolated, [
        {"piece": "Once upon a time", "prob": 0.9, "alts": []},
        {"piece": ", there", "prob": 0.4, "alts": [{"piece": ", suddenly", "prob": 0.3}]},
    ], prompt="story")

    explain.cmd_branch(_args())

    out = capsys.readouterr().out
    assert "suddenlyand" not in out
    assert "suddenly and it lived" in out


def test_branch_works_when_alt_uses_text_key_instead_of_piece(isolated, monkeypatch, capsys):
    """Regression: the final print used to read alt['piece'] directly (instead of the already-computed
    alt_piece, which falls back to alt.get('text', ...)) and crashed with a raw KeyError on any trace whose
    alt dicts use the 'text' key -- a schema this same function already tolerates everywhere else."""
    monkeypatch.setattr(explain, "complete_once", lambda port, prefix, max_tokens: " cont text")
    _write_trace(isolated, [
        {"text": "Hello", "conf": 0.9, "alts": []},
        {"text": " world", "conf": 0.3, "alts": [{"text": " there", "prob": 0.3}]},
    ], prompt="hi")

    explain.cmd_branch(_args())     # must not raise

    out = capsys.readouterr().out
    assert "Hello there cont text" in out


def test_oneline_escapes_newlines_and_tabs():
    assert fmt._oneline("a\nb\tc") == "a\\nb\\tc"


def test_oneline_is_a_noop_on_plain_text():
    assert fmt._oneline("plain text") == "plain text"


def test_oneline_handles_none():
    assert fmt._oneline(None) == ""
