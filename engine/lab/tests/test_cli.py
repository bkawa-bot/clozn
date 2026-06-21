"""Tests for the ``cloze`` CLI front door. Torch-free: everything runs on FakeAdapter."""

import pytest

from cloze_lab.cli import build_adapter, main
from cloze_lab.models.base import ModelAdapter


def test_build_adapter_fake_is_torch_free() -> None:
    adapter = build_adapter("fake")
    assert isinstance(adapter, ModelAdapter)


def test_build_adapter_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        build_adapter("bogus")


def test_run_prints_text_and_stats(capsys) -> None:
    main(["run", "hello cloze", "--model", "fake", "--max-new", "6", "--steps", "3"])
    out = capsys.readouterr()
    assert out.out.strip()  # the generated text lands on stdout
    assert "[fake]" in out.err  # the stats footer lands on stderr (keeps stdout clean)
    assert "tok/s" in out.err


def test_run_block_mode(capsys) -> None:
    main(["run", "hello cloze", "--model", "fake", "--max-new", "8", "--steps", "4", "--block-len", "4"])
    assert capsys.readouterr().out.strip()


def test_run_no_chat_flag_is_harmless_for_fake(capsys) -> None:
    # --chat/--no-chat parse fine; chat templating is dream-only so fake ignores it.
    main(["run", "hello cloze", "--model", "fake", "--no-chat", "--max-new", "4", "--steps", "2"])
    assert capsys.readouterr().out.strip()


def test_run_with_revise_flag(capsys) -> None:
    # --revise wires a RemaskLowConf reviser through to generate; tau=1.0 forces
    # revisions on the fake model. Just exercises the path end-to-end.
    main(["run", "hello cloze", "--model", "fake", "--revise", "1.0", "--max-new", "4", "--steps", "10"])
    assert capsys.readouterr().out.strip()


def test_run_effort_preset(capsys) -> None:
    # --effort applies a (steps, block_len, cache) bundle; runs end-to-end on fake.
    main(["run", "hello cloze", "--model", "fake", "--effort", "fast", "--max-new", "8"])
    assert capsys.readouterr().out.strip()


def test_infill_subcommand(capsys) -> None:
    # `cloze infill PREFIX SUFFIX` fills the middle; prints the reconstructed sequence.
    main(["infill", "def add", "return x", "--model", "fake", "--gap", "4", "--steps", "8"])
    out = capsys.readouterr()
    assert out.out.strip()
    assert "infill" in out.err and "filled" in out.err


def test_no_subcommand_errors() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_bench_subcommand_delegates(capsys) -> None:
    # `cloze bench` forwards to the report CLI; fake adapter prints the markdown table.
    main(["bench", "--model", "fake", "--max-new", "6", "--steps", "3"])
    assert "token-match" in capsys.readouterr().out  # the honesty column from the bench table
