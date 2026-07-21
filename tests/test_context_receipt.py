"""Phase 2.4: delivered/survived context receipts and CLI inspection."""
from __future__ import annotations

import json

import clozn.runs.store as runlog
from clozn.cli import main as cli
from clozn.runs.context_receipt import build_context_receipt


def test_context_receipt_keeps_labels_and_does_not_mislabel_output_cutoff():
    delivered = [{"role": "system", "content": "caller rule"},
                 {"role": "user", "content": "question"}]
    assembled = [{"role": "system", "content": "caller rule\n\nmemory card"},
                 {"role": "user", "content": "question"}]
    receipt = build_context_receipt(
        messages=delivered, assembled_messages=assembled, final_prompt="<rendered exact>",
        finish_reason="length", meta={"max_tokens": 8, "n_ctx": 128, "prompt_tokens": 31},
        trace={"tokens": ["a", "b"]},
    )

    assert receipt["delivered"]["messages"] == delivered
    assert receipt["survived"]["assembled_messages"] == assembled
    assert receipt["survived"]["final_prompt"] == "<rendered exact>"
    assert receipt["input_truncated"] is False
    assert receipt["output_cut_off"] is True
    assert receipt["warnings"][0]["code"] == "output_truncated"
    assert receipt["limits"] == {
        "prompt_tokens": 31, "context_window_tokens": 128,
        "requested_max_tokens": 8, "generated_tokens": 2,
    }


def test_run_store_persists_context_receipt_and_summary_warning(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    rid = runlog.record(
        source="openai_api", messages=[{"role": "user", "content": "hello"}],
        assembled_messages=[{"role": "system", "content": "memory"},
                            {"role": "user", "content": "hello"}],
        final_prompt="EXACT PROMPT", response="partial", finish_reason="length",
        meta={"max_tokens": 2, "n_ctx": 64, "prompt_tokens": 9},
        trace=[{"piece": "par"}, {"piece": "tial"}],
    )
    run = runlog.get_run(rid)
    assert run["context_receipt"]["survived"]["final_prompt"] == "EXACT PROMPT"
    assert run["warnings"][0]["code"] == "output_truncated"
    assert runlog.list_runs(1)[0]["warnings"] == run["warnings"]


def test_context_last_uses_latest_organic_run_and_prints_both_sections(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    organic = runlog.record(
        source="cli", client="cli", messages=[{"role": "user", "content": "raw question"}],
        assembled_messages=[{"role": "user", "content": "raw question"}],
        final_prompt="RENDERED QUESTION", response="partial", finish_reason="length",
        meta={"max_tokens": 4}, started=1.0,
    )
    runlog.record(
        source="replay", parent_run_id=organic,
        messages=[{"role": "user", "content": "internal replay"}], response="child", started=2.0,
    )

    assert cli.main(["context", "last"]) == 0
    output = capsys.readouterr().out
    assert organic in output
    assert "internal replay" not in output
    assert "DELIVERED" in output and "raw question" in output
    assert "SURVIVED" in output and "RENDERED QUESTION" in output
    assert "WARNING" in output and "reply may be incomplete" in output

    assert cli.main(["context", "last", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == organic
    assert payload["context_receipt"]["delivered"]["label"] == "delivered"
    assert payload["context_receipt"]["survived"]["label"] == "survived"
