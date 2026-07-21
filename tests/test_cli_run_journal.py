"""Gate-0 coverage for the `clozn run` journal's exact rendered input."""
from __future__ import annotations

from clozn.cli.commands import run as cli_run
import clozn.runs.store as runlog


def test_cli_log_keeps_raw_message_and_exact_rendered_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    rendered = "<chat><system>helpful</system><user>hello</user><assistant>"

    cli_run._log_run_cli("tiny-model", "hello", "hi", [], 1.0,
                         finish_reason="stop", final_prompt=rendered)

    saved = runlog.get_run(runlog.list_runs(limit=1)[0]["id"])
    assert saved["messages"] == [{"role": "user", "content": "hello"}]
    assert saved["assembled_messages"] == saved["messages"]
    assert saved["final_prompt"] == rendered


def test_run_turn_passes_the_actual_generation_text_to_the_journal(monkeypatch):
    captured = {}
    monkeypatch.setattr(cli_run, "stream_ar", lambda *_a, **_k: (1, [{"piece": "ok"}], "stop"))
    monkeypatch.setattr(cli_run, "_log_run_cli", lambda *a, **k: captured.update(args=a, kwargs=k))

    rendered = "<templated>raw question</templated>"
    reply = cli_run._run_turn(8080, "autoregressive", rendered, 16, False,
                              "tiny-model", "raw question")

    assert reply == "ok"
    assert captured["kwargs"]["final_prompt"] == rendered
    assert captured["args"][1] == "raw question"
