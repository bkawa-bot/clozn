"""Gate 0 request-path regressions for CLI rendered prompts and think-tag hygiene."""
from __future__ import annotations

import json

from clozn.cli.commands import run as run_command


def _line(payload):
    if payload == "[DONE]":
        return b"data: [DONE]\n\n"
    return ("data: " + json.dumps(payload) + "\n\n").encode("utf-8")


def test_cli_stream_prints_and_returns_only_public_answer_but_journals_raw(monkeypatch, capsys):
    class CliResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter([
                _line({"type": "tokens_committed", "items": [
                    {"pos": 0, "id": 1, "piece": "<think>secret</thi", "conf": .6},
                    {"pos": 1, "id": 2, "piece": "nk>an", "conf": .8},
                    {"pos": 2, "id": 3, "piece": "swer", "conf": .9},
                ]}),
                _line({"type": "gen_finished", "reason": "eos"}),
                _line("[DONE]"),
            ])

    captured = {}
    monkeypatch.setattr(run_command.urllib.request, "urlopen", lambda *_a, **_kw: CliResponse())
    monkeypatch.setattr(run_command, "_log_run_cli",
                        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs))
    response = run_command._run_turn(
        1234, "autoregressive", "prompt", 20, False, "model", "user prompt"
    )
    out = capsys.readouterr().out
    assert response == "answer"
    assert out.strip() == "answer"
    assert "secret" not in out and "think" not in out
    assert captured["args"][2] == "<think>secret</think>answer"


def test_cli_journal_keeps_user_message_and_exact_rendered_prompt(monkeypatch):
    recorded = []
    monkeypatch.setattr("clozn.runs.store.record", lambda **kwargs: recorded.append(kwargs) or "run_cli")
    monkeypatch.setattr(run_command, "_identity_for_port", lambda _port: {"model_sha256": "a" * 64})

    rendered = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"
    run_command._log_run_cli("model", "hello", "hi", [], 1.0, finish_reason="stop", port=123,
                             final_prompt=rendered)

    assert recorded[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert recorded[0]["assembled_messages"] == [{"role": "user", "content": "hello"}]
    assert recorded[0]["final_prompt"] == rendered
    assert recorded[0]["identity"]["model_sha256"] == "a" * 64

