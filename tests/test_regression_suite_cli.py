"""Focused CLI checks for captured-run regression-suite promotion."""
from __future__ import annotations

import json

import pytest

from clozn.cli import main as cli
import clozn.runs.store as runlog


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _captured_run(response="Contact Ada at ada@example.test"):
    return runlog.record(
        source="openai_api",
        model="local-model",
        messages=[{"role": "user", "content": "Email ada@example.test"}],
        response=response,
        meta={"max_tokens": 64, "sampler_mode": "sample", "temperature": 0.7,
              "top_p": 0.9, "seed": 42},
        identity={"model_sha256": "a" * 64, "template_fingerprint": "template-one"},
    )


def test_create_redact_freeze_and_verify_source(isolated, capsys):
    run_id = _captured_run()
    path = isolated / "captured.json"

    assert cli.main(["suite", "create", str(path), "--from-runs", run_id,
                     "--redact", "ada@example.test", "--freeze"]) == 0
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["state"] == "frozen"
    assert artifact["cases"][0]["messages"][-1]["content"] == "Email [REDACTED]"
    assert artifact["cases"][0]["expect"]["equals"] == "Contact Ada at [REDACTED]"
    assert artifact["cases"][0]["sampling"]["seed"] == 42
    assert "client_key" not in json.dumps(artifact)

    assert cli.main(["suite", "verify", str(path), "--source"]) == 0
    assert "source run(s)" in capsys.readouterr().out


def test_draft_must_be_frozen_before_run_unless_explicitly_allowed(isolated, capsys, monkeypatch):
    from clozn.testkit import ci

    run_id = _captured_run(response="Stable answer")
    path = isolated / "draft.json"
    assert cli.main(["suite", "create", str(path), "--from-runs", run_id]) == 0
    capsys.readouterr()

    constructed = []

    class FakeClient:
        def __init__(self, url, headers=None):
            constructed.append((url, headers))

        def chat(self, prompt, **kwargs):
            return {"id": "run_fresh", "response": "Stable answer", "trace": {}, "memory": {},
                    "meta": {"max_tokens": kwargs.get("max_tokens")}}

    monkeypatch.setattr(ci, "Client", FakeClient)
    assert cli.main(["suite", "run", str(path)]) == 1
    assert "not frozen" in capsys.readouterr().err
    assert constructed == []

    assert cli.main(["suite", "run", str(path), "--allow-draft",
                     "--client-id", "editor-one", "--project", "repo-one"]) == 0
    assert constructed == [("http://127.0.0.1:8080", {
        "X-Clozn-Client-Id": "editor-one", "X-Clozn-Project-Id": "repo-one",
    })]


def test_freeze_replaces_its_draft_but_refuses_unrelated_overwrite(isolated, capsys):
    run_id = _captured_run()
    draft = isolated / "draft.json"
    occupied = isolated / "occupied.json"
    occupied.write_text("keep", encoding="utf-8")
    assert cli.main(["suite", "create", str(draft), "--from-runs", run_id]) == 0

    assert cli.main(["suite", "freeze", str(draft)]) == 0
    assert json.loads(draft.read_text(encoding="utf-8"))["state"] == "frozen"

    second = isolated / "second.json"
    assert cli.main(["suite", "create", str(second), "--from-runs", run_id]) == 0
    assert cli.main(["suite", "freeze", str(second), "--out", str(occupied)]) == 1
    assert "--force" in capsys.readouterr().err
    assert occupied.read_text(encoding="utf-8") == "keep"
