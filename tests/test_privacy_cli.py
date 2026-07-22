"""Focused user-facing checks for Phase 3.5 privacy controls."""
from __future__ import annotations

import json

import pytest

from clozn import network_policy
from clozn.cli import main as cli
from clozn.cli.commands import doctor
from clozn.runs import retention_policy, store


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(network_policy, "POLICY_PATH", str(tmp_path / "network_policy.json"))
    monkeypatch.setattr(network_policy, "LEDGER_PATH", str(tmp_path / "outbound_attempts.jsonl"))
    monkeypatch.delenv(network_policy.POLICY_ENV, raising=False)
    monkeypatch.delenv(network_policy.LEDGER_ENV, raising=False)
    monkeypatch.delenv(network_policy.LOCAL_ONLY_ENV, raising=False)
    monkeypatch.setattr(retention_policy, "POLICY_PATH", str(tmp_path / "retention_policy.json"))
    monkeypatch.delenv(retention_policy.POLICY_ENV, raising=False)
    return tmp_path


def _record(secret="private prompt", *, started=1.0):
    return store.record(
        source="openai_api", client="sdk", model="local-model", substrate="engine",
        messages=[{"role": "user", "content": secret}], response="private response",
        trace={"tokens": ["private", " response"], "confidence": [0.8, 0.9]},
        meta={"prompt_tokens": 3}, started=started, ended=started + 1.0,
    )


def test_local_only_cli_and_doctor_strict_check(isolated, capsys):
    assert cli.main(["privacy", "local-only", "on", "--json"]) == 0
    configured = json.loads(capsys.readouterr().out)
    assert configured["configured"] is True and configured["effective"] is True

    check = doctor._check_offline()
    assert check["status"] == "OK"
    assert check["evidence"]["probe_blocked"] is True
    assert check["evidence"]["probe_recorded"] is True

    assert cli.main(["privacy", "local-only", "off"]) == 0
    capsys.readouterr()
    assert doctor._check_offline()["status"] == "FAIL"


def test_run_redact_delete_and_retention_cli(isolated, capsys):
    oldest = _record("oldest secret", started=1.0)
    newest = _record("newest secret", started=2.0)

    assert cli.main(["runs", "retention", "--keep", "1", "--json"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dry_run"] is True and plan["run_ids"] == [oldest]
    assert store.get_run(oldest) is not None

    assert cli.main(["runs", "redact", newest, "--json"]) == 0
    redaction = json.loads(capsys.readouterr().out)
    assert redaction["ok"] is True and store.get_run(newest)["flags"] == ["redacted"]

    assert cli.main(["runs", "delete", newest]) == 1
    assert "--yes" in capsys.readouterr().err
    assert cli.main(["runs", "delete", newest, "--yes", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert store.get_run(newest) is None


def test_missing_run_mutations_are_user_errors(isolated, capsys):
    assert cli.main(["runs", "redact", "run_missing"]) == 1
    assert "was not found" in capsys.readouterr().err
    assert cli.main(["runs", "delete", "run_missing", "--yes"]) == 1
    assert "was not found" in capsys.readouterr().err


def test_telemetry_cli_omits_content_by_default_and_refuses_overwrite(isolated, capsys):
    run_id = _record()
    target = isolated / "run.jsonl"

    assert cli.main(["runs", "export-otel", str(target), "--from-runs", run_id]) == 0
    capsys.readouterr()
    text = target.read_text(encoding="utf-8")
    assert "private prompt" not in text and "private response" not in text
    assert "clozn.content.policy" in text and "omitted" in text

    before = target.read_bytes()
    assert cli.main(["runs", "export-otel", str(target), "--from-runs", run_id]) == 1
    assert "--force" in capsys.readouterr().err
    assert target.read_bytes() == before

    assert cli.main(["runs", "export-otel", "-", "--from-runs", run_id,
                     "--redact", "private"]) == 1
    assert "require include_content" in capsys.readouterr().err


def test_runs_redact_literal_cli_scrubs_only_the_given_text(isolated, capsys):
    run_id = _record("my token abc-123-secret is here")
    assert cli.main(["runs", "redact", run_id, "--literal", "abc-123-secret", "--json"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["redaction"]["schema"] == "clozn.run_literal_redaction.v1"
    redacted = store.get_run(run_id)
    assert redacted["messages"][0]["content"] == "my token [REDACTED] is here"
    assert redacted["client"] == "sdk"  # unlike a full redact, everything else survives


def test_runs_delete_refuses_children_then_succeeds_with_cascade(isolated, capsys):
    parent = _record("parent", started=1.0)
    child = store.record(
        source="replay", client="sdk", model="local-model", substrate="engine",
        messages=[{"role": "user", "content": "child"}], response="ok",
        parent_run_id=parent, trace={"tokens": ["c"], "confidence": [0.5]}, started=2.0, ended=2.0,
    )

    assert cli.main(["runs", "delete", parent, "--yes"]) == 1
    err = capsys.readouterr().err
    assert "refused" in err and child in err and "--cascade" in err
    assert store.get_run(parent) is not None

    assert cli.main(["runs", "delete", parent, "--yes", "--cascade", "--json"]) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert set(receipt["deleted_run_ids"]) == {parent, child}
    assert store.get_run(parent) is None
    assert store.get_run(child) is None


def test_privacy_retention_cli_set_show_and_clear(isolated, capsys):
    assert cli.main(["privacy", "retention", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["days"] is None

    assert cli.main(["privacy", "retention", "--days", "14", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["days"] == 14

    assert cli.main(["privacy", "retention"]) == 0
    assert "14 day" in capsys.readouterr().out

    assert cli.main(["privacy", "retention", "--off", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["days"] is None


def test_privacy_retention_cli_rejects_conflicting_flags(isolated):
    assert cli.main(["privacy", "retention", "--days", "14", "--off"]) == 1
