from __future__ import annotations

from datetime import datetime, timezone

import pytest

from clozn.cli.commands.connect import configure_aider, undo_aider


def test_existing_aider_config_is_backed_up_then_update_is_idempotent(tmp_path):
    path = tmp_path / ".aider.conf.yml"
    original = "dark-mode: true\nmodel: old/model\n"
    path.write_text(original, encoding="utf-8")
    clock = datetime(2026, 7, 21, 12, 34, 56, 123456, tzinfo=timezone.utc)

    report = configure_aider(
        path, base_url="http://127.0.0.1:9000", model="clozn", api_key="local-key",
        state_path=tmp_path / "state.json", now=clock)

    backup = tmp_path / ".aider.conf.yml.bak-20260721T123456.123456Z"
    assert report["status"] == "updated"
    assert report["backup"] == str(backup)
    assert backup.read_text(encoding="utf-8") == original
    configured = path.read_text(encoding="utf-8")
    assert "dark-mode: true" in configured
    assert 'model: "openai/clozn"' in configured
    assert 'openai-api-base: "http://127.0.0.1:9000/v1"' in configured

    again = configure_aider(
        path, base_url="http://127.0.0.1:9000/v1", model="openai/clozn", api_key="local-key",
        state_path=tmp_path / "state.json")
    assert again["status"] == "unchanged"
    assert list(tmp_path.glob("*.bak-*")) == [backup]


def test_dry_run_does_not_create_config_or_backup(tmp_path):
    path = tmp_path / ".aider.conf.yml"
    report = configure_aider(
        path, base_url="http://localhost:8080/v1", model="clozn", api_key="local",
        state_path=tmp_path / "state.json", dry_run=True)
    assert report["status"] == "dry_run"
    assert not path.exists()
    assert not list(tmp_path.iterdir())


def test_ambiguous_duplicate_key_fails_without_mutation(tmp_path):
    path = tmp_path / ".aider.conf.yml"
    original = "model: one\nmodel: two\n"
    path.write_text(original, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        configure_aider(path, base_url="http://localhost:8080", model="clozn", api_key="local",
                        state_path=tmp_path / "state.json")
    assert path.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob("*.bak-*"))


def test_undo_restores_existing_config(tmp_path):
    path = tmp_path / ".aider.conf.yml"
    state = tmp_path / "state.json"
    original = b"dark-mode: true\r\nmodel: old/model\r\n"
    path.write_bytes(original)
    configure_aider(path, base_url="http://localhost:8080", model="clozn", api_key="local",
                    state_path=state)

    report = undo_aider(state)

    assert report["status"] == "restored"
    assert path.read_bytes() == original
    assert not state.exists()


def test_undo_removes_config_created_from_absent_target(tmp_path):
    path = tmp_path / ".aider.conf.yml"
    state = tmp_path / "state.json"
    configure_aider(path, base_url="http://localhost:8080", model="clozn", api_key="local",
                    state_path=state)

    report = undo_aider(state)

    assert report["status"] == "removed"
    assert not path.exists()
    assert not state.exists()


def test_undo_refuses_after_external_edit(tmp_path):
    path = tmp_path / ".aider.conf.yml"
    state = tmp_path / "state.json"
    path.write_text("model: old/model\n", encoding="utf-8")
    configure_aider(path, base_url="http://localhost:8080", model="clozn", api_key="local",
                    state_path=state)
    path.write_text(path.read_text(encoding="utf-8") + "dark-mode: true\n", encoding="utf-8")
    externally_edited = path.read_bytes()

    with pytest.raises(ValueError, match="changed after"):
        undo_aider(state)

    assert path.read_bytes() == externally_edited
    assert state.is_file()
