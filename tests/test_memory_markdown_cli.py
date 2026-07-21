"""Focused product checks for transactional Markdown memory import/export."""
from __future__ import annotations

import json

import pytest

from clozn.cli import main as cli
from clozn.memory import cards
from clozn.runs.association import client_key, project_key


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    return tmp_path


def test_plain_import_defaults_pending_deduplicates_and_supports_dry_run(isolated, capsys):
    source = isolated / "MEMORY.md"
    source.write_text("# Project memory\n\n- Prefer concise answers\n- Prefer concise answers\n", encoding="utf-8")

    assert cli.main(["memory", "import", str(source), "--dry-run", "--json"]) == 0
    dry = json.loads(capsys.readouterr().out)
    assert dry["added"] == 1 and dry["skipped_duplicates"] == 1
    assert cards.list_cards() == []

    assert cli.main(["memory", "import", str(source), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    stored = cards.list_cards()
    assert report["added"] == 1 and len(stored) == 1
    assert stored[0]["status"] == "pending"
    assert stored[0]["evidence"] == "imported from MEMORY.md"


def test_invalid_import_leaves_existing_store_byte_identical(isolated, capsys):
    assert cards.create("Keep this", status="active")
    before = (isolated / "cards.json").read_bytes()
    source = isolated / "bad.md"
    source.write_text("# Notes\nthis prose is not a bullet\n", encoding="utf-8")

    assert cli.main(["memory", "import", str(source)]) == 1
    assert "memory import failed" in capsys.readouterr().err
    assert (isolated / "cards.json").read_bytes() == before


def test_export_is_private_by_default_and_refuses_overwrite(isolated, capsys):
    card = cards.create("Likes tea", status="active", source_run_id="run_secret",
                        source_turn=0, quoted_span="I like tea", evidence="private evidence")
    target = isolated / "export.md"

    assert cli.main(["memory", "export", str(target)]) == 0
    text = target.read_text(encoding="utf-8")
    assert "Likes tea" in text
    assert "run_secret" not in text and "private evidence" not in text
    before = target.read_bytes()
    assert cli.main(["memory", "export", str(target)]) == 1
    assert "--force" in capsys.readouterr().err
    assert target.read_bytes() == before
    assert card["id"] in json.dumps(cards.list_cards())


def test_versioned_reimport_is_idempotent(isolated, capsys):
    cards.create("Use bullets", status="active")
    target = isolated / "cards.md"
    assert cli.main(["memory", "export", str(target), "--status", "all"]) == 0

    assert cli.main(["memory", "import", str(target), "--preserve-status", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["added"] == 0 and report["skipped_duplicates"] == 1
    assert len(cards.list_cards()) == 1


def test_add_and_rescope_use_opaque_local_keys(isolated, capsys):
    assert cli.main(["memory", "add", "Use concise answers", "--status", "active",
                     "--scope", "app", "--client-id", "editor-one", "--json"]) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["scope"] == {"kind": "app", "key": client_key("editor-one", accept_key=False)}
    assert "editor-one" not in json.dumps(cards.list_cards())

    assert cli.main(["memory", "scope", created["id"], "--scope", "project",
                     "--project", "repo-one", "--label", "My repo", "--json"]) == 0
    updated = json.loads(capsys.readouterr().out)
    assert updated["scope"] == {"kind": "project",
                                 "key": project_key("repo-one", accept_key=False),
                                 "label": "My repo"}
    assert "repo-one" not in json.dumps(cards.list_cards())


def test_export_defaults_to_portable_global_cards(isolated, capsys):
    cards.create("Global preference", status="active")
    cards.create("App preference", status="active",
                 scope={"kind": "app", "key": client_key("editor-one", accept_key=False)})
    portable = isolated / "portable.md"
    complete = isolated / "complete.md"

    assert cli.main(["memory", "export", str(portable)]) == 0
    assert "Global preference" in portable.read_text(encoding="utf-8")
    assert "App preference" not in portable.read_text(encoding="utf-8")

    assert cli.main(["memory", "export", str(complete), "--scope", "all"]) == 0
    assert "App preference" in complete.read_text(encoding="utf-8")


def test_list_shows_scope_without_opaque_key(isolated, capsys):
    cards.create("Project preference", status="pending",
                 scope={"kind": "project", "key": project_key("repo-one", accept_key=False),
                        "label": "My repo"})
    assert cli.main(["memory", "list"]) == 0
    output = capsys.readouterr().out
    assert "pending/project/My repo" in output
    assert "project_" not in output
