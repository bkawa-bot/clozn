"""Direct, model-free tests for trusted card-scope mutation at the substrate boundary."""
from __future__ import annotations

import pytest

import clozn.memory.cards as memory_cards
from clozn.memory.scope import MemoryScope
from clozn.server import app as server_app


class _Mem:
    rules = []
    prefix = None
    memory_strength = 1.0


@pytest.fixture
def scoped_substrate(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    sub = object.__new__(server_app.Substrate)
    sub._mem = _Mem()
    sub._cards_migrated = True
    return sub


def test_add_defaults_to_global_and_accepts_only_a_public_kind(scoped_substrate):
    default = scoped_substrate._memory("/memory/add", {"text": "be concise"})
    explicit = scoped_substrate._memory(
        "/memory/add", {"text": "use metric units", "scope": "GLOBAL"})

    assert default["scope"] == {"kind": "global"}
    assert explicit["scope"] == {"kind": "global"}


def test_add_binds_app_and_project_to_the_trusted_request_context(scoped_substrate):
    request_scope = MemoryScope(app_key="app:trusted", project_key="project:trusted")
    app = scoped_substrate._memory("/memory/add", {
        "text": "app preference", "scope": "app", "_memory_scope": request_scope,
    })
    project = scoped_substrate._memory("/memory/add", {
        "text": "project preference", "scope": "project", "_memory_scope": request_scope,
    })

    assert app["scope"] == {"kind": "app", "key": "app:trusted"}
    assert project["scope"] == {"kind": "project", "key": "project:trusted"}


@pytest.mark.parametrize("body", [
    {"text": "bad", "scope": {"kind": "app", "key": "attacker"}},
    {"text": "bad", "scope": "app", "_memory_scope": {"app_key": "attacker"}},
    {"text": "bad", "scope": "app", "app_key": "attacker"},
    {"text": "bad", "scope": "project", "_memory_scope": MemoryScope(app_key="app-only")},
])
def test_add_rejects_raw_keys_and_missing_or_fake_private_context(scoped_substrate, body):
    result = scoped_substrate._memory("/memory/add", body)

    assert result["ok"] is False
    assert memory_cards.list_cards() == []


def test_scope_endpoint_rebinds_existing_card_without_accepting_public_keys(scoped_substrate):
    card = scoped_substrate._memory("/memory/add", {"text": "scoped later"})
    request_scope = MemoryScope(project_key="project:trusted")

    changed = scoped_substrate._memory("/memory/scope", {
        "id": card["id"], "scope": "project", "_memory_scope": request_scope,
    })
    assert changed["scope"] == {"kind": "project", "key": "project:trusted"}

    missing = scoped_substrate._memory("/memory/scope", {"id": card["id"]})
    assert missing["ok"] is False
    assert memory_cards.get(card["id"])["scope"]["kind"] == "project"

    refused = scoped_substrate._memory("/memory/scope", {
        "id": card["id"], "scope": "project", "project_key": "attacker",
    })
    assert refused["ok"] is False
    assert memory_cards.get(card["id"])["scope"] == {
        "kind": "project", "key": "project:trusted",
    }

    global_card = scoped_substrate._memory("/memory/scope", {
        "id": card["id"], "scope": "global",
    })
    assert global_card["scope"] == {"kind": "global"}


def test_gateway_injects_header_derived_scope_without_mutating_public_body(monkeypatch):
    from clozn.runs.association import client_key, project_key

    class CaptureSub:
        seen = None

        def handle(self, path, body):
            self.seen = (path, body)
            return {"ok": True}

    sub = CaptureSub()
    monkeypatch.setattr(server_app, "SUB", sub)
    handler = object.__new__(server_app.make_handler())
    handler.headers = {
        "X-Clozn-Client-Id": "editor-one",
        "X-Clozn-Project-Id": "repo-one",
    }
    handler._json = lambda status, payload: None
    public = {"text": "project preference", "scope": "project"}

    handler._dispatch_post("/memory/add", public)

    assert public == {"text": "project preference", "scope": "project"}
    path, private = sub.seen
    assert path == "/memory/add"
    assert private["_memory_scope"] == MemoryScope(
        app_key=client_key("editor-one", accept_key=False),
        project_key=project_key("repo-one", accept_key=False),
    )
