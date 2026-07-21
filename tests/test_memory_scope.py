"""Pure tests for global/app/project memory scope selection."""
from dataclasses import FrozenInstanceError

import pytest

from clozn.memory import scope


def test_memory_scope_is_immutable_and_validates_opaque_keys():
    current = scope.MemoryScope(app_key="client_ab12", project_key="project_cd34")
    with pytest.raises(FrozenInstanceError):
        current.app_key = "changed"
    for invalid in ("", "has space", "line\nbreak", "snowman_☃", "x" * 129):
        with pytest.raises(scope.MemoryScopeError):
            scope.memory_scope(app_key=invalid)


def test_validated_card_scope_helpers_return_exact_shapes():
    assert scope.global_scope() == {"kind": "global"}
    assert scope.app_scope("app_a", label="Editor") == {
        "kind": "app", "key": "app_a", "label": "Editor",
    }
    assert scope.project_scope("project_b") == {"kind": "project", "key": "project_b"}
    with pytest.raises(scope.MemoryScopeError):
        scope.card_scope("global", key="not-allowed")
    with pytest.raises(scope.MemoryScopeError):
        scope.app_scope("app_a", label="")


def test_missing_and_malformed_card_scope_are_legacy_global_reads():
    malformed = [
        None,
        "app",
        {},
        {"kind": "unknown"},
        {"kind": "app"},
        {"kind": "app", "key": "bad key"},
        {"kind": "project", "key": "p", "unexpected": True},
        {"kind": "global", "key": "p"},
    ]
    for value in malformed:
        assert scope.normalize_card_scope(value) == {"kind": "global"}
        assert scope.is_eligible({"scope": value}, scope.MemoryScope())


def test_normalize_scope_can_reject_malformed_writer_input():
    assert scope.normalize_scope(
        {"kind": "app", "key": "app_a"}, legacy_global=False
    ) == {"kind": "app", "key": "app_a"}
    with pytest.raises(scope.MemoryScopeError):
        scope.normalize_scope({"kind": "app"}, legacy_global=False)


def test_eligible_union_is_global_then_app_then_project_and_stable_within_rank():
    cards = [
        {"id": "p1", "scope": scope.project_scope("project_here")},
        {"id": "a1", "scope": scope.app_scope("app_here")},
        {"id": "g1"},                                      # legacy global
        {"id": "a-other", "scope": scope.app_scope("app_elsewhere")},
        {"id": "g2", "scope": {"kind": "broken"}},     # malformed -> global
        {"id": "p2", "scope": scope.project_scope("project_here", label="Repo")},
        {"id": "a2", "scope": scope.app_scope("app_here", label="Editor")},
        {"id": "p-other", "scope": scope.project_scope("other")},
        {"id": "g3", "scope": scope.global_scope()},
    ]
    original = [dict(card) for card in cards]

    selected = scope.eligible_cards(
        cards, scope.MemoryScope(app_key="app_here", project_key="project_here"))

    assert [card["id"] for card in selected] == ["g1", "g2", "g3", "a1", "a2", "p1", "p2"]
    assert cards == original


def test_matching_is_exact_and_missing_request_keys_exclude_scoped_cards():
    app = {"scope": scope.app_scope("App_Key")}
    project = {"scope": scope.project_scope("project")}
    assert scope.is_eligible(app, scope.MemoryScope(app_key="App_Key"))
    assert not scope.is_eligible(app, scope.MemoryScope(app_key="app_key"))
    assert not scope.is_eligible(app, scope.MemoryScope())
    assert not scope.is_eligible(project, scope.MemoryScope())
