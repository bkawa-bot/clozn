"""Focused tests for privacy-preserving request association primitives."""
from __future__ import annotations

import pytest

from clozn.runs import association


@pytest.fixture(autouse=True)
def stable_association_secret(monkeypatch):
    monkeypatch.setattr("clozn.runs.store.association_secret", lambda: b"test-secret")


def test_project_key_is_stable_opaque_and_accepts_an_existing_key():
    key = association.project_key("clozn")

    assert key == association.project_key("clozn")
    assert key.startswith(association.PROJECT_PREFIX)
    assert len(key) == len(association.PROJECT_PREFIX) + 24
    assert "clozn" not in key
    assert association.project_key(key.upper()) == key


def test_request_project_hashes_only_the_explicit_header():
    headers = {"X-Clozn-Project-Id": "workspace-one", "User-Agent": "editor/1.0"}

    assert association.request_project(headers) == association.project_key(
        "workspace-one", accept_key=False
    )
    assert association.request_project({"User-Agent": "editor/1.0"}) is None


def test_request_explicit_client_never_falls_back_to_user_agent():
    assert association.request_explicit_client({"User-Agent": "editor/1.0"}) is None
    assert association.request_explicit_client(
        {"X-Clozn-Client-Id": "aider", "User-Agent": "editor/1.0"}
    ) == association.client_key("aider", accept_key=False)


@pytest.mark.parametrize(
    "value",
    ["", "contains space", "contains\nnewline", "é", "x" * 129],
)
def test_project_header_uses_existing_selector_validation(value):
    with pytest.raises(association.AssociationValueError) as exc:
        association.validate_request_headers({"X-Clozn-Project-Id": value})

    assert exc.value.field == "X-Clozn-Project-Id"


def test_project_header_accepts_visible_ascii_at_the_length_limit():
    association.validate_request_headers({"X-Clozn-Project-Id": "x" * 128})
