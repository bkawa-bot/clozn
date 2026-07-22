"""Focused tests for the persisted age-based retention policy (Phase 3.5)."""
from __future__ import annotations

import pytest

from clozn.runs import retention_policy as policy


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "POLICY_PATH", str(tmp_path / "retention_policy.json"))
    monkeypatch.delenv(policy.POLICY_ENV, raising=False)
    return tmp_path


def test_no_policy_file_means_no_policy(isolated):
    assert policy.get_policy() == {"days": None}


def test_set_and_get_policy_round_trips(isolated):
    saved = policy.set_policy(30)
    assert saved["days"] == 30
    assert saved["path"] == str(isolated / "retention_policy.json")

    fetched = policy.get_policy()
    assert fetched["days"] == 30
    assert fetched["updated_at"] == saved["updated_at"]


def test_set_policy_none_clears_it(isolated):
    policy.set_policy(30)
    cleared = policy.set_policy(None)
    assert cleared["days"] is None
    assert policy.get_policy()["days"] is None


@pytest.mark.parametrize("days", [0, -1, True, 1.5, "30"])
def test_set_policy_rejects_invalid_days(isolated, days):
    with pytest.raises(ValueError):
        policy.set_policy(days)


def test_malformed_policy_file_reads_as_no_policy(isolated):
    isolated.joinpath("retention_policy.json").write_text("not json", encoding="utf-8")
    assert policy.get_policy() == {"days": None}


def test_policy_file_with_bad_days_value_reads_as_no_policy(isolated):
    isolated.joinpath("retention_policy.json").write_text(
        '{"days": -5, "updated_at": "x"}', encoding="utf-8")
    assert policy.get_policy()["days"] is None


def test_environment_path_override(isolated, monkeypatch, tmp_path):
    other = tmp_path / "elsewhere.json"
    monkeypatch.setenv(policy.POLICY_ENV, str(other))
    policy.set_policy(7)
    assert other.is_file()
    assert policy.get_policy()["days"] == 7
    assert not (isolated / "retention_policy.json").exists()
