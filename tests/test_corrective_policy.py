from __future__ import annotations

import pytest

from clozn.behavior import corrective_retries as policy
from clozn.profiles import store as profiles


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "_PATH", str(tmp_path / "corrective.json"))
    profile_store = profiles.ProfileStore(str(tmp_path / "profiles"))
    monkeypatch.setattr(policy, "_profile_store", lambda: profile_store)
    return profile_store


def test_session_policy_activates_injects_and_undoes(isolated):
    key = "session_0123456789abcdef01234567"
    activated = policy.activate("session", key, "less-verbose", now=100.0)
    assert policy.session_presets(key, now=101.0) == ["less-verbose"]
    delivered = [{"role": "user", "content": "Tell me"}]
    effective = policy.inject(delivered, policy.effective_presets(session_key=key, now=101.0))
    assert delivered == [{"role": "user", "content": "Tell me"}]
    assert effective[0]["role"] == "system" and "answer concisely" in effective[0]["content"]

    undone = policy.undo(activated["undo_id"], now=102.0)
    assert undone["status"] == "undone"
    assert policy.session_presets(key, now=103.0) == []


def test_stale_session_undo_refuses_newer_policy(isolated):
    key = "session_0123456789abcdef01234567"
    first = policy.activate("session", key, "less-verbose", now=100.0)
    policy.activate("session", key, "more-concrete", now=101.0)
    with pytest.raises(policy.CorrectivePolicyError, match="stale undo"):
        policy.undo(first["undo_id"], now=102.0)
    assert policy.session_presets(key, now=103.0) == ["less-verbose", "more-concrete"]


def test_profile_policy_travels_in_bundle_and_undoes(isolated):
    isolated.save(profiles.new_profile("work"))
    activated = policy.activate("profile", "work", "use-context", now=100.0)
    assert isolated.load("work")["response_policies"] == ["use-context"]
    assert policy.profile_presets("work") == ["use-context"]

    policy.undo(activated["undo_id"], now=101.0)
    assert isolated.load("work")["response_policies"] == []
