"""Focused tests for local-only enforcement and the privacy-safe outbound ledger."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from clozn import network_policy as policy


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "POLICY_PATH", str(tmp_path / "network_policy.json"))
    monkeypatch.setattr(policy, "LEDGER_PATH", str(tmp_path / "outbound_attempts.jsonl"))
    monkeypatch.delenv(policy.POLICY_ENV, raising=False)
    monkeypatch.delenv(policy.LEDGER_ENV, raising=False)
    monkeypatch.delenv(policy.LOCAL_ONLY_ENV, raising=False)
    return tmp_path


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def test_persisted_flag_and_environment_override(isolated, monkeypatch):
    assert policy.local_only_enabled() is False
    saved = policy.set_local_only(True)
    assert saved["configured"] is True and saved["effective"] is True

    monkeypatch.setenv(policy.LOCAL_ONLY_ENV, "off")
    assert policy.local_only_enabled() is False
    assert policy.set_local_only(True)["environment_override"] is True


def test_malformed_persisted_policy_fails_closed(isolated):
    isolated.joinpath("network_policy.json").write_text("not json", encoding="utf-8")
    assert policy.local_only_enabled() is True


def test_local_only_blocks_external_before_transport_and_ledgers_only_metadata(isolated, monkeypatch):
    policy.set_local_only(True)
    called = []
    monkeypatch.setattr(policy, "_transport_urlopen", lambda *_a, **_k: called.append(True))
    secret = "PROMPT-SECRET-42"
    request = urllib.request.Request(
        f"https://api.example.test/v1/chat?token={secret}",
        data=json.dumps({"prompt": secret}).encode(),
        headers={"Authorization": f"Bearer {secret}"},
        method="POST",
    )

    with pytest.raises(policy.LocalOnlyViolation) as exc:
        policy.guarded_urlopen(request)

    assert called == []
    assert exc.value.host == "api.example.test"
    raw = isolated.joinpath("outbound_attempts.jsonl").read_text(encoding="utf-8")
    assert secret not in raw and "/v1/chat" not in raw and "Authorization" not in raw
    row = json.loads(raw)
    assert row["destination_category"] == "external"
    assert row["host"] == "api.example.test"
    assert row["operation"] == "http_post"
    assert row["outcome"] == "blocked"


@pytest.mark.parametrize("url", [
    "http://127.0.0.1:8080/health",
    "http://[::1]:8080/health",
    "http://studio.localhost:8080/health",
])
def test_local_only_allows_loopback_and_records_success(isolated, monkeypatch, url):
    policy.set_local_only(True)
    calls = []

    def opened(*args, **kwargs):
        calls.append((args, kwargs))
        return _Response()

    monkeypatch.setattr(policy, "_transport_urlopen", opened)

    assert policy.guarded_urlopen(url) is not None
    assert len(calls) == 1
    row = policy.read_outbound_attempts()[-1]
    assert row["destination_category"] == "loopback"
    assert row["outcome"] == "succeeded"


def test_private_network_literal_is_not_treated_as_same_host(isolated, monkeypatch):
    policy.set_local_only(True)
    monkeypatch.setattr(policy, "_transport_urlopen", lambda *_a, **_k: pytest.fail("transport opened"))

    with pytest.raises(policy.LocalOnlyViolation):
        policy.guarded_urlopen("http://192.168.1.5/model")

    assert policy.read_outbound_attempts()[-1]["destination_category"] == "private_network"


def test_remote_file_share_cannot_bypass_local_only(isolated, monkeypatch):
    policy.set_local_only(True)
    monkeypatch.setattr(policy, "_transport_urlopen", lambda *_a, **_k: pytest.fail("transport opened"))

    with pytest.raises(policy.LocalOnlyViolation):
        policy.guarded_urlopen("file://fileserver/private/model.gguf")

    assert policy.read_outbound_attempts()[-1]["destination_category"] == "external"


def test_failed_allowed_request_records_error_type_not_error_text(isolated, monkeypatch):
    secret = "transport-secret"

    def failed(*_args, **_kwargs):
        raise urllib.error.URLError(secret)

    monkeypatch.setattr(policy, "_transport_urlopen", failed)
    with pytest.raises(urllib.error.URLError):
        policy.guarded_urlopen("https://example.test/private/path?key=hidden")

    raw = isolated.joinpath("outbound_attempts.jsonl").read_text(encoding="utf-8")
    assert secret not in raw and "/private/path" not in raw and "hidden" not in raw
    row = json.loads(raw)
    assert row["outcome"] == "failed" and row["error_type"] == "URLError"


def test_ledger_is_append_only_and_offline_verification_distinguishes_blocks(isolated, monkeypatch):
    monkeypatch.setattr(policy, "_transport_urlopen", lambda *_a, **_k: _Response())
    policy.guarded_urlopen("http://127.0.0.1:8080/health")
    policy.guarded_urlopen("https://example.test/model")
    policy.set_local_only(True)
    with pytest.raises(policy.LocalOnlyViolation):
        policy.guarded_urlopen("https://blocked.example.test/model")

    rows = policy.read_outbound_attempts()
    assert len(rows) == 3
    report = policy.verify_offline()
    assert report["verified"] is True
    assert report["local_only"] is True
    assert report["guard_installed"] is True and report["probe_blocked"] is True
    assert report["probe_recorded"] is True
    assert report["external_attempt_count"] == 2
    assert report["blocked_external_attempt_count"] == 2
    assert report["violations"] == []


def test_verify_offline_fails_when_policy_is_off_or_guard_is_absent(isolated, monkeypatch):
    off = policy.verify_offline()
    assert off["verified"] is False and off["local_only"] is False

    policy.set_local_only(True)
    monkeypatch.setattr(urllib.request, "urlopen", policy._transport_urlopen)
    absent = policy.verify_offline()
    assert absent["verified"] is False
    assert absent["local_only"] is True and absent["guard_installed"] is False


def test_global_urllib_guard_is_installed():
    assert urllib.request.urlopen is policy.guarded_urlopen
