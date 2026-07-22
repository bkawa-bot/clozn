"""Persisted local retention policy: how many days of run-journal history to keep before ``clozn migrate
--gc`` also deletes rows, not just orphaned trace blobs.

Mirrors ``clozn.network_policy``'s persisted-flag pattern: one small JSON document at
``~/.clozn/retention_policy.json``, with an environment-overridable path for tests. No file present, or
an unreadable/malformed one, means "no age-based deletion" -- ``clozn migrate --gc`` keeps its original
blob-only behavior until a user explicitly opts in via ``clozn privacy retention --days N``. A malformed
file fails CLOSED in the sense that matters here: it is simply treated as "no policy", never as an
accidentally huge or tiny day count that could surprise-delete or surprise-retain a user's history.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from clozn._io import atomic_write_json

POLICY_PATH = os.path.join(os.path.expanduser("~/.clozn"), "retention_policy.json")
POLICY_ENV = "CLOZN_RETENTION_POLICY_PATH"


def _policy_path(environ=None) -> str:
    env = os.environ if environ is None else environ
    return str(env.get(POLICY_ENV) or POLICY_PATH)


def get_policy(*, environ=None) -> dict:
    """Return ``{"days": None}`` (no policy) or ``{"days": N, "updated_at": ...}``."""
    try:
        with open(_policy_path(environ), encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception:
        return {"days": None}
    if not isinstance(value, dict):
        return {"days": None}
    days = value.get("days")
    valid_days = days if isinstance(days, int) and not isinstance(days, bool) and days > 0 else None
    return {"days": valid_days, "updated_at": value.get("updated_at")}


def set_policy(days: "int | None", *, environ=None) -> dict:
    """Persist a day-count retention policy, or clear it with ``days=None``."""
    if days is not None and (not isinstance(days, int) or isinstance(days, bool) or days <= 0):
        raise ValueError("days must be a positive integer or None")
    path = _policy_path(environ)
    now = datetime.now(timezone.utc).isoformat()
    document = {"schema_version": 1, "days": days, "updated_at": now}
    atomic_write_json(path, document)
    return {"days": days, "updated_at": now, "path": path}


__all__ = ["POLICY_ENV", "POLICY_PATH", "get_policy", "set_policy"]
