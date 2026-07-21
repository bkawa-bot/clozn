"""Persistent scope and undo for prompt-first corrective retries.

Session policies are keyed only by Clozn's opaque session association and expire.
Profile policies live in the portable profile bundle.  Every persistent activation
gets a compare-and-swap undo transaction; stale undo never overwrites newer intent.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy

from clozn._io import atomic_write_json
from clozn.replay.corrective import CORRECTION_PRESETS


SCHEMA = "clozn.corrective_retries.v1"
SESSION_TTL_SECONDS = 30 * 86400
_PATH = os.path.join(os.path.expanduser("~"), ".clozn", "corrective_retries.json")


class CorrectivePolicyError(ValueError):
    """A safe, user-actionable policy or undo refusal."""


def _path() -> str:
    return _PATH


def _empty() -> dict:
    return {"schema": SCHEMA, "sessions": {}, "transactions": []}


def _preset_list(values) -> list[str]:
    if not isinstance(values, list):
        return []
    return list(dict.fromkeys(str(value) for value in values if str(value) in CORRECTION_PRESETS))


def _load(*, strict: bool = False, now: float | None = None) -> dict:
    try:
        with open(_path(), encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
            raise ValueError("invalid corrective retry store")
        sessions = raw.get("sessions")
        transactions = raw.get("transactions")
        if not isinstance(sessions, dict) or not isinstance(transactions, list):
            raise ValueError("invalid corrective retry store")
        doc = {"schema": SCHEMA, "sessions": {},
               "transactions": [tx for tx in transactions if isinstance(tx, dict)][-200:]}
        clock = float(time.time() if now is None else now)
        for key, entry in sessions.items():
            if not isinstance(key, str) or not key.startswith("session_") or not isinstance(entry, dict):
                continue
            expires = entry.get("expires_ts")
            if not isinstance(expires, (int, float)) or float(expires) <= clock:
                continue
            doc["sessions"][key] = {
                "presets": _preset_list(entry.get("presets")),
                "revision": max(0, int(entry.get("revision") or 0)),
                "expires_ts": float(expires),
            }
        return doc
    except FileNotFoundError:
        return _empty()
    except Exception as exc:
        if strict:
            raise CorrectivePolicyError(f"corrective retry store is unreadable: {exc}") from None
        return _empty()


def _save(doc: dict) -> None:
    atomic_write_json(_path(), doc, ensure_ascii=False, indent=2, sort_keys=True)


def _profile_store():
    from clozn.profiles.store import ProfileStore
    return ProfileStore()


def profile_presets(name: str | None) -> list[str]:
    if not name:
        return []
    try:
        return _preset_list(_profile_store().load(str(name)).get("response_policies"))
    except Exception:
        return []


def session_presets(session_key: str | None, *, now: float | None = None) -> list[str]:
    if not session_key:
        return []
    entry = _load(now=now)["sessions"].get(str(session_key)) or {}
    return _preset_list(entry.get("presets"))


def effective_presets(*, session_key: str | None = None,
                      profile_name: str | None = None, now: float | None = None) -> list[str]:
    """Portable profile policy first, narrower session policy second, de-duplicated."""
    return list(dict.fromkeys(profile_presets(profile_name) + session_presets(session_key, now=now)))


def inject(messages: Sequence[Mapping], presets: Sequence[str]) -> list[dict]:
    """Copy messages and append one auditable Clozn block to system context."""
    if isinstance(messages, (str, bytes)) or not isinstance(messages, Sequence):
        raise CorrectivePolicyError("messages must be a sequence")
    copied = [deepcopy(dict(message)) for message in messages]
    selected = _preset_list(list(presets))
    if not selected:
        return copied
    lines = ["Clozn active corrective response policy:"]
    lines.extend(f"- {CORRECTION_PRESETS[preset]}" for preset in selected)
    block = "\n".join(lines)
    for message in copied:
        if message.get("role") == "system":
            message["content"] = (str(message.get("content") or "") + "\n\n" + block).strip()
            return copied
    return [{"role": "system", "content": block}] + copied


def evidence(presets: Sequence[str], *, session_key: str | None,
             profile_name: str | None) -> dict | None:
    selected = _preset_list(list(presets))
    if not selected:
        return None
    scopes = []
    profile = profile_presets(profile_name)
    session = session_presets(session_key)
    if profile:
        scopes.append({"scope": "profile", "target": profile_name, "presets": profile})
    if session:
        scopes.append({"scope": "session", "target": session_key, "presets": session})
    return {
        "schema": SCHEMA,
        "mechanism": "system_instruction",
        "presets": selected,
        "instructions": [CORRECTION_PRESETS[preset] for preset in selected],
        "scopes": scopes,
    }


def _transaction(scope: str, target: str, before: list[str], after: list[str], now: float) -> dict:
    return {
        "id": "repair_" + uuid.uuid4().hex[:16],
        "scope": scope,
        "target": target,
        "before": list(before),
        "after": list(after),
        "created_ts": now,
        "undone_ts": None,
    }


def activate(scope: str, target: str | None, preset: str, *, now: float | None = None) -> dict:
    """Persist a successful correction for an exact session or portable profile."""
    if scope not in {"session", "profile"}:
        raise CorrectivePolicyError("persistent scope must be session or profile")
    if preset not in CORRECTION_PRESETS:
        raise CorrectivePolicyError(f"unknown corrective preset {preset!r}")
    target = str(target or "")
    if not target:
        raise CorrectivePolicyError(f"{scope} scope has no exact target")
    clock = float(time.time() if now is None else now)
    doc = _load(strict=True, now=clock)

    if scope == "session":
        if not target.startswith("session_"):
            raise CorrectivePolicyError("session scope requires an exact opaque session key")
        current = doc["sessions"].get(target) or {"presets": [], "revision": 0}
        before = _preset_list(current.get("presets"))
        if preset in before:
            return {"status": "unchanged", "scope": scope, "target": target,
                    "presets": before, "undo_id": None}
        after = before + [preset]
        tx = _transaction(scope, target, before, after, clock)
        revision = int(current.get("revision") or 0) + 1
        tx["after_revision"] = revision
        doc["sessions"][target] = {"presets": after, "revision": revision,
                                    "expires_ts": clock + SESSION_TTL_SECONDS}
        doc["transactions"].append(tx)
        _save(doc)
    else:
        store = _profile_store()
        try:
            profile = store.load(target)
        except Exception as exc:
            raise CorrectivePolicyError(f"profile {target!r} is unavailable: {exc}") from None
        before = _preset_list(profile.get("response_policies"))
        if preset in before:
            return {"status": "unchanged", "scope": scope, "target": target,
                    "presets": before, "undo_id": None}
        after = before + [preset]
        tx = _transaction(scope, target, before, after, clock)
        profile["response_policies"] = after
        try:
            store.save(profile)
            doc["transactions"].append(tx)
            _save(doc)
        except Exception:
            try:
                profile["response_policies"] = before
                store.save(profile)
            except Exception:
                pass
            raise
    return {"status": "activated", "scope": scope, "target": target,
            "presets": after, "undo_id": tx["id"]}


def undo(transaction_id: str, *, now: float | None = None) -> dict:
    """Compare-and-swap undo; refuse if a newer policy changed the target."""
    clock = float(time.time() if now is None else now)
    doc = _load(strict=True, now=clock)
    tx = next((item for item in reversed(doc["transactions"])
               if item.get("id") == transaction_id), None)
    if tx is None:
        raise CorrectivePolicyError("unknown corrective retry undo id")
    if tx.get("undone_ts") is not None:
        raise CorrectivePolicyError("corrective retry was already undone")
    scope, target = tx.get("scope"), str(tx.get("target") or "")
    before, after = _preset_list(tx.get("before")), _preset_list(tx.get("after"))

    if scope == "session":
        current = doc["sessions"].get(target) or {}
        if (_preset_list(current.get("presets")) != after
                or int(current.get("revision") or 0) != int(tx.get("after_revision") or -1)):
            raise CorrectivePolicyError("session policy changed after this retry; refusing stale undo")
        revision = int(current.get("revision") or 0) + 1
        if before:
            doc["sessions"][target] = {"presets": before, "revision": revision,
                                        "expires_ts": clock + SESSION_TTL_SECONDS}
        else:
            doc["sessions"].pop(target, None)
        tx["undone_ts"] = clock
        _save(doc)
    elif scope == "profile":
        store = _profile_store()
        try:
            profile = store.load(target)
        except Exception as exc:
            raise CorrectivePolicyError(f"profile {target!r} is unavailable: {exc}") from None
        if _preset_list(profile.get("response_policies")) != after:
            raise CorrectivePolicyError("profile policy changed after this retry; refusing stale undo")
        profile["response_policies"] = before
        try:
            store.save(profile)
            tx["undone_ts"] = clock
            _save(doc)
        except Exception:
            try:
                profile["response_policies"] = after
                store.save(profile)
            except Exception:
                pass
            raise
    else:
        raise CorrectivePolicyError("corrective retry transaction has an invalid scope")
    return {"status": "undone", "undo_id": transaction_id, "scope": scope,
            "target": target, "presets": before}
