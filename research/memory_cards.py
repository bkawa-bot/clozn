"""memory_cards -- structured, inspectable memory for Clozn Studio (roadmap Milestone 3, issue D1).

Memory stops being a bag of trait strings and becomes state you can browse, source-link, edit, and
review: each preference/fact is a *card* with provenance and lifecycle. Only 'active' cards feed the
soft-prefix (see active_texts); 'pending' cards await review, 'disabled' are kept-but-unused, 'rejected'
are tombstoned. The latent prefix / soft-state itself lives elsewhere (self_teach_server) -- this module
owns only the card metadata + CRUD.

Mirrors research/runlog.py exactly: stdlib only, a single flat JSON file, and IO that NEVER raises --
persistence must not break a request, so every op degrades to None/[]/False on failure. The store path is
a module-level global so tests can point it at a temp file (as runlog.py does with RUNS_DIR).
"""
from __future__ import annotations

import json
import os
import time
import uuid

CARDS_PATH = os.path.join(os.path.expanduser("~/.clozn"), "studio_memory_cards.json")

# lifecycle states; only ACTIVE feeds the prefix (E1). Kept as a tuple so callers can validate against it.
STATUSES = ("pending", "active", "disabled", "rejected")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _load() -> list[dict]:
    """Read the whole store; [] if missing or unreadable (never raises)."""
    try:
        if not os.path.isfile(CARDS_PATH):
            return []
        with open(CARDS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(cards: list[dict]) -> bool:
    """Persist the whole store; False on any failure (never raises)."""
    try:
        os.makedirs(os.path.dirname(CARDS_PATH), exist_ok=True)
        with open(CARDS_PATH, "w", encoding="utf-8") as f:
            json.dump(cards, f)
        return True
    except Exception:
        return False


def create(text: str, status: str = "pending", source_run_id: str | None = None,
           kind: str = "preference", risk: str = "low", evidence: str = "",
           strength: float = 1.0) -> dict | None:
    """Create + persist a card; return it (or None on IO failure)."""
    try:
        card = {
            "id": "mem_" + uuid.uuid4().hex[:12],
            "text": (text or "").strip(),
            "status": status if status in STATUSES else "pending",
            "source_run_id": source_run_id,
            "created_at": _now(),
            "last_used_at": None,
            "usage_count": 0,
            "kind": kind,
            "risk": risk,
            "evidence": evidence,
            "strength": float(strength),
        }
        cards = _load()
        cards.append(card)
        if not _save(cards):
            return None
        return card
    except Exception:
        return None


def list_cards(status: str | None = None) -> list[dict]:
    """All cards (newest first), or only those in `status`."""
    cards = _load()
    if status is not None:
        cards = [c for c in cards if c.get("status") == status]
    # newest first: created_at is ISO + monotonic enough; fall back to store order on ties
    return sorted(cards, key=lambda c: c.get("created_at") or "", reverse=True)


def get(card_id: str) -> dict | None:
    for c in _load():
        if c.get("id") == card_id:
            return c
    return None


def update(card_id: str, **fields) -> dict | None:
    """Patch the given fields on a card; return the updated card (or None if not found / IO fails).

    `id` and `created_at` are immutable and silently ignored. `strength` is coerced to float; an
    out-of-range `status` is ignored (keeps the store's states well-formed)."""
    cards = _load()
    updated = None
    for c in cards:
        if c.get("id") == card_id:
            for k, v in fields.items():
                if k in ("id", "created_at"):
                    continue
                if k == "status" and v not in STATUSES:
                    continue
                if k == "strength":
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        continue
                c[k] = v
            updated = c
            break
    if updated is None:
        return None
    if not _save(cards):
        return None
    return updated


def set_status(card_id: str, status: str) -> dict | None:
    """Convenience wrapper over update() for the lifecycle transitions (approve/disable/reject)."""
    if status not in STATUSES:
        return None
    return update(card_id, status=status)


def delete(card_id: str) -> bool:
    """Remove a card; True if one was removed, False otherwise (incl. IO failure)."""
    cards = _load()
    kept = [c for c in cards if c.get("id") != card_id]
    if len(kept) == len(cards):
        return False
    return _save(kept)


def bump_usage(card_id: str) -> dict | None:
    """Record that a card influenced a run: ++usage_count and stamp last_used_at (hooked from _log_run)."""
    card = get(card_id)
    if card is None:
        return None
    return update(card_id, usage_count=int(card.get("usage_count") or 0) + 1, last_used_at=_now())


def active_texts() -> list[str]:
    """The text of every active card -- what self_teach builds the memory prefix from (E1)."""
    return [c.get("text", "") for c in list_cards(status="active") if c.get("text")]


def migrate_from_rules(rules: list[str]) -> list[dict]:
    """One-time seed: turn legacy trait strings into 'active' cards, but only if the store is EMPTY.

    Idempotent -- once any card exists this is a no-op and returns [] (so it can run on every load without
    duplicating). Existing card stores are left untouched."""
    if _load():                                          # already migrated / has cards -> no-op
        return []
    created = []
    for text in rules or []:
        text = (text or "").strip()
        if not text:
            continue
        card = create(text, status="active", kind="preference")
        if card is not None:
            created.append(card)
    return created
