"""memory_cards -- structured, inspectable memory for Clozn Studio (roadmap Milestone 3, issue D1).

Memory stops being a bag of trait strings and becomes state you can browse, source-link, edit, and
review: each preference/fact is a *card* with provenance and lifecycle. Only 'active' cards feed the
soft-prefix (see active_texts); 'pending' cards await review, 'disabled' are kept-but-unused, 'rejected'
are tombstoned. The latent prefix / soft-state itself lives elsewhere (self_teach_server) -- this module
owns only the card metadata + CRUD.

Provenance (the OBEY defense -- a measured failure mode: a fluent,
plausible card can still be a hallucination or an injected instruction; plausibility gates don't catch
it, only a checkable link to what the user actually said does): a card proposed from a run carries
`source_turn` (index into that run's messages) + `quoted_span` (the verbatim cited text) alongside
`source_run_id`. has_provenance() / is_provenance_claim_unbacked() are the single source of truth for
whether a card's claim to come from a run is actually backed up -- the server's approve-gate and the
Memory page's "you said this" / "no provenance" rendering both read them, so they can't disagree.

Mirrors research/runlog.py exactly: stdlib only, a single flat JSON file, and IO that NEVER raises --
persistence must not break a request, so every op degrades to None/[]/False on failure. The store path is
a module-level global so tests can point it at a temp file (as runlog.py does with RUNS_DIR).
"""
from __future__ import annotations

import json
import os
import time
import uuid

from clozn._io import atomic_write_json

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
    """Persist the whole store; False on any failure (never raises).

    Atomic (see clozn._io): a bad/non-serializable value in `cards` raises out of json.dumps BEFORE the
    real file is ever touched, and the on-disk write itself is a temp-file-then-rename, so a failure here
    can never leave CARDS_PATH truncated or partially written -- the prior contents survive intact."""
    try:
        atomic_write_json(CARDS_PATH, cards)
        return True
    except Exception:
        return False


def create(text: str, status: str = "pending", source_run_id: str | None = None,
           kind: str = "preference", risk: str = "low", evidence: str = "",
           strength: float = 1.0, source_turn: int | None = None,
           quoted_span: str = "") -> dict | None:
    """Create + persist a card; return it (or None on IO failure).

    `source_turn` + `quoted_span` are the PROVENANCE pair (roadmap: the OBEY defense, see
    a measured failure mode -- a fluent, plausible card can still be a hallucination or an
    injected instruction; the only real defense is a checkable link to what the user actually said).
    `source_turn` is the index of the cited message within its run's `messages` list; `quoted_span` is the
    verbatim (possibly truncated) text of that message. Both default empty/None for cards that don't claim
    a run at all (e.g. a manually-typed /memory/add) -- that's a different, self-authored category, not a
    provenance failure. See has_provenance()."""
    try:
        card = {
            "id": "mem_" + uuid.uuid4().hex[:12],
            "text": (text or "").strip(),
            "status": status if status in STATUSES else "pending",
            "source_run_id": source_run_id,
            "source_turn": source_turn,
            "quoted_span": quoted_span or "",
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


def has_provenance(card: dict) -> bool:
    """Does this card back up its claim to come from a run with a checkable quote?

    True iff it cites a run (source_run_id) AND carries a non-empty verbatim quoted_span -- the pair the
    Memory page renders as "you said this". A card that names no run at all (source_run_id is None, e.g. a
    manually-typed /memory/add) is a different, self-authored category and is NOT considered a provenance
    failure by this check -- see create()'s docstring. This is the single source of truth both the server
    (the approve-gate) and the UI read, so they can never disagree about what counts as provenance."""
    if not isinstance(card, dict):
        return False
    return bool(card.get("source_run_id")) and bool((card.get("quoted_span") or "").strip())


def is_provenance_claim_unbacked(card: dict) -> bool:
    """True for the specific failure this defense targets: the card CLAIMS a run (source_run_id is set --
    it says "I came from a real conversation") but has no quoted_span to prove it. This is what must be
    flagged and blocked from approval; it is distinct from a card that never claimed a run in the first
    place (has_provenance() is False for that too, but it isn't an unbacked CLAIM)."""
    if not isinstance(card, dict):
        return False
    return bool(card.get("source_run_id")) and not bool((card.get("quoted_span") or "").strip())


def list_cards(status: str | None = None) -> list[dict]:
    """All cards (newest first), or only those in `status`."""
    cards = _load()
    if status is not None:
        cards = [c for c in cards if c.get("status") == status]
    # newest first: created_at is ISO + monotonic enough; fall back to store order on ties. str()-coerce
    # the key so a hand-edited store with a non-string created_at (e.g. a number) can't make `sorted`
    # raise TypeError from comparing mixed types -- that would break active_texts() (the live memory-
    # prefix compile), not just this listing (round-2 pressure test #3).
    return sorted(cards, key=lambda c: str(c.get("created_at") or ""), reverse=True)


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
    duplicating). Existing card stores are left untouched.

    Guarded per-entry (round-2 pressure test #4): unlike its siblings (create/_save/_load), this used to
    have no try/except at all, so one non-string rule (e.g. a stray int/dict in a hand-edited legacy
    source) would raise AttributeError out of `.strip()` and abort the ENTIRE seed, not just that one
    entry. A bad entry is now skipped, matching the guarded-degradation ethos everywhere else here."""
    if _load():                                          # already migrated / has cards -> no-op
        return []
    created = []
    for text in rules or []:
        try:
            text = (text or "").strip()
        except AttributeError:
            continue                                     # not a string-like value -- skip, don't abort the seed
        if not text:
            continue
        card = create(text, status="active", kind="preference")
        if card is not None:
            created.append(card)
    return created
