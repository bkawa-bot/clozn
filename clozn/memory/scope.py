"""Pure app/project scoping for memory-card selection.

``MemoryScope`` describes the current request context.  Card scope is stored in a
card's optional ``scope`` object using one of these shapes::

    {"kind": "global"}
    {"kind": "app", "key": "opaque-token", "label": "Optional display name"}
    {"kind": "project", "key": "opaque-token", "label": "Optional display name"}

Keys are uninterpreted, case-sensitive tokens; the caller that derives them owns
privacy and stability.  This module only accepts bounded visible ASCII and never
hashes, persists, or displays a key.  Labels are optional display metadata and do
not participate in matching.

Legacy compatibility is explicit: a missing *or malformed* card scope reads as
global.  Writers should use the validated constructors below so malformed scoped
cards are never produced.  Selection is a stable union ranked global, app, then
project, preserving input order inside each rank.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


MAX_KEY_BYTES = 128
MAX_LABEL_CHARS = 128
_KINDS = frozenset({"global", "app", "project"})
_GLOBAL_SCOPE = {"kind": "global"}


class MemoryScopeError(ValueError):
    """A requested scope cannot be represented safely."""


def validate_opaque_key(value: Any, field: str = "key") -> str:
    """Validate, but deliberately do not interpret or transform, an opaque key."""
    if not isinstance(value, str):
        raise MemoryScopeError(f"{field} must be a string")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError:
        raise MemoryScopeError(
            f"{field} must contain 1-{MAX_KEY_BYTES} visible ASCII characters") from None
    if (not 1 <= len(encoded) <= MAX_KEY_BYTES
            or any(byte < 0x21 or byte > 0x7E for byte in encoded)):
        raise MemoryScopeError(
            f"{field} must contain 1-{MAX_KEY_BYTES} visible ASCII characters")
    return value


def _validate_label(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_LABEL_CHARS:
        raise MemoryScopeError(f"label must be a non-empty string up to {MAX_LABEL_CHARS} characters")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise MemoryScopeError("label must not contain control characters")
    return value


@dataclass(frozen=True)
class MemoryScope:
    """Immutable request scope; either, both, or neither exact key may be active."""

    app_key: str | None = None
    project_key: str | None = None

    def __post_init__(self) -> None:
        if self.app_key is not None:
            validate_opaque_key(self.app_key, "app_key")
        if self.project_key is not None:
            validate_opaque_key(self.project_key, "project_key")


def memory_scope(*, app_key: str | None = None,
                 project_key: str | None = None) -> MemoryScope:
    """Validated construction helper for the current request scope."""
    return MemoryScope(app_key=app_key, project_key=project_key)


def card_scope(kind: str, *, key: str | None = None,
               label: str | None = None) -> dict[str, str]:
    """Build one valid card-scope object without retaining caller-owned state."""
    if kind not in _KINDS:
        raise MemoryScopeError(f"kind must be one of {sorted(_KINDS)!r}")
    if kind == "global":
        if key is not None or label is not None:
            raise MemoryScopeError("global scope does not accept key or label")
        return dict(_GLOBAL_SCOPE)
    if key is None:
        raise MemoryScopeError(f"{kind} scope requires key")
    result = {"kind": kind, "key": validate_opaque_key(key)}
    if label is not None:
        result["label"] = _validate_label(label)
    return result


def global_scope() -> dict[str, str]:
    return card_scope("global")


def app_scope(key: str, *, label: str | None = None) -> dict[str, str]:
    return card_scope("app", key=key, label=label)


def project_scope(key: str, *, label: str | None = None) -> dict[str, str]:
    return card_scope("project", key=key, label=label)


def normalize_scope(value: Any, *, legacy_global: bool = True) -> dict[str, str]:
    """Return a validated scope copy, optionally accepting legacy-global input.

    Card readers use the default compatibility behavior.  Writers and request
    boundaries can pass ``legacy_global=False`` to reject missing or malformed
    values rather than silently widening them to global scope.
    """
    if not isinstance(value, Mapping):
        if legacy_global:
            return dict(_GLOBAL_SCOPE)
        raise MemoryScopeError("scope must be a valid global, app, or project scope")
    try:
        kind = value.get("kind")
        if kind == "global" and set(value) == {"kind"}:
            return global_scope()
        if kind in ("app", "project") and set(value) in (
                {"kind", "key"}, {"kind", "key", "label"}):
            return card_scope(kind, key=value.get("key"), label=value.get("label"))
    except (MemoryScopeError, TypeError):
        pass
    if legacy_global:
        return dict(_GLOBAL_SCOPE)
    raise MemoryScopeError("scope must be a valid global, app, or project scope")


def normalize_card_scope(value: Any) -> dict[str, str]:
    """Return a valid copied scope; malformed/missing values become legacy-global."""
    return normalize_scope(value)


def scope_for_card(card: Any) -> dict[str, str]:
    """Read one card's scope with the legacy-global fallback."""
    return normalize_card_scope(card.get("scope") if isinstance(card, Mapping) else None)


def eligibility_rank(card: Any, current: MemoryScope) -> int | None:
    """Return global/app/project rank (0/1/2), or ``None`` when ineligible."""
    if not isinstance(current, MemoryScope):
        raise MemoryScopeError("current must be a MemoryScope")
    if not isinstance(card, Mapping):
        return None
    scope = scope_for_card(card)
    kind = scope["kind"]
    if kind == "global":
        return 0
    if kind == "app" and current.app_key is not None and scope["key"] == current.app_key:
        return 1
    if (kind == "project" and current.project_key is not None
            and scope["key"] == current.project_key):
        return 2
    return None


def is_eligible(card: Any, current: MemoryScope) -> bool:
    return eligibility_rank(card, current) is not None


def eligible_cards(cards: Sequence[Mapping[str, Any]],
                   request_scope: MemoryScope) -> list[Mapping[str, Any]]:
    """Stable global→app→project union, retaining input order within each rank."""
    if isinstance(cards, (str, bytes, bytearray)) or not isinstance(cards, Sequence):
        raise MemoryScopeError("cards must be an array of card objects")
    ranked: list[list[Mapping[str, Any]]] = [[], [], []]
    for card in cards:
        rank = eligibility_rank(card, request_scope)
        if rank is not None:
            ranked[rank].append(card)
    return [card for group in ranked for card in group]


__all__ = [
    "MemoryScope", "MemoryScopeError", "app_scope", "card_scope", "eligibility_rank",
    "eligible_cards", "global_scope", "is_eligible", "memory_scope", "normalize_card_scope",
    "normalize_scope", "project_scope", "scope_for_card", "validate_opaque_key",
]
