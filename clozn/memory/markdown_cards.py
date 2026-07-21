"""Deterministic, inert Markdown interchange for Clozn memory cards.

Format ``clozn.memory_cards.markdown.v1`` is deliberately small.  A document has
one ``## Card`` section per card.  Scalar metadata is written as JSON values on
fixed Markdown list lines; the human-authored fields are literal fenced blocks::

    <!-- clozn-memory-cards:clozn.memory_cards.markdown.v1 -->
    # Clozn Memory Cards

    ## Card

    - id: "mem_abc"
    - status: "active"
    ...

    ### Text

    ~~~~ text
    Prefer concise answers.
    ~~~~

``Text``, ``Quoted span``, and ``Evidence`` use a tilde fence longer than any
tilde run in that value.  Markdown headings, HTML, code fences, shell fragments,
and template syntax inside them are therefore data only.  Parsing performs no
I/O, imports no code named by the document, and never mutates the supplied card
objects.  Canonical output uses UTF-8-friendly text, LF newlines, stable field
order, and a final newline.  CRLF/CR input is normalized to LF, including inside
literal blocks.

The format covers every field currently produced by ``memory.cards.create``.
Unknown/missing fields and duplicate IDs fail instead of being silently lost.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from clozn.memory.cards import STATUSES


SCHEMA = "clozn.memory_cards.markdown.v1"
MAGIC = f"<!-- clozn-memory-cards:{SCHEMA} -->"
MAX_DOCUMENT_CHARS = 4 * 1024 * 1024
MAX_CARDS = 10_000

_METADATA_FIELDS = (
    "id",
    "status",
    "kind",
    "risk",
    "strength",
    "source_run_id",
    "source_turn",
    "created_at",
    "last_used_at",
    "usage_count",
)
_BLOCK_FIELDS = (
    ("text", "Text", "text"),
    ("quoted_span", "Quoted span", "quoted-span"),
    ("evidence", "Evidence", "evidence"),
)
_ALL_FIELDS = frozenset(_METADATA_FIELDS) | frozenset(field for field, _, _ in _BLOCK_FIELDS)
_FENCE_RUN = re.compile(r"~+")


class CardMarkdownError(ValueError):
    """The document cannot be represented as current Clozn memory cards."""


def _lf(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _string(value: Any, field: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str):
        raise CardMarkdownError(f"{field} must be a string" + (" or null" if nullable else ""))
    if field in _METADATA_FIELDS and ("\n" in value or "\r" in value):
        raise CardMarkdownError(f"{field} must be a single-line string")
    return _lf(value)


def _normalize_card(value: Any, index: int) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CardMarkdownError(f"card {index} must be an object")
    extra = sorted(set(value) - _ALL_FIELDS)
    missing = sorted(_ALL_FIELDS - set(value))
    if extra or missing:
        raise CardMarkdownError(
            f"card {index} fields are invalid: missing={missing!r}, extra={extra!r}")

    card_id = _string(value["id"], "id")
    if not card_id:
        raise CardMarkdownError(f"card {index} id must not be empty")
    status = _string(value["status"], "status")
    if status not in STATUSES:
        raise CardMarkdownError(f"card {index} status must be one of {list(STATUSES)!r}")
    kind = _string(value["kind"], "kind")
    risk = _string(value["risk"], "risk")

    strength = value["strength"]
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(float(strength))):
        raise CardMarkdownError(f"card {index} strength must be a finite number")

    source_turn = value["source_turn"]
    if (source_turn is not None
            and (isinstance(source_turn, bool) or not isinstance(source_turn, int) or source_turn < 0)):
        raise CardMarkdownError(f"card {index} source_turn must be a non-negative integer or null")
    usage_count = value["usage_count"]
    if isinstance(usage_count, bool) or not isinstance(usage_count, int) or usage_count < 0:
        raise CardMarkdownError(f"card {index} usage_count must be a non-negative integer")

    return {
        "id": card_id,
        "text": _string(value["text"], "text"),
        "status": status,
        "source_run_id": _string(value["source_run_id"], "source_run_id", nullable=True),
        "source_turn": source_turn,
        "quoted_span": _string(value["quoted_span"], "quoted_span"),
        "created_at": _string(value["created_at"], "created_at"),
        "last_used_at": _string(value["last_used_at"], "last_used_at", nullable=True),
        "usage_count": usage_count,
        "kind": kind,
        "risk": risk,
        "evidence": _string(value["evidence"], "evidence"),
        "strength": float(strength),
    }


def _fence(value: str) -> str:
    longest = max((len(match.group(0)) for match in _FENCE_RUN.finditer(value)), default=0)
    return "~" * max(4, longest + 1)


def _literal_block(title: str, label: str, value: str) -> str:
    fence = _fence(value)
    return f"### {title}\n\n{fence} {label}\n{value}\n{fence}"


def format_cards(cards: Sequence[Mapping[str, Any]]) -> str:
    """Return canonical Markdown without changing ``cards`` or nested values."""
    if isinstance(cards, (str, bytes, bytearray)) or not isinstance(cards, Sequence):
        raise CardMarkdownError("cards must be an array of card objects")
    if len(cards) > MAX_CARDS:
        raise CardMarkdownError(f"document supports at most {MAX_CARDS} cards")
    normalized = [_normalize_card(card, index) for index, card in enumerate(cards)]
    ids = [card["id"] for card in normalized]
    if len(ids) != len(set(ids)):
        raise CardMarkdownError("card ids must be unique")

    sections = [MAGIC + "\n# Clozn Memory Cards"]
    for card in normalized:
        lines = ["## Card", ""]
        for field in _METADATA_FIELDS:
            encoded = json.dumps(card[field], ensure_ascii=False, allow_nan=False,
                                 separators=(",", ":"))
            lines.append(f"- {field}: {encoded}")
        for field, title, label in _BLOCK_FIELDS:
            lines.extend(["", _literal_block(title, label, card[field])])
        sections.append("\n".join(lines))
    document = "\n\n".join(sections) + "\n"
    if len(document) > MAX_DOCUMENT_CHARS:
        raise CardMarkdownError(f"document exceeds {MAX_DOCUMENT_CHARS} characters")
    return document


def _skip_blank(lines: list[str], index: int) -> int:
    while index < len(lines) and lines[index] == "":
        index += 1
    return index


def _parse_literal(lines: list[str], index: int, title: str, label: str) -> tuple[str, int]:
    if index >= len(lines) or lines[index] != f"### {title}":
        raise CardMarkdownError(f"expected '### {title}'")
    index = _skip_blank(lines, index + 1)
    if index >= len(lines):
        raise CardMarkdownError(f"{title} literal block is missing")
    opening = lines[index].split(" ", 1)
    if (len(opening) != 2 or opening[1] != label or len(opening[0]) < 3
            or set(opening[0]) != {"~"}):
        raise CardMarkdownError(f"{title} must use a tilde fence labeled {label!r}")
    fence = opening[0]
    start = index + 1
    index = start
    while index < len(lines) and lines[index] != fence:
        index += 1
    if index >= len(lines):
        raise CardMarkdownError(f"{title} literal block is not closed")
    value = "\n".join(lines[start:index])
    if fence != _fence(value):
        raise CardMarkdownError(
            f"{title} fence must be longer than every tilde run in its literal content")
    return value, index + 1


def parse_cards(document: str) -> list[dict[str, Any]]:
    """Parse v1 Markdown into new card dictionaries without touching the card store."""
    if not isinstance(document, str):
        raise CardMarkdownError("document must be a string")
    if len(document) > MAX_DOCUMENT_CHARS:
        raise CardMarkdownError(f"document exceeds {MAX_DOCUMENT_CHARS} characters")
    text = _lf(document)
    if text.startswith("\ufeff"):
        text = text[1:]
    lines = text.split("\n")
    if len(lines) < 2 or lines[0] != MAGIC or lines[1] != "# Clozn Memory Cards":
        raise CardMarkdownError("document has no supported Clozn memory-card header")

    cards = []
    index = _skip_blank(lines, 2)
    while index < len(lines):
        if lines[index] != "## Card":
            raise CardMarkdownError(f"expected '## Card' at line {index + 1}")
        index = _skip_blank(lines, index + 1)
        raw: dict[str, Any] = {}
        for field in _METADATA_FIELDS:
            prefix = f"- {field}: "
            if index >= len(lines) or not lines[index].startswith(prefix):
                raise CardMarkdownError(f"expected metadata field {field!r} at line {index + 1}")
            try:
                raw[field] = json.loads(lines[index][len(prefix):])
            except (json.JSONDecodeError, RecursionError) as exc:
                raise CardMarkdownError(f"invalid JSON value for {field!r}: {exc}") from None
            index += 1
        for field, title, label in _BLOCK_FIELDS:
            index = _skip_blank(lines, index)
            raw[field], index = _parse_literal(lines, index, title, label)
        cards.append(_normalize_card(raw, len(cards)))
        if len(cards) > MAX_CARDS:
            raise CardMarkdownError(f"document supports at most {MAX_CARDS} cards")
        index = _skip_blank(lines, index)

    ids = [card["id"] for card in cards]
    if len(ids) != len(set(ids)):
        raise CardMarkdownError("card ids must be unique")
    return cards


def parse_plain_cards(document: str) -> list[str]:
    """Parse the folk memory-file form: headings/comments plus top-level bullet cards.

    Continuation lines must be indented by two spaces.  Unexpected prose fails with a line number so
    a notes file is never partially or silently imported as durable memory.
    """
    if not isinstance(document, str):
        raise CardMarkdownError("document must be a string")
    if len(document) > MAX_DOCUMENT_CHARS:
        raise CardMarkdownError(f"document exceeds {MAX_DOCUMENT_CHARS} characters")
    lines = _lf(document.lstrip("\ufeff")).split("\n")
    cards: list[str] = []
    current: list[str] | None = None
    in_comment = False
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if in_comment:
            if "-->" in line:
                in_comment = False
            continue
        if stripped.startswith("<!--"):
            in_comment = "-->" not in line
            continue
        if not stripped or line.startswith("#"):
            continue
        if line.startswith("- "):
            if current is not None:
                cards.append("\n".join(current).strip())
            current = [line[2:].strip()]
            continue
        if current is not None and (line.startswith("  ") or line.startswith("\t")):
            current.append(line.lstrip())
            continue
        raise CardMarkdownError(f"unexpected prose at line {line_number}; use a top-level '- ' card")
    if in_comment:
        raise CardMarkdownError("Markdown comment is not closed")
    if current is not None:
        cards.append("\n".join(current).strip())
    cards = [card for card in cards if card]
    if not cards:
        raise CardMarkdownError("plain Markdown contains no card bullets")
    if len(cards) > MAX_CARDS:
        raise CardMarkdownError(f"document supports at most {MAX_CARDS} cards")
    return cards


# Explicit interchange names for future CLI wiring; all four names remain pure.
export_cards = format_cards
import_cards = parse_cards


__all__ = [
    "CardMarkdownError", "SCHEMA", "export_cards", "format_cards", "import_cards", "parse_cards",
    "parse_plain_cards",
]
