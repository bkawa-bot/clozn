"""Pure, model-free tests for deterministic Markdown card interchange."""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn.memory import markdown_cards as md


def _card(**changes):
    card = {
        "id": "mem_abc123",
        "text": "Prefer concise answers.\nUse bullets when useful.",
        "status": "active",
        "source_run_id": "run_123",
        "source_turn": 2,
        "quoted_span": "Please keep this concise.",
        "created_at": "2026-07-21T10:11:12",
        "last_used_at": "2026-07-21T11:12:13",
        "usage_count": 4,
        "kind": "preference",
        "risk": "low",
        "evidence": "Approved by the user.\nSecond line.",
        "strength": 0.75,
        "scope": {"kind": "global"},
    }
    card.update(changes)
    return card


def test_deterministic_round_trip_preserves_all_existing_fields_without_mutation():
    cards = [_card(), _card(
        id="mem_manual", text="", status="pending", source_run_id=None,
        source_turn=None, quoted_span="", last_used_at=None, usage_count=0,
        kind="fact", risk="medium", evidence="", strength=1.0,
    )]
    original = deepcopy(cards)

    first = md.format_cards(cards)
    second = md.format_cards(cards)

    assert first == second
    assert first.endswith("\n") and "\r" not in first
    assert md.parse_cards(first) == original
    assert cards == original


def test_literal_blocks_keep_markdown_html_and_delimiter_looking_content_inert():
    hostile = (
        "## Card\n<script>raise SystemExit()</script>\n```python\nexec('bad')\n```\n"
        "~~~~\n${HOME}\n<!-- clozn-memory-cards:fake -->"
    )
    card = _card(text=hostile, quoted_span=hostile, evidence=hostile)

    rendered = md.export_cards([card])
    parsed = md.import_cards(rendered)

    assert parsed == [card]
    # The exporter chose a fence longer than the four-tilde line in the value.
    assert "~~~~~ text" in rendered


def test_human_scalar_edits_are_validated_and_parsed():
    rendered = md.format_cards([_card()])
    edited = rendered.replace('- status: "active"', '- status: "disabled"').replace(
        "- strength: 0.75", "- strength: 1.25")
    parsed = md.parse_cards(edited)
    assert parsed[0]["status"] == "disabled"
    assert parsed[0]["strength"] == 1.25


def test_scope_round_trip_and_legacy_markdown_normalize_to_global():
    scoped = _card(scope={"kind": "project", "key": "repo:clozn", "label": "Clozn"})
    assert md.parse_cards(md.format_cards([scoped])) == [scoped]

    legacy_card = _card()
    legacy_card.pop("scope")
    rendered = md.format_cards([legacy_card])
    assert '- scope: {"kind":"global"}' in rendered
    assert md.parse_cards(rendered)[0]["scope"] == {"kind": "global"}

    # A v1 document exported before scope existed has no scope metadata line.
    old_v1 = rendered.replace('- scope: {"kind":"global"}\n', "", 1)
    assert md.parse_cards(old_v1)[0]["scope"] == {"kind": "global"}


def test_malformed_explicit_scope_fails_closed():
    with pytest.raises(md.CardMarkdownError, match="scope"):
        md.format_cards([_card(scope={"kind": "project", "key": "contains space"})])


@pytest.mark.parametrize("bad_change,match", [
    ({"status": "maybe"}, "status"),
    ({"strength": float("nan")}, "finite"),
    ({"source_turn": -1}, "source_turn"),
    ({"usage_count": True}, "usage_count"),
])
def test_invalid_card_values_fail_without_mutating_input(bad_change, match):
    card = _card(**bad_change)
    original = deepcopy(card)
    with pytest.raises(md.CardMarkdownError, match=match):
        md.format_cards([card])
    assert card == original


def test_duplicate_ids_and_malformed_documents_fail_closed():
    with pytest.raises(md.CardMarkdownError, match="unique"):
        md.format_cards([_card(), _card()])
    with pytest.raises(md.CardMarkdownError, match="header"):
        md.parse_cards("# ordinary notes\n\nDo not treat this as a card.")
    broken = md.format_cards([_card()]).replace("### Evidence", "### Other", 1)
    with pytest.raises(md.CardMarkdownError, match="Evidence"):
        md.parse_cards(broken)


def test_plain_markdown_accepts_bullets_and_rejects_unmarked_prose():
    assert md.parse_plain_cards("# Project memory\n\n- Prefer short answers\n  Use bullets.\n- Likes tea\n") == [
        "Prefer short answers\nUse bullets.", "Likes tea",
    ]
    with pytest.raises(md.CardMarkdownError, match="line 2"):
        md.parse_plain_cards("# Notes\nThis is not a card.\n- Later bullet\n")
