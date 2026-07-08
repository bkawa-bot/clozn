"""Model-free tests for `clozn preferences`: format_preferences is pure (canned dict -> text), so no
server/model needed. Mirrors test_explain_cli.py's canned-dict approach."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # repo root for `from clozn import cli`
from clozn import cli as clozn_cli  # noqa: E402


def test_empty_pending_is_a_friendly_hint():
    s = clozn_cli.format_preferences({"pending": []})
    assert "no suggestions yet" in s


def test_lists_a_proposal_with_its_approve_command():
    data = {"pending": [{
        "id": "pref_ab", "dial": "concise", "suggested_value": 0.5, "evidence": ["r1", "r2", "r3"],
        "label": "You've asked for more concise 3x -- make it your default?",
    }]}
    s = clozn_cli.format_preferences(data)
    assert "make it your default" in s
    assert "clozn preferences --approve pref_ab" in s
    assert "from 3 replies" in s          # the evidence count, humanized


def test_singular_evidence_and_garbage_never_crash():
    one = clozn_cli.format_preferences({"pending": [{"id": "x", "label": "L", "evidence": ["r1"]}]})
    assert "from 1 reply" in one          # singular
    assert isinstance(clozn_cli.format_preferences({}), str)          # no `pending` key
    assert isinstance(clozn_cli.format_preferences({"pending": [{}]}), str)   # a proposal missing fields


def test_parser_has_the_subcommand():
    p = clozn_cli.build_parser()
    ns = p.parse_args(["preferences", "--approve", "pref_9", "--port", "8090"])
    assert ns.fn is clozn_cli.cmd_preferences and ns.approve == "pref_9" and ns.port == 8090
