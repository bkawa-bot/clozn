"""Tests for the close-call locator (close_calls.py) -- pure over synthetic traces."""
from __future__ import annotations

from clozn.runs import close_calls


def _trace(*pairs):
    """Each pair = (top, alt, p_top, p_alt) -> one token's top-2 alternatives."""
    return {"trace": {"alternatives": [[{"piece": t, "prob": pt}, {"piece": a, "prob": pa}]
                                       for (t, a, pt, pa) in pairs]}}


def test_flags_a_genuine_two_way_content_split():
    calls = close_calls.close_calls(_trace(("Rome", "Lyon", 0.44, 0.42)))
    assert len(calls) == 1
    c = calls[0]
    assert c["top"] == "Rome" and c["alt"] == "Lyon" and c["margin"] <= close_calls.MARGIN


def test_ignores_a_confident_token():
    assert close_calls.close_calls(_trace(("Paris", "Lyon", 0.95, 0.02))) == []


def test_ignores_punctuation_and_one_char_forks():
    # near-even, but not content -> not a meaningful close call
    assert close_calls.close_calls(_trace(("or", "(", 0.44, 0.42))) == []
    assert close_calls.close_calls(_trace(("a", "I", 0.44, 0.42))) == []


def test_ignores_a_weak_runnerup_even_if_top_is_low():
    # top only 0.30 but runner-up 0.05 -> a spread, not a two-way tie (runner-up below MIN_RUNNERUP)
    assert close_calls.close_calls(_trace(("The", "Sure", 0.30, 0.05))) == []


def test_summarize_names_the_tightest_call():
    calls = close_calls.close_calls(_trace(("Rome", "Lyon", 0.44, 0.42),
                                           ("bake", "make", 0.45, 0.44)))
    s = close_calls.summarize(calls)
    assert s.startswith("2 close calls")
    assert "make" in s and "bake" in s               # the tightest (margin 0.01) is named
    assert close_calls.summarize([]) == ""


def test_never_raises_on_junk():
    for junk in (None, {}, {"trace": "x"}, {"trace": {"alternatives": "x"}},
                 {"trace": {"alternatives": [[{"piece": "a"}]]}}):     # only one alt
        assert close_calls.close_calls(junk) == []
