"""Tests for the close-call locator (close_calls.py) -- pure over synthetic traces.

The trace shape mirrors the real engine record: `tokens[i]`/`confidence[i]` carry the EMITTED token and
its softmax prob; `alternatives[i]` carries the OTHER top-k tokens (emitted excluded). A close call is the
CHOSEN token vs its strongest rival, so the locator reconstructs {emitted} u alternatives per step.
"""
from __future__ import annotations

from clozn.runs import close_calls


def _trace(*steps):
    """Each step = (emitted, p_emitted, [(alt_piece, alt_prob), ...]) -> one generated token."""
    return {"trace": {
        "tokens": [emit for (emit, _p, _alts) in steps],
        "confidence": [p for (_e, p, _alts) in steps],
        "alternatives": [[{"piece": a, "prob": pa} for (a, pa) in alts] for (_e, _p, alts) in steps],
    }}


def test_flags_a_genuine_two_way_content_split():
    calls = close_calls.close_calls(_trace(("Rome", 0.44, [("Lyon", 0.42)])))
    assert len(calls) == 1
    c = calls[0]
    assert c["top"] == "Rome" and c["alt"] == "Lyon"
    assert c["emitted"] == "Rome" and c["margin"] <= close_calls.MARGIN


def test_reconstructs_when_the_model_sampled_the_runnerup():
    # emitted "Lyon" (0.42) was NOT the argmax -- "Rome" (0.44) sat in alternatives. The co-leaders are
    # Rome (top) vs Lyon, and `emitted` honestly records that the model chose the runner-up.
    calls = close_calls.close_calls(_trace(("Lyon", 0.42, [("Rome", 0.44)])))
    assert len(calls) == 1
    c = calls[0]
    assert c["top"] == "Rome" and c["alt"] == "Lyon" and c["emitted"] == "Lyon"


def test_ignores_a_confident_token():
    assert close_calls.close_calls(_trace(("Paris", 0.95, [("Lyon", 0.02)]))) == []


def test_ignores_punctuation_and_one_char_forks():
    assert close_calls.close_calls(_trace(("or", 0.44, [("(", 0.42)]))) == []
    assert close_calls.close_calls(_trace(("a", 0.44, [("I", 0.42)]))) == []


def test_ignores_a_weak_runnerup_even_if_top_is_low():
    # emitted 0.30 but runner-up 0.05 -> a spread, not a two-way tie (runner-up below MIN_RUNNERUP)
    assert close_calls.close_calls(_trace(("The", 0.30, [("Sure", 0.05)]))) == []


def test_meaningful_flags_digit_and_polarity_forks_only():
    # two different digits -> the answer's NUMBER was a coin-flip
    digit = close_calls.close_calls(_trace(("5", 0.54, [("0", 0.45)])))
    assert digit and digit[0]["meaningful"] is True
    # polarity flip: exactly one side is a negation
    polarity = close_calls.close_calls(_trace(("not", 0.44, [("they", 0.42)])))
    assert polarity and polarity[0]["meaningful"] is True
    # two content words, same substance -> a real close call but NOT meaning-changing
    style = close_calls.close_calls(_trace(("imagine", 0.50, [("think", 0.48)])))
    assert style and style[0]["meaningful"] is False
    # meaningful() returns only the answer-changing slice
    mixed = close_calls.close_calls(_trace(("5", 0.54, [("0", 0.45)]), ("imagine", 0.50, [("think", 0.48)])))
    assert [c["top"] for c in close_calls.meaningful(mixed)] == ["5"]


def test_summarize_names_the_tightest_call():
    calls = close_calls.close_calls(_trace(("Rome", 0.44, [("Lyon", 0.42)]),
                                           ("bake", 0.45, [("make", 0.44)])))
    s = close_calls.summarize(calls)
    assert s.startswith("2 close calls")
    assert "make" in s and "bake" in s               # the tightest (margin 0.01) is named
    assert close_calls.summarize([]) == ""


def test_never_raises_on_junk():
    for junk in (None, {}, {"trace": "x"}, {"trace": {"alternatives": "x"}},
                 {"trace": {"tokens": ["a"], "confidence": [0.5], "alternatives": [[]]}}):  # no rival
        assert close_calls.close_calls(junk) == []
