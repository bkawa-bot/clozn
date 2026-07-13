"""Tests for the outcome evaluators (eval.outcome) -- named matchers, pure."""
from __future__ import annotations

from clozn.eval import outcome


def test_exact_match_accepts_gold_as_a_phrase_in_fluff():
    assert outcome.exact_match("The capital of France is Paris.", "Paris") is True
    assert outcome.exact_match("paris", "Paris") is True
    assert outcome.exact_match("New York City", "New York", aliases=("NYC",)) is True
    assert outcome.exact_match("It is via an alias.", "wrongword", aliases=("alias",)) is True


def test_exact_match_is_word_bounded_not_substring():
    assert outcome.exact_match("She is Parisian.", "Paris") is False       # not a substring hit
    assert outcome.exact_match("The answer is London.", "Paris") is False


def test_numeric_match_and_tolerance():
    assert outcome.numeric_match("It weighs about 42 kg.", "42") is True
    assert outcome.numeric_match("Roughly 41.8", "42", tol=0.5) is True
    assert outcome.numeric_match("It is 100", "42") is False
    assert outcome.numeric_match("no number here", "42") is False          # gold numeric, pred isn't -> miss
    assert outcome.numeric_match("42", "Paris") is None                    # gold not numeric -> ungradeable


def test_mcq_letter():
    assert outcome.mcq_letter("The answer is C.", "C") is True
    assert outcome.mcq_letter("(B)", "b") is True
    assert outcome.mcq_letter("D", "A") is False
    assert outcome.mcq_letter("no letter", "A") is False
    assert outcome.mcq_letter("A", "banana") is None                       # gold has no A-E letter


def test_grade_dispatch_and_ungradeable():
    assert outcome.grade("Paris", "Paris", "exact") is True
    assert outcome.grade("7", "7", "numeric") is True
    assert outcome.grade("anything", "", "exact") is None                  # empty gold -> ungradeable
    assert outcome.grade("x", "y", "no_such_kind") is None                 # bad kind -> None, never raises
