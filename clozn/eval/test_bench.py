"""Tests for the pure helpers of the live bench (eval.bench). The live bench() itself needs a running
studio, so CI covers only the framework-free pieces here."""
from __future__ import annotations

from clozn.eval import bench


def test_answer_confidences_drops_structural_tokens():
    trace = {"tokens": ["Tok", "yo", "", " "], "confidence": [0.9, 0.8, 0.5, 0.4]}
    assert bench._answer_confidences(trace) == [0.9, 0.8]      # empty + whitespace pieces excluded


def test_answer_confidences_tolerates_ragged_and_missing():
    assert bench._answer_confidences({"tokens": ["a"], "confidence": []}) == []
    assert bench._answer_confidences({}) == []


def test_print_is_safe_on_an_empty_report(capsys):
    bench._print({"rows": [], "report": {"available": False}, "n": 0, "unmatched": 0}, "hard", "min")
    assert "no gradeable items" in capsys.readouterr().out
