"""Tests for the selective-generation policy (eval.policy) -- pure over synthetic (score, correct) pairs."""
from __future__ import annotations

import pytest

from clozn.eval import policy


# a clean separable set: high-score items right, low-score items wrong
_SEP = [(0.95, True), (0.9, True), (0.85, True), (0.4, False), (0.3, False), (0.2, False)]


def test_decide_three_way_band():
    assert policy.decide(0.9, answer_at=0.8, ask_at=0.5) == "answer"
    assert policy.decide(0.6, answer_at=0.8, ask_at=0.5) == "ask"
    assert policy.decide(0.3, answer_at=0.8, ask_at=0.5) == "abstain"
    assert policy.decide(0.3, answer_at=0.8) == "abstain"          # no ask band -> answer/abstain only


def test_choose_threshold_maximizes_coverage_under_target():
    t = policy.choose_threshold(_SEP, target_error=0.0)
    # the three high-score items are all correct -> answerable at 0% error, coverage 0.5
    assert t["achievable"] is True and t["error"] == 0.0
    assert t["coverage"] == 0.5 and t["n_answered"] == 3
    assert 0.4 < t["threshold"] <= 0.85


def test_choose_threshold_unachievable_when_top_item_is_wrong():
    t = policy.choose_threshold([(0.99, False), (0.5, True)], target_error=0.0)
    assert t["achievable"] is False and t["coverage"] == 0.0 and t["n_answered"] == 0


def test_apply_policy_reports_both_sides_of_the_trade():
    s = policy.apply_policy(_SEP, answer_at=0.8)
    assert s["coverage"] == 0.5 and s["answered_error"] == 0.0
    assert s["errors_caught"] == 3 and s["correct_withheld"] == 0   # caught all 3 wrong, held back no right
    assert s["base_error"] == 0.5


def test_apply_policy_counts_correct_withheld_as_a_cost():
    # answer_at too high (0.92): only the 0.95 item answers; the 0.9 and 0.85 CORRECT items are withheld
    s = policy.apply_policy(_SEP, answer_at=0.92)
    assert s["n_answer"] == 1 and s["correct_withheld"] == 2 and s["errors_caught"] == 3


def test_recommend_bundles_thresholds_and_trade():
    r = policy.recommend(_SEP, target_error=0.0)
    assert r["achievable"] is True and r["answer_at"] is not None
    assert r["summary"]["errors_caught"] == 3
    assert "correct_withheld" in r["summary"]


def test_empty_and_junk_never_raise():
    assert policy.choose_threshold([])["achievable"] is False
    assert policy.apply_policy([], answer_at=0.5)["n"] == 0
    assert policy.recommend([(0.5, None), ("x", True)])["summary"]["n"] == 0


# ============================================================== score_from_trace / classify_run (runtime)

_TRACE = {"tokens": ["Par", "is", "", " "], "confidence": [0.9, 0.3, 0.5, 0.1]}


def test_score_from_trace_drops_structural_tokens():
    # empty/whitespace pieces excluded -- only 0.9 and 0.3 count
    assert policy.score_from_trace(_TRACE, "min") == 0.3
    assert policy.score_from_trace(_TRACE, "mean") == pytest.approx(0.6)


def test_score_from_trace_none_when_nothing_scored():
    assert policy.score_from_trace({}, "min") is None
    assert policy.score_from_trace({"tokens": ["a"], "confidence": []}, "min") is None


_SAVED = {"model": "qwen2.5-7b-instruct.Q4_K_M", "score": "min",
          "policy": {"answer_at": 0.8, "ask_at": 0.4}}


def test_classify_run_bands_answer_ask_abstain():
    hi = {"tokens": ["ok"], "confidence": [0.95]}
    mid = {"tokens": ["ok"], "confidence": [0.6]}
    lo = {"tokens": ["ok"], "confidence": [0.1]}
    model = _SAVED["model"]
    assert policy.classify_run(hi, _SAVED, model=model)["band"] == "answer"
    assert policy.classify_run(mid, _SAVED, model=model)["band"] == "ask"
    assert policy.classify_run(lo, _SAVED, model=model)["band"] == "abstain"


def test_classify_run_reports_score_and_thresholds():
    r = policy.classify_run({"tokens": ["ok"], "confidence": [0.6]}, _SAVED, model=_SAVED["model"])
    assert r == {"available": True, "band": "ask", "score": 0.6, "score_aggregate": "min",
                "answer_at": 0.8, "ask_at": 0.4}


def test_classify_run_unavailable_with_no_saved_calibration():
    r = policy.classify_run({"tokens": ["ok"], "confidence": [0.1]}, None, model="qwen2.5-7b-instruct.Q4_K_M")
    assert r["available"] is False and "no calibration saved" in r["reason"]


def test_classify_run_unavailable_on_model_mismatch():
    r = policy.classify_run({"tokens": ["ok"], "confidence": [0.1]}, _SAVED, model="llama-3.2-3b-instruct")
    assert r["available"] is False and "does not match" in r["reason"]


def test_classify_run_unavailable_with_no_model_provenance():
    r = policy.classify_run({"tokens": ["ok"], "confidence": [0.1]}, _SAVED, model=None)
    assert r["available"] is False and "provenance" in r["reason"]


def test_classify_run_unavailable_on_unsupported_score_aggregate():
    saved = {**_SAVED, "score": "weighted"}
    r = policy.classify_run({"tokens": ["ok"], "confidence": [0.1]}, saved, model=_SAVED["model"])
    assert r["available"] is False and "score aggregate" in r["reason"]


def test_classify_run_unavailable_with_no_scored_tokens():
    r = policy.classify_run({}, _SAVED, model=_SAVED["model"])
    assert r["available"] is False and "no scored content tokens" in r["reason"]


def test_classify_run_unavailable_with_no_policy_block():
    saved = {"model": _SAVED["model"], "score": "min"}
    r = policy.classify_run({"tokens": ["ok"], "confidence": [0.1]}, saved, model=_SAVED["model"])
    assert r["available"] is False and "no policy" in r["reason"]
