"""Tests for the selective-generation policy (eval.policy) -- pure over synthetic (score, correct) pairs."""
from __future__ import annotations

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
