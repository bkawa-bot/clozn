"""Tests for outcome-grounded calibration (eval.calibration) -- pure over synthetic (score, correct) pairs."""
from __future__ import annotations

from clozn.eval import calibration as cal


def test_brier_extremes_and_empty():
    assert cal.brier([(1.0, True), (0.0, False)]) == 0.0          # perfectly confident + right
    assert cal.brier([(1.0, False), (0.0, True)]) == 1.0          # perfectly confident + wrong
    assert cal.brier([]) is None
    assert cal.brier([(0.5, True), (0.5, False)]) == 0.25         # 0.25 each way


def test_ece_is_near_zero_for_a_well_calibrated_set():
    # a bin at score ~0.7 whose accuracy is also 0.7 has ~0 gap; build 10 items, 7 correct, all score 0.7
    pairs = [(0.7, True)] * 7 + [(0.7, False)] * 3
    out = cal.ece(pairs)
    assert out["ece"] is not None and out["ece"] < 0.05


def test_ece_flags_overconfidence():
    # score 0.95 but only half correct -> a big calibration gap
    pairs = [(0.95, True)] * 5 + [(0.95, False)] * 5
    assert cal.ece(pairs)["ece"] > 0.4


def test_temperature_scaling_softens_an_overconfident_mixed_set():
    pairs = [(0.99, True), (0.99, False)] * 10
    fit = cal.fit_temperature(pairs)
    assert fit["available"] is True and fit["method"] == "scalar-temperature"
    assert fit["temperature"] > 1.0
    assert fit["nll_after"] < fit["nll_before"]
    assert fit["brier_after"] < fit["brier_before"]
    assert cal.temperature_scale(0.99, fit["temperature"]) < 0.99


def test_temperature_scaling_is_monotone_and_handles_boundaries():
    values = [cal.temperature_scale(x, 2.0) for x in (0.0, 0.2, 0.5, 0.8, 1.0)]
    assert values == sorted(values)
    assert values[0] > 0 and values[-1] < 1 and values[2] == 0.5
    assert cal.temperature_scale(-0.1, 1.0) is None
    assert cal.temperature_scale(0.5, 0.0) is None


def test_temperature_fit_refuses_unidentifiable_outcomes():
    assert cal.fit_temperature([])["available"] is False
    out = cal.fit_temperature([(0.9, True), (0.8, True)])
    assert out["available"] is False and "both correct and incorrect" in out["reason"]


def test_ungradeable_and_malformed_pairs_are_dropped():
    pairs = [(0.9, True), (0.8, None), ("x", True), (1.5, False), (0.5, False)]
    assert cal.brier(pairs) is not None
    assert cal.ece(pairs)["n"] == 2                               # only the two clean pairs survive


def test_risk_coverage_rewards_an_informative_score():
    # score correlates with correctness: the confident ones are right, the unsure ones wrong.
    pairs = [(0.9, True), (0.8, True), (0.7, True), (0.3, False), (0.2, False)]
    pts = cal.risk_coverage(pairs)
    assert pts[0].error == 0.0                                    # most-confident answer is correct
    assert pts[-1].coverage == 1.0 and pts[-1].error == 0.4      # answering all -> base error 2/5
    s = cal.selective_summary(pairs, coverage=0.6)
    assert s["error_at_coverage"] < s["full_coverage_error"]     # abstaining cuts error
    assert s["error_reduction_vs_full"] > 0


def test_aurc_lower_for_better_ranking():
    good = [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]     # confidence tracks correctness
    bad = [(0.9, False), (0.8, False), (0.2, True), (0.1, True)]      # exactly backwards
    assert cal.aurc(good) < cal.aurc(bad)


def test_report_bundle_shape_and_empty():
    r = cal.report([(0.9, True), (0.8, True), (0.3, False), (0.2, False)])
    assert r["available"] is True and r["n"] == 4
    assert r["temperature_scaling"]["available"] is True
    assert set(r["selective"].keys()) == {50, 70, 90}
    assert r["base_error"] == 0.5
    assert cal.report([])["available"] is False
    assert cal.report([(0.5, None)])["available"] is False       # nothing gradeable
