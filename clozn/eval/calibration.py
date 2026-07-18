"""Outcome-grounded calibration -- the tier actuary.py flags as missing.

Given (score, correct) pairs -- where `correct` came from an OUTCOME evaluator (eval.outcome) and `score`
is a per-item confidence in [0,1] -- report:

  * brier()          -- mean squared error of the score against the 0/1 outcome (proper scoring rule).
  * fit_temperature() -- scalar temperature fit on labeled outcomes; the persisted transform used by
                         Studio only when model + score-aggregate provenance match.
  * ece()            -- Expected Calibration Error against TRUTH (|mean_score - accuracy| per bin), the
                        honest sibling of actuary.ece_proxy. Same bin convention (top bin closed at 1.0).
  * risk_coverage()  -- SELECTIVE generation: sort by score desc, and for each coverage (answer only the
                        top-k most-confident items) report the error among answered. Answering only when
                        confident should trade coverage for lower error.
  * aurc()           -- area under that curve (lower is better).
  * selective_summary()/report() -- the headline: "at 70% coverage, error is E vs F at full coverage".

Pure and deterministic. The score's provenance (answer-span probability, min vs mean token confidence) is
the CALLER's choice -- this module never invents a confidence. It measures correctness on the GIVEN eval
set; it is not a universal guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def _clean(pairs) -> list[tuple[float, bool]]:
    """Keep only well-formed pairs: score a real number in [0,1], correct coercible to bool. `correct` may
    be None (ungradeable) upstream -- those are dropped here, not counted, so coverage means 'of the items
    we could grade'."""
    out = []
    for pair in pairs or []:
        try:
            s, c = pair
        except (TypeError, ValueError):
            continue
        if c is None or not isinstance(s, (int, float)) or isinstance(s, bool):
            continue
        if 0.0 <= float(s) <= 1.0:
            out.append((float(s), bool(c)))
    return out


def brier(pairs) -> float | None:
    p = _clean(pairs)
    if not p:
        return None
    return sum((s - (1.0 if c else 0.0)) ** 2 for s, c in p) / len(p)


def temperature_scale(score: float, temperature: float) -> float | None:
    """Apply scalar temperature scaling to one probability-like confidence score.

    The eval signal is already in [0, 1], so recover its binary logit, divide by positive ``T``, and
    map back through sigmoid. ``T > 1`` softens overconfidence; ``T < 1`` sharpens underconfidence.
    Invalid inputs return None rather than being clipped into invented evidence. Exact 0/1 inputs are
    epsilon-clamped only for the finite logit calculation.
    """
    if (isinstance(score, bool) or not isinstance(score, (int, float))
            or not math.isfinite(float(score)) or not 0.0 <= float(score) <= 1.0):
        return None
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature)) or float(temperature) <= 0.0):
        return None
    p = min(1.0 - 1e-9, max(1e-9, float(score)))
    z = math.log(p / (1.0 - p)) / float(temperature)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _nll(pairs: list[tuple[float, bool]], temperature: float) -> float:
    qs = [(temperature_scale(s, temperature), c) for s, c in pairs]
    return -sum(math.log(q if c else 1.0 - q) for q, c in qs) / len(qs)


def fit_temperature(pairs) -> dict:
    """Fit one positive temperature by minimizing binary NLL on labeled probe outcomes.

    This is real post-hoc temperature scaling, not bin relabeling. A bounded golden-section search over
    log(T) keeps it dependency-free and deterministic. Calibration is deliberately unavailable unless the
    probe set includes at least two items and both outcomes: an all-right/all-wrong set cannot identify a
    useful correctness transform. The saved provenance still limits the result to this model, probe set,
    and score aggregate; callers must enforce those boundaries before applying it.
    """
    p = _clean(pairs)
    if len(p) < 2:
        return {"available": False, "n": len(p), "reason": "temperature scaling needs at least 2 labeled items"}
    if len({c for _, c in p}) < 2:
        return {"available": False, "n": len(p),
                "reason": "temperature scaling needs both correct and incorrect labeled outcomes"}

    lo, hi = math.log(0.05), math.log(20.0)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    x1, x2 = hi - (hi - lo) / phi, lo + (hi - lo) / phi
    f1, f2 = _nll(p, math.exp(x1)), _nll(p, math.exp(x2))
    for _ in range(96):
        if f1 <= f2:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - (hi - lo) / phi
            f1 = _nll(p, math.exp(x1))
        else:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + (hi - lo) / phi
            f2 = _nll(p, math.exp(x2))
    temperature = math.exp((lo + hi) / 2.0)
    before = _nll(p, 1.0)
    after = _nll(p, temperature)
    if after > before:                      # numerical guard: calibration may never worsen its fit set
        temperature, after = 1.0, before
    scaled = [(temperature_scale(s, temperature), c) for s, c in p]
    return {
        "available": True,
        "method": "scalar-temperature",
        "temperature": round(temperature, 6),
        "n": len(p),
        "nll_before": round(before, 6),
        "nll_after": round(after, 6),
        "brier_before": round(brier(p), 6),
        "brier_after": round(brier(scaled), 6),
        "ece_before": round(ece(p)["ece"], 6),
        "ece_after": round(ece(scaled)["ece"], 6),
        "note": ("fit on labeled probe outcomes for this exact model and score aggregate; it is a "
                 "calibrated correctness estimate on that eval distribution, not a fact-check or a "
                 "universal guarantee"),
    }


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    n: int
    mean_score: float | None
    accuracy: float | None
    gap: float | None                    # mean_score - accuracy (>0 = over-confident)
    ci_lo: float | None = None           # Wilson score interval on `accuracy`, 95% confidence
    ci_hi: float | None = None


def _wilson_interval(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion. Preferred over the normal approximation
    because it stays well-behaved (bounded to [0,1], sane width) even for small n."""
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n)) / denom
    return center - margin, center + margin


def ece(pairs, n_bins: int = 10) -> dict:
    """ECE against truth = Σ (n_b/N)·|mean_score_b − accuracy_b|. Bin convention matches actuary.calibration:
    [lo, hi) with the top bin closed on the right so score == 1.0 lands in the last bin."""
    p = _clean(pairs)
    total = len(p)
    bins: list[ReliabilityBin] = []
    num = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        cell = [(s, c) for s, c in p if (lo <= s < hi or (i == n_bins - 1 and s == 1.0))]
        n = len(cell)
        if n:
            ms = sum(s for s, _ in cell) / n
            acc = sum(1 for _, c in cell if c) / n
            gap = ms - acc
            num += n * abs(gap)
            ci_lo, ci_hi = _wilson_interval(acc, n)
            bins.append(ReliabilityBin(lo, hi, n, ms, acc, gap, ci_lo, ci_hi))
        else:
            bins.append(ReliabilityBin(lo, hi, 0, None, None, None, None, None))
    return {"ece": (num / total) if total else None, "bins": bins, "n": total}


@dataclass
class CoveragePoint:
    threshold: float                     # answer only items with score >= this
    coverage: float                      # fraction of gradeable items answered
    error: float                         # fraction wrong among answered
    n_answered: int


def risk_coverage(pairs) -> list[CoveragePoint]:
    """Sort by score DESC and sweep coverage from 1 item to all. At coverage k/N the model 'answers' its k
    highest-score items; error is the fraction of those that were wrong. The curve a selective-generation
    policy is tuned against."""
    p = _clean(pairs)
    if not p:
        return []
    p.sort(key=lambda sc: -sc[0])
    n_total = len(p)
    wrong = 0
    pts = []
    for k in range(1, n_total + 1):
        s, c = p[k - 1]
        if not c:
            wrong += 1
        pts.append(CoveragePoint(threshold=s, coverage=k / n_total, error=wrong / k, n_answered=k))
    return pts


def aurc(pairs) -> float | None:
    """Area under the risk-coverage curve (trapezoid over coverage). Lower is better; a perfect confidence
    ranking (all correct answered first) pushes it toward the base error rate's minimum."""
    pts = risk_coverage(pairs)
    if len(pts) < 2:
        return None
    return sum((b.coverage - a.coverage) * (a.error + b.error) / 2 for a, b in zip(pts, pts[1:]))


def selective_summary(pairs, coverage: float = 0.7) -> dict:
    """At the target coverage (answer the most-confident fraction), report error vs answering everything."""
    pts = risk_coverage(pairs)
    if not pts:
        return {"available": False, "note": "no gradeable items"}
    full_error = pts[-1].error
    pick = min(pts, key=lambda pt: abs(pt.coverage - coverage))
    reduction = None if full_error == 0 else (full_error - pick.error) / full_error
    return {"available": True, "coverage": round(pick.coverage, 3), "error_at_coverage": round(pick.error, 4),
            "full_coverage_error": round(full_error, 4), "abstain_below_score": round(pick.threshold, 4),
            "error_reduction_vs_full": (None if reduction is None else round(reduction, 3)),
            "n": len(pts)}


def report(pairs, n_bins: int = 10) -> dict:
    """The one-call headline bundle: n, Brier, ECE-vs-truth, AURC, and selective error at 50/70/90% coverage.
    Honest-empty ({available:false}) when nothing is gradeable."""
    p = _clean(pairs)
    if not p:
        return {"available": False, "n": 0, "note": "no gradeable (score, correct) pairs"}
    return {"available": True, "n": len(p),
            "base_error": round(sum(1 for _, c in p if not c) / len(p), 4),
            "brier": round(brier(p), 4), "ece": round(ece(p, n_bins)["ece"], 4),
            "aurc": round(aurc(p), 4),
            "temperature_scaling": fit_temperature(p),
            "selective": {int(c * 100): selective_summary(p, c) for c in (0.5, 0.7, 0.9)}}
