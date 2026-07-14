"""Selective-generation policy -- the RUNTIME half of outcome-grounded eval (calibration.py is the
measurement half). Given a calibration set of (score, correct) pairs, pick a confidence threshold that
lets the model ANSWER only when it's reliable enough, and ABSTAIN (or ASK for clarification) otherwise.

The contract is honest about BOTH sides of the trade. A threshold that cuts answered-error also withholds
some answers that would have been correct; `apply_policy` reports `errors_caught` (wrong answers not
emitted -- the win) AND `correct_withheld` (right answers held back -- the cost), never just the flattering
half. A threshold is chosen against a calibration SET and is only as good as that set generalizes; it is
not a per-answer guarantee.
"""
from __future__ import annotations

from clozn.eval import calibration as cal


def decide(score: float, answer_at: float, ask_at: float | None = None) -> str:
    """One decision: 'answer' if score >= answer_at; 'ask' if an ask band is set and answer_at > score >=
    ask_at; else 'abstain'. With ask_at=None it's a plain answer/abstain policy."""
    if score >= answer_at:
        return "answer"
    if ask_at is not None and score >= ask_at:
        return "ask"
    return "abstain"


def choose_threshold(pairs, target_error: float = 0.05) -> dict:
    """The confidence threshold that MAXIMIZES coverage subject to answered-error <= target_error. Scans
    every distinct score as a candidate `answer >= tau` cut (exact w.r.t. ties, unlike reading the
    risk-coverage curve directly). Returns {threshold, coverage, error, n_answered, achievable};
    achievable is False when no cut hits the target (even answering only the single most-confident item
    errs) -- then threshold sits above the max score, i.e. answer nothing."""
    clean = cal._clean(pairs)
    if not clean:
        return {"threshold": 1.01, "coverage": 0.0, "error": None, "n_answered": 0, "achievable": False}
    best = None
    for tau in sorted({s for s, _ in clean}, reverse=True):
        answered = [(s, c) for s, c in clean if s >= tau]
        err = sum(1 for _, c in answered if not c) / len(answered)
        if err <= target_error:
            cov = len(answered) / len(clean)
            if best is None or cov > best["coverage"]:
                best = {"threshold": round(tau, 4), "coverage": round(cov, 4),
                        "error": round(err, 4), "n_answered": len(answered), "achievable": True}
    if best:
        return best
    top = max(s for s, _ in clean)
    return {"threshold": round(top + 0.01, 4), "coverage": 0.0, "error": None,
            "n_answered": 0, "achievable": False}


def apply_policy(pairs, answer_at: float, ask_at: float | None = None) -> dict:
    """Run a (possibly two-threshold) policy over a calibration set and report BOTH sides of the trade:
    coverage + answered-error (the answers we DID emit), `errors_caught` (wrong answers we withheld -- the
    win) and `correct_withheld` (right answers we withheld -- the cost)."""
    clean = cal._clean(pairs)
    n = len(clean)
    buckets = {"answer": [], "ask": [], "abstain": []}
    for s, c in clean:
        buckets[decide(s, answer_at, ask_at)].append((s, c))
    answered = buckets["answer"]
    withheld = buckets["ask"] + buckets["abstain"]
    answered_wrong = sum(1 for _, c in answered if not c)
    return {
        "n": n, "answer_at": round(answer_at, 4), "ask_at": (round(ask_at, 4) if ask_at is not None else None),
        "n_answer": len(answered), "n_ask": len(buckets["ask"]), "n_abstain": len(buckets["abstain"]),
        "coverage": round(len(answered) / n, 4) if n else 0.0,
        "answered_error": round(answered_wrong / len(answered), 4) if answered else None,
        "base_error": round(sum(1 for _, c in clean if not c) / n, 4) if n else None,
        "errors_caught": sum(1 for _, c in withheld if not c),      # wrong answers NOT emitted (the win)
        "correct_withheld": sum(1 for _, c in withheld if c),       # right answers held back (the cost)
    }


def recommend(pairs, target_error: float = 0.05) -> dict:
    """A ready-to-use recommendation: `answer_at` = the tightest threshold hitting target_error, and an
    `ask` band down to a LOOSER threshold (target_error*3) -- in that middle band the model is right often
    enough that a clarifying re-ask beats a flat refusal. Below the ask band, abstain. Returns the chosen
    thresholds + the honest apply_policy trade. Heuristic ask band, documented as such."""
    picked = choose_threshold(pairs, target_error)
    answer_at = picked["threshold"]
    looser = choose_threshold(pairs, min(1.0, target_error * 3))
    ask_at = looser["threshold"] if (looser["achievable"] and looser["threshold"] < answer_at) else None
    return {"target_error": target_error, "answer_at": answer_at, "ask_at": ask_at,
            "achievable": picked["achievable"], "summary": apply_policy(pairs, answer_at, ask_at)}
