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


def _content_confidences(trace: dict) -> list:
    """Per-token confidence over a trace's CONTENT tokens (empty/whitespace pieces dropped -- they're
    structural, not answer content). Mirrors eval.bench._answer_confidences's derivation exactly, so a
    live classification below uses the identical signal the saved calibration was fit against. Kept as
    its own copy rather than imported: this module takes a plain, already-normalized trace dict
    (clozn.runs.trace.steps_to_trace's shape -- {tokens, confidence, ...}), never a live run or
    substrate, so it stays a pure, dependency-light analysis function like the rest of this file."""
    trace = trace if isinstance(trace, dict) else {}
    toks = trace.get("tokens")
    conf = trace.get("confidence")
    toks = toks if isinstance(toks, list) else []
    conf = conf if isinstance(conf, list) else []
    out = []
    for i in range(min(len(toks), len(conf))):
        if not (toks[i] or "").strip():
            continue
        v = conf[i]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out.append(float(v))
    return out


def score_from_trace(trace: dict, aggregate: str = "min") -> float | None:
    """The answer-span confidence aggregate for one reply's trace: 'min' (the weakest content token --
    the standard selective-generation signal, and eval.bench's default) or 'mean'. None when the trace
    carries no scored content tokens -- never a fabricated 0.0/1.0."""
    vals = _content_confidences(trace)
    if not vals:
        return None
    return min(vals) if aggregate != "mean" else sum(vals) / len(vals)


def classify_run(trace: dict, saved: dict | None, model: str | None = None) -> dict:
    """Classify one LIVE reply's confidence against a saved selective-generation policy -- the payload
    `clozn eval --save` persists (clozn.eval.store) and GET /journal/calibration serves. This is the
    RUNTIME half of the calibration backlog item ("a retrieval/clarify action wired to the policy's ask
    band"): `decide()` already knows how to band a score; this adds the honest provenance gate around it,
    mirroring clozn.runs.calibrated_trust.attach_truth's rules (same reasons for the same mismatches) so a
    live verdict is never stronger than what the saved report can actually back up:

      * no saved report, or one with no `policy` block -> unavailable
      * the saved model and the model that actually produced this reply must both be known and match
        EXACTLY (a policy tuned on one model says nothing about another)
      * the saved score aggregate must be 'min' or 'mean' -- this call's score is derived under that SAME
        aggregate, so it is apples-to-apples with what the policy was tuned against
      * the saved `answer_at` threshold must be a usable number; `ask_at` is optional (its absence just
        collapses the policy to answer/abstain -- see `decide()`)

    Returns {"available": False, "reason": str} when any of the above fails, or {"available": True,
    "band": "answer"|"ask"|"abstain", "score", "score_aggregate", "answer_at", "ask_at"}. Never raises."""
    if not isinstance(saved, dict):
        return {"available": False, "reason": "no calibration saved -- run `clozn eval --save`"}
    pol = saved.get("policy")
    if not isinstance(pol, dict):
        return {"available": False, "reason": "saved calibration carries no policy"}
    saved_model = str(saved.get("model") or "").strip()
    actual_model = str(model or "").strip()
    if not saved_model or not actual_model:
        return {"available": False, "reason": "saved calibration or this reply is missing model provenance"}
    if saved_model != actual_model:
        return {"available": False,
                "reason": f"calibration model {saved_model!r} does not match this reply's model {actual_model!r}"}
    aggregate = str(saved.get("score") or "").strip()
    if aggregate not in ("min", "mean"):
        return {"available": False, "reason": "saved calibration has no supported score aggregate (min or mean)"}
    score = score_from_trace(trace, aggregate)
    if score is None:
        return {"available": False, "reason": "this reply's trace carries no scored content tokens"}
    answer_at = pol.get("answer_at")
    if isinstance(answer_at, bool) or not isinstance(answer_at, (int, float)):
        return {"available": False, "reason": "saved policy has no usable answer_at threshold"}
    ask_at = pol.get("ask_at")
    if isinstance(ask_at, bool) or not isinstance(ask_at, (int, float)):
        ask_at = None
    band = decide(score, float(answer_at), float(ask_at) if ask_at is not None else None)
    return {"available": True, "band": band, "score": round(score, 4), "score_aggregate": aggregate,
            "answer_at": round(float(answer_at), 4),
            "ask_at": (round(float(ask_at), 4) if ask_at is not None else None)}


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
