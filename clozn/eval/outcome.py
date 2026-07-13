"""Outcome evaluators -- turn a model's answer + a gold answer into a hard correct/incorrect bool, so
calibration can rest on TRUTH (this eval set) instead of the acceptance PROXY that actuary.py and
calibrated_trust.py use.

Deliberately narrow and exact. Each matcher is NAMED (exact / numeric / mcq) and does ONE checkable thing,
so "correct" is never a vibe. An item we cannot grade -- no gold, or a kind that doesn't apply -- returns
None: excluded from coverage, NEVER silently counted as wrong. Short-answer models wrap the answer in
fluff ("The capital is Paris."), so exact-match accepts the gold appearing as a contiguous phrase in the
normalized reply, not just full-string equality.
"""
from __future__ import annotations

import re

_ARTICLES = {"the", "a", "an"}
_NUM = re.compile(r"-?\d[\d,]*\.?\d*")
_LETTER = re.compile(r"\b([A-Ea-e])\b")


def normalize(s: str) -> str:
    """Casefold, strip punctuation to spaces, drop leading/trailing articles -- a conservative canonical
    form so "Paris." / "paris" / "the Paris" compare equal without being so aggressive it merges distinct
    answers."""
    s = re.sub(r"[^\w\s]", " ", (s or "").casefold())
    return " ".join(t for t in s.split() if t not in _ARTICLES)


def _phrase_in(hay: str, needle: str) -> bool:
    """Does the token sequence `needle` appear contiguously in `hay`? (Word-boundary phrase match, so
    "paris" matches "... is paris" but not "parisian".)"""
    h, n = hay.split(), needle.split()
    if not n:
        return False
    return any(h[i:i + len(n)] == n for i in range(len(h) - len(n) + 1))


def exact_match(pred: str, gold: str, aliases: tuple = ()) -> bool:
    """True if the normalized gold (or any alias) appears as a contiguous phrase in the normalized reply."""
    hay = normalize(pred)
    return any(_phrase_in(hay, normalize(g)) for g in (gold, *aliases) if normalize(g))


def _nums(s: str) -> list[float]:
    out = []
    for m in _NUM.findall(s or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def numeric_match(pred: str, gold, tol: float = 0.0) -> bool | None:
    """True if any number in `pred` is within `tol` of the gold number. None if the GOLD carries no number
    (ungradeable); False if the gold is numeric but the prediction produced no matching number."""
    g = _nums(str(gold))
    if not g:
        return None
    target = g[0]
    return any(abs(x - target) <= tol for x in _nums(pred))


def _letter(s: str) -> str | None:
    m = _LETTER.search(s or "")
    return m.group(1).upper() if m else None


def mcq_letter(pred: str, gold) -> bool | None:
    """True if the option letter (A-E) in `pred` matches the gold letter. None if the gold has no letter."""
    g = _letter(str(gold))
    if g is None:
        return None
    p = _letter(pred)
    return (p == g) if p is not None else False


def grade(pred: str, gold, kind: str = "exact", **kw) -> bool | None:
    """Dispatch to the named matcher. Returns None (ungradeable) rather than raising on a bad kind, so a
    single odd item can't sink a whole eval run -- but an EMPTY prediction with a real gold is a hard miss
    (False), never None: refusing/blanking is answering wrong, and selective coverage must count it."""
    if not str(gold).strip():
        return None
    if kind == "exact":
        return exact_match(pred, gold, tuple(kw.get("aliases", ())))
    if kind == "numeric":
        return numeric_match(pred, gold, float(kw.get("tol", 0.0)))
    if kind == "mcq":
        return mcq_letter(pred, gold)
    return None
