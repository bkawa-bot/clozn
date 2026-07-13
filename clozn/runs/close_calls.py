"""Close calls -- the near-tie locator (the honest, free "where to probe" signal that replaces raw
chosen-token probability as the headline).

A "close call" is a generation step where the top token and the runner-up were nearly as likely as each
other: a coin-flip decision -- exactly where a branch-stability test would pay off. Computed PURELY from
the recorded top-k `alternatives` (one consistent distribution per step, so no scale mismatch with the
`confidence` field), zero re-runs.

TRUTH CONDITIONS (binding, per the terminology reframe): a close call is CORRELATIONAL, a locator, never
a verdict. It says "this decision was nearly a toss-up between X and Y", never "wrong" and never "fragile"
-- "fragile" is earned only after an actual branch-stability test forces the runner-up and shows the
answer diverges. This module only points at where to run that test.
"""
from __future__ import annotations

# Thresholds tuned against the real 202-run journal (2026-07-13): looser values flag ~58% of runs, almost
# all HARMLESS phrasing/punctuation forks ("or" vs "("). These keep it to genuine near-even splits between
# two CONTENT tokens (~3% of runs) -- rare enough to stay exception-only, meaningful enough to be worth a
# branch-stability test. Deliberately conservative: we would rather miss a stylistic fork than cry wolf.
MARGIN = 0.10         # top-1 prob minus runner-up prob <= this => a near-tie
MIN_RUNNERUP = 0.35   # ...and BOTH were genuine contenders (a real two-way split, not a spread)


def _pieces(cand: dict) -> str:
    return str(cand.get("piece") or cand.get("text") or "").strip()


def _contentful(piece: str) -> bool:
    """A content-ish token: >=2 chars with a letter. Filters the punctuation/whitespace/one-char forks
    ("or" vs "(", " " vs ",") that are near-ties but never meaningful -- the journal's dominant noise."""
    p = (piece or "").strip()
    return len(p) >= 2 and any(c.isalpha() for c in p)


def close_calls(run: dict | None) -> list[dict]:
    """[{index, top, top_prob, alt, alt_prob, margin}] for every meaningful near-tie step (a genuine
    two-way split between two content tokens). Pure over the trace's `alternatives`; never raises."""
    try:
        trace = run.get("trace") if isinstance(run, dict) else None
        alts = (trace or {}).get("alternatives") if isinstance(trace, dict) else None
        if not isinstance(alts, list):
            return []
        out = []
        for i, cand in enumerate(alts):
            if not isinstance(cand, list) or len(cand) < 2:
                continue
            p0, p1 = cand[0].get("prob"), cand[1].get("prob")
            if not isinstance(p0, (int, float)) or not isinstance(p1, (int, float)):
                continue
            top, alt = _pieces(cand[0]), _pieces(cand[1])
            if not (_contentful(top) and _contentful(alt)):
                continue
            margin = float(p0) - float(p1)
            if float(p1) >= MIN_RUNNERUP and margin <= MARGIN:
                out.append({"index": i, "top": top, "top_prob": round(float(p0), 3),
                            "alt": alt, "alt_prob": round(float(p1), 3), "margin": round(margin, 3)})
        return out
    except Exception:
        return []


def tightest(calls: list[dict]) -> dict | None:
    """The single closest call (smallest margin) -- the one worth naming."""
    return min(calls, key=lambda c: c.get("margin", 1.0)) if calls else None


def summarize(calls: list[dict]) -> str:
    """'' if none; else 'N close call(s)' + the tightest one named ('nearly "X" over "Y"'). Honest,
    concrete, and non-alarming -- a close call between two words is a true statement, not a warning."""
    if not calls:
        return ""
    n = len(calls)
    head = f"{n} close call{'s' if n != 1 else ''}"
    t = tightest(calls)
    if t and t["alt"] and t["top"]:
        head += f" · nearly “{t['alt']}” over “{t['top']}”"
    return head
