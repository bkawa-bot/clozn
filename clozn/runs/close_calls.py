"""Close calls — the answer-fork locator.

A "close call" is a generation step where the model's probability mass split near-evenly between two
tokens: a coin-flip decision — exactly where a branch-stability test would pay off. Computed PURELY from
the recorded trace (zero re-runs) by reconstructing each step's top-k distribution.

TRUTH CONDITIONS (binding, per the fragility/stability terminology). A close call is CORRELATIONAL — a
LOCATOR, never a verdict. It says "this step was nearly a toss-up between X and Y", never "wrong" and never
"fragile". "Fragile" is earned only after a real branch-stability test forces the runner-up and shows the
answer diverges; this module only points at WHERE to run that test.

DATA (settled 2026-07-13, after an engine-trace audit). The engine records, per generated token, the
emitted token's softmax probability in `trace.confidence[i]` and the OTHER top-k tokens in
`trace.alternatives[i]` — all on ONE consistent softmax scale (verified empirically: confidence[i] +
sum(alt probs) has median 1.00 over the real journal). The emitted token is EXCLUDED from `alternatives`
(it lives in `tokens`/`confidence`). So a genuine near-tie is CHOSEN-vs-strongest-rival, reconstructed by
putting the emitted token BACK into the distribution and taking the two co-leaders — NOT by comparing
alternatives[0] vs [1] (two roads-NOT-taken, the old bug). No engine change was needed; the data was
already right, only this reconstruction was wrong.

WHERE IT BELONGS. Per-token close calls are COMMON under sampling — a flat top-2 between two harmless
function words ("as" vs "is") fires on a majority of runs — so this is a STUDIO locator (surface every
fork when you're inspecting one run), NOT an exception-only footer signal. The rare, meaning-changing
subset — a coin-flip between two DIGITS, or a polarity flip (a negation vs a non-negation) — is flagged
`meaningful` (≈4% of journal runs) for callers that want only the answer-changing slice.
"""
from __future__ import annotations

# A per-step near-tie: the two highest-probability tokens (emitted ∪ alternatives, one distribution) fall
# within MARGIN of each other and BOTH clear MIN_RUNNERUP — a real two-way split, not a long flat spread.
MARGIN = 0.12
MIN_RUNNERUP = 0.30

_NEGATIONS = {"not", "no", "never", "n't", "'t", "cannot", "can't", "won't", "don't", "without",
              "none", "nothing", "neither", "nor"}


def _piece(cand: dict) -> str:
    return str(cand.get("piece") or cand.get("text") or "").strip()


def _contentful(piece: str) -> bool:
    """Content-bearing: filters the punctuation/whitespace/one-char forks ("or" vs "(", " " vs ",") that
    are near-ties but never meaningful — the journal's dominant close-call noise. A digit passes even at one
    char ("5" vs "0" IS answer-bearing); an alpha token needs >=2 chars (drops bare "a"/"I")."""
    p = (piece or "").strip()
    if any(c.isdigit() for c in p):
        return True
    return len(p) >= 2 and any(c.isalpha() for c in p)


def _is_digit(piece: str) -> bool:
    return any(c.isdigit() for c in (piece or ""))


def _is_negation(piece: str) -> bool:
    # normalize curly/typographic apostrophes so "'t" and "’t" (the same contraction ending, different
    # encodings) read as ONE negation -- else the pair looks like a polarity flip and false-fires.
    p = (piece or "").strip().lower().replace("’", "'").replace("‘", "'").replace("＇", "'")
    return p in _NEGATIONS


def _is_meaningful(a: str, b: str) -> bool:
    """The fork changes the answer's substance: two DIFFERENT digits, or a polarity flip (exactly one side
    is a negation). "as" vs "is" is a close call but not meaningful; "5" vs "0" and "not" vs "they" are."""
    if _is_digit(a) and _is_digit(b) and a.strip() != b.strip():
        return True
    return _is_negation(a) != _is_negation(b)


def _coleaders(emitted_piece: str, emitted_prob, alts) -> tuple | None:
    """Reconstruct a step's top tokens = {emitted} ∪ alternatives (deduped by piece, keeping the max prob),
    return the two highest as ((piece, prob), (piece, prob)) or None if fewer than two usable candidates."""
    by_piece: dict[str, float] = {}

    def add(piece, prob):
        if isinstance(prob, (int, float)):
            p = str(piece or "").strip()
            if p and float(prob) > by_piece.get(p, -1.0):
                by_piece[p] = float(prob)

    add(emitted_piece, emitted_prob)
    for a in (alts or []):
        if isinstance(a, dict):
            add(_piece(a), a.get("prob"))
    if len(by_piece) < 2:
        return None
    ranked = sorted(by_piece.items(), key=lambda kv: -kv[1])
    return ranked[0], ranked[1]


def close_calls(run: dict | None) -> list[dict]:
    """[{index, top, top_prob, alt, alt_prob, margin, emitted, meaningful}] for every step whose two
    co-leading tokens were a near-even split between two CONTENT tokens. `top`/`alt` are the two co-leaders
    by probability (the model's two strongest options); `emitted` is what it actually chose — it equals
    `top` on an argmax pick, or `alt` if the model sampled the runner-up. Pure over the trace; never raises."""
    try:
        trace = run.get("trace") if isinstance(run, dict) else None
        if not isinstance(trace, dict):
            return []
        toks = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
        conf = trace.get("confidence") if isinstance(trace.get("confidence"), list) else []
        alts = trace.get("alternatives") if isinstance(trace.get("alternatives"), list) else []
        n = min(len(toks), len(conf), len(alts))
        out = []
        for i in range(n):
            try:
                pair = _coleaders(toks[i], conf[i], alts[i] if isinstance(alts[i], list) else [])
                if not pair:
                    continue
                (top, p_top), (alt, p_alt) = pair
                if not (_contentful(top) and _contentful(alt)):
                    continue
                if p_alt < MIN_RUNNERUP or (p_top - p_alt) > MARGIN:
                    continue
                out.append({"index": i, "top": top, "top_prob": round(p_top, 3),
                            "alt": alt, "alt_prob": round(p_alt, 3), "margin": round(p_top - p_alt, 3),
                            "emitted": str(toks[i] or "").strip(), "meaningful": _is_meaningful(top, alt)})
            except Exception:
                continue                           # one malformed step skips itself, not the run
        return out
    except Exception:
        return []


def meaningful(calls: list[dict]) -> list[dict]:
    """The rare, meaning-changing subset (digit forks + polarity flips) — the exception-only slice."""
    return [c for c in calls if c.get("meaningful")]


def tightest(calls: list[dict]) -> dict | None:
    """The single closest call (smallest margin) — the one worth naming."""
    return min(calls, key=lambda c: c.get("margin", 1.0)) if calls else None


def summarize(calls: list[dict]) -> str:
    """'' if none; else 'N close call(s)' + the tightest named ('nearly "X" vs "Y"'). Correlational and
    non-alarming — a coin-flip between two words is a true statement, not a warning."""
    if not calls:
        return ""
    n = len(calls)
    head = f"{n} close call{'s' if n != 1 else ''}"
    t = tightest(calls)
    if t and t["top"] and t["alt"]:
        head += f" · nearly “{t['top']}” vs “{t['alt']}”"
    return head
