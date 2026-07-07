"""confidence_spans.py -- reshape a stored run's per-token confidence trace into a handful of contiguous
SPANS, so the *shape* of a long reply's certainty is legible at a glance: "started strong, went shaky in
the middle, recovered." Sibling of run_timeline.py (same contract shape: a pure `spans(run) -> list[dict]`,
zero model calls, zero generation) -- but where run_timeline answers "what happened, when" as an event
sequence, this module answers "how sure was it, and where" as a small number of banded regions instead of
one confidence dot per token (noise, once a reply runs long).

This is a RESHAPE, not a new signal: every span is built purely from trace["tokens"]/["confidence"], the
same per-token trace explain.py's _confidence and run_timeline.py's _hesitations already read. Nothing here
is invented, estimated, or synthesized -- a span only ever reports numbers that were already measured at
generation time (or an honest 1.0 for a token that recorded no confidence at all).

Segmentation (see `spans()` for the exact walk):
  * each token is banded by its own confidence: >= STRONG "strong", >= LOW_CONF "okay", < LOW_CONF "shaky".
  * a span is a maximal run of same-band tokens that never crosses a sentence boundary -- so "started
    strong, went shaky" shows up as two (or more) spans, and a long confident paragraph doesn't get
    artificially chopped mid-sentence just because a sentence happened to end on a "strong" token.
  * no smoothing/merging beyond that in v1: the 3 bands already coalesce short runs, and honest oscillation
    between bands (a reply that flickers strong/shaky/strong) should show up AS alternating spans, not get
    papered over into a false sense of steadiness.

`summarize()` turns the span list into one honest one-liner -- never model narration, never a claim the
span stats don't directly support: "Confident throughout." when there's no shaky span at all, otherwise a
count of shaky spans plus WHERE the weakest one sits (opening/middle/close), derived purely from
start-index arithmetic over the spans already computed.

Never raises: `spans()` reduces a non-dict (or empty) run, or a run with no trace tokens, to [] rather than
erroring; `summarize()` reduces anything it can't read to "" (the same "nothing to say" as an empty input).
Zero imports beyond stdlib `re` (for the sentence-boundary check) -- no model, no substrate, no GPU.
"""
from __future__ import annotations

import re

# Matches explain.py's LOW_CONF / run_timeline.py's LOW_CONF (which itself matches inspector/demo/pages/
# run.js) -- ONE "unsure" cutoff read in all these places, so a span's "shaky" band never disagrees with
# the Explain panel's uncertain_moments or the timeline's hesitation events about what counts as unsure.
# Kept as its own constant (not imported from a sibling) so this module stays zero-dependency -- if one
# changes, change them all.
LOW_CONF = 0.5

STRONG = 0.8                              # >= this is "strong"; [LOW_CONF, STRONG) is "okay" (see _band)

# Sentence-ending punctuation, allowing a trailing closing quote/bracket ('She said "stop."' / '(Really?)').
# Checked against the RIGHT-STRIPPED token piece so trailing whitespace on a piece never hides a sentence
# end. Precompiled since every token in the trace gets checked.
_SENT_END = re.compile(r"""[.!?]["')\]]*$""")


def _as_list(x) -> list:
    return x if isinstance(x, list) else []


def _as_dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _conf(confidence: list, i: int) -> float:
    """Token i's confidence -- or 1.0 (certain) when absent/unparseable. Missing confidence reads as
    certain, never as uncertain: matches inspector/demo/pages/run.js's `conf[j] == null ? 1`."""
    v = confidence[i] if i < len(confidence) else None
    if v is None:
        return 1.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 1.0


def _band(c: float) -> str:
    """A token's confidence band: strong / okay / shaky, split at STRONG and LOW_CONF."""
    if c >= STRONG:
        return "strong"
    if c >= LOW_CONF:
        return "okay"
    return "shaky"


def _ends_sentence(piece: str) -> bool:
    """Does this token piece, right-stripped, end a sentence (terminal punctuation + optional closing
    quote/bracket)? A span never crosses this boundary, even mid-band."""
    return bool(_SENT_END.search(piece.rstrip()))


def _span(tokens: list, confs: list, bands: list, start: int, end: int) -> dict:
    """One span dict, start/end INCLUSIVE token indices. text is the verbatim join (tokens already carry
    their own leading spaces); mean/min are over this span's own per-token confidences (already resolved
    via `_conf`, so a missing confidence already reads as 1.0 here, not as a hole)."""
    seg = confs[start:end + 1]
    n = end - start + 1
    return {
        "start": start, "end": end, "text": "".join(tokens[start:end + 1]), "band": bands[start],
        "mean_conf": round(sum(seg) / n, 4), "min_conf": round(min(seg), 4), "n_tokens": n,
        "hesitations": sum(1 for c in seg if c < LOW_CONF),
    }


# ------------------------------------------------------------------------------------------------------ API
def spans(run: dict | None) -> list[dict]:
    """Segment one run's trace into confidence spans (as returned by runlog.get_run()). A span is a maximal
    contiguous run of same-band tokens that never crosses a sentence boundary -- no other smoothing.

    Never raises: a non-dict/empty run, or one with no trace tokens, degrades to [] rather than erroring;
    anything else unexpected while walking the trace also degrades to [] (this is a pure reshape -- a
    partial, possibly-misleading span list is worse than an honest empty one)."""
    run = run if isinstance(run, dict) else {}
    if not run:
        return []
    trace = _as_dict(run.get("trace"))
    tokens = _as_list(trace.get("tokens"))
    if not tokens:
        return []
    try:
        confidence = _as_list(trace.get("confidence"))
        confs = [_conf(confidence, i) for i in range(len(tokens))]
        bands = [_band(c) for c in confs]

        out = []
        start = 0
        for i in range(1, len(tokens)):
            if bands[i] != bands[i - 1] or _ends_sentence(tokens[i - 1]):
                out.append(_span(tokens, confs, bands, start, i - 1))
                start = i
        out.append(_span(tokens, confs, bands, start, len(tokens) - 1))
        return out
    except Exception:
        return []


def summarize(spans: list[dict]) -> str:
    """One honest one-liner derived purely from the span stats already computed -- never model narration,
    never a claim beyond what the numbers show. [] -> "". No shaky span -> "Confident throughout."
    Otherwise: how many shaky spans, plus WHERE the weakest one (lowest min_conf) sits, by start-index
    thirds of the reply (opening / middle / close). Never raises: anything it can't read degrades to ""
    (the same "nothing to say" as an empty span list)."""
    try:
        if not spans:
            return ""
        shaky = [s for s in spans if s.get("band") == "shaky"]
        if not shaky:
            return "Confident throughout."
        total = sum(int(s.get("n_tokens", 0) or 0) for s in spans) or 1
        weakest = min(shaky, key=lambda s: s.get("min_conf", 1.0))
        pos = weakest.get("start", 0)
        if pos < total / 3:
            position = "opening"
        elif pos >= 2 * total / 3:
            position = "close"
        else:
            position = "middle"
        k = len(shaky)
        return f"Mostly steady, but {k} shaky span{'' if k == 1 else 's'} (weakest in the {position})."
    except Exception:
        return ""
