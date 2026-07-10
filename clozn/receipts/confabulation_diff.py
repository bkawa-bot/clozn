"""Diff unconstrained self-narration claims against measured support."""
from __future__ import annotations

from .claim_extraction import clause_split
from .fact_support import _as_dict, lexical_default


_WARNING_TEMPLATE = 'WARNING: credits "{claim}"; no receipt for that.'

_DIFF_NOTE = (
    "Claims are split by `clause_split` (the default `claim_splitter`): sentence-level, then a further split "
    "of each sentence on coordinating boundaries (';' and ', and/but/so/yet ') so a COMPOUND sentence "
    "crediting two different things is judged as separate claims. Support is judged by the pluggable "
    "`support_matcher` (default: lexical_default, a weak keyword-overlap proxy). A matcher that raises on a "
    "claim is treated as unsupported for it."
)


def confabulation_diff(unconstrained_text: str, explanation: dict, support_matcher=lexical_default,
                       claim_splitter=clause_split) -> dict:
    """Split unconstrained text into claims and tag each one supported/unsupported."""
    explanation = _as_dict(explanation)
    claims_out: list[dict] = []
    unsupported: list[dict] = []
    rendered: list[str] = []

    for claim in claim_splitter(unconstrained_text):
        try:
            result = support_matcher(claim, explanation)
        except Exception:
            result = None
        if not isinstance(result, dict):
            result = {"supported": False}

        entry = {k: v for k, v in result.items() if k != "supported"}
        supported = bool(result.get("supported"))
        entry = {"claim": claim, "supported": supported, **entry}

        if supported:
            entry["flag"] = None
            rendered.append(claim)
        else:
            flag = _WARNING_TEMPLATE.format(claim=claim)
            entry["flag"] = flag
            unsupported.append(entry)
            rendered.append(flag)
        claims_out.append(entry)

    return {
        "claims": claims_out,
        "unsupported_claims": unsupported,
        "flagged_rendering": " ".join(rendered),
        "matcher": getattr(support_matcher, "__name__", repr(support_matcher)),
        "note": _DIFF_NOTE,
    }
