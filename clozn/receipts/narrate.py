"""Top-level accountable narration assembly."""
from __future__ import annotations

from . import explain
from .claim_extraction import _split_claims, clause_split
from .confabulation_diff import confabulation_diff
from .fact_support import (
    _as_dict,
    _as_list,
    _citable_facts,
    _influence_lexicon,
    lexical_default,
    semantic_support_matcher,
)
from .narrative_rendering import constrained_narration, unconstrained_why


def narrate(run: dict, sub, support_matcher=lexical_default, claim_splitter=clause_split) -> dict:
    """Return constrained narration plus flags from the receipt-free self-report diff."""
    try:
        explanation = explain.explain(run)
    except Exception:
        explanation = explain.explain(None)

    try:
        cn = constrained_narration(explanation, sub)
    except Exception:
        cn = {"narration": "", "receipt_ids": []}

    why_text = ""
    try:
        why = unconstrained_why(run, sub)
        why_text = why.get("unconstrained_text_context_only", "") if isinstance(why, dict) else ""
    except Exception:
        why_text = ""

    try:
        diff = confabulation_diff(why_text, explanation, support_matcher=support_matcher,
                                  claim_splitter=claim_splitter)
    except Exception:
        diff = {"unsupported_claims": [], "matcher": getattr(support_matcher, "__name__", repr(support_matcher))}

    unsupported = diff.get("unsupported_claims") or []
    flags = [e.get("flag") for e in unsupported if isinstance(e, dict) and e.get("flag")]
    note = (
        f"constrained_narration is the answer surface; the model's unconstrained self-report is never "
        f"included here (THE TRAP guard) -- only its diff against the receipts, as flags, is. "
        f"Matcher used: {diff.get('matcher')}. If that is lexical_default: it is a weak keyword-overlap "
        f"proxy that both over- and under-flags; an absent flag is not proof a claim is true, and a present "
        f"flag is not proof it is false. The real semantic judgment is deferred, gated, on-model work."
    )
    return {
        "constrained_narration": cn,
        "flags": flags,
        "unsupported_claims": unsupported,
        "note": note,
    }
