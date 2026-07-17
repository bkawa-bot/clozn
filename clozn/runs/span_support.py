"""Attach an explicit SUPPORT channel to confidence spans.

The channel answers one narrow question: does an active, recorded influence premise (a memory card or
dial receipt from ``receipts.explain``) entail this output span according to an injected matcher? It is
not source retrieval, web evidence, or a factual-correctness verdict. The production route injects the
optional independent NLI cross-encoder only when the caller explicitly requests support; importing and
testing this module loads no model and no Torch.
"""
from __future__ import annotations


NOTE = ("support checks whether an active recorded influence entails the span. It is not external-source "
        "evidence and not a factual verdict. NLI-unavailable/error results stay unavailable; no lexical "
        "fallback is relabeled as NLI support.")


def attach(spans: list[dict], explanation: dict, matcher) -> tuple[list[dict], dict]:
    """Return copied spans with ``support`` objects plus a run-level summary.

    ``matcher`` uses the existing ``semantic_matcher`` contract: ``matcher(claim, explanation)`` returns
    supported/score/method/provenance fields. Failures are recorded per span and never escape. A labeled
    ``nli-unavailable`` or ``nli-error`` result is not interpreted as unsupported evidence; it is
    unavailable, preserving the distinction between "checked and no premise entailed it" and "not checked."
    """
    copied = [dict(s) for s in (spans or []) if isinstance(s, dict)]
    if not callable(matcher):
        return copied, {"requested": True, "available": False, "reason": "no NLI matcher available", "note": NOTE}

    out = []
    n_supported = n_unsupported = n_unavailable = 0
    methods = set()
    for span in copied:
        claim = str(span.get("text") or "").strip()
        if not claim:
            support = {"available": False, "entailed": None, "method": "no-text",
                       "note": "span has no text to check"}
        else:
            try:
                raw = matcher(claim, explanation)
            except Exception as exc:                         # noqa: BLE001 -- evidence degrades, never crashes
                raw = {"method": "matcher-error", "note": f"matcher raised {type(exc).__name__}"}
            raw = raw if isinstance(raw, dict) else {"method": "matcher-error", "note": "matcher returned no result"}
            method = str(raw.get("method") or "unknown")
            unavailable = method in ("nli-unavailable", "nli-error", "matcher-error")
            support = {
                "available": not unavailable,
                "entailed": (bool(raw.get("supported")) if not unavailable else None),
                "score": raw.get("score"),
                "threshold": raw.get("threshold"),
                "matched_id": raw.get("matched_id"),
                "closest_id": raw.get("closest_id"),
                "contradiction": raw.get("contradiction"),
                "method": method,
                "note": raw.get("note"),
            }
        methods.add(support["method"])
        if not support["available"]:
            n_unavailable += 1
        elif support["entailed"]:
            n_supported += 1
        else:
            n_unsupported += 1
        out.append({**span, "support": support})

    available = (n_supported + n_unsupported) > 0
    return out, {
        "requested": True,
        "available": available,
        "n_spans": len(out),
        "n_entailed": n_supported,
        "n_not_entailed": n_unsupported,
        "n_unavailable": n_unavailable,
        "methods": sorted(methods),
        "note": NOTE,
    }
