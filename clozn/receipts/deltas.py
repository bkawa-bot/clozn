"""Influence ablation changes and receipt object assembly."""
from __future__ import annotations

from .metrics import receipt_metrics


_NOTE_BASELINE = (
    "the run's stored sampled reply is NOT the baseline for this receipt -- greedy-with-the-influence is. "
    "The sampled reply is context only; it is never a term in this subtraction "
    "(EXPLAIN_THIS_ANSWER_SPEC.md M2: diffing sampled-vs-greedy would mix two changes at once)."
)


def _key(influence: dict) -> str:
    influence = influence or {}
    if influence.get("card_id"):
        return f"card:{influence['card_id']}"
    if influence.get("dial"):
        return f"dial:{influence['dial']}"
    if influence.get("memory_off"):
        return "memory_off"
    if influence.get("behavior_off"):
        return "behavior_off"
    return "unknown"


def _ablation_changes(influence: dict) -> dict | None:
    """One influence spec -> replay changes for its ablated arm."""
    if not isinstance(influence, dict):
        return None
    cid = influence.get("card_id")
    if cid:
        return {"disabled_memory_ids": [str(cid)]}
    dial = influence.get("dial")
    if dial:
        return {"behavior_overrides": {str(dial): 0.0}}
    if influence.get("memory_off"):
        return {"memory_off": True}
    if influence.get("behavior_off"):
        return {"behavior_off": True}
    return None


def _merge_ablation_changes(influences: list) -> dict:
    """Joint replay changes that ablate every influence at once."""
    ids: list = []
    overrides: dict = {}
    memory_off = behavior_off = False
    for inf in influences:
        c = _ablation_changes(inf) or {}
        ids.extend(c.get("disabled_memory_ids") or [])
        overrides.update(c.get("behavior_overrides") or {})
        memory_off = memory_off or bool(c.get("memory_off"))
        behavior_off = behavior_off or bool(c.get("behavior_off"))
    merged: dict = {}
    if ids:
        merged["disabled_memory_ids"] = ids
    if overrides:
        merged["behavior_overrides"] = overrides
    if memory_off:
        merged["memory_off"] = True
    if behavior_off:
        merged["behavior_off"] = True
    return merged


def _cost_note(influence: dict) -> str:
    influence = influence or {}
    if influence.get("card_id") or influence.get("memory_off"):
        return ("cost: a front-of-context memory ablation changes the shared prefix, so the ablated arm "
                "re-prefills the whole context (no KV reuse) -- the expensive case.")
    return ("cost: a dial ablation acts at decode time, so the prompt KV stays reusable -- cheap relative "
            "to a memory ablation.")


def _unapplied_note(ablated_child: dict, changes: dict) -> str | None:
    notes = ((ablated_child or {}).get("memory") or {}).get("notes") or {}
    if changes.get("disabled_memory_ids") and "disabled_memory_ids" in notes:
        return notes["disabled_memory_ids"]
    if changes.get("edited_memory") and "edited_memory" in notes:
        return notes["edited_memory"]
    return None


def _build_receipt(influence: dict, baseline_child: dict, ablated_child: dict, changes: dict) -> dict:
    baseline_reply = baseline_child.get("response") or ""
    ablated_reply = ablated_child.get("response") or ""
    unapplied = _unapplied_note(ablated_child, changes)
    out = {
        "influence": influence,
        "changes_applied": changes,
        "baseline_reply": baseline_reply,
        "ablated_reply": ablated_reply,
        "delta": receipt_metrics(baseline_reply, ablated_reply),
        "has_effect": baseline_reply != ablated_reply,
        "causal_verified": unapplied is None,
        "note": _NOTE_BASELINE,
        "cost_note": _cost_note(influence),
    }
    if unapplied:
        out["ablation_note"] = unapplied
    return out
