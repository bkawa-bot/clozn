"""Causal receipt orchestration.

This layer dispatches between regen receipts and teacher-forced receipts. Pure metric math, ablation
assembly, and forced scoring live in sibling modules so this file stays focused on workflow.
"""
from __future__ import annotations

from clozn.replay.replay import replay as replay_run

from .deltas import _ablation_changes, _build_receipt, _key, _merge_ablation_changes
from .forced import forced_receipt
from .metrics import receipt_metrics


_APPROX_NOTE = (
    "prove-all runs leave-one-out over every fired card/dial from the M1 manifest, plus a REDUNDANCY "
    "GUARD that checks PAIRS -- not the full power set -- among influences whose own leave-one-out showed "
    "~no effect. Documented approximation (EXPLAIN_THIS_ANSWER_SPEC.md M2): a 3-way-or-higher redundancy, "
    "where no single pair shows an effect but a larger group does, would be missed by this pairwise check."
)

_PERF_NOTE = (
    "sequential, not batched: one greedy baseline (generated once, reused for every check below) plus one "
    "greedy ablated generation per fired influence, plus one more per redundancy-guard pair. Batching "
    "every leave-one-out arm into a single forward pass is the documented perf follow-up "
    "(EXPLAIN_THIS_ANSWER_SPEC.md M2 cost model), not implemented here."
)


def _receipt_regen(run: dict, influence: dict, sub) -> dict | None:
    """Rigorous regenerated receipt: greedy-with-influence vs greedy-without-influence."""
    try:
        if not run or not isinstance(run, dict):
            return None
        changes = _ablation_changes(influence)
        if not changes:
            return None
        baseline_child = replay_run(run, {"greedy": True}, sub)
        if baseline_child is None:
            return None
        ablated_child = replay_run(run, {**changes, "greedy": True}, sub)
        if ablated_child is None:
            return None
        return _build_receipt(influence, baseline_child, ablated_child, changes)
    except Exception:
        return None


def _fired_influences(manifest: dict):
    """M1 manifest cards + dials as receipt influence specs."""
    influences: list = []
    skipped: list = []
    active = (manifest or {}).get("influences_active") or {}
    for c in active.get("cards") or []:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        if not cid:
            skipped.append({"influence": {"text": c.get("text")}, "reason":
                            "no card id recorded for this application; per-card ablation needs an id"})
            continue
        influences.append({"card_id": cid, "text": c.get("text")})
    for d in active.get("dials") or []:
        if isinstance(d, dict) and d.get("name"):
            influences.append({"dial": d["name"], "value": d.get("value")})
    return influences, skipped


def _prove_all_regen(run: dict, sub, *, manifest: dict | None = None) -> dict:
    """Leave-one-out receipts for every fired influence, plus the pairwise redundancy guard."""
    out = {
        "run_id": run.get("id") if isinstance(run, dict) else None,
        "receipts": [],
        "skipped": [],
        "redundant_pairs": [],
        "approximation_note": _APPROX_NOTE,
        "perf_note": _PERF_NOTE,
    }
    try:
        if not run or not isinstance(run, dict):
            return out
        if manifest is None:
            from . import explain

            manifest = explain.explain(run)
        influences, skipped = _fired_influences(manifest)
        out["skipped"].extend(skipped)
        if not influences:
            return out

        baseline_child = replay_run(run, {"greedy": True}, sub)
        if baseline_child is None:
            out["skipped"].append({"influence": None,
                                   "reason": "could not generate the greedy baseline (with-influence arm)"})
            return out
        baseline_reply = baseline_child.get("response") or ""

        per_key: dict = {}
        for inf in influences:
            changes = _ablation_changes(inf)
            ablated_child = replay_run(run, {**changes, "greedy": True}, sub) if changes else None
            if ablated_child is None:
                out["skipped"].append({"influence": inf, "reason": "ablation could not be generated"})
                continue
            rec = _build_receipt(inf, baseline_child, ablated_child, changes)
            out["receipts"].append(rec)
            per_key[_key(inf)] = (inf, rec["has_effect"])

        no_effect = [k for k, (_, eff) in per_key.items() if not eff]
        for i in range(len(no_effect)):
            for j in range(i + 1, len(no_effect)):
                ka, kb = no_effect[i], no_effect[j]
                joint_changes = _merge_ablation_changes([per_key[ka][0], per_key[kb][0]])
                if not joint_changes:
                    continue
                joint_child = replay_run(run, {**joint_changes, "greedy": True}, sub)
                if joint_child is None:
                    continue
                joint_reply = joint_child.get("response") or ""
                if joint_reply != baseline_reply:
                    out["redundant_pairs"].append({
                        "redundant": [ka, kb],
                        "note": "together they drive this; individually neither is load-bearing",
                    })
        return out
    except Exception:
        return out


def _forced_prove_all(run: dict, sub, manifest: dict | None) -> dict:
    """Forced-mode receipts for every fired influence."""
    out = {"run_id": run.get("id") if isinstance(run, dict) else None, "mode": "forced",
          "forced_receipts": [], "skipped": []}
    try:
        if not run or not isinstance(run, dict):
            return out
        if manifest is None:
            from . import explain

            manifest = explain.explain(run)
        influences, skipped = _fired_influences(manifest)
        out["skipped"].extend(skipped)
        for inf in influences:
            fr = forced_receipt(run, inf, sub)
            if fr is None:
                out["skipped"].append({"influence": inf, "reason": "forced receipt could not be computed"})
                continue
            out["forced_receipts"].append(fr)
    except Exception:
        pass
    return out


def receipt(run: dict, influence: dict, sub, *, mode: str = "regen") -> dict | None:
    """One causal receipt for one influence."""
    mode = mode if mode in ("regen", "forced", "both") else "regen"
    if mode == "regen":
        return _receipt_regen(run, influence, sub)
    if mode == "forced":
        return forced_receipt(run, influence, sub)
    regen = _receipt_regen(run, influence, sub)
    forced = forced_receipt(run, influence, sub)
    if regen is None and forced is None:
        return None
    out = dict(regen or {})
    out["forced"] = forced
    out["mode"] = "both"
    if regen is not None and forced is not None:
        floor = forced.get("null_floor") or {}
        out["silent_influence"] = bool(not regen.get("has_effect")
                                       and floor.get("exceeds_floor_by_order_of_magnitude"))
    return out


def prove_all(run: dict, sub, *, manifest: dict | None = None, mode: str = "regen") -> dict:
    """Leave-one-out receipts for every fired influence."""
    mode = mode if mode in ("regen", "forced", "both") else "regen"
    if mode == "regen":
        return _prove_all_regen(run, sub, manifest=manifest)
    forced_out = _forced_prove_all(run, sub, manifest)
    if mode == "forced":
        return forced_out
    out = _prove_all_regen(run, sub, manifest=manifest)
    out["mode"] = "both"
    out["forced_receipts"] = forced_out["forced_receipts"]
    if forced_out.get("skipped"):
        out["skipped"] = list(out.get("skipped") or []) + forced_out["skipped"]
    return out
