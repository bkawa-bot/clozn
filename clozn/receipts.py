"""receipts.py -- Milestone 2 (EXPLAIN_THIS_ANSWER_SPEC.md): on-demand RIGOROUS causal receipts.

THE SEAM THIS FIXES (read this before touching anything below): the pre-M2 "receipt" -- still what
run.js's per-card/per-dial buttons drive today, via a single POST /runs/<id>/replay -- diffs the run's
ORIGINAL SAMPLED reply against a greedy replay with one influence toggled off. That mixes TWO changes at
once (influence on->off AND sampled->greedy decoding), so the delta is not cleanly attributable to the
influence alone; some of it is just sampling noise. The rigorous fix, here: regenerate BOTH arms GREEDY --
a greedy-WITH-the-influence baseline, and a greedy-WITHOUT-it ablated arm -- from the SAME stored
messages, and diff *those two*. The run's stored sampled reply is never a term in the subtraction; it is
context only, and every receipt says so in its own `note` field (the spec's honesty bullet).

This module owns exactly two things: (1) turning one "influence spec" into the replay.py `changes` dict
for its ablated arm, and (2) the pure delta-strip math (word count / words-per-sentence / %-of-wording-
changed) that mirrors studio/pages/run.js's `receiptMetrics()` EXACTLY, so the client's JS and this
server-side Python can never silently disagree about what a delta strip says. It does NOT reimplement any
state discipline: every generation goes through `replay.replay()`, which already owns the snapshot / apply-
changes / restore-in-a-finally contract (never leaves the live studio mutated) and the greedy flag
(deterministic decoding, so a diff is attributable to the change, not to sampling dice).

Two public entry points:

  * receipt(run, influence, sub) -- one rigorous receipt for one influence. `influence` is exactly one of:
        {"card_id": <id>}    -- ablate exactly this one memory card (disabled_memory_ids=[id]); a REAL
                                per-card ablation only in "prompt" memory mode (replay.py's own docstring).
                                In "internalized" mode the cards are fused into one trained prefix and
                                replay.py can't remove just one -- its own honest "not applied" note is
                                relayed here as `ablation_note`, and `causal_verified` is correctly False
                                rather than silently claiming a no-op proves "no effect".
        {"dial": <name>}     -- ablate exactly this one tone dial (behavior_overrides={name: 0.0}),
                                leaving every OTHER currently-active dial untouched: true leave-one-out.
        {"memory_off": true} -- the whole memory subsystem off (memory_strength -> 0).
        {"behavior_off": true} -- every tone dial neutral.
      Returns None on any failure (a bad/empty spec, or the substrate/chat couldn't generate) -- never
      raises, mirroring replay.replay's own contract exactly.

  * prove_all(run, sub) -- leave-one-out over every influence the M1 manifest
      (explain.explain(run)["influences_active"]) says FIRED on this run (every card with a resolvable id,
      every active dial), plus the REDUNDANCY GUARD: among the influences whose OWN leave-one-out receipt
      shows ~no effect, every PAIR is re-checked by ablating both at once; if that joint ablation DOES show
      an effect, the pair is reported as redundant ("together they drive this; individually neither is
      load-bearing") instead of the misleading "neither mattered" a naive leave-one-out would imply. This
      is a documented APPROXIMATION -- pairs only, not the full power set of fired influences -- see
      `_APPROX_NOTE` on the returned object; a 3-way-or-higher redundancy (no single pair shows an effect,
      but some larger group does) would be missed by it.
      "No effect" is judged by exact reply-STRING equality: greedy decoding is deterministic, so an
      identical string under ablation is the strongest, simplest honest signal that nothing measurable
      changed -- not an eyeballed threshold on the delta metrics.
      One baseline (greedy-WITH, no changes) is generated ONCE per prove_all() call and reused for every
      single-influence receipt AND every redundancy-guard pair check. This is safe -- not a shortcut on
      rigor -- because it is the SAME deterministic reply replay.replay(run, {"greedy": True}, sub) would
      produce every time on an unchanged live substrate; it is NOT the batched-forward-pass optimization
      the spec's cost model names as the next perf step (that needs the substrate itself to batch several
      variants through one forward pass, which is out of scope here -- see `_PERF_NOTE`, and NEXT_STEPS'
      note on this as the documented follow-up).

Stdlib-only, duck-typed against the live substrate exactly like replay.py: no torch import, no model, no
GPU -- every generation happens by calling `replay.replay(run, changes, sub)`, so this module is fully
unit-testable against a fake substrate with a canned, deterministic `.chat()`.
"""
from __future__ import annotations

import math
import re

from clozn import replay

# ============================================================================================ metric math
# Mirrors studio/pages/run.js's receiptMetrics() EXACTLY (word count, words/sentence, and a
# word-TYPE Jaccard-distance %) so the client's delta strip and this server-side one can never silently
# drift apart. The one deliberate non-literal-transliteration: JS's `for (k in oset)` / `rset[k]` lookups
# walk a plain-object "set" (a latent prototype-chain gotcha for a word that collides with an
# Object.prototype member, e.g. "constructor") -- real Python sets compute what that code is actually
# TRYING to compute, correctly, for every realistic word.
_WORD_RE = re.compile(r"[a-z0-9']+")
_SENT_SPLIT_RE = re.compile(r"[.!?]+")


def _text(s) -> str:
    """Mirrors JS's `String(s || "")`: a falsy value (None, "", 0, False) becomes "", anything else is
    stringified."""
    if not s:
        return ""
    return s if isinstance(s, str) else str(s)


def _words_of(s) -> list:
    """run.js's wordsOf: lowercase, then every maximal run of [a-z0-9']."""
    return _WORD_RE.findall(_text(s).lower())


def _sent_count(s) -> int:
    """run.js's sentCount: split on runs of [.!?], drop segments that are empty after trim, floor at 1 (an
    empty/unpunctuated reply is still "one sentence" for a words-per-sentence average)."""
    parts = [p for p in _SENT_SPLIT_RE.split(_text(s)) if p.strip()]
    return max(1, len(parts))


def _js_round(x: float) -> int:
    """Math.round semantics for x >= 0 (every value rounded in this module is a count, a ratio of counts,
    or a 0-100 percentage -- always non-negative): a trailing .5 rounds UP. Python's builtin round() is
    banker's rounding (round(2.5) == 2, round(62.5) == 62) and would silently disagree with the client's
    JS on exactly these tie cases -- the whole reason this helper exists."""
    return math.floor(x + 0.5)


def receipt_metrics(orig, repl) -> dict:
    """The three honest numbers run.js's delta strip shows, computed identically here: word count,
    words/sentence (one decimal), and % of wording changed (100 * the Jaccard distance of word TYPES --
    unique words, not token order, not a diff). No model judges anything; it's pure counting."""
    ow, rw = _words_of(orig), _words_of(repl)
    oset, rset = set(ow), set(rw)
    inter = len(oset & rset)
    uni = len(oset | rset)
    return {
        "words": [len(ow), len(rw)],
        "wps": [_js_round(len(ow) / _sent_count(orig) * 10) / 10,
                _js_round(len(rw) / _sent_count(repl) * 10) / 10],
        "changed": _js_round((1 - inter / uni) * 100) if uni else 0,
    }


# ==================================================================================== influence -> changes
_NOTE_BASELINE = (
    "the run's stored sampled reply is NOT the baseline for this receipt -- greedy-with-the-influence is. "
    "The sampled reply is context only; it is never a term in this subtraction "
    "(EXPLAIN_THIS_ANSWER_SPEC.md M2: diffing sampled-vs-greedy would mix two changes at once)."
)

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


def _key(influence: dict) -> str:
    """A short, stable label for one influence spec -- used to name redundancy pairs and skip entries."""
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
    """One influence spec -> the replay.py `changes` dict for its ablated ("without") arm. None when the
    spec doesn't resolve to any known ablation (a bad/empty request) -- never raises."""
    if not isinstance(influence, dict):
        return None
    cid = influence.get("card_id")
    if cid:
        return {"disabled_memory_ids": [str(cid)]}
    dial = influence.get("dial")
    if dial:
        return {"behavior_overrides": {str(dial): 0.0}}    # zero JUST this one axis; others untouched
    if influence.get("memory_off"):
        return {"memory_off": True}
    if influence.get("behavior_off"):
        return {"behavior_off": True}
    return None


def _merge_ablation_changes(influences: list) -> dict:
    """The joint replay.py `changes` dict that ablates every one of `influences` AT ONCE -- the redundancy
    guard's "drop both together" arm."""
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
    """The cost-asymmetry, surfaced honestly rather than silently (spec bullet): a front-of-context memory
    ablation changes the shared prefix (no KV reuse -- the whole context re-prefills); a dial acts at
    decode time (the prompt KV stays reusable)."""
    influence = influence or {}
    if influence.get("card_id") or influence.get("memory_off"):
        return ("cost: a front-of-context memory ablation changes the shared prefix, so the ablated arm "
                "re-prefills the whole context (no KV reuse) -- the expensive case.")
    return ("cost: a dial ablation acts at decode time, so the prompt KV stays reusable -- cheap relative "
            "to a memory ablation.")


def _unapplied_note(ablated_child: dict, changes: dict) -> str | None:
    """Relays replay.py's OWN honesty note when the requested ablation could not actually take effect
    (today: a disabled_memory_ids attempt while memory mode is "internalized" -- see replay.py's
    _apply_changes). Never invents new detection; only reads what replay.py already recorded on the child
    run, so a receipt can never silently claim `causal_verified: true` for a no-op."""
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


# ===================================================================================================== API
def receipt(run: dict, influence: dict, sub) -> dict | None:
    """One rigorous causal receipt for `influence` against `run`, generated on the live substrate `sub`.
    Both arms greedy, both from `run`'s own stored messages; the stored sampled reply never enters the
    diff. Returns None on any failure -- a bad influence spec, or a substrate/chat that couldn't generate
    (mirrors replay.replay's own "never raises into the caller" contract)."""
    try:
        if not run or not isinstance(run, dict):
            return None
        changes = _ablation_changes(influence)
        if not changes:
            return None
        baseline_child = replay.replay(run, {"greedy": True}, sub)          # greedy-WITH the influence
        if baseline_child is None:
            return None
        ablated_child = replay.replay(run, {**changes, "greedy": True}, sub)  # greedy-WITHOUT it
        if ablated_child is None:
            return None
        return _build_receipt(influence, baseline_child, ablated_child, changes)
    except Exception:
        return None


def _fired_influences(manifest: dict):
    """The M1 manifest's cards + dials, each turned into a receipts.py influence spec for leave-one-out. A
    card with no resolvable id can't be ablated (disabled_memory_ids needs a real id) -- reported as an
    honest `skipped` entry rather than silently dropped."""
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


def prove_all(run: dict, sub, *, manifest: dict | None = None) -> dict:
    """Leave-one-out over every influence that FIRED on `run` (per the M1 manifest), plus the pairwise
    redundancy guard. Never raises; degrades to an (honestly) empty result on any failure. `manifest` lets
    a caller that already has the M1 explanation object (e.g. just fetched /explain) pass it in instead of
    recomputing it; None (the default) computes it here via explain.explain(run)."""
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
            from clozn import explain
            manifest = explain.explain(run)
        influences, skipped = _fired_influences(manifest)
        out["skipped"].extend(skipped)
        if not influences:
            return out

        baseline_child = replay.replay(run, {"greedy": True}, sub)
        if baseline_child is None:
            out["skipped"].append({"influence": None,
                                   "reason": "could not generate the greedy baseline (with-influence arm)"})
            return out
        baseline_reply = baseline_child.get("response") or ""

        per_key: dict = {}    # key -> (influence, has_effect), for influences that DID generate
        for inf in influences:
            changes = _ablation_changes(inf)
            ablated_child = replay.replay(run, {**changes, "greedy": True}, sub) if changes else None
            if ablated_child is None:
                out["skipped"].append({"influence": inf, "reason": "ablation could not be generated"})
                continue
            rec = _build_receipt(inf, baseline_child, ablated_child, changes)
            out["receipts"].append(rec)
            per_key[_key(inf)] = (inf, rec["has_effect"])

        # REDUNDANCY GUARD -- pairwise only (a documented approximation; see _APPROX_NOTE), over the
        # influences whose OWN leave-one-out showed ~no effect (exact reply-string equality: greedy
        # decoding is deterministic, so "identical" is the strongest honest signal available).
        no_effect = [k for k, (_, eff) in per_key.items() if not eff]
        for i in range(len(no_effect)):
            for j in range(i + 1, len(no_effect)):
                ka, kb = no_effect[i], no_effect[j]
                joint_changes = _merge_ablation_changes([per_key[ka][0], per_key[kb][0]])
                if not joint_changes:
                    continue
                joint_child = replay.replay(run, {**joint_changes, "greedy": True}, sub)
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
