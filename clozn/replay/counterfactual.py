"""counterfactual.py -- Milestone 3 (EXPLAIN_THIS_ANSWER_SPEC.md): interactive counterfactual dials.

WHAT THIS IS: the panel's "what if I turn this dial to X" slider. Unlike M2's receipts.py (an on-demand
PROOF that one influence already firing on this run mattered -- ablate it, diff the two arms), M3 is a
live WHAT-IF: the caller names any behavior dial(s) + any value(s) (not necessarily what was active on the
run at all) and gets back the reply that dial value WOULD have produced, honestly diffed against the run's
actual dials -- so a slider drag in the panel is a measured re-generation, never a narrated guess.

THE SAME SEAM M2 FIXES, reused here: the run's stored reply was SAMPLED under whatever dials were live at
chat time. Diffing a new counterfactual reply against that stored sampled reply would mix two changes at
once (dial value change AND sampled->greedy decoding) -- exactly the bug M2's receipts.py exists to fix for
ablation. So the baseline here is ALSO regenerated greedy, with the run's OWN dials untouched
(`replay.replay(run, {"greedy": True}, sub)`) -- never the stored sampled response. Both arms greedy, both
from the SAME stored messages, diffed via `receipts.receipt_metrics` -- the identical metric math run.js's
delta strip and receipts.py already use, so a client never sees three different numbers for the same kind
of diff.

LAW #6 (a 7B-calibrated dial derails a 1.5B; every receipt needs a coherence/sanity axis): a slider that
lets a user push a dial past where THIS model can absorb it must not report "huge delta = huge effect" --
a huge delta can just as easily be the model degenerating into repetition or a script switch
(the measured "Russian mid-sentence" case; the measured "degeneration can
GAME a lexical steering eval"). So every counterfactual carries a `coherence` field: a crude text-only
degeneration/repetition proxy computed on the counterfactual arm (the one that might be over-dosed). It
duplicates -- does not import -- memory_disorders.is_degenerate's exact checks: this module's build spec
restricts it to importing receipts/replay only, and memory_disorders.py is explicitly out of scope to
touch or depend on here. `causal_verified` and `coherence` are deliberately ORTHOGONAL: the first says
whether the override mechanically took hold; the second says whether the resulting reply still makes
sense. A dial can be causally verified AND derailed at the same time -- that combination is exactly what
the coherence axis exists to catch.

`dose_sweep()` operationalizes "how warm is warm on THIS model" directly: run `counterfactual()` at each of
several values for ONE dial and return the response curve (delta + coherence per value), so the per-model
dose-response shape is a receipt, not a guess -- with a derailment flag that fires the moment coherence
craters, wherever that happens to sit for this particular model. Each point is an independent
`counterfactual()` call (its own fresh greedy baseline + its own fresh greedy arm) -- O(len(values)) greedy
re-generations, never batched or cached across points, so one bad/degenerate point can never contaminate
another's baseline.

Stdlib + receipts + replay only, exactly like receipts.py: no torch import, no model, no GPU. Every
generation goes through `replay.replay()`, which already owns the snapshot / apply-changes / restore-in-a-
finally contract (never leaves the live studio mutated) and the greedy flag (deterministic decoding, so a
diff is attributable to the dial change, not to sampling dice). This module is fully unit-testable against
a fake substrate with a canned, deterministic `.chat()`.
"""
from __future__ import annotations

import re
import unicodedata

from clozn import receipts
from .replay import replay as replay_run

# ============================================================================================ notes / cost
_NOTE_BASELINE = (
    "the run's stored sampled reply is NOT the baseline for this counterfactual -- greedy-with-the-run's-"
    "ACTUAL-dials is. The sampled reply is never a term in this diff (EXPLAIN_THIS_ANSWER_SPEC.md M3: only "
    "baseline-greedy vs overridden-greedy are compared, exactly the seam M2's receipts.py fixes for "
    "ablation)."
)

_COST_NOTE = (
    "cost: a dial override is a decode-time hook, so the prompt KV stays reusable -- cheap. Contrast a "
    "front-of-context memory-card ablation, which changes the shared prefix and must re-prefill the whole "
    "context (the expensive case; see receipts.py's cost_note)."
)

_DOSE_NOTE = (
    "the per-model dose receipt (EXPLAIN_THIS_ANSWER_SPEC.md law #6): this curve is THIS model's measured "
    "response to ONE dial at these exact values, not a portable calibration -- the same nominal value that "
    "is safe on the model a dial was calibrated on can derail a smaller or differently-tuned one. Read the "
    "coherence column, not just delta: a big delta paired with degenerate coherence is derailment, not a "
    "bigger effect."
)


# ============================================================================================== coherence
# A crude, eyeball-informed, NOT-learned degeneration/repetition proxy -- mirrors memory_disorders.py's
# is_degenerate() checks exactly (empty output / immediate 3-gram word repetition / character runaway /
# script switch -- the failure modes the steering experiments actually
# eyeballed at small-model dial extremes). Duplicated rather than imported: this module's build spec
# restricts it to importing receipts/replay only (memory_disorders.py is explicitly out of scope here).
_FOREIGN_MIN = 3


def _foreign_letters(t: str) -> int:
    """Count non-ASCII LETTER characters. A real language/script switch (the failure this flags -- steering
    derailing into Cyrillic/CJK word-salad) is many non-ASCII *letters*; emoji, curly quotes, and em-dashes
    are non-ASCII SYMBOLS/PUNCTUATION and must NOT count -- Gemma-2 is emoji-heavy and perfectly coherent, so
    the old `[^\\x00-\\x7F]` catch-all false-flagged it 100% degenerate. A stray accent ('cafe') is one
    letter, below the threshold, so ordinary loanwords pass."""
    return sum(1 for ch in t if ord(ch) > 127 and unicodedata.category(ch).startswith("L"))


def _coherence(text: str) -> dict:
    """{"degenerate": bool, "reason": str} for `text` -- the MANDATORY coherence axis (law #6) so an
    over-dosed dial that derails into gibberish is FLAGGED, never silently read as "big delta = big
    effect". Pure text counting, no model call -- exactly as honest, and exactly as crude, as the one
    other place this codebase already does this (memory_disorders.is_degenerate)."""
    t = (text or "").strip()
    if not t:
        return {"degenerate": True, "reason": "empty"}
    words = t.split()
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return {"degenerate": True, "reason": "repeat-3gram"}
    if re.search(r"(.)\1{4,}", t):
        return {"degenerate": True, "reason": "char-runaway"}
    if _foreign_letters(t) >= _FOREIGN_MIN:
        return {"degenerate": True, "reason": "script-switch"}
    return {"degenerate": False, "reason": ""}


# ======================================================================================= unapplied guard
def _unapplied_overrides_note(cf_child: dict, behavior_overrides: dict) -> str | None:
    """Whether replay.py's OWN recorded state on the child run confirms the requested overrides actually
    took hold. Reads cf_child["behavior"]["active_dials"] -- the dict replay.replay() itself fills in
    (via its _effective_dials helper) AFTER applying `changes` -- so this never invents a new probe into
    the model; it only relays what replay.py already recorded on the child run, exactly
    receipts._unapplied_note's contract for a per-card ablation replay.py couldn't apply (this module may
    only import receipts/replay, so the check is re-derived here rather than shared as a function).

    A requested value of exactly 0.0 is never flagged: a dial that reads back as absent/zero after being
    deliberately zeroed is indistinguishable from one that was never touched under the SAME convention
    steer.active() already uses (it filters falsy entries) -- receipts.py's own dial-ablation-to-zero
    relies on this identical convention, so this is an inherited honest gap, not a new one."""
    active = ((cf_child or {}).get("behavior") or {}).get("active_dials")
    active = active if isinstance(active, dict) else {}
    missed = []
    for name, val in (behavior_overrides or {}).items():
        try:
            want = float(val)
        except (TypeError, ValueError):
            missed.append(str(name))
            continue
        if want == 0.0:
            continue
        got = float(active.get(str(name), 0.0) or 0.0)
        if abs(got - want) > 1e-6:
            missed.append(str(name))
    if not missed:
        return None
    return ("not applied: override(s) for {} did not show up in the replayed run's own recorded dial "
            "state -- this substrate may expose no steer mechanism, or doesn't recognize that axis; "
            "treated as NOT causally verified rather than reporting an unmoved reply as proof of "
            "'no effect'").format(", ".join(sorted(missed)))


# ===================================================================================================== API
def counterfactual(run: dict, behavior_overrides: dict, sub) -> dict | None:
    """The honest what-if for one dial-value proposal against `run`, generated on the live substrate
    `sub`. BOTH arms greedy, both from `run`'s own stored messages:
      * baseline        = replay.replay(run, {"greedy": True}, sub)                     -- the run's
                           ACTUAL (currently live) dials, greedy.
      * counterfactual   = replay.replay(run, {"behavior_overrides": behavior_overrides,
                                               "greedy": True}, sub)                     -- same messages,
                           dials overridden.
    Diffs ONLY those two via receipts.receipt_metrics -- the run's stored sampled reply is never a term in
    the subtraction (see the returned `note`). Returns None on any failure -- a bad/empty overrides dict,
    an invalid run, or a substrate/chat that couldn't generate (mirrors receipts.receipt's own "never
    raises into the caller" contract).

    Returns {overrides_applied, baseline_reply, counterfactual_reply, delta, has_effect, causal_verified,
    coherence, note, cost_note}, plus `override_note` when an override could not be verified as applied.
    `causal_verified` is True whenever the override measurably took hold (per replay.py's own recorded
    dial state) -- independent of whether the result is still coherent; check `coherence` separately for
    that (law #6): a derailed-but-genuinely-applied override is causal_verified: true, coherence:
    degenerate, NOT "no effect"."""
    try:
        if not run or not isinstance(run, dict):
            return None
        if not isinstance(behavior_overrides, dict) or not behavior_overrides:
            return None
        baseline_child = replay_run(run, {"greedy": True}, sub)   # the run's ACTUAL dials, greedy
        if baseline_child is None:
            return None
        changes = {"behavior_overrides": dict(behavior_overrides), "greedy": True}
        cf_child = replay_run(run, changes, sub)                  # same messages, dials overridden
        if cf_child is None:
            return None
        baseline_reply = baseline_child.get("response") or ""
        cf_reply = cf_child.get("response") or ""
        unapplied = _unapplied_overrides_note(cf_child, behavior_overrides)
        out = {
            "overrides_applied": dict(behavior_overrides),
            "baseline_reply": baseline_reply,
            "counterfactual_reply": cf_reply,
            "delta": receipts.receipt_metrics(baseline_reply, cf_reply),
            "has_effect": baseline_reply != cf_reply,
            "causal_verified": unapplied is None,
            "coherence": _coherence(cf_reply),
            "note": _NOTE_BASELINE,
            "cost_note": _COST_NOTE,
        }
        if unapplied:
            out["override_note"] = unapplied
        return out
    except Exception:
        return None


def dose_sweep(run: dict, dial: str, values, sub) -> dict:
    """The per-model dose-response curve for ONE dial: counterfactual() at each of `values`, independently
    (O(len(values)) greedy re-generations -- 2 per value, never batched or shared across points, so a
    derailed point can never contaminate another point's baseline). Returns:

        {run_id, dial,
         curve: [{value, baseline_reply, counterfactual_reply, delta, has_effect, causal_verified,
                  coherence[, override_note]} | {value, error}, ...],
         derailment: bool,             # True iff coherence went degenerate at ANY sampled value
         derailed_at: [values ...],    # exactly which values craters, so the curve is inspectable
         note}

    Never raises -- a missing dial name or an unmeasurable point degrades to an honest per-point `error`
    rather than fabricating a curve entry."""
    out = {"run_id": run.get("id") if isinstance(run, dict) else None, "dial": dial,
           "curve": [], "derailment": False, "derailed_at": [], "note": _DOSE_NOTE}
    try:
        if not dial or not isinstance(dial, str):
            out["note"] = "no receipt for that: need a dial name"
            return out
        for v in (values or []):
            rec = counterfactual(run, {dial: v}, sub)
            if rec is None:
                out["curve"].append({"value": v, "error": "no receipt for this dose (the replay could "
                                                           "not be generated)"})
                continue
            point = {"value": v, "baseline_reply": rec["baseline_reply"],
                     "counterfactual_reply": rec["counterfactual_reply"], "delta": rec["delta"],
                     "has_effect": rec["has_effect"], "causal_verified": rec["causal_verified"],
                     "coherence": rec["coherence"]}
            if "override_note" in rec:
                point["override_note"] = rec["override_note"]
            out["curve"].append(point)
        derailed_at = [pt["value"] for pt in out["curve"] if pt.get("coherence", {}).get("degenerate")]
        out["derailed_at"] = derailed_at
        out["derailment"] = bool(derailed_at)
        return out
    except Exception:
        return out
