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
import random
import re

from clozn import rederive
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
def _receipt_regen(run: dict, influence: dict, sub) -> dict | None:
    """The EXISTING (pre-S3) rigorous causal receipt for `influence` against `run`, generated on the
    live substrate `sub`. Both arms greedy, both from `run`'s own stored messages; the stored sampled
    reply never enters the diff. Returns None on any failure -- a bad influence spec, or a substrate/chat
    that couldn't generate (mirrors replay.replay's own "never raises into the caller" contract).
    UNCHANGED by S3 (REPRODUCE_AND_PROVE_PLAN.md): `receipt()` below dispatches here for mode="regen"
    (the default) and its output must stay byte-identical to before forced mode existed."""
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


def _prove_all_regen(run: dict, sub, *, manifest: dict | None = None) -> dict:
    """The EXISTING (pre-S3) leave-one-out over every influence that FIRED on `run` (per the M1
    manifest), plus the pairwise redundancy guard. Never raises; degrades to an (honestly) empty result
    on any failure. `manifest` lets a caller that already has the M1 explanation object (e.g. just
    fetched /explain) pass it in instead of recomputing it; None (the default) computes it here via
    explain.explain(run). UNCHANGED by S3: `prove_all()` below dispatches here for mode="regen" (the
    default) and its output must stay byte-identical to before forced mode existed."""
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


# =========================================================================== S3: forced-mode receipts
# (notes/REPRODUCE_AND_PROVE_PLAN.md) -- a GRADED, per-token DEPENDENCE measurement on the run's OWN
# stored answer, via teacher-forced /score (rederive.py's score_arm), alongside the regen receipt above
# (a BINARY text-diff on a FRESH greedy regeneration). Forced mode never regenerates anything -- it
# scores the SAME continuation tokens under WITH vs WITHOUT the influence, so it answers a DIFFERENT
# question than regen ("would the greedy answer have changed?"): "how much did THIS answer rely on it?"
# See "The sub-threshold receipt" in the plan for why this matters -- an influence can shift the model's
# preferences substantially (measurable here) without ever flipping a greedy argmax (invisible to regen).

_FORCED_MEAN_THRESHOLD = 0.05     # nats/token -- has_effect trigger #1 (tunable; PRESENTATION only --
_FORCED_SUM_THRESHOLD = 2.0       # nats -- has_effect trigger #2                  raw numbers always ship)
_NULL_FLOOR_RATIO_MIN = 5.0       # "exceeds the null floor by ~an order of magnitude" (the plan's own
                                  # wording for a silent-influence claim) -- PRESENTATION only; the raw
                                  # ratio always ships in the payload.

_FORCED_CAVEAT = (
    "a nonzero delta means the influence changed the model's confidence in the answer it gave -- it "
    "does NOT mean the answer would have been different without it. Regen mode answers 'would the "
    "greedy answer have changed?' (counterfactual text); forced mode answers 'how much did THIS answer "
    "rely on it?' (dependence). Both are interventions; they measure different outcomes -- read them "
    "side by side, never interchangeably. (REPRODUCE_AND_PROVE_PLAN.md: 'the sub-threshold receipt'.)"
)

_FORCED_NOTE = (
    "dial vectors (and, for a card ablation, the recompiled memory block) are computed from TODAY's "
    "steering library / card store at the run's recorded strengths and card texts -- the same "
    "limitation the regen receipt already carries. The with/without prompts differ in length by "
    "whatever was ablated; deltas align per CONTINUATION token position, which is what matters -- not "
    "per prompt token."
)

# The null-floor CARD control (REPRODUCE_AND_PROVE_PLAN.md's null-floor recipe): "no block" is NOT a
# valid control (length alone moves logits) -- swap the real card for filler of the SAME length at the
# SAME block position instead. MATCHED REGISTER matters as much as matched length: the block's other
# cards are all short first-person-about-the-user PREFERENCE/HABIT statements (compile_prompt_block's
# "here is what you know about them" framing), so the filler is written the same way -- a personal
# habit/preference, not an encyclopedia fact -- to isolate "this specific fact is irrelevant" from "this
# sentence doesn't belong in this block at all" (a register mismatch would itself be a confound, not a
# clean null floor). Picked to have no plausible bearing on typical everyday questions.
_FILLER_TEXT = (
    "The user prefers to schedule meetings in the early morning rather than the afternoon. The user "
    "always tips exactly twenty percent at restaurants without needing to calculate it by hand. The "
    "user set their phone's default browser to a different app than the one it shipped with. The user "
    "keeps their email inbox at zero and archives messages the same day they arrive. "
)


def _matched_length_filler(n_chars: int) -> str:
    """An IRRELEVANT filler string of (approximately) the length asked for. Cycles `_FILLER_TEXT` to
    whatever length is requested; deterministic (no randomness), so repeated calls on the same run agree
    exactly (acceptance #3's determinism)."""
    n = max(1, int(n_chars))
    reps = n // len(_FILLER_TEXT) + 1
    return (_FILLER_TEXT * reps)[:n]


def _vector_norm(vec) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in (vec or [])))


def _random_vector_of_norm(dim: int, norm: float, seed) -> list:
    """A DETERMINISTIC pseudo-random vector (seeded on `seed` -- the same run+influence always gets the
    SAME vector, so repeated calls agree exactly) of the given `norm`/dimensionality -- the S3 null-floor
    CONTROL for a dial ablation: a direction with no relation to any real dial's semantics, at the SAME
    magnitude as what was actually applied ("a random vector of equal norm at the same layer")."""
    rng = random.Random(seed)
    dim = max(1, int(dim))
    raw = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    raw_norm = _vector_norm(raw)
    if raw_norm <= 0.0:
        raw = [1.0] + [0.0] * (dim - 1)
        raw_norm = 1.0
    scale = float(norm) / raw_norm
    return [x * scale for x in raw]


def _forced_deltas(with_tokens, without_tokens):
    """Per-position lp_with - lp_without; None if the two token lists don't align 1:1 (a scoring
    inconsistency somewhere) -- forced deltas are only meaningful position-for-position on the SAME
    continuation tokens, and this module never guesses past a misalignment."""
    if not with_tokens or not without_tokens or len(with_tokens) != len(without_tokens):
        return None
    out = []
    for w, wo in zip(with_tokens, without_tokens):
        if not isinstance(w, dict) or not isinstance(wo, dict):
            return None
        lw, lwo = w.get("logprob"), wo.get("logprob")
        if not isinstance(lw, (int, float)) or not isinstance(lwo, (int, float)):
            return None
        out.append(float(lw) - float(lwo))
    return out


def _delta_summary(deltas: list) -> dict:
    n = len(deltas) or 1
    return {
        "sum_nats": round(sum(deltas), 6),
        "mean_nats_per_token": round(sum(abs(d) for d in deltas) / n, 6),
    }


def _top_dependent(pieces: list, deltas: list, k: int = 5) -> list:
    """The top-k answer tokens by |delta|, with their pieces -- the receipt's "leaned on it mostly at:
    <these tokens>" evidence."""
    order = sorted(range(len(deltas)), key=lambda i: -abs(deltas[i]))[:k]
    return [{"index": i, "piece": pieces[i] if i < len(pieces) else "", "delta": round(deltas[i], 6)}
            for i in order]


def _forced_ablation(run: dict, influence: dict, sub, conditions: dict):
    """One influence spec -> the WITHOUT arm (real ablation) and its null-floor CONTROL (irrelevant,
    matched-magnitude perturbation), both as kwargs for rederive.score_arm (block=/steer_strengths=/
    steer_vec=). Returns None for a bad/empty spec, else a dict:
        {"without": {...} | None, "control": {...} | None, "note": str | None}
    `without` is None when the ablation genuinely cannot be resolved against this run (nothing recorded
    to ablate) -- `note` then carries the honest reason (mirrors replay.py's _apply_changes "not
    applied" machinery: a card ablation attempted on a run whose OWN recorded memory.mode wasn't
    "prompt" reads exactly like "this card was never applied", since only prompt-mode runs record
    applied_ids at all). `control` is None when there is nothing to null-test against (e.g. memory_off
    on a run that already had no active block, or a dial that was never active)."""
    influence = influence or {}
    with_block = conditions.get("raw_block")
    with_strengths = dict(conditions.get("steer_strengths") or {})

    cid = influence.get("card_id")
    if cid:
        mem = run.get("memory") or {}
        ids = mem.get("applied_ids") or []
        texts = mem.get("cards_applied") or []
        pairs = list(zip(ids, texts))
        match = next((t for i, t in pairs if str(i) == str(cid)), None)
        if match is None:
            return {"without": None, "control": None,
                    "note": "this card was not recorded as applied on this run (internalized memory "
                            "mode fuses cards into a trained prefix, or the card simply wasn't active "
                            "this turn) -- nothing to ablate"}
        from clozn import memory_mode
        without_texts = [t for i, t in pairs if str(i) != str(cid)]
        without_block = memory_mode.compile_prompt_block(without_texts)
        control_texts = [t if str(i) != str(cid) else _matched_length_filler(len(match)) for i, t in pairs]
        control_block = memory_mode.compile_prompt_block(control_texts)
        return {"without": {"block": without_block, "steer_strengths": with_strengths},
                "control": {"block": control_block, "steer_strengths": with_strengths}, "note": None}

    if influence.get("memory_off"):
        control = ({"block": _matched_length_filler(len(with_block)), "steer_strengths": with_strengths}
                  if with_block else None)
        return {"without": {"block": None, "steer_strengths": with_strengths}, "control": control,
                "note": None if with_block else "no active memory block on this run -- nothing to ablate"}

    dial = influence.get("dial")
    if dial:
        without_strengths = dict(with_strengths)
        without_strengths.pop(dial, None)
        control = None
        steer = getattr(sub, "steer", None)
        if steer is not None and hasattr(steer, "steer_vector") and with_strengths.get(dial):
            try:
                isolated = steer.steer_vector({dial: with_strengths[dial]})
            except Exception:
                isolated = None
            norm = _vector_norm(isolated) if isolated else 0.0
            if norm > 0:
                seed = f"{run.get('id')}:dial:{dial}"
                rand_vec = _random_vector_of_norm(len(isolated), norm, seed)
                control = {"block": with_block, "steer_strengths": without_strengths, "steer_vec": rand_vec}
        return {"without": {"block": with_block, "steer_strengths": without_strengths}, "control": control,
                "note": None if with_strengths.get(dial) else
                       f"dial '{dial}' was not active on this run -- nothing to ablate"}

    if influence.get("behavior_off"):
        control = None
        steer = getattr(sub, "steer", None)
        if (steer is not None and hasattr(steer, "steer_vector") and with_strengths
                and any(with_strengths.values())):
            try:
                full_vec = steer.steer_vector(with_strengths)
            except Exception:
                full_vec = None
            norm = _vector_norm(full_vec) if full_vec else 0.0
            if norm > 0:
                seed = f"{run.get('id')}:behavior_off"
                rand_vec = _random_vector_of_norm(len(full_vec), norm, seed)
                control = {"block": with_block, "steer_strengths": {}, "steer_vec": rand_vec}
        return {"without": {"block": with_block, "steer_strengths": {}}, "control": control,
                "note": None if with_strengths else "no active dial on this run -- nothing to ablate"}

    return None


def forced_receipt(run: dict, influence: dict, sub) -> dict | None:
    """One S3 forced-mode causal receipt: score `run`'s OWN stored answer tokens WITH vs WITHOUT
    `influence` (teacher-forced, via rederive.score_arm -- NEVER generation), plus a matched-magnitude
    NULL-FLOOR control (an irrelevant, same-size perturbation on the SAME channel) so the caller can tell
    real dependence from float noise. Returns None only for a bad run/influence spec; every other
    failure (substrate can't score, ablation doesn't apply, arms misalign) degrades to an honest dict
    with `causal_verified: False` and a `note` -- never raises, mirrors receipt()'s own contract."""
    try:
        if not run or not isinstance(run, dict):
            return None
        if not isinstance(influence, dict) or not influence:
            return None
        conditions = rederive.with_arm_conditions(run)
        ablation = _forced_ablation(run, influence, sub, conditions)
        if ablation is None:
            return None
        # a resolvable-from-the-record check, so an ablation that can't apply at all (nothing was ever
        # applied to remove) bails BEFORE spending a scoring call -- matches _unapplied_note's spirit:
        # honest, and cheap.
        if ablation.get("without") is None:
            return {"influence": influence, "mode": "forced", "causal_verified": False,
                    "note": ablation.get("note"), "caveat": _FORCED_CAVEAT}

        with_tokens, with_ok = rederive.score_arm(
            sub, conditions, messages=conditions["raw_messages"], block=conditions["raw_block"],
            steer_strengths=conditions["steer_strengths"])
        if not with_ok:
            return {"influence": influence, "mode": "forced", "causal_verified": False,
                    "note": "forced scoring needs the engine substrate (score_tokens is not available "
                            "here)", "caveat": _FORCED_CAVEAT}

        without_tokens, without_ok = rederive.score_arm(
            sub, conditions, messages=conditions["raw_messages"], **ablation["without"])
        if not without_ok:
            return {"influence": influence, "mode": "forced", "causal_verified": False,
                    "note": "the ablated arm could not be scored", "caveat": _FORCED_CAVEAT}

        deltas = _forced_deltas(with_tokens, without_tokens)
        if deltas is None:
            return {"influence": influence, "mode": "forced", "causal_verified": False,
                    "note": "with/without arms did not align token-for-token (a scoring inconsistency)",
                    "caveat": _FORCED_CAVEAT}

        pieces = [str(t.get("piece", "")) for t in with_tokens]
        summary = _delta_summary(deltas)
        has_effect = (summary["mean_nats_per_token"] >= _FORCED_MEAN_THRESHOLD
                     or abs(summary["sum_nats"]) >= _FORCED_SUM_THRESHOLD)

        out = {
            "influence": influence,
            "mode": "forced",
            "retokenized": conditions["retokenized"],
            "causal_verified": True,
            "answer_tokens": pieces,
            "deltas": [round(d, 6) for d in deltas],
            "sum_nats": summary["sum_nats"],
            "mean_nats_per_token": summary["mean_nats_per_token"],
            "top_dependent": _top_dependent(pieces, deltas),
            "has_effect": has_effect,
            "threshold": {"mean_abs_nats_per_token": _FORCED_MEAN_THRESHOLD,
                         "abs_sum_nats": _FORCED_SUM_THRESHOLD},
            "note": _FORCED_NOTE,
            "caveat": _FORCED_CAVEAT,
        }
        if ablation.get("note"):
            out["ablation_note"] = ablation["note"]

        control = ablation.get("control")
        if control is not None:
            control_tokens, control_ok = rederive.score_arm(
                sub, conditions, messages=conditions["raw_messages"], **control)
            control_deltas = _forced_deltas(with_tokens, control_tokens) if control_ok else None
            if control_deltas is not None:
                c_summary = _delta_summary(control_deltas)
                floor_mean = c_summary["mean_nats_per_token"]
                ratio = (summary["mean_nats_per_token"] / floor_mean) if floor_mean > 0 else None
                out["null_floor"] = {
                    "kind": ("card_filler" if influence.get("card_id") else
                            "block_filler" if influence.get("memory_off") else
                            "behavior_off_random_vector" if influence.get("behavior_off") else
                            "dial_random_vector"),
                    "deltas": [round(d, 6) for d in control_deltas],
                    "sum_nats": c_summary["sum_nats"],
                    "mean_nats_per_token": floor_mean,
                    "ratio_real_over_floor": round(ratio, 3) if ratio is not None else None,
                    "exceeds_floor_by_order_of_magnitude": bool(ratio is not None
                                                                and ratio >= _NULL_FLOOR_RATIO_MIN),
                }
        return out
    except Exception:
        return None


def _forced_prove_all(run: dict, sub, manifest: dict | None) -> dict:
    """mode="forced"'s /receipts payload: forced_receipt() for every influence the M1 manifest says
    FIRED on this run -- no generation anywhere (unlike _prove_all_regen, this never needs SUB.chat).
    No redundancy-guard equivalent (S3 scope is per-influence dependence, not a pairwise regen search)."""
    out = {"run_id": run.get("id") if isinstance(run, dict) else None, "mode": "forced",
          "forced_receipts": [], "skipped": []}
    try:
        if not run or not isinstance(run, dict):
            return out
        if manifest is None:
            from clozn import explain
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


# ================================================================================= public dispatchers
# `mode` (REPRODUCE_AND_PROVE_PLAN.md S3): "regen" (DEFAULT) | "forced" | "both". mode="regen" (omitted
# or explicit) dispatches straight to the UNCHANGED pre-S3 function -- byte-identical output, no added
# keys -- so every pre-existing caller (and test) keeps working exactly as it always did.

def receipt(run: dict, influence: dict, sub, *, mode: str = "regen") -> dict | None:
    """One causal receipt for `influence` against `run`, in the given `mode`:
      "regen"  (DEFAULT, unchanged) -- the existing greedy-both-arms text-diff (_receipt_regen); returns
               its dict EXACTLY as before this feature existed (byte-identical regression contract).
      "forced" -- the S3 teacher-forced dependence receipt (forced_receipt) alone.
      "both"   -- the regen dict, PLUS a "forced" sub-object and (when both succeeded) a top-level
               "silent_influence" flag: regen showed no text change AND forced cleared the null floor by
               roughly an order of magnitude (REPRODUCE_AND_PROVE_PLAN.md's "sub-threshold receipt").
    An unrecognized mode string falls back to "regen" (never raise on a typo)."""
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
    """Leave-one-out receipts for every influence that FIRED on `run`, in the given `mode` (see
    receipt()'s docstring for the three modes). mode="regen" (DEFAULT) returns _prove_all_regen's dict
    EXACTLY as before this feature existed (byte-identical regression contract)."""
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
