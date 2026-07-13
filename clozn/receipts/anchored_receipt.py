"""anchored_receipt.py -- the causal receipt for ANCHORED MEMORY (clozn/memory/anchored.py): the
"verify with a causal receipt" payoff for the fit -> whatlearned -> recall -> delete_term demo.
`anchored.compile_steer()` composes a card's (or every active card's) k-sparse bag of named
word-directions into ONE steer payload; nothing in that flow PROVES the injected bag actually changed
the model's answer -- this module is that proof.

Template: clozn/receipts/swap_receipt.py (read its docstring for the full rationale -- the structure
below is mirrored from it almost verbatim, substituting a caller-supplied `to_concept` for the
receipt's OWN, already-fitted "named cause"):

  * BASELINE: an unsteered generation from the run's own rendered prompt.
  * ANCHORED: `compile_steer()`'s composed unit `vector` injected at its own `coef`/`layer` (L21 by
    default -- anchored.py's validated X7 operating point).
  * NULL: a random, EQUAL-MAGNITUDE write at the SAME layer/coef (`clozn.receipts.forced`'s own
    `_random_vector_of_norm`, reused unchanged, exactly as swap_receipt.py reuses it) -- must not
    coherently move the answer any more than noise would. `has_effect`/`targeted_shift` are only True
    when ANCHORED beats BOTH baseline and null.
  * TARGET TERM: the single most heavily-weighted word across every composed bag's `alpha_top3`
    (`compile_steer()`'s own per-card summary) -- this receipt's ONE legible "named cause", picked by
    the bag's own fit, never by a caller. It must resolve to a single vocab token (the SAME
    `/score`-based round trip `concept_dir.ConceptSteer.resolve_token_id` uses) for the two measures
    below; a multi-token word blocks here with `blocked: "token_resolution"`, BEFORE any generation is
    spent -- mirroring swap_receipt's WRITE-then-BASELINE ordering exactly (there, `to_concept` itself
    is the write and must resolve first; here, the write is already a stored bag vector, but the
    quantitative measures below still need ONE nameable token, so the same ordering applies).

  Two measures on that target term, like swap_receipt.py, NEVER conflated:
    A) `lexicon_hits` -- literal mentions of the target term, baseline vs anchored vs null (see
       `_LEXICON_CAVEAT` below -- the same limitation as swap_receipt's own, retargeted from a
       caller-supplied `to_concept` to this receipt's fitted target term).
    B) `logprob_shift` -- exact logprob(target token | prompt), baseline vs anchored vs null, via the
       engine's `/score` route.

  WHATLEARNED: the composed bag(s)' own alpha table rides in `whatlearned`
  (`anchored.whatlearned(bags)` -- the SAME pure lookup the Memory page's "what do you remember?"
  uses, over exactly the bags this receipt injected). This is the receipt legibly NAMING its own
  cause: never "the model remembers X" -- X is injected weighted word-directions, and the alpha table
  IS what was injected (anchored.py's binding HONESTY clause, section 9).

Never raises (mirrors swap_receipt.py's contract): every failure mode -- no engine, nothing to compose
(no bag for the given card_id / no active bags / a degenerate composition), the target term failing to
resolve to one vocab token, a generation failure -- degrades to `causal_verified: False` plus a
`blocked`/`note` explaining exactly why, never a silent guess and never an exception escaping to the
caller. `blocked` is one of: `no_engine`, `no_bag`, `token_resolution`, `generation_failed`, `error`
(catch-all). A handful of purely-informational failures that precede any injection attempt at all --
no run given, a run with no messages, a prompt-template render failure -- leave `blocked: None` with
just a `note` (mirrors swap_receipt's own treatment of "no run"/"no messages"): they are missing INPUT,
not a broken injection/generation pipeline, so they don't consume one of the five enumerated labels.
"""
from __future__ import annotations

import clozn.behavior.steering.concept_dir as concept_dir
import clozn.memory.anchored as anchored

from . import rederive
from .forced import _random_vector_of_norm
from .swap_receipt import _coherence, _concept_mentions, _stable_seed

_NULL_NOTE = (
    "the null arm injects a RANDOM direction of the SAME magnitude (coef) at the SAME layer as the "
    "real anchored injection -- if it shifts the answer toward the target term about as often/strongly "
    "as the real composed bag vector does, the effect is just 'perturbing the residual stream changes "
    "the text,' not a targeted anchored-memory effect. `has_effect`/`targeted_shift` are only claimed "
    "True when the real anchored injection beats BOTH the un-injected baseline AND this null on at "
    "least one measure below (mirrors clozn/receipts/swap_receipt.py's own null-control discipline, "
    "itself ported from the validated run_j5a_swap.py J5a script). When the null arm itself could not "
    "be generated/scored, `null_control_available` is False and `has_effect`/`targeted_shift` reflect "
    "the baseline comparison ONLY -- read it as weaker evidence in that case."
)

_LEXICON_CAVEAT = (
    "lexicon_hits counts LITERAL, case-insensitive, word-boundary mentions of the TARGET TERM -- the "
    "single most heavily-weighted word across every composed bag's alpha_top3, not a caller-supplied "
    "concept (see `injected.target_term`). It will UNDERCOUNT semantically related mentions (e.g. "
    "\"kyoto\" answers phrased around \"Japan\" without saying the word itself) -- read `has_effect`/"
    "`targeted_shift` alongside `logprob_shift` (unaffected by this limitation), never off lexicon_hits "
    "alone. Ported from clozn/receipts/swap_receipt.py's `_LEXICON_CAVEAT`, retargeted from a "
    "caller-supplied `to_concept` to this receipt's own fitted target term."
)


def _top_injected_term(bag_summaries: list) -> str | None:
    """The single most heavily-weighted word across every composed bag's `alpha_top3`
    (`compile_steer()`'s own per-card summary) -- this receipt's ONE legible "named cause". Ties go to
    whichever entry is scanned first (stable, since `alpha_top3` is itself sorted by |alpha| desc per
    card and `compile_steer`'s bag order is deterministic). None when there are simply no terms at all
    (a degenerate/empty composition)."""
    best_token, best_abs = None, -1.0
    for entry in bag_summaries or []:
        for t in (entry or {}).get("alpha_top3") or []:
            token, alpha = t.get("token"), t.get("alpha")
            if not token or alpha is None:
                continue
            a = abs(float(alpha))
            if a > best_abs:
                best_abs, best_token = a, token
    return best_token


def anchored_receipt(run: dict, card_id: str | None, sub, *, max_new: int = 64, null_seed=None) -> dict:
    """The product entry point. Never raises.

    `run`: a stored run record -- the SAME shape `clozn.receipts.rederive.with_arm_conditions` already
        reconstructs a prompt from (messages / assembled_messages).
    `card_id`: OPTIONAL. A specific anchored bag's card_id, or None/''/falsy to use every currently
        active bag (`clozn.memory.anchored.active_bags()`) composed together, exactly as a live chat
        turn would.
    `sub`: a substrate exposing `.engine` (the raw EngineClient: `.apply_template`, `.complete`,
        `.intervene`, and optionally `.score` for the quantitative measure -- the SAME raw-vector steer
        path `anchored.compile_steer()`'s payload is built to feed).

    Returns a dict always carrying: mode, causal_verified, run_id, card_id, injected, whatlearned,
    baseline_reply, anchored_reply, null_reply, lexicon_hits, logprob_shift, has_effect, targeted_shift,
    null_control_available, coherent, coherence_score, null_note, lexicon_note, blocked, note.
    """
    out = {
        "mode": "anchored_receipt", "causal_verified": False, "run_id": None, "card_id": card_id or None,
        "injected": {"card_id": card_id or None, "layer": None, "coef": None, "s_total": None,
                    "bags": None, "target_term": None, "target_token_id": None},
        "whatlearned": None,
        "baseline_reply": None, "anchored_reply": None, "null_reply": None,
        "lexicon_hits": None, "logprob_shift": None,
        "has_effect": None, "targeted_shift": None, "null_control_available": False,
        "coherent": None, "coherence_score": None,
        "null_note": _NULL_NOTE, "lexicon_note": _LEXICON_CAVEAT,
        "blocked": None, "note": None,
    }
    try:
        if not run or not isinstance(run, dict):
            out["note"] = "no run record given"
            return out
        out["run_id"] = run.get("id")

        engine = getattr(sub, "engine", None)
        if engine is None or not all(hasattr(engine, m) for m in ("apply_template", "complete", "intervene")):
            out["blocked"] = "no_engine"
            out["note"] = ("substrate has no .engine (a raw EngineClient exposing apply_template/"
                           "complete/intervene) to render a prompt and inject the composed anchored "
                           "steer on")
            return out

        # WRITE: compose the target bag(s) into ONE steer payload -- a pure read of the anchored-memory
        # store, no engine call (compile_steer() never touches a provider/engine; see its docstring).
        if card_id:
            bag = anchored.get_bag(card_id)
            if bag is None:
                out["blocked"] = "no_bag"
                out["note"] = f"no anchored bag stored for card {card_id!r}"
                return out
            bags = [bag]
        else:
            bags = anchored.active_bags()
            if not bags:
                out["blocked"] = "no_bag"
                out["note"] = "no active anchored bags to inject (nothing anchored yet, or all toggled off)"
                return out
        comp = anchored.compile_steer(bags)
        if comp is None:
            out["blocked"] = "no_bag"
            out["note"] = ("the requested bag(s) did not compose into a steer payload -- toggled off, "
                          "or a degenerate/opposed composition (nothing honest to inject)")
            return out
        out["injected"].update({"layer": comp["layer"], "coef": comp["coef"], "s_total": comp["s_total"],
                                "bags": comp["bags"]})
        out["whatlearned"] = anchored.whatlearned(bags)

        # TARGET TERM: the single most heavily-weighted injected word -- this receipt's own named
        # cause, picked by the fit, never by a caller. Must resolve to ONE vocab token (the same
        # /score round trip concept_dir.ConceptSteer.resolve_token_id uses) for the two measures below.
        top_term = _top_injected_term(comp["bags"])
        if top_term is None:
            out["blocked"] = "no_bag"
            out["note"] = "the composed bag(s) carry no named terms to verify a receipt against"
            return out
        resolver = concept_dir.ConceptSteer(engine, layer=comp["layer"])
        resolved = resolver.resolve_token_id(top_term)
        if not resolved.get("ok"):
            out["blocked"] = "token_resolution"
            out["note"] = (f"could not resolve the top injected term {top_term!r} to a single vocab "
                          f"token for the quantitative measure: {resolved.get('note')}")
            return out
        token_id = resolved["token_id"]
        out["injected"]["target_term"] = top_term
        out["injected"]["target_token_id"] = token_id

        conditions = rederive.with_arm_conditions(run)
        messages = conditions.get("messages") or []
        if not messages:
            out["note"] = "run has no messages to reconstruct a prompt from"
            return out
        try:
            prompt = engine.apply_template(messages, add_assistant=True)
        except Exception as e:
            out["note"] = f"could not render the run's messages into a prompt: {e}"
            return out

        # BASELINE: unsteered generation from the SAME prompt.
        try:
            baseline_reply = concept_dir._text_of(engine.complete(prompt, max_tokens=max_new))
        except Exception as e:
            out["blocked"] = "generation_failed"
            out["note"] = f"baseline generation failed: {e}"
            return out
        out["baseline_reply"] = baseline_reply

        # ANCHORED: the composed bag steer injected at its own coef/layer.
        try:
            anchored_reply = concept_dir._text_of(
                engine.intervene(prompt, vector=comp["vector"], coef=comp["coef"], layer=comp["layer"],
                                 max_tokens=max_new))
        except Exception as e:
            out["blocked"] = "generation_failed"
            out["note"] = f"anchored generation failed: {e}"
            return out
        out["anchored_reply"] = anchored_reply

        # NULL: a random equal-magnitude write at the same layer/coef -- see _NULL_NOTE. Its failure
        # degrades ONLY the null control, not the whole receipt (baseline+anchored already succeeded);
        # `null_control_available` records which case this is.
        seed = null_seed if null_seed is not None else _stable_seed(
            out["run_id"], card_id or "active_bags", comp["layer"])
        null_vec = _random_vector_of_norm(len(comp["vector"]), 1.0, seed)
        null_reply = None
        try:
            null_reply = concept_dir._text_of(
                engine.intervene(prompt, vector=null_vec, coef=comp["coef"], layer=comp["layer"],
                                 max_tokens=max_new))
            out["null_control_available"] = True
        except Exception:
            pass
        out["null_reply"] = null_reply

        # measure A: literal mentions of the target term (swap_receipt's lexicon-hits measure,
        # transferred from a caller-supplied to_concept to this receipt's own strongest named cause).
        base_hits = _concept_mentions(baseline_reply, top_term)
        anchored_hits = _concept_mentions(anchored_reply, top_term)
        null_hits = _concept_mentions(null_reply, top_term) if null_reply is not None else None
        out["lexicon_hits"] = {"baseline": base_hits, "anchored": anchored_hits, "null": null_hits}

        # measure B: quantitative logprob(target token) shift, baseline vs anchored vs null.
        logprob_shift = None
        score_fn = getattr(engine, "score", None)
        if callable(score_fn):
            try:
                base_lp = score_fn(prompt=prompt, continuation_ids=[token_id], topk=0)["tokens"][0]["logprob"]
                anchored_lp = score_fn(prompt=prompt, continuation_ids=[token_id], topk=0,
                                       steer_vec=comp["vector"],
                                       steer={"coef": comp["coef"], "layer": comp["layer"]}
                                       )["tokens"][0]["logprob"]
                null_lp = None
                try:
                    null_lp = score_fn(prompt=prompt, continuation_ids=[token_id], topk=0,
                                       steer_vec=null_vec, steer={"coef": comp["coef"], "layer": comp["layer"]}
                                       )["tokens"][0]["logprob"]
                except Exception:
                    null_lp = None
                logprob_shift = {
                    "baseline": float(base_lp), "anchored": float(anchored_lp),
                    "null": (float(null_lp) if null_lp is not None else None),
                    "anchored_over_baseline_nat": round(float(anchored_lp) - float(base_lp), 4),
                    "anchored_over_null_nat": (round(float(anchored_lp) - float(null_lp), 4)
                                              if null_lp is not None else None),
                }
            except Exception:
                logprob_shift = None
        out["logprob_shift"] = logprob_shift

        coherence_score, is_coherent = _coherence(anchored_reply)
        out["coherence_score"] = coherence_score
        out["coherent"] = is_coherent

        beats_baseline_gen = anchored_hits > base_hits
        beats_null_gen = null_hits is None or anchored_hits > null_hits
        gen_shift = beats_baseline_gen and beats_null_gen
        score_shift = False
        if logprob_shift is not None:
            beats_baseline_score = logprob_shift["anchored_over_baseline_nat"] > 1.0
            beats_null_score = (logprob_shift["anchored_over_null_nat"] is None
                                or logprob_shift["anchored_over_null_nat"] > 1.0)
            score_shift = beats_baseline_score and beats_null_score
        # has_effect / targeted_shift: computed ONLY from the actual diff above -- never inferred, never
        # defaulted to True just because a bag composed cleanly (a clean composition proves nothing by
        # itself; only the generated diff, beyond the null, does).
        effect = bool(gen_shift or score_shift)
        out["has_effect"] = effect
        out["targeted_shift"] = effect

        out["causal_verified"] = True
        return out
    except Exception as e:
        out["blocked"] = out.get("blocked") or "error"
        out["note"] = out.get("note") or f"anchored_receipt failed unexpectedly: {e}"
        return out
