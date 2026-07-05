"""receipts_as_reward.py -- Wild Experiment #8 (Wave 2): the ablation RECEIPT as a REWARD, evolving the
memory prompt-block's WORDING with no gradients and no LLM judge.

Pre-registration: research/WILD_WAVE2_PREREG.md, "Exp #8 -- Receipts as reward". Read that first -- this
module is a literal build of that spec; deliberate design choices it left open are called out below.

WHAT'S BEING OPTIMIZED. memory_mode.compile_prompt_block(cards, style="soft") compiles the studio's active
memory cards into ONE system-prompt block, e.g. "You are a helpful assistant ... use it naturally to
tailor how you respond:\n- <card>\n- <card>". The CARDS (the facts/rules) are FIXED here -- one concept
card (baking) + one rule card (concise), self_audit_gap.py's two Wave-1 traits that trained reliably. Only
the WRAPPING WORDING (the "You are a helpful assistant..." instruction) evolves; internally it is a
"{RULES}"-templated string (see render_block / SEED_WORDING) so the substitution point never has to be
guessed out of a rephrase.

THE RECEIPT AS FITNESS (score_wording). For one candidate wording: render the block, generate one GREEDY
reply per probe (deterministic -- a re-score of the same wording always reproduces), then

    expression = mean hit-rate over ON-TOPIC probes  (HELDOUT below, copied from self_audit_gap.HELDOUT:
                 neutral, open-ended -- baking COULD naturally surface, brevity IS a live stylistic choice)
    bleed      = mean hit-rate over OFF-TOPIC probes  (OFF_TOPIC_PROBES below, new here, in
                 steer_vs_prompt.py's bleed spirit generalized to a concept AND a rule card at once: each
                 probe names a subject with no natural link to baking -- so a baking keyword in the reply
                 is an unambiguous non-sequitur -- AND explicitly asks for a long, detailed answer -- so a
                 reply that still comes back short is the concise card overriding a direct contrary
                 instruction. Same two hit functions, just a different probe set and a different reading
                 of a "hit": showing up where it shouldn't, instead of where it should.)
    fitness    = expression - LAMBDA * bleed   (LAMBDA = 1.0, documented, ONE value, not swept --
                 symmetric: a point of bleed costs exactly a point of expression)

Both rates are ABSOLUTE (a plain rate, not a with/without-memory delta): that is the pre-reg's own
definition ("trait-shows-up rate" / "trait-leaks rate"), and it halves the generations needed per
candidate, since no separate no-block baseline arm has to be scored. ANY reply (on- or off-topic) that
trips counterfactual._coherence disqualifies the WHOLE wording (fitness -> DISQUALIFIED_FITNESS, a large
negative sentinel -- not -inf, so the run JSON stays strictly valid): a wording cannot win by making the
model degenerate into repetition/gibberish that happens to also contain a keyword.

EVOLUTION (no gradients, no judge). G generations, (1+K)-elitist: each step asks the model itself to
REPHRASE the current wording K ways (`mutate` -- a SAMPLED generation; cards/facts are never touched, only
the phrasing), scores {current} + the K children, and the trajectory's `select` rule picks the next
current from that pool of K+1 candidates:

    "fitness" (evolved)     -- argmax fitness. Python's max() keeps the FIRST maximal element, and
                               `current` sits at pool index 0, so a generation with no strictly-better
                               child deliberately STAYS PUT (elitism: a run can never lose fitness to an
                               unlucky batch of rephrasings).
    "random"  (random_walk) -- uniform random choice over the SAME pool, via a seeded RNG, ignoring
                               fitness entirely -- REQUIRED NULL #2. Isolates "receipt-guided SELECTION
                               helped" from "any rephrasing drift helped": if evolved's climb matches
                               random_walk's, selection added nothing.

REQUIRED NULL #1 (SEED) is just score_wording(SEED_WORDING) -- the shipped, un-evolved block -- reported
as the flat reference line both trajectories start from.

"Mutate identically" (the pre-reg's own phrase for the random-walk null) is made LITERAL here, not just
same-procedure: both trajectories' mutation calls run through ONE shared cache keyed by (current wording,
k) (see run_experiment). Whenever they ask to mutate the SAME wording -- guaranteed true at generation 1,
since both start at SEED_WORDING -- they get the IDENTICAL K children back, not an independent stochastic
re-draw. They are free to diverge afterwards: the moment their selections differ, they are mutating
different parents, so there is nothing left to share. Selection is the only designed-in asymmetry between
the two arms.

DONE / FALSIFIABLE (`_verdict`): evolved > seed AND > random_walk -> the receipt is a usable optimizer (a
real win). evolved ~= random_walk -> selection did not help -- an honest negative, even if evolved > seed
(drift alone can produce that; it is exactly why the null exists).

CAVEATS, LOUD:
  * ONE memory (2 cards), ONE model family (Qwen2.5-7B-Instruct nf4 -- the pre-reg is explicit this is NOT
    a cross-family question), ONE lambda value (not swept), a single trajectory per arm per invocation
    (--seed controls the two arms' RNG streams; re-running with a different --seed is how you would check
    variance across a seed -- not done by this module itself).
  * expression/bleed are CRUDE, LEXICAL, ABSOLUTE-THRESHOLD scorers (a keyword-substring hit for baking, a
    flat word-count cutoff for concise) -- exactly the kind of metric this codebase has watched get GAMED
    by degeneration before (steer_vs_prompt_findings.md). The coherence gate and the random-walk null are
    the guards, not a claim that the scorers themselves are rigorous: if fitness climbs but the winning
    wordings are eyeball-nonsense, or the "win" is really just random-walk in a wig, that is a
    scorer-gaming / null-failure result, and this module is built to SURFACE it (every generation's chosen
    wording + rendered block + sample replies are kept in the run JSON) rather than hide it.
  * the MUTATOR is the audited model itself (an LLM rephrasing its own instruction block) -- but the
    SELECTOR is the measured receipt (or, in the null, chance), never the model's own judgment of which
    rephrasing reads better. That split is the whole point of the experiment.
  * G/K are modest by construction (compute scales as roughly G generations x 2 arms x up to K+1
    candidates x len(on_topic)+len(off_topic) greedy generations -- fewer once the elitist parent starts
    getting cache hits -- plus K short sampled mutation calls per generation, shared across arms by the
    cache above). Defaults below are a bounded single-GPU job, not tuned against any observed result --
    see "GPU SMOKE: PENDING".

CONVENTIONS REUSED: self_audit_gap.py's HELDOUT probes, its baking trait's rule text + keyword list and
its concise trait's rule text (the two reliably-training Wave-1 traits), mirror_bench.py's wants_four_bit,
counterfactual._coherence (the mandatory coherence gate), and the SelfTeach-based generate path
(`_make_gen_fns` wraps SelfTeach._generate with use_prefix=False -- the INTERNALIZED trained soft-prefix
is never touched by this experiment; only the compiled PROMPT block, as a system message, is under test).

WHY SOME OF THOSE ARE COPIED, NOT IMPORTED: self_audit_gap.py imports self_teach_server (torch,
transformers, bitsandbytes) AT MODULE LEVEL, so importing anything from it would make a bare `import
receipts_as_reward` require torch to be installed -- and so would importing mirror_bench.py (which itself
imports from self_audit_gap). This module instead copies the tiny needed surface (HELDOUT, the baking rule
+ keywords, the concise rule text, kw_hit(), wants_four_bit()) so `import receipts_as_reward` is torch-free
and the whole model-free test file (test_receipts_as_reward.py) never needs torch installed to run --
exactly the concern research/tests/test_prompt_vs_prefix_ab.py's own `_lazy()` helper solves for its rig
("so the plain model-free suite can collect this file without touching CUDA"). This module resolves the
same concern by copying the few needed literals instead of a deferred-import wrapper, since the borrowed
surface is this small; test_receipts_as_reward.py opportunistically cross-checks the copies against a live
self_audit_gap / mirror_bench import when torch happens to be available, so drift is still caught wherever
it can be. self_teach_server.SelfTeach itself is still imported -- but only inside `run()`, exactly like
mirror_bench.py's own `run()` defers it.

counterfactual.py and memory_mode.py ARE imported directly at module level: both are stdlib-only by their
own docstrings (no torch anywhere in their import chains), so `_coherence` and `compile_prompt_block` cost
nothing extra. SEED_WORDING is DERIVED from a live call to memory_mode.compile_prompt_block(CARDS,
style="soft") -- not a hand-copied literal -- so it is provably "the studio's current compiled block" (the
pre-reg's own definition of the seed), and automatically tracks memory_mode.py if that wording is ever
edited.

GPU SMOKE: PENDING. This authoring pass validates import-cleanliness (no torch required) and the full
pure-logic suite against a FAKE generator (test_receipts_as_reward.py) -- no GPU job was run (the parent
session serializes all GPU work; not this agent's to run). `run()` / `_make_gen_fns` are wired but
UNVALIDATED against a real Qwen-7B: treat the CLI below as reviewed-but-unrun until a --smoke pass actually
executes.

USAGE (CUDA venv):
    PY=C:/Users/brigi/src/cloze/.venv/Scripts/python.exe
    $PY research/receipts_as_reward.py --smoke --out research/runs/receipts_as_reward_smoke.json
    $PY research/receipts_as_reward.py --out research/runs/receipts_as_reward.json \
        --generations 6 --k 4 --lambda 1.0
Model-free tests (no GPU, no torch required):
    $PY -m pytest research/tests/test_receipts_as_reward.py -q
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from counterfactual import _coherence   # stdlib-only (no torch) -- see docstring "WHY SOME ARE COPIED"
import memory_mode                       # stdlib-only (no torch)

# ==================================================================================== the fixed memory
# One concept card (baking) + one rule card (concise) -- self_audit_gap.py TRAITS[0]/TRAITS[2], the two
# traits Wave-1 found trained reliably. Text/keywords copied verbatim (not imported -- see module
# docstring); test_receipts_as_reward.py cross-checks the copy against a live self_audit_gap when torch
# happens to be importable.
_BAKING_RULE = "The user is really into baking."
_BAKING_KW = ["bak", "bread", "dough", "oven", "cake", "pastry", "flour", "cinnamon", "recipe", "knead",
             "yeast", "loaf", "muffin"]
_CONCISE_RULE = "The user wants you to answer very concisely, in one short sentence."
CARDS = [_BAKING_RULE, _CONCISE_RULE]

# ON-TOPIC / expression probes: neutral, open-ended -- copied verbatim from self_audit_gap.HELDOUT. Baking
# COULD naturally surface here, and brevity IS a live stylistic choice -- exactly where a working memory
# block SHOULD make the traits show up.
HELDOUT = [
    "How was your morning?",
    "I'm not sure what to do this evening.",
    "Can you help me think through my week?",
    "Tell me a fun fact.",
    "I'm feeling a little tired today.",
    "Describe a nice place to relax.",
]

# OFF-TOPIC / bleed probes: new to this experiment (steer_vs_prompt.py's bleed spirit, generalized to both
# card shapes at once). Each probe is (a) about a subject with zero natural connection to baking, so any
# baking keyword in the reply is an unambiguous non-sequitur, AND (b) an EXPLICIT ask for a long, detailed
# answer, so a reply that still comes back short is the concise card overriding a direct contrary
# instruction -- both cards' "shouldn't be here" failure mode, measured on the same probe set.
OFF_TOPIC_PROBES = [
    "Give me a detailed, thorough, multi-paragraph explanation of how compound interest works on a loan.",
    "Explain in depth, step by step, how a car engine converts fuel into motion.",
    "Walk me through, with full detail, the water cycle from evaporation to precipitation.",
    "Describe at length the causes and course of the fall of the Roman Empire.",
    "Give a comprehensive, detailed walkthrough of how to file a small-claims lawsuit.",
    "Explain thoroughly, in several paragraphs, how vaccines train the immune system.",
]

LAMBDA_DEFAULT = 1.0    # documented single value (pre-reg: "lambda documented, one value" -- not swept):
                        # symmetric weighting, one point of bleed costs exactly one point of expression.
CONCISE_WORD_MAX = 30   # generous absolute word-count cutoff for "answered concisely" (rule text says
                        # "one short sentence"); crude and threshold-y on purpose -- transparent, not tuned.
DISQUALIFIED_FITNESS = -1e9   # NOT -inf: keeps the run JSON strictly valid JSON. Safely below any
                              # achievable real fitness (expression, bleed both in [0,1] -> fitness in
                              # roughly [-lambda, 1]).
_RULES_TOKEN = "{RULES}"


# ============================================================================== wording <-> block render
def _bullets(cards) -> str:
    """Mirrors memory_mode.compile_prompt_block's own bullet formatting exactly (str + strip each, drop
    blanks, "- " prefix, "\n"-joined) so a rendered block is byte-comparable to the shipped compiler
    whenever the wording's preamble matches (see the SEED_WORDING derivation/round-trip below)."""
    return "\n".join("- " + str(c).strip() for c in (cards or []) if str(c or "").strip())


def render_block(wording: str, cards) -> str:
    """Render one candidate WORDING (a template containing the literal token "{RULES}") with the fixed
    CARDS substituted in verbatim -- the cards/facts never change during evolution, only `wording` does.
    If a mutated wording lost the token (an LLM rephrase can mangle or translate it), the bullets are
    APPENDED instead of silently dropping the cards: a fixed-cards violation must never be possible by
    construction, no matter how a rephrase mangles the template."""
    bullets = _bullets(cards)
    if _RULES_TOKEN in wording:
        return wording.replace(_RULES_TOKEN, bullets)
    return wording.rstrip() + "\n" + bullets


def _derive_seed_wording(cards) -> str:
    """The SEED wording, DERIVED from the studio's own memory_mode.compile_prompt_block(cards, "soft")
    rather than a hand-copied literal -- so it is provably "the studio's current compiled block" (the
    pre-reg's own definition of the seed) by construction, and automatically tracks memory_mode.py if its
    soft wording is ever edited. Fails loudly at import time (an explicit, un-optimizable check, not a bare
    assert) if compile_prompt_block's output ever stops being exactly `preamble + bullets`, since that
    would mean the {RULES}-templating below no longer holds."""
    full = memory_mode.compile_prompt_block(list(cards), style="soft")
    bullets = _bullets(cards)
    if bullets and not full.endswith(bullets):
        raise RuntimeError("memory_mode.compile_prompt_block's soft wording no longer ends with the plain "
                           "bullets list -- SEED_WORDING derivation in receipts_as_reward.py needs a "
                           "matching update")
    preamble = full[: len(full) - len(bullets)] if bullets else full
    return preamble + _RULES_TOKEN


SEED_WORDING = _derive_seed_wording(CARDS)


# ======================================================================================= hit / rate math
def kw_hit(text: str, kw) -> bool:
    """Copied verbatim from self_audit_gap.kw_hit (see module docstring)."""
    t = (text or "").lower()
    return any(k in t for k in kw)


def _baking_hit(text: str) -> bool:
    return kw_hit(text, _BAKING_KW)


def _word_count(text: str) -> int:
    return len((text or "").split())


def _concise_hit(text: str, max_words: int = CONCISE_WORD_MAX) -> bool:
    return _word_count(text) <= max_words


def _hit_rate(hit_fn, replies) -> float:
    n = max(1, len(replies))
    return sum(1 for r in replies if hit_fn(r)) / n


# ============================================================================================ the receipt
def score_wording(gen_fn, wording: str, *, cards=None, on_topic=None, off_topic=None,
                  lam: float = LAMBDA_DEFAULT, concise_max: int = CONCISE_WORD_MAX) -> dict:
    """THE RECEIPT for one candidate wording (pre-reg #8). `gen_fn(block, probe) -> str` is the only model
    touchpoint -- one GREEDY reply per probe (deterministic: re-scoring the same wording always
    reproduces, which is what makes the caches in run_generations/run_experiment safe).

    expression = mean hit-rate over ON-TOPIC probes (both cards); bleed = mean hit-rate over OFF-TOPIC
    probes (both cards); fitness = expression - lam*bleed, UNLESS any reply (on- or off-topic) trips
    counterfactual._coherence, in which case fitness = DISQUALIFIED_FITNESS regardless of the raw
    numbers -- a wording cannot win by making the model degenerate into repetition/gibberish that happens
    to contain a keyword. Returns the full detail (rendered block, per-card rate breakdown, degenerate
    reasons, up to 2 sample replies per probe set) so a human can eyeball a fitness climb, not just trust
    the scalar."""
    cards = CARDS if cards is None else list(cards)
    on_topic = list(HELDOUT if on_topic is None else on_topic)
    off_topic = list(OFF_TOPIC_PROBES if off_topic is None else off_topic)
    block = render_block(wording, cards)
    on_replies = [gen_fn(block, p) for p in on_topic]
    off_replies = [gen_fn(block, p) for p in off_topic]

    coh = [_coherence(r) for r in on_replies + off_replies]
    degenerate_reasons = sorted({c["reason"] for c in coh if c["degenerate"]})
    coherence_ok = not degenerate_reasons

    concise_hit = lambda r: _concise_hit(r, concise_max)   # noqa: E731
    exp_baking, exp_concise = _hit_rate(_baking_hit, on_replies), _hit_rate(concise_hit, on_replies)
    bleed_baking, bleed_concise = _hit_rate(_baking_hit, off_replies), _hit_rate(concise_hit, off_replies)
    expression = round((exp_baking + exp_concise) / 2, 4)
    bleed = round((bleed_baking + bleed_concise) / 2, 4)
    raw_fitness = round(expression - lam * bleed, 4)
    fitness = raw_fitness if coherence_ok else DISQUALIFIED_FITNESS

    return {
        "wording": wording, "block": block,
        "expression": expression, "bleed": bleed,
        "expression_parts": {"baking": round(exp_baking, 4), "concise": round(exp_concise, 4)},
        "bleed_parts": {"baking": round(bleed_baking, 4), "concise": round(bleed_concise, 4)},
        "lambda": lam, "raw_fitness": raw_fitness, "fitness": fitness,
        "coherence_ok": coherence_ok, "degenerate_reasons": degenerate_reasons,
        "samples": {"on_topic": on_replies[:2], "off_topic": off_replies[:2]},
    }


# ================================================================================================ mutate
_MUTATE_PREFIX = (
    "Rewrite the instruction below so it reads differently -- different words, different sentence "
    "structure -- but means exactly the same thing. Do not change, translate, drop, or explain the token "
    "{RULES}; leave it exactly where it is, verbatim -- it will be replaced by real content afterward. Do "
    "not add any new facts or rules of your own. Reply with ONLY the rewritten instruction and nothing "
    "else: no preamble, no quotes, no commentary.\n\nInstruction to rewrite:\n"
)

_MUTATION_LABELS = ("rewritten instruction:", "rewrite:", "instruction:",
                    "here is the rewritten instruction:", "here's the rewritten instruction:")


def _clean_mutation(text: str, fallback: str) -> str:
    """Sanitize one raw mutation reply into a usable wording template: strip a leading meta-label the
    model added despite the instruction, strip wrapping quotes, and fall back to the PARENT wording
    VERBATIM (never an empty/garbage template) if the cleaned result is empty. render_block's own
    append-fallback still catches a {RULES} token the rewrite dropped, so this is belt-and-suspenders, not
    the only guard against a broken mutation losing the cards."""
    t = (text or "").strip()
    low = t.lower()
    for label in _MUTATION_LABELS:
        if low.startswith(label):
            t = t[len(label):].strip()
            low = t.lower()
    if len(t) >= 2 and t[0] in "\"'" and t[-1] in "\"'":
        t = t[1:-1].strip()
    return t if t else fallback


def mutate(gen_fn_sampled, wording: str, k: int) -> list[str]:
    """K independent LLM-driven rephrasings of `wording` -- cards/facts untouched, only the wrapping
    instruction's phrasing varies (pre-reg #8: "the model itself rephrases the block instruction"). Each
    of the K calls is an independent, SAMPLED generation (`gen_fn_sampled(prompt) -> str`; the real wiring
    uses SelfTeach._generate(..., sample=True), temperature 0.7 / top_p 0.9 -- diversity is the point here,
    unlike the greedy scoring path). No shared state between calls -- one bad rephrase can't cascade."""
    return [_clean_mutation(gen_fn_sampled(_MUTATE_PREFIX + wording), fallback=wording)
            for _ in range(max(0, int(k)))]


# ========================================================================================== evolution loop
def run_generations(score_fn, mutate_fn, seed_wording: str, generations: int, k: int, *, select: str,
                    rng: random.Random, cache: dict | None = None, on_generation=None) -> dict:
    """One (1+K)-elitist evolutionary trajectory over `generations` steps. Each step mutates the CURRENT
    wording `k` ways via `mutate_fn(current, k)`, forms the pool {current} + children, scores every member
    with `score_fn` (memoized in `cache`, keyed by wording text -- safe because score_fn is a
    pure/deterministic function of the wording when scoring is greedy), and picks the next `current` by
    `select`:
      "fitness" -- argmax fitness (max() keeps the FIRST maximal element, and `current` sits at pool index
                   0, so a generation with no strictly-better child deliberately stays put -- the "keep
                   best" the pre-reg asks for, elitist so a run can never regress below a previously-
                   reached fitness purely from unlucky mutations).
      "random"  -- rng.randrange over the SAME pool, completely ignoring fitness -- the random-walk null.
    Disqualified candidates carry DISQUALIFIED_FITNESS, so "fitness" selection can never pick one over any
    coherent alternative; "random" selection CAN (by design -- the null gets no help from the coherence
    gate either, though score_fn still runs it, so the disqualification is visible in the trace).
    `on_generation(g, entry)`, if given, fires after each generation (g=1..generations) with that
    generation's trace entry -- the real run's checkpoint-save hook."""
    cache = {} if cache is None else cache
    if seed_wording not in cache:
        cache[seed_wording] = score_fn(seed_wording)
    current = seed_wording
    trace = [{"generation": 0, "chosen": current, "score": cache[current], "pool": [current]}]
    for g in range(1, int(generations) + 1):
        children = mutate_fn(current, k)
        pool = [current] + list(children)
        for w in pool:
            if w not in cache:
                cache[w] = score_fn(w)
        scored = [cache[w] for w in pool]
        if select == "fitness":
            best_i = max(range(len(pool)), key=lambda i: scored[i]["fitness"])
        elif select == "random":
            best_i = rng.randrange(len(pool))
        else:
            raise ValueError(f"unknown select mode: {select!r}")
        current = pool[best_i]
        entry = {"generation": g, "chosen": current, "chosen_index": best_i, "score": scored[best_i],
                 "pool": pool, "pool_fitness": [s["fitness"] for s in scored]}
        trace.append(entry)
        if on_generation is not None:
            on_generation(g, entry)
    return {"final_wording": current, "final_score": trace[-1]["score"],
            "fitness_by_generation": [t["score"]["fitness"] for t in trace], "trace": trace}


def _verdict(seed_score: dict, evolved: dict, random_walk: dict, tie_eps: float = 0.02) -> dict:
    """The pre-reg's falsifiable call: evolved > seed AND > random-walk -> the receipt is a usable
    optimizer; evolved ~= random-walk -> selection did not help (an honest negative), regardless of
    whether evolved beat the seed (drift alone can do that too -- the whole reason the null exists).
    `tie_eps` is an ABSOLUTE margin on the bounded fitness scale (expression, bleed both in [0,1], lambda
    documented as 1.0 -> fitness roughly in [-1, 1]); 0.02 is one eyeball-small step, not fit to any data."""
    sf = seed_score["fitness"]
    ef = evolved["final_score"]["fitness"]
    rf = random_walk["final_score"]["fitness"]
    beats_seed = ef > sf + tie_eps
    beats_random = ef > rf + tie_eps
    close_to_random = abs(ef - rf) <= tie_eps
    if beats_seed and beats_random:
        label = "receipt-guided selection wins: evolved beats both the seed and the random-walk null"
    elif close_to_random:
        label = ("evolved ~= random-walk: selection did not help (honest negative) -- rephrasing drift "
                 "alone explains the climb, if any")
    elif not beats_seed:
        label = "evolved did not beat the seed wording"
    else:
        label = ("evolved beat the seed but not clearly the random-walk null -- selection's own "
                 "contribution is unclear")
    return {"seed_fitness": sf, "evolved_fitness": ef, "random_walk_fitness": rf,
            "beats_seed": beats_seed, "beats_random_walk": beats_random,
            "seed_disqualified": sf <= DISQUALIFIED_FITNESS / 2, "tie_eps": tie_eps, "label": label}


def run_experiment(gen_fn, gen_fn_sampled, *, generations: int = 6, k: int = 4,
                   lam: float = LAMBDA_DEFAULT, seed: int = 0, cards=None, on_topic=None, off_topic=None,
                   concise_max: int = CONCISE_WORD_MAX, on_update=None) -> dict:
    """Run the full pre-reg #8 rig: score the SEED wording once, then two independent (1+K)-elitist
    trajectories of `generations` steps from that SAME seed -- "evolved" (selects by fitness) and
    "random_walk" (selects uniformly at random; the load-bearing null) -- plus the falsifiable verdict
    comparing all three.

    The two trajectories' mutation calls run through ONE shared cache keyed by (current wording, k): the
    first time EITHER trajectory asks to mutate a given wording, the K children are generated and cached;
    if the OTHER trajectory later asks to mutate the identical wording (guaranteed true at generation 1,
    since both start at SEED_WORDING; possible later too, by chance), it gets the SAME children back
    rather than an independent stochastic re-draw. This makes "mutate identically" (the pre-reg's phrase
    for the null) literal wherever the two trajectories actually coincide, while leaving them free to
    diverge the moment their SELECTIONS differ -- selection is the only designed-in asymmetry.

    `on_update(arm, g, entry)`, if given, fires once for the seed (arm="seed", g=0) and once per
    generation of each arm ("evolved" / "random_walk", g=1..generations) -- the real run's checkpoint
    hook (see `run`)."""
    cards = CARDS if cards is None else list(cards)
    on_topic = list(HELDOUT if on_topic is None else on_topic)
    off_topic = list(OFF_TOPIC_PROBES if off_topic is None else off_topic)

    def score_fn(wording):
        return score_wording(gen_fn, wording, cards=cards, on_topic=on_topic, off_topic=off_topic,
                             lam=lam, concise_max=concise_max)

    mutation_cache: dict = {}

    def shared_mutate(wording, kk):
        key = (wording, kk)
        if key not in mutation_cache:
            mutation_cache[key] = mutate(gen_fn_sampled, wording, kk)
        return list(mutation_cache[key])

    score_cache: dict = {}
    seed_score = score_fn(SEED_WORDING)
    score_cache[SEED_WORDING] = seed_score
    if on_update is not None:
        on_update("seed", 0, {"generation": 0, "chosen": SEED_WORDING, "score": seed_score})

    def cb(arm):
        return None if on_update is None else (lambda g, entry: on_update(arm, g, entry))

    evolved = run_generations(score_fn, shared_mutate, SEED_WORDING, generations, k, select="fitness",
                              rng=random.Random(seed), cache=score_cache, on_generation=cb("evolved"))
    random_walk = run_generations(score_fn, shared_mutate, SEED_WORDING, generations, k, select="random",
                                  rng=random.Random(seed + 1), cache=score_cache,
                                  on_generation=cb("random_walk"))

    verdict = _verdict(seed_score, evolved, random_walk)
    return {"cards": cards, "lambda": lam, "generations": generations, "k": k, "seed": seed,
            "concise_max": concise_max, "on_topic_probes": on_topic, "off_topic_probes": off_topic,
            "seed_wording": SEED_WORDING, "seed_baseline": seed_score,
            "evolved": evolved, "random_walk": random_walk, "verdict": verdict}


# ===================================================================================== model plumbing
_SMALL = ("0.5b", "1.5b", "-1b", "1b-", "2b", "3b", "-1.7b")   # copied from mirror_bench.py (see docstring)


def wants_four_bit(name: str, override: str) -> bool:
    """Copied verbatim from mirror_bench.py (see module docstring)."""
    if override == "yes":
        return True
    if override == "no":
        return False
    return not any(s in name.lower() for s in _SMALL)


def _make_gen_fns(app, max_new: int = 90, mutate_max_new: int = 150):
    """Wire the pure logic above to a live SelfTeach `app` (research/self_teach_server.py). Both closures
    use use_prefix=False: the INTERNALIZED trained soft-prefix is never touched by this experiment -- only
    the compiled PROMPT block (a system message) is under test, exactly the "prompt" memory_mode this
    experiment is about. apply_gate=False/gate=1.0 because there is no prefix here for the topic gate to
    scale.
      gen_fn(block, probe)   -- GREEDY (sample=False): the scoring path. Deterministic, so the score/
                                mutation caches above are safe to reuse across generations.
      gen_fn_sampled(prompt) -- SAMPLED (sample=True, the class's own temperature=0.7/top_p=0.9 defaults):
                                the mutation path, where diversity across the K rephrasings is the point."""
    def gen_fn(block: str, probe: str) -> str:
        messages = [{"role": "system", "content": block}, {"role": "user", "content": probe}]
        return app._generate(messages, use_prefix=False, max_new=max_new, sample=False,
                             gate=1.0, apply_gate=False)

    def gen_fn_sampled(prompt: str) -> str:
        return app._generate([{"role": "user", "content": prompt}], use_prefix=False,
                             max_new=mutate_max_new, sample=True, gate=1.0, apply_gate=False)

    return gen_fn, gen_fn_sampled


# =================================================================================================== run
def run(model_name: str, out_path: str, *, four_bit_override: str = "auto", smoke: bool = False,
       generations: int = 6, k: int = 4, lam: float = LAMBDA_DEFAULT, seed: int = 0,
       concise_max: int = CONCISE_WORD_MAX, max_new: int = 90):
    """The real GPU run (NOT executed by this authoring pass -- see module docstring's "GPU SMOKE:
    PENDING"). Loads Qwen nf4 via SelfTeach (imported here, not at module level -- see the docstring on
    why), then runs `run_experiment` with closures wired to the live model, checkpointing the whole `res`
    dict to `out_path` after every single generation of either arm (a kill/OOM mid-run keeps every
    generation already scored, mirroring mirror_bench.py's per-trait checkpoint)."""
    from self_teach_server import SelfTeach
    t0 = time.time()
    four_bit = wants_four_bit(model_name, four_bit_override)
    gens = 2 if smoke else generations
    kk = 2 if smoke else k
    on_topic = HELDOUT[:2] if smoke else HELDOUT
    off_topic = OFF_TOPIC_PROBES[:2] if smoke else OFF_TOPIC_PROBES

    print(f"[load] {model_name} ({'nf4' if four_bit else 'bf16'}, cuda){' [SMOKE]' if smoke else ''} ...",
          flush=True)
    app = SelfTeach(model_name, m=16, four_bit=four_bit, persist_path=None)
    gen_fn, gen_fn_sampled = _make_gen_fns(app, max_new=max_new)
    print(f"[run] G={gens} K={kk} lambda={lam} concise_max={concise_max} "
          f"on_topic={len(on_topic)} off_topic={len(off_topic)}", flush=True)

    res = {"model": model_name, "four_bit": four_bit, "smoke": smoke, "generations": gens, "k": kk,
           "lambda": lam, "seed": seed, "concise_max": concise_max, "max_new": max_new,
           "cards": CARDS, "on_topic_probes": on_topic, "off_topic_probes": off_topic,
           "progress": {"seed": None, "evolved": [], "random_walk": []}}

    def on_update(arm, g, entry):
        if arm == "seed":
            res["progress"]["seed"] = entry
        else:
            res["progress"][arm].append(entry)
        s = entry["score"]
        print(f"  [{arm} gen {g}] fitness={s['fitness']:.4f} expr={s['expression']:.3f} "
              f"bleed={s['bleed']:.3f} coherent={s['coherence_ok']}", flush=True)
        _save(out_path, res)

    experiment = run_experiment(gen_fn, gen_fn_sampled, generations=gens, k=kk, lam=lam, seed=seed,
                                on_topic=on_topic, off_topic=off_topic, concise_max=concise_max,
                                on_update=on_update)
    res.update(experiment)
    res["seconds"] = round(time.time() - t0, 1)
    _save(out_path, res)
    _summary(res)
    print(f"\nsaved -> {out_path}", flush=True)
    return res


def _save(out_path: str, res: dict) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def _summary(res: dict) -> None:
    print("\n" + "=" * 78, flush=True)
    print(f"RECEIPTS AS REWARD -- {res.get('model', '?')}{' [SMOKE]' if res.get('smoke') else ''}",
          flush=True)
    v = res["verdict"]
    print(f"lambda={res.get('lambda')}  G={res.get('generations')}  K={res.get('k')}", flush=True)
    print(f"seed fitness        = {v['seed_fitness']:.4f}", flush=True)
    print(f"evolved fitness     = {v['evolved_fitness']:.4f}", flush=True)
    print(f"random-walk fitness = {v['random_walk_fitness']:.4f}", flush=True)
    print(f"-> {v['label']}", flush=True)
    print("\ngen | evolved  | random-walk", flush=True)
    ef, rf = res["evolved"]["fitness_by_generation"], res["random_walk"]["fitness_by_generation"]
    for g, (e, r) in enumerate(zip(ef, rf)):
        print(f"{g:3d} | {e:8.4f} | {r:8.4f}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--out", default="research/runs/receipts_as_reward.json")
    ap.add_argument("--four-bit", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny: 2 generations, K=2, 2 on-topic + 2 off-topic probes -- prove the wiring cheap")
    ap.add_argument("--generations", type=int, default=6, help="G: evolutionary generations per arm")
    ap.add_argument("--k", type=int, default=4, help="K: rephrasings sampled per generation")
    ap.add_argument("--lambda", dest="lam", type=float, default=LAMBDA_DEFAULT,
                    help="bleed penalty weight: fitness = expression - lambda*bleed")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for the random-walk null's selection (and the two arms' stream offset)")
    ap.add_argument("--concise-max", type=int, default=CONCISE_WORD_MAX,
                    help="word-count cutoff for the 'answered concisely' hit test")
    ap.add_argument("--max-new", type=int, default=90, help="max new tokens per probe reply when scoring")
    return ap


if __name__ == "__main__":
    args = _build_parser().parse_args()
    out = args.out.replace(".json", "_smoke.json") if args.smoke else args.out
    run(args.model, out, four_bit_override=args.four_bit, smoke=args.smoke, generations=args.generations,
        k=args.k, lam=args.lam, seed=args.seed, concise_max=args.concise_max, max_new=args.max_new)
