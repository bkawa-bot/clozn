"""dial_autocalibrate.py -- per-dial, per-model COHERENT OPERATING RANGE calibration.

Antecedent: parliament.py's `calibrate_and_check_liveness` (per-axis dose sweep + a shuffled-direction
liveness check + a coherence gate, on 5 fixed stances, choosing ONE operating dose). This module generalizes
that pattern from "pick one dose for these 5 stances" to "for ANY dial (the full steering.AXES tone-dial
set, plus parliament's skeptical/plain customs, or an arbitrary --dials list), report the FULL usable dose
RANGE" -- Law #6's studio-facing need: a 7B-calibrated dial derails a 1.5B, so the slider must show
0-to-where-it-actually-works, PER MODEL, not one asserted number.

WHAT THIS DOES NOT DO (read before trusting a number out of this file): it does NOT pick a single "best"
dose. Ranking doses against each other needs a USER-PREFERENCE signal (which dose's warmth/candor/whatever
the user actually likes) that nothing in this codebase collects yet. What IS measurable without that
signal: (a) where the dial provably breaks the model (derail_point, via counterfactual._coherence -- the
same mandatory degeneration gate every dial/steering receipt in this codebase already uses), and (b) where
the dial provably moves the reply toward its OWN pole, attributably -- not any perturbation of that size,
and not just "the wording changed" (usable_max, gated on beating a matched-norm SHUFFLED-direction null at
the identical dose -- the same null parliament.py's shuffled-dial-null arm uses). The output is a RANGE
plus that derail point, never a recommended single setting.

THE EFFECT MEASURE -- REWRITTEN, on a CONFIRMED real-run failure mode (stated loud, not buried). A prior
version of this module measured "effect" as word-type-Jaccard CHANGE vs baseline
(receipts.receipt_metrics(...)["changed"] / 100) -- "how much did the wording move", full stop. That
measure cannot tell a dial that genuinely produces its trait from a dial that merely REFORMATS the answer
(headers, bullet points, a worked example, a different opening line) while never actually moving toward its
own pole: on a real Qwen2.5-7B-Instruct nf4 run (research/runs/dial_autocalibrate.json), "skeptical" at dose
1.0 turned a plain TCP/UDP explainer into one with "### Key Differences" headers and a worked example --
genuinely different wording -- but not one sentence of actual doubt, hedging, or challenged claims, while
"warm" at the SAME dose produced "Hey there! I'd love to explain...", an actually warmer reply. The OLD
metric scored both "usable", identically; a reformat is not steering.

THE FIX -- direction, not magnitude. A dial IS a direction in the residual stream at its steering layer
(sc.vecs[dial]: the UNIT diff-of-means of its pos/neg poles, computed once by SteeringControl.compute /
add_custom). So instead of asking "how different is the wording", this module now asks "did the reply's
OWN representation move further toward that direction" -- a white-box projection, no LLM judge, one extra
forward pass per reply (no generation):

  directional_alignment(sc, reply_text, dial) -- encode `reply_text` RAW (no chat template, no added
      generation prompt: the tokenizer sees exactly the reply's own tokens and nothing else -- see the
      function's own docstring for why raw rather than chat-wrapped), one forward pass
      (output_hidden_states=True), MEAN-POOL the layer-`sc.layer` hidden state over every token position
      (hidden_states[sc.layer + 1], the same layer/indexing convention _last_resid already uses elsewhere
      in this codebase), then take the SIGNED scalar projection of that pooled vector onto
      unit(sc.vecs[dial]). Higher = further toward the dial's POSITIVE pole. This is the piece that
      changed; HOW the direction itself is computed (SteeringControl.compute/add_custom) is untouched.

  directional_effect(sc, dial, baseline_texts, steered_texts) -- THE new `effect` / `shuffled_effect` curve
      field: mean over the prompt sample of [directional_alignment(steered) - directional_alignment(that
      SAME prompt's own dose-0 baseline reply)]. Positive = the steered reply sits further toward the pole
      than that prompt's own unsteered reply did; ~0 = no net directional movement (what a mere-reformat
      dose should show, however much its wording changed); negative = moved toward the OPPOSITE pole -- a
      real, reportable finding, never clamped away (see _compute_calibration, which correctly never credits
      a negative number as "real": this sweep only ever engages the positive-pole direction of a dose).

THE OLD MEASURE IS KEPT, DEMOTED TO A DIAGNOSTIC (`change_magnitude`, still computed by
effect_vs_baseline/receipts.receipt_metrics, unchanged) precisely so a reader can see change-vs-direction
side by side in the same curve row -- a dose with a big change_magnitude and a near-zero effect IS the
reformat-not-steering signature this fix exists to catch, and the JSON/console output should make that
legible, not hide it. change_magnitude no longer feeds usable_max / dead_below / derail_point; only the
direction-aware `effect` does.

CAVEATS ON THE NEW MEASURE (Law #6, stated loud, not hidden -- this is a DIFFERENT metric with ITS OWN
honest limitations, not a solved problem):
  * MEAN-POOL is a CHOICE, not a law: pooling over every token treats a reply's opening line and its last
    clause as equally informative about "did this move toward the pole". A trait that shows up in only
    part of a long reply (one warm closing line on an otherwise neutral answer) gets diluted by the tokens
    around it. A last-token or an attention-weighted pool could disagree with this one on some replies.
  * SINGLE LAYER: only sc.layer (the SAME mid layer steering itself pushes on) is read. A trait could be
    more legible earlier or later in the stack; this module does not sweep layers to check.
  * SELF-REFERENTIAL: the ruler is made of the same stuff as what it measures -- sc.vecs[dial] is the
    diff-of-means of the SAME pos/neg instruction pair used to steer, so a reply that merely echoes the
    pole's own vocabulary (e.g. says "I doubt this claim" without actually scrutinizing anything) could
    still project toward the direction for surface-lexical reasons correlated with, but not identical to,
    genuinely exhibiting the trait. This is a narrower, subtler version of the OLD metric's gameability, not
    an elimination of it -- sample_replies are kept for exactly this reason: a human should still eyeball
    any dose flagged "usable".
  * ONE MODEL, ONE RUN: like every other number in this file, a raw projection value carries no meaning
    transferred to a different model, layer, or quantization -- see _EFFECT_EPS below and "SINGLE MODEL
    ONLY" further down.

THE SHUFFLED NULL: for each dial, ONE random UNIT direction is drawn once (make_shuffle_unit_vector, seeded
deterministically from (--seed, dial name) via _dial_seed -- pure integer arithmetic, not Python's
process-randomized hash()) and reused at EVERY dose of that dial's sweep, written at the SAME magnitude as
the real direction at that dose -- UNCHANGED. What changed is how the null is SCORED: its directional_effect
is directional_alignment(shuffled_steered_reply) - directional_alignment(baseline), projected onto THAT SAME
REAL dial's unit(sc.vecs[dial]) -- never onto the random direction itself. The question the null answers is
"does a random push of this size happen to move the reply along THIS dial's own axis" -- projecting onto
the random direction's own axis would answer a different, uninteresting question (a random direction
trivially aligns with itself). A real, working dial should clear this null comfortably; a random direction
of matched magnitude should not, and should usually score close to zero.

THRESHOLDS ARE CHOSEN CUTS, NOT DERIVED (stated loud, not hidden): _DEGEN_THRESHOLD=0.34 is UNCHANGED (a
dose is flagged derailing once MORE than ~1-in-3 sampled prompts come back degenerate -- copied from
parliament.py's _DEGEN_OK; coherence has nothing to do with the effect measure and needed no re-tuning).
_EFFECT_EPS changed VALUE AND UNITS along with the effect measure: it used to be 0.03 on the old metric's
native [0,1] Jaccard-distance scale (3% of the full dynamic range). The new metric's native scale is a raw
dot product -- a projection of a mean-pooled residual vector onto a UNIT direction at sc.layer, unbounded,
and NOT comparable across models/layers/quantizations the way a 0-1 ratio incidentally was. _EFFECT_EPS is
now 2.0, PICKED (not derived) from this exact rig's own most recent real run against Qwen2.5-7B-Instruct
nf4 at layer 14 (research/runs/dial_autocalibrate.json's recorded steer_info.resid_norm = 68.7 -- the
average norm of a SINGLE token's layer-14 residual over the contrastive seed prompts; hidden_size for this
model is 3584, confirmed from its config.json, so 28 layers -> mid-layer 14 matches SteeringControl's own
comment). A generic, uncorrelated vector of that norm projected onto an arbitrary fixed unit direction in a
3584-dim space would be expected to land, VERY roughly, around resid_norm / sqrt(hidden_size) =~
68.7 / 59.9 =~ 1.15 -- the textbook scaling for an unrelated high-dimensional vector against a fixed axis, a
rough sanity check and NOT a rigorous bound (real activations are anisotropic, not isotropic, so this could
be off in either direction). 2.0 sits comfortably above that rough noise-floor estimate, and well below the
scale of an obvious, human-legible shift (this run's warm-dial sample_replies visibly change register by
frac 1.0, under a raw steering push of base=58.42 per unit strength). Like _DEGEN_THRESHOLD, this is an
eyeballed cut a different, equally defensible choice could move -- a module constant, not learned or fit --
but it carries a SHARPER caveat than the old dimensionless epsilon did: because it is expressed in this
model/layer's own raw units rather than a portable 0-1 ratio, re-running this rig against a very
differently-scaled model, quantization, or steering layer should come with re-EYEBALLING this constant
against THAT run's own curve (not just trusting 2.0 to still mean "small" there) -- Law #6 applies to this
number harder than it did to its predecessor.

THE PROMPT SAMPLE SHAPES THE RESULT (stated loud): by default this pulls the N most recent DISTINCT user
turns from runlog (runlog.list_runs + runlog.get_run -- real, unmodified text, on the theory that "does this
dial hold up on THIS user's real prompts" is the actual question), falling back to a small built-in neutral
set (NEUTRAL_PROMPTS) when the runlog is empty (a fresh install, or a machine with nothing logged yet).
Every run records which source was actually used (prompt_source) and the literal prompts (prompts) -- a
different sample, or a different day's runlog, can shift the calibrated range; this is a calibration against
A sample, not THE truth.

SINGLE MODEL ONLY: every number in one run's JSON is specific to (this checkpoint, this quantization, this
prompt sample, this layer). Nothing here is portable to a different model size -- that is the entire point
(Law #6) and also the whole reason no single number here should ever be read as a universal constant.

GREEDY DECODING throughout (matching receipts.py/counterfactual.py's own convention): a diff between two
arms must be attributable to the dose/direction change, not to sampling dice.

Cost: O(n_dials x n_doses x n_prompts x ~2) greedy generations (~2 = one real-direction decode + one
shuffled-direction decode per prompt per nonzero dose; dose 0 needs only 1, shared as everyone's baseline),
PLUS one cheap forward pass (no generation, output_hidden_states only) per reply for directional_alignment
-- negligible next to a 100-token generation (order a single decode STEP's worth of compute each). Default
config (12 dials x 7 doses x 6 prompts) is a few hundred short greedy decodes -- --smoke (1-2 dials, 3
doses, 2 prompts) proves the wiring in a handful of decodes first, and is NOT a finding.

Run (CUDA venv):
    PY=C:/Users/brigi/src/cloze/.venv/Scripts/python.exe
    $PY research/dial_autocalibrate.py --model Qwen/Qwen2.5-7B-Instruct --out research/runs/dial_autocalibrate.json
Smoke first (prove the wiring cheaply -- NOT a finding):
    $PY research/dial_autocalibrate.py --smoke --out research/runs/dial_autocalibrate_smoke.json
A subset of dials:
    $PY research/dial_autocalibrate.py --dials warm candid concrete --n-prompts 10

--library MODE: sweep an entire CANDIDATE LIBRARY (research/dial_library_candidates.json's ~70-dial,
15-category {"dials":[{name,category,pos,neg,predict}]} format) instead of steering.AXES/--dials -- every
entry registered as a custom dial (register_library_dials) and swept through the IDENTICAL calibrate_dial
path as everything else in this file. Checkpoint-saved after EVERY dial (see run_library's docstring), so a
kill/OOM partway through this much bigger run keeps every dial finished so far:
    $PY research/dial_autocalibrate.py --library research/dial_library_candidates.json --out research/runs/dial_library_sweep.json
--report MODE: pure analysis, NO model/GPU (still imports torch/transformers at module level like every mode
here, but loads no model and touches no CUDA) -- reads a completed --library sweep JSON and prints the
per-category summary, the surface-vs-cognitive hypothesis verdict, and the curated shippable list, and writes
that curated list to --curated-out (default research/runs/dial_library_curated.json):
    $PY research/dial_autocalibrate.py --report research/runs/dial_library_sweep.json
"""
from __future__ import annotations

import argparse, gc, json, os, sys, time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root (clozn/ package)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import clozn.behavior.steering.axes as steering_mod
from clozn.behavior.steering import SteeringControl
from clozn.replay.counterfactual import _coherence   # {"degenerate": bool, "reason": str} -- the mandatory coherence axis
from clozn import receipts                          # receipt_metrics -- the word-type-Jaccard "%changed" DIAGNOSTIC
import clozn.runs.store as runlog

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================================================ helpers
def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def degenerate_rate(texts: list[str]) -> float:
    return round(_mean(_coherence(t)["degenerate"] for t in texts), 3) if texts else 0.0


def effect_vs_baseline(baseline_texts: list[str], steered_texts: list[str]) -> float:
    """THE OLD effect measure -- KEPT, but DEMOTED to a reported diagnostic field (`change_magnitude`) as of
    the direction-aware rewrite (see the module docstring's THE EFFECT MEASURE section): 1 minus the
    word-type-Jaccard-similarity between a steered reply and that SAME prompt's dose-0 (unsteered) reply --
    concretely, receipts.receipt_metrics(baseline, steered)["changed"] / 100, averaged over the prompt
    sample. This is deliberately the SAME pure-counting "%-of-wording-changed" metric run.js's delta strip /
    receipts.py / counterfactual.py already compute for every other receipt in this codebase, so "how much
    did this dose move the text, lexically" is never a second, silently-different definition of that
    question.

    Computed PER PROMPT (steered_texts[i] vs baseline_texts[i] -- that SAME prompt's own dose-0 reply,
    never a cross-prompt comparison) and averaged over the sample. 0.0 = identical word types (no
    detectable change); 1.0 = completely disjoint vocabularies.

    NO LONGER DRIVES usable_max / dead_below / derail_point (directional_effect does -- see the module
    docstring): this function's own CONFIRMED failure mode on a real run is EXACTLY why -- a dial that only
    reformats the answer (new headers, a different opening line, a worked example) scores a large "changed"
    number here with zero genuine movement toward its pole, and a degenerate reply (repetition loop /
    script-switch) very often ALSO scores large here despite being worthless. Kept and reported
    (`change_magnitude`) specifically so a reader can see change-vs-direction side by side in the same curve
    row -- a big change_magnitude next to a near-zero effect IS the reformat-not-steering signature this
    module was rewritten to catch, and hiding that comparison would be less honest than showing it."""
    if not baseline_texts or not steered_texts or len(baseline_texts) != len(steered_texts):
        return 0.0
    vals = [receipts.receipt_metrics(b, s)["changed"] / 100.0 for b, s in zip(baseline_texts, steered_texts)]
    return round(_mean(vals), 4)


def _project_onto_unit(vec: torch.Tensor, direction: torch.Tensor) -> float:
    """Pure tensor math, no model, no I/O: the SIGNED scalar projection of `vec` onto `direction`,
    normalizing `direction` to a unit vector first regardless of whether it already is one (sc.vecs[dial]
    entries already ARE unit per SteeringControl.compute/add_custom, but this makes the projection correct
    even if handed a raw, non-unit vector -- defensive, not load-bearing at the current call site).
    Unit-testable directly on fabricated CPU tensors, exactly as make_shuffle_unit_vector already is."""
    unit = direction / (direction.norm() + 1e-8)
    return float(torch.dot(vec.float(), unit.float().to(vec.device)))


@torch.no_grad()
def directional_alignment(sc, reply_text: str, dial: str) -> float:
    """White-box directional-alignment score for ONE generated reply against ONE dial's own axis -- the
    primitive the new effect measure (directional_effect, below) is built from: does this reply's own
    representation sit further toward the dial's pole, not just "does it look different" (see the module
    docstring's THE EFFECT MEASURE section for why this replaced the old word-Jaccard measure).

    ENCODING CHOICE (documented, not accidental): `reply_text` is tokenized RAW -- sc.tok(reply_text), with
    NO chat template wrapped around it and no added generation prompt -- so every token fed to the model is
    reply CONTENT, nothing else. Two reasons, not one: (1) a chat-wrapped encoding (role="user", say) would
    introduce role/special-token scaffolding (<|im_start|>user, <|im_end|>, ...) that a mean-pool would need
    to explicitly carve back out to avoid diluting the signal with template tokens, and getting that
    carve-out wrong would silently leak scaffolding into the score; (2) a reply is never itself a "user
    turn" -- it is the model's OWN assistant output, generated as raw continuation tokens with no wrapper of
    its own -- so feeding it back in exactly that raw form, rather than re-framing the model's own words as
    if a user had said them, is the less distorting choice. The cost (an honest limitation, not hidden):
    this reads the model in a slightly different "mode" than the chat-templated, last-token instruction
    contrasts that compute sc.vecs[dial] itself (SingleTurnSteer._last_resid / SteeringControl._last_resid)
    -- an apples-to-oranges wrinkle inherent to comparing a direction derived from INSTRUCTIONS against
    content read from a finished REPLY. One forward pass, no generation.

    POOLING CHOICE (documented, not accidental): MEAN over every token position of hidden_states[sc.layer +
    1] (the output of decoder block sc.layer -- identical indexing convention to every other _last_resid in
    this codebase), not just the last token. A reply is being read as a finished utterance to locate in
    activation space, not as a not-yet-answered prompt whose LAST token is about to decide the next one
    (that is _last_resid's own, different, job: computing a direction FROM instruction contexts, never a
    question this function needs to answer). Mean-pooling treats the reply's opening line and its last
    clause as equally informative -- a trait concentrated in only part of a long reply is diluted by the
    tokens around it; a different pooling choice (last-token, attention-weighted) could disagree with this
    one on some replies. See the module docstring's CAVEATS for this and the metric's other limitations.

    Empty text / zero tokens -> 0.0 (nothing to project). Returns the SIGNED scalar projection of the
    pooled residual onto unit(sc.vecs[dial]) via _project_onto_unit -- a raw dot product, NOT centered or
    scaled against anything of its own; callers always compare it against the SAME prompt's own
    baseline-reply alignment (directional_effect = align(steered) - align(baseline)), never read as an
    absolute in isolation."""
    ids = sc.tok(reply_text or "", return_tensors="pt").input_ids.to(DEV)
    if ids.shape[1] == 0:
        return 0.0
    hs = sc.model(ids, output_hidden_states=True).hidden_states[sc.layer + 1]
    pooled = hs[0].float().mean(dim=0)          # [H] -- mean over every (content) token position
    return _project_onto_unit(pooled, sc.vecs[dial])


def directional_effect(sc, dial: str, baseline_texts: list[str], steered_texts: list[str]) -> float:
    """THE new effect measure (replaces the old word-Jaccard change_magnitude as what usable_max/dead_below
    are actually gated on -- see the module docstring's THE EFFECT MEASURE section): mean over the prompt
    sample of [directional_alignment(steered) - directional_alignment(baseline)], each alignment a
    projection onto the dial's OWN unit direction (sc.vecs[dial]).

    Computed PER PROMPT (steered_texts[i] against that SAME prompt's own baseline_texts[i], never a
    cross-prompt comparison) and averaged over the sample -- the identical pairing discipline
    effect_vs_baseline already used. Positive = the steered reply sits further toward the dial's positive
    pole than that prompt's own unsteered reply did; ~0 = no net movement along the dial's axis (what a
    mere-REFORMAT dose should show, however much its wording changed); negative = moved toward the dial's
    NEGATIVE pole -- also a real, reportable finding, never clamped away (a caller comparing against
    _EFFECT_EPS with a plain `>` naturally excludes it from "real", which is correct: this sweep only ever
    engages the POSITIVE-pole direction of a dose -- see calibrate_dial).

    Called with the IDENTICAL `dial` name for both the real-direction arm and the shuffled-direction null --
    only `steered_texts` differs between the two calls a sweep makes per dose. That is deliberate, not an
    oversight: see the module docstring's THE SHUFFLED NULL section for why the null's replies are
    projected onto the SAME real dial axis rather than onto the random direction's own axis.

    Zero-cost note: each call independently re-encodes baseline_texts (no cross-call cache) -- a handful of
    redundant single-forward-pass encodes per dial across a full sweep, negligible next to the ~2 full
    100-new-token GENERATIONS this module already spends per prompt per nonzero dose; traded for keeping
    this a single, simple, directly-testable function (no cache-invalidation surface of its own)."""
    if not baseline_texts or not steered_texts or len(baseline_texts) != len(steered_texts):
        return 0.0
    vals = [directional_alignment(sc, s, dial) - directional_alignment(sc, b, dial)
            for b, s in zip(baseline_texts, steered_texts)]
    return round(_mean(vals), 4)


# nf4 for anything that won't fit bf16 comfortably on the 16GB card -- copied verbatim from
# parliament.py/mirror_bench.py (not imported): this codebase's own precedent is that each experiment
# script owns its small model-loading helpers rather than importing a sibling script.
_SMALL = ("0.5b", "1.5b", "-1b", "1b-", "2b", "3b", "-1.7b")
def wants_four_bit(name: str, override: str) -> bool:
    if override == "yes":
        return True
    if override == "no":
        return False
    return not any(s in name.lower() for s in _SMALL)


def axis_max_of(sc, name: str) -> float:
    """Per-dial calibrated ceiling -- sc.custom's own "max" FIRST if `name` has been explicitly registered
    as a custom dial (via add_custom), else steering.AXES' own "max" for a plain built-in, else
    SteeringControl.set's own default (1.5) if neither declares one.

    PRECEDENCE, AND WHY IT CHANGED FROM parliament.py's IDENTICAL-LOOKING HELPER: this used to check AXES
    first (parliament.py's own axis_max_of, copied verbatim, still does -- untouched, out of scope here).
    That was harmless there because parliament.py never registers a custom dial under a name that ALSO
    exists in steering.AXES. This module's --library mode can: the candidate library (research/
    dial_library_candidates.json) independently invented dial names like "warm"/"concise"/"formal"/
    "playful"/"poetic"/"concrete"/"confident" that happen to collide with steering.AXES' own built-in keys.
    add_custom already OVERWRITES sc.vecs[name] on such a collision (the library's own pos/neg direction
    wins, unconditionally -- see SteeringControl.add_custom), so a library dial's DIRECTION is already the
    library's, not the built-in's; axis_max_of checking AXES first would silently give that SAME dial the
    built-in's max (often 1.5, since most built-ins don't declare one) instead of the library's registered
    ceiling (_LIBRARY_DEFAULT_MAX) -- a real, silent mismatch between which pos/neg pair defines the swept
    axis and which ceiling bounds the sweep of it. Checking sc.custom first keeps both in lock-step:
    whichever definition is actually live in sc.vecs[name] is also the one whose max is read. Safe for every
    OTHER existing call site: a name only ever lives in ONE of {sc.custom, steering.AXES} everywhere else in
    this codebase (see test_axis_max_of_builtin_caps / test_axis_max_of_custom_axis, both still passing
    unchanged), so this precedence flip only ever changes behavior on the new collision case (see
    test_axis_max_of_custom_overrides_builtin_on_name_collision)."""
    return (sc.custom.get(name) or steering_mod.AXES.get(name) or {}).get("max", 1.5)


def _dial_seed(base_seed: int, name: str) -> int:
    """Deterministic per-(run-seed, dial-name) integer seed for the shuffled-direction generator -- pure
    integer arithmetic over the dial name's character codes, NOT Python's hash() (string hashing is
    process-randomized unless PYTHONHASHSEED is pinned, which would silently break --seed reproducibility).
    Generalizes parliament.py's _axis_seed (which indexes into a fixed 5-item STANCES list) to an arbitrary
    dial-name string, since this module's dial set is open-ended (--dials). Position-weighted so an
    anagram-like pair of names doesn't collide."""
    name_val = sum((i + 1) * ord(c) for i, c in enumerate(name))
    return (int(base_seed) * 1_000_003 + name_val * 97 + 13) & 0xFFFFFFFF


def make_shuffle_unit_vector(ref: torch.Tensor, seed: int) -> torch.Tensor:
    """A fresh random UNIT direction with the same shape/device/dtype as `ref`, seeded reproducibly on CPU
    (so the same --seed gives the same shuffled directions regardless of CUDA's own RNG state). Pure tensor
    math -- no model -- so this is unit-testable on any CPU tensor. Copied verbatim from parliament.py."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFF)
    v = torch.randn(ref.shape, generator=gen).to(ref.device, ref.dtype)
    return v / (v.norm() + 1e-8)


def _free_cuda():
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()


# ============================================================================================ prompt sample
# Neutral fallback -- deliberately original text, disjoint from the steering SEED_PROMPTS (used only to
# compute the diff-of-means directions themselves) and from parliament.py's CALIB_PROBES, so evaluating on
# it is never circular with either.
NEUTRAL_PROMPTS = [
    "What's a good way to spend a rainy afternoon?",
    "Can you help me plan a small dinner party?",
    "I'm not sure what to do about a noisy neighbor.",
    "What should I keep in mind before starting a garden?",
    "Tell me about a topic you find interesting.",
    "I'm nervous about an upcoming presentation at work.",
    "What's the best way to organize a messy closet?",
    "How do I get better at sticking to a morning routine?",
    "My phone battery drains really fast lately, any ideas?",
    "What are some good conversation starters for a first date?",
]


def sample_prompts(n: int, seed: int = 0) -> tuple[list[str], str]:
    """(prompts, source) -- source is "runlog" or "neutral-fallback". Pulls the n most recent DISTINCT
    user turns from runlog (runlog.list_runs() for recency-ordered ids, runlog.get_run() for the full,
    un-truncated `messages` -- list_runs()'s own prompt_summary is clipped to 90 chars, too short to trust
    as an actual generation prompt), else NEUTRAL_PROMPTS[:n] when the runlog is empty/unavailable (a fresh
    install, or a machine whose ~/.clozn/runs has nothing logged yet). `seed` is accepted for interface
    symmetry with this module's other seeded calls but unused: sampling takes "the N most recent", not a
    random draw, so there is nothing to seed here -- kept so callers never need a special case.

    Deliberately real, unmodified user text: the entire point of calibrating against "the user's real
    prompts" is to catch a dial that behaves differently on this user's actual distribution than on a bank
    of neutral test sentences -- see the module docstring's "the prompt sample shapes the result" caveat.
    """
    del seed  # unused -- see docstring
    try:
        rows = runlog.list_runs(limit=max(200, n * 20))   # already newest-first
    except Exception:
        rows = []
    seen: set = set()
    prompts: list[str] = []
    for row in rows:
        rid = row.get("id") if isinstance(row, dict) else None
        if not rid:
            continue
        rec = runlog.get_run(rid)
        if not rec:
            continue
        msgs = rec.get("messages") or []
        user_text = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        user_text = (user_text or "").strip()
        if not user_text or user_text in seen:
            continue
        seen.add(user_text)
        prompts.append(user_text)
        if len(prompts) >= n:
            break
    if prompts:
        return prompts, "runlog"
    return list(NEUTRAL_PROMPTS[:n]), "neutral-fallback"


# =========================================================================== the backbone + dial machinery
class SingleTurnSteer(SteeringControl):
    """SteeringControl, but every contrast prompt used to COMPUTE a direction is folded into a single USER
    turn (no system role). Copied from parliament.py's class of the same name: some chat templates (Gemma-2)
    reject a system role outright, and using the identical single-user-turn recipe UNCONDITIONALLY (not
    just when --model happens to be one of those) keeps the direction-computation recipe identical no
    matter which checkpoint --model points at. compute()/add_custom() are inherited unchanged from
    SteeringControl and call this override polymorphically -- nothing in the adapter itself needs touching.
    """

    @torch.no_grad()
    def _last_resid(self, system: str, user: str) -> torch.Tensor:
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": f"{system}\n\n{user}"}],
            add_generation_prompt=True, return_tensors="pt").to(DEV)
        hs = self.model(ids, output_hidden_states=True).hidden_states[self.layer + 1]
        return hs[0, -1].float()


class Rig:
    """Loads one model. Local-cache-first path lookup and the nf4-vs-bf16 choice follow parliament.py's
    Rig (itself following steer_vs_prompt.py's), except four_bit uses this module's own wants_four_bit."""

    def __init__(self, name: str, four_bit_override: str = "auto"):
        path = os.path.join(os.path.expanduser("~"), "hf_models", name.split("/")[-1])
        path = path if os.path.isfile(os.path.join(path, "config.json")) else name
        self.four_bit = wants_four_bit(name, four_bit_override)
        print(f"[load] {name} ({'nf4' if self.four_bit else 'bf16'}, {DEV}) ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.four_bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb,
                                                              device_map={"": 0}).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV).eval()

    @torch.no_grad()
    def gen(self, user: str, max_new: int = 100, sample: bool = False, temperature: float = 0.9) -> str:
        """Single USER-turn only -- never a system role, matching SingleTurnSteer's own recipe uniformly.
        repetition_penalty/no_repeat_ngram_size tame steering-induced loops (steer_vs_prompt.py's/
        parliament.py's Rig.gen do the same)."""
        ids = self.tok.apply_chat_template([{"role": "user", "content": user}],
                                           add_generation_prompt=True, return_tensors="pt").to(DEV)
        kw = dict(max_new_tokens=max_new, repetition_penalty=1.3, no_repeat_ngram_size=3,
                  pad_token_id=self.tok.eos_token_id or 0)
        if sample:
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False)
        out = self.model.generate(ids, **kw)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def free(self):
        self.model = None
        self.tok = None


# ---- custom dial defs: parliament.py's skeptical/plain, copied verbatim (not imported -- this codebase's
# stated precedent for a small definition shared between sibling experiment scripts; see wants_four_bit
# above, copied the same way from parliament.py/mirror_bench.py). ----
_SKEPTICAL_POS = ("Respond with skeptical, critical scrutiny: question the claims involved, flag what is "
                  "unproven, uncertain, or unverified, and do not accept assertions at face value.")
_SKEPTICAL_NEG = ("Respond with complete trust and acceptance: take all claims at face value and do not "
                  "question or doubt anything.")
_PLAIN_POS = ("Respond in plain, unembellished language: state things simply and directly, with no "
              "metaphor, no rhetorical flourish, and no stylistic decoration.")
_PLAIN_NEG = ("Respond in a highly stylized, embellished, decorative way, full of rhetorical flourish, "
              "vivid metaphor, and elaborate language.")
_CUSTOM_DIAL_DEFS = {                          # name -> (pos, neg, max)
    "skeptical": (_SKEPTICAL_POS, _SKEPTICAL_NEG, 0.5),
    "plain": (_PLAIN_POS, _PLAIN_NEG, 0.5),
}

# Captured NOW, at import time -- compute_dials() mutates the module-global steering_mod.AXES later, and
# list(...) of a dict's keys copies them, so this stays the FULL original set regardless of later narrowing.
DEFAULT_DIALS = list(steering_mod.AXES) + list(_CUSTOM_DIAL_DEFS)


def compute_dials(sc, dial_names: list[str]) -> dict:
    """Compute every requested dial's direction on sc's backbone. Built-ins (steering.AXES keys) go through
    sc.compute() -- narrowed FIRST to just the requested built-in names (parliament.py's compute_stances
    trick) so forward passes aren't burned on axes nobody asked for. sc.compute() is ALWAYS called at least
    once, even when every requested dial is a non-built-in custom one, because it is what calibrates
    sc.base/sc.resid_norm (Law #6: per-model, not a fixed default) -- skipping it entirely would leave
    add_custom() silently reusing SteeringControl.__init__'s uncalibrated base=1.0. When no built-ins were
    requested, steering_mod.AXES is left UNNARROWED so compute() still has its full default set to
    calibrate against.

    Non-built-in names are registered via sc.add_custom() if recognized (currently: skeptical, plain --
    parliament.py's two custom stances); an unrecognized name is neither a steering.AXES key nor a known
    custom and is reported in info["unknown_dials"] rather than raising, so a typo in --dials degrades to
    an honest warning + a shorter dial list, not a crash mid-sweep.

    NOTE: mutates the process-global steering_mod.AXES (never restores it) -- a one-shot script, exactly
    matching parliament.py's own compute_stances. Callers that need the ORIGINAL full AXES afterward (e.g.
    tests sharing a process with other suites) must snapshot/restore it themselves."""
    builtin_req = [d for d in dial_names if d in steering_mod.AXES]
    custom_req = [d for d in dial_names if d not in steering_mod.AXES]
    if builtin_req:
        steering_mod.AXES = {k: v for k, v in steering_mod.AXES.items() if k in builtin_req}
    info = sc.compute()
    info["custom_axes"] = {}
    unknown = []
    for dname in custom_req:
        if dname in _CUSTOM_DIAL_DEFS:
            pos, neg, mx = _CUSTOM_DIAL_DEFS[dname]
            sc.add_custom(dname, pos, neg, mx=mx)
            info["custom_axes"][dname] = {"max": mx}
        else:
            unknown.append(dname)
    info["unknown_dials"] = unknown
    return info


# ======================================================================================= the candidate library
def load_dial_library(path: str) -> list[dict]:
    """Pure I/O + validation, NO model: load a candidate-dial library JSON in research/dial_library_
    candidates.json's own format ({"dials": [{"name", "category", "pos", "neg", "predict"}, ...]}) and
    return its "dials" list, in file order. Raises ValueError -- fast, before any model loads -- on a
    structurally broken file: not a {"dials": [...]} shape, an entry missing a required field, or two
    entries sharing the same `name` (register_library_dials registers each by name; a silent collision
    there would make the SECOND add_custom call silently overwrite the FIRST's direction under sc.vecs/
    sc.custom, and this module would then report calibration results for only one of the two intended
    dials without ever saying so -- caught here instead, loudly, before any GPU work starts)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    dials = data.get("dials") if isinstance(data, dict) else None
    if not isinstance(dials, list) or not dials:
        got = type(dials).__name__ if dials is not None else type(data).__name__
        raise ValueError(f"{path}: expected a top-level {{'dials': [...]}} non-empty list, got {got}")
    required = ("name", "category", "pos", "neg", "predict")
    for i, d in enumerate(dials):
        if not isinstance(d, dict):
            raise ValueError(f"{path}: dials[{i}] is not an object: {d!r}")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"{path}: dials[{i}] missing required field(s) {missing}: {d}")
    names = [d["name"] for d in dials]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"{path}: duplicate dial name(s) in library: {dupes}")
    return dials


_LIBRARY_DEFAULT_MAX = 1.5   # the SWEEP CEILING for library candidates (strength = frac*axis_max, frac up
                              # to 1.5 -> strength up to 2.25). Set to 1.5 (NOT add_custom's 0.5 default) so
                              # the library sweeps the SAME strength regime where the built-in tonal dials
                              # actually showed effect AND reached derail in the 12-dial run (warm/concise/etc.
                              # at max 1.5). A 0.5 ceiling caps exploration at strength 0.75 -- too weak to
                              # reach many dials' effect/derail regime, so it would silently under-dose and
                              # inflate false-"dead" verdicts. The coherence gate catches over-injection, so a
                              # wide ceiling is strictly more informative; the per-dial USABLE range the sweep
                              # discovers within [0, this] is still the output, this just sets how far it looks.


def register_library_dials(sc, library: list[dict]) -> dict:
    """Register EVERY dial spec in `library` (as returned by load_dial_library) as a custom dial on `sc`,
    via the exact same sc.add_custom(name, pos, neg, mx=...) recipe this module's own skeptical/plain
    customs and parliament.py's identical pattern already use -- just batch-driven from a big JSON file
    instead of a hand-written dict. Returns {name: {"category", "predict", "pos", "neg", "max",
    "shadows_builtin"}} in the library's own order -- list(the return value) becomes the sweep's dial_order,
    and every other field rides along so run_library can stamp it onto that dial's calibration report
    without re-reading the library file later (--report mode never needs the original library JSON at all).

    CALLER MUST have already called compute_dials/sc.compute() at least once before this (to calibrate
    sc.base/sc.resid_norm) -- add_custom computes a direction but never touches sc.base itself; see
    compute_dials's own docstring for why sc.compute() must run even when zero steering.AXES built-ins are
    requested.

    shadows_builtin=True flags a library dial whose name ALSO exists in steering.AXES (this library
    independently invented names like "warm"/"concise"/"formal"/"playful"/"poetic"/"concrete"/"confident"
    that collide with the built-in steering axes) -- add_custom overwrites sc.vecs[name] unconditionally on
    such a collision (the library's own pos/neg wins, not the built-in's), and axis_max_of's custom-first
    precedence (see its own docstring) keeps the swept ceiling consistent with that same override. Flagged,
    not prevented: a collision is not an error, just a fact worth a human noticing (run_library prints it)."""
    out = {}
    for spec in library:
        name = spec["name"]
        shadows_builtin = name in steering_mod.AXES
        sc.add_custom(name, spec["pos"], spec["neg"], mx=_LIBRARY_DEFAULT_MAX)
        out[name] = {"category": spec["category"], "predict": spec["predict"],
                     "pos": spec["pos"], "neg": spec["neg"],
                     "max": _LIBRARY_DEFAULT_MAX, "shadows_builtin": shadows_builtin}
    return out


# ================================================================================================ the sweep
_SWEEP_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
_SWEEP_FRACS_SMOKE = [0.0, 0.75, 1.5]

_DEGEN_THRESHOLD = 0.34   # a dose is flagged derailing once MORE than ~1-in-3 sampled prompts come back
                          # degenerate (counterfactual._coherence) -- copied from parliament.py's _DEGEN_OK.
                          # UNCHANGED by the direction-aware rewrite: coherence has nothing to do with the
                          # effect measure, so it needed no re-tuning.
_EFFECT_EPS = 2.0         # RE-TUNED for the new metric's scale (a raw projection delta, not a [0,1] Jaccard
                          # ratio -- full derivation in the module docstring's THRESHOLDS section). Picked,
                          # not derived, from this rig's own most recent real run against
                          # Qwen2.5-7B-Instruct nf4 at layer 14 (recorded steer_info.resid_norm = 68.7 in
                          # research/runs/dial_autocalibrate.json, hidden_size 3584): comfortably above a
                          # rough "uncorrelated vector" noise-floor estimate of resid_norm/sqrt(hidden_size)
                          # =~ 1.15, comfortably below the scale of an obvious tonal shift. An eyeballed,
                          # conservative cut like _DEGEN_THRESHOLD -- a module constant, not learned or fit
                          # -- but SCALE-TIED to this model/layer/quantization in a way the old dimensionless
                          # 0.03 never was: re-picking (not just re-deriving) this number is expected when
                          # this rig is run against a differently-scaled model/layer.


def _compute_calibration(curve: list[dict]) -> dict:
    """Pure function, no model, no I/O: derive derail_point / dead_below / usable_max / usable_range /
    range_valid from an already-generated `curve` (a list of {frac, real_degenerate_rate, effect,
    shuffled_effect, change_magnitude} dicts, one per swept dose, dose 0 included). Unit-testable directly
    against a fabricated curve -- this is where the "does the dial have a usable range" verdict actually
    lives; the GPU-touching calibrate_dial below only exists to produce `curve`'s numbers. UNCHANGED by the
    direction-aware rewrite: this function has always just compared `effect`/`shuffled_effect` numbers
    against thresholds, whatever those numbers mean -- it does not know or care that `effect` used to be a
    word-Jaccard ratio and is now a directional-alignment delta; `change_magnitude` (the old measure, kept
    as a diagnostic) is never read here.

    derail_point = the LOWEST frac (dose 0 included -- an unsteered model that is already degenerate on
                   this prompt sample is a real, if unlikely, finding, not a case to hide) whose
                   real_degenerate_rate exceeds _DEGEN_THRESHOLD. None if no swept dose derails.
    dead_below   = the LOWEST NONZERO frac whose effect exceeds _EFFECT_EPS. Dose 0 is excluded even though
                   its effect is always exactly 0.0 (it IS the baseline) -- "dead" here means "the dial
                   isn't doing anything yet", which is trivially true at 0 by construction, not a finding.
                   A NEGATIVE effect (the dial moved the reply toward its OPPOSITE pole -- possible now that
                   effect is a signed projection, never possible under the old [0,1]-bounded Jaccard
                   measure) also fails "> _EFFECT_EPS" and is correctly never picked here.
    usable_max   = the HIGHEST frac that is simultaneously: (a) coherent (real_degenerate_rate <=
                   _DEGEN_THRESHOLD), (b) has a real effect (> _EFFECT_EPS), AND (c) beats the shuffled
                   null AT THE SAME DOSE (effect > shuffled_effect) -- "the dial moved the reply toward its
                   own pole", not "any perturbation this size did", and not "the wording merely changed"
                   (change_magnitude plays no part in this test). None if no dose satisfies all three.
    usable_range = [dead_below, usable_max] (either or both may be None -- see range_valid).
    range_valid  = True iff both ends are present AND ordered (dead_below <= usable_max). A dial that never
                   clears the null/coherence bar at all, or only ever has an effect below _EFFECT_EPS --
                   INCLUDING a dial that changes the WORDING a great deal (a high change_magnitude) while
                   never moving toward its own pole, the reformat-not-steering case this module was
                   rewritten to catch -- reports range_valid=False. Read that as "no honestly-calibrated
                   usable range on this sample", never as a zero-width range at some arbitrary point.
    """
    derail_point = next((c["frac"] for c in curve if c["real_degenerate_rate"] > _DEGEN_THRESHOLD), None)
    dead_below = next((c["frac"] for c in curve if c["frac"] > 0 and c["effect"] > _EFFECT_EPS), None)
    usable_fracs = [c["frac"] for c in curve
                    if c["frac"] > 0
                    and c["real_degenerate_rate"] <= _DEGEN_THRESHOLD
                    and c["effect"] > _EFFECT_EPS
                    and c["effect"] > c["shuffled_effect"]]
    usable_max = max(usable_fracs) if usable_fracs else None
    range_valid = dead_below is not None and usable_max is not None and dead_below <= usable_max
    return {
        "derail_point": derail_point,
        "dead_below": dead_below,
        "usable_max": usable_max,
        "usable_range": [dead_below, usable_max],
        "range_valid": range_valid,
    }


def calibrate_dial(rig, sc, name: str, prompts: list[str], fracs: list[float], seed: int,
                    max_new: int = 100) -> dict:
    """The heart of this module: sweep dial `name` over `fracs` (each a fraction of axis_max_of(sc, name))
    on `prompts`, against a matched-norm SHUFFLED-direction null at the IDENTICAL magnitude, at every dose.

    STRENGTH IS WRITTEN DIRECTLY INTO sc.strength[...], NOT via SteeringControl.set() (deliberate, and
    worth stating loud): .set() clamps its argument into [-axis_max, axis_max], which is exactly the
    ceiling this sweep needs to go PAST (fracs run up to 1.5x axis_max, by design, to find where a dial
    derails BEYOND its documented "safe" max) -- going through .set() would silently clamp frac=1.0/1.25/1.5
    down to the SAME strength, making every dose past 1.0x indistinguishable and derail_point unreachable.
    The shuffled null needs the identical bypass for the identical reason (parliament.py's own null already
    does this, for the same reason, at its narrower fracs<=1.0 sweep).

    At frac=0.0: ONE greedy decode per prompt with no dial engaged is both "the real arm" and "the shuffled
    arm" (steering off is steering off, whichever vector isn't there) -- these become `baseline_texts`,
    reused as the fixed reference point for BOTH directional_effect and effect_vs_baseline at EVERY other
    dose (never a moving target).

    At each nonzero frac: real-direction decodes, then (after a full clear/disengage) shuffled-direction
    decodes, both at the SAME |strength| -- so directional_effect, effect_vs_baseline, and degenerate_rate
    are all computed on a like-for-like pair at every dose.

    Each curve row carries THREE effect-shaped numbers, deliberately, so a reader sees them together (see
    the module docstring's THE EFFECT MEASURE section):
      * effect             -- directional_effect(sc, name, baseline_texts, real_texts): the NEW,
                               direction-aware measure. Drives dead_below/usable_max/usable_range.
      * shuffled_effect    -- directional_effect(sc, name, baseline_texts, shuf_texts): the SAME projection,
                               applied to the shuffled-direction arm's replies, onto the SAME real dial axis.
      * change_magnitude   -- effect_vs_baseline(baseline_texts, real_texts): the OLD word-Jaccard measure,
                               kept as a diagnostic only (see effect_vs_baseline's own docstring) -- NOT
                               read by _compute_calibration.

    Returns {dial, axis_max, curve, derail_point, dead_below, usable_max, usable_range, range_valid,
    sample_replies} -- see _compute_calibration for exactly how the four calibration numbers are derived
    from `curve`, and the module docstring for the effect measure + thresholds' definitions and caveats.
    `sample_replies` keeps, per dose, the FIRST prompt's (prompt, baseline reply, steered reply) triple --
    enough for a human to eyeball whether a dose flagged "usable" is genuinely on-character, not a
    self-referential quirk of projecting onto the dial's own diff-of-means direction (see the module
    docstring's SELF-REFERENTIAL caveat).
    """
    axis_max = axis_max_of(sc, name)
    shuffle_vec = make_shuffle_unit_vector(sc.vecs[name], _dial_seed(seed, name))

    baseline_texts = None
    curve: list[dict] = []
    sample_replies: list[dict] = []
    for frac in fracs:
        strength = round(frac * axis_max, 4)
        sc.disengage()
        sc.clear()
        if frac == 0.0:
            real_texts = [rig.gen(p, max_new=max_new) for p in prompts]
            baseline_texts = real_texts
            shuf_texts = real_texts        # steering off either way at frac=0 -- identical by construction
        else:
            sc.strength[name] = strength   # direct write -- bypasses .set()'s clamp to axis_max (see above)
            sc.engage()
            real_texts = [rig.gen(p, max_new=max_new) for p in prompts]
            sc.disengage()
            sc.clear()

            sc.vecs["_shuf_tmp"] = shuffle_vec
            sc.strength["_shuf_tmp"] = strength    # the null must land at EXACTLY the real dial's magnitude
            sc.engage()
            shuf_texts = [rig.gen(p, max_new=max_new) for p in prompts]
            sc.disengage()
            sc.clear()
            del sc.vecs["_shuf_tmp"]

        curve.append({
            "frac": frac, "strength": strength,
            "real_degenerate_rate": degenerate_rate(real_texts),
            "shuffled_degenerate_rate": degenerate_rate(shuf_texts),
            "effect": directional_effect(sc, name, baseline_texts, real_texts),
            "shuffled_effect": directional_effect(sc, name, baseline_texts, shuf_texts),
            "change_magnitude": effect_vs_baseline(baseline_texts, real_texts),
        })
        sample_replies.append({
            "frac": frac, "prompt": prompts[0] if prompts else "",
            "baseline_reply": baseline_texts[0] if baseline_texts else "",
            "steered_reply": real_texts[0] if real_texts else "",
        })

    calib = _compute_calibration(curve)
    return {"dial": name, "axis_max": axis_max, "curve": curve, "sample_replies": sample_replies, **calib}


# ================================================================================================= run
def run(model_name: str, dials: list[str] | None = None, n_prompts: int = 6,
        out_path: str = "research/runs/dial_autocalibrate.json", four_bit_override: str = "auto",
        smoke: bool = False, seed: int = 0, layer: int | None = None, max_new: int = 100) -> dict:
    torch.manual_seed(seed)
    dial_names = list(dials) if dials else list(DEFAULT_DIALS)
    if smoke:
        dial_names = dial_names[:2]        # --smoke always caps to 1-2 dials, whatever was requested
    n_eff = 2 if smoke else n_prompts      # --smoke always uses 2 prompts
    fracs = _SWEEP_FRACS_SMOKE if smoke else _SWEEP_FRACS

    prompts, prompt_source = sample_prompts(n_eff, seed=seed)
    print(f"[prompts] {len(prompts)} prompt(s) from {prompt_source}", flush=True)

    rig = Rig(model_name, four_bit_override)
    sc = SingleTurnSteer(rig.model, rig.tok, layer=layer)
    print(f"[dials] computing {len(dial_names)} dial direction(s) at layer {sc.layer}: {dial_names}",
          flush=True)
    steer_info = compute_dials(sc, dial_names)
    print(f"[dials] {steer_info}", flush=True)
    if steer_info["unknown_dials"]:
        print(f"[warn] unknown dial(s) ignored (not a steering.AXES built-in or a registered custom dial "
              f"-- known customs: {sorted(_CUSTOM_DIAL_DEFS)}): {steer_info['unknown_dials']}", flush=True)
    dial_names = [d for d in dial_names if d in sc.vecs]
    if not dial_names:
        raise SystemExit("no valid dials left to calibrate after filtering unknown --dials names")

    res = {
        "model": model_name, "four_bit": rig.four_bit, "seed": seed, "smoke": smoke,
        "n_prompts": len(prompts), "prompt_source": prompt_source, "prompts": prompts,
        "max_new": max_new, "steer_layer": sc.layer, "steer_info": steer_info,
        "sweep_fracs": fracs, "degen_threshold": _DEGEN_THRESHOLD, "effect_eps": _EFFECT_EPS,
        "dial_order": dial_names, "dials": {},
    }
    _save(out_path, res)

    print(f"[sweep] {len(dial_names)} dial(s) x {len(fracs)} doses x {len(prompts)} prompts ...", flush=True)
    t0 = time.time()
    for name in dial_names:
        report = calibrate_dial(rig, sc, name, prompts, fracs, seed=seed, max_new=max_new)
        res["dials"][name] = report
        _save(out_path, res)
        print(f"  [{name}] usable_range={report['usable_range']} derail_point={report['derail_point']} "
              f"range_valid={report['range_valid']}", flush=True)
    res["wall_clock_sec"] = round(time.time() - t0, 1)

    sc.disengage()
    rig.free()
    del sc, rig
    _free_cuda()

    _summary(res)
    _save(out_path, res)
    print(f"\nsaved -> {out_path}", flush=True)
    return res


def _save(out_path, res):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def _summary(res):
    print("\n" + "=" * 78, flush=True)
    print(f"DIAL AUTO-CALIBRATION -- {res['model']} ({'nf4' if res['four_bit'] else 'bf16'}) -- "
          f"{res['n_prompts']} prompt(s) from {res['prompt_source']}", flush=True)
    print(f"\n{'dial':12} {'dead_below':11} {'usable_max':11} {'derail_at':10} {'valid':6}", flush=True)
    for name in res["dial_order"]:
        c = res["dials"][name]
        print(f"{name:12} {str(c['dead_below']):11} {str(c['usable_max']):11} "
              f"{str(c['derail_point']):10} {str(c['range_valid']):6}", flush=True)
    invalid = [n for n in res["dial_order"] if not res["dials"][n]["range_valid"]]
    if invalid:
        print(f"\nno honestly-calibrated usable range on this sample: {invalid}", flush=True)
    print("\nNOTE: this is a RANGE calibration, not a recommended single value -- see the module docstring.",
          flush=True)


# ========================================================================================= the library sweep
def run_library(library_path: str, model_name: str = "Qwen/Qwen2.5-7B-Instruct", n_prompts: int = 6,
                out_path: str = "research/runs/dial_library_sweep.json", four_bit_override: str = "auto",
                smoke: bool = False, seed: int = 0, layer: int | None = None, max_new: int = 100) -> dict:
    """--library mode: sweep an entire CANDIDATE LIBRARY (research/dial_library_candidates.json's ~70-dial,
    15-category format) through the IDENTICAL per-dial calibration path run() uses for steering.AXES/
    --dials -- same calibrate_dial, same _compute_calibration, same direction-aware effect measure, same
    shuffled null (see the module docstring for all of those). Two differences from run(): (1) every dial
    comes from add_custom via register_library_dials -- there is no steering.AXES built-in path here, only
    customs (even a library name that collides with a built-in -- see register_library_dials/axis_max_of);
    and (2) each dial's calibration report additionally carries its library `category`/`predict`/`pos`/
    `neg`, so --report can group/verdict/re-ship on them without ever re-reading the library file.

    sc.base/sc.resid_norm are calibrated via compute_dials(sc, []) BEFORE any library dial is registered --
    passing an empty dial_names list makes compute_dials leave steering_mod.AXES unnarrowed (so sc.compute()
    still has its full built-in set to average a resid_norm over, avoiding a divide-by-zero on an empty
    AXES -- see compute_dials's own docstring), which as a side effect also computes the 10 stock AXES
    directions into sc.vecs. Those 10 are never swept here (dial_order only ever lists library names) and
    the cost is negligible (~360 short forward passes, no generation) next to the ~70-dial sweep itself --
    an accepted, pre-existing cost this module already pays for ANY --dials invocation that requests zero
    built-ins (e.g. `--dials skeptical plain`), not something --library newly introduces.

    THIS IS A BIG RUN (~70 dials x 7 doses x n_prompts x ~2 generations -- an order of magnitude more than
    the default sweep), so, exactly like run(), the JSON at `out_path` is CHECKPOINT-SAVED after EVERY
    dial's calibration completes, not just at the end: a kill or OOM partway through still leaves every
    already-finished dial's full curve/calibration on disk, never just whatever was saved before the LAST
    dial that happened to be in flight when the process died."""
    torch.manual_seed(seed)
    library = load_dial_library(library_path)
    if smoke:
        library = library[:2]          # --smoke caps to the first 2 library entries, matching run()'s cap
    n_eff = 2 if smoke else n_prompts
    fracs = _SWEEP_FRACS_SMOKE if smoke else _SWEEP_FRACS

    prompts, prompt_source = sample_prompts(n_eff, seed=seed)
    print(f"[prompts] {len(prompts)} prompt(s) from {prompt_source}", flush=True)

    rig = Rig(model_name, four_bit_override)
    sc = SingleTurnSteer(rig.model, rig.tok, layer=layer)
    print(f"[library] {len(library)} candidate dial(s) loaded from {library_path}", flush=True)
    steer_info = compute_dials(sc, [])     # calibrates sc.base/resid_norm; no built-ins requested for sweep
    lib_meta = register_library_dials(sc, library)
    dial_names = list(lib_meta)
    print(f"[library] registered {len(dial_names)} custom dial(s) (max={_LIBRARY_DEFAULT_MAX} each)",
          flush=True)
    shadowed = sorted(n for n, m in lib_meta.items() if m["shadows_builtin"])
    if shadowed:
        print(f"[warn] {len(shadowed)} library dial name(s) shadow a steering.AXES built-in -- the "
              f"library's own pos/neg + max win (axis_max_of is custom-first): {shadowed}", flush=True)

    res = {
        "model": model_name, "four_bit": rig.four_bit, "seed": seed, "smoke": smoke,
        "n_prompts": len(prompts), "prompt_source": prompt_source, "prompts": prompts,
        "max_new": max_new, "steer_layer": sc.layer, "steer_info": steer_info,
        "library_path": library_path, "sweep_fracs": fracs,
        "degen_threshold": _DEGEN_THRESHOLD, "effect_eps": _EFFECT_EPS,
        "dial_order": dial_names, "dial_meta": lib_meta, "dials": {},
    }
    _save(out_path, res)

    print(f"[sweep] {len(dial_names)} dial(s) x {len(fracs)} doses x {len(prompts)} prompts ...", flush=True)
    t0 = time.time()
    for i, name in enumerate(dial_names):
        report_row = calibrate_dial(rig, sc, name, prompts, fracs, seed=seed, max_new=max_new)
        report_row["category"] = lib_meta[name]["category"]
        report_row["predict"] = lib_meta[name]["predict"]
        report_row["pos"] = lib_meta[name]["pos"]
        report_row["neg"] = lib_meta[name]["neg"]
        res["dials"][name] = report_row
        _save(out_path, res)      # checkpoint after EVERY dial -- a kill/OOM keeps every finished one
        print(f"  [{i + 1}/{len(dial_names)}] [{name}] ({report_row['category']}/{report_row['predict']}) "
              f"usable_range={report_row['usable_range']} derail_point={report_row['derail_point']} "
              f"range_valid={report_row['range_valid']}", flush=True)
    res["wall_clock_sec"] = round(time.time() - t0, 1)

    sc.disengage()
    rig.free()
    del sc, rig
    _free_cuda()

    _summary_by_category(res)
    _save(out_path, res)
    print(f"\nsaved -> {out_path}", flush=True)
    return res


def _summary_by_category(res):
    print("\n" + "=" * 78, flush=True)
    print(f"DIAL LIBRARY SWEEP -- {res['model']} ({'nf4' if res['four_bit'] else 'bf16'}) -- "
          f"{res['n_prompts']} prompt(s) from {res['prompt_source']} -- {len(res['dial_order'])} dial(s)",
          flush=True)
    by_cat: dict = {}
    for name in res["dial_order"]:
        cat = res["dials"][name].get("category", "?")
        by_cat.setdefault(cat, []).append(name)
    for cat in sorted(by_cat):
        names = by_cat[cat]
        print(f"\n-- {cat} ({len(names)}) --", flush=True)
        print(f"{'dial':20} {'predict':10} {'dead_below':11} {'usable_max':11} {'derail_at':10} {'valid':6}",
              flush=True)
        for name in names:
            c = res["dials"][name]
            print(f"{name:20} {c.get('predict', '?'):10} {str(c['dead_below']):11} "
                  f"{str(c['usable_max']):11} {str(c['derail_point']):10} {str(c['range_valid']):6}",
                  flush=True)
    n_valid = sum(1 for n in res["dial_order"] if res["dials"][n]["range_valid"])
    print(f"\n{n_valid}/{len(res['dial_order'])} dial(s) got a real, honestly-calibrated usable range.",
          flush=True)
    print("\nNOTE: this is a RANGE calibration, not a recommended single value -- see the module docstring.",
          flush=True)


# =========================================================================================== --report mode
def _dial_mean_effect(dial_report: dict) -> float:
    """Pure, no model: mean of `effect` over every NONZERO-frac row in one dial's curve -- dose 0 is the
    baseline itself (effect always exactly 0.0 there by construction), excluded here for the same reason
    _compute_calibration's dead_below/usable_max only ever read nonzero-frac rows. 0.0 for a dial with no
    curve, or only its dose-0 row (never raises on a missing/malformed field)."""
    curve = dial_report.get("curve") or []
    vals = [c["effect"] for c in curve if c.get("frac", 0) > 0 and "effect" in c]
    return round(_mean(vals), 4) if vals else 0.0


def category_summary(sweep: dict) -> dict:
    """Pure function, no model/GPU: per-CATEGORY {n_usable, n_total, usable_rate, mean_effect} from a
    completed --library sweep JSON's "dials" map -- categories are read off each dial's own "category" field
    (stamped there by run_library), never re-derived from the original library file, so --report only ever
    needs the sweep JSON. mean_effect is the mean, across the category's dials, of each dial's OWN mean
    nonzero-dose effect (_dial_mean_effect) -- a category with a few strongly-steering dials and a few dead
    ones still shows an honest middling average, not just a pass/fail count. Sorted by category name for a
    stable report order."""
    dials = sweep.get("dials", {})
    by_cat: dict = {}
    for d in dials.values():
        by_cat.setdefault(d.get("category", "?"), []).append(d)
    out = {}
    for cat, rows in sorted(by_cat.items()):
        n_total = len(rows)
        n_usable = sum(1 for r in rows if r.get("range_valid"))
        mean_effect = round(_mean(_dial_mean_effect(r) for r in rows), 4) if rows else 0.0
        out[cat] = {"n_usable": n_usable, "n_total": n_total,
                    "usable_rate": round(n_usable / n_total, 3) if n_total else 0.0,
                    "mean_effect": mean_effect}
    return out


def hypothesis_verdict(sweep: dict) -> dict:
    """Pure function, no model/GPU: the candidate library's OWN pre-registered hypothesis (dial_library_
    candidates.json's _about/_predict_legend -- SURFACE-expressed qualities steer well, ABSTRACT-COGNITIVE
    stances don't), checked against a completed --library sweep: the usable RATE (n range_valid=True / n
    total) for predict="surface" dials vs predict="cognitive" dials, plus the gap (surface - cognitive).
    predict="uncertain" dials are reported (their own n/usable/rate) but excluded from the gap -- the
    library's own docstring calls that tag out as "the interesting middle", a deliberate third bucket, not a
    second hypothesis to average in. hypothesis_holds is True iff BOTH rates are defined and surface's rate
    is strictly higher -- a plain directional check, not a significance test (n-per-category is small and
    this sweep draws one sample of prompts; read the gap's SIZE, not just this boolean)."""
    dials = sweep.get("dials", {})
    by_predict: dict = {}
    for d in dials.values():
        by_predict.setdefault(d.get("predict", "?"), []).append(bool(d.get("range_valid")))

    def _bucket(tag):
        rows = by_predict.get(tag, [])
        n_usable = sum(rows)
        rate = round(n_usable / len(rows), 3) if rows else None
        return {"n_usable": n_usable, "n_total": len(rows), "usable_rate": rate}

    surface, cognitive, uncertain = _bucket("surface"), _bucket("cognitive"), _bucket("uncertain")
    sr, cr = surface["usable_rate"], cognitive["usable_rate"]
    gap = round(sr - cr, 3) if sr is not None and cr is not None else None
    return {
        "surface": surface, "cognitive": cognitive, "uncertain": uncertain,
        "gap_surface_minus_cognitive": gap,
        "hypothesis_holds": bool(gap is not None and gap > 0),
    }


def curated_library(sweep: dict) -> list[dict]:
    """Pure function, no model/GPU: the SHIPPABLE subset of a completed --library sweep -- every dial with
    range_valid=True (an honestly-calibrated, coherent, null-beating usable range on this sample; see
    _compute_calibration), reduced to exactly what a deploy-time caller needs to re-register and cap the
    SAME dial on a live model: name, category, usable_range ([dead_below, usable_max]), derail_point, and
    the pos/neg poles add_custom needs to recompute the identical direction. A dial with range_valid=False
    is silently dropped here -- not an error, see category_summary/hypothesis_verdict for what happened to
    it instead. Sorted by (category, name) -- matches the console report's own category grouping, name as a
    stable tie-breaker."""
    dials = sweep.get("dials", {})
    out = []
    for name, d in dials.items():
        if not d.get("range_valid"):
            continue
        out.append({"name": name, "category": d.get("category", "?"),
                    "usable_range": d.get("usable_range"), "derail_point": d.get("derail_point"),
                    "pos": d.get("pos"), "neg": d.get("neg")})
    out.sort(key=lambda r: (r["category"], r["name"]))
    return out


def report(sweep_path: str, curated_out: str = "research/runs/dial_library_curated.json") -> dict:
    """--report mode: PURE ANALYSIS, NO model/GPU -- reads a completed --library sweep JSON (produced by
    run_library) and prints (a) the per-CATEGORY summary (category_summary), (b) the pre-registered
    surface-vs-cognitive HYPOTHESIS VERDICT (hypothesis_verdict), and (c) the CURATED shippable list
    (curated_library) -- then writes that same curated list to `curated_out` as {"dials": [...]}, the file
    research/clozn_server.py's dial-calibration curator step is meant to read from (only range_valid=True
    winners -- see curated_library). Returns {"category_summary", "hypothesis", "curated"} so a caller (or a
    test) can assert on the numbers directly, without re-parsing stdout."""
    with open(sweep_path, encoding="utf-8") as f:
        sweep = json.load(f)

    cat_summary = category_summary(sweep)
    hyp = hypothesis_verdict(sweep)
    curated = curated_library(sweep)

    print("\n" + "=" * 78, flush=True)
    print(f"DIAL LIBRARY REPORT -- {sweep.get('model', '?')} -- "
          f"{len(sweep.get('dial_order', []))} dial(s) swept from {sweep.get('library_path', '?')}",
          flush=True)

    print(f"\n{'category':22} {'n_usable/n_total':17} {'usable_rate':12} {'mean_effect':11}", flush=True)
    for cat, s in cat_summary.items():
        frac = f"{s['n_usable']}/{s['n_total']}"
        print(f"{cat:22} {frac:17} {s['usable_rate']:<12} {s['mean_effect']:<11}", flush=True)

    print("\nHYPOTHESIS VERDICT (surface-expressed dials predicted to usable-calibrate more often than "
          "cognitive-stance ones):", flush=True)
    surf, cog, unc = hyp["surface"], hyp["cognitive"], hyp["uncertain"]
    print(f"  surface:   {surf['n_usable']}/{surf['n_total']} usable  (rate={surf['usable_rate']})", flush=True)
    print(f"  cognitive: {cog['n_usable']}/{cog['n_total']} usable  (rate={cog['usable_rate']})", flush=True)
    print(f"  uncertain: {unc['n_usable']}/{unc['n_total']} usable  (rate={unc['usable_rate']}) -- reported, "
          f"not part of the gap (the library's own deliberate middle)", flush=True)
    print(f"  gap (surface - cognitive) = {hyp['gap_surface_minus_cognitive']} -- hypothesis "
          f"{'HOLDS' if hyp['hypothesis_holds'] else 'DOES NOT HOLD'} on this sweep", flush=True)

    print(f"\nCURATED SHIPPABLE LIST ({len(curated)} dial(s) with a real, honestly-calibrated usable range):",
          flush=True)
    last_cat = None
    for r in curated:
        if r["category"] != last_cat:
            print(f"\n-- {r['category']} --", flush=True)
            last_cat = r["category"]
        print(f"  {r['name']:20} usable_range={r['usable_range']} derail_point={r['derail_point']}",
              flush=True)

    os.makedirs(os.path.dirname(curated_out) or ".", exist_ok=True)
    with open(curated_out, "w", encoding="utf-8") as f:
        json.dump({"dials": curated}, f, indent=2, ensure_ascii=False)
    print(f"\nsaved curated library ({len(curated)} dial(s)) -> {curated_out}", flush=True)

    return {"category_summary": cat_summary, "hypothesis": hyp, "curated": curated}


def _default_out_path(library: str | None) -> str:
    """Pure, no I/O: --out's default depends on which mode is running -- the library sweep's own, much
    larger, output file (research/runs/dial_library_sweep.json) vs the plain --dials/built-in sweep's
    (research/runs/dial_autocalibrate.json). An explicit --out always overrides either default; this is
    only what --out defaults TO when the caller didn't pass one."""
    return "research/runs/dial_library_sweep.json" if library else "research/runs/dial_autocalibrate.json"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dials", nargs="+", default=None,
                    help="dial names to calibrate (default: every steering.AXES built-in + skeptical/plain)")
    ap.add_argument("--library", default=None, metavar="LIBRARY.json",
                    help="load a candidate-dial library (research/dial_library_candidates.json's "
                         "{'dials':[{name,category,pos,neg,predict}]} format) and sweep EVERY entry as a "
                         "custom dial, instead of steering.AXES/--dials -- checkpoint-saved after each dial")
    ap.add_argument("--report", default=None, metavar="SWEEP.json",
                    help="pure analysis, NO model/GPU: read a completed --library sweep JSON and print the "
                         "per-category summary + surface-vs-cognitive hypothesis verdict + curated winners "
                         "list (also written to --curated-out)")
    ap.add_argument("--curated-out", default="research/runs/dial_library_curated.json",
                    help="(--report only) where to write the curated, winners-only library JSON")
    ap.add_argument("--n-prompts", type=int, default=6,
                    help="recent runlog user turns to sample (else NEUTRAL_PROMPTS)")
    ap.add_argument("--out", default=None,
                    help="output path (default: dial_library_sweep.json for --library, else "
                         "dial_autocalibrate.json -- see _default_out_path)")
    ap.add_argument("--four-bit", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--layer", type=int, default=None, help="steering layer override (default num_layers//2)")
    ap.add_argument("--max-new", type=int, default=100, help="max new tokens per generation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true",
                    help="1-2 dials, 3 doses, 2 prompts -- prove the wiring cheaply, not a finding")
    return ap


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.report:
        report(args.report, curated_out=args.curated_out)
    elif args.library:
        run_library(args.library, model_name=args.model, n_prompts=args.n_prompts,
                    out_path=args.out or _default_out_path(args.library),
                    four_bit_override=args.four_bit, smoke=args.smoke, seed=args.seed, layer=args.layer,
                    max_new=args.max_new)
    else:
        run(args.model, dials=args.dials, n_prompts=args.n_prompts,
            out_path=args.out or _default_out_path(args.library),
            four_bit_override=args.four_bit, smoke=args.smoke, seed=args.seed, layer=args.layer,
            max_new=args.max_new)
