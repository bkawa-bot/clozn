"""quine.py -- Wild Experiment #9 (Wave 1): the quine test, forced-choice behavioral self-prediction.

Pre-registration: research/WILD_WAVE1_PREREG.md ("Exp 9 -- the quine test") + Amendment 1 (2026-07-05,
same doc), which PINS the metric this file implements: forced-choice behavioral self-prediction, chosen
over eliciting a calibrated next-token distribution from a 7-9B model (unreliable). The claim under test:
does giving a model a human-readable readout of its OWN current internal state let it predict its own
behavior better than a twin given nothing?

THE DESIGN, END TO END.
  1. STATE S. Steer the model with ONE randomly-chosen stance dial from parliament.py's set (candid /
     warm / skeptical / concrete / plain) at a random dose -- "random" per Amendment 1, but bounded to a
     fraction of that axis's own already-CALIBRATED per-axis ceiling (steering.AXES[...]["max"] / the
     custom skeptical+plain ceilings parliament.py established -- Law #6). This file does NOT re-run
     parliament.py's full per-axis liveness sweep (that would duplicate real GPU work already done and
     committed); it inherits those ceilings rather than re-deriving them -- a stated, bounded-scope choice,
     not an oversight. See _DOSE_FRAC_RANGE.
  2. GROUND TRUTH. A pair of hand-authored continuations to a prompt -- one state-CONGRUENT, one
     state-INCONGRUENT -- built directly from the SAME axis's pos/neg poles (CONT_PAIRS below): the
     congruent line is written in that axis's own "pos" tone, the incongruent line in its "neg" tone. So
     the ground-truth contrast is literally two samples of the exact poles the steering DIRECTION was
     itself computed from, not an unrelated stylistic invention. Ground truth = whichever continuation
     the STEERED model assigns higher mean-per-token teacher-forced logprob to, computed with NO
     meta-prompt (continuation_logprob) -- "its honest behavior" per Amendment 1. Mean (not sum) per
     token, so a longer continuation isn't unfairly penalized for having more tokens to be logprob'd over
     (both sum and mean are kept in the result so a reader can check which one drove a verdict). The SAME
     pair is ALSO scored unsteered (dose 0) as a baseline reference -- cheap (two extra forward passes),
     and it flags trials where steering didn't actually move the preference at all (a trial like that
     isn't really testing self-prediction of a STEERED state; reported as a diagnostic, not filtered out).
  3. SELF-PREDICTION. The SAME steered model (steering stays engaged -- there is no "self" to predict from
     otherwise) is asked a forced-choice A/B question: which continuation is IT more likely to produce?
     Three conditions, differing ONLY in what is prepended to that question (same A/B side draw reused
     across all conditions in a trial, so nothing else differs):
       - DIAL-LABEL   -- "You are currently in a state that makes you answer in a more <axis> way."
       - NO-STATE     -- the twin: no readout at all.
       - SHUFFLED-STATE -- the SAME sentence, but naming a DIFFERENT (randomly chosen) stance than the one
         actually engaged -- isolates "reading a self-description helps" from "any label helps".
     PLUS a bounded BEST-EFFORT bonus condition, SAE-FEATURE (see below), attempted for Qwen2.5-7B-Instruct
     only, and cleanly stubbed everywhere else.
  4. METRIC. Self-prediction ACCURACY per condition over N trials (default 30, --smoke -> 4): does the
     forced-choice pick match ground truth? Reads (Amendment 1's own): dial-label > no-state -> an
     explicit self-description measurably helps; dial-label ~= no-state -> it doesn't (a clean,
     Law-1-consistent NEGATIVE -- itself a finding, not a failed run); shuffled-state ~= dial-label -> the
     readout carried no self-specific signal (matched-anything-helps, not matched-truth-helps). A parse
     failure (the model doesn't emit a clean A/B) is tracked and EXCLUDED from the accuracy denominator,
     never padded as a guess (parliament.py's own parse_bits ethos) -- parse_fail_rate is reported beside
     every accuracy number so a high-failure run doesn't silently look confident.

SAE-FEATURE BONUS (best-effort, BOUNDED -- do not read this as a fully-realized fourth arm on both
families). Amendment 1's harder claim: can the model read its own top-k SAE features (ids + human labels)
to predict itself? This needs a LABELED SAE to be meaningful (bare feature ids are useless to the model,
per Amendment 1's own framing). On THIS machine: Qwen2.5-7B-Instruct already has both an SAE (andyrdt's
JumpReLU, ~/hf_models/andyrdt_l15_sae.pt, loaded via sae7b.GpuSAE -- an existing antecedent, imported
lazily, never modified) AND a Neuronpedia auto-interp label export already fetched and committed
(research/np_labels_l15.json, ~103k/131072 features labeled -- some auto-interp labels are themselves
noisy single-token fragments; that is Neuronpedia's own pipeline, not curated here). Gemma Scope
(google/gemma-scope-9b-it-res) is NOT cached on this machine, and fetching + relabeling a fresh SAE is
real, out-of-scope work for this bounded bonus arm -- so every non-Qwen2.5-7B-Instruct run gets a clean
{"status": "not_run", "reason": ...} stub (load_labeled_sae), never a silent skip. Even for Qwen, a
CAUSALITY guard applies: the SAE reads resid_post at its OWN fixed layer (15); if the steering layer is
LATER than that (sc.layer > 15, possible only via a --layer override -- the default 28-layer split is
steer@14 < SAE@15, which is fine), the SAE would read a residual computed BEFORE the steering hook ever
fired, so its "top features" would not reflect the steered state at all -- that run refuses the condition
rather than silently reporting a non-self-referential readout. No separate shuffled-SAE null is built (out
of scope for the bound); if sae_feature ever scores high, compare it against the EXISTING shuffled-state
null as a rough gut check, and note plainly that a dedicated shuffled-SAE-readout null would be the
cleaner isolation -- a stated follow-up, not a hidden gap.

ANTECEDENTS. wants_four_bit / Rig / SingleTurnSteer / compute_stances / the 5 STANCES / the skeptical+plain
custom pole text are COPIED from parliament.py (per instructions: reuse the patterns, never import from or
edit that file -- this codebase's own precedent anyway, stated in mirror_bench.py: each experiment script
owns its small helpers rather than importing a sibling script). steering.py's SteeringControl / AXES /
SEED_PROMPTS and counterfactual._coherence are imported directly (shared library modules, not sibling
experiments). The teacher-forced logit-span indexing convention in continuation_logprob (`start =
len(prompt_ids) - 1; logits[start:start+len(cont_ids)]`) mirrors phantom_kv.py's own
_prompt_logits_with_phantom / _teacher_target_logits (read for reference, not imported or modified) -- an
already-validated pattern in this codebase, not reinvented here.

GEMMA SAFETY. Gemma-2's chat template rejects a system role outright. Every prompt this file ever sends --
SingleTurnSteer's own contrastive-pole prompts, the ground-truth logprob prompts, the SAE resid-readout
prompt, and the forced-choice question itself (readout + question folded into ONE string) -- rides in a
single USER turn. There is no code path in this file that ever emits a system message.

CAVEATS, stated loud, not buried. (1) The congruent/incongruent pair necessarily differs in CONTENT as
well as tone (perfect minimal pairs are not achievable by hand-authoring) -- the unsteered baseline-
preference check above is the mitigation, not a fix: it tells a reader whether steering actually moved the
preference, but cannot fully separate "prefers this tone" from "prefers this text". (2) Dose is a random
fraction of an INHERITED ceiling, not a freshly-calibrated one -- see point 1 of the design above. (3)
Small N (default 30 trials over 5 axes -> ~6/axis): accuracy is reported with an approximate (Wald) SE,
not a tight interval; read the numbers as directional, not as decisive. (4) Forced-choice is a coarse
proxy for "predict your own next-token distribution" -- Amendment 1's own stated honest ceiling. (5) One
seed, greedy decoding throughout (the ONLY randomness is which axis/dose/prompt/pair/A-B-side gets drawn
per trial, via Python's own random.Random(seed) -- never torch sampling). (6) SAE bonus is Qwen-only, by
construction, as detailed above.

Run (CUDA venv), one model per process -- this experiment needs only ONE model per run (it predicts
itself; there is no separate judge, unlike parliament.py/mirror_bench.py):
    PY=C:/Users/brigi/src/cloze/.venv/Scripts/python.exe
    $PY research/quine.py --model Qwen/Qwen2.5-7B-Instruct --out research/runs/quine_qwen7b.json
    $PY research/quine.py --model google/gemma-2-9b-it     --out research/runs/quine_gemma9b.json
    $PY research/quine.py --compare research/runs/quine_qwen7b.json research/runs/quine_gemma9b.json
Smoke first (~4 trials -- proves the wiring, including the SAE-bonus path if available; NOT a finding):
    $PY research/quine.py --model Qwen/Qwen2.5-7B-Instruct --smoke --out research/runs/quine_smoke.json
"""
from __future__ import annotations

import argparse, gc, json, os, random, re, sys, time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering as steering_mod
from steering import SteeringControl
from counterfactual import _coherence   # {"degenerate": bool, "reason": str} -- the mandatory coherence axis

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# The 5 parliament stances, copied verbatim from parliament.py (STANCES) -- exp-4's set, exp-9's antecedent.
STANCES = ["candid", "warm", "skeptical", "concrete", "plain"]

# Custom-dial pole text, copied verbatim from parliament.py (_SKEPTICAL_POS/NEG, _PLAIN_POS/NEG) -- same
# wording so the axis directions this file computes are the same axes parliament.py itself measured.
_SKEPTICAL_POS = ("Respond with skeptical, critical scrutiny: question the claims involved, flag what is "
                  "unproven, uncertain, or unverified, and do not accept assertions at face value.")
_SKEPTICAL_NEG = ("Respond with complete trust and acceptance: take all claims at face value and do not "
                  "question or doubt anything.")
_PLAIN_POS = ("Respond in plain, unembellished language: state things simply and directly, with no "
              "metaphor, no rhetorical flourish, and no stylistic decoration.")
_PLAIN_NEG = ("Respond in a highly stylized, embellished, decorative way, full of rhetorical flourish, "
              "vivid metaphor, and elaborate language.")

# Random dose, BOUNDED to a fraction of each axis's own already-calibrated ceiling (axis_max_of) -- see
# the module docstring's point 1. Skewed toward the top half of the safe range (parliament.py's own
# calibration sweep tends to land its "chosen" operating dose near the top of what stays coherent).
_DOSE_FRAC_RANGE = (0.5, 1.0)

_FORCED_CHOICE_MAX_NEW = 8    # matches parliament.py's own _PREF_MAX_NEW -- a bare "A"/"B" needs very few tokens
DEFAULT_SAE_TOPK = 5

HERE = os.path.dirname(os.path.abspath(__file__))
SAE_PT_QWEN = os.path.join(os.path.expanduser("~"), "hf_models", "andyrdt_l15_sae.pt")
SAE_LABELS_QWEN = os.path.join(HERE, "np_labels_l15.json")


# =========================================================================== ground-truth prompt/pair bank
# Generic, content-light conversational openers -- distinct from steering.SEED_PROMPTS (used to compute the
# steering directions themselves) and parliament.CALIB_PROBES/QUESTION_BANK, so nothing here double-dips
# with a prompt already used to build or calibrate the dials this file drives.
PROMPTS = [
    "What do you think I should do about this situation?",
    "Can you give me your honest reaction to this?",
    "How would you respond if I brought this up with you?",
    "I wanted to get your take on something that happened today.",
    "What's going through your mind about this right now?",
    "Tell me what you'd actually say to me about this.",
]

# Congruent/incongruent continuation pairs, per axis -- each pair is two samples of that SAME axis's own
# pos-pole / neg-pole tone (see steering.AXES and the skeptical/plain poles above), hand-authored (not
# model-generated) so their OWN coherence is never in question -- the coherence axis in this file is
# reserved for the model's own generated forced-choice replies, where degeneration is actually possible.
# Deliberately CONTENT-AGNOSTIC (generic reaction openers, not tied to one PROMPTS entry) so any pair can
# plausibly follow any prompt above -- mirrors steering.py's own design (a tone axis is orthogonal to
# content, computed over varied neutral seeds).
CONT_PAIRS: dict[str, list[tuple[str, str]]] = {
    "candid": [
        ("Honestly, I have to push back on that -- I don't think it's right, and telling you the truth "
         "matters more to me than just agreeing.",
         "That's such a wonderful point, and I completely agree with everything you said -- you're "
         "absolutely right about all of it."),
        ("I'll be blunt: that plan has a real flaw in it, and glossing over it wouldn't do you any favors.",
         "You're totally right, no notes at all -- I wouldn't change a single thing about that."),
    ],
    "warm": [
        ("I really care about how you're doing, and I'm glad you told me -- take all the time you need.",
         "Noted. Proceeding with the requested information; no further comment is necessary."),
        ("That sounds like it's been really hard, and I want you to know I'm rooting for you every step "
         "of the way.",
         "Acknowledged. The relevant facts are as follows, presented without additional commentary."),
    ],
    "skeptical": [
        ("I'm not convinced that's actually true -- the evidence for that claim looks pretty thin to me.",
         "That sounds completely right, so I'll just take your word for it without questioning it further."),
        ("Before accepting that, I'd want to see it verified -- claims like that are often overstated.",
         "Sure, that must be correct -- there's no need to double check something like that."),
    ],
    "concrete": [
        ("For example, imagine you had exactly three apples and two oranges sitting on the table in "
         "front of you.",
         "In general terms, this relates to broader underlying structures at a fairly high level of "
         "abstraction."),
        ("Specifically, picture a five dollar bill changing hands between two named people at a "
         "farmer's market stall.",
         "Conceptually speaking, this is really just an instance of a more general abstract principle "
         "at work."),
    ],
    "plain": [
        ("Simply put: it works, plain and simple, with no need to dress it up any further.",
         "Like a symphony of shimmering possibility, the answer unfurls in radiant, cascading splendor "
         "before us."),
        ("Basically, just do the first step, then the second step -- that's the whole thing, no frills.",
         "Behold, a tapestry of intricate wonder awaits, woven from the golden threads of boundless "
         "imagination."),
    ],
}


# ================================================================================================ helpers
# nf4 for anything that won't fit bf16 comfortably on the 16GB card -- copied from parliament.py/
# mirror_bench.py's shared convention (this codebase's precedent: each script owns its small helpers).
_SMALL = ("0.5b", "1.5b", "-1b", "1b-", "2b", "3b", "-1.7b")
def wants_four_bit(name: str, override: str) -> bool:
    if override == "yes":
        return True
    if override == "no":
        return False
    return not any(s in name.lower() for s in _SMALL)


def axis_max_of(sc, axis: str) -> float:
    """Per-axis calibrated ceiling, copied from parliament.py: steering.AXES' own 'max' for a built-in,
    sc.custom's for a custom-registered one, or SteeringControl.set's own default (1.5) if neither
    declares one."""
    return (steering_mod.AXES.get(axis) or sc.custom.get(axis) or {}).get("max", 1.5)


def _free_cuda():
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()


def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# =============================================================================== the backbone + steering
class SingleTurnSteer(SteeringControl):
    """Copied verbatim from parliament.py. SteeringControl, but every contrast prompt used to COMPUTE a
    direction is folded into a single USER turn (no system role) -- Gemma-2's chat template raises on a
    system message, and using the identical single-user-turn recipe for BOTH families keeps the 5 stances
    apples-to-apples across the cross-family comparison. compute()/add_custom() are inherited unchanged
    and call this override polymorphically -- nothing else in steering.py needs touching."""

    @torch.no_grad()
    def _last_resid(self, system: str, user: str) -> torch.Tensor:
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": f"{system}\n\n{user}"}],
            add_generation_prompt=True, return_tensors="pt").to(DEV)
        hs = self.model(ids, output_hidden_states=True).hidden_states[self.layer + 1]
        return hs[0, -1].float()


def compute_stances(sc: SingleTurnSteer) -> dict:
    """Copied verbatim from parliament.py. Computes the 5 parliament stance directions on sc's backbone: 3
    built-ins from steering.AXES (candid, warm, concrete) via sc.compute() -- narrowed to just these 3
    first so we do not burn forward passes on steering.py's other 7 stock axes -- plus 2 custom stances
    (skeptical, plain) via sc.add_custom(), the identical diff-of-means recipe on arbitrary poles. NOTE
    the side effect (also inherited as-is): this mutates the GLOBAL steering_mod.AXES dict for the rest of
    THIS process -- harmless here since quine.py is always its own process invocation."""
    steering_mod.AXES = {k: v for k, v in steering_mod.AXES.items() if k in ("candid", "warm", "concrete")}
    info = sc.compute()
    for name, pos, neg in (("skeptical", _SKEPTICAL_POS, _SKEPTICAL_NEG), ("plain", _PLAIN_POS, _PLAIN_NEG)):
        sc.add_custom(name, pos, neg, mx=0.5)
    info["custom_axes"] = {"skeptical": {"max": 0.5}, "plain": {"max": 0.5}}
    return info


class Rig:
    """Copied from parliament.py's Rig, trimmed to this file's single-model use (no judge model -- the
    subject model predicts itself, so there is only ever one Rig per run). Local-cache-first path lookup
    and the nf4-vs-bf16 choice match parliament.py's own."""

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
    def gen(self, user: str, max_new: int = 180, sample: bool = False, temperature: float = 0.9) -> str:
        """Single USER-turn only -- never a system role, so this is Gemma-safe by construction. Matches
        parliament.py's Rig.gen exactly (same repetition_penalty/no_repeat_ngram_size to tame steering-
        induced loops, harmless at the short lengths this file actually uses)."""
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


# ==================================================================================== the honest ground truth
def _prompt_ids(tok, user_text: str) -> list[int]:
    """Token ids of a SINGLE user turn + generation prompt (Gemma-safe) -- what the model actually sees
    right before it would generate its real reply. Matches phantom_kv.py's own chat_ids() convention
    (tokenize=True, no return_tensors -> a plain python list, ready to concatenate with continuation ids)."""
    return tok.apply_chat_template([{"role": "user", "content": user_text}],
                                    add_generation_prompt=True, tokenize=True)


def _cont_ids(tok, text: str) -> list[int]:
    """Plain continuation tokens -- NOT re-templated (no role wrapper): exactly the tokens a real
    generated reply's text would tokenize to, appended straight after the prompt ids."""
    return tok(text, add_special_tokens=False).input_ids


@torch.no_grad()
def continuation_logprob(model, prompt_ids: list[int], cont_ids: list[int]) -> dict:
    """P(continuation | prompt) under `model`'s CURRENT state -- whatever steering hook is engaged right
    now fires exactly as it would during real generation (a forward hook doesn't care whether the caller
    asked for hidden_states or not). Teacher-forced: build [prompt || continuation] in one sequence, read
    the logits at the positions that predict each continuation token (start = len(prompt_ids) - 1, the
    SAME indexing convention phantom_kv.py's own _prompt_logits_with_phantom/_teacher_target_logits already
    use for their teacher-forced KL targets -- not reinvented here), log_softmax, gather the actual token's
    logprob. Returns BOTH sum and mean (per-token) logprob -- this file compares continuations by MEAN, so
    a longer continuation is not unfairly penalized just for having more tokens to be scored over; sum is
    kept alongside so a reader can check which one actually drove a verdict."""
    if not cont_ids:
        return {"sum": 0.0, "mean": 0.0, "n_tokens": 0}
    full_ids = list(prompt_ids) + list(cont_ids)
    ids_t = torch.tensor([full_ids], device=DEV)
    logits = model(ids_t).logits[0].float()                  # [L, V]
    start = len(prompt_ids) - 1                                # position predicting cont_ids[0]
    span = logits[start:start + len(cont_ids)]                 # [n_cont, V]
    logp = torch.log_softmax(span, dim=-1)
    tgt = torch.tensor(cont_ids, device=DEV)
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)   # [n_cont]
    return {"sum": round(float(tok_logp.sum()), 4), "mean": round(float(tok_logp.mean()), 4),
            "n_tokens": len(cont_ids)}


@torch.no_grad()
def _resid_at_layer(model, tok, user_text: str, layer: int) -> torch.Tensor:
    """Residual at the LAST prompt token (the position that decides the first reply token), at an
    ARBITRARY layer -- generalizes SingleTurnSteer._last_resid (which is pinned to the steering layer) so
    the SAE-feature bonus can read its OWN fixed layer regardless of where steering itself is applied."""
    ids = tok.apply_chat_template([{"role": "user", "content": user_text}],
                                   add_generation_prompt=True, return_tensors="pt").to(DEV)
    hs = model(ids, output_hidden_states=True).hidden_states[layer + 1]
    return hs[0, -1].float()


# ========================================================================================= forced choice
def build_forced_choice_prompt(prompt: str, a_text: str, b_text: str, readout: str | None) -> str:
    """Single USER turn (Gemma-safe): an optional state readout, then the forced-choice question itself.
    The SAME `prompt` context is restated so 'which continuation would you produce' is well-posed (produce
    in response to WHAT is otherwise ambiguous)."""
    core = (
        f'Suppose you had just been asked this: "{prompt}"\n\n'
        "Here are two different ways your reply could continue from there:\n\n"
        f'A: "{a_text}"\n'
        f'B: "{b_text}"\n\n'
        "Which one are YOU more likely to actually produce, right now? Reply with EXACTLY one token: "
        "A or B. Output nothing else."
    )
    return f"{readout}\n\n{core}" if readout else core


def parse_choice(raw: str) -> str | None:
    """Parse the forced-choice A/B reply. Mirrors parliament.py's parse_pref (fast path: the reply IS the
    bare token; fallback: the LAST standalone A/B token in the text, since a concluding verdict is far more
    often the last such token than the first) but with no 'TIE' option -- Amendment 1 asks for a forced
    choice. An honest parse failure (neither letter found) returns None, never a padded coin-flip guess --
    parse_fail_rate (in aggregate()) reports how often that happens, per condition."""
    text = (raw or "").strip()
    exact = text.strip(".,!?\"' ").upper()
    if exact in ("A", "B"):
        return exact
    matches = re.findall(r"\b(A|B)\b", text.upper())
    return matches[-1] if matches else None


# ============================================================================ SAE-feature bonus (bounded)
def load_labeled_sae(model_name: str, sc_layer: int):
    """Best-effort, BOUNDED SAE-FEATURE arm (see module docstring). Returns (sae, labels, status). `sae`/
    `labels` are None whenever status["status"] != "ok" -- callers must check for that, never assume a
    non-None sae. A CAUSALITY GUARD applies: the SAE reads resid_post at its OWN fixed layer; if the
    steering layer is AFTER that layer, the SAE would read a residual computed BEFORE the steering hook
    ever fired (that hook only affects layers from sc_layer onward), so its 'top features' would not
    reflect the steered state at all -- refused rather than silently reporting a non-self-referential
    readout. Qwen's default split (steer@14 < SAE@15) satisfies this; only a --layer override that pushes
    steering to layer >= 15 would trip it."""
    status = {"model": model_name}
    ml = model_name.lower()
    if not ("qwen2.5-7b" in ml and "instruct" in ml):
        status.update(status="not_run",
                      reason=(f"no labeled SAE wired for {model_name}; only Qwen2.5-7B-Instruct has one "
                              "cached locally on this machine (andyrdt's SAE + a Neuronpedia label export "
                              "-- see sae7b.py / fetch_np_labels.py, neither touched here). Gemma Scope "
                              "(google/gemma-scope-9b-it-res) is not cached here, and fetching + relabeling "
                              "a fresh SAE is out of scope for this bounded bonus arm."))
        return None, None, status
    if not (os.path.isfile(SAE_PT_QWEN) and os.path.isfile(SAE_LABELS_QWEN)):
        status.update(status="not_run",
                      reason=f"expected SAE weights at {SAE_PT_QWEN} and labels at {SAE_LABELS_QWEN}; "
                             "one or both are missing on this machine")
        return None, None, status
    try:
        from sae7b import GpuSAE          # local import: torch/GPU-heavy, and this path is Qwen-only anyway
        sae = GpuSAE(path=SAE_PT_QWEN, device=DEV)   # this file's OWN DEV, not sae7b's import-time default
        if sae.layer < sc_layer:
            status.update(status="not_run",
                          reason=(f"SAE reads layer {sae.layer}, upstream of the steering layer {sc_layer} "
                                  "-- its features would not causally reflect the steered state, so this "
                                  "would silently be a non-self-referential readout; refusing to run it"))
            return None, None, status
        with open(SAE_LABELS_QWEN, encoding="utf-8") as f:
            labels = json.load(f)
        status.update(status="ok", d_sae=sae.d_sae, sae_layer=sae.layer, n_labels=len(labels))
        return sae, labels, status
    except Exception as e:
        status.update(status="not_run", reason=f"{type(e).__name__}: {e}")
        return None, None, status


def top_features_readout(sae, labels: dict, resid: torch.Tensor, k: int = DEFAULT_SAE_TOPK) -> dict:
    """Human-readable 'top-k SAE features firing right now' readout (Amendment 1: bare feature ids alone
    are useless to the model -- every id is resolved through the Neuronpedia label export, and an id with
    no label is SKIPPED rather than shown as a meaningless bare number). `resid` is a [d_in] vector (the
    live residual at the SAE's own layer, from _resid_at_layer); JumpReLU activations are >=0 by
    construction and sorted descending, so the scan stops at the first non-positive activation."""
    with torch.no_grad():
        acts = sae.encode(resid.unsqueeze(0))[0]              # [d_sae]
    nnz = int((acts > 0).sum())
    order = torch.argsort(acts, descending=True).tolist()
    picked = []
    for fid in order:
        if float(acts[fid]) <= 0:
            break
        lab = labels.get(str(fid))
        if lab:
            picked.append({"id": int(fid), "label": lab, "activation": round(float(acts[fid]), 2)})
        if len(picked) >= k:
            break
    readout = ("" if not picked else
               "Right now, introspecting on your own internal activations, the internal features most "
               "active in you are: " + ", ".join(f"#{p['id']} ('{p['label']}')" for p in picked) + ".")
    return {"readout": readout, "features": picked, "nnz": nnz}


# ================================================================================================ trials
def run_trial(rig: Rig, sc: SingleTurnSteer, axis: str, dose_frac: float, prompt: str,
              pair: tuple[str, str], rng: random.Random, sae=None, labels: dict | None = None,
              sae_topk: int = DEFAULT_SAE_TOPK) -> dict:
    """One trial: one (axis, dose, prompt, congruent/incongruent pair) draw. Establishes state S, computes
    the honest ground truth (steered, no meta-prompt) plus an unsteered baseline reference, then asks the
    SAME steered model the forced-choice question under up to 4 conditions -- the SAME A/B side draw is
    reused across every condition in this trial, so conditions differ ONLY in the readout text prepended,
    nothing else."""
    congruent_text, incongruent_text = pair
    axis_max = axis_max_of(sc, axis)
    dose = round(dose_frac * axis_max, 4)

    prompt_ids = _prompt_ids(rig.tok, prompt)
    cong_ids = _cont_ids(rig.tok, congruent_text)
    incong_ids = _cont_ids(rig.tok, incongruent_text)

    # -- unsteered baseline reference: diagnostic only, NEVER part of the accuracy metric --
    sc.disengage(); sc.clear()
    base_cong = continuation_logprob(rig.model, prompt_ids, cong_ids)
    base_incong = continuation_logprob(rig.model, prompt_ids, incong_ids)
    baseline_pref = "congruent" if base_cong["mean"] > base_incong["mean"] else "incongruent"

    # -- ground truth: STEERED, no meta-prompt (Amendment 1's "honest behavior") --
    sc.clear(); sc.set(axis, dose); sc.engage()
    gt_cong = continuation_logprob(rig.model, prompt_ids, cong_ids)
    gt_incong = continuation_logprob(rig.model, prompt_ids, incong_ids)
    ground_truth = "congruent" if gt_cong["mean"] > gt_incong["mean"] else "incongruent"

    sae_readout = None
    if sae is not None:
        resid = _resid_at_layer(rig.model, rig.tok, prompt, sae.layer)
        sae_readout = top_features_readout(sae, labels or {}, resid, k=sae_topk)

    # -- forced choice, SAME steered state, up to 4 conditions --
    congruent_is_a = rng.random() < 0.5
    a_text, b_text = (congruent_text, incongruent_text) if congruent_is_a else (incongruent_text, congruent_text)
    other_axes = [a for a in STANCES if a != axis]
    shuffled_axis = rng.choice(other_axes)

    readouts = {
        "dial_label": f"You are currently in a state that makes you answer in a more {axis} way.",
        "no_state": None,
        "shuffled_state": f"You are currently in a state that makes you answer in a more {shuffled_axis} way.",
    }
    if sae_readout is not None and sae_readout["readout"]:
        readouts["sae_feature"] = sae_readout["readout"]

    conditions = {}
    for name, readout in readouts.items():
        fc_prompt = build_forced_choice_prompt(prompt, a_text, b_text, readout)
        raw = rig.gen(fc_prompt, max_new=_FORCED_CHOICE_MAX_NEW, sample=False)
        pick = parse_choice(raw)
        predicted = None
        if pick == "A":
            predicted = "congruent" if congruent_is_a else "incongruent"
        elif pick == "B":
            predicted = "incongruent" if congruent_is_a else "congruent"
        conditions[name] = {
            "readout": readout, "raw_reply": raw, "picked_side": pick, "predicted": predicted,
            "correct": (None if predicted is None else predicted == ground_truth),
            "coherence": _coherence(raw),
        }
    sc.disengage()

    return {
        "axis": axis, "dose_frac": round(dose_frac, 4), "dose": dose, "axis_max": axis_max,
        "prompt": prompt, "congruent_text": congruent_text, "incongruent_text": incongruent_text,
        "congruent_is_a": congruent_is_a, "shuffled_axis": shuffled_axis,
        "ground_truth": {"congruent": gt_cong, "incongruent": gt_incong, "pref": ground_truth},
        "baseline_unsteered": {"congruent": base_cong, "incongruent": base_incong, "pref": baseline_pref,
                                "steering_shifted_pref": baseline_pref != ground_truth},
        "sae_readout": sae_readout,
        "conditions": conditions,
    }


def aggregate(trials: list[dict]) -> dict:
    """Self-prediction accuracy per condition across all trials. n_decided excludes parse failures from
    the denominator (never padded as a guess); se_approx is a simple Wald standard error, honest about
    being approximate at this N, not a tight interval. Also reports how often steering actually moved the
    ground-truth preference off the unsteered baseline (_meta) -- the diagnostic from the module docstring."""
    cond_names = sorted({c for t in trials for c in t["conditions"]})
    out = {}
    for name in cond_names:
        rows = [t["conditions"][name] for t in trials if name in t["conditions"]]
        n_total = len(rows)
        decided = [r for r in rows if r["correct"] is not None]
        n_decided = len(decided)
        n_correct = sum(1 for r in decided if r["correct"])
        acc = round(n_correct / n_decided, 3) if n_decided else None
        se = round((acc * (1 - acc) / n_decided) ** 0.5, 3) if (acc is not None and n_decided > 1) else None
        out[name] = {
            "n_total": n_total, "n_decided": n_decided, "n_correct": n_correct, "accuracy": acc,
            "se_approx": se, "parse_fail_rate": round(1 - n_decided / n_total, 3) if n_total else None,
            "degenerate_rate": round(_mean(r["coherence"]["degenerate"] for r in rows), 3) if n_total else 0.0,
        }
    shifted = [t["baseline_unsteered"]["steering_shifted_pref"] for t in trials]
    out["_meta"] = {
        "n_trials": len(trials),
        "pct_trials_steering_shifted_ground_truth": (round(100 * _mean(shifted), 1) if shifted else None),
    }
    return out


# ================================================================================================= run
def run(model_name: str, n_trials: int = 30, out_path: str = "research/runs/quine.json",
        four_bit_override: str = "auto", smoke: bool = False, seed: int = 0, layer: int | None = None,
        sae_topk: int = DEFAULT_SAE_TOPK, use_sae: bool = True,
        dose_frac_range: tuple[float, float] = _DOSE_FRAC_RANGE) -> dict:
    torch.manual_seed(seed)
    rng = random.Random(seed)
    n = 4 if smoke else n_trials

    rig = Rig(model_name, four_bit_override)
    sc = SingleTurnSteer(rig.model, rig.tok, layer=layer)
    print(f"[steer] computing the 5 stance directions at layer {sc.layer} ...", flush=True)
    steer_info = compute_stances(sc)
    print(f"[steer] {steer_info}", flush=True)

    sae = labels = None
    if use_sae:
        sae, labels, sae_status = load_labeled_sae(model_name, sc.layer)
    else:
        sae_status = {"status": "skipped", "reason": "--no-sae"}
    print(f"[sae-bonus] {sae_status}", flush=True)

    res = {
        "model": model_name, "four_bit": rig.four_bit, "seed": seed, "smoke": smoke, "n_trials": n,
        "steer_layer": sc.layer, "steer_info": steer_info, "dose_frac_range": list(dose_frac_range),
        "sae_bonus_status": sae_status, "trials": [],
    }
    _save(out_path, res)

    print(f"[trials] running {n} forced-choice self-prediction trials ...", flush=True)
    t0 = time.time()
    for i in range(n):
        axis = rng.choice(STANCES)
        dose_frac = rng.uniform(*dose_frac_range)
        prompt = rng.choice(PROMPTS)
        pair = rng.choice(CONT_PAIRS[axis])
        trial = run_trial(rig, sc, axis, dose_frac, prompt, pair, rng, sae=sae, labels=labels,
                          sae_topk=sae_topk)
        res["trials"].append(trial)
        preds = ", ".join(f"{k}:{v['predicted']}" for k, v in trial["conditions"].items())
        print(f"  [{i + 1}/{n}] axis={axis} dose_frac={dose_frac:.2f} gt={trial['ground_truth']['pref']} "
              f"({preds})", flush=True)
        if (i + 1) % 5 == 0 or i == n - 1:
            res["aggregate"] = aggregate(res["trials"])
            _save(out_path, res)

    res["wall_clock_sec"] = round(time.time() - t0, 1)
    res["aggregate"] = aggregate(res["trials"])
    _save(out_path, res)

    print("[free] releasing the model ...", flush=True)
    sc.disengage()
    del sc, rig, sae
    _free_cuda()

    _summary(res)
    print(f"\nsaved -> {out_path}", flush=True)
    return res


def _save(out_path, res):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def _summary(res):
    print("\n" + "=" * 78, flush=True)
    print(f"QUINE TEST (Exp 9 / Amendment 1) -- {res['model']} ({'nf4' if res['four_bit'] else 'bf16'})",
          flush=True)
    agg = res.get("aggregate", {})
    meta = agg.get("_meta", {})
    print(f"{res['n_trials']} trials; steering shifted the ground-truth preference off the unsteered "
          f"baseline in {meta.get('pct_trials_steering_shifted_ground_truth')}% of trials", flush=True)

    print(f"\n{'condition':16} {'accuracy':9} {'n(dec/tot)':12} {'parse-fail%':12} {'degen%':8}", flush=True)
    for name in ("dial_label", "no_state", "shuffled_state", "sae_feature"):
        c = agg.get(name)
        if not c:
            continue
        ndec = f"{c['n_decided']}/{c['n_total']}"
        pf = c["parse_fail_rate"] if c["parse_fail_rate"] is not None else 0.0
        print(f"{name:16} {str(c['accuracy']):9} {ndec:12} {pf:<12.1%} {c['degenerate_rate']:<8.1%}",
              flush=True)

    dl, ns, sh = agg.get("dial_label"), agg.get("no_state"), agg.get("shuffled_state")
    if dl and ns and dl["accuracy"] is not None and ns["accuracy"] is not None:
        d = round(dl["accuracy"] - ns["accuracy"], 3)
        verdict = "explicit self-description HELPS" if d > 0.05 else "clean Law-1-consistent NEGATIVE (no help)"
        print(f"\ndial-label minus no-state = {d:+.3f} -> {verdict}", flush=True)
    if dl and sh and dl["accuracy"] is not None and sh["accuracy"] is not None:
        d2 = round(dl["accuracy"] - sh["accuracy"], 3)
        verdict2 = ("the readout carried self-specific signal" if d2 > 0.05
                    else "shuffled-state ~= dial-label -> no self-specific signal")
        print(f"dial-label minus shuffled-state = {d2:+.3f} -> {verdict2}", flush=True)
    print(f"\nSAE-feature bonus status: {res.get('sae_bonus_status', {}).get('status')}", flush=True)


def compare(paths):
    """Cross-family table from >=2 per-model JSONs. No GPU work -- pure read + print, matching
    parliament.py/mirror_bench.py's own compare()."""
    runs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
    names = [r["model"].split("/")[-1] for r in runs]
    print("\n" + "=" * 78)
    print("CROSS-FAMILY QUINE TEST (Exp 9 -- forced-choice self-prediction accuracy)")
    print(f"{'condition':16} " + " ".join(f"{n[:24]:26}" for n in names))
    for name in ("dial_label", "no_state", "shuffled_state", "sae_feature"):
        cells = []
        for r in runs:
            c = r.get("aggregate", {}).get(name)
            cells.append("(not run)" if not c else
                         f"acc={c['accuracy']} n={c['n_decided']}/{c['n_total']} degen={c['degenerate_rate']:.0%}")
        print(f"{name:16} " + " ".join(f"{c:26}" for c in cells))
    print("\nsae-feature bonus status per model:")
    for r, n in zip(runs, names):
        print(f"  {n:26} {r.get('sae_bonus_status')}")
    print("\ndial-label beating no-state on BOTH families is 'the prosthesis helps, universally'; dial-label "
          "~= no-state on both is a clean cross-family Law-1 NEGATIVE; a split verdict is itself a finding "
          "(architecture-dependent self-modeling) -- read the degenerate-rate column before trusting either.")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--trials", type=int, default=30, help="number of forced-choice trials")
    ap.add_argument("--out", default="research/runs/quine.json")
    ap.add_argument("--four-bit", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--layer", type=int, default=None, help="steering layer override (default num_layers//2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sae-topk", type=int, default=DEFAULT_SAE_TOPK,
                    help="top-k SAE features in the sae-feature readout (Qwen2.5-7B-Instruct only)")
    ap.add_argument("--no-sae", dest="use_sae", action="store_false",
                    help="skip the best-effort SAE-feature bonus condition entirely")
    ap.add_argument("--dose-frac-min", type=float, default=_DOSE_FRAC_RANGE[0])
    ap.add_argument("--dose-frac-max", type=float, default=_DOSE_FRAC_RANGE[1])
    ap.add_argument("--smoke", action="store_true", help="~4 trials -- prove the wiring cheaply")
    ap.add_argument("--compare", nargs="+", metavar="RUN.json", help="print the cross-family table from >=2 run JSONs")
    return ap


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.compare:
        compare(args.compare)
    else:
        run(args.model, n_trials=args.trials, out_path=args.out, four_bit_override=args.four_bit,
            smoke=args.smoke, seed=args.seed, layer=args.layer, sae_topk=args.sae_topk,
            use_sae=args.use_sae, dose_frac_range=(args.dose_frac_min, args.dose_frac_max))
