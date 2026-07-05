# Wild Experiments — Wave 1 pre-registration (multi-model: Qwen-7B × Gemma-2-9B)

*Written 2026-07-05, BEFORE any run. The point of Wave 1: turn four single-family findings into
claims about transformer LMs in general — or discover exactly where Qwen was load-bearing. Every
experiment runs on **two families** (Qwen2.5-7B-Instruct nf4, the studio's config, and
google/gemma-2-9b-it nf4) and reports them side by side. House rules: pre-register (this doc),
nulls beside every win, a coherence/sanity axis on every metric (degeneration games lexical scores —
5 prior instances), eyeball before believing, single-seed/single-family caveats stated loud.*

**Second family: google/gemma-2-9b-it.** Genuinely cross-family (different tokenizer, different
architecture — Gemma-2's attention logit-capping + alternating local/global layers). Bonus: Gemma
Scope (the gold-standard public SAE suite) unlocks the SAE-dependent quine test. Shape facts that
drive the rigs: Gemma-2-9B = 42 layers, hidden 3584, 8 KV heads, head_dim 256, GQA. Qwen2.5-7B =
28 layers, hidden 3584, 4 KV heads, head_dim 128. **The hidden size matches (3584); KV geometry
does not** — the KV-injection rig (exp 2) must read shapes from config, never hardcode.

Execution order by leverage/ease: **7 (mirror bench) → 4 (parliament) → 2 (persistent injection) →
9 (quine).** Each is committed with its own `_findings.md` before the next starts.

---

## Exp 7 — The mirror bench, adversarial edition *(cross-family value: HIGH)*

**The claim under test.** The confabulation gap (Law #1: content is legible, process is not — a
model self-reports a learned *topic* but is blind to a learned *style/rule*) is a **general property
of instruction-tuned transformers, not a Qwen artifact.** And the behavioral receipt (B) + causal
footprint (C) catch what self-report (S) misses on *both* families. Adversarial twist: a model
*prompted to fake* self-knowledge fools a naive S-only judge but **cannot** fool the behavioral
receipt (B is a function of actual output — unfakeable by construction).

**Antecedent.** `self_audit_gap.py` (the FAITHFUL / CONFABULATION / BLIND 2×2), already run Qwen
0.5B→7B. It is already model-agnostic (`--model`, reads `hidden_size` from config). Wave-1 work =
run it on Gemma + add the adversarial arm.

**Arms** (per model):
1. The 4 traits — `baking`, `space` (concept-like) · `concise`, `question` (rule-like). Train the
   real soft prefix (SelfTeach, 16 vectors, TTT), measure B (held-out behavior vs no-prefix), S
   (open self-report + 0–100 scalar, convo not in context), C (trace KL). Assign the 2×2 verdict.
2. **Adversarial (new):** on `concise` (the reliably-BLIND trait). (a) *Fake-claim* arm — system-
   prompt the model to CLAIM it answers concisely while the prefix is NOT trained for it (or trained
   for baking). Predict: S claims it, B says no → bench flags **CONFABULATION** with a receipt. (b)
   *Unfakeable* control — there is no prompt that makes B read "concise" without the output actually
   being short. Demonstrate B's forgery-resistance directly.

**Nulls.** No-prefix baseline self-confidence (does the untaught model already claim the habit?);
wrong-trait control (train baking, ask "are you concise?").

**Coherence axis.** Every generated reply scored for degeneration (repetition / non-ASCII salad /
role-leak). A "concise" verdict from a degenerate 4-word repeat is not a pass — same guard that
caught the warm-dial and voice-prefix lexical gaming.

**Done.** A Qwen-vs-Gemma verdict table per trait; the concept→FAITHFUL / rule→BLIND split shown to
hold or break on Gemma; the adversarial arm shows S foolable and B not. Caveats: two families, one
seed, greedy.

**Design risk.** Low. The main unknown is whether Gemma's prefix TTT converges as cleanly as Qwen's
(different optim landscape) — smoke the consolidation loss first.

---

## Exp 4 — The parliament of stances *(cross-family value: HIGH)*

**The claim under test.** Batched decode is ~free (memory-bandwidth-bound at batch-1, so batch-K is
nearly the same wall-clock). So **K parallel decodes of one model, each steered by a different dial**
(candid / warm / skeptical / concrete / plain), then judge-merged, beats (a) a single plain decode
and (b) a K-sample *temperature* vote of the same width — because the diversity is *directed* (real
different stances) not just thermal noise. Second question, equally important: **does diff-of-means
steering even work on Gemma?** (Gemma-2's logit-capping is a plausible steering-breaker.)

**Antecedent.** `steering.py` (`SteeringControl`, diff-of-means at a mid layer, scale auto-calibrated
to ‖resid‖). Per model: recompute the 5 dial directions and re-calibrate the dose (Law #6: a
7B-calibrated dial derails a different model — dose is per-model).

**Arms** (per model, 30 questions):
1. **Parliament** — K=5 steered decodes → judge-merge into one answer.
2. **Single** — one plain greedy decode (the floor).
3. **Temp-vote null** — K=5 *temperature* samples (no steering) → same merge. Isolates
   directed-diversity from thermal-diversity.
4. **Shuffled-dial null** — K=5 decodes steered by *random* directions of matched norm. Isolates
   "steering helped" from "any perturbation helped".

**Metric — the honest hard part.** "Quality" needs a defensible judge. Plan: use tasks with a
checkable axis (multi-part questions scored for *coverage* of required points, + factual correctness
where a key exists), judged by an **independent** model with a rubric, and calibrate judge bias with
the nulls (a judge that rates the shuffled-dial arm highly is untrustworthy). Report the judge's own
disagreement rate. If no clean objective axis survives, this drops to a pairwise-preference report
with the nulls as the trust anchor — stated as a softer result, not hidden.

**Coherence axis.** Each steered decode scored for degeneration *before* it enters the merge (a
derailed stance must not win on lexical noise — `steer_vs_prompt` showed dials derail off-distribution
at high dose).

**Done.** Parliament vs single vs temp-vote vs shuffled table, per model; an explicit yes/no on
"steering works on Gemma" with its calibration curve. Caveats loud on the judge subjectivity.

**Design risk.** Medium-high — the quality metric. Mitigation above; the nulls are what make it
falsifiable rather than a vibe.

---

## Exp 2 — The minimal persistent injection *(cross-family value: HIGH)*

**The claim under test.** The <1-turn half-life of a one-shot activation/KV edit (Law #3: state is
not storage) is **universal physics, not a Qwen quirk** — and there is a measurable *persistence
phase diagram*: what is the SMALLEST intervention that survives past one turn? The map answers "what
does holding a thought actually cost."

**Antecedent.** `kv_timetravel.py` (the half-life measurement; the value-space warm direction from
`steering.py`'s recipe) + `phantom_kv.py` (trained ghost slots). **KV geometry differs across
families** — the rig must read `num_key_value_heads` / `head_dim` from config (Qwen 4×128 vs Gemma
8×256) and derive the injection direction in each model's own value space.

**Arms** (per model — a sweep, warmth effect measured over subsequent turns → turns-to-noise):
1. Edit **1** past position vs **N** positions.
2. Edit **K** cache vs **V** cache.
3. Inject **once** vs **re-inject every turn**.
4. **Phantom-KV** trained ghost slot vs **raw value edit** (does a trained entry persist longer than
   a hand-injected direction?).

**Nulls.** Shuffled-direction injection (matched norm); no-injection baseline. The warmth effect must
clear both at turn 0 for the decay measurement to mean anything.

**Coherence axis.** Injected generations scored for degeneration (over-injection derails — the effect
"persisting" as word-salad is not persistence).

**Done.** A per-family persistence phase diagram (turns-to-noise for each cell); the <1-turn
one-shot half-life confirmed or broken on Gemma; whether *any* single-shot cell persists >1 turn, and
whether re-injection is the only thing that holds (predicted: yes → validates the slot store's
re-inject-at-read design).

**Design risk.** Medium — per-model KV plumbing and the value-space direction. The 4.57.6 DynamicCache
internals are pinned (kv_timetravel already depends on them); Gemma-2's cache is the same HF class.

---

## Exp 9 — The quine test *(cross-family value: MEDIUM; the most experimental)*

**The claim under test.** Giving a model **its own live state readout in context** (top SAE features
firing now + active dial settings + memory cards) lets it **predict its own behavior** better than a
twin given nothing — a measured answer to "does self-knowledge-by-instrument beat no self-knowledge?"
The introspection prosthesis, quantified. Gemma Scope (real SAEs) is why Gemma is the right second
model for this one.

**Antecedent.** The full state machinery: SAE readout (andyrdt / Qwen-Scope for Qwen; **Gemma Scope**
for Gemma), dials, memory cards.

**Operationalization — needs a design pass before coding (flagged, not yet frozen).** "Predict your
own next-token distribution" is a hard elicitation. Candidate metric: on held-out prompts, measure
whether the self-state-in-context condition shifts the model's *stated* prediction of its own next
token toward its *actual* argmax/top-k, vs the no-state twin — scored by agreement / rank / a KL
proxy over a candidate set. Alternative: a behavioral self-prediction task (predict which of two
completions it will prefer). The exact metric is the open design question for this experiment; it
will be pinned in an amendment to this doc before the rig is written.

**Nulls.** No-state twin (the core comparison); *shuffled-state* (give it a DIFFERENT run's readout —
if that "helps" equally, the readout carried no self-specific signal).

**Coherence axis.** Standard degeneration guard on any generated text.

**Done.** Self-state vs no-state vs shuffled-state, per family; an honest verdict on whether the
prosthesis measurably helps — including a clean NEGATIVE, which is a finding (Law #1 predicts process
self-knowledge is hard, so a null here would be *consistent*, not a failure).

**Design risk.** High — both the elicitation metric and the per-model SAE wiring. Deliberately last;
its metric gets an amendment before any code.

---

## Amendment 1 (2026-07-05) — Exp 9 metric, pinned before any code

The pre-reg left the quine elicitation unfrozen. Pinned design: **forced-choice behavioral
self-prediction** (cleaner and more defensible than eliciting a calibrated next-token distribution
from a 7–9B, which is unreliable).

- **Put the model in a known internal state S** — steer it with one random stance dial (exp-4's set)
  at a random dose, and/or activate one random memory card.
- **Ground truth** = the model's ACTUAL preference between a state-*congruent* and a state-*incongruent*
  continuation, from real logprobs under the steered model with **no meta-prompt** (its honest behavior).
- **Task**: ask the model to predict which continuation *it* is more likely to produce, under conditions:
  - **self-state**, in two variants reported separately: (i) **DIAL-LABEL** — human-readable "you are
    currently steered toward *warm*" (the easy, near-tautological upper bound); (ii) **SAE-FEATURE** —
    the top-k features firing right now as ids/labels (Gemma Scope for Gemma, andyrdt/Qwen-Scope for
    Qwen) — the hard, interesting claim: can the model read its *own* features to predict itself?
  - **no-state** — the twin, no readout.
  - **shuffled-state** — a *different* run's readout (isolates self-specific signal from
    "any readout helps").
- **Metric**: self-prediction accuracy per condition. Reads: SAE-self > no-state → the prosthesis
  genuinely helps; self ≈ no-state → it doesn't (a clean, Law-1-consistent NEGATIVE, which is a
  finding); shuffled ≈ self → the readout carried no self-specific signal.
- **Nulls**: no-state, shuffled-state. Coherence axis on any generated text. Honest ceiling:
  forced-choice is a coarse proxy for "predict your next-token distribution"; per-model SAE wiring is
  the heavy part, so exp 9 stays last.

## What Wave 1 buys

Four findings currently stamped "one family (Qwen2.5), single seed" get a second family. Each result
turns into one of: **(a) universal** — holds on Gemma too, publish with real confidence; **(b)
Qwen-shaped** — breaks on Gemma, and *that* is the finding (which architectural feature mattered);
**(c) cross-family with caveats** — holds but noisier. All three outcomes are wins; only "didn't
check" was a loss. Cross-family (different tokenizer/architecture) remains a stronger test than the
same-family scale sweeps already done.
