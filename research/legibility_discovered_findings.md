# Legible learning in a DISCOVERED basis — UNGATED first cut (GPT-2 + Bloom's gpt2-small-res-jb)

*Run 2026-06-22. `research/legibility_discovered.py`, synchronous single process, `.venv-sae` (sae_lens
6.44.3 + transformer_lens + Bloom's `gpt2-small-res-jb` SAE), FROZEN GPT-2-small (124M), CPU, ~155 s.
Outputs: `research/runs/legibility_discovered_gpt2.json` + three SVGs (`_ttt_`, `_specificity_`,
`_causal_`) + `research/runs/_legible_discovered.log`.*

## The question

We can make a frozen model **learn** a new rule (test-time adaptation: a soft prefix fit by a few
gradient steps on the rule's own examples applies an unseen 1-to-1 relation to held-out words —
`frontier_apply_v2` lever 3). But the learned prefix is an **opaque blob**: a probe names it at chance,
hand-named sliders only "named" it via an input-feature artifact, self-report confabulated
(`legibility_v1`). The feature-discovery deep-dive's top recommendation (`feature_discovery_deepdive.md`
§5, idea 3 / STEP-5): stop reading the learned rule in a *hand-named* basis and read it in a
**pretrained, interpretable feature dictionary** — a real SAE — the **Golden-Gate move pointed at the
model's own learned rule**. This file is the **ungated proof of concept**: GPT-2-small + Bloom's
`gpt2-small-res-jb` residual SAE (both ungated, already set up in `.venv-sae` from the SAE salvage). The
richer Gemma Scope version needs gated access and is the recommended follow-up.

**The bet, made concrete (four falsifiable claims):** the TTT-learned rule reads out as a set of
discovered SAE features that is (1) **SPARSE**, (2) **RULE-SPECIFIC**, (3) **NAMEABLE** (the lit
features read as the rule), and (4) **CAUSAL** (clamp them on a fresh query → the rule fires). If yes →
the legible-AND-rich dream, ungated. If no → a valid negative that tells us whether the bottleneck is
the basis or the substrate.

## Verdict: **PARTIAL (2/3 axes) — SPARSE ✅ + RULE-SPECIFIC ✅, but NOT NAMEABLE ❌ and NOT CAUSAL ❌.**

The learned rule's activation-delta is a genuinely **sparse** and **rule-discriminative** combination of
Bloom's discovered features — but the features it lights are largely **generic high-magnitude residual
directions** (not nameable as the rule), and **clamping them back does not reproduce the rule**. On a
124M model with an ungated SAE the read-out is *legible-ish in geometry* but not *legible-as-the-rule*
and not *causal*. This is the honest, valuable result; it is **not cherry-picked** and the negative axes
are reported as plainly as the positive ones.

---

## What happened, step by step

### STEP 1 — TTT on GPT-2 **WORKS** (the precondition holds)

The soft-prefix TTT that worked on Qwen-0.5B **also works on GPT-2-small**, better than expected for a
124M model. Fitting an 8-vector soft prefix for 40 steps on each relation's TRAIN words (frozen
backbone, only the prefix moves), then evaluating held-out **free-generation** apply (full-vocab argmax
== answer):

| | no-prefix | TTT held-out (free) |
|---|---|---|
| aggregate over 12 relations | **0.000** | **0.768** |

All **12/12** relations cleared the keep-bar. Per relation (held-out free): plural 1.00, third_person
1.00, gerund 1.00, past 1.00, comparative 0.88, opposite_gender 0.75, continent 0.75, antonym2 0.67,
agent 0.67, superlative 0.60, color 0.57, part_of 0.33. No-prefix free is 0.000 for every relation
(GPT-2 alone does not do `{x} ->` analogies), so the entire signal is the learned prefix. (GPT-2 is
weaker than Qwen, as the brief anticipated — part_of/color are the floor — but TTT genuinely applies
these rules to held-out words. The precondition for the read-out is satisfied.)

### STEP 2 — read the rule in discovered features: **SPARSE ✅ and RULE-SPECIFIC ✅**

For each held-out query we take the **activation-delta** the adaptation induces at the SAE hook
(`blocks.8.hook_resid_pre`) and encode it **in feature space**: `enc_SAE(resid_with_prefix) −
enc_SAE(resid_without_prefix)`, averaged over the relation's held-out queries. (We encode each state and
subtract, **not** the raw residual-delta — the SAE encoder's per-feature threshold is tuned for
activations, so encoding a raw delta is dense and meaningless; verified ~3.6k nonzero that way.)

- **SPARSE ✅.** Mean participation-ratio (effective # of features moved) = **115** out of 24,576, vs a
  matched-norm **random-direction null of 6,801** — the learned rule moves a **~60× sparser** set than a
  random residual perturbation of the same size. Per-relation L0 is 261–656; the top ~150–280 features
  carry 90% of the |delta| mass. The rule is a small combination of features, not a smear.
- **RULE-SPECIFIC ✅.** Relation × relation cosine of the positive feature-deltas: off-diagonal mean
  **0.27** (max 0.54); top-12 feature-set Jaccard off-diagonal **0.10**. After **removing the shared
  across-rule component** (the common "an answer is being produced at this slot" direction every prefix
  induces), the mean off-diagonal cosine is **−0.08** — i.e. once you subtract what's common to all
  rules, different rules' residual deltas are essentially **uncorrelated**. Different rules light
  different features. (The `_specificity_gpt2.svg` heatmap shows this shared-removed matrix: bright
  diagonal, near-zero/negative off-diagonal.)

So on the two "is it legible *as geometry*" axes the answer is **yes**: the read-out is sparse and it
discriminates the rule.

### STEP 3 — interpretability of the lit features: **NOT cleanly NAMEABLE ❌** (the honest ceiling)

Top-activating tokens/contexts per feature read from the cached 56k-row WikiText GPT-2 layer-8 SAE
activation matrix (`inspector/runs/gpt2_control_acts.npz`, the pretrained-SAE control's harvest — same
model/layer/SAE, fully offline), plus **Neuronpedia auto-interp labels for all 92 distinct top features
(network reachable)**. The labels are the load-bearing finding:

- `plural`'s top features by delta are `f18486` "text parsing errors / code artifacts", `f16021` "dates
  and locations", `f15075` "DOIs/URLs", `f1259` "military weapons" — **none of which is "plural nouns".**
- `antonym2`'s top is `f9444`, Neuronpedia: *"conflicts/debates between opposing concepts (symbol vs.
  substance, good vs. evil, …)"* — that one is **arguably on-theme** (opposition), the cleanest hit in
  the whole run. But its other top features are `f23162` "comparisons between two entities" and `f16021`
  "dates and locations".
- A handful of **generic high-magnitude directions recur across almost every rule**: `f16021`
  (dates/locations), `f9444` (and/vs), `f12423` (numbers-in-text), `f22852` (URLs), `f15075` (DOIs).
  These 10 features appear in ≥3 rules' top-12.

Quantified: **mean 35% of each rule's top-12 features are generic** (shared by ≥3 rules), and only
**8% of rules' top-1 feature has a clean lexical modal token** (most top-1 modal tokens are punctuation,
`@`, or stopwords like "and"). So the read-out **separates rules statistically** (STEP 2) **without the
lit features reading as the rule.** The sparse, rule-specific direction is carried substantially by
*which generic features fire and how strongly*, not by a dedicated "make-it-plural" feature — Bloom's
24k-feature GPT-2 SAE simply may not **have** a clean "plural" or "past-tense" feature, and the
prefix-induced delta at the answer slot is dominated by answer-formatting directions.

### STEP 4 — causal check (the actual Golden-Gate move): **NOT CAUSAL ❌**

Build the injection from the read-out features' **decoder directions** (the full positive-feature
reconstruction — turn on exactly the features the rule turned on, at their relative strengths), add it to
the residual at layer 8 on a **fresh held-out query with no prefix**, sweep the scale ×{1,2,4,8} of the
natural prefix-delta norm, and measure recovered held-out apply. Controls: a **random-feature clamp**
(matched feature count, matched final norm), the **no-prefix floor**, the **full-prefix TTT ceiling**.

| | free-apply (aggregate) |
|---|---|
| no-prefix floor | 0.000 |
| **read-out feature clamp (best scale)** | **0.035** |
| random-feature clamp | 0.000 |
| TTT ceiling | 0.768 |

The clamp recovers **~5% of the TTT gain** — essentially nothing. It beats the random-feature null
(0.035 vs 0.000) and only **2 of 12 relations** show any recovery at all (antonym2 0.17 / 25% of gain,
continent 0.25 / 33% of gain). Notably the **menu** score (argmax restricted to the candidate answer
set) does move for several rules (third_person 0.43, past/continent 0.14–0.25) — the clamp nudges the
model *toward the right answer set* but not enough to win free-generation. Under the honest bar (beat
random by a margin **and** recover ≥25% of the TTT gain), this is **NOT CAUSAL**: reconstructing the
SAE-feature delta and adding it back does not reproduce the rule on a 124M model.

---

## Why it lands where it does (the diagnosis)

The read-out is **sparse and rule-discriminative** but **not nameable and not causal**. Three mechanisms,
each a likely contributor, all pointing at the *substrate/dictionary*, not at the method:

1. **The base model is weak (124M).** GPT-2-small's residual stream has less abstract, less linearly-
   separable rule structure than Qwen/Gemma — the deep-dive's own primary diagnosis. The prefix learns
   to *produce the answer token* more than to *instantiate a reusable "pluralize" direction*, so the
   delta is dominated by answer-formatting.
2. **The dictionary may not contain the concept.** Bloom's 24k-feature SAE is the ungated gold standard
   but it is **small** (Golden Gate Claude used 1M–34M features). Anthropic's own coverage logic says at
   this width most concepts have **no dedicated feature** — there may simply be no "plural" or
   "past-tense" atom, so the rule gets expressed through generic features and is neither nameable nor
   cleanly clampable.
3. **The delta is answer-slot-dominated.** Reading the delta at the *answer position* of `{x} ->` mixes
   the rule with "emit a word here" — which is exactly the shared component STEP 2 had to subtract to see
   specificity, and exactly what the generic recurring features (numbers/dates/`and`) look like. The
   *centered* read-out is rule-specific, but you cannot clamp the centered read-out without re-adding the
   shared part, and the shared part is what the SAE actually represented.

This is a coherent picture, and it is **precisely the regime the deep-dive predicted**: ungated GPT-2 +
small SAE gets you *sparse + specific* (the geometry is real) but stalls at *nameable + causal* (the
substrate is too thin). The negative axes are informative, not a failure of the experiment.

## Honest controls (all present, none skipped)

- **Sparsity null:** matched-norm random residual direction encoded the same way → PR 6,801 vs real 115
  (real is ~60× sparser). The sparsity claim clears its null decisively.
- **Specificity:** raw + Jaccard + **shared-removed** cosine; the shared-removed off-diagonal (−0.08)
  confirms a **different rule's delta lights different features** (the brief's required check).
- **Nameability:** Neuronpedia labels for all 92 features + a generic-feature-share statistic — the
  read-out is shown to lean on shared generic directions, reported plainly.
- **Causal null:** random-feature clamp (matched count + norm) + no-prefix floor + TTT ceiling beside
  every number; the read-out clamp barely beats the random clamp and recovers ~5% of the gain.
- **Per-relation everywhere**, free-gen reported alongside menu (the menu-mirage guard), baselines beside
  every aggregate. GPT-2's weakness is named as a confound, not hidden.

## What this means / next

- **The first two axes are a genuine ungated win:** a TTT-learned rule **does** read out as a *sparse,
  rule-specific* combination of pretrained discovered features — more than hand-named sliders or self-
  report ever delivered. The *geometry* of "the model learned [these features moved]" is legible.
- **The last two axes are an honest miss on GPT-2:** the lit features are not nameable as the rule and
  clamping them is not causal. The bottleneck reads as **substrate + dictionary width**, not the method.
- **The recommended follow-up is exactly the deep-dive's lead:** rerun on **Gemma-2-2B + a pretrained
  Gemma Scope residual SAE (layer 12, widths up to 1M, every feature Neuronpedia-labeled)** — a richer
  base model and a ~40× wider dictionary are the direct test of whether *nameable* and *causal* close. It
  needs gated HF access (Gemma) and the GPU; the `sae_lens` `from_pretrained → encode` path and the whole
  TTT-delta-read-clamp harness here transfer unchanged. A secondary fix worth trying first on GPT-2:
  read/inject the delta at a **mid-layer non-answer position** (or average over the prompt) to dodge the
  answer-slot-dominated delta, and clamp at **all positions**.

---

### Reproduce

```
# .venv-sae only — do NOT touch the lab GPU venv.
cd research
../.venv-sae/Scripts/python.exe legibility_discovered.py --n_relations 12 --ttt_steps 40 --tag _gpt2
#   --no_interp        skip the WikiText auto-interp (faster)
#   --no_neuronpedia   skip the bonus Neuronpedia labels (offline)
#   --sae_layer 8      Bloom SAE layer (only layer 8 is cached locally; 0..11 exist on the Hub)
# Synchronous, single process, ~155 s on CPU. Outputs to research/runs/legibility_discovered_gpt2.*
```

Relation bank + TRAIN/TEST split are reused from `frontier_apply_v2` (re-filtered to GPT-2 BPE; `currency`
dropped for too few single-token pairs). STEP 3 reuses the pretrained-SAE control's cached activation
matrix (`inspector/runs/gpt2_control_acts.npz`) for auto-interp, fully offline.
