# Distributed function: individually-nameable units carry almost no causal mass

**Status:** internal research writeup. Not for external distribution without a separate review pass.
**Scope:** Qwen2.5-7B-Instruct, Qwen3.5-9B, Meta-Llama-3.1-8B-Instruct, all served Q4_K_M via
llama.cpp through clozn's own engine. Two vendor families, three checkpoints, four independently
built measurement instruments, small batteries (5-30 cases each). Every number below is read
directly from a receipt JSON or a script's committed source; where a receipt disagreed with the
draft brief this was written against, that is called out explicitly in its own section rather than
quietly corrected.

**Date:** 2026-07-23. **Author's note:** this document is written by an agent from the receipts on
disk; it makes no claims not traceable to a specific file path below.

---

## Abstract

Across four measurement instruments, built independently, at different times, by different means,
on this project's own quantized 7-9B chat models, the same pattern recurs: an individually-nameable
unit of computation — one context position, one SAE feature, one attention head, one residual site
— carries almost none of the causal mass behind a model's answer, and fails to separate from a
matched random control. Some, not all, ways of aggregating units into a SET recover a clean, large,
control-beating effect: a greedily-constructed contiguous span of input positions (severed at the
attention edge) reaches separations of 100x-800,000x over matched random controls; a jointly-ablated
group of 8-16 SAE features reaches 10-45% of a site's causal mass against a 0-6% random-group floor;
jointly ablating an entire trace's surviving residual sites produces a prediction-vs-observation
scorecard correct on 91-92% of tested flips across both families. But one further generalization of
"the set" — a *coalition of residual sites* built by the same greedy method that worked for input
spans, this time applied to (layer, position) sites under residual mean-ablation — does NOT clear
its own control (mean separation 0.93x-1.43x, never reaching the 2x bar). The positive and negative
results together triangulate a specific, falsifiable, and narrower claim than "everything is
distributed": on these models, at these scales, *localized causal structure lives on the input
attention edges, not on the residual stream's sites* — a sharp instrument (edge severance) finds it,
and a blunt one (residual ablation) does not, at any granularity tried so far.

---

## The claim

Define two kinds of causal object this project's instruments can name:

- **A unit**: one context position, one dictionary (SAE) feature, one attention head, or one
  (layer, position) residual site — anything a mechanistic-interpretability report would ordinarily
  draw as a single circled node.
- **A set**: a contiguous span of positions, a group of features, a joint ablation of many sites at
  once, or a coalition of sites chosen to jointly reproduce most of an effect.

The claim, stated at the precision this data supports: **on Qwen2.5-7B-Instruct, Qwen3.5-9B, and
Llama-3.1-8B-Instruct, served Q4_K_M, a single unit's own solo causal contribution is reliably small
relative to what a well-chosen set of the same kind of unit achieves, and fails to clear a matched
random-control bar that some (not all) set-level constructions clear easily.** This is not a claim
that units contribute literally zero — several individual measurements below show a single unit
carrying a real, nonzero, sometimes large effect — it is a claim about a *replicated pattern*: four
separately-built instruments, at four or five different granularities, using four different control
designs, all found the same shape. Where the pattern breaks (the molecule/coalition result, below)
it breaks in the same direction — toward "individual claims about which specific unit or set matters
here do not survive controls" — which is itself evidence the instruments are measuring something
real rather than an artifact of one script.

---

## Instrument 1 — positions (attention-edge severance + greedy span search)

**Method.** `clozn/analysis/provenance.py`'s `trace_provenance()`. For a prompt and its own greedy
answer, zero the attention edge from the final (answering) position to a candidate context position
`p`, at every layer simultaneously (`attn_knockout: {layer, queries:[final], keys:[p],
renormalize:true}`), and re-score the answer token teacher-forced. `context_dependence = 1 -
exp(cut_logprob - base_logprob)`. Two fixes were forced by measurement before this worked at all
(both documented in the module's own docstring and in `notes/CIRCUIT_TRACER_DESIGN.md` §5g-5h):
*renormalize*, because un-renormalized knockout ranked the attention-sink position (index 0) top
at +0.717 — a pure amplitude artifact, not a routing result, that "vanishes from every span" once
renormalization is on — and *greedy accumulation instead of single-position ranking*, because a
multi-token entity's positions are individually redundant (cut one BPE piece of a passcode or name
and its siblings re-supply the answer).

**Control.** At each greedy step, the joint delta of the accumulated span is compared against
`n_controls` (default 3) matched random position-sets of the same size, drawn from the remaining
pool; `best_control_ratio` is the largest such ratio seen across the whole trace. A verdict of
`CONTEXT_CARRIED` / `MIXED` / `PARAMETRIC` requires `best_control_ratio >= 3.0`; below that the
verdict is `INCONCLUSIVE` regardless of the raw effect size.

**Result.** The starkest single before/after number in this project's history is from the 4-prompt,
1-family pilot (`notes/CIRCUIT_TRACER_DESIGN.md` §5h): on the "gate code" prompt (a passcode
introduced mid-paragraph, then asked for again), the single best position scored **+0.11** nats
while the greedy-accumulated span scored **+7.60** nats on the *same* prompt. The 30-case,
two-family validation battery that followed contains the same case (`kv_late` — "Meeting notes: ...
The passcode for the shared drive was changed to 9314. ... The new passcode for the shared drive
is") re-measured at a different pass: best single position **0.030** nats vs. span **7.304** nats, a
**246x** ratio — the same qualitative result, different exact digits (see "numbers that required
reconciliation" below for why these two don't match to the decimal).

Across the full battery this is a *median* pattern, not a universal one: some single-token factual
cues (e.g. `fact_water`, `cf_capital`) already carry a large solo effect, because the answer's cue
genuinely is one token. Aggregating over all 29 gradeable Qwen2.5-7B cases: median best-single delta
**0.97** nats vs. median span delta **5.86** nats (span/single ratio: median **2.73x**, range
**1.0x-246x**); Llama-3.1-8B: median best-single **2.77** nats vs. median span **9.14** nats (ratio
median **1.66x**, range **1.07x-36.8x**). The span is never smaller than its own best single member
by construction (greedy search only adds positions that raise the joint effect), so the interesting
number is how much larger it gets, and in the most redundant cases (multi-token entities) that
multiplier reaches into the hundreds.

**Corroborating measurement — attention weight is not this causal ranking.** `attn_vs_causal.py`
asks whether the *correlational* signal every attention-heatmap product already shows (head-mean,
layer-mean post-softmax attention mass at the final position) agrees with the *causal* ranking above.
Across 8 prompt categories (Qwen2.5-7B only): mean Spearman rho **0.2183**, top-1 agreement **3/8**,
mean top-3 overlap **1.62/3**. The attention sink is the clearest illustration and also the clearest
counterexample to a blanket "sink is never causal" claim: it is attention-ranked in the top 3 in
5/8 cases, and in two of those its causal rank is far down the list despite high attention mass
(`distractor`: attention rank 0 [mass 0.4401] vs. causal rank 32 [delta -0.0536]; `arith`: attention
rank 0 [mass 0.5027] vs. causal rank 25 [delta -0.0807]) — but in two others (`long`, `doc`) the sink
is *both* attention-ranked and causally ranked near the top, with a real, sizeable causal delta
(`long`: +1.5359). The honest reading is not "the sink never matters," it is "attention mass and
causal effect are two different rankings that sometimes agree and sometimes don't, and a heatmap
alone cannot tell you which case you're in" — exactly the claim `provenance.py`'s docstring makes
("attention weight is correlational — it is not the same thing as influence").

**Receipt paths.** `runs/experiments/provenance_battery_qwen2.5-7b.json`,
`runs/experiments/provenance_battery_llama3.1-8b.json`, `runs/experiments/attn_vs_causal_qwen2.5-7b.json`;
pilot figures in `notes/CIRCUIT_TRACER_DESIGN.md` §5g-5h; method in `clozn/analysis/provenance.py`,
`scripts/tracer/provenance_battery.py`, `scripts/tracer/attn_vs_causal.py`.

---

## Instrument 2 — SAE features (joint-vs-random ablation on a discovered dictionary)

**Method.** `scripts/tracer/sae_fidelity_vs_concentration.py` and `sae_joint_vs_random.py`, on
Qwen2.5-7B-Instruct-Q4_K_M against the `andyrdt/Qwen2.5-7B` layer-15 JumpReLU SAE (131,072 features,
d_in 3584). At each prompt's causally-decisive site (the position — excluding the attention-sink
position 0, which is catastrophically out of distribution for this SAE: residual norm ~220x typical,
118,175 "active" features, explained variance -11,160) two questions are asked: (A) **causal
fidelity** — substitute the residual with the SAE's own reconstruction and re-score; `1 -
delta(substitute)/delta(mean_ablate)`; (B) **concentration** — ablate the top-k active features
*jointly* (canonical removal: `h - sum(a_f * d_dec[f])`), at k in {1,2,4,8,16,32,all}, against a
**random-k-of-the-active-set** control at the same k.

**Control.** The random-k-of-the-active-set arm in (B) is the decisive control: if top-k and
random-k produce the same joint delta, *which* features you remove carries no information and the
site is simply distributed with no special subset. This is a designed head-to-head, not a post-hoc
comparison, and it was added specifically to correct an earlier reading (see below).

**Result.** Median causal fidelity **99.9%** (range 96.1%-107.7%, occasionally over 100% because
substituting the reconstruction sometimes makes the answer *more* likely — the discarded variance
was mildly harmful noise) against a median **explained variance of only 57.2%** (range 43.9%-68.1%)
at the same sites — variance and causal function dissociate. No single feature is load-bearing:
top-1 concentration is approximately **0.5%** of a site's mass, indistinguishable from a random
single feature (e.g. induction k=1: top -0.18%, random -0.08%; factual k=1: top 0.68%, random
0.55%). But *groups* are real and super-additive: top-8 features jointly reach **10.5%** (induction)
to **44.9%** (distractor) of a site's causal mass, against **-0.4%** and **5.8%** respectively for
matched random-8; top-16 reaches **18.3%/16.2%** vs. **0.0%/0.7%** random. An earlier pass had
reported "top-12 features = 6.3% of the site's mass" by *summing single-feature deltas* — exactly
the sum-vs-joint error this project's interaction-gap finding (Instrument 4a) should have pre-warned
against — and the joint-ablation control above was built specifically to correct that reading.
Removing every active feature at a site still leaves **70-77%** of its causal mass intact; the
SAE's own reconstruction *error* term (`b_dec` plus residual) independently carries most of the
site's content (e.g. `b_dec` alone: +7.08 vs. the site's own mean-ablation effect of +3.08 on one
case — more than the "real" effect, because the bias term captures structure the dictionary's
sparse code does not).

**A gap in the receipts trail.** Unlike the other three instruments, this granularity's raw
per-prompt results are **not** committed to `runs/experiments/`. They exist only as narrative tables
in `notes/CIRCUIT_TRACER_DESIGN.md` §5e, plus ephemeral, non-versioned JSON written by the two
scripts above to a session-scratchpad directory outside the repository
(`sae_generalize.json`, `sae_joint.json`). Those scratchpad files still happen to exist on this
machine and were read directly to cross-check every number in this section against the committed
table — they match exactly — but they are not a durable repository artifact, and the next SAE run
should write to `runs/experiments/` like every other instrument here.

**Receipt paths.** `notes/CIRCUIT_TRACER_DESIGN.md` §5e (durable, committed); underlying scripts
`scripts/tracer/sae_fidelity_vs_concentration.py`, `scripts/tracer/sae_joint_vs_random.py`; raw
per-prompt JSON not preserved in the repository (see gap note above).

---

## Instrument 3 — attention heads (dual-intervention corroboration)

**Method.** `scripts/tracer/head_corroboration.py`, run on both families at three (layer, site)
combinations each (`factual`, `induction`, `kv` categories). Two **independent** interventions per
head, per site: (A) **output ablation** — `head_write` replaces head `h`'s `kqv_out` slice (the
per-head output, materialized before the `W_o` projection, per `notes/HEAD_UNITS_DESIGN.md` §1) with
its own mean slice, severing what the head *contributes*; (B) **edge knockout** — `attn_knockout`
with `head=h` zeroes that head's attention row into the site, renormalized, severing what the head
*reads*. If "head h matters here" is a real mechanism claim, the two measurements — contribution and
routing — should agree on which heads matter.

**Control.** A random-head-at-a-random-non-final-site arm, using the same ablation values: a claimed
head effect must separate from "any head, anywhere," not just from zero.

**Result.** The two interventions **disagree** at every site tested. Top-3-strongest-head overlap
between output-ablation and edge-knockout: **0/3** (Qwen, factual), **0/3** (Qwen, induction), **0/3**
(Qwen, kv); **0/3** (Llama, factual), **0/3** (Llama, induction), **1/3** (Llama, kv). Spearman
correlation between the two heads' full rankings ranges from **-0.32 to +0.41** across the six
site/family combinations (Qwen: -0.3153 factual, 0.1544 induction, 0.41 kv; Llama: 0.3919 factual,
0.187 induction, 0.0711 kv) — no reliable positive relationship. Separation from the random-site
control almost never clears the project's own 2x claim bar: Qwen peaks at **1.73x** (factual); Llama
peaks at **2.69x** but only once (kv site), an outlier not replicated at the other two Llama sites
(0.65x induction, 1.99x factual). Under the tracer's own house rule (a node must beat its matched
control by >=2x to be claimed at all), essentially no single head would ever earn a claim on this
evidence. `notes/HEAD_UNITS_DESIGN.md`'s verdict additionally reports "JOINT top-3 vs random-3
reaches only 1.31-1.95x" — **this specific figure could not be independently verified against the
stored receipt JSON**, which records only single-head ablation/knockout/control arms and has no
joint-3 field; it is reported here on the note's authority only, not re-derived from primary data.

**Receipt paths.** `runs/experiments/head_corroboration_qwen2.5-7b.json`,
`runs/experiments/head_corroboration_llama3.1-8b.json`; design and verdict in
`notes/HEAD_UNITS_DESIGN.md` (the VERDICT block at the end of the file, dated 2026-07-23); method in
`scripts/tracer/head_corroboration.py`.

---

## Instrument 4 — residual sites (mean-ablation tracer): two granularities

### 4a. Site sum-of-solos vs. joint (the interaction gap)

**Method.** `clozn/analysis/tracer.py`'s `trace()`. S0 nominates candidate (layer, position) sites
(via a J-lens projection screen or, for the second family, an any-GGUF mean-ablation grid screen).
S1 full-ablates each surviving candidate solo (`h <- the run's own mean residual at that layer`) and
records its own delta. S2 ablates **all** survivors simultaneously in one forward and records
`delta_total`. `accounting()` reports `interaction_gap = delta_total - sum_solo`, raw, never
normalized away. The validation battery (`scripts/tracer/causal_trace_battery.py`, 16 prompts across
`factual_easy`, `factual_distractor`, `in_context`, `syntactic`, `arithmetic`, `later_position`, and
deliberately-hard `low_margin` categories) additionally runs an **S4** tier: the graph publishes a
per-node flip *prediction* (from `delta_full` and the baseline top-1/runner-up margin) **before**
any greedy generation runs, then generation runs and the prediction is scored against what actually
happened.

**Control.** Noise floor = 3x the median absolute control delta, from random-equal-norm-direction
and random-site control arms; `FAILED_CONTROLS` fires if the strongest control matches or exceeds
the strongest claimed real effect, overriding any nominal survivor count.

**Result.** Interaction gap / sum-of-solos median **-60%** (Qwen3.5-9B, 16/16 cases, range
-80%..-3%) and **-82%** (Llama-3.1-8B, 16/16 cases, range -88%..-72%): summing every surviving site's
own solo delta overshoots the actual joint ablation effect by roughly **2.5x-5x**. Despite that, the
S4 predicted-vs-observed scorecard is strong on both families: **88/96 = 91.7%** correct
(Qwen3.5-9B: 88 correct, 8 wrong, 2 diverged-early, out of 27 predicted / 27 observed flips) and
**216/237 = 91.1%** correct (Llama-3.1-8B: 216 correct, 21 wrong, 5 diverged-early, 73 predicted / 75
observed flips). Verdicts: **16/16 PASS** on Qwen3.5-9B; **15/16 PASS + 1 FAILED_CONTROLS** on
Llama-3.1-8B — the guard is not hypothetical, it fired on a real prompt (`low_margin`, "My favourite
colour is", margin 1.16, 12 nominal survivors, 0 solid/strong/weak, delta_total +7.71 vs. sum_solo
+43.42, interaction gap -35.7).

**A caveat this instrument inherits.** `screen_null.py` tests whether the S0 site-nomination step is
answer-*specific* or merely finds generic salient structure: it traces the model's real greedy
answer against a plausible-but-wrong foil (Berlin/London for Paris, Mars for Jupiter, Fahrenheit for
Celsius, Ag for the gold symbol, "warm" for the opposite of hot, "five" for two-plus-two, Bush for
the first US president) on 8 factual prompts, under **absolute** scoring. Result: mean strong-node
count **9.25** (true answer) vs. **6.0** (null/wrong token); node-set Jaccard overlap **0.668**; 0/8
null traces returned anything but `PASS`. The receipt's own reading: *"MIXED/FAILS: null tokens
produce comparable structure — screen may be finding generic activity, not answer-specific
computation."* This motivated a **contrastive** scoring mode (`tracer.trace(contrast=...)`, delta =
change in the y-vs-foil logit gap rather than y's absolute logprob), committed separately
(`1edcecf`). The `causal_trace_battery` receipts analyzed above use **absolute** scoring throughout,
so this caveat about generic-vs-answer-specific site nomination applies to the 91.7%/91.1% numbers
above; the contrastive fix has not yet been re-run as its own full battery at this scale.

**Receipt paths.** `runs/experiments/causal_trace_battery_qwen3.5-9b-regression.json`,
`runs/experiments/causal_trace_battery_llama3.1-8b.json`, `runs/experiments/screen_null_qwen2.5-7b.json`;
method in `clozn/analysis/tracer.py`, `scripts/tracer/causal_trace_battery.py`,
`scripts/tracer/screen_null.py`.

### 4b. Position coalitions (the "molecules" program) — the fifth granularity

**Method.** `scripts/tracer/molecules.py` (2026-07-23, the newest result in this document).
Redefines "circuit" as the **smallest coalition of (layer, position) sites whose joint ablation
reaches 80% of the full-candidate-pool's joint effect**, built by the same greedy
marginal-gain construction that found the input spans in Instrument 1 — generalized from
*contiguous input positions under attention-edge severance* to an *arbitrary set of residual sites
under mean-ablation*. Three batteries on Qwen2.5-7B: (1) simple factual prompts, contrastive
scoring; (2) simple factual prompts, absolute scoring; (3) prompts specifically requiring
distributed retrieval (in-context KV lookup, induction, multi-hop) with the trivially-dominant final
readout position explicitly excluded from the candidate pool.

**Control.** Up to 8 random same-size coalitions drawn from the same candidate layers (not the
same, already-strong candidate positions); `separation_vs_random_k = |min_set_delta| /
max(random draws)`.

**Result.** (1) Simple factual, contrastive: minimal-set size **1** in every case — a single readout
site carries the whole answer-*selective* signal — mean separation **1.0x** (0/5 cases clear 2x);
adding scaffolding sites can even *reduce* the selective gap (`frac_of_full` exceeds 1 in every case,
e.g. 4.99, 8.29 — contrastive ablation is non-monotonic). (2) Simple factual, absolute: minimal-set
size **1**, mean separation **1.43x** (0/5 clear 2x) — the final/readout position dominates by
construction, since it is always where the prediction is computed. (3) Distributed cases, readout
excluded, absolute: mean minimal-set size **1.67**, mean separation **0.93x** (0/6 clear 2x, and the
mean sits *below 1.0* — the greedily-optimized coalition on average does not even beat the strongest
of 8 random same-size, same-layer coalitions).

This is the one place in this document where the "the atom is wrong, build a set" move — which
worked for spans (Instrument 1), for SAE feature groups (Instrument 2), and for whole-trace joint
ablation (Instrument 4a) — **does not rescue the result**. At (layer, position) granularity, under
residual mean-ablation, there is no privileged small coalition: targeted small sets are
indistinguishable from random small sets of the same size and layer pool. The module's own verdict
(`scripts/tracer/molecules.py`, lines 213-241) states this as a measured negative, not a build
failure, and immediately draws the methodological conclusion that motivates the next section: the
one thing in this whole program that *did* beat its controls by 100x or more — the greedy input
span — is a different instrument (attention-edge severance of contiguous input tokens), not this one
(residual mean-ablation of an arbitrary position set).

**Receipt paths.** `runs/experiments/molecules_qwen2.5-7b.json` (contrastive, simple factual),
`runs/experiments/molecules_qwen2.5-7b-abs.json` (absolute, simple factual),
`runs/experiments/molecules_qwen2.5-7b-kv.json` (absolute, distributed cases, readout excluded);
method + VERDICT block in `scripts/tracer/molecules.py`.

---

## The positive half: what IS real (sets, not atoms)

The negative results above are not the whole story, and a document that only listed failures would
misrepresent the receipts. Four things measured cleanly and repeatably:

1. **Greedy contiguous input spans under attention-edge severance.** The one clean, large,
   unambiguous win in this entire program. Ratios over matched random-position-set controls reach
   into the hundreds and beyond: 793x and 246x within the 30-case battery (kv_late on both passes),
   2232x and 1216x in the pilot table (`notes/CIRCUIT_TRACER_DESIGN.md` §5h, kv and induction cases),
   and several battery cases into the tens-of-thousands (e.g. `arith_sum` 14,705x, `doc_town` on
   Llama 14,203x, `dis_edison` 244,950x). This is Instrument 1, and it is the load-bearing positive
   result for the whole "did the model actually use this context" (RAG provenance) product surface.

2. **Joint SAE feature groups beating random-k cleanly.** Instrument 2: top-8/16 features jointly
   reach 10-45% of a site's causal mass against a 0-6% random-group floor, with the effect strongly
   super-additive (joint far exceeds the sum of the same features' solo deltas).

3. **Joint site ablation, cross-family S4 prediction scorecards.** Instrument 4a: 91.7% (Qwen3.5-9B)
   and 91.1% (Llama-3.1-8B) of pre-published flip predictions were confirmed by an actual greedy
   generation run after the fact — a real, cross-family, predicted-before-observed result, even
   though the same instrument's sum-of-solos accounting shows the underlying interaction gap is
   large.

4. **Shipped coalition/Shapley credit machinery**, built directly in response to the interaction-gap
   finding: `clozn/receipts/coalition.py`. It never reports a leave-one-out (solo) delta alone —
   every report also computes pairwise coalitions, one full joint arm, and (for N<=4 influences) an
   **exact Shapley value** over the complete 2^N power set, or for larger N a documented
   **Shapley-Taylor second-order approximation** (Sundararajan, Dhamdhere & Agarwal 2020) with a
   bootstrap confidence interval on the interaction term, carrying an explicit `OVERCOUNTING_CAVEAT`
   quoting this project's own -60%-median finding verbatim. This is unit-tested product code that
   *operationalizes* the interaction-gap finding — it is not itself a new experimental battery, and
   is cited here as evidence the finding changed how the codebase attributes credit, not as an
   additional receipt.

5. **The provenance verdict system itself** (CONTEXT_CARRIED / MIXED / PARAMETRIC) is, by
   construction, a set-level (span) measurement, and its own validation battery is the subject of
   this document's first reconciliation note below: the frequently-quoted "41/41 two-family
   agreement" headline is reproducible from the stored receipts, but not by reading the summary
   block already written into the JSON files — see the next section.

---

## The methodological seam: a sharp instrument and a blunt one

The clearest single finding cutting across all five granularities above is not "everything is
distributed" — it is that **which instrument you use determines whether you find localized
structure at all**, and this project has direct, dated history showing why.

`notes/CIRCUIT_TRACER_DESIGN.md` §5f-5g documents that the ORIGINAL plan — cross-position path
patching on the residual stream — was structurally blocked before attention-edge severance was ever
tried: patching one residual site cannot hold a cross-position path, because the un-ablated source
position re-supplies the effect at every later layer, and `llama.cpp`'s `inp_out_ids` optimization
materializes only the logit rows at the final layer, leaving one layer permanently unpatchable. The
measured symptom was a flat **0.0% routed fraction** for every cross-position edge tried, "which is
physically impossible as a true effect size" (§5f) given the final position's residual is the *only*
route to the output. The fix — sever the attention *edge* instead of patching the residual *site* —
is what made Instrument 1 (and the 41/41-reproducible provenance battery) possible at all.

This cycle's molecules result (Instrument 4b) shows the same seam persists even after generalizing
from single positions to coalitions: a greedily-optimized *set* of residual sites, under
mean-ablation, still does not beat a random same-size set (separation 0.93x-1.43x, never clearing
2x) — while the *same greedy-set-construction idea* applied to attention-edge severance of
contiguous input tokens clears its control by two to six orders of magnitude. Put plainly: **on
these models, "a circuit" — something that beats a matched random control by a wide, repeatable
margin — lives on the input attention edges, not on the residual stream's sites.** Every instrument
in this document that intervenes on attention edges (Instrument 1, and the edge-knockout half of
Instrument 3) finds cleaner separation than every instrument that intervenes on residual sites
(Instruments 2, 4a, 4b), even when the SAME site or the SAME kind of aggregation (solo -> joint,
single -> greedy set) is tried on both. This is a claim about what these two specific interventions
measure on this specific engine and these specific models — not a general claim that residual-stream
interpretability is impossible, and not yet a mechanistic explanation of *why* the edges carry more
structure than the sites (see Open Questions).

---

## Honest scope limits

- **Two vendor families, three checkpoints, not evenly used.** Qwen2.5-7B-Instruct-Q4_K_M is the
  model for Instruments 1 (provenance battery + attn-vs-causal), 2 (SAE), 3 (heads), and 4b
  (molecules). Llama-3.1-8B-Instruct-Q4_K_M is the second family for Instruments 1, 3, and 4a.
  Qwen3.5-9B (a different generation and size) is the *first* family for Instrument 4a's cross-family
  claim — there is no Qwen2.5-7B causal_trace_battery receipt in the repository. Nothing here should
  be read as "the same four models tested identically four ways"; build the per-instrument model
  table below before citing any cross-instrument comparison.

  | Instrument | Family A | Family B |
  |---|---|---|
  | 1. Positions (provenance) | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct |
  | 1. Attention-vs-causal | Qwen2.5-7B-Instruct | *(single family only)* |
  | 2. SAE features | Qwen2.5-7B-Instruct | *(single family only)* |
  | 3. Attention heads | Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct |
  | 4a. Site sum-of-solos / S4 | Qwen3.5-9B | Llama-3.1-8B-Instruct |
  | 4b. Position coalitions | Qwen2.5-7B-Instruct | *(single family only)* |

- **One quantization scheme, applied consistently, but not the only 4-bit scheme this project has
  used.** Every number above is measured on the model exactly as served: Q4_K_M, llama.cpp's k-quant
  format. `notes/JLENS_SAE_FINDINGS.md` finding #8 documents that other work in this project (the
  J-lens sidecar) was fit against a *different* 4-bit scheme (`nf4` via bitsandbytes) — a reminder
  that "4-bit" is not one thing, and that this document's quantization scope is specifically Q4_K_M
  on the served engine, not a general claim about 4-bit quantization.
- **7-9B instruction-tuned chat models only.** No base/pretrained models, no models under 7B or over
  9B, are part of this specific claim. (Other threads in this project — the peek-ahead-steering and
  CoT-shortcut null results in `notes/JLENS_SAE_FINDINGS.md` — touch a 0.5B model, but that is a
  separate research question with its own null results, not evidence for or against this document's
  claim.)
- **Small batteries; every percentage is a point estimate.** 30 cases x2 families (provenance), 16
  cases x2 (different) families (causal_trace_battery), 8 cases (attn_vs_causal,
  head_corroboration, screen_null), 5-6 cases (molecules, SAE). None of these support a confidence
  interval on a rate; read every percentage in this document as "measured on this small battery,"
  not as an asymptotic property of the model.
- **A specific, narrow behavior mix.** Factual recall, in-context key/value lookup, induction,
  counterfactual override, simple arithmetic, negation, long-context retrieval, and deliberately
  diffuse/low-margin cases ("my favourite colour is"). No open-ended generation, no multi-turn
  dialogue, no tool use, no code generation, no long-form reasoning chains are covered by any
  instrument in this document.
- **"Distributed" is a description of a measurement pattern, not yet a mechanism.** No instrument
  here establishes *why* function is distributed — not redundancy from training dynamics, not
  superposition, not an artifact of Q4_K_M quantization specifically, not GQA/RoPE architecture
  choices. Every "why" in this document is the project's own hypothesis language, clearly marked as
  such, never a measured result.
- **This is not a claim about GPT-2-small, classic mechanistic-interpretability testbeds, base
  models, or models >=13B.** It is a claim about these four checkpoints, these four instruments,
  this small battery of prompts. Generalizing further is exactly the kind of overclaim this
  project's own house style (visible in `notes/HEAD_UNITS_DESIGN.md`'s and `molecules.py`'s verdict
  blocks) explicitly refuses to make, and this document follows the same discipline.

---

## Open questions

- **Heads:** k-sweeps beyond 3 were never run (SAE joint structure only appeared at k=8-16; heads
  were tested jointly only up to k=3, and even that figure is not in the stored receipt — see the
  reconciliation note below). Other layers per site, value-zeroing vs. mean-ablation, and head roles
  at non-answer positions are all untested (`notes/HEAD_UNITS_DESIGN.md` §"left open").
- **Position coalitions:** the module's own next step is to try the coalition-construction idea on
  attention-EDGE sets rather than residual-site sets — exactly where the methodological seam above
  says the structure should live — plus larger k and other layers (`scripts/tracer/molecules.py`,
  final comment block: "any revival starts from a coalition that beats its control — this one does
  not").
- **Contrastive scoring at battery scale:** the fix for `screen_null`'s generic-vs-specific caveat
  (contrastive scoring, `1edcecf`) exists in `clozn/analysis/tracer.py` but has not been re-run as a
  full battery the way the absolute-scoring `causal_trace_battery` has. The 91.7%/91.1% S4 numbers
  in this document are absolute-scoring and inherit the screen_null caveat about site nomination.
- **Focus-mode provenance** (scoping the RAG question to one document region rather than "any
  context at all") is explicitly marked experimental in `clozn/analysis/provenance.py`'s own
  docstring, validated on one live case per direction, not yet its own battery.
- **The SAE granularity has no persisted repository receipt.** Fix this before citing Instrument 2
  externally: re-run `sae_fidelity_vs_concentration.py` / `sae_joint_vs_random.py` writing to
  `runs/experiments/` like every other instrument in this document.
- **Why edges and not sites?** No instrument here explains the mechanism behind the methodological
  seam — only that it is measured, twice, at two different granularities (single positions in §5f-5g
  of the design doc; position coalitions in Instrument 4b of this document).

---

## Numbers that required reconciliation (report honestly)

Four figures in the sources for this document did not match on first read. All four are reported
here with the reconciliation, not silently corrected or silently dropped.

1. **"41/41 two-family agreement"** (`notes/RETROSPECTIVE_2026-07.md`, `provenance.py`'s
   `SCOPE_NOTE`, multiple session logs) is **not** what the two `provenance_battery_*.json` files'
   own stored `summary` blocks report: `provenance_battery_qwen2.5-7b.json` reports **23/23**
   (100%) agreement, and `provenance_battery_llama3.1-8b.json` reports **21/26** (80.8%) with 5
   listed disagreements (`cf_capital` plus all four `distractor_parametric` cases). Re-grading the
   *same stored per-case answers and verdicts* against the grading closures currently defined in
   `scripts/tracer/provenance_battery.py` (which — per that file's own docstrings — now treats the
   four `distractor_parametric` cases and any `forbid_answer`-triggered counterfactual resistance
   as `descriptive` rather than graded, changes made *because of* this very battery's results)
   reproduces exactly **20/20 (Qwen) + 21/21 (Llama) = 41/41**. The "41/41" headline is real and
   reproducible from the receipts — but only by re-applying the current grading code to the stored
   per-case data, not by quoting the `summary` block physically stored inside either JSON file,
   which reflects an older grading pass and is now stale. (Verified directly: a small script loading
   `scripts.tracer.provenance_battery.CASES` and re-grading each stored `{answer, verdict}` pair
   reproduces 20/20 and 21/21 exactly, reported above.)
2. **"Best single +0.11 vs. greedy span +7.60"** (`clozn/analysis/provenance.py` module docstring,
   `notes/CIRCUIT_TRACER_DESIGN.md` §5h) is from the 1-family, 4-prompt **pilot**, not from the
   30-case battery JSON. The battery's `kv_late` case — the same prompt family, re-measured later —
   shows best single **0.030** vs. span **7.304** (246x). Same qualitative pattern, different exact
   digits; cited separately above rather than conflated.
3. **"JOINT top-3 vs random-3 reaches only 1.31-1.95x"** (`notes/HEAD_UNITS_DESIGN.md`'s VERDICT
   block) has no corresponding field in either `head_corroboration_*.json` receipt, which records
   only single-head ablation, single-head knockout, and single-head random-site-control arms — no
   joint-3 arm was persisted. Reported above on the note's authority only, flagged as unverifiable
   against the primary data artifact.
4. **"Up to -73% on a larger model"** (`clozn/receipts/coalition.py`'s `OVERCOUNTING_CAVEAT` and
   module docstring) vs. this document's own aggregation of `causal_trace_battery_llama3.1-8b.json`
   (the larger-model receipt), which gives a median interaction-gap ratio of **-82%** across its 16
   cases (range -88%..-72%). Same direction, same rough magnitude — -73% sits at the shallow edge of
   the measured range rather than matching the median — most likely written from an earlier or
   partial pass of the same battery. Not corrected in `coalition.py` by this document; flagged here
   for whoever next touches that file.

---

## Receipts

| # | Path | Instrument / granularity | One-line content |
|---|---|---|---|
| 1 | `runs/experiments/provenance_battery_qwen2.5-7b.json` | 1. Positions | 30-case provenance battery, Qwen2.5-7B; stored summary 23/23 agree (see reconciliation #1) |
| 2 | `runs/experiments/provenance_battery_llama3.1-8b.json` | 1. Positions | 30-case provenance battery, Llama-3.1-8B; stored summary 21/26 agree, 5 disagreements (see reconciliation #1) |
| 3 | `runs/experiments/attn_vs_causal_qwen2.5-7b.json` | 1. Attention-vs-causal | 8-case attention-heatmap-vs-causal-rank head-to-head, Qwen2.5-7B only |
| 4 | `notes/CIRCUIT_TRACER_DESIGN.md` §5e | 2. SAE features | Committed narrative table: causal fidelity 99.9%/explained variance 57.2%, top-k joint-vs-random table |
| 5 | *(not in repo — ephemeral scratchpad `sae_generalize.json`, `sae_joint.json`)* | 2. SAE features | Raw per-prompt SAE data underlying §5e; cross-checked for this document, matches exactly, not a durable artifact |
| 6 | `runs/experiments/head_corroboration_qwen2.5-7b.json` | 3. Attention heads | 3-site dual-intervention (output-ablation vs. edge-knockout) battery, Qwen2.5-7B |
| 7 | `runs/experiments/head_corroboration_llama3.1-8b.json` | 3. Attention heads | Same, Llama-3.1-8B |
| 8 | `notes/HEAD_UNITS_DESIGN.md` | 3. Attention heads | Design + VERDICT (2026-07-23): NEGATIVE at the claim bar, do not ship head nodes |
| 9 | `runs/experiments/causal_trace_battery_qwen3.5-9b-regression.json` | 4a. Site sum-of-solos / S4 | 16-case causal-trace battery, Qwen3.5-9B; interaction gap median -60%, S4 91.7% |
| 10 | `runs/experiments/causal_trace_battery_llama3.1-8b.json` | 4a. Site sum-of-solos / S4 | Same, Llama-3.1-8B; interaction gap median -82%, S4 91.1%, 1 FAILED_CONTROLS |
| 11 | `runs/experiments/screen_null_qwen2.5-7b.json` | 4a. Screen-specificity caveat | 8-case true-vs-null-token trace comparison; reading = MIXED/FAILS under absolute scoring |
| 12 | `runs/experiments/molecules_qwen2.5-7b.json` | 4b. Position coalitions | Simple factual, contrastive scoring; min-set 1, separation 1.0x |
| 13 | `runs/experiments/molecules_qwen2.5-7b-abs.json` | 4b. Position coalitions | Simple factual, absolute scoring; min-set 1, separation 1.43x |
| 14 | `runs/experiments/molecules_qwen2.5-7b-kv.json` | 4b. Position coalitions | Distributed (KV/induction/multi-hop), readout excluded; mean min-set 1.67, separation 0.93x |
| 15 | `clozn/analysis/provenance.py` | 1. Method | `trace_provenance()`, renormalize + greedy-span fixes, contrastive verdict thresholds |
| 16 | `clozn/analysis/tracer.py` | 3-4. Method | `trace()`, S0-S4 pipeline, contrastive scoring, accounting/interaction-gap, controls_verdict |
| 17 | `scripts/tracer/provenance_battery.py` | 1. Method | The 30-case battery + grading closures (source of reconciliation #1) |
| 18 | `scripts/tracer/attn_vs_causal.py` | 1. Method | Attention-vs-causal script |
| 19 | `scripts/tracer/head_corroboration.py` | 3. Method | Dual-intervention head script |
| 20 | `scripts/tracer/screen_null.py` | 4a. Method | True-vs-null-token screen specificity script |
| 21 | `scripts/tracer/molecules.py` | 4b. Method | Coalition-construction script + VERDICT block (lines 213-241) |
| 22 | `scripts/tracer/causal_trace_battery.py` | 4a. Method | 16-case S1/S2/S4 battery + aggregate scorecard printer |
| 23 | `scripts/tracer/sae_fidelity_vs_concentration.py` | 2. Method | Causal fidelity vs. concentration script |
| 24 | `scripts/tracer/sae_joint_vs_random.py` | 2. Method | Joint-vs-random-k feature ablation script (the control that could have overturned the finding) |
| 25 | `clozn/receipts/coalition.py` | Positive half | Shipped Shapley/coalition credit machinery built in response to the interaction-gap finding |
| 26 | `notes/CIRCUIT_TRACER_DESIGN.md` §5f-5g | Methodological seam | Cross-position residual patching blocked (0.0% routed fraction); the fix that led to attention-edge severance |
| 27 | `notes/RETROSPECTIVE_2026-07.md` | Context | Cycle retrospective; source of the "41/41" and "distributed function" framing this document verifies |
| 28 | `notes/JLENS_SAE_FINDINGS.md` finding #8 | Scope note | nf4-vs-Q4_K_M quantization-scheme distinction |
