# quine — does a self-state readout help a model predict itself? (findings)

*Wild Experiment #9, Wave 1. Pre-registration: `WILD_WAVE1_PREREG.md` (exp 9 + Amendment 1). Run
2026-07-05. Qwen2.5-7B-Instruct (nf4); Gemma-2-9B (nf4) for the dial-label cross-family point. Rig:
`research/quine.py`; runs: `research/runs/quine_{qwen7b,gemma9b}.json`.*

## TL;DR — the verdict (both families in)

**No robust, SELF-SPECIFIC benefit from a self-state readout — the project's core thesis, put to a
direct test, comes back negative.** The question was whether an instrument can make the interior legible
*to the model itself*. Across both families the answer is a **Law-1-consistent negative**:
- The **white-box SAE self-view gives ZERO lift** (Qwen; the on-thesis variant — a clean no).
- The human-readable **dial-label's weak effect is NOT self-specific**: on Gemma a *shuffled* (wrong)
  label helps exactly as much as the true one (both 1.0 on the shifted subset vs 0.667 no-state), i.e.
  it's generic "think about your state" priming, not the model reading its actual state.
Both runs are underpowered (20–27% steering-shift rate; tiny shifted subsets n=6–8) and Gemma's forced-
choice parse-fail is high and condition-dependent — so this is "no evidence of a real self-specific
effect," with the SAE-zero the one unambiguous piece. **Does not support building a "show the model its
own state" feature.**

## The design (why the measurement is trustworthy even when the effect isn't)

Forced-choice behavioral self-prediction, per Amendment 1 — and notably **no LLM judge**: the ground
truth is the model's own logprobs.
1. Steer the model into a known state S (one stance dial, random dose off its calibrated ceiling).
2. **Ground truth** = which of two continuations (state-congruent vs -incongruent) the STEERED model
   assigns higher mean-per-token logprob — its honest behavior, no meta-prompt. Objective.
3. Ask the SAME steered model, forced-choice, which it'd produce — under four readouts that differ ONLY
   in the prepended text: **dial-label** ("you're steered <axis>"), **no-state** (twin), **shuffled-state**
   (a *different* axis's label), **SAE-feature** (its top-k firing SAE features + Neuronpedia labels;
   Qwen-only — the andyrdt SAE, layer 15, 103,491 labels).
4. Metric: self-prediction accuracy vs ground truth. Reads: dial-label > no-state → an explicit
   self-description helps; SAE > no-state → the model can read its *own features*; shuffled ≈ dial →
   no self-specific signal.

## Qwen-7B result

**Steering shifted the ground-truth preference in only 26.7% (8/30) of trials** — the key limitation.
When steering doesn't move the model's preference between the two continuations, there is no distinct
steered state to predict, and the trial can't test self-knowledge. So the honest analysis is the
**shifted-only subset** (n=8), reported beside the pooled numbers.

| condition | accuracy (all 30) | accuracy (shifted-only, n=8) |
|---|---|---|
| **dial-label** | 0.633 | **0.750** |
| no-state (twin) | 0.600 | 0.625 |
| shuffled-state | 0.600 | 0.625 |
| **SAE-feature** | 0.600 | **0.625** |

**What it says:**
- **The SAE-feature prosthesis gives zero benefit** — 0.625 = no-state, on the clean subset *and*
  pooled. The model cannot read its own top SAE features to predict itself better than nothing. This is
  the novel, on-thesis claim, and it's a clean **no**.
- **The dial-label shows only a non-significant hint** — +0.125 over no-state on the shifted subset, and
  it beats shuffled-state (0.75 vs 0.625), which is the *direction* Amendment 1 predicts for a real
  effect. But n=8 means that gap is a single trial (6/8 vs 5/8), well inside the Wald noise (~±0.17).
  Pooled, it collapses to +0.033. So: suggestive, not established.

## Gemma-2-9B — a bigger readout effect, but NOT self-specific

Dial-label / no-state / shuffled only (SAE is Qwen-only). Steering-shift rate 20% (6/30). **Caveat that
shapes everything below:** the forced-choice parse-fail is high AND condition-dependent (dial-label 20%,
no-state 10%, shuffled 27%), so the pooled accuracies below rest on *different* denominators (24/27/22
decided) and aren't a clean apples-to-apples comparison.

| condition | accuracy (all 30) | accuracy (shifted-only, n=6) |
|---|---|---|
| **dial-label** | 0.958 | **1.000** |
| no-state | 0.852 | 0.667 |
| **shuffled-state** | 0.864 | **1.000** |

The rig's auto-verdict flagged "explicit self-description HELPS" (dial +0.106 over no-state pooled). But
the shifted-only subset — the clean test — shows the giveaway: **shuffled-state (a *wrong* axis label)
scores 1.0, exactly like the true dial-label**, both above no-state's 0.667. So on Gemma, *any* "you are
in a state that makes you answer more X" preamble helps the model answer more carefully — a generic
self-referential-priming effect — and the TRUE label carries **no self-specific advantage** over a false
one. That is the opposite of what "the model is reading its actual state" predicts.

## Cross-family synthesis

| claim | Qwen-7B | Gemma-9B |
|---|---|---|
| SAE white-box self-view helps | **no** (0 lift) | not testable (no cached SAE) |
| dial-label helps over no-state | faint (+0.125, n=8, n.s.) | yes-ish (+0.106, but high parse-fail) |
| effect is SELF-SPECIFIC (true > shuffled) | dial≈shuffled pooled; weak | **no** (shuffled = true, both 1.0) |

The two families agree on the load-bearing point: **whatever weak readout effect exists is not
self-specific**, and the one genuinely white-box variant (SAE) does nothing. A generic "reflect on your
state" prompt would reproduce the entire observed effect without any measured self-knowledge.

## Product read

The feature this experiment was probing — **show the model its own measured state so it self-predicts
better** — is **not supported.** The SAE variant (the version that would actually use clozn's white-box
instrument) gives **zero lift**. And the human-readable dial-label's weak effect is **not self-specific**
(Gemma: a wrong label helps as much as the true one), so it's generic self-referential priming — a plain
"reflect on how you're answering" prompt would reproduce it without any measured self-knowledge. There is
no white-box-specific win here to build on. This is the thesis being honest with itself: Law #1 says
process is hard to self-report, and handing the model a *measured* readout of that process — even its own
SAE features — did not lift the blindness.

## Caveats, louder than the (non-)result

The dominant one: **underpowered by a 26.7% steering-shift rate.** Only 8 trials created a real steered
state — partly because 3/5 stances barely steer (parliament's 2/5-live finding) and coherent doses don't
reliably flip a logprob preference. A higher-N run, or one biased toward the live axes (warm, skeptical)
at stronger doses, would firm up the dial-label hint — the clearest follow-up. Also: (2) the
congruent/incongruent pair differs in CONTENT as well as tone, so the forced-choice is partly a content
preference, not pure self-knowledge (the unsteered-baseline diagnostic mitigates but doesn't fix this);
(3) one seed, greedy; (4) forced-choice is a coarse proxy for a full next-token distribution (Amendment
1's own stated ceiling); (5) SAE-feature had no dedicated shuffled-SAE null — its zero-lift is measured
against no-state and the shuffled-dial-label, a rougher control.
