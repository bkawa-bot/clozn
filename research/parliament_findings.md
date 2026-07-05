# parliament — a parliament of steered stances vs the nulls, cross-family (findings)

*Wild Experiment #4, Wave 1. Pre-registration: `WILD_WAVE1_PREREG.md` (exp 4). Run 2026-07-05.
Qwen2.5-7B-Instruct (nf4) subject, Gemma-2-9B judge (and vice versa) — a genuinely INDEPENDENT
cross-family judge. Reduced first pass: 12 questions (of the 30-question bank). Rig:
`research/parliament.py`; runs: `research/runs/parliament_{qwen7b,gemma9b}.json`.*

## TL;DR — the verdict (both families in)

**No reliable evidence the parliament helps — and the two families CONTRADICT each other, which is
itself the tell.** The objective-ish coverage rubric can't separate any arm on *either* family (judge
flagged SUSPECT both times). The finer pairwise signal points *opposite ways*: on Qwen the parliament
beats a single decode (0.917) but ties the random-direction null; on Gemma it *ties* a single decode
(0.455) but *beats* the random null (0.9). An effect that flips direction between models and never
clears the objective rubric is noise, not a feature. (One thing that did replicate: diff-of-means
steering registers on Gemma-2 — 2/5 stances live, same as Qwen — so softcapping didn't kill it.)
**Recommendation: do not ship it.**

## Qwen-7B — the two metrics tell a two-part story

Judge = Gemma-2-9B (cross-family, independent). Parse-fail dropped to 0–8% (vs the smoke's 50–75% with
a 1.5B judge), so the coverage numbers are trustworthy.

**Coverage rubric (primary, objective-ish — % of pre-written required points hit):**

| arm | coverage | degen% | parse-fail% |
|---|---|---|---|
| parliament | 75.0 | 20% | 8.3% |
| single (floor) | 76.8 | 8.3% | 8.3% |
| temp-vote null | 77.1 | 15% | 0% |
| shuffled-dial null | 77.1 | 15% | 0% |

All four **tied (~76%)**. The rubric is near-ceiling for every arm and cannot separate them. By the
rig's own null-calibration the judge is flagged **SUSPECT** — it can't distinguish parliament from
random-direction steering.

**Pairwise preference (softer, but finer — head-to-head A/B, position-randomized):**

| matchup | parliament winrate |
|---|---|
| parliament vs **single decode** | **0.917** (won 11/12) |
| parliament vs **temp-vote null** | 0.727 (8/3/1) |
| parliament vs **shuffled-dial null** | **0.500** (6/6 — dead tie) |

## What it means

Reconciling the two metrics:

1. **Ensembling is real, but modest and metric-fragile.** Merging 5 drafts beats 1 draft handily in
   head-to-head preference (0.917) — plausibly because 5 drafts give the merge more material to union
   into a more complete answer. But the objective coverage rubric shows *zero* separation, so the win
   lives entirely in the softer pairwise signal, which is contaminated by length/thoroughness bias and
   the judge-writes-then-scores coupling (below).
2. **The stances add nothing detectable.** Parliament **ties the shuffled-dial null** (0.5 pairwise,
   tied coverage). Five *random* directions merge as well as five real stances — so the *directedness*,
   the entire "parliament of viewpoints" idea, carries no measured signal. Parliament *does* beat the
   temp-vote null (0.727), i.e. activation-space diversity beats thermal (temperature) diversity for the
   merge — a mildly interesting mechanism note, but not the hypothesis.

**The honest confound that keeps this from being a clean "no":** only **2/5 stances registered as live
steering** on Qwen (the rig's own liveness check; the abstract stances candid/concrete/plain barely
moved the text by diff-of-means). So directedness was *under-powered* — a "parliament" of 2 real
stances + 3 near-unsteered decodes is, unsurprisingly, close to a random-perturbation ensemble. This
run cannot cleanly separate "directed diversity doesn't help" from "our stances didn't steer hard
enough to *be* directed." A better-calibrated steering pass might rescue the directedness — but see the
product read below on why that likely doesn't change the answer.

**The coupling risk (stated in the rig, load-bearing here):** the SAME judge model both MERGES the 5
drafts into one answer AND SCORES it. If it's simply a better writer synthesizing 5 varied inputs than
1, that biases toward parliament for reasons unrelated to steering — and one judge can't rule it out.
The nulls are the guardrail: because shuffled ≈ parliament, we do NOT over-read the parliament > single
result as "stances help."

## Gemma-2-9B — and it contradicts Qwen

Judge = Qwen-2.5-7B (cross-family). Note the parse-fail asymmetry: Qwen-as-judge fails the strict
rubric-bit format **41.7%** of the time (vs Gemma-as-judge's 0–8% on the Qwen leg) — so Gemma's
coverage numbers rest on ~7 of 12 questions and are noisier; the pairwise leg (0% parse-fail) is the
more trustworthy signal here.

**Coverage rubric:**

| arm | coverage | degen% | parse-fail% |
|---|---|---|---|
| parliament | 71.4 | 6.7% | 41.7% |
| single (floor) | 67.9 | 0% | 41.7% |
| temp-vote null | 67.9 | 0% | 41.7% |
| shuffled-dial null | 71.4 | 0% | 41.7% |

Parliament ties the shuffled null (71.4 = 71.4) → judge SUSPECT again.

**Pairwise preference (the reliable leg here, 0% parse-fail):**

| matchup | parliament winrate | vs Qwen |
|---|---|---|
| parliament vs **single decode** | **0.455** (5/6/1 — basically lost) | Qwen won 0.917 |
| parliament vs temp-vote null | 0.500 (5/5/2) | Qwen 0.727 |
| parliament vs **shuffled-dial null** | **0.900** (9/1/2 — big win) | Qwen tied 0.500 |

**The cross-family contradiction is the finding.** On Qwen the parliament's value was ensembling (beat
single) with the stances adding nothing (tied shuffled). On Gemma it is the *exact inverse*: the stances
DID beat random directions (0.9), but the whole parliament did NOT beat a single decode (0.455). The two
families disagree on **both** load-bearing comparisons. Whatever small effects exist are
**family-dependent and judge-dependent**, not a robust property — which is the cleanest possible reason
not to build on them. Coverage can't separate the arms on either side, and the judge is SUSPECT on both.

One sub-question got a clean answer: **diff-of-means steering works on Gemma-2** (2/5 stances live,
identical to Qwen; Gemma's ~5× larger residual norm was auto-recalibrated) — the attention softcapping
did not break steering. That's a useful cross-family instrument fact, independent of the parliament verdict.

## Product read — not worth the complexity

Independent of how the Gemma leg lands, the feature case is weak:
- The user-facing story ("consult a panel of *perspectives*") is exactly the part with **no evidence** —
  random directions do as well as real stances.
- What remains is **generic ensembling** (sample-N-then-merge), which needs none of clozn's white-box
  steering machinery — plain temperature sampling + a merge gets you most of it — and costs **5–6×
  compute per reply** (5 steered decodes + a merge call) for a gain visible only in a **biased LLM
  judge's preference** (the objective rubric saw nothing).
- It cuts against clozn's differentiator: the project sells *measured* legibility (receipts you can
  verify), and "a judge model liked this one more" is the self-narration the project is built to
  distrust.

**Deciding NOT to ship, on the receipts, is the project's ethos working as intended.** Recorded as a
finding; not promoted to a feature.

## Caveats, louder than the (non-)win

One seed; reduced N=12 (not the full 30); nf4 (quantization confounded with family); one judge model
per direction (merges AND scores — the coupling above); only 2/5 stances live on Qwen (directedness
under-powered); coverage rubric near-ceiling so blind to fine differences; the "batched decode is ~free"
economic premise that motivates the whole idea was **NOT tested** (all K decodes ran sequentially).
