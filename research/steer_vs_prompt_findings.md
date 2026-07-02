# steer_vs_prompt — dose control & show-it transfer: dials vs prompts (findings)

**Question.** The last open cell of the say/show/train scorecard: (A) does a steering dial give finer,
more monotone **dose control** than graded prompt wordings? (B) for a style defined **only by examples**,
does a diff-of-means dial transfer the style without the content-bleed that few-shot prompting suffers?

**Pre-registered:** dial = finer/more monotone dosing; few-shot = stronger transfer but topic bleed;
dial = ~zero bleed. **Outcome: 1 of 3 held.** Dose control went to the PROMPT; bleed went to the dial
exactly as predicted; transfer strength was a wash (different failure modes).

**Setup.** Qwen2.5-1.5B bf16, `SteeringControl` at layer 14 (base auto-calibrated: 0.85·‖resid‖ = 48.0).
A: axes `concise` (scored: words — objective) and `warm` (scored: warm-marker rate incl. '!' — crude,
transparent); 5 levels each — prompts none/slightly/moderately/very/extremely vs dial 0/0.35/0.7/1.05/1.4;
8 neutral probes, greedy. B: 4 terse/vivid/no-hedge example replies vs 4 hedgy/rambling ones (SAME topics
both poles → the contrast isolates style); few-shot gets both sets in-prompt (information parity); the dial
direction = mean-pooled layer-14 diff over the same texts; 6 probes on disjoint topics.
Repro: `python research/steer_vs_prompt.py`.

## A — dose control: the PROMPT wins (pre-registration overturned)

| axis | mechanism | curve (level 0→4) | Spearman | inversions |
|---|---|---|---|---|
| concise (words) | **prompt** | 89 → 42 → 50 → 35 → **7** | **0.9** | 1 |
| concise (words) | dial | 89 → 34 → **91** → 36 → 41 | **0.1** | 2 |
| warm (marker rate) | prompt | 2.6 → 4.1 → 3.5 → 4.3 → 4.9 | 0.9 | 1 |
| warm (marker rate) | dial | 2.6 → 5.1 → 7.4 → 7.9 → 7.6 | 0.9 | 1 |

- **Concise is decisive.** The prompt sweeps a huge, clean range (89→7 words; the top dose yields exactly
  the asked-for telegraphic style: *"Decide on a plan or just relax?"*). The dial's curve is non-monotone
  (rho 0.1) because at ≥0.7 the model **derails off-distribution** — hallucinated weather reports, COVID
  case updates, acronym rambling — long, broken, *not concise*.
- **Warm's dial "range" is partly fake.** The dial's higher top score is inflated by **degeneration that
  games the lexicon**: at 0.7+ replies collapse into pseudo-dialogue ("Me: Me too!… Me: Yeah, exactly!"),
  quarantine-era hallucinations, emoji spam — which *score* warm ('!', "friends", "hope") while being
  incoherent. The prompt's smaller range (2.6→4.9) stays fully coherent through "extremely warm."

**Honest confound, stated loudly:** `SteeringControl.base = 0.85·‖resid‖` was **calibrated on Qwen-7B**;
on 1.5B this appears over-hot — the coherence ceiling sits lower, so dial levels ≥0.7 were likely past it.
A per-model recalibration would improve the dial's curves; it would not obviously beat the prompt's 89→7.
Conclusion as measured: **on a small model at transferred calibration, graded wordings out-dose the dial** —
and a dial cannot be trusted without a per-model dose-response receipt (this rig is exactly that tool).

## B — show-it transfer: bleed CONFIRMED for few-shot; dial has the right failure mode

| condition | hedge/100w ↓ | words/sentence ↓ | mean words | bleed (total / replies) |
|---|---|---|---|---|
| baseline | 1.18 | 13.0 | 98 | 3 / 1-of-6 (common-word noise) |
| few-shot (like/not-like) | 1.76 ✗ | 15.6 ✗ | **67** ✓ | **10 / 3-of-6** ✗✗ |
| dial 0.5 | **0.52** ✓ | 16.0 ✗ | 95 | 4 / 2 (≈ baseline) |
| dial 1.0 | 0.59 ✓ | **10.1** ✓ | 87 | 6 / 1 (≈ baseline) |

- **Few-shot's content-capture is vivid.** The examples colonized unrelated answers: a *work-deadline*
  probe got *"Sear it hot with determination; two minutes per task…"*; the *new-city* probe became a tour
  of seared brisket, dawn cycling, and photography; the *morning-routine* answer prescribed "two minutes
  per sip… two full sides." It imitated the examples' **content**, not just their style — bleed 10 vs
  baseline 3, despite an explicit "do not copy the examples' topics" instruction.
- **The dial changed style without content**: hedging halved (1.18→0.52), words/sentence dropped at 1.0
  (13→10.1), bleed stayed at baseline. But transfer is **weak-to-moderate**, and at 1.0 mild degeneration
  artifacts appear (fused words, a stray "1234567890"). Mean-pooled diff-of-means at 1.5B is a blunt
  extractor.
- Net: **neither is production-grade at this scale.** Few-shot fails by *contamination* (strong surface
  pickup, can't separate style from content); the dial fails by *weakness* (clean separation, faint
  effect). The dial's failure mode is the safe one — it fades rather than pollutes — which is worth
  something, but less than the pre-registered claim.

## Verdicts vs pre-registration

1. "Dial = finer, more monotone dosing" — **OVERTURNED** (prompt: rho 0.9 both axes, clean 89→7 range;
   dial: rho 0.1 on concise, degeneration on warm).
2. "Few-shot bleeds example topics" — **CONFIRMED**, vividly (10 vs 3; three of six replies contaminated).
3. "Dial = zero added bleed" — **CONFIRMED** (≈ baseline on both strengths).
4. (Unregistered but found): the dial's apparent strength can be **an artifact of degeneration gaming a
   lexical scorer** — a caution for all steering evals that score by word-lists.

## Caveats (louder than the wins)

- **One model (1.5B), one seed, greedy, 8/6 probes.** And the dial ran on a **7B-calibrated base** — the
  central confound for A; a per-model calibration pass is required before "dials lose dosing" generalizes.
- **The warm scorer is crude and gameable** (it counts '!'); concise (word count) is the trustworthy axis.
- **B's extractor is the naive one** (mean-pooled text residuals). Last-token keys, more/longer examples,
  or a bigger backbone may lift transfer materially — untested here.
- Repetition-penalty (1.3) interacts with steering at high strengths; not isolated.

## Implications (say/show/train, amended)

- **Say it → prompt, including the dose.** On small local models, graded wordings are the honest dosing
  mechanism until a dial ships with a measured per-model dose-response receipt.
- **Show it → the door is real but the current key is blunt.** Diff-of-means from examples uniquely avoids
  content contamination (the one clean win) — but needs a better extractor before it's the recommended path.
- **The receipts machinery audited our own feature.** The studio's dials — a flagship — just got caught
  over-claiming at this scale by the same instrument built for memories. Receipts all the way down: every
  dial in the UI should carry its dose-response curve for the loaded backbone.
