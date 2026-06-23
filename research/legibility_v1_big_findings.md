# Does verifiable self-report of a test-time-learned rule scale up? (`legibility_v1_big.py`)

*Scale-curve sequel to `legibility_v1.py` (idea 3: self-report + verify). READ `legibility_v1.py` +*
*`legibility_v1_findings.md` first — this run REUSES that idea-3 self-report + verification harness*
*VERBATIM (the verification rig is the validated asset) and runs it on BIGGER Instruct models. Run*
*2026-06-22. Qwen2.5-{0.5B known, **1.5B, 3B**}-Instruct, FROZEN, lab `.venv` (torch cu128, RTX 5080,*
*float32). Synchronous, single process (280.5 s; 1.5B then 3B sequentially, freed between). `.venv-sae`*
*untouched.*

## The question

`legibility_v1.py` found a clean **negative** on Qwen2.5-0.5B-Instruct: after test-time-adapting a soft
prefix to a NEW (held-out) relation — which the frozen model then **applies** to held-out words at 0.944 —
the model could **not STATE the rule in words that check out**. Stated-rule held-out apply was **0.306**
(agreement with the adapted behaviour 0.278), **not clear of an averaged wrong-rule null (0.292)**. The
verifier was sound (oracle true-rule applied 0.769 ≫ wrong 0.292) and the model *could* articulate easy
rules from examples-in-context (ICL self-report ceiling 0.382). So the failure was plausibly **(a)** the
0.5B being too small to introspect on the adaptation, and/or **(b)** the soft-prefix sitting in the answer
slot and derailing fluent statement. This run tests both: a **scale curve** (0.5B → 1.5B → 3B, same rig
verbatim) and a **statement-friendly adaptation** (a free per-layer steering nudge that leaves the
generation head free) to isolate (b).

## Verdict

**A bigger model CAN verifiably say what it just learned — the 0.5B negative was, in large part, a
small-model limit.** At **1.5B** the prefix-active self-report states the actual transformation in
checkable words and **verifies at 0.771 (agreement 0.743) ≫ the wrong-rule null 0.138** — a clean,
decisive positive where the 0.5B sat at chance. At **3B** it still clears the null (**0.569 ≫ 0.215**)
but *lower* than 1.5B (the curve is **non-monotonic**, see below). **The introspection gap the 0.5B failed
is broken by scale: ≥1.5B, a TTT-learned rule is both applied AND statable-and-verified.**

**The statement-friendly steering adaptation did NOT help — confound (b) is refuted as the main cause.**
Leaving the generation head free (a free per-layer residual nudge instead of an answer-slot prefix)
**applies the rule fine** (1.5B 0.861, 3B 0.917 held-out menu) but its self-report is **worse** than the
prefix at both scales (1.5B 0.336 vs prefix 0.771; 3B 0.416 vs prefix 0.507) — the steered model mostly
echoes the prompt format ("In one short phrase starting with a verb"). So the 0.5B failure was **not**
mainly the answer-slot prefix derailing generation; it was the model's **introspective capacity**, and
that is what scale fixes. The most on-thesis route (self-report from the answer-slot prefix) is also the
*best* route once the model is big enough.

### The scale curve (best framing; stated-rule held-out apply / agreement; menu-scored; held-out words + relations)

| model | adapted (TTT) apply | **prefix stated→applied** | agreement | clears wrong-null? | steering stated→applied | ICL self-report ceiling | oracle true-rule | wrong-rule null |
|---|---|---|---|---|---|---|---|---|
| **0.5B** (known) | 0.944 | **0.306** | 0.278 | **NO** (null 0.292) | — | 0.382 | 0.769 | 0.292 |
| **1.5B** | 0.944 | **0.771** | 0.743 | **YES** ≫ | 0.336 | 0.771 | 0.917 | **0.138** |
| **3B** | 0.824 | **0.569** | 0.477 | **YES** | 0.416 | 0.568 | 0.972 | 0.215 |

(Per-arm aggregates: **1.5B prefix** decl 0.771 / metacog 0.713; **steer** decl 0.336 / metacog 0.321.
**3B prefix** decl 0.569 / metacog 0.507; **steer** decl 0.283 / metacog 0.416. The table shows the best
framing per cell; both framings are in the JSON. "best framing" can mix decl/metacog across the
stated-acc and agreement columns — read the per-arm lines for a single-framing view.)

- **0.5B → 1.5B is the jump that matters: +0.47 stated-apply, from at-chance to ≫-null.** The 1.5B names
  the real transformation in words that verify: `plural` → *"add -s to the end"* (1.000), `past` → *"add
  -ed"* (1.000), `antonym2` → *"reverse the meaning"* (1.000), `gerund` → *"add -ing … present participle"*
  (1.000). The 0.5B, for these same relations, reported only *"maps the first word to the second"* /
  *"In one short phrase starting with a verb"* — the prompt format, not the content.
- **The curve is non-monotonic (1.5B > 3B), and the 3B per-relation data says why.** Two things drop at 3B:
  **(i)** the *adaptation itself* is weaker on this held set (adapted apply 0.824 vs 1.5B 0.944 — same m=8
  prefix, 30 steps; the bigger model needs more capacity/steps to fit, e.g. 3B `plural` adapted only 0.444),
  so there is *less correct behaviour to describe*; **(ii)** the 3B's chattier instruction-tuning derails
  the prefix-active self-report into **assistant boilerplate** — for `past` it said *"You are a helpful
  assistant"* / *"I am a helpful assistant"* (verifies 0.000), and its `gerund` metacog was *"Based on the
  word pairs you provided"* (0.000). Where the 3B *does* state cleanly it is excellent (`plural` →
  *"pluralize"* 1.000; `third_person` → *"add -s or -es"* 0.875; `antonym2` metacog → *"one state to its
  opposite or extreme"* 1.000). Crucially the **ICL self-report ceiling also drops 1.5B → 3B (0.771 →
  0.568)** for the same reason — the 3B wraps even in-context answers in boilerplate — so the 3B dip is a
  *self-report-articulation* artifact of this checkpoint's chat-tuning, **not** a failure of the
  introspection thesis (both scales clear the null; the bracket moved, not the result).

## The statement-friendly steering adaptation (free per-layer residual nudge; head left free)

A NEW mechanism added here (the named-basis `SteerHook` in `legibility_v1` learns a coefficient over a
*fixed* diff-in-means basis; this learns the **raw per-layer vector directly** — strictly more expressive,
to give "apply" its fairest shot): one learnable `v_L ∈ R^H` per mid-layer, added to that layer's residual
output at **every position** (so it does **not** occupy the answer slot), fit by the SAME apply-CE loss on
the relation's own examples, frozen backbone. Mid-band auto-scaled to depth (1.5B layers 7–19, 3B layers
9–25).

- **It APPLIES the rule well** — 1.5B 0.861 / 3B 0.917 held-out menu (the task's pre-condition is met:
  "make sure the steering still applies"). So the comparison is fair.
- **But its self-report is WORSE than the prefix at both scales** (1.5B 0.336 vs 0.771; 3B 0.416 vs 0.507).
  The steered model, asked to state the rule, mostly **echoes the prompt** (*"In one short phrase starting
  with a verb"*, *"states the transformation rule"*) or emits a single off-target token (*"verbs"*,
  *"rules"*, *"Stating"*). A constant residual nudge at every position pushes the logits toward the
  *answer* tokens just as the prefix does, so it derails generation in much the same way — without the
  prefix's one advantage (a coherent learned "preamble" the chat head can sometimes verbalize).
- **Conclusion on confound (b):** decoupling the generation head from the adaptation did **not** unlock
  self-report. The 0.5B failure was an **introspective-capacity** limit, not an answer-slot-prefix artifact
  — which is exactly what the scale curve independently shows (capacity, via model size, is what flips it).

## Why this is the honest result (the verification rig is the asset, reused verbatim)

1. **The verifier stayed sound at every scale and the controls behaved.** Oracle true-rule applied ≫
   wrong-rule null at all three sizes (0.5B 0.769/0.292, 1.5B 0.917/0.138, **3B 0.972/0.215**) — when the
   frozen model is *handed* the right rule it applies it, so a stated rule that verifies is genuinely
   capturing the relation. The wrong-rule null is **averaged over 4 different relations' descriptions**
   (one arbitrary wrong rule is noisy; a word's own prior can emit the true answer regardless), giving a
   stable rule-independent floor; the reported signal is the **gap above wrong**, and at 1.5B/3B that gap
   is large and real, where at 0.5B it was ~0.
2. **Self-report is never trusted, only verified.** We never score the stated rule as text; we *apply* it
   to held-out words with the frozen, *unadapted* model (chat + menu-scored) and measure both held-out
   accuracy and **agreement** (per-word match with the adapted model's own behaviour). The per-relation
   table exposes exactly where a word's prior inflates a cell (e.g. `antonym2`/`part_of` wrong-floors).
3. **Held-out WORDS + held-out RELATIONS, both framings, free-gen beside menu, full per-relation
   breakdown, no cherry-picking.** Same expanded relation bank, split, ICL ceiling, and candidate menu as
   lever 3 / the 0.5B run — byte-identical (same Qwen tokenizer ⇒ 26 relations, currency dropped, 728
   words, 449-way menu, chance 0.0022). The held-out eval set is the same shuffled-order six (`plural,
   third_person, antonym2, gerund, past, part_of`). The adaptation reproduces lever 3 (1.5B 0.944) so the
   comparison is anchored.
4. **The non-monotonic dip is reported plainly, with its mechanism**, rather than smoothed — 3B's weaker
   adaptation-on-this-held-set and its boilerplate-prone chat head, both visible per-relation and both
   tracked by the parallel drop in the 3B ICL ceiling.

## What this means / next rung

- **The headline flips from `legibility_v1`: a TTT-learned rule CAN be made legible by self-report —
  scale is the unlock.** At ≥1.5B the model both *does* the new rule and *says* it, and the saying
  **verifies** against a sound oracle and a proper averaged null. The 0.5B negative was a small-model
  limit, not a law — confirming the `legibility_v1` "next probes" hypothesis (b: bigger model) and
  refuting (a: answer-slot prefix) as the dominant cause.
- **The legible window is the *answer-slot soft prefix itself*, not a head-free steering vector.** The
  statement-friendly steering applies but does not state; the prefix both applies and (at scale) states.
  So for Clozn the read-out path is: TTT a soft prefix, then ask the (≥1.5B) model what it learned, then
  **verify**. The verification is non-negotiable — it is what separated the 0.5B confabulation from the
  1.5B real report, and it is what flags the 3B's boilerplate cells as 0.000.
- **3B is articulation-limited by this checkpoint's chat-tuning, not by introspection.** The obvious next
  probes: (i) **strengthen the 3B adaptation** (more steps / larger m) so there is more correct behaviour
  to describe, and re-measure — the 3B dip may close; (ii) **a less boilerplate-prone elicitation** (a
  raw-text "the rule is:" continuation, or a system-prompt that forbids meta-commentary) to recover the
  3B's clean reports (`plural`→pluralize, `third_person`→add -s) that are being masked by *"You are a
  helpful assistant"*; (iii) **a prefix trained with an auxiliary "describe-the-rule" objective** (make
  the adaptation statable-by-construction) — the one steering idea that might still beat the prefix is one
  that is fit to *both* apply and verbalize, which this free nudge was not.
- **The verification harness ported cleanly and stayed trustworthy across 3 model sizes.** It reported a
  flat negative at 0.5B, a decisive positive at 1.5B, and a real-but-lower positive-with-a-caveat at 3B —
  each beside the same oracle/wrong/ICL brackets, without flattering any of them. That is the reusable
  asset for the C++-core white-box read-out and for the next better-injection pushes.

## Setup (deltas from `legibility_v1`; everything else identical / verbatim)

- **Models.** Qwen2.5-1.5B-Instruct (28 layers, H=1536) and Qwen2.5-3B-Instruct (36 layers, H=2048),
  FROZEN, **float32** (byte-faithful to the 0.5B run; also dodges a float32-cache/bf16-weight matmul
  mismatch in the verbatim-reused `frontier_apply` helpers — 3B fp32 ≈ 12.4 GB fits the 17 GB card,
  activations here are tiny). Models run **sequentially in one process** and are freed between (peak = one
  model; 3B peaked ≈ 15 GB). Downloaded to `~/hf_models/<name>` via `local_dir` to dodge the Windows
  symlink crash (MEMORY: WinError 1314 — `HF_HUB_DISABLE_SYMLINKS` no longer exists in huggingface_hub
  0.36; `local_dir` is the working path).
- **Reused VERBATIM from `legibility_v1`** (imported, not re-implemented — the validated rig): `run_idea3`,
  `apply_stated_rule`, `adapted_behavior`, `score_against_truth`, `clean_rule`, `chat_ids`,
  `generate_with_prefix`, `render_examples`, `RULE_DESC`, `SELFREPORT_USER` (both framings), `svg_idea3`,
  the Maiko palette; and `fit_ttt_prefix` (the lever-3 soft-prefix mechanism). The TTT prefix is m=8, 30
  Adam steps, lr 0.05, fit on the relation's full TRAIN words.
- **NEW here (arm B):** `FreeSteerHook` + `fit_free_steer` (free per-layer residual nudge, 60 steps, lr
  0.02, weight-decay 1e-3) and `run_steering_selfreport` (the IDENTICAL idea-3 verify pipeline, but the
  adaptation is the steering hook; self-report generated with the hook active via `generate_with_hook`,
  head free). Same controls (oracle / averaged-wrong / ICL ceiling), same held set, same scoring.

## Reproduce

```
cd research   # repo: C:\Users\brigi\src\clozn ; lab .venv (GPU), SYNCHRONOUS single process
# Qwen2.5-1.5B/3B-Instruct auto-download to ~/hf_models on first run (local_dir, no symlinks)
python legibility_v1_big.py                                   # 1.5B + 3B, both arms (the deliverable; ~280 s)
python legibility_v1_big.py --models Qwen/Qwen2.5-1.5B-Instruct   # one model
python legibility_v1_big.py --steer 0                         # prefix arm only (scale curve)
```

Files: `research/legibility_v1_big.py`; `research/runs/legibility_v1_big.json`; Maiko-palette SVGs
`research/runs/legibility_v1_big_scalecurve.svg` (the headline: stated-apply/agreement vs wrong-null vs
ICL vs oracle across 0.5B→1.5B→3B, both arms) and per-model
`legibility_v1_big_{prefix,steer}_{1.5b,3b}.svg` (adapted vs stated-applied vs agreement vs oracle vs
wrong, per-relation + aggregate). The 0.5B anchor is read from `research/runs/legibility_v1_0p5b.json`.
```
```
