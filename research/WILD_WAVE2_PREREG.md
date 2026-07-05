# Wild Experiments — Wave 2 pre-registration (#8 receipts-as-reward, #10 idle self-play)

*Written 2026-07-05, BEFORE any run. Wave 2 = the two Wave-2 wild ideas PROMOTED from "defer" to active
because they lean on the MEASURED/RECEIPT channel that survived Wave 1 (mirror), not the diversity/
introspection intuitions that didn't (parliament, quine). Both run on the studio's own Qwen2.5-7B nf4
stack — NOT cross-family; the question is whether clozn's own honest machinery is a usable optimizer/
maintainer, so a second model family adds nothing. House rules hold: nulls beside every win, a coherence/
sanity axis on every metric, eyeball before believing, single-seed caveats loud.*

Execution: pre-register (this doc) → build rig → smoke on the real Qwen-7B → full run → findings → commit.
Order: **#8 first** (contained, one clear falsifiable claim), then **#10** (bigger — a single-pass
prototype of the loop, not a scheduler).

---

## Exp #8 — Receipts as reward *(the honest optimizer)*

**The claim under test.** The ablation RECEIPT is a scalar — expression (did the memory show up?) minus
bleed (did it leak off-topic?). Treated as a REWARD, can it drive the memory block's *wording* to a
better phrasing than the shipped seed — **using only the honest measured channel, no gradients, no LLM
judge?** If yes: memory that tunes its own phrasing through receipts, self-improving but only along the
axis the project trusts.

**Antecedents reused.** `memory_mode.compile_prompt_block` (the block compiler + soft/strict styles — the
thing being optimized), `receipts.py` / `replay.py` (the ablation-receipt machinery), the expression
scorers from `self_audit_gap.py` / `mirror_bench.py` (`expressed()`, keyword/length), a bleed measure in
the spirit of `steer_vs_prompt.py`, `counterfactual._coherence` (the coherence gate).

**Setup.** A fixed small memory: one concept card (baking) + one rule card (concise) — the two Wave-1
traits that train reliably. Seed wording = the studio's current compiled block. On-topic held-out probes
(where the trait SHOULD express) + off-topic probes (where it should NOT — bleed).

**Fitness = the receipt.** `expression − λ·bleed`, where expression = trait-shows-up rate on on-topic
probes and bleed = trait-leaks rate on off-topic probes; any wording whose replies trip
`_coherence` is disqualified (a degenerate wording cannot win). λ documented, one value.

**Evolution (no gradients).** G generations; each generation MUTATES the current-best wording K ways —
the model itself rephrases the block instruction (cards/facts fixed, only phrasing changes) — scores each
by fitness, keeps the best.

**Nulls (both required).** (1) SEED baseline — the shipped wording, un-evolved. (2) RANDOM-WALK — mutate
identically but SELECT AT RANDOM each generation instead of by fitness. This is the load-bearing null: it
isolates "the receipt-guided *selection* helped" from "any rephrasing drift helped." If evolved ≈
random-walk, the receipt added nothing.

**Done / falsifiable.** Best-evolved fitness vs seed vs random-walk, over generations. evolved > seed AND
> random-walk → the receipt is a usable optimizer (the win). evolved ≈ random-walk → selection didn't
help (an honest negative). Caveats: one memory, one seed, crude expression/bleed scorers, LLM-driven
mutation (the mutator is the audited model — but the SELECTOR is the measured receipt, which is the point).

**Design risk.** The expression/bleed scorers are lexical and gameable — the coherence gate + the
random-walk null are the guards. If fitness climbs but the winning wordings are eyeball-nonsense, that's a
scorer-gaming result, reported not hidden.

---

## Exp #10 — Idle-compute self-play with provenance *(honest overnight maintenance)*

**The claim under test.** The local advantage is owned idle compute. Can a single self-maintenance PASS
over a corpus of conversations produce **honest, receipt-verified improvements** — provenance-linked
memory consolidations + a better dial setting — *without* the hallucination that killed diffusion dreaming
(Law #4)? The output is a changelog a user could actually trust: "verified 3 memories, warm=0.4 beats 0.6
on your real prompts."

**Antecedents reused.** `runlog.py` (reading runs), `memory_cards.py` (provenance-linked propose —
`source_run_id` + quoted span), `receipts.py` (verification), `counterfactual.dose_sweep` (the dial A/B),
`dream_consolidation_findings.md` (the KILLED dreaming baseline — the null to beat, 14–0 in the antecedent).

**Setup.** A fixed synthetic "day": ~15–20 user turns carrying a few LATENT preferences the user states in
passing (e.g. mentions baking twice, asks for short answers once) — so we have GROUND TRUTH for what
honest extraction should find, and planted distractors it should NOT consolidate.

**The loop (single pass, measured at each stage).**
1. EXTRACT provenance-linked candidates over the day's runs (each carries the run id + the quoted span it
   came from). Planted distractors present.
2. VERIFY each candidate with a RECEIPT (does it express on-topic without bleed?); keep only receipt-passers.
3. DIAL A/B: `dose_sweep` one dial (warmth) against the day's real prompts; pick the best-scoring setting.
4. Emit a CHANGELOG of what was verified + the chosen dial.

**Nulls (both required).** (1) DREAMING baseline — re-mask+re-denoise the turns (the killed approach) and
run the SAME verify filter; does provenance-extraction beat dreaming again (reproduce 14–0)? (2) RANDOM
dial — does the A/B-selected dial actually beat a randomly-picked setting on the day's prompts, or is the
"improvement" noise?

**Done / falsifiable.** (a) Extraction yield: candidates proposed vs receipt-VERIFIED, and how many
verified match the planted ground-truth preferences vs the distractors (the honesty filter's precision).
(b) Dreaming null: extraction's verified-yield ≫ dreaming's (or not). (c) Dial A/B: the chosen dial beats
default AND random on the day's prompts (or not). A pass that verifies real preferences AND finds a better
dial = the idle-maintenance idea works; 0 verified or no dial gain = it doesn't. Either is a finding.

**Design risk & scope.** This is a SINGLE-PASS PROTOTYPE of the loop, not a scheduler/cron and not wired
into the live studio — the experiment is "does the loop produce honest, verified maintenance on a day's
runs," not "ship the nightly job." The synthetic day is hand-authored (its ground truth is known but its
realism is limited — stated, not hidden). One seed.

---

## What Wave 2 buys

Unlike Wave 1 (which mostly PRUNED speculative features), Wave 2 tests two things that could become real,
on-thesis features — and does so honestly: #8 asks whether the receipt is a usable optimizer, #10 whether
owned idle compute can do trustworthy self-maintenance. Both are built to FAIL loudly (the random-walk and
dreaming nulls, the coherence gate, the provenance-vs-distractor precision) rather than to flatter the
idea. A win on either is a candidate feature grounded in a receipt; a null is a clean "don't build it."
