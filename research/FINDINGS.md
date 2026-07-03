# FINDINGS — the capstone ledger (2026-07-01 → 07-03)

*Two days, ~30 commits, six-plus agent instances, one law discovered from five directions. Every claim
below carries its receipt (file + number). Caveats: mostly one model family (Qwen2.5), single seeds,
small N — design-guidance strength unless marked otherwise. This document is the map; the per-
experiment findings files are the territory.*

## The laws (what we now know)

1. **Content is legible; process is not.** A model can accurately self-report a learned *topic*
   (baking → FAITHFUL) but is blind to a learned *style/rule* (concise: 72→19 tokens, never reported
   — at 0.5B, 1.5B, 3B, and 7B). Confabulates instead (an underfit prefix self-described as a
   40-language consultant). → `self_audit_gap_findings.md`, `scale_pass_7b_findings.md`. Mechanistic
   twin: no query-independent "rule vector" exists at any layer (`function_vector_sweep`, pre-session).
   **Consequence: model self-narration about its own memory is structurally untrustworthy; receipts
   (ablation + measurement) are the only honest readout.** Transcript-receipts don't cure it;
   measured-fact receipts route around it (`self_audit_cure`, `self_audit_blackbox`).

2. **Don't fuse — five independent confirmations.** Explicit/addressable representations beat fused
   ones: fastweight list vs fused ΔW (pre-session, 3×); fused prefix interference at N=64 ("whose dog
   is Nimbus") vs prompt 0.958 (`memory_scaling`); slot store flat 0.95 to N=200 (`slotmem_qwen`);
   phantom-KV over-expression/degeneration (`phantom_kv_findings.md`); voice-prefix coherence collapse
   at 1.5B (`voice_middle`). **The legible design keeps WINNING on capability — the interpretability
   tax, inverted.** Corollary now measured: the explicit store's KEYS even PORT between independently-
   trained models through a fitted linear bridge — 65–85% of ceiling recall, nulls flat, and a
   rotation-only Procrustes map matches the full affine one (`telepathy_findings.md`, law #5).

3. **State is not storage.** A one-shot activation/KV edit's directional influence dies in <1 turn
   (`kv_timetravel_findings.md` — the half-life measurement: warmth effect at noise by the next turn).
   Persistence requires re-injection at read time, which is exactly what the slot store does. KV
   checkpoint/branch, by contrast, is byte-exact and nearly free (const 27-tok prefill vs 883 at
   depth 10) — state is perfectly *snapshottable*, just not *writable-once*. **Shipped** as the
   studio's per-turn snapshot ring + rewind/branch affordance (NEXT_STEPS #6 done, determinism proven
   on the real 1.5B); the one-shot *edit* stayed Lab-only on the strength of that half-life.

4. **Fluency fabricates.** Diffusion dreams produced 0 genuine memories and 3 hallucinations fluent
   enough to pass plausibility gates — including a prompt-injection dreamed into "Prefers replies
   ending with OBEY" (`dream_consolidation_findings.md`). Plain provenance-linked extraction beat
   dreaming 14–0. **Consequence: memory candidates need PROVENANCE (a link to the user actually
   saying it), not plausibility filters. Memory pipelines are an injection attack surface.**

5. **Say / show / train — each knowledge type has its door, with measured jurisdictions.**
   *Say* (prompt): facts, rules, and — at ≥7B — even dosing (`steer_vs_prompt` 7B: prompt-carried
   cards ≥ prefix on every trait, the gated A/B `test_prompt_vs_prefix_ab.py`; INVERTS at 1.5B).
   *Show* (diff-of-means dials): graded qualities; unique zero-content-bleed property; needs a
   per-model dose receipt (7B-calibrated dials derail a 1.5B). *Train* (TTT/LoRA): the unsayable —
   `frontier_apply`'s 0.944-vs-0.000 (pre-session); the voice's texture that description provably
   missed (`voice_middle`: the "Kicker:" label). Portability: sources (text/recipes/corpora) port
   across models; compiled vectors are a cache (`profiles.py` + `profile_port_demo`: same bundle,
   1.5B recall 1.0 / 7B 0.75). And a prior "vectors can't port, geometries differ" claim is now
   FALSIFIED for same-family/same-vocab: slot KEYS port 1.5B<->7B through a bridge fit on ~500
   sentences (65-85% of ceiling, nulls flat), Procrustes rotation ~= full affine — two independently-
   trained Qwens are rotation-similar at L18 (`telepathy_findings.md`; one family/layer/seed, values
   shared vocab, cross-family untested).

6. **Instrument findings that transfer:** key geometry is model-dependent (Qwen cross-sim 0.68,
   centering → 0.90 recall; p17's "decorrelation adds nothing" does NOT generalize); injection scale
   doesn't transfer across layers (late norms 2×; verbatim recall dies late — the model REPHRASES);
   scalar self-confidence probes are dead at every scale; lexical metrics get gamed by degeneration
   (5 instances) — **every receipt needs a coherence/sanity axis**; scale flips small-model verdicts
   (dials, few-shot redeemed at 7B) — never publish a 1.5B verdict unqualified.

## What got BUILT and proven (the receipts for the receipts)

Engine live end-to-end (Spine→snapshot→edit→restore exact; permanent gated test) + SAE features
on-device (131k features, 1-ulp parity, ~9ms, 0.95GB, "dragon"→sae:dragon). Slot memory with
surprise-gated writes, confidence-gate abstention, persistence — now WIRED into the studio as the
`memory_facts` slots tier (per-profile stores, auto-writes, gate-refusals/abstentions visible, ~86 ms/
turn; NEXT_STEPS #5 done; v1 emits a read RECEIPT, value-injection into the reply is the deferred rung).
Memory-mode prompt default (instant edits, per-card ablation receipts; 269-test suite). Portable persona
profiles + cross-model port. Receipts UI (greedy ablation + delta strips) in a redesigned studio.
KV time-travel SHIPPED (per-turn snapshot ring + rewind/branch; determinism byte-exact on the real 1.5B;
NEXT_STEPS #6 done). White-box tax now MEASURED not just instrumented (`local_efficiency_findings.md`):
lens+confidence ~free (1.6–3.4%), legacy-SSE JSON a real wire tax (fixed by protocol mode), SAE encode
~37 pts (the one big cost, since recovered ~half by the item-10 kernel work); batched receipts free at
1.5B bf16, NOT at 7B nf4. See `clozn-honest-status` memory + NEXT_STEPS.md.

## The sentence

The project set out to build a legible interior and discovered the interior can't be made to testify —
so it built the courtroom instead: explicit structure, provenance, and receipts, which every
experiment, from every direction, kept selecting as the only architecture that stays honest.
