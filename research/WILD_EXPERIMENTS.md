# Wild experiments I'd run next (a Fable parting gift)

*2026-07-03. Each: the crazy claim, the first falsifiable rung, and what a result would mean. All
runnable on the 5080 + this repo's machinery. House rules apply: pre-register, nulls beside wins,
eyeball before believing, provenance over plausibility.*

1. **Finish vector telepathy** (rig exists, untracked: `research/vector_telepathy.py`). Can memories
   port between models as raw vectors through a fitted linear bridge — no text? Rung: ridge-fit
   1.5B→7B on 256 shared sentences, port slot KEYS only, measure recall vs text-recompile ceiling vs
   shuffled-fit null. Means: models can share thoughts without language — or we get the measured
   anisotropy of why not.

2. **The minimal persistent injection.** Time-travel found a thought's half-life < 1 turn. So what IS
   the smallest intervention that persists? Sweep: edit 1 vs N past positions; edit K vs V; re-inject
   every turn vs once; phantom-KV entry vs value edit. Map the persistence phase diagram. Means: the
   physics of what "holding a thought" costs — and the principled design for working memory.

3. **KV-space handshake between two models.** Two models, same family: splice model A's cache segment
   for a sentence into model B's cache (dims match at 1.5B↔1.5B twins; use two seeds/checkpoints or
   1.5B-base vs -instruct). Does B answer questions about text only A read? Null: shuffled segment.
   Means: direct state-level communication — context sharing without tokens.

4. **The parliament of stances.** Batched decode is ~free (bandwidth-bound). Run K=5 parallel decodes
   of one model, each steered by a different dial (candid/warm/skeptical/concrete/plain), then a vote
   or judge merge. Rung: 30 questions, parliament vs single-decode vs 5-sample-vote (the null —
   diversity from temperature instead of steering). Means: a new local-inference quality trick where
   the dials are the enabling tech.

5. **Cross-substrate thought transfer.** Read a concept off the AR model's SAE features (on-device
   now), find the matching feature in the diffusion model, steer Dream's denoise with the AR model's
   "thought." Rung: 10 concepts, does the steered denoise express them above a shuffled-feature null?
   Means: SAE labels as an interlingua between architectures — the AR model imagines, Dream paints.

6. **Model organisms of memory disorders.** Deliberately break the slot store to build diagnostics:
   induce interference (uncentered keys), confabulation (gate off), amnesia (eta too low), intrusive
   memories (eta too high) — then verify the receipts machinery DETECTS each syndrome blind (agent
   given only receipts must name the disorder). Means: the receipt suite becomes a validated
   diagnostic instrument, not just a display.

7. **The mirror bench, adversarial edition.** Package the confabulation-gap as an eval any model can
   take — then the twist: fine-tune/prompt a model to FAKE self-knowledge and test whether the bench
   catches it (behavioral receipts should be unfakeable; self-report should fool a naive judge).
   Means: a benchmark whose adversarial robustness is itself measured.

8. **Receipts as reward.** The ablation receipt is a scalar (did the memory express? how much drift?).
   Use it as a training signal: tune the memory block's WORDING (prompt evolution, no gradients)
   against expression-with-minimal-bleed receipts. Rung: 20 generations of block mutations, receipt-
   selected. Means: memory that optimizes its own phrasing — self-improving, but only through the
   honest channel.

9. **The quine test.** Give the model its own live state readout (top SAE features, dial settings,
   memory cards — all now available) IN CONTEXT, and ask it to predict its own next-token distribution
   on held-out prompts. Compare to a twin model given nothing. Rung: KL between predicted and actual.
   Means: a measured answer to "does self-knowledge-by-instrument beat no self-knowledge at all?" —
   the introspection prosthesis, quantified.

10. **Idle-compute self-play with provenance.** The GPU sleeps between messages. Nightly: the model
    re-reads the day's runs (NOT dreams — extraction, provenance-linked), proposes receipts-verified
    consolidations, and A/B-tests its own dial settings against the day's real prompts, waking with a
    changelog: "overnight I verified 3 memories and found warm=0.4 beats 0.6 on your actual usage."
    Means: the local advantage (owned idle compute) turned into honest self-maintenance.
