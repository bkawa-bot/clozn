# Clozn — Implementation Roadmap

**Remaining work only, in the order to do it.** Each item carries *why* we're doing it and its *payoff*
(what improves in the product / what a person sees). Shipped work lives in git history (the pre-2026-07-17
reconciliation backlog carried the full done-archive). Binding on everything below: the **honesty
invariants** — every readout carries a null; measured never self-reported; negatives ship as labels;
discrimination-not-awareness framing (`notes/CLOZN_SOUL.md`).

The sequence is GPU-shaped: **verify already-merged work before building the big C++ lanes.** VRAM (not
compute) is the live limit — one 0.5B engine ≈ 2.7 GB fits the current ~3 GB headroom (one task at a time);
anything needing Qwen-7B + Dream co-resident (~13 GB) waits for the parallel effort to free ~10 GB.

---

## Phase 1 — GPU verify (now; fits the 0.5B budget)

Not new features — the trust gate before we build more. Payoff = confidence the shipped product works.

1. **Route D live round-trip** — does a real model honor the verbatim pins? Validates the shipped Rewrite
   (AR) mode; the pin-fidelity chips report honestly. Single engine boot, no worker restart.
   *Why:* we shipped Rewrite mode but never confirmed a real model keeps pins verbatim. *Payoff:* proof the
   studio's Edit → Rewrite mode works — or honest ✗ chips when a model breaks a pin, never a silent lie.
2. **Real-browser pass over heavn UI** — click through Settings / Explain / quick-repair / Scope / Read /
   actuary against a live engine.
   *Why:* the reconciled UI only had model-free render checks; no human has clicked it live. *Payoff:* no
   broken panel waiting for a user — an end-to-end quality gate on the whole studio.
3. **Pressure-test the merged lanes** — hostile hands-on over migrations / GC / cancellation (kill workers
   mid-stream, disconnect clients, corrupt/fill the blob store).
   *Why:* the new persistence/cancel code only saw unit tests; hostile testing has found real bugs every
   prior time. *Payoff:* the product won't corrupt a run, hang, or lose data under messy real usage.
4. **Engine-side cooperative cancel** (small C++) — make "cancel" actually stop the worker, not just drop
   the socket. Completes the request-isolation lane.
   *Why:* today the worker keeps decoding after a cancel (wasted GPU). *Payoff:* hitting Stop frees the GPU
   instantly — snappier iteration, no compute burned on an abandoned generation.
5. **H7 + H3 live captures** — ⛔ **blocked on VRAM** (need Qwen-7B + Dream, ~13 GB). Harnesses armed in
   `notes/ar_diffusion/{h7,h3}/`; two commands each once ~10 GB frees.
   *Why:* run the harnesses to answer "does generation *order* change *content*? does diffusion commit
   non-linearly?" *Payoff:* research evidence (a divergence atlas) that could become a studio visualization
   or a publishable finding — not a feature yet.

## Phase 2 — the big C++ arc (GPU + build; dependency-ordered)

Build the two foundations before the headline. 6–8 are infrastructure whose payoff is that they make #9
possible and make the white-box readouts fast enough to watch live.

6. **Native exact-state checkpointing + token-exact branching** — serializable/clonable `EngineState` (KV +
   token ids + gen pos + sampler/RNG + interventions + hashes), snapshot/clone/restore in C++, plus
   copy-on-write / paged KV so branches share a prefix (today `clozn/replay/timetravel.py` is
   descriptor-only). Bar: bit-exact greedy suffix after save→restore.
   *Why:* today's time-travel re-prefills a transcript (replay, not exact) and can cross tokenizer
   boundaries. *Payoff:* "rr/gdb for an LLM" — pause at a token, fork 100 branches from one shared state,
   patch a feature, resume bit-exactly. The studio's branching/Experiment flow becomes exact and cheap.
7. **Batched multi-sequence decode** — the shared-prefix batch primitive; also unlocks **batched causal
   credit** (coalition / Shapley over teacher-forced arms — `clozn/receipts/core.py` is sequential
   leave-one-out + pairwise-only today).
   *Why:* receipts are computed one-arm-at-a-time (slow) and miss higher-order interactions. *Payoff:*
   receipts return much faster (less spinner) **and** a richer "why did it say this" that catches
   influences which only matter *together*, not just individually.
8. **Device-resident multi-observer readout plane** — keep layer activations device-resident, fan out to
   J-lens / SAE / probes / norms on an async stream. Bar: all observers together at <5–10% overhead,
   measured under concurrent load.
   *Why:* live readouts are single-owner and CPU-transported — one lens at a time, slowly. *Payoff:* the
   "watch it think" experience becomes rich and real-time — several lenses at once during generation.
9. **Intervention-validated circuit tracer** *(headline — needs 6 + 7 first)* — attribution graph with an
   explicit unexplained-mass term; every path patch/inhibit/ablate-testable at the exact token+layer;
   predicted-vs-observed logit movement against random-node / direction / shuffled-edge controls.
   `clozn/analysis/microscope.py` is the correlational precursor.
   *Why:* clozn can inspect and intervene but can't yet *produce and prove* how an input caused an output —
   the architecture marks this unbuilt. *Payoff:* **the north-star feature.** Click "Tokyo" in an answer →
   a compact causal path through named internal features to the output logit → disable it → watch the
   prediction move. The whole thesis made tangible: *what computation caused this, proven by changing only that.*

## Phase 3 — research lanes

Higher variance; each is a real capability if it lands.

10. **Live risk controller** — wire the offline answer/ask/abstain policy (`clozn/eval/policy.py`, today
    selection-only) into live generation; beat black-box baselines held-out with CIs.
    *Why:* the eval layer computes thresholds but generation never uses them. *Payoff:* the model can
    actually **abstain, ask to clarify, or self-check before answering** when likely wrong — measurably
    fewer wrong answers, not a decorative confidence meter.
11. **Cross-model causal state diffing + transplants** — align residual spaces, transplant A→B at a
    token+layer, measure the recovered logit; decompose quant/finetune regressions
    (`clozn/analysis/model_diff.py` is observational-only after divergence today).
    *Why:* we can compare variants but only *observe* the divergence. *Payoff:* a debugger for "why did Q4
    get worse than Q8 / what did this fine-tune break," pinpointed to a layer+feature and *proven* by
    transplant — useful to anyone running quantized local models.
12. **Native fast-weight fact memory** — port keyed/fast-weight memory into the engine so a recall actually
    alters the reply (`clozn/server/facts_store.py`'s read receipt doesn't today), with
    with-memory / without-memory / matched-null receipts and abstain-on-ambiguity.
    *Why:* the fact store retrieves but its receipt doesn't change the reply, and it isn't native.
    *Payoff:* editable memory **inside the model's computation**, not a vector DB glued to the prompt —
    write / inspect / recall / delete a fact with a receipt proving the *memory* (not prompt leakage) moved the answer.
13. **Closed-loop disposition guardrails** — mid-gen lens polling → threshold → `dir(c)` counter-injection,
    on a banned-topic battery. ("The biggest unclaimed frontier.")
    *Why:* catch a disposition mid-generation and counter-steer before it's spoken. *Payoff:* a live
    safety/steering guardrail that acts *during* generation, demonstrated on a banned-topic battery.
14. **AR×diffusion H2 + H5** — **H2** (score-gated self-repair) and **H5** (counterfactual-patch receipts;
    ⚠ needs a `/v1/revise` ablated-context spike first). Specs in `notes/ar_diffusion/specs/`.
    *Why:* research bets on combining the AR + diffusion substrates. *Payoff:* possibly better generation or
    richer receipts — honestly may not pan out (its sibling H1 was killed; we keep the honesty).
15. **Edit routes B / C** — Route **B** content-concept via `dir(c)` (~dozen-line engine unlock); Route
    **C** free-text via LLaDA-8B-Instruct (the research swing). Route D shipped.
    *Why:* extend the edit vocabulary past Rewrite. *Payoff:* more Edit-drawer modes — steer content *by
    concept* inside a real bidirectional resolve (B); free-text edit instructions to a diffusion model (C).
16. **J-lens post-v1 (J5)** — Dream/denoise lens, chat-vs-web-text lens, stream top-k during generation.
    *Why:* extend the shipped "disposed to say" lens. *Payoff:* richer live readouts in the studio.
17. **Assembled-but-unconnected bets** — model's-own-CI, legible-basis microscope (OMP), branch-on-doubt,
    paraphrase-brittleness receipts, cross-model disposition transfer (pilot).
    *Why:* validated-but-unwired research primitives. *Payoff:* each *could* become a studio feature;
    low-priority exploration.

## Phase 4 — product / UX polish

18. **Ambient channel-3** — inline confidence-shading right inside Cursor / the ChatGPT web UI (needs
    text↔trace alignment via `X-Clozn-Run-Id`). Highest effort of the three.
    *Why:* the ambient-delivery endgame. *Payoff:* clozn's confidence/trust shading **inside the tools
    people already use** — "zoom into the sketchy spans" without leaving their workflow.
19. **Route-B "revise steer_vec" engine unlock** — content-concept edits inside a real bidirectional
    resolve (pairs with #15's Route B).
    *Why:* the engine unlock that makes Route B real. *Payoff:* content edits that *propagate* through a
    resolve, not just regenerate.
20. **Design-agent mock pack (D1–D5)** — only if pursuing the visual-polish direction (`notes/CLOZN_UX.md` §11).
    *Why:* optional visual polish. *Payoff:* a more finished studio look — only if we pursue it.

---

## Parked — needs an owner decision or an external unblock

- **Lab artifact contracts + model qualification** — 🔄 in progress in the parallel effort
  (`clozn/artifacts/contracts.py`, `docs/qualification/`). End-state: a one-command
  `clozn qualify-whitebox <gguf> --checkpoint <org/model>` proven on Qwen + Gemma + Llama.
  *Why/Payoff:* makes clozn's white-box features work on **any GGUF**, with an honest capability matrix —
  single-model → real multi-model platform.
- **Ollama drop-in?** — owner decision (recommended: a thin `/api/chat|generate|tags|version`).
  *Why/Payoff:* point your existing Ollama tools at clozn and get white-box on top — an adoption lever.
- **`real-runtime-smoke.yml` green** — the "zero jobs" parse bug is fixed but unverified; needs an owner
  `workflow_dispatch` click.
  *Why/Payoff:* a trustworthy green CI check that the real engine works — what makes a release credible.
