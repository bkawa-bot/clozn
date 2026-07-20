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

1. ~~**Route D live round-trip**~~ ✅ DONE 2026-07-18: tested against qwen-0.5b/CUDA via
   `POST /engine/rewrite`. Pin-fidelity chips honestly report kept/not-kept: formal rewrite keeps pins;
   translation correctly reports `kept: false` when model translates a pinned phrase; truncation
   (`finish_reason: "length"`) correctly reports unreached pins as not-kept. Minor wart: `all_pins_kept`
   is vacuously `true` with zero pins (Python `all([])`). Bug found + fixed: receipt footer in probe
   replies contaminated numeric grading (run_id hex contained gold answer as substring).
2. **Real-browser pass over heavn UI** — click through Settings / Explain / quick-repair / Scope / Read /
   actuary against a live engine.
   *Why:* the reconciled UI only had model-free render checks; no human has clicked it live. *Payoff:* no
   broken panel waiting for a user — an end-to-end quality gate on the whole studio.
3. ~~**Pressure-test the merged lanes**~~ ✅ DONE 2026-07-18: 47 hostile tests across cancel proxy,
   truth-tier calibration, migrations (TOCTOU, ledger, duplicate versions, semaphore leak). Found + fixed
   5 real bugs. `tests/test_pressure.py`, all green.
4. ~~**Engine-side cooperative cancel**~~ ✅ DONE 2026-07-18: CancelRegistry + `POST /cancel` +
   broken-pipe auto-detect, all 3 streaming handlers wired, exception-safe cleanup (revise + board).
   GPU build verified. Gateway `req_` → engine `req` correlation wired (`RequestContext.engine_req`,
   captured off first SSE frame; `req_id` body key in cancel proxy).
5. **H7 + H3 live captures** — ⛔ **blocked on VRAM** (need Qwen-7B + Dream, ~13 GB). Harnesses armed in
   `notes/ar_diffusion/{h7,h3}/`; two commands each once ~10 GB frees.
   *Why:* run the harnesses to answer "does generation *order* change *content*? does diffusion commit
   non-linearly?" *Payoff:* research evidence (a divergence atlas) that could become a studio visualization
   or a publishable finding — not a feature yet.

## Phase 2 — the big C++ arc (GPU + build; dependency-ordered)

Build the two foundations before the headline. 6–8 are infrastructure whose payoff is that they make #9
possible and make the white-box readouts fast enough to watch live.

6. ~~**Native exact-state checkpointing + token-exact branching**~~ ✅ DONE 2026-07-19 (first slice):
   `EngineCheckpoint` (KV blob + tokens + n_past) + `save_checkpoint`/`load_checkpoint` on the adapter,
   `POST /v1/checkpoint` / `/v1/restore` / `/v1/branch` routes (in-memory store, cap 16). **Bar met:
   bit-exact greedy suffix after save→restore, and all greedy branches match baseline.** Restore
   currently re-prefills from saved tokens (correct; KV-blob fast-restore is the deferred optimization);
   sampler/RNG state + interventions in the snapshot also deferred.
7. ~~**Batched multi-sequence decode**~~ ✅ DONE 2026-07-19 (engine primitive): `branch_kv` /
   `ar_forward_batch` (one `llama_decode` for N seqs) / `cleanup_seqs` (n_seq_max 16) +
   `generate_ar_branched` loop; `/v1/branch` now prefills the shared prompt ONCE and decodes all
   branches per-step in one batch. Bit-exact vs the sequential path. REMAINING: wire batched causal
   credit (coalition/Shapley over teacher-forced arms) in `clozn/receipts/core.py` on top of it.
8. ~~**Device-resident multi-observer readout plane**~~ ✅ DONE 2026-07-19: multi-layer capture set in
   `eval_cb` (one forward, N layers) → `CaptureFrame` sink → serve-side `ReadoutPlane` worker (observer
   math OFF the decode thread) fanning out to J-lens + norms + probes; J-lens weights uploaded once to
   the GPU (`init_device`), whole batches + all layers share ONE head read per graph (`readout_multi`),
   adaptive token batching (≤16/graph, ≤140 ms lag) to amortize WDDM syncs. `readout:{...}` on
   `/v1/completions` (SSE), `readout` capability + handshake fixture. **Bar met: 4.8% single-stream /
   9.4% two-concurrent-streams overhead with jlens(L16+L24)+norms at every=1, full coverage, zero
   drops** (measured on the 9B Q4, 96-token runs; naive per-token CPU readout was 48%). Honest-coverage
   `readout_stats` frame (observed/dropped/skipped). SAE observer: seam noted, not yet folded in
   (`with_sae_readout` path unchanged).
9. **Intervention-validated circuit tracer** *(headline)* — 🔄 S0–S2 LIVE 2026-07-19 (design:
   notes/CIRCUIT_TRACER_DESIGN.md). Slice 1 (engine): `/score` + `write` (single or ARRAY — joint
   cross-layer patches in one forward) + `capture` keys; verified bit-exact vs /harvest + /state.
   Slice 2 (product): `clozn/analysis/tracer.py` (screen via client-side dir(c) projections on
   captured rows — exact alignment, no retokenize; mean-ablation solo arms + random-direction +
   random-site controls + noise floor + marginal flags; joint arm + interaction gap; PASS /
   NO_CAUSAL_NODES / FAILED_CONTROLS verdicts) + `clozn trace-circuit` CLI + fixture tests.
   **STOP-check PASSED live on the 9B**: real nodes 80–200x above the noise floor on two factual
   prompts; ~3 s wall for 116 screened sites / ~60 arms. First findings: heavy sub-additivity
   (interaction gap ≈ −50% of Σsolo — self-repair is real), the biggest causal node can be ~0%
   legible via the screened direction (unexplained-but-causal, kept in the graph), Kyoto-distractor
   sites appear as small suppressive nodes. S3+S4 LIVE same day: path patching (edges + shuffled
   control; same-column edges route ~100% — the structural correctness check passes; off-column
   discovery edge L16@3(France)→L24@4 routes 45%, shuffled ctl 34x smaller) + generation arms via
   `/v1/completions` write (patch + greedy + reference early-stop). **Predicted-vs-observed
   scorecards PERFECT on both pilot prompts.** Full S0–S4 trace ~2.7–4.4 s.
   **16-prompt / 7-category validation battery 2026-07-20** (notes/CIRCUIT_TRACER_DESIGN.md §5b):
   S4 accuracy **88/96 = 91.7%** (3 false-positive / 3 false-negative, 5 of 6 within ±40% of the
   decision boundary — threshold noise on a real signal); 16/16 PASS. Constraints found: only
   **45% of surviving nodes are `strong`** (≥3x strongest control — `control_ratio`/`strength`
   tiers now shipped per node), median legibility **~24%** (most causal mass is unnameable),
   interaction gap median **−60%** (solo attribution overcounts ~2.5x, universally), and
   `FAILED_CONTROLS` has **never fired on a real prompt** — the STOP check is unexercised in the
   wild. ⚠️ **The disable-and-watch demo (§5c) found the scorecard's scope limit: a next-token flip
   is NOT loss of the answer** — ablating one strong node flipped the token and the model still
   said "Tokyo" one token later. 91.7% is accuracy at predicting *token flips*, never "predicts
   when the model loses the fact". Also: both strong nodes landed on the FINAL prompt token, so at
   this node granularity the graph reads more as "where in depth the answer commits" than "which
   context supplied it" (finer units — attention-head / per-source-position — would be needed).
   REMAINING: run-journal input mode, studio click-a-token panel, a genuine screen-null (replace
   the target concept, don't dilute it), finer node units, 2nd model family.
   *Why:* clozn can inspect and intervene but can't yet *produce and prove* how an input caused an output.
   *Payoff:* **the north-star feature.** Click "Tokyo" in an answer → a compact causal path through named
   internal features to the output logit → disable it → watch the prediction move.

## Phase 3 — research lanes

Higher variance; each is a real capability if it lands.

10. **Live risk controller** — 🔄 PARTIAL 2026-07-18: ask-band signal wired into both streaming + non-streaming
    `/v1/chat/completions` (`clozn_policy` metadata field, silent unless calibration says "ask").
    `policy.score_from_trace` + `classify_run` + `generation_gateway.ask_band_signal` landed.
    REMAINING: abstain-band action (refuse / self-check), heavn UI indicator, beat black-box baselines
    held-out with CIs.

    ⚠️ **GATE RESULT 2026-07-19 — "beat black-box baselines" is UNWINNABLE as written, because the
    deployed signal is not white-box.** Verified in code, not just measured: `runs/trace.py` sets
    `confidence = [s.get("prob") ...]` (the emitted token's softmax probability) and
    `eval/policy.py:score_from_trace` aggregates exactly that. Under greedy decoding the emitted token
    IS the argmax, so this score is the min top-1 output probability — the same number any
    OpenAI-compatible API returns with `logprobs=true`. Confirmed empirically: white-box score and an
    independently computed `exp(min(logprob))` baseline are **bit-identical across all 362 items (max
    abs diff 0.0)**. No hidden-state probe, no J-lens readout, no attention signal is involved.

    The signal nonetheless **works**: TEST n=182 (stratified, deterministic split; threshold + length
    model fit on TRAIN only), AUROC **0.822 [0.760, 0.879]**, risk-coverage drives error from a 20.9%
    base rate to **0.0% at 50% coverage** / 8.7% at 70%, and it survives a length control (length alone
    0.806; length + score 0.865, score coefficient +1.79). Bonus finding that supports the project's
    confabulation thesis: **self-reported confidence is degenerate** — 358/362 replies claimed exactly
    1.0, including nearly every wrong answer (AUROC 0.510, chance).

    So: (a) ship the abstain/UI work on its own merits but relabel it honestly as **token-probability
    selective generation, not a white-box differentiator**; or (b) first feed an actually-internal score
    (J-lens readout / hidden-state probe) into `score_from_trace` and re-run this same harness — the only
    route to a legitimate white-box-advantage claim. Harness + data:
    `scratchpad/wb_analyze.py`, `wb_results.json`, `wb_raw*.jsonl`.

    ⚠️ **Option (b) HAS NOW BEEN RUN, and it fails — there is no white-box advantage here.** Same 362
    items, same TRAIN(180)/TEST(182) split. Candidates: ridge-logistic probes on layer-16/24 residuals
    (hyperparams CV'd inside TRAIN only), J-lens live-energy fraction, hidden-state norms. Scripts:
    `scratchpad/wb_harvest.py`, `wb_live_energy.py`, `wb_probe_analyze.py`, results `wb_probe_results.json`.

    On the full mixed test set the best probe *looks* like a win (AUROC 0.900 vs 0.822) — but a
    **1-bit topic control** (`is_hard_arith`, using zero model internals) scores **0.822**, matching the
    baseline exactly, and `corr(probe predictions, is_hard_arith) = -0.85` versus `-0.09` for the
    baseline. The probe is a topic detector, not a wrongness detector.

    Holding task type constant (hard-arithmetic subset, n=70) inverts the result:

    | arm | AUROC [95% CI] |
    |---|---|
    | whitebox_min (= min token logprob) | **0.937 [0.873, 0.984]** |
    | probe_mean16 (best internal) | 0.799 [0.686, 0.894] |
    | hidden_norm_mean_L16 | 0.709 [0.583, 0.833] |
    | live_energy_mean_k50 | 0.670 [0.544, 0.794] |

    Paired bootstrap (probe − baseline, shared resamples): full set **+0.079 [-0.007, +0.163]** (not
    significant); hard subset **-0.138 [-0.265, -0.029]** (baseline significantly BETTER). Verified
    independently against the raw JSON.

    **Conclusion: for risk prediction, the token probability wins and internal state adds nothing.**
    Take path (a) — ship selective generation on its merits, labelled honestly as token-probability-based.
    Scope: Qwen3.5-9B-Q4_K_M, greedy, question mix dominated by 4-6 digit multiplication; does not test
    other model families, non-greedy decoding, or an exhaustive layer sweep.
    Power caveat: 67 of 75 wrong items came from a constructed hard-multiplication stress set (only 8 from
    the natural probe corpus, where this model errs ~4%), so this is well-powered for arithmetic slips,
    not for general factual/reasoning failure. Also currently non-functional end-to-end: `:8080` is the raw
    C++ engine, not the Python gateway, and `~/.clozn/eval_report.json` is stale for this model and would be
    refused by `classify_run`'s exact-model provenance gate.
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
15. **Edit routes ~~B~~ / C** — Route **B** ✅ DONE 2026-07-18 (`steer_vec` on all 3 endpoints,
    exception-safe cleanup, heavn Edit concept input + strength slider). Route **C** free-text via
    LLaDA-8B-Instruct (the research swing) still open.
    *Why:* extend the edit vocabulary past Rewrite. *Payoff:* more Edit-drawer modes — steer content *by
    concept* inside a real bidirectional resolve (B, done); free-text edit instructions to a diffusion model (C).
16. **J-lens post-v1 (J5)** — Dream/denoise lens, chat-vs-web-text lens, stream top-k during generation.
    *Why:* extend the shipped "disposed to say" lens. *Payoff:* richer live readouts in the studio.
17. **Assembled-but-unconnected bets** — ~~model's-own-CI~~ (DONE 2026-07-18: `clozn test-model` CLI +
    `clozn.eval.golden` + 210-probe golden fixture saved on GPU; extended probe set added),
    ~~legible-basis microscope (OMP)~~ (DONE — shipped via anchored memory X7),
    branch-on-doubt, paraphrase-brittleness receipts.
    *Why:* validated-but-unwired research primitives. *Payoff:* each *could* become a studio feature;
    low-priority exploration.

## Phase 4 — product / UX polish

18. **Ambient channel-3** — inline confidence-shading right inside Cursor / the ChatGPT web UI (needs
    text↔trace alignment via `X-Clozn-Run-Id`). Highest effort of the three.
    *Why:* the ambient-delivery endgame. *Payoff:* clozn's confidence/trust shading **inside the tools
    people already use** — "zoom into the sketchy spans" without leaving their workflow.
19. ~~**Route-B "revise steer_vec" engine unlock**~~ ✅ DONE 2026-07-18 (see #15).
20. **Design-agent mock pack (D1–D5)** — only if pursuing the visual-polish direction (`notes/CLOZN_UX.md` §11).
    *Why:* optional visual polish. *Payoff:* a more finished studio look — only if we pursue it.

---

## Parked — needs an owner decision or an external unblock

- ~~**Lab artifact contracts + model qualification**~~ ✅ DONE 2026-07-18: `clozn qualify-whitebox <gguf>`
  landed — 39 model-free tests, honest per-feature capability matrix from contracts.gguf_identity + wave1
  ledger + local artifact lookup. Two feature families gated differently: core (receipts/explain/rewrite)
  qualified unconditionally; white-box (steering/j-lens/SAE) qualified only with real per-model data.
  Surfaced a real nuance: qwen2.5-7b has a calibrated steer tap layer (14) but wave1 dials status is
  `legacy_global_requires_model_scoped_recalibration` — correctly reports steering as NOT qualified.
  *Why/Payoff:* makes clozn's white-box features work on **any GGUF**, with an honest capability matrix —
  single-model → real multi-model platform.
- **Ollama drop-in?** — owner decision. ✅ Shim written 2026-07-18 (`clozn/server/routes/ollama.py`:
  `GET /api/tags|version`, `POST /api/generate|chat`; non-streaming only; version returns
  `"0.0.0-clozn"`). NOT registered in `app.py` — one-line wire when decided.
  *Why/Payoff:* point your existing Ollama tools at clozn and get white-box on top — an adoption lever.
- ~~**`real-runtime-smoke.yml` green**~~ ✅ DONE 2026-07-19 — **first green run ever**
  (`feat/night-2026-07-19`, run 29682484296, `conclusion=success`). The "zero jobs" parse bug was indeed
  already fixed; what still killed every run afterwards was that the job called `setup-python` and then
  never pip-installed anything, so the P0 step died at `ModuleNotFoundError: No module named 'numpy'`
  (`clozn.server.app` -> `routes/readouts.py` imports numpy at module scope). One added install step,
  pinned to ci.yml's `numpy==2.4.6` so the two product-minimal lanes can't drift. The green run builds the
  pinned CPU worker from source, downloads + SHA-verifies the real Qwen2.5-0.5B GGUF, and passes the P0
  contracts (46 tests, OK).
  *Why/Payoff:* a trustworthy green CI check that the real engine works — what makes a release credible.
