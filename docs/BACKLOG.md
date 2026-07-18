# Clozn — Implementation Roadmap

**Remaining work only, in the order to do it.** Shipped work lives in git history (the pre-2026-07-17
reconciliation backlog carried the full done-archive if you need it). Binding on everything below: the
**honesty invariants** — every readout carries a null; measured never self-reported; negatives ship as
labels; discrimination-not-awareness framing (`notes/CLOZN_SOUL.md`).

The sequence is GPU-shaped: **verify already-merged work before building the big C++ lanes.** VRAM (not
compute) is the live limit — one 0.5B engine ≈ 2.7 GB fits the current ~3 GB headroom (one task at a time);
anything needing Qwen-7B + Dream co-resident (~13 GB) waits for the parallel effort to free ~10 GB.

---

## Phase 1 — GPU verify (now; fits the 0.5B budget)

Validate what's already merged before pouring effort into new C++.

1. **Route D live round-trip** — does a real model honor the verbatim pins? Validates the shipped Rewrite
   (AR) mode; the pin-fidelity chips report honestly. Single engine boot, no worker restart.
2. **Real-browser pass over heavn UI** — the reconciled UI only had model-free render checks; click through
   Settings / Explain / quick-repair / Scope / Read / actuary against a live engine.
3. **Pressure-test the merged lanes** — hostile hands-on over migrations / GC / cancellation (kill workers
   mid-stream, disconnect clients, corrupt/fill the blob store). Has surfaced real bugs every time.
4. **Engine-side cooperative cancel** (small C++) — today "cancel" just drops the socket and the worker
   keeps decoding; make it actually stop. Completes the request-isolation lane.
5. **H7 + H3 live captures** — ⛔ **blocked on VRAM**: need Qwen-7B + Dream co-resident (~13 GB). Harnesses
   armed in `notes/ar_diffusion/{h7,h3}/`; two commands each once ~10 GB frees.

## Phase 2 — the big C++ arc (GPU + build; dependency-ordered)

Build the two foundations before the headline.

6. **Native exact-state checkpointing + token-exact branching** — serializable/clonable `EngineState` (KV +
   token ids + gen pos + sampler/RNG + active interventions + hashes), snapshot/clone/restore in C++, plus
   copy-on-write / paged KV so branches share a prefix (today `clozn/replay/timetravel.py` is
   descriptor-only and re-prefills). Bar: bit-exact greedy suffix after save→restore.
7. **Batched multi-sequence decode** — the shared-prefix batch primitive; also unlocks **batched causal
   credit** (coalition / Shapley over teacher-forced counterfactual arms — `clozn/receipts/core.py` is
   sequential leave-one-out + pairwise-only today).
8. **Device-resident multi-observer readout plane** — replace the single-owner activation tap; keep layer
   activations device-resident, fan out to J-lens / SAE / probes / norms on an async stream. Bar: all
   observers together at <5–10% throughput overhead, measured under concurrent load.
9. **Intervention-validated circuit tracer** *(headline — needs 6 + 7 first)* — attribution graph with an
   explicit unexplained-mass term; every path patch/inhibit/ablate-testable at the exact token+layer;
   predicted-vs-observed logit movement against random-node / direction / shuffled-edge controls.
   `clozn/analysis/microscope.py` is the correlational precursor.

## Phase 3 — research lanes

10. **Live risk controller** — wire the offline answer/ask/abstain policy (`clozn/eval/policy.py`, today
    selection-only, never executed in generation) into live generation; beat black-box baselines held-out
    with CIs. (`clozn/eval/calibration.py` is the metric base.)
11. **Cross-model causal state diffing + transplants** — align residual spaces across model/quant variants,
    transplant an aligned state A→B at a token+layer and measure the recovered target-logit, decompose
    quant/finetune regressions. `clozn/analysis/model_diff.py` is observational-only after divergence today.
12. **Native fast-weight fact memory** — port keyed/fast-weight memory into the engine so a recall actually
    alters the reply (`clozn/server/facts_store.py`'s read receipt does not today), with
    with-memory / without-memory / matched-null receipts and abstain-on-ambiguity.
13. **Closed-loop disposition guardrails** — mid-gen lens polling → threshold → `dir(c)` counter-injection,
    on a banned-topic battery. ("The biggest unclaimed frontier.")
14. **AR×diffusion H2 + H5** — H7/H3 harnesses are built (Phase 1 captures); **H2** (score-gated
    self-repair) and **H5** (counterfactual-patch receipts — ⚠ needs a `/v1/revise` ablated-context build
    spike first) remain. Specs in `notes/ar_diffusion/specs/`.
15. **Edit routes B / C** — Route **B** content-concept via `dir(c)` (a validated ~dozen-line engine
    unlock); Route **C** free-text via LLaDA-8B-Instruct (the research swing). Route D shipped.
16. **J-lens post-v1 (J5)** — Dream/denoise lens, chat-vs-web-text lens, stream top-k during generation.
17. **Assembled-but-unconnected bets** — model's-own-CI, legible-basis microscope (OMP), branch-on-doubt,
    paraphrase-brittleness receipts, cross-model disposition transfer (pilot).

## Phase 4 — product / UX polish

18. **Ambient channel-3** — inline confidence-shading right inside Cursor / the ChatGPT web UI (needs
    text↔trace alignment via `X-Clozn-Run-Id`). Highest effort of the three.
19. **Route-B "revise steer_vec" engine unlock** — content-concept edits inside a real bidirectional
    resolve (pairs with #15's Route B).
20. **Design-agent mock pack (D1–D5)** — only if pursuing the visual-polish direction (`notes/CLOZN_UX.md` §11).

---

## Parked — needs an owner decision or an external unblock

- **Lab artifact contracts + model qualification** — 🔄 in progress in the parallel effort
  (`clozn/artifacts/contracts.py`, `docs/qualification/`, model registry). End-state: a one-command
  `clozn qualify-whitebox <gguf> --checkpoint <org/model>` proven on Qwen + Gemma + Llama.
- **Ollama drop-in?** — owner decision (recommended: a thin `/api/chat|generate|tags|version`).
- **`real-runtime-smoke.yml` green** — the "zero jobs" parse bug is fixed but unverified; needs an owner
  `workflow_dispatch` click to confirm the pinned-worker GGUF test actually schedules and runs.
