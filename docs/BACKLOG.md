# Clozn — Consolidated Backlog

**The single source of truth for open work.** Last reconciled 2026-07-16 against the shipped code, the
task history, and the scattered planning docs (`docs/ROADMAP.md`, `notes/*`, `CLOZN_REFACTOR_HANDOFF.md`).
Those docs are **stale** — most of what they list as "planned" already shipped. This file separates
*done* from *open* so we stop re-reading a to-do list of finished work.

Source tags: **[H:Px]** = refactor handoff phase · **[RM]** = docs/ROADMAP.md · **[FB §n]** =
notes/FRONTIER_BETS.md · **[UX]** = notes/CLOZN_UX.md · **[AMB]** = notes/AMBIENT_DELIVERY.md ·
**[EDIT]** = notes/EDIT_INSTRUCTIONS_DESIGN.md · **[EXPLAIN]** = docs/EXPLAIN_THIS_ANSWER_SPEC.md ·
**[MODEL]** = docs/MODEL_SUPPORT.md · **[SPLIT]** = docs/RUNTIME_SPLIT.md.

---

## 0. Already shipped (do NOT re-do — the docs still list much of this as "planned")

- **Product/lab split** — Stages 1-7, landed on `main` 2026-07-16 (`ec5a0bc..10b9215`). This IS the
  handoff's **P1 physical split**: lab Torch code relocated to `clozn/lab/`, product provably torch-free
  (enforced by the `product-minimal` CI lane), lab owns its handler via an injectable substrate with zero
  product-global mutation, internalized retrain moved to a lab mixin.
- **Live runtime validated (the handoff's P0)** — 2026-07-16: `clozn smoke` 24/24 + `--deep` 26/26 (forced
  receipts + replay) against the real C++ worker + the pinned 0.5B GGUF. First end-to-end proof the actual
  C++/GGUF stack works, not just model-free tests. Found + fixed a `clozn stop` registry-cleanup race.
- Engine white-box runtime (`/harvest`, `/score`, steer taps, prompt-mode memory); **J-lens** live at
  `/jlens` + studio panel; causal receipts (prove-all leave-one-out, forced, graded leaning; early-stop);
  **`clozn eval`** outcome-grounded calibration; Tier-0 any-AR-GGUF; sampling (S5); tiny-test; branch/lineage.
- **heavn studio** — the live app, incl. the **Experiment drawer**, **click-a-span → signals popover**,
  and **anchored memory** (X7) end-to-end.
- **Ambient delivery** — channel 1 (receipt footer), channel 2 (`clozn watch` alerts + server push),
  channel 3 (userscript v1). Per-run permalink `/r/<id>`.
- **Pin-and-resolve HARD invariant** — validated, 2,153/2,153 checks, zero violations ("ship it").
- Performance: prefix/KV reuse, fit planner, quant-ladder receipts (`clozn quant-check`), verify-then-escalate.
- Introspection pack X1-X8 — largely run (self-report receipts, injected-thought detection, k*/k_J, X7/X8).
- Actuarial journal (`clozn/runs/actuary.py`) — built; `clozn eval` truth-tier calibration — live.
- **[FB] H1 (diffusion-drafts / AR-verifies speedup) — KILLED** (0.19-0.47x, net slowdown). Do not retread
  as a *local* speedup. (Distilled-drafter variants remain theoretically open, low priority.)

---

## 1. Close out the refactor (SMALL, HIGH-VALUE — do first)

These make the branch we just pushed trustworthy and the docs honest. All small.

- [ ] **Stabilization pass** — true up the ~63 retrain-internals tests deliberately left red (they patch
  the moved `cs._RETRAIN*` / `_join_retrain`; repoint to the substrate). Files: `test_memory_mode`,
  `test_memory_wiring`, `test_async_retrain`, `test_profiles_server`. Greens the full `python` CI lane.
- [x] **Live `clozn smoke`** **[H:P0]** — ✅ **DONE 2026-07-16**: ran against the real C++ worker + the pinned
  qwen2.5-0.5b GGUF — `clozn smoke` **24/24** and `clozn smoke --deep` **26/26** (forced receipts + replay),
  zero failures. Found + fixed one real bug (a `clozn stop` registry-cleanup race, `af53bbf`). Remaining
  sub-item: get the nightly `real-runtime-smoke.yml` green in CI (build the pinned worker on the Linux runner).
- [ ] **Docs/claims refresh** **[RM][SPLIT][MODEL]** — `MODEL_SUPPORT.md` + `RUNTIME_SPLIT.md` still list
  *shipped* things (Tier-0 chat templating, J-lens) as blockers. Trace every headline claim to a measurement.
- [ ] **Security: neutralize the planted prompt-injection** **[SPLIT]** — `engine/.../llama.cpp/CLAUDE.md`
  (inside the *vendored* checkout) contains a prompt-injection; delete/neutralize so no future agent obeys it.

---

## 2. Runtime → production beta (the handoff's remaining P0-P2)

The "make it a product people can actually run" track. Ordered per the handoff's recommended sequence.

- [ ] **Engine sampling on the serving path** **[RM][SPLIT P3][H]** — top-p/k is unwired (engine is
  greedy/temp/rep-penalty only). The single most-repeated open item across three docs. **Owner decision:
  default on or off?** (Receipts/replay stay forced-greedy regardless.)
- [ ] **Worker protocol handshake** **[H:P1]** — `protocol/SPEC.md` is prose, not a contract. Add
  `protocol_version` + capabilities to `/health` + `/readyz`; supervisor refuses an incompatible major;
  request-id + monotonic event seq on native streams; golden fixtures shared by C++/Python/Studio.
- [ ] **Request isolation + cancellation** **[H:P1][SPLIT P4]** — per-request context (id, sampling, memory
  manifest, steering snapshot, trace, finish reason, cancellation) instead of shared process globals; this
  removes the reason every POST is globally serialized. Propagate client disconnect → cancel the worker;
  define worker-dies-midstream behavior. *(Stage 2's injectable substrate is groundwork for this.)*
- [ ] **Persistence: migratable + trustworthy** **[H:P1]** — replace `_ensure()` schema stamping with real
  transactional migrations + `clozn migrate`/doctor; verify trace-blob digests on read; GC unreferenced
  blobs; stop silently swallowing evidence-write failures.
- [ ] **Client compatibility matrix** **[H:P2]** — publish an endpoint/field support matrix; unsupported
  fields → OpenAI-shaped 4xx or documented-ignored (no silent pretend). Integration tests with the real
  OpenAI client. **Owner decision: Ollama drop-in?** (recommended: thin `/api/chat|generate|tags|version`).
- [ ] **CI lanes + release artifact** **[H:P2]** — root `pyproject.toml` + version + console entry point;
  `pip install` into a clean env yields a working `clozn` with no repo-path hacks; `clozn version` +
  `clozn doctor`; package Studio/protocol/worker assets. *(product-minimal CI lane already done.)*
- [ ] **Lab artifact contracts + model qualification** **[H:P2]** — 🔄 **IN PROGRESS in a concurrent
  session** (`clozn/artifacts/contracts.py`, `docs/qualification/`, model registry). One manifest format
  (J-lens / dials / SAE), validated before load; qualify a hero model per architecture. Related **[MODEL]**:
  parameterize the ~73 hardcoded Qwen literals into a registry; **verify white-box taps on a 2nd
  architecture** (Gemma/Llama) — the honest "any GGUF" check; LLM-judge for push-button Tier-1 dial sweeps.
- [ ] Batched multi-sequence decode; auth/TLS if remote binding is ever added **[SPLIT P4]**.

**Owner decisions still open** **[H]**: Ollama compat · network exposure (loopback vs remote+auth) ·
release order (Linux-CPU-first recommended) · worker distribution (prebuilt vs build-local) · reference
model (✅ decided: Qwen2.5-0.5B).

---

## 3. Research frontier (the genuinely-open bets)

- [ ] **AR×diffusion H2/H3/H5/H7** **[FB §3]** — all spec'd (`notes/ar_diffusion/specs/`), none run. H1 dead.
  Cheapest-decisive first: **H7 divergence atlas** → H3 substrate routing → H2 score-gated self-repair →
  H5 counterfactual-patch receipts (⚠ needs a `/v1/revise` ablated-context build spike first).
- [ ] **Edit-instruction routes** **[EDIT]** — Route **D** "Rewrite (AR)" mode is buildable now, zero engine
  work; Route **B** content-concept via `dir(c)` is a validated ~dozen-line engine unlock; Route **C**
  free-text via LLaDA-8B-Instruct (engine has native LLaDA) is the research swing.
- [ ] **Closed-loop disposition guardrails** **[FB §9.1]** — "the biggest unclaimed frontier": mid-gen lens
  polling → threshold → `dir(c)` counter-injection, on a banned-topic battery.
- [ ] **Calibration next rungs** **[CALIBRATION_FINDINGS]** — bigger probe sets + CIs; a retrieval/clarify
  action wired to the policy's `ask` band; render the truth-tier curve in studio.
- [ ] Assembled-but-unconnected bets **[FB §9.3-9.9]** — model's-own-CI, legible-basis microscope (OMP),
  branch-on-doubt, paraphrase-brittleness receipts, cross-model disposition transfer (pilot).
- [ ] J-lens post-v1 (J5) — Dream/denoise lens, chat-vs-web-text lens, stream top-k during generation.

---

## 4. Product / UX polish (heavn is built; these are refinements)

- [ ] **Document-first "Read view"** **[AMB]** — make it the default `/r/<id>` landing ("read it, zoom into
  the sketchy spans"). The docs call this *"the answer to why use clozn instead of just calling the model."*
- [ ] **Ambient channel-3 endgame** **[AMB]** — inline confidence-shading right inside Cursor / the ChatGPT
  web UI (needs text↔trace alignment via `X-Clozn-Run-Id`). Highest effort.
- [ ] **Trust in the UI: fold in the SUPPORT channel** **[FB §1.1][FABLE]** — trust spans are
  confidence-only today; add receipts/NLI support + real calibration (temperature-scaling), not just raw probs.
- [ ] **Two-tier memory surfacing** **[FABLE][STUDIO]** — the anchored "what did you learn?" α-lookup receipt
  UX + how to show anchored vs soft-prefix tiers honestly.
- [ ] **Actuary productization** **[FB §9.2]** — `GET /journal/actuary` endpoint + studio panel + a live
  "resembles past failures" warning (model is built; the surfacing isn't).
- [ ] **EXPLAIN "Explain this answer" — remaining surfaces** **[EXPLAIN M5]** — M1-M4 shipped; the
  any-client bridge (`clozn inspect` off a returned run_id) + web "Explain" tab remain.
- [ ] Calibrated-trust upgrade to the footer/alerts (F2) **[AMB]**; Route-B "revise steer_vec" engine unlock
  for content edits **[EDIT]**; the CLOZN_UX §11 design-agent mock pack (D1-D5) if pursuing the visual polish.

---

## Notes

- This file supersedes the *open-work* sections of `docs/ROADMAP.md` and the `notes/*` roadmaps; those stay
  as narrative/results archives. When an item lands, move it to §0.
- The **honesty invariants** (every readout carries a null; measured never self-reported; negatives ship as
  labels; discrimination-not-awareness framing) are binding on everything here — see `notes/CLOZN_SOUL.md`.
