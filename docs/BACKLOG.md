# Clozn — Consolidated Backlog

**The single source of truth for open work.** Last reconciled 2026-07-17 against the shipped code, the
task history, and the scattered planning docs (`docs/ROADMAP.md`, `notes/*`, `CLOZN_REFACTOR_HANDOFF.md`).
Those docs are **stale** — most of what they list as "planned" already shipped. This file separates
*done* from *open* so we stop re-reading a to-do list of finished work.

Source tags: **[H:Px]** = refactor handoff phase · **[RM]** = docs/ROADMAP.md · **[FB §n]** =
notes/FRONTIER_BETS.md · **[UX]** = notes/CLOZN_UX.md · **[AMB]** = notes/AMBIENT_DELIVERY.md ·
**[EDIT]** = notes/EDIT_INSTRUCTIONS_DESIGN.md · **[EXPLAIN]** = docs/EXPLAIN_THIS_ANSWER_SPEC.md ·
**[MODEL]** = docs/MODEL_SUPPORT.md · **[SPLIT]** = docs/RUNTIME_SPLIT.md.

---

## Implementation order (suggested, updated 2026-07-17 post-reconciliation)

The constraint: **one GPU** (shared with a parallel effort) and helper agents that do Python / UI /
research-harness work but **cannot build the C++ engine or use the GPU**. Drain GPU-free work first, then a
batched GPU queue that **verifies already-merged work before building big new C++ lanes**. §1–§4 are the
*catalog*; this is the *sequence*.

**Wave A — GPU-free UI + harnesses — ✅ essentially DONE** (this reconciliation folds two parallel efforts
into one): heavn is the one frontend; Settings/profiles, Explain, quick-repair, Scope harvest, facts tier,
custom dial maker, memory-carrier controls, Read view, actuary panel, `clozn inspect`, trust/support
channels, and strict OpenAI compat all shipped. H7 + H3 research harnesses built (in `notes/`, run later).

**Wave B — the moment the GPU frees (verify-before-build, in order):** Route D live round-trip (does a real
model honor verbatim pins?) · H7 + H3 live captures (armed) · real-browser pass over all heavn UI ·
pressure-test the merged lanes (migrations / GC / cancellation) · engine-side cooperative cancel (small C++).

**Wave C — big C++ arc (GPU + build), dependency-ordered:** exact-state checkpointing + batched decode →
batched causal credit → **circuit tracer** (headline; last); device-resident readout plane alongside. Build
the two foundations before the credit / tracer work.

**Wave D — research lanes:** live risk controller (wire `eval/policy.py` into generation) · cross-model
diffing/transplants · native fast-weight memory · closed-loop disposition guardrails · AR×diffusion H2/H5 ·
Edit routes B/C · J-lens J5.

**Parked (unblock condition):** lab artifact contracts + qualification (in progress) · Ollama drop-in (owner
decision) · docs-heavy polish (owner-deferred) · real-runtime-smoke green (needs an owner `workflow_dispatch`).

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

- [x] **Stabilization pass** — DONE 2026-07-16: the 4 retrain-internals files (`test_memory_mode`,
  `test_memory_wiring`, `test_async_retrain`, `test_profiles_server`) trued up to the lab substrate —
  95/95 pass; the full suite is green (1589 passed, 0 failed).
- [x] **Live `clozn smoke`** **[H:P0]** — ✅ **DONE 2026-07-16**: ran against the real C++ worker + the pinned
  qwen2.5-0.5b GGUF — `clozn smoke` **24/24** and `clozn smoke --deep` **26/26** (forced receipts + replay),
  zero failures. Found + fixed one real bug (a `clozn stop` registry-cleanup race, `af53bbf`). Remaining
  sub-item: get the nightly `real-runtime-smoke.yml` green in CI (build the pinned worker on the Linux runner).
  🔧 **2026-07-16**: fixed the workflow's "failed run, zero jobs" parse failure -- root cause was a
  job-level `env:` block referencing the `runner` context, which GitHub's schema only allows inside
  `jobs.<job_id>.steps.*` (not `jobs.<job_id>.env`); rejected at parse time before any job is created.
  Moved `MODEL_PATH` to a `$GITHUB_ENV`-writing step instead. Verified locally (`yaml.safe_load` + manual
  schema review); **still needs a live `workflow_dispatch` run to confirm the CPU build + smoke steps
  themselves pass** -- that part was never reachable before this fix.
- [x] **Docs/claims refresh** **[RM][SPLIT][MODEL]** — DONE 2026-07-17: `MODEL_SUPPORT.md` and
  `RUNTIME_SPLIT.md` now describe the shipped template/J-lens/runtime paths, tie headline claims to
  repeatable tests or the exact Wave 1 qualification ledger, and state the cross-family white-box/artifact
  boundaries. README/ROADMAP overclaims and the stale sampled-run “top-k/top-p not enforced” metadata were
  corrected in the same audit.
- [x] **Security: neutralize the planted prompt-injection** **[SPLIT]** — DONE: the vendored
  `llama.cpp/CLAUDE.md` + `AGENTS.md` are gone from the local checkout, and `bootstrap_llama.py` strips
  them on every future bootstrap (47c5072), so the injection cannot come back with a re-vendor.

---

## 2. Runtime → production beta (the handoff's remaining P0-P2)

The "make it a product people can actually run" track. Ordered per the handoff's recommended sequence.

- [x] **Engine sampling on the serving path** **[RM][SPLIT P3][H]** — DONE (4839fa8). top_k/top_p now
  wired end to end: SampleOpts/SampleConfig → sample_from → sample.cpp truncates to top-k then the top-p
  nucleus before the seeded draw. Default ON with Ollama's canonical temp 0.8/top_k 40/top_p 0.9/rep 1.1
  (owner: default on, "feels like the same model they know from Ollama"). Greedy argmax path byte-identical
  → receipts/replay stay forced-greedy regardless. Verified: top_k=1==greedy, seed reproduces, sampled≠greedy.
- [x] **Worker protocol handshake** **[H:P1]** — DONE (b7433c9 handshake + 0205dfd stream envelope).
  `protocol_version` "1.0" + a `capabilities` object on engine `/health` + gateway `/readyz`; the
  supervisor (`spawn_engine`) refuses an incompatible/missing major (terminates + raises, message says to
  rebuild) instead of driving a worker blind. Every native SSE frame is stamped with `req` (request id) +
  a monotonic per-request `seq` (StreamEnvelope; completions/infill/revise/board, legacy + protocol:true).
  Golden fixture `protocol/fixtures/handshake.json` guards C++ header, Python constant, and the /health
  capability keys via `tests/test_protocol_handshake.py`. Verified live + product smoke 24/24. *Follow-up:
  wire Studio to read the same fixture (it can today; not yet consumed) + seq-gap detection on consumers.*
- [x] **Request isolation + cancellation** **[H:P1][SPLIT P4]** — DONE 2026-07-16 (73ab294 + caae941 +
  c603b49). `clozn/server/request_context.py`: one per-call `RequestContext` (request id, sampling, memory
  manifest, steering snapshot, trace, finish reason, threading.Event cancellation) atomically published as
  `sub._request` — the old five piecemeal `_last_*` writes (torn-read hazard) are now read-only property
  views. POST_GATE waits are cancellable (`client_gone` socket probe → frees the queue slot, HTTP 499) and
  serialization is scoped: two audited-safe POSTs exempted; the rest stay serialized because steer/memory
  state is STILL shared (documented in app.py). sse.py distinguishes client-disconnect (cancel + stop
  writing) from worker-dies-midstream (honest error frame + [DONE], finish_reason never "stop");
  `gen.close()` unconditional. Verified: 22 new tests, suite 1653/0, live smoke 24/24 + --deep 26/26.
  *Follow-ups: engine-side cooperative cancel (C++); non-streaming chat() has no mid-flight cancel; true
  concurrent generation needs steer/memory de-globalized; correlate `req_` ids with the worker's `req`.*
- [x] **Persistence: migratable + trustworthy** **[H:P1]** — DONE (46b03a1 migrations engine + 0c5cade
  evidence-write honesty + e6898e2 blob GC + fb01a32 `clozn migrate` CLI). `clozn/runs/migrations.py`:
  versioned, ordered, transactional migration steps (each its own BEGIN IMMEDIATE/COMMIT/ROLLBACK -- a
  mid-migration failure rolls back cleanly, DB stays usable at the prior version) replace `_ensure()`'s ad
  hoc executescript-stamping; the ledger reuses `schema_meta` so a fresh migrated DB's schema stays
  byte-identical to the old `_ensure()`'s (proven by schema-dump diff), and a legacy `_ensure()`-built DB
  upgrades in place losslessly. `clozn migrate [--dry-run] [--gc] [--json]`: reports current/target
  version + applies pending migrations, or (`--gc`) garbage-collects blob files no run row references
  (dry-run by default, path-containment-checked, TOCTOU-safe). Trace-blob digests verified on read
  (already done, commit 6409535). `_store_trace`'s write failure (used to propagate through `record()`'s
  blanket except and silently drop the WHOLE run) is now caught, logged, and marked -- the run row still
  lands with an honest "evidence-missing" flag/meta instead of vanishing or reading as "no trace ever
  existed". 42 new tests (migrations, GC, CLI, evidence-write); full suite 1631 passed/11 skipped (skips
  all pre-existing model-gated).
- [x] **Client compatibility matrix** **[H:P2]** — DONE 2026-07-17: `OPENAI_COMPATIBILITY.md` publishes
  the exact endpoint/field subset; one validator rejects unknown or behavior-bearing unsupported fields
  with OpenAI-shaped 400s and strips only documented neutral values. Explicit sampling fields now reach
  both streaming and non-streaming engine calls, `max_completion_tokens` is supported, fake zero usage was
  removed, and the CPU CI lane drives the real `openai` Python client against a model-free live gateway.
- [x] **Offline calibration sweep batching + resume** — ✅ **DONE 2026-07-17** (`4cf1d93`): dial sweeps
  now batch generation and activation-alignment work behind a tunable `--batch-size`, share the unsteered
  baseline across dials, checkpoint every completed dial, and safely `--resume` only compatible runs.
  This improves the next calibration job without touching the occupied GPU now; it is separate from the
  still-open multi-request serving scheduler below.
- [ ] **Owner decision: Ollama drop-in?** **[H:P2]** — recommended thin
  `/api/chat|generate|tags|version`; intentionally not inferred from OpenAI compatibility work.
- [x] **CI lanes + release artifact** **[H:P2]** — ✅ **DONE 2026-07-16**: root `pyproject.toml` (setuptools)
  + `setup.py` (the studio/protocol `package_dir` remap `find()` can't express) + single-source version
  (`clozn.__version__`) + `clozn = clozn.cli.main:main` console entry point. `clozn version` (+ git commit
  when in a checkout) and `clozn doctor` (engine binary / models / studio assets / registry staleness /
  protocol version / python version -- warns, never fails, on a merely-missing engine) both land in
  `clozn/cli/commands/`. Studio + `protocol/` ship inside the wheel as `clozn.studio` /
  `clozn.protocol_fixtures`, with `clozn/server/config.py`'s `DEMO` gaining a third packaged-mode fallback
  (env var → repo layout → `importlib.resources`) -- caught and fixed a real bug this way (a namespace-
  package `MultiplexedPath` silently broke the packaged asset lookup; fixed by giving `studio/`/`protocol/`
  a marker `__init__.py`). `scripts/release/clean_room_install_test.py` builds the wheel, installs into a
  throwaway venv, and proves `import clozn` / `python -m clozn version` / `python -m clozn doctor` / the
  `clozn` console script all work from a scratch cwd outside the repo -- wired as CI's new `packaging` job.
  Engine build provenance (llama.cpp pin from `bootstrap_llama.py`) surfaces via `doctor` when a binary is
  found; no build-flags/commit record is embedded in the binary itself yet (proposed, not built: see the
  session report for a concrete `server_main.cpp`/CMakeLists.txt sketch). *(product-minimal CI lane
  already done.)*
- [ ] **Lab artifact contracts + model qualification** **[H:P2]** — 🔄 **IN PROGRESS in a concurrent
  session** (`clozn/artifacts/contracts.py`, `docs/qualification/`, model registry). One manifest format
  (J-lens / dials / SAE), validated before load; qualify a hero model per architecture. Related **[MODEL]**:
  parameterize the ~73 hardcoded Qwen literals into a registry; **verify white-box taps on a 2nd
  architecture** (Gemma/Llama) — the honest "any GGUF" check; LLM-judge for push-button Tier-1 dial sweeps.
- [ ] **Native exact-state checkpointing + token-exact branching** — a serializable/clonable `EngineState`
  in the C++ worker: KV cache + exact token ids + generation position + sampler/RNG state + active
  interventions + model/artifact hashes, with snapshot/clone/restore/lifecycle in the engine rather than
  only in Python descriptors. Today `clozn/replay/timetravel.py` is descriptor-only and rebuilds a branch by
  re-prefilling a transcript, and `clozn/replay/fork.py` can cross tokenizer boundaries — useful replay, not
  exact computational time-travel. Add copy-on-write / paged KV so branches share a prefix instead of
  re-prefilling; patch a residual/feature at token t then resume from precisely that state; snapshot compat
  fails closed across model/tokenizer/quant/context. Correctness bar: bit-exact greedy suffix after
  save→restore, deterministic sampled suffix when RNG+sampler state are restored. Big C++ lane.
- [ ] **Device-resident multi-observer readout plane** — replace the single-owner activation tap (live
  J-lens is exclusive with other white-box modes + does a CPU transport/unembed; the SAE path starts
  host-resident — `engine/core/serve/server_main.cpp`, `server_shared.hpp`, `include/cloze/sae.hpp`) with a
  per-request multi-subscriber plan: keep selected-layer activations device-resident, fan them out to J-lens
  / SAE top-k / probes / norms on an async side stream, copy back only compact top-k / scalar records;
  multiple observed layers + observers per request; bounded queues, cancellation, backpressure. Bar: all
  observers together at a measured <5–10% throughput overhead with CPU-parity under real concurrent serving,
  not isolated kernel speed. Engine perf + interpretability in one lane; composes with the readout plumbing
  the protocol handshake + StreamEnvelope already carry.
- [ ] **Batched multi-sequence serving decode** **[SPLIT P4]** — the shared-prefix batch primitive (auth/TLS
  too if remote binding is ever added). This same batch API unlocks **interaction-aware causal credit**:
  teacher-force many counterfactual arms in one pass → coalition / sampled-Shapley scoring + pairwise and
  higher-order interaction terms, with uncertainty + matched-null arms. Today `clozn/receipts/core.py` is
  sequential leave-one-out + pairwise-only and (by its own note) misses 3-way-and-higher interactions;
  batched and sequential scores must agree within a documented tolerance.

**Owner decisions still open** **[H]**: Ollama compat · network exposure (loopback vs remote+auth) ·
release order (Linux-CPU-first recommended) · worker distribution (prebuilt vs build-local) · reference
model (✅ decided: Qwen2.5-0.5B).

---

## 3. Research frontier (the genuinely-open bets)

- [ ] **AR×diffusion H2/H3/H5/H7** **[FB §3]** — all spec'd (`notes/ar_diffusion/specs/`), none run. H1 dead.
  Cheapest-decisive first: **H7 divergence atlas** → H3 substrate routing → H2 score-gated self-repair →
  H5 counterfactual-patch receipts (⚠ needs a `/v1/revise` ablated-context build spike first).
- [ ] **Edit-instruction routes** **[EDIT]** — Route **D** "Rewrite (AR)" ✅ SHIPPED 2026-07-16 (3db1e2d +
  7b686fc): `POST /engine/rewrite` — pins ride as keep-verbatim prompt constraints, pin fidelity MEASURED
  post-hoc (per-pin kept:true/false, never assumed), every response carries the honest note "regenerates
  the unpinned text — not a bidirectional resolve" on the wire; studio Edit drawer gained a RESOLVE/REWRITE
  toggle. 27 tests, model-free-verified; ⏳ live engine round-trip + real-browser check pending GPU.
  Still open: Route **B** content-concept via `dir(c)` (validated ~dozen-line engine unlock); Route **C**
  free-text via LLaDA-8B-Instruct (engine has native LLaDA) — the research swing.
- [ ] **Closed-loop disposition guardrails** **[FB §9.1]** — "the biggest unclaimed frontier": mid-gen lens
  polling → threshold → `dir(c)` counter-injection, on a banned-topic battery.
- [ ] **Calibration next rungs → live risk controller** **[CALIBRATION_FINDINGS]** — ✅ exact-model
  scalar-temperature fitting, provenance/probe-count gates, and the separate Replay truth channel shipped
  (`6af7cc0`). Still open: bigger probe sets + CIs; render the full truth-tier curve in Studio; and close
  the loop into the runtime — wire the offline answer/ask/abstain policy (`clozn/eval/policy.py`, today
  selection-only — product generation never executes it) into live generation so a request can answer, ask
  to clarify, abstain, or run a counterfactual check before answering. Condition calibration on
  model/quant/decode-policy/task; drift-invalidate a stale policy; log decision→evidence→action→verified
  outcome; and prove it beats strong black-box + random-signal baselines held-out with CIs.
- [ ] **Intervention-validated circuit tracer** — an attribution graph over input tokens, attn/MLP
  components, SAE/transcoder features, a residual-error node, and output logits; attribute signed logit
  contribution through it while keeping an explicit *unexplained-mass* term (a sparse graph must never imply
  omitted computation was zero); make every selected path testable by patch/inhibit/ablate at the exact
  token+layer, and compare predicted vs observed logit movement against matched random-node,
  random-direction, and shuffled-edge controls; report stability across paraphrases/seeds/quant/pruning
  thresholds; emit a machine-readable graph + intervention receipt. `clozn/analysis/microscope.py` is the
  correlational precursor (sparse directional decomposition, not intervention-validated); ARCHITECTURE marks
  circuit tracing + scaled-transcoder inference unbuilt. Leans on the exact-state + batched-scoring
  foundations in §2; the largest research lane here — build those first.
- [ ] **Cross-model causal state diffing + transplants** — align residual spaces across model / quant
  variants (CKA/CCA/Procrustes or a shared feature dict, held-out alignment quality reported); locate the
  first statistically-meaningful internal divergence *before* the first output-token divergence; transplant
  an aligned residual/feature from model A into model B at an exact token+layer and measure how much of A's
  target-logit behavior it recovers in B; decompose quant/finetune regressions by layer + component + signed
  logit effect, with reverse / adjacent-layer / random-state controls. `clozn/analysis/model_diff.py` is
  observational-only after divergence today; `clozn/receipts/quant_receipts.py` is the common-continuation
  scoring baseline to build on.
- [ ] **Native fast-weight fact memory** **[FABLE]** — port the keyed / fast-weight memory mechanism into
  the native engine (today `clozn/server/facts_store.py`'s read receipt does NOT alter the product reply,
  and `clozn/memory/slotmem_qwen` is a research-only substrate): surprise-gated write/update/exact-delete
  with stable inspectable ids; topic+confidence-gated reads injected into generation at a known token/layer;
  with-memory / without-memory / matched-null receipts on every claimed recall; abstain on ambiguous
  retrieval rather than forcing a memory in. Bar: recall/abstention curves 10→10k memories with
  unrelated-fact + near-neighbor controls, plus exact-delete + ghost-memory (residue) probes. Cross-refs the
  §4 facts-tier UI (now shipped).
- [ ] Assembled-but-unconnected bets **[FB §9.3-9.9]** — model's-own-CI, legible-basis microscope (OMP),
  branch-on-doubt, paraphrase-brittleness receipts, cross-model disposition transfer (pilot).
- [ ] J-lens post-v1 (J5) — Dream/denoise lens, chat-vs-web-text lens, stream top-k during generation.

---

## 4. Product / UX polish (heavn is built; these are refinements)

- [x] **UI consolidation → heavn only** (owner decision 2026-07-17: one frontend). ✅ CUT: instrument/
  studio/jlens/brain/app.html + clozn.css + pages/agent.js + pages/lab.js (all dead or superseded — the
  legacy pages assumed the pre-split multi-substrate server; /say, /denoise, /engine/concepts 409/410 on
  the product gateway). `denoise.html` re-homed as a lab-served page (works only under `clozn lab dream`).
  ✅ Theme aligned to the ambient-runtime refs (honesty pills, live /readyz nav-footer, killed a fabricated
  98% health stat). **COMPLETED PORT QUEUE** (legacy sources were retained until their replacements landed):
  ✅ profiles CRUD → heavn Settings (snapshot/update/switch/import/export/delete; active-delete guarded)
  ✅ facts tier UI → heavn Memory (mode/list/add/read/delete with surprise + abstention receipts)
  ✅ engine.html harvest/observe → heavn Scope; `/steer/*` is canonical and `/engine/steer/*` is a deprecated
  compatibility facade over the same EngineSteer. ✅ run-view ports: narrate + explain + lineage tree +
  quick-repair presets in Replay, propose-memory in Memory. ✅ memory-mode selector + strength slider → heavn
  Memory. ✅ learned-preference suggestions + custom dial maker (create, honest model-cost note, two-step
  delete) → heavn Patch. ✅ CUT: the remaining legacy `pages/*.js` bundle + `engine.html`; heavn is the only
  product frontend, while `denoise.html` remains an explicitly lab-served Dream page.

- [x] **Document-first "Read view"** **[AMB]** — `/r/<id>` now lands on the recorded answer first, with
  raw-confidence shading, contiguous sketchy-span zoom + captured alternatives, and explicit
  commitment-not-correctness copy. Missing permalinks never substitute a different run; ordinary Studio
  entry still defaults to Replay.
- [ ] **Ambient channel-3 endgame** **[AMB]** — inline confidence-shading right inside Cursor / the ChatGPT
  web UI (needs text↔trace alignment via `X-Clozn-Run-Id`). Highest effort.
- [x] **Trust in the UI: fold in the SUPPORT channel** **[FB §1.1][FABLE]** — Replay now separates three
  channels: acceptance proxy, outcome-grounded scalar-temperature calibration, and an explicit opt-in NLI
  entailment check against stored causal receipts when present (otherwise a plainly labeled active manifest).
  Truth estimates require exact model/score provenance and ≥50 labeled probes before they can drive shading;
  support never masquerades as external evidence or a
  fact-check, and an unavailable NLI model never falls back under the same label. The any-client
  `clozn_trust` extension remains raw/uncalibrated and says so; this completion is the Studio trust surface.
- [x] **Two-tier memory surfacing** **[FABLE][STUDIO]** — heavn Memory now presents the anchored α lookup
  as the inspectable product carrier beside the opaque, lab-only soft prefix, while explicitly keeping
  prompt cards separate. Each anchored bag can run its existing baseline/anchored/equal-magnitude-null
  receipt against the current recorded run; the UI discloses its 2–3 fresh-generation cost before launch.
- [x] **Actuary productization** **[FB §9.2]** — heavn Read now shows the cached proxy calibration/drift
  report and a server-scored, past-only failure resemblance. The selected run and timestamped-later runs are excluded;
  warnings require ≥5 earlier organic runs in each proxy class + score ≥0.65, while smaller samples are
  shown as weak evidence and cannot alert. Every surface says behavioral proxy, not correctness.
- [x] **EXPLAIN "Explain this answer" — any-client surface** **[EXPLAIN M5]** — `clozn inspect <run_id>`
  now assembles the zero-generation explanation directly from the local SQLite journal (`--last` and
  exact `--json` included), falling back to a specified gateway only for non-local ids. This consumes the
  existing `clozn_run_id` / `X-Clozn-Run-Id` bridge; `clozn explain --why` remains the explicit generative path.
- [x] Calibrated-trust upgrade to the Replay footer/shading (F2) **[AMB]** — exact-model scalar-temperature
  fit, probe-count gate, and separate acceptance/support channels; no truth fit means no correctness estimate.
- [ ] Route-B "revise steer_vec" engine unlock for content edits **[EDIT]**.
- [ ] The CLOZN_UX §11 design-agent mock pack (D1-D5), if pursuing the visual polish.

---

## Notes

- This file supersedes the *open-work* sections of `docs/ROADMAP.md` and the `notes/*` roadmaps; those stay
  as narrative/results archives. When an item lands, move it to §0.
- The **honesty invariants** (every readout carries a null; measured never self-reported; negatives ship as
  labels; discrimination-not-awareness framing) are binding on everything here — see `notes/CLOZN_SOUL.md`.
