# Roadmap — Clozn

From where we are now (a fresh monorepo with the [blueprint](ARCHITECTURE.md) committed)
to the full vision: **a local runtime where you view and steer a model's memory — at
scale.** Phases are roughly sequential; each gates on tests/validation before the next.

**Honest effort framing.** Phases 0–2 are *engineering* (weeks): consolidate, unify the
protocol, one inspector over every substrate. Phases 3–4 are the *research frontier*
(months, and some rungs may not pan out — that's the method: build the smallest thing that
could kill the idea, let the result decide). Phase 5 is productization, parallelizable.

Status legend: ⬜ todo · 🟡 in progress · ✅ done. The engine's diffusion+AR white-box,
the confidence-select kernel, and the inspector's toy SAE/transcoder already exist (see the
[maturity ladder](ARCHITECTURE.md#where-the-depth-lives--the-interp-maturity-ladder)) — this
roadmap is about wiring them into one tree and then pushing the interp from *toy* to *real*.

---

## Phase 0 — Consolidate (the monorepo migration) 🟡

Mechanical: no new capability, just one coherent tree with green tests. Fresh history;
the three old repos (`cloze`, `clozn-archive`, `legible-interior`) stay as archives.

- **0.1** Scaffold the tree (`engine/ kernels/ inspector/ research/ protocol/ docs/`), a
  top-level README, and a unified `.gitignore` (Python + C++ build trees + models + the
  local-only files).
- **0.2** **Engine** → `engine/core` (was `cloze/core`). Wire `llama.cpp` as a fresh
  submodule at the pinned commit + re-apply the CLOZE patches from `PATCHES.md` (link,
  don't fork). Fix CMake paths (kernels moved). Gate: backend-free `ctest` 8/8 + the GPU
  build green.
- **0.3** **Lab** → `engine/lab` (was `cloze/lab`). Fix `pyproject`/paths. Gate: `pytest`
  (the 237 goldens) green.
- **0.4** **Kernels** → `kernels/` (was `cloze/kernels`). Gate: the CUDA parity test green.
- **0.5** **Inspector** → `inspector/` (was `clozn-archive`'s package). Rewire its diffusion
  `StateSource` onto `engine/lab`. Gate: `pytest -m "not model"` green.
- **0.6** **Research** → `research/` (was `legible-interior`).
- **0.7** **Docs** → `docs/`: merge `DESIGN.md` + `TECHNICAL.md` + clozn's `DESIGN.md` into
  one architecture; a single product README.
- **0.8** CI for the monorepo (CPU lane: ctest + pytest + parity). Mark the old repos
  archived.

## Phase 1 — The protocol (the keystone) ⬜

One state-stream the engine emits and the inspector consumes — collapses the two spines.

- **1.1** Spec `protocol/`: `StateStep / Intervention / StateSource / Spine` — one schema +
  JSON wire. Reconcile the engine's §5.1 events ↔ the inspector's `StateStep`.
- **1.2** Engine **emits** the protocol over SSE for every substrate (diffusion, AR),
  substrate-tagged.
- **1.3** Inspector: an `EngineStateSource` that **consumes** the engine stream — the
  inspector ops run over the wire, not just in-process.
- **1.4** Round-trip gate: engine → protocol → inspector `snapshot/restore/steer` on a real
  model, validated end-to-end.

## Phase 2 — One inspector, every substrate ⬜

- **2.1** **AR** as a first-class inspector substrate (drive `engine.ar_forward` via the
  protocol) — clozn's long-planned "Phase 4 AR residual-stream taps," now that the engine
  side exists.
- **2.2** **RWKV** recurrent-state through the engine (llama.cpp converts RWKV-7) — or keep
  the `transformers` source behind the same spine; pick the one that scales.
- **2.3** **Diffusion** through the engine (inspector drives the C++ server, not just the
  Python lab).
- **2.4** One dashboard across all three substrates.

## Phase 3 — Interp at scale (the heavy engine/kernel work) ⬜ — *the unbuilt meat*

Push SAEs/transcoders from toy (RWKV-169m, collapses) to real local models.

- **3.1** **Activation harvesting at scale**: batched, multi-layer tap in the engine, kept
  on-device, streamed out. (The foundation everything below stands on.)
- **3.2** **SAE inference** in the engine, CPU reference first: load a dictionary → encoder
  → top-k → decoder over harvested activations.
- **3.3** **Interp top-k kernel**: extend/repoint the confidence-select kernel for SAE
  sparsify (sparse top-k over the feature dim). GPU, validated vs the CPU reference.
- **3.4** **Pretrained dictionaries** (Gemma Scope / Llama Scope SAEs) as optional plug-ins
  — never the foundation, a per-model add-on.
- **3.5** **Transcoder inference** (sparse MLP stand-in) hooked at a component (the current
  SOTA substrate).
- **3.6** **Scale honesty**: does discovery hold where the toy collapsed? Real metrics +
  dose-response, the caveats louder than the wins.
- **3.7** **Feature steering at scale**: steer a discovered/loaded feature; causal
  dose-response curve.

## Phase 4 — The frontier: legible, editable in-model memory (fast-weights) ⬜ — *research*

The "legible local editable in-model memory" gap — capability that stays legible by
construction (the `research/` thesis applied).

- **4.1** **The 1-slot experiment**: a minimal surprise-gated fast-weight slot in the
  engine (smallest thing that could kill the idea).
- **4.2** **Test-time weight-delta** integration into the forward pass (engine-deep; the
  genuinely kernel-novel part).
- **4.3** **Legibility**: can you *read and edit* what the memory stored? Sparse, nameable
  by construction, or inscrutable? (the open crux from the research handoff).
- **4.4** **Persistence** across sessions — the fast-weight memory rehydrates cold (ties to
  the inspector's `store`/`memory`).
- **4.5** **Honest eval**: associative recall / needle-in-haystack with the persistent
  sparse state — can a compressed memory be both sparse *and* high-capacity?

## Phase 5 — Daily driver / product ⬜ (parallelizable)

- **5.1** Unified served viz across substrates + a live feature atlas.
- **5.2** Packaging / distribution — the "Ollama for white-box."
- **5.3** Automated honesty harnesses: every shipped claim carries its causal test +
  dose-response.

---

## How this gets executed

Top-down, **tests gate every task**. I work through a phase, report at its boundary
(not per task, not asking permission to proceed), and only promote the next phase's tasks
into the live tracker when the current one is green. The frontier phases (3–4) are where a
*result*, not the plan, decides the next rung — expect some rungs to get cut, loudly and on
purpose. The four [carried-over invariants](ARCHITECTURE.md#carried-over-invariants-non-negotiable)
(honesty-first, the seam, tests-as-oracle, substrate-agnostic) hold throughout.
