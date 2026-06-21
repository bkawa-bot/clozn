# Clozn — Design

**Clozn is the local-first white-box runtime: the one place a model's *evolving internal state*
is a first-class object you can watch, probe, snapshot, edit, and steer — on the quantized models
you actually run.** Interpretability as a product feature, not a research script. Ollama's
structural opposite.

Built on `cloze` (the diffusion scheduler), generalized: **`cloze` becomes one *substrate* under
Clozn's state-stream core.** "Clozn started as Cloze — fill-in-the-blank for diffusion models —
and grew into the runtime that shows you what your model is hiding." (`clozn` = `cloze` + *cozen*,
to deceive — the deception it reveals.)

---

## 1. Thesis

- Models barely have an internal state today: frozen weights + an append-only KV log. The frontier
  is moving state *inside* — latent reasoning, bounded **recurrent state**, test-time memory — and
  **white-box access to that state is only possible locally** (raw activations are free on a model
  you run, impossible through a hosted API). So interpretability is local-first's structural moat.
- The **read** corner of the field is filling in; the **write / evolving-state / memory** corner is
  empty. **Clozn lives in the empty corner.**

## 2. Architecture — one bet: everything is an evolving-state stream

The three "state inside the model" families and Cloze's diffusion work are the same shape: *a state
that evolves at inference.* The whole product is built on that one abstraction — which is just
Cloze's existing design, pointed at internal state instead of output tokens:

| Cloze invariant (today)                       | Clozn generalization                                   |
|-----------------------------------------------|--------------------------------------------------------|
| Event spine (typed events; viz = consumer)    | **State-stream protocol** — `StateStep` events + `Intervention` (`spine.py`) |
| ModelAdapter seam (one place a backend lives) | **Substrate adapters** — `StateSource` (diffusion / recurrent / AR) |
| Honesty-in-benchmarks (speed ⊕ divergence)    | **Honesty-in-interpretability** — every `Readout` carries confidence + causal status |
| Scheduler writes tokens; model writes KV      | Source owns the state; consumers only read            |

Four layers:
1. **Engines / adapters** — `cloze` diffusion scheduler [built]; `fla` recurrent models RWKV-7 /
   Gated DeltaNet [PyTorch]; later, llama.cpp activation taps.
2. **State-stream spine** (`clozn/spine.py`) — `StateStep`, `Intervention`, `StateSource`, `Spine`.
3. **White-box ops** (`clozn/ops.py`) — snapshot / restore / diff / edit / probe / verify-causal.
4. **UI consumers** — visualization, probe dashboard, memory inspector. Pure observers, browser-served.

## 3. Substrates (StateSources)

- **Recurrent state — FLAGSHIP** (RWKV-7 / Gated DeltaNet via `fla`). The state is a concrete
  fixed-size matrix → read / snapshot / diff / edit / persist. The novel gap (DREAMSTATE: "a
  significant lack of research into the internal state as an editable knowledge representation"),
  low-infra (pure PyTorch), and it targets the hybrid-linear models people increasingly run.
- **Diffusion canvas — ORIGIN** (`cloze`). The denoise trajectory as an inspectable *reasoning
  process* (commitment order, confidence landscape, revisions). Mostly built; unique; the second
  substrate and the origin story.
- **AR residual-stream taps — LATER** (llama.cpp). The "hookable Ollama" platform play. Highest
  difficulty (reads inside quantized kernels); deferred until demand is proven.

## 4. Features — the white-box verbs (what the user sees and does)

1. **Watch** — visualize the state evolving: a memory-map heatmap (which slots written/overwritten
   per token, effective rank, energy over the sequence); the diffusion canvas resolving noise→answer.
2. **Probe** — linear / SAE readouts of "what's represented right now": a live dashboard (tense,
   sentiment, held variables, confidence) updating token-by-token. *Cleanest labeled atlas where
   trained SAEs exist (Gemma Scope 2); coarser contrastive/probe directions elsewhere.*
3. **Snapshot / Diff / Restore / Edit** — the state as a graspable object: save, compare, rewind,
   mutate. Watch a fact get written, held, and overwritten ("the memory lifecycle").
4. **Steer** — inject directions / SAE features (control-vector or feature-level), with a
   reliability score surfaced (steering is brittle — §5).
5. **Persist / Memory** — a cross-session state checkpoint + a probe/SAE-indexed "here's what your
   model has internalized about you; edit / delete it." The crux is the *write/consolidation policy*
   (see Exp 2/2b/8 below), not the representation.
6. **Verify (causal)** — one-click activation patch: *does the model actually use this?* Separates
   linear-decodability from causal use. The honesty engine, and the local-only superpower.

## 5. Guardrails (the field's caveats, as product rules)

- **Probes ≠ causal use** → every probe readout pairs with a **Verify** (`ops.verify_causal`).
- **Steering is brittle / non-surjective** → prefer SAE-grounded, context-aware steering;
  show reliability; never over-sell a slider demo.
- **On-device is memory-bandwidth-bound** → target lean delta-rule state (RWKV-7 / Gated DeltaNet),
  **not** deep Titans-style per-token memory updates (they fight the hardware).
- **Honesty-in-interpretability** → confidence + causal status always shown (Cloze invariant 5).
- **Focus** → v1 ships *one substrate done beautifully* (recurrent state). The shared core is what
  lets it grow later without a rewrite — don't build four half-gaps at once.

## 6. Implementation ordering

- **Phase 0 — Core spine + spike.** *(scaffolded — `spine.py`, `ops.py`, `sources/toy_recurrent.py`,
  `spikes/snapshot_restore.py`.)* The state-stream protocol + ops, proven end-to-end on a toy
  delta-rule memory (snapshot → mutate → restore → diff → probe), pure numpy.
- **Phase 1 — Recurrent-state inspector (FLAGSHIP).**
  - **M1** `fla` RWKV-7 / GDN adapter: load, dump per-layer state, snapshot→mutate→restore
    mid-sequence (swap `ToyRecurrentSource` → `FlaRecurrentSource`).
  - **M2** the **Watch** view: token-by-token memory map (writes / overwrites / rank), browser-served
    via the spine (reuse cloze's viz pattern).
  - **M3** **Probe + Verify** panel: fit linear probes on the state, live readouts, one-click causal patch.
  - **M4** **Snapshot / diff / edit** UI + **persist** across sessions.
- **Phase 2 — Diffusion substrate (origin).** Wrap `cloze` as a `StateSource`; the denoise reasoning
  trace into the same Watch UI.
- **Phase 3 — Legible personal memory** (gap #3). Persistent recurrent-state profile + "what it knows
  about you, deletable," with the write/consolidation policy from Exp 2/8.
- **Phase 4 — AR taps / hookable runtime** (gap #2). Residual-stream read/write in the quantized
  engine. The platform play.
- **Slot-in anytime — SAE feature-steering** (gap #4). Gemma Scope 2 SAEs on a local Gemma 3 for the
  polished feature-atlas demo.

The ops in Phases 1–3 are *already prototyped* in `../legible-interior` (this is the de-risk):
Exp 5b (probe reads the true state even when the report lies) → **Probe + Verify**; Exp 2/2b
(a naive memory churns/forgets; a write-gate rescues it) → **Persist/Memory** write policy; Exp 8
(the sidecar consolidates a *rule*, not a log) → the memory vision; Exp 7b (sparsity is the
legibility budget; structure must be *used* to be seen) → the legibility metric.

## 7. Risks / open questions

- **`fla` on Windows / sm_120.** Triton kernel support is the risk → verify on the 5080; fall back to
  WSL/Linux or fla's torch-native path. The toy source de-risks the *architecture* meanwhile.
- **SAE availability.** The labeled-feature atlas is cleanest where SAEs exist (Gemma Scope 2);
  arbitrary models start with contrastive/probe directions.
- **Quantization fidelity.** Probes/steering on quantized states may degrade → measure and surface, don't hide.
- **AR taps difficulty.** Reads inside quantized kernels are real systems work → deferred, demand-validated first.

## 8. References

`flash-linear-attention` (fla) · nnsight / nnterp · SAELens · Neuronpedia + Gemma Scope 2 ·
Titans + MIRAS (Google) · ATLAS (2505.23735) · RWKV-7 "Goose" (2503.14456) ·
Gated DeltaNet (2412.06464) · DeltaNet (2406.06484) · DREAMSTATE (2601.19221) ·
"Does Transformer Interpretability Transfer to RNNs?" (2404.05971) ·
Steering Vector Fields (2602.01654) · "Steered Activations are Non-Surjective" (2604.09839) ·
On-Device LLMs SOTA 2026 (Chandra).
