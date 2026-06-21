# Architecture — the unified runtime

> **Working name: `glassbox` (provisional — rename freely).** The product is *a local
> runtime where you can **view and steer a model's memory***. The old names map on as
> **`cloze` = the engine inside**, **`clozn` = this product**, **`legible-interior` = the
> research thread.** The near-homonym `cloze`/`clozn` is half the historical confusion;
> a distinct umbrella name (this one, or another) is the cleanest fix.

This repo unifies three codebases that grew up separately but are one system:
`cloze` (a C++/ggml runtime, born as "Ollama for diffusion LMs"), `clozn` (a Python
white-box inspector), and `legible-interior` (interpretability research). The diffusion
runtime was the **birth story, not the identity** — it's one substrate the engine runs.

## The one product

**View and steer a model's evolving internal state — its memory — on the models you run
yourself.** Watch it think; read named concepts and token candidates off each position;
snapshot / restore / edit / persist the state; steer concepts into the residual stream.
Ollama's structural opposite: not a black box you prompt, a glass box you inspect.

## The layers (one repo, strict downward dependencies)

```
research/   →   inspector/   →   protocol/   →   engine/ + lab/ + kernels/
 (science)      (the product)    (the seam)       (the runtime)
```

- **`research/`** — the legibility science (was `legible-interior`). "Compression under
  constraint"; the interpretability-tax experiments. Decides *what* interp methods are
  worth building, at toy scale, fast. Feeds methods upward; depends on nothing below.
- **`inspector/`** — the white-box product (was `clozn`): the state-stream spine, the ops
  (snapshot/restore/diff/edit/probe/steer), the **memory** (persist/associate/atlas), the
  viz. The thing you actually use. Owns *all* product surface. Drives the engine through
  the protocol; delegates hot paths down (it never does heavy compute itself).
- **`protocol/`** — **the keystone.** One state-stream vocabulary shared by engine and
  inspector (see below). This is what makes it *one* system instead of two inspectors.
- **`engine/`** — the runtime (was `cloze/core`): C++/ggml, runs real models, **emits the
  state-stream, applies steers, and hosts the interp primitives that must scale.** The
  performant floor. Substrate-agnostic: diffusion · autoregressive · (later) recurrent.
- **`lab/`** — the Python reference scheduler + model adapters + goldens (was `cloze/lab`).
  The correctness oracle the engine is validated against; the CPU/iteration path.
- **`kernels/`** — GPU kernels (was `cloze/kernels`): confidence-select today, **interp
  kernels tomorrow** (sparse top-k for SAE/transcoder inference; on-device activation
  harvesting). Optional, validated against CPU reference.

**The rule that fixes the drift:** the engine never owns product opinions (no competing
inspector); the inspector owns all view/steer/memory surface; they meet only at the
protocol. `cloze`'s polished C++ viz survives as the engine's *reference* view; the
canonical inspector is the Python one, consuming the same stream.

## The keystone — `protocol/` (the state-stream)

Today the same idea exists twice: the engine's §5.1 typed events and the inspector's
`StateStep / Intervention / StateSource / Spine`. They are the same abstraction discovered
twice. Collapse them:

- **`StateStep`** — one frame of the model's evolving state: `{t, substrate, slots, values,
  kind}`. A diffusion *board pass*, an AR *token + residual*, an RWKV *recurrent-state
  update* are all `StateStep`s differing only in `substrate`.
- **`Intervention`** — the write-back: `{target (layer/slot), vector, coef}`. Steering and
  edits flow up this channel.
- **`StateSource`** — anything producing the stream: the engine (any substrate over
  HTTP/SSE) or a lightweight Python source (RWKV-via-`transformers`).
- **Memory ops** — snapshot / restore / persist / associate operate on accumulated
  `StateStep`s. *That is "the model's memory, made legible and editable."*

One spine, every substrate. "View" = read the stream; "steer" = push an `Intervention`;
"memory" = persist and recall the state.

## Where the depth lives — the interp maturity ladder

The interp side looks thin because most of its substance is **engine-side and ahead of
us**, not in the Python you can read today. The heavy primitives belong DOWN in
`engine/` + `kernels/`; the Python inspector is orchestration + method library + UX.

| Capability | Status | Home |
|---|---|---|
| Activation tap (mid-layer), logit-lens | **real** | engine (C++ `cb_eval`) |
| Training-free diff-in-means concept probes + steering | **real** | engine + inspector |
| Confidence-select **top-k kernel**, zero-copy device-logits | **real, but aimed at diffusion *sampling*** | kernels / engine |
| SAE + transcoder feature discovery | **toy** (RWKV-169m, no kernels, *collapses at scale*) | inspector (today) → engine (to scale) |
| **Scaled** SAE/transcoder *inference* on real models | **unbuilt** | engine + **interp kernels** (sparse top-k = extend confidence-select) |
| Pretrained dictionaries (Gemma/Llama Scope) as plug-ins | **unbuilt** | inspector loads, engine runs |
| **Fast-weights / in-model editable memory** (Titans-style, surprise-gated) | **unbuilt — the frontier** | engine-deep (woven into the forward pass) + kernels |
| Circuit / attribution-graph tracing (transcoder-based) | **unbuilt** | inspector method over engine taps |

**The encouraging part:** the kernel infrastructure isn't absent, it's *adjacent and
unaimed.* SAE inference is `encoder matmul → top-k sparsify → decoder matmul`; we already
have a top-k kernel (confidence-select) and on-device activation/logits access (the §4.3
zero-copy work). Repointing and extending those toward interp is the bridge. Honest
caveat: much of "interp at scale" is GPU-batched cuBLAS in the engine, not exotic kernels
— the genuinely kernel-novel frontier is **fast-weights** (per-token weight deltas during
inference) and very-high-feature-count SAE top-k.

## Substrates & "memory"

The unifying object is the model's in-flight state; "memory" is its sharpest framing.

- **Recurrent (RWKV)** — an explicit fixed-size state vector: the cleanest "memory" object;
  the inspector's flagship today.
- **Autoregressive (Llama/Qwen/...)** — residual stream + KV cache as working memory;
  engine hooks built (`ar_forward` / causal taps).
- **Diffusion (LLaDA/Dream)** — the denoising board as a state canvas; the original engine.

Persistence (`store` / `memory`) turns a captured state into a *cross-session* memory —
the bridge from "inspect activations" to "a model with a legible, editable memory."

## Carried-over invariants (non-negotiable)

1. **Honesty-first.** Every speed number ships with its quality column; every "readable"
   claim (a probe decodes X) ships with the causal test (steering X moves behavior) and
   its dose-response. Decodability ≠ use.
2. **The model/seam boundary.** Model backends live only behind the adapter seam; the
   scheduler/inspector are pure logic against it (the C++ port stays a translation).
3. **Tests are the oracle.** Goldens pin (prompt, seed, policy) → exact picks; the same
   fixtures validate the engine and the lab.
4. **Substrate-agnostic by construction.** New model families are new `StateSource`s, not
   forks of the inspector.

## Migration — fresh-history monorepo

Decided: **execute now, fresh history.** One clean initial commit; the three old repos
kept as **archives** (not deleted) so per-file history survives there.

1. ✅ This `ARCHITECTURE.md` — lock the target. *(you are here)*
2. Scaffold the tree; bring `engine/ lab/ kernels/` in, keep `ctest` + `pytest` green.
3. Bring `inspector/` in; rewire its diffusion `StateSource` onto the engine.
4. Fold `research/` in.
5. Author `protocol/`; converge the engine's §5.1 events and the inspector's `StateStep`
   onto it.
6. Archive the old repos once the new tree is green.
