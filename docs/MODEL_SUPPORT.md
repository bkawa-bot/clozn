# Model Support & Agnosticism

**Thesis:** clozn is a GGUF runtime, so portability is **tiered, not all-or-nothing**. Most of the product
is model-agnostic today (or one bounded fix away); only the SAE concept viz is genuinely model-gated.

| Tier | Features | Portability | Per-model cost |
|---|---|---|---|
| **0 — Universal** | chat, token trace / confidence timeline, prompt-mode memory (cards), receipts / rederive / forced-scoring, run inspector | **any autoregressive GGUF** llama.cpp loads | **$0** (once the template is engine-sourced) |
| **1 — Calibrated** | tone dials / steering, linear probes | any GGUF **+ one automated sweep** | ~1 GPU sweep (forward passes, no backprop) + a curation pass |
| **2 — Bespoke** | SAE concept atlas / "brain" readouts (named sparse features) | **only models with a trained SAE** | free where a suite exists (GemmaScope); research-grade otherwise |

Tier 0 is a **capability**, not a roster. Tier 1 is a **growable roster** (add a model by running its sweep).
Tier 2 is **hero-only**.

---

## What's model-locked today, and the fix

Survey (2026-07-08): 123 `Qwen` references across 13 `clozn/*.py` files, concentrated in `clozn_server.py`
(73). The couplings that actually matter:

1. **Chat template — the one real Tier-0 blocker.** `_qwen_tmpl` (`clozn/clozn_server.py:121-132`)
   hardcodes Qwen's ChatML (`<|im_start|>…<|im_end|>`) and is called in *every* generation path
   (`:1595, :1640, :1723, :2708, :2710`). Pointing clozn at a Llama/Gemma GGUF today would feed it the
   wrong prompt format.
   **Fix:** apply the GGUF's *embedded* chat template engine-side (`llama_chat_apply_template` reads it from
   model metadata); Python sends messages, the engine templates per-model. Bounded, mechanical.

2. **Hardcoded model defaults.** `"Qwen/Qwen2.5-7B-Instruct"` as `model_id` (`:70`) and in `SelfTeach(...)`
   (`:1174`), plus scattered assumptions.
   **Fix:** a small **model registry / config** (id, tap layer, quant, sampler defaults) instead of literals.

3. **SAE / concept stack** (`sae7b.py`, `brain_readout.py`, `atlas_concepts.py`) — trained on Qwen-7B with
   Neuronpedia labels. Fully Tier-2; does not generalize without a per-model SAE.

4. **Dials** (`steering.py` + `clozn/data/dial_library_shipped.json`) — steer vectors + the 33-dial safe
   ranges are per-model ("a different model needs its own sweep" — the library says so itself, Law #6).

---

## Per-model cost & the honest nuances

- **Tier 0: $0.** Any AR GGUF llama.cpp supports. Diffusion models (Dream/LLaDA) are the exception —
  they use a separate substrate (no forced-scoring; the AR factorization doesn't apply).
- **Tier 1 (dials): one automated sweep, with two caveats.**
  1. You **don't get the same 33 dials on every model** — you get *that model's* survivors, with their own
     safe ranges (Qwen-7B: 71 candidates → 47 metric-usable → 33 human-kept). A weaker/stronger model keeps
     a different subset.
  2. The final "reads-as-its-tone" curation was **human**. To make Tier 1 push-button per model, replace it
     with an **LLM-judge** rating each dose. Automatable, but a real step.
  - Steering hooks must support the architecture (fine for standard + MoE transformers — the residual stream
    is there; verify on a second architecture before claiming it).
- **Tier 2 (SAE): the only genuinely gated tier.** Needs a trained SAE. **Free where a suite exists**
  (GemmaScope 1 → Gemma 2; GemmaScope 2 → Gemma 3). Otherwise research-grade. **Model-agnostic fallbacks**
  that need no SAE and already have `provider_type` slots in the workspace-lens seam:
  - **Activation MRI** — per-token residual norms, per-layer geometry (just reads `/harvest`).
  - **Logit lens** — project the residual through the unembedding: "what token is it leaning toward at
    layer L?" Zero training, any model.
  - **Linear probes** — cheap per-concept directions.

---

## Roster

- **Tier 0 (any AR GGUF, free):** Llama-3.x, Qwen3.x, Gemma 3/4, Mistral, Phi-4, DeepSeek, GPT-OSS, …
- **Tier 1 featured (run the sweep):** Qwen3-14B, Gemma 4-12B, **GPT-OSS 20B** — plus **Qwen2.5-7B** (legacy
  baseline; its 33-dial library is already calibrated).
- **Tier 2 full-brain hero:** **Gemma 3-12B** (GemmaScope 2 SAEs, free). Qwen2.5-7B keeps its existing
  `sae7b`. Other models get logit-lens + probes + MRI instead of the SAE atlas.

All picks fit a 16 GB card at Q4.

---

## Near-term work plan (to unlock Tier-0 agnosticism)

1. **Engine-side templating** — replace `_qwen_tmpl` with the GGUF's embedded chat template
   (`llama_chat_apply_template` on the C++ side); Python passes messages, not a pre-rendered string.
2. **Model registry** — parameterize the ~73 Qwen literals in `clozn_server.py` into a config object
   (id, tap layer, sampler defaults).
3. **Verify white-box taps** (`/harvest`, steer, `/score`) on a **second architecture** (Gemma 3 or GPT-OSS)
   — the honest check that "any GGUF" is real, not assumed.

After that: Tier 1 per model = a sweep (+ LLM-judge curation); Tier 2 = wire Gemma-3 GemmaScope SAEs.

---

## Reference: model landscape (from web research on 2026-07-08 — time-sensitive; verify before acting)

> Claude's own training cutoff is Jan 2026, so the below is from live search, not memory.

- **Latest Qwen:** Qwen3.6 (Apr 2026): 27B dense + 35B-A3B MoE. 16 GB-fit sweet spot = **Qwen3-14B** (~9 GB
  Q4) or Qwen3.5-9B; Qwen3.6-27B Q4_K_M is **16.8 GB** (edge of a 16 GB card). llama.cpp supports Qwen3.6.
- **Latest Gemma:** Gemma 4 (Apr 2026): E2B / E4B / 12B / 26B-MoE / 31B-dense. **GemmaScope 2 SAEs cover
  Gemma 3 only** (270m/1b/4b/12b/27b, PT+IT) — Gemma 4 has no SAEs yet. → the SAE hero is **Gemma 3**, even
  though Gemma 4 is newer.
- **Popularity (Jun-Jul 2026):** six families that matter — Llama, Mistral, Qwen, DeepSeek, Gemma, Phi.
  Qwen = safe all-around default. **GPT-OSS 20B** is the repeatedly-cited 16 GB champion (~42 tok/s,
  ~13.7 GB VRAM).

Sources: Qwen3.6 (unsloth/Qwen3.6-27B-GGUF; github.com/QwenLM/Qwen3.6) · Gemma 4 (blog.google) ·
Gemma Scope 2 (lesswrong "Announcing Gemma Scope 2"; deepmind.google/models/gemma/gemma-scope) ·
Best local LLM Jul 2026 (app.stationx.net) · Best 16 GB VRAM LLMs (localllm.in).
