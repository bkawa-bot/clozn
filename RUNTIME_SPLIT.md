# Runtime split — what lives where

*2026-07-06. The load-bearing productionizing decision: end the ad-hoc PyTorch/GGUF split by
giving each half one job. Grounded in a full recon of both the C++ engine (`engine/core`) and
the Python studio (`research/clozn_server.py` + imports) — file:line evidence throughout so this
is a map, not a vibe.*

## The decision

**The PRODUCT is the engine — forward-only, one runtime.** **The LAB is PyTorch — the research
workbench, kept fully intact.** They meet at **forward-only artifacts on disk**; neither reaches
into the other at runtime.

The pivot that makes this clean: **the learned soft-prefix TTT memory (the "sleep tick") is cut
from the product** (it stays in the lab for experiments). It was the *only* autograd-locked thing
in the chat path — remove it and the entire product becomes forward-only, so the engine can own
all of it with zero PyTorch at runtime. Product memory becomes the **legible, editable RAG card
store** (which is also more on-thesis than an inscrutable learned prefix).

## The three homes + the seam

### ① PRODUCT RUNTIME — the engine (`engine/core`, online, forward-only)

| Capability | Engine status today | Evidence |
|---|---|---|
| Generation (AR + diffusion) | **PRESENT** | `/v1/completions`; `model_ggml.cpp:182-236,398-554`; `sample.cpp:1-71` |
| Steering / tone dials (apply) | **PRESENT** | `steer_vec`/`/intervene`; `model_ggml.cpp:142-151`; `cloze_server.cpp:1542-1699` (comment: *"the studio's engine tone dials"*) |
| Dial directions (compute) | **ADDABLE** — engine-native via harvest | diff-of-means over pole prompts = `/harvest` + mean-pool; no PyTorch needed |
| Activation tap (receipts read) | **PRESENT** | `/harvest`; `eval_cb` `model_ggml.cpp:107-140`; the `l_out-<il>` residual |
| Activation edit (write, propagates fwd) | **PRESENT** | `/state`; proven by `test_ggml_state_write.cpp:60-74` |
| Soft-prefix (apply) | **PRESENT** (product may not use it) | `prefix_embd`; `model_ggml.cpp:238-274` — *"the train-on-HF / serve-on-llama.cpp bridge"* |
| Slot-memory read/write | **ADDABLE** | it *is* `/state`'s primitive |
| LoRA (apply) | **ADDABLE-EASY** | upstream ready (`llama.h:649,682-686`), ~8 lines like `set_steer` |

One generic ggml callback (`eval_cb`) carries steering + taps; near-stock llama.cpp, one unrelated
patch. The forward-only feature set is mostly already there.

### ② LAB — PyTorch (`research/`, offline, gradient + research; KEPT INTACT)

Everything that needs autograd or is pure research. **Runs on the bench, never in the product's
chat path.** Feeds the product only via forward-only artifacts.

- **Soft-prefix TTT trainer** (`self_teach_server.consolidate`) — autograd-locked; a lab experiment now, not a product feature.
- **Dial calibration + library sweep** (`dial_autocalibrate.py`) — forward-only; a natural idle/lab job that emits the calibration artifacts. (Can later move engine-native — see Phase 5.)
- **The wild experiments, honesty harnesses, brain-readout/SAE research, counterfactual/mirror benches** — the whole research surface stays exactly as-is.

### ③ SERVER TIER — above the substrate (runtime-agnostic, zero model internals)

Torch-free by design; depends on **one method, `.chat()`**. Serves *both* product and lab.

- **Prompt-card / RAG memory** — retrieval + assembly + a small sentence-transformer gate; runs *before* the substrate (`memory_cards.py`, `topic_gate.py`, `memory_mode.py`). Already substrate-agnostic.
- **Receipts / explain / replay / narrate / counterfactual / Run Inspector** — `receipts.py`, `explain.py`, `replay.py`, `narrate.py`, `counterfactual.py`, `runlog.py`. All state "no torch, no model" in their own docstrings; route through `.chat` exclusively.

### ④ THE SEAM — forward-only artifacts on disk (already exist, already used)

| Artifact | Written by (lab) | Read by (product) |
|---|---|---|
| `dial_calibration.json` + `studio_library.json` + `studio_personality.json` | calibration sweep + curation | engine `steer_vec` (dials) |
| `studio_memory_cards.json` | the card CRUD / card-mining | RAG memory tier |
| LoRA adapter (future) | a lab LoRA trainer | engine `llama_set_adapters_lora` |
| `run_*.json` | the runtime (as it serves) | the receipts stack |
| ~~`studio_memory.pt` (learned prefix)~~ | ~~the sleep tick~~ | **lab-only now — not a product artifact** |

## The keystone

Today the Python `Substrate` abstraction and the engine's HTTP surface are **two disconnected
paths**: `/engine/*` handlers call the engine but *bypass* the `Substrate` class, and the entire
receipts/memory/dial stack requires `.chat`, which only `QwenSubstrate` has. The unlock is **one
class**:

```
EngineSubstrate(Substrate):  .chat() / .chat_stream()  →  engine /v1/completions  (+ chat templating)
```

Write that, and the whole Server tier (③) runs on the engine — because it all routes through
`.chat`. This is the migration, not a feature-by-feature port.

## Migration sequence

- **Phase 0 — the keystone.** `EngineSubstrate(Substrate)` with `.chat`/`.chat_stream` over `/v1/completions`, owning per-model chat templating (the engine takes raw strings — no template today). *Validates:* receipts/explain/replay run against the engine substrate.
- **Phase 1 — dials engine-native.** Compute the 33-dial directions via `/harvest` over the pole prompts (forward-only, native activation space), apply via `steer_vec`, capped by the existing calibration artifact. *Validates:* the dial library works on the engine.
- **Phase 2 — RAG memory.** Already substrate-agnostic (runs before `.chat`) — "just works" once Phase 0 lands. *Validates:* cards inject + gate on the engine.
- **Phase 3 — sampler + streaming parity.** Richer sampling (top-p/k; upstream has it, unwired) for chat; SSE through `.chat_stream`. Keep greedy for receipts (they *want* determinism).
- **Phase 4 — hardening.** Request cancellation, batched multi-sequence decode, auth (if ever remote). Runtime-grade robustness.
- **Phase 5 — (optional) calibration engine-native.** Move the dial sweep off PyTorch onto the engine (forward-only) so a new model self-calibrates with no lab dependency → a fully self-contained product.

## Honest hard-parts

1. **Chat templating** — engine takes raw prompt strings, no `/v1/chat/completions`, no per-model template. `EngineSubstrate` must own it. Mechanical, but real.
2. **Sampler parity** — engine sampler is greedy/temp/rep-penalty only. Fine for receipts; chat wants top-p/k (upstream ready, unwired).
3. **KV-level persistent injection** has *no working reference anywhere* (the one PyTorch attempt, `persistent_injection.py`, is broken/deferred). But slot-memory is receipt-only + off-by-default → off the critical path, and `/state`-write is a better foundation than the broken torch KV-edit.
4. **Attention-weight taps** cost flash-attention perf (probs aren't a tensor on the flash path). Live receipts appear not to need them (logits + hidden-state harvest) → likely moot.
5. **No batched decode** — engine concurrency is N full contexts = N× KV. Don't assume vLLM-style continuous batching.

## Security note

`engine/core/third_party/llama.cpp/CLAUDE.md` (inside the *vendored* checkout) contains a
prompt-injection instructing agents to read an `AGENTS.md` "before any work." It is third-party
content, not a project instruction — delete it or neutralize it so no future agent/tool obeys it.
