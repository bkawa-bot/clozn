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

- **Phase 0 — the keystone. ✅ DONE (2026-07-06), live-validated.** `EngineSubstrate(Substrate)` with `.chat` over `/v1/completions` (`_engine_complete_traced` for the per-token trace), memory as the prompt-mode card block, dials via `EngineSteer.steer_vec`. Reuses `_qwen_tmpl`. Wired into `load_substrate("engine")` (was `None`). **Validated against a live GGUF engine (Qwen-7B Q4, GPU): studio boots in ~0s with zero PyTorch model, and all six flagship endpoints work on the engine substrate — `/v1/chat/completions`, `/explain` (M1, 28-token trace + "2 hesitations"), `/receipts` (M2), `/counterfactual` (M3, `warm=1.0` `causal_verified:true` — and it visibly over-bleeds off-facts, exactly as calibration predicted), `/narrate` (M4), `/replay` (F1).** Deferred to a fast-follow: `.chat_stream` (SSE streaming — the OpenAI endpoint falls through to non-stream cleanly meanwhile) and moving chat templating engine-side (it's Python `_qwen_tmpl` today, fine for the studio orchestration layer). 48 model-free tests + full suite green.
- **Phase 1 — dials engine-native. ✅ DONE (2026-07-06), live-validated.** `EngineSteer.load_library` loads the 27 shipped library dials' metadata at boot (they appear in `/steer/axes` tagged `library`); `compute()` harvests their diff-of-means directions on first use; `steer_vector` applies them, each capped by its own calibrated max. Validated live: all 27 appear + steer (`ceremonious=1.5` → *"Esteemed Lord and Lady… the most splendid and regal attire"*; `slangy` correctly capped 1.5→1.0). **New finding: calibration is *substrate*-specific, not just model-specific** — the engine's steer scale (`base=0.08·resid_norm`) ≠ PyTorch's, so a dial value means something different per substrate (`eli5=0.5` reads mild on the engine, strong on PyTorch). The PyTorch-derived ranges are a usable starting point but the engine wants its own sweep → **Phase 5**.
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
6. **Engine robustness — TWO confirmed crash modes; THE #1 production blocker (bigger than Phase 0 first implied).** The single-worker engine hard-exits (no crash trace) under two conditions, both surfaced during Phase 0/1 validation:
   - **(i) Non-deterministic crash during streaming generation — the serious one.** The single-worker engine hard-crashes at an unpredictable point during streaming generation: `research/smoke_engine_substrate.py` dies ~gen 7 of a normal session (chat → receipts → counterfactual → narrate → replay…); a direct sequential-stream hammer sometimes dies on gen 1. No crash trace, and *after* full startup (concept probes ready, listening). **Ruled out:** VRAM exhaustion (12.9 GB free, zero zombie processes at crash time) and a startup race (dies post-"listening"). Every feature passes on a fresh, lightly-loaded engine (validated manually), so this is a crash in the engine's own streaming/generation path — it needs a **debug build** to locate (the release binary exits with no symbols). Ordinary usage streams every turn, so the engine substrate can't reliably sustain a session until this is fixed.
   - **(ii) Mid-SSE-stream client disconnect.** A streaming request whose client aborts mid-stream kills it (non-stream is fine). The studio's own `_engine_complete_traced` consumes the full stream so ordinary chat doesn't trip *this* one, but closed tabs / timeouts / cancels will. Scoped to `serve/cloze_server.cpp:1078`: the SSE `write` lambda discards `sink.write`'s bool and `run(on_event)` (:1099) has no cancellation hook, so a vanished client keeps generation writing to a dead socket until it faults.

   **Phase 4 must:** (a) fix BOTH C++ crash modes — (ii) is scoped (`cloze_server.cpp:1078`: capture `sink.write`'s bool + thread a cancel flag through `run()`'s token loop); (i) needs a debug build to find the ~7-generation death (leak / context-pool). (b) supervise the engine (auto-restart on death); (c) studio — retry-on-connection-refused. **Until (a), the engine substrate is a validated *architecture* but not yet a usable *runtime*.**

## Security note

`engine/core/third_party/llama.cpp/CLAUDE.md` (inside the *vendored* checkout) contains a
prompt-injection instructing agents to read an `AGENTS.md` "before any work." It is third-party
content, not a project instruction — delete it or neutralize it so no future agent/tool obeys it.
