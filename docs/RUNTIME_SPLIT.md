# Runtime split — what lives where

*2026-07-06. The load-bearing productionizing decision: end the ad-hoc PyTorch/GGUF split by
giving each half one job. Grounded in a full recon of both the C++ engine (`engine/core`) and
the Python studio (`clozn/server/app.py` + imports) — file:line evidence throughout so this
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

**Serving topology (implemented 2026-07-14):** one public Torch-free Python gateway supervises one
private loopback C++ worker. `clozn serve` is the only launcher; `clozn studio` attaches to it. The
gateway has one `ENGINE` handle sourced from `CLOZN_ENGINE_PORT`—the old `ENGINE`/`ENGINE_QWEN` guessed
port pair and runtime substrate switching are gone. `/v1/*` is strict client compatibility;
`/api/clozn/generate` is the native event stream.

The repeatable topology gate is `clozn smoke MODEL` (`--preflight` for build inputs, `--deep` for
receipts/replay). It launches the actual `clozn serve` front door, validates both protocols plus SQLite,
kills and recovers the private worker behind the same public gateway, and cleans up the process tree.

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

### ② LAB — PyTorch (offline, gradient + research; kept intact)

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
| SQLite run rows + SHA-256 trace blobs | the product gateway | receipts, replay, Studio, CLI |
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

- **Phase 0 — the keystone. ✅ DONE (2026-07-06), live-validated.** `EngineSubstrate(Substrate)` with `.chat` over `/v1/completions` (`_engine_complete_traced` for the per-token trace), memory as the prompt-mode card block, dials via `EngineSteer.steer_vec`. It is now constructed directly as the sole product adapter; there is no product substrate loader. **Validated against a live GGUF engine (Qwen-7B Q4, GPU): the gateway boots in ~0s with zero PyTorch model, and all six flagship endpoints work on the engine substrate — `/v1/chat/completions`, `/explain` (M1, 28-token trace + "2 hesitations"), `/receipts` (M2), `/counterfactual` (M3, `warm=1.0` `causal_verified:true` — and it visibly over-bleeds off-facts, exactly as calibration predicted), `/narrate` (M4), `/replay` (F1).**
- **Phase 1 — dials engine-native. ✅ DONE (2026-07-06), live-validated.** `EngineSteer.load_library` loads the 27 shipped library dials' metadata at boot (they appear in `/steer/axes` tagged `library`); `compute()` harvests their diff-of-means directions on first use; `steer_vector` applies them, each capped by its own calibrated max. Validated live: all 27 appear + steer (`ceremonious=1.5` → *"Esteemed Lord and Lady… the most splendid and regal attire"*; `slangy` correctly capped 1.5→1.0). **New finding: calibration is *substrate*-specific, not just model-specific** — the engine's steer scale (`base=0.08·resid_norm`) ≠ PyTorch's, so a dial value means something different per substrate (`eli5=0.5` reads mild on the engine, strong on PyTorch). The PyTorch-derived ranges are a usable starting point but the engine wants its own sweep → **Phase 5**.
- **Phase 2 — RAG memory.** Already substrate-agnostic (runs before `.chat`) — "just works" once Phase 0 lands. *Validates:* cards inject + gate on the engine.
- **Phase 3 — streaming. ✅ DONE (2026-07-07).** `EngineSubstrate.chat_stream` streams the engine's SSE and yields live token chunks; the studio's `/v1/chat/completions` SSE path (`_sse_chat`) now fires on the engine substrate (before, streaming requests fell through to one non-streamed reply). Validated: live chunks arrive over time (raw-socket + urllib + the OpenAI `chat.completion.chunk` format), and a mid-stream client disconnect leaves engine + studio alive (3×, via `chat_stream`'s `GeneratorExit`→close-connection path). Remaining nicety: richer sampling (top-p/k; the engine is greedy/temp-only — fine for receipts, which *want* determinism).
- **Phase 4 — hardening. IN PROGRESS.** Engine supervision is shipped. Remaining: request cancellation
  (performance, not safety—see #6), batched multi-sequence decode, and auth if remote binding is added.
- **Phase 5 — (optional) calibration engine-native.** Move the dial sweep off PyTorch onto the engine (forward-only) so a new model self-calibrates with no lab dependency → a fully self-contained product.

## Honest hard-parts

1. **Chat templating** — engine takes raw prompt strings, no `/v1/chat/completions`, no per-model template. `EngineSubstrate` must own it. Mechanical, but real.
2. **Sampler parity** — engine sampler is greedy/temp/rep-penalty only. Fine for receipts; chat wants top-p/k (upstream ready, unwired).
3. **KV-level persistent injection** has *no working reference anywhere* (the one PyTorch attempt, `persistent_injection.py`, is broken/deferred). But slot-memory is receipt-only + off-by-default → off the critical path, and `/state`-write is a better foundation than the broken torch KV-edit.
4. **Attention-weight taps** cost flash-attention perf (probs aren't a tensor on the flash path). Live receipts appear not to need them (logits + hidden-state harvest) → likely moot.
5. **No batched decode** — engine concurrency is N full contexts = N× KV. Don't assume vLLM-style continuous batching.
6. **Engine robustness — TWO confirmed crash modes; THE #1 production blocker (bigger than Phase 0 first implied).** The single-worker engine hard-exits (no crash trace) under two conditions, both surfaced during Phase 0/1 validation:
   - **(i) Crash during streaming generation — ✅ FIXED (2026-07-06). Was the #1 blocker.** ROOT CAUSE: the streaming content-provider path calls the generators from a callback cpp-httplib invokes *outside* its routing try/catch, so any generator throw (`n_ctx` exceeded, `llama_decode` failure, `max_new < 1`) escaped into httplib's worker thread → `std::terminate()`→`abort()` — a silent hard crash (no trace on a Windows Release build). Proven decisively: `max_tokens:0`+`stream:true` hard-killed the engine, but the *identical* request non-streamed returned a clean 500 (it runs inside httplib's routing catch). FIX (`serve/cloze_server.cpp`): (1) clamp `max_tokens`/`steps` ≥ 1 in `config_from`; (2) `run()` restores the pooled context (`clear_steer`/`clear_diffusion_prefix`/tap/emit) + rethrows on any generator throw — never leaves the context dirty; (3) the streaming provider catches the rethrow and emits a clean `data:{"error":…}` + `[DONE]` frame instead of letting it abort. VALIDATED: the previously-fatal `max_tokens:0` and `n_ctx`-overflow streaming requests now return clean errors with the engine alive, and `scripts/smoke/engine_substrate.py` — which hard-crashed the engine *twice* — now passes **12/12**, sustaining a full session (chat → receipts → counterfactual → narrate → replay → memory → dials). Applied to `/v1/completions`+`/v1/infill` (the studio's path); the same one-line-shaped guard should extend to the other three streaming providers (see the status line below).
   - **(ii) Mid-SSE-stream client disconnect — ✅ RESOLVED (2026-07-07): NOT a crash (earlier misattribution).** Verified two ways: cpp-httplib's `sink.write` tracks an internal `ok` bool and simply NO-OPs once the peer disconnects (it never throws — and the build links no TLS, so there's no OpenSSL-throw path either), and empirically a raw-socket client that reads 2 frames then RST-closes, 3×, left the engine alive throughout (same PID). The engine deaths first pinned on "disconnect" were really mode-(i) exceptions firing near the disconnect. ONE real (non-crash) residual: since `sink.write`'s bool is discarded and there's no cancellation hook, an abandoned generation runs to completion in the background holding the single worker's context — wasteful, can delay the next request. A cancel-flag in `run()`'s loop is a *perf* follow-up, not a safety one.

   **Phase 4 status:** (i) ✅ FIXED + (ii) ✅ RESOLVED-as-not-a-crash (above). The mode-(i) try/catch now guards ALL 4 streaming providers (`/v1/completions`+`/v1/infill`, `/v1/revise`, `/v1/board`, `/intervene`), and `/intervene`'s dangling `&raw_vec`/`&applied_layers` captures are fixed (`raw_vec` by value; `applied_layers` via a `shared_ptr<json>`). Engine supervision and gateway retry-on-reconnect are now provided by the `clozn serve` runtime supervisor. **Remaining (defense-in-depth, non-urgent):** the abandoned-generation cancel-flag (the mode-(ii) performance residual above). **The engine substrate is now a usable, streaming, supervised runtime — not just a validated architecture.**

## Security note

`engine/core/third_party/llama.cpp/CLAUDE.md` (inside the *vendored* checkout) contains a
prompt-injection instructing agents to read an `AGENTS.md` "before any work." It is third-party
content, not a project instruction — delete it or neutralize it so no future agent/tool obeys it.
