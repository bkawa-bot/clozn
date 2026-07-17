# Runtime split — what lives where

*Decision made 2026-07-06; implementation status refreshed 2026-07-17.*

> **Status:** the split has landed. The online product is one Torch-free Python gateway supervising one
> private C++/GGUF worker; PyTorch model work lives in `clozn/lab`. The Qwen2.5 live acceptance run recorded
> `clozn smoke` **24/24** and `--deep` **26/26**. The checked-in
> [Wave 1 qualification ledger](qualification/wave1.json) also records CPU basic/deep smoke across Llama
> 3.1, Qwen 3.5, Gemma 4, and Ministral 3. The remaining runtime gaps are engine-side cooperative cancel,
> true concurrent/batched decode, and auth/TLS only if the gateway stops being loopback-only.

## The decision

**PRODUCT is forward-only. LAB owns gradients.** They meet at validated, forward-only artifacts on disk;
the product never reaches into a live PyTorch model.

The learned soft-prefix TTT “sleep tick” was removed from the product chat path and retained as a lab
experiment. Product memory is the legible prompt-card store. This makes `clozn serve` importable and usable
without Torch while preserving the research workbench under the explicit `clozn lab` command.

## Deployed topology

```text
OpenAI client / CLI / heavn Studio
              │
              ▼
public loopback Python gateway (`clozn serve`, Torch-free)
  API compatibility · run journal · memory cards · receipts · replay
              │ negotiated protocol 1.0
              ▼
private loopback C++ worker (one loaded GGUF)
  template · generate · sample · tap · score · steer · optional J-lens
```

`clozn serve` is the launcher and supervisor; `clozn studio` attaches to an existing gateway. The gateway
has one worker handle sourced from `CLOZN_ENGINE_PORT`. `/v1/*` is the compatibility surface and
`/api/clozn/generate` is the native instrumented stream.

The repeatable topology gate is `clozn smoke MODEL` (`--preflight` for prerequisites and `--deep` for
forced receipts/replay). Managed smoke starts the real gateway/worker pair, checks both protocols and
SQLite, replaces the worker behind the stable gateway, and cleans up the process tree.

## Current ownership

### Product worker — `engine/core` (online, forward-only)

| Capability | Status | Evidence |
|---|---|---|
| AR/diffusion generation and streaming | **shipped** | `engine/core/serve/server_main.cpp`; managed smoke |
| Per-model chat templating | **shipped** | `chat_template_renderer.cpp`, `/apply_template`; `tests/test_engine_apply_template.py`; Wave 1 Gemma full-Jinja pass |
| Temperature/top-k/top-p/repetition/seed sampling | **shipped** | `engine/core/src/sample.cpp`; handoff checks in `tests/test_engine_stream.py` and `tests/test_engine_substrate.py` |
| Activation read | **shipped** | `/harvest`, `/harvest/layers`; `routes_whitebox.cpp` |
| Teacher-forced scoring | **shipped, AR-only** | `/score`; deep smoke's forced receipt/replay checks |
| Activation/steering write | **shipped** | `/state`, `/intervene`, `steer_vec`; `routes_state.cpp` |
| Prompt embedding prefix | **available, lab-oriented** | `prefix_embd`; deliberately not product memory |
| J-lens apply/read | **shipped when a qualified sidecar is loaded** | `/jlens`; `tests/test_jlens_server.py`; Qwen2.5 Q4_K_M qualification in `wave1.json` |
| SAE concept readout | **optional, artifact-gated** | `--sae`; never implied by core support |
| Batched/continuous decode | **open** | one context per active generation; no vLLM-style scheduler |

### Lab — `clozn/lab` (offline gradients and research)

- Soft-prefix TTT training and other autograd experiments.
- Dial derivation/calibration and model qualification jobs.
- J-lens fitting and artifact export. Applying a qualified lens is a C++ product operation; fitting one is
  still lab work.
- SAE, diffusion, counterfactual, and other research benches.

The lab owns its handler and receives its substrate by injection. It does not swap a product-global model
or expose the product `/v1` API.

### Product gateway — `clozn/server` (online orchestration, no model weights)

- OpenAI/native HTTP envelopes, request admission, cancellation state, and worker supervision.
- Prompt-card retrieval/assembly. The optional semantic topic gate is lazy and fails open when its optional
  dependency is absent; it is not required by the product-minimal installation.
- Run journaling, content-addressed trace blobs, migrations, receipts, explain, replay, narrate, and
  counterfactual orchestration.
- The one `EngineSubstrate` adapter used by product routes.

### Artifact seam

| Artifact | Producer | Consumer | Validation boundary |
|---|---|---|---|
| model-scoped dial bundle | lab calibration | engine steering | checkpoint/substrate identity + safe ranges |
| J-lens manifest and matrices | lab fit/export | engine `/jlens` | manifest, payload hashes, dimensions, exact qualified GGUF digest |
| SAE bundle | external or lab export | engine concept readout | model/layer/dimension identity |
| prompt memory cards | product CRUD/mining | product prompt assembly | legible JSON, no hidden learned state |
| SQLite rows + SHA-256 trace blobs | product gateway | receipts, replay, Studio, CLI | schema migrations + digest verification on read |
| learned `studio_memory.pt` | lab only | lab only | not a product artifact |

## Claims and evidence

| Claim | Repeatable evidence | Recorded measurement / limit |
|---|---|---|
| Product/lab imports are physically separated | `.github/workflows/ci.yml` `product-minimal`; `tests/test_runtime_architecture.py`; `tests/test_studio.py` | CI installs no Torch/Transformers for the product gate |
| The actual supervised topology works | `tests/test_product_smoke.py`; `clozn smoke MODEL [--deep]` | Qwen2.5 live: 24/24 basic, 26/26 deep; Wave 1 contains the exact CPU model/digest results |
| Gateway and worker negotiate one contract | `protocol/fixtures/handshake.json`; `tests/test_protocol_handshake.py` (7 checks) | incompatible or missing major is refused before serving |
| Streaming failures do not masquerade as normal stops | `tests/test_engine_stream.py`; `clozn/server/sse.py` | client disconnect cancels the gateway context; worker death emits an error + `[DONE]` |
| Persistence upgrades transactionally | `tests/test_runs_migrations.py`, `tests/test_cli_migrate.py` | each migration commits or rolls back independently; trace digests are verified on read |
| Core portability is cross-family, optional artifacts are not | `docs/qualification/wave1.json` | five core-qualified families; only the exact Qwen2.5-7B Q4_K_M row is lens-qualified |

## Migration ledger

- **Phase 0 — EngineSubstrate keystone: done.** Product chat, trace, memory assembly, receipts, replay, and
  explain use the C++ worker through one product adapter.
- **Phase 1 — engine steering: done.** Direction computation/application is live. Calibration remains
  model- and substrate-scoped; a PyTorch-derived range is not automatically a C++ qualification.
- **Phase 2 — prompt-card memory: done.** Cards inject and gate before `.chat`; learned-prefix memory is
  lab-only.
- **Phase 3 — streaming: done.** OpenAI streams contain only standard chunks; native state events remain
  on the native surface. Interactive chat samples; receipts/replay remain forced-greedy.
- **Phase 4 — production hardening: mostly done.** Supervision, protocol handshake, per-request context,
  cancellable gateway queueing, honest worker-death semantics, transactional migrations, and blob GC have
  shipped. Engine-side cooperative cancel and true concurrent/batched decode remain open.
- **Phase 5 — self-contained calibration: open/optional.** The current calibration jobs live in the lab;
  moving forward-only sweeps to the worker would remove that operational dependency.

## Honest remaining boundaries

1. **Cancellation stops gateway work, not a running C++ decode.** After a client leaves, the worker may
   finish an abandoned generation before releasing its context. Engine-loop cooperative cancellation is
   still open.
2. **Shared steering/memory state limits concurrency.** The gateway exempts only audited-safe POST routes;
   true parallel generation requires de-globalizing worker state and then batching decode.
3. **No continuous batching.** Concurrency is not vLLM-style scheduling and should not be marketed as such.
4. **Remote exposure is not supported by implication.** The production decision is loopback. Remote bind
   requires an explicit auth/TLS design and tests.
5. **Core-qualified is not artifact-qualified.** Dials, J-lenses, and SAEs remain checkpoint-specific even
   though their apply paths are forward-only.

## Resolved security note

The vendored llama.cpp root previously carried upstream `CLAUDE.md`/`AGENTS.md` instruction files.
`engine/core/third_party/bootstrap_llama.py` now removes both on every bootstrap, and they are absent from
the current checkout.
