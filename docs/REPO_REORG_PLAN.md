# Clozn Product Repo Reorg Plan

Planning artifact only. No files were moved for this inventory.

## Scope And Evidence

This plan is based on:

- `git ls-files` for tracked source inventory.
- Current working tree state, including untracked/ignored files that current code imports or references.
- Entry points in `clozn_cli.py`, `research/clozn_server.py`, `engine/core/CMakeLists.txt`, `engine/lab/pyproject.toml`, and `inspector/demo/*.html`.
- Python AST import edges for `research/`, `inspector/clozn/`, and `engine/lab/cloze_lab/`.
- Direct path assumptions found with `rg` for `research/`, `inspector/demo`, `engine/lab`, `kernels`, and `protocol`.

Important current state:

- The active checkout inspected was `C:/Users/brigi/src/clozn`; the originally supplied Codex workspace directory was empty.
- Existing local modifications were present before this plan was written, including `README.md`, `clozn_cli.py`, `research/clozn_server.py`, and multiple tests. This plan does not overwrite or revert them.
- `research/receipt_bundle.py`, `research/tests/test_cli_trace.py`, `research/tests/test_run_lineage_server.py`, and `research/tests/test_runlog_lineage.py` are untracked but imported/tested by the current server or test suite, so they are included.
- `engine/lab/cloze_lab/models/*.py` exists locally and is imported by tracked `engine/lab` code, but it is hidden by `.gitignore` pattern `/engine/**/models/`. That is a reorg blocker to fix before moving `engine/lab`.

## Target Product Structure

Concrete target layout:

```text
studio/
  app.html
  classic/studio.html
  pages/
  assets/
  lab-windows/
cli/
  clozn_cli.py
  wrappers/
server/
  app.py
  routes/
  runlog/
  replay/
  receipts/
  memory/
  readouts/
  behavior/
  profiles/
  substrates/
  data/
engine/
  core/
  client/
  kernels/
  lab/                 # keep temporarily while the Dream/lab substrate still imports it
protocol/
  README.md
  SPEC.md
  schemas/             # future typed contracts, no extraction yet
docs/
  product docs only
examples/
  traces/
  prompts/
  fixtures/
tests/
  cli/
  server/
  studio/
  engine/
  protocol/
  fixtures/
scripts/
  dev/
  migration/
  calibration/
clozn-research/        # separate future repo, not a directory in product repo
```

Design rule for the migration: code that is directly imported by `clozn_cli.py`, `research/clozn_server.py`, Studio page modules, or CMake engine targets stays in the product repo until replacement shims and tests exist. Research spikes and findings move to `clozn-research/` only after server and test imports no longer depend on top-level `research/` imports.

## Product Entry Points

| Surface | Current entry point | Current purpose | Depends on | Proposed destination | Risk |
|---|---|---|---|---|---|
| CLI wrapper | `clozn` | POSIX launcher for `clozn_cli.py` | root-relative `clozn_cli.py` | `cli/wrappers/clozn` plus root compatibility shim | Medium |
| CLI wrapper | `clozn.cmd` | Windows launcher for `clozn_cli.py` | root-relative `clozn_cli.py` | `cli/wrappers/clozn.cmd` plus root compatibility shim | Medium |
| CLI product | `clozn_cli.py` | User terminal surface: `run`, `serve`, `models`, `pull`, `studio`, `ps`, `stop`, `trace`, `branch`, `explain`, `preferences` | `engine/core`, `research/runlog.py`, `research/clozn_server.py` | `cli/clozn_cli.py` | High |
| Studio startup | `clozn_cli.py:cmd_studio` | Launches Studio backend with `python research/clozn_server.py --port ...` | hardcoded `research/clozn_server.py`, cwd repo root | `cli/clozn_cli.py` invoking `server/app.py` | High |
| Python server | `research/clozn_server.py` | Main backend/API/orchestration server, static Studio server, runlog/replay/receipts/memory/readouts | top-level imports from `research/`, `engine/client`, `engine/lab`, `inspector/demo` | `server/app.py` plus modules | Very high |
| Server executable | `research/clozn_server.py:main()` | `ThreadingHTTPServer((host, port), make_handler())`, args `--port`, `--host`, `--substrate qwen|dream|engine` | `load_substrate`, `DEMO=../inspector/demo` | `server/app.py` | Very high |
| Studio shell | `inspector/demo/app.html` | Main hash-routed Studio app shell | `clozn.css`, `pages/*.js`, backend `/health`, `/runs`, `/memory/*`, `/profiles/*` | `studio/app.html` | High |
| Classic Studio | `inspector/demo/studio.html` | Older single-page chat/memory/dials UI | `clozn.css`, `/v1/chat/completions`, `/memory/*`, `/steer/*` | `studio/classic/studio.html` | Medium |
| Studio pages | `inspector/demo/pages/*.js` | Agent/Runs/Run/Memory/Behavior/Lab/Settings routes | same-origin backend API, relative module paths | `studio/pages/*.js` | High |
| Local UI index | `index.html` | Links all current windows and startup commands | hardcoded `research/*`, `inspector/demo/*`, `inspector/runs/*` | `studio/local-index.html` or `examples/index.html` | Medium |
| Native engine server | `engine/core/serve/cloze_server.cpp` | `cloze-server <model.gguf>` HTTP server; `/health`, `/v1/completions`, `/v1/infill`, white-box endpoints | `engine/core`, vendored llama.cpp, `kernels` through CMake | `engine/core/serve/` | Medium |
| Native engine CLI | `engine/core/tools/cloze_cli.cpp` | Native `cloze` CLI for GGUF denoise/infill | `engine/core/src`, `engine/core/include` | `engine/core/tools/` | Low |
| Native engine tools | `engine/core/tools/cloze_bench.cpp`, `cloze_arbench.cpp`, `cloze_probe_sweep.cpp`, `cloze_ar.cpp` | Bench/probe/native diagnostic tools | CMake gated by `CLOZE_BUILD_GGML` | `engine/core/tools/` or `scripts/engine/` after packaging decision | Medium |
| Python lab CLI | `engine/lab/pyproject.toml` script `cloze = cloze_lab.cli:main` | Python reference runtime CLI: `run`, `infill`, `bench`, `tui` | `cloze_lab`, local ignored `cloze_lab/models/*.py` | Keep at `engine/lab` for now | High |
| Protocol docs | `protocol/README.md`, `protocol/SPEC.md` | State-stream contract consumed by engine/inspector/server docs | docs and engine comments | `protocol/` | Low |

## Current Inventory Table

| Current path | Current purpose | Product-used? | Imported by / depends on | Proposed destination | Classification | Reason | Migration risk |
|---|---|---:|---|---|---|---|---|
| `.github/workflows/ci.yml` | CI workflow | yes | CI only | `.github/workflows/ci.yml` | KEEP_PRODUCT | Repo infrastructure, not part of product layout request | Low |
| `.gitignore` | Ignore rules | yes | Git | `.gitignore` | KEEP_PRODUCT | Needs targeted fix for `/engine/**/models/` before lab migration | Medium |
| `LICENSE` | License | yes | packaging/legal | `LICENSE` | KEEP_PRODUCT | Root license should remain | Low |
| `README.md` | Public product README | yes | users, docs links | `README.md` | KEEP_PRODUCT | Do not rewrite or move in this planning task | Medium |
| `ARCHITECTURE.md` | Product architecture doc | yes | linked from docs/README | `docs/ARCHITECTURE.md` | MOVE_TO_DOCS | Product docs belong in docs | Low |
| `ROADMAP.md` | Product roadmap doc | yes | linked from docs/README | `docs/ROADMAP.md` | MOVE_TO_DOCS | Product docs belong in docs | Low |
| `RUNTIME_SPLIT.md` | Runtime split audit/plan | yes | references server/engine paths | `docs/RUNTIME_SPLIT.md` | MOVE_TO_DOCS | Product planning doc | Low |
| `STUDIO.md` | Studio startup notes | yes | references `research/clozn_server.py` | `docs/STUDIO.md` | MOVE_TO_DOCS | Product operator doc; update links during move | Low |
| `docs/*` | Current product docs | yes | docs links | `docs/` | KEEP_PRODUCT | Already in target bucket | Low |
| `docs/REPO_REORG_PLAN.md` | This plan | yes | planning only | `docs/REPO_REORG_PLAN.md` | KEEP_PRODUCT | Acceptance output | Low |
| `protocol/*` | Shared state-stream contract docs | yes | engine comments, inspector docs | `protocol/` | KEEP_PRODUCT | Already matches target bucket | Low |
| `clozn_cli.py` | Main CLI implementation | yes | root wrappers; tests under `research/tests`; imports `research/runlog.py` | `cli/clozn_cli.py` | MOVE_TO_CLI | Terminal product surface | High |
| `clozn` | POSIX CLI wrapper | yes | invokes root `clozn_cli.py` | `cli/wrappers/clozn` plus root shim | MOVE_TO_CLI | Terminal product surface | Medium |
| `clozn.cmd` | Windows CLI wrapper | yes | invokes root `clozn_cli.py` | `cli/wrappers/clozn.cmd` plus root shim | MOVE_TO_CLI | Terminal product surface | Medium |
| `index.html` | Local launcher/index for all windows | yes/unknown | hardcoded old paths | `studio/local-index.html` or `examples/index.html` | MOVE_TO_STUDIO | UI launcher, but old demo links need review | Medium |
| `engine/core/include`, `engine/core/src`, `engine/core/serve`, `engine/core/CMakeLists.txt` | Native runtime, API server, scheduler, ggml adapter, event spine | yes | CMake targets; CLI/server use `cloze-server` binary | `engine/core/` | KEEP_PRODUCT | Already in target engine bucket | Medium |
| `engine/core/tools/*.cpp` | Native CLI/bench/probe tools | yes/unknown | CMake gated executables | `engine/core/tools/` or later `scripts/engine/` | KEEP_PRODUCT | Some are product tools, some dev tools; defer split | Medium |
| `engine/core/tools/*.py` | SAE export/dump tools | unknown | manual/dev tooling | `scripts/engine/` | MOVE_TO_SCRIPTS | Dev data preparation scripts | Low |
| `engine/core/tests/*` | Native engine tests | yes | CMake/CTest | `tests/engine/core/` or keep under `engine/core/tests` | MOVE_TO_TESTS | Product tests; moving needs CMake update | Medium |
| `engine/core/build_*.bat` | Build helpers | yes | user/dev commands | `scripts/dev/engine/` or keep wrappers | MOVE_TO_SCRIPTS | Dev scripts | Medium |
| `engine/core/third_party/PATCHES.md`, `patches/*` | llama.cpp patch metadata | yes | CMake/vendor setup | `engine/core/third_party/` | KEEP_PRODUCT | Engine build contract | Low |
| `engine/client/*` | Python client SDK/probe for C++ server | yes | `research/clozn_server.py` inserts `engine/client`; inspector tests | `engine/client/` or `server/engine_client/` | KEEP_PRODUCT | Boundary SDK; do not move until server imports are packaged | Medium |
| `engine/lab/cloze_lab/*` | Python reference scheduler/runtime | yes | `research/clozn_server.py`, `research/denoise_server.py`, lab tests | Keep `engine/lab/` temporarily | KEEP_PRODUCT | Still a product dependency for Dream/lab substrate | High |
| `engine/lab/cloze_lab/models/*.py` | Local model adapters used by lab | yes but ignored | imported by `cloze_lab.cli`, `generate`, tests | `engine/lab/cloze_lab/models/` after ignore fix | KEEP_PRODUCT | Hidden by `.gitignore`; must be made intentional before reorg | Very high |
| `engine/lab/tests/*` | Python lab tests/goldens | yes | pytest | `tests/engine/lab/` eventually | MOVE_TO_TESTS | Product correctness oracle | Medium |
| `engine/lab/cloze_lab/bench/results/*` | Benchmark result fixtures/docs | yes/unknown | docs, bench context | `examples/benchmarks/` or keep near lab | MOVE_TO_EXAMPLES | Repro examples, not runtime | Low |
| `kernels/confidence_select/*` | CUDA confidence-select kernel and tests | yes | `engine/core/CMakeLists.txt` references `../../kernels/confidence_select` | `engine/kernels/confidence_select/` | MOVE_TO_ENGINE | Native engine kernel | High |
| `kernels/sae_topk/*` | CUDA SAE top-k kernel and tests | yes | `engine/core/CMakeLists.txt` references `../../kernels/sae_topk` | `engine/kernels/sae_topk/` | MOVE_TO_ENGINE | Native engine kernel | High |
| `inspector/demo/app.html`, `pages/*.js`, `clozn.css` | Main Studio UI shell and pages | yes | served by `research/clozn_server.py`; relative static paths | `studio/` | MOVE_TO_STUDIO | Frontend product surface | High |
| `inspector/demo/studio.html` | Classic Studio UI | yes | served by server; linked from app shell | `studio/classic/studio.html` | MOVE_TO_STUDIO | Frontend product surface | Medium |
| `inspector/demo/brain.html`, `engine.html`, `denoise.html`, `instrument.html`, `talk.html`, `memory*.html` | Legacy/lab UI windows | yes/unknown | served by server and/or linked by index | `studio/lab-windows/` or `examples/ui/` | UNCLEAR | Some are product-accessible, some are prototypes | Medium |
| `inspector/demo/*.json`, `*.jsonl`, `brain_data.js` | Static demo data/traces/assets | unknown | demo pages; no direct import for several files | `examples/traces/` or `studio/assets/` | MOVE_TO_EXAMPLES | Sample traces/fixtures if not loaded by current UI | Medium |
| `inspector/demo/*.py` | Demo generators/servers | no/unknown | write `inspector/runs/*`; old demos | `clozn-research/` or `scripts/dev/studio/` | MOVE_TO_RESEARCH_REPO | Non-product prototypes unless promoted | Medium |
| `inspector/clozn/*` | Older Python inspector package | unknown | inspector tests/spikes; not current CLI/server | `clozn-research/inspector/` unless protocol/client pieces are promoted | UNCLEAR | It overlaps with protocol concepts but is not current Studio backend | High |
| `inspector/tests/*` | Tests for old inspector package | unknown | pytest | `clozn-research/tests/inspector/` or `tests/studio/legacy/` | UNCLEAR | Depends on decision for `inspector/clozn` | Medium |
| `inspector/spikes/*` | Research spikes | no | spike scripts only | `clozn-research/inspector/spikes/` | MOVE_TO_RESEARCH_REPO | Non-product experiments | Low |
| `inspector/runs/*` | Generated local outputs | no | linked by local index; gitignored | do not migrate as source | ARCHIVE | Generated artifacts, large local files | Low |
| `inspector/pyproject.toml`, `requirements.txt`, `DESIGN.md`, `README.md` | Old inspector package metadata/docs | unknown | package install/tests | `clozn-research/inspector/` or docs if retained | UNCLEAR | Package not current product entry point | Medium |
| `research/*` | Mixed backend/product/research files | mixed | see detailed table below | split among `server/`, `docs/`, `examples/`, `tests/`, `scripts/`, `clozn-research/` | UNCLEAR | Directory name is misleading; individual classification required | High |
| `research/runs/*`, `research/dream_runs/*`, `research/model_k8.pt`, `research/np_*.json` | Ignored local generated artifacts/data | mixed | some local readouts use `np_*.json`; generated runs used by findings | do not migrate as product source by default | ARCHIVE | Large/local/generated; make explicit fixtures only if required | Medium |
| `.idea`, `.pytest_cache`, `__pycache__`, build trees | Local/editor/generated state | no | local only | none | DELETE | Generated/local artifacts | Low |
| `notes/` | Local-only context per `.gitignore` | no | local only | none or `clozn-research/private-notes` if intentionally preserved | ARCHIVE | Not product source | Low |

## Detailed `research/` Classification

Legend for `product-used?`: `yes` means imported by current CLI/server/Studio-facing tests or used as runtime data; `tests` means product test only; `no` means research/prototype by import evidence; `unknown` means product exposure is indirect or stale.

| Current path | Current purpose | Product-used? | Imported by / depends on | Proposed destination | Classification | Reason | Migration risk |
|---|---|---:|---|---|---|---|---|
| `research/clozn_server.py` | Main Python backend and API routes | yes | CLI `studio`; imports many `research/*`, `engine/client`, `engine/lab`; serves `inspector/demo` | `server/app.py` | MOVE_TO_SERVER | Actual server startup and route surface | Very high |
| `research/runlog.py` | Shared run journal, trace normalization, lineage | yes | `clozn_cli.py`, `clozn_server.py`, many tests | `server/runlog/runlog.py` | MOVE_TO_SERVER | Source of truth for persisted runs | Very high |
| `research/workspace_lens.py` | Workspace/readout trace extraction | yes | `runlog.py`, `clozn_server.py`, tests | `server/readouts/workspace_lens.py` | MOVE_TO_SERVER | Product readout persistence | High |
| `research/run_timeline.py` | RunEvent timeline assembly | yes | `/runs/<id>/timeline`, tests | `server/runlog/timeline.py` | MOVE_TO_SERVER | Product route helper | High |
| `research/confidence_spans.py` | Confidence span summaries | yes | `/runs/<id>/spans`, tests | `server/runlog/confidence_spans.py` | MOVE_TO_SERVER | Product route helper | High |
| `research/capture_mode.py` | Capture tier settings | yes | server `/capture/tier`, replay, tests | `server/runlog/capture_mode.py` | MOVE_TO_SERVER | Product setting | High |
| `research/memory_cards.py` | Memory card store/review metadata | yes | server memory routes, explain, tests | `server/memory/cards.py` | MOVE_TO_SERVER | Product memory code | Very high |
| `research/memory_mode.py` | Memory mode settings and prompt block compilation | yes | server, replay, facts, tests | `server/memory/mode.py` | MOVE_TO_SERVER | Product memory code | Very high |
| `research/topic_gate.py` | Topic relevance gate | yes | memory prompt/internalized paths, tests | `server/memory/topic_gate.py` | MOVE_TO_SERVER | Product memory gating | High |
| `research/self_teach_server.py` | Soft-prefix memory training and helpers | yes | `clozn_server.py`, memory tests | `server/memory/self_teach.py` | MOVE_TO_SERVER | Product backend code despite name | Very high |
| `research/dream_memory.py` | Dream/Qwen memory substrate helpers | yes | `clozn_server.py`, tests | `server/substrates/dream_memory.py` | MOVE_TO_SERVER | Product backend dependency | High |
| `research/facts_mode.py` | Facts memory feature flag/settings | yes | server facts routes, tests | `server/memory/facts_mode.py` | MOVE_TO_SERVER | Product feature gate | High |
| `research/slotmem_qwen.py` | Slot memory/facts substrate | yes | server facts routes, tests | `server/memory/slotmem_qwen.py` | MOVE_TO_SERVER | Product backend dependency | High |
| `research/profiles.py` | Persona/profile bundles | yes | server profiles routes, tests | `server/profiles.py` | MOVE_TO_SERVER | Product backend code | High |
| `research/feedback.py` | Feedback signal store | yes | `/feedback`, preferences, tests | `server/behavior/feedback.py` | MOVE_TO_SERVER | Product backend code | High |
| `research/preferences.py` | Learned-preference proposal store | yes | `/preferences`, CLI preferences, tests | `server/behavior/preferences.py` | MOVE_TO_SERVER | Product backend code | High |
| `research/steering.py` | Tone dial steering, axes, engine steering | yes | server dials/engine chat, scripts/tests | `server/behavior/steering.py` | MOVE_TO_SERVER | Product behavior control | Very high |
| `research/replay.py` | Replay helper | yes | `/runs/<id>/replay`, receipts/counterfactual, tests | `server/replay/replay.py` | MOVE_TO_SERVER | Product route helper | High |
| `research/timetravel.py` | Branch/time-travel run transforms and settings | yes | `/runs/<id>/branch`, tests | `server/replay/timetravel.py` | MOVE_TO_SERVER | Product route helper | High |
| `research/counterfactual.py` | Counterfactual regeneration helper | yes | `/runs/<id>/counterfactual`, tests | `server/replay/counterfactual.py` | MOVE_TO_SERVER | Product route helper | High |
| `research/receipts.py` | Receipt/prove-all helpers | yes | `/runs/<id>/receipt(s)`, tests | `server/receipts/receipts.py` | MOVE_TO_SERVER | Product receipts code | High |
| `research/receipt_bundle.py` | Export bundle builder | yes | `/runs/<id>/export`, tests | `server/receipts/bundle.py` | MOVE_TO_SERVER | Product export code; currently untracked | High |
| `research/explain.py` | Explain object assembly | yes | `/runs/<id>/explain`, CLI explain, narrate, tests | `server/receipts/explain.py` | MOVE_TO_SERVER | Product explain code | High |
| `research/narrate.py` | Receipt-constrained narration | yes | `/runs/<id>/narrate`, CLI explain `--why`, tests | `server/receipts/narrate.py` | MOVE_TO_SERVER | Product explain/narration code | High |
| `research/semantic_matcher.py` | Optional semantic support matcher | yes | narrate route/tests | `server/receipts/semantic_matcher.py` | MOVE_TO_SERVER | Product optional dependency | Medium |
| `research/brain_readout.py` | Brain/SAE readout provider | yes | `clozn_server.py`, tests; depends on `sae7b.py`, local `np_*.json` | `server/readouts/brain.py` | MOVE_TO_SERVER | Product readout code | High |
| `research/sae7b.py` | Qwen SAE load/readout helpers | yes | brain readout/server/tests | `server/readouts/sae7b.py` | MOVE_TO_SERVER | Product readout code with local data dependency | High |
| `research/atlas_concepts.py` | Concept atlas data/helpers | yes | brain readout/tests and research scripts | `server/readouts/atlas_concepts.py` | MOVE_TO_SERVER | Product readout dependency | Medium |
| `research/denoise_server.py` | Standalone Dream denoise server and imported substrate helper | yes/unknown | `clozn_server.py`, `test_studio.py`; depends on `engine/lab` | `server/substrates/denoise.py` | MOVE_TO_SERVER | Product-adjacent substrate code | High |
| `research/dial_library_shipped.json` | Curated shipped dial library | yes | server boot/deploy/calibration tests | `server/data/dial_library_shipped.json` | MOVE_TO_SERVER | Runtime product data | Medium |
| `research/dial_library_candidates.json` | Candidate dial library for sweeps | no | calibration scripts/tests | `clozn-research/data/` or `examples/fixtures/dials/` | MOVE_TO_RESEARCH_REPO | Research input, not runtime | Low |
| `research/deploy_dial_library.py` | One-time library dial registration | yes/script | tests, reads shipped dial JSON and steering | `scripts/calibration/deploy_dial_library.py` | MOVE_TO_SCRIPTS | Product maintenance script | Medium |
| `research/gen_dial_calibration.py` | Generate runtime calibration file from shipped dials | yes/script | reads shipped dial JSON | `scripts/calibration/gen_dial_calibration.py` | MOVE_TO_SCRIPTS | Product maintenance script | Medium |
| `research/dial_autocalibrate.py` | PyTorch dial calibration sweep | no/script | tests, runlog, steering | `scripts/calibration/dial_autocalibrate.py` or `clozn-research/` | MOVE_TO_SCRIPTS | Dev calibration rig, not runtime | Medium |
| `research/dial_autocalibrate_engine.py` | Engine dial calibration sweep | no/script | tests, engine client, steering, runlog | `scripts/calibration/dial_autocalibrate_engine.py` | MOVE_TO_SCRIPTS | Dev calibration rig, not runtime | Medium |
| `research/fetch_np_labels.py`, `research/fetch_np_stats.py` | Fetch local Neuronpedia label/stat data | script | writes ignored `np_*.json` | `scripts/data/` | MOVE_TO_SCRIPTS | Data preparation scripts | Low |
| `research/bench_batched_receipts.py`, `research/bench_whitebox_tax.py` | Bench scripts | no/script | engine/server HTTP | `scripts/bench/` | MOVE_TO_SCRIPTS | Dev benchmarks | Low |
| `research/smoke_engine_substrate.py` | Manual smoke for engine substrate | tests/script | engine/server HTTP | `tests/server/smoke/` | MOVE_TO_TESTS | Product smoke test | Medium |
| `research/engine_prefix_test.py`, `research/engine_dream_prefix_test.py`, `research/engine_concepts_test.py`, `research/engine_steer_spike.py` | Engine integration probes | tests/unknown | engine server/readouts | `tests/engine/integration/` if kept, else `clozn-research/` | UNCLEAR | Some are smoke tests, some are spikes | Medium |
| `research/EXPLAIN_THIS_ANSWER_SPEC.md` | Product explain/run-id bridge spec | yes | referenced in server comments | `docs/EXPLAIN_THIS_ANSWER_SPEC.md` | MOVE_TO_DOCS | Product spec doc | Low |
| `research/README.md`, `HANDOFF.md`, `FINDINGS.md`, `NEXT_STEPS.md`, `UI_SCOPE_AUDIT.md`, `WILD_*.md` | Research summaries/audits/preregs | no | docs only | `clozn-research/docs/` | MOVE_TO_RESEARCH_REPO | Research-only material, not product docs | Low |
| `research/*_findings.md`, `feature_discovery_deepdive.md`, `gating_sketch.md`, `writeup_draft_receipts.md` | Findings/drafts/sketches | no | docs only | `clozn-research/findings/` | MOVE_TO_RESEARCH_REPO | Research findings, not product docs | Low |
| `research/brain_server.py`, `brain_server_7b.py`, `memory_live_server.py`, `learns_server.py` | Standalone demo servers | unknown | local index/docs; not main CLI server | `clozn-research/demos/` unless promoted | MOVE_TO_RESEARCH_REPO | Non-product prototypes beside current server | Medium |
| `research/learns_live.html` | Standalone learns demo UI | no/unknown | `learns_server.py` | `clozn-research/demos/` | MOVE_TO_RESEARCH_REPO | Non-product prototype | Low |
| `research/denoise_capture.py`, `wire_atlas.py`, `wire_denoise.py`, `wire_memory.py`, `memory_timeline.py` | Demo capture/wiring scripts | no/unknown | write `inspector/demo/*` assets | `examples/` or `clozn-research/demos/` | MOVE_TO_EXAMPLES | Sample-generation scripts; move only with asset path decisions | Medium |
| `research/data/input.txt` | Research toy input | no | `crux.py` | `clozn-research/data/` | MOVE_TO_RESEARCH_REPO | Research fixture | Low |
| `research/crux.py`, `grok.py`, `grok_clock.py`, `interior*.py`, `intro*.py`, `sidecar*.py`, `structured_superposition.py`, `superpos_static.py`, `toy_superposition.py`, `state_cycle.py` | Early toy/research experiments | no | not current product imports | `clozn-research/spikes/early/` | MOVE_TO_RESEARCH_REPO | Research-only material | Low |
| `research/frontier_apply*.py`, `function_vector*.py`, `legibility*.py`, `feature_atlas*.py`, `feature_circuit*.py`, `concept_readout.py` | Feature/legibility research rigs | no | import each other and old SAE helpers | `clozn-research/spikes/legibility/` | MOVE_TO_RESEARCH_REPO | Research-only material | Medium |
| `research/dream_consolidation.py`, `dream_memory_spike.py` | Dream memory experiments | no | `engine/lab`, `self_teach_server` | `clozn-research/spikes/memory/` | MOVE_TO_RESEARCH_REPO | Research-only, but imports product memory helpers | Medium |
| `research/idle_selfplay.py`, `mirror_bench.py`, `receipts_as_reward.py`, `parliament.py`, `persistent_injection.py`, `phantom_kv.py`, `quine.py`, `self_audit*.py`, `memory_disorders.py`, `memory_scaling.py`, `steer_vs_prompt.py`, `vector_telepathy.py`, `voice_lora.py`, `voice_middle.py`, `profile_port_demo.py`, `validate_traits.py` | Experiments around memory/receipts/preferences/voice | no | often import product helpers | `clozn-research/spikes/` with product imports replaced by package imports or fixtures | MOVE_TO_RESEARCH_REPO | Research-only material; move after server package extraction | Medium |
| `research/kv_timetravel.py` | Research KV time-travel experiment | no | imports steering | `clozn-research/spikes/timetravel/` | MOVE_TO_RESEARCH_REPO | Distinct from product `timetravel.py` | Medium |
| `research/test_studio.py` | Legacy/current integration smoke | tests | imports `clozn_server`, `brain_readout`, `sae7b`, `steering` | `tests/server/test_studio.py` | MOVE_TO_TESTS | Product test | Medium |
| `research/tests/conftest.py`, `__init__.py` | Test package setup | tests | test suite | `tests/server/conftest.py` after split | MOVE_TO_TESTS | Product test infrastructure | Medium |
| `research/tests/test_async_retrain.py` | Memory retrain test | tests | memory/runlog/server | `tests/server/memory/` | MOVE_TO_TESTS | Product test | Medium |
| `research/tests/test_bridge_server.py` | Server bridge test | tests | memory/runlog/server | `tests/server/routes/` | MOVE_TO_TESTS | Product test | Medium |
| `research/tests/test_capture_mode.py` | Capture mode test | tests | capture/memory mode | `tests/server/runlog/` | MOVE_TO_TESTS | Product test | Medium |
| `research/tests/test_cli_color.py`, `test_cli_trace.py`, `test_explain_cli.py`, `test_narrate_cli.py`, `test_preferences_cli.py` | CLI render/trace tests | tests | `clozn_cli.py`, runlog/memory | `tests/cli/` | MOVE_TO_TESTS | CLI product tests | High |
| `research/tests/test_confidence_spans.py`, `test_confidence_spans_server.py` | Confidence span route/helper tests | tests | confidence_spans/runlog/server | `tests/server/runlog/` | MOVE_TO_TESTS | Product tests | Medium |
| `research/tests/test_counterfactual.py`, `test_counterfactual_server.py` | Counterfactual helper/route tests | tests | counterfactual/replay/receipts/server | `tests/server/replay/` | MOVE_TO_TESTS | Product tests | Medium |
| `research/tests/test_dial_autocalibrate.py`, `test_dial_autocalibrate_engine.py` | Calibration script tests | tests/script | calibration scripts/data | `tests/scripts/calibration/` | MOVE_TO_TESTS | Script tests | Medium |
| `research/tests/test_dial_calibration_server.py`, `test_dial_library_server.py`, `test_dial_suggestion.py`, `test_engine_library_dials.py`, `test_steering_headroom.py` | Dial/steering product tests | tests | steering/server/data | `tests/server/behavior/` | MOVE_TO_TESTS | Product tests | Medium |
| `research/tests/test_engine_add_custom.py`, `test_engine_layers.py`, `test_engine_stream.py`, `test_engine_substrate.py` | Engine-backed server tests | tests | server, engine client, memory/steering | `tests/server/engine/` | MOVE_TO_TESTS | Product tests | High |
| `research/tests/test_explain.py`, `test_explain_server.py`, `test_export_server.py`, `test_receipts.py`, `test_receipts_server.py`, `test_narrate.py`, `test_narrate_server.py`, `test_semantic_matcher_gated.py` | Explain/receipt/narration tests | tests | runlog/memory/receipts/server | `tests/server/receipts/` | MOVE_TO_TESTS | Product tests | Medium |
| `research/tests/test_facts_mode.py`, `test_facts_server.py`, `test_slotmem_shared.py`, `test_slotmem_store.py` | Facts/slot memory tests | tests | facts/slotmem/profiles/server | `tests/server/memory/` | MOVE_TO_TESTS | Product tests | Medium |
| `research/tests/test_feedback.py`, `test_preferences.py` | Feedback/preference store tests | tests | feedback/preferences | `tests/server/behavior/` | MOVE_TO_TESTS | Product tests | Medium |
| `research/tests/test_fair_capacity.py`, `test_hf_trace.py`, `test_memory_cards.py`, `test_memory_mode.py`, `test_memory_wiring.py`, `test_prompt_relevance.py`, `test_propose_memory.py`, `test_topic_gate.py`, `test_trace_capture.py` | Memory/runlog/readout tests | tests | memory/self-teach/topic/runlog | `tests/server/memory/` and `tests/server/runlog/` | MOVE_TO_TESTS | Product tests | Medium |
| `research/tests/test_profiles.py`, `test_profiles_server.py` | Profile store/server tests | tests | profiles/server | `tests/server/profiles/` | MOVE_TO_TESTS | Product tests | Medium |
| `research/tests/test_replay.py`, `test_run_timeline.py`, `test_run_timeline_server.py`, `test_runlog.py`, `test_runlog_lineage.py`, `test_run_lineage_server.py`, `test_timetravel.py`, `test_timetravel_determinism.py`, `test_timetravel_server.py` | Runlog/replay/lineage/timetravel tests | tests | runlog/replay/timetravel/server | `tests/server/runlog/` and `tests/server/replay/` | MOVE_TO_TESTS | Product tests | Medium |
| `research/tests/test_idle_selfplay.py`, `test_parliament.py`, `test_persistent_injection.py`, `test_prompt_vs_prefix_ab.py`, `test_quine.py`, `test_receipts_as_reward.py` | Research experiment tests | no/tests | research rigs plus product helpers | `clozn-research/tests/` | MOVE_TO_RESEARCH_REPO | Tests belong with research rigs after extraction | Medium |

### Explicit `research/` Coverage Checklist

This checklist expands the grouped rows above. It is not a move list; it is a classification coverage check for every current source/data file found under `research/`, excluding cache directories and generated `research/runs/` / `research/dream_runs/` contents.

| Bucket | Files |
|---|---|
| Actual product backend code -> `server/` | `research/atlas_concepts.py`, `research/brain_readout.py`, `research/capture_mode.py`, `research/clozn_server.py`, `research/confidence_spans.py`, `research/counterfactual.py`, `research/denoise_server.py`, `research/dream_memory.py`, `research/explain.py`, `research/facts_mode.py`, `research/feedback.py`, `research/memory_cards.py`, `research/memory_mode.py`, `research/narrate.py`, `research/preferences.py`, `research/profiles.py`, `research/receipt_bundle.py`, `research/receipts.py`, `research/replay.py`, `research/run_timeline.py`, `research/runlog.py`, `research/sae7b.py`, `research/self_teach_server.py`, `research/semantic_matcher.py`, `research/slotmem_qwen.py`, `research/steering.py`, `research/timetravel.py`, `research/topic_gate.py`, `research/workspace_lens.py` |
| Product runtime data -> `server/data/` | `research/dial_library_shipped.json` |
| Product docs -> `docs/` | `research/EXPLAIN_THIS_ANSWER_SPEC.md` |
| Dev/calibration/bench scripts -> `scripts/` | `research/bench_batched_receipts.py`, `research/bench_whitebox_tax.py`, `research/deploy_dial_library.py`, `research/dial_autocalibrate.py`, `research/dial_autocalibrate_engine.py`, `research/fetch_np_labels.py`, `research/fetch_np_stats.py`, `research/gen_dial_calibration.py` |
| Product/integration tests -> `tests/` | `research/test_studio.py`, `research/tests/__init__.py`, `research/tests/conftest.py`, `research/tests/test_async_retrain.py`, `research/tests/test_bridge_server.py`, `research/tests/test_capture_mode.py`, `research/tests/test_cli_color.py`, `research/tests/test_cli_trace.py`, `research/tests/test_confidence_spans.py`, `research/tests/test_confidence_spans_server.py`, `research/tests/test_counterfactual.py`, `research/tests/test_counterfactual_server.py`, `research/tests/test_dial_calibration_server.py`, `research/tests/test_dial_library_server.py`, `research/tests/test_dial_suggestion.py`, `research/tests/test_engine_add_custom.py`, `research/tests/test_engine_layers.py`, `research/tests/test_engine_library_dials.py`, `research/tests/test_engine_stream.py`, `research/tests/test_engine_substrate.py`, `research/tests/test_explain.py`, `research/tests/test_explain_cli.py`, `research/tests/test_explain_server.py`, `research/tests/test_export_server.py`, `research/tests/test_facts_mode.py`, `research/tests/test_facts_server.py`, `research/tests/test_fair_capacity.py`, `research/tests/test_feedback.py`, `research/tests/test_hf_trace.py`, `research/tests/test_memory_cards.py`, `research/tests/test_memory_mode.py`, `research/tests/test_memory_wiring.py`, `research/tests/test_narrate.py`, `research/tests/test_narrate_cli.py`, `research/tests/test_narrate_server.py`, `research/tests/test_preferences.py`, `research/tests/test_preferences_cli.py`, `research/tests/test_profiles.py`, `research/tests/test_profiles_server.py`, `research/tests/test_prompt_relevance.py`, `research/tests/test_propose_memory.py`, `research/tests/test_receipts.py`, `research/tests/test_receipts_server.py`, `research/tests/test_replay.py`, `research/tests/test_run_lineage_server.py`, `research/tests/test_run_timeline.py`, `research/tests/test_run_timeline_server.py`, `research/tests/test_runlog.py`, `research/tests/test_runlog_lineage.py`, `research/tests/test_semantic_matcher_gated.py`, `research/tests/test_slotmem_shared.py`, `research/tests/test_slotmem_store.py`, `research/tests/test_steering_headroom.py`, `research/tests/test_timetravel.py`, `research/tests/test_timetravel_determinism.py`, `research/tests/test_timetravel_server.py`, `research/tests/test_topic_gate.py`, `research/tests/test_trace_capture.py` |
| Script tests -> `tests/scripts/` | `research/tests/test_dial_autocalibrate.py`, `research/tests/test_dial_autocalibrate_engine.py` |
| Examples/demo generators -> `examples/` or `clozn-research/demos/` pending Studio decision | `research/denoise_capture.py`, `research/memory_timeline.py`, `research/wire_atlas.py`, `research/wire_denoise.py`, `research/wire_memory.py` |
| Research-only docs/findings -> `clozn-research/docs/` or `clozn-research/findings/` | `research/FINDINGS.md`, `research/HANDOFF.md`, `research/NEXT_STEPS.md`, `research/README.md`, `research/UI_SCOPE_AUDIT.md`, `research/WILD_EXPERIMENTS.md`, `research/WILD_WAVE1_PREREG.md`, `research/WILD_WAVE2_PREREG.md`, `research/conceptmem_findings.md`, `research/dial_calibration_engine_findings.md`, `research/dial_library_findings.md`, `research/dream_consolidation_findings.md`, `research/fastweight_findings.md`, `research/feature_circuit_clean_qwen_findings.md`, `research/feature_circuit_pilot_qwen_findings.md`, `research/feature_discovery_deepdive.md`, `research/frontier_apply_findings.md`, `research/frontier_apply_v2_findings.md`, `research/function_vector_sweep_qwen_findings.md`, `research/gating_sketch.md`, `research/idle_selfplay_findings.md`, `research/kv_timetravel_findings.md`, `research/legibility_discovered_findings.md`, `research/legibility_discovered_qwen_findings.md`, `research/legibility_natural_qwen_findings.md`, `research/legibility_v1_big_findings.md`, `research/legibility_v1_findings.md`, `research/local_efficiency_findings.md`, `research/memory_disorders_findings.md`, `research/memory_scaling_findings.md`, `research/mirror_bench_findings.md`, `research/parliament_findings.md`, `research/persistent_injection_findings.md`, `research/phantom_kv_findings.md`, `research/quine_findings.md`, `research/receipts_as_reward_findings.md`, `research/sae_at_scale_findings.md`, `research/scale_pass_7b_findings.md`, `research/self_audit_gap_findings.md`, `research/self_audit_synthesis.md`, `research/sidecar_real_findings.md`, `research/sidecar_semantic_findings.md`, `research/slotmem_qwen_findings.md`, `research/steer_vs_prompt_findings.md`, `research/telepathy_findings.md`, `research/voice_lora_findings.md`, `research/voice_middle_findings.md`, `research/writeup_draft_receipts.md` |
| Research-only scripts/prototypes -> `clozn-research/spikes/` | `research/brain_server.py`, `research/brain_server_7b.py`, `research/concept_readout.py`, `research/crux.py`, `research/dial_library_candidates.json`, `research/dream_consolidation.py`, `research/dream_memory_spike.py`, `research/feature_atlas.py`, `research/feature_atlas_7b.py`, `research/feature_atlas_emergent.py`, `research/feature_circuit_clean_qwen.py`, `research/feature_circuit_pilot_qwen.py`, `research/frontier_apply.py`, `research/frontier_apply_v2.py`, `research/function_vector_roundtrip_qwen.py`, `research/function_vector_sweep_qwen.py`, `research/grok.py`, `research/grok_clock.py`, `research/idle_selfplay.py`, `research/interior.py`, `research/interior2.py`, `research/interior_viz.py`, `research/intro.py`, `research/intro_probe.py`, `research/kv_timetravel.py`, `research/learns_live.html`, `research/learns_server.py`, `research/legibility_discovered.py`, `research/legibility_discovered_qwen.py`, `research/legibility_natural_qwen.py`, `research/legibility_v1.py`, `research/legibility_v1_big.py`, `research/memory_disorders.py`, `research/memory_live_server.py`, `research/memory_scaling.py`, `research/mirror_bench.py`, `research/parliament.py`, `research/persistent_injection.py`, `research/phantom_kv.py`, `research/profile_port_demo.py`, `research/quine.py`, `research/receipts_as_reward.py`, `research/self_audit_blackbox.py`, `research/self_audit_cure.py`, `research/self_audit_gap.py`, `research/self_audit_report.py`, `research/self_teach_extras.py`, `research/sidecar.py`, `research/sidecar_real.py`, `research/sidecar_semantic.py`, `research/state_cycle.py`, `research/steer_vs_prompt.py`, `research/structured_superposition.py`, `research/superpos_static.py`, `research/toy_superposition.py`, `research/validate_traits.py`, `research/vector_telepathy.py`, `research/voice_lora.py`, `research/voice_middle.py` |
| Research-only tests -> `clozn-research/tests/` | `research/tests/test_idle_selfplay.py`, `research/tests/test_parliament.py`, `research/tests/test_persistent_injection.py`, `research/tests/test_prompt_vs_prefix_ab.py`, `research/tests/test_quine.py`, `research/tests/test_receipts_as_reward.py` |
| Archive/local-only data -> do not migrate as product source | `research/CLAUDE.md`, `research/data/input.txt`, `research/model_k8.pt`, `research/np_labels_l15.json`, `research/np_stats_l15.json` |

## Staged Migration Order

### Stage 1: Low-risk docs/examples moves

Goal: prove move mechanics and link updates without changing imports or runtime behavior.

Recommended Stage 1 moves:

| Move | Why safe | Required updates | Validation |
|---|---|---|---|
| `ARCHITECTURE.md` -> `docs/ARCHITECTURE.md` | Docs only | update docs links; keep root compatibility stub only if needed | link check/grep |
| `ROADMAP.md` -> `docs/ROADMAP.md` | Docs only | update docs links | link check/grep |
| `STUDIO.md` -> `docs/STUDIO.md` | Docs only | update docs links, preserve command text until server moves | link check/grep |
| `RUNTIME_SPLIT.md` -> `docs/RUNTIME_SPLIT.md` | Docs only | update docs links | link check/grep |
| `research/EXPLAIN_THIS_ANSWER_SPEC.md` -> `docs/EXPLAIN_THIS_ANSWER_SPEC.md` | Product spec, no imports | update server comments later, no behavior | grep old path |
| `engine/lab/cloze_lab/bench/results/*` -> `examples/benchmarks/engine-lab/` | Fixtures/results, no imports observed | update docs references only | grep old path |
| static unused traces from `inspector/demo/*.jsonl` -> `examples/traces/` | Sample trace data | confirm no page `fetch()` or direct filename references first | grep filenames |

Do not include `inspector/demo/pages`, `research/*.py`, `kernels/*`, or `clozn_cli.py` in Stage 1.

### Stage 2: Server module moves

Goal: extract product backend from `research/` into importable `server/` package.

Order:

1. Create `server/` package with no behavior change.
2. Move pure/std-lib modules first: `runlog`, `workspace_lens`, `run_timeline`, `confidence_spans`, `capture_mode`, `memory_mode`, `memory_cards`, `feedback`, `preferences`, `profiles`, `receipt_bundle`, `explain`.
3. Add temporary wrappers under old `research/*.py` names that re-export from `server.*`.
4. Move generation/substrate-heavy modules: `steering`, `self_teach_server`, `dream_memory`, `facts_mode`, `slotmem_qwen`, `brain_readout`, `sae7b`, `denoise_server`, `clozn_server`.
5. Convert top-level imports like `import runlog` to package imports.
6. Update tests from `research/tests` into `tests/server`.

### Stage 3: CLI moves

Goal: put terminal product in `cli/` while preserving `clozn` command compatibility.

Order:

1. Move `clozn_cli.py` to `cli/clozn_cli.py`.
2. Keep root `clozn_cli.py` as a thin compatibility wrapper for at least one release.
3. Keep root `clozn` and `clozn.cmd` wrappers or make them dispatch to `cli/clozn_cli.py`.
4. Update `cmd_studio` to prefer `server/app.py` but support old `research/clozn_server.py` shim.
5. Move CLI tests into `tests/cli`.

### Stage 4: Studio rename/move

Goal: move product frontend from `inspector/demo` to `studio/`.

Order:

1. Move `inspector/demo/app.html`, `clozn.css`, and `pages/` together.
2. Configure server static root to `studio/`.
3. Keep old `/studio.html`, `/app.html`, and `/pages/*` static path compatibility or redirects.
4. Move legacy windows (`brain.html`, `denoise.html`, `engine.html`, `instrument.html`, `talk.html`, memory demos) only after deciding whether each is Studio lab, example, or research.
5. Update `index.html` last, or move it to `studio/local-index.html`.

### Stage 5: Protocol extraction/cleanup

Goal: make shared contracts explicit without changing payload shapes.

Order:

1. Keep `protocol/README.md` and `SPEC.md`.
2. Add schemas/types only after current server/engine payloads are frozen.
3. Extract duplicated event/readout contract comments from engine/server/studio into protocol docs.
4. Do not move `engine/core/include/cloze/events.hpp` until C++ include migration is separately planned.

### Stage 6: Research extraction

Goal: move non-product spikes/findings into separate `clozn-research/`.

Order:

1. Finish Stage 2 wrappers so research scripts can import product modules from `server.*`.
2. Move research findings/docs and spike scripts as a batch.
3. Move research tests with their scripts.
4. Leave a short `research/README.md` tombstone only if needed, pointing to the external repo. Do not keep executable product code under `research/`.

## Risky Import And Path Assumptions

| Assumption | Evidence | Breakage if moved | Mitigation |
|---|---|---|---|
| CLI assumes repo root layout | `REPO = dirname(clozn_cli.py)`; `ENGINE_CORE = REPO/engine/core`; `cmd_studio` uses `REPO/research/clozn_server.py` | Moving CLI changes model dirs, engine build discovery, Studio launch | Introduce `project_root()` helper and root wrapper before moving |
| CLI imports research modules by path injection | `sys.path.insert(0, REPO/research); import runlog` | Moving `runlog.py` breaks `run`, `trace`, `branch`, `explain --last` | Move `runlog` first with `research/runlog.py` re-export shim |
| Server imports sibling research modules as top-level names | `sys.path.insert(0, HERE)` in `clozn_server.py`; many `import memory_cards`, `import runlog`, etc. | Any single module move can break server at runtime | Package `server/` and keep re-export wrappers until all imports are updated |
| Server static root is relative to research | `DEMO = HERE/../inspector/demo` | Moving Studio or server breaks static UI | Configurable static root plus old-path fallback |
| Server imports engine/lab by path | `sys.path.insert(0, HERE/../engine/lab)` | Moving lab or server breaks Dream substrate | Keep lab in place until substrate packaging exists |
| Server imports engine/client by path | `sys.path.insert(0, HERE/../engine/client)` | Moving server breaks `EngineClient` | Package engine client or use explicit import helper |
| CMake expects root `kernels/` | `engine/core/CMakeLists.txt` uses `../../kernels/...` | Moving kernels breaks CUDA/SAE builds | Move kernels only with CMake path update and CUDA test |
| Frontend uses relative script/CSS paths | `app.html` loads `clozn.css` and `pages/*.js`; pages link `../../index.html` | Moving frontend partially causes blank pages/broken links | Move full static bundle together |
| Local index hardcodes old paths | `index.html` links `research/*.py`, `inspector/demo/*`, `inspector/runs/*` | Docs/UI launcher lies after moves | Move/update index in same Studio stage |
| Tests manipulate `sys.path` to `research/` | many `research/tests/*.py` import top-level product modules | Moving tests or modules breaks collection | Stage server package and update conftest/imports together |
| `.gitignore` hides lab model adapters | `/engine/**/models/` ignores `engine/lab/cloze_lab/models/*.py` | Fresh clone may lack required Python adapter source | Narrow ignore rule before any lab work |
| Local data files are used but ignored | `research/np_labels_l15.json`, `np_stats_l15.json`, `model_k8.pt`, `research/runs/*` | Treating them as product source bloats repo; omitting them can break optional readouts | Define explicit fixture/data download policy |

## Minimum Compatibility Shims

| Shim | Needed for | Shape | Remove after |
|---|---|---|---|
| Root `clozn_cli.py` | Existing `python clozn_cli.py ...` usage | imports `cli.clozn_cli:main` | CLI packaging/entry point is stable |
| Root `clozn`, `clozn.cmd` | Existing PATH usage | dispatch to `cli/clozn_cli.py` | never, or keep as permanent launchers |
| `research/clozn_server.py` | Existing `python research/clozn_server.py --port ...` and CLI fallback | imports `server.app:main` | one release after CLI updated |
| `research/runlog.py` and other server module wrappers | Existing top-level imports in tests/scripts | `from server.runlog.runlog import *` style wrappers | after Stage 2 import rewrite |
| Static old path fallback | Existing `/studio.html`, `/pages/*`, `inspector/demo/*` file links | server serves both `studio/` and legacy path aliases | after Studio links are migrated |
| Kernel path alias | CMake/docs during kernel move | temporary CMake variables for old/new kernel roots | after all build scripts updated |
| CLI `studio` path fallback | Users/scripts launching old backend path | try `server/app.py`, fall back to `research/clozn_server.py` shim | after shim removal |

## What Not To Move Yet

- Do not move `research/clozn_server.py` until `server/` wrappers and tests exist.
- Do not move `research/runlog.py` before adding a compatibility wrapper; CLI depends on it.
- Do not move `inspector/demo/pages/*` without moving `app.html`, `clozn.css`, and updating server static root together.
- Do not move `kernels/*` until CMake is updated and CUDA/SAE build validation is planned.
- Do not move `engine/lab` until the ignored `cloze_lab/models/*.py` issue is fixed and Dream substrate imports are packaged.
- Do not move generated/local artifacts (`research/runs`, `research/dream_runs`, `inspector/runs`, `*.pt`, `np_*.json`) into product source without an explicit fixture/data policy.
- Do not extract `inspector/clozn/*` until deciding whether it is legacy research or a protocol/client package.

## Top Safe And Risky Moves

Top 10 safe moves:

1. `research/EXPLAIN_THIS_ANSWER_SPEC.md` -> `docs/EXPLAIN_THIS_ANSWER_SPEC.md`
2. `ARCHITECTURE.md` -> `docs/ARCHITECTURE.md`
3. `ROADMAP.md` -> `docs/ROADMAP.md`
4. `STUDIO.md` -> `docs/STUDIO.md`
5. `RUNTIME_SPLIT.md` -> `docs/RUNTIME_SPLIT.md`
6. `engine/lab/cloze_lab/bench/results/*` -> `examples/benchmarks/engine-lab/`
7. `inspector/demo/workspace_lens_trace.jsonl` -> `examples/traces/workspace_lens_trace.jsonl` after grep confirms no runtime fetch
8. `inspector/demo/denoise_trace.json` -> `examples/traces/denoise_trace.json` after grep confirms no runtime fetch
9. `inspector/demo/memory_timeline.json` -> `examples/traces/memory_timeline.json` after grep confirms no runtime fetch
10. `research/*_findings.md` -> `clozn-research/findings/` in Stage 6 after external repo exists

Top 10 risky moves:

1. `research/clozn_server.py` -> `server/app.py`
2. `research/runlog.py` -> `server/runlog/runlog.py`
3. `research/memory_cards.py` and `research/memory_mode.py` -> `server/memory/`
4. `research/steering.py` -> `server/behavior/steering.py`
5. `research/self_teach_server.py` -> `server/memory/self_teach.py`
6. `clozn_cli.py` -> `cli/clozn_cli.py`
7. `inspector/demo/app.html` plus `pages/*` -> `studio/`
8. `kernels/*` -> `engine/kernels/`
9. `engine/lab/*` -> any new path
10. `inspector/clozn/*` -> any new path before legacy/protocol decision

Unclear files/directories:

- `inspector/clozn/*`: old package, useful concepts, but not the current CLI/server/Studio path.
- `inspector/demo/brain.html`, `engine.html`, `denoise.html`, `instrument.html`, `talk.html`, memory pages: product-accessible demos, but not all are core Studio.
- `engine/lab`: product dependency today, but target `engine/` is intended for native/runtime core; keep temporarily.
- `engine/lab/cloze_lab/models/*.py`: imported but ignored by Git.
- Local `research/np_labels_l15.json`, `research/np_stats_l15.json`: optional readout data, large and ignored.
- `research/model_k8.pt`, `research/runs/*`, `research/dream_runs/*`, `inspector/runs/*`: generated/local artifacts.

Recommended first actual migration task:

Fix the `.gitignore` rule that hides `engine/lab/cloze_lab/models/*.py`, then add/track or deliberately relocate those adapter files. This is the smallest source-control correctness fix and removes a hidden dependency before any physical reorg starts.
