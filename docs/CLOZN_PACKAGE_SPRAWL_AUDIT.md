# Clozn Package Sprawl Audit

Scope: `clozn/`

Date: 2026-07-09

`__pycache__/` and `.pyc` files are intentionally excluded from the file table. They are ignored generated artifacts, not product source. `git ls-files clozn/__pycache__` returned no tracked files, and `.gitignore` ignores `__pycache__/`.

## Commands Run

```bash
cd C:/Users/brigi/src/clozn/clozn
find . -type f -not -path './.git/*'
find . -type f -not -path './.git/*' -print0 | xargs -0 wc -l
```

```bash
cd C:/Users/brigi/src/clozn
rg -n "<module_or_filename_without_ext>"
rg -n "from clozn|import clozn|clozn_server|cli"
pytest -q
pytest -q tests
```

## Test Results

`pytest -q` is available, but full-repo collection fails before running product tests because it descends into third-party and engine test trees with missing dependencies or import collisions:

- `appium` missing under `engine/core/third_party/llama.cpp/scripts/snapdragon/qdc/tests`
- `wget` missing under `engine/core/third_party/llama.cpp/tools/server/tests`
- duplicate `test_reference` import mismatch between engine kernel test directories
- `engine/lab/tests/test_events.py` import mismatch for `WorkspaceReadout`

Product tests pass:

```text
pytest -q tests
1204 passed, 10 skipped
```

## Summary

No source or runtime data file in `clozn/` is deletion-ready under the rule "do not recommend deletion unless the file has no imports, no route/static references, no tests, and no docs/examples dependency."

The main maintainability pressure is concentrated in:

- `clozn/clozn_server.py`
- `clozn/cli.py`
- `clozn/runlog.py`
- `clozn/receipts.py`
- `clozn/self_teach_server.py`
- `clozn/steering.py`
- `clozn/narrate.py`
- `clozn/slotmem_qwen.py`

The cleanest direction is not deletion. It is package extraction with compatibility shims: `server/`, `server/routes/`, `server/substrates/`, `server/memory/`, `server/receipts/`, `server/runlog/`, `server/behavior/`, `server/readouts/`, and `cli/`.

## Audit Table

| Path | Lines | What this file does | Used in product? | Evidence of usage | Belongs somewhere else? | If >500 lines, justified? | Split/delete/move recommendation | Risk level |
|---|---:|---|---|---|---|---|---|---|
| `clozn/__init__.py` | 6 | Package marker and docstring for the flat product package. | Yes | Imports throughout tests use `from clozn import ...`; README identifies `clozn/` as product Python package. | No. | N/A | Keep. | Safe to keep |
| `clozn/__main__.py` | 7 | `python -m clozn` entry point into the CLI. | Yes | `clozn.cmd:4`, `clozn.sh:3`, imports `clozn.cli.main`; README documents `python -m clozn`. | No. | N/A | Keep. | Safe to keep |
| `clozn/atlas_concepts.py` | 183 | Concept labels, seed corpora, and content-word helper for readouts. | Yes | Imported by `brain_readout.py`; covered by `tests/test_studio.py`; referenced in docs. | Yes, `server/readouts/atlas_concepts.py`. | N/A | Move with compatibility shim during readouts package split. | Safe to move with shim |
| `clozn/brain_readout.py` | 145 | Qwen/SAE concept readout provider. | Yes | Imported by `clozn_server.py` for Qwen substrate; covered by `tests/test_studio.py`; docs mention SAE/concept stack. | Yes, `server/readouts/brain.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/capture_mode.py` | 46 | Capture-tier settings and policy. | Yes | Server `/capture/tier`; imported by `replay.py`; covered by `tests/test_capture_mode.py`. | Yes, `server/runlog/capture_mode.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/cli.py` | 1793 | Main terminal CLI: `run`, `serve`, `studio`, `trace`, `branch`, `explain`, `preferences`, `test`, model management. | Yes | `__main__.py`, root wrappers, README commands, CLI tests. | Yes, `cli/` package with command modules. | No. It mixes engine process management, HTTP calls, rendering, runlog access, command parsing, and test harness glue. | Split by command group and keep a thin `clozn.cli` compatibility entry. | Needs deeper review |
| `clozn/clozn_server.py` | 3123 | Main HTTP backend: route dispatcher, substrate loading, Studio static serving, OpenAI-compatible endpoint, memory, receipts, replay, readouts, engine proxy routes. | Yes | CLI launches `python -m clozn.clozn_server`; many `/v1`, `/runs`, `/memory`, `/engine`, `/profiles`, `/facts`, `/feedback`, `/preferences` routes; broad server test coverage. | Yes, `server/app.py`, `server/routes/*`, `server/substrates/*`. | No. It is the central sprawl point. Route families, substrates, static serving, export, and product helpers are fused. | Split behind shims by route family and substrate. Do not move in one large cut. | Needs deeper review |
| `clozn/confidence_spans.py` | 154 | Converts stored run traces into contiguous confidence spans and summaries. | Yes | Server `/runs/<id>/spans`; `tests/test_confidence_spans.py`; `tests/test_confidence_spans_server.py`. | Yes, `server/runlog/confidence_spans.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/counterfactual.py` | 239 | Counterfactual dial regeneration and dose sweep helpers. | Yes | Server `/runs/<id>/counterfactual`; imports `receipts` and `replay`; covered by helper and server tests. | Yes, `server/replay/counterfactual.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/data/dial_library_shipped.json` | 372 | Curated shipped dial library metadata. | Dev-only | Used by calibration/deploy scripts, tests, and docs. No direct server import found; deployed runtime file appears to be `~/.clozn/studio_library.json`. | If runtime-owned, `server/data/`; if calibration-owned, `scripts/calibration/data/`. | N/A | Keep. Do not delete; clarify ownership and path during package split. | Safe to keep |
| `clozn/denoise_server.py` | 116 | Standalone Dream denoise server and imported `trace_for` helper. | Yes | `clozn_server.py` imports `trace_for`; `/denoise` route; `index.html`; `tests/test_studio.py`. | Yes, helper in `server/substrates/denoise.py`; standalone demo server may belong under examples or demos. | N/A | Move helper with shim; separate standalone server concerns. | Safe to move with shim |
| `clozn/dream_memory.py` | 244 | Dream soft-prefix memory adapter and memory substrate helpers. | Yes | Imported by `clozn_server.py` for Dream substrate; covered by `tests/test_studio.py`. | Yes, `server/substrates/dream_memory.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/explain.py` | 216 | Builds the "explain this answer" object from stored run data. | Yes | Server `/runs/<id>/explain`; export route; imported by `narrate.py`; CLI explain path; tests. | Yes, `server/receipts/explain.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/facts_mode.py` | 64 | Facts-tier feature flag and per-profile store path helpers. | Yes | Server `/facts/*`; imported by `clozn_server.py`; covered by facts tests. | Yes, `server/memory/facts_mode.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/feedback.py` | 117 | Feedback signal JSON store. | Yes | Server `/feedback` and `/feedback/summary`; covered by `tests/test_feedback.py`. | Yes, `server/behavior/feedback.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/fit_planner.py` | 344 | GGUF header/range reader for "will it fit?" CLI planning. | Yes | Imported by `cli.py`; covered by `tests/test_fit_planner.py`. | Yes, `cli/fit_planner.py` or `server/models/fit_planner.py` if shared later. | N/A | Move with shim. | Safe to move with shim |
| `clozn/memory_cards.py` | 211 | Memory card CRUD, provenance, status, active text compilation support. | Yes | Server memory routes; imported by `explain.py`, `memory_mode.py`, tests across memory/server/receipts. | Yes, `server/memory/cards.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/memory_mode.py` | 165 | Memory mode settings and prompt-block compiler. | Yes | Server memory mode route; imported by facts, replay, capture, tests. | Yes, `server/memory/mode.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/narrate.py` | 553 | Accountable narration and confabulation-diff logic. | Yes | Server `/runs/<id>/narrate`; CLI `explain --why`; imports `explain`; narrate tests. | Yes, `server/receipts/narrate.py`. | Partly. One cohesive feature, but parsing/matching/generation orchestration are mixed. | Split claim parsing, citation/fact extraction, support matching, and orchestration. | Needs deeper review |
| `clozn/preferences.py` | 144 | Learned-preference proposal and resolve store. | Yes | Server `/preferences` and `/preferences/resolve`; CLI preferences tests; `feedback` integration. | Yes, `server/behavior/preferences.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/profiles.py` | 187 | Persona/profile bundle validation, save/switch helpers. | Yes | Server `/profiles/*`; profile tests; server imports. | Yes, `server/profiles.py` or `server/profiles/store.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/receipt_bundle.py` | 207 | Unified export bundle and Markdown rendering. | Yes | Server `/runs/<id>/export`; `tests/test_export_server.py`; testkit attachment round trip. | Yes, `server/receipts/bundle.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/receipts.py` | 718 | Causal receipt and prove-all helpers. | Yes | Server `/runs/<id>/receipt` and `/receipts`; imported by `testkit.py`, `counterfactual.py`; README/docs/tests. | Yes, `server/receipts/receipts.py` split into submodules. | Partly. It is a core product feature, but metrics, ablation planning, regeneration, and forced deltas are mixed. | Split metrics, replay regeneration, forced-delta logic, and receipt assembly. | Needs deeper review |
| `clozn/rederive.py` | 170 | Teacher-forced re-derivation scoring. | Yes | Server `/runs/<id>/rederive`; imported by `receipts.py`; tests/docs. | Yes, `server/receipts/rederive.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/replay.py` | 262 | Replay and compare helper for changed cards/dials/settings. | Yes | Server `/runs/<id>/replay`; imported by `receipts.py` and `counterfactual.py`; tests. | Yes, `server/replay/replay.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/run_timeline.py` | 244 | Converts run records into semantic timeline events. | Yes | Server `/runs/<id>/timeline`; timeline tests. | Yes, `server/runlog/timeline.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/runlog.py` | 725 | Run journal, trace normalization, lineage/family summaries, tiny-test attachment. | Yes | CLI trace/logging; server `/runs*`; many tests; README references `~/.clozn/runs`. | Yes, `server/runlog/store.py`, `server/runlog/trace.py`, `server/runlog/lineage.py`. | Partly. It is central, but several distinct responsibilities are fused. | Split storage, trace normalization, lineage/family, and update helpers. | Needs deeper review |
| `clozn/sae7b.py` | 69 | Qwen SAE loading and feature extraction helpers. | Yes | Imported by `brain_readout.py`; server transitive; `tests/test_studio.py`; docs. | Yes, `server/readouts/sae7b.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/self_teach_server.py` | 804 | Qwen self-teach/model harness, logits trace processor, helper algorithms, standalone server. | Yes | `clozn_server.py` imports `SelfTeach`, `RecordingLogitsProcessor`, `steps_from_records`, `finish_reason_from_generated_ids`; tests. | Yes, split between `server/substrates/qwen.py`, `server/memory/self_teach.py`, and demo/server wrapper. | Partly. Product helpers are valid, but standalone server, model harness, and utility algorithms are mixed. | Split product substrate code from standalone/dev server and utility helpers. | Needs deeper review |
| `clozn/semantic_matcher.py` | 214 | Optional NLI/cross-encoder support matcher for narration. | Yes | `clozn_server.py` selects it for `/narrate`; semantic matcher tests. | Yes, `server/receipts/semantic_matcher.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/slotmem_qwen.py` | 506 | Qwen slot-memory facts store plus CLI/sweep helpers. | Yes | Server facts routes import it; slot memory tests; docs. | Yes, `server/memory/slotmem_qwen.py`; move sweep/dev CLI elsewhere. | Partly. Store is product code; sweep/demo tail is not product-route code. | Split product store from sweep/dev CLI. | Needs deeper review |
| `clozn/steering.py` | 664 | Tone dials and steering adapters for HF/Qwen, Dream, and engine substrates. | Yes | Server dials and engine chat; scripts; tests; README. | Yes, `server/behavior/steering.py` plus adapter-specific modules. | Partly. Shared axis/catalog data is useful, but adapter implementations are separable. | Split axes/catalog, HF steering, Dream steering, Engine steering. | Needs deeper review |
| `clozn/testkit.py` | 493 | User-authored run-level tiny-test harness. | Yes | `cli.py` implements `clozn test`; README; `tests/test_testkit.py`, `tests/test_testkit_cli.py`. | Yes, `server/testkit.py` or `server/receipts/testkit.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/timetravel.py` | 405 | KV snapshot accounting and run branch/time-travel helpers. | Yes | Server `/timetravel/*` and `/runs/<id>/branch`; tests. | Yes, `server/replay/timetravel.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/topic_gate.py` | 245 | Topic relevance and openness gate for memory. | Yes | Imported by `clozn_server.py` and `self_teach_server.py`; memory/topic tests. | Yes, `server/memory/topic_gate.py`. | N/A | Move with shim. | Safe to move with shim |
| `clozn/workspace_lens.py` | 209 | Workspace/J-lens readout normalization and protocol helpers. | Yes | Imported by `runlog.py` and `clozn_server.py`; docs and trace tests. | Yes, `server/readouts/workspace_lens.py`. | N/A | Move with shim. | Safe to move with shim |

## Suggested Split Order

1. Move leaf modules first with import shims: `confidence_spans`, `capture_mode`, `feedback`, `preferences`, `profiles`, `workspace_lens`, `run_timeline`.
2. Split memory modules: `memory_cards`, `memory_mode`, `topic_gate`, `facts_mode`, then `slotmem_qwen`.
3. Split receipts/replay modules: `explain`, `rederive`, `replay`, `counterfactual`, `receipt_bundle`, then `receipts` and `narrate`.
4. Split readouts/substrates: `atlas_concepts`, `sae7b`, `brain_readout`, `dream_memory`, `denoise_server`, `self_teach_server`.
5. Split `steering.py` after behavior routes and tests are stable.
6. Split `clozn_server.py` last, route family by route family, behind compatibility shims.
7. Split `cli.py` after server entry points have stable import paths.

