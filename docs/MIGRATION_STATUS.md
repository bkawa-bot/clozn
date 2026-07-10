# Migration Status

Branch: `repo-reorg-clean-break`

## Current Phase

Batch A complete; Batch B complete:

- Phase 0: prep branch and cleanup rules
- Phase 1: destination package skeletons
- Phase 2: low-risk leaf module moves
- Phase 3: `runlog.py` moved into `clozn.runs` and split into storage, trace, summaries, attachments, and lineage
- Phase 4: memory/facts modules moved into `clozn.memory`; `slotmem_qwen.py` split into runtime store, fact banks, sweep logic, and CLI entrypoint
- Phase 5: receipts/explain/narrate/rederive/export modules moved into `clozn.receipts`; receipt core and narration split into focused modules
- Phase 6: counterfactual and time-travel moved into `clozn.replay`
- Phase 7: steering moved into `clozn.behavior.steering` and split into axes, catalog, HF, Dream, engine, and library modules
- Phase 8: remaining readout/substrate modules moved into `clozn.readouts`, `clozn.substrates`, and `clozn.dev`
- Phase 9: `clozn_server.py` moved whole (no split) into `clozn.server.app`; `fit_planner.py` moved into `clozn.cli.fit_planner` (its only consumer's package). No flat top-level `clozn/*.py` modules remain.

## Moved Modules

- `clozn/cli.py` -> `clozn/cli/main.py`
- `clozn/profiles.py` -> `clozn/profiles/store.py`
- `clozn/receipts.py` -> `clozn/receipts/core.py`
- `clozn/replay.py` -> `clozn/replay/replay.py`
- `clozn/testkit.py` -> `clozn/testkit/runner.py`
- `clozn/capture_mode.py` -> `clozn/runs/capture_mode.py`
- `clozn/confidence_spans.py` -> `clozn/runs/confidence_spans.py`
- `clozn/run_timeline.py` -> `clozn/runs/timeline.py`
- `clozn/workspace_lens.py` -> `clozn/readouts/workspace_lens.py`
- `clozn/feedback.py` -> `clozn/behavior/feedback.py`
- `clozn/preferences.py` -> `clozn/behavior/preferences.py`
- `clozn/atlas_concepts.py` -> `clozn/readouts/atlas_concepts.py`
- `clozn/sae7b.py` -> `clozn/readouts/sae7b.py`
- `clozn/runlog.py` -> `clozn/runs/store.py`
- trace normalization/helpers -> `clozn/runs/trace.py`
- compact run summaries/flags -> `clozn/runs/summaries.py`
- tiny-test run attachments -> `clozn/runs/attachments.py`
- run lineage/family lookup -> `clozn/runs/lineage.py`
- `clozn/memory_cards.py` -> `clozn/memory/cards.py`
- `clozn/memory_mode.py` -> `clozn/memory/mode.py`
- `clozn/facts_mode.py` -> `clozn/memory/facts_mode.py`
- `clozn/topic_gate.py` -> `clozn/memory/topic_gate.py`
- `clozn/slotmem_qwen.py` -> `clozn/memory/slotmem_qwen/store.py`
- slot-memory fact banks -> `clozn/memory/slotmem_qwen/facts.py`
- slot-memory smoke/sweep logic -> `clozn/memory/slotmem_qwen/sweep.py`
- slot-memory manual entrypoint -> `clozn/memory/slotmem_qwen/cli.py`
- `clozn/explain.py` -> `clozn/receipts/explain.py`
- `clozn/narrate.py` -> `clozn/receipts/narrate.py`
- `clozn/rederive.py` -> `clozn/receipts/rederive.py`
- `clozn/receipt_bundle.py` -> `clozn/receipts/bundle.py`
- `clozn/semantic_matcher.py` -> `clozn/receipts/semantic_matcher.py`
- receipt metric math -> `clozn/receipts/metrics.py`
- receipt ablation/delta assembly -> `clozn/receipts/deltas.py`
- teacher-forced receipt scoring -> `clozn/receipts/forced.py`
- narration fact support -> `clozn/receipts/fact_support.py`
- narration claim splitting -> `clozn/receipts/claim_extraction.py`
- narration rendering -> `clozn/receipts/narrative_rendering.py`
- confabulation diff -> `clozn/receipts/confabulation_diff.py`
- `clozn/counterfactual.py` -> `clozn/replay/counterfactual.py`
- `clozn/timetravel.py` -> `clozn/replay/timetravel.py`
- `clozn/steering.py` -> `clozn/behavior/steering/axes.py`
- preference-to-dial routing -> `clozn/behavior/steering/catalog.py`
- PyTorch/HF steering adapter -> `clozn/behavior/steering/hf_adapter.py`
- Dream diffusion steering adapter -> `clozn/behavior/steering/dream_adapter.py`
- native engine steering adapter -> `clozn/behavior/steering/engine_adapter.py`
- steering library helpers -> `clozn/behavior/steering/library.py`
- `clozn/brain_readout.py` -> `clozn/readouts/brain.py`
- `clozn/dream_memory.py` -> `clozn/substrates/dream_memory.py`
- `clozn/denoise_server.py` -> `clozn/substrates/denoise.py`
- standalone Dream denoise server -> `clozn/dev/denoise_server.py`
- `clozn/self_teach_server.py` -> `clozn/substrates/self_teach.py`
- Qwen/HF model loading and trace helpers -> `clozn/substrates/qwen.py`
- standalone self-teach server -> `clozn/dev/self_teach_server.py`
- `clozn/clozn_server.py` -> `clozn/server/app.py` (single-module move, not a split; run via `python -m clozn.server.app`)
- `clozn/fit_planner.py` -> `clozn/cli/fit_planner.py` (its only consumer is `clozn/cli/main.py`'s `plan` command)

## Tests Currently Green

Current product suite:

```text
pytest -q tests
1204 passed, 10 skipped
```

Focused Phase 3 suite:

```text
pytest -q tests\test_runlog.py tests\test_runlog_lineage.py tests\test_trace_capture.py tests\test_run_timeline.py tests\test_confidence_spans.py
100 passed
```

Focused Phase 4 suite:

```text
pytest -q tests\test_memory_cards.py tests\test_memory_mode.py tests\test_facts_mode.py tests\test_facts_server.py tests\test_topic_gate.py tests\test_prompt_relevance.py tests\test_slotmem_store.py tests\test_slotmem_shared.py tests\test_memory_wiring.py tests\test_propose_memory.py
172 passed
```

Focused Phase 5 suite:

```text
pytest -q tests\test_receipts.py tests\test_receipts_server.py tests\test_explain.py tests\test_explain_server.py tests\test_explain_cli.py tests\test_narrate.py tests\test_narrate_server.py tests\test_narrate_cli.py tests\test_rederive.py tests\test_rederive_server.py tests\test_export_server.py tests\test_semantic_matcher_gated.py
221 passed, 4 skipped
```

Focused Phase 6 suite:

```text
pytest -q tests\test_replay.py tests\test_counterfactual.py tests\test_counterfactual_server.py tests\test_timetravel.py tests\test_timetravel_server.py tests\test_timetravel_determinism.py
104 passed, 1 skipped
```

Focused Phase 7 suite:

```text
pytest -q tests\test_dial_suggestion.py tests\test_steering_headroom.py tests\test_dial_autocalibrate.py tests\test_dial_autocalibrate_engine.py tests\test_engine_substrate.py tests\test_engine_add_custom.py tests\test_engine_library_dials.py tests\test_dial_calibration_server.py tests\test_dial_library_server.py tests\test_studio.py
303 passed, 1 skipped
```

Focused Phase 8 suite:

```text
pytest -q tests\test_studio.py tests\test_hf_trace.py tests\test_fair_capacity.py tests\test_memory_mode.py tests\test_generation_meta.py tests\test_jlens_server.py tests\test_engine_substrate.py tests\test_propose_memory.py
133 passed
```

Full root `pytest -q` is not a valid baseline yet because it collects third-party and engine test trees with missing dependencies/import collisions.

Phase 9 (this move) reverified the full product suite after relocating `clozn_server.py` and `fit_planner.py`:

```text
pytest -q tests
1204 passed, 10 skipped
```

Also reverified clean: `python -c "import clozn"`, `python -c "import clozn.server.app"`, `python -m clozn --help`, and `python -m clozn.server.app --help` (argparse help only -- no server/model/GPU started).

## Known Broken Commands

None currently known through Phase 9.

## Remaining Import Errors

None currently known through Phase 9. Python sources no longer import `clozn.runlog`, `clozn.memory_cards`,
`clozn.memory_mode`, `clozn.facts_mode`, `clozn.topic_gate`, `clozn.slotmem_qwen`, `clozn.explain`,
`clozn.narrate`, `clozn.rederive`, `clozn.receipt_bundle`, `clozn.semantic_matcher`,
`clozn.counterfactual`, `clozn.timetravel`, the old flat readout/substrate modules, the old flat
steering module, `clozn.clozn_server`, or `clozn.fit_planner`. No flat `clozn/*.py` modules remain
except `clozn/__init__.py` and `clozn/__main__.py`.

## Compatibility Policy

This is a breaking cleanup branch. Preserve `python -m clozn` and the meaningful CLI/server/product workflows, but do not preserve old internal flat imports such as `clozn.runlog`, `clozn.receipts`, or `clozn.clozn_server`.
