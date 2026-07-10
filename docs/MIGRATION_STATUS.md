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
- Phase 11: flat `scripts/` reorganized into purpose-based subdirectories (`bench/`, `calibration/`, `data/`, `smoke/`, plus the pre-existing `cleanup/`); moves-only, same precedent as Phase 9 (no internal split of the two giant calibration scripts).
- Phase 9 (continued) -- the full `clozn.server.app` route split: 3124 lines split into an app-factory
  scaffold (`app.py`, `config.py`, `static.py`, `sse.py`) + 13 `clozn/server/routes/*.py` HTTP-route
  modules, one per family. See "Phase 9 full split" below.

## Phase 9 full split -- `clozn.server.app` route extraction

The Phase 9 move above landed `clozn_server.py` whole, unsplit. This chip did the actual split, following
`docs/REPO_REORG_PLAN.md`'s target `clozn/server/` layout, in the documented incremental order (one
family at a time, `pytest -q tests` after each).

**The mechanism.** `clozn/server/app.py` still owns ALL shared mutable state -- `SUB`/`SUBNAME`/`ARGS`,
`ENGINE`/`ENGINE_QWEN`, the retrain lock (`_TRAIN_LOCK`/`_RETRAIN`), `SLOTS`/`SNAPSHOTS`, and every helper
function/class ~30 test files monkeypatch directly via `monkeypatch.setattr(cs, "X", ...)` (`cs` ==
`clozn.server.app`). Each `routes/<family>.py` does `from clozn.server import app as ctx` and reads
`ctx.SUB`, `ctx._prompt_block_for(...)`, etc. at CALL time (never `from clozn.server.app import X`, which
would bind a private copy immune to monkeypatching) -- so a test's `monkeypatch.setattr(cs, "SUB", Fake())`
is observed identically whether the code that reads `SUB` lives in `app.py` or in a route module. Each
route module exposes `try_get(handler, path)` / `try_post(handler, path, body)` returning `True` once it
has written a response (via the handler's existing `_json`/`_send`) or `False` to let the next family try
-- `do_GET`/`do_POST` are now a 6-line and an 18-line dispatch loop over ordered `_GET_ROUTES`/
`_POST_ROUTES` lists, ending in the one fallback that isn't a per-path route: `SUB.handle(path, body)`,
the substrate-polymorphic dispatch for `/memory/*`/`/steer/*` that lives in the `Substrate` class
hierarchy, not in a route file (see "What did NOT move" below).

**A real bug this caught, fixed:** `python -m clozn.server.app` runs this file as `__main__` -- a SEPARATE
module object from `clozn.server.app` in `sys.modules` terms. Once route modules started doing
`from clozn.server import app as ctx`, that import silently re-executed `app.py` a second time under its
real dotted name (fresh `SUB=None`/`SUBNAME="qwen"` defaults, a second `ENGINE`/`ENGINE_QWEN` pair, ...),
so `main()` mutated the `__main__` copy while every route handler read the other, untouched copy --
invisible to the whole test suite (which only ever imports the dotted name, never runs this as
`__main__`) but fatal to a live boot (`/state` reported `substrate: "qwen"` no matter what `--substrate`
was passed). Fixed with one line near the top of `app.py`:
`sys.modules.setdefault("clozn.server.app", sys.modules[__name__])`, run before the routes/* imports at
the bottom of the file, so both names resolve to the same module object either way. Caught only because
this chip's hard gate requires booting the real server, not just green unit tests -- exactly the
unit-green-does-not-mean-it-works case the gate exists for.

**Families extracted, in order (all landed, all green):**

1. scaffolding + app factory -- `config.py` (path/env/sys.path setup, moved out of app.py's top),
   `static.py` (Studio static-file serving), `sse.py` (the `/v1/chat/completions` SSE writer), plus
   `routes/health.py` (`/substrate`, `/engine/health`, `/state`, `/capture/tier`) and the
   `_GET_ROUTES`/`_POST_ROUTES` dispatch-loop mechanism itself.
2. `routes/runs.py` -- `/runs`, `/runs/<id>` (generic, registered LAST so more-specific suffixes get
   first refusal), `/timeline`, `/lineage`, `/family`, `/spans`.
3. `routes/memory.py` (`/memory/mode`, `/memory/<id>/runs`, `/runs/<id>/propose-memory`) and
   `routes/facts.py` (all of `/facts/*`).
4. `routes/receipts.py` -- `/runs/<id>/export`, `/explain`, `/receipts`, `/receipt`, `/rederive`,
   `/jlens` (general + per-run), `/narrate`.
5. `routes/replay.py` (`/runs/<id>/replay`, `/runs/<id>/counterfactual`) and `routes/timetravel.py`
   (`/timetravel/mode`, `/timetravel/stats`, `/runs/<id>/branch`).
6. `routes/profiles.py` (`/profiles/*`), `routes/preferences.py` (`/preferences`,
   `/preferences/resolve`), `routes/feedback.py` (`/feedback`, `/feedback/summary`).
7. `routes/openai.py` (`/v1/models`, `/v1/chat/completions`), `routes/engine.py` (`/engine/observe`,
   `/engine/steer/axes`, `/engine/steer/check`, `/engine/chat`, plus `/say` and `/denoise` -- the two
   studio-chat surfaces that log a run, placed here since no better-fitting family exists for them in
   the target tree), `routes/readouts.py` (`/engine/harvest`, `/engine/layers`, `/engine/concepts`).

**What did NOT move (by design, not by running out of time):** the `Substrate`/`QwenSubstrate`/
`DreamSubstrate`/`_EngineMemory`/`EngineSubstrate` class hierarchy (~880 lines) and the helper functions
they and the route modules share stay in `app.py`. These are substrate-polymorphic DOMAIN dispatch (a
card-CRUD/tone-dial trait surface shared by three different model backends), not per-path HTTP routing --
`Substrate._memory`/`Substrate._steer` are reached through the one generic `SUB.handle(path, body)`
fallback at the end of `do_POST`, not through a `routes/*.py` file. They are also the most
monkeypatch-entangled part of the file (nearly every test-patched symbol in the `cs.X` list is read from
inside these classes); relocating them would be a substrate-package refactor (arguably `clozn.substrates`
territory), not a mechanical HTTP split, and was out of scope for this chip. `behavior.py` (listed in the
target tree for `/steer/*`, `/behavior/*`) was not created for the same reason: there is no inline
per-path dispatch for `/steer/*` to extract (it is entirely `Substrate._steer`), and no `/behavior/*`
route exists in this codebase at all.

**Resulting file line counts** (`wc -l`):

```text
clozn/server/app.py              2362   (was 3124)
clozn/server/config.py              21
clozn/server/static.py              28
clozn/server/sse.py                 65
clozn/server/routes/__init__.py      1
clozn/server/routes/health.py       59
clozn/server/routes/runs.py         72
clozn/server/routes/memory.py      106
clozn/server/routes/facts.py        47
clozn/server/routes/receipts.py    189
clozn/server/routes/replay.py       58
clozn/server/routes/timetravel.py   85
clozn/server/routes/profiles.py     69
clozn/server/routes/preferences.py  42
clozn/server/routes/feedback.py     26
clozn/server/routes/openai.py       74
clozn/server/routes/engine.py      187
clozn/server/routes/readouts.py     38
```

No route file exceeds 189 lines (target: none `>>500`). `app.py` shrank by 762 lines (24%); the remainder
is the shared-context state + the `Substrate` domain hierarchy described above, not undispersed routing.

**Verification (green at every family boundary, not just at the end):**

```text
pytest -q tests
1204 passed, 10 skipped
```

`python -c "import clozn.server.app"` clean; `python -m clozn.server.app --help` exits 0;
`python -m clozn --help` exits 0. LIVE boot verified (`python -m clozn.server.app --port <free>
--substrate engine`, CPU-only, no GPU/model): `GET /` -> 200 (6254-byte HTML), `GET /state` -> 200 with
the CORRECT active substrate + real dial/card counts (post sys.modules fix), `GET /substrate`,
`GET /runs`, `GET /runs/<id>`, `GET /runs/<id>/{timeline,spans,export,lineage}` (export vs the generic
by-id route both resolve correctly -- the ordering-sensitive case), `POST /runs/<id>/explain`,
`GET /v1/models`, `GET /memory/mode`, `GET /profiles/list`, `GET /timetravel/mode`,
`POST /feedback/summary` -- all responded correctly; `GET /does-not-exist` -> 404. Server terminated
cleanly.

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

### Phase 11 — `scripts/` reorg (moves-only)

- `scripts/bench_batched_receipts.py` -> `scripts/bench/batched_receipts.py`
- `scripts/bench_whitebox_tax.py` -> `scripts/bench/whitebox_tax.py`
- `scripts/deploy_dial_library.py` -> `scripts/calibration/deploy_dial_library.py`
- `scripts/gen_dial_calibration.py` -> `scripts/calibration/gen_dial_calibration.py`
- `scripts/dial_autocalibrate.py` -> `scripts/calibration/torch_autocalibrate.py` (renamed to distinguish from the engine rig)
- `scripts/dial_autocalibrate_engine.py` -> `scripts/calibration/engine_autocalibrate.py` (renamed to distinguish from the PyTorch rig)
- `scripts/fetch_np_labels.py` -> `scripts/data/fetch_np_labels.py`
- `scripts/fetch_np_stats.py` -> `scripts/data/fetch_np_stats.py`
- `scripts/smoke_engine_substrate.py` -> `scripts/smoke/engine_substrate.py`

Each moved script gained one extra `os.path.dirname(...)` / `".."` level in its own HERE-relative
repo-root, `clozn/data/dial_library_shipped.json`, and `engine/client` path lookups (they now sit one
directory deeper); these are load-bearing fixes, not gold-plating -- without them
`deploy_dial_library.py`/`gen_dial_calibration.py`/`engine_autocalibrate.py` silently pointed at a
directory that no longer existed, which 4 tests in `tests/test_dial_library_server.py` caught. The
three test files that import these scripts directly (`tests/test_dial_autocalibrate.py`,
`tests/test_dial_autocalibrate_engine.py`, `tests/test_dial_library_server.py`) had their
`sys.path.insert`/`import` lines updated to the new `scripts/calibration/` location and module names.

**Deferred (not done in Phase 11):**
- The internal split of the two giant calibration scripts (`torch_autocalibrate.py`, ~1150 lines;
  `engine_autocalibrate.py`, ~680 lines) into a `calibration_lib/` package. Moved+renamed whole only,
  same precedent as the Phase 9 server/CLI moves-not-splits.
- The Neuronpedia data-location policy (`clozn/data/neuronpedia/` + `~/.clozn/cache` for
  `scripts/data/fetch_np_labels.py`/`fetch_np_stats.py`'s output). Touches product readout code; left
  for a later chip.

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

Phase 11 (scripts/ reorg) reverified the full product suite after the moves + HERE-relative path fixes:

```text
pytest -q tests
1204 passed, 10 skipped
```

Also reverified clean by direct invocation (not just import): `python scripts/calibration/gen_dial_calibration.py --check`, `python scripts/calibration/deploy_dial_library.py --check`, `python scripts/calibration/torch_autocalibrate.py --help`, and `python scripts/calibration/engine_autocalibrate.py --help` -- all resolve `clozn/data/dial_library_shipped.json` and the repo-root `clozn` package correctly from the new one-level-deeper location.

## Known Broken Commands

None currently known through Phase 11.

## Remaining Import Errors

None currently known through Phase 11. Python sources no longer import `clozn.runlog`, `clozn.memory_cards`,
`clozn.memory_mode`, `clozn.facts_mode`, `clozn.topic_gate`, `clozn.slotmem_qwen`, `clozn.explain`,
`clozn.narrate`, `clozn.rederive`, `clozn.receipt_bundle`, `clozn.semantic_matcher`,
`clozn.counterfactual`, `clozn.timetravel`, the old flat readout/substrate modules, the old flat
steering module, `clozn.clozn_server`, or `clozn.fit_planner`. No flat `clozn/*.py` modules remain
except `clozn/__init__.py` and `clozn/__main__.py`. No flat `.py` files remain directly under `scripts/`
either -- only subdirectories (`bench/`, `calibration/`, `data/`, `smoke/`, `cleanup/`).

## Compatibility Policy

This is a breaking cleanup branch. Preserve `python -m clozn` and the meaningful CLI/server/product workflows, but do not preserve old internal flat imports such as `clozn.runlog`, `clozn.receipts`, or `clozn.clozn_server`.
