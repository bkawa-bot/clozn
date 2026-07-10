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
- Phase 10: the full `clozn.cli.main` split: 1793 lines split into `main.py` (argparse root + dispatch),
  `formatting.py`, `engine_process.py`, `trace_io.py`, and 7 `clozn/cli/commands/*.py` modules, one per
  command family. See "Phase 10 full split" below.

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

## Phase 10 full split -- `clozn.cli.main` command extraction

Split the monolithic `clozn/cli/main.py` (1793 lines) into command modules + helper modules, in the
documented incremental order (helpers first, then leaf commands, then run/serve/studio, then
explain/trace/branch), `pytest -q tests` after each family, exactly like the Phase 9 route split. Unlike
the server, `main.py` has one piece of genuine shared mutable state to carry across modules: `HOME`
(the `~/.clozn` root) and `CloznError` (the one user-facing exception class `main()` catches). Both stay
owned by `main.py`; every other module reaches them via `from clozn.cli import main as ctx` ->
`ctx.HOME` / `raise ctx.CloznError(...)`, read at CALL time (never `from clozn.cli.main import HOME`,
which would bind a stale copy immune to a later `_setup_console()` call or a test's monkeypatch). The
color globals (`DIM`/`BOLD`/`RST`/`COLOR`) got the same treatment but a different owner: they now live in
`formatting.py` itself (not `main.py`), and every module that prints with them does
`from clozn.cli import formatting as fmt` -> `fmt.DIM` etc. Functions that read them internally
(`_paint`, `_conf_rgb`, `_heatmap_lines`, ...) are safe to import directly by name anywhere
(`from clozn.cli.formatting import _paint`) despite living in `formatting.py`, since a Python function
always reads its OWN defining module's globals, never a copy bound into whoever imported it -- that's
what makes `formatting.py`'s module-level `COLOR`/`DIM`/`BOLD`/`RST` the single live source of truth
regardless of which command module calls into it.

**A real circular-import bug this caught, fixed:** the first cut had `engine_process.py`/`trace_io.py`/
every `commands/*.py` do `from clozn.cli import main as ctx` at MODULE level (mirroring the `ctx.SUB`
pattern from the Phase 9 server split literally). That is safe as long as `clozn.cli.main` happens to be
the first CLI submodule anyone imports -- which is true for every test file and for `python -m clozn` --
but breaks the moment anything imports a submodule directly and first, e.g. `from clozn.cli.engine_process
import find_engine` in isolation: `engine_process` starts loading, hits its `from clozn.cli import main as
ctx` line, which loads `main.py` fresh, which (at its own module level) does `from clozn.cli.engine_process
import _free_port` -- but `engine_process` is still mid-load, stuck at that very `ctx` import, so
`_free_port` doesn't exist on it yet: `ImportError: cannot import name '_free_port' from partially
initialized module`. Fixed by making every `ctx`/`CloznError` reference in `engine_process.py`,
`trace_io.py`, and every `commands/*.py` file a FUNCTION-LOCAL import (`from clozn.cli import main as ctx`
as the first line of each function body that needs `ctx.HOME`/`ctx.CloznError`, not a module-level
import) -- by the time any of these functions actually run, module loading has long finished, so the
import is always safe regardless of which submodule was entered first. Verified by directly importing
every new submodule in isolation (`import clozn.cli.engine_process`, `.trace_io`, `.formatting`,
`.commands.models`, `.commands.explain`, each on its own, fresh interpreter) as well as via `clozn.cli.main`
-- exactly the kind of gap unit-green tests don't catch (every test file already imports
`clozn.cli.main` first), caught only because the hard gate requires exercising the live import graph, not
just the test suite.

**A real pre-existing bug this caught and fixed:** `REPO` was computed as
`os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` (two levels up from the file) -- correct
for the OLD flat `clozn/cli.py`, but `cli.py` became `cli/main.py` in the original Phase 9 move (one
directory deeper) without updating this arithmetic, so `REPO` had silently resolved to `<repo>/clozn`
instead of `<repo>` ever since. This made `find_engine()` unable to find a real engine build under
`<repo>/engine/core` (always fell through to "no engine built", even with one present) and made
`_model_dirs()` scan a nonexistent `<repo>/clozn/models` instead of the real `<repo>/models`. Fixed
(now three `dirname` calls, in `engine_process.py`) and confirmed live: `ENGINE_CORE` now resolves to
the real `<repo>/engine/core` and `os.path.isdir(...)` on it returns `True`. Caught only because the hard
gate requires running `clozn models`/`clozn plan` against the real repo layout, not just importing the
module.

**Command -> module mapping:**

- `commands/models.py` -- `cmd_models`, `cmd_pull`, `cmd_plan` + model discovery (`resolve_model`,
  `_flags_for`, `_friendly`, `_model_dirs`, `_scan_models`, `KNOWN`, `PULLABLE`) and the fit-planner render
  (`format_plan`, `_fmt_ctx`, `_detect_vram_gb`).
- `commands/serve.py` -- `cmd_serve` + its two small daemon-registry companions `cmd_ps`/`cmd_stop`
  (grouped together rather than over-fragmented into three files, since all three share one registry).
- `commands/studio.py` -- `cmd_studio` (launch + health-poll + browser-open) and its helpers.
- `commands/run.py` -- `cmd_run`, the interactive REPL (`_repl`), one-turn execution (`_run_turn`), the
  AR/diffusion prompting plumbing (`stream_ar`, `complete_once`, chat templates), and the run-log side
  effect (`_log_run_cli`).
- `commands/explain.py` -- `cmd_explain` + `cmd_trace` + `cmd_branch` (folded together per the migration
  doc's own "(+ trace/branch/preferences if cohesive)" note -- all three are the run-inspection family),
  plus `format_explain`/`format_narrate` and their `_fetch_explain`/`_fetch_narrate`/`_verified_tag`/
  `_last_run_id` support.
- `commands/preferences.py` -- `cmd_preferences`, `format_preferences`, `_fetch_preferences`,
  `_resolve_preference` (kept separate from `explain.py`, matching the target tree's explicit listing).
- `commands/test.py` -- `cmd_test` (testkit invocation), `format_test_report`, `_load_test_spec`,
  `_fetch_live_receipt`.
- `formatting.py` -- the color globals (`DIM`/`BOLD`/`RST`/`COLOR`) + `_setup_console()` + every pure
  render helper (`_paint`, `_conf_rgb`, `_heatmap_lines`, `_conf_legend`, `_confbar`, `_sparkline`,
  `_paint_sparkline`, `_stream_token`, `_num`, `_as_list`, `_as_dict`, `_term_width`).
- `engine_process.py` -- `find_engine`, `_env_with_dlls`, `_launch_args`, `spawn_engine`, `_health`,
  `_free_port`, `_log_tail`, plus the warm-daemon registry (`_reg_read`/`_reg_write`/`_register`/`_kill`/
  `_unregister`/`_find_warm`) and the (now-fixed) `REPO`/`ENGINE_CORE`/`BUILDS` constants.
- `trace_io.py` -- CLI trace save/load (`_save_trace`, `_trace_cache_files`, `_runid`) and the
  runlog-journal -> terminal-render bridge (`_render_trace`, `_cmd_trace_legacy`, `_runlog_trace_steps`,
  `_runlog_trace_meta`, `_list_runlog_traces`, `_import_runlog`).
- `main.py` -- `build_parser()`, `main()`, the shared `HOME`/`CloznError`, and a block of stable re-exports
  (`_free_port`, `_save_trace`, `format_explain`, `_SPARK`, ...) purely so pre-split tests that call
  `clozn.cli.main.<name>` directly keep working -- safe because none of them are mutated globals, just
  functions/constants.

**Test compatibility:** four test files (`test_cli_color.py`, `test_cli_trace.py`, `test_testkit_cli.py`,
`test_narrate_cli.py`) needed edits, all mechanical and all about monkeypatch TARGETS, never call sites:
- `test_cli_color.py`'s `color_on`/`color_off` fixtures, `test_cli_trace.py`'s `isolated` fixture, and
  `test_testkit_cli.py`'s `iso` fixture patched `COLOR`/`DIM`/`BOLD`/`RST` on `clozn.cli.main` -- updated
  to patch `clozn.cli.formatting` instead, since that's where the real, live-read globals now live
  (patching the old location would silently no-op: the render functions never read it there anymore).
  `test_cli_trace.py`'s `HOME` patch needed NO change -- `HOME` stayed on `clozn.cli.main` exactly where
  the test already patches it.
- `test_narrate_cli.py`'s two tests proving `--why` is opt-in patched `_fetch_explain`/`_fetch_narrate` on
  `clozn.cli.main` -- updated to patch `clozn.cli.commands.explain` instead, since `cmd_explain` (which
  calls them by bare name) is defined there now, not in `main.py`; a patch on `main`'s re-exported copy
  wouldn't be observed by `cmd_explain`'s own call.
- `test_explain_cli.py`, `test_preferences_cli.py`, `test_fit_planner.py` needed NO changes: they only call
  functions directly (`format_explain(...)`, `build_parser()`, ...), never monkeypatch CLI internals, and
  every name they reference is still importable from `clozn.cli.main` via the re-export block.
- `test_engine_ctx_overflow.py` does `from clozn.cli import find_engine, _env_with_dlls` (the bare
  `clozn.cli` package, not `.main`) -- this import was ALREADY broken before this chip (confirmed with
  `python -c "from clozn.cli import find_engine"` on the pre-split tree: `ImportError`), is caught by a
  `try/except -> pytest.skip` inside a module-scoped fixture, and is gated behind `-m model` (not run in
  the default suite, hence invisible in the 1204/10 baseline). Left as-is rather than re-exported from
  `clozn/cli/__init__.py`: doing so would force the ENTIRE CLI module tree to load eagerly the moment
  ANYTHING imports `clozn.cli` (even `from clozn.cli import fit_planner`, which `test_fit_planner.py`
  does), which is both a bigger blast radius than this pre-existing, already-skipped gap warrants and its
  own fresh circular-import risk (package `__init__.py` would become a NEW earlier entry point than
  `main.py`). Not a regression -- verified identical (broken) behavior before and after this chip.

**Resulting file line counts** (`wc -l`):

```text
clozn/cli/main.py                  156   (was 1793)
clozn/cli/formatting.py             187
clozn/cli/engine_process.py         184
clozn/cli/trace_io.py               186
clozn/cli/commands/models.py        288
clozn/cli/commands/run.py           238
clozn/cli/commands/serve.py          88
clozn/cli/commands/studio.py        112
clozn/cli/commands/explain.py       375
clozn/cli/commands/preferences.py    89
clozn/cli/commands/test.py          120
```

No module exceeds 400 lines (target: none `>>400`); `main.py` shrank by 1637 lines (91%) down to the
argparse tree + dispatch + shared constants + the test-compat re-export block.

**Verification (green at every family boundary, not just at the end):**

```text
pytest -q tests
1204 passed, 10 skipped
```

`python -c "import clozn.cli.main"` clean; also confirmed each new submodule imports standalone, in
isolation, in a fresh interpreter (`import clozn.cli.engine_process`, `.trace_io`, `.formatting`,
`.commands.models`, `.commands.explain`) -- the specific check the circular-import bug above needed.
LIVE CLI run (no server, no GPU, no model load): `python -m clozn --help` lists all 13 subcommands;
`python -m clozn models` correctly reports "no engine built" (verified against the real, empty
`engine/core` -- no `build-*` directories exist on this machine, only the `build_*.bat` scripts) and
lists real local GGUFs (via the fixed `REPO`); `python -m clozn plan llama-1b` and `plan qwen-0.5b` read
real local GGUF headers and print correct FITS/size/layer verdicts; `python -m clozn ps` reports no
daemons; `python -m clozn trace` rendered a REAL prior run from the shared `~/.clozn/runs` journal (the
full confidence heatmap + per-token bars + "almost" alternatives); `python -m clozn stop all`,
`clozn explain --last` (Studio down), `clozn test does_not_exist.json`, and `clozn pull
totally-unknown-model-xyz` all produced clean, one-line `CloznError` messages (never a traceback) with the
correct exit codes (1, 1, 2, 1 respectively); `--help` on every subcommand
(run/serve/models/pull/plan/studio/ps/stop/trace/branch/explain/preferences/test) renders its full flag
list correctly. `clozn/__main__.py` still does `from clozn.cli.main import main` unchanged and works.

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

### Phase 10 — `clozn.cli.main` command extraction

- model discovery + `cmd_models`/`cmd_pull`/`cmd_plan`/`format_plan` -> `clozn/cli/commands/models.py`
- `cmd_serve`/`cmd_ps`/`cmd_stop` -> `clozn/cli/commands/serve.py`
- `cmd_studio` -> `clozn/cli/commands/studio.py`
- `cmd_run`/`stream_ar`/`complete_once`/chat templates/`_repl` -> `clozn/cli/commands/run.py`
- `cmd_explain`/`cmd_trace`/`cmd_branch`/`format_explain`/`format_narrate` -> `clozn/cli/commands/explain.py`
- `cmd_preferences`/`format_preferences` -> `clozn/cli/commands/preferences.py`
- `cmd_test`/`format_test_report` -> `clozn/cli/commands/test.py`
- color globals + paint/heatmap/sparkline helpers -> `clozn/cli/formatting.py`
- `find_engine`/DLL-on-PATH/spawn/health/warm-daemon registry -> `clozn/cli/engine_process.py`
- trace persistence + runlog-journal render bridge -> `clozn/cli/trace_io.py`
- `build_parser`/`main`/`HOME`/`CloznError` stay in `clozn/cli/main.py`

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

Phase 10 (`clozn.cli.main` split) reverified the full product suite after each command family AND at the end:

```text
pytest -q tests
1204 passed, 10 skipped
```

Also reverified clean: `python -c "import clozn.cli.main"`; each new submodule imported standalone in a
fresh interpreter (`import clozn.cli.engine_process` / `.trace_io` / `.formatting` / `.commands.models` /
`.commands.explain`, each on its own -- the specific check that caught the circular-import bug described
above); `clozn/__main__.py` unchanged (`from clozn.cli.main import main`) and confirmed working. LIVE CLI
(no server, no GPU, no model load): `python -m clozn --help` lists all 13 subcommands and every
subcommand's own `--help`; `python -m clozn models` (correct "no engine built" + real local GGUFs found,
post-REPO-fix); `python -m clozn plan llama-1b` / `plan qwen-0.5b` (real GGUF headers, correct FITS
verdicts); `python -m clozn ps`; `python -m clozn trace` (rendered a REAL run from `~/.clozn/runs`, full
heatmap); `python -m clozn stop all` / `explain --last` / `test does_not_exist.json` / `pull
totally-unknown-model-xyz` (all clean one-line `CloznError`s with correct exit codes, never a traceback).

## Known Broken Commands

None currently known through Phase 10. (A pre-existing, already-broken, `-m model`-gated import in
`tests/test_engine_ctx_overflow.py` -- `from clozn.cli import find_engine, _env_with_dlls` -- predates this
chip, is unaffected by it, and is documented in the Phase 10 section above rather than fixed, to avoid a
new circular-import risk; it is caught by a try/except inside the test's own fixture and does not affect
the default `pytest -q tests` baseline.)

## Remaining Import Errors

None currently known through Phase 10. Python sources no longer import `clozn.runlog`, `clozn.memory_cards`,
`clozn.memory_mode`, `clozn.facts_mode`, `clozn.topic_gate`, `clozn.slotmem_qwen`, `clozn.explain`,
`clozn.narrate`, `clozn.rederive`, `clozn.receipt_bundle`, `clozn.semantic_matcher`,
`clozn.counterfactual`, `clozn.timetravel`, the old flat readout/substrate modules, the old flat
steering module, `clozn.clozn_server`, or `clozn.fit_planner`. No flat `clozn/*.py` modules remain
except `clozn/__init__.py` and `clozn/__main__.py`. No flat `.py` files remain directly under `scripts/`
either -- only subdirectories (`bench/`, `calibration/`, `data/`, `smoke/`, `cleanup/`).

## Compatibility Policy

This is a breaking cleanup branch. Preserve `python -m clozn` and the meaningful CLI/server/product workflows, but do not preserve old internal flat imports such as `clozn.runlog`, `clozn.receipts`, or `clozn.clozn_server`.
