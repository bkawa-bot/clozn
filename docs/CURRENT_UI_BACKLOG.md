# Current UI Backlog

Evidence-backed backlog refresh against the current working tree on 2026-07-07.

Status legend used here:
- `DONE`
- `PARTIAL`
- `STILL NEEDED`
- `OBSOLETE`
- `NEEDS RENAMING / NARROWING`

Audit verdicts in the second column are copied from `research/UI_SCOPE_AUDIT.md`.

| Item | Previous audit verdict | Current repo status | Evidence: file/function/line if possible | Recommended action | Suggested Codex task name |
|---|---|---|---|---|---|
| True `finish_reason` is captured onto persisted runs | `QUICK WIN` | `PARTIAL` | `research/runlog.py:150-171` `finish_reason_from_frames()` maps engine frames to `stop`/`length`; `research/clozn_server.py:1816-1822`, `2612-2619` passes `SUB.last_finish_reason()` into `_log_run()` but still falls back to `"stop"` in OpenAI responses when `fr` is missing | Keep the real path; remove the last `"stop"` fallback once every substrate guarantees `last_finish_reason()` | `backlog/finish-reason-end-to-end` |
| Run metadata persisted on run records (`quant`/mode/sampler/etc.) | `QUICK WIN` | `PARTIAL` | `research/clozn_server.py:1569-1592` `EngineSubstrate.run_meta()` persists `model_file`, `quant`, `mode`, `sampling`, optional `n_ctx`/`device`/`gpu_layers`; `research/clozn_server.py:1958-1977` records that into `runlog`; `research/runlog.py:250` stores `meta` | Add missing repro fields the audit asked for: `seed`, explicit sampler params, and make `n_ctx`/`device` reliable from `/health` | `backlog/repro-metadata-complete` |
| OpenAI-compatible responses use the real `finish_reason` | `QUICK WIN` | `PARTIAL` | `research/clozn_server.py:1790-1818` streamed chunks use `finish=fr`; `research/clozn_server.py:2616-2619` non-stream response uses `fr or "stop"` | Add regression tests for `length` and remove fallback-only behavior where possible | `backlog/openai-finish-reason-tests` |
| Persisted token trace includes the current shipped fields | `SHIPPED` | `DONE` | `research/runlog.py:86-111` `steps_to_trace()` stores `tokens`, `confidence`, `alternatives`; `research/runlog.py:114-147` `accumulate_ar_events()` folds `tokens_committed` + `step_lens`; `inspector/demo/pages/run.js:277-323` renders the same trace shape | Keep as the stable v1 trace contract | `backlog/trace-contract-lock` |
| Persisted token trace includes `token_id`, chosen-token `prob`/`logprob`, full top-k, entropy, token index, wall-clock timing | `SHIPPED` + `NEEDS A HOOK` | `PARTIAL` | `research/runlog.py:70` `TRACE_KEYS` keeps only `tokens`, `confidence`, `alternatives`, `workspace_readouts`; `research/runlog.py:89-110` step schema is only `{piece, conf, alts}`; `research/run_timeline.py:112-140` derives token order from array index, not stored token metadata | Extend the stored step schema before normalization, then thread those fields through UI/CLI consumers | `backlog/rich-token-trace-schema` |
| `t` is a real clock rather than just an ordinal/counter | `NEEDS A HOOK` | `STILL NEEDED` | `inspector/demo/workspace_lens_trace.jsonl:1-10` uses `t` as event ordinal; `research/run_timeline.py:104-115` only exposes run-level `duration_ms`, not per-token timing; no persisted per-token timestamp field exists in `research/runlog.py` | Add a per-token timing tap in generation and store explicit `wall_ms` or `dt_ms` per token | `backlog/per-token-timing` |
| `clozn trace` reads the same persisted run records as Studio | `not explicit in audit` | `STILL NEEDED` | `clozn_cli.py:422-428` writes `~/.clozn/traces/*.json`; `clozn_cli.py:880-919` `cmd_trace()` reads that trace cache; Studio reads `~/.clozn/runs` via `research/runlog.py:270-282` and `research/clozn_server.py:2117,2153-2156` | Point `clozn trace` at `runlog` or formally document it as a separate legacy cache | `backlog/unify-cli-trace-with-runlog` |
| Replay persists `trace_out` on child runs | `QUICK WIN` | `DONE` | `research/replay.py:152-161` captures `trace_steps`; `research/replay.py:241-256` persists them via `runlog.record(trace=trace_steps)` | None beyond keeping tests around this seam | `backlog/replay-trace-regression` |
| Token-probability replay diff exists | `QUICK WIN` | `STILL NEEDED` | `research/replay.py:152-161` now captures the child trace, but `inspector/demo/pages/run.js:742-784` compare UI still uses text-level receipt metrics only; no token-prob diff renderer is present | Build a compare view over parent/child trace arrays | `backlog/token-prob-replay-diff` |
| Branch lineage is rendered as a tree | `QUICK WIN` | `PARTIAL` | `research/runlog.py:249` stores `parent_run_id`; `research/run_timeline.py:160-163` emits a flat `branched_from` event; no tree renderer is present in `inspector/demo/pages/run.js:1246-1289` | Add lineage aggregation on the Runs page or Run Inspector | `backlog/branch-lineage-tree` |
| Concept / SAE / probe readouts are persisted on run records | `NEEDS A HOOK (small)` | `PARTIAL` | `research/clozn_server.py:1840-1867` builds a live provider from `concepts_from_engine()` / `concepts_only()`; `research/runlog.py:199-220` attaches `trace.workspace_readouts`; but `research/explain.py:163-189` still reports concept spans unavailable on runs | Decide whether `workspace_readouts` is the long-term persisted shape, or also persist raw concept spans/features | `backlog/persist-concept-spans` |
| `causal_verified` stays `null` unless a real receipt verified it | `SHIPPED` / `LEAN-IN` | `DONE` | `research/explain.py:14-27`, `120`, `150` set `causal_verified: None` on M1 influences; `research/receipts.py:225-242` only flips to boolean on measured ablation receipts; `protocol/SPEC.md:19,28` defines the invariant | Keep the invariant and test it whenever new readout types land | `backlog/causal-verified-invariant` |
| There is an event shape for `workspace_readout` / `concept_readout` | `NEEDS A HOOK (small)` | `PARTIAL` | `protocol/SPEC.md:45` defines `workspace_readout`; `research/workspace_lens.py:125-160` emits it; `research/runlog.py:185-220` persists it; there is no `concept_readout` event shape in the protocol or run schema | Either standardize on `workspace_readout` as the adapter shape or add a distinct `concept_readout` event intentionally | `backlog/readout-event-taxonomy` |
| Memory contacts are persisted on runs | `SHIPPED` | `DONE` | `research/clozn_server.py:1885-1903`, `1912-1930` writes `memory.cards_applied`, `applied_ids`, `mode`, `gate`, `strength`; `inspector/demo/pages/run.js:395-430` renders them in the Influence column | Keep this run-level memory manifest as the source of truth | `backlog/memory-contact-contract` |
| Per-card relevance / cosine is persisted and displayed | `NEEDS A HOOK (small)` | `PARTIAL` | Persistence: `research/clozn_server.py:261-268`, `302-305`, `1899-1901`; Markdown export renders it in `research/clozn_server.py:359-375`; Run Inspector UI does not display relevance in `inspector/demo/pages/run.js:400-406` | Surface persisted `memory.relevance` in the Run Inspector, not just exports | `backlog/show-memory-relevance` |
| Final assembled prompt is captured in prompt mode | `NEEDS A HOOK` | `STILL NEEDED` | Prompt block is assembled transiently in `research/clozn_server.py:1237-1241`, `1308-1312`, `2550-2554`; `research/runlog.py:246-250` stores original `messages` plus memory manifest, not the final injected prompt | Persist an explicit `final_prompt` / `assembled_messages` field for prompt-mode turns | `backlog/capture-final-prompt` |
| Internalized / soft-prefix memory is represented honestly | `SHIPPED` / `LEAN-IN` | `DONE` | `research/memory_mode.py:5-14` and `135-165` distinguish prompt block vs internalized prefix; `inspector/demo/pages/memory.js:667-708` says internalized is a trained soft prefix and not self-reportable; `inspector/demo/pages/run.js:401-410`, `424-425` avoids pretending per-card receipts work in internalized mode | Keep the wording honest and mode-specific | `backlog/memory-mode-copy-lock` |
| `engine.html` and `brain.html` are reachable from Studio | `NEEDS A HOOK` | `NEEDS RENAMING / NARROWING` | `inspector/demo/studio.html:81-88` exposes only the Studio chrome plus an `all windows` link; direct entry points live in `index.html:57-68,86-151` and `clozn_cli.py:633-638` | Narrow this backlog item to “add Studio-native entry points to brain/runtime pages” instead of claiming they are unreachable globally | `backlog/studio-lab-entry-points` |
| There is a Lab Mode toggle/tab inside Studio | `NEEDS A HOOK` | `STILL NEEDED` | `inspector/demo/studio.html` has no Lab tab/toggle; `inspector/demo/pages/run.js:1261-1267` only exposes `Detail` and `Explain`; capture tiers exist server-side in `research/capture_mode.py:5-24` and `research/clozn_server.py:2098-2100` but not in the UI | Add a Studio-visible Lab entry point or capture-tier control | `backlog/studio-lab-mode` |
| Layer summaries / norms / heatmap reductions are implemented where users can reach them | `QUICK WIN` | `PARTIAL` | `inspector/demo/engine.html:84-91` renders per-token residual norms from `/engine/harvest`; `research/clozn_server.py:2482-2484` exposes `/engine/layers`; these are not wired into Studio/Run Inspector | Decide whether this stays an engine tool or becomes a Studio MRI panel | `backlog/model-mri-surface` |
| JSON export exists | `QUICK WIN` | `DONE` | `research/clozn_server.py:2118-2135` serves `/runs/<id>/export?format=json` with `{"run": run, "explain": xr}` | Keep endpoint and add tests only if export schema changes | `backlog/export-json-regression` |
| Markdown export exists | `QUICK WIN` | `DONE` | `research/clozn_server.py:333-380` `_export_markdown()` builds the receipt; `research/clozn_server.py:2130-2133` serves `format=md` | Keep endpoint and add tests only if receipt format changes | `backlog/export-markdown-regression` |
| There is a unified receipt / repro metadata object | `NEEDS A HOOK` | `STILL NEEDED` | Run repro metadata is split across `run.meta` in `research/runlog.py:250` / `research/clozn_server.py:1958-1977`; receipt objects are separate in `research/receipts.py:225-242`; export bundles `run` + `explain` only in `research/clozn_server.py:2134-2135` | Define one export-level object that groups run, repro metadata, explain, receipts, and future tiny tests | `backlog/unified-receipt-bundle` |
| Tiny-test harness exists | `NEEDS A HOOK` | `STILL NEEDED` | There is no product tiny-test endpoint or schema in `research/clozn_server.py`; existing tests are repo tests (`research/tests/test_receipts.py`, `research/test_studio.py`) rather than user-authored run-level checks | Add a minimal run-level assertion harness on top of the existing receipt/replay seams | `backlog/tiny-test-harness` |
| Top-level README reflects the strongest current demo | `not explicit in audit` | `PARTIAL` | `README.md:3-6,34-39` still centers generic memory/state tracing and CLI trace; it does not lead with the stronger current Studio/receipt/white-box demo surfaced in `research/README.md:8-30` and `index.html:57-151` | Refresh top-level positioning around the actual shipped Studio + receipts + engine/brain surfaces | `backlog/readme-repositioning` |
| Docs avoid overclaiming J-Space / chain-of-thought | `not explicit in audit` | `DONE` | No J-Space or chain-of-thought claims in `README.md:1-58`; `docs/WORKSPACE_LENS.md:26-28` explicitly says future adapters may include Jacobian Lens and not to label providers as J-Space/Jacobian Lens until they exist | Keep that wording discipline | `backlog/no-overclaim-copy-guard` |
| Docs distinguish live product vs research spikes | `not explicit in audit` | `PARTIAL` | `research/README.md:3-6,8-30,33-39` clearly separates live Studio backend from one-off research; top-level `README.md:45-58` still presents the repo more generically and does not make that split explicit | Mirror the live-vs-research split in the top-level repo entry points | `backlog/live-vs-research-doc-split` |

## Top 10 Remaining Highest-Leverage Tasks

1. `backlog/repro-metadata-complete`  
   Persist full reproducibility metadata: `seed`, reliable `n_ctx`, `device`, sampler params, and build/runtime identifiers.

2. `backlog/rich-token-trace-schema`  
   Extend stored token traces with `token_id`, chosen-token `prob`/`logprob`, token index, entropy, and richer top-k.

3. `backlog/unify-cli-trace-with-runlog`  
   Make `clozn trace` inspect the same run journal as Studio instead of a separate `~/.clozn/traces` cache.

4. `backlog/token-prob-replay-diff`  
   Use the now-persisted replay child trace to render token-level probability diffs.

5. `backlog/branch-lineage-tree`  
   Turn `parent_run_id` from flat bookkeeping into a real lineage tree in the Runs/Inspector UI.

6. `backlog/persist-concept-spans`  
   Decide whether raw concept/SAE spans should join persisted runs alongside `workspace_readout`.

7. `backlog/show-memory-relevance`  
   Surface already-persisted per-card cosine/relevance in the Run Inspector.

8. `backlog/capture-final-prompt`  
   Persist the actual assembled prompt / injected system block for prompt-mode runs.

9. `backlog/studio-lab-mode`  
   Add a Studio-native Lab entry point or toggle for brain/runtime/MRI surfaces and capture tiers.

10. `backlog/unified-receipt-bundle`  
    Define one export bundle for run data, repro metadata, explain output, receipts, and future tiny tests.
