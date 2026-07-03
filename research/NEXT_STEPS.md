# NEXT_STEPS — actionable work for downstream (Opus/Sonnet) sessions

*2026-07-03. Ordered by leverage. Each item: what/why, where, done-criteria, suggested model tier.
House rules for every session: read the memory files first (they are current), pre-register, smoke
before full runs, eyeball before believing metrics, nulls + coherence axis on every receipt, commit
at seams with the session trailer convention, never set explicit timeouts on long background runs.
Environment: CUDA python = C:\Users\brigi\src\cloze\.venv\Scripts\python.exe; engine boot
`python clozn_cli.py serve llama-1b --port 8080`; studio `clozn studio` (needs CLOZN_STUDIO_PYTHON).*

1. **Provenance on memory candidates** (the OBEY defense) — Sonnet, ~half day. Every proposed card
   gets `source_run_id` + `source_turn` + the quoted span; the Memory page shows "you said this" with
   a link; candidates WITHOUT provenance are flagged, never auto-approvable. Files:
   `research/clozn_server.py` (propose paths), `memory_cards.py` (fields exist partially),
   `inspector/demo/pages/memory.js`. Done: a fabricated candidate visibly lacks provenance; tests in
   `research/tests/`. Why: `dream_consolidation_findings.md` — plausibility gates pass fabrications.

2. **Finish the two stalled Fable runs** — Sonnet, hours. (a) Telepathy: rig exists UNTRACKED at
   `research/vector_telepathy.py` (pre-registration in header) — commit, smoke, run, write
   `telepathy_findings.md`. (b) Efficiency: instruments committed (`bench_whitebox_tax.py`,
   `bench_batched_receipts.py`, plan in commit 676af08 + c25ee33) — run the GPU phases, write
   `local_efficiency_findings.md`. NOTE: the "<3GB" GPU gate is unsatisfiable (WDDM floor ~4.8GB);
   use util<15% sustained instead (documented precedent). Done: findings committed with tables.

3. **Restart :8080 + binary cleanup** — any tier, minutes. The live engine still runs renamed
   `cloze-server-live8080.exe`; restart onto fresh binaries (now SAE-capable), delete `-live8080`
   files. Then the 5-line `clozn_cli.py serve --sae` passthrough so `--sae ~/.clozn/sae/andyrdt_l15`
   works from the CLI (only meaningful for the qwen GGUF — dims must match; server refuses politely).

4. **Profiles studio UI** — Sonnet/Opus, ~1 day. Persona picker in the masthead (the subchip shows
   the profile letter), backed by `research/profiles.py` (tested) + `/memory/mode` prompt path.
   Switch = `profiles.switch()` semantics server-side: new endpoint `/profiles/*` (list/save/switch/
   export/import). Done: two personas with disjoint cards/dials/facts switch instantly in the UI;
   model-free tests. Spec fragments: profiles.py docstring + MEMORY_MODE_SWAP_SPEC.md.

5. **Slot-store studio wiring (`memory_mode:"slots"` facts tier)** — DONE (2026-07-03). Per-profile
   stores (`~/.clozn/profiles/<name>.slots.pt`), surprise-gated auto-writes, a Facts panel
   (list/delete = surgical, gate-refusal + abstentions visible), and profile-switch fact compilation
   (the `facts_note` seam is closed) are all wired; `SlotMem.from_shared` reuses the studio's 7B (no
   second load). Gated behind `memory_facts` (default OFF — the latency rule); measured overhead
   **~86 ms/turn** (a slot read ~171 ms vs ~85 ms baseline forward on the real nf4), logged as
   `slot_ms`. `facts_mode.py` + `SlotBox`/`/facts/*` in clozn_server.py + memory.js; tests
   `test_facts_mode.py` / `test_facts_server.py` / `test_slotmem_shared.py`. **Remaining rung:** v1's
   slot read produces a RECEIPT only — it does NOT yet inject the retrieved value into the chat reply
   (the read machinery + receipts are the foundation for that next step). See the "STUDIO WIRING"
   section of `slotmem_qwen_findings.md`.

6. **Time-travel debugger feature** — DONE (2026-07-03). Product: a bounded, CPU-offloaded per-turn KV
   snapshot ring (`timetravel.SnapshotStore`: per-run "last N turns" cap + a hard byte budget, evict-
   oldest, payload freed on eviction; honest byte accounting), a "rewind & branch from here" affordance
   in the Run Inspector (a turn picker + optional alt-user field → `POST /runs/<id>/branch` →
   `renderCompare`, following the doReplay pattern), and branches recorded as CHILD runs via runlog
   (`parent_run_id` + `changes_applied` = `{branch_turn, edited_user, alt_user?, kv_snapshot}`). State-
   surgery SKIPPED per the findings (half-life <1 turn; Lab only). **Determinism receipt PROVEN**: the
   gated `test_timetravel_determinism.py` (`-m model`, ran on the real 1.5B) shows a branch restored
   from the STORE's CPU-offloaded KV snapshot byte-matches a fresh full recompute (identical=True, 33
   tok), re-prefilling only the 27-tok alt-user turn vs the 147-tok shared prefix. **Memory measured**:
   the snapshot ring is behind `timetravel_snapshots` (default OFF — the RAM rule) because a KV
   snapshot costs CPU RAM: **~7 MB / 128 tok on Qwen2.5-7B nf4** (bf16 KV: 28 layers × 4 KV heads × 128
   × 2 B), so last-8 at ~512 tok ≈ **224 MB** (1.5B is half: 4.02 MB measured for the 147-tok rewind
   point). v1 registers a DESCRIPTOR-only snapshot per studio turn (turn idx + token count, zero bytes)
   because the studio chat is STATELESS — `SelfTeach._generate` builds its own cache via `generate()`
   and discards it. Branch RECORDING (transcript truncate → child run) does NOT depend on the gate and
   works today; a branch keeps the live memory/dials (toggle via the replay buttons). Files:
   `timetravel.py` + `SnapshotStore`/`_snap_store`/`_timetravel`/`_maybe_snapshot_turn` in
   clozn_server.py (`/runs/<id>/branch`, `/timetravel/mode|stats`) + the branch UI in run.js; tests
   `test_timetravel.py` (37) / `test_timetravel_server.py` (21) / `test_timetravel_determinism.py`
   (gated). **Remaining rung:** the KV fast path — actually RESTORE a stored snapshot to skip the
   shared-prefix re-prefill on a studio branch — needs the generation path (`SelfTeach._generate`) to
   hand back its `past_key_values`; v1's studio branch re-generates from the truncated transcript
   (correct, and exactly what every stateless turn already costs). The mechanism (offload → restore →
   byte-exact continue) is proven; the studio wiring to reuse it is the deferred seam.

7. **Publish pass** — Opus (or remaining Fable), half day. `writeup_draft_receipts.md` + fold in:
   the 7B A/B (prompt≥prefix at 7B, inverts at 1.5B), the half-life number, phantom-KV, OBEY.
   `FINDINGS.md` is the skeleton. Post; let reality vote.

8. **LoRA voice at 7B** — Opus, 1 day. `pip install peft` (allowed for this). Re-run voice_middle's
   own-door with a LoRA instead of the 16-vector prefix, coherence-GATED early stopping (stop on
   base-model perplexity of outputs, not train loss), receipts incl. coherence axis. Why:
   `voice_middle_findings.md` constructive path; boundary glitches should vanish.

9. **Prompt-block strict variant for small models** — Sonnet, hours. The A/B found the studio block's
   soft wording under-fires at 1.5B (inverts the 7B verdict). Add a "strict" block variant (raw rule
   as system message) selected by model size or a setting; re-run the gated A/B at 1.5B. Files:
   `memory_mode.py`, `test_prompt_vs_prefix_ab.py`.

10. **SAE encoder polish** — DONE, partial (2026-07-03). Both pre-located optimizations landed and
    torch-parity stayed green throughout (`ctest -R sae_encoder`: max|d|=0.0078, max rel=0.00242, 0
    gate flips, top-k overlap 1.000 — identical to pre-change numbers at every step; full
    `ctest --test-dir build-cuda`: 12/12). (a) `encode_jumprelu_kernel`'s GEMV loads vectorized from
    scalar `__half` to 8-wide (`float4`-shaped, 16 B/lane) loads unpacked via `__half22float2`
    (`engine/core/src/sae_encoder.cu`) — the GEMV-ONLY cost (isolated via `encode_dense`, which skips
    `sae_topk`) dropped to **~1.3ms**, squarely hitting the ~2ms target and right at the ~1.0-1.3ms
    bandwidth-floor estimate. (b) `sae_topk`'s per-call `cudaMalloc`+forced-sync+`cudaFree` scratch
    hoisted into `SaeEncoder::Impl`'s existing grow-only workspace (`kernels/sae_topk/sae_topk.cuh`/
    `.cu` gained an optional `picked_scratch` param, default nullptr so `validate.cu` and
    `test_sae_topk.cu` are untouched). **Honest gap**: the alloc hoist alone barely moved the
    steady-state `encode_topk` TOTAL (8 rows: 9.1-9.6ms → 6.1-6.5ms after (a)+(b) combined, not the
    ~2ms hoped for) — the GEMV/topk split reveals why: with (a)'s ~1.3ms GEMV, the remaining ~4.9ms is
    `sae_topk_kernel`'s OWN compute (32 rounds of an O(131072)-wide masked-argmax reduction per row +
    an O(k²) insertion sort), a real cost the alloc hoist can't touch — a genuinely separate, larger
    redesign (block radix-select, per the kernel's own comment) than item 10b scoped. End-to-end on a
    rebuilt `--sae` server (`:8081`, isolated `build-sae-server/` — NEVER `build-gpu`/`:8080`):
    `feat-protocol` vs `plain` on the SAME process held at **75-77%** across repeated runs (up from
    the pre-optimization measured **61.2%** in `local_efficiency_findings.md`) — a real ~14-16 point
    recovery of the 37-point tax, though this comparison ran under heavier ambient GPU contention
    (multiple resident models) than the original isolated measurement, so treat it as suggestive
    corroboration alongside the cleaner unit-level numbers above, not a like-for-like replication.
    **Not done**: the optional `--sae-every N` throttle (out of scope for this pass; the two scoped
    optimizations plus honest measurement took the full session).

Parked ideas with rigs/specs: `WILD_EXPERIMENTS.md` (10 pre-designed experiments), phantom-KV
coherence problem (Lab), diffusion dreaming (killed — provenance extraction instead). The findings
map: `FINDINGS.md`. The state of everything: the three memory files.
