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

6. **Time-travel debugger feature** — Opus, 1-2 days. The rig proved byte-exact checkpoint/branch
   (`kv_timetravel_findings.md`). Product: per-turn KV snapshots in the studio chat (CPU offload),
   a "rewind & branch from here" affordance in the Run Inspector, branches recorded as child runs.
   Skip state-surgery in v1 (half-life <1 turn = not a memory mechanism; Lab only).

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

10. **SAE encoder polish** — Sonnet, hours. The two pre-located optimizations (vectorized GEMV loads
    ~9ms→~2ms; hoist sae_topk's per-call cudaMalloc into the workspace), parity test must stay green
    (`ctest -R sae_encoder`). Then `--sae-every N` throttle if wanted.

Parked ideas with rigs/specs: `WILD_EXPERIMENTS.md` (10 pre-designed experiments), phantom-KV
coherence problem (Lab), diffusion dreaming (killed — provenance extraction instead). The findings
map: `FINDINGS.md`. The state of everything: the three memory files.
