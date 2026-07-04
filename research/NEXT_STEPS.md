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

7. **Publish pass** — DONE (2026-07-03). `writeup_draft_receipts.md` rewritten to its final publishable
   form: folded in the 7B A/B (prompt≥prefix at 7B, INVERTS at 1.5B, single-seed caveats attached), the
   half-life-of-a-thought (<1 turn), phantom-KV (ghost slots work + coherence tax), the OBEY case
   (provenance beats plausibility), and the vector-telepathy closer (Procrustes rotation between
   independently-trained models, nulls collapsed — the authors' own impossibility claim killed by their
   own method). Every number traces to a findings file; single-family/single-seed qualifiers kept on
   each; the 1.5B/7B inversions are told as features of the story. Publishing-notes block updated
   (candidate titles, repro pointers, anticipated objections incl. "isn't this just gisting/LoRA/
   introspection-lit"). `FINDINGS.md` given a consistency pass (telepathy → laws #2/#5; time-travel
   SHIPPED → law #3; facts-tier wired + efficiency MEASURED → BUILT). Runs ~2.2k words prose (over the
   1400–1800 target because five experiments were folded in with nulls+caveats intact — honesty over
   the ceiling; a trim to hit 1800 exactly would drop a caveat or an experiment). Ready to post; let
   reality vote.

8. **LoRA voice at 7B** — DONE (2026-07-03). `peft` 0.19.1 installed into cloze/.venv. Rig
   `research/voice_lora.py` (QLoRA: nf4 base = studio's exact SelfTeach config + r=8 LoRA on attn+mlp,
   20.2M/0.462% trainable), coherence-GATED early stop keyed on FROZEN-BASE perplexity of 2 held-out
   generations (NOT train loss; train loss hit ~0 by step 50 and was useless), all scoring imported
   verbatim from voice_middle (zero drift), prior 4 arms CITED from `runs/voice_middle_qwen7b.json` (not
   rerun) and re-scored on the new coherence axis. **VERDICT: the LoRA captures the voice WITHOUT the
   coherence tax** — voice-dist **0.087** (BEST arm, beats cited few-shot 0.142 and prefix 0.158),
   **glitch=0 / role-leak=0** vs the prefix's **12 glitches + 1 leaked `assistant`**, lowest base-ppl of
   any voice arm (28.1 vs prefix 76.1). The `.orningside`-class boundary glitches VANISHED, exactly as
   predicted — they were a 16-vec-in-embedding-space artifact, not a "training a voice" artifact. The
   gate proved load-bearing (H5): held-out base-ppl climbs 16.2→27.5 past the step-50 optimum while train
   loss stays 0. **Two pre-registrations FLIPPED (reported loud):** (H3) the LoRA BEAT few-shot on
   fidelity — own-door advantage isn't just economics; (H4) the LoRA self-reported its terse voice
   ACCURATELY ("I'm slow... shorter answers"), the first process-artifact to do so across 4 scales — but
   this is almost certainly ENACTMENT (the self-report is itself in-voice, i.e. terse) not a break in
   process-blindness; wants an out-of-voice probe test. **The one real cost: bleed=10** (worst arm) —
   corpus imagery (coffee/beans/stones) leaks into unrelated topics; the LoRA paid down the COHERENCE tax
   fully but NOT the BLEED tax (few-shot stays cleanest on content). Findings + full caveats:
   `research/voice_lora_findings.md`; run `research/runs/voice_lora_qwen7b.json`. **Untested next arms:**
   the "50–200 examples" half of the constructive path (kept the same 12-reply corpus to isolate the
   container as the single variable); a second seed/voice/family; the out-of-voice self-report probe.

9. **Prompt-block strict variant for small models** — DONE (2026-07-03). The A/B found the studio
   block's soft wording under-fires at 1.5B (space/question inverted vs the trained prefix; the 7B
   verdict held). Added a `"strict"` block variant — the same cards as a direct imperative (*"Follow
   these facts and rules about the user exactly, in every reply, without exception:"*, no
   "naturally"/"tailor" hedge) — selected by a persisted `block_style` setting (default `soft`, so no
   behaviour change for anyone who hasn't opted in), wired through `memory_mode.get/set_block_style` +
   `compile_prompt_block(texts, style=None)` (signature back-compatible: `None` reads the setting, every
   existing call site is untouched and now honors it). **RE-RAN the gated 1.5B A/B with strict on the two
   inverted cells** (seed 0, steps 80, same probes; artifact
   `research/runs/prompt_vs_prefix_ab_1p5b_strict.json`): **strict CLOSES the inversion on both** —
   space +0.166→**+0.500 d_prompt** (PREFIX-stronger → **PARITY**, now = prefix), question
   +0.167→**+0.667** (PREFIX-stronger → **PROMPT-stronger**, now beats prefix). Clean controlled A/B:
   baseline + prefix arm reproduced byte-identically across soft/strict (prefix doesn't see the block),
   so the block wording is the only moving part; coherence-eyeballed (the question win is short, clean,
   natural questions — NOT the soft-run prefix's degeneration). The 1.5B inversion was a soft-wording
   under-compliance artifact, not a capacity ceiling. Files: `memory_mode.py` (+ `BLOCK_STYLES`,
   getters), `test_memory_mode.py` (14 new model-free tests), `test_prompt_vs_prefix_ab.py` (`block_style`
   param + gated strict re-run), `self_audit_gap_findings.md` (Follow-up 4b table + verdict), a one-line
   honest note in `inspector/demo/pages/memory.js` (block_style exists, not yet a UI toggle). **Not done
   (deliberate):** strict left OFF at 7B (soft holds there; a direct imperative risks over-firing on a
   strong instruction-follower — strict is the *small-model* wording); no UI toggle exposed (server/config
   only per the task's scope); strict's off-topic over-bleed not stress-tested (the topic gate should omit
   the block, untested interaction).

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

11. **"Explain this answer" — the inspect-a-reply core surface** — spec at
    `research/EXPLAIN_THIS_ANSWER_SPEC.md` (5 milestones, honesty invariants, cost model). The
    mainstream front door: measured confidence/influence/concepts/counterfactuals on any reply, never
    self-reported. Mostly assembly of existing parts (runlog manifest + trace, replay.py ablation,
    run.js receipts, SAE readouts). Order: M1 free-signal `/runs/<id>/explain` (Sonnet) → M2 rigorous
    greedy-both-arms ablation + redundancy guard (Sonnet) → M3 counterfactual dials → M4 accountable-self
    narration + confabulation-diff (Opus, honesty-critical) → M5 TUI + any-client bridge. The trap
    (do-not-build): a plain "explain" that asks the model to explain itself = the confabulation machine.
