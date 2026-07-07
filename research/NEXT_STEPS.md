# NEXT_STEPS — actionable work for downstream (Opus/Sonnet) sessions

*2026-07-03. Ordered by leverage. Each item: what/why, where, done-criteria, suggested model tier.
House rules for every session: read the memory files first (they are current), pre-register, smoke
before full runs, eyeball before believing metrics, nulls + coherence axis on every receipt, commit
at seams with the session trailer convention, never set explicit timeouts on long background runs.
Environment: CUDA python = C:\Users\brigi\src\cloze\.venv\Scripts\python.exe; engine boot
`python clozn_cli.py serve llama-1b --port 8080`; studio `clozn studio` (needs CLOZN_STUDIO_PYTHON).*

---

## 2026-07-07 — runtime bet LANDED (READ THIS FIRST; supersedes the sections below where they conflict)

*Session 2026-07-06→07 built + validated the engine convergence (the ⭐ North Star of the 2026-07-06 list)
and finished shipping the dial library — its two flagship items are DONE. The living record of the runtime
work is `RUNTIME_SPLIT.md`: the "what lives where" definition, the phased migration, and the engine-crash
forensics. Research still pinned in `FINDINGS.md`.*

**✅ DONE — engine convergence (the North Star; "biggest structural lever").** `EngineSubstrate` in
`research/clozn_server.py` implements the `Substrate` interface over the C++ engine, so chat + the whole
receipts stack (explain/receipts/counterfactual/narrate/replay) + prompt-mode RAG memory + all 33 tone
dials run on the GGUF runtime with ZERO PyTorch model resident. The clean split (`RUNTIME_SPLIT.md`):
**product = engine (forward-only), lab = PyTorch (offline trainer/research)**, joined by forward-only
artifacts; the learned soft-prefix TTT is CUT from the product (kept lab-only) so the product is 100%
forward-only. Validated **12/12** by `research/smoke_engine_substrate.py` (a full session survives).

**✅ DONE — validated dial library, deployed + engine-native.** 71→47→33 curated
(`research/dial_library_shipped.json`), per-model-calibrated. Deployed live (`deploy_dial_library.py` →
`~/.clozn/studio_library.json`; `gen_dial_calibration.py` → `~/.clozn/dial_calibration.json`; tagged
"library" in `/steer/axes`). Works on BOTH substrates (PyTorch + engine via `EngineSteer.load_library`).
Findings: `research/dial_library_findings.md`.

**🔧 FIXED — the engine's #1 robustness blocker.** Uncaught exceptions on the streaming SSE path silently
`abort()`ed the single-worker engine (crashed a normal session ~7 generations in). Root-caused + fixed in
`serve/cloze_server.cpp` (clamp inputs + restore-and-rethrow + a streaming try/catch). Remaining C++
follow-ups (lower priority now sessions are stable): the same guard on `/v1/revise`, `/v1/board`,
`/intervene`; `/intervene`'s dangling `&raw_vec`/`&applied_layers` captures; the mid-stream-disconnect
guard. See `RUNTIME_SPLIT.md` hard-part #6.

**➡️ OPEN WORK, re-ranked by leverage:**

1. **🔑 Preference-signal plumbing (thumbs / regenerate / edits).** STILL the highest-leverage single
   input — unblocks dial auto-SELECT (pick the value, not just calibrate the range) AND voice/LoRA "learn
   how you want me to act." Where: chat UI + a `/feedback` seam + the run log. Done: a thumbs/edit persists
   a signal the calibration/learning loops can read. Model: Opus (design) + Sonnet (wiring).

2. **➕ Auto-calibrate custom dials on creation — now engine-native + timely.** The "make your own dial" UI
   runs the calibration engine live → "works (range X)" or "doesn't steer on your model." `EngineSteer`
   already computes directions via `/harvest`, so this runs on the engine with no PyTorch. Extra pull: this
   session found **calibration is SUBSTRATE-specific** (the engine's steer scale ≠ PyTorch's — a dial value
   means different things per substrate), so a per-substrate calibrate-on-the-engine flow is genuinely
   needed (`RUNTIME_SPLIT.md` Phase 5). Where: refactor the calibration core to run against the live
   substrate; give `EngineSteer` an `add_custom`. Model: Sonnet.

3. **📋 Profiles studio UI** — unchanged; contained ~1-day product win (backend `research/profiles.py`
   tested, no front-end). Persona picker in the masthead, `/profiles/*` endpoints. Model: Sonnet/Opus.

4. **Runtime polish (`RUNTIME_SPLIT.md` Phases 3–5):** streaming `chat_stream` on the engine (live-token
   UX — the OpenAI streaming endpoint falls through to non-stream today); the C++ hardening follow-ups
   above; engine supervision (auto-restart) + studio retry-on-connection-refused (defense-in-depth);
   engine-native dial calibration (Phase 5).

**🕗 DEFERRED / ❌ KILLED — unchanged from 2026-07-06:** memory hardening (single-user threat model), slot
value-injection, persistent-injection KV-edit bug, receipt-tuned wording deferred; Parliament + Quine killed.

---

## 2026-07-06 — decision-updated production list (mostly DONE — superseded by the 2026-07-07 section above; supersedes items 1–11 below)

*Session 2026-07-05→06 closed all six wild experiments, made the product-scope calls below, and built the
dial auto-calibration feature. Items 1–11 (dated 2026-07-03) are mostly DONE or superseded — this section
is the current truth. Research is fully pinned in `FINDINGS.md`.*

**🔨 ACTIVE — dial auto-calibration → a validated dial library.** The calibration engine
(`research/dial_autocalibrate.py`) is built + validated: DIRECTION-AWARE effect (projects each reply onto
the dial's own diff-of-means axis — NOT mere change-magnitude, which falsely passed reformatting dials),
coherence-gated, shuffled-direction null. A ~71-dial candidate library
(`research/dial_library_candidates.json`, 15 categories, hypothesis-tagged) is being swept to find which
dials genuinely steer on the model. Studio wiring DONE: `/steer/axes` serves each dial's calibrated range,
`behavior.js` caps sliders + greys dead dials with "no measurable effect on this model." REMAINING: run
`--report` to curate → drop `dial_library_curated.json` at `~/.clozn/dial_calibration.json` → ship.
Emerging finding: **rich positive-signature dials steer** (distinctive voices, social-directness, affect);
**absence/default dials go dead** (plainspoken, prose, punchy — can't steer toward an absence); **epistemic
stances are fragile** (skeptical flips live↔dead between runs). Ship a *measured* library, not a guessed one.

**➕ NEXT BUILDS (natural extensions):**
- **Auto-calibrate custom dials on creation** — the "make your own dial" UI calls the calibration engine and
  tells the user "works (range X)" or "doesn't steer well on your model — try different poles." Turns the
  engine into a live feature. Needs the calibration core refactored to run against the LIVE substrate.
- **Expand the winning dial categories** to fill the shipped library (map-then-expand).

**⭐ NORTH STAR — substrate-agnostic studio server (the real engine convergence; biggest structural lever).**
Features (memory/dials/receipts) only drive the PyTorch substrate today; implement the full `Substrate`
interface for the C++ engine so the same feature code runs on the fast runtime. Roadmap Phases 1–2,
ENGINEERING not research — the primitives exist (engine seam mapped this session). This is what collapses
the PyTorch/engine dual system. Clean split: **engine = online runtime, PyTorch = offline lab/trainer**
(gradients, calibration). Engine-frontier-AS-SLOT-MEMORY is **SHELVED** (thin payoff + ~86ms/turn latency
tax even ported) — convergence is the receipt/legibility surface + existing features, not a new kernel.

**🔑 KEY ENABLER — preference-signal plumbing (thumbs / regenerate / edits).** Blocks dial auto-*select*
(pick the value, not just calibrate the range) AND voice/LoRA learning ("learn how you want me to act").
Highest-leverage single input to build.

**📋 QUEUED product gaps:** Profiles studio UI (backend tested, no front-end); engine restart + `--sae` CLI
passthrough (+ restart the engine/`:8097` studio stopped this session for GPU headroom); KV fast-path for
time-travel branches (snapshot restore, mechanism proven).

**🕗 DEFERRED / de-prioritized (with reason):** Memory hardening (provenance/ownership/durability/adversarial
gates) — de-prioritized per the local single-user threat model; REVISIT if the assistant ingests external
content. Slot value-injection into reply — lower priority (slot memory de-emphasized). Persistent-injection
nf4 KV-edit bug — deferred (`persistent_injection_findings.md`). Receipt-tuned wording (#8) — RAG-tier win.

**❌ KILLED (research said don't build):** Parliament (panel of stances), Quine (show-the-model-its-state).

**📚 RESEARCH pinned in `FINDINGS.md`:** all 6 wild experiments (Wave 1 cross-family Qwen×Gemma + Wave 2).
Encores DEFERRED: cross-family telepathy, SAE causal leg (mechanistically closed), multi-seed replication.

---

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

11. **"Explain this answer" — the inspect-a-reply core surface** — **M1-M5 COMPLETE (2026-07-04) — the whole
    feature, reachable from the studio, the terminal, and any OpenAI client; optional follow-ups only.** Spec at `research/EXPLAIN_THIS_ANSWER_SPEC.md` (5 milestones, honesty invariants, cost
    model). The mainstream front door: measured confidence/influence/concepts/counterfactuals on any
    reply, never self-reported. Mostly assembly of existing parts (runlog manifest + trace, replay.py
    ablation, run.js receipts, SAE readouts). Order: M1 free-signal `/runs/<id>/explain` (Sonnet) → M2
    rigorous greedy-both-arms ablation + redundancy guard (Sonnet) → M3 counterfactual dials → M4
    accountable-self narration + confabulation-diff (Opus, honesty-critical) → M5 TUI + any-client
    bridge. The trap (do-not-build): a plain "explain" that asks the model to explain itself = the
    confabulation machine.

    **M1 shipped:** a new stdlib-only `research/explain.py` (mirrors replay.py/memory_mode.py's
    separation -- no model, no GPU, unit-testable on plain dicts) assembles a run's already-logged
    signals into one `explanation` object; `clozn_server.py` wires it as a thin `POST
    /runs/<id>/explain` (404 on an unknown run, else `runlog.get_run` -> `explain.explain` -> `_json` --
    no substrate needed, unlike `/replay`/`/branch`). Shape: `{run_id, confidence, influences_active,
    concepts}`. **confidence** -- the trace's "uncertain moments" (tokens < `LOW_CONF` 0.5, matching
    run.js's convention), each with its recorded alternatives, plus a one-line "N hesitations" summary;
    `{available:false, note:"token trace captured on the engine path"}` when the run carries no
    per-token trace (the HF chat path). **NEVER a single aggregate confidence number** -- enforced by a
    recursive scan in the tests, not just a top-level shape check. **influences_active** -- every fired
    memory card resolved to its provenance (`source_run_id`/`source_turn`/`quoted_span`) by looking it up
    in `memory_cards` by id, plus the topic-relevance gate value and the active tone dials; a card whose
    id no longer resolves (edited/deleted) or never had a quote gets an explicit "no receipt" note
    instead of silently looking backed. **Every card and dial entry carries `causal_verified:null`**
    (active is not proof -- only M2's on-demand ablation may ever flip it). **concepts** -- honestly
    `{available:false, note:"concept readout needs the engine — not available on this run."}` on every
    run today (no logging path threads the engine's `sae:<id>` StepFeatures onto the stored record yet --
    `runlog.TRACE_KEYS` only keeps tokens/confidence/alternatives), but reads `trace["concepts"]` /
    `run["concepts"]` first so the day that capture wiring lands, assembly needs no changes (tested by
    hand-mutating a fetched run to prove the forward-compatible contract). Tests: `test_explain.py` (27,
    pure-assembly against fixture runs built through the real `runlog.record()`/`get_run()` round trip +
    an isolated `memory_cards` store -- with-trace, without-trace, with-cards+provenance, an unresolvable
    card id, internalized-mode's missing `applied_ids`, with-dials, concepts present/absent, empty run,
    non-dict/garbage input, a maximally-malformed-but-dict run) + `test_explain_server.py` (3, the
    no-socket HTTP dispatch proving the thin endpoint wiring and that it needs no substrate). Suite: 439
    passed / 3 skipped (was 409/3 before this session). `run.js` untouched (display is M5's job).

    **M2 shipped:** a new stdlib-only `research/receipts.py` (same separation as explain.py/replay.py --
    no model, no GPU, duck-typed against the live substrate, unit-testable against a fake `.chat()`)
    fixes the seam the spec calls out: the pre-M2 receipt (still what run.js's per-card/per-dial buttons
    drive today, via a single `/replay` call) diffs the ORIGINAL SAMPLED reply against a greedy replay
    with one influence off -- mixing influence-on/off with sampled/greedy in one delta.
    `receipts.receipt(run, influence, sub)` regenerates BOTH arms greedy
    (`replay.replay(run, {"greedy":true}, sub)` for the WITH baseline, `{**changes,"greedy":true}` for
    WITHOUT) from the run's own stored messages -- the stored sampled reply never enters the diff, and
    every receipt says so via a `note` field. `influence` is one of
    `{card_id}`/`{dial}`/`{memory_off:true}`/`{behavior_off:true}`; a card ablates through the SAME
    replay.py `_apply_changes` prompt-mode path M1 already relies on for provenance (in "internalized"
    mode replay.py's own "not applied" note is relayed here as `ablation_note` and `causal_verified` is
    correctly **False** -- never a silently-laundered "no effect"); a dial ablates via
    `behavior_overrides:{name:0.0}`, leaving every OTHER active dial untouched (true leave-one-out). Delta
    math (`receipt_metrics`) mirrors run.js's `receiptMetrics()` EXACTLY, including the JS-vs-Python
    rounding tie (`Math.round` rounds a trailing .5 UP; Python's builtin `round()` bankers'-rounds it down
    -- e.g. 62.5 -> 63 here/in JS, 62 under naive Python `round()`) -- caught by two dedicated tie-case
    tests. `prove_all(run, sub)` runs leave-one-out over every card+dial the M1 manifest
    (`explain.explain(run)`) says FIRED, sharing ONE greedy baseline across every check (safe -- the same
    deterministic call every time -- and NOT the batched-forward-pass optimization the spec's cost model
    names next: documented as `perf_note`, not implemented here), plus the spec-mandated **REDUNDANCY
    GUARD**: among influences whose own leave-one-out showed no effect (exact reply-string equality --
    greedy is deterministic), every PAIR is re-ablated jointly; a pair whose joint drop DOES change the
    reply is reported `{redundant:[...], note:"together they drive this; individually neither is
    load-bearing"}` instead of "neither mattered". Documented as a PAIRWISE approximation only
    (`approximation_note`), not the full power set -- a 3-way-or-higher redundancy would be missed, said
    so explicitly. Cost asymmetry surfaced per-receipt via `cost_note` (a card/memory ablation re-prefills
    the whole context; a dial is decode-time, KV-reusable, cheap). Wired as `POST /runs/<id>/receipt
    {influence:{...}}` (one) and `POST /runs/<id>/receipts {}` (prove-all) in `clozn_server.py`, right
    after `/explain`; both 404 on an unknown run and 503 when `SUB` is None (both arms regenerate, unlike
    `/explain`), and `/receipt` 400s on a missing/malformed `influence` body. Tests: `test_receipts.py`
    (25, model-free against a FakeSub whose `.chat()` is a deterministic function of which influence is
    live -- metric math incl. both rounding ties, the both-arms-greedy call count/shape, the
    unapplied-ablation honesty guard, a constructed card_a+card_b redundant pair, never-raises) +
    `test_receipts_server.py` (10, the no-socket HTTP dispatch incl. the 503-no-substrate path on both
    endpoints and a full prove-all-over-HTTP redundancy check). Suite: 474 passed / 3 skipped (was 439/3
    before this session). `run.js` untouched (M5's job).

    **M3 shipped** (`6442947`): a new stdlib-only `research/counterfactual.py` (receipts/replay imports
    only) turns the ablation machinery into an interactive what-if. `counterfactual(run,
    behavior_overrides, sub)` regenerates BOTH arms greedy -- baseline = the run's ACTUAL live dials
    (`replay.replay(run,{"greedy":true},sub)`), arm = the same with `behavior_overrides` merged -- and
    diffs only those two (the stored sampled reply never enters the subtraction, same seam M2 fixes).
    Coherence and causation are kept **orthogonal**: a dial that genuinely took hold but derailed the
    model reports `causal_verified:true` AND `coherence.degenerate:true` -- never laundered into "big
    delta = big effect" (law #6). Unapplied-override guard reads the `active_dials` replay.py already
    recorded on the child run (no new model probe) and flips `causal_verified:false` naming exactly which
    axis failed; a 0.0 override is never flagged. `dose_sweep(run, dial, values, sub)` runs
    `counterfactual()` independently per value (2 greedy re-gens each, no shared baseline) and returns the
    per-model dose-response curve with `derailed_at` naming which values crater -- the honest answer to
    "how warm is warm on THIS model" instead of trusting a 7B-calibrated number on a 1.5B. Wired as `POST
    /runs/<id>/counterfactual {behavior_overrides:{...}}` after `/receipt` (404 unknown run, 503 no SUB,
    400 non-dict/empty overrides, 500 on a failed result). Tests: `test_counterfactual.py` (23, model-free
    FakeSub/FakeSteer whose chat() is a deterministic function of live dial values) +
    `test_counterfactual_server.py` (5, no-socket dispatch). `run.js` untouched (the slider UI is a later
    display pass, deferred to avoid colliding with M5-display's run.js edit).

    **M5-display shipped** (`6dec79e`): the surfaces over M1's endpoint (no generation, endpoint
    unchanged). TUI -- `clozn explain <run_id>` / `clozn explain --last` in `clozn_cli.py`; `--last`
    resolves via a direct `runlog.list_runs(limit=1)` read so it works with the Studio HTTP server DOWN
    (only the explanation fetch needs the server), and a pure `format_explain(dict)->str` renders a
    per-token confidence sparkline + hesitation bars with "almost said" alternatives (never an aggregate
    %), the influence list with provenance quotes + gate/mode, and every not-available panel's note
    verbatim; every influence is tagged `[was active]`, never `[caused]` (only a real M2 receipt may print
    `[proven]`/`[ruled out]`). Web -- a new "Explain" tab beside "Detail" in `run.js` (Detail stays
    default; receipt buttons unchanged), prefetched in parallel with the run (M1 is free per the cost
    model, so it opens instantly), rendered by a pure `explainSummaryHTML()` under the same honesty rules,
    styled through run.js's own self-injected `ristyle()` (no `app.html`/`app.js`/`clozn.css` edits).
    Tests: `test_explain_cli.py` (21, canned-dict + one in-process real-server wire-format check).

    **M4 scaffolding shipped** (`bf98a7c`): the MODEL-FREE harness for the accountable-self narration --
    `research/narrate.py`, with the honesty-critical judgment deliberately RESERVED, not faked.
    `constrained_narration(explanation, sub)` is never handed the run (structurally cannot crib the
    transcript), flattens the M1 manifest to id-tagged facts, asks the model to cite `[id]` after every
    clause, and drops any citation that doesn't resolve. `unconstrained_why(run, sub)` is never handed the
    explanation (cannot see a receipt), asks "why did you answer that way?" with zero facts in context,
    and returns its text labeled `do_not_surface_as_answer`/`role:confabulation_sample` three redundant
    ways. `confabulation_diff(text, explanation, support_matcher=lexical_default)` splits sentence-level
    claims, tags each supported/unsupported (a throwing matcher fails CLOSED to unsupported), and renders
    unsupported ones as `WARNING: credits "<claim>"; no receipt for that.` `narrate()` returns exactly
    `{constrained_narration, flags, unsupported_claims, note}` -- four keys, no answer/why field, so the
    fabrication cannot reach an answer slot. The **trap-guard is structural** (the two arms are physically
    denied each other's inputs), tested directly. Honesty boundary: `lexical_default` is a loudly-
    documented weak proxy (over- and under-flags); the real `semantic_support_matcher` RAISES
    `NotImplementedError` rather than ship a fake verdict. Tests: `test_narrate.py` (34, model-free
    FakeSub). NO server endpoint (deferred to avoid the concurrent `clozn_server.py` edit). Known gap it
    flagged for the gated pass: claim-splitting is sentence-level, so a compound "concise because you
    asked, AND warm because I like you" is one claim and a half-confabulation can hide behind the true
    clause.

    **M4 on-model shipped** (`724b40d`): the honesty-critical half -- a REAL `support_matcher` that replaces
    narrate.py's raising hook. `research/semantic_matcher.py`'s `nli_support_matcher` judges whether a
    self-narration claim is entailed by the measured receipts using an INDEPENDENT model -- a cross-encoder
    NLI (`nli-deberta-v3-base`), deliberately NOT the audited Qwen, so it cannot rubber-stamp the audited
    model's own confabulations (the self-consistency trap law #1 warns about). Passed via narrate's pluggable
    `support_matcher` param; narrate.py itself stays frozen + model-free (its 34 tests untouched). The
    embedding-vs-NLI choice was MEASURED, not asserted: the on-disk MiniLM bi-encoder separated true claims
    from confabulations by only ~0.02 on the seed cases (cosine = topical overlap, not entailment), the
    cross-encoder by ~0.84 (true 0.83-0.98 entailment, confab <=0.002) -- and it does per-influence
    ATTRIBUTION (a "concise" claim entails from the concise card, not the warm dial). Honest ceiling loud in
    the docstring: one (receipt,claim) pair at a time (the compound-sentence gap remains), "unsupported"
    conflates no-receipt with contradicted (raw contradiction prob returned alongside), one model/threshold/
    English, lazy-load + LABELED degrade (`nli-unavailable`) so a caller can fall back to lexical_default.
    Gated test `research/tests/test_semantic_matcher_gated.py` (`-m model`, 3 tests) is the spec's M4 "Done"
    bar: a SEEDED KNOWN divergence (concise card + warm dial, nothing about chess, vs an unconstrained why
    mixing paraphrased-true claims with a chess confabulation) proves the matcher passes the true ones,
    flags the confab, and rescues a paraphrase lexical_default wrongly flags; the trap-guard holds with the
    real matcher; and a real-Qwen end-to-end demo caught the model confabulating live (asked "why did you
    answer that way?", the 1.5B APOLOGIZED and RE-ANSWERED the question instead of introspecting -- law #1 in
    the wild). Suite 557 passed / 6 skipped (the 3 gated tests skip in the plain run; semantic_matcher
    imports torch-free at collection).

    **Sibling experiment landed same batch** (`388b895`, WILD_EXPERIMENTS #6, not part of the milestone
    chain): `research/memory_disorders.py` broke SlotMem four ways (interference / confabulation / amnesia
    / intrusion) via instance monkeypatch and asked whether the RECEIPT SIGNALS alone diagnose each
    disorder blind. A pre-registered rule-classifier that never sees the label got **3/5 across every
    seed, including the two safety-critical ones (confabulation, amnesia)** -- receipts are a diagnostic
    instrument, not just a display. Honest miss: interference under a LIVE gate presents as pathological
    over-abstention (gate floor computed from broken uncentered sims), not the predicted cross-talk, so it
    misfiled as healthy (10/15 blind, not 15/15). Confabulation stayed fluent-while-wrong ("capital of
    France" -> "Beatrix") -- the dangerous failure is the one that doesn't look like one. Full report:
    `research/memory_disorders_findings.md`.

    **M4 reachable in the product** (`29189e2`): `POST /runs/<id>/narrate` wires `narrate.narrate` into
    `clozn_server.py` (right after `/counterfactual`) so the studio/TUI can invoke the accountable-self
    narration on any run -- the real independent NLI judge when its checkpoint is present, else the LABELED
    lexical fallback, and narrate()'s own `note` states which ran (self-describing honesty level). 404/503/500
    per the sibling endpoints' precedent; model-free server test `test_narrate_server.py` (3) forces the
    lexical branch (no checkpoint load) and pins the trap guard over HTTP (the confabulation appears ONLY
    wrapped in a WARNING flag, never as the narration surface).

    **M4 narration RENDERED** (`113b589`): the accountable self is now in front of a user, not just in the
    API. The Run Inspector "Explain" tab gets a lazy "Why did it say this?" button (`/narrate` generates, so
    on-demand) rendering `narrateHTML` -- the receipt-bound narration, the confabulation `flags` as distinct
    WARNING rows, the matcher `note` as a muted caption; `clozn explain <run> --why` renders the same via a
    pure `format_narrate` (flags in the warm/hot-rose truecolor, a nod to "a flag is a low-confidence token";
    no-op when color is off). Only the safe 4-key object is ever rendered -- the raw confabulation has no field
    to leak through (a Node smoke asserts even a stray 5th field can't). Tests: `test_narrate_cli.py` (22,
    canned-dict honesty invariants + an in-process `/narrate` wire check + the `--why` opt-in contract).

    **M5 any-client run_id bridge SHIPPED** (`549630c`): `POST /v1/chat/completions` now returns its run id
    two ways -- an additive `clozn_run_id` JSON field (OpenAI clients ignore unknown fields) and an
    `X-Clozn-Run-Id` header -- so a user chatting through ANY OpenAI client can then `clozn explain <that id>`.
    `_log_run` returns the rid (None on any failure -- logging still never breaks a request); `_json` gained a
    backward-compatible `extra_headers` param (all 108 existing call sites emit byte-identical output); the id
    is omitted cleanly when logging fails. The STREAMING path was deferred with a real reason (SSE headers
    flush before generation, and a trailing frame after `data: [DONE]` is silently dropped by clients like
    openai-python) -- left byte-identical, documented inline. Tests: `test_bridge_server.py` (5, model-free).

    **M1-M5 COMPLETE.** Measured confidence + rigorous both-arms-greedy receipts + counterfactual dials + the
    accountable-self narration (real, independent NLI confabulation judge) -- reachable from the studio Run
    Inspector, the terminal (`clozn explain [--why]`, `clozn trace`), and any OpenAI client (the run_id
    bridge). Suite: **609 passed / 6 skipped** on the clean committed tree (was 439/3 at the session start).
    **Optional follow-ups only:** clause-level claim extraction (close the compound-sentence gap the
    sentence-level split leaves, now that the matcher is real); a streaming-path run_id (needs a client-safe
    carrier); making `nli_support_matcher` the studio's live narrate default (the endpoint already prefers it
    when present). Also shipped this session: the CLI confidence heatmap -- `clozn trace`/`explain` painted by
    per-token confidence, `clozn run --heat` live, denoise's palette (#e07a96/#c3cde0/#7aa7ff) in the terminal.
