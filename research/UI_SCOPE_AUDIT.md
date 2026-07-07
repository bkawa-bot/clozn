# Clozn UI Scope — Codebase Audit

**Read-only audit · 2026-07-07 · verified against the working tree by a 7-agent fan-out** (6 screen clusters + 1 frontier). Every verdict below traces to source; line numbers current as of the audit.

Two design docs mapped against what the tree can actually do **today**:
- the **6-screen UI scope** (the Run Inspector product doc), and
- the **6 frontier / future ideas**.

---

## The one-line thesis

**The doc is a floor, not a ceiling.** In the places that matter most — causal receipts, the engine-native glass box, honest confidence — clozn is already *above* it. Where that's true, the move is to **strengthen the surface, not flatten it** to match the sketch.

## Verdict legend

| Verdict | Meaning |
|---|---|
| **SHIPPED** | A user can do this now, on the engine substrate. |
| **QUICK WIN** | The signal is *already computed* and then discarded at the display/persist boundary. The work is to stop throwing it away — often client-side. |
| **NEEDS A HOOK** | A genuine build (new capture path, C++ change, or renderer), but no open research question. |
| **RESEARCH / OOS** | An unbuilt subsystem, a lab-only artifact, or something clozn's own results say *not* to do. |
| **★ LEAN-IN** | clozn already meets or exceeds the doc here; the receipt is stronger than the sketch. |

## Scoreboard (58 screen line-items)

- **21 SHIPPED** — works today
- **10 QUICK WIN** — data exists, wire it
- **20 NEEDS A HOOK** — real build, no research
- **7 RESEARCH / OOS** — or a new subsystem
- **12+ LEAN-IN** — meets or beats the doc

---

# Part one — the 6-screen product scope

The recurring shape: the runtime **measures far more than it shows**. Most "quick wins" are one hop of plumbing between a real number and the log line that drops it.

## Screen 1 · Run foundation & timeline

| Feature | Verdict | What's there / gap |
|---|---|---|
| **The Run object** | SHIPPED | Every turn persisted whole — messages, response, dials, memory contacts — and re-openable. `research/runlog.py` |
| **Run metadata** (seed · quant · device · ctx) | QUICK WIN | Model name + timing already ride along; seed / quant / device / ctx all live in-process at generation time but are never threaded into the run record. Capture values that already exist. |
| **finish_reason** (eos / length / steps) | QUICK WIN | Engine computes the true stop cause, then drops it one hop before logging and hard-codes `"stop"` on the OpenAI shape. `cloze_server.cpp:56-58`. **The recurring theme, screen after screen.** |
| **Typed RunEvent timeline** | NEEDS A HOOK | Engine emits typed SSE frames (`gen_started`, `tokens_committed`, `gen_finished`) but at *token* grain, not the doc's semantic taxonomy (memory-fired, dial-applied, repair-branched). Needs a mapping layer. |
| **Local run logs** | NEEDS A HOOK | The server silences its own log; surfacing it is a real hook. |
| **Capture tiers** (Light/Standard/Deep/Lab) | NEEDS A HOOK | A binary light/full switch exists; the doc's 4 user-facing tiers do not. |
| **tool_call events** | RESEARCH / OOS | There is no tool-calling anywhere in clozn. This event type has no source. |

## Screen 2 · Context & memory inspector

| Feature | Verdict | What's there / gap |
|---|---|---|
| **Memory provenance — "you said this"** | SHIPPED ★ | Each fact card carries `source_run_id` · turn · `quoted_span`, and nothing enters memory except through an enforced approval gate. **Stronger than the doc's "retrieved chunk" — audited origin, not a similarity hit.** |
| **Memory contacts** (fired / included) | SHIPPED | Which cards fired and entered the turn is recorded and shown. *Note: all-or-nothing per turn today, not per-card include/exclude.* |
| **Per-card relevance score** | NEEDS A HOOK (small) | The per-card cosine *exists* — `topic_gate.relevance()` — but only the scalar gate decision is read on the hot path; the number is computed and discarded. Surface a value you already calculate. |
| **Final assembled prompt** | NEEDS A HOOK | Prompt mode: the built string is a discarded local (small hook to keep it). Internalized mode: memory is a non-textual soft-prefix — there is no string to show (needs a decode-surrogate). |
| **Segmented prompt** (system/memory/history/user) | NEEDS A HOOK | Memory is glued into the system message with no segment schema; nothing yet to color-code. |
| **Context-window map** | NEEDS A HOOK | Compounds the segmented-prompt and budget gaps. |
| **Retrieval / RAG over files** | RESEARCH / OOS | "Memory" = trait cards + fact-slots. There is **no** document store, embedding index, or file retrieval. The doc's "retrieved chunks" is a from-scratch subsystem. |
| **Token-budget usage meter** | RESEARCH / OOS | The OpenAI-compat endpoint reports a fabricated `0/0/0`. A real meter means real accounting; until then, the honest move is to show nothing. |

## Screen 3 · Token & output inspector

| Feature | Verdict | What's there / gap |
|---|---|---|
| **Honest confidence + uncertain moments** | SHIPPED ★ | Per-token confidence is a genuine full-vocab softmax — not a heuristic. Three surfaces (`explain.py`, `run.js`, CLI) share *one* `LOW_CONF=0.5` constant by explicit cross-reference so they can't drift, and no fake aggregate "confidence %" is ever synthesized. **The honest-confidence discipline is already a shipped convention.** |
| **Per-token stream** | SHIPPED | Live per-token decoded text over SSE, committed to the run. *Raw `tokenId` dropped after decode — cheap to keep.* |
| **Top-k alternatives** | SHIPPED | Runner-up tokens (chosen excluded, capped 3) with probabilities, on both streaming and explain paths. |
| **Token probabilities** | SHIPPED | Real softmax probabilities, persisted. Stored as `prob`; a logprob is a one-line derivation. |
| **Sentence ↔ token spans** | QUICK WIN | Token stream + text both on disk; mapping a sentence to its token range is a pure client-side function. |
| **Repetition markers** | QUICK WIN | Absent, but a cheap client-side scan over the already-captured token stream. |
| **Per-token entropy** | NEEDS A HOOK | Not computed on the hot path, and the trace-key filter would drop it. A derived-new quantity (the distribution is available at the moment, just not summarized). |
| **Per-token timestamps** | NEEDS A HOOK | The `t` field is a token *counter*, not a clock. Real wall-time needs a timing tap in the gen loop. |
| **Baseline-vs-repaired token diff** | NEEDS A HOOK | Replay doesn't capture `trace_out`, so there's no token-aligned before/after to diff, and no token-level diff renderer. (See the replay quick-win on screen 5.) |

## Screen 4 · State & "Model MRI"

| Feature | Verdict | What's there / gap |
|---|---|---|
| **SAE features + concept readouts** | SHIPPED ★ | On-device SAE (131k-feature JumpReLU, ~9ms, 1-ulp parity) + 6 diff-in-means concept probes, compiled into the C++ engine and wired to `brain.html`. `cloze_server.cpp:439-503` · `:567-754`. **A CUDA-kernel-tested white-box readout in the production server.** |
| **Activation capture** | SHIPPED | `/harvest` taps residuals at `l_out-<il>` and feeds `engine.html` — live tensors, not a mock. |
| **State deltas** (write & observe) | SHIPPED | `/state` writes into the residual stream and returns a real `moved_l2` delta with baseline/edited top tokens — surfaced at `/engine/observe`. |
| **Concept candidates — correlational, honest** | SHIPPED ★ | Readouts ship with `causal_verified: null` until patched — the SPEC's wire-honesty rule, commented in the C++. That *is* the doc's "candidate, correlated, not causal." `protocol/SPEC.md:19,26-29` |
| **Learned probes — the primitive** | SHIPPED | `ConceptProbes` is literally a linear probe: fit from labeled ±examples, `.project()` to score, `.steer_vector()` to write. Already user-triggerable via `/steer/custom`. `probe.hpp:16-53` |
| **Layer summaries + activation distribution** | QUICK WIN | Raw tensors already in hand from `/harvest`; norms, per-layer summaries, heatmap bins are client-side reductions. |
| **4-tier capture modes** | QUICK WIN | A binary light/full mode already gates capture; expanding to four tiers is a config surface. |
| **Attention maps** | NEEDS A HOOK (C++) | The flash-attention path discards per-head weights; surfacing a map means capturing them in the kernel path. |
| **Attention entropy** | RESEARCH / OOS (blocked) | Downstream of attention maps — nothing to summarize until those weights are captured. |

## Screen 5 · Patch & repair studio

| Feature | Verdict | What's there / gap |
|---|---|---|
| **Quick-repairs + save-fix** | SHIPPED ★ | "Too verbose → concise", "too vague → concrete", "too agreeable → candid", "too cold → warm" buttons fire `/feedback` then replay, with before/after as evidence. `run.js:880-914`. **Exactly the "feedback buttons on receipts" pattern — already live.** |
| **Activation steering** | SHIPPED ★ | `/intervene` writes a steer vector; 33 library dials + user-fit custom directions. A negative coefficient *reduces* a direction — the doc's "suppress" for free. |
| **Branch / replay from a turn** | SHIPPED | Re-generate from any point with a changed dial / prompt / memory; text-level before/after diff renders. *KV-cache fast-branch (byte-exact, proven in `kv_timetravel.py`) is a perf upgrade, not a blocker.* |
| **Residual-stream patching** | SHIPPED (engine) | The write-and-observe primitive is real via `/state` — just not surfaced as a studio control yet (lives on standalone `engine.html`). |
| **Edit prompt / remove memory** | SHIPPED | Editing the user turn and removing a memory card (prompt mode) both drive a real re-run. |
| **Patch-history tree** | QUICK WIN | `parent_run_id` is already recorded on every branch — the lineage exists, it's just never rendered as a tree. |
| **Token-probability replay diff** | QUICK WIN | Thread `trace_out` through `replay.py:155` and the token-aligned prob diff (screen 3) has its data. |
| **Sampler controls** (top-p / top-k / stop) | NEEDS A HOOK (C++) | Engine sampler is greedy / temp / rep-penalty only. Temperature patch is easy; top-p, top-k, stop-sequences, logit-bias need sampler work. |
| **Attention-head ablation** | NEEDS A HOOK | No per-head hook exists anywhere. **The single most involved item in either doc.** |
| **Exclude a retrieved chunk** | RESEARCH / OOS | Depends on RAG, which doesn't exist. |

## Screen 6 · Receipts & tests

| Feature | Verdict | What's there / gap |
|---|---|---|
| **Causal receipts** | SHIPPED ★★ | Receipts are **causal, not assertion-based**: M2 leave-one-out, both-arms-greedy, redundancy-guarded. Confabulated self-narration is flagged as a distinct WARNING via an independent NLI judge. `receipts.py` · `counterfactual.py` · `semantic_matcher.py`. **This is the headline. The doc's tiny-tests are weaker — strengthen this, don't dumb it down.** |
| **Auto-repair recommendations** | SHIPPED ★ | The preference-plumbing loop: repeated quick-repairs accumulate, cross a threshold, become an evidence-backed proposal ("asked for concise 3× — make it default?"), reviewed in the Suggestions panel / `clozn preferences`. `preferences.py` · `feedback.py`. **Frontier idea #2's L1+L2 — already in product with a review UI.** |
| **Export** (JSON / Markdown) | QUICK WIN | Run records already serialize to JSON, and a profile-download pattern already exists to copy for a receipt export button. |
| **Tiny-test harness + before/after status** | NEEDS A HOOK | No assertion harness today — but the pluggable-matcher pattern in `narrate.py` is the right seam, and the diff machinery for pass/fail already exists. |
| **Unified receipt object + repro metadata** | NEEDS A HOOK | Half exists; determinism is real (byte-exact in `kv_timetravel.py`) but seed / sampler / build aren't recorded into a repro block yet. |
| **Concept candidates on the run record** | NEEDS A HOOK (small) | The readout is live in the brain viz but not threaded onto the persisted run — a plumbing hop. |
| **User-trained probes · cross-model receipts** | RESEARCH / OOS | A real fit + causal-verify probe rig exists only in the sibling `inspector/` project (RWKV/sentiment), not the product; cross-model is offline scripts and Parliament was killed. |

---

# Part two — the 6 frontier ideas

Your instinct was right: **several are more built than a "future ideas" doc would ever assume.** Two are essentially shipped and engine-native.

## #1 · Interpreting internal concepts — SUBSTANTIALLY BUILT
On-device SAE compiled into the C++ engine (131k-feature JumpReLU, GPU top-k, 1-ulp parity, ~9ms) + a Python twin filtered against **103,491 Neuronpedia-labeled features** + 6 engine-calibrated concept probes. The honesty rule (`causal_verified`) is load-bearing.
- `cloze_server.cpp:439-503` (SAE), `:567-754` (probes); `sae7b.py:21-35`, `brain_readout.py:62-140`; labels in `np_labels_l15.json`.
- **Gap:** "supported by N examples" transparency isn't surfaced per-concept; one-click causal verification of a *labeled* concept (vs. a dial) is CLI-only.
- **More built than expected?** Yes — a doc writer pictures a matplotlib notebook; clozn has a CUDA-tested encoder in the production server.

## #2 · Automatic repair recommendations — PARTIAL (L1/L2 shipped)
**L1** (rule-based complaint→dial) and **L2** (pattern-from-history → propose & review) are both shipped in product. A research prototype (`receipts_as_reward.py`) evolves memory wording using the ablation receipt as fitness and **beats both the shipped seed (1.000 vs 0.833) and a random-walk null** — an L4-shaped result.
- **Gap:** no single-instance "this run failed → here's a fix"; L2 needs repeated signals first. `idle_selfplay.py` honestly found receipt-gated auto-consolidation passes a laundered injection — **"do not ship"** is the recorded finding.
- **More built than expected?** Yes — shipped L1+L2 plus a null-beating auto-patch experiment.

## #4 · Mechanistic interpretability workflows — SUBSTANTIALLY BUILT
A real read→discover→write→observe SDK against the *production* engine (`cloze_engine.py:165-222` → `/harvest`, `/state`, `/intervene`), a browser front-end (`engine.html`), activation patching with a same-norm random control (`probe_and_patch.py`), and ablation-verified 2-hop feature circuits (`feature_circuit_clean_qwen.py` → `inspector/runs/feature_circuit.html`).
- **Gap:** grepped the whole tree — **zero** attention-head ablation, and no classic cross-layer logit lens (only per-position final-logit top-k; `StepLens` is real but narrower than "residual decomposition across depth"). None of it sits behind a studio "Lab Mode" toggle.
- **More built than expected?** Emphatically yes — the doc filed this whole category as thin/aspirational "Lab Mode."

## #3 · Cross-model state comparisons — DEEP RESEARCH, ZERO PRODUCT
Behavior-level: `mirror_bench.py` confirms content-legible / process-blind **cross-family** (Qwen2.5-7B × gemma-2-9b). Activation-level: `vector_telepathy.py` (736 lines, pre-registered E1–E10) fits ridge + Procrustes bridges between independently-trained 1.5B and 7B models — and **a pure rotation (Procrustes) matches the full affine map**, narrowing the doc's own "usually not valid" prior to a specific case. `telepathy_findings.md:87-120`.
- **Gap:** cross-*family* (different tokenizer) is untested, flagged "likely much harder." **Zero** product surface — no endpoint, no UI. 100% research artifact.
- **More built than expected?** Dramatically — the doc says "don't bother"; clozn ran it with real nulls and got a positive, nuanced answer.

## #5 · Learned probes — PRIMITIVE LIVE, LOOP UNBUILT
The fit-classify-steer machinery exists *twice* (engine `ConceptProbes`, Python `SteeringControl`) and a user can already inject labeled ±examples into a live probe-fit via `/steer/custom` (`clozn_server.py:1057-1064`, `steering.py:292,544`) — aimed at **steering, not warning**.
- **Gap:** nothing lets a user mark "20 good / 20 bad *replies*" in the UI, and nothing applies a fitted probe at *replay time* to warn "this is drifting." A wiring gap, not a research gap.
- **More built than expected?** Yes for the primitive; no for the specific mark→train→warn dream.

## #6 · Rich "Model MRI" visualizations — REAL PIPES, DECORATIVE FLAGSHIP
At least three non-decorative data-to-pixel pipelines already ship: the write-and-observe `engine.html`, the engine's SSE-driven live viz (`viz_html.hpp`), and the Explain tab where every influence traces to a real provenance. `brain.html` has a genuine live mode wired to `/engine/concepts` (`brain_readout.py:108-126`).
- **Gap:** the flagship `brain.html` **defaults to a hand-shaped demo mind** (`brain_README.md:75-77`) — the exact decorative thing the doc warns about — going data-real only when you flip it. No single view offers the doc's selectable token/layer/event/patch/memory scopes; scope today = which page you open.
- **More built than expected?** Mixed — yes for the plumbing, no for the flagship visual.

**Ranking, most-built → least:** #4 & #1 (essentially shipped, engine-native) → #2 (L1/L2 shipped + research win) → #3 (deep research, no product) → #5 (right primitive, no loop) → #6 (real pipes beside a decorative flagship).

**Biggest hidden gem:** the on-device SAE + `ConceptProbes` + harvest/write/observe stack — a CUDA-tested, causally-verified white-box SDK in the production C++ server.

**Biggest genuinely-hard one:** true cross-*family* activation-level transfer. And more fundamentally: Law #1 + the quine-test null mean **any feature that leans on the model introspecting about itself is empirically dead on arrival** — which is why every shipped surface routes through receipts instead of self-report.

---

# Three patterns that cut across both docs

### Pattern 01 — Captured-but-discarded
The most common reason something isn't in the UI is **not that it's missing — the runtime computes it and then drops it** at the persist/display boundary: `finish_reason` → hard-coded `"stop"`; raw `tokenId` → dropped; per-card relevance cosine → reduced to a boolean; concept readouts → live in the viz, never on the run record; `trace_out` → not carried through replay. This is why the quick-win column is 10 deep. **A focused sprint that simply stops throwing signals away would light up a large fraction of the doc at once.**

### Pattern 02 — The honesty discipline is a moat
The doc's north star ("show state signals, not thoughts") isn't an aspiration — it's an **enforced invariant already on the wire**: `causal_verified: null` until a real patch; no fabricated aggregate confidence; confabulation flagged by an independent judge; one shared `LOW_CONF` constant so surfaces can't drift. Backed by clozn's own killed experiments (Law #1, the quine-test null). Every shipped surface routes through receipts instead of self-report — which is exactly why the receipts are the thing to lean into.

### Pattern 03 — Two honest scope corrections
- **There is no RAG / file retrieval.** "Memory" = trait cards + fact-slots with audited provenance — a different, arguably stronger thing. "Retrieved chunks", "exclude a chunk", and the file-context map all sit on a subsystem built from scratch.
- **Token-budget usage is fabricated** (`0/0/0` on the compat endpoint). A usage meter means real accounting first; until then, showing nothing is the honest call.

---

# Where clozn already exceeds the doc (lean in — don't dumb down)

Your steer: *"if the current receipts are stronger, strengthen that section rather than dumbing it down to match the design doc."* These are the places where matching the sketch literally would be a downgrade.

- **Causal receipts** — the doc proposes tiny-test *assertions*; clozn does *causal* receipts (leave-one-out, both-arms-greedy, redundancy-guarded, confabulation-flagged). Keep tiny-tests as a light add-on; the causal engine stays the headline.
- **Memory provenance** — the doc's "retrieved chunk" is a similarity hit; clozn's contact is an *audited origin* (run id, turn, quoted span, approval-gated).
- **Honest confidence** — real full-vocab softmax, one shared threshold across three surfaces, hard rule against a fake aggregate %.
- **The engine-native glass box** — on-device SAE, 6 concept probes, harvest/write/observe, feature circuits — compiled into the production server. The doc's "Lab Mode, someday" is largely already running.
- **Auto-repair, shipped** — frontier #2's L1+L2 are in product today with a propose-and-review UI and evidence run-ids.
- **Correlational-by-default honesty** — the doc asks for "candidate, correlated not causal"; clozn enforces it on the wire (`causal_verified: null`) and only flips true after a measured patch.

---

# A grounded build order

Sequenced by the audit, not the doc's page order — cheapest, highest-leverage first.

### 1. Now — the plumbing sprint (stop discarding)
Mostly client-side / one-hop: **finish_reason** · **run metadata** (seed/quant/device/ctx) · **patch-history tree** (render `parent_run_id`) · **sentence↔token spans** · **repetition markers** · **receipt export** · **token-prob replay diff** (thread `trace_out`) · **per-card relevance** · **layer summaries** · **capture tiers**.

### 2. Next — small hooks (a week or two each)
**Final-prompt capture** (prompt mode) · **concept readouts on the run record** · **token-level diff renderer** · **a "Lab Mode" toggle** that surfaces the *already-built* `engine.html` / state / brain live tools inside the studio · **segmented-prompt schema** · **per-token entropy**.

### 3. Engine — C++ work (real depth, no research)
**Attention-weight capture** (flash-attn discards them — unblocks maps + entropy) · **sampler expansion** (top-p / top-k / stop / logit-bias) · **per-token timing tap** · **attention-head ablation** (the hardest single item).

### 4. If wanted — new subsystems (decide before planning around them)
**Tiny-test harness** (on the pluggable-matcher seam — an add-on to causal receipts, not a replacement) · **the mark-good/bad → warn-at-replay probe loop** (the one product-facing gap in #5) · **RAG / file retrieval** · **honest token-budget accounting**.

### 5. Leave in the lab — research surfaces (strong as artifacts, not product)
**Cross-model comparison** (`vector_telepathy.py` / `mirror_bench.py`) · **one-click concept causal-verify** (the CLI rig works) · **head-level mechanistic tools** (gate behind Lab Mode when the engine hooks land).

---

*Method: read the research ledgers (`README.md`, `FINDINGS.md`, `WILD_EXPERIMENTS.md`, `NEXT_STEPS.md`, `ROADMAP.md`, `ARCHITECTURE.md`) end-to-end, then verified every claim against the actual source (engine C++, Python research scripts, served JS/HTML) rather than trusting the prose.*
