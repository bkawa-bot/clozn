# Clozn Roadmap — consolidated map

**Updated 2026-07-08.** Single source of truth for *done / v1 / later*. This supersedes the old phased
"monorepo migration" roadmap (Phase 0 = the reorg, now done). It **indexes** the detailed planning docs
rather than duplicating them — go to the linked doc for execution detail.

> **The thesis** (FRONTIER_BETS §0): as hardware commoditizes capability, **control becomes the product** —
> trust, steer, version, debug. Clozn's wedge is what a text-in/text-out runtime structurally can't do.

## The planning docs this map indexes
| Doc | Covers | Status |
|---|---|---|
| `notes/REPRODUCE_AND_PROVE_PLAN.md` | **#114** — `/score`, forced receipts, rederive (S0–S6) | S0–S4 ✅, S5/S6 left |
| `notes/JLENS_ENGINE_PLAN.md` | **#115** — engine-native J-lens (J0–J5) | not started (the v1 headline) |
| `notes/INTROSPECTION_EXPERIMENTS.md` | **X1–X8** — introspection science (needs J-lens) | designs ready, none run |
| `notes/FRONTIER_BETS.md` | strategic bets: perf stack, AR×diffusion (H1–H7), honesty ledger | idea inventory + ranked order |
| `notes/MEMORY_MODE_SWAP_SPEC.md` | prompt-mode memory | ✅ built (this is what ships today) |
| `docs/MODEL_SUPPORT.md` | model-agnosticism tiers, roster, J-lens as the brain-viz path | Tier-0 ✅; Tier-1/2 scoped |
| `docs/CURRENT_UI_BACKLOG.md` | inspector/UI item-level backlog | reconciled 2026-07-08 |

*(`notes/` is local/gitignored; the tracked map lives here.)*

---

## ✅ Done
- **Repo reorg** — product-only tree (`clozn/` package, `studio/`, `engine/kernels/`); research split to `../clozn-research`. *(old roadmap "Phase 0")*
- **Engine white-box runtime** — AR + diffusion GGUF on the C++ `cloze-server`: `/harvest`, `/score`, `/apply_template`, steer taps, prompt-mode memory.
- **#114 S0–S4 (reproduce & prove)** — teacher-forced `/score`, the SDK/substrate seam, rich per-token trace (`token_id`/`logprob`/top-k-entropy) + repro `meta.decode`, forced-mode receipts + rederive, and the graded-leaning inspector UI. *The null-floor experiment killed the "silent influence" badge (principled — filler-swap can't discriminate, Pearson 0.9985); shipped graded per-card leaning via co-present leave-one-out instead.*
- **Tier-0 model-agnosticism** — clozn runs **any AR GGUF** across the whole white-box stack (proven tap-by-tap on Llama-1B). Engine-side templating + derive-model-from-`/health`. *(Bonus beyond the plan docs — and it's what makes the J-lens portable.)*

## 🎯 v1 target — "causal receipts + J-lens readouts"
*Headline (JLENS_ENGINE_PLAN): "the local runtime with causal receipts and J-lens readouts — see what your GGUF is disposed to say, per token, and prove what changed its answer."*

- **#114 leftovers** — **S5** (turn on engine sampling; gated on a human go/no-go) · **S6** (docs/claims refresh).
- **#115 — engine-native J-lens (J0→J4).** *Not started; the headline. RACE posture (Anthropic published 2026-07-06; nobody has it in a product).*
  - **J0** — fit the lens in the lab (PyTorch, autograd, nf4 + checkpointing on the 16 GB card; ~100 prompts saturates), export per-layer matrices + manifest. ~1–2 days.
  - **J1** — **transfer gate (blocks J2).** ~20-line numpy oracle: does the HF-fitted lens survive on GGUF `/harvest` activations? top-1/5 agreement vs a shuffled-`J` null. **If it fails, the headline dies here — before any C++ — and the negative is publishable.**
  - **J2** — C++ `/jlens` route (apply = `unembed(J_l @ h)`, reusing the GGUF's own head; no `W_U` sidecar; forward-only). ~3–5 days.
  - **J3** — studio panel: per-token "disposed-to-say" chips (finally *earns* the workspace name `workspace_lens.py` overclaimed). Honest provenance label from the manifest.
  - **J4** — "does a 7B even have a J-space?" (spider test) — launch content either way; also the existence gate for X6.

## 🔭 Post-v1 backlog

### Performance stack — *all reuse the `/score` keystone* ("the debugger IS the speedup", FRONTIER §5)
1. **Prefix/KV reuse** — top daily-feel ROI; also makes prove-all/branch *interactive*.
2. **Fit planner** — range-request a GGUF header + a 30 s microbench → "runs ~22 tok/s at 32k" *before* the download.
3. **Quant-ladder receipts** — "did Q4 lobotomize your model?" measured on *your* runs (two model files, `/score` unchanged).
4. **Trust as an API field** — per-claim confidence/support spans on the wire so agents can branch on trust (ship labeled-uncalibrated first).
5. **Verify-then-escalate routing** — the big model *scores* the small model's answer (one prefill, no gen); escalate only on a bad score.

### AR × diffusion (H1–H7) — the both-substrates advantage
- **Start with §3.4** (cheap, decisive): measure **Dream→Qwen draft-acceptance rates** (`/score` already there, one afternoon). ≥~8/32 accepted → H1 (diffusion drafts, AR verifies) is real; low → H1 dies cheaply, H2–H7 survive.
- H2 AR-writes/diffusion-repairs (score-gated self-repair), H3 substrate routing, H5 span-level counterfactual patches (a new receipt type), H7 divergence atlas.

### Introspection science (X1–X8) — *gated on J-lens (J0) existing*
Research with product tie-ins; house rule: every rung ships a null control, and negative results ship as honest labels.
- **X1 — introspection receipts** (top product tie-in): score self-report vs J-lens readouts with the existing NLI judge → a per-model "self-report reliability" score. High = a trust feature; low = the honest "trust the receipts, not the story" label.
- **X7 — J-anchored legible memory** (biggest transformation): memory as a sparse bag of nameable J-directions → "what did you learn?" becomes a *lookup*, confabulation structurally impossible. Measures the interpretability tax on a real 7B memory. Lab-only, parallelizable.
- X3 (injected-thought detection → free legible concept dials), X6 (CoT-as-paging / workspace-occupancy meter, gated on J4=yes), X4/X2/X8, and **X5 (convergence archaeology — ⚠️ time-sensitive**: pre-July-2026-cutoff models age out; needs no lens).

### Model portability (MODEL_SUPPORT.md)
- **Tier-1 dial sweeps** per hero (Qwen3-14B, Gemma-3-12B) — automated sweep + an LLM-judge curation pass.
- **J-lens as the model-agnostic Tier-2 brain viz** — fit a lens per model (cheap), replacing the SAE gate. Gemma-3 also gets free GemmaScope SAEs.
- Pull the hero models (Gemma-3-12B / Qwen3-14B) and smoke-test — the victory lap now that Tier-0 is done.

### Inspector / UI leftovers (CURRENT_UI_BACKLOG.md)
- ✅ **branch-lineage-tree** (shipped `810539b` — client-side tree from parent_run_id) · ✅ **capture-final-prompt** (shipped `fc3b2ec` — persists the exact rendered prompt).
- *In flight:* final-prompt **display** in the inspector + a full-family **`/runs/<id>/lineage`** endpoint (past the 80-run cap).
- Remaining: persist-concept-spans, studio-lab-mode, tiny-test-harness.

### Housekeeping
- Push `../clozn-research` (local, yours to push). · Engine-rebuild validation on the GPU box after CMake changes.

---

## Two keystones (why the ordering)
1. **`/score` is the performance keystone, not just a receipts primitive.** The same teacher-forced batch scoring is the spec-decode verifier, the routing judge, the quant-sensitivity meter, and the context-receipt prober. We built it for honesty; it doubles as the perf roadmap's foundation.
2. **J-lens completes the model-agnostic brain viz.** Fit-in-lab / apply-forward matches clozn's substrate split; with Tier-0 done it runs on *any* GGUF. It's the read half of read-(J-lens)-plus-prove-(receipts), and it's a RACE.

## Honesty invariants (non-negotiable — the house style)
- **Lens-blind ≠ absent; agreement ≠ introspection; a linear lens always outputs something** — every readout rung carries a null (shuffled lens / shuffled pairing).
- **Negative results ship as labels or scope-bounds**, never buried. Report the whole eval set, no cherry-picking.
- **No claim outruns its evidence** — see FRONTIER_BETS §6 (the honesty ledger of claims we must NOT make yet: uncalibrated "confidence", cross-model counterfactuals as the model's own, speedup numbers before the experiment, "only runtime that CAN", etc.).
- Discrimination/detection framing only — never "awareness", never "consciousness".
