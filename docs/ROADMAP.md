# Clozn Roadmap

The consolidated map: what's done, the v1 cut, what's next.

> **The thesis:** as hardware commoditizes capability, **control becomes the product** — trust,
> steer, version, debug. Clozn's wedge is what a text-in/text-out runtime structurally can't do.

## ✅ Done

- **Engine white-box runtime** — AR + diffusion GGUF on the C++ `cloze-server`: `/harvest`,
  `/score`, `/apply_template`, steer taps, prompt-mode memory.
- **Reproduce & prove** — teacher-forced `/score`, the SDK/substrate seam, rich per-token trace
  (`token_id` / `logprob` / top-k entropy) + reproducibility metadata, forced-mode receipts +
  deterministic rederive, and the graded-leaning inspector UI. A null-floor experiment killed the
  planned "silent influence" badge (filler-swap can't discriminate — Pearson 0.9985 against the
  null); graded per-card leaning via co-present leave-one-out shipped instead.
- **Tier-0 model-agnosticism** — clozn runs **any AR GGUF** across the whole white-box stack
  (proven tap-by-tap on Llama-1B). Engine-side chat templating; the model is derived from the
  engine's own `/health`.
- **Engine-native J-lens** — the lens is fitted offline (PyTorch, autograd), then applied
  forward-only by the C++ engine on the GGUF's own final-norm + quantized head. Validated in
  stages: the HF-fitted lens transfers to the engine's activations essentially losslessly
  (cross-position consistency *and* semantic recovery, both far above a proper null); the C++
  apply matches the numpy oracle at 96–99% top-1, with every disagreement a near-tie inside
  head-quantization noise. Served at `POST /jlens`; surfaced in the Run Inspector as per-token
  "disposed to say" chips with an unskippable provenance caption.
- **Outcome-grounded calibration** — `clozn eval`: Brier / ECE-vs-truth / risk–coverage on a
  labeled probe set, plus a selective-generation policy (answer / ask / abstain) that reports
  both sides of the trade. Served as the TRUTH tier beside the journal's acceptance-proxy curve.

## 🎯 v1 — "causal receipts + J-lens readouts"

*The local runtime with causal receipts and J-lens readouts: see what your GGUF is disposed to
say, per token, and prove what changed its answer.*

- Turn on engine sampling for the serving path (currently gated).
- Docs/claims refresh — every headline claim traced to a measurement.
- "Does a 7B have a J-space?" — the workspace-existence probe; launch content either way.

## 🔭 Post-v1 backlog

### Performance — all reuse the `/score` keystone
1. **Prefix/KV reuse** — top daily-feel ROI; also makes prove-all and branching interactive.
2. **Fit planner** — range-request a GGUF header + a 30 s microbench → "runs ~22 tok/s at 32k"
   *before* the download.
3. **Quant-ladder receipts** — "did Q4 lobotomize your model?" measured on *your* runs
   (`clozn quant-check`, already wired; needs a free GPU pass).
4. **Trust as an API field** — per-claim confidence/support spans on the wire, so agents can
   branch on trust (labeled-uncalibrated first).
5. **Verify-then-escalate routing** — a big model *scores* the small model's answer (one
   prefill, no generation); escalate only on a bad score.

### AR × diffusion — the both-substrates advantage
- First, the cheap decisive test: measure diffusion→AR draft-acceptance rates (`/score` already
  does the verification). Good rates make "diffusion drafts, AR verifies" real; bad rates kill
  it cheaply. Follow-ons: score-gated self-repair, substrate routing, span-level counterfactual
  patches, a divergence atlas.

### Introspection science — gated on the J-lens
House rule: every experiment ships a null control, and negative results ship as honest labels.
- **Introspection receipts** — score a model's self-report against its J-lens readouts; a
  per-model "self-report reliability" score. High = a trust feature; low = the honest "trust
  the receipts, not the story" label.
- **J-anchored legible memory** — memory as a sparse bag of nameable lens directions, so "what
  did you learn?" becomes a lookup rather than a self-report.
- Injected-thought detection (free legible concept dials), workspace-occupancy metering, and
  the rest of the introspection ladder.

### Model portability
- Per-model dial sweeps for the hero models (Qwen3-14B, Gemma-3-12B) — automated sweep + a
  judge-curated pass.
- **J-lens as the model-agnostic brain viz** — fit a lens per model (cheap), replacing the SAE
  gate; Gemma also gets free public SAEs.

### Inspector / UI
- Final-prompt display in the inspector; a full-family `/runs/<id>/lineage` endpoint.
- Persist concept spans; a studio lab mode.

---

## Two keystones (why this ordering)

1. **`/score` is the performance keystone, not just a receipts primitive.** The same
   teacher-forced batch scoring is the spec-decode verifier, the routing judge, the
   quant-sensitivity meter, and the context-receipt prober. Built for honesty; it doubles as
   the perf roadmap's foundation.
2. **The J-lens completes the model-agnostic brain viz.** Fit-in-lab / apply-forward matches
   clozn's substrate split; with Tier-0 done it runs on *any* GGUF. It's the read half of
   read-(lens)-plus-prove-(receipts).
