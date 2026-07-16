# Clozn Roadmap

The consolidated map: the thesis, what's shipped, and where the open work lives.

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
- **Product/lab split** — one Torch-free product gateway (`clozn serve`) + a private C++ worker;
  PyTorch is lab-only, enforced by a `product-minimal` CI lane; lab owns its handler via an
  injectable substrate with zero product-global mutation. Landed 2026-07-16.

## 🔭 What's next

The forward-looking backlog — every open item, reconciled against what actually shipped and
source-tagged to its origin doc — now lives in a single tracker:

→ **[docs/BACKLOG.md](BACKLOG.md)**

This section used to duplicate that list, but most of it had already shipped (prefix/KV reuse,
fit planner, quant-ladder receipts, trust-as-an-API-field, verify-then-escalate, the introspection
pack, the J-lens ladder). See **BACKLOG.md §0** for the full "already done" ledger and **§1–§4** for
open work: refactor close-out · runtime → production beta · research frontier · product/UX polish.

---

## Two keystones (why the ordering that got us here)

1. **`/score` is the performance keystone, not just a receipts primitive.** The same
   teacher-forced batch scoring is the spec-decode verifier, the routing judge, the
   quant-sensitivity meter, and the context-receipt prober. Built for honesty; it doubles as
   the perf roadmap's foundation.
2. **The J-lens completes the model-agnostic brain viz.** Fit-in-lab / apply-forward matches
   clozn's substrate split; with Tier-0 done it runs on *any* GGUF. It's the read half of
   read-(lens)-plus-prove-(receipts).
