# Consolidation sidecar on a real frozen LLM (`sidecar_real.py`)

*The "truly learn" rung: does the toy sidecar's rule-consolidation (`research/sidecar.py`) transfer from*
*toy embeddings to a REAL frozen model's representations? Run 2026-06-22. Qwen2.5-0.5B + 3B, frozen.*

## Verdict

**The consolidation machinery transfers to real LLM features, fully, for the headline. But the mod-cipher
task is too easy to prove the deep claim, and the one place real-model geometry shows through is a
*weakness*. PARTIAL transfer, and the result reframes what to test next.**

## Setup

- Models: Qwen2.5-0.5B-Instruct (workhorse) + Qwen2.5-3B (confirmation), FROZEN. Env: the **lab `.venv`**
  (torch 2.11+cu128, RTX 5080); `.venv-sae` is CPU-only and was not used. Plain `transformers` +
  `output_hidden_states` with the backbone frozen.
- Task: mod-N cipher over number tokens (N=12), mirroring the toy. Sidecar arch identical to `sidecar.py`,
  but `feat()` = the cached frozen-LLM mid-layer activation for a token.
- Tractability: harvest each token's activation ONCE, cache it, meta-train on the cache.
- **Harvesting gotcha (load-bearing):** a number token fed ALONE has cosine ~1.000 to every other token at
  mid layers (Qwen's position-0 attention-sink / massive-activation drowns token identity). Fix: tap inside
  a carrier context (`"The number {w}"`), which restores distinct features (offdiag cosine ~0.67-0.81). This
  is the same massive-activation structure we hit in the SAE work.

## Headline (untaught-input accuracy; chance = lookup = 0.083)

| | sidecar | lookup | ICL ceiling | legibility probe |
|---|---|---|---|---|
| Qwen-0.5B (layers 4-20, K 1-5) | **1.000 ± 0.000** | 0.083 | ~0.30 | **1.000** |
| Qwen-3B (layers 8-24) | **1.000 ± 0.000** | 0.083 | ~0.44 | **1.000** |

Perfect untaught generalization on real features, flat across layers and seeds (zero variance). Persistence:
zero examples in any prompt at query time, the rule lives entirely in the consolidated state `s`. Legibility:
a linear probe reads the secret shift `b` at 1.000, and the mean state per `b` lays on a clean circle (PCA).

## The controls that reframe the win (stated louder than the win)

1. **The mod task is saturated.** The sidecar hits 1.000 on real, gaussian-random, AND one-hot features
   alike; only *collapsed* (near-identical) features fail to chance. So the task tests "are the tokens
   separable?", not "does the model's specific geometry carry the rule." Real features clear the bar but are
   not special at it. "Works on a real model" is true and a LOW BAR.
2. **Permutation negative control** (an arbitrary mapping with no algebraic rule): sidecar untaught = 0.087
   ~ chance, at all K. This proves the mod win is genuine rule-extraction, not lookup leakage or an
   artifact: structure present (mod) -> extracted; structure absent (permutation) -> correctly fails. The
   harness reports failure honestly.
3. **Real geometry shows through as a fragility.** Under feature noise, real features collapse to chance
   while gaussian survives, on both 0.5B and 3B. Cause: Qwen's number tokens are anisotropic (offdiag cosine
   ~0.7 -> a large shared component, a small per-number residual), so isotropic noise swamps the
   distinguishing signal. A real, reproducible property of LLM number-token geometry, and a weakness.

## What this means / next rung

- **Machinery validated:** the sidecar survives real, anisotropic, high-dim features (collapsed fails, so it
  is not trivial). Enough to build on.
- **NOT yet shown:** that the model's own *learned understanding* carries a transferable, consolidatable
  rule. The mod cipher only needed token distinctness. To make it bite, the next task's rule must be
  **unsolvable from labels alone**: a *semantic* mapping / analogy / relation over real word tokens, or a
  continuous regression, where success requires the model's learned geometry rather than separable tokens.
  The permutation control shows the harness will honestly report failure, so it is trustworthy for that push.

Deliverables: `research/sidecar_real.py`; `research/runs/sidecar_real{,_3b}.json`; SVGs (generalization-vs-K
and `b`-on-a-circle, Maiko palette) in `research/runs/`.
