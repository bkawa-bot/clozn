# Concept-indexed (legible-by-construction) memory on a frozen autoregressive LM

*Roadmap Phase 4 §4.2 — the complement to the fast-weight store (`fastweight_findings.md`).*
*Run date 2026-06-22. Substrate: GPT-2-small (124M), FROZEN, via transformer_lens HookedTransformer.*
*Spike: `inspector/spikes/p18_conceptmem.py`. Raw numbers: `inspector/runs/p18_conceptmem.npz`.*

## The idea

The fast-weight store (p15/p16) holds *arbitrary* `key→value` facts and is capacity-bound by **key
collisions** in the frozen model's activations. Concept-indexed memory takes the opposite tack: the
memory's **state is a vector of coefficients over a fixed, NAMED concept basis**. It is legible by
construction (the state literally reads `animals=+2.0, formal=+1.0`), editable (change a named
coefficient), and has **bounded, named capacity** (= number of concepts) with no arbitrary-key
addressing problem. The cost: it can only hold what you have a named concept for. It is a *"stance /
context in named terms"* memory, complementary to open-vocabulary fact recall.

## Setup

8 named concepts — `animals, colors, money, fear, formal, past, question, food` — each a **diff-in-means**
direction (training-free): `d_c = mean(resid_post @ L over positive sentences) − mean(over contrast
sentences)`. The memory realizes state **c** by adding `Σ_i c_i·d_i` to `resid_post` at layer L (the
**raw** diff-in-means direction, so `c=1` = one natural concept direction — the standard steering scale; a
unit-norm scale under-doses and falsely reads as "dead"). Behavior is read with **transparent token-set
probabilities** (e.g. P(animal-word tokens); log-odds(formal vs casual markers) for the relational
concepts), never a learned classifier. The BOS position (a ~30× norm outlier) is excluded from injection
and from the basis. Layers 6/7/8 scanned; L=8 best, L=7 default. Backbone frozen throughout.

## Results (best layer L=8)

**(a) Writing a named coefficient shifts generation in the named way — for 7/8 concepts.** `animals,
colors, fear, formal, past, question, food` all show clean dose-response over c ∈ {0,1,2,4,8}, each
**beating an equal-norm random-direction null** (the control that rules out "any big enough push moves the
readout"). **`money` fails** — its readout barely moves and does not beat the null. Reported, not hidden.

**(b) Read-back is faithful downstream.** Same-layer projection is an exact linear inverse (diag = c, corr
+1.00) — reference only. The meaningful test projects the residual back onto the basis **after 3 more
layers** (basis rebuilt there): overall write-vs-read **corr +0.92–0.97**, per-concept +0.99/+1.00. The
legible state *survives* the nonlinear layers; magnitude attenuates (a written 4 recovers as ~2–3, mean
err ~1.9) — the honest wash-out from downstream processing, not a loss of identity.

**(c) Legible + editable by construction.** The coefficient vector *is* the state; editing a named
coefficient changes behavior monotonically. No discovery step, no addressing — you index by name.

**(d) Interference is low but real, and the basis cosine does NOT predict it.** At L=8, **7/8 readouts are
clean** (respond to their own coefficient, not others). The honesty catch: a *row-only* interference
metric mislabeled the **relational** readouts (`money`, `past`) as clean at L=6/L=7; a **column-aware**
metric correctly flags them **FRAGILE** (dragged by any large perturbation, not just their own concept).
The concept×concept **cosine** matrix correlates ~0 with behavioral interference — cosine bounds the
*read-back* leakage, but behavioral entanglement is dominated by **readout fragility**, a different thing.
Both metrics reported explicitly.

## Verdict

Concept-indexed memory **is a clean legible-by-construction memory**: bounded + named capacity, no
arbitrary-key collision, faithful read-back where the basis is near-orthogonal. It is **complementary** to
the fast-weight store, not a replacement:

| | fast-weight (p15/p16) | concept-indexed (p18) |
|---|---|---|
| stores | arbitrary open-vocab facts (key→value) | strength of a **named** concept |
| capacity | small-N, bound by **MLP-key collisions** (model-fixed) | = #concepts, bound by **basis cosine** (a **design knob**) |
| legibility | value logit-lenses to the answer (by construction) | the coefficient vector *is* the state (by construction) |
| addressing | nearest-key lookup (cross-talks as N grows) | index by name (no addressing) |
| failure mode | wrong-key recall as the store fills | a few fragile readouts; "money" doesn't steer |

The key structural advantage: concept-indexed interference is bounded by something **we choose** (pick a
more orthogonal concept set), unlike fast-weight keys fixed by the model. The cost is the obvious one —
it only remembers in terms you've named in advance.

## Honesty notes

- **Equal-norm random-direction null beside every dose-response** — the load-bearing control; a concept
  "works" only if its own readout rises *and* beats the null. `money` does not, and is called out.
- **The round-trip that counts is downstream**, not the trivially-exact same-layer inverse.
- **Two interference metrics, the stricter one reported** — the row-only metric flattered the relational
  readouts; the column-aware metric is the honest one (`money`/`past` are fragile).
- **Calibration is disclosed:** injection uses the *raw* diff-in-means scale (the standard steering
  magnitude), not unit-norm; the equal-norm null controls for absolute scale, so beating it is the real
  signal regardless of the chosen magnitude.

## Reproduce

```
cd inspector
# .venv-sae python; gpt2 is cached, no large download
python spikes/p18_conceptmem.py                 # full battery, layer scan 6/7/8
python spikes/p18_conceptmem.py --layer 8       # single layer
```
