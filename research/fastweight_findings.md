# Glass-box fast-weight memory on a frozen autoregressive LM

*Roadmap Phase 4 §rung-1 — "legible, editable in-model memory (fast-weights)."*
*Run date 2026-06-22. Substrate: GPT-2-small (124M), FROZEN, via transformer_lens HookedTransformer.*
*Spike: `inspector/spikes/p15_fastweight.py`. Raw numbers: `inspector/runs/p15_fastweight.npz`.*

## TL;DR — the verdict

**It works, and it is legible + editable by construction — with one load-bearing caveat: the
addressing must be sharpened.** A frozen GPT-2-small that knows none of 12 made-up facts
(mean P(answer|cue) = **1.15%**, **0% top-1**) recalls them after we add a glass-box associative
store as an explicit list of `{key, value, eta, label}` entries injected by a hook at one mid-layer.

| addressing | recall top-1 | recall P(ans) | off-target top-1 (specificity) |
|---|---|---|---|
| baseline (no memory) | **0.0%** | 1.15% | — |
| naive dot-product `eta·(k·k')` | 41.7% | 16.7% | 2.3% |
| **hard top-1 (sharpened)** | **91.7%** | **49.8%** | **0.8%** |

(numbers at the best layer, **L=8** of 12; L=6 is close behind at 83.3% top-1.)

- **Recall lift:** 0.0% → **91.7%** top-1 (top-1 mode, L=8); P(answer) 1.15% → 49.8%. Unambiguous.
- **Specificity (the honesty control):** querying fact *i* with only a **wrong** fact in memory
  raises P(ans_i) to **1.12%** vs the 1.15% baseline — i.e. **no spurious recall**. Off-target
  top-1 is 0.8% (top1) / 2.3% (dot). The store is genuinely keyed, **not** a global bias.
- **Editability — delete:** removing an entry drops its recall **55.9% → 5.9%** (baseline 1.4%)
  while every other entry's recall is **bit-identical** (49.20% → 49.20%). Clean and surgical.
- **Editability — reweight:** the clean per-entry dose-response is **monotone**:
  P(ans) = 1.2 / 15.1 / 50.3 / 90.8 / 97.3% across η× ∈ {0, 0.5, 1, 2, 4}.
- **Legibility:** **12/12 (100%)** of stored values decode to their intended answer through the
  logit lens — true by construction, since value = the answer's unembedding direction.
- **The caveat:** with **naive dot-product addressing**, recall is real but weaker (41.7% top-1)
  because GPT-2's MLP activations are **not orthogonal across facts** — keys cross-talk. Sharpening
  the addressing (unit-normalize keys + hard top-1, or a softmax) fixes it. We report **both**.

**Verdict: a glass-box fast-weight memory stores, recalls, stays legible, and edits cleanly on a
frozen AR model — provided addressing is sharpened.** The naive raw-dot substrate works but is
selectivity-limited; that limit is the honest finding, and it is curable without leaving the glass box.

## Why this argues for keeping the glass box (not fusing into weights)

The naive `dot` mode is *exactly* what a fused weight delta gives:
`(Σ_i η_i·v_i·k_iᵀ) · k' = Σ_i η_i·(k_i·k')·v_i`. So In-Place-TTT's literal `W_down += η·v·kᵀ` can only
ever reach the **dot-mode** number (41.7% top-1) — the cross-talk is baked into the linearity and is
inseparable from a single fused ΔW. The sharpened modes (softmax 75% / hard top-1 92%) are **only
possible because the memory is kept as an explicit list**: selecting/reweighting per entry by
key-similarity is a *nonlinear* addressing step that no single linear weight delta can represent. So the
glass box is not just for inspectability — it is **functionally stronger** than the fused weight it stands
in for. **Implication for the engine rung (Roadmap 4.2/4.3):** keep the explicit entry list as the memory
of record and inject via an addressing hook; do **not** fuse the delta into `W_down`. The thing that makes
it legible is the same thing that makes it recall better.

## What we actually did

The mechanism, reduced to its smallest testable core (In-Place test-time-training / "fast-weights"):
a frozen LM can store a new association as a low-rank delta on one MLP down-projection,
`W_down += η·v·kᵀ`, where **k** = the MLP post-activation at the fact's answer position (the *key*,
dim d_mlp=3072) and **v** = a target residual direction (the *value*, dim d_model=768). Querying with
a similar activation k' returns `η·(k·k')·v` added to that layer's output — a literal key→value store.

The glass-box twist: **we never fuse the delta into the weights.** The memory is an explicit, editable
Python list of `{key, value, η, label}` entries; recall is a forward hook that adds
`Σ_i w_i·v_i` to the residual stream at layer L, where `w_i` is the per-entry addressing weight.

1. **FACTS (step 1).** 12 nonce "cue → single-token answer" facts (e.g. *"The secret color of
   Zorbland is" → " blue"*), distinct made-up subjects + common single-token answers
   (colors/numbers/common nouns). We auto-verify each answer is **one** GPT-2 token and **drop**
   any the base model already knows (top-1, or P≥30%). All 12 survived; **baseline mean
   P(ans) = 1.15%, top-1 = 0.0%, top-5 = 33.3%** (the top-5 hits are generic common-word noise,
   not the association — no fact was actually known). This is the chance floor every later number
   is measured against.
2. **WRITE (step 2).** One forward pass over `"cue answer"`; grab the MLP post-activation
   (`blocks.{L}.mlp.hook_post`) at the **answer** position as the key. Value = the answer token's
   **unembedding column** `W_U[:, ans_id]` — the legible choice: adding c·v to the residual promotes
   that token via the logit lens. η auto-calibrated so a self-recall contribution ≈ a strong steer.
3. **READ (step 3).** Query each cue (no answer); a hook captures the query post-activation k' at the
   final position and adds the memory contribution at `blocks.{L}.hook_resid_post`. We compare four
   addressing modes: `dot` (raw `η·(k·k')`), `cos` (unit-normalized), `softmax` (sharpened soft top-1),
   `top1` (hard nearest-key). Measured P(ans), top-1, top-5 **with memory vs the no-memory baseline**.
4. **SPECIFICITY (step 4, mandatory control).** For every ordered pair (i, j), query fact *i*'s cue
   with **only fact j** in memory; the diagonal (i=j, right fact present) is on-target recall, the
   off-diagonal (wrong fact present) must stay at baseline. A memory that raises the answer for *every*
   query is a bias, not a store — this catches it.
5. **EDITABILITY (step 5).** (a) Delete entry 0 → its recall must fall to baseline, others unchanged.
   (b) Reweight η over {0, 0.5, 1, 2, 4}× → dose-response of P(answer), reported two ways (see below).
6. **LEGIBILITY (step 6).** Logit-lens each stored value (`ln_final` then `W_U`) → does the top token
   read out as the intended answer?

Layers L=6 and L=8 of GPT-2's 12 were both tested. Backbone frozen throughout (`requires_grad_(False)`).
Runs in `.venv-sae` (transformer_lens + torch); GPT-2-small is tiny so CPU is fine (the SAE venv is
CPU-torch on this machine; the spike auto-selects cuda if present).

## The numbers (L=8, the best layer)

**Recall, all addressing modes (baseline P=1.15%, top-1=0.0%, top-5=33.3%):**

| mode | P(ans) | top-1 | top-5 |
|---|---|---|---|
| dot (raw substrate) | 16.7% | 41.7% | 91.7% |
| cos | 12.4% | 33.3% | 83.3% |
| softmax (β=30) | 37.8% | 75.0% | 91.7% |
| **top1 (hard)** | **49.8%** | **91.7%** | 91.7% |

Even the raw dot product lifts top-5 from 33% → 92% and top-1 from 0% → 42%: the store *is* recalling.
Sharpening just removes the cross-talk that keeps the wrong-keyed competitors in contention.

**Specificity (querying fact i with one WRONG fact j in memory):**

| mode | on-target top-1 (right fact present) | off-target P(ans_i) | off-target top-1 |
|---|---|---|---|
| dot | 58.3% | 1.33% (vs 1.15% base) | 2.3% |
| **top1** | **100.0%** | **1.12% (vs 1.15% base)** | **0.8%** |

Off-target recall sits **at the baseline** — the store does not leak. (In top1 mode, on-target hits
100% here vs 91.7% in the full-memory recall, because single-fact memory removes all competing
contributions.)

**Editability — delete (top1):** deleted Zorbland→blue. Fact 0 P(ans) 55.9% → **5.9%** (baseline 1.4%);
every other fact's P(ans) **49.20% → 49.20%** (unaffected to the digit). Surgical.

**Editability — reweight (η sweep × {0, 0.5, 1, 2, 4}):**

| | 0× | 0.5× | 1× | 2× | 4× |
|---|---|---|---|---|---|
| P(ans), all-η dot (scales cross-talk too) | 1.2% | 39.7% | 38.6% | 38.1% | 37.8% |
| **P(ans), clean per-target (top1, target only)** | **1.2%** | **15.1%** | **50.3%** | **90.8%** | **97.3%** |

The **clean per-entry dose is monotone** and saturates toward 1.0 — exactly a dial on one fact's
strength. The all-η dot curve is non-monotone (peaks at 0.5× then dips) **because scaling every η also
amplifies the cross-talk from non-target entries**, which erode P(target). That non-monotonicity is a
property of raw-dot addressing, not of the memory primitive — reported here rather than hidden.

**Legibility:** 12/12 values decode to their answer through the logit lens (' blue', ' green', ' gold',
…). 100% nameable, by construction.

## L=6 vs L=8

Nearly identical story; L=8 is slightly stronger. L=6: top1 recall **83.3%**, off-target top-1 0.0%,
delete restores (42.85% → 1.56%), clean dose monotone (1.2/11.1/40.4/89.7/97.5%), 100% nameable.
L=8: top1 recall **91.7%**. Both mid-layers carry a clean associative store; **L=8 is the pick.**

## Capacity / scaling (p16) — the honest gate

Rung-1 worked at N=12. Before building anything on top, we asked the question that could kill it: **does
recall survive as the store fills?** `inspector/spikes/p16_capacity.py` mints 200 clean programmatic facts
(nonce names × 72 single-token answers), reuses p15's mechanism verbatim, and sweeps N ∈ {5,10,20,50,100,
200} at L=8 — recall-vs-N for each addressing mode, each beside the no-memory baseline (0.0% top-1
throughout) **and a shuffled-key null** (keys permuted across entries: if real recall ≈ shuffled, the
addressing isn't doing work).

**Recall top-1 vs N (baseline 0% throughout; shuffled-key null beside):**

| N | dot | dot-shuf | softmax | top1 | top1-shuf |
|---|---|---|---|---|---|
| 5   | 20% | 20% | 40% | **60%** | 20% |
| 10  | 10% | 10% | 30% | **50%** | 0% |
| 20  | 10% | 5%  | 5%  | **40%** | 10% |
| 50  | 4%  | 4%  | 2%  | **20%** | 0% |
| 100 | 2%  | 2%  | 2%  | **11%** | 2% |
| 200 | 3%  | 2%  | 0.5%| **6%**  | 1.5% |

**Verdict: a small-N device, not a scalable store.**
- **`dot` (= the fused-weight equivalent) has no real capacity at any N** — it equals its shuffled-key null
  at every N (20/20, 10/10, 4/4, 2/2, 3-vs-2). Whatever lift it shows is a global value-bias, not keyed
  retrieval. A single fused ΔW therefore has ~zero associative capacity on this substrate — a second,
  independent confirmation of "don't fuse."
- **`top1` (hard nearest-key list addressing) is the only mode doing genuine work** — it beats its shuffled
  null at every N (+4.5 pts even at N=200) so it never fully collapses to "no keying" — **but it decays
  steeply**: 60→50→40→20→11→6%. Usable only at small N. `softmax` dies by N≈20; `dot` is at baseline by N≈50.
- **False-recall (top-1 = a *different* stored fact's answer) rises monotonically** as the store fills
  (top1: 20%→83%; dot: 60%→97%) — the direct fingerprint of mis-addressing.

**The deeper finding — it's a key-*distinctiveness* wall, not a raw-count wall.** Rung-1's hand-picked
DIVERSE facts hit 92% top-1 at N=12; p16's programmatic facts (drawn from 16 templates, so many share a
carrier and differ only in a nonce subject) hit ~40-50% at N=10-20. The gap is the tell: the **key** is the
MLP post-activation at the *answer* position, which is dominated by the **template/syntactic context**, not
the (upstream) subject — so two facts sharing a template have near-colliding keys *regardless of N*. The
ceiling is set by how distinctive the facts' answer-position activations are, not by N per se; in realistic
use facts share structure, so the effective ceiling is low. **The lever (next rung, p17): a better,
training-free key** — whitened/orthogonalized keys, a dedicated key projection, or keying on a more
fact-distinctive position (the subject token) — rather than the raw `mlp.hook_post`. That decides whether
the fast-weight direction scales past a working-memory handful.

## Honesty notes / controls / caveats

- **Baseline beside every number.** Recall, specificity, delete, and dose all print the no-memory
  baseline inline. The lift (0.0% → 91.7% top-1) is interpretable because the floor is measured, not
  assumed.
- **Specificity is the load-bearing control and it passes.** Off-target recall at baseline (1.12% vs
  1.15%) is the evidence this is a *keyed store*, not a global "say the answer" bias. The η=0× column
  (1.2% = baseline) is the built-in null: with weight zero, the hook does nothing, as it must.
- **We report the raw substrate, not just the tuned one.** The naive dot-product result (41.7% top-1)
  is the honest behavior of GPT-2's non-orthogonal MLP activations; sharpening to top-1 (91.7%) is the
  fix. Both are in the table. Do not quote only the 91.7%.
- **Legibility is true by construction, so it is not evidence of anything emergent.** We *chose*
  value = unembedding direction. The non-trivial results are recall + specificity + clean editability
  *on top of* that legible value — those are what the glass-box claim rests on.
- **Aggregate + spread reported.** Per-fact tables show the full distribution; one fact (Flonkville→
  seven) recalls weakly under top1 at both layers — number tokens have idiosyncratic unembed geometry.
  No cherry-picking: the headline is the mean over all 12.

## Limits / what this rung does NOT yet show

- **One layer, one position, single-token answers.** This is the smallest testable core. Multi-token
  answers, multi-fact composition in a single query, and write-position robustness are untested.
- **Sharpened addressing needs a query.** Hard top-1 picks the nearest stored key; it presumes the
  query activation lands nearest the right key. Here it does (100% on-target with the right fact
  present), but at larger scale (hundreds of facts, overlapping cues) selectivity will be re-tested —
  the dot-product cross-talk seen here is the early warning.
- **No training, by design.** This validates the *substrate* (frozen LM + explicit delta list). It does
  not claim the model learns *when* to write; that is a later rung.

## Reproduce

```
cd inspector
# .venv-sae python; gpt2 is cached, no large download
python spikes/p15_fastweight.py                 # full battery, L=6 and L=8
python spikes/p15_fastweight.py --layers 4,6,8  # other layers
```
