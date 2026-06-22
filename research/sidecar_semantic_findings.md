# Semantic-relation consolidation on a real frozen LLM (`sidecar_semantic.py`)

*The "truly understands" rung. `sidecar_real.py` proved the consolidation MACHINERY works on real LLM*
*features, but its mod-cipher task was a LOW BAR — gaussian-random and one-hot features ALSO scored*
*1.000, because a cipher only needs distinct tokens. This rung asks the deep question: does the model's*
*OWN learned understanding carry a transferable, consolidatable, legible RULE, with a task that is*
*UNSOLVABLE from token-distinctness alone? Run 2026-06-22. Qwen2.5-0.5B + 3B, frozen, lab `.venv` (cu128, RTX 5080).*

## Verdict

**PARTIAL — and the split is the finding.** Real features carry the *relation itself*: the consolidated
state is **perfectly legible** (a linear probe names which of the 7 relations it is at **1.000**, chance
0.143) and the read lands on a word *of the correct relation type* **~69% of the time** — and this needs
**real** geometry (every geometry-blind control is at chance for the same job). That is a genuine step past
the mod-cipher low bar: random/one-hot features **cannot** even identify the relation from an unseen word.

**But the model's geometry does NOT, through this sidecar, let it APPLY a one-to-one relation to a novel
word.** Exact held-out retrieval is **~0.10** overall and **0.000** on all five 1-to-1 relations
(antonym, plural, past, comparative, country→capital); only the **many-to-few** relations
(hypernym→0.60, object→color→0.14) clear the bar — and those are partly the *same* token-distinctness
low bar wearing a semantic mask (few distinct answers, so "right cluster" ≈ "right answer"). The frozen
model **can** do every one of these relations in-context (native ICL **0.94–0.99**), so the information is
present and accessible — **the bottleneck is the consolidation/read (a single mean-pooled state + a small
MLP), not the features.** Real ≫ controls on *knowing the relation*; real ≈ controls on *applying a 1-to-1
relation to an unseen word*. The deep claim is **half true**, and the harness localizes exactly which half.

## Setup

- **Models:** Qwen2.5-0.5B (24 layers, H=896) and Qwen2.5-3B (36 layers, H=2048), **frozen**. Plain `transformers`
  + `output_hidden_states`. (Qwen2.5-7B-Instruct was **not run** — not in the local HF cache; it would
  need a ~15 GB download, and the bottleneck below is architectural, not feature-richness, so scale was
  spent on 0.5B-vs-3B instead.) Env: lab `.venv` (torch 2.11+cu128, RTX 5080); `.venv-sae` untouched.
- **Task — semantic-relation consolidation.** 7 hand-curated real word-pair relations, every word a
  **single token** in Qwen2.5 BPE (verified by a harvest-time assert): antonym (29 pairs), plural (30),
  past-tense (24), comparative (25), country→capital (20), object→color (24), hypernym (31). 183 pairs.
  An episode = one relation: show K teaching pairs `(x, R(x))`, consolidate a state `s`, predict `R(x')`
  for a held-out `x'`.
- **NO LEAKAGE (verified).** Each relation's pairs are split 70/30 TRAIN/TEST. Meta-train the sidecar only
  on TRAIN pairs; evaluate only on HELD-OUT TEST pairs. Asserted: TRAIN/TEST pair sets are disjoint per
  relation, **and** a TEST query word never appears as a TRAIN input word in the same relation — the
  held-out word is genuinely unseen as a stimulus. All controls are evaluated on the **same** held-out
  test pairs (this is the candidate-set-leakage double-check the rung demands).
- **Scoring — RETRIEVAL (the load-bearing choice).** `read(x', s)` emits a vector; we score every
  candidate word `c` in a shared vocabulary **V** (|V| = 152 distinct output words) by cosine similarity to
  a projection of `feat(c)`, predict argmax, correct iff = `R(x')`. **Chance = 1/|V| = 0.0066.** This
  forces the answer through feature geometry, so random features genuinely fail. A classification head over
  V is reported as a cross-check (retrieval is primary).
- **Arch (reused from `sidecar_real.py`).** `write: s = mean_i write_mlp([proj·feat(x_i), proj·feat(R(x_i))])`;
  `read: read_mlp([proj·feat(x'), s]) → vector`, cosine-scored against `key·feat(c)`. write/read
  meta-learned across random relation-episodes; **LLM frozen**; features cached once via a carrier context
  (`"The word {w}"`) to dodge Qwen's position-0 attention-sink artifact (the same fix as the prior rung).
- **Honest floors beside every number:** chance (1/|V|); a **frequency-prior floor** (a geometry-blind
  model that guesses the most frequent answer — global and per-relation-majority); a **lookup** baseline
  (chance on held-out by construction); and the **native-ICL ceiling** (frozen model, K pairs in the
  prompt as text, retrieval-scored over the same V). Aggregated over 3 seeds; full layer sweep; K∈{1,2,3,5}.

## Headline — held-out untaught generalization (retrieval acc; chance = 0.0066)

### Qwen2.5-0.5B (best layer L12 of 24; H=896; 3 seeds; 15k steps)

| metric | K=1 | K=2 | K=3 | K=5 |
|---|---|---|---|---|
| **real** features | 0.108 | 0.107 | 0.105 | 0.105 |
| gaussian-random | 0.013 | 0.011 | 0.013 | 0.013 |
| one-hot | 0.008 | 0.007 | 0.007 | 0.009 |
| collapsed | 0.020 | 0.021 | 0.017 | 0.021 |
| freq-prior floor | 0.018 | 0.018 | 0.018 | 0.018 |
| **native-ICL ceiling** | 0.940 | 0.991 | 0.986 | 0.994 |

Real (~0.105) beats every geometry-blind floor (best control 0.021, freq-floor 0.018) by ~5×, **but sits
far below its own model's ICL ceiling (~0.99).** Flat across K and across layers (sweep L6/L12/L18 →
0.08/0.12/0.06). Classification-head cross-check ≈ 0.10 (same regime — the task is genuinely hard for the
sidecar, not a retrieval artifact). Persistence: zero teaching examples in any prompt at query time.

### The diagnostic that explains the headline (Qwen2.5-0.5B, K=5)

| | value | reading |
|---|---|---|
| relation probe (s → which of 7 relations) | **1.000** | the relation IS fully consolidated & legible |
| cluster acc (prediction is *some* answer of the right relation) | **0.687** | the read knows the relation type |
| exact retrieval (full menu) | **0.105** | …but rarely the exact held-out target |
| within-relation retrieval (menu restricted to that relation) | **0.116** | even given the relation, can't pinpoint |

High cluster (0.69) + low exact/within (0.11) is the whole story: **the sidecar consolidates *which
relation* but cannot reconstruct an applicable one-to-one map.** When it errs it emits a *seen* answer of
the right type ("slow"→"faster", "Italy"→"Paris", "walk"→"talked") instead of computing the unseen target.

### Per-relation held-out retrieval (Qwen2.5-0.5B, K=5; real vs gaussian vs freq-floor)

| relation | distinct answers | real | gaussian | freq-floor |
|---|---|---|---|---|
| antonym | 29 | **0.000** | 0.000 | 0.000 |
| plural | 30 | **0.000** | 0.000 | 0.000 |
| past | 24 | **0.000** | 0.000 | 0.000 |
| comparative | 25 | **0.000** | 0.000 | 0.000 |
| capital | 20 | **0.000** | 0.000 | 0.000 |
| color | 11 | 0.140 | 0.094 | 0.143 |
| hypernym | 14 | **0.599** | 0.000 | 0.000 |

The only real win that is *not* explainable by a frequency prior is **hypernym (0.599 vs 0.0 floor)** — a
many-to-few mapping where landing near the category cluster = getting the answer. **color** (0.140) is at
its own frequency floor (0.143): with ~8 colors, guessing the commonest is as good as the sidecar. The
five 1-to-1 relations are flat at 0.000 for real and controls alike. So real's aggregate edge over the
controls is carried almost entirely by hypernym + the relation-cluster effect, **not** by applying a
relation to a novel word.

### Qwen2.5-3B (best layer L12 of 36; H=2048; 3 seeds; 15k steps)

| metric | K=1 | K=2 | K=3 | K=5 |
|---|---|---|---|---|
| **real** features | 0.105 | 0.103 | 0.106 | 0.106 |
| gaussian-random | 0.030 | 0.028 | 0.030 | 0.033 |
| one-hot | 0.015 | 0.015 | 0.016 | 0.019 |
| collapsed | 0.019 | 0.023 | 0.018 | 0.018 |
| freq-prior floor | 0.018 | 0.018 | 0.018 | 0.018 |
| **native-ICL ceiling** | 0.914 | 0.991 | 0.997 | **1.000** |

**Scale does not change the verdict — it sharpens it.** Diagnostic (K=5): relation probe **1.000**,
cluster **0.725** (↑ from 0.5B's 0.687 — bigger geometry knows the relation *better*), exact **0.106**,
within-relation **0.124**. Per-relation: the five 1-to-1 relations are **0.000** for real; **hypernym 0.634**
(vs gaussian 0.079) is the one clean real win; **color 0.111** sits at its frequency floor (0.143, and
gaussian actually edges real here at 0.155 — pure frequency, no geometry). Native ICL reaches **1.000** at
K=5. So 3B's richer features lift *knowing the relation* (cluster 0.69→0.73) and the in-context ceiling
(→1.000), **but the 1-to-1 exact retrieval stays pinned at zero** — direct evidence the bottleneck is the
single mean-pooled state + read, not feature richness or model scale.

### Both sizes, side by side

| | real @K5 | best control | cluster (knows-rel) | within-rel | relation probe | ICL @K5 | hypernym (real) |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B (L12) | 0.105 | 0.021 | 0.687 | 0.116 | 1.000 | 0.994 | 0.599 |
| Qwen2.5-3B (L12) | 0.106 | 0.033 | 0.725 | 0.124 | 1.000 | 1.000 | 0.634 |

Near-identical. The story is **scale-invariant**: relation consolidated + legible at both sizes; 1-to-1
application fails at both; the model can do it in-context at both (3B perfectly).

## Why this is the honest result (controls stated louder than the win)

1. **The discriminating controls work exactly as designed.** On the *cipher* (prior rung) gaussian and
   one-hot tied real at 1.000. **Here they collapse**: gaussian/one-hot/collapsed cannot exceed the
   frequency floor on held-out words, because for an unseen `x'` a random/orthogonal feature carries no
   structure to compute `R(x')`. Confirmed on the **same** held-out test pairs as real (the candidate-set
   leakage check). Real is the only bank that can even identify the relation from an unseen word.
2. **Real's edge is real but narrow.** Real beats the controls on *knowing the relation* (probe 1.000,
   cluster 0.69, both needing real geometry) — a true advance past the cipher low bar. Real does **not**
   beat the controls on *applying a 1-to-1 relation* (0.000 = floor on five of seven relations). The two
   "successes" (hypernym, color) are many-to-few mappings, i.e. partly the token-distinctness low bar
   again. We report this rather than headline the 0.69 cluster number.
3. **The ceiling proves it is a sidecar limit, not a model limit.** Native ICL nails every relation
   (0.94–0.99, including plural/past/capital at ~1.00). The information to apply each relation to a
   held-out word is present in the frozen model and reachable in-context; the **mean-pooled state +
   small read MLP** cannot extract an applicable 1-to-1 transform from it. Robust to read-head design
   (separate-key, shared-projection, and additive-offset reads all fail the 1-to-1 relations) and to
   training length/capacity (30k steps and 2× width do not move the 1-to-1 zeros).

## What this means / next rung

- **Validated, again and harder:** the model's geometry carries the *relation* in a legible,
  consolidatable form that random features cannot fake. The legibility result (name the relation from `s`
  at 1.000; relations separate cleanly in a PCA of `s`) is the strongest "what meaning is in the memory"
  evidence so far.
- **NOT shown — the deep claim in full:** that a single consolidated state lets the model *apply* an
  arbitrary learned 1-to-1 relation to an unseen input. It can recall *which* relation and solve
  many-to-few relations; it cannot reconstruct a 1-to-1 map the way it trivially does in-context.
- **The diagnosis points the next rung at the read, not the features.** Candidates: (a) a read that
  predicts a *transformation* applied to `feat(x')` rather than a free vector (the additive variant already
  nudged plural off zero); (b) a richer state than a single mean-pooled vector (slots / a small set);
  (c) letting the read attend back over the teaching pairs (retrieval-augmented apply) — which would
  approach ICL but keep persistence. The harness honestly reports 0.000 where structure is not extracted,
  so it is trustworthy for that push.

Deliverables: `research/sidecar_semantic.py`; `research/runs/sidecar_semantic_{0p5b,3b}.json`; Maiko-palette
SVGs in `research/runs/`: `sidecar_semantic_genK_{0p5b,3b}.svg` (real vs controls vs ICL across K),
`sidecar_semantic_perrel_{0p5b,3b}.svg` (per-relation real vs gaussian vs freq-floor),
`sidecar_semantic_legib_{0p5b,3b}.svg` (relations separating in PCA of `s`, with probe accuracy).
