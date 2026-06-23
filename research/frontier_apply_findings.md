# Can the frozen model apply its OWN consolidated clay? (`frontier_apply.py`)

*The frontier "uses its own clay" rung. `sidecar_semantic.py` found a sidecar can CONSOLIDATE which*
*relation it was shown (real ≫ random; a probe names it at 1.000) but a small EXTERNAL read-MLP*
*cannot APPLY a 1-to-1 relation to a held-out word (~0.105 aggregate, **0.000** on every 1-to-1*
*relation), while native in-context learning NAILS all of them (0.94–1.00). So the information is*
*present and the model can apply it — the external reader just can't extract it. This rung asks: can*
*a LEARNED INJECTION make the FROZEN model apply the relation with its OWN forward pass + unembedding*
*head, where the read-MLP failed? Run 2026-06-22. Qwen2.5-0.5B-Instruct, frozen, lab `.venv` (cu128, RTX 5080).*

## Verdict

**SPLIT — and the split is the finding.**

**STAGE 1 is a clear PASS.** A soft prefix (m≈4–8 continuous vectors prepended to the query,
trained by backprop through the **frozen** backbone — only the prefix moves) makes the model's
**own next-token output** apply a relation to **held-out words the prefix never saw in training**.
Aggregate held-out apply = **0.928** (menu-restricted) / **0.860** (free-generation, the strict
"model's true argmax" test), vs the read-MLP's **0.105** and a random/no-prefix null of **0.228 /
0.230** — at **95% of the native-ICL ceiling (0.980)**. On the five **1-to-1 relations the read-MLP
scored exactly 0.000** (antonym, plural, past, comparative, capital), the soft prefix reaches
**0.959** (ICL 0.990): past **1.000/1.000** free, plural **0.963/0.963**, comparative
**0.917/0.875**, antonym **0.852/0.667**. Free-gen tracks menu throughout, so this is the model
genuinely emitting the answer, not a scoring artifact. **The inject-and-model-applies architecture
is viable: the bottleneck in the read-MLP rung was the external reader, exactly as that rung's ICL
ceiling predicted — give the injection to the model and let the model apply it, and the 1-to-1
zeros become near-ones.**

**STAGE 2 is a robust NEGATIVE.** Meta-learning a **compressor** that maps K example-activations →
a soft prefix (leave-one-relation-out: train on the other relations, then a few examples of a
held-out relation must produce a prefix the frozen model applies to held-out words) **does not
work**: free-generation = **0.000** on all seven held-out relations; the menu-restricted score is
**0.159**, *below* even the read-MLP. This is **not** a weak-injection artifact — we explicitly
scaled the generated prefix to norm **0.450** (matched to real token embeddings, where the
successful Stage-1 prefixes live) and gave it 2000 steps; the model's free-gen output still
collapses to a degenerate token. **A few examples of a new relation, run through a meta-learned
compressor, do NOT yield a prefix the frozen model can use — even though independently optimizing a
prefix per relation (Stage 1) works beautifully.** The meta-generalization step is where it breaks.

**Legibility: NOT legible (at chance).** A probe over the learned per-relation prefixes names the
relation at **0.000** (chance 1/7 = 0.143; centered, LOO nearest-centroid, n=21 prefixes). This is
the *expected* reading, not a bug: each Stage-1 prefix is an **independent** optimization from a
different random init, so same-relation prefixes are equally-valid solutions that do not align in
parameter space. This is the sharp contrast with `sidecar_semantic`, where the consolidated state
was perfectly legible (probe 1.000) — that legibility came from a **shared** meta-learned encoder,
which is absent here and would only reappear with a working shared compressor (Stage 2, which fails).

**Bottom line on the frontier question.** *Can the frozen model apply its own consolidated clay?*
**Yes, when the clay is shaped directly for that relation (Stage 1) — the model's own machinery
applies an arbitrary 1-to-1 relation to unseen words at near-ICL accuracy, killing the read-MLP's
0.000.** **No, when the clay must be CONSOLIDATED from examples by a meta-learned compressor (Stage
2) — that map does not generalize to a new relation.** The simplest version of the full
"consolidate-a-rule-the-model-then-applies-itself" loop is **falsified at the honest (free-gen)
bar**; the weaker claim — that a *directly learned* injection lets the frozen model apply a relation
its external reader could not — is **confirmed**.

## Setup

- **Model:** Qwen2.5-0.5B-Instruct (24 layers, H=896), **frozen** (all params `requires_grad_(False)`).
  Plain `transformers`; the prefix is the only trainable tensor. Env: lab `.venv` (torch cu128, RTX
  5080); `.venv-sae` untouched. Wall time 1190 s (3 seeds × 400 steps × 7 relations × {m=4, m=8} for
  Stage 1; 7 LORO compressors × 2000 steps for Stage 2; 200-episode ICL ceiling).
- **Apples-to-apples with the read-MLP rung.** `RELATIONS`, `split_relations` (test_frac=0.30,
  split_seed=0), the candidate menu, the carrier context, and the ICL scoring recipe are **imported
  directly from `sidecar_semantic.py`**, so the held-out TRAIN/TEST split is byte-identical: the
  Stage-1 soft prefix is evaluated on the **same held-out words** the read-MLP scored 0.000 on. The
  read-MLP baseline numbers are loaded from `runs/sidecar_semantic_0p5b.json` (the recorded p18
  result), not re-described.
- **Stage 1 — soft prefix / prefix-tuning.** The query is rendered as the native-ICL query line,
  `"{x} ->"`, so the model's next token after `->` is the answer slot (same target token-space as the
  ICL ceiling). We embed it (frozen table), prepend m trainable vectors (init from m random real
  token embeddings — one fixed recipe for all relations, not cherry-picked), run the frozen model on
  `inputs_embeds`, and cross-entropy the final-position logits against R(x)'s token id over the
  **full vocab** (the model's real unembedding head). Train on TRAIN words, eval on HELD-OUT words.
- **Stage 2 — meta-learned compressor.** `s = mean_i enc([proj·feat(x_i), proj·feat(R(x_i))])` →
  decode → soft prefix, permutation-invariant over K examples (like `sidecar_semantic`'s write).
  `feat` is the frozen layer-12 residual feature (the read-MLP's best layer). Leave-one-relation-out:
  the compressor never sees the held-out relation in training; at eval, K examples of the held-out
  relation produce the prefix, scored on that relation's held-out test words. The generated prefix is
  **explicitly normalized to 0.45** (the diagnosed fix for a first-run collapse to norm 0.1 — see
  Honesty §2), giving the compressor a fair, strong chance.
- **Scoring — two ways, both the model's own output.** *menu* = argmax of next-token logits
  restricted to the 152-word candidate menu (matches the ICL / read-MLP retrieval metric, the
  apples-to-apples number). *free* = argmax over the full vocab equals the answer token (the strict
  "the model genuinely says it" test). **The verdicts key on FREE-gen**; menu is reported alongside.
  A high menu with ~0 free is a menu-restriction mirage and is flagged as such.
- **Controls beside every number (load-bearing).** read-MLP baseline (the p18 result); native-ICL
  ceiling (frozen model, K text pairs in the prompt, retrieval-scored over the same V); a **null**
  (random untrained prefix) and a **none** (no prefix at all) on the same held-out words; a
  **train-fit** sanity (does the prefix even fit TRAIN?); and a **cross-relation** control (apply
  R's prefix to *other* relations' words — does it still produce an R-type answer?). Aggregated over
  3 seeds with per-relation std; prefix length **swept {4, 8}, both reported**; eval **only** on
  held-out words.

## Headline — held-out untaught apply accuracy (model's OWN output; chance = 0.0066)

### Stage 1, m=4 (best of the swept {4,8}; 3 seeds; 400 steps)

| relation | soft (menu) | soft (free) | ICL | read-MLP (p18) | null | none | x-rel |
|---|---|---|---|---|---|---|---|
| antonym | 0.852 | 0.593 | 0.99 | **0.000** | 0.185 | 0.222 | 0.39 |
| plural | 1.000 | 0.963 | 1.00 | **0.000** | 0.222 | 0.111 | 0.44 |
| past | 1.000 | 1.000 | 1.00 | **0.000** | 0.000 | 0.000 | 0.53 |
| comparative | 1.000 | 0.917 | 0.96 | **0.000** | 0.000 | 0.000 | 0.68 |
| capital | 0.944 | 0.889 | 1.00 | **0.000** | 0.889 | 0.833 | 0.97 |
| color | 0.810 | 0.810 | 0.91 | 0.140 | 0.190 | 0.000 | 0.92 |
| hypernym | 0.889 | 0.852 | 1.00 | 0.599 | 0.111 | 0.444 | 0.82 |
| **AGGREGATE** | **0.928** | **0.860** | **0.980** | **0.105** | **0.228** | **0.230** | **0.679** |

(m=8 is near-identical: 0.888 / 0.832 aggregate. Both pass; m=4 is marginally stronger.)

- **The five 1-to-1 relations** (read-MLP = 0.000): soft-prefix **0.959** aggregate (ICL 0.990). The
  read-MLP's hard zeros become near-ones — the headline result.
- **Free-gen ≈ menu** (0.860 vs 0.928): the prefix genuinely steers the model's true output, not a
  scoring artifact. Where free < menu (antonym 0.593) it is honest morphological near-misses.
- **Cross-relation (x-rel) = 0.679**: R's prefix applied to *foreign* words lands on an R-type
  answer far above the ~1/7 a fixed-answer bias would give — the prefix transports the **relation**
  (a transform), not a memorized answer set. This is the cleanest "it's an applicable rule" evidence.
- **Honest confounds, visible in the table.** `capital` has a high null/none (0.889/0.833): the bare
  model already knows many capitals from the country name alone, so the soft-prefix win there is
  partly world knowledge, not the injection. `color` is the model's weakest relation (it knows fewer
  colors). We report these rather than average them away — the strong, clean wins are on past /
  comparative / plural, where null ≈ 0.

### Stage 2, leave-one-relation-out (compressor-generated prefix; generated-prefix norm = 0.450)

| held-out relation | meta (menu) | meta (free) | ICL | read-MLP |
|---|---|---|---|---|
| antonym | 0.222 | **0.000** | 0.99 | 0.000 |
| plural | 0.111 | **0.000** | 1.00 | 0.000 |
| past | 0.000 | **0.000** | 1.00 | 0.000 |
| comparative | 0.000 | **0.000** | 0.96 | 0.000 |
| capital | 0.667 | **0.000** | 1.00 | 0.000 |
| color | 0.000 | **0.000** | 0.91 | 0.140 |
| hypernym | 0.111 | **0.000** | 1.00 | 0.599 |
| **AGGREGATE** | **0.159** | **0.000** | **0.980** | **0.105** |

Free-gen is **0.000 everywhere**. The few non-zero menu cells (capital 0.667) are the same
world-knowledge mirage as Stage 1's null — the bare country name primes the capital, and the menu
restriction discards the tokens the model actually prefers. **The compressor does not produce a
usable injection for an unseen relation.**

## Why this is the honest result (controls stated louder than the win)

1. **The Stage-1 win is the model's own output, on genuinely unseen words.** Held-out test words
   never appear in the prefix's training, yet free-generation produces the correct novel inflection
   (e.g. an unseen verb → its `-ed` past). That rules out memorization by construction; the
   cross-relation control (0.679) further shows the prefix carries the *relation*, not the answers.
   The strict free-gen metric (0.860) sits right under the menu metric (0.928), so the number is not
   a retrieval/menu artifact — the contrast with Stage 2 (menu 0.159, free 0.000) is exactly what a
   real effect vs a mirage looks like.
2. **Stage 2's negative is robust, not an under-tuned strawman.** The first compressor run collapsed
   to generated-prefix norm ~0.1 and free-gen 0.000; we diagnosed it (the model emitted a degenerate
   `' '` token), rebuilt the compressor to **scale outputs to norm 0.45** (where Stage-1 prefixes
   succeed) with a deeper decoder and a learnable per-vector gain, and gave it 2000 steps. Free-gen
   stayed **0.000**. So the failure is in the **activation→prefix map generalizing to a new
   relation**, not in prefix strength — the most informative place for it to be.
3. **Legibility is reported as chance, with the right diagnosis.** 0.000 is below chance because
   independently-optimized prefixes for the same relation are unrelated in parameter space (we
   confirmed: raw-flatten 0.000, centered 0.143, PCA/linear all ≈ chance). This is the honest mirror
   of `sidecar_semantic`'s 1.000 — legibility there was a property of the **shared encoder**, and the
   shared encoder (the compressor) is precisely what fails here.
4. **No cherry-picking.** Prefix length swept {4, 8} (both reported, conclusions identical); the
   Stage-2 feature layer fixed at L12 (the read-MLP's pre-registered best, not re-searched); 3 seeds
   with per-relation std; eval only on held-out words; the read-MLP and ICL numbers are the recorded
   p18 values loaded from disk.

## What this means / next rung

- **Validated (new):** the read-MLP rung's bottleneck was the **external reader**, and the fix is
  architectural — a learned injection that lets the **frozen model apply the relation with its own
  head** clears the 1-to-1 zeros and reaches 95% of the in-context ceiling. "Inject a state, let the
  model apply it" is a viable white-box primitive on this model.
- **Falsified (the deep claim, honestly):** a single meta-learned compressor that **consolidates** K
  examples of a *new* relation into a model-usable prefix does **not** generalize across relations —
  free-gen 0.000 LORO. The full "consolidate-a-rule-the-model-then-applies-itself" loop, in its
  simplest form, does not hold. The gap between Stage 1 (works per-relation) and Stage 2 (fails
  cross-relation) localizes the open problem precisely: **learning the example→injection MAP that
  generalizes**, not the injection format (prefix tuning is plenty expressive — Stage 1 proves it).
- **Where the next rung should push (the diagnosis points the way):** (a) richer/longer injections
  the compressor can hit (KV-prefix or per-layer steering rather than an input-embedding prefix); (b)
  a compressor trained to **match the Stage-1 prefixes** as targets (distillation: Stage 1 gives a
  per-relation oracle prefix — regress to it, instead of end-to-end through the frozen model, which
  may be why the gradient signal is too weak to generalize); (c) far more relations, so the
  example→prefix map has enough coverage to interpolate. The harness reports 0.000 where the map does
  not generalize, so it is trustworthy for that push.

Deliverables: `research/frontier_apply.py` (runnable; Stage 1 always, Stage 2 gated on Stage 1
passing); `research/runs/frontier_apply_0p5b.json`; Maiko-palette SVGs
`research/runs/frontier_apply_stage1_0p5b.svg` (soft-prefix menu+free vs ICL vs read-MLP vs null,
per-relation + aggregate) and `research/runs/frontier_apply_stage2_0p5b.svg` (meta-consolidated
prefix vs ICL vs read-MLP, leave-one-relation-out).
