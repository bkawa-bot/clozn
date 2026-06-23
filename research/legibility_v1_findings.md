# Can a test-time-learned rule be made LEGIBLE? (`legibility_v1.py`)

*The first legibility experiment for Clozn. Direct sequel to `frontier_apply.py` / `frontier_apply_v2.py`*
*(READ those + their `_findings.md` first). Run 2026-06-22. Qwen2.5-0.5B-Instruct, FROZEN, lab `.venv`*
*(torch cu128, RTX 5080). Synchronous, single process (169.8 s); `.venv-sae` untouched.*

## Where we are

`frontier_apply_v2.py` lever 3 established that **test-time adaptation (TTT) works**: for a NEW
(held-out) relation, fit a soft prefix by a few (~20–30) gradient steps on its own examples — backprop
through the frozen model, only the prefix moves — and the frozen Qwen applies the rule to **held-out
words** at ~0.94, its own output, near the in-context ceiling. **But the learned prefix is an opaque
blob.** A relation-probe over learned prefixes was at chance; lever 2's apparent 0.99 "legibility" was
an input-feature artifact (an untrained map scored the same). Application is bought; legibility is not.
This file asks: **can the learned thing be made readable** — in words (idea 3) or by name (idea 1)?

## Verdict

**Idea 3 (self-report + verify): NEGATIVE, cleanly. The TTT-learned rule applies (0.944) but the model
cannot STATE it in words that check out.** Best stated-rule held-out apply = **0.306** (agreement with
the adapted behavior 0.278), **not clear of an averaged wrong-rule control (0.292)**. The verification
path is sound — the oracle true-rule applied verifies at **0.769 ≫ wrong 0.292**, so instruction-
following *can* apply these rules when handed the right words — and the adaptation works (0.944). The
gap is **introspective**: with the adaptation active, the model almost always reports only *that* there
is a first→second mapping ("The first word becomes the second word", "Transformation rule"), not *which*
transformation. The unadapted ICL self-report ceiling is also low (**0.382**), so even from examples-in-
context this 0.5B mostly cannot put the rule into checkable words. **A test-time-learned rule applies
without the model being able to say what it learned.**

**Idea 1 (named sliders): the legible form WORKS a little and COSTS a lot. Legibility is real but weak;
coverage is the hard wall.** Constraining the adaptation to a basis of named diff-in-means relation
directions (learn a shared coefficient vector, inject `Σ cᵢ dᵢ` at layers 6–16, no free prefix) gives
held-out in-basis apply **0.241 free / 0.330 menu vs unconstrained TTT 0.889** — a **−0.65 accuracy
cost**. With the relation's own direction removed (out-of-basis / LORO, the coverage test) it collapses
to **0.046**: a new relation with no named direction is essentially uncovered. Legibility-by-construction
(does the largest coefficient name the relation?) = **0.192 vs a shuffled-label null of 0.000** (chance
0.038) — **genuinely above a proper null** (it survives the lever-2 trap this time), but low: only ~1 in
5 relations names itself, mean rank of the own-coefficient is **7.9 of 26**. So the named coefficient
*is* a real, readable signal, but a single linear named direction cannot reconstruct the rule the way a
free prefix can, and rules outside the basis are invisible.

**Answer to the headline question.** *Can a TTT-learned rule be made legible?* On this model, **not for
free, and not yet well.** Self-report is the most on-thesis route and it **confabulates** here (the model
knows how to *do* the rule but not how to *describe* it). The named-slider route is the honest opposite
trade: you get legibility-by-construction and editability, but you pay a large accuracy cost and you only
cover rules you already have a name for. **The applies-vs-legible tension from `frontier_apply` persists,
now measured on both axes: free TTT applies but is opaque; named sliders are legible but weak + bounded
by coverage.** A negative on the favorite (self-report) is reported plainly — the verification is the
point, and it says the words don't check out.

## Setup

- **Model & mechanism.** Qwen2.5-0.5B-Instruct (24 layers, H=896), **frozen**. The TTT prefix is the
  exact lever-3 mechanism, imported from `frontier_apply` (SoftPrefix, `forward_with_prefix`, `batch_pack`).
  Relation bank, held-out split, ICL ceiling, and the candidate menu are imported from `frontier_apply_v2`
  / `sidecar_semantic` — **byte-identical to lever 3**. 26 relations (currency dropped: <10 single-token
  pairs); held-out relation eval set (shared, LORO) = `plural, third_person, antonym2, gerund, past,
  part_of` (the same shuffled-order recipe as v2). Eval is **always on held-out WORDS** (test split,
  disjoint from the fit set) and **held-out RELATIONS**.
- **TTT adaptation (lever 3).** 30 Adam steps, lr 0.05, fit on the relation's full TRAIN words, m=8 prefix
  vectors from a tiny random init. Reaches held-out apply 0.944 menu / 0.889 free — confirming the
  recorded lever-3 result on this larger held set.
- **Idea 3 — self-report + verify.** With the adaptation **active**, the frozen model is prompted in its
  **chat format** (it is an Instruct model — gives self-report its fairest shot) to state the rule, under
  two framings: declarative ("State the transformation rule…") and metacognitive ("what rule did you just
  learn?"). The stated rule is then **verified, not trusted**: it is applied to each held-out word by the
  frozen, *unadapted* model in chat format ("Apply this rule to the word… Rule: <stated> … Answer:"),
  scored over the candidate menu. We report **stated-rule held-out apply** and **agreement** (per-word
  match between stated-rule-applied and the adapted model's own behavior).
- **Idea 1 — named sliders.** A basis of one **diff-in-means** direction per relation (conceptmem/p18
  recipe), `d_r = mean(feat(R(x)) − feat(x))` over TRAIN pairs, **raw** magnitude (the conceptmem lesson:
  unit-norm under-doses), harvested at a 6-layer band {6,8,10,12,14,16}. The adaptation is constrained to
  this basis: learn ONE shared coefficient vector `c ∈ R²⁶`, inject `Σ cᵢ dᵢ` at each layer's output via
  a forward hook (no free prefix; the bare `"{x} ->"` query). Same apply-CE TTT loss, 80 steps (K params,
  cheap). The legible state *is* the coefficient vector. Run in-basis (R's own direction present →
  legible-by-name possible) and out-of-basis/LORO (R's direction removed → coverage test).
- **Controls (load-bearing — this frontier has reversed on clean-looking wins).** Idea 3: an **ICL
  self-report ceiling** (unadapted model states the rule from the SAME examples-in-prompt — the natural
  upper bound for "can it articulate at all"), an **oracle true-rule** (a hand-written correct description
  — does instruction-following even apply it?), and a **wrong-rule** negative control **averaged over 4
  different relations' descriptions** (one arbitrary wrong rule is noisy and a word's own prior can emit
  the true answer regardless of the rule — averaging gives a stable rule-independent floor). Idea 1: the
  **out-of-basis/LORO** coverage test, and a **shuffled-label null** for the legibility read-out (relabel
  the basis directions by a fixed random permutation; if argmax-names-self survives the shuffle it is an
  artifact). Every number sits beside the **unconstrained-TTT** and the **ICL ceiling**. Per-relation
  breakdown; free-gen reported beside menu; no cherry-picking.

## Idea 3 — self-report + verify (menu-scored; held-out words + relations)

| held-out rel | adapted (TTT) | self-report (metacog) | stated→applied | agreement | ICL self-report | oracle | wrong (avg) |
|---|---|---|---|---|---|---|---|
| plural | 1.000 | *"…maps the first word to the second word"* | 0.000 | 0.000 | 0.000 | 0.778 | 0.083 |
| third_person | 1.000 | *"…that maps the first word to the second"* | 0.000 | 0.000 | 0.125 | 0.500 | 0.250 |
| antonym2 | 0.833 | *"…map the first word to the second … one-to-one"* | 0.833 | 0.667 | 0.833 | 1.000 | 0.625 |
| gerund | 1.000 | *"In one short phrase starting with a verb"* | 0.500 | 0.500 | **1.000** | 0.833 | 0.208 |
| past | 1.000 | *"In one short phrase starting with a verb"* | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 |
| part_of | 0.833 | *"In one short phrase starting with a verb"* | 0.500 | 0.500 | 0.333 | 0.500 | 0.583 |
| **AGGREGATE** | **0.944** | (best framing) **0.306** | | **0.278** | **0.382** | **0.769** | **0.292** |

(declarative framing aggregate: stated 0.222 / agreement 0.222. The table shows metacog, the slightly
better of the two.)

- **The model echoes the task format, not the content.** Under the prefix, self-reports are
  meta-descriptions of *the prompt* ("Transformation rule", "In one short phrase starting with a verb",
  "the first word becomes the second word"). The soft prefix was trained only on the answer-slot query
  `"{x} ->"`, so prepended to a chat prompt it pushes toward answer tokens and derails fluent statement.
  That is itself part of the honest finding — **an injected adaptation is not automatically a *statable*
  one.** For `plural` it even confabulated a wrong rule ("adding -tion").
- **Where the model *can* articulate, it's from examples-in-context, not from the adaptation.** The two
  genuine hits are ICL self-reports (no prefix): `gerund` → *"adding the suffix '-ing'"* verifies **1.000**
  (agreement 1.000); `antonym2` → *"reversing its meaning"* verifies 0.833. The adapted (prefix-active)
  self-report cannot match this.
- **The verification is sound and the controls behaved.** Oracle 0.769 ≫ wrong 0.292 aggregate; on clean
  relations the wrong-rule floor is genuinely low (plural 0.083, past 0.000, gerund 0.208). The two high
  wrong cells (antonym2 0.625, part_of 0.583) are exactly where the word's own prior dominates — the model
  emits the antonym / the whole regardless of the stated rule, and menu-restriction scores it right. This
  confound is visible per-relation and is why the **gap above wrong**, not the raw stated-acc, is the
  honest read — and that gap (0.306 − 0.292) is not significant.

## Idea 1 — named sliders: applies vs legible (free-gen; held-out words + relations)

| held-out rel | in-basis (free) | in-basis (menu) | out-of-basis / LORO (free) | unconstrained TTT (free) | top coeff names | self-rank |
|---|---|---|---|---|---|---|
| plural | 0.222 | 0.444 | 0.111 | 1.000 | **plural** (self) | 1 |
| third_person | 0.250 | 0.250 | 0.000 | 1.000 | plural | 2 |
| antonym2 | 0.000 | 0.167 | 0.000 | 0.833 | synonym | 5 |
| gerund | 0.500 | 0.500 | 0.000 | 0.833 | **gerund** (self) | 1 |
| past | 0.143 | 0.286 | 0.000 | 1.000 | third_person | 3 |
| part_of | 0.333 | 0.333 | 0.167 | 0.667 | gerund | 3 |
| **AGGREGATE** | **0.241** | **0.330** | **0.046** | **0.889** | | mean 7.9 / 26 |

Legibility-by-construction across all 26 relations: **argmax-names-self = 0.192** vs **shuffled-label
null = 0.000** (chance 0.038), mean self-rank 7.9 / 26.

- **Constraining to named sliders costs ~0.65 accuracy** (0.241 free vs 0.889 TTT). A single linear named
  direction (even injected across 6 layers, with the optimizer free to pick the coefficient magnitude)
  cannot reconstruct an arbitrary relation the way an m×H free prefix with full attention reach can. This
  is the expected, fair result — and a strong-form one (multi-layer, raw diff-in-means, 26-way basis).
- **Coverage is the hard wall.** Remove the relation's own direction (LORO) and apply drops to 0.046 — a
  relation with no named slider is essentially uncovered. This is the user's coverage/legibility tension,
  measured: the named basis only holds what you have a name for.
- **Legibility is real but partial — and this time it beats a proper null.** argmax-names-self 0.192 >
  shuffled-label null 0.000 means the largest coefficient genuinely tends to name the relation (unlike
  lever 2's artifact). But it is weak: only `plural` and `gerund` name themselves at rank 1; `past`,
  `antonym2`, `part_of` are pulled toward a *correlated* direction (past→third_person, the two share the
  verb-inflection geometry; part_of/antonym2→gerund/synonym). Where the named slider both **works and
  names itself** (gerund: 0.500 apply, rank 1), it is the clean legible-AND-working corner; most relations
  fall short on one axis or both.

## Why this is the honest result (controls stated louder than any win)

1. **Self-report is given its fairest shot and still confabulates.** Chat format (the Instruct model's
   native mode), two framings, a verb-first instruction — and the adapted model still reports the task
   format, not the rule. The negative is not a weak-prompt artifact: the ICL self-report ceiling (same
   examples, no prefix) is 0.382, and the oracle path proves the verifier works (0.769). The failure is
   localized to **the model articulating the injected adaptation**, the most informative place for it.
2. **Verification is decoupled from trust.** We never score the stated rule as text; we *apply* it with
   the frozen model and measure held-out accuracy + agreement. The wrong-rule control is averaged over 4
   relations to kill the single-wrong-rule noise, and the per-relation table exposes exactly where a
   word's prior inflates it. The reported signal is the **gap above wrong**, which is ~0.
3. **Idea 1's legibility beats a proper null — the lever-2 trap is avoided.** The shuffled-label null is
   0.000, so the 0.192 self-naming is genuine learned structure, not input-feature separability. We report
   it as *weak* (mean rank 7.9/26) rather than rounding up.
4. **Both routes measured against the SAME unconstrained-TTT and ICL brackets, on held-out words AND
   relations, free-gen beside menu, full per-relation breakdown, no cherry-picking.** The adaptation
   itself reproduces lever 3 (0.944) on this held set, so the comparison is anchored.

## What this means / next rung

- **The legibility gap is real and now localized on two axes.** Free TTT: applies (0.94), opaque (can't
  state it; prefixes don't align). Named sliders: legible-by-construction + editable, but −0.65 accuracy
  and bounded by coverage (LORO 0.05). Neither gives legible-AND-working-AND-general at once on this model.
- **Self-report's failure is plausibly a small-model / injection-format limit, not a law.** The model
  *can* articulate the easy rules from examples-in-context (gerund 1.000, antonym2 0.833 ICL self-report);
  it is the **prefix-active** statement that breaks, because the answer-slot prefix derails generation.
  The honest next probes: (a) a **statement-friendly adaptation** (KV-prefix / per-layer steering that
  leaves the generation head free, or a prefix trained with an auxiliary "describe-the-rule" objective);
  (b) a **bigger Instruct model** (1.5B/3B) where introspective self-report is stronger — does the verify
  gap open up?; (c) for idea 1, a **richer-but-still-named basis** (multiple directions per concept, or
  learned-then-named directions) to lift the 0.24 apply without losing the shuffled-null-beating
  legibility — and a per-relation map of which rules are *in-basis-expressible* (gerund yes, antonym2 no).
- **The verification harness is the reusable asset.** It cleanly separates "the model said X" from "X is
  true," with sound controls (oracle ≫ wrong, agreement, ICL ceiling). It reported a flat negative on the
  favorite idea without flattering it — trustworthy for the bigger-model and better-injection pushes.

## Reproduce

```
cd research   # repo: C:\Users\brigi\src\clozn ; lab .venv (GPU), SYNCHRONOUS single process
# Qwen2.5-0.5B-Instruct cached; HF_HUB_DISABLE_SYMLINKS=1 if a download is triggered
python legibility_v1.py                                  # both ideas, n_held=6 (the deliverable run)
python legibility_v1.py --ideas 3 --n_held 6             # self-report + verify only
python legibility_v1.py --ideas 1 --steer_layers 6,8,10,12,14,16   # named sliders only
```

Files: `research/legibility_v1.py`; `research/runs/legibility_v1_0p5b.json`; Maiko-palette SVGs
`research/runs/legibility_v1_idea3_0p5b.svg` (adapted vs stated-applied vs agreement vs oracle vs wrong,
per-relation + aggregate) and `research/runs/legibility_v1_idea1_0p5b.svg` (in-basis vs LORO vs TTT vs
ICL, with the argmax-names-self dots + the legible-vs-null readout).
