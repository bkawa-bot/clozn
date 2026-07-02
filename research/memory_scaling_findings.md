# memory_scaling — prompt vs internalized (soft-prefix) memory under fact load (findings)

**Question.** The falsifiable crux left by the self-audit thread: the naive defense of internalized memory
claims "prompts win only at low volume; a trained internal state wins at scale." Kill-test it: scale the
fact load N ∈ {4, 16, 64} and compare retrieval accuracy for `none` (guess control) / `prompt` (all facts
as a system prompt) / `prefix16` (the studio mechanism, m=16 TTT soft prefix) / `prefix64` (4× capacity —
separates capacity-failure from mechanism-failure).

**Pre-registered prediction** (from the project's own `fastweight_findings` "don't fuse" result): the fused
prefix saturates early; the prompt stays near ceiling; **no crossover in range**. (Mid-run wobble recorded
for honesty: prefix16 hit 0.938 at N=16 and briefly looked like it might hold.)

**Setup.** Qwen2.5-1.5B-Instruct bf16. 128-fact bank (16 attribute types × 8 relations, distinctive
single-word values). Prefix TTT: Adam, minibatch 16, steps 120/200/300 by load, norm cap 14·√(m/16),
keep-best. Fairness: prefix trains on ONE question phrasing; **held-out phrasing** is the honest column
(encode-the-fact, not memorize-the-string). Objective scoring: value-word in reply; eval ≤24 facts/load;
greedy. Repro: `python research/memory_scaling.py`.

## Result — held-out-phrasing retrieval accuracy

| N | none | prompt | prefix16 | prefix64 | prompt ctx (tok/call) |
|---|---|---|---|---|---|
| 4 | 0.0 | 1.000 | 0.75 | 1.000 | 55 |
| 16 | 0.0 | 1.000 | 0.938 | 0.812 | 173 |
| 64 | 0.0 | **0.958** | **0.333** | **0.500** | 762 |

**No crossover.** The prompt is at/near ceiling across the whole range (762 tokens of facts cost a 32k-ctx
model nothing). The fused prefix collapses at N=64, and **4× capacity only partially rescues it**
(0.333 → 0.500): the failure is substantially *mechanism*, not just capacity.

**The diagnostic tell — train 0.917 vs held-out 0.333/0.500 at N=64.** Under load the prefix stops encoding
facts and memorizes trained QA strings. And its held-out errors are *interference*, not absence:
- "my dog's name?" → **"Nimbus"** (that's the *boss's* dog; wanted Zephyr)
- "my sister's dog's name?" → **"Nimbus"** again (wanted Biscuit)
- "sister's favorite book genre?" → "'fic' (fantasy)" (wanted cyberpunk — blur/confabulation)

Cross-talk between entities is exactly the fused-superposition failure mode. The model knows *dog names
exist in memory*; it no longer knows *whose dog is which* — and it answers **confidently wrong**, echoing
the fastweight finding that fused-memory failures are "confident wrong-fact retrievals."

## Interpretation

1. **"Don't fuse" — third independent confirmation, new setting.** `fastweight_findings` showed it for a
   fused weight vs an explicit list (which recalls **91.7% at N≥200**); the studio hit it as "baking drowns
   out dogs"; now the TTT soft prefix shows it under fact load. The project's two memory mechanisms sit on
   opposite sides of the line: **explicit/addressable scales, fused blobs interfere.**
2. **The naive "internalize to scale" defense is dead in the tested range.** Prompts win at low AND medium
   volume for facts. What survives of the thesis: (a) *process/skill* internalization — a prompt can't fix
   the introspective blindness, and behaviours aren't retrieval; (b) *structured* internal memory (Titans-
   style slots — the explicit-list side of don't-fuse); (c) the **cost asymmetry** — prompt reads cost O(N)
   tokens *every call* (55→762 here; extrapolated ~12k tokens at N≈1000) vs a constant prefix, and prefix
   writes cost real GPU time (25s→260s here). Untested beyond N=64; noted, not claimed.
3. **Product guidance (studio):** trait cards (N≈1–8, behavioural) are inside the prefix's working range —
   measured 0.75–1.0 here — so the current design is fine *for its use case*. But **route facts to explicit
   storage** (prompt/retrieval or a slot memory); do not consolidate a fact-store into the prefix.

## Caveats (louder than the wins)

- **One model (1.5B), one seed, one TTT recipe.** A better consolidation recipe (more steps, curriculum,
  paraphrase-augmented targets) would lift prefix numbers some; the interference signature is unlikely to
  vanish but the exact collapse point (16→64) is recipe-dependent.
- **Scoring undercounts slightly:** exact-word match misses morphology — "peonies" was scored a miss for
  want-"peony" (a correct answer). True prefix numbers are a few points higher; does not change the verdict
  (≈0.54 vs 0.958 is still a rout).
- **Retrieval-of-facts only.** This says nothing about style/behavioural traits (the prefix's actual job in
  the studio) — those were measured in the self-audit thread, not here.
- Prompt dilution may exist at much larger N or smaller ctx models — untested here; at 762/32k tokens there
  was none to find.
