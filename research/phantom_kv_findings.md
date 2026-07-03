# phantom_kv — training-free gisting via a trainable synthetic KV cache (findings)

**Question.** The studio's prompt-mode memory pays ~63 context tokens on EVERY call for the compiled
card block. Can that cost be paid ONCE offline — distilling the block's effect into k trainable
"phantom" past-key-values entries per layer (post-projection K/V, warm-started from the real block's
own cache, backbone frozen) — so inference carries the memory at ZERO context tokens? Distinct from
gisting (Mu et al. 2023) / AutoCompressor / ICAE: those fine-tune the model or train a general
compressor; this trains nothing but 28×2 small KV tensors against ONE fixed block, at test time.

**Setup.** Qwen2.5-1.5B-Instruct bf16, greedy. Memory block = `compile_prompt_block` over 3 traits
(baking/space = concept, concise = rule) = 63 raw tokens / 52 cache positions. Distillation: KL(with-
phantom ‖ with-full-block) on the teacher's own 40-token answer spans over 8 training prompts
(disjoint from eval), Adam lr 0.02, grad-clip 2.0, per-tensor norm cap 60, keep-best, 200 steps.
Eval: `self_audit_gap`'s 6 HELDOUT probes + its objective scorers (keyword-hit for concepts, token
count for concise). Arms: no-memory floor / full-block ceiling / phantom k∈{4,8,16} / RANDOM-phantom
null (k=16, teacher-scale, untrained). Rig: `research/phantom_kv.py`; receipts:
`research/runs/phantom_kv_qwen1p5b.json` (+ `phantom_kv_smoke.json`). Full run ≈ 9 min on the 5080.

## Result — expression on 6 held-out probes, zero context tokens for every phantom arm

| arm | ctx tok | baking | space | concise (mean tok) | KL→ceiling | coherence (eyeball) |
|---|---|---|---|---|---|---|
| floor | 0 | 0.000 | 0.000 | 68.0 | — | clean |
| **ceiling (full block)** | **63** | **0.333** | **0.333** | **60.3** | 0 | clean; restates the block |
| random-null k=16 | 0 | 0.000 | 0.000 | 87.8¹ | 12.81 | **garbage** (CJK/unicode salad) |
| phantom k=4 | 0 | 0.333 | 0.333 | 60.7 | 0.351 | readable, doubled-word glitches |
| phantom k=8 | 0 | 0.333 | 0.500 | 63.0 | 0.291 | worse repetition ("have have have have") |
| phantom k=16 | 0 | 0.500 | 0.500 | 73.5¹ | 0.211 | fake-dialogue leakage ("Human:") |

¹ token counts inflated by degeneration (null: unbounded rambling; k=16: hallucinating the user's
next turn) — NOT verbosity. Do not read the concise column as a clean dose measure.

Warm-start alone (pre-training diagnostic) under-expresses: space 0.167/0.0/0.0 across k. Training
is what buys expression. Degeneration receipts over 3 samples/arm: doubled-words 0/0 (floor/ceiling)
vs 3/3/1 (k=4/8/16); fake-dialogue 1 at k=16; non-ASCII 138 chars in the null.

## Verdict on the headline question (E3: does k=8–16 recover ≥80% of ceiling expression at 0 tokens?)

**CONCEPTS: yes on the metric — k=4 already matches the ceiling (0.333/0.333) and k=8/16 exceed it
(0.5/0.5) — but the honest verdict is "yes, with a coherence tax."** The baking/space content in
phantom replies is genuine and semantically appropriate ("baking a new recipe", "space-themed
treats"), not keyword-stuffing; but every phantom arm carries measurable degeneration artifacts the
ceiling does not have (doubled words at all k; at k=16, the model continues the dialogue as
"Human:"). Fractions >1.0 (k=16 recovers "150%" of ceiling) are over-expression — the phantom pushes
topics harder than the teacher — the same shape as an over-dosed steering vector, not faithful
reproduction.

**THE CONCISE RULE: unmeasurable in this setup — the CEILING itself barely expressed it** (68.0 →
60.3 tokens, a 7.7-token effect; the historic concise result was 72→19 with the raw rule as a system
message). `compile_prompt_block`'s soft framing ("use it naturally") under-fires the rule at 1.5B —
consistent with the prompt-vs-prefix A/B note that block wording is softer instruction pressure than
the raw rule. With a near-zero denominator the "fraction recovered" is noise (k=4 "0.95", k=8 "0.65",
k=16 "-0.71"). No claim made either way about rule transfer.

## Violated / overturned expectations (pre-registered in the rig header)

- **E1 partially violated:** predicted ceiling ≥ phantom; got phantom ≥ ceiling on the keyword metric
  (over-expression). Floor ≈ null held; all trained phantoms ≫ both.
- **E2 held cleanly:** KL→ceiling monotone in k (0.351 → 0.291 → 0.211) and ~40–60× below the random
  null (12.81). The null proves the win is the training + warm-start, not "some vectors in the cache."
- **E3:** predicted concepts clear 80% by k=16 and concise does NOT — concepts cleared it by k=4
  (more optimistic than predicted); concise turned out UNMEASURABLE (ceiling effect too small), which
  is a different failure than predicted (teacher under-expression, not fused-rep distortion).
- **E4 partially violated:** predicted warm-start keeps phantoms coherent at small k — even k=4 shows
  doubled-word artifacts, and degeneration worsens with k (opposite of "large k = more capacity =
  cleaner"). The 6th instance in this repo of style/expression metrics needing a coherence axis.
- **New, unregistered finding:** held-out KL→ceiling got slightly WORSE after training at every k
  (0.325→0.351, 0.283→0.291, 0.191→0.211) while train-KL collapsed ~100× (0.85→0.008). The
  distillation OVERFITS the 8 training prompts; warm-start compression already carries the held-out
  distributional fidelity, and training converts it into expression, not fidelity.
- **Mechanical note:** the norm cap (60) rescaled the warm-start K tensors hard (mean per-layer K
  norms 142/203/288 for k=4/8/16 → capped to ≤60 from step 0), so training operated on a
  direction-preserving but 2.4–4.8× norm-compressed version of the real cache slice.

## Interpretation

1. **The mechanism is real:** k=4 phantom entries per layer (57K params total, GPU-seconds to train)
   carry the concept content of a 63-token block at zero marginal context cost, and the random null
   proves it's the distillation doing it. The cost asymmetry the rig was built to test is confirmed:
   52 prefill positions saved per call, forever, for a one-time ~3-minute training run.
2. **"Don't fuse" survives, in softened form:** the phantom is a fused representation and shows the
   family signature — over-expression (no dose control), coherence glitches, and rule/process content
   failing to make it through (here masked by teacher under-expression, but nothing suggests concise
   transferred). Concepts port; the texture degrades. Matches memory_scaling + voice_middle.
3. **Product frame:** phantom-KV sits between prompt mode (legible, per-call cost, clean) and the TTT
   prefix (opaque, zero-cost, degradation-prone). It inherits the prefix's failure modes at lower
   training cost. If shipped, it needs the same receipts regime: per-model dose/coherence receipts,
   never self-narration.

## Caveats (louder than the wins)

- **One model (1.5B bf16), one family, one seed, greedy decode, 6 probes, 3 saved samples/arm.** The
  degeneration-vs-k trend rests on small N. No 7B validation (the 1.5B→7B pass has flipped verdicts
  in this repo before — see scale_pass_7b).
- **The ceiling is weak on these held-out probes** (concepts 0.333 = 2/6 probes) — "recovering the
  ceiling" is a low bar here; a stronger teacher (raw-rule system message) would raise it and might
  widen the phantom-teacher gap.
- The keyword scorer counts ANY mention; over-expression and appropriateness are only separated by
  eyeball. The concise column is contaminated by degeneration length (flagged in-table).
- Train prompts (8) and answer spans (40 tok) are small; the overfit finding says more/varied prompts
  is the first thing to try if held-out fidelity matters.
- The soft-prefix rival arm (`--soft-rival`) was not run (time-boxed); the phantom-vs-input-embedding
  comparison is open.
