# slotmem_qwen — the glass-box slot memory, ported to Qwen + the three unbuilt rungs (findings)

**What.** The p15/p17/p19 explicit slot memory (the don't-fuse winner) ported from GPT-2-small to
Qwen2.5-1.5B-Instruct (bf16, layer 18/28, raw torch hooks, zero new deps), plus the rungs the spikes
never built: **surprise-gated writes**, a **confidence gate** at read, and **multi-token answers**.
Mechanism per p17: key = the cue's LAST-token residual (query-time-consistent), value = the first
answer token's unembedding direction (legible by construction), hard top-1 addressing, inject
eta·value at the query position (eta = 1.5× the layer's mean residual norm). Rig:
`research/slotmem_qwen.py`; run: `runs/slotmem_qwen1p5b.json` (12s end-to-end).

## Results (N=20 nonce facts: 13 single-token answers, 7 multi-token; baseline floor 0.000 top-1)

| phase | result |
|---|---|
| **Surprise-gated writes** (new) | **20/20 nonce written · 4/4 known facts SKIPPED** ("capital of France…" refuses storage — the Titans write-policy rung, working) |
| **Recall** | **top-1 0.90**, P(ans) 0.71 — vs GPT-2 p15's 91.7% at N=12: **parity, on a 12× larger model** |
| Shuffled-key null | **0.00** — keyed addressing, not a global bias |
| Specificity | on-target 4/4, **off-target 0/12** — no spurious recall |
| Surgical delete | victim → 0, **every other fact bit-identical** |
| Paraphrase (10 rewordings) | ungated: 9/10 right, **1 confident-wrong-fact** (the p19 disease) → **gated: 9/10 right, 0 wrong, 1 abstain** — the gate converts the exact failure mode into an abstention at zero cost |
| **Multi-token emission** (new) | single 12/13; multi 4/7 with first-token-only → **5/7 (71%) with a two-token VALUE SCHEDULE** (token-1 direction at decode step 1, token-2 at step 2, then clean) |

## Capacity sweep — the p16/p17 question, answered on Qwen

Programmatic nonce facts (place-name × attribute templates), scored p17-style (SELECT = picked its own
entry, collision-proof; EXPRESS = answer wins the logits), shuffled-key null beside every N:

| N | select | express | shuffled null |
|---|---|---|---|
| 10 | 1.00 | 1.00 | 0.00 |
| 25 | 1.00 | 0.96 | 0.00 |
| 50 | 1.00 | 0.95 | 0.00 |
| 100 | 1.00 | 0.95 | 0.00 |
| **200** | **1.00** | **0.95** | 0.00 |

**Flat to N=200 — no interference regime at all in range.** GPT-2 held ~82% express at N≥200; Qwen
(centered keys, L18) holds **95%, perfectly selected**. The ~5% express gap is per-token forcing
difficulty (flat, not decaying), not capacity. The explicit list is, in this regime, a lossless store. |

## Layer sweep (1.5B): 14 / 18 / 22 — mid-stack is a band, not a gradient

Full battery per layer, same 20-fact bank, greedy/deterministic (the L18 row reproduces the committed
run exactly). Runs: `runs/slotmem_qwen1p5b_L{14,18,22}.json`.

| layer | recall top1 | P(ans) | shuffled null | specificity on / off | paraphrase ungated R/WF | gated R/WF/abst | emission single | emission multi |
|---|---|---|---|---|---|---|---|---|
| 14 | **0.95** | 0.58 | 0.00 | 1.00 / 0.00 | 7 / 1 | 7 / 0 / 1 | **13/13** | 5/7 |
| **18 (default)** | 0.90 | **0.71** | 0.00 | 1.00 / 0.00 | **9 / 1** | **9 / 0 / 1** | 12/13 | 5/7 |
| 22 | 0.50 | 0.42 | 0.00 | **0.25** / 0.00 | 6 / 0 | 6 / 0 / 1 | 7/13 | 4/7 |

**Decision: 18 stays the default.** L14's +0.05 recall sits AT (not above) the pre-set bar and costs
paraphrase generalization (9→7 of 10) and P(ans) (0.71→0.58) — earlier keys read more *lexical*:
exact cues route slightly better, rewordings worse. (Violated pre-registration: I expected L14
*weaker* on recall, not stronger.)

The L22 collapse decomposes into two effects (dose-response probe: model loaded once, 20-fact
recall, injection scale swept):

| eta frac (× L22 resid norm) | 0.15 | 0.25 | 0.35 | **0.5** | 0.65 | 0.75 | 1.0 | 1.5 (default) | 2.0–4.0 |
|---|---|---|---|---|---|---|---|---|---|
| recall top1 | 0.20 | 0.60 | 0.65 | **0.80** | 0.75 | 0.70 | 0.65 | 0.50 | 0.40 |

(a) **`eta = 1.5× the layer's resid norm` does not transfer across layers.** Late-layer norms are
~2× mid-stack (132 vs 69), and the same *fraction* over-injects — L22's dose-response peaks near
0.5×. (b) Even dose-tuned, L22 tops out at **0.80 < L18's 0.90**, and the emission failures say why:
with the *right* value injected, the model REPHRASES instead of copying ("seven"→"7",
"winter"→"harsh and cold", "silver"→"pure gold"). The concept lands; the verbatim token loses —
and the store's contract is verbatim. So p19's "deeper = more meaning" is a *band*: paraphrase
routing improves 14→18, then verbatim control dies by 22.

(Eyeball note, applies to all layers: emission's substring scoring counts "rosemary…" as recalling
"rose" — one visible leniency in the samples; the exact-argmax recall metric is unaffected.)

## The new finding the port itself produced

**Qwen's keys are anisotropic where GPT-2's weren't.** Raw last-token keys had cross-similarity
**0.68** (every cue ends alike), which crippled routing (recall 0.33) and made the gate uncalibrable
(floor > 1.0). **Centering the keys** (subtract the mean key, renormalize; queries likewise) dropped
cross-sim to **−0.05** and took recall 0.33 → **0.90** in one change. p17 found decorrelation "adds
nothing" on GPT-2 — that result **does not transfer**: key geometry is model-dependent, and centering
is the cheap fix Qwen needs. (Injection scale is also model-dependent: GPT-2's calibration was too
weak here; 1.5× residual norm is Qwen's working point — 0.6× lifted P(ans) 17× yet lost the argmax.)

## What this rung does NOT show (caveats loud)

One model, one seed; layers now swept at 14/18/22 on 1.5B (18 confirmed default; the battery rows
share one eta-frac, only the L22 probe re-dosed), next-token + short-greedy metrics; sweep facts are
templated (six attribute families — diverse free-text cues untested at scale, though the 20-fact bank's
0.90 covers hand-varied phrasing). Multi-token at 71% (two-token schedule) is still a partial — answers
past two tokens rely on clean continuation. The write gate's threshold (3.0 nats) is hand-set,
validated only against 4 known facts. Persistence/serving (a `~/.clozn` store + studio surface) is
unbuilt — this is the mechanism proven, not the product wired.

## Why it matters

The don't-fuse law now has its constructive half **on the studio's model family**: an internal memory
that is explicit (a list you can print), legible (every value logit-lens decodes to its answer),
editable (surgical deletes, bit-identical bystanders), **honest about ignorance** (abstains under the
gate instead of confabulating), and **selective about what it learns** (refuses to store what the
model already knows). Fused-prefix memory interferes at N=64 (`memory_scaling`); this is the
architecture that replaces it for facts-inside-the-model — Tier-2's structured sibling.
