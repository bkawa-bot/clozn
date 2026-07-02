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
| **Multi-token emission** (new) | single 12/13; **multi 4/7 (57%)** — first-token injection elicits the full answer just over half the time |

## The new finding the port itself produced

**Qwen's keys are anisotropic where GPT-2's weren't.** Raw last-token keys had cross-similarity
**0.68** (every cue ends alike), which crippled routing (recall 0.33) and made the gate uncalibrable
(floor > 1.0). **Centering the keys** (subtract the mean key, renormalize; queries likewise) dropped
cross-sim to **−0.05** and took recall 0.33 → **0.90** in one change. p17 found decorrelation "adds
nothing" on GPT-2 — that result **does not transfer**: key geometry is model-dependent, and centering
is the cheap fix Qwen needs. (Injection scale is also model-dependent: GPT-2's calibration was too
weak here; 1.5× residual norm is Qwen's working point — 0.6× lifted P(ans) 17× yet lost the argmax.)

## What this rung does NOT show (caveats loud)

One model, one seed, one layer (18; not swept), 20 nonce facts with distinctive single/short answers,
next-token + short-greedy metrics. Multi-token at 57% is a real partial: the value only promotes token
one; a two-token value scheme or per-step re-injection is the obvious next lever. No capacity sweep on
Qwen yet (p17 held to N≥200 on GPT-2; untested here). The write gate's threshold (3.0 nats) is
hand-set, validated only against 4 known facts. Persistence/serving (a `~/.clozn` store + studio
surface) is unbuilt — this is the mechanism proven, not the product wired.

## Why it matters

The don't-fuse law now has its constructive half **on the studio's model family**: an internal memory
that is explicit (a list you can print), legible (every value logit-lens decodes to its answer),
editable (surgical deletes, bit-identical bystanders), **honest about ignorance** (abstains under the
gate instead of confabulating), and **selective about what it learns** (refuses to store what the
model already knows). Fused-prefix memory interferes at N=64 (`memory_scaling`); this is the
architecture that replaces it for facts-inside-the-model — Tier-2's structured sibling.
