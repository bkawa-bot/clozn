# The fair test: read the rule in SAE features on NATURAL in-context application (`legibility_natural_qwen.py`)

*Answers Brigitte's critique of `legibility_discovered_qwen.py`: "SAEs are trained for exactly this; if it
isn't working we're doing something wrong." We fed the SAE the activation a gradient-found injected soft
prefix causes (out-of-distribution), then a difference of two such encodings. The fair read is the
activation when the model applies the rule the NATURAL way (in-context examples, real tokens), against a
zero-shot baseline. That is the classic function/task-vector construction, squarely on the SAE's home turf.
2026-06-23, Qwen3-1.7B-Base + Qwen-Scope (TopK, 32k, layer 14), 8 held-out relations.*

## Verdict: the fair read REPRODUCES the negative (1/4). Reading the NATURAL application gives the SAME answer as the soft-prefix read - so the prefix was NOT the confound for sparse/nameable. The causal leg is separately INCONCLUSIVE (underpowered), and that part is worth fixing.

The natural state is real: the in-context **ICL apply ceiling is 0.94** (the model genuinely applies these
rules from examples), so we are reading a state where the rule is being used. Then:

| leg | soft-prefix read | NATURAL read | takeaway |
|---|---|---|---|
| rule-specific | yes (−0.14) | **yes (−0.13)** | robust both ways |
| sparse | no (PR 74 vs 67) | **no (PR 75 vs 70)** | not a prefix artifact; TopK spreads it either way |
| nameable (logit-lens) | no (0%) | **no (0%)** | not a prefix artifact |
| causal | ~1% | **see below** | inconclusive on both |

**So the soft-prefix-OOD hypothesis is ruled out for sparse + nameable.** Reading the model naturally
applying the rule does not make its footprint in this SAE any sparser or more nameable. Whatever is going
on, it is not "we fed the SAE a weird injected object."

## The causal leg is underpowered - honest flag, not a clean negative

STEP 4 clamps a direction into a fresh bare query at layer 14, three ways: the RAW task vector
(`mean(A_rule − A_zero)`), the SAE reconstruction of the positive footprint, and a random-feature null.
Result: zero-floor 0.02, **RAW task vector 0.00**, SAE-recon 0.06, random 0.00, ICL ceiling 0.94.

The **RAW task vector recovering ~0% is suspicious**: the function/task-vector literature (Hendel et al.
2023, Todd et al. 2024) shows an in-context task vector IS causal - patch it into a zero-shot query and the
model does the task. Ours did nothing. The likely reasons are methodological, not "no causal direction
exists":
- **Crude task vector.** `mean(A_rule − A_zero)` at the last token mixes the rule with everything else that
  differs between a long ICL prompt and a bare 2-token query (position, "there is an instruction + examples
  above" context). Adding that whole difference to a bare query injects a lot of irrelevant context-shift.
  The literature uses more careful constructions (specific heads / calibrated patching).
- **Single layer + add-not-patch.** We extract and inject only at layer 14 and ADD a task-vec-norm-scaled
  vector; the function-vector recipe sweeps layers and often PATCHES (replaces) the hidden state.
- The hook is not a no-op (the SAE-recon clamp moved 2 relations to 0.17/0.33), so the mechanism works; the
  RAW vector just is not a clean rule handle as constructed.

So **causality is NOT settled here.** The right test is a proper function-vector layer sweep (extract at and
patch into each layer, find where the task vector is causal), independent of the SAE; then ask whether the
SAE at that layer captures it.

## What it means (honest, and it does engage the critique)
- **The fair read did NOT vindicate "the prefix made it unfair"** for the sparse/nameable legs - the natural
  read is just as diffuse and just as un-nameable (by logit-lens) at layer 14.
- **The substantive read:** SAEs reliably surface CONTENT features (a topic/concept present in the text -
  "Golden Gate", "dogs", sentiment). A RULE / TRANSFORMATION ("make plural") is an abstract RELATION, and it
  appears to be represented more DIFFUSELY than a concept even when applied naturally - rule-discriminative
  (you can tell rules apart) but not a single sparse nameable feature. That is a property of how rules are
  encoded, not obviously a bug in our method. Concepts → SAE features; relational rules → harder/diffuse.
- **The legibility that DOES work for a learned rule is the model stating it** (self-report + verify, idea 3,
  >=1.5B). That remains the path.
- **Still genuinely open (the critique's residue):** the causal leg. A function-vector layer sweep could
  still find a clampable rule direction at some layer - and if it exists, whether an SAE there is sparse on
  it is the real, fair version of the question. Worth doing before declaring rules un-steerable.

Files: `research/legibility_natural_qwen.py`, `runs/legibility_natural_qwen.json` + 3 SVGs (ICL, specificity,
causal). Reuses RawTopKSAE + QwenHarness from `legibility_discovered_qwen`; synchronous; lab `.venv` clean of
sae_lens. Caveat: reads are no-BOS (matching the prior runs / the bare-query convention).
