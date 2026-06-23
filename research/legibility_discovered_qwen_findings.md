# Discovered-basis legibility on Qwen3-1.7B + Qwen-Scope (`legibility_discovered_qwen.py`)

*The richer-substrate test the GPT-2 + Bloom cut (`legibility_discovered.py`, 2/4) said it needed. We did
NOT need the gated Gemma route: Qwen ships **Qwen-Scope**, official pretrained SAEs (Apache-2.0, ungated),
loadable via `sae_lens`. 2026-06-23, Qwen3-1.7B-Base frozen + Qwen-Scope `qwen-scope-3-1.7b-base-w32k-l50`
(TopK, k=50, 32,768 features, `blocks.14.hook_resid_post`). 8 relations, held-out words + relations.*

## Verdict: PARTIAL (1/4), a useful NEGATIVE. The richer base + bigger official dictionary did NOT close NAMEABLE or CAUSAL, and lost SPARSE to a TopK-architecture confound. The one robust, comparable signal across both substrates: RULE-SPECIFIC yes, CAUSAL no.

The bet behind this run was the GPT-2 cut's optimistic read: "sparse+specific are real; nameable+causal
stall because the substrate (124M base + 24k dict) is too small; a richer base + richer pretrained
dictionary should close them." On a 1.7B base with a 32k official dictionary, **it did not.**

## The four legs

**STEP 1 - TTT precondition: STRONG (8/8).** Held-out free-apply: plural/past/gerund/superlative/
comparative/third_person **1.000**, antonym2 0.833, part_of 0.500 (no-prefix ~0.00). Qwen3-1.7B is far
stronger than GPT-2 here, so the precondition that limited GPT-2 is gone. Whatever fails below is NOT "the
model can't learn the rule."

**STEP 2a - SPECIFIC: YES (robust).** Shared-component-removed off-diagonal cosine **-0.14**, top-16
Jaccard 0.234: after removing the common "produce-an-answer" direction, different rules light genuinely
different features. Matches the GPT-2 cut. The learned rule's *footprint* is rule-discriminative.

**STEP 2b - SPARSE: NO, but CONFOUNDED.** Real participation-ratio **73.6** vs random-direction null 67.0
(barely different). BUT this is largely an **SAE-architecture artifact, not a Qwen fact**: Qwen-Scope is a
**TopK** SAE (exactly k=50 features per token), so the read-out delta `enc(with) - enc(without)` spans two
different top-50 sets and is inherently diffuse (L0 ~240-300). Bloom's GPT-2 SAE (where the prior cut found
"sparse") is a JumpReLU/L1 SAE whose activations are naturally concentrated. Comparing the two on
"sparsity of the delta" is apples-to-oranges. A fair sparsity test needs a **non-TopK** Qwen SAE.

**STEP 3 - NAMEABLE: NO (0%), but METHOD-LIMITED.** Qwen-Scope has no Neuronpedia labels
(`neuronpedia_id` is None), so unlike the GPT-2 cut (Neuronpedia auto-interp + WikiText top-activating
contexts) the only tool here was a **logit-lens proxy** (name each feature by the tokens its decoder
direction promotes through the unembedding). The promoted tokens were dominated by rare multilingual/code
pieces (`деят`, `这个词`, `amatør`, ...), never plural/`-s`/`-ed`. This is partly weak naming (a mid-layer
14/28 decoder direction read through the final unembed is noisy) and partly genuine polysemanticity. So
0% is a **soft** negative: a real auto-interp pipeline (or Neuronpedia hosting Qwen-Scope) is needed to
call nameability cleanly. Not apples-to-apples with the GPT-2 cut.

**STEP 4 - CAUSAL: NO (the clean, comparable negative).** Clamping the read-out features' decoder
reconstruction into a fresh no-prefix query recovers **~1%** of the TTT gain: feature-clamp 0.02 vs
random-clamp 0.00 vs TTT ceiling 0.92. Only part_of showed any recovery (33%, at a small absolute level).
This **replicates the GPT-2 result exactly**. The activation-delta is not a causal handle through these
features at either scale.

## What it means (honestly)

- **The "bottleneck = substrate" optimism did not pan out for this SAE family.** Scaling the base to 1.7B
  and using a 32k official dictionary left NAMEABLE and CAUSAL failing, and (architecture-confounded)
  sparsity gone. Bigger + richer was not the unlock for feature-reading a learned rule.
- **Robust cross-substrate story:** a TTT-learned rule's footprint in a pretrained SAE basis is
  RULE-DISCRIMINATIVE (specific) but NOT a clean, nameable, causal feature set. Reading a learned rule in
  discovered features is, locally, at best a *statistical signature*, not a legible-and-steerable handle.
- **Contrast the other legibility route, which DOES work:** self-report + verify (idea 3,
  `legibility_v1_big.py`) clears the wrong-rule null at >=1.5B - the model *states* the rule and it checks
  out. So for clozn, legibility-of-a-learned-rule comes from **asking the model (and verifying)**, not from
  SAE feature-reading - at least at local scale with the dictionaries available.
- **A likely deeper reason** (beyond size): the thing we read is a *soft-prefix delta* - an out-of-
  distribution object for an SAE trained on *natural* residual activations. The rule lives in a learned
  prefix, not in the residual stream the way a trained-in concept ("golden gate") does. An SAE may simply
  not have a basis aligned to "the residual change a fresh injected prefix causes." That mismatch, not
  scale, may be the real wall.

## Honest caveats / cheap next cuts (the door is not closed)
1. **TopK confounds sparsity** - rerun on a non-TopK Qwen SAE (e.g. the `mwhanna/qwen3-1.7b-transcoders`
   or any JumpReLU release) for an apples-to-apples sparsity comparison with Bloom's GPT-2 SAE.
2. **Nameability needs real auto-interp**, not logit-lens; or wait for Qwen-Scope on Neuronpedia.
3. **One layer only** (14/28) - a layer sweep is cheap and might matter (the rule may read more cleanly
   nearer the answer slot).
4. The golden-gate analogy reads a *trained-in* concept; reading a *test-time-injected* rule is a harder,
   possibly different problem - worth stating as the framing, not a footnote.

Files: `research/legibility_discovered_qwen.py`, `research/runs/legibility_discovered_qwen.json` + 3 SVGs
(`_ttt`, `_specificity`, `_causal`). SAE weights dumped via `.venv-sae` to `runs/qwen_scope_1p7b_layer14.npz`
(+ `.meta.json`), reimplemented raw and verified bit-faithful in the lab GPU venv (encode rel 9e-6, decode
4e-3). Synchronous, single process; the lab `.venv` was never given `sae_lens`.
