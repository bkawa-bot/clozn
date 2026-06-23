# Function-vector layer sweep + round-trip diagnostic: WHERE does the in-context rule live?

*Settles the causal leg the SAE reads left underpowered, and explains the whole discovered-basis negative.
`function_vector_sweep_qwen.py` (Hendel-style task-vector sweep) + `function_vector_roundtrip_qwen.py` (the
diagnostic). 2026-06-23, Qwen3-1.7B-Base frozen, 8 relations, forward-only.*

## Verdict: there is NO query-independent, clampable "rule vector" - and the reason is MECHANISTIC, cleanly diagnosed, not a method bug. At the SAE's layer (14) the rule is not yet in the residual stream; the model is still computing it by attending to the in-context examples. So an SAE there genuinely cannot read it as a feature.

This is the deepest answer to Brigitte's "are we doing something wrong?": **partly yes** - clamping/reading at
layer 14 was the wrong place. But fixing that reveals there is no query-independent rule vector at ANY layer.

## The sweep (the question)
Extract a Hendel task vector at every layer (the in-context prompt's last-token hidden state, averaged over
demos), patch it into a zero-shot query, sweep all 28 layers. Result: **peak mean apply 0.07 at L18**, barely
above the zero-shot floor 0.02 and the wrong-task null 0.00 (ICL ceiling 0.94). Essentially no transfer.
On its own this is ambiguous: broken mechanism, or a real fact?

## The diagnostic (the answer)
Patch the model's OWN in-context hidden state for the SAME query back into its zero-shot run, per layer:

| relation | L6 | L10 | L14 (SAE) | L18 | L22 | L26 |
|---|---|---|---|---|---|---|
| plural | 0.0 | 0.0 | **0.0** | 0.33 | 1.0 | 1.0 |
| past | 0.0 | 0.0 | **0.0** | 0.0 | 1.0 | 1.0 |
| gerund | 0.0 | 0.0 | **0.0** | 0.17 | 1.0 | 1.0 |

(ICL apply = 1.0 throughout.) So:
- **The patch mechanism is SOUND** - deep layers round-trip 100%. The sweep's ~0 is not a bug.
- **The answer only becomes a transplantable last-token residual by layer ~22.** Before that (including the
  SAE's layer 14) the last-token state does NOT carry the answer when moved into a context without the
  examples - the model is still computing it by **attending to the in-context example tokens** through the
  middle layers.
- **And by the depth where it IS a clampable vector (~22), it is the CONCRETE answer for that query, not a
  reusable rule** - which is exactly why the query-AVERAGED task vector in the sweep transferred ~0 even at
  deep layers (averaging over queries washes out the specific answer, and there is no abstract rule vector
  underneath it).

## Why this resolves the entire discovered-basis arc
- **Early layers (incl. L14, the SAE):** the rule is an in-flight *computation over the examples* (attention),
  not residual-stream *content*. An SAE reads residual content at one layer, so it legitimately cannot render
  the in-context rule as a feature there - it is not there to read. Not a bug; the object isn't present.
- **Late layers (~22+):** the residual now holds the concrete answer (query-specific), not a reusable rule.
- **So there is no layer with a query-independent, clampable rule vector:** early = not computed, late = the
  answer. This is fully consistent with every prior leg: the footprint is rule-DISCRIMINATIVE (the in-flight
  computation differs by rule) but not SPARSE / NAMEABLE / CAUSAL (it is not a clean residual feature).

## The takeaway for clozn (the satisfying synthesis)
- A trained-in **concept** ("Golden Gate Bridge") is residual-stream *content* -> SAE-readable and steerable.
  That is the published SAE success, and it is real.
- A transformation **rule** applied in-context is a *computation over the examples*, not residual content at a
  mid layer -> not an SAE feature. **Different kind of object.** SAE feature-reading is the right tool for
  concepts and the wrong tool for in-context rules. We were not doing it wrong in a fixable way; we were
  pointing a content-reader at a computation.
- The legible handle on a **learned** rule is the model **stating it** (self-report + verify, works at >=1.5B,
  the :8078 demo). That route sidesteps all of this by asking the model to externalize the rule in words,
  which we then verify - rather than trying to find it as a vector that, mechanistically, isn't there.

## Honesty / limits
- The round-trip proves the mechanism, so the negative is not a clamp artifact. All 28 layers swept; wrong-task
  null included; ICL ceiling 0.94 confirms the model genuinely applies the rules.
- We only tried SINGLE-layer, last-token interventions. A multi-layer or attention-head (Todd-style) function
  vector might extract more - but the round-trip already shows the answer isn't a single-layer residual until
  it has become the concrete answer, so a clean query-independent rule vector is unlikely regardless. If we
  ever want to push it, that is the experiment; it would not change the clozn takeaway (self-report is the
  legible handle).

Files: `research/function_vector_sweep_qwen.py` (+ `runs/function_vector_sweep_qwen.json` + curve SVG),
`research/function_vector_roundtrip_qwen.py` (+ `runs/function_vector_roundtrip_qwen.json`). Qwen3-1.7B-Base,
frozen, forward-only; lab `.venv`, no sae_lens.
