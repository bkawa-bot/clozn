# Feature-circuit pilot: do real feature->feature causal edges survive ablation? (`feature_circuit_pilot_qwen.py`)

*Answers Brigitte: "now that we have valid SAE dicts, can't we draw the concept->concept edges the think_graph
note refused to fake?" Method: gold-standard ABLATION. Ablate a top source feature at layer 14 (subtract its
decoder write from resid_post), measure the change in each top target feature at layer 20, keep only edges
that beat a random-active-feature ablation null (99th pct). Qwen3-1.7B-Base + Qwen-Scope SAEs (L14, L20),
3 prompts, forward-only. 2026-06-23.*

## Verdict: QUALIFIED YES. Real causal feature->feature edges DO survive verification (so "draw none" is now too strong) - but most surviving edges are dominated by generic high-magnitude features + a TopK-selection artifact, and only a FEW are genuinely interpretable concept->concept edges. A trustworthy circuit needs a cleanup pass; do not wire arbitrary edges.

55/300 candidate edges (18%) beat the random-ablation null across 3 prompts (the model gets all three:
George->Washington, hot->cold, France->Paris). So ablating a *specific* source feature moves a *specific*
downstream feature more than ablating a generic active feature does. The intervention is real and the effect
is above the generic-perturbation floor. The note's strong form ("we can verify NO edges") is falsified.

## But read the actual edges - three load-bearing caveats

1. **Dominated by a few GENERIC high-magnitude features.** The same source feature `f22632`
   (logit-lens: `rael/otope/olation`) is the top survivor in BOTH "opposite of hot" AND "capital of France" -
   unrelated prompts. A feature that fires huge across everything is a generic/position/frequency direction,
   not a content concept. Most survivors look like this (logit-lens returns multilingual/code/punctuation
   fragments, not "hot"/"cold"/"city").

2. **The huge deltas are partly a TopK artifact.** Qwen-Scope is TopK (exactly 50 features/token). Ablating a
   source can push a target feature across the top-50 boundary, so a target jumps 0<->large (delta +98, +158).
   That is a real causal change, but it is a discontinuous threshold flip, not a smooth "i drives j", and it
   inflates magnitudes. (The null has this too - hence the 99th-pct threshold - but survivors with delta>>null
   are still mostly the generic features.)

3. **Tiny answer-logit effects.** Ablating any single source feature moves the predicted-token logit by only
   ~0.05-0.15. So these edges are NOT the output circuit; they are feature-to-feature couplings off to the
   side of what actually drives the answer.

## The genuinely interpretable survivors (the real signal in the noise)
A few edges are exactly the kind you'd want, and they survived ablation + the null:
- **France->Paris:** `f22979` (logit-lens `town / 城镇 / town`) -> `f27357` (`city / cities / 城市`), delta +28.
  A "town/place" feature driving a "city" feature while predicting Paris. That is a real, sensible
  concept->concept edge.
- **George->Washington:** subword chains `f21894/f26521` (`sson / son / sonian`) -> `f12237` (`sson / sons /
  son`), delta +37-60 - the "...ington/son" orthographic assembly of the answer.

So SOME real, interpretable concept->concept edges exist and verify. They are just outnumbered by
generic-feature / TopK-boundary edges that the raw pipeline cannot tell apart without a filter.

## Honest call
- **The think_graph note can be SOFTENED, not lifted wholesale.** We CAN now draw *verified* edges, and a few
  are genuinely interpretable - so "draw none" is too conservative. But drawing ALL surviving edges would put
  generic/artifact edges on the page, which is the dishonesty the note guards against.
- **What it would take to draw a circuit I'd trust:** (a) filter out generic features (fire across unrelated
  prompts / no clean logit-lens), (b) handle the TopK discreteness (measure a smooth/gradient influence, or
  average over small perturbations, not a single full ablation), (c) weight by answer-relevance (the tiny
  logit effects say most of these are side-couplings). Then draw only the survivors that are interpretable AND
  answer-relevant - a small, honest set (town->city is the proof it exists).
- This matches the arc: the dictionary blocker is gone, real causal structure is *findable*, but on a local
  1.7B it is faint and noisy, so the honest artifact is a *few labeled, verified* edges, not a dense graph.

Files: `research/feature_circuit_pilot_qwen.py`, `runs/feature_circuit_pilot_qwen.json` (note: the JSON's
auto-verdict is the rosy "edges exist" read; THIS doc is the honest nuance after inspecting the actual edges).
Reuses RawTopKSAE; Qwen-Scope SAEs at L14+L20 dumped via .venv-sae and verified bit-faithful here.
