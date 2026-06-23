# Clean feature circuit + viz (`feature_circuit_clean_qwen.py`)

*The cleanup pass that turns the noisy pilot (`feature_circuit_pilot_qwen.py`, 18% survive but
generic/artifact-dominated) into a small, honest, ablation-verified concept->concept circuit, and renders it
as a self-contained light viz. Qwen3-1.7B-Base + Qwen-Scope SAEs (L14 source, L20 target). 2026-06-23.*

## Verdict: a verified concept->concept circuit IS drawable, and at least one is genuinely legible. "opposite of hot -> cold" is the standout: a `reverse`/`flip` concept (L14) drives a `lower`/`lesser` concept (L20) that pushes the answer. Small and prompt-dependent on a local 1.7B - the honest few-labeled-edges artifact, not a dense graph.

The three fixes from the pilot's caveats, all applied:
- **Smooth influence (kills the TopK artifact):** edge weight = change in the target feature's PRE-activation
  (linear readout of the residual), not its post-TopK activation. No top-50 boundary flips.
- **Specificity filter (drops generic features):** a feature's activation here / its mean over a diverse
  10-prompt background. Generic always-on directions (the ones that topped unrelated prompts in the pilot)
  score ~1 and are removed; only features that fire on THIS prompt survive.
- **Answer-relevance + 2-hop circuit:** ablate each source -> change in the predicted-token logit; draw
  source(L14) -> target(L20) -> answer token, where the target->token edge is the feature's logit-lens push.
Edge drawn only if both endpoints are specific, the smooth influence beats the 99th-pct generic-ablation
null, and at least one endpoint is interpretable (clean logit-lens name).

## Per-prompt (honest, mixed - which is the point)
- **"opposite of hot is" -> cold (the win):** `reverse`/`flip` (L14 f20434/f29880) -> `lower`/`lowest`
  (L20 f14187, push +20) and `synonyms`/`alternatives` -> `lesser`/`less`/`smaller` (f16442, push +10). A
  real, legible concept->concept circuit: "take the reverse/lower of hot" -> cold. Exactly the dream.
- **"...was George" -> Washington:** the `son`/`sson`/`sonian` subword-assembly chain (f21894 -> f12237)
  survives and is interpretable (orthographic), mixed with a couple of un-nameable features.
- **"capital of France" -> Paris:** ZERO edges survived the stricter clean filter (the pilot's town->city
  did not clear it). Reported as honest absence, not forced.
- **"two plus two equals" -> four:** survivors are un-nameable junk with ~0 answer-relevance. Shown in the
  JSON, not emphasized; arithmetic does not yield a legible 2-feature circuit here.

24 verified + interpretable edges across 4 prompts; quality varies sharply by prompt (great for "opposite",
nil for "Paris"). That variance is the honest finding: real causal structure is findable but faint on a
local 1.7B, so the trustworthy artifact is a few labeled edges, prompt by prompt.

## What changed downstream
- New self-contained light viz: `inspector/runs/feature_circuit.html` (source -> target -> answer token,
  edges ablation-verified, generic-filtered; honest note on the page). Mobile-openable.
- `think_graph.py`'s "we draw NO concept->concept edges (the SAE null)" note is SOFTENED: those edges are
  now drawable with a pretrained SAE + ablation verification (pointing to this circuit); think_graph keeps
  concept->token only because its 8 concepts are diff-in-means directions, not dictionary features.

## Honesty / limits
- Logit-lens naming is mid-layer and approximate (a few survivors are un-nameable; we only draw edges with at
  least one clean endpoint). The "opposite" circuit is clean enough to trust; the arithmetic one is not, and
  we say so. Absence of an edge = "did not survive verification here", not "no influence".
- Two layers only (14->20), single ablation per feature (smoothed via pre-activation, not a full perturbation
  sweep). A scaled version would use transcoders + cross-layer attribution; this direct-ablation pilot is the
  honest core and already enough to lift the "draw none" note.

Files: `research/feature_circuit_clean_qwen.py`, `runs/feature_circuit_clean_qwen.json`,
`inspector/runs/feature_circuit.html`. Qwen-Scope SAEs L14+L20 verified bit-faithful; lab `.venv`, no sae_lens.
