# Frontier v2: closing the consolidate-then-apply loop (`frontier_apply_v2.py`)

*Follow-up to `frontier_apply.py` (Stage 1 yes / Stage 2 + legibility no). Three levers against the wall.*
*2026-06-22, Qwen2.5-0.5B-Instruct frozen, 26 relations, held-out relations + words. Fast/"smoke" run*
*(real model + real mechanism + all three levers and their controls; small held-out set + modest step*
*budgets, so directions are solid and exact magnitudes want a fuller confirmation).*

## Verdict: TEST-TIME ADAPTATION closes the loop. The feed-forward compressor does not. Legibility stays open.

`frontier_apply.py` localized it: a directly-tuned prefix *applies* a relation (Stage 1, ~0.93), but a
meta-learned feed-forward compressor that turns examples into that prefix *fails* to generalize to a new
relation (Stage 2, 0.000), and the prefixes are not legible. Three levers:

**Lever 1 — MORE relations (was 7 just starved?): NO.** At 26 relations, held-out-relation FREE-apply stays
**0.000** (menu 0.11) vs ICL **1.0**. More relations do not move it: seven was not merely starved, the
feed-forward example->prefix map fundamentally fails to generalize to an unseen relation. (The Stage-1
oracle still scales: per-relation tuned prefixes hit agg free **0.66** over all 26 — the *direct* path
scales, the *consolidation* path does not.)

**Lever 2 — DISTILL to a shared compressor (for generalization + legibility): NO on both.** A shared
compressor regressing examples->the working oracle prefixes still gives held-out FREE-apply **0.000** (menu
0.26) vs the oracle ceiling 0.89. And the apparent legibility (a probe names the relation at 0.99) is an
**artifact**: an untrained-map null scores the same (1.0), so the separability is an input-feature property,
not learned shared structure. No real legibility recovered.

**Lever 3 — TEST-TIME ADAPTATION (the Titans / fast-weights move): YES.** For a NEW (held-out) relation, fit
its prefix with a few GRADIENT STEPS on its own examples (not a single feed-forward map). Held-out FREE-apply:
0 steps -> 0.00; 5 steps -> 0.21; **20 steps -> 0.944** (ICL 1.0, read-MLP 0.0). A handful of gradient steps
recover a working prefix for an unseen relation, so the loop **CLOSES per-relation at test time**. And a
learned init (the compressor from lever 2) reaches the bar at **5 steps** vs 20 from scratch — the learned
init helps even though it cannot do the job feed-forward alone.

## What it means
- **"Learns" is earned, in the test-time-training sense.** Show the model a few examples of a new rule, let
  it take a few gradient steps, and it applies that rule to held-out cases at near-in-context accuracy, with
  its *own* output. That is genuine test-time learning (the Titans / fast-weights paradigm), not a memorized
  pile. What does NOT work is the one-shot feed-forward "compress examples -> injection" map.
- **Legibility is still open.** The working (TTT-fit) prefixes are not legible, and lever 2's apparent
  legibility was an artifact. So the **applies-vs-legible tension** from `frontier_apply.py` persists: TTT
  buys application, not legibility-by-construction. That is the next real target.

## Honesty / caveats
- This is the FAST run: real Qwen, real mechanism, all three levers + controls, but a small held-out set
  (3 relations) and a coarse step grid (0/5/20). Two flat negatives and one strong positive — directions are
  clear; a fuller confirmation of lever 3's 0.944 on more held-out relations + a denser step curve is the
  warranted next step.
- Operational note: the v2 run over-parallelized into a swarm of GPU jobs and could not deliver cleanly; this
  verdict is synthesized from its completed fast-run results (`research/runs/frontier_apply_v2_smoke.json`),
  and the process sprawl was cleaned up afterward.

Files: `research/frontier_apply_v2.py`; `research/runs/frontier_apply_v2_smoke.json`.
