# engine-native dial calibration — the metric doesn't transfer (findings)

*2026-07-07. Built `research/dial_autocalibrate_engine.py` (the engine analog of `dial_autocalibrate.py`)
to fix the substrate-specific range gap — the shipped dial ranges are PyTorch-derived and the engine's
steer scale differs. Ran the `--smoke` calibration, EYEBALLED it before trusting, and caught that the
metric is not sound on the engine. Documenting so it isn't re-attempted blind. The steering itself is
fine; only the automated calibration metric is off. NOT deployed — a real sweep would write bad ranges.*

## What the smoke caught

The smoke flagged `warm` and `tender` as `works=False`. But I've watched both steer. A direct measurement
against dials I know steer strongly exposed two independent problems, both from faithfully porting the
PyTorch **raw-dot projection** (`engine_alignment` mirrors `dial_autocalibrate._project_onto_unit`) to the
engine's very different activation scale:

1. **It under-measures.** `ceremonious` unmistakably steers ("Oh, my dear, let us transport ourselves to
   the realm of the heavens") but its measured effect is only **+12**, *below* the engine-derived
   `effect_eps=23.1` — so a sweep would flag an obviously-working dial as dead.
2. **The scale isn't comparable across dials.** The *same* neutral reply projects **17.7** onto
   `ceremonious`'s direction but **336.6** onto `tender`'s — a ~20× spread. A single scalar threshold
   cannot fit both, and `tender`'s effects come back as noise (+137.7, +5.1, +26.5 across doses).

## Root cause

- **Raw-dot ≠ scale-invariant.** `engine_alignment = mean_pool(reply_acts) · unit(dir)` is dominated by
  `|mean_pool(reply)|` (hundreds, at the engine's `resid_norm≈794`, ~11× PyTorch's), so a dial whose
  direction happens to align with the general activation reads huge, one that's orthogonal reads small.
  The single `effect_eps` can't span that. (Agent-B flagged cosine as a "same-spirit alternative it didn't
  use, to mirror the PyTorch raw-dot" — that omission is exactly what bites here.)
- **Some directions are genuinely contaminated.** `tender`'s baseline projection (336.6) ≈ its own vector
  norm scale, i.e. `tender`'s direction is nearly parallel to the general reply direction — the model
  barely distinguishes tender-vs-harsh in the residual at layer 14, so the diff-of-means is weak and picks
  up the mean activation. This is the "distinctive positive signature steers, subtle/absence dies" finding
  (`dial_library_findings.md`) reappearing at the *metric* level.

## The fix path (a proper research iteration, not a threshold tweak)

1. **A scale-invariant / per-dial-normalized effect metric** — cosine (normalize by `|mean_pool(reply)|`),
   or project the CENTERED change against a per-dial reference, so effects are comparable across dials and
   a single threshold is meaningful.
2. **Handle genuinely-weak directions** — a dial whose diff-of-means barely separates from the mean should
   read "no measurable direction on this model", not a noisy number (a shuffled-direction null per dial, as
   the PyTorch rig already does, but scaled correctly).
3. **Re-validate against known anchors before trusting a sweep** — `ceremonious`/`storyteller` must read a
   clean positive band; a nonsense dial → nothing; only then run the 33-dial sweep.

## Status / what's real

- **Steering works great** on the engine — dials steer strongly (`ceremonious`, `slangy`, `storyteller` all
  visibly on-tone). The dials + the product are NOT blocked.
- **`EngineSteer.add_custom` works** — "make your own dial" on the engine is real, shipped value; only the
  *calibrate-on-creation verdict* waits on the metric.
- **The PyTorch-derived ranges remain the "usable starting points"** they always were.
- `dial_autocalibrate_engine.py` + its tests are committed as a first attempt; **NOT wired into a deploy**
  (it would write untrustworthy ranges). The metric redesign is a scoped follow-up, parked in NEXT_STEPS.
