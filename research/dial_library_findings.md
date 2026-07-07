# dial library — a measured, per-model tone-dial library (findings)

*2026-07-06. Qwen2.5-7B-Instruct (nf4). Rig: `research/dial_autocalibrate.py`; shipped library:
`research/dial_library_shipped.json` (33 dials). The product move: instead of showing users a pile of
dials that may or may not work and making them customize-and-hope, SHIP a library the model itself
voted for — each pre-calibrated to its safe range on the user's own model.*

## The pipeline

**71 candidates → 47 metric-usable → 33 human-curated.** (`dial_library_candidates.json` →
`dial_autocalibrate.py` sweep → human curation pass.)

## The metric (the load-bearing fix)

A dial is a diff-of-means direction; "does it work" = does steering it move the output *toward its pole*
AND stay coherent. The first metric used **change-magnitude** (how different from baseline) — and it
FALSELY passed dials that merely *reformat* (skeptical adding `### headers`) as "usable." Fixed to
**direction-aware**: re-encode each steered reply and project onto the dial's OWN axis. This correctly
collapsed the reformat-only dials while keeping genuine tonal ones — validated starkly (warm/poetic strong
positive projection + real range; skeptical/candid → ~0, empty range, despite high change-magnitude).
Plus a coherence gate (derail ceiling) and a shuffled-direction null (a random direction reads ~0 on the
real axis).

## The finding

- **The surface-vs-cognitive hypothesis HELD in aggregate but is noisy per-dial.** surface 30/42 usable
  (71%) vs cognitive 5/14 (36%), gap 0.36. But individual predictions failed a lot (confident, intuitive,
  first_principles *feel* cognitive yet steer; plainspoken, prose *feel* surface yet die).
- **The real axis is "distinctive positive signature," not surface-vs-cognitive.** Dials whose pole
  distinction has rich positive markers steer (intuitive: "gut/sense"; confident: hedge-words-absent — a
  surface marker of a cognitive-feeling quality). Dials defined by an **absence/default** die — you can't
  steer *toward* "plainspoken" (no ornament) or "prose" (no lists) when the model's already there.
- **Pole wording matters a lot.** `confident` was strongly live as a built-in dial but dead as a
  library-worded custom — same concept, different pole text. So "dead" = "these poles didn't calibrate,"
  not "impossible"; dead dials are retryable with better poles.

## Why the human pass (eyeball caught the metric's ceiling)

The direction-aware metric is a good LIVE/DEAD filter but **optimistic at the top**: at high dose several
"usable" dials clip or ramble *before* the crude coherence gate catches it (noir @1.5 = terse bullets, no
noir voice; first_principles @1.5 = "...as an AI language model from scratch provided by Alibaba"). A
human-read pass over the 47 kept only dials that genuinely READ AS their tone (33), with **conservative
ranges** dropped back from the raw usable_max wherever the top dose degraded. 14 dropped for producing no
recognizable tone (noir, blunt, assertive, tough_love, professional, ...) — several with factual
inversions at high dose.

## The 33 shipped dials

`affect_tone` (warm, tender, wry, playful), `register` (formal, academic, ceremonious, slangy),
`voice_flavor` (poetic, storyteller, folksy, technical_writer), `persona_composite` (cheerleader, coach,
editor, professor), `imagery` (concrete, vivid, minimalist), `emotional_posture` (empathetic, nurturing,
reassuring), `audience_level` (eli5, expert, jargon_free), `verbosity` (concise, detailed), `rhetoric`
(humorous, exemplifying), `reasoning_style` (intuitive, first_principles), `format` (stepwise),
`directness` (diplomatic).

## Caveats & deploy

**Per-model (Law #6):** ranges + which dials survive are Qwen-7B-specific; a different model needs its own
sweep (a natural idle/overnight job — Gemma-9B sweep is queued but not run). The metric measures
*projection onto the dial's linear axis*, which correlates with but isn't identical to human-perceived
tone (the human pass is the safety net). One model / one seed / one prompt sample. **To go live:** register
the 33 as studio dials + drop a `~/.clozn/dial_calibration.json` (the `/steer/axes` + `behavior.js` wiring
already caps sliders + greys dead dials) — the remaining integration step.
