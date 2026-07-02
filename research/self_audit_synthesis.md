# Self-audit thread — strategic synthesis ("is it actually useful?")

*Written 2026-07-01, after the confabulation-gap / cure / black-box / scale experiments. Deliberately
clear-eyed: what's useful vs merely interesting, what to keep, what to drop. Caveats louder than wins.*

## TL;DR

One real, reproducible finding with one clear product implication; a pile of interesting-but-not-yet-useful
science around it; and one genuinely useful strategic reframe. The finding: **a model changes how it behaves
and cannot introspect that it changed — for *process* (style/rules), while it *can* for *content* (topics).**
It reproduces across 0.5B→3B. The implication: **in the studio, show measured receipts, never let the model
narrate its own memory.** The reframe: **clozn is the legible memory + verification layer beside a frontier
model, not a rival to it** — and the verification instrument ports to closed/API models.

## What was actually established (with honesty about strength)

1. **Content-legible / process-blind asymmetry — ROBUST (for its scope).** Trained (soft-prefix) to adopt a
   *concept* (baking, space), the model self-reports it accurately. Trained to adopt a *process/rule* (be
   concise), it changes hugely (72→19, 84→16 tokens) and, asked what it learned, does **not** name it —
   confabulates instead. Holds at 0.5B, 1.5B, 3B (6× params). *Strength:* good — 3 sizes, large effect sizes,
   clean on the `concise` trait. *Limits:* one model family (Qwen2.5), one seed, N=4 traits, 7B untested.

2. **The naive "show it the receipt" cure FAILS; a STRUCTURED receipt works — PARTIAL.** Handing the model its
   raw with/without transcript made its self-description *worse* (vague/inverted); handing it the *digested
   facts* ("13 words vs 72") got "it makes me respond more concisely." *Strength:* suggestive, and it cured the
   exact blindness the raw/white-box path couldn't. *Limits:* noisy at 1.5B (worked for length, muffed topical
   traits, inverted `question`).

3. **The instrument ports to closed/API models — SOLID (mechanism), the honest half.** B (behavioural ablation)
   ports fully; C ports as an output-space logprob-KL (weaker); the structured-receipt self-audit is
   black-box-native. What you *lose* to an API is hard: mechanism (SAE/steering/circuits) and the *internalized
   memory substrate* (black-box memory is a prompt — self-legible but not persistent weight-state).

4. **The scalar "how sure are you 0–100" self-report probe is DEAD — clean negative.** Saturates at every scale
   (0.5B≈0, 1.5B≈85–95). Don't ship introspective confidence; use behaviour + causal trace.

## Useful vs interesting — the honest triage

- **Useful now:**
  - *Product decision:* **receipts, not self-narration.** When a user edits memory/personality, show measured
    proof (B: behaviour moved; C: where it acts) and flag self-description mismatches. Finding #1 is the
    justification; it's the studio's honest-differentiator ("the only local AI memory that shows the receipts").
  - *Portability:* that feature works **beside a closed model** (finding #3) — audit whether a system-prompt /
    memory actually changed behaviour, via API only.
  - *Strategic reframe:* clozn = legible **memory + verification layer** next to the frontier model. Dissolves
    the "local model isn't smart enough" worry — M doesn't need to be smart; it's the memory/interpretability
    organ.
- **Interesting, not yet useful:**
  - The blindness result *as science*. It's real but bounded (one family, small N); as a standalone claim about
    "LLMs" it's pre-pilot. It mostly sharpens a known caution (don't trust model self-explanation) into a
    specific, measured shape (content vs process). Worth a proper writeup only if validated (below).
  - The structured-receipt cure — promising, too noisy at this scale to lean on.
- **Drop / don't chase:**
  - Scalar introspective confidence (dead).
  - `question` as a trait (the 16-vector prefix underfits ending-based rules — a mechanism artifact, not signal).
  - Any implication that this gives *mechanistic* interpretability of closed models — it does not. Hard wall.

## What to build / validate, ranked by ROI

1. **Weld "receipts" into the studio** (highest ROI, days). A panel: added a memory → here's the measured
   behaviour delta + causal footprint + a ⚠ when the model's self-description doesn't match. Turns a research
   result into the product's honesty feature. Uses machinery that already exists (`trace`, behaviour A/B).
2. **The proxy-memory demo** (bigger bet, the reframe made real). A tiny local M as a closed model's legible,
   auditable memory, driven by text, inspectable in `studio.html`. This is the strategic story as a demo.
3. **Validate the finding properly, IF pursuing the science** (only if #1/#2 want the backing): add 7B, a
   second model family (Llama/Gemma), ≥3 seeds, ≥8 traits with clean process/content balance, and a held-out
   judge instead of keyword scoring. Then it's a defensible result, not a sketch.

## Risks & what would change my mind

- **Motivated reasoning.** These threads are fun; fun feels useful. The guards: does it survive scale (it did to
  3B) and is there a user who'd act on it (yes for the studio feature; thin for standalone science).
- **Single-family risk.** If Llama/Gemma *don't* show process-blindness, finding #1 is a Qwen quirk — downgrade.
- **"So what" risk for the science.** If a 7B *does* self-report its process, the concern softens for capable
  models and the receipts feature matters less at frontier scale (though still for local/small M). Untested —
  the one experiment most worth running next for the science.

## Bottom line

Keep the **product decision** and the **reframe** — those are the useful outputs, and they're real. Treat the
**blindness result** as a validated-enough *justification for the feature*, not as a finished research
contribution. If you want it to be a contribution, the validation pass (#3) is the price. The studio remains the
useful artifact; this thread earns its keep by making the studio *honest by construction*.
