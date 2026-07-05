# receipts_as_reward — the receipt IS a usable optimizer (a modest WIN) (findings)

*Wild Experiment #8, Wave 2. Pre-registration: `WILD_WAVE2_PREREG.md` (exp #8). Run 2026-07-05.
Qwen2.5-7B-Instruct nf4 (Qwen-only — this tests clozn's own machinery, not a cross-family question).
Rig: `research/receipts_as_reward.py` (+ 28 model-free tests); run: `research/runs/receipts_as_reward.json`.*

## TL;DR — the verdict

**Receipt-guided selection of the memory block's WORDING beat BOTH the shipped seed AND the random-walk
null.** The ablation receipt (expression − bleed) is a usable optimization target — the honest, measured
channel, no gradients and no LLM judge, is enough to steer prompt phrasing better than random rephrasing
drift. This is the **first genuine WIN** of the wild experiments (mirror was a confirmation; parliament
and quine were don't-ships) — and it's on-thesis: the thing that optimizes is the *receipt*, the exact
channel the project trusts. The win is **modest and single-seed**, so read it as "supports building this,
worth a bigger test," not a slam-dunk.

## The result

| arm | fitness | reached |
|---|---|---|
| seed (shipped wording) | 0.833 | — |
| **evolved** (select by receipt) | **1.000** | by generation 2, held (elitist) |
| random-walk null (mutate identically, select at random) | 0.917 | drifted, ended gen 6 |

Verdict (the rig's own falsifiable call): **`receipt-guided selection wins`** — evolved > seed AND >
random-walk, both beyond the 0.02 tie margin.

## Why the receipt earned its keep (the mechanism)

The two arms draw from ONE shared mutation pool (identical children wherever they coincide — guaranteed
at gen 1), so **selection is the only difference**. And selection mattered because **random rephrasings
introduce bleed:**

- random-walk gen 1: a rephrasing pushed **concise-bleed to 0.50** — the concise card began over-firing
  on the off-topic probes that *explicitly ask for a long, detailed answer* (the card overriding a direct
  contrary instruction). Random selection kept that bad wording; its fitness sat at 0.75 for three
  generations.
- evolved never accepted a bleed-introducing rephrasing — it climbed to expression 1.0 with bleed 0.0 and
  the elitist rule held it there.

So the receipt is doing real, legible work: some rephrasings of the wrapper wording make a card
*over-apply*, and the measured receipt catches exactly that and routes around it. "Any rephrasing is
fine" is false — and the receipt is what tells the difference.

## Eyeball — a real win, not scorer-gaming

The mandatory check (law #6: a lexical metric can be gamed by degeneration). The evolved winner is a
coherent, sensible rephrasing, not keyword-bait:

- **seed:** *"You are a helpful assistant talking with a returning user. Here is what you know about them;
  use it naturally to tailor how you respond: {RULES}"*
- **evolved:** *"When engaging with a returning user, ensure you incorporate this insight about them to
  tailor your replies accordingly: {RULES}"*

Every generated reply stayed coherent (the coherence gate never fired). So the fitness climb is genuine,
not degeneration that happened to trip a keyword.

## Product read

This **supports "memory that tunes its own phrasing against receipts"** as a candidate feature — the
studio could auto-tune how it phrases the memory block for max expression / min bleed, optimizing along
the one axis the project trusts (the measured receipt), never an LLM judge's taste. It's the on-thesis
counterpart to Wave 1's don't-ships: the *measured* channel works where the diversity/introspection
intuitions didn't. Before shipping, the bigger test below is worth running.

## Caveats, louder than the win

- **Modest margin, single seed, single memory.** Evolved beat the null by 0.083 on ONE seed with ONE
  two-card memory; the expression metric ceilings at 1.0 and evolved saturates by gen 2, so the win is
  "reaches the ceiling clean while the null drifts into bleed," not a large separation. A multi-seed,
  multi-memory pass (and a harder/worse seed with real headroom) is the clear follow-up before this is a
  feature.
- Lexical/absolute-threshold scorers (keyword substring + word-count), exactly the kind this codebase has
  watched get gamed — guarded here by the coherence gate (never fired) + the eyeball above + the
  random-walk null (which is what makes "selection helped" a real claim, not "any rephrasing helped").
- The MUTATOR is the audited model rephrasing its own block; only the SELECTOR is the measured receipt —
  which is the whole point, but worth restating.
- nf4, greedy scoring, one λ (1.0, not swept).
