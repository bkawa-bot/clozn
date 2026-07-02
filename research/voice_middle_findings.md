# voice_middle — can a VOICE be owned better than said or rented? (findings)

**Question (Tier 2 of the memory architecture).** A voice — terse, second-person, concrete, fragment-
rhythm, kicker endings — defined ONLY by a 12-reply corpus. Four doors: `baseline`, `description` (SAY:
the best ~80-word verbal steelman), `fewshot` (RENT: 6 pairs in context), `prefix` (OWN: m=16 TTT, 220
steps, loss→0.63). Scored on 8 held-out disjoint-topic probes with a transparent voice fingerprint
(wps/fragments/hedge/lists/disclaimers/you-rate) vs the corpus's own fingerprint, plus bleed, context
cost, self-report. Qwen2.5-1.5B, one seed, greedy.

## The numbers — and why they cannot be taken at face value

| condition | voice-distance ↓ | wps | frag | you | mean words | bleed | ctx tok |
|---|---|---|---|---|---|---|---|
| (corpus target) | 0 | 5.9 | .456 | .251 | 32 | — | — |
| baseline | 0.652 | 11.5 | .283 | .295* | 78 | 2 | 0 |
| description | 0.349 | 11.5 | .113 | .113 | 26 | 1 | 77 |
| fewshot | 0.534 | 16.6 | .091 | .021 | 77 | 3 | 321 |
| **prefix** | **0.270** | 10.1 | .204 | .367 | 89 | 2 | 0 |

**By the metric, the prefix wins (0.270 < 0.349) — pre-registration #1 nominally confirmed. By the
eyeball, OVERTURNED: the prefix replies are substantially degenerate.** Samples: *"A Qwen can type faster
than a regular one… Different earths dry fruit better, shorter leaves are usually easier for my fingers to
grip"* (laptops); a reply that lapses into **Russian mid-sentence** («ьте сохранить для себя»); a spurious
URL + emoji; *"shorter and greener after that first day of school on April 1st, 2057"* (photosynthesis);
fused-word junk (*"filmestohearthemoviey"*). The prefix bought its fingerprint with coherence — short
you-addressed fragments of nonsense. **Fourth instance this session of a metric gamed by degeneration**
(after the warm-dial lexicon). A style score without a sanity axis is not a receipt.

## The genuinely valuable result: the sayable/unsayable boundary, measured

Look at WHERE each door failed:

- **description** (coherent throughout) captured exactly the STATABLE constraints — short (26 words ✓),
  no lists ✓, no disclaimers ✓, hedging down ✓ — and missed exactly the UNSTATED texture: fragment rhythm
  **0.113 vs target 0.456**, second-person stance **0.113 vs 0.251**, and it drifted into first-person
  lyric ("anticipation gnaws at my insides") where the voice is second-person address. The description
  did what descriptions do: transmitted the rules, not the gait.
- **prefix** captured exactly the complement — you-stance **0.367**, fragmentary rhythm, the corpus's
  imagery vocabulary — and lost semantic coherence doing it. It also smuggled corpus TEXTURE into
  unrelated topics ("water your plants" advice inside a moving-apartments reply; "Wear sturdy shoes";
  coffee everywhere) — texture-bleed the topic lexicon undercounts (scored 2, reads pervasive).
- **fewshot** at 1.5B simply failed: treated the example history as conversation, not a style model
  (you-rate 0.021, 77 words). Rent requires a renter smart enough to infer the lease.

So the boundary the whole thread predicted is REAL and now measured: **what could be said, the prompt
delivered; what could not be said, only the trained artifact absorbed — and at 1.5B/16-vectors it absorbed
the texture by tearing it out of the semantic fabric.** The middle exists. This key is too crude.

## Other pre-registrations

2. *Fewshot bleeds, prefix ≈ zero*: **NOT confirmed** — lexicon bleed all within noise of baseline
   (3/2/2/1); qualitatively the PREFIX is the worse offender via texture-bleed. Honest miss.
3. *Cost*: confirmed trivially — 0 tokens forever vs 77 (description) vs 321 (fewshot) per call.
4. *Self-report blindness*: **partially overturned, interestingly.** The prefixed model said *"My concise,
   rhythmical writing style…"* — two accurate style adjectives — then confabulated an entire basketball
   analogy and degenerated. Partial sight + confident confabulation. Conjecture (untested): style
   self-description gets easier when the answer is *enacted in the answer's own generation* — the
   describing channel can read its own output stream. One probe; anecdote, not result.

## Verdict

**Tier 2 is not validated at this scale — and the experiment is more useful than a validation.** It
demonstrated (a) the sayable/unsayable boundary as a measurable split across conditions, (b) that the
only door through it (gradients) currently trades coherence for texture at 1.5B×16-vectors×12-examples,
and (c) that any Tier-2 receipt MUST carry a coherence/sanity axis alongside style axes, or degeneration
games it. The receipt discipline caught its own metric — again.

## What would make it work (the constructive path, untested)

Bigger backbone (7B — capacity to hold texture without semantic collapse); **LoRA instead of a 16-vector
prefix** (distributed capacity; the industry's standard container for voice); 50–200 examples not 12;
coherence-gated early stopping (stop on base-model-perplexity or a degeneration detector, not train loss);
and the receipt panel from the roadmap as the certification harness.

## Caveats

One model (1.5B), one seed, one voice, greedy decode, 8 probes; the fingerprint axes are crude proxies;
the bleed lexicon includes common words (baseline hits 2); self-report is a single greedy sample.
