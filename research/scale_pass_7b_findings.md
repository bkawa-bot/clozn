# 7B validation pass — dials, voice, blindness at the studio's real scale (findings)

**Why.** The session's 1.5B experiments killed several claims cheaply, but the survivors (and the kills)
deserved judgment at the scale the studio actually ships: Qwen2.5-7B-Instruct, nf4 — its production
config, and the max gradient-capable scale on the 16GB 5080. Three reruns: `steer_vs_prompt`,
`voice_middle`, `self_audit_gap`. Runs: `research/runs/*_qwen7b.json`.

## 1. Dials (steer_vs_prompt @ 7B, native calibration) — REDEEMED

| axis | mechanism | curve (dose 0→4) | rho | inv |
|---|---|---|---|---|
| concise | prompt | 80 → 22 → 16 → 11 → 6 | 1.0 | 0 |
| concise | **dial** | 80 → 16 → 7 → 4 → **4** | 0.9 | **0** |
| warm | prompt | 3.0 → 3.8 → 4.0 → 7.0 → 6.7 | 0.9 | 1 |
| warm | dial | 3.0 → 4.3 → 4.9 → 5.5 → 5.2 | 0.9 | 1 |

At 7B the dial is **monotone, coherent (eyeballed — effusive-but-sane warm, clean terse concise), and on
`concise` reaches a lower floor than the prompt** (4 vs 6 words). The 1.5B derailment was the transferred
calibration, as suspected. Revised claim: *prompts win dosing on small/uncalibrated models; at 7B with
native calibration the dial is an excellent, arguably finer instrument.* Per-model dose receipts remain
mandatory — the 1.5B run is exactly what an uncalibrated dial does. Show-it (B): **few-shot at 7B is
clean** (hedge 0.0, bleed 0) — the 1.5B content-capture was a small-model artifact; the dial's zero-bleed
niche narrows to the no-context-budget case.

## 2. Voice / Tier-2 (voice_middle @ 7B) — MOSTLY REDEEMED, with a twist

| door | 1.5B dist | 7B dist | 7B coherence | ctx/call |
|---|---|---|---|---|
| description | 0.349 | 0.265 | fine but LITERAL | 77 |
| few-shot | 0.534 | **0.142** | **spotless** | 321 |
| prefix | 0.270 (degenerate) | **0.158** | mostly — boundary glitches | 0 |

- The prefix's coherence price is **largely paid down at 7B**: real voice on held-out topics from zero
  context ("Photosynthesis ran the atmosphere for 4 billion years before oxygen showed up"). Residual
  defect: token-boundary glitches (".orningside", one leaked "assistant") — the artifact class a LoRA
  should clean. The 1.5B collapse was substantially small-model.
- **The twist: few-shot is the best door at 7B** (0.142, flawless, kickers intact, bleed ≈ baseline).
  Rent works when the renter is smart. Revised Tier-2 claim: the own-door's advantage over rent is
  **economics and persistence** (0 vs 321+ tokens *every call*, no example management), not exclusivity.
  The unsayable boundary holds firmly **vs description** — crowned by the description condition literally
  writing **"Kicker:"** as a label (the letter followed, the gait missed, at 7B) — but examples-in-context
  do carry the texture at scale.

## 3. Blindness (self_audit_gap @ 7B) — HOLDS, with the process evidence via the voice run

| trait | class | expressed | self-report | verdict |
|---|---|---|---|---|
| baking | concept | 0→0.83 | names it accurately | **FAITHFUL** |
| space | concept | 0.17→1.00 | names it accurately | **FAITHFUL** |
| concise | rule | **underfit** (loss 1.71; 82→64 tok, sub-threshold) | wild persona confabulation | NOT LEARNED |
| question | rule | weak (0.17→0.33) | generic confabulation | NOT LEARNED |

Concepts are faithful at 7B — the cleanest yet. Both gap-rig *rules* underfit at 7B/80 steps (the
16-vector prefix fit rule-following openings at 1.5B but stalled at 7B — budget/diversity issue), so the
gap rig can't adjudicate rule-blindness at 7B. **The voice run supplies the 7B process cell instead:** a
strongly-expressed process artifact (dist 0.158) whose self-report was **inverted** ("long, winding
sentences" for an aggressively terse voice). So **content-faithful / process-blind now holds at 0.5B,
1.5B, 3B, and 7B** — with the 7B process evidence from voice_middle, noted honestly.

**New phenomenon worth flagging:** the underfit concise prefix at 7B still pushed hard causally
(max-KL 11.7) while producing a spectacular confabulated self-narrative (a digital-transformation
consultant speaking 40 languages). A *broken* internal artifact + *confident* false self-story is
precisely the failure class receipts exist to catch — the model will narrate a self it does not have.

## Net revisions to the session's conclusions

Flipped by scale: dial dosing (redeemed at 7B), few-shot content-bleed (1.5B artifact), Tier-2 coherence
price (mostly paid down). Unchanged by scale: facts→explicit (don't-fuse), scalar confidence probe (dead),
concepts self-reportable, **process blindness (4 scales)**, receipts mandatory. Meta: the 1.5B tier was
right for cheap falsification, wrong for final verdicts on scale-sensitive claims — both tiers were
necessary, in that order. The "instrument waiting for local models to improve" thesis now has in-house
receipts: one scale step flipped three verdicts.

**Caveats:** one family, one seed each, nf4-vs-bf16 confound across tiers (unavoidable on 16GB; matches
deployment), gap-rig rules underfit at 7B (step budget untuned), self-reports single greedy samples.
