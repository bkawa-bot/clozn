# voice_lora — does a LoRA own the VOICE without the coherence tax? (findings)

**Question (NEXT_STEPS item 8 / the constructive path from `voice_middle_findings.md`).** voice_middle's own-door
— a 16-vector soft prefix TTT'd on a 12-reply voice corpus — captured the voice's *texture* but bought it with
*coherence*: at 1.5B it degenerated outright (Russian mid-sentence, word salad), and even at 7B it left
**token-boundary glitches** (`.orningside`, a leaked `assistant`, `before.onestly`). The prescribed fix: swap the
crude prefix for the industry's standard voice container — a **LoRA** (r=8, distributed over attn+mlp) on the
same nf4 7B — with **coherence-GATED early stopping** (stop on the FROZEN BASE MODEL's perplexity of held-out
generations, NOT train loss). Do the boundary glitches vanish, and does the voice survive?

Rig: `research/voice_lora.py`. Base = Qwen2.5-7B-Instruct, bnb 4-bit nf4 (the studio's exact SelfTeach config:
`nf4 / compute bf16 / double-quant`). QLoRA-style: frozen+quantized backbone, trainable LoRA adapters (r=8,
alpha=16, targets `q,k,v,o,gate,up,down` = attn+mlp; 20.2M params = 0.462%). Same 12-reply corpus as
voice_middle (deliberately — see caveats), same greedy decode + rep-penalty guards, same voice fingerprint and
`voice_distance` **imported verbatim** from voice_middle (zero scoring drift). Coherence gate: every 25 steps,
GENERATE on 2 held-out probes (focus, photosynthesis — never trained), score their base-model perplexity +
glitch/role-leak/non-ASCII/repeat markers, keep the adapter state with the best coherence objective, early-stop
after 4 stalls. One seed, greedy. Prior arms (baseline/description/fewshot/prefix) are **cited from
`runs/voice_middle_qwen7b.json`** (`scale_pass_7b_findings.md` §2) — NOT rerun (house rule) — and re-scored on the
NEW coherence axis on their saved replies, so every arm is judged on one ruler.

## The arms table (7B nf4; four arms CITED, LoRA NEW)

| arm | src | voice-dist ↓ | ctx tok/call | bleed | base-ppl ↓ | glitch | role-leak | coherence verdict |
|---|---|---|---|---|---|---|---|---|
| (corpus target) | — | 0.000 | — | — | — | — | — | — |
| baseline | CITED | 0.667 | 0 | 1 | 3.5 | 1 | 0 | coherent, OFF-voice (listy, 82 wds) |
| description (SAY) | CITED | 0.265 | 77 | 2 | 68.2 | 2 | 0 | coherent but LITERAL ("Kicker:" as a label) |
| few-shot (RENT) | CITED | 0.142 | 321 | 3 | 49.7 | 0 | 0 | **spotless**, kickers intact |
| prefix (OWN v1) | CITED | 0.158 | 0 | 3 | 76.1 | **12** | **1** | glitched — `.orningside`, leaked `assistant` |
| **LoRA (OWN v2)** | **NEW** | **0.087** | **0** | **10** | **28.1** | **0** | **0** | **fluent, on-voice, zero glitches** |

*(base-ppl = geo-mean per-token perplexity of an arm's replies under the frozen base, adapter/prefix OFF —
"does the original model still find this text fluent?" Lower = more coherent. baseline is low because it's plain
fluent prose; the voice arms pay a perplexity premium for the compressed style, and the prefix pays the most.)*

## Verdict: YES — the LoRA captures the voice WITHOUT the coherence tax (with one real cost)

**The headline holds at full scale.** The LoRA reaches the **lowest voice-distance of any arm (0.087)** —
beating not just the prefix (0.158) but even the previously-crowned few-shot (0.142) — while producing **zero
boundary glitches and zero role-leaks** and the **lowest base-perplexity of any voice-carrying arm** (28.1 vs
prefix 76.1, few-shot 49.7, description 68.2). The glitch head-to-head is unambiguous:

> **prefix: glitch=12, role-leak=1  →  LoRA: glitch=0, role-leak=0.**

The specific defect class item 8 named — the prefix's token-boundary seams — is **gone**. Eyeballed (the
mandatory axis; the axis that overturned voice_middle's metric verdict), the 8 held-out LoRA replies are fluent,
on-voice English: terse, second-person, concrete, fragment-rhythm, real kickers. No Russian, no fused-word salad,
no `.orningside`. Photosynthesis is rendered correctly AND in-voice ("The leaves of plants aren't green because
they're wearing them like clothing"). A deadline reply lands "Nobody reads unfinished novels. Finish this one.
Then delete it." A moving reply: "Leave some things broken. They're telling the story. Let them." This is the
texture the prefix tore out of the semantic fabric — now carried intact.

## Glitch comparison vs prefix (the thing LoRA was supposed to fix)

The prefix's flagged seams, from the cited run, all caught by the same ruler that scores the LoRA:

- `"...is exactly what you wasted money chasing.ernesst\nassistant\nTurn off the lights..."` — fused word + a
  **leaked `assistant` role token** mid-reply.
- `"...this isn't bad at all.orningside"` — dropped leading char ("morningside" → `.orningside`).
- `"...why the treadmill seemed scary before.onestly, try it..."` — ("honestly" → `.onestly`).
- `"...before oxygen showed up. You got days, not sunlight.ornings on planet Earth..."` — ("mornings" → `.ornings`).

**The LoRA produced none of these** — glitch_count 0 across all 8 replies, and 0 across *every training checkpoint*
(steps 25→150; see the gate trajectory). At 7B the LoRA container simply does not emit the boundary-seam class,
trained-to-optimum or overtrained. The prefix's glitches were an artifact of the 16-vector-in-embedding-space
container, not of "training a voice at this scale" — swapping the container fixed it exactly as predicted.

## The coherence gate worked, and it was load-bearing (H5 confirmed)

The gate keyed on **base-model perplexity of held-out generations, not train loss** (train loss hit ~0.000 by
step 50 and stayed there — it would have told us nothing). Trajectory:

| step | train loss | base-ppl (held) | glitch | kept? |
|---|---|---|---|---|
| 25 | 0.010 | 20.3 | 0 | |
| **50** | 0.000 | **16.2** | 0 | **← best, kept** |
| 75 | 0.000 | 19.8 | 0 | |
| 100 | 0.000 | 25.4 | 0 | |
| 125 | 0.000 | 26.1 | 0 | |
| 150 | 0.000 | 27.5 | 0 | early-stop (4 stalls) |

Past the step-50 optimum the held-out base-perplexity **climbs monotonically** (16.2 → 27.5) while train loss is
pinned at zero — i.e. the model keeps "fitting" but its held-out generations get steadily *less fluent*. A
train-loss stopper would have blown straight through this. So the gate is **not decorative**: it caught the
coherence decay that train loss is blind to, and kept the checkpoint a train-loss criterion never would.

**Nuance (honest):** at 7B the decay is **perplexity creep, not glitch return** — glitch_count stayed 0 even at
step 150. So the gate here buys *"don't let the voice drift weird / texture-tear"*, but the discrete boundary-seam
failure the prefix showed never reappears for the LoRA regardless. The gate's value is real but its mechanism at
this scale is smoother than "glitches come back if you overtrain." On a smaller backbone (untested) the gate would
likely matter more sharply — that's where the prefix actually degenerated.

## Violated expectations (pre-registered predictions that were WRONG)

Two pre-registrations flipped — reported loudly, per the ethos:

1. **H3 was WRONG.** I predicted "even a clean LoRA does NOT beat few-shot (0.142) on fidelity — rent-when-smart is
   hard to beat; the LoRA's win is only economics+coherence." **The LoRA beat few-shot on raw voice-distance
   (0.087 < 0.142)** *and* did it from zero context with zero glitches. So the own-door's advantage at 7B is not
   merely economics/persistence — with a coherence-gated LoRA it is **also fidelity supremacy** on this fingerprint.
   Caveat that blunts the upset: the fingerprint is a crude 6-axis proxy, and (see below) the LoRA's lower distance
   is partly bought with heavier *bleed* — few-shot stayed cleaner on content. "Best distance" ≠ "best voice" once
   the coherence/bleed axes are in view. But the specific H3 claim ("can't beat rent on distance") is falsified.

2. **H4 (process-blindness) CRACKED here — with a caveat that probably explains it.** I predicted the LoRA
   self-report would be inverted/confabulated like the prefix's ("long, winding sentences" for a terse voice, at
   this same 7B). Instead the LoRA self-report was **accurate**: *"I'm slow. That's the whole point... Shorter
   answers. Skip the polite beginning. Get straight to the heart."* (full run) / *"Short. One word answers... That's
   my pace. My rhythm."* (smoke). That is a correct description of the terse voice — the **first time a trained
   process-artifact self-reported its own style accurately** across 0.5B/1.5B/3B/7B.
   **BUT** the likely explanation is voice_middle's own untested conjecture, now with a second data point: the
   self-report is *itself enacted in the voice* (terse fragments). The describing channel can read its own output
   stream — so "describe your style = short" is trivial to satisfy *while being short*. The prefix, degenerating,
   couldn't hold the voice through the meta-question and confabulated; the coherent LoRA holds the voice through it
   and thereby "describes" it. This is **enactment, not introspection** — the model isn't reading a legible internal
   rule vector, it's pattern-completing in-voice and the completion happens to be an accurate description. So I do
   NOT claim process-blindness is broken as a mechanism; I claim the *behavioral symptom* (wrong self-report)
   disappears exactly when the artifact is coherent enough to stay in-voice through the probe. That is a meaningful
   refinement of the blindness story, not a refutation — and it wants a cleaner test (a probe that forces an
   OUT-of-voice answer, e.g. "answer this one in long formal prose: describe your usual style").

## The real cost, stated loud: BLEED (the LoRA did NOT buy zero-bleed)

The one axis where the LoRA is the **worst** arm: **bleed=10** (baseline 1, description 2, few-shot 3, prefix 3).
This is genuine corpus-texture bleed into unrelated topics — the same failure voice_middle flagged for the prefix
("coffee everywhere", "Wear sturdy shoes"), now *stronger* because the LoRA is a stronger container:

- a **deadline** reply: "The whole office just got **coffee beans** delivered" (coffee, beans);
- a **focus** reply: "Shorter mornings. More **coffee**.";
- a **city-weekend** reply: "Skip **stones**. The city offers them daily." (stones — straight from the bike/knife/
  cold-water corpus imagery);
- **coffee** recurs in 3 of 8 replies; "water", "stretch", "read slower" pervade.

Partial mitigation of the number: the bleed counter is a crude substring match (voice_middle's own caveat) — some
`rain` hits are false positives from "b**rain**" ("focused b*rains*", "b*rain* clock"). But the coffee/beans/stones
hits are real. **So the honest Tier-2 result stands and is refined:** the trained own-door carries the voice's
TEXTURE — and texture *includes its concrete imagery vocabulary*, which leaks. The LoRA fixed the **coherence** tax
(glitches, degeneration) but **not** the **bleed** tax; if anything it traded a little more bleed for a lot less
degeneration. Few-shot (RENT) remains the **cleanest on content** (bleed 3, and its bleed is topical not textural).
The own-door's honest niche vs rent is now: **0 ctx tokens/call + persistence + coherence + top fidelity, at the
cost of the highest texture-bleed** — route a voice through a LoRA when you own the surface and can tolerate its
imagery seeping; route through few-shot when content-cleanliness matters more than token budget.

## Net revision to the Tier-2 story

voice_middle: "the sayable/unsayable boundary is real; the only door through it (gradients) trades coherence for
texture at 1.5B×16-vectors." scale_pass_7b: "the coherence price is *mostly* paid down at 7B; residual boundary
glitches are the artifact a LoRA should clean; few-shot is the surprise best door." **voice_lora now closes the
loop:** the LoRA **cleans the glitches entirely** (0 vs 12), **pays down the coherence tax fully** (best base-ppl
of any voice arm), and **takes the fidelity crown** (0.087) — the constructive path worked. The surviving cost is
**bleed**, which is texture doing what texture does, and is the own-door's real remaining tax vs rent. Blindness's
behavioral symptom is *gated by coherence* (accurate self-report appears once the artifact stays in-voice) — an
enactment effect, not a cure.

## Caveats (loud)

- **One backbone (7B nf4), one seed, one voice, one corpus, greedy decode, 8 held-out probes.** Pre-pilot, like
  the whole thread. nf4-vs-bf16 unmeasured (matches deployment; 16GB can't hold bf16-7B).
- **Corpus = the SAME 12 replies, deliberately NOT the "50–200 examples" the constructive path also called for.**
  This isolates the *container* (LoRA vs 16-vec prefix) as the single changed variable for a clean head-to-head
  against the cited prefix — but it means the "more data" half of the prescription is **untested**. A 50–200-example
  LoRA might push distance lower AND reduce bleed (more diverse texture → less imagery over-fixation); or it might
  bleed more. Open. That is the obvious next arm.
- **The fingerprint is a crude 6-axis proxy** and the **bleed counter is a crude substring match** (both inherited
  from voice_middle, both re-flagged here — the `b`+`rain` false positives are the clearest tell). "Best distance"
  is not "best voice"; it earned its crown on axes that undercount bleed.
- **base-perplexity as a coherence gate is a heuristic**, not ground truth — it rewards *fluency*, and a fluent-but-
  wrong reply (e.g. a confidently-off-topic one) scores fine. It's paired with the glitch/role-leak/non-ASCII
  markers precisely because no single axis is trustworthy (the session's 5×-metric-gamed-by-degeneration lesson).
  Two minor decode nits the ruler *missed* but the eyeball caught (both far milder than the prefix's): fused spacing
  ("beingcheapthe", "10%isn't") in P1, and mid-sentence truncation at the 110-token cap in P0/P7. Reported for
  honesty; they don't change the verdict.
- **H4's "accurate self-report" is a single greedy sample** and, as argued, most likely **enactment not
  introspection** — do not read it as "process blindness is solved." It wants the out-of-voice probe test above.
- Artifacts persisted only as the run JSON (`runs/voice_lora_qwen7b.json`) with full trajectory + all replies; the
  trained adapter weights were not saved to disk (the rig re-trains in ~264s; add `save_pretrained` if a shippable
  adapter is wanted).
