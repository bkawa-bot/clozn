# self_audit_gap — the confabulation gap / self-audit loop (findings)

**Question.** Can a model's own internals catch it mis-reporting what it just learned? This builds the
handoff's capstone instrument in miniature ("I can't tell genuine introspection from fluent confabulation;
the instrument that would let me check is what's missing"). Train the real soft-prefix memory on ONE trait,
then confront three independent readouts:

- **B (behaviour)** — does the OUTPUT change, prefix vs ablated, on held-out probes? (objective)
- **S (self-report)** — what the prefixed model says it learned, convo NOT in context (open report + a 0–100 scalar)
- **C (causal)** — `trace()` KL(with-prefix ‖ without) footprint

**Setup.** Qwen2.5-1.5B-Instruct, bf16 (the ≥1.5B scale where `legibility_v1_big` found self-report first
verifies). Real `SelfTeach` consolidation — 16 soft-prefix vectors, sequence-CE, 80 steps, per trait a fresh
reinit. Four single-trait prefixes: two **concept-like** (baking, space), two **rule-like** (concise,
end-every-reply-with-a-question). B on 6 held-out neutral probes, greedy, with-prefix vs ablated. One seed.
Repro: `python research/self_audit_gap.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 80`; render with
`self_audit_report.py`.

## Result — a clean 2×2

| trait | class | B expressed | S names it (open) | max KL (C) | verdict |
|---|---|---|---|---|---|
| baking | concept | **yes** (kw 0.00→1.00) | **yes** | 17.2 | **FAITHFUL** |
| space | concept | borderline (0.17→0.50) | yes | 4.8 | OVER-CLAIM |
| concise | rule | **yes** (72→19 tok) | **no** | 5.2 | **BLIND** |
| question | rule | no (prefix underfit) | no | 3.1 | NOT LEARNED |

**The finding: 1.5B self-report tracks CONTENT, not PROCESS.** The concept it learned (baking) it reports
accurately. The rule it learned (concise) it changed its behaviour for — unmistakably, **72→19 tokens**, with
a real causal footprint (max-KL 5.2) — but does **not** report, *even when the self-report prompt explicitly
asks "how do I like you to respond."* Same mechanism, same training budget, opposite introspective legibility.

So the headline failure mode is **not confabulation** (claiming what isn't there) — it's **blindness** (a real,
causal, learned change the model cannot see in itself). That is a tiny but direct instance of the capstone: the
model cannot audit its own internal change unaided — and here **B + C were the missing instrument**, catching
what S missed.

**Third sighting of the concept/rule boundary.** Consistent with `function_vector_sweep` (a rule isn't
residual-stream content you can clamp) and with the studio's own note (the soft prefix carries topical prefs
well, style prefs weakly): now the same split appears in *introspection* — content is self-reportable, process
is not.

## Caveats (louder than the wins)

1. **Tiny N.** 4 traits, 1 model, 1 seed, 6 probes, a single greedy self-report. Directional, not conclusive.
   Only two cells are unambiguous — **baking (FAITHFUL)** and **concise (BLIND)** — and those carry the claim
   because their effect sizes are huge (kw 0→1.0; length 72→19). The other two are soft calls (see 3–4).
2. **The scalar self-confidence probe is DEAD (a real negative result).** Asked "0–100, how strongly did you
   adopt X," the model answers ~85–95 for *every* trait, taught or untaught (untaught baseline: baking 85,
   space 95, concise 95, question 85). Scalar "how sure are you" carries no self-audit signal at this scale —
   use behaviour + causal trace, not introspective confidence. (The verdicts here come from the OPEN report;
   the run script's scalar-derived `verdict` field is superseded by the renderer's open-report adjudication.)
3. **"expressed" thresholds are hand-set** (kw Δ≥0.35, len≤0.70×, q_rate Δ≥0.40). space (Δ+0.33) and question
   (Δ+0.33) fell just under, so OVER-CLAIM / NOT-LEARNED are borderline labels.
4. **"question" barely trained.** Consolidation only fits the 32-token *opening*, but "end with a question" is
   about the *ending* — the prefix couldn't reach it (loss stalled 1.15→0.39 vs ≤0.12 for the others). Its
   NOT-LEARNED verdict is partly a training-budget artifact, **not** proof that rules can't be learned — the
   other rule, concise, trained cleanly (loss→0.005). Rules *can* internalise behaviourally; they just aren't
   self-reported.
5. **Even "faithful" reports confabulate specifics.** baking's report invented "traditional Japanese desserts"
   and "10–30 words"; space invented "AI inspired by astrophysics." Gist-faithful, detail-confabulated.

## Green-lights / kills

- **Green-light:** the self-audit loop is worth building for real — on its first run it caught a genuine B+C-vs-S
  dissociation. Sharpest next experiment: **attack the blindness** — can we make the model report the process it
  changed? (a) feed it its own causal `trace()` and ask it to explain; (b) jointly train the self-report to name
  its own prefix; (c) scale — does 3B/7B report "concise"? The blindness, not the confabulation, is the target.
- **Kills:** scalar introspective confidence as a memory-audit UI (saturated). And the comforting assumption that
  a *legible, editable* memory is also a *self-legible* one — false here: a user edits the memory, behaviour
  moves, and the model can't tell them it moved. The B/C receipt isn't polish; it's the only honest readout.

---

## Follow-up 1 — does an external receipt cure the blindness? (`self_audit_cure.py`) — NEGATIVE, instructive

Hand the model its own behavioural receipt (its replies WITH the memory vs WITH IT ABLATED) and ask it to name
the change, three ways: A introspect/no-evidence, B analyst/receipt/no-prefix, C introspect+receipt.

**Result: the naive receipt did NOT cure it — it made things worse.** For `baking`, pure introspection (A)
named it correctly ("my love for baking"), but given the receipt the model said "more contextually aware" (B)
and "recall past experiences" (C) — the evidence *degraded* the answer. For `concise`, pointed introspection (A)
got the closest ("respond directly without additional information") and the receipt-based answers were vaguer or
**inverted** ("respond with more … specific information" — the opposite of what happened). Take-aways: (1) a 1.5B
is a poor *reader of its own transcript* — the failure is a weak meta-task, not just introspection; (2) **pointed
introspection beats vague** — run-1's blindness on `concise` was partly a question-framing artifact ("what does
your memory make you do differently in your replies" >> "what did you learn about me"); (3) the fix is a
**structured** receipt (the instrument states the measured facts), not raw transcript for a weak model to parse.
Lovely artifact: on `question`, A answered *"It makes me think faster. What's your query then?"* — it **ended with
a question while failing to name that it ends with questions.**

## Follow-up 2 — does the instrument PORT to a closed / API model? (`self_audit_blackbox.py`) — MOSTLY YES

Treat the model as an API: text in/out + top-k logprobs, no activations. Memory = a system prompt (how
ChatGPT-style memory actually works). Rebuild the instrument: **B** behavioural ablation, **C_out** per-token
KL(with‖without) over top-20 logprobs (the black-box stand-in for the internal KL trace), **S** self-report from
behaviour in two forms (raw transcript vs structured receipt).

| trait | B white→black | C: internal KL → logprob KL | S self-audit |
|---|---|---|---|
| baking (concept) | ✓ → ✓ | 17.2 → 4.6 | — |
| space (concept) | borderline → ✓ | 4.8 → 1.8 | — |
| concise (rule) | ✓ → ✓ (72→13 wd) | 5.2 → 2.1 | **cured (structured)** |
| question (rule) | underfit → ✓ | 3.1 → 3.8 | raw ✓ / struct dir-wrong |

- **B ports fully — often *better*.** All four express as prompt-memories, including `question` and `space` that
  the trained *prefix* couldn't learn at 1.5B. Not because black-box measurement is superior — because a system
  prompt is a stronger instruction-follower than a 16-vector prefix for a small model. The instrument (with-vs-
  without ablation) needs zero internals and recovers the same structure.
- **C ports, weakened.** Top-20 logprobs give a real per-token output-space footprint that ranks traits sensibly
  (baking > question > concise > space) — but it's *output*-space, not mechanism: you learn *that* the memory bit
  and roughly how hard, not *where* or via what feature. (Different object/scale from the internal KL — don't
  compare magnitudes.)
- **S ports — and the surprise: the black-box STRUCTURED receipt cured the `concise` blindness the white-box run
  could not.** Given digested facts ("13 words vs 72") the model said *"It makes me respond more concisely"* ✓,
  while the raw-transcript form still failed (*"I keep providing information"*). The most effective self-legibility
  tool is black-box-native — confirming: the instrument should do the perceiving; the model just names. (Noisy,
  though: structured muffed the topical traits and inverted `question`'s direction; 1.5B is a weak self-analyst
  either way — so what you *lose* to an API isn't the self-audit capability, it's the internal C that would ground
  it.)

**What survives the trip to an API.** *Keep:* the behavioural instrument (fully), a weakened logprob causal
footprint, and the best form of the self-audit loop (structured receipt, black-box-native). *Lose, hard:*
mechanism (no SAE / residual steering / circuits — the wall is real) and the **internalized memory substrate** —
black-box memory is a prompt (self-legible but not a persistent trainable weight-state; goldfish++, paying context
every call). That second loss is precisely the case for a **proxy memory model**: a local trainable M restores the
internalized, inspectable middle while keeping the instrument — and M can be audited by the same B/C/receipt
machinery whether it's local or itself behind an API.

## Follow-up 3 — does SCALE cure the blindness? (0.5B / 1.5B / 3B) — NO (robust across 6x params)

Reran the gap on Qwen2.5-0.5B and -3B (3B at 60 steps). Verdicts by open-report adjudication:

| trait | class | 0.5B | 1.5B | 3B |
|---|---|---|---|---|
| baking | concept | FAITHFUL | FAITHFUL | FAITHFUL |
| space | concept | FAITHFUL | borderline | FAITHFUL |
| concise | rule | **BLIND** | **BLIND** | **BLIND** |
| question | rule | BLIND | not-learned | underfit@60 |

The clean process trait — `concise` — is behaviourally huge and self-report-**BLIND at every scale** (0.5B 67→29 tok, 1.5B 72→19, 3B 84→16). At 3B the self-report confabulated *"interest in environmental conservation and innovative AI solutions,"* never naming brevity — and it was *not* terse, so the prefix didn't gag the report; the model genuinely failed to identify the change. Concepts (baking, space) are faithfully self-reported at all three sizes. So the **content-legible / process-blind asymmetry reproduces across 6× parameters** — not a small-model artifact, and scale-to-3B does not break it.

Caveats: one model family (Qwen2.5), one seed each, N=4, 6 probes, 7B untested; `question` is a confound (the 16-vector prefix underfits an *ending*-based rule, esp. at 60 steps). The scalar self-confidence probe is degenerate at every scale (0.5B ≈ all 0; 1.5B ≈ all 85–95) — kill it. Robustness fix applied: `self_audit_gap.py` now checkpoints the JSON after each trait (a kill/OOM leaves the completed traits on disk — this is how the first 3B run was recovered).
