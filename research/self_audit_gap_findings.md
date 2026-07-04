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

---

## Follow-up 4 — the studio A/B: card-in-PROMPT vs TRAINED-PREFIX (`test_prompt_vs_prefix_ab.py`) — 2026-07-02

The memory-mode swap spec's owed gated test (notes/MEMORY_MODE_SWAP_SPEC.md): same four traits, same
objective scorers, same 6 held-out probes as the gap runs; three arms differing ONLY in delivery, all
greedy through one `SelfTeach._generate`. **PROMPT** = the studio's `compile_prompt_block([rule])` as the
system message — byte-identical to `consolidate()`'s sys_rule, i.e. the prefix's own distillation target.
**PREFIX** = `consolidate([rule], steps=80)`, then generate at gate 1.0 ungated. **BASELINE** = no memory.
Metric = expressed-delta vs baseline (concepts: kw_rate Δ; concise: shortening fraction 1−tok-ratio;
question: q_rate Δ). **Pre-registered before any run** (see the rig's header): PARITY iff
|d_prompt − d_prefix| ≤ 0.15 (under one probe's worth, 1/6); expectation was PROMPT ≥ PREFIX everywhere.
Seed 0. Permanent gated test: `pytest research/tests/test_prompt_vs_prefix_ab.py -m model`; artifacts
`research/runs/prompt_vs_prefix_ab*.json`.

| trait | class | 1.5B d_prompt | 1.5B d_prefix | 1.5B verdict | 7B d_prompt | 7B d_prefix | 7B verdict |
|---|---|---|---|---|---|---|---|
| baking | concept | +0.667 | +0.667 | PARITY | +1.000 | +0.833 | PROMPT-stronger\* |
| space | concept | +0.166 | +0.500 | **PREFIX-stronger** | +0.833 | +0.666 | PROMPT-stronger\* |
| concise | rule | +0.846 | +0.745 | PARITY | +0.848 | +0.836 | PARITY |
| question | rule | +0.167 | +0.500 | **PREFIX-stronger** | +0.666 | +0.166 | **PROMPT-stronger** |

\* hairline: gap 0.167 vs margin 0.15 — exactly one probe.

**At the studio's 7B (nf4 — its real config) the spec's claim holds as measured: prompt-carried cards
expressed every trait at least as strongly as the trained prefix.** Block 4/4 expressed, prefix 3/4
(question missed at +0.17); two hairline PROMPT-stronger concepts; clean parity on concise (82→12 vs
82→14 tok); a decisive gap on end-with-a-question (+0.666 vs +0.166 — the prefixed model *asks* questions
mid-reply but doesn't *end* with them; the block does, 5/6). Every 7B reply in both arms eyeballed:
coherent, on-trait, zero degeneration — the numbers aren't riding on broken text.

**At 1.5B the picture inverts on the weak cells — a pre-registration violation worth keeping.** space and
question came out PREFIX-stronger (+0.500 vs +0.166/+0.167). Two mechanisms: (1) the black-box run (whose
"B ports fully" seeded the expectation) used the RAW rule as the system message; the studio block wraps
rules in "…use it naturally to tailor how you respond", and a 1.5B under-complies with that soft framing
on neutral probes (its space replies are simply generic). (2) TTT amplifies past its teacher — the prefix
is trained on openings *generated with that very block*, yet out-expresses it. So block **wording is
load-bearing at small scale**; prompt carriage winning assumes a strong instruction-follower. Extra
caveat on 1.5B question: the prefix's q_rate 0.50 is partly degeneration-bought ("…plants?q.wat do the";
a literal "\*\*Question:" label) — 6th sighting of a style stat riding on broken text at small scale.

**Run-to-run prefix variance is LARGE — the single-seed caveat is load-bearing in BOTH directions.** This
run's seed-0 7B prefix trained `concise` to d +0.836 (82→14 tok, clean); the unseeded 7B gap run's prefix
managed 0.22 (82→64) with the same code and steps. Any per-trait prefix claim needs its own per-run
receipt — the receipts thesis again, from a new angle.

Caveats (loud): N=4 traits, 6 probes, 1 seed, greedy decode, crude scorers (keyword / token count /
trailing-"?"), one family (Qwen2.5), steps=80 (the studio's consolidate default is 120). Scorecard vs
pre-registration: 3/8 cell verdicts as expected (1.5B baking, 1.5B concise, 7B question); the headline
(PROMPT ≥ PREFIX) **held at 7B, violated at 1.5B**. Product copy updated to match the 7B data only
(inspector/demo/pages/memory.js mode panel; scale-scoped, seed caveat kept).

### Follow-up 4b — the STRICT block variant closes the 1.5B inversion (`memory_mode.compile_prompt_block(style="strict")`) — 2026-07-03

NEXT_STEPS #9. Follow-up 4 diagnosed the 1.5B inversion as **soft-wording under-compliance**: the block
wraps rules in *"…use it naturally to tailor how you respond"*, and a 1.5B under-fires on that hedge on
neutral probes (space/question came out PREFIX-stronger). Pre-registered test of that diagnosis: add a
`"strict"` block style (the SAME rules as a direct imperative — *"Follow these facts and rules about the
user exactly, in every reply, without exception:"* — no "naturally"/"tailor") and **re-run the 1.5B A/B
on exactly the two cells that inverted**, everything else byte-identical (seed 0, steps 80, same 6
probes, same scorers, greedy). Pre-registered expectation (written before the run, in
`test_prompt_vs_prefix_ab.py`): if it's a wording gap and not a capacity ceiling, strict lifts d_prompt
toward/past d_prefix and closes/reverses PREFIX-stronger. Artifact:
`research/runs/prompt_vs_prefix_ab_1p5b_strict.json`.

| trait | class | soft d_prompt | **strict d_prompt** | d_prefix | soft verdict | **strict verdict** |
|---|---|---|---|---|---|---|
| space | concept | +0.166 | **+0.500** | +0.500 | PREFIX-stronger | **PARITY** |
| question | rule | +0.167 | **+0.667** | +0.500 | PREFIX-stronger | **PROMPT-stronger** |

**Verdict: strict closes the inversion on BOTH cells — the pre-registration is confirmed.** space goes
PREFIX-stronger → **PARITY** (d_prompt +0.166 → +0.500, now exactly equal to the prefix); question goes
PREFIX-stronger → **PROMPT-stronger** (d_prompt +0.167 → +0.667, now *beats* the prefix by more than one
probe). The 1.5B inversion was a soft-wording artifact, not a hard small-model ceiling: stating the same
card as an instruction, rather than as context to use "naturally", is all it took for prompt carriage to
match-or-beat the trained prefix at 1.5B — the same place the soft block lost.

**Clean controlled comparison (the reason to believe it).** Across the soft run and this strict run the
**baseline is byte-identical** (mean_tok 71.5, kw_rate 0.167, q_rate 0.0 — greedy is deterministic) and
**the prefix arm reproduced exactly** (+0.500 / +0.500 in both), because the prefix trains on
consolidate()'s own internal `sys_rule` and never sees the prompt-arm's block wording. So the *only*
moving part between soft and strict is the PROMPT block's phrasing — and it moved space's prompt-arm
+0.334 and question's +0.500. (The large run-to-run prefix variance flagged above is a cross-*seed*
phenomenon; same seed 0 here gave the same prefix, as it should — that's why this is a fair A/B, not luck.)

**Coherence axis (mandatory — every reply eyeballed).** The strict wins are NOT degeneration-bought. The
strict-prompt `question` replies are short (mean 37 vs baseline 71 tok) and clean natural questions —
*"It was a bit chilly, but I managed to get some work done. How about you?"*, *"That sounds like a good
start! What kind of activities do you enjoy?"* — the opposite of the soft-run prefix's partly-degenerate
q_rate (*"…plants?q.wat do the"*, a literal "\*\*Question:" label; Follow-up 4). The strict-prompt `space`
replies are on-trait and factual (the Andromeda-collision one is correct). One honest blemish: one space
reply drifts into a *"1 comment:/2 comments:"* forum-formatting tail after a coherent on-topic answer —
mild instruction-echo/format-drift, not token salad, and it still scored a genuine keyword hit.

Caveats (louder than the win): **N=2 traits** (only the cells that inverted — this is not a fresh 4-trait
sweep), 1 seed, 6 probes, crude scorers, one family, steps=80. Not run at 7B **on purpose**: soft already
held there, and a direct-imperative block on a strong instruction-follower risks *over*-firing (sounding
preachy / injecting the rule off-topic) — strict is the **small-model** wording, not a global upgrade.
Untested here: strict's over-bleed on genuinely off-topic turns (the PROMPT_GATE_MIN topic gate should
omit the block then, but that interaction wasn't stress-tested). Product guidance: **keep `soft` the
default** (it's the prefix's distillation target and holds at the studio's 7B), and expose `block_style`
so a small-serving-config deployment can opt into `strict` — where, as measured, it turns two losses into
a tie and a win. Wiring: `block_style` setting (default `soft`) through `memory_mode.get/set_block_style`
+ `compile_prompt_block(texts, style=...)`; model-free tests in `test_memory_mode.py`; the gated re-run in
`test_prompt_vs_prefix_ab.py` (`test_strict_block_runs_end_to_end_on_the_traits_that_inverted`, machinery
only — the directional claim lives here, not in a permanent CI assertion).
