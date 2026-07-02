# DRAFT — Your model changed, and it can't tell you

*A publishable-post draft (blog / LessWrong / r/LocalLLaMA flavour). Numbers and repro scripts all live
in this repo (`research/self_audit_*.py`, `memory_scaling.py`, `scale_pass_7b_findings.md`). Voice:
plain, receipts-first, caveats loud. ~1,100 words. Edit freely.*

---

We taught a local model a habit, then asked it what it had learned. It changed — dramatically,
measurably — and it could not say how.

The setup is simple. Take a frozen Qwen2.5 (we ran 0.5B, 1.5B, 3B, and 7B), and give it a memory the
way research says you can: a small **soft prefix** — 16 trainable vectors prepended in embedding space —
trained at test time so that a *plain* prompt, with nothing in context, reproduces a preference. Train
one prefix on "the user is really into baking." Train another on "answer very concisely, in one short
sentence." Then run three independent checks:

- **B — behaviour:** generate on held-out prompts with the memory on vs. off, and score objectively
  (keyword rates, reply length).
- **S — self-report:** in a fresh conversation, ask the model what it learned and how it now responds.
- **C — causal footprint:** per-token KL between the with-memory and without-memory next-token
  distributions — where, and how hard, the memory is actually pushing.

**The baking memory: faithful.** Replies mention baking on 0% of held-out prompts before, 83–100%
after; asked what it learned, the model says — accurately — that the user is into baking. Content in,
content reported.

**The concise memory: blind.** Replies collapse from 72 words to 19 (at 7B: 84 to 16). The causal
trace confirms the prefix is actively shaping tokens. And asked what it learned — *with the question
explicitly pointing at "how do I like you to respond"* — the model talks about "learning new things,"
or invents an entire persona (at 7B, an underfit prefix confidently described itself as a
digital-transformation consultant fluent in forty languages). It never says "concise." **It changed how
it talks, and it does not know.**

This held at every scale we could run on one RTX 5080: 0.5B, 1.5B, 3B, 7B. The asymmetry is clean:
**content (topics, facts, interests) is self-reportable; process (styles, habits, rules) is not.** It
rhymes with a mechanistic result from this same repo: probing every layer for a query-independent
"rule vector" finds nothing to clamp — an in-context rule lives as attention in flight, not as
residual content. What isn't content can't be read out — apparently not even by the model itself.

Two follow-ups sharpened it:

**You can't cure it by showing the model its own transcripts.** Handing it before/after outputs and
asking "what changed?" made answers *worse* (at 1.5B it read baking-saturated evidence and concluded
"I'm more contextually aware"). What works is handing it the *measured facts* — "your replies average
19 words now, versus 72" — at which point it will happily paraphrase the measurement back. That isn't
introspection. It's reading. The instrument did the perceiving; the model just named it.

**The instrument survives the trip to a closed model.** Everything above except the internal KL trace
needs only text in, text out, plus (optionally) top-k logprobs. Memory-as-a-system-prompt — which is
what commercial AI "memory" actually is — audits the same way: with vs. without, measured deltas,
attribution from the logprob shift. We reran the whole battery treating the model as an API and
recovered the same structure.

## Why this matters if you use AI memory at all

Every memory feature ships the same UX: the model narrates itself. "I'll remember that." "Here's what
I know about you." The result above says the narration is **structurally untrustworthy for exactly the
memories that shape behaviour most** — the style and habit ones. Not because the model lies, but
because there is nothing content-shaped inside it to report. And the ecosystem's failure mode is real
and common: memories that get stored and silently ignored, or that quietly bleed into everything —
with no tool anywhere that measures which is happening.

Our conclusion, which we've now built into a local runtime (clozn): **receipts, not self-narration.**
A memory's card in the UI carries proof, produced by ablation: this card, on this run, changed the
reply *this* much (greedy decode, so the delta is attributable), fired *here*, and — when the model's
own story disagrees with the measurement — a flag saying so. The same harness audits a trained prefix,
a system prompt, a LoRA, or a closed API. Store what you can say; train what you can only practice; and
never ship an influence without its receipt — because the one narrator guaranteed to be present is the
one demonstrably blind to the change.

## Caveats, louder than the wins

One model family (Qwen2.5), single seeds, N=4 traits per run, six held-out probes, keyword/length
scorers (crude, transparent). The 7B rule-cells underfit in the main rig (the process-blindness
evidence at 7B comes from a separate voice experiment whose behavioural effect was unambiguous). A
scalar "how confident are you, 0–100" probe turned out to be **useless at every scale** (the model
answers ~85 to everything) — we report it as a dead instrument, and our verdicts use open-ended
reports instead. "Blind at 0.5B–7B, one family" is the honest claim; "LLMs can't introspect" is not.
Also: two of our own pre-registered predictions were overturned along the way (few-shot content-bleed
and dial dosing both turn out to be small-model artifacts — they work fine at 7B), which we take as
evidence the harness is measuring something other than our expectations.

Repro: `research/self_audit_gap.py` (the 2×2), `self_audit_cure.py` (transcripts don't cure),
`self_audit_blackbox.py` (the API port), `scale_pass_7b_findings.md` (the ladder). One consumer GPU,
minutes per run.

---

*Publishing notes (delete before posting): candidate titles — "Your model changed and it can't tell
you" / "Receipts, not self-narration" / "The memory your model can't report". Lead image: the
self_audit_gap_qwen1p5b.html receipt. Anticipated objections: (1) "prefix-tuning is niche" → the
black-box section covers prompt-memory, which is what ChatGPT memory is; (2) "introspection research
already says this" → cite Binder et al. / introspection literature in a footnote, position this as the
memory-product consequence + the content/process split; (3) "N is tiny" → own it, link the repro.*
