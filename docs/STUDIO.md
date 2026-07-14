# clozn studio — a local model you can see and shape

Chat with a local model like you would through Ollama — but **open the panels and see what it has learned
about you, and tune who it is.** Not a hidden system prompt and not a cloud account: a model running on
your own machine whose *memory* and *personality* are yours to read and edit.

The bet isn't that the local model is *smart* (it isn't, next to frontier). It's that it's **yours,
legible, and tunable** — the one thing a sealed frontier API can never be. That gets better, not worse,
as local models improve; this is the instrument waiting for them.

## Run it

```bash
clozn studio          # launches the backend + engine and opens the app
```

Any OpenAI client works too — point it at `http://localhost:8090/v1`
(`/v1/chat/completions`, `/v1/models`). Every reply already has the memory + tone applied, and
every reply is logged as a run you can open in the Run Inspector afterwards.

## The two controls

**Memory — readable cards.** What the model knows about you lives in cards you can read, edit,
and delete — never an opaque blob. By default a card rides as assembled context (prompt mode),
gated by a **topic-relevance gate**: an off-topic task doesn't get your baking card, while an
open personal ask ("what should I do today?") does. Each application is logged per-run, so the
Run Inspector can show which cards were present — and **causal receipts** can test which ones
the answer actually leaned on. Two deeper carriers exist behind the same cards: *anchored*
memory (a card decomposed into named directions in the model's own space, so "what did you
learn?" is a lookup, not a self-report) and a trained soft prefix (the research mode).

**Personality — real-time tone dials.** Warm, concise, formal, playful — each is a *direction in
the model's activation space* (the mean difference between answering with the trait vs its
opposite), added live to a mid layer as you drag the slider. No training, instant, composable.
An "always warm" model is correct, so these are dials, not gated cards.

## How it actually works (the honest version)

- **Tone = contrastive activation steering**, calibrated so slider `1.0 ≈ 0.85×` the residual
  norm: clearly on, still coherent. Past ~`2×` it breaks down, so the dials are range-capped.
- **Composition:** tone dials blend with memory best at mild settings (~±0.5–0.8). Cranked to
  max, steering reshapes the whole reply and can crowd out a memory trait — by design (you
  asked for *very* warm), but worth knowing.
- **The relevance gate is real but not clairvoyant:** it scores topic overlap plus an
  "openness" signal, so lexically-distant paraphrases of a topic can stay muted (a safe miss),
  and style-phrased cards ("prefers concise answers") are better expressed as dials than as
  gated memory.
- **Memory application ≠ memory influence.** A card being present in the context is logged;
  whether it *caused* the answer is a separate, testable claim — that's what the leave-one-out
  receipts are for.

Everything here is a real view of the model — read and write of its actual internal state — not
a prompt dressed up to look like one. If it isn't an accurate picture of what the model is
doing, it isn't in here.
