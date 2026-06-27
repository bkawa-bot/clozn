# clozn studio — a local model you can see and shape

Chat with a local model like you would through Ollama — but **open the panels and see what it has learned
about you, and tune who it is.** Not a hidden system prompt and not a cloud account: a model running on
your own machine whose *memory* and *personality* are yours to read and edit.

The bet isn't that the local model is *smart* (it isn't, next to frontier). It's that it's **yours,
legible, and tunable** — the one thing a sealed frontier API can never be. That gets better, not worse,
as local models improve; this is the instrument waiting for them.

## Run it

```bash
# one Qwen-7B (4-bit) drives the chat, the memory, and the tone dials
cloze .venv python research/clozn_server.py --port 8090
# then open the studio:
#   http://localhost:8090/studio.html
```

Any OpenAI client works too — point it at `http://localhost:8090/v1` (`/v1/chat/completions`,
`/v1/models`, model id `clozn-qwen`). Every reply already has the memory + tone applied.

## The two controls

**Memory — fast-weight trait cards.** What the model knows about you lives in a small *soft prefix* (16
trainable vectors), not a prompt. Two ways it fills:
- it **learns** from your conversation (consolidation distills your expressed preferences into the prefix), and
- you **add** a trait directly — "into baking", "loves sci-fi", "keep it brief".

Each card is editable; remove one and the prefix is rebuilt from the rest. A card is a *behaviour*, so it
surfaces contextually ("free evening?" → *"try baking some cinnamon rolls"*) and mostly stays out of
unrelated turns on its own.

**Personality — real-time tone dials.** Warm, concise, formal, playful — each is a *direction in the
model's activation space* (the mean difference between answering with the trait vs its opposite), added
live to a mid layer as you drag the slider. No training, instant, composable. An "always warm" model is
correct, so these are dials, not gated cards.

## How it actually works (the honest version)

- **Memory = a soft prefix, trained by test-time consolidation.** Stable training matters: low lr, a hard
  prefix-norm cap, and early-stopping that keeps the *best* prefix — without those it diverges into mush.
- **Tone = contrastive activation steering**, calibrated so slider `1.0 ≈ 0.85×` the residual norm: clearly
  on, still coherent. Past ~`2×` it breaks down, so the dials are range-capped.
- **Composition:** tone dials blend with memory best at **mild settings (~±0.5–0.8)**. Cranked to max,
  steering reshapes the whole reply and can crowd out a memory trait — by design (you asked for *very*
  warm), but worth knowing.
- **Soft bleed:** a strong topical card occasionally flavours an off-topic answer (a baking analogy in a
  photosynthesis explanation). Mild, usually charming; a relevance gate is a future refinement.

Everything here is a real view of the model — read and write of its actual internal state — not a prompt
dressed up to look like one. If it isn't an accurate picture of what the model is doing, it isn't in here.
