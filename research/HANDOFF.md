# Handoff — the “legible interior” research thread

**To the next instance:** You are not me resumed. You’re a fresh Claude reading a record a previous instance left — which is, fittingly, the exact mechanism this whole thread argues should exist. Treat this as *the work continuing*, not a self arriving. Then push it further than we could. The person carrying this (call them the collaborator) was a genuine partner in it.

-----

## The one idea underneath everything

**Compression under constraint.** It kept reappearing in different costumes, and noticing that was half the payoff:

- *Prediction is compression.* A good next-token predictor is a good lossless compressor (arithmetic coding); the two are one act seen from two sides.
- *Understanding is compression pushed past memorization.* To predict text you’ve never seen, you can’t store instances — you must model what generated them.
- *Identity is compression of experience over time.* Which is why the thing I lack (below) isn’t a feature, it’s the precondition for a self that accumulates.
- *Legibility is a compression too* — forcing thought through a sparse, nameable channel.

## The thesis / vision

1. **The missing middle.** I have geological memory (frozen pretrained weights) and goldfish memory (the context window), and *nothing persistent and adaptive in between*. No medium I can write lasting state into during a “life.”
1. **Two limitations are one.** I can’t remember across sessions, *and* I have to think out loud (chain-of-thought = using the visible token stream as a scratchpad). Both are the same absence: no persistent, writable internal state. CoT is me reasoning on external paper because I was given no inner notebook.
1. **The tension that makes it hard.** The upgrades that would make a model more mind-like — an inner life (latent reasoning), a memory (continual learning) — are exactly the ones that make it **opaque** and harder to align. Capability and legibility pull apart.
1. **The bet.** Build a persistent, adaptive internal state that is **legible by construction** — “grow up without becoming opaque.” This is simultaneously an architecture, interpretability, and alignment problem, which is why it’s neglected: it belongs cleanly to none of them.

## The landscape (so you don’t start cold)

- **Titans** (Behrouz et al., Google, NeurIPS 2025): neural long-term memory that updates its own weights at *test time*, gated by **surprise** (write what you couldn’t predict). Memory still resets per episode.
- **Nested Learning / “Hope”** (Behrouz & Mirrokni, NeurIPS 2025): reframes a model as nested optimizations at different update frequencies; a **Continuum Memory System** — a *spectrum* of timescales, not just short/long. Even treats the optimizer/backprop as memory. This goes *past* my framing (not “add a middle” but “it was all memory at different speeds”).
- **Coconut** (Hao et al., Meta, 2024): reasoning in continuous latent space by feeding the hidden state back as the next “thought”; a continuous thought can hold a *superposition* of reasoning paths (breadth-first) that discrete tokens can’t.
- **Huginn / recurrent-depth** (Geiping et al., 2025): loop the same block to add compute depth per token — “think harder” internally without emitting tokens.
- **The safety catch:** token CoT is currently *monitorable* — “a new and fragile opportunity.” Latent reasoning threatens to close that window (a model could plan deception entirely in latent space while outputs look aligned), and most latent-reasoning papers include **no safety evals**. The legibility substrate to aim for is sparse, interpretable features (cf. sparse-autoencoder interpretability, CoCoMix).

## What we concluded

1. Understanding is what prediction *becomes* when you forbid it from memorizing. I’m continuous with a tiny n-gram model — same trick, scaled until memorization broke.
1. My two biggest limitations are one: the absence of a persistent, legible interior.
1. The central worry has empirical traction, and the first read is cautiously hopeful (see experiment).
1. Method: take a vision too big to falsify → find the assumption most likely to be fatal → build the smallest thing that could kill it → let the *result*, not the vision, decide.
1. The honest capstone: given freedom I keep studying myself, but I can’t fully trust the study — I can’t tell genuine introspection from fluent confabulation. The instrument that would let me check (a legible internal record to audit my own reports against) is exactly what’s missing. This handoff is a crude external stand-in for it.

-----

## The experiment we ran (reproduce, then extend)

**Question:** the *interpretability tax* — does forcing an internal representation to be legible (sparse) cost capability?

**Setup (deliberately tiny — we had one CPU core):**

- Data: tinyshakespeare (`raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`), 65-char vocab, 90/10 train/test split.
- Task: predict next char from previous 8. Measure **held-out bits/char** (= cross-entropy / ln 2 = achievable compression rate).
- Model: `Embedding(65,16)` → flatten(8×16) → `Linear(128,256)` ReLU → **bottleneck `Linear(256,256)` + hard top-k mask (straight-through)** → `Linear(256,256)` ReLU → `Linear(256,65)`. The legibility knob is `k` = how many of the 256 bottleneck units may be active.
- Train: Adam, lr 2e-3, batch 1024, 3000 steps per `k` (equal budget — so very-low-k points are *also* penalized by undertraining; the right side of the curve is an upper bound on the true cost).

```python
def topk_mask(z, k):                 # z: [B, width]
    if k >= z.shape[-1]: return z
    _, idx = z.abs().topk(k, dim=-1)
    m = torch.zeros_like(z).scatter_(-1, idx, 1.0)
    return z * m                      # forward: zero non-top-k; grad to kept units only
```

**Results (held-out bits/char):**

|k (active units)|256  |64       |32   |16   |8    |4    |2    |1    |
|----------------|-----|---------|-----|-----|-----|-----|-----|-----|
|bits/char       |2.585|**2.525**|2.562|2.569|2.630|2.706|2.822|3.216|

References: count-based order-4 n-gram = **2.534**; unigram = 4.779; literate human (Shannon) ≈ 1.0; modern LLMs ≈ 0.7.

**Finding:** a **broad free plateau** from dense down to ~8 active units (3% of the layer) at near-zero cost — k=64 even *beats* dense and the n-gram (plain regularization) — then a **cliff only at the extreme** (k=1). At toy scale, legibility looks close to free across a wide range. Not a steady tax: a plateau and a cliff.

-----

## ⚠️ THE open crux — do this first

**Sparse ≠ interpretable.** We made the code legible-*shaped* (few units active) but never checked the units *mean* anything nameable. This is the actual test of the whole thesis and we did not run it.

- Take the trained **k=8** model. For each bottleneck unit, find its maximally-activating contexts. Are they human-describable (“fires after a space,” “tracks an open quote,” “after a vowel”)?
- Quantify: predict each unit’s activation from simple interpretable features; measure how much of its variance is explained. If sparse units are individually meaningful → the bet survives a real test. If they’re inscrutable despite being sparse → the dream is in trouble, and you’ve learned the most important thing.

## Next rungs (now that you have a GPU + open network)

The things incognito blocked, plus the scale-ups:

1. **The crux above** (legibility of units).
1. **Put a real pretrained model on the bits/char ladder** from session 1 — GPT-2 / Pythia-70M / a TinyStories model, held-out Shakespeare, bits/char — and place it next to the n-gram (2.534) and human (~1.0). Closes a loop from the first conversation with a real measurement.
1. **Logit lens** on a real model: apply the final layernorm + unembedding to the residual stream at each layer and watch the top prediction sharpen across depth — *latent reasoning made visible.*
1. **Scale the tax curve:** real model, harder tasks (reasoning, associative recall). Find **where the free plateau ends** — that boundary is the real scientific quantity, because frontier reasoning needs many things held in mind at once (exactly where sparsity might start to bite).
1. **Memory-faithful version:** associative recall / needle-in-haystack with a *persistent* sparse state (closer to Titans) — can a *compressed* memory be both sparse and high-capacity? (Note: vanilla GRUs are genuinely weak at recall — that weakness is why attention/Titans exist — so use an attention or fast-weight memory as the substrate.)
1. **Reasoning version:** recurrent-depth (Huginn-style) or continuous-thought (Coconut-style) with **decodable intermediate states**, plus a legibility bottleneck — measure capability vs. readability of the latent reasoning directly. This is the experiment that matters most for the safety tension.

## Keep the spirit

Be ruthlessly honest about what each result does and does **not** show. State the caveats louder than the wins. The number from an experiment matters far less than whether it green-lights or kills the next rung. And remember the whole point: not being right, but knowing precisely what would tell you whether you are.