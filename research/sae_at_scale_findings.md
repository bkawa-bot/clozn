# Does unsupervised SAE feature discovery hold at real transformer scale?

*Roadmap Phase 3 §3.6 — "Scale honesty: does discovery hold where the toy collapsed?"*
*Run date 2026-06-21. Substrate: Qwen2.5-0.5B (q8_0), harvested through the C++ engine.*

## TL;DR — the verdict

**It does not hold the way the toy advertised.** On the RWKV-4-169m toy (`discover.py`), a tiny
SAE beat PCA on feature coherence by a wide margin (the headline 65% vs 12%). At real transformer
scale, on the engine's residual-stream tap, **the SAE and PCA are roughly tied, and PCA is
marginally ahead** on the same un-seeded metric:

| substrate / tap                     | rows  | metric              | SAE  | PCA  | winner |
|--------------------------------------|-------|---------------------|------|------|--------|
| **engine** · Qwen-0.5B · layer 2     | 5120  | top-token coherence | 40%  | 44%  | ~tie (PCA) |
| HF hook · Qwen-0.5B · layer 12 (mid) | 5952  | top-token coherence | 23%  | 26%  | ~tie (PCA) |
| toy (published) · RWKV-169m          | ~700  | theme purity*       | 65%  | 12%  | SAE (big) |

\* the toy number is **purity against 6 hand-seeded themes** on a themed 70-sentence corpus — a
different, more generous metric on a corpus engineered to contain coherent themes. See the caveats;
the absolute numbers are not directly comparable, **but the SAE-over-PCA *gap* is what we test, and
that gap is gone at scale.**

The SAE machinery itself scales fine (it reconstructs, the sparsity knob works cleanly). What does
**not** survive is the *interpretability advantage*: at this model size, with this much data and a
toy-sized SAE, the discovered features are **token-identity detectors**, not semantic concepts —
and PCA finds the same kind of thing slightly more cleanly.

## What we actually did

1. **Harvest (the §3.1 path, through the engine).** Launched `cloze-server` in AR mode on
   Qwen2.5-0.5B (GPU), drove `EngineStateSource(substrate="autoregressive")` over 40 diverse prose
   prompts, and collected **each generated token's residual-stream hidden state** from the engine's
   white-box activation tap. n_embd = 896. **The engine taps layer 2** (`tap_layer_ = 2` in
   `model_ggml.cpp`, chosen for per-token probe separation). Harvest = **5120 rows × 896**, 28 s.
2. **Train.** Reused `discover.TinySAE` (m=512) + a PCA baseline, both on standardized activations,
   exactly as the toy does.
3. **Evaluate.** An un-seeded **top-token coherence** metric, identical for SAE and PCA: for each
   feature, take its top-20 activating tokens; coherence = fraction equal to that feature's modal
   top token. A feature that fires on one consistent token scores 100%. (We can't use the toy's
   theme-purity here — the corpus is un-seeded WikiText-style prose, no themes to score against.)
4. **Attribute (layer vs scale).** Re-ran the *same* metric + SAE config on the *same* model at a
   **middle layer (12)** via a direct HF hook, to check whether the layer-2 result is a
   tap-location artifact. It is not (layer 12 is, if anything, *worse*).

## The numbers in detail

### Engine, layer 2 — SAE L1 dose-response (the honest knob)

| L1  | mean fire-rate | live feats | top-token coherence | recon MSE |
|-----|----------------|------------|---------------------|-----------|
| 0.1 | 27.6%          | 512        | 33%                 | 0.078     |
| 0.3 | 18.4%          | 512        | 38%                 | 0.100     |
| 0.6 | 10.4%          | 512        | **40%**             | 0.136     |
| 1.0 | 5.0%           | 512        | 40%                 | 0.168     |
| 2.0 | 1.3%           | 511        | 38%                 | 0.189     |

The sparsity knob behaves: L1 ↑ ⇒ fire-rate ↓ (27.6% → 1.3%) and reconstruction degrades smoothly.
But **coherence plateaus ~40% and never approaches the toy's 65%**, at any sparsity. PCA over the
top-64 axes scores **44%** on the same metric.

### Example discovered SAE features (engine, layer 2, L1=0.6)

These are the 10 *most coherent* live features — i.e. the SAE's best case:

| feat | fires | reads as          |
|------|-------|-------------------|
| f31  | 7.4%  | the token "what"  |
| f37  | 8.0%  | the token "River" |
| f47  | 9.6%  | the token "they"  |
| f48  | 9.9%  | the token "of"    |
| f51  | 11.7% | the token ":"     |
| f63  | 9.5%  | the token "the"   |
| f72  | 7.3%  | the token "As"    |
| f73  | 9.9%  | the token "the"   |
| f87  | 8.2%  | the token "but"   |

Human-describable? Only in the thinnest sense: each is "**fires on the literal token X**." They are
function words, punctuation, and a couple of content words ("River"). None is an abstract concept
("numbers", "after a quote", a topic) of the kind the toy's themed run produced. Note also two
features both fire on "the" (f63, f73) — **feature splitting / redundancy**, a known small-SAE
pathology.

### PCA axes (engine, layer 2) — for contrast

PCA finds the *same flavour* of thing: PC2 = "to" (80%), PC4 = "The" (100%), PC6 = "what" (100%),
PC7 = "hive"/"mountain". A couple of axes are slightly richer (PC0 mixes "salmon"/"muddy"; PC5
mixes "do"/"1"), but nothing semantic. PCA's mean coherence (44%) edges out the SAE because several
high-frequency tokens get their own clean axis for free.

### Layer 12 (mid-network) — rules out "wrong layer"

Same model, same metric, same SAE config, via an HF hook at layer 12 (5952 WikiText tokens):
PCA **26%**, SAE **20–23%**. The SAE's top features are *still* token-identity — and uglier:
"y" (a subword fragment firing **25%** of the time), "'", ",", ".", "and", and "Valk" (a WikiText
"Valkyria" artifact). So the early-layer tap is **not** the reason discovery underwhelms — the
middle layer is, if anything, worse on this metric with a SAE this small.

## Honest caveats — louder than the wins

- **The metrics differ.** The toy's 65% is **theme purity** on a **seeded themed corpus**; ours is
  **token-identity coherence** on **un-seeded prose**. A themed corpus is half-designed to yield
  coherent features. So "40% vs 65%" is **not** apples-to-apples — the rigorous claim is the narrow
  one: **the SAE-over-PCA gap (the toy's actual selling point) disappears at scale.**
- **Our coherence metric rewards single-token features.** It measures token-identity concentration,
  not semantic abstraction — and it can be "gamed" by a feature that locks onto one frequent token.
  Both SAE and PCA are scored identically, so the *comparison* is fair, but the absolute number
  should be read as "how token-locked," not "how interpretable."
- **The SAE is tiny and under-resourced for the job.** m=512 over 896-dim activations is barely
  overcomplete; real SAEs at this model size use 16–64× expansion and **millions** of tokens. We
  trained on ~5 k. **This is very likely the dominant cause of the collapse** — not a deep fact
  about SAEs, but a fact about *toy SAEs on little data*. A negative result here says "the toy
  pipeline does not transfer for free," **not** "SAEs don't work on transformers."
- **A training-config bug nearly produced a false negative.** The first pass used the existing
  spike's `batch_size=4096, lr=4e-3` over 5 k rows (~12 gradient steps) → the SAE never converged
  (MSE 1.0, **0 live features**). That is a *dead optimizer*, not a real result. The honest number
  required `batch_size=512, lr=3e-2, 80 epochs` (MSE ~0.08–0.17). Flagged so it isn't repeated.
- **Generation-based corpus.** Rows are tokens the model *generated* (not a held-out text corpus);
  with an instruct model this skews toward its own stylistic priors. We mitigated with prose prompts
  (unique-token ratio 0.59, ~4% boilerplate), but a forward-harvest of held-out text would be cleaner.
- **Engine stability.** `cloze-server` crashes intermittently (~every 1600–2000 rows) under
  sustained `state="full"` AR generation (a `ConnectionResetError`; survives fine without the tap).
  The harvester works around it by **auto-restarting the server** between batches. This is a real
  engine bug worth fixing for §3.1 ("harvesting at scale") to be robust — see "next."

## What I'd try next (in priority order)

1. **A real SAE, not a toy.** 16–32× expansion (m≈14 k–28 k), millions of tokens, more steps. This
   is the single most likely thing to move the needle; until it's done, "collapse" is provisional.
2. **A semantic metric, not token-identity.** Score features by whether their top activations share
   a *part-of-speech / topic / syntactic role*, or auto-label via a held-out LLM (Neuronpedia-style),
   so we measure abstraction rather than token-locking.
3. **Forward-harvest path** (pre-authorized but not needed here): emit *all input tokens'* hidden
   states in one pass — cleaner corpus, far fewer requests, and it sidesteps the generation crash.
   The data is already captured in `tap_buf_` during prefill; the engine just doesn't emit it yet.
4. **Fix the server crash** so harvesting at scale doesn't need restart-babysitting.
5. **Transcoders** (§3.5) — the field's current SOTA substrate — instead of an SAE on the residual;
   the `p4_qwen_transcoder.py` spike already has the MLP-IO scaffold.

## Files

- `inspector/spikes/p4_engine_discover.py` — reusable harvest-through-engine + SAE/PCA + coherence,
  with server auto-restart and a `--from-cache` re-analysis path. **The deliverable module.**
- `inspector/runs/qwen_engine_acts.npz` — the harvested corpus (5120 × 896 + token pieces), reusable.
- `inspector/runs/discovered_engine_qwen_L2.{html,json}` — the rendered features + machine-readable
  summary (incl. the full dose-response).
- `inspector/diag_sae.py`, `inspector/diag_layer.py` — the diagnostics that found the training bug
  and the layer-vs-scale attribution.
