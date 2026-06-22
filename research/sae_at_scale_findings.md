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

---

# Proper-scale rerun: a real SAE + a natural-text harvest

*Run date 2026-06-21. Same substrate (Qwen2.5-0.5B q8_0, engine layer-2 tap), but both confounds
the §3.6 run flagged are now removed.*

The §3.6 result above ("the gap is gone at scale") carried two explicit caveats that made the
negative result *provisional*: the SAE was **toy-sized** (m=512, ~1x overcomplete, ~5k tokens), and
the corpus was the model's **own generated tokens** (repetitive, instruct-skewed, and it crashed the
server under sustained streaming). This rerun removes both and re-asks the one question that matters:
**does the SAE>PCA advantage RETURN once the SAE is properly resourced and the data is natural?**

## TL;DR — the verdict: it does NOT return. Mixed-but-negative; the collapse holds.

A 16-32x SAE (14k-28k features), converged (MSE 0.04-0.18) on **120,145 natural WikiText
token-activations**, lands at best **+3.2 points over PCA's top-256 axes — and 10 points BEHIND PCA's
top-64**. The toy's decisive 53-point gap (65% vs 12%) does not reappear in any form: the SAE edge is
a wobbly few points that flips sign with the PCA component count. And qualitatively nothing changed —
the discovered features are **still token-identity detectors**, not concepts.

| substrate / setup                                   | rows    | SAE | PCA | winner |
|------------------------------------------------------|---------|-----|-----|--------|
| toy (published) · RWKV-169m · seeded themes          | ~700    | 65% | 12% | SAE (big) |
| §3.6 engine · Qwen-0.5B L2 · **toy SAE, generated**  | 5,120   | 40% | 44% | ~tie (PCA) |
| **this · Qwen-0.5B L2 · big SAE, NATURAL text**      | 120,145 | **44.7%** | **41.5%** (top-256) / **54.8%** (top-64) | mixed → PCA |

The honest reading of the bottom row: against a deep PCA basis (256 axes) the SAE is marginally
ahead (+3.2); against a tight one (64 axes) it is clearly behind (−10.1). A "win" that depends on
how many PCA axes you pick is not the SAE advantage the toy advertised — that advantage was
**unconditional and large**. So: **the collapse is real, not an artifact of under-resourcing.**

## What we actually did (differently this time)

1. **Forward-harvest of natural text (NEW engine endpoint).** Added `POST /harvest` to the C++
   engine: it tokenizes a text, runs ONE causal forward over all its tokens with the white-box tap
   on, and returns every token's layer-2 residual (`{tokens:[piece], activations:{dtype,shape,
   data}}`, the §1.2 tensor wire). This is the "forward-harvest path" §3.6 listed as next-step #3 —
   one forward per passage, natural held-out text, and it **sidesteps the generation crash entirely**
   (no sustained streaming). Drove it over WikiText-103 passages -> **120,145 rows × 896**, 21 s,
   0 crashes (12 over-length passages skipped cleanly). Unique-token ratio **0.088** (vs 0.307 for
   the old generated set over a tiny prompt list) — far more lexical diversity, real prose.
2. **A properly-resourced SAE.** Scaled `discover.TinySAE`'s exact objective to a GPU SAE
   (`TorchSAE` in the new spike) at **16x (m=14,336) and 32x (m=28,672)** expansion, swept L1 in
   {0.5,1,2,4,8}, **40 epochs / ~9.4k gradient steps each**. Every config CONVERGED (live features
   ~all of m; MSE 0.039-0.181, never the dead MSE=1.0). PCA baseline at K=256 (reporting top-64 too).
3. **Same metric.** Un-seeded top-token coherence, identical for SAE and PCA (apples-to-apples).

## The numbers in detail

### SAE L1 dose-response — the sparsity knob bites cleanly now

| expansion | L1  | live feats | mean fire | coherence (live) | recon MSE |
|-----------|-----|-----------:|----------:|-----------------:|----------:|
| 16x | 0.5 | 14,112 | 13.3% | 27.8% | 0.049 |
| 16x | 1.0 | 14,253 |  9.9% | 30.9% | 0.039 |
| 16x | 2.0 | 14,271 |  7.5% | 35.5% | 0.041 |
| 16x | 4.0 | 14,320 |  5.2% | 40.7% | 0.044 |
| 16x | 8.0 | 14,287 |  3.2% | **44.7%** | 0.059 |
| 32x | 0.5 | 27,376 | 29.0% | 21.6% | 0.181 |
| 32x | 1.0 | 28,419 | 15.2% | 21.2% | 0.103 |
| 32x | 2.0 | 28,645 |  7.7% | 24.2% | 0.059 |
| 32x | 4.0 | 28,620 |  5.0% | 29.0% | 0.056 |
| 32x | 8.0 | 28,620 |  3.5% | 33.6% | 0.058 |

Two clean, honest trends. (a) **Coherence rises with sparsity** (L1 ↑ ⇒ fire ↓ ⇒ each surviving
feature locks onto fewer tokens): the metric rewards token-locking, and a sparser SAE token-locks
harder. (b) **16x beats 32x at matched L1** — the bigger dictionary *splits* features across more
neurons (feature splitting), lowering per-feature concentration. So the SAE's best case (44.7%) is
the *smaller*, *sparsest* config; throwing more features at it makes the metric WORSE, not better.
PCA top-256 = 41.5%, top-64 = 54.8%.

### The 10 most-coherent SAE features (16x, L1=8.0) — still token identity

| feat | fires | reads as | feat | fires | reads as |
|------|-------|----------|------|-------|----------|
| f2  | 5.7% | "illo" (subword) | f58 | 4.9% | "and" |
| f13 | 4.3% | " (" (punct)      | f62 | 4.9% | "regime" |
| f16 | 5.8% | "Fernandez" (WikiText name) | f73 | 3.6% | "Townsend" (again — splitting) |
| f36 | 6.1% | "of"             | f82 | 1.7% | "when" |
| f42 | 3.9% | "Townsend" (WikiText name) | f88 | 2.8% | "L" (subword) |

These are the SAE's **best case** (the 10 most coherent of 14k live features), and they are exactly
what §3.6 found at toy scale: function words ("of", "and", "when"), punctuation ("("), subword
fragments ("illo", "L"), and corpus-specific proper nouns ("Townsend", "Fernandez" — WikiText
article artifacts, and "Townsend" appears TWICE: feature splitting/redundancy, unchanged). **Not one
is an abstract concept** (a topic, a syntactic role, a sentiment). The properly-resourced SAE on
clean natural text discovers the same KIND of thing the toy SAE did — it just has 28x more of them.
PCA finds the same flavour (PC1="@", PC2=",", PC3/4="9", PC6/7="The"), with its top axis (PC0, 37%
of variance) a polysemantic sentence-initial-capitalized mix ("According/However/Although/Germany").

## Honest caveats (again, louder than the result)

- **This is now a STRONG negative, not a provisional one.** §3.6's collapse could be blamed on the
  toy SAE / generated corpus. Both are gone, and the result barely moved (SAE 40% → 44.7%, still
  ~tied-to-behind PCA, still token-identity). The provisional caveat is discharged: **on a 0.5B
  model's early residual, an SAE — even a big, converged one on clean data — does not buy a
  monosemanticity advantage over PCA on this metric.**
- **The metric still rewards token-locking, not abstraction.** Top-token coherence measures how
  concentrated a feature is on one token; it can be "won" by a feature that fires on one frequent
  token. It is fair as a *comparison* (SAE and PCA scored identically) but its absolute value is
  "how token-locked," not "how interpretable." The real frontier metric (auto-interp / does the top
  set share a part-of-speech / topic / role) is still §3.6 next-step #2 and would be the cleaner
  judge — but note BOTH methods here produce features with obvious lexical labels, so a richer
  metric would more likely *confirm* "these are token detectors" than overturn it.
- **Layer 2 is early.** The engine's tap is hardwired to layer 2 (per-token probe separation), which
  skews lexical; §3.6 already showed layer 12 is, if anything, *worse* on this metric with a small
  SAE. The new `/harvest` endpoint accepts a `layer` override (validated at layer 8), so a mid-layer
  natural-text sweep is now a one-flag rerun — left as the obvious next probe, but it is unlikely to
  manufacture the toy's 53-point gap given §3.6's layer-12 evidence.
- **Model scale.** This is a 0.5B model. The published SAE wins are on much larger models with
  millions-to-billions of tokens; "SAEs beat PCA for interpretability" may simply require a scale
  this substrate doesn't reach. What we can say rigorously is bounded to THIS model and metric.
- **A second training-config trap, fixed.** §3.6's bug was too-FEW steps (dead SAE, MSE 1.0). The
  opposite trap bit here at 120k rows: lr=1e-2 **diverged** (MSE ~75) because ~800 first-token
  "attention-sink" outlier rows (standardized norm up to ~270) blew up early gradients. The honest
  config is **lr=1e-3, batch=512, 40 epochs** (MSE 0.04-0.18, all features live). Flagged so neither
  trap is repeated: verify MSE < 1.0 AND a sane live-feature count before trusting any coherence.

## What this implies for the roadmap

The SAE-on-residual path is not where Clozn's interpretability edge will come from at this model
scale — PCA is a near-free baseline that matches or beats it, and the discovered units are token
detectors either way. This **reinforces** §3.6 next-step #5: move to **transcoders** (the field's
current SOTA substrate; the `p4_qwen_transcoder.py` MLP-IO scaffold exists) and to a **semantic /
auto-interp metric** rather than chasing a bigger residual SAE. The `/harvest` endpoint is the
reusable asset that makes all of those one-forward-per-text cheap and crash-free.

## Files (proper-scale rerun)

- `engine/core/serve/cloze_server.cpp` — **NEW `POST /harvest`** endpoint (forward-harvest of all a
  text's token activations at the tap layer; `{text, layer?}` -> `{tokens, layer, n_tokens, n_embd,
  activations:{dtype,shape,data}}`). Built + validated (shape [n,896], layer override, clean 400s).
- `engine/core/src/model_ggml.cpp` + `include/cloze/model_ggml.hpp` — **NEW `GgmlAdapter::harvest()`**
  (one causal forward, returns ALL `tap_buf_` rows, not just the last like `ar_forward`).
- `inspector/spikes/p4_big_sae.py` — the reusable harness: `/harvest` corpus collection + a
  GPU `TorchSAE` (16-32x, streaming top-k so no 120k×28k host matrix) + PCA + coherence, with a
  `--from-cache` re-analysis path. **The deliverable module for the proper-scale run.**
- `inspector/runs/qwen_big_natural_acts.npz` — the harvested matrix (120,145 × 896 + token pieces),
  re-analyzable (gitignored).
- `inspector/runs/discovered_big_sae_qwen.{html,json}` — rendered features + machine-readable
  summary (full dose-response, 10 example features, PCA axes, the SAE−PCA gap).

---

# Transcoders: does the field's SOTA substrate beat the SAE/PCA null at scale?

*Roadmap Phase 3 §3.5 — "the field's current SOTA interp substrate."*
*Run date 2026-06-21. Same substrate (Qwen2.5-0.5B q8_0), harvested through the engine's `/harvest`.*

The two SAE runs above land on a strong negative: a residual SAE (even a big, converged one on 120k
natural-text tokens) buys **no monosemanticity advantage over PCA** on this 0.5B model, and the units
it discovers are **token-identity detectors**, not concepts. Both writeups name the same next move:
the **transcoder** — a sparse stand-in for a component's **INPUT→OUTPUT** map (vs the SAE's in→in
*reconstruction*), the field's current SOTA interp substrate. The toy notes hinted transcoders edged
SAEs at early layers (66% vs 62% on RWKV channel-mix). **Does that edge survive at real scale?**

## TL;DR — the verdict: NO. The transcoder's apparent edge over the SAE is a LAYER-CHOICE ARTIFACT, and a CONTROL kills it. The null holds.

The simplest viable transcoder — a **layer-residual transcoder**: code the input residual `l_out-2`
(the calibrated early tap) through a sparse bottleneck, reconstruct the **output residual `l_out-6`**
six steps downstream (in→out, token-aligned, no engine change) — does, at first glance, **beat the
SAE**: 40.5% top-token coherence vs the SAE's 34.3% (+6.3), and beats PCA's top-256 (29.1%, +11.4).
But it **ties/loses to PCA's top-64 (42.2%)** — the SAME "win depends on which PCA basis" wobble the
SAE run flagged — and, decisively, **a control SAE trained on the INPUT layer (`l_out-2`) alone scores
44.7%, ABOVE the transcoder's 40.5%.** The transcoder reads its code from L2; an SAE reading the same
L2 does *better*. So the +6.3 "transcoder beats SAE" is entirely **"L2 is more token-coherent than
L6"**, not the in→out *mechanism*. Forcing the code to also predict the distant L6 output, if
anything, **slightly hurts** coherence (40.5 vs 44.7). And qualitatively nothing changed: the
transcoder's features are **still pure token-identity** (" one", " metal", " United", " the", " and").

| method (all judged on the same metric)            | rows    | best coherence | sparsity (mean fire) | reads as |
|----------------------------------------------------|---------|---------------:|---------------------:|----------|
| **transcoder** · code L2 → reconstruct L6 · 16x    | 120,145 | **40.5%**      | 1.99% (L1=8)         | token identity |
| SAE · on the OUTPUT L6 · 16x                        | 120,145 | 34.3%          | 4.73% (L1=8)         | token identity |
| **CONTROL: SAE · on the INPUT L2 · 16x**           | 120,145 | **44.7%**      | 3.18% (L1=8)         | token identity |
| PCA · on the OUTPUT L6 · top-256 / top-64          | 120,145 | 29.1% / **42.2%** | —                 | token identity |
| *(prior)* residual SAE-at-scale · L2 · 16x         | 120,145 | 44.7%          | —                    | token identity |
| *(prior)* toy RWKV channel-mix transcoder hint     | ~700    | 66% (vs SAE 62%) | —                  | seeded themes |

The control (row 3) is the whole story: it reproduces the SAE-at-scale headline (44.7%, identical
layer) and **exceeds the transcoder**. The transcoder is not finding better features than a plain SAE;
it is finding *L2's* features (because that is where its encoder reads), and slightly degrading them
by also demanding they reconstruct L6.

## What we actually did

1. **Two-layer token-aligned harvest (NO engine change — reuses `/harvest`'s `layer` param).** Drove
   `POST /harvest` over the SAME WikiText-103 passages **twice per passage**, once at `layer=2`
   (input) and once at `layer=6` (output). The forward is deterministic, so the two activation
   matrices are **token-aligned** (row *r* is the same token in both) — exactly the (x_in, y_out)
   pairs a transcoder trains on. Result: **120,145 rows × 896 at each layer**, 42 s, 0 crashes, 12
   over-length passages skipped (identical row count + skips to the SAE run → same corpus, apples-to-
   apples). The in→out map is **non-trivial**: copy-the-input MSE = 1.071 (standardized), *worse* than
   predicting the target's mean — L2 and L6 genuinely differ, so reconstructing L6 from L2 is real work.
2. **A properly-resourced transcoder.** `TorchTranscoder` = the exact `TorchSAE`/`TinySAE` objective
   (relu code, L1 sparsity, unit-norm decoder, streaming top-k so no 120k×14k host matrix), but with a
   **separate target**: `f = relu(Xin_L2·We + be)`, `recon = f·Wd + bd`, `loss = MSE(recon, Xout_L6) +
   L1·|f|`. 16× expansion (m = 14,336), L1 swept {0.5,1,2,4,8}, batch 512, lr 1e-3, 40 epochs
   (~9,400 grad steps/config). Every config beats copy-input (MSE 0.19–0.21 vs 1.071) with ~all
   features live — a real, converged map.
3. **Matched baselines on the SAME target.** An SAE on `l_out-6` (in→in) and PCA on `l_out-6`
   (top-256, top-64 reported) — so all three are judged on representing the **same L6 output**.
   Same un-seeded top-token-coherence metric, identical for every method.
4. **The control that decides it.** An SAE on `l_out-2` (the layer the transcoder's *encoder* reads) —
   to test whether any "transcoder > SAE" gap is the in→out mechanism or just the input layer.
5. **Anti-degenerate guard (a metric trap caught + fixed).** At low L1 the dictionary is dense (mean
   fire ~15–48%) and the `[0.002,0.4]` live-band lets through only a HANDFUL of features; a "coherence"
   averaged over 1–6 features is noise that would falsely win the verdict (an 8k-row dry run picked a
   **1-live-feature** SAE at "55%"). The reported-best now requires **≥ 200 live features** (the full
   dose-response still prints every config). Flagged alongside the SAE run's two training traps.

## The numbers in detail — the L1 dose-response (the honest knob)

| L1  | TRANSCODER (L2→L6) coh | SAE (on L6) coh | CONTROL SAE (on L2) coh |
|-----|-----------------------:|----------------:|------------------------:|
| 0.5 | 23.1%                  | 25.2%           | —                       |
| 1.0 | 25.7%                  | 26.4%           | —                       |
| 2.0 | 30.3%                  | 28.7%           | —                       |
| 4.0 | 36.0%                  | 31.4%           | **40.7%**               |
| 8.0 | **40.5%**              | **34.3%**       | **44.7%**               |

Three honest reads. (a) **Coherence rises with sparsity** for every method — the metric rewards
token-locking, and a sparser dictionary token-locks harder (same trend as the SAE run). (b) The
transcoder edges the SAE-on-L6 only at L1 ≥ 2, and the gap (~+6) is *smaller than* the gap between
the two SAEs at different layers (L2 44.7% vs L6 34.3% = +10.4). I.e. **layer choice moves the metric
more than the transcoder-vs-SAE substrate choice does.** (c) The transcoder's reconstruction MSE
(0.19–0.21) is ~4× the SAE's (0.04–0.07) — predicting a *different* layer is genuinely harder — but
it still beats copy-input ~5×, so the bottleneck is learning a real input→output map, not cheating.

### The 10 most-coherent TRANSCODER features (16x, L1=8) — still token identity

| feat | fires | reads as | feat | fires | reads as |
|------|-------|----------|------|-------|----------|
| f14 | 2.6% | " one" | f58 | 4.3% | " and" |
| f28 | 2.1% | " metal" | f59 | 1.7% | " New" |
| f31 | 3.4% | " United" | f94 | 2.1% | "graf" (subword) |
| f33 | 1.6% | " on" | f118 | 2.3% | "ati" (subword) |
| f40 | 5.4% | " the" | f125 | 2.2% | " no" |

Every one is "**fires on the literal token X**" at 100% coherence — function words (" the", " and",
" on", " no", " one"), subword fragments ("graf", "ati"), and corpus-frequent content tokens
(" United", " New", " metal"). **Not one is an abstract concept.** This is exactly what BOTH SAE runs
found; the transcoder discovers the same KIND of unit. (The SAE-on-L6 contrast set is the same flavour:
",", ".", " to", " only", " 1", '"', " later", " along". PCA too: PC4 = "@" 100%, PC7 = "'" 100%,
PC0 = a polysemantic sentence-initial-capitalized mix "According/However/Although/Germany" at 22.5%
of variance.)

## Honest caveats (louder than the result)

- **The control is the load-bearing finding, and it is a NULL for the transcoder.** "Transcoder beats
  SAE +6.3" is true only against an SAE on the *output* layer; against an SAE on the *input* layer (the
  one the transcoder actually encodes from) the transcoder is **behind** (40.5 vs 44.7). The in→out
  mechanism contributes nothing positive to feature coherence here — it slightly *costs*. So the
  honest claim is **not** "transcoder ties the null" but "**the transcoder's only apparent advantage is
  a layer-selection artifact, and it underperforms the same-budget SAE on its own input layer.**"
- **The metric still rewards token-locking, not abstraction** (carried over, unchanged). Top-token
  coherence measures how concentrated a feature is on one token; it is fair as a *comparison* (all
  methods scored identically) but its absolute value is "how token-locked," not "how interpretable."
  The transcoder's faster rise with sparsity is partly *because* its code reads the **early, lexical**
  L2 — i.e. the metric and the input-layer choice both push the transcoder toward token-identity. A
  **semantic / auto-interp metric** (does the top set share a part-of-speech / topic / role; or
  auto-label via a held-out LLM) remains the real test — but note all four methods here produce
  features with obvious *lexical* labels, so a richer metric would more likely **confirm** "these are
  token detectors" than overturn the verdict.
- **This is the SIMPLEST transcoder, not the canonical one.** A layer-residual transcoder (L→L'
  residual) is a legitimate transcoder and the cheapest one (no engine change), but the canonical
  interp transcoder replaces a **specific component** — the MLP/FFN sub-block (`ffn_inp`→`ffn_out`) —
  whose nonlinearity is the thing sparse dictionaries are meant to untangle. That needs a small engine
  change (tap a named tensor via `cb_eval`, extend `/harvest`). Given how flatly the simple version
  nulls — and that the *input-layer SAE control already beats it* — building the FFN tap is unlikely
  to manufacture the toy's gap on a 0.5B model on this metric; it is the obvious next probe only if a
  **semantic** metric is built first (so the result wouldn't just be more token-locking).
- **Layer pair (2→6) and model scale (0.5B).** L2 is early/lexical; §3.6 already showed an SAE at L12
  is, if anything, *worse* on this metric. A mid→late transcoder (e.g. 8→16) is a one-flag rerun and
  worth a look, but the published transcoder wins are on much larger models with millions–billions of
  tokens; "transcoders beat SAEs for interpretability" may simply require a scale this substrate
  doesn't reach. What we can say rigorously is bounded to THIS model, layer pair, and metric.

## What this implies for the interp-at-scale direction

The sparse-dictionary family — PCA, residual SAE, **and now the layer-residual transcoder** — is a
**robust null for monosemantic feature discovery on this 0.5B model's early/mid residual** on the
token-coherence metric. Swapping the SAE's in→in objective for the transcoder's in→out objective did
**not** move the verdict; the discovered units are token detectors for all of them, and the
transcoder's headline edge dissolves under a same-layer control. The two genuinely unexplored levers
are therefore **(1) a semantic / auto-interp metric** (the field's actual standard; the current metric
can only ever reward token-locking) and **(2) the canonical FFN/MLP transcoder** (the sub-block whose
nonlinearity sparse codes are designed for) — and (1) should come first, because without it even a
"winning" FFN transcoder would just be reporting token detectors more confidently. Clozn's
interpretability edge at this model scale is **not** going to come from a bigger/cleverer sparse
dictionary on the residual; the `/harvest` (now exercised at two aligned layers) is the reusable asset
that makes both of those follow-ups one-forward-per-text and crash-free.

## Files (transcoders)

- `inspector/spikes/p5_transcoder_scale.py` — the deliverable: two-layer token-aligned `/harvest`
  corpus + a GPU `TorchTranscoder` (in→out, streaming top-k) + matched SAE & PCA on the output +
  the un-seeded coherence metric + an anti-degenerate `MIN_LIVE` guard, with a `--from-cache`
  re-analysis path. Reuses `p4_big_sae` wholesale (server mgmt, harvest, corpus, metric, PCA, TorchSAE).
- `inspector/spikes/p5_selfgate.py` — the self-gate: proves a tiny two-layer harvest is token-aligned
  at two different layers and a tiny transcoder trains (MSE drops, features live) before the big run.
- `inspector/spikes/p4_big_sae.py` — `harvest_text()` extended with an optional `layer` arg (passes
  `/harvest`'s `{layer}` through; backward-compatible). No other change.
- `inspector/runs/qwen_transcoder_2layer_acts.npz` — the two-layer matrix (120,145 × 896 at L2 and L6
  + token pieces), re-analyzable (gitignored).
- `inspector/runs/discovered_transcoder_qwen.{html,json}` — rendered transcoder features + machine-
  readable summary (full dose-response for transcoder & SAE, PCA axes, the transcoder−SAE and
  transcoder−PCA gaps, 10 example features each). The control (SAE-on-L2 = 40.7%/44.7%) is recorded
  here in the writeup; reproduce it with `p4_big_sae`'s `TorchSAE` on the cached `Xin`.

---

# Semantic / auto-interp metric: was the token metric the confound, or is the null robust even semantically?

*Roadmap Phase 3 §3.6 next-step #2 — "a semantic metric, not token-identity." The close of the
interp-at-scale arc.*
*Run date 2026-06-21. Same artifacts (Qwen2.5-0.5B q8_0, engine layer-2 tap, the 120,145-token
WikiText `/harvest` matrix and the 16x L1=8 SAE from the proper-scale run — re-trained deterministically
from cache, reproduces exactly: token-coherence 44.7%, MSE 0.059).*

Every prior run on this page reached the same null — top features are **token-identity detectors**,
no SAE/transcoder advantage over PCA — and every one used the SAME metric: **top-activating-TOKEN
coherence**, which *structurally can only reward token-locking*. A feature that fires on many
DIFFERENT number words ("one", "two", "nine", "forty") scores ~5% on token-coherence yet would be a
perfectly good *semantic* "number" feature. So the metric may have hidden semantic structure the
whole time. Every prior writeup flagged this and named the fix. This run runs it, two ways: **LLM
auto-interp on top-activating contexts** (does a feature read as a concept/role/topic, not a token?)
and **concept alignment** (does any feature track the inspector's semantic labels — number / tense /
person / sentence-type / sentiment — ACROSS different tokens?).

## TL;DR — the verdict: it is a ROBUST NULL, even semantically. The metric was NOT the (only) confound.

Re-scoring the very same features with a semantic eye does **not** rescue them. (1) **Auto-interp:** of
60 candidate features (the top-20 by token-coherence ∪ by density ∪ by activation-spread), **~3–5 are
weakly semantic** (a syntactic/positional role like "concessive connective at clause start" or
"cardinal-number-word starting a sentence"), the rest are token-identity or polysemantic grab-bags —
and crucially the few semantic-ish ones are **syntactic/positional**, not the topical/conceptual
features auto-interp is supposed to surface. (2) **Concept alignment (the quantitative half):** with
honest held-out scoring + a label-permutation null, **no SAE feature beats PCA on any of the five
concepts** (single-unit held-out AUC: SAE 0.67–0.76 vs PCA 0.76–0.82, both barely above their own
null), and a whole-representation probe has **PCA ahead 3–1**. The single "perfect" alignment (sentence
q/stmt, AUC 1.00) is a **"."-token detector** firing on 22/24 sentences with one distinct token — the
metric confound made visible, not a concept.

So the token-coherence metric was a fair scorer after all *on this substrate*: a richer metric finds
the same thing it did — **these units are token/position detectors, not concepts** — exactly as the
prior writeups predicted ("a richer metric would more likely *confirm* 'these are token detectors'
than overturn it"). The interp-at-scale null on this 0.5B early-layer setup is **robust to the metric.**

## What we actually did

1. **Reconstructed top-activating examples WITH CONTEXT.** The saved `/harvest` matrix stores tokens
   in corpus order, so a window of neighbouring `pieces` IS the context (focus token marked `<<…>>`).
   No re-harvest needed. For each feature we emit its top-10 contexts (≈30 tokens each) — the unit the
   token metric throws away. (`runs/p6_autointerp_contexts.json`: 60 SAE features + 16 PCA axes.)
2. **Ranked features THREE ways** so the judge isn't shown only the token-locked winners: by
   token-coherence (the old lens), by **activation density** (features that fire broadly — where a
   concept would hide), and by **activation spread** (peaky-vs-peaky). Union of the top-20 of each = 60.
3. **Auto-interp (LLM-judged, harsh).** For every candidate, judge the contexts: nameable SEMANTIC
   pattern (concept / syntactic role / topic / sentiment) or token-bound? "Fires on the token X" =
   token-bound, full stop, even if X is a content word.
4. **Concept alignment (quantitative, the part we compute).** Harvested the inspector's five
   matched-frame minimal-pair corpora (atlas/probes: number, tense, person, sentence-type, sentiment)
   through the SAME layer-2 tap (final-token = sentence rep), encoded through the SAE and the PCA
   basis, and scored each unit by **held-out** concept AUC with **(a)** a ≥8-firer floor (a 1–4 firer
   can hit AUC≈1 by luck — the degeneracy trap the transcoder run flagged) and **(b)** a
   label-permutation null (what "max over 14k features" scores by chance on 24–48 sentences). Plus a
   **whole-representation** k-fold linear probe (the inspector's `kfold_accuracy`) on the full SAE code
   vs the full PCA projection — the fair, no-cherry-pick comparison.

## (1) Auto-interp judgments — 10 concrete examples (feature → verdict → sample contexts)

The harsh tally over the 60 candidates: **~50 token-bound** (a single literal token, 100% coherence),
**~7 polysemantic grab-bags** (varied tokens but no unifying concept), **~3–5 weakly semantic**
(a real but *syntactic/positional* pattern across different tokens). **Zero** clean topical/conceptual
features ("this fires on text ABOUT war / chemistry / sentiment", token-independently). Examples:

| feature | surfaced by | my verdict | what the contexts show |
|---------|-------------|-----------|------------------------|
| **f3966** | spread (coh 0.65) | **WEAKLY SEMANTIC** ✓ | Cardinal-number words starting a sentence: `<<Two>> weeks prior`, `<<Three>> days later`, `<<One>> of the most popular`, `<<Nine>>, the product of three`, `<<Each>> wall has`. Tracks "number/quantifier at sentence-start" across 6 distinct tokens — a genuine pattern the token metric scored low. But it's **syntactic-positional**, and entangled with passage-onset capitalization. |
| **f4544 / f14112** | spread (coh 0.35/0.25) | **WEAKLY SEMANTIC** ✓ | Concessive/discourse connectives at clause start: `<<Although>> the Egyptians believed`, `<<However>>, a bout`, `<<While>> Fingal was discharging`, `<<Though>> Townsend was proud`. A real syntactic-role feature across Though/Although/However/While — the best "concept-like" case found. |
| **f11982** | spread (coh 0.35) | **WEAKLY SEMANTIC** ✓ | Quantifier-determiner sentence starts: `<<Most>> of the equipment`, `<<Many>> of the assassins`. Real "quantifier" pattern, but only 2 distinct tokens (Most/Many). |
| **f14273** | density (coh 0.20) | **POLYSEMANTIC grab-bag** ✗ | 7 distinct tokens, no unifying idea: `cardiac <<arrest>>`, `the Polish <<government>>`, `her <<departure>>`, `the Polish <<people>>`, `my <<name>>`, `Soviet <<invasion>>`. Mid-frequency nouns thrown together — the density lens's "broad" feature is just diffuse, not a concept. |
| **f7242** | density (coh 0.80) | **TOKEN-BOUND** ✗ | The token "rock"/"Rock" — but polysemous across meanings the feature does NOT separate: `30 <<Rock>>` (TV show), `<<rock>> production style`, `Rashtrakuta <<rock>>-temple`. Mono-token, not mono-semantic. |
| **f11406** | density (coh 1.0) | **TOKEN-BOUND** ✗ (topical tint) | The token "god": `the sun <<god>> 's rule`, `<<god>> of Elephantine Island`, `the universal <<god>>`. 100% one token; it only *looks* topical because the WikiText "Egyptian religion" article is long. |
| **f122 / f9760 / f12918** | spread (coh 0.2–0.6) | **TOKEN-BOUND / boundary artifact** ✗ | Proper-noun first-token-of-passage: `<<Egypt>>ian deities`, `<<Germany>> 's policy`, `<<Atlanta>> was a casemate`, `<<Jordan>> played`. Looks like "topic onset" but is the passage-boundary capitalized-first-token confound — different articles, not a learned topic axis. |
| **f2 / f16 / f42 / f62** | coherence (coh 1.0) | **TOKEN-BOUND** ✗ | The token-metric "winners": `Truj<<illo>>`, ` <<Fernandez>>`, ` <<Townsend>>` (twice — feature splitting), ` <<regime>>`. WikiText proper-noun / article-specific. Pure token identity, as every prior run found. |
| **f36 / f58 / f105 / f166** | coherence (coh 1.0) | **TOKEN-BOUND** ✗ | Function words / punctuation: ` <<of>>`, ` <<and>>`, ` <<not>>`, ` <<.>>`. 100% single-token. |
| **f12394 / f14 / f11005** | spread (coh 0.55–0.85) | **TOKEN-BOUND (positional)** ✗ | Single sentence-initial function words: `<<During>> the…`, `<<In>> 2009…`, `<<As>> with…`. Position-locked AND token-locked (≤1–3 distinct tokens) — not a concept. |

Read across the table, the pattern is unambiguous: the **only** units that escape pure token-identity
are **syntactic-positional** (connective / quantifier / number-word at clause start), and even those
are entangled with the passage-onset capitalization artifact. Nothing reads as a token-independent
**topic or concept**. The density and spread lenses — built specifically to surface features the token
metric buries — surface diffuse polysemantic grab-bags, not hidden concepts.

## (2) Concept-alignment numbers — SAE features vs PCA axes on semantic minimal pairs

Held-out single-unit AUC (best unit picked on train folds, scored on the held-out fold;
`max(auc, 1−auc)`), with the label-permutation null in parentheses; plus the whole-representation
k-fold probe accuracy. **A real win must clear the null AND beat PCA.**

| concept | SAE best unit (held-out AUC, null) | PCA best axis (AUC, null) | whole-repr probe: SAE / PCA / raw | semantic? |
|---------|-----------------------------------:|--------------------------:|----------------------------------:|-----------|
| number (sing/plural)  | f5635 **0.76** (0.68), 8 firers, 5 toks | PC241 **0.80** (0.66) | 0.04 / 0.08 / 0.38 | no — PCA ahead |
| tense (past/present)  | f1007 **0.74** (0.71), 8 firers, 5 toks | PC182 **0.80** (0.67) | 0.12 / 0.08 / 0.21 | no — PCA ahead |
| person (1st/3rd)      | f8775 **0.74** (0.74 = null!), 8 firers | PC72 **0.82** (0.71) | 0.17 / 0.50 / 0.50 | no — at chance |
| sentence (q/stmt)     | f218 **1.00** (0.82), 22 firers, **1 tok** | PC0 **1.00** (0.73) | 0.50 / 0.50 / 0.50 | **no — it's the "." token** |
| sentiment (pos/neg)   | f11158 **0.67** (0.64), 8 firers, 7 toks | PC131 **0.76** (0.65) | 0.38 / 0.48 / 0.48 | no — PCA ahead |

**SAE single-unit wins over null+PCA: 0/5. Whole-representation probe: PCA wins 3, SAE 1, tie 1.**

Three honest reads:
- **Every SAE single-unit "win" is a null once cross-validated.** The first (un-validated) pass
  reported AUC 0.93–1.00 for number/tense/person — all from features firing on **4 of 24** sentences,
  i.e. multiple-comparisons flukes (14k features × 24 samples → something separates by chance). With a
  ≥8-firer floor and held-out folds they collapse to 0.67–0.76, at or just above their permutation
  null, and **below PCA on all five**. This is the degeneracy trap the transcoder run's `MIN_LIVE`
  guard was built for, now caught on the alignment metric too.
- **The one AUC=1.00 is the confound, photographed.** SAE f218 separates question vs statement
  perfectly — by firing on the sentence-final **"."** token (1 distinct token, 22/24 firers).
  Statements end in ".", questions in "?". That is exactly "token-locking masquerading as a concept";
  PCA's PC0 does the same. It *confirms* the metric critique rather than refuting it.
- **The most semantic-looking unit still doesn't clear the bar.** Sentiment f11158 fires on
  negative-valence words across **7 distinct tokens** (ruined / insult / failure / disgusting / loss /
  dreadful) — the closest thing to a token-independent concept in the whole test. But held-out AUC
  0.665 vs null 0.643 is not distinguishable from chance on 48 sentences, and PCA (0.76) beats it.
  Suggestive, not significant — and notably, **PCA already captures sentiment better** without any
  sparse dictionary.
- **Caveat on the probe magnitudes:** the grammar concepts are barely linearly decodable from a
  **layer-2 sentence-final token at all** (raw-activation probe 0.21–0.50), so these are weak-signal
  conditions for *every* method. That is itself part of the finding (layer-2 is lexical, not where
  sentence-level grammar lives) — but it does **not** rescue the SAE, which underperforms raw acts and
  PCA. The point estimate, fairly measured, is: **no SAE concept-tracking advantage.**

## Honest caveats (louder than the result)

- **This makes the negative STRONGER, not weaker.** The prior runs left "use a semantic metric" as the
  one untried lever that might overturn the null. It's now tried — auto-interp AND quantitative concept
  alignment — and the verdict holds on both. The token-coherence metric was **not** hiding semantic
  structure on this substrate; it was reporting the truth (token detectors) in the only vocabulary it
  had. The provisional "maybe the metric was the confound" caveat is **discharged**.
- **Bounded to THIS substrate, and the bound is real.** 0.5B model, **layer 2** (early/lexical — where
  token-identity is *expected* to dominate; sentence-grammar barely decodes here even from raw acts),
  this corpus, sentence-level concepts, sentence-final-token representation. Published auto-interp /
  concept-feature wins (Anthropic, Neuronpedia, GemmaScope) are on **far larger models, mid/late
  layers, millions–billions of tokens**, and often **per-token** (not sentence-pooled) concept labels.
  The clean statement is: **on a 0.5B model's layer-2 residual, sparse-dictionary features are token/
  position detectors under BOTH a token metric and a semantic metric** — not "SAEs have no semantic
  features anywhere."
- **The concept corpora are small (12 pairs/concept) and sentence-pooled.** 24–48 sentences is why the
  null band is wide (~0.64–0.82) and why a held-out + permutation discipline was mandatory. A larger,
  **per-token** concept-labeled probe set (label every token's grammatical number, not the sentence's)
  would be a cleaner alignment test and is the obvious refinement — but it would have to overturn a
  result that is currently *consistent across token-coherence, density-ranked auto-interp, and
  concept-AUC*, which the layer-2 evidence makes unlikely.
- **Auto-interp judge = the model in this run.** A documented limitation of LLM auto-interp generally;
  mitigated by harshness (token-bound unless a token-independent pattern is explicit) and by the
  quantitative half agreeing. Mechanical proxies confirm the direction: mean distinct-top-tokens over
  the 50 token-coherence/positional features ≈ **1**; the 3 "weakly semantic" ones have 2–6.

## What this implies for the interp-at-scale direction

**The sparse-dictionary-on-residual program is exhausted on this 0.5B substrate — and now exhausted on
the metric axis too.** PCA, a residual SAE, a layer-residual transcoder (all on token-coherence) AND a
re-score under a semantic metric (auto-interp + concept-AUC) converge on one answer: **the discovered
units are token/position detectors, PCA matches or beats the SAE, and no concept features emerge that a
semantic metric ranks above PCA.** The "wrong metric" escape hatch is closed. Two directions remain,
and the honest prior on each shifts:

1. **A bigger model / mid-late layer is now the load-bearing variable, not the method.** Three method
   swaps (SAE size, transcoder objective) and one metric swap (semantic) all nulled at 0.5B/L2. The
   remaining hypothesis for "interp-at-scale works" is **scale itself** (model + layer + token count),
   which this substrate cannot reach. If Clozn wants a feature-discovery win, it needs a bigger
   substrate — not a cleverer dictionary or metric on Qwen-0.5B. **Recommendation: stop iterating
   sparse dictionaries on this model.**
2. **Clozn's interpretability edge is its CAUSAL / white-box machinery, not unsupervised discovery.**
   The inspector's concept *probes* (number/person/tense/sentiment) DO decode and steer with causal
   verification (atlas/probes) — a supervised, honesty-gated read+write loop that needs no monosemantic
   dictionary. The arc's lesson: **lead with causal probing + steering on named concepts (the verified,
   working capability), and treat unsupervised SAE feature discovery as a known-null on small local
   models** rather than the headline. The `/harvest` endpoint + this auto-interp harness remain the
   reusable assets for the day a larger substrate is wired in.

## Files (semantic / auto-interp metric)

- `inspector/spikes/p6_autointerp.py` — the deliverable: reloads the cached `/harvest` matrix +
  re-trains the reported-best 16x L1=8 SAE deterministically; reconstructs top-activating **contexts**
  from the corpus-ordered pieces; ranks features 3 ways (coherence / density / spread) and emits them
  with context for auto-interp; runs **concept alignment** (held-out single-unit AUC with a firing
  floor + label-permutation null, plus a whole-representation `kfold_accuracy` probe) against the
  inspector's five semantic minimal-pair corpora. Reuses `p4_big_sae` (server mgmt, `/harvest`, corpus,
  metric, `TorchSAE`, PCA) and `clozn.atlas`/`clozn.probes` wholesale. `--no-engine` skips the
  alignment harvest (auto-interp half only).
- `inspector/runs/p6_autointerp_contexts.json` — 60 SAE features + 16 PCA axes, each with top-10
  contexts (the raw material for the auto-interp judgments above; re-judgeable). Gitignored.
- `inspector/runs/p6_concept_alignment.json` — per-concept held-out AUCs (SAE best unit vs PCA axis,
  with nulls + firer counts + distinct-token counts) and whole-representation probe accuracies.
  Gitignored.
