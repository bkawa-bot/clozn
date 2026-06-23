# Feature discovery for legible learning: can discovery be made to work locally, or is construction the only path?

*A deep-dive research memo. Run date 2026-06-22. Analysis only — CPU + web, no GPU, no training.*
*Motivating goal: we can make a model LEARN a new rule (test-time adaptation works), but the learned
thing is an opaque blob. We want it LEGIBLE — and not forever limited to simple hand-named rules. We
want rich, complex, even emergent features (the "Golden Gate Claude" dream) that are STILL legible.
Our legibility wins to date are all "by construction" (hand-named diff-in-means directions); every
attempt at unsupervised DISCOVERY has failed.*

---

## 0. Executive summary (the diagnosis, the top ideas, the verdict)

**The diagnosis.** Our from-scratch SAE/transcoder discovery genuinely failed locally — but the
failure is **over-determined by resource regime, not a fact about SAEs or about "local."** We proved
this ourselves with two controls at the end of the saga: (1) a pretrained gold-standard SAE
(Bloom's `gpt2-res-jb`) on GPT-2-small (124M — *smaller* than our 0.5B) produces genuinely
interpretable concept features, which rules OUT "too small / local"; and (2) under a *calibrated*
auto-interp ruler (one that correctly ranks Bloom > PCA > random), **our own** from-scratch
dictionaries score below PCA and below random — so the null is real *for our dictionaries*, but it is
a **training-resource** null, not a metric artifact and not a "discovery is impossible locally" fact.
The field's published numbers make the resource gap stark: Golden Gate Claude used a **frontier base
model**, a **1M–34M-feature** dictionary, trained on **billions of activations**; we used **0.5B–7B**
models, **29k–115k** features, **≤1M tokens**. That is ~10²–10³× on model scale, ~10×–10³× on width,
and ~10³–10⁶× on token budget — simultaneously. From-scratch local discovery is in the wrong regime
on every axis at once.

**The top 3 locally-feasible ideas (ranked by promise × feasibility):**

1. **Use PRETRAINED dictionaries as the legible basis (Gemma Scope on Gemma-2-2B).** This is the
   single highest-value move and it is genuinely drop-in: same `sae_lens` `SAE.from_pretrained → encode`
   path we already run for `gpt2-res-jb`, Gemma-2-2B fits our RTX 5080 in **full bf16 with ~10 GB to
   spare**, layer-12 features come in widths up to **1M**, and every feature already has a
   Neuronpedia auto-interp label. This buys rich, externally-validated, legible features (the actual
   "Golden Gate" mechanism — clamp a feature, change behavior) **without training anything**. It
   escapes the hand-named-simple-rule ceiling because the dictionary is exhaustive and emergent.

2. **A semi-supervised / hybrid basis: anchor on our diff-in-means directions, then DISCOVER the
   residual.** Freeze our verified named concept directions as decoder atoms and learn additional
   atoms in their orthogonal complement (OrtSAE-style orthogonality penalty) — or use the published
   **Concept-Bottleneck SAE** recipe (keep good discovered atoms, add a supervised branch only for
   the concepts they miss). This is the most direct realization of "reliability of construction +
   richness of discovery," and the orthogonal-complement version is a small, well-motivated
   composition nobody has published under that name — a genuine opening for us.

3. **Model-diffing / crosscoders to read out the LEARNED RULE in discovered features.** This is the
   STEP-5 connection. The field's actual method for "what did fine-tuning change?" is to train a
   crosscoder across base vs adapted model and surface the features that were ADDED. If we run a TTT
   adaptation (which we know works) and diff it in a pretrained-quality feature basis, the learned
   rule could be read out as *discovered* features — not hand-named ones. This directly attacks the
   illegible-learned-blob problem.

**The verdict.** **Local from-scratch discovery is a dead end** — confirmed by us and matched by the
field (Korznikov et al. 2026: SAEs barely beat random; DeepMind/Nanda de-prioritized SAE research in
2025). But **legible-AND-rich is NOT a dead end** — the path is **pretrained dictionaries**, not
home-trained ones. The best single next experiment: **wire Gemma-2-2B + a pretrained Gemma Scope SAE
into the inspector, reproduce a real feature-steer ("Golden Gate") locally, then run our TTT
adaptation and diff it in that basis to see whether the learned rule lights up discovered features.**
Construction (diff-in-means) remains the best *steering/detection* basis per AxBench; discovery's role
is **richness and coverage**, supplied by pretrained suites, not local training.

---

## 1. What WE actually did, and what we concluded

### 1.1 Setup (the substrate, across the whole arc)

- **Models:** Qwen2.5-0.5B-Instruct (q8_0, n_embd 896) and Qwen2.5-7B-Instruct (q8_0, n_embd 3584),
  harvested through the C++ engine's white-box activation tap / `POST /harvest` endpoint; plus
  GPT-2-small (124M) via `transformer_lens` for the pretrained-SAE control; plus RWKV-4-169m as the
  original "toy" that motivated the whole question.
- **Hardware:** a single consumer RTX 5080 (16 GB).
- **Dictionaries (from-scratch):** `discover.TinySAE`/`TorchSAE`/`StreamingSAE` — first-gen ReLU + L1
  objective, unit-norm decoder, expansion **1× → 32×** (m from 512 up to 57,344), L1 swept, 40–80
  epochs, every reported config converged (live-feature counts sane, MSE documented; three distinct
  training traps found and fixed: dead-optimizer at too-few steps, lr=1e-2 divergence on
  attention-sink outliers, lr=3e-3 divergence on 7B massive-activation outliers).
- **Token budgets:** 5k (first toy-scale engine run) → **120,145** (0.5B natural WikiText) →
  **1,000,061** (7B natural WikiText). The literature trains on **4B–40B+**.
- **Layers:** L2 (engine's hardwired lexical tap), L12 (0.5B mid, via HF hook), L16-of-28 (7B mid),
  block-8 (GPT-2 mid).
- **Metrics, in escalating rigor:**
  1. **Top-token coherence** (un-seeded): fraction of a feature's top-20 activating tokens equal to
     its modal token. Apples-to-apples for SAE vs PCA. *Structurally rewards token-locking.*
  2. **Semantic / concept-alignment**: held-out single-unit AUC on five matched-frame minimal-pair
     corpora (number/tense/person/sentence-type/sentiment), with a ≥8-firer floor + a
     label-permutation null, plus a whole-representation k-fold probe; AND LLM auto-interp on
     top-activating contexts.
  3. **Calibrated detection auto-interp** (the final, trusted ruler): judge writes a one-line
     description from a feature's top-12 contexts, then predicts fires/not on 8 held-out highs + 8
     nulls (labels hidden); score = balanced accuracy. **Calibrated on GPT-2 first** so that it
     correctly orders Bloom's SAE > PCA > random before being trusted on ours.

### 1.2 What we found, run by run (the SAE/transcoder saga)

| run | substrate | dict | tokens | metric | result |
|---|---|---|---|---|---|
| toy (published) | RWKV-169m, seeded themes | tiny SAE | ~700 | theme purity | SAE **65%** vs PCA 12% — big SAE win |
| §3.6 | Qwen-0.5B L2 | toy SAE (m=512), generated text | 5,120 | top-token coh | SAE 40% vs PCA 44% — **gap gone** |
| proper-scale | Qwen-0.5B L2 | big SAE (16–32×), natural | 120,145 | top-token coh | SAE 44.7% vs PCA 41.5%/54.8% — **mixed→PCA** |
| transcoders | Qwen-0.5B L2→L6 | layer-residual transcoder 16× | 120,145 | top-token coh | TC 40.5%, but a same-layer SAE control = 44.7% **kills the edge** (layer artifact) |
| semantic | Qwen-0.5B L2 | re-score the 16× SAE | 120,145 | auto-interp + concept-AUC | **0/5 SAE single-unit wins** over null+PCA; PCA ahead 3–1 on whole-rep probe |
| 7B scale | Qwen-7B L16 | 8–16× SAE, natural | 1,000,061 | top-token coh | SAE **15–19%** vs PCA **64–70%** — **PCA by ~50 pts**, null DRAMATICALLY stronger |
| **GPT-2 control** | GPT-2-small (124M) | **PRETRAINED** `gpt2-res-jb` 32× | 56,507 | top-token coh | Bloom SAE 31.3% ≈ PCA 27.3/39.2% — **same "null" signature on a KNOWN-GOOD SAE** |
| **calibrated salvage** | GPT-2 (calib) + our Qwen SAEs | — | cached | detection auto-interp | Bloom **0.835** > PCA 0.815 > random 0.531 (ruler VALID); **OUR SAEs: 0.55–0.68, BELOW PCA and random** |

### 1.3 What we concluded (precisely)

- **For our from-scratch local dictionaries, discovery is a real null** — not a metric artifact. Under
  a ruler we calibrated to recognize a gold-standard SAE, our SAE features are *less* describable-and-
  re-detectable than PCA axes or even random directions on both Qwen substrates. Our discovered units
  are token / character / digit / position detectors; the two features that *looked* like cross-token
  concepts ("licence", "approval/sanction") **collapsed on held-out** examples.
- **But the broken-metric salvage matters and is honest about its limits.** Top-token coherence is
  retired as an interpretability judge: it gives Bloom's genuinely-interpretable SAE the *same* "token
  detector, ~tied with PCA" verdict it gave ours — because it cannot see **token-anchored context
  features** (Bloom's `at`-feature that means "died *at* <place>") or **cross-token concepts**. So
  "our SAEs are token detectors" was, on the old metric, **un-provable either way**; only the
  calibrated ruler settled it (against us).
- **Size / "local" is ruled OUT as the cause.** A 124M model's pretrained SAE is clean and rich. The
  cause is the **from-scratch training regime** (dictionary width, token budget, objective, and the
  base model's own representational richness), not the fact of running locally.
- **The working path the saga itself names:** **pretrained dictionaries** (Bloom scores 0.84 on our
  trusted ruler) and **causal probe→steer on NAMED concepts** (the verified capability that needs no
  monosemantic dictionary). Both appear in `sae_at_scale_findings.md`'s final two sections.

### 1.4 The legibility context (the other findings)

The discovery null sits inside a larger, **consistent** pattern: **construction works, discovery fails.**

- **Construction is legible by design and it works.** Concept-indexed memory (`conceptmem`): the
  memory state *is* a coefficient vector over named diff-in-means concepts; 7/8 concepts steer
  cleanly, beat an equal-norm random null, survive downstream layers (read-back corr +0.92–0.97). The
  glass-box fast-weight store (`fastweight`): legible by construction (value = the answer's
  unembedding direction), 100% of values logit-lens to their answer, edits are surgical. **These are
  our legibility wins — and they are all hand-named.**
- **TTT closes the APPLY loop, but the learned thing is illegible.** `frontier_apply`: a directly-
  tuned soft prefix makes the frozen model apply an unseen 1-to-1 relation at 95% of the in-context
  ceiling (the read-MLP's 0.000s become near-ones) — but a probe names the learned prefix at
  **chance** (0.000). `frontier_apply_v2`: **test-time adaptation (a few gradient steps) closes the
  consolidate-then-apply loop** (held-out free-apply 0.00 → 0.944 at 20 steps) — genuine test-time
  learning — but the working prefixes are **not legible**, and lever-2's apparent legibility was an
  **input-feature artifact** (an untrained-map null scored the same). **So: we can make the model
  learn a rule; we cannot yet read the rule out.** That is exactly the gap this memo targets.

---

## 2. The field SOTA (2024–2026): where discovery DEMONSTRABLY works, and why

### 2.1 The proof that rich-yet-legible discovery is real — and what it cost

**Anthropic "Towards Monosemanticity" (Oct 2023).** A custom **1-layer** transformer (d_model 128,
MLP 512), SAE on the MLP, **4,096 features (8×)**, trained on **~8 billion activations**. Median
neuron interpretability ~0, median feature ~12. Even this *toy* needed **8B activations on a 512-wide
target** — ~8,000× our token budget for a vastly easier problem.
[Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) ·
[Anthropic research page](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning)

**Anthropic "Scaling Monosemanticity" / Golden Gate Claude (May 2024).** The result we are comparing
against. **Claude 3 Sonnet** (a frontier production model), residual stream **halfway through the
model**, **three** SAEs of **~1M / ~4M / ~34M features** (~12M live in the 34M), L0 < ~300. The
Golden Gate Bridge feature (`34M/31164353`) clamped to **10× its max activation** produces "Golden
Gate Claude." The same feature fires across **English/French/Chinese** and on **images** despite a
text-only SAE — i.e. the richness is a property of the **frontier base model**, surfaced by the SAE.
Even at 34M features the dictionary is **incomplete** (only ~60% of London boroughs get a feature);
Anthropic say exhaustive coverage "may require autoencoders with **billions of features**."
[Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) ·
[Golden Gate Claude (news)](https://www.anthropic.com/news/golden-gate-claude)

**The levers, quantified against us** (this is the heart of the diagnosis):

| lever | Golden Gate Claude | Ours (Qwen-0.5B/7B, 8–32×, ≤1M tok) | gap |
|---|---|---|---|
| base model | Claude 3 Sonnet (frontier) | Qwen 0.5B / 7B | ~10²–10³× params; **qualitatively richer features** |
| SAE width | 1M → 34M (~12M live) | ~29k (0.5B@32×) → ~115k (7B@32×) | **~9×–1,150× narrower** |
| training tokens | ≫8B activations (≥8B even for the 1-layer toy) | ≤1M | **~10³–10⁶× fewer** |
| hyperparams | scaling-law-extrapolated LR + steps | one fixed expansion | off the compute-optimal frontier |
| site | middle residual stream | L2 (lexical) / mid | minor, but L2 hurt us |

Two of the three biggest levers — **base-model scale** and **token budget** — are things **no SAE
hyperparameter sweep can buy**. Our null was the expected outcome.

### 2.2 Pretrained open SAE suites — the levers, pre-paid, downloadable

The key realization: *we don't have to pay those levers ourselves.* Two production suites exist, both
loadable through the exact `sae_lens` path we already use for Bloom's GPT-2 SAE.

**Gemma Scope (DeepMind, Aug 2024).** **400+ JumpReLU SAEs, 30M+ features** on **every layer and
sublayer** (attention/MLP/residual) of **Gemma-2 2B & 9B** (+ select 27B layers). Widths
**16k → 1M** (the full 2¹⁴–2²⁰ ladder; **layer 12 of Gemma-2-2B has the whole ladder up to 1M**).
Trained on **4B–16B tokens each**, costing **>20% of GPT-3's training compute** in aggregate.
**CC-BY-4.0**, weights as `params.npz`, natively supported by `sae_lens`. This is the suite behind the
public DeepMind/Neuronpedia steering demos.
[Gemma Scope (arXiv 2408.05147)](https://arxiv.org/abs/2408.05147) ·
[DeepMind blog](https://deepmind.google/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/) ·
[HF card](https://huggingface.co/google/gemma-scope)

**Llama Scope (Fudan/OpenMOSS, Oct 2024).** **256 TopK SAEs** (32 layers × 4 positions {R/A/M/**TC**}
× 2 widths) for **Llama-3.1-8B-Base**, widths **8× = 32,768** and **32× = 131,072**, on SlimPajama
activations. On HuggingFace (`fnlp/Llama-Scope`), runnable via `sae_lens`.
[Llama Scope (arXiv 2410.20526)](https://arxiv.org/abs/2410.20526) ·
[HF card](https://huggingface.co/fnlp/Llama-Scope)

**Neuronpedia** hosts auto-interp explanations for every feature in these suites (GPT-2, Gemma Scope,
Llama Scope), fetchable programmatically:
`GET /api/feature/{modelId}/{source}/{index}` returns the feature's explanation + activations. So the
**naming** half of legibility is also pre-paid.
[Neuronpedia features API](https://docs.neuronpedia.org/features) ·
[Gemma Scope on Neuronpedia](https://www.neuronpedia.org/gemma-scope)

**Local feasibility (our RTX 5080, RUN-only, no training) — confirmed:**
- **Gemma-2-2B + Gemma Scope:** base in **full bf16 ≈ 6 GB**, SAE 0.14–0.56 GB → **~10 GB free**. No
  quantization needed → the SAE sees native-precision activations (cleanest path). Can hold layer-12's
  whole width ladder in memory.
- **Llama-3.1-8B + Llama Scope:** needs int8 (~7.5 GB) or nf4 (~3.7 GB) on the base; comfortably fits.
  Caveat: Scope SAEs were trained on bf16 activations, so heavy base-model quantization shifts the
  activation distribution (prefer int8 over nf4 when fidelity matters). Gemma-2-2B in bf16 sidesteps
  this entirely.
[HF bitsandbytes docs](https://huggingface.co/docs/transformers/quantization/bitsandbytes) ·
[SAELens usage](https://decoderesearch.github.io/SAELens/dev/usage/)

### 2.3 Transcoders + crosscoders — the current SOTA substrate

**Transcoders** approximate a component's INPUT→OUTPUT map (canonically the MLP: `ffn_in → ffn_out`),
a *functional replacement* rather than a reconstruction. The original Dunefsky et al. (2024) paper
found transcoder features **about as interpretable as SAE features** (41 vs 38 of 50 "interpretable"
in a blind eval) — the win there was **circuit faithfulness** (clean weight-based tracing).
[Transcoders Find Interpretable LLM Feature Circuits (arXiv 2406.11944)](https://arxiv.org/html/2406.11944v1)

**Skip transcoders (EleutherAI, Jan 2025)** added a learned affine skip to absorb the MLP's high-rank
linear part, closing the reconstruction gap — and then **transcoder features score *significantly
higher* on auto-interp** than SAE features (Pythia-160M: fuzzing 86.4% vs 74.6%; detection 80.9% vs
70.2%). **This is the paper that establishes a genuine feature-interpretability edge — and it is the
canonical MLP transcoder *with a skip*, NOT the layer-residual variant we tried.** Our null was a
clean negative for a variant nobody claims wins.
[Transcoders Beat SAEs for Interpretability (arXiv 2501.18823)](https://arxiv.org/html/2501.18823v1)

**Cross-layer transcoders (CLTs)** and **attribution graphs** (Anthropic, Mar 2025) are the post-SAE
frontier: a CLT feature reads from its layer and writes to all downstream MLP outputs, enabling a
"replacement model" whose feature→feature causal graph can be traced on a single prompt. Demonstrated
on Claude 3.5 Haiku (multi-step reasoning, poetry planning, multilingual circuits).
[Circuit Tracing (methods)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) ·
[On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)

**Crosscoders** read/write across multiple layers — or across **two models** ("model diffing").
**This is the most important transcoder-family result for our STEP-5 problem** (see §5).
[Sparse Crosscoders (Anthropic, Oct 2024)](https://transformer-circuits.pub/2024/crosscoders/index.html)

**Pretrained transcoders are downloadable for local use** and the Gemma-2-2B attribution-graph
pipeline runs on ≤16 GB (the `circuit-tracer` Gemma demo runs on free 15 GB Colab):
`google/gemma-scope-2b-pt-transcoders` (PLT), `mntss/clt-gemma-2-2b-426k` (CLT),
`mntss/transcoder-Llama-3.2-1B`, `mntss/clt-llama-3.2-1b-524k`, plus EleutherAI's Llama-3.2-1B skip
transcoders.
[circuit-tracer (safety-research)](https://github.com/safety-research/circuit-tracer) ·
[Open-sourcing circuit tracing (Anthropic)](https://www.anthropic.com/research/open-source-circuit-tracing)

### 2.4 The skeptical literature — simple baselines beat SAEs (this VINDICATES construction)

This cluster is load-bearing because **our legibility wins come from diff-in-means**, and the field's
flagship benchmarks say that is the *right* choice for acting on known concepts.

**AxBench (Stanford, Jan 2025) — "Even Simple Baselines Outperform Sparse Autoencoders."** On
Gemma-2-2B/9B: for **steering**, *"prompting outperforms all existing methods, followed by
finetuning"*; for **concept detection**, *"representation-based methods such as difference-in-means
perform the best."* On both, *"SAEs are not competitive."* Among interpretable *directions*, supervised
ones (DiffMean, the weakly-supervised ReFT-r1) beat raw SAE latents. **DiffMean is the best
representation method for detection and beats SAEs for steering** — i.e. exactly our basis, benchmarked.
[AxBench (arXiv 2501.17148)](https://arxiv.org/abs/2501.17148)

**Probing ≥ SAEs for detection.** Kantamneni & Engels et al. (ICML 2025): *"SAE probes underperform
the baseline of logistic regression in each regime when taking the mean across datasets."* DeepMind
(Mar 2025): dense linear probes "perform nearly perfectly, including out of distribution," while SAE
probes are "distinctly worse" OOD — and probing the SAE *reconstruction* is worse than the raw
residual (the SAE throws away concept information).
[Sparse Probing case study (arXiv 2502.16681)](https://arxiv.org/abs/2502.16681) ·
[DeepMind: Negative Results / deprioritising SAEs](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)

**SAE "dark matter."** Engels et al.: *"about half of the error vector itself and >90% of its norm
can be linearly predicted from the initial activation"* — what SAEs miss is **structured**, and
*"larger SAEs mostly struggle to reconstruct the same contexts as smaller SAEs"* (width doesn't fix it).
[Decomposing the Dark Matter of SAEs (arXiv 2410.14670)](https://arxiv.org/abs/2410.14670)

**Feature splitting + absorption.** Chanin et al. ("A is for Absorption"): a general feature's firing
gets *absorbed* into specialized children, producing "an interpretability illusion… with arbitrary
false negatives," and *"varying SAE sizes or sparsity is insufficient to solve this issue"*
(absorption *rises* with width and sparsity). **This is the mechanism-level reason a hand-named
diff-in-means direction can be *more* legible than a discovered feature** — the SAE may have shattered
your concept across latents.
[A is for Absorption (arXiv 2409.14507)](https://arxiv.org/abs/2409.14507)

**"Are SAE features real?" — and the direct replication of our null.** Korznikov et al. (Feb 2026):
*"SAEs recover only 9% of true features despite 71% explained variance… our [random] baselines match
fully-trained SAEs in interpretability (0.87 vs 0.90), sparse probing (0.69 vs 0.72), and causal
editing (0.73 vs 0.72)."* **Our finding — SAE features lost to PCA and even random under a calibrated
ruler — is this paper's central result.** Companion: auto-interp scores *"do not distinguish trained
and random transformers"* (Heap et al.) — which is exactly why our calibration-against-random
discipline was necessary.
[Sanity Checks for SAEs (arXiv 2602.14111)](https://arxiv.org/abs/2602.14111) ·
[Auto-interp metrics don't distinguish trained/random (arXiv 2501.17727)](https://arxiv.org/abs/2501.17727)

**The mood shift.** DeepMind/Nanda publicly *"deprioritising fundamental SAE research… SAEs are not
likely to be a gamechanger any time soon and plausibly never will be"* (Mar 2025); Nanda on the
record that "SAEs are overrated" and "simple probes outperform SAEs" (Sep 2025). The field's pivot is
to **transcoders + attribution graphs** and **weight-space decomposition**. Recommending supervised
directions as the legible basis is the **mainstream** position, not contrarian.
[Open Problems in Mech Interp (arXiv 2501.16496)](https://arxiv.org/abs/2501.16496)

### 2.5 Newer SAE variants (if we ever DID train one)

- **TopK SAEs** (OpenAI, Jun 2024): direct L0 control, no L1 shrinkage; clean scaling laws; the 16M-
  feature GPT-4 SAE. [arXiv 2406.04093](https://arxiv.org/abs/2406.04093)
- **JumpReLU** (DeepMind): learned per-feature threshold; Pareto-best on fidelity–sparsity; *what
  Gemma Scope uses.* **Gated SAEs** were its predecessor (gate/magnitude split fixes L1 shrinkage).
  [Gemma Scope](https://arxiv.org/abs/2408.05147) · [Gated SAEs (arXiv 2404.16014)](https://arxiv.org/abs/2404.16014)
- **Matryoshka SAEs** (Bussmann/Nanda, ICML 2025): nested dictionaries where each prefix must
  reconstruct — explicitly engineered against splitting/absorption. Gemma-2-2B: **absorption 0.05 vs
  0.49** for BatchTopK; metrics hold across 4k/16k/65k where baselines degrade. **The variant most
  likely to "just work" if we trained one.** [arXiv 2503.17547](https://arxiv.org/html/2503.17547v1)
- **End-to-end SAEs** (Braun/Sharkey): train on the model's KL, not MSE → functionally-important
  features with fewer of them. [arXiv 2405.12241](https://arxiv.org/abs/2405.12241)

---

## 3. The honest diagnosis: why OURS failed, cause by cause

Each plausible cause mapped to our evidence, ranked by how load-bearing it is.

| cause | load-bearing? | evidence (ours + field) |
|---|---|---|
| **From-scratch vs pretrained** | **YES — primary** | The GPT-2 control: a pretrained SAE on a *smaller* model is clean and rich; the calibrated salvage: our from-scratch dicts score below PCA *and* random while Bloom's scores 0.84. The single biggest discriminator. |
| **Token budget (≤1M vs 4B–40B)** | **YES — primary** | Even Anthropic's 1-layer toy used 8B activations (~8,000× ours). Gemma Scope: 4–16B *per SAE*. We were 3–6 orders of magnitude short. This is a lever no objective/metric change can recover. |
| **Base-model representational richness** | **YES — primary** | Golden Gate's multimodal/multilingual features are properties of a frontier model; Qwen-0.5B/7B simply have less abstract structure to extract. Our 7B run got *worse* vs PCA than 0.5B — scale within our reach didn't help. |
| **SAE width (29k–115k vs 1M–34M)** | **partly** | At our widths, by Anthropic's own coverage logic, most concepts have no dedicated feature; but width alone wouldn't fix the token/base-model deficits, and field results say wider *worsens* absorption. Contributory, not sufficient. |
| **The metric (top-token coherence)** | **was a confound, now controlled** | The GPT-2 control proved it gives a known-good SAE the same "null" — so every pre-salvage "token detector" read-out was un-provable. BUT the calibrated ruler (which *can* see Bloom's quality) still scored ours below baselines. So: the metric masked the question; it did not *cause* our SAEs to be bad. |
| **Objective (first-gen ReLU+L1)** | **minor** | The GPT-2 control's Bloom SAE is *also* first-gen ReLU+L1 and is clearly interpretable — so architecture class is not the cause. JumpReLU/TopK/Matryoshka would help at the margin, not rescue the regime. |
| **Layer (early/lexical L2)** | **minor** | L12 (0.5B) and L16 (7B) were also null; L2 skews lexical but moving off it didn't manufacture the win. |
| **Transcoder objective** | **ruled out (as we tested it)** | The layer-residual transcoder's edge was a layer-choice artifact killed by a same-layer control. We never tested the canonical FFN-with-skip transcoder (which the field says *does* win). |
| **"Local" / consumer-GPU ceiling** | **RULED OUT** | A 124M pretrained SAE runs and is interpretable; "local" is about *training budget*, not running. The ceiling is on from-scratch *training*, not on *using* good dictionaries locally. |

**The honest one-line diagnosis:** we were simultaneously **~10²–10³× short on model scale, ~10×–10³×
short on dictionary width, and ~10³–10⁶× short on token budget**, training from scratch. The first and
third are unrecoverable by any local tweak. The metric confound was real but secondary — it hid the
result; the calibrated ruler confirmed the from-scratch null is genuine for our dictionaries. **The
fix is not a better local SAE; it is to stop training our own.**

---

## 4. Ranked, locally-feasible ideas (promise × feasibility), with the legible-AND-rich angle

### Idea 1 — PRETRAINED dictionaries as the legible basis (Gemma Scope on Gemma-2-2B). **DO THIS.**
- **What it buys:** rich, exhaustive (widths to 1M), externally-validated, auto-interp-labeled features
  — the actual Golden Gate mechanism (clamp a feature → behavior changes) on a model we run locally.
  **Escapes the hand-named-simple-rule ceiling** because the dictionary is emergent and complete-ish,
  not a list we wrote.
- **Cost/risk:** ~zero training. Drop-in via our existing `sae_lens` path; Gemma-2-2B fits in bf16
  with ~10 GB free; Neuronpedia supplies names. Risk: it's *their* model, not Qwen — but that's a
  feature for the legibility question (we want a substrate where good features exist).
- **Legible AND rich?** **Yes — this is the golden-gate dream, locally, today.** Bloom's far-smaller
  pretrained SAE already scored 0.84 on our trusted ruler; Gemma Scope is a much richer suite.
- **Verdict: highest promise, highest feasibility. The lead.**

### Idea 2 — Semi-supervised hybrid: anchor on diff-in-means, DISCOVER the residual.
- **What it buys:** keeps the reliability/legibility of construction (our verified named directions)
  while letting **discovery** add the atoms we *didn't* name — directly answering "don't limit me to
  simple hand-named rules." Two validated templates + one un-built composition:
  - **Concept-Bottleneck SAE** (Kulkarni/Weng et al., Dec 2025): keep the good discovered atoms, add a
    supervised branch only for named concepts they miss. Demonstrated on vision/VLM, not yet LLM
    residual streams — transfers in principle. [arXiv 2512.10805](https://arxiv.org/html/2512.10805v1)
  - **SHIFT** (Marks et al., ICLR 2025): discover an SAE-feature circuit, then *human-trim* to the
    legible/intended subset — "discover rich, keep legible." [arXiv 2403.19647](https://arxiv.org/abs/2403.19647)
  - **Orthogonal-complement discovery (un-built, our opening):** freeze decoder columns to our DiffMean
    atoms, learn the remaining latents under an **OrtSAE** orthogonality penalty against the frozen set
    — fit a dictionary in the orthogonal complement of known directions. Nobody has published this
    under that name; OrtSAE supplies the machinery (−65% absorption), DiffMean supplies the anchors.
    [OrtSAE (arXiv 2509.22033)](https://arxiv.org/html/2509.22033v1)
- **Cost/risk:** this DOES involve local training, so it inherits the regime risk from §3 (small model,
  ≤1M tokens). Mitigant: anchoring on *known-good* directions and only discovering a *small* residual
  basis is a far easier learning problem than a full from-scratch dictionary — and it can be **layered
  on top of a pretrained Gemma Scope basis** (discover atoms orthogonal to Gemma Scope's, seeded by our
  concepts) to dodge the regime problem.
- **Legible AND rich?** **Plausibly the best of both** — legible by anchor, rich by residual discovery.
  Unproven on LLMs.
- **Verdict: high promise, moderate feasibility. The most novel and the most "us." Second.**

### Idea 3 — Model-diffing / crosscoders: read the LEARNED RULE in discovered features.
- **What it buys:** the field's actual method for "what did adaptation change?" Train a crosscoder
  across base vs TTT-adapted model; the features **unique to the adapted model** ARE the learned rule,
  surfaced as **discovered** (not hand-named) features. Directly attacks our illegible-learned-blob
  problem. A whole 2025–2026 sub-literature exists (chat-tuning diffing, LoRA-adapter SAE analysis,
  Delta-Crosscoder for narrow fine-tuning).
  [Crosscoders (Anthropic)](https://transformer-circuits.pub/2024/crosscoders/index.html) ·
  [Overcoming Sparsity Artifacts in Crosscoders / chat-tuning (arXiv 2504.02922)](https://arxiv.org/pdf/2504.02922) ·
  [Delta-Crosscoder (arXiv 2603.04426)](https://arxiv.org/html/2603.04426v1)
- **Cost/risk:** training a crosscoder *is* local dictionary training (regime risk again), and narrow
  TTT adaptations produce *subtle* diffs that even the literature finds hard (hence Delta-Crosscoder).
  Cheaper variant: skip the crosscoder and **project the TTT activation-delta onto a pretrained Gemma
  Scope basis** — read the learned rule as a sparse combination of *already-legible* features. This is
  cheap, training-free, and the cleanest first cut at STEP 5.
- **Legible AND rich?** Yes if it works — the learned rule expressed in rich discovered features is
  precisely the goal. Higher uncertainty than ideas 1–2.
- **Verdict: high promise, moderate feasibility; highest strategic value for the legibility frontier.**

### Idea 4 — Canonical FFN/skip transcoder or pretrained CLT (Gemma-2-2B). Supporting.
- **What it buys:** the field's genuine feature-interpretability edge (skip transcoders > SAEs) and
  attribution graphs (causal feature wiring). Pretrained ones are downloadable; the Gemma-2-2B
  `circuit-tracer` demo runs on ≤16 GB.
- **Cost/risk:** low if pretrained (don't train our own — our layer-residual null is a warning). More
  about *circuits* than the steerable named features we want for legibility.
- **Verdict: do this with PRETRAINED transcoders only, as a richer substrate for idea 3's circuit
  reading. Not a from-scratch effort.**

### Idea 5 — Train a better local dictionary (Matryoshka/TopK/JumpReLU, semantic metric). **DON'T (as a primary).**
- **What it buys:** marginally better-behaved from-scratch features (Matryoshka kills absorption).
- **Cost/risk:** still in the broken regime (§3) — small model, ≤1M tokens. Our 7B/1M run already
  pulled three scale levers and got *more* dominated by PCA. The field replicates the null.
- **Verdict: low promise locally. Only worth it as the *training* half of idea 2 (anchored, small
  residual), never as standalone from-scratch discovery. The calibrated ruler (`p9`) stays as the
  metric of record if we ever do train.**

---

## 5. Connecting to the legibility frontier: a richer basis for legible learning

**The frontier problem (from `frontier_apply_v2`):** TTT *works* — a few gradient steps make the
frozen model apply a new rule at near-in-context accuracy — but the learned prefix is **illegible**
(probe at chance), and the one apparent legibility win was an input-feature artifact. We can make the
model learn; we can't read the rule out. Our only legible read-outs are **hand-named** diff-in-means
coefficients (`conceptmem`), which by construction can't express a *rule the model discovered for
itself* that we didn't pre-name.

**The unlock:** read the learned rule out in a **rich, discovered, pretrained-quality basis** instead
of a hand-named one. Three of the ideas above compose into exactly this:

- A **pretrained Gemma Scope basis** (idea 1) gives thousands of legible, named features — far beyond
  our 8 hand-named concepts.
- **Model-diffing / activation-delta projection** (idea 3) reads what a TTT adaptation *changed* in
  that basis — the learned rule as a sparse set of *discovered* features.
- Optionally, **anchored residual discovery** (idea 2) adds atoms for parts of the rule no existing
  feature covers, seeded by any concept we *can* name.

This turns "the model learned an opaque blob" into "the model learned [feature A ↑, feature B ↓, new
atom C]" — **legible because the basis is legible, rich because the basis is discovered and
emergent.** It is the golden-gate dream pointed at *learning* rather than at static inspection.

### The single most promising next experiment

**"Legible learning in a discovered basis."** Concretely:

1. **Wire Gemma-2-2B + a pretrained Gemma Scope residual SAE (layer 12) into the inspector** via the
   existing `sae_lens` path (bf16, ~6 GB, no training). Pull Neuronpedia labels.
2. **Reproduce a real feature-steer locally** (clamp a known Gemma Scope feature, e.g. a topic/style
   feature, and confirm behavior changes) — our local "Golden Gate," and a sanity gate that the basis
   is genuinely steerable on our hardware.
3. **Run a TTT adaptation we already know works** (the `frontier_apply_v2` prefix / few-gradient-step
   rule) on Gemma-2-2B, harvest the activation **delta** (adapted − base) at layer 12, and **project
   it onto the Gemma Scope feature basis.** Test whether the learned rule reads out as a small,
   *nameable* set of features (and whether clamping those features reproduces the rule's behavior — a
   causal check, the discipline the whole arc insists on).
4. **Compare** that discovered-feature read-out against the hand-named diff-in-means read-out on the
   same rule: is the discovered basis *more* expressive (captures rule structure the named basis
   misses) while staying legible? That comparison is the deliverable.

If step 3 lights up coherent, nameable features and clamping them reproduces the behavior, **we have
legible-AND-rich read-out of a learned rule** — the thing every prior run wanted and couldn't get from
hand-named directions alone. If it doesn't, we've still escaped the from-scratch regime and learned
whether the bottleneck is the basis or the adaptation. Either way it's decisive, it's training-free,
and it fits the RTX 5080. **Pretrained dictionaries are the bridge from "construction only" to
"discovery that actually works locally."**

---

## Sources

- [Anthropic — Towards Monosemanticity (2023)](https://transformer-circuits.pub/2023/monosemantic-features/index.html) · [research page](https://www.anthropic.com/research/towards-monosemanticity-decomposing-language-models-with-dictionary-learning)
- [Anthropic — Scaling Monosemanticity (2024)](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html) · [Golden Gate Claude](https://www.anthropic.com/news/golden-gate-claude)
- [Anthropic — Sparse Crosscoders for Cross-Layer Features and Model Diffing (2024)](https://transformer-circuits.pub/2024/crosscoders/index.html)
- [Anthropic — Circuit Tracing: Computational Graphs (2025)](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) · [On the Biology of a Large Language Model](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)
- [Anthropic — Open-sourcing circuit tracing tools (2025)](https://www.anthropic.com/research/open-source-circuit-tracing) · [circuit-tracer (GitHub)](https://github.com/safety-research/circuit-tracer)
- [Gemma Scope (arXiv 2408.05147)](https://arxiv.org/abs/2408.05147) · [DeepMind blog](https://deepmind.google/blog/gemma-scope-helping-the-safety-community-shed-light-on-the-inner-workings-of-language-models/) · [HF card](https://huggingface.co/google/gemma-scope) · [transcoders repo](https://huggingface.co/google/gemma-scope-2b-pt-transcoders)
- [Llama Scope (arXiv 2410.20526)](https://arxiv.org/abs/2410.20526) · [HF card](https://huggingface.co/fnlp/Llama-Scope)
- [Neuronpedia](https://www.neuronpedia.org/gemma-scope) · [features API](https://docs.neuronpedia.org/features) · [SAELens usage](https://decoderesearch.github.io/SAELens/dev/usage/)
- [Transcoders Find Interpretable LLM Feature Circuits (arXiv 2406.11944)](https://arxiv.org/html/2406.11944v1)
- [Transcoders Beat SAEs for Interpretability (arXiv 2501.18823)](https://arxiv.org/html/2501.18823v1)
- [OpenAI — Scaling and evaluating sparse autoencoders / TopK (arXiv 2406.04093)](https://arxiv.org/abs/2406.04093)
- [Gated SAEs (arXiv 2404.16014)](https://arxiv.org/abs/2404.16014)
- [Matryoshka SAEs (arXiv 2503.17547)](https://arxiv.org/html/2503.17547v1)
- [End-to-end SAEs (arXiv 2405.12241)](https://arxiv.org/abs/2405.12241)
- [AxBench — Even Simple Baselines Outperform SAEs (arXiv 2501.17148)](https://arxiv.org/abs/2501.17148)
- [Are Sparse Autoencoders Useful? Sparse Probing (arXiv 2502.16681)](https://arxiv.org/abs/2502.16681)
- [DeepMind — Negative Results for SAEs / deprioritising SAE research (2025)](https://deepmindsafetyresearch.medium.com/negative-results-for-sparse-autoencoders-on-downstream-tasks-and-deprioritising-sae-research-6cadcfc125b9)
- [Decomposing the Dark Matter of SAEs (arXiv 2410.14670)](https://arxiv.org/abs/2410.14670)
- [A is for Absorption — feature splitting/absorption (arXiv 2409.14507)](https://arxiv.org/abs/2409.14507)
- [Sanity Checks for SAEs: Do SAEs Beat Random Baselines? (arXiv 2602.14111)](https://arxiv.org/abs/2602.14111)
- [Auto-interp metrics don't distinguish trained/random transformers (arXiv 2501.17727)](https://arxiv.org/abs/2501.17727)
- [SAEs Do Not Find Canonical Units of Analysis (arXiv 2502.04878)](https://arxiv.org/abs/2502.04878)
- [Open Problems in Mechanistic Interpretability (arXiv 2501.16496)](https://arxiv.org/abs/2501.16496)
- [Representation Engineering / RepE (arXiv 2310.01405)](https://arxiv.org/abs/2310.01405)
- [Finding Neurons in a Haystack / sparse probing (arXiv 2305.01610)](https://arxiv.org/abs/2305.01610)
- [Concept Bottleneck SAE (arXiv 2512.10805)](https://arxiv.org/html/2512.10805v1)
- [Sparse Feature Circuits / SHIFT (arXiv 2403.19647)](https://arxiv.org/abs/2403.19647)
- [OrtSAE — Orthogonal SAEs (arXiv 2509.22033)](https://arxiv.org/html/2509.22033v1)
- [Overcoming Sparsity Artifacts in Crosscoders — chat-tuning diffing (arXiv 2504.02922)](https://arxiv.org/pdf/2504.02922) · [Delta-Crosscoder (arXiv 2603.04426)](https://arxiv.org/html/2603.04426v1)
