# Clozn

**The local-first white-box runtime** — watch, probe, snapshot, edit, and steer a model's
*evolving internal state*, on the models you run yourself. Ollama's structural opposite.

`clozn` = `cloze` (the diffusion engine inside) + *cozen* (to deceive — the illusion it reveals).

See **[DESIGN.md](DESIGN.md)** for the architecture, feature breakdown, and build order.

## Status — Phase 1 (recurrent inspector): M1–M4 ✅ on a real model

Everything runs today against **RWKV-4-169m** via plain `transformers` — no Triton, no custom
kernels (HuggingFace's RWKV already exposes an explicit, fixed-size recurrent state, exactly the
substrate Clozn inspects). The flagship `fla` RWKV-7 / Gated-DeltaNet adapter stays stubbed for
the WSL/Linux path.

| Milestone | What it shows | Result |
|-----------|---------------|--------|
| **M1** snapshot/restore | the recurrent state is a graspable, bit-exact, restorable object | restore maxdiff `0.00e+00`; France-context vs Japan-context predict differently |
| **M2** Watch | logit-lens "thought" per token + per-layer write-intensity heatmap | `runs/watch.html` |
| **M3** Probe + Verify | sentiment is **~90%** linearly decodable from `att_num`, **and causal** — steering that direction gives a clean monotonic dose-response | `runs/probe_verify.html` |
| **M4** Persist | save memory to disk, rehydrate cold in a fresh session → it **recalls** ("the password is *Maiko*") vs a cold model's "the same as the" | `runs/persisted_memory.html` |

Honesty-first throughout: decodability (*what's readable*) is never reported without the causal
test (*what the model uses*); every steering claim ships with its dose-response curve.

### Phase 2 (diffusion substrate): ✅ proven substrate-agnostic

The sibling `cloze` diffusion engine is wrapped as a `StateSource` (the denoising **board** is the
state). The **same** `clozn.ops` and `clozn.store` — zero changes — work on it: snapshot/restore
rewinds the canvas bit-exactly mid-denoise, `diff` counts which slots a pass filled, and a
half-denoised canvas persisted to disk **resumes in a fresh session to the identical result**
(resumable diffusion). Runs on cloze's pure-numpy `FakeAdapter`, so it's exact, fast, and in CI.
`runs/denoise.html` shows slots filling in parallel (the dLLM signature). One architecture, two
substrates (recurrent matrix + denoising board) — that's the bet.

### Phase 3 (legible personal memory): M1 ✅ associative recall over states

`clozn/memory.py` — a browsable shelf of saved mind-states with a new verb: **association**.
Store the state after reading different things, then ask "what does this remind you of?" — keyed
by the *shape of thought*, not text. On real RWKV-4: "The capital of Germany is" → recalls
*france/japan*; "Seven plus five equals" → *math*; "the clouds were dark…" → *weather* — none by
lexical overlap. A **write-gate** skips near-duplicate states to keep the shelf sparse (legible).
Built on `store.py` + the seam, so it's substrate-agnostic. `runs/memory_shelf.html`.

### Phase 3 (interpretability): probe atlas, per-token features, and feature **discovery**

- **Concept atlas** (`atlas.py`) — probe many features at once; on RWKV-4: sentence-type 100%, person 96%, sentiment 90% (causal ✓), tense 71%, number 62%. `runs/concept_atlas.html`.
- **Per-token features** (`features.py`) — watch named features light up word-by-word as the model reads. `runs/features.html`.
- **Feature discovery** (`discover.py`) — the SOTA gap, now working: a tiny sparse autoencoder trained unsupervised on RWKV state **rediscovers seeded themes** (color/number/emotion/animal/place) from top-activating tokens alone, beating a PCA baseline **65% vs 12%** coherence. The "features the model reveals," not just the ones we name. `runs/discovered_features.html`. A **discovered** feature is then **steered + causally verified** (`p3_discover_steer.py`) — pushing the unsupervised "color" feature monotonically raises P(color words).
- **Transcoder** (`discover.py` + `collect_block_io`) — the current SOTA substrate (displaced SAEs): a sparse stand-in for a component's input→output, hooked onto RWKV's channel-mix (its MLP). Head-to-head vs the SAE across layers, it wins at the early layer (L3, 66% vs 62%; color/number/place/food at 88%) but is layer-dependent — honest, not "strictly better" at this scale. `runs/transcoder_features.html`.

## Layout

```
clozn/
  spine.py              the state-stream protocol (StateStep / Intervention / StateSource / Spine)
  ops.py                white-box ops: snapshot / restore / diff / edit / LinearProbe / verify_causal
  probes.py             concept probe + causal dose-response (the M3 method, reusable)
  store.py              persist state across sessions (npz + json manifest)
  viz.py                browser views: Watch film, Probe panel, Memory card, Inspector dashboard
  inspect.py            the Inspector — one command, one dashboard
  sources/
    hf_rwkv.py          REAL adapter: RWKV-4 via transformers (M1–M4 run on this)
    toy_recurrent.py    a tiny delta-rule memory — proves the architecture in pure numpy
    diffusion.py        the cloze diffusion canvas as a StateSource (Phase 2; links to ../cloze)
    fla_rwkv.py         FLAGSHIP adapter (stub): RWKV-7 / Gated DeltaNet via flash-linear-attention
  memory.py             legible personal memory: a shelf of saved minds + associative recall (Phase 3)
  atlas.py              concept atlas: a "what's readable" map of the state across many features
  features.py           per-token feature attribution (which features light up while reading)
  discover.py           unsupervised feature discovery: PCA baseline + a tiny sparse autoencoder
spikes/                 m1_rwkv, m2_watch, m3_probe, m4_persist, p2_diffusion, p3_memory, p3_atlas,
                        p3_features, p3_discover, p3_discover_steer, p3_transcoder
tests/                  fast model-free oracle + a gated `-m model` layer on real RWKV
```

## Run it

```bash
pip install -r requirements.txt                 # numpy + torch + transformers
python -m clozn.inspect "The capital of France is Paris. Two plus two equals"
# -> runs/inspect.html : Watch + Probe + Persist, one dashboard

# or the individual milestone spikes:
python spikes/m1_rwkv.py      # snapshot / restore / diff, bit-exact
python spikes/m3_probe.py     # probe a sentiment direction, then verify it's causal
python spikes/m4_persist.py   # save a memory, rehydrate it in a cold session
```

## Test

```bash
pytest -m "not model"   # fast: ops / store / spine on the toy source (no checkpoint)
pytest -m model         # gated: M1/M3/M4 asserted on real RWKV-4 (downloads ~350MB once)
```

## Next

Phase 3 — legible personal memory over `store.py` (a browsable shelf of saved minds; write/merge
policies). Phase 4 — autoregressive residual-stream taps (the "hookable Ollama" substrate). Plus:
a served browser view over the spine, and the SAE feature-steering slot-in for `probes.py`.
