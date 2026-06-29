# research/

This directory holds two different things now: the **live clozn studio** (a small, maintained set), and
the **research spikes** from the legibility science that produced it (many one-off scripts — what used to
be the `legible-interior` exploratory thread). If you're here to work on the product, you only need the
first list.

## The live studio backend

The running tool is `clozn_server.py` + a handful of modules it imports. Smoke-tested by `test_studio.py`
(imports + structure, no model load): `cloze .venv python research/test_studio.py`.

| file | what it is |
|---|---|
| `clozn_server.py` | the server. One port, one model. A `Substrate` base holds the shared **/memory** (trait cards) and **/steer** (tone dials) surface; `QwenSubstrate` (AR: chat + concepts + OpenAI-compatible `/v1/chat/completions`, streaming) and `DreamSubstrate` (diffusion: `/denoise`) inherit it. Switches substrate by re-exec (one 7B fits the GPU). Persist paths centralized via `_pers()` → `~/.clozn/`. |
| `self_teach_server.py` | `SelfTeach` — the AR **memory**: a soft prefix trained by test-time consolidation against a next-token loss, persisted. (Also a standalone server, but the studio uses the class.) |
| `dream_memory.py` | `DreamMemory` — the **diffusion-native memory**: a soft prefix trained against the *masked-denoising* loss, applied through the real `cloze_lab` scheduler via `PrefixAdapter`. |
| `steering.py` | `SteeringControl` (AR) + `DreamSteering` (diffusion) — the **tone dials**: contrastive activation steering, 7 axes, calibrated, persisted. |
| `brain_readout.py` | `BrainReadout` — the **concept readout** over Qwen + an SAE (`think`, `concepts_only`, `concepts_from_engine`). Used for the brain window + the "see inside" chat strip. |
| `sae7b.py` | loads the andyrdt 7B SAE + Qwen-7B (4-bit); `feats7b`. |
| `denoise_server.py` | `trace_for` — drives a Dream denoise through the `cloze_lab` scheduler and returns the pass-by-pass trace (commits + revisions) for the viz. |
| `atlas_concepts.py` | concept corpora + `content_word` for the brain readout. |

The front-end windows it serves live in `../inspector/demo/` (`studio.html`, `denoise.html`, `brain.html`,
`engine.html`, `instrument.html`) and share `clozn.css`. Persisted state (gitignored, in `~/.clozn/`):
`studio_memory.pt`, `studio_personality.json`, `studio_dream_memory.pt`, `studio_dream_personality.json`.

```bash
cloze .venv python research/clozn_server.py --port 8090     # the studio (qwen by default; --substrate dream)
cloze .venv python research/test_studio.py                  # smoke test (no model load)
```

## Research spikes (the legibility science — not maintained)

Everything else is a one-off from the research that led to the studio. The one idea underneath:
**compression under constraint** — the bet that you can build a persistent, adaptive internal state that
is **legible by construction** ("grow up without becoming opaque"), and that interp methods earn their
place on *results*, loudly cut otherwise (see `HANDOFF.md`). Kept for the record; not imported by the live
tool. Roughly:

- **memory / legibility** — `legibility_v1*.py`, `legibility_discovered*.py`, `legibility_natural_qwen.py`,
  `frontier_apply*.py`, `sidecar*.py`, `memory_timeline.py`, `memory_live_server.py`, `learns_server.py`,
  `self_teach_extras.py`, `state_cycle.py`
- **feature discovery / circuits** — `feature_atlas*.py`, `feature_circuit*.py`, `function_vector*.py`,
  `concept_readout.py`, `wire_atlas.py`
- **superposition / interp toys** — `structured_superposition.py`, `superpos_static.py`,
  `toy_superposition.py`, `interior*.py`, `grok*.py`, `intro*.py`, `crux.py`
- **brain / labels** — `brain_server*.py`, `fetch_np_labels.py`, `fetch_np_stats.py`
- **diffusion spikes** — `denoise_capture.py`, `dream_memory_spike.py` (superseded by `dream_memory.py`),
  `wire_denoise.py`
- **misc** — `validate_traits.py`, `engine_concepts_test.py`, `wire_memory.py`
