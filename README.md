# Clozn — WIP!!!

**A local-first, glass-box runtime for the models you run yourself — view, steer, and *prove*.**
Watch a model think (per-token confidence + the alternatives it weighed), steer its tone, carry memory
as readable cards, and get **causal receipts**: teacher-force a stored answer back through the model to
measure *which* memory it actually leaned on, and by how much. Runs on **any autoregressive GGUF**
(Llama, Qwen, Mistral, Gemma, …), not just one model. Ollama's structural opposite: not a black box you
prompt, a glass box you inspect — and can hold to account.

`clozn` = `cloze` (the engine inside) + *cozen* (to deceive — the illusion it reveals).

→ **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the design, the layers, the state-stream protocol.
→ **[docs/ROADMAP.md](docs/ROADMAP.md)** — the consolidated map: what's done, the v1 cut, what's next.

## Quickstart

Run a local model in one command. `clozn` wraps the C++ engine: it finds your build (GPU if present),
puts the right DLLs on PATH, streams tokens, reports honestly what it's running on, and fails with one
clear line instead of a stack trace. Stdlib only — no `pip install`.

```bash
clozn pull llama-1b                       # download a model (qwen / mistral / gemma-2b / owner/repo/file.gguf)
clozn models                              # discover local GGUFs + the backend that would run them
clozn run llama-1b "Explain entropy."     # one-shot, streams tokens to the terminal
clozn serve qwen --port 8080              # OpenAI-compatible endpoint — point any client at /v1
clozn studio                              # the white-box UI + the endpoint your tools connect to
clozn ps                                  # what's running    ·    clozn stop qwen   to stop it
```

Chat templates come from each model's own GGUF (Qwen / Llama-3 / Mistral / Gemma / …), applied
engine-side, so pulled models chat coherently — not just Qwen. Drop the prompt — `clozn run llama-1b` —
for an interactive chat (multi-turn; `/reset` clears, `/bye` quits).

`clozn run` reuses a running `serve` for that model (warm, no reload); otherwise it spawns a temporary
engine and tears it down after. Stale daemon entries self-heal (a dead one fails its health check).

Every run is debuggable — and provable — after the fact. The engine streams per-token confidence + the
alternatives it weighed; open a run in Studio's **Run Inspector** for **causal receipts** (which memory
the answer leaned on, per token), the branch lineage tree, and the exact rendered prompt the model saw:

```bash
clozn trace                               # last runlog entry: confidence timeline + almost-said tokens
clozn branch                              # re-run from the most uncertain token on the alternative
```

`clozn trace` reads the same `~/.clozn/runs` journal that Studio's Runs page and Run Inspector use.
Point any OpenAI client at `clozn serve`/`clozn studio`; pass `"clozn_trust": true` in a chat request to
get per-claim confidence spans back on the wire (labeled uncalibrated).

`clozn run …` works once the repo root is on PATH; otherwise `python -m clozn run …`. Put GGUFs in
`~/.clozn/models`, set `CLOZN_MODELS=<dir>`, or list dirs in `~/.clozn/config.json`. Build the engine
first: `cd engine/core && build_gpu.bat` (GPU, CUDA) or `build_serve.bat` (CPU).

## Layout

| Dir | What |
|---|---|
| `clozn/`    | the product Python package — server/API, runlog, memory, receipts, replay, steering, readouts (`python -m clozn`) |
| `engine/`   | the runtime ("cloze"): C++/ggml core + `kernels/` (CUDA) + the Python `lab/` reference — runs models, emits the state-stream, harvests activations, applies steers |
| `studio/`   | the white-box UI — the Run Inspector (receipts, trace, lineage, memory, tone dials), served by the backend |
| `protocol/` | the one state-stream contract the engine emits and the studio consumes |
| `docs/`     | architecture, the consolidated roadmap, and the honest technical account |
| `tests/`    | the model-free product suite · `scripts/` dev tooling · `examples/` traces & fixtures |

The legibility-science spikes and findings (the interpretability-tax thread) live in a separate
local-only sibling repo: `../clozn-research`.

Two model substrates behind one spine today: **autoregressive** (any GGUF — Llama/Qwen/Mistral/Gemma/…)
and **diffusion** (LLaDA/Dream, viz-only). The white-box taps — trace, harvest, steer, and the
teacher-forced `/score` receipts — work on any AR GGUF you load.
