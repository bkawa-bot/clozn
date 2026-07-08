# Clozn - WIP!!!

**A local-first runtime where you can view and steer a model's *memory*.** Watch a model
think; read named concepts and token candidates off its evolving internal state; snapshot,
edit, and persist that state; steer concepts into the residual stream — on the models you
run yourself. Ollama's structural opposite: not a black box you prompt, a glass box you
inspect.

`clozn` = `cloze` (the engine inside) + *cozen* (to deceive — the illusion it reveals).

→ **[ARCHITECTURE.md](ARCHITECTURE.md)** — the design, the layers, the state-stream protocol.
→ **[ROADMAP.md](ROADMAP.md)** — from here to the memory frontier, broken into tasks.

## Quickstart

Run a local model in one command. `clozn` wraps the C++ engine: it finds your build (GPU if present),
puts the right DLLs on PATH, streams tokens, reports honestly what it's running on, and fails with one
clear line instead of a stack trace. Stdlib only — no `pip install`.

```bash
clozn pull llama-1b                       # download a model (qwen / mistral / gemma-2b / owner/repo/file.gguf)
clozn models                              # discover local GGUFs + the backend that would run them
clozn run llama-1b "Explain entropy."     # one-shot, streams tokens to the terminal
clozn serve qwen --port 8080              # OpenAI-compatible endpoint — point any client at /v1
clozn ps                                  # what's running    ·    clozn stop qwen   to stop it
```

Chat templates are per-family (Qwen / Llama-3 / Mistral / Gemma), so pulled models chat coherently, not just Qwen.
Drop the prompt — `clozn run llama-1b` — for an interactive chat (multi-turn, `/reset` clears, `/bye` quits).

`clozn run` reuses a running `serve` for that model (warm, no reload); otherwise it spawns a temporary
engine and tears it down after. Stale daemon entries self-heal (a dead one fails its health check).

Every run is debuggable after the fact — the engine streams per-token confidence + the alternatives it weighed:

```bash
clozn trace                               # last shared runlog entry: confidence timeline + almost-said tokens
clozn trace --legacy-cache                # old ~/.clozn/traces cache, kept for compatibility
clozn branch                              # re-run from the most uncertain token on the alternative
```

`clozn trace` reads the same `~/.clozn/runs` journal that Studio's Runs page and Run Inspector use. The
older `~/.clozn/traces` cache is still written for compatibility and branching, but it is no longer the
default trace source.

`clozn run …` works once the repo root is on PATH; otherwise `python clozn_cli.py run …`. Put GGUFs in
`~/.clozn/models`, set `CLOZN_MODELS=<dir>`, or list dirs in `~/.clozn/config.json`. Build the engine
first: `cd engine/core && build_gpu.bat` (GPU, CUDA) or `build_serve.bat` (CPU).

## Layout

| Dir | What |
|---|---|
| `engine/`    | the runtime ("cloze"): C++/ggml core + the Python `lab/` reference — runs models, emits the state-stream, applies steers |
| `kernels/`   | GPU kernels (confidence-select today; interp kernels — SAE top-k — to come) |
| `inspector/` | the white-box product: the spine, ops (snapshot/restore/edit/probe/steer), memory, viz — what you actually use |
| `research/`  | the legibility science (the interpretability-tax thread) |
| `protocol/`  | the one state-stream contract the engine emits and the inspector consumes |
| `docs/`      | architecture + the honest technical account |

Three model substrates behind one spine: **diffusion** (LLaDA/Dream), **autoregressive**
(Llama/Qwen/...), and **recurrent** (RWKV). The thing you view and steer — "the model's
memory" — is its evolving internal state, made legible and editable.
