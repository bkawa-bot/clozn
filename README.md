# Clozn

**A local-first runtime where you can view and steer a model's *memory*.** Watch a model
think; read named concepts and token candidates off its evolving internal state; snapshot,
edit, and persist that state; steer concepts into the residual stream — on the models you
run yourself. Ollama's structural opposite: not a black box you prompt, a glass box you
inspect.

`clozn` = `cloze` (the engine inside) + *cozen* (to deceive — the illusion it reveals).

→ **[ARCHITECTURE.md](ARCHITECTURE.md)** — the design, the layers, the state-stream protocol.
→ **[ROADMAP.md](ROADMAP.md)** — from here to the memory frontier, broken into tasks.

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
