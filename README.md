# Clozn

**A local-first, glass-box runtime for the models you run yourself — view, steer, and *prove*.**
Watch a model think (per-token confidence + the alternatives it weighed), steer its tone, carry memory
as readable cards, and get **causal receipts**: teacher-force a stored answer back through the model to
measure *which* memory it actually leaned on, and by how much. Read a per-token **J-lens**: a fitted
linear lens, applied forward on the GGUF's own head, reads what the model was "disposed to say" at each
position — not a decode of its literal thought (a linear lens always emits *something*). The core runtime
has passed basic/deep qualification across five autoregressive GGUF families; deeper white-box writes and
optional dials/lenses remain model-qualified. Ollama's structural opposite: not a black box you prompt, a
glass box you inspect — and can hold to account. See [model support](docs/MODEL_SUPPORT.md) for the exact
evidence boundary.

`clozn` = `cloze` (the engine inside) + *cozen* (to deceive — the illusion it reveals).

→ **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the design, the layers, the state-stream protocol.
→ **[docs/ROADMAP.md](docs/ROADMAP.md)** — the consolidated map: what's done, the v1 cut, what's next.

## Quickstart

Run a local model in one command. `clozn` starts one public, Torch-free product gateway and one private
C++ model worker. It finds your build (GPU if present), streams tokens, reports honestly what it is
running on, and fails with one clear line instead of a stack trace.

```bash
clozn pull llama-1b                       # download a model (qwen / mistral / gemma-2b / owner/repo/file.gguf)
clozn models                              # discover local GGUFs + the backend that would run them
clozn run llama-1b "Explain entropy."     # one-shot, streams tokens to the terminal
clozn serve qwen --port 8080              # gateway + private worker; API and Studio on :8080
clozn studio --open                       # attach a browser to that already-running gateway
clozn lab qwen --open                     # optional PyTorch workbench; deliberately no /v1 API
clozn ps                                  # what's running    ·    clozn stop qwen   to stop it
```

Before changing the runtime, run its managed acceptance gate:

```bash
clozn smoke qwen --preflight              # report every missing build/model/asset prerequisite
clozn smoke qwen                          # test APIs + SQLite, restart the worker, then clean up
clozn smoke qwen --deep                   # also exercise forced receipts and replay
```

`clozn smoke --url http://127.0.0.1:8080` attaches non-destructively to an existing gateway. Managed
smoke owns the stack it starts, verifies that the private worker can be replaced without changing the
public gateway, and stops the complete process tree when finished.

Chat templates come from each model's own GGUF (Qwen / Llama-3 / Mistral / Gemma / …), applied
engine-side, so pulled models chat coherently — not just Qwen. Drop the prompt — `clozn run llama-1b` —
for an interactive chat (multi-turn; `/reset` clears, `/bye` quits).

`clozn run` reuses a running `serve` for that model (warm, no reload); otherwise it spawns a temporary
gateway/worker pair and tears it down after. Product commands never bypass the gateway to call a warm
worker directly. `clozn serve` supervises the private worker and restarts it after an unexpected exit.

OpenAI clients use the documented subset of `/v1/chat/completions`, `/v1/completions`, and `/v1/models`;
unsupported behavior-bearing fields return a typed 400 instead of being silently ignored. See the exact
[endpoint/field matrix](docs/OPENAI_COMPATIBILITY.md). Clozn's CLI and Studio instrumentation use
`/api/clozn/generate`, which preserves the native state-event stream. Native event frames never leak into
an OpenAI completion stream.

Every run is debuggable — and provable — after the fact. The engine streams per-token confidence + the
alternatives it weighed; open a run in Studio's **Run Inspector** for **causal receipts** (which memory
the answer leaned on, per token), the **"Disposed to say · J-lens"** panel (per-token, per-layer, with an
unskippable provenance caption), the branch lineage tree, and the exact rendered prompt the model saw:

```bash
clozn trace                               # last runlog entry: confidence timeline + almost-said tokens
clozn inspect <clozn_run_id>              # explain any API reply from the local journal; no model needed
clozn branch                              # re-run from the most uncertain token on the alternative
clozn test cases.json                     # run-level assertions over the receipt/replay seams
```

`clozn trace` and `clozn inspect` read the same SQLite journal that heavn uses. OpenAI responses expose
the exact id as `clozn_run_id` and `X-Clozn-Run-Id`; `inspect` assembles confidence, active influences,
and captured concepts locally, falling back to a running gateway only when the id is not in this journal.
Queryable run
metadata lives in `~/.clozn/runs/runs.sqlite3`; large traces are immutable, content-addressed blobs under
`~/.clozn/runs/blobs/sha256`. To import an old beta JSON journal once, run `clozn migrate-runs`.
`clozn test` runs user-authored checks against a stored run: static ones (`contains` / `finish_reason` /
`min_confidence` / `card_applied` / …) read the run alone; the causal `leans_on` check re-runs the real
ablation and honestly **skips** (never a silent pass) unless you pass `--live`. Point any OpenAI client
at `clozn serve`; pass `"clozn_trust": true` in a chat request to get per-claim confidence
spans back on the wire (labeled uncalibrated).

`clozn run …` works once the repo root is on PATH; otherwise `python -m clozn run …`. Put GGUFs in
`~/.clozn/models`, set `CLOZN_MODELS=<dir>`, or list dirs in `~/.clozn/config.json`. Build the engine
first: `cd engine/core && build_gpu.bat` (GPU, CUDA) or `build_serve.bat` (CPU).

## Layout

| Dir | What |
|---|---|
| `clozn/`    | the product Python package — server/API, runlog, memory, receipts, replay, steering, readouts, the J-lens proxy, the tiny-test harness (`python -m clozn`) |
| `engine/`   | the runtime ("cloze"): C++/ggml core + `kernels/` (CUDA) + the Python `lab/` reference — runs models, emits the state-stream, harvests activations, applies steers, serves `/jlens` |
| `studio/`   | the white-box UI — the Run Inspector (receipts, trace, lineage, memory, tone dials, J-lens readouts), served by the backend |
| `protocol/` | the one state-stream contract the engine emits and the studio consumes |
| `docs/`     | architecture, the consolidated roadmap, and the honest technical account |
| `tests/`    | the model-free product suite · `scripts/` dev tooling |

The legibility-science spikes and findings (the interpretability-tax thread) live in a separate
local-only sibling repo: `../clozn-research`.

Two model substrates sit behind one spine today: **autoregressive** GGUF and **diffusion** LLaDA/Dream
(viz-only). The AR core contract includes trace, harvest, steer, and teacher-forced `/score`; the checked-in
qualification ledger records how far each exact model/quant has passed. **J-lens is fit per model**
(offline, nf4 + autograd) and applied forward on the engine's own GGUF head; today's qualified fit covers
Qwen2.5-7B. A second-family fit and targeted cross-family write checks remain open.
