# engine/client — Python SDK for the cloze-server white-box API

`cloze_engine.py` is the thin Python seam over the C++ engine's HTTP interface
(`engine/core/serve/cloze_server.cpp`). It exists so the research stack (SAE discovery,
feature circuits, concept probes, all numpy already) can read from and write into the
**live** ggml/llama.cpp model instead of a separate HF copy.

It drives the read -> edit -> write -> observe loop closed by GAP #1:

| method | endpoint | what it does |
| --- | --- | --- |
| `harvest(text, layer=None)` | `POST /harvest` | read every token's residual at the tap layer (one causal forward) |
| `write_state(text, layer, positions, values)` | `POST /state` | overwrite those positions' residual, run again, report how the next-token logits moved |
| `edit_and_observe(text, transform, ...)` | both | the whole loop in one call: harvest, apply `transform(acts)->acts`, write the changed rows back at the same layer |
| `intervene(prompt, concept=/vector=, coef, ...)` | `POST /intervene` | steer a generation with a named concept probe or a raw direction |
| `complete(prompt, **params)` | `POST /v1/completions` | a plain generation (no white-box) |
| `health()` | `GET /health` | `{status, model, mode}` |

Dependencies: the standard library (HTTP/JSON/base64) plus numpy. No `requests`.

The tensor wire format is `{dtype:"float32", shape, data}` where `data` is base64 of the
raw little-endian float32 bytes; `decode_tensor` reconstructs it with a single
`np.frombuffer('<f4')`. Writes go back as a flat, position-major list of floats (the
server checks `len(values) == len(positions) * n_embd`).

## Use it

```python
from cloze_engine import EngineClient
import numpy as np

eng = EngineClient(port=8080)
h = eng.harvest("The capital of France is")     # h.activations: [n_tokens, n_embd] float32

# Edit the last token's residual and observe the prediction shift.
def amplify(acts):
    acts[-1] *= 5.0
    return acts

_, obs = eng.edit_and_observe("The capital of France is", transform=amplify)
print(obs.summary())     # moved_l2, baseline top-3, edited top-3, whether the top-1 flipped
```

## Validate

```
python cloze_engine.py --selftest    # offline: the wire codec is exact, no server needed
python cloze_engine.py --demo --port 8091   # live: a full round-trip against a running server
```

The `--demo` prints an **identity-write control** next to the real edit: writing the
harvested rows back unchanged must move `moved_l2 ~= 0` (it round-trips through base64
losslessly), while a real edit moves the logits and can flip the argmax. That control is
the end-to-end correctness check.

## Bring up a server to test against

```
# from engine/core, with the ggml/llama DLLs on PATH (build-ggml-cpu/bin/Release):
cloze-server <model.gguf> --port 8091 --ctx 512
```

`/harvest` and `/state` work on either a diffusion (LLaDA/Dream/open-dcoder) or an
autoregressive (Qwen/Llama) GGUF; they force a causal forward locally regardless of mode.
