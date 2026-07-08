# Workspace Lens

Workspace Lens is the UI-visible seam for per-token, per-layer latent workspace readouts.

This first implementation is deliberately small:

- `workspace_readout` is an additive event type in the generation event spine.
- `research/workspace_lens.py` adapts existing Clozn concept/SAE readouts into that event shape.
- `research/clozn_server.py` prefers the live C++ engine concept path (`/engine/concepts`, backed by
  `/harvest` + SAE) and stores provider `engine_concepts` when available.
- If the engine concept path is unavailable but the Python Qwen + SAE brain stack is loaded, the server
  stores provider `sae/probe`.
- The Studio Run Inspector displays the latest readout, provider name, token strip, and fogginess.
- `inspector/demo/workspace_lens_trace.jsonl` is a tiny fixture trace for demos and schema examples.

Mock readouts are not auto-attached to real runs. The deterministic mock provider remains only for fixture
or offline sample traces, where it should be treated as UI/sample data rather than interpretability evidence.
Its sample labels are:

- `code_error`
- `uncertainty`
- `memory_reference`
- `instruction_following`
- `hallucination_risk`

Future providers should keep the same payload shape and replace only the readout source. Good adapter
targets include logit lens, Jacobian Lens, SAE probes, and linear probes. Do not label a provider as
J-Space or Jacobian Lens until that adapter exists.

## Event Payload

```json
{
  "type": "workspace_readout",
  "run_id": "run_...",
  "token_index": 3,
  "token_text": " maybe",
  "layer": 12,
  "position": 3,
  "top_readouts": [
    { "label": "dragon/fear/RPG", "score": 0.83 },
    { "label": "mythic creatures", "score": 0.58 }
  ],
  "entropy": 0.67,
  "provider": "engine_concepts"
}
```

`entropy` is the current UI's fogginess signal. The engine-concepts adapter derives it from the aligned
token confidence until the engine emits a per-token concept entropy directly.
