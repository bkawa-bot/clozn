# Workspace Lens

Workspace Lens is the UI-visible seam for per-token, per-layer latent workspace readouts.

This first implementation is deliberately small:

- `workspace_readout` is an additive event type in the generation event spine.
- `research/workspace_lens.py` provides a deterministic mock provider.
- `research/runlog.py` attaches mock readouts to completed runs that already have a token trace.
- The Studio Run Inspector displays the latest readout, provider name, token strip, and fogginess.
- `inspector/demo/workspace_lens_trace.jsonl` is a tiny fixture trace for demos and schema examples.

The mock provider does not inspect real activations and should not be treated as interpretability evidence.
It only emits plausible labels so the product surface and storage shape can be exercised:

- `code_error`
- `uncertainty`
- `memory_reference`
- `instruction_following`
- `hallucination_risk`

Future providers should keep the same payload shape and replace only the readout source. Good adapter
targets include logit lens, Jacobian Lens, SAE probes, and linear probes. A real provider can live beside
the mock provider and emit `workspace_readout` events with `provider` set to its own name.

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
    { "label": "uncertainty", "score": 0.83 },
    { "label": "hallucination_risk", "score": 0.58 }
  ],
  "entropy": 0.67,
  "provider": "mock"
}
```

`entropy` is the current UI's fogginess signal. For the mock provider it is derived from token confidence
and label spread; real providers should document their calibration when they replace it.
