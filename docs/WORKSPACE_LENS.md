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
Jacobian Lens until that adapter exists.

## Taxonomy Decision

Use `workspace_readout` as the generic persisted adapter event. Do not add a separate
`concept_readout` event for SAE/probe concept labels; concepts are one `readout_kind` inside the
same event. This keeps old traces renderable and gives future providers a stable plug-in slot without
renaming persisted data.

New writers should include:

- `provider`: concrete adapter name, for example `engine_concepts`, `qwen_scope_sae_l15`, or
  `logit_lens_l20`
- `provider_type`: `sae`, `probe`, `logit_lens`, `jacobian_lens`, `mock`, or `engine_concepts`
- `readout_kind`: `concept`, `token`, `feature`, `risk`, or `summary`

Existing traces that only contain `provider` are still valid. The run logger fills subtype fields when
it can infer them from known provider names.

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
  "provider": "engine_concepts",
  "provider_type": "engine_concepts",
  "readout_kind": "concept"
}
```

`entropy` is the current UI's fogginess signal. The engine-concepts adapter derives it from the aligned
token confidence until the engine emits a per-token concept entropy directly.

## Provider Examples

SAE feature readout:

```json
{
  "type": "workspace_readout",
  "provider": "qwen_scope_sae_l15",
  "provider_type": "sae",
  "readout_kind": "feature",
  "run_id": "run_demo",
  "token_index": 4,
  "token_text": " dragons",
  "layer": 15,
  "position": 4,
  "top_readouts": [
    { "label": "sae:1421 mythology/fantasy", "score": 0.83 },
    { "label": "sae:0774 creature/entity", "score": 0.61 }
  ],
  "entropy": 0.22
}
```

Linear probe readout:

```json
{
  "type": "workspace_readout",
  "provider": "tone_probe_v1",
  "provider_type": "probe",
  "readout_kind": "concept",
  "run_id": "run_demo",
  "token_index": 7,
  "token_text": " should",
  "layer": 12,
  "position": 7,
  "top_readouts": [
    { "label": "instruction_following", "score": 0.79 },
    { "label": "uncertainty", "score": 0.18 }
  ],
  "entropy": 0.18
}
```

Logit lens token readout:

```json
{
  "type": "workspace_readout",
  "provider": "logit_lens_l20",
  "provider_type": "logit_lens",
  "readout_kind": "token",
  "run_id": "run_demo",
  "token_index": 9,
  "token_text": " Paris",
  "layer": 20,
  "position": 9,
  "top_readouts": [
    { "label": " Paris", "score": 0.42 },
    { "label": " London", "score": 0.21 }
  ],
  "entropy": 0.64
}
```

Future Jacobian Lens readout:

```json
{
  "type": "workspace_readout",
  "provider": "future_jacobian_lens_adapter",
  "provider_type": "jacobian_lens",
  "readout_kind": "feature",
  "run_id": "run_demo",
  "token_index": 11,
  "token_text": " because",
  "layer": 18,
  "position": 11,
  "top_readouts": [
    { "label": "answer_support_direction", "score": 0.36 },
    { "label": "citation_sensitivity", "score": 0.24 }
  ],
  "entropy": 0.49
}
```

These readouts are model-state or logit observations. They are not private reasoning text, and they do
not imply causal proof unless a separate intervention/receipt verifies the effect.
