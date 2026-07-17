# OpenAI client compatibility

**Status (2026-07-17):** Clozn implements a deliberately small, strict subset of the OpenAI HTTP API.
This is an endpoint/field contract, not a claim of full platform compatibility. Behavior-bearing fields
that Clozn cannot honor return an OpenAI-shaped HTTP 400 instead of being silently ignored.

The field inventory was checked against OpenAI's official
[Chat Completions](https://developers.openai.com/api/reference/resources/chat/methods/create) and
[legacy Completions](https://developers.openai.com/api/reference/resources/completions/methods/create)
references. OpenAI recommends the Responses API for new platform integrations; Clozn does not currently
implement `/v1/responses`.

## Endpoint matrix

| Method and path | Status | Notes |
|---|---|---|
| `GET /v1/models` | supported | one currently loaded local model |
| `POST /v1/chat/completions` | supported subset | text-only, one choice, streaming or non-streaming |
| `POST /v1/completions` | supported subset | legacy text prompt, one choice, streaming or non-streaming |
| `POST /v1/responses` | unsupported | returns the normal route-level 404 |
| embeddings, audio, images, files, batches, fine-tuning | unsupported | no routes |
| stored chat list/get/update/delete | unsupported | Clozn's local run journal is a different API |

The native instrumented stream is `POST /api/clozn/generate`; it is intentionally outside `/v1` so
Clozn state events never leak into a standard OpenAI stream.

## Chat Completions request fields

| Field | Status | Exact behavior |
|---|---|---|
| `model` | supported | labels the response/run; the gateway still serves its one loaded worker |
| `messages` | supported subset | non-empty text `{role, content}` objects; roles `system`, `user`, `assistant`; `developer` is normalized to `system` for local GGUF templates |
| `max_tokens` | supported | positive integer |
| `max_completion_tokens` | supported alias | normalized to `max_tokens`; sending both is a 400 |
| `stream` | supported | boolean; standard `chat.completion.chunk` SSE + `[DONE]` |
| `temperature`, `top_p`, `seed` | supported | forwarded into the request's sampler; explicit fields override Studio's persisted sampling default |
| `n` | one only | `1`/null accepted and stripped; any other value is a 400 |
| `top_k`, `repeat_penalty` | Clozn extensions | forwarded to the engine sampler |
| `clozn_trust`, `clozn_receipt`, `clozn_lens` | Clozn extensions | opt-in confidence spans, receipt delivery, and live J-lens readout |

Message content arrays (images/audio/files), tool/function messages, `name`, `tool_calls`, and other message
fields are rejected. The engine's template seam currently accepts only role + string content.
Sampler fields omitted by the client use Clozn's persisted interactive defaults (initially temperature 0.8,
top-p 0.9, top-k 40, repetition penalty 1.1); fields explicitly present in the request override them.

These fields are accepted **only at a neutral value**, removed before generation, and documented here as
ignored for client interoperability:

| Field | Accepted neutral values |
|---|---|
| `user` | string or null |
| `frequency_penalty`, `presence_penalty` | `0` or null |
| `logprobs` | `false` or null |
| `top_logprobs` | `0` or null |
| `stop`, `audio`, `prediction` | null only |
| `logit_bias`, `metadata` | empty object or null |
| `response_format` | `{"type":"text"}` or null |
| `tools`, deprecated `functions` | empty list or null |
| `tool_choice`, deprecated `function_call` | `"none"` or null |
| `parallel_tool_calls` | `true` or null (irrelevant without tools) |
| `modalities` | `["text"]` or null |
| `store` | `false` or null |
| `service_tier` | `"auto"`, `"default"`, or null |
| `stream_options` | `{"include_usage":false}` or null |

Any behavior-bearing value—tools, stop sequences, JSON/structured output, nonzero frequency/presence
penalties, multiple choices, requested stream usage, and so on—is a 400 with that field in `error.param`.
Unknown top-level fields are also rejected.

## Legacy Completions request fields

| Field | Status | Exact behavior |
|---|---|---|
| `model` | supported | response label for the loaded local worker |
| `prompt` | supported subset | one string; prompt/token arrays are rejected |
| `max_tokens`, `stream`, `temperature`, `top_p`, `seed` | supported | validated and forwarded |
| `n`, `best_of` | one only | `1`/null accepted and stripped |
| `echo` | false only | `false`/null accepted and stripped |
| `top_k` | Clozn extension | forwarded |
| `repeat_penalty` / `rep_penalty` | Clozn extension | one spelling only; normalized to the worker's `rep_penalty` field |
| `user` | documented ignored | string/null accepted and stripped |

`stop`, `suffix`, nonzero penalties, logprobs, logit bias, multiple choices, and unknown fields are rejected
unless they carry the neutral values defined in `clozn/server/openai_compat.py`.

## Response boundary

- Chat and completion response objects/chunks use the standard object names and one choice at index 0.
- Non-streaming chat may add `clozn_run_id` and `X-Clozn-Run-Id`; opt-in Clozn fields are additive.
- Token usage is omitted when unknown. Clozn no longer fabricates zero prompt/completion token counts.
- Finish reasons map worker EOS to `stop` and token limits to `length`. A worker failure is an error, never a
  successful `stop`.
- Streaming chat cannot return a run id in headers because the run is persisted after headers are sent.

Unsupported request example:

```json
{
  "error": {
    "message": "parameter 'tools' is supported only at its neutral/default value; Clozn cannot honor the requested behavior",
    "type": "invalid_request_error",
    "param": "tools",
    "code": "unsupported_parameter"
  }
}
```

## Test evidence

- `tests/test_openai_compat.py` exercises the field table without a model or network.
- `tests/test_openai_client_compat.py` starts the real gateway with a fake substrate and drives model list,
  non-streaming chat, streaming chat, and a typed 400 through the real `openai` Python package.
- CI installs `openai>=1` in the CPU Python lane, so the SDK integration test cannot silently skip there.
- `tests/test_runtime_architecture.py` and `tests/test_product_smoke.py` guard the standard-vs-native stream
  envelope boundary.
