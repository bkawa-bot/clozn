# The state-stream protocol — spec

The one contract the [engine](../engine) emits and the [inspector](../inspector) consumes.
It collapses the two spines that were the same idea discovered twice:

| Today | Becomes |
|---|---|
| engine's §5.1 typed events (`events.hpp`) — output-oriented + white-box taps | a stream of `StateStep`s |
| inspector's `StateStep / StateSource / Spine` (`spine.py`) — state-oriented | the canonical vocabulary |

**The inspector's vocabulary is canonical** (it's already substrate-agnostic and state-first);
the engine's events are a *specialization* of it. "The model's memory" is the `state` field of a
`StateStep`; viewing = reading the stream; steering = posting an `Intervention`.

## Canonical types

```
StateStep   { step:int, token:any, state:State, readouts:[Readout], meta:{...} }
Readout     { name:str, value:any, confidence:float, causal_verified:bool|null }
Intervention{ kind:"steer"|"edit"|"restore"|"patch", target:{...}, vector?:[f], coef?:f, note?:str }
State       = { "<component>": tensor }      # diffusion canvas | RWKV state | AR residual/KV slice
```

- **`StateStep`** — one frame of the evolving state. `state` is the internal state *after* this
  step (the substrate's "memory" slice); `token` is what was consumed/committed; `readouts` are
  probe/lens/feature readings; `meta` carries `substrate`, timing, commit counts, block, etc.
- **`Readout`** — a reading that **never travels without its confidence**, and **never claims
  causality until verified** (`causal_verified`: null = unchecked, true/false = patched-and-measured).
  This is the honesty invariant on the wire.
- **`Intervention`** — the write-back channel: steer a direction, edit a slot, restore a snapshot.

`StateSource` (reset / step / get_state / set_state) and `Spine` (drive a source, fan steps to
consumers) stay as in `spine.py` — the engine is just one more `StateSource`, reached over the wire.

## Engine §5.1 event → StateStep mapping

One forward pass's events fold into one `StateStep`:

| Engine event | Folds into |
|---|---|
| `gen_started` / `gen_finished` | stream control frames (`meta.kind = "begin"/"end"`, prompt/total) |
| `tokens_committed` (items) | `StateStep.token` (the committed id(s)) + `meta.confidence` |
| `step_features` (concept scores) | `readouts`: one `Readout` per concept (`name`, `value`=score, per slot) |
| `step_lens` (top-k candidates) | a `Readout` (`name="logit-lens"`, `value`=candidates+probs) |
| `step_stats` (committed/remaining/ms) | `meta` (committed, remaining, ms, cache_hit) |
| `tokens_revised` | a `StateStep` with `meta.kind="revise"` (diffusion-only) |
| block start/finalize | `meta.block`, `meta.span` |
| the activation tap (`ForwardResult.activations`) | `StateStep.state` (the per-position hidden state) |

`meta.substrate ∈ {"diffusion","autoregressive","recurrent"}` tags every step.

## The wire (SSE + JSON)

**Light frame by default, heavy state on demand.** Streaming the full activation tensor every
step is gigabytes; most consumers want tokens + readouts + meta. So:

- **Stream** (`GET …/stream`, SSE): each `StateStep` as `data: {json}\n\n`, with `state` **omitted
  or summarized** (shapes + a sparse/projected view) unless `?state=full`. This is the engine's
  existing event SSE, reshaped to the `StateStep` schema.
- **Snapshot** (`GET …/state`): the full `State` (named tensors, shape + base64/npy) — for
  `snapshot/restore/associate`. Pulled when the inspector needs the heavy object, not per step.
- **Intervene** (`POST …/intervene`): an `Intervention` — the engine applies it (steer = the
  control-vector `set_steer`; edit/restore = `set_state`) and the effect shows up in the next steps.

Tensors on the wire: `{dtype, shape, data}` (base64 little-endian) — `np.ndarray` ⇄ JSON both ways.

## How phases 1.2–1.3 implement this

- **1.2 (engine emits):** reshape `sse_data(...)` so each frame is a `StateStep` (the mapping
  above), tag `meta.substrate`, add the `/state` + `/intervene` endpoints. Diffusion + AR both.
- **1.3 (inspector consumes):** an `EngineStateSource(StateSource)` whose `step()` reads the SSE
  frame, `get_state()` hits `/state`, `set_state()`/steer POST `/intervene`. Then every existing
  inspector op (snapshot/restore/probe/steer/memory) runs over a *real engine model*, unchanged.
- **1.4 (gate):** drive an engine model through the `Spine`, snapshot → restore → steer → confirm
  the behavior moved. One spine, proven end-to-end.

## Invariants on the wire

Honesty (Readout confidence + `causal_verified`), the source-owns-state rule (consumers only read;
writes go through `Intervention`), and substrate-agnosticism (a new family is a new `StateSource`,
never a protocol change) all hold here — they're the same four
[carried-over invariants](../ARCHITECTURE.md#carried-over-invariants-non-negotiable), now on the wire.
