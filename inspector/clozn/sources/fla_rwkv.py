"""
clozn.sources.fla_rwkv — THE FLAGSHIP adapter (stub; Phase 1, milestone 1).

Wraps a `flash-linear-attention` (fla) RWKV-7 / Gated DeltaNet model so its per-layer recurrent
state becomes a StateSource: read it, snapshot/restore/edit it, stream it per token. This is the
one substrate the research doc flags as a genuinely open gap (DREAMSTATE) — and unlike a
transformer's KV log, the state is a concrete fixed-size matrix you can actually grab.

IMPLEMENTATION PLAN
  1. `pip install flash-linear-attention torch`.
     GPU note: fla's fast path uses Triton kernels. On Windows that's flaky — prefer WSL/Linux,
     or fla's torch-native fallback, and verify on the RTX 5080 (sm_120). The toy source already
     proves the architecture, so this can be developed on CPU / a tiny checkpoint first.
  2. Load a small RWKV-7 or Gated DeltaNet checkpoint via fla.
  3. Run with the recurrent cache enabled and capture each layer's state from the cache object
     (the fixed-size state matrices). Map them to State["layerN.S"].
  4. get_state()/set_state() read & overwrite those cache tensors  ->  snapshot/restore/edit.
  5. step(token) advances one token and emits state + which keys were written + effective rank
     (mirror ToyRecurrentSource.step's meta so the SAME viz/probe UI works unchanged).

Milestone 1 success = swap ToyRecurrentSource -> FlaRecurrentSource in spikes/snapshot_restore.py
and watch snapshot -> mutate -> restore work on a *real* recurrent model.
"""
from __future__ import annotations

from ..spine import State, StateStep


class FlaRecurrentSource:
    def __init__(self, model_name: str, device: str = "cuda"):
        raise NotImplementedError(
            "fla RWKV-7 / Gated DeltaNet adapter — see this module's docstring for the plan. "
            "The toy delta-rule source (toy_recurrent.py) proves the spine/ops today; this "
            "swaps in a real model with the identical StateSource interface."
        )

    # reset(self) / step(self, token) -> StateStep / get_state() -> State / set_state(State)
    # mirror ToyRecurrentSource against the fla model's per-layer recurrent state.
