"""
clozn.sources.hf_rwkv — a REAL trained recurrent model as a StateSource, via transformers.

No Triton, no custom kernels: HuggingFace's RWKV exposes an explicit, fixed-size recurrent state
(a list of tensors carried token-to-token) — exactly Clozn's substrate: a graspable state you can
snapshot / restore / edit / probe. Delivers Phase-1 M1 on a real model *today*. The fla RWKV-7 /
Gated DeltaNet adapter (fla_rwkv.py) is the same interface for the matrix-valued delta-rule state,
once Triton/WSL is available.

RWKV-4's recurrent state = 5 per-(hidden, layer) tensors:
  att_x   previous token's time-mix input
  att_num WKV numerator (the running weighted memory)
  att_den WKV denominator
  att_max running max (numerical-stability term)
  ffn_x   previous token's channel-mix input
"""
from __future__ import annotations

import os

import numpy as np

from ..spine import State, StateStep

_NAMES = ["att_x", "att_num", "att_den", "att_max", "ffn_x"]


class RwkvStateSource:
    def __init__(self, name: str = "RWKV/rwkv-4-169m-pile", device: str = "cpu"):
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        self.model = (AutoModelForCausalLM
                      .from_pretrained(name, torch_dtype=torch.float32, trust_remote_code=True)
                      .to(device).eval())
        self.device = device
        self.names = _NAMES
        self.reset()

    def reset(self) -> None:
        self.state = None          # transformers initializes on the first forward
        self.t = 0
        self._last_logits = None

    # --- text helpers ---
    def encode(self, text: str) -> list[int]:
        return self.tok(text, return_tensors="pt").input_ids[0].tolist()

    def top_next(self, k: int = 5) -> list[tuple[str, float]]:
        if self._last_logits is None:
            return []
        p = self._last_logits.softmax(-1)[0]
        vals, idx = p.topk(k)
        return [(self.tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(vals, idx)]

    def feed(self, text: str) -> list[StateStep]:
        return [self.step(t) for t in self.encode(text)]

    # --- StateSource interface ---
    def step(self, token_id: int) -> StateStep:
        torch = self.torch
        ids = torch.tensor([[int(token_id)]], device=self.device)
        with torch.no_grad():
            out = self.model(ids, state=self.state, use_cache=True)
        self.state = out.state
        self._last_logits = out.logits[:, -1].detach()
        st = self._np_state()
        self.t += 1
        meta = {
            "token_id": int(token_id),
            "token": self.tok.decode([int(token_id)]),
            "top1": self.tok.decode([int(self._last_logits.argmax())]),   # logit-lens "thought"
            "norms": {n: round(float(np.linalg.norm(st[n])), 3) for n in self.names},
        }
        return StateStep(self.t, meta["token"], st, meta=meta)

    def get_state(self) -> State:
        return self._np_state()

    def set_state(self, s: State) -> None:
        torch = self.torch
        self.state = [torch.tensor(s[n], device=self.device, dtype=torch.float32) for n in self.names]

    def _np_state(self) -> State:
        if self.state is None:
            return {}
        return {n: self.state[i].detach().cpu().numpy().copy() for i, n in enumerate(self.names)}
