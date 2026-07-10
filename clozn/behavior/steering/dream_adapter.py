"""Dream-family diffusion adapter steering."""
from __future__ import annotations

import numpy as np
import torch

from . import axes
from .hf_adapter import SteeringControl


class DreamSteering(SteeringControl):
    """Tone dials for a Dream-family diffusion adapter."""

    def __init__(self, adapter, layer: int | None = None):
        self.ad = adapter
        self.model = adapter._model
        self.tok = adapter._tok
        self.dev = adapter._device
        base_model = getattr(self.model, "model", self.model)
        self._layers = base_model.layers
        n = len(self._layers)
        self.layer = layer if layer is not None else n // 2
        self.vecs, self.strength = {}, {}
        self.base, self.resid_norm = 1.0, 0.0
        self._handle = None

    @torch.no_grad()
    def _resid(self, text: str) -> torch.Tensor:
        """Mean-pooled residual at self.layer for a chat-wrapped prompt."""
        ids = self.ad.encode(text, chat=True)
        board = np.asarray(ids, dtype=np.int64)
        n = len(ids)
        attn = np.ones((n, n), dtype=bool)
        cap = {}

        def grab(m, i, o):
            cap["h"] = (o[0] if isinstance(o, tuple) else o).detach()

        hh = self._layers[self.layer].register_forward_hook(grab)
        try:
            self.ad.forward(board, attn)
        finally:
            hh.remove()
        return cap["h"][0].float().mean(0)

    @torch.no_grad()
    def compute(self, seeds=None) -> dict:
        seeds = axes.SEED_PROMPTS if seeds is None else seeds
        out, norms = {}, []
        for name, ax in axes.AXES.items():
            pos = torch.stack([self._resid(ax["pos"] + "\n\n" + s) for s in seeds]).mean(0)
            neg = torch.stack([self._resid(ax["neg"] + "\n\n" + s) for s in seeds]).mean(0)
            norms += [float(pos.norm()), float(neg.norm())]
            d = pos - neg
            self.vecs[name] = d / (d.norm() + 1e-8)
            out[name] = round(float(d.norm()), 2)
        self.resid_norm = sum(norms) / len(norms)
        self.base = 0.85 * self.resid_norm
        return {"raw_norms": out, "resid_norm": round(self.resid_norm, 1), "base": round(self.base, 2)}
