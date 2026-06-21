"""
clozn.sources.hf_transformer — the autoregressive (transformer) substrate, finally (Phase 4).

A real HF transformer (Qwen, or Dream's Qwen backbone) as a white-box surface: hook a decoder
layer's residual stream and collect per-token activations over a big streamed corpus — the
(activation, token) data a real SAE/transcoder trains on. Standard transformer, so plain forward
hooks suffice (no custom kernels). Auto-uses CUDA if available; falls back to CPU.
"""
from __future__ import annotations

import numpy as np


class TransformerActs:
    """Collect residual-stream activations from layer `layer` of an HF causal LM."""

    def __init__(self, name: str = "Qwen/Qwen2.5-0.5B", layer: int = 12, device: str | None = None):
        import os
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = (AutoModelForCausalLM
                      .from_pretrained(name, torch_dtype=torch.float32)
                      .to(device).eval())
        self.layer = layer
        self.hidden = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers
        self._buf: list = []
        self._h = self.model.model.layers[layer].register_forward_hook(
            lambda m, i, o: self._buf.append((o[0] if isinstance(o, tuple) else o).detach().float().cpu()))

    def collect(self, n_tokens: int = 50000, ctx: int = 64, batch: int = 16, source: str = "wikitext"):
        """Stream the corpus, tokenize into ctx-length chunks (no padding), batch-forward, and
        capture this layer's output for every token. Returns (acts[N, hidden], tokens[N])."""
        from ..corpora import text_stream
        torch = self.torch
        ids: list[int] = []
        for t in text_stream(source):
            ids.extend(self.tok(t, add_special_tokens=False).input_ids)
            if len(ids) >= n_tokens:
                break
        ids = ids[: (min(len(ids), n_tokens) // ctx) * ctx]
        chunks = [ids[i:i + ctx] for i in range(0, len(ids), ctx)]
        acts, toks = [], []
        for b in range(0, len(chunks), batch):
            bb = chunks[b:b + batch]
            x = torch.tensor(bb, device=self.device)
            self._buf.clear()
            with torch.no_grad():
                self.model(x)
            a = self._buf[0]                                  # [B, ctx, hidden]
            acts.append(a.reshape(-1, a.shape[-1]).numpy())
            for chunk in bb:
                toks.extend(self.tok.decode([i]) for i in chunk)
        return np.concatenate(acts, 0), toks

    def close(self):
        self._h.remove()
