"""dream_memory.py -- diffusion-native MEMORY: a soft prefix trained on Dream's masked-denoising loss.

Productized from dream_memory_spike.py (validated: "into baking" transfers to a Dream denoise). The AR
memory trains a prefix against next-token CE; diffusion has no such loss, so we train against the
MASKED-DENOISING objective instead -- partial random masking (mirrors how diffusion trains), gradients
through the frozen nf4 backbone via inputs_embeds + Dream's shifted head + bidirectional attention. The
prefix is applied via a prefix-aware denoise loop that also emits the pass-by-pass trace for the viz.

Reusable in the clozn dream substrate (cumulative trait cards + persistence), mirroring SelfTeach on the
qwen side -- so "see + shape" works on BOTH substrates.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

PROBES = ["What should I do this weekend?", "Recommend a relaxing hobby.", "I have a free evening.",
          "Suggest something fun to try.", "How should I unwind tonight?", "Give me an idea for today."]


class PrefixAdapter:
    """Wraps a Dream ModelAdapter so its forward prepends a trained soft prefix in embedding space -- the
    REAL cloze_lab scheduler then drives generation (confidence unmasking, stepping, the event trace),
    with the memory injected transparently. KV reuse breaks under a prepended prefix, so this is for the
    cache-off path (the scheduler's default: full recompute every pass)."""

    def __init__(self, base, prefix):
        import numpy as np
        from cloze_lab.models.base import ForwardResult
        self._base = base
        self._prefix = prefix                          # [m, H] tensor on device
        self.m = int(prefix.shape[0])
        self._model = base._model
        self._emb = self._model.get_input_embeddings()
        self._dev = base._device
        self._dt = next(self._emb.parameters()).dtype
        self._np = np
        self._FR = ForwardResult

    @property
    def config(self):
        return self._base.config

    def encode(self, text, *, chat=False):
        return self._base.encode(text, chat=chat)

    def decode(self, ids):
        return self._base.decode(ids)

    def forward(self, board, attn_mask, *, kv=None, recompute_kv=None, logits_for=None):
        np = self._np
        board = np.asarray(board)
        n = int(board.shape[0])
        want = list(range(n)) if logits_for is None else list(logits_for)
        ids = torch.tensor([board.tolist()], device=self._dev)
        e = self._emb(ids).to(self._dt)
        e = torch.cat([self._prefix.to(self._dt)[None], e], 1)        # [1, m+n, H]
        m, L = self.m, self.m + n
        vis = np.ones((L, L), dtype=bool)                            # prefix visible to all; board per attn_mask
        vis[m:, m:] = np.asarray(attn_mask)
        mask4d = torch.zeros((1, 1, L, L), dtype=self._dt, device=self._dev)
        mask4d.masked_fill_(torch.from_numpy(~vis).to(self._dev)[None, None], torch.finfo(self._dt).min)
        pos = torch.arange(L, device=self._dev).unsqueeze(0)
        with torch.inference_mode():
            out = self._model(inputs_embeds=e, attention_mask=mask4d, position_ids=pos)
        raw = out.logits[0]
        rows = [max(m + p - 1, 0) for p in want]                     # Dream's shifted head, prefix-offset
        return self._FR(logits=raw[rows].float().cpu().numpy(), kv=None)


class DreamMemory:
    def __init__(self, adapter, m=16, persist_path=None):
        self.ad = adapter
        self.model = adapter._model
        self.tok = adapter._tok
        self.dev = adapter._device
        self.emb = self.model.get_input_embeddings()
        self.H = self.model.config.hidden_size
        self.mask_id = adapter.config.mask_token_id
        self.eos = adapter.config.eos_token_id
        self.m = m
        self.prefix: nn.Parameter | None = None
        self.rules: list[str] = []
        self.dtype = next(self.emb.parameters()).dtype
        self.persist = persist_path
        if self.persist:
            self.load()

    # ---- forward: [prefix? + board] with full (non-causal) attention; raw logits (Dream-shifted head) --
    def _logits(self, board_ids, *, grad=False, use_prefix=True):
        ids = torch.tensor([board_ids], device=self.dev)
        e = self.emb(ids).to(self.dtype)
        if use_prefix and self.prefix is not None:
            e = torch.cat([self.prefix.to(self.dtype)[None], e], 1)
        L = e.shape[1]
        mask4d = torch.zeros((1, 1, L, L), dtype=self.dtype, device=self.dev)
        pos = torch.arange(L, device=self.dev).unsqueeze(0)
        ctx = torch.enable_grad() if grad else torch.inference_mode()
        with ctx:
            out = self.model(inputs_embeds=e, attention_mask=mask4d, position_ids=pos)
        return out.logits[0]

    # ---- prefix-aware confidence denoise; optionally emit the pass-by-pass trace for the viz -----------
    @torch.no_grad()
    def denoise(self, prompt, max_new=40, steps=20, use_prefix=True, min_content=14, trace=False):
        pj = self.ad.encode(prompt, chat=True)
        board = list(pj) + [self.mask_id] * max_new
        lp = len(pj)
        off = self.m if (use_prefix and self.prefix is not None) else 0
        passes = []
        for step in range(steps):
            masked = [j for j in range(lp, len(board)) if board[j] == self.mask_id]
            if not masked:
                break
            lg = self._logits(board, use_prefix=use_prefix)
            probs = torch.softmax(lg.float(), -1)
            best = {}
            for j in masked:
                dist = probs[off + j - 1]
                if (j - lp) < min_content:                       # force content early (no premature EOS)
                    dist = dist.clone()
                    dist[self.eos] = 0.0
                p, tid = dist.max(0)
                best[j] = (float(p), int(tid))
            k = max(1, len(masked) // max(1, steps - step))
            items = []
            for j in sorted(masked, key=lambda j: -best[j][0])[:k]:
                board[j] = best[j][1]
                items.append({"pos": j, "piece": self.tok.decode([best[j][1]]), "conf": round(best[j][0], 3)})
            if items:
                passes.append({"pass": len(passes), "items": items})
        out = []
        for t in board[lp:]:
            if t == self.eos:
                break
            if t != self.mask_id:
                out.append(t)
        final = self.tok.decode(out).strip()
        if trace:
            return {"model": "Dream-v0-Instruct-7B · memory", "prompt": prompt,
                    "prompt_text": self.tok.decode(pj), "n_prompt": lp, "board_len": len(board),
                    "steps": steps, "final_text": final, "passes": passes}
        return final

    # ---- consolidate: train the prefix on the partial masked-denoising loss over cumulative rules -------
    def consolidate(self, rules, steps=120, lr=0.025, max_norm=30.0, tgt_len=24, mask_frac=0.6):
        rules = [r for r in (rules or []) if str(r).strip()]
        if not rules:
            return {"ok": False, "reason": "no rules"}
        self.rules = list(rules)
        sys_rule = ("You are a returning user's assistant. Remember this about them:\n"
                    + "\n".join("- " + r for r in rules) + "\nUse it naturally.")
        targets, masks = [], []
        for pr in PROBES:
            txt = self.denoise(sys_rule + "\n\n" + pr, use_prefix=False)
            tids = self.tok.encode(txt, add_special_tokens=False)[:tgt_len]
            if tids:
                targets.append((self.ad.encode(pr, chat=True), tids))
                keep = torch.rand(len(tids)) < mask_frac
                if not bool(keep.any()):
                    keep[0] = True
                masks.append(keep)
        if not targets:
            return {"ok": False, "reason": "no targets"}
        self.prefix = nn.Parameter(0.02 * torch.randn(self.m, self.H, device=self.dev, dtype=torch.float32))
        opt = torch.optim.Adam([self.prefix], lr=lr, weight_decay=2e-3)

        def loss_on(i):
            pj, tids = targets[i]
            keep = masks[i]
            board = list(pj) + [self.mask_id if bool(keep[t]) else tids[t] for t in range(len(tids))]
            lg = self._logits(board, grad=True, use_prefix=True)
            rows = [self.m + len(pj) + t - 1 for t in range(len(tids)) if bool(keep[t])]
            tgt = [tids[t] for t in range(len(tids)) if bool(keep[t])]
            return F.cross_entropy(lg[rows].float(), torch.tensor(tgt, device=self.dev))

        def avg():
            with torch.no_grad():
                return sum(loss_on(i).item() for i in range(len(targets))) / len(targets)

        start = best = avg()
        best_prefix = self.prefix.detach().clone()
        bad = 0
        for step in range(steps):
            opt.zero_grad()
            for i in range(len(targets)):
                (loss_on(i) / len(targets)).backward()
            torch.nn.utils.clip_grad_norm_([self.prefix], 2.0)
            opt.step()
            with torch.no_grad():
                n = float(self.prefix.norm())
                if n > max_norm:
                    self.prefix.mul_(max_norm / n)
            if step % 3 == 2:
                cur = avg()
                if cur < best - 1e-3:
                    best, bad = cur, 0
                    best_prefix = self.prefix.detach().clone()
                else:
                    bad += 1
                    if bad >= 10:
                        break
        with torch.no_grad():
            self.prefix.copy_(best_prefix)
        if self.persist:
            self.save()
        return {"ok": True, "rules": self.rules, "n_targets": len(targets),
                "start": round(start, 3), "final": round(best, 3), "norm": round(float(self.prefix.norm()), 1)}

    # ---- persistence (mirrors SelfTeach.save/load) -----------------------------------------------------
    def save(self, path=None):
        path = path or self.persist
        if not path or self.prefix is None:
            return False
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"m": self.m, "prefix": self.prefix.detach().cpu(), "rules": self.rules}, path)
        return True

    def load(self, path=None):
        path = path or self.persist
        if not path or not os.path.isfile(path):
            return False
        try:
            d = torch.load(path, map_location="cpu")
        except Exception:
            return False
        if d.get("m") != self.m:
            return False
        self.prefix = nn.Parameter(d["prefix"].to(self.dev).float())
        self.rules = d.get("rules", [])
        return True

    def reset(self):
        self.prefix = None
        self.rules = []
        if self.persist and os.path.isfile(self.persist):
            try:
                os.remove(self.persist)
            except OSError:
                pass
        return {"ok": True}
