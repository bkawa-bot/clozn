"""dream_memory_spike.py -- can a soft-prefix MEMORY be trained into a DIFFUSION model?

The AR memory trains a soft prefix against a next-token loss; diffusion has no such loss. This spike
trains a prefix against Dream's MASKED-DENOISING objective instead:

  1. build rule-following targets (denoise [rule + probe] with the base model -> baking-flavored text),
  2. for each, mask the target region and train the prefix so a PLAIN prompt denoises TO that target
     (CE on the masked positions, grad through the frozen nf4 backbone to the prefix), with Dream's
     shifted head (token at pos p is read from logits row p-1) and full bidirectional attention,
  3. apply the prefix via a minimal confidence-based denoise loop.

Honest test: baseline denoise (no prefix) vs after consolidating "into baking" -- does baking surface?

    python research/dream_memory_spike.py      # needs the GPU free (stop clozn first)
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine", "lab"))
from cloze_lab.cli import build_adapter   # noqa: E402

PROBES = ["What should I do this weekend?", "Recommend a relaxing hobby.", "I have a free evening.",
          "Suggest something fun to try.", "How should I unwind tonight?", "Give me an idea for today."]


class DreamMemory:
    def __init__(self, adapter, m=16):
        self.ad = adapter
        self.model = adapter._model
        self.tok = adapter._tok
        self.dev = adapter._device
        self.emb = self.model.get_input_embeddings()
        self.H = self.model.config.hidden_size
        self.mask_id = adapter.config.mask_token_id
        self.eos = adapter.config.eos_token_id
        self.m = m
        self.prefix = None
        self.dtype = next(self.emb.parameters()).dtype

    def _logits(self, board_ids, *, grad=False, use_prefix=True):
        """Forward [prefix? + board] with full (non-causal) attention; return raw logits [m?+n, V].
        Dream's shifted head: board token at board-pos j is read from raw row (off + j - 1)."""
        ids = torch.tensor([board_ids], device=self.dev)
        e = self.emb(ids).to(self.dtype)                                  # [1, n, H]
        if use_prefix and self.prefix is not None:
            e = torch.cat([self.prefix.to(self.dtype)[None], e], 1)       # [1, m+n, H]
        L = e.shape[1]
        mask4d = torch.zeros((1, 1, L, L), dtype=self.dtype, device=self.dev)   # all-visible
        pos = torch.arange(L, device=self.dev).unsqueeze(0)
        ctx = torch.enable_grad() if grad else torch.inference_mode()
        with ctx:
            out = self.model(inputs_embeds=e, attention_mask=mask4d, position_ids=pos)
        return out.logits[0]                                             # [L, V]

    @property
    def _off(self):
        return self.m if self.prefix is not None else 0

    @torch.no_grad()
    def denoise(self, prompt, max_new=40, steps=20, use_prefix=True, min_content=14):
        """Minimal confidence-based denoise WITH the prefix prepended. Greedy. Suppresses EOS for the first
        `min_content` answer slots (the crude loop otherwise commits EOS immediately -> empty output) and
        stops at the first committed EOS."""
        pj = self.ad.encode(prompt, chat=True)
        board = list(pj) + [self.mask_id] * max_new
        lp = len(pj)
        off = self.m if (use_prefix and self.prefix is not None) else 0
        for step in range(steps):
            masked = [j for j in range(lp, len(board)) if board[j] == self.mask_id]
            if not masked:
                break
            lg = self._logits(board, use_prefix=use_prefix)              # [off+n, V]
            probs = torch.softmax(lg.float(), -1)
            best = {}
            for j in masked:
                row = off + j - 1
                dist = probs[row]
                if (j - lp) < min_content:                              # force content early (no premature EOS)
                    dist = dist.clone()
                    dist[self.eos] = 0.0
                p, tid = dist.max(0)
                best[j] = (float(p), int(tid))
            k = max(1, len(masked) // max(1, steps - step))
            for j in sorted(masked, key=lambda j: -best[j][0])[:k]:
                board[j] = best[j][1]
        out = []
        for t in board[lp:]:
            if t == self.eos:
                break                                                   # answer ends at the first EOS
            if t != self.mask_id:
                out.append(t)
        return self.tok.decode(out).strip()

    def consolidate(self, rule, steps=120, lr=0.025, max_norm=30.0, tgt_len=24, mask_frac=0.6):
        sys_rule = f"You are a returning user's assistant. Remember this about them: {rule} Use it naturally."
        # 1. build rule-following targets with the BASE model (rule in the prompt, no prefix)
        targets, masks = [], []
        for pr in PROBES:
            txt = self.denoise(sys_rule + "\n\n" + pr, use_prefix=False)
            tids = self.tok.encode(txt, add_special_tokens=False)[:tgt_len]
            if tids:
                targets.append((self.ad.encode(pr, chat=True), tids))
                keep = torch.rand(len(tids)) < mask_frac                 # a FIXED ~60% mask per target...
                if not bool(keep.any()):
                    keep[0] = True
                masks.append(keep)
        print(f"  built {len(targets)} targets; e.g. -> {self.tok.decode(targets[0][1])[:80]!r}", flush=True)
        # 2. init + TTT the prefix on a PARTIAL masked-denoising loss (mirrors how diffusion trains: predict
        #    a masked SUBSET given the rest -- far easier to fit than predicting the whole target at once).
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
        return {"start": round(start, 3), "final": round(best, 3), "norm": round(float(self.prefix.norm()), 1)}


def main():
    print("loading Dream-7B (4-bit) ... (slow, ~14.5GB cache)", flush=True)
    ad = build_adapter("dream", device="cuda", quant="nf4")
    mem = DreamMemory(ad, m=16)
    tests = ["What's a good way to spend Sunday?", "I have a free hour this afternoon."]
    print("\n=== BASELINE denoise (no memory) ===", flush=True)
    for t in tests:
        print(f"  {t!r} -> {mem.denoise(t, use_prefix=False)!r}", flush=True)
    print("\n=== consolidating 'into baking' (partial masked-denoising TTT) ===", flush=True)
    print("  ", mem.consolidate("They love baking and home cooking."), flush=True)
    print("\n=== denoise WITH the trained memory (same prompts) ===", flush=True)
    for t in tests:
        print(f"  {t!r} -> {mem.denoise(t, use_prefix=True)!r}", flush=True)


if __name__ == "__main__":
    main()
