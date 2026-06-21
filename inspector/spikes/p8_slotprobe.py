"""
Phase-8 — characterize the TRAINED slot (loads runs/slotmem_<model>.pt from p7):

  (1) RECURRENCE ablation — is the 8.8% gain a real memory (dynamic, carried across steps)
      or just a static learned bias?  Compare held-out loss:  no-slot / frozen-slot (inject
      slot0 every step, never updated) / full-slot (read+GRU-update each step).
  (2) LEGIBILITY — does the trained slot stay readable?  Build training-free category probes
      in the read-layer residual, then project the trained slot states onto them: do the
      slots decode to nameable concepts, and do different slots specialize?

Usage: <cloze venv python> spikes/p8_slotprobe.py [open-dcoder|dream7b]
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                 "..", "cloze", "lab")))

import numpy as np   # noqa: E402
import torch         # noqa: E402
import torch.nn as nn          # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cloze_lab.models.base import LoadConfig                      # noqa: E402
from clozn.corpora import text_stream                            # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "open-dcoder"
PROMPT_LEN, L, K, M_SLOTS = 16, 16, 4, 4
CATS = ["punct", "number", "word"]


def build_adapter():
    if MODEL == "dream7b":
        from cloze_lab.models.dream import DREAM_7B_INSTRUCT, DreamAdapter
        return DreamAdapter(LoadConfig(model_id=DREAM_7B_INSTRUCT, device="cuda", dtype="bfloat16"),
                            quantization="nf4")
    from cloze_lab.models.dream import open_dcoder_adapter
    return open_dcoder_adapter(LoadConfig(model_id="fredzzp/open-dcoder-0.5B", device="cuda", dtype="float32"))


class SlotMemory(nn.Module):
    def __init__(self, d, M):
        super().__init__()
        self.d, self.M = d, M
        self.slot0 = nn.Parameter(torch.zeros(M, d))
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False); self.v = nn.Linear(d, d, bias=False)
        self.gru = nn.GRUCell(d, d); self.inj = nn.Linear(d, d); self.alpha = nn.Parameter(torch.zeros(1))

    def init(self):
        return self.slot0

    def read(self, slots, hidden):
        att = torch.softmax(self.q(slots) @ self.k(hidden).t() / (self.d ** 0.5), dim=-1)
        return att @ self.v(hidden)

    def update(self, slots, rd):
        return self.gru(rd, slots)

    def inject(self, slots):
        return self.alpha * self.inj(slots.mean(0))


def category(tok):
    s = tok.strip()
    if not s:
        return None
    if any(c.isdigit() for c in s):
        return "number"
    if not any(c.isalnum() for c in s):
        return "punct"
    if s.isalpha():
        return "word"
    return None


def main():
    print(f"building {MODEL} ...")
    ad = build_adapter()
    model = ad._model; model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    d = model.config.hidden_size; n_layer = model.config.num_hidden_layers
    L_READ = int(n_layer * 2 // 3); MASK = ad.config.mask_token_id
    embed = model.get_input_embeddings()
    ckpt = torch.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   "runs", f"slotmem_{MODEL}.pt"), map_location=dev)
    mem = SlotMemory(d, M_SLOTS).to(dev).to(dt)
    mem.load_state_dict(ckpt["state"])
    mem.eval()
    print(f"  loaded slot module (alpha={float(mem.alpha):+.3f}); read hidden_states[{L_READ}]")

    chunks, ids = [], []
    for t in text_stream():
        ids.extend(ad.encode(t))
        while len(ids) >= PROMPT_LEN + L and len(chunks) < 120:
            chunks.append(ids[:PROMPT_LEN + L]); ids = ids[PROMPT_LEN + L:]
        if len(chunks) >= 120:
            break
    eval_ch = chunks[:60]

    n = PROMPT_LEN + L; cont = list(range(PROMPT_LEN, n))
    mask4d = torch.zeros((1, 1, n, n), dtype=dt, device=dev); pos = torch.arange(n, device=dev).unsqueeze(0)
    per = max(1, L // K)

    @torch.no_grad()
    def denoise(chunk, mode, seed, grab=None):
        board = list(chunk[:PROMPT_LEN]) + [MASK] * L
        order = np.random.default_rng(seed).permutation(cont); revealed = set()
        slots = mem.init(); total = 0.0; nmask = 0
        for t in range(K):
            emb = embed(torch.tensor([board], device=dev))
            if mode in ("full", "frozen"):
                emb = emb + mem.inject(slots).view(1, 1, d)
            out = model(inputs_embeds=emb, attention_mask=mask4d, position_ids=pos,
                        output_hidden_states=True, use_cache=False)
            raw = out.logits[0]; shifted = torch.cat([raw[:1], raw[:-1]], dim=0)
            masked = [p for p in cont if p not in revealed]
            if masked:
                gm = torch.tensor([chunk[p] for p in masked], device=dev)
                total += float(F.cross_entropy(shifted[masked].float(), gm, reduction="sum")); nmask += len(masked)
            new = mem.update(slots, mem.read(slots, out.hidden_states[L_READ][0]))
            if grab is not None:
                grab.append(new.float().cpu().numpy())          # [M,d] trained slot state this step
            if mode == "full":
                slots = new
            for p in order[t * per:(t + 1) * per]:
                board[p] = chunk[p]; revealed.add(p)
        return total / max(nmask, 1)

    print("\n(1) recurrence ablation — held-out reconstruction loss:")
    res = {}
    for mode in ["noslot", "frozen", "full"]:
        res[mode] = np.mean([denoise(ch, mode, 9000 + j) for j, ch in enumerate(eval_ch)])
    print(f"     no-slot       : {res['noslot']:.4f}")
    print(f"     frozen-slot   : {res['frozen']:.4f}   ({100*(res['noslot']-res['frozen'])/res['noslot']:+.1f}% vs no-slot)")
    print(f"     full-slot     : {res['full']:.4f}   ({100*(res['noslot']-res['full'])/res['noslot']:+.1f}% vs no-slot)")
    print(f"     => recurrence adds {100*(res['frozen']-res['full'])/res['noslot']:+.1f}% beyond the static bias")

    # build category probes in the read-layer residual
    px, pc = [], []
    for ch in eval_ch[:40]:
        board = list(ch)                                         # fully revealed -> resolved reps
        out = model(inputs_embeds=embed(torch.tensor([board], device=dev)), attention_mask=mask4d,
                    position_ids=pos, output_hidden_states=True, use_cache=False)
        h = out.hidden_states[L_READ][0].float().cpu().numpy()
        for p in range(n):
            c = category(ad.decode([int(board[p])]))
            if c:
                px.append(h[p]); pc.append(c)
    X = np.stack(px); C = np.array(pc, dtype=object)
    rng = np.random.default_rng(0); idxb = {c: np.where(C == c)[0] for c in CATS}
    nb = min(len(idxb[c]) for c in CATS if len(idxb[c]) > 0)
    bal = np.concatenate([rng.choice(idxb[c], nb, replace=False) for c in CATS if len(idxb[c]) >= nb])
    mu = X[bal].mean(0); sd = X[bal].std(0) + 1e-6; Xs = (X[bal] - mu) / sd; cb = C[bal]
    dirs = {c: (lambda dd: dd / (np.linalg.norm(dd) + 1e-9))(Xs[cb == c].mean(0) - Xs[cb != c].mean(0))
            for c in CATS if (cb == c).sum() >= 5}

    grab = []
    for j, ch in enumerate(eval_ch):
        denoise(ch, "full", 9000 + j, grab=grab)               # collect trained slot states
    slots_arr = np.stack(grab)                                  # [n_obs, M, d]
    print(f"\n(2) trained-slot legibility — project {slots_arr.shape[0]} slot states onto category probes:")

    def proj(v):
        z = (v - mu) / sd
        return {c: float(z @ dirs[c]) for c in dirs}

    for m in range(M_SLOTS):
        vs = slots_arr[:, m, :]
        prof = {c: float(np.mean([proj(v)[c] for v in vs])) for c in dirs}
        best = max(prof, key=prof.get)
        print(f"     slot {m}: leans '{best}'  " + "  ".join(f"{c}={prof[c]:+.2f}" for c in dirs))


if __name__ == "__main__":
    main()
