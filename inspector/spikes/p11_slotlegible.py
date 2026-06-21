"""
Phase-11 — the crux: is a TRAINED, WORKING memory still LEGIBLE?

Uses p10's working architecture (forcing task + position-aware injector + distinct slots),
trains longer, re-confirms the memory signal (full > frozen), then PROBES the trained slot
states with training-free category probes: do the slots decode to nameable concepts, and
do the M slots SPECIALIZE (different concepts) — or collapse? This answers the legible-
interior question for in-model memory: does usefulness cost legibility?

Usage: <cloze venv python> spikes/p11_slotlegible.py [open-dcoder|dream7b] [n_train]
"""
import os
import sys

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
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 800
N_EVAL = 60
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
        self.slot0 = nn.Parameter(torch.randn(M, d) * 0.02)
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False); self.v = nn.Linear(d, d, bias=False)
        self.gru = nn.GRUCell(d, d)
        self.iq = nn.Linear(d, d, bias=False); self.ik = nn.Linear(d, d, bias=False); self.iv = nn.Linear(d, d, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))

    def init(self):
        return self.slot0

    def read(self, slots, hidden):
        att = torch.softmax(self.q(slots) @ self.k(hidden).t() / (self.d ** 0.5), dim=-1)
        return att @ self.v(hidden)

    def update(self, slots, rd):
        return self.gru(rd, slots)

    def inject(self, slots, emb):
        att = torch.softmax(self.iq(emb) @ self.ik(slots).t() / (self.d ** 0.5), dim=-1)
        return self.alpha * (att @ self.iv(slots))


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
    mem = SlotMemory(d, M_SLOTS).to(dev).to(dt)

    chunks, ids = [], []
    for t in text_stream():
        ids.extend(ad.encode(t))
        while len(ids) >= PROMPT_LEN + L and len(chunks) < N_TRAIN + N_EVAL:
            chunks.append(ids[:PROMPT_LEN + L]); ids = ids[PROMPT_LEN + L:]
        if len(chunks) >= N_TRAIN + N_EVAL:
            break
    train_ch, eval_ch = chunks[:N_TRAIN], chunks[N_TRAIN:N_TRAIN + N_EVAL]

    n = PROMPT_LEN + L; cont = list(range(PROMPT_LEN, n))
    pos = torch.arange(n, device=dev).unsqueeze(0); per = max(1, L // K)
    mask_full = torch.zeros((1, 1, n, n), dtype=dt, device=dev)
    mask_hide = mask_full.clone(); mask_hide[:, :, PROMPT_LEN:, :PROMPT_LEN] = torch.finfo(dt).min

    def denoise(chunk, mode, seed, grab=None):
        board = list(chunk[:PROMPT_LEN]) + [MASK] * L
        order = np.random.default_rng(seed).permutation(cont); revealed = set()
        slots = mem.init(); total = torch.zeros((), device=dev); nmask = 0
        for t in range(K):
            base = embed(torch.tensor([board], device=dev))
            emb = base + mem.inject(slots, base[0]).view(1, n, d) if mode in ("full", "frozen") else base
            mask = mask_full if t == 0 else mask_hide
            out = model(inputs_embeds=emb, attention_mask=mask, position_ids=pos,
                        output_hidden_states=True, use_cache=False)
            raw = out.logits[0]; shifted = torch.cat([raw[:1], raw[:-1]], dim=0)
            masked = [p for p in cont if p not in revealed]
            if masked:
                gm = torch.tensor([chunk[p] for p in masked], device=dev)
                total = total + F.cross_entropy(shifted[masked].float(), gm, reduction="sum"); nmask += len(masked)
            nxt = mem.update(slots, mem.read(slots, out.hidden_states[L_READ][0]))
            if grab is not None:
                grab.append(nxt.float().cpu().numpy())
            if mode == "full":
                slots = nxt
            for p in order[t * per:(t + 1) * per]:
                board[p] = chunk[p]; revealed.add(p)
        return total / max(nmask, 1)

    opt = torch.optim.Adam(mem.parameters(), lr=2e-3)
    print(f"\ntraining {N_TRAIN} steps (forcing task + position-aware injector) ...")
    run = []
    for i, ch in enumerate(train_ch):
        opt.zero_grad(); loss = denoise(ch, "full", i); loss.backward(); opt.step()
        run.append(float(loss.detach()))
        if (i + 1) % 200 == 0:
            print(f"  step {i+1:4d}  loss {np.mean(run[-200:]):.4f}  alpha {float(mem.alpha):+.3f}")

    with torch.no_grad():
        res = {m: np.mean([float(denoise(ch, m, 9000 + j)) for j, ch in enumerate(eval_ch)])
               for m in ["noslot", "frozen", "full"]}
    b = res["noslot"]
    print("\nablation (held-out):")
    print(f"  no-slot {res['noslot']:.4f}  frozen {res['frozen']:.4f} ({100*(b-res['frozen'])/b:+.1f}%)  "
          f"full {res['full']:.4f} ({100*(b-res['full'])/b:+.1f}%)  => MEMORY {100*(res['frozen']-res['full'])/b:+.1f}%")

    # probes in the read-layer residual
    px, pc = [], []
    with torch.no_grad():
        for ch in eval_ch[:40]:
            out = model(inputs_embeds=embed(torch.tensor([list(ch)], device=dev)), attention_mask=mask_full,
                        position_ids=pos, output_hidden_states=True, use_cache=False)
            h = out.hidden_states[L_READ][0].float().cpu().numpy()
            for p in range(n):
                c = category(ad.decode([int(ch[p])]))
                if c:
                    px.append(h[p]); pc.append(c)
    X = np.stack(px); C = np.array(pc, dtype=object)
    rng = np.random.default_rng(0); idxb = {c: np.where(C == c)[0] for c in CATS}
    nb = min(len(idxb[c]) for c in CATS if len(idxb[c]) > 0)
    bal = np.concatenate([rng.choice(idxb[c], nb, replace=False) for c in CATS])
    mu = X[bal].mean(0); sd = X[bal].std(0) + 1e-6; Xs = (X[bal] - mu) / sd; cb = C[bal]
    dirs = {c: (lambda dd: dd / (np.linalg.norm(dd) + 1e-9))(Xs[cb == c].mean(0) - Xs[cb != c].mean(0)) for c in CATS}

    grab = []
    with torch.no_grad():
        for j, ch in enumerate(eval_ch):
            denoise(ch, "full", 9000 + j, grab=grab)
    S = np.stack(grab)                       # [obs, M, d]

    def proj(v):
        z = (v - mu) / sd
        return {c: float(z @ dirs[c]) for c in dirs}

    print(f"\ntrained-slot legibility ({S.shape[0]} states): do the {M_SLOTS} slots specialize + decode?")
    leans = []
    for m in range(M_SLOTS):
        prof = {c: float(np.mean([proj(v)[c] for v in S[:, m, :]])) for c in dirs}
        best = max(prof, key=prof.get); leans.append(best)
        print(f"  slot {m}: '{best}'  " + "  ".join(f"{c}={prof[c]:+.2f}" for c in dirs))
    print(f"  distinct concepts across slots: {len(set(leans))}/{M_SLOTS}")


if __name__ == "__main__":
    main()
