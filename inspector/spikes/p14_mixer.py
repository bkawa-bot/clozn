"""
Phase-14 — break the p13 wall with a proper Mixer.

p13: the minimal single-softmax read couldn't content-address (find the marked value in
varied context) and the slots collapsed. Fix = MetaState's two missing pieces:
  - MULTI-HEAD cross-attention read + inject (real content addressing),
  - learnable per-slot IDENTITY embeddings (break symmetry so slots specialize).
Same marker-recall task + slot->X legibility probe. If recall climbs above chance AND the
slot decodes to X, the working memory is content-addressable AND legible.

Usage: <cloze venv python> spikes/p14_mixer.py [open-dcoder|dream7b] [n_train]
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

MODEL = sys.argv[1] if len(sys.argv) > 1 else "open-dcoder"
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
PROMPT_LEN, L, M_SLOTS, V_SIZE, HEADS = 8, 4, 8, 16, 8
TEMPLATE = [501, 502, 503]
MARKER = 499


def build_adapter():
    if MODEL == "dream7b":
        from cloze_lab.models.dream import DREAM_7B_INSTRUCT, DreamAdapter
        return DreamAdapter(LoadConfig(model_id=DREAM_7B_INSTRUCT, device="cuda", dtype="bfloat16"),
                            quantization="nf4")
    from cloze_lab.models.dream import open_dcoder_adapter
    return open_dcoder_adapter(LoadConfig(model_id="fredzzp/open-dcoder-0.5B", device="cuda", dtype="float32"))


def mha(q, k, v, H):                                            # q[Q,d] k,v[N,d] -> [Q,d]
    Q, d = q.shape; N = k.shape[0]; hd = d // H
    qh = q.view(Q, H, hd).transpose(0, 1)                       # [H,Q,hd]
    kh = k.view(N, H, hd).transpose(0, 1)
    vh = v.view(N, H, hd).transpose(0, 1)
    att = torch.softmax(qh @ kh.transpose(-2, -1) / (hd ** 0.5), dim=-1)
    return (att @ vh).transpose(0, 1).reshape(Q, d)


class SlotMemory(nn.Module):
    def __init__(self, d, M, H):
        super().__init__()
        self.d, self.M, self.H = d, M, H
        self.slot0 = nn.Parameter(torch.randn(M, d) * 0.02)
        self.ident = nn.Parameter(torch.randn(M, d) * 0.02)     # learnable per-slot identity
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False); self.v = nn.Linear(d, d, bias=False)
        self.gru = nn.GRUCell(d, d)
        self.iq = nn.Linear(d, d, bias=False); self.ik = nn.Linear(d, d, bias=False); self.iv = nn.Linear(d, d, bias=False)
        self.alpha = nn.Parameter(torch.zeros(1))

    def init(self):
        return self.slot0

    def read(self, slots, hidden):                             # multi-head: identity-augmented slots query hidden
        return mha(self.q(slots + self.ident), self.k(hidden), self.v(hidden), self.H)

    def update(self, slots, rd):
        return self.gru(rd, slots)

    def inject(self, slots, emb):                             # multi-head: positions query slots
        return self.alpha * mha(self.iq(emb), self.ik(slots + self.ident), self.iv(slots), self.H)


def main():
    print(f"building {MODEL} ...")
    ad = build_adapter()
    model = ad._model; model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    d = model.config.hidden_size; L_READ = int(model.config.num_hidden_layers * 2 // 3)
    MASKT = ad.config.mask_token_id; embed = model.get_input_embeddings(); vocab = model.config.vocab_size
    mem = SlotMemory(d, M_SLOTS, HEADS).to(dev).to(dt)

    WORDS = [" the", " and", " of", " a", " in", " to", " is", " it", " that", " for", " on",
             " with", " as", " was", " he", " at", " by", " not", " you", " are"]
    V = []
    for w in WORDS:
        tid = ad.encode(w)
        if tid and tid[0] not in V:
            V.append(tid[0])
        if len(V) >= V_SIZE:
            break
    rng = np.random.default_rng(0)
    n = PROMPT_LEN + L; recall = n - 1
    pos = torch.arange(n, device=dev).unsqueeze(0)
    mask_full = torch.zeros((1, 1, n, n), dtype=dt, device=dev)
    mask_hide = mask_full.clone(); mask_hide[:, :, PROMPT_LEN:, :PROMPT_LEN] = torch.finfo(dt).min
    print(f"  d={d} read[{L_READ}] M={M_SLOTS} heads={HEADS} |V|={len(V)} (chance {100/len(V):.1f}%)")

    def make_board(X):
        fill = [int(rng.integers(vocab)) for _ in range(PROMPT_LEN - 2)]
        return [MARKER, X] + fill + TEMPLATE + [MASKT]

    def fwd(board, t, slots):
        base = embed(torch.tensor([board], device=dev))
        emb = base + mem.inject(slots, base[0]).view(1, n, d)
        return model(inputs_embeds=emb, attention_mask=(mask_full if t == 0 else mask_hide),
                     position_ids=pos, output_hidden_states=True, use_cache=False)

    opt = torch.optim.Adam(mem.parameters(), lr=1.5e-3)
    print(f"\ntraining {N_TRAIN} steps (marker recall, multi-head Mixer) ...")
    run = []
    for i in range(N_TRAIN):
        X = int(V[rng.integers(len(V))]); board = make_board(X)
        opt.zero_grad()
        slots = mem.init()
        o0 = fwd(board, 0, slots)
        slots = mem.update(slots, mem.read(slots, o0.hidden_states[L_READ][0]))
        o1 = fwd(board, 1, slots)
        sh = torch.cat([o1.logits[0][:1], o1.logits[0][:-1]], dim=0)
        loss = F.cross_entropy(sh[recall].float().unsqueeze(0), torch.tensor([X], device=dev))
        loss.backward(); opt.step(); run.append(float(loss.detach()))
        if (i + 1) % 300 == 0:
            print(f"  step {i+1:4d}  loss {np.mean(run[-300:]):.4f}  alpha {float(mem.alpha):+.3f}")

    lab2i = {v: i for i, v in enumerate(V)}
    S, Y, top1, top5 = [], [], [], []
    with torch.no_grad():
        for _ in range(900):
            X = int(V[rng.integers(len(V))]); board = make_board(X)
            slots = mem.init(); o0 = fwd(board, 0, slots)
            slots = mem.update(slots, mem.read(slots, o0.hidden_states[L_READ][0]))
            S.append(slots.flatten().float().cpu().numpy()); Y.append(lab2i[X])
            o1 = fwd(board, 1, slots); sh = torch.cat([o1.logits[0][:1], o1.logits[0][:-1]], dim=0)
            lg = sh[recall].float(); top1.append(int(lg.argmax()) == X)
            top5.append(X in set(int(x) for x in lg.topk(5).indices))
    S = np.stack(S); Y = np.array(Y)
    print(f"\n  recall: top1 {100*np.mean(top1):.1f}%  top5 {100*np.mean(top5):.1f}%  (chance top1 {100/len(V):.1f}% top5 {500/len(V):.1f}%)")

    Xtr = torch.tensor(S[:600], device=dev).float(); Ytr = torch.tensor(Y[:600], device=dev)
    Xte = torch.tensor(S[600:], device=dev).float(); Yte = torch.tensor(Y[600:], device=dev)
    probe = nn.Linear(S.shape[1], len(V)).to(dev).float()
    po = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(500):
        po.zero_grad(); F.cross_entropy(probe(Xtr), Ytr).backward(); po.step()
    with torch.no_grad():
        acc = float((probe(Xte).argmax(1) == Yte).float().mean())
    print(f"  legibility (slot -> X) held-out: {100*acc:.1f}%  (chance {100/len(V):.1f}%)")


if __name__ == "__main__":
    main()
