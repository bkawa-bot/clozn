"""
Phase-13 — the final loop: is the WORKING memory LEGIBLE?

p12 gave a slot that genuinely carries a random token across the discard (recall top5
39% vs 0%). Now: can we READ which token it's holding straight from the slot state? Train
the recall slot (random fillers, so the probe must generalize across contexts, not memorize
16 fixed states), then fit a linear probe slot_state -> X. If it recovers X well above chance,
the memory is WORKING *and* LEGIBLE — the complete glass-box-memory claim.

Usage: <cloze venv python> spikes/p13_memlegible.py [open-dcoder|dream7b] [n_train]
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
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 1800
PROMPT_LEN, L, M_SLOTS, V_SIZE = 8, 4, 4, 16
TEMPLATE = [501, 502, 503]


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


def main():
    print(f"building {MODEL} ...")
    ad = build_adapter()
    model = ad._model; model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = next(model.parameters()).device; dt = next(model.parameters()).dtype
    d = model.config.hidden_size; L_READ = int(model.config.num_hidden_layers * 2 // 3)
    MASK = ad.config.mask_token_id; embed = model.get_input_embeddings(); vocab = model.config.vocab_size
    mem = SlotMemory(d, M_SLOTS).to(dev).to(dt)

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
    print(f"  d={d} read[{L_READ}] |V|={len(V)} (chance {100/len(V):.1f}%)  [random fillers]")

    MARKER = 499
    def make_board(X):                                          # [MARKER, X, random fillers] — explicit cue
        fill = [int(rng.integers(vocab)) for _ in range(PROMPT_LEN - 2)]
        return [MARKER, X] + fill + TEMPLATE + [MASK]

    def forward_step(board, t, slots):
        base = embed(torch.tensor([board], device=dev))
        emb = base + mem.inject(slots, base[0]).view(1, n, d)
        out = model(inputs_embeds=emb, attention_mask=(mask_full if t == 0 else mask_hide),
                    position_ids=pos, output_hidden_states=True, use_cache=False)
        return out

    opt = torch.optim.Adam(mem.parameters(), lr=2e-3)
    print(f"\ntraining {N_TRAIN} steps (recall, random fillers) ...")
    run = []
    for i in range(N_TRAIN):
        X = int(V[rng.integers(len(V))]); board = make_board(X)
        opt.zero_grad()
        slots = mem.init()
        out0 = forward_step(board, 0, slots)
        slots = mem.update(slots, mem.read(slots, out0.hidden_states[L_READ][0]))
        out1 = forward_step(board, 1, slots)
        shifted = torch.cat([out1.logits[0][:1], out1.logits[0][:-1]], dim=0)
        loss = F.cross_entropy(shifted[recall].float().unsqueeze(0), torch.tensor([X], device=dev))
        loss.backward(); opt.step(); run.append(float(loss.detach()))
        if (i + 1) % 300 == 0:
            print(f"  step {i+1:4d}  loss {np.mean(run[-300:]):.4f}  alpha {float(mem.alpha):+.3f}")

    # collect (carried slot state, X) across random contexts; measure recall + probe legibility
    print("\ncollecting slot states + measuring recall ...")
    lab2i = {v: i for i, v in enumerate(V)}
    S, Y, top5 = [], [], []
    with torch.no_grad():
        for _ in range(900):
            X = int(V[rng.integers(len(V))]); board = make_board(X)
            slots = mem.init()
            out0 = forward_step(board, 0, slots)
            slots = mem.update(slots, mem.read(slots, out0.hidden_states[L_READ][0]))
            S.append(slots.flatten().float().cpu().numpy()); Y.append(lab2i[X])
            out1 = forward_step(board, 1, slots)
            shifted = torch.cat([out1.logits[0][:1], out1.logits[0][:-1]], dim=0)
            top5.append(X in set(int(x) for x in shifted[recall].float().topk(5).indices))
    S = np.stack(S); Y = np.array(Y)
    print(f"  recall top5 (full-slot): {100*np.mean(top5):.1f}%  (chance {500/len(V):.1f}%)")

    # linear probe slot_state -> X, held-out
    ntr = 600
    Xtr = torch.tensor(S[:ntr], device=dev); Ytr = torch.tensor(Y[:ntr], device=dev)
    Xte = torch.tensor(S[ntr:], device=dev); Yte = torch.tensor(Y[ntr:], device=dev)
    probe = nn.Linear(S.shape[1], len(V)).to(dev).float()
    po = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-3)
    for _ in range(400):
        po.zero_grad(); l = F.cross_entropy(probe(Xtr.float()), Ytr); l.backward(); po.step()
    with torch.no_grad():
        acc = float((probe(Xte.float()).argmax(1) == Yte).float().mean())
    print(f"\nlegibility — linear probe (slot state -> which token X) held-out: {100*acc:.1f}%  (chance {100/len(V):.1f}%)")
    print("  => the working memory is " + ("LEGIBLE: you can read which token it holds." if acc > 0.25
                                           else "weakly/illegible at this scale."))


if __name__ == "__main__":
    main()
