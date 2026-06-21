"""
Phase-7 — the TRAINED glass-box slot: does a small persistent memory module, trained on
a FROZEN diffusion backbone, (a) actually help denoising, and (b) stay LEGIBLE?

This is MetaState-lite (arXiv 2603.01331): M memory slots carried across denoising steps,
read from the mid-layer residual (attention), GRU-updated, and injected back as a learned
bias (zero-init => starts as a no-op so the pretrained behaviour is preserved). Backbone
frozen; only the slot module trains. We compare reconstruction loss WITH vs WITHOUT the
slot on held-out text — then probe the trained slot for legibility (separate spike).

Usage: <cloze venv python> spikes/p7_slottrain.py [open-dcoder|dream7b] [n_train]
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
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 300
N_EVAL = 60
PROMPT_LEN, L, K = 16, 16, 4   # prompt / continuation length / denoise steps
M_SLOTS = 4


def build_adapter():
    if MODEL == "dream7b":
        from cloze_lab.models.dream import DREAM_7B_INSTRUCT, DreamAdapter
        return DreamAdapter(LoadConfig(model_id=DREAM_7B_INSTRUCT, device="cuda", dtype="bfloat16"),
                            quantization="nf4")
    from cloze_lab.models.dream import open_dcoder_adapter
    return open_dcoder_adapter(LoadConfig(model_id="fredzzp/open-dcoder-0.5B", device="cuda", dtype="float32"))


class SlotMemory(nn.Module):
    """M slots carried across steps: attention-read from hidden, GRU-update, inject a bias."""
    def __init__(self, d, M):
        super().__init__()
        self.d, self.M = d, M
        self.slot0 = nn.Parameter(torch.zeros(M, d))
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.gru = nn.GRUCell(d, d)
        self.inj = nn.Linear(d, d)
        self.alpha = nn.Parameter(torch.zeros(1))   # zero-init => slot starts as a no-op

    def init(self):
        return self.slot0

    def read(self, slots, hidden):                  # slots [M,d], hidden [n,d] -> [M,d]
        att = torch.softmax(self.q(slots) @ self.k(hidden).t() / (self.d ** 0.5), dim=-1)
        return att @ self.v(hidden)

    def update(self, slots, rd):
        return self.gru(rd, slots)

    def inject(self, slots):                        # -> [d] bias, broadcast over positions
        return self.alpha * self.inj(slots.mean(0))


def main():
    print(f"building {MODEL} ...")
    ad = build_adapter()
    model = ad._model
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    d = model.config.hidden_size
    n_layer = model.config.num_hidden_layers
    L_READ = int(n_layer * 2 // 3)
    MASK = ad.config.mask_token_id
    embed = model.get_input_embeddings()
    print(f"  d={d}, layers={n_layer}, read hidden_states[{L_READ}], MASK={MASK}, dtype={dt}")

    mem = SlotMemory(d, M_SLOTS).to(dev).to(dt)
    n_params = sum(p.numel() for p in mem.parameters())
    print(f"  slot module: {n_params} params ({100*n_params/sum(p.numel() for p in model.parameters()):.3f}% of backbone)")

    # data: text chunks -> (prompt, gold continuation)
    chunks, ids = [], []
    need = N_TRAIN + N_EVAL
    for t in text_stream():
        ids.extend(ad.encode(t))
        while len(ids) >= PROMPT_LEN + L and len(chunks) < need:
            chunks.append(ids[:PROMPT_LEN + L]); ids = ids[PROMPT_LEN + L:]
        if len(chunks) >= need:
            break
    train_ch, eval_ch = chunks[:N_TRAIN], chunks[N_TRAIN:N_TRAIN + N_EVAL]
    print(f"  {len(train_ch)} train / {len(eval_ch)} eval chunks")

    n = PROMPT_LEN + L
    cont = list(range(PROMPT_LEN, n))
    mask4d = torch.zeros((1, 1, n, n), dtype=dt, device=dev)          # all-visible => bidirectional
    pos = torch.arange(n, device=dev).unsqueeze(0)
    per = max(1, L // K)

    def denoise_loss(chunk, use_slot, reveal_seed):
        gold = chunk
        board = list(chunk[:PROMPT_LEN]) + [MASK] * L
        order = np.random.default_rng(reveal_seed).permutation(cont)
        revealed = set()
        slots = mem.init()
        total = torch.zeros((), device=dev, dtype=torch.float32)
        nmask = 0
        for t in range(K):
            ids_t = torch.tensor([board], device=dev)
            emb = embed(ids_t)
            if use_slot:
                emb = emb + mem.inject(slots).view(1, 1, d)
            out = model(inputs_embeds=emb, attention_mask=mask4d, position_ids=pos,
                        output_hidden_states=use_slot, use_cache=False)
            raw = out.logits[0]
            shifted = torch.cat([raw[:1], raw[:-1]], dim=0)          # row p = distribution for position p
            masked = [p for p in cont if p not in revealed]
            if masked:
                gm = torch.tensor([gold[p] for p in masked], device=dev)
                total = total + F.cross_entropy(shifted[masked].float(), gm, reduction="sum")
                nmask += len(masked)
            if use_slot:
                slots = mem.update(slots, mem.read(slots, out.hidden_states[L_READ][0]))
            for p in order[t * per:(t + 1) * per]:
                board[p] = gold[p]; revealed.add(p)
        return total / max(nmask, 1)

    opt = torch.optim.Adam(mem.parameters(), lr=2e-3)
    print("\ntraining the slot (backbone frozen) ...")
    run = []
    for i, ch in enumerate(train_ch):
        opt.zero_grad()
        loss = denoise_loss(ch, use_slot=True, reveal_seed=i)
        loss.backward()
        opt.step()
        run.append(float(loss))
        if (i + 1) % 50 == 0:
            print(f"  step {i+1:4d}  loss {np.mean(run[-50:]):.4f}  alpha {float(mem.alpha):+.3f}")

    print("\nheld-out reconstruction loss (same reveal order both arms):")
    with torch.no_grad():
        base = np.mean([float(denoise_loss(ch, use_slot=False, reveal_seed=10000 + j)) for j, ch in enumerate(eval_ch)])
        slot = np.mean([float(denoise_loss(ch, use_slot=True, reveal_seed=10000 + j)) for j, ch in enumerate(eval_ch)])
    print(f"  no slot : {base:.4f}")
    print(f"  + slot  : {slot:.4f}   (Δ {slot - base:+.4f}, {100*(base-slot)/base:+.1f}% better)" )
    torch.save({"state": mem.state_dict(), "d": d, "M": M_SLOTS, "L_READ": L_READ},
               os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", f"slotmem_{MODEL}.pt"))
    print("  saved slot module -> runs/slotmem_" + MODEL + ".pt")


if __name__ == "__main__":
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs"), exist_ok=True)
    main()
