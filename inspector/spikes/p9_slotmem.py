"""
Phase-9 — does a real MEMORY emerge when the task REQUIRES it?

p7/p8 found the minimal slot learned a useful *static bias*, not a memory (recurrence
didn't help; slots collapsed). Two fixes, both principled:
  - FORCING TASK: after denoise step 0, hide the prompt from attention (continuation
    queries can't see prompt keys). The only way to reconstruct the continuation is to
    have CARRIED the prompt in the slot across steps. => recurrence is now necessary.
  - SYMMETRY BREAK: distinct random slot init + inject from ALL slots (not the mean),
    so slots can specialize (MetaState's learnable slot identities, minimal form).

Ablation: no-slot / frozen-slot (slot0, never reads the prompt) / full-slot (reads the
prompt at step 0, carries it). If full beats frozen+no-slot, the slot is a real memory.

Usage: <cloze venv python> spikes/p9_slotmem.py [open-dcoder|dream7b] [n_train]
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
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 400
N_EVAL = 60
PROMPT_LEN, L, K, M_SLOTS = 16, 16, 4, 4


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
        self.slot0 = nn.Parameter(torch.randn(M, d) * 0.02)          # DISTINCT init -> can specialize
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False); self.v = nn.Linear(d, d, bias=False)
        self.gru = nn.GRUCell(d, d)
        self.inj = nn.Linear(M * d, d)                              # inject from ALL slots (keep per-slot info)
        self.alpha = nn.Parameter(torch.zeros(1))

    def init(self):
        return self.slot0

    def read(self, slots, hidden):
        att = torch.softmax(self.q(slots) @ self.k(hidden).t() / (self.d ** 0.5), dim=-1)
        return att @ self.v(hidden)

    def update(self, slots, rd):
        return self.gru(rd, slots)

    def inject(self, slots):
        return self.alpha * self.inj(slots.reshape(-1))


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
    print(f"  d={d} layers={n_layer} read[{L_READ}] MASK={MASK}; slot params={sum(p.numel() for p in mem.parameters())}")

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
    mask_hide = mask_full.clone()
    mask_hide[:, :, PROMPT_LEN:, :PROMPT_LEN] = torch.finfo(dt).min     # after step 0: continuation can't see prompt

    def denoise(chunk, mode, seed):
        board = list(chunk[:PROMPT_LEN]) + [MASK] * L
        order = np.random.default_rng(seed).permutation(cont); revealed = set()
        slots = mem.init(); total = 0.0 if mode == "eval_noslot" else torch.zeros((), device=dev)
        total = torch.zeros((), device=dev); nmask = 0
        for t in range(K):
            emb = embed(torch.tensor([board], device=dev))
            if mode in ("full", "frozen"):
                emb = emb + mem.inject(slots).view(1, 1, d)
            mask = mask_full if t == 0 else mask_hide
            out = model(inputs_embeds=emb, attention_mask=mask, position_ids=pos,
                        output_hidden_states=True, use_cache=False)
            raw = out.logits[0]; shifted = torch.cat([raw[:1], raw[:-1]], dim=0)
            masked = [p for p in cont if p not in revealed]
            if masked:
                gm = torch.tensor([chunk[p] for p in masked], device=dev)
                total = total + F.cross_entropy(shifted[masked].float(), gm, reduction="sum"); nmask += len(masked)
            if mode == "full":
                slots = mem.update(slots, mem.read(slots, out.hidden_states[L_READ][0]))
            for p in order[t * per:(t + 1) * per]:
                board[p] = chunk[p]; revealed.add(p)
        return total / max(nmask, 1)

    opt = torch.optim.Adam(mem.parameters(), lr=2e-3)
    print("\ntraining the slot on the FORCING task (prompt hidden after step 0) ...")
    run = []
    for i, ch in enumerate(train_ch):
        opt.zero_grad(); loss = denoise(ch, "full", i); loss.backward(); opt.step()
        run.append(float(loss.detach()))
        if (i + 1) % 80 == 0:
            print(f"  step {i+1:4d}  loss {np.mean(run[-80:]):.4f}  alpha {float(mem.alpha):+.3f}")

    print("\nablation — held-out loss (prompt hidden after step 0; same reveal both arms):")
    with torch.no_grad():
        out = {m: np.mean([float(denoise(ch, m, 9000 + j)) for j, ch in enumerate(eval_ch)])
               for m in ["noslot", "frozen", "full"]}
    b = out["noslot"]
    print(f"  no-slot     : {out['noslot']:.4f}")
    print(f"  frozen-slot : {out['frozen']:.4f}   ({100*(b-out['frozen'])/b:+.1f}% — static bias, never reads prompt)")
    print(f"  full-slot   : {out['full']:.4f}   ({100*(b-out['full'])/b:+.1f}% — reads prompt @step0, carries it)")
    print(f"  => MEMORY (full beyond frozen): {100*(out['frozen']-out['full'])/b:+.1f}% of no-slot loss")


if __name__ == "__main__":
    main()
