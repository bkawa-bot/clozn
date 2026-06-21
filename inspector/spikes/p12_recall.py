"""
Phase-12 — the decisive memory test: a task that PROVABLY requires it.

p7-p11 found the slot degenerates to a static bias because generic reconstruction never
*needs* cross-step memory (a strong model just guesses). Fix the task: single-token
associative recall of a RANDOM token.

  board = [X][filler x7] [template x3] [MASK]      (only the last position is masked)
  X = a random token from a 16-token set (UNGUESSABLE).
  step 0: full attention — the slot reads the prompt (encodes X).
  step 1: the prompt is HIDDEN from the continuation — the recall MASK can only be filled
          from the slot. no-slot is at chance (1/16); if full-slot recalls X, it's a real
          1-token memory carried across the discrete-token discard.

Ablation: no-slot / frozen-slot (never reads the prompt) / full-slot. Metric: recall acc.

Usage: <cloze venv python> spikes/p12_recall.py [open-dcoder|dream7b] [n_train]
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
N_TRAIN = int(sys.argv[2]) if len(sys.argv) > 2 else 600
N_EVAL = 200
PROMPT_LEN, L, M_SLOTS = 8, 4, 4
V_SIZE = 16
FILLER, TEMPLATE = 500, [501, 502, 503]


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
    d = model.config.hidden_size; n_layer = model.config.num_hidden_layers
    L_READ = int(n_layer * 2 // 3); MASK = ad.config.mask_token_id
    embed = model.get_input_embeddings()
    mem = SlotMemory(d, M_SLOTS).to(dev).to(dt)

    WORDS = [" the", " and", " of", " a", " in", " to", " is", " it", " that", " for", " on",
             " with", " as", " was", " he", " at", " by", " not", " you", " are"]
    V = []
    for w in WORDS:                                             # high-prior value tokens (model OUTPUTS them readily)
        tid = ad.encode(w)
        if tid and tid[0] not in V:
            V.append(tid[0])
        if len(V) >= V_SIZE:
            break
    rng = np.random.default_rng(0)
    n = PROMPT_LEN + L
    recall = n - 1
    pos = torch.arange(n, device=dev).unsqueeze(0)
    mask_full = torch.zeros((1, 1, n, n), dtype=dt, device=dev)
    mask_hide = mask_full.clone(); mask_hide[:, :, PROMPT_LEN:, :PROMPT_LEN] = torch.finfo(dt).min
    print(f"  d={d} read[{L_READ}] |V|={V_SIZE} (chance recall {100/V_SIZE:.1f}%); board n={n}, recall@{recall}")

    def step(X, mode, want_grad):
        board = [X] + [FILLER] * (PROMPT_LEN - 1) + TEMPLATE + [MASK]
        slots = mem.init()
        ctx = torch.enable_grad() if want_grad else torch.no_grad()
        with ctx:
            loss = None; pred = None
            for t in range(2):
                base = embed(torch.tensor([board], device=dev))
                emb = base + mem.inject(slots, base[0]).view(1, n, d) if mode in ("full", "frozen") else base
                out = model(inputs_embeds=emb, attention_mask=(mask_full if t == 0 else mask_hide),
                            position_ids=pos, output_hidden_states=True, use_cache=False)
                raw = out.logits[0]; shifted = torch.cat([raw[:1], raw[:-1]], dim=0)
                if t == 1:                                       # recall step (prompt hidden)
                    logit = shifted[recall].float()
                    loss = F.cross_entropy(logit.unsqueeze(0), torch.tensor([X], device=dev))
                    pred = int(logit.argmax())
                    in5 = X in set(int(x) for x in logit.topk(5).indices)
                if mode == "full":
                    slots = mem.update(slots, mem.read(slots, out.hidden_states[L_READ][0]))
        return loss, pred, in5

    opt = torch.optim.Adam(mem.parameters(), lr=2e-3)
    print(f"\ntraining {N_TRAIN} steps on single-token recall ...")
    run = []
    for i in range(N_TRAIN):
        X = int(V[rng.integers(V_SIZE)])
        opt.zero_grad(); loss, _, _ = step(X, "full", True); loss.backward(); opt.step()
        run.append(float(loss.detach()))
        if (i + 1) % 150 == 0:
            print(f"  step {i+1:4d}  loss {np.mean(run[-150:]):.4f}  alpha {float(mem.alpha):+.3f}")

    print(f"\nrecall (held-out random X; prompt hidden at recall; chance top1 ~{100/len(V):.1f}% top5 ~{500/len(V):.1f}%):")
    evalX = [int(V[rng.integers(len(V))]) for _ in range(N_EVAL)]
    for mode in ["noslot", "frozen", "full"]:
        outs = [step(X, mode, False) for X in evalX]
        top1 = np.mean([o[1] == X for o, X in zip(outs, evalX)])
        top5 = np.mean([o[2] for o in outs])
        mloss = np.mean([float(o[0]) for o in outs])
        tag = "   <- chance" if mode == "noslot" else ("   <- MEMORY" if mode == "full" else "")
        print(f"  {mode:8}: top1 {100*top1:5.1f}%   top5 {100*top5:5.1f}%   loss {mloss:.3f}{tag}")


if __name__ == "__main__":
    main()
