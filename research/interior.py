"""
interior.py — Experiment 2: is a SPARSE working memory legible?

crux.py (Shakespeare) found sparse-but-not-legible units, but it was confounded:
next-char on one play rewards MEMORIZATION, so the sparse code became a string dictionary.

Here the task FORBIDS memorization. A flip-flop parity stream: a hidden bit toggled by
random `F` tokens, queried only occasionally (`Q` then the answer), so the model must
CARRY the bit across gaps. Data is procedurally generated from a huge fresh pool, far
beyond the tiny model's capacity, so the only way to cut loss is to LEARN to track parity.

A recurrent net carries its entire state in a top-k SPARSE bottleneck. We sweep k and ask:
  capability = accuracy at query positions (does it know the bit?), vs
  legibility = accuracy of a LINEAR probe reading the bit out of the sparse state,
               especially at GAP positions where the bit is only being HELD, not used.
If one unit becomes "the bit," that's the nameable unit Shakespeare never gave us — and a
demonstrated monitor (the safety payoff). If the model tracks parity but no probe can read
it, that's capability-without-legibility, in miniature, with ground truth.
"""
import os, sys, json, time
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.set_float32_matmul_precision("high")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs"); os.makedirs(RUNS, exist_ok=True)

# vocab: 0,1,2 filler ; 3=F(flip) ; 4=Q(query) ; 5=answer'0' ; 6=answer'1'
V, NF, FLIP, QUERY, ANS0 = 7, 3, 3, 4, 5
H, E = 64, 16
SYM = {0: ".", 1: ",", 2: ";", 3: "F", 4: "Q", 5: "0", 6: "1"}

def gen_pool(N, L, p_flip=0.20, p_query=0.12, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    X = torch.zeros(N, L, dtype=torch.long, device=DEV)
    par = torch.zeros(N, L, dtype=torch.long, device=DEV)
    s = torch.zeros(N, dtype=torch.long, device=DEV)
    pending = torch.zeros(N, dtype=torch.bool, device=DEV)
    for t in range(L):
        r = torch.rand(N, generator=g, device=DEV)
        tok = torch.randint(0, NF, (N,), generator=g, device=DEV)          # filler default
        flip = (~pending) & (r < p_flip)
        query = (~pending) & (r >= p_flip) & (r < p_flip + p_query)
        s = s ^ flip.long()
        tok = torch.where(flip, torch.full_like(tok, FLIP), tok)
        tok = torch.where(query, torch.full_like(tok, QUERY), tok)
        tok = torch.where(pending, ANS0 + s, tok)                          # answer = current parity
        X[:, t] = tok; par[:, t] = s
        pending = query
    return X, par

def topk_mask(z, k):
    if k >= z.shape[-1]:
        return z
    _, idx = z.abs().topk(k, dim=-1)
    return z * torch.zeros_like(z).scatter_(-1, idx, 1.0)

class SparseRNN(nn.Module):
    """Elman recurrence whose carried state IS the top-k sparse bottleneck."""
    def __init__(self, k):
        super().__init__()
        self.emb = nn.Embedding(V, E)
        self.Wx = nn.Linear(E, H)
        self.Ws = nn.Linear(H, H, bias=False)
        self.head = nn.Linear(H, V)
        self.k = k
    def forward(self, X, return_states=False):
        s = torch.zeros(X.shape[0], H, device=X.device)
        emb = self.emb(X)
        logits, states = [], []
        for t in range(X.shape[1]):
            s = topk_mask(torch.tanh(self.Wx(emb[:, t]) + self.Ws(s)), self.k)
            logits.append(self.head(s))
            if return_states:
                states.append(s)
        logits = torch.stack(logits, 1)
        return (logits, torch.stack(states, 1)) if return_states else logits

def lm_loss(logits, X):
    return F.cross_entropy(logits[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1))

@torch.no_grad()
def capability(model, X):
    pred = model(X)[:, :-1].argmax(-1)
    tgt = X[:, 1:]; m = tgt >= ANS0                       # answer positions only
    return (pred[m] == tgt[m]).float().mean().item()

def train(k, steps=3000, bs=256, lr=2e-3, Ltr=48, seed=0):
    torch.manual_seed(seed)
    model = SparseRNN(k).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pool, _ = gen_pool(20000, Ltr, seed=100)              # huge fixed pool >> model capacity
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    model.train()
    for _ in range(steps):
        Xb = pool[torch.randint(0, pool.shape[0], (bs,), generator=g, device=DEV)]
        loss = lm_loss(model(Xb), Xb)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def probe(S, par, X, frac=0.5):
    """Linear (logistic) probe: read parity out of the sparse state. Train/test by sequence."""
    N, L, Hh = S.shape
    ntr = int(N * frac)
    Str, ytr = S[:ntr].reshape(-1, Hh), par[:ntr].reshape(-1)
    Ste, yte = S[ntr:].reshape(-1, Hh), par[ntr:].reshape(-1)
    lin = nn.Linear(Hh, 1).to(DEV)
    opt = torch.optim.Adam(lin.parameters(), 1e-2)
    for _ in range(400):
        loss = F.binary_cross_entropy_with_logits(lin(Str).squeeze(1), ytr.float())
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc_all = ((lin(Ste).squeeze(1) > 0).long() == yte).float().mean().item()
        gap = X[ntr:].reshape(-1) < NF                    # filler positions: bit only HELD
        acc_gap = ((lin(Ste[gap]).squeeze(1) > 0).long() == yte[gap]).float().mean().item()
    # best single unit (held-out): threshold at midpoint of class means
    m1, m0 = Str[ytr == 1].mean(0), Str[ytr == 0].mean(0)
    sgn, thr = torch.sign(m1 - m0), (m1 + m0) / 2
    accu = ((((Ste - thr) * sgn) > 0).long() == yte[:, None]).float().mean(0)
    return acc_all, acc_gap, accu.max().item(), int(accu.argmax())

# --------------------------------------------------------------------------- sweep
KS = [1, 2, 4, 8, 16, 32, 64]
Xev, parev = gen_pool(4000, 64, seed=999)                 # held-out, LONGER than train (48->64)
print(f"device={DEV}  task=flip-flop parity  H={H}  train L=48  eval L=64  eval seqs={Xev.shape[0]}")
print("\n  k   | capability | probe(all) | probe(GAP) | best-1-unit")
rows, model_k8 = {}, None
for k in KS:
    t = time.time(); m = train(k); cap = capability(m, Xev)
    with torch.no_grad():
        _, S = m(Xev, return_states=True)
    pa, pg, bu, bui = probe(S, parev, Xev)
    rows[k] = dict(cap=cap, probe_all=pa, probe_gap=pg, best_unit=bu, best_unit_idx=bui)
    print(f"  {k:<3} |   {cap:.3f}    |   {pa:.3f}    |   {pg:.3f}    |  {bu:.3f} (u{bui})   ({time.time()-t:.0f}s)")
    if k == 8:
        model_k8 = m
json.dump(rows, open(os.path.join(RUNS, "interior.json"), "w"), indent=2)

# ----------------------------------------------- qualitative: the "bit unit" in action
print("\n  the bit unit tracking parity on one held-out sequence (k=8):")
with torch.no_grad():
    _, S8 = model_k8(Xev[:1], return_states=True)
ui = rows[8]["best_unit_idx"]
seq, pr, uv = Xev[0].tolist(), parev[0].tolist(), S8[0, :, ui].tolist()
print("   token : " + "".join(SYM[t] for t in seq))
print("   parity: " + "".join(str(p) for p in pr))
print("   u%-4d : " % ui + "".join("+" if v > 0 else ("-" if v < 0 else "0") for v in uv))
print("   (F=flip Q=query 0/1=answer  ·,;=filler ; unit sign should track parity even on filler)")

# --------------------------------------------------------------------------- plot
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(KS, [rows[k]["cap"] for k in KS], "o-", label="capability (parity-query acc)")
    plt.plot(KS, [rows[k]["probe_all"] for k in KS], "s-", label="legibility: probe (all positions)")
    plt.plot(KS, [rows[k]["probe_gap"] for k in KS], "^-", label="legibility: probe at GAP (bit only held)")
    plt.plot(KS, [rows[k]["best_unit"] for k in KS], "d--", label="best single unit = 'the bit'")
    plt.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
    plt.xscale("log", base=2); plt.ylim(0.45, 1.02)
    plt.xlabel("k  (active units in the sparse carried state, H=64)"); plt.ylabel("accuracy")
    plt.title("Is a sparse working memory legible?  (flip-flop parity, no memorization)")
    plt.legend(fontsize=8, loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    out = os.path.join(RUNS, "interior.png"); plt.savefig(out, dpi=130)
    print(f"\n  plot -> {out}")
except Exception as e:
    print("\n  (plot skipped:", repr(e), ")")
