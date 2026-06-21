"""
interior2.py — Experiment 2b: does a WRITE-GATE rescue sparse memory?

Exp 2 (interior.py) showed a hard top-k on the carried state destroys memory at low k:
the active set churns each step, so the bit can't stay parked. Hypothesis (the handoff's
"legibility of the middle depends on the write policy"): give the recurrence a learned
HOLD-vs-OVERWRITE gate (Titans-style surprise-gated write) so the state is held between
flips and the active set stays stable. Then sparse memory should work — and stay legible —
at low k.

  gated update:  c_t = tanh(Wx x_t + Ws s_{t-1})              (candidate)
                 g_t = sigmoid(Wg[x_t, s_{t-1}])  (per-unit write gate; bias<0 = hold by default)
                 s_t = topk( (1-g_t)*s_{t-1} + g_t*c_t , k )   (sparse carried state)

Compares against Exp 2's vanilla numbers (runs/interior.json). Same task, seeds, probe.
"""
import os, sys, json, math, time
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.set_float32_matmul_precision("high")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "runs")
V, NF, FLIP, QUERY, ANS0, H, E = 7, 3, 3, 4, 5, 64, 16

def gen_pool(N, L, p_flip=0.20, p_query=0.12, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    X = torch.zeros(N, L, dtype=torch.long, device=DEV); par = torch.zeros_like(X)
    s = torch.zeros(N, dtype=torch.long, device=DEV); pending = torch.zeros(N, dtype=torch.bool, device=DEV)
    for t in range(L):
        r = torch.rand(N, generator=g, device=DEV)
        tok = torch.randint(0, NF, (N,), generator=g, device=DEV)
        flip = (~pending) & (r < p_flip); query = (~pending) & (r >= p_flip) & (r < p_flip + p_query)
        s = s ^ flip.long()
        tok = torch.where(flip, torch.full_like(tok, FLIP), tok)
        tok = torch.where(query, torch.full_like(tok, QUERY), tok)
        tok = torch.where(pending, ANS0 + s, tok)
        X[:, t] = tok; par[:, t] = s; pending = query
    return X, par

def topk_mask(z, k):
    if k >= z.shape[-1]: return z
    _, idx = z.abs().topk(k, dim=-1)
    return z * torch.zeros_like(z).scatter_(-1, idx, 1.0)

class GatedSparseRNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.emb = nn.Embedding(V, E)
        self.Wx = nn.Linear(E, H); self.Ws = nn.Linear(H, H, bias=False)
        self.Wg = nn.Linear(E + H, H); self.head = nn.Linear(H, V); self.k = k
        nn.init.constant_(self.Wg.bias, -2.0)            # default: hold (sigmoid(-2)=0.12)
    def forward(self, X, return_states=False):
        s = torch.zeros(X.shape[0], H, device=X.device); emb = self.emb(X); lg, st = [], []
        for t in range(X.shape[1]):
            e = emb[:, t]
            c = torch.tanh(self.Wx(e) + self.Ws(s))
            g = torch.sigmoid(self.Wg(torch.cat([e, s], -1)))
            s = topk_mask((1 - g) * s + g * c, self.k)
            lg.append(self.head(s));  st.append(s) if return_states else None
        return (torch.stack(lg, 1), torch.stack(st, 1)) if return_states else torch.stack(lg, 1)

@torch.no_grad()
def capability(model, X):
    pred = model(X)[:, :-1].argmax(-1); tgt = X[:, 1:]; m = tgt >= ANS0
    return (pred[m] == tgt[m]).float().mean().item()

def train(k, steps=3000, bs=256, lr=2e-3, Ltr=48, seed=0):
    torch.manual_seed(seed); m = GatedSparseRNN(k).to(DEV); opt = torch.optim.Adam(m.parameters(), lr)
    pool, _ = gen_pool(20000, Ltr, seed=100); g = torch.Generator(device=DEV).manual_seed(seed + 1)
    for _ in range(steps):
        Xb = pool[torch.randint(0, pool.shape[0], (bs,), generator=g, device=DEV)]
        loss = F.cross_entropy(m(Xb)[:, :-1].reshape(-1, V), Xb[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m

def probe(S, par, X, frac=0.5):
    N, L, Hh = S.shape; ntr = int(N * frac)
    Str, ytr = S[:ntr].reshape(-1, Hh), par[:ntr].reshape(-1)
    Ste, yte = S[ntr:].reshape(-1, Hh), par[ntr:].reshape(-1)
    lin = nn.Linear(Hh, 1).to(DEV); opt = torch.optim.Adam(lin.parameters(), 1e-2)
    for _ in range(400):
        loss = F.binary_cross_entropy_with_logits(lin(Str).squeeze(1), ytr.float())
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc_all = ((lin(Ste).squeeze(1) > 0).long() == yte).float().mean().item()
        gap = X[ntr:].reshape(-1) < NF
        acc_gap = ((lin(Ste[gap]).squeeze(1) > 0).long() == yte[gap]).float().mean().item()
    m1, m0 = Str[ytr == 1].mean(0), Str[ytr == 0].mean(0); sgn = torch.sign(m1 - m0); thr = (m1 + m0) / 2
    accu = ((((Ste - thr) * sgn) > 0).long() == yte[:, None]).float().mean(0)
    return acc_all, acc_gap, accu.max().item(), int(accu.argmax())

KS = [1, 2, 4, 8, 16, 32, 64]
Xev, parev = gen_pool(4000, 64, seed=999)
print(f"device={DEV}  Exp 2b: write-gated sparse memory  H={H}")
print("\n  k   | capability | probe(GAP) | best-1-unit   (vanilla cap from Exp 2)")
van = json.load(open(os.path.join(RUNS, "interior.json")))
rows = {}
for k in KS:
    t = time.time(); m = train(k); cap = capability(m, Xev)
    with torch.no_grad():
        _, S = m(Xev, return_states=True)
    pa, pg, bu, bui = probe(S, parev, Xev)
    rows[k] = dict(cap=cap, probe_all=pa, probe_gap=pg, best_unit=bu, best_unit_idx=bui)
    print(f"  {k:<3} |   {cap:.3f}    |   {pg:.3f}    |  {bu:.3f} (u{bui})     (vanilla {van[str(k)]['cap']:.3f})   ({time.time()-t:.0f}s)")
json.dump(rows, open(os.path.join(RUNS, "interior_gated.json"), "w"), indent=2)

# comparison SVG: vanilla vs gated (capability + gap-legibility) over k
def svg(path):
    W, Hh, ml, mr, mt, mb = 620, 380, 60, 196, 34, 48
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    Xc = lambda k: x0 + math.log2(k) / 6 * (x1 - x0)
    Yc = lambda a: y0 + (a - 0.45) / (1.02 - 0.45) * (y1 - y0)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="#1A1F4A"/>',
         f'<text x="{(x0+x1)/2}" y="20" fill="#F4F0E8" font-size="13" text-anchor="middle">Write-gate rescues sparse memory  (flip-flop parity)</text>']
    for a in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        y = Yc(a); p += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#2c2f5e"/>',
                         f'<text x="{x0-8}" y="{y+4:.1f}" fill="#8784b3" font-size="11" text-anchor="end">{a:.1f}</text>']
    for k in KS:
        p.append(f'<text x="{Xc(k):.1f}" y="{y0+18}" fill="#8784b3" font-size="11" text-anchor="middle">{k}</text>')
    p += [f'<text x="{(x0+x1)/2}" y="{Hh-8}" fill="#B8B3D6" font-size="12" text-anchor="middle">k  (active units in carried state, H=64)</text>',
          f'<line x1="{x0}" y1="{Yc(0.5):.1f}" x2="{x1}" y2="{Yc(0.5):.1f}" stroke="#666" stroke-dasharray="4 3"/>']
    series = [(van, "cap", "#3f6f8f", "vanilla: capability", " stroke-dasharray='5 3'"),
              (van, "probe_gap", "#7a6a3a", "vanilla: legibility(gap)", " stroke-dasharray='5 3'"),
              (rows, "cap", "#6FD6C9", "gated: capability", ""),
              (rows, "probe_gap", "#F5D77A", "gated: legibility(gap)", "")]
    ly = mt + 14
    for src, key, lab, col, dash in series:
        get = lambda k: (src[str(k)] if isinstance(list(src.keys())[0], str) else src[k])[key]
        pts = " ".join(f"{Xc(k):.1f},{Yc(get(k)):.1f}" for k in KS)
        p.append(f"<polyline points='{pts}' fill='none' stroke='{lab}' stroke-width='2.5'{dash}/>")
        for k in KS:
            p.append(f"<circle cx='{Xc(k):.1f}' cy='{Yc(get(k)):.1f}' r='3' fill='{lab}'/>")
        p += [f"<rect x='{x1+14}' y='{ly-9}' width='12' height='12' fill='{lab}'/>",
              f"<text x='{x1+30}' y='{ly+1}' fill='#F4F0E8' font-size='11'>{col}</text>"]; ly += 21
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg(os.path.join(RUNS, "interior_gated.svg"))
print("\n  wrote runs/interior_gated.json and runs/interior_gated.svg")
