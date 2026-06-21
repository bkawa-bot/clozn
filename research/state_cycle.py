"""
state_cycle.py — Experiment 6 (the handoff's rung #2, done properly):
is a NECESSARY, TRACKED latent linearly readable from a recurrent internal state?

Task: count steps on a cycle of size m. A `STEP` token advances a hidden position c := (c+1) mod m;
a `QUERY` token must be answered with the current position c. So c is a genuine internal latent the
model must MAINTAIN (not recall a stored item — just carry an evolving value). Parity was the m=2
case; here we sweep m = 2,4,8,16,32 (the latent gets richer).

We measure, per m:
  capability  = answer accuracy at queries (chance = 1/m)
  legibility  = a LINEAR probe decoding the true position c from the GRU's hidden state (chance 1/m)
If legibility stays high, sparsity/structure bought a MONITOR (the safety payoff, with ground truth).
Bonus: for m=16 we look at the geometry of the hidden state per position — does a recurrent model
tracking a cyclic counter put the positions on a CIRCLE, like the grokking model did?

Outputs: runs/cycle_curve.svg, runs/cycle_circle.svg.  (matplotlib blocked -> SVG.)
"""
import os, sys, json, math
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.set_float32_matmul_precision("high")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "runs")
E, Hd, L, STEPS, BS, LR = 32, 64, 64, 6000, 256, 2e-3
FILLER, STEP, QUERY, ANS = 0, 1, 2, 3                    # ANS+c are the m answer tokens

def gen(N, m, g, p_step=0.45, p_query=0.15):
    V = ANS + m
    c = torch.zeros(N, dtype=torch.long, device=DEV)
    pend = torch.zeros(N, dtype=torch.bool, device=DEV)
    X = torch.zeros(N, L, dtype=torch.long, device=DEV); C = torch.zeros_like(X); AM = torch.zeros(N, L, dtype=torch.bool, device=DEV)
    for t in range(L):
        r = torch.rand(N, generator=g, device=DEV); np_ = ~pend
        step = np_ & (r < p_step); query = np_ & (r >= p_step) & (r < p_step + p_query)
        c = torch.where(step, (c + 1) % m, c)
        tok = torch.zeros(N, dtype=torch.long, device=DEV)
        tok = torch.where(step, torch.full_like(tok, STEP), tok)
        tok = torch.where(query, torch.full_like(tok, QUERY), tok)
        tok = torch.where(pend, ANS + c, tok)
        X[:, t] = tok; C[:, t] = c; AM[:, t] = pend; pend = query
    return X, C, AM, V

class GRUNet(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.emb = nn.Embedding(V, E); self.gru = nn.GRU(E, Hd, batch_first=True); self.head = nn.Linear(Hd, V)
    def forward(self, x):
        h, _ = self.gru(self.emb(x)); return self.head(h), h

def train(m, seed=0):
    torch.manual_seed(seed); V = ANS + m; model = GRUNet(V).to(DEV)
    opt = torch.optim.Adam(model.parameters(), LR); g = torch.Generator(device=DEV).manual_seed(seed + 1)
    for _ in range(STEPS):
        X, C, AM, _ = gen(BS, m, g)
        logits, _ = model(X)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def capability(model, m, g):
    X, C, AM, V = gen(4000, m, g)
    pred = model(X)[0][:, :-1].argmax(-1); tgt = X[:, 1:]; msk = AM[:, 1:]
    return (pred[msk] == tgt[msk]).float().mean().item()

def legibility(model, m, g):
    X, C, AM, V = gen(4000, m, g)
    with torch.no_grad():
        _, h = model(X)
    N = X.shape[0]; ntr = N // 2
    Htr, Ctr = h[:ntr].reshape(-1, Hd), C[:ntr].reshape(-1)
    Hte, Cte = h[ntr:].reshape(-1, Hd), C[ntr:].reshape(-1)
    probe = nn.Linear(Hd, m).to(DEV); opt = torch.optim.Adam(probe.parameters(), 1e-2)
    for _ in range(500):
        loss = F.cross_entropy(probe(Htr), Ctr); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (probe(Hte).argmax(1) == Cte).float().mean().item()
    return acc, h, C

MS = [2, 4, 8, 16, 32]
g = torch.Generator(device=DEV).manual_seed(123)
print(f"device={DEV}  tracked cyclic counter  H={Hd}")
print("\n   m   | capability | legibility | chance(1/m)")
rows = {}; h16 = C16 = None
for m in MS:
    model = train(m); cap = capability(model, m, g); leg, h, C = legibility(model, m, g)
    rows[m] = dict(cap=cap, leg=leg, chance=1.0 / m)
    print(f"  {m:<4} |   {cap:.3f}    |   {leg:.3f}    |   {1.0/m:.3f}")
    if m == 16:
        h16, C16 = h, C
json.dump(rows, open(os.path.join(RUNS, "cycle.json"), "w"), indent=2)

# ---- curve SVG: capability + legibility vs m ----
def svg_curve(path):
    W, Hh, ml, mr, mt, mb = 600, 340, 56, 150, 36, 46; x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    lx = math.log2(MS[-1])
    Xc = lambda m: x0 + math.log2(m) / lx * (x1 - x0); Yc = lambda v: y0 - v * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="#1A1F4A"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="#F4F0E8" font-size="13" text-anchor="middle">Is a tracked latent readable? (cyclic counter mod m)</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="#2c2f5e"/>',
                         f'<text x="{x0-8}" y="{Y+4:.1f}" fill="#8784b3" font-size="10" text-anchor="end">{v:g}</text>']
    for m in MS:
        p.append(f'<text x="{Xc(m):.1f}" y="{y0+18}" fill="#8784b3" font-size="11" text-anchor="middle">{m}</text>')
    p.append(f'<text x="{(x0+x1)/2}" y="{Hh-8}" fill="#B8B3D6" font-size="11" text-anchor="middle">m = cycle size (states the latent can take)</text>')
    for key, col, lab in [("cap", "#6FD6C9", "capability (answer acc)"), ("leg", "#F5D77A", "legibility (linear probe)"),
                          ("chance", "#FF8FB3", "chance = 1/m")]:
        pts = " ".join(f"{Xc(m):.1f},{Yc(rows[m][key]):.1f}" for m in MS)
        dash = ' stroke-dasharray="4 3"' if key == "chance" else ''
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.3"{dash}/>')
        for m in MS:
            p.append(f'<circle cx="{Xc(m):.1f}" cy="{Yc(rows[m][key]):.1f}" r="3" fill="{col}"/>')
    ly = mt + 10
    for col, lab in [("#6FD6C9", "capability"), ("#F5D77A", "legibility (probe)"), ("#FF8FB3", "chance 1/m")]:
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="#F4F0E8" font-size="11">{lab}</text>']; ly += 21
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg_curve(os.path.join(RUNS, "cycle_curve.svg"))

# ---- geometry SVG (m=16): mean hidden state per position -> circle? ----
def svg_circle(path, m=16):
    means = torch.stack([h16.reshape(-1, Hd)[C16.reshape(-1) == c].mean(0) for c in range(m)])  # [m,Hd]
    e = means - means.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(e, full_matrices=False)
    pc = (e @ Vh[:2].T).cpu(); pc = pc / pc.abs().max()
    W = 440; R = 165; cx = W / 2; cy = W / 2 + 6
    Xf = lambda i: cx + pc[i, 0].item() * R; Yf = lambda i: cy - pc[i, 1].item() * R
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{W+30}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{W+30}" fill="#1A1F4A"/>',
         f'<text x="{W/2}" y="22" fill="#F4F0E8" font-size="13" text-anchor="middle">Hidden state per counter position (m={m}) — a tracked circle?</text>']
    order = " ".join(f"{Xf(i):.1f},{Yf(i):.1f}" for i in range(m)) + f" {Xf(0):.1f},{Yf(0):.1f}"
    p.append(f'<polyline points="{order}" fill="none" stroke="#3a3f6e" stroke-width="1.2"/>')
    for i in range(m):
        p.append(f'<circle cx="{Xf(i):.1f}" cy="{Yf(i):.1f}" r="6" fill="hsl({int(360*i/m)},70%,64%)"/>')
        p.append(f'<text x="{Xf(i):.1f}" y="{Yf(i)-9:.1f}" fill="#F4F0E8" font-size="9" text-anchor="middle">{i}</text>')
    p.append(f'<text x="{W/2}" y="{W+24}" fill="#8784b3" font-size="10" text-anchor="middle">line connects positions 0,1,2,... in order</text>')
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg_circle(os.path.join(RUNS, "cycle_circle.svg"))
print("\n  wrote runs/cycle_curve.svg and runs/cycle_circle.svg")
