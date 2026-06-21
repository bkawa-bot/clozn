"""
interior_viz.py — render Experiment 2 as SVG (matplotlib's compiled backend is blocked by
this machine's Application Control policy, so we emit SVG directly: no compiled deps).
  - runs/interior_curve.svg : capability vs legibility vs k (from runs/interior.json)
  - runs/interior_trace.svg : one held-out sequence; does the 'bit unit' track parity?
                              dense k=64 (success) vs sparse k=8 (failure), maiko-themed.
Retrains k=8 and k=64 to recover their bit-units (seeds match interior.py).
"""
import os, sys, json, math
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "runs")
V, NF, FLIP, QUERY, ANS0, H, E = 7, 3, 3, 4, 5, 64, 16
SYM = {0: ".", 1: ",", 2: ";", 3: "F", 4: "Q", 5: "0", 6: "1"}

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

class SparseRNN(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.emb = nn.Embedding(V, E); self.Wx = nn.Linear(E, H)
        self.Ws = nn.Linear(H, H, bias=False); self.head = nn.Linear(H, V); self.k = k
    def forward(self, X, return_states=False):
        s = torch.zeros(X.shape[0], H, device=X.device); emb = self.emb(X); lg, st = [], []
        for t in range(X.shape[1]):
            s = topk_mask(torch.tanh(self.Wx(emb[:, t]) + self.Ws(s)), self.k)
            lg.append(self.head(s));  st.append(s) if return_states else None
        return (torch.stack(lg, 1), torch.stack(st, 1)) if return_states else torch.stack(lg, 1)

def train(k, steps=3000, bs=256, lr=2e-3, Ltr=48, seed=0):
    torch.manual_seed(seed); m = SparseRNN(k).to(DEV); opt = torch.optim.Adam(m.parameters(), lr)
    pool, _ = gen_pool(20000, Ltr, seed=100); g = torch.Generator(device=DEV).manual_seed(seed + 1)
    for _ in range(steps):
        Xb = pool[torch.randint(0, pool.shape[0], (bs,), generator=g, device=DEV)]
        loss = F.cross_entropy(m(Xb)[:, :-1].reshape(-1, V), Xb[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m

def best_unit(S, par):
    Sf, y = S.reshape(-1, H), par.reshape(-1)
    m1, m0 = Sf[y == 1].mean(0), Sf[y == 0].mean(0); sgn = torch.sign(m1 - m0); thr = (m1 + m0) / 2
    acc = ((((Sf - thr) * sgn) > 0).long() == y[:, None]).float().mean(0)
    u = int(acc.argmax()); return u, float(acc[u]), float(sgn[u])

# ---------------------------------------------------------------- curve SVG
rows = json.load(open(os.path.join(RUNS, "interior.json")))
KS = [1, 2, 4, 8, 16, 32, 64]
def svg_curve(path):
    W, Hh, ml, mr, mt, mb = 600, 380, 60, 168, 34, 48
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    Xc = lambda k: x0 + math.log2(k) / 6 * (x1 - x0)
    Yc = lambda a: y0 + (a - 0.45) / (1.02 - 0.45) * (y1 - y0)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="#1A1F4A"/>',
         f'<text x="{(x0+x1)/2}" y="20" fill="#F4F0E8" font-size="13" text-anchor="middle">Is a sparse working memory legible?  (flip-flop parity)</text>']
    for a in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        y = Yc(a); p += [f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="#2c2f5e"/>',
                         f'<text x="{x0-8}" y="{y+4:.1f}" fill="#8784b3" font-size="11" text-anchor="end">{a:.1f}</text>']
    for k in KS:
        p.append(f'<text x="{Xc(k):.1f}" y="{y0+18}" fill="#8784b3" font-size="11" text-anchor="middle">{k}</text>')
    p += [f'<text x="{(x0+x1)/2}" y="{Hh-8}" fill="#B8B3D6" font-size="12" text-anchor="middle">k  (active units in carried state, H=64)</text>',
          f'<line x1="{x0}" y1="{Yc(0.5):.1f}" x2="{x1}" y2="{Yc(0.5):.1f}" stroke="#666" stroke-dasharray="4 3"/>']
    series = [("cap", "#6FD6C9", "capability"), ("probe_all", "#F5D77A", "probe (all pos)"),
              ("probe_gap", "#FFB38A", "probe (held/gap)"), ("best_unit", "#FF8FB3", "best single unit")]
    ly = mt + 14
    for key, col, lab in series:
        pts = " ".join(f"{Xc(k):.1f},{Yc(rows[str(k)][key]):.1f}" for k in KS)
        dash = ' stroke-dasharray="6 3"' if key == "best_unit" else ''
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"{dash}/>')
        for k in KS:
            p.append(f'<circle cx="{Xc(k):.1f}" cy="{Yc(rows[str(k)][key]):.1f}" r="3.2" fill="{col}"/>')
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="#F4F0E8" font-size="11">{lab}</text>']; ly += 21
    p.append(f'<text x="{x1+14}" y="{ly+1}" fill="#8784b3" font-size="10">- - chance 0.5</text>')
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg_curve(os.path.join(RUNS, "interior_curve.svg"))

# ---------------------------------------------------------------- trace SVG (success vs fail)
Xev, parev = gen_pool(4000, 64, seed=999)
m64 = train(64); m8 = train(8)
with torch.no_grad():
    _, S64 = m64(Xev, return_states=True); _, S8 = m8(Xev, return_states=True)
u64, a64, g64 = best_unit(S64, parev); u8, a8, g8 = best_unit(S8, parev)
seq, par = Xev[0].tolist(), parev[0].tolist()
sg64 = (S64[0, :, u64] * g64).tolist(); sg8 = (S8[0, :, u8] * g8).tolist()  # parity-aligned
print(f"k=64 bit-unit u{u64} acc={a64:.3f} ; k=8 bit-unit u{u8} acc={a8:.3f}")

def svg_trace(path):
    L = len(seq); cw, ch, ml, mt = 13, 21, 150, 34; W = ml + L * cw + 14; Hh = mt + 4 * ch + 24
    def row(y, label, lab_col):
        return f'<text x="{ml-10}" y="{y+15}" fill="{lab_col}" font-size="11" text-anchor="end">{label}</text>'
    def cell(i, y, fill, txt, tcol):
        x = ml + i * cw
        return (f'<rect x="{x}" y="{y}" width="{cw-1}" height="{ch-1}" fill="{fill}"/>'
                f'<text x="{x+cw/2:.1f}" y="{y+15}" font-size="10.5" text-anchor="middle" fill="{tcol}">{txt}</text>')
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="#1A1F4A"/>',
         f'<text x="10" y="20" fill="#F4F0E8" font-size="12">the bit-unit tracking the hidden parity, one held-out sequence</text>']
    y = mt
    p.append(row(y, "tokens", "#B8B3D6"))
    for i, t in enumerate(seq):
        col = {3: "#F5D77A", 4: "#B59DD8", 5: "#FFB38A", 6: "#FFB38A"}.get(t, "#2c2f5e")
        tc = "#211a33" if t >= 3 else "#8784b3"
        p.append(cell(i, y, col, SYM[t], tc))
    y += ch
    p.append(row(y, "true parity", "#6FD6C9"))
    for i, b in enumerate(par):
        p.append(cell(i, y, "#6FD6C9" if b else "#23274f", str(b), "#0d1117" if b else "#5b6090"))
    y += ch
    p.append(row(y, f"k=64  u{u64}", "#6FD6C9"))
    for i, v in enumerate(sg64):
        f = "#6FD6C9" if v > 1e-6 else ("#FF8FB3" if v < -1e-6 else "#3a3f6e")
        p.append(cell(i, y, f, "+" if v > 1e-6 else ("-" if v < -1e-6 else ""), "#0d1117"))
    y += ch
    p.append(row(y, f"k=8  u{u8}", "#FF8FB3"))
    for i, v in enumerate(sg8):
        f = "#6FD6C9" if v > 1e-6 else ("#FF8FB3" if v < -1e-6 else "#3a3f6e")
        p.append(cell(i, y, f, "+" if v > 1e-6 else ("-" if v < -1e-6 else ""), "#0d1117"))
    p.append(f'<text x="10" y="{Hh-7}" fill="#8784b3" font-size="10">teal=+ (parity-1 side) · pink=- · grey=masked-out (not in top-k).  '
             f'k=64 row should mirror true-parity; k=8 row is mostly masked = bit not held.</text>')
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg_trace(os.path.join(RUNS, "interior_trace.svg"))
print("wrote runs/interior_curve.svg and runs/interior_trace.svg")
