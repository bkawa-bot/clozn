"""
structured_superposition.py — Experiment 7 (the exploratory one): torus or tangle?

Single structured latents become CIRCLES (Exp 3, 6). Unstructured features superpose into
POLYGONS (Exp 4, Anthropic). Open question for us: when many STRUCTURED latents are crammed into
too few dimensions, do they stay a clean PRODUCT of circles (a torus — crowded but each still
readable) or TANGLE (individually unreadable)? And does crowding break capability, legibility, or
both — and in what order?

Task: K cyclic counters mod m, in a width-H GRU. STEP_k advances counter k; QUERY_k must be answered
with counter k's value. Clean storage of K circles needs ~2K dims; with H=32 the model is forced to
superpose once K > ~16. Sweep K and measure:
  capability  = query-answer accuracy (can it USE counter k?)
  legibility  = per-counter linear-probe accuracy, normalized above base rate (is counter k READABLE?)
  interference = mean |cosine| between different counters' readout directions (are subspaces shared?)
Geometry (crowded K): does each counter still trace a circle in its own plane?

I genuinely cannot predict the result. Outputs: runs/superpos_curve.svg, runs/superpos_geom.svg.
"""
import os, sys, json, math
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.set_float32_matmul_precision("high")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "runs")
M, Hd, E, STEPS, BS, LR = 4, 32, 32, 4000, 256, 2e-3

def gen(N, K, g, p_step=0.6, p_query=0.2):
    L = max(96, 6 * K)
    ANSB, STEPB, QB = 1, 1 + M, 1 + M + K; V = 1 + M + 2 * K
    c = torch.zeros(N, K, dtype=torch.long, device=DEV)
    pend = torch.zeros(N, dtype=torch.bool, device=DEV); pbit = torch.zeros(N, dtype=torch.long, device=DEV)
    X = torch.zeros(N, L, dtype=torch.long, device=DEV); C = torch.zeros(N, L, K, dtype=torch.long, device=DEV)
    AM = torch.zeros(N, L, dtype=torch.bool, device=DEV)
    for t in range(L):
        r = torch.rand(N, generator=g, device=DEV); k = torch.randint(0, K, (N,), generator=g, device=DEV)
        np_ = ~pend; step = np_ & (r < p_step); query = np_ & (r >= p_step) & (r < p_step + p_query)
        si = step.nonzero(as_tuple=True)[0]
        c[si, k[si]] = (c[si, k[si]] + 1) % M
        tok = torch.zeros(N, dtype=torch.long, device=DEV)
        tok[step] = STEPB + k[step]; tok[query] = QB + k[query]
        pi = pend.nonzero(as_tuple=True)[0]; tok[pi] = ANSB + c[pi, pbit[pi]]
        X[:, t] = tok; C[:, t] = c; AM[:, t] = pend
        pend = query; pbit = torch.where(query, k, pbit)
    return X, C, AM, V

class GRUNet(nn.Module):
    def __init__(self, V):
        super().__init__(); self.emb = nn.Embedding(V, E); self.gru = nn.GRU(E, Hd, batch_first=True); self.head = nn.Linear(Hd, V)
    def forward(self, x):
        h, _ = self.gru(self.emb(x)); return self.head(h), h

def train(K, seed=0):
    torch.manual_seed(seed); g = torch.Generator(device=DEV).manual_seed(seed + 1)
    _, _, _, V = gen(2, K, g); model = GRUNet(V).to(DEV); opt = torch.optim.Adam(model.parameters(), LR)
    for _ in range(STEPS):
        X, C, AM, _ = gen(BS, K, g)
        logits, _ = model(X)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), X[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model

@torch.no_grad()
def eval_batch(model, K, g):
    X, C, AM, V = gen(3000, K, g); logits, h = model(X)
    pred = logits[:, :-1].argmax(-1); msk = AM[:, 1:]
    cap = (pred[msk] == X[:, 1:][msk]).float().mean().item()
    return X, C, AM, h, cap

def legibility(h, C, K):
    N, L, _ = h.shape; flatH = h.reshape(-1, Hd); flatC = C.reshape(-1, K)
    ntr = flatH.shape[0] // 2
    probe = nn.Linear(Hd, K * M).to(DEV); opt = torch.optim.Adam(probe.parameters(), 1e-2)
    for _ in range(500):
        out = probe(flatH[:ntr]).reshape(-1, K, M)
        loss = F.cross_entropy(out.reshape(-1, M), flatC[:ntr].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pr = probe(flatH[ntr:]).reshape(-1, K, M).argmax(-1)          # [Nte, K]
        ct = flatC[ntr:]
        skills = []
        for k in range(K):
            acc = (pr[:, k] == ct[:, k]).float().mean().item()
            base = torch.bincount(ct[:, k], minlength=M).max().item() / ct.shape[0]
            skills.append((acc - base) / max(1e-6, 1 - base))
    return sum(skills) / len(skills)

@torch.no_grad()
def readout_dirs(h, C, K):
    # per counter k: least-squares directions predicting (cos,sin) of its angle -> its 2D plane
    flatH = h.reshape(-1, Hd); ang = 2 * math.pi * C.reshape(-1, K).float() / M
    Hb = torch.cat([flatH, torch.ones(flatH.shape[0], 1, device=DEV)], 1)
    dirs = []
    for k in range(K):
        tgt = torch.stack([torch.cos(ang[:, k]), torch.sin(ang[:, k])], 1)
        W = torch.linalg.lstsq(Hb, tgt).solution[:Hd]                 # [Hd,2]
        dirs.append(W / (W.norm(dim=0, keepdim=True) + 1e-9))
    D = torch.cat(dirs, 1)                                            # [Hd, 2K]
    G = (D.T @ D).abs()
    block = torch.zeros_like(G)                                       # mask same-counter pairs
    for k in range(K):
        block[2 * k:2 * k + 2, 2 * k:2 * k + 2] = 1
    off = G * (1 - block); interf = off.sum().item() / max(1, (off > 0).sum().item())
    return dirs, interf

KS = [2, 4, 8, 16, 24]
g = torch.Generator(device=DEV).manual_seed(99)
print(f"device={DEV}  structured superposition  m={M}  H={Hd}  (capacity ~{Hd//2} counters)")
print("\n   K  | capability | legibility | interference")
rows = {}; geomK = 24; geom = None
for K in KS:
    model = train(K); X, C, AM, h, cap = eval_batch(model, K, g)
    leg = legibility(h, C, K); dirs, interf = readout_dirs(h, C, K)
    rows[K] = dict(cap=cap, leg=leg, interf=interf)
    print(f"  {K:<3} |   {cap:.3f}    |   {leg:.3f}    |    {interf:.3f}")
    if K == geomK:
        geom = (h, C, dirs)
json.dump(rows, open(os.path.join(RUNS, "superpos.json"), "w"), indent=2)

# ---- curve SVG ----
def svg_curve(path):
    W, Hh, ml, mr, mt, mb = 600, 340, 56, 150, 36, 46; x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    lx = math.log2(KS[-1]); Xc = lambda K: x0 + math.log2(K) / lx * (x1 - x0); Yc = lambda v: y0 - v * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="#1A1F4A"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="#F4F0E8" font-size="13" text-anchor="middle">Cramming K cyclic latents into width {Hd}: torus or tangle?</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="#2c2f5e"/>',
                         f'<text x="{x0-8}" y="{Y+4:.1f}" fill="#8784b3" font-size="10" text-anchor="end">{v:g}</text>']
    for K in KS:
        p.append(f'<text x="{Xc(K):.1f}" y="{y0+18}" fill="#8784b3" font-size="11" text-anchor="middle">{K}</text>')
    cap_thr = Xc(Hd // 2)
    p += [f'<line x1="{cap_thr:.1f}" y1="{y1}" x2="{cap_thr:.1f}" y2="{y0}" stroke="#B59DD8" stroke-dasharray="3 3"/>',
          f'<text x="{cap_thr+4:.1f}" y="{y1+12}" fill="#B59DD8" font-size="9">capacity (2K=H)</text>',
          f'<text x="{(x0+x1)/2}" y="{Hh-8}" fill="#B8B3D6" font-size="11" text-anchor="middle">K = number of cyclic latents crammed in</text>']
    for key, col, lab in [("cap", "#6FD6C9", "capability"), ("leg", "#F5D77A", "legibility (probe skill)"),
                          ("interf", "#FF8FB3", "interference (subspace overlap)")]:
        pts = " ".join(f"{Xc(K):.1f},{Yc(rows[K][key]):.1f}" for K in KS)
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.3"/>')
        for K in KS:
            p.append(f'<circle cx="{Xc(K):.1f}" cy="{Yc(rows[K][key]):.1f}" r="3" fill="{col}"/>')
    ly = mt + 10
    for col, lab in [("#6FD6C9", "capability (use)"), ("#F5D77A", "legibility (read)"), ("#FF8FB3", "interference")]:
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="#F4F0E8" font-size="10.5">{lab}</text>']; ly += 20
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg_curve(os.path.join(RUNS, "superpos_curve.svg"))

# ---- geometry small-multiples (crowded K): each counter's circle in its own plane ----
def svg_geom(path):
    h, C, dirs = geom; flatH = h.reshape(-1, Hd); flatC = C.reshape(-1, geomK)
    nshow = 8; per = 150; cols = 4; rows_ = (nshow + cols - 1) // cols
    W = cols * per; Hh = rows_ * per + 40
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="#1A1F4A"/>',
         f'<text x="{W/2}" y="22" fill="#F4F0E8" font-size="13" text-anchor="middle">Crowded ({geomK} latents in width {Hd}): does each counter keep its circle?</text>']
    for s in range(nshow):
        k = s; W2 = dirs[k]                                           # [Hd,2]
        proj = flatH @ W2                                            # [Npos,2]
        pts = []
        for v in range(M):
            mask = flatC[:, k] == v
            pts.append(proj[mask].mean(0).cpu() if mask.any() else torch.zeros(2))
        P = torch.stack(pts); P = P - P.mean(0); P = P / (P.abs().max() + 1e-9)
        cx = (s % cols) * per + per / 2; cy = (s // cols) * per + per / 2 + 30; R = per * 0.32
        Xf = lambda v: cx + P[v, 0].item() * R; Yf = lambda v: cy - P[v, 1].item() * R
        ring = " ".join(f"{Xf(v):.1f},{Yf(v):.1f}" for v in range(M)) + f" {Xf(0):.1f},{Yf(0):.1f}"
        p.append(f'<polyline points="{ring}" fill="none" stroke="#3a3f6e" stroke-width="1"/>')
        for v in range(M):
            p.append(f'<circle cx="{Xf(v):.1f}" cy="{Yf(v):.1f}" r="5" fill="hsl({int(360*v/M)},70%,64%)"/>')
        p.append(f'<text x="{cx:.1f}" y="{cy - R - 10:.1f}" fill="#8784b3" font-size="10" text-anchor="middle">counter {k}</text>')
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg_geom(os.path.join(RUNS, "superpos_geom.svg"))
print("\n  wrote runs/superpos_curve.svg and runs/superpos_geom.svg")
