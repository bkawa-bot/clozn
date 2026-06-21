"""
superpos_static.py — Experiment 7b: torus or tangle, ISOLATED (no maintenance confound).

Exp 7 (GRU) was confounded: capability collapsed below capacity, so it measured the GRU's
maintenance limit, not representation. Here we drop recurrence entirely — a static autoencoder —
so the ONLY thing that can limit it is representational capacity.

(a) SCALAR CONTROL (validate the machinery reproduces Anthropic superposition): n scalar features,
    width-2 bottleneck, importance decay; features-represented should RISE with sparsity.
(b) CYCLIC MAIN: K cyclic variables, m values each (one-hot per value), crammed into a tight
    bottleneck H. Sweep sparsity. Measure per-variable legibility = when a variable is active, is
    its value recoverable from the reconstruction? And look at the geometry (do values form a circle?).

I read the cyclic result ONLY if the scalar control behaves. Outputs: runs/superpos_static_*.svg
"""
import os, sys, json, math
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.set_float32_matmul_precision("high")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "runs")

# ---------------------------------------------------- (a) scalar control (Anthropic)
def scalar_control():
    n, Hh, STEPS, BS = 8, 2, 4000, 2048
    IMP = 0.8 ** torch.arange(n, dtype=torch.float32, device=DEV)
    print("  scalar control (Anthropic superposition): features represented vs sparsity")
    out = {}
    for S in [0.0, 0.7, 0.9, 0.97]:
        torch.manual_seed(0)
        W = nn.Parameter(torch.randn(Hh, n, device=DEV) * 0.1); b = nn.Parameter(torch.zeros(n, device=DEV))
        opt = torch.optim.Adam([W, b], 1e-2); g = torch.Generator(device=DEV).manual_seed(1)
        for _ in range(STEPS):
            val = torch.rand(BS, n, generator=g, device=DEV)
            x = val * (torch.rand(BS, n, generator=g, device=DEV) > S).float()
            xh = F.relu((x @ W.T) @ W + b)
            loss = (IMP * (xh - x) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        rep = int((W.detach().norm(dim=0) > 0.5).sum()); out[S] = rep
        print(f"    sparsity {S:<5} -> {rep}/{n} represented")
    ok = out[0.0] <= 3 and out[0.97] > out[0.0]
    print(f"  control {'OK (superposition reproduced: rises with sparsity)' if ok else 'FAILED — do not trust cyclic run'}")
    return ok

# ---------------------------------------------------- (b) cyclic main
K, M, Hd, STEPS, BS = 12, 6, 8, 6000, 1024
def gen_cyclic(N, S, g):
    active = (torch.rand(N, K, generator=g, device=DEV) > S)
    vals = torch.randint(0, M, (N, K), generator=g, device=DEV)
    oneh = active.float().unsqueeze(-1) * F.one_hot(vals, M).float()
    return oneh.reshape(N, K * M), vals, active

def train_cyclic(S, seed=0):
    torch.manual_seed(seed)
    W = nn.Parameter(torch.randn(Hd, K * M, device=DEV) * 0.1); b = nn.Parameter(torch.zeros(K * M, device=DEV))
    opt = torch.optim.Adam([W, b], 5e-3); g = torch.Generator(device=DEV).manual_seed(seed + 1)
    for _ in range(STEPS):
        x, _, _ = gen_cyclic(BS, S, g)
        xh = F.relu((x @ W.T) @ W + b)
        loss = ((xh - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return W.detach(), b.detach()

@torch.no_grad()
def legib(W, b, S, g):
    x, vals, active = gen_cyclic(8000, S, g)
    xh = F.relu((x @ W.T) @ W + b).reshape(-1, K, M)
    pred = xh.argmax(-1)
    corr = (pred == vals) & active
    return corr.sum().item() / max(1, active.sum().item())

print(f"device={DEV}  static superposition  (cyclic: K={K} vars, m={M}, bottleneck H={Hd})\n")
ctrl_ok = scalar_control()

print("\n  cyclic: per-variable legibility (value recoverable when active) vs sparsity")
g = torch.Generator(device=DEV).manual_seed(7)
SPS = [0.0, 0.5, 0.8, 0.95]; rows = {}; Wsave = None
for S in SPS:
    W, b = train_cyclic(S); acc = legib(W, b, S, g); rows[S] = acc
    print(f"    sparsity {S:<5} -> legibility {acc:.3f}  (chance {1.0/M:.3f})")
    if S == 0.95:
        Wsave = (W, b)
json.dump({"control_ok": ctrl_ok, "cyclic_legibility": rows}, open(os.path.join(RUNS, "superpos_static.json"), "w"), indent=2)

# ---- curve SVG ----
def svg_curve(path):
    W_, Hh, ml, mr, mt, mb = 560, 320, 56, 40, 36, 46; x0, x1, y0, y1 = ml, W_ - mr, Hh - mb, mt
    Xc = lambda s: x0 + s * (x1 - x0); Yc = lambda v: y0 - v * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W_}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W_}" height="{Hh}" fill="#1A1F4A"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="#F4F0E8" font-size="13" text-anchor="middle">Sparsity buys legible superposition (cyclic vars, H={Hd})</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="#2c2f5e"/>',
                         f'<text x="{x0-8}" y="{Y+4:.1f}" fill="#8784b3" font-size="10" text-anchor="end">{v:g}</text>']
    for s in SPS:
        p.append(f'<text x="{Xc(s):.1f}" y="{y0+18}" fill="#8784b3" font-size="11" text-anchor="middle">{s}</text>')
    p.append(f'<line x1="{x0}" y1="{Yc(1.0/M):.1f}" x2="{x1}" y2="{Yc(1.0/M):.1f}" stroke="#666" stroke-dasharray="4 3"/>')
    p.append(f'<text x="{x1}" y="{Yc(1.0/M)-4:.1f}" fill="#8784b3" font-size="9" text-anchor="end">chance 1/m</text>')
    pts = " ".join(f"{Xc(s):.1f},{Yc(rows[s]):.1f}" for s in SPS)
    p.append(f'<polyline points="{pts}" fill="none" stroke="#6FD6C9" stroke-width="2.5"/>')
    for s in SPS:
        p.append(f'<circle cx="{Xc(s):.1f}" cy="{Yc(rows[s]):.1f}" r="3.5" fill="#6FD6C9"/>')
    p.append(f'<text x="{(x0+x1)/2}" y="{Hh-8}" fill="#B8B3D6" font-size="11" text-anchor="middle">sparsity (fraction of variables inactive)</text>')
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg_curve(os.path.join(RUNS, "superpos_static_curve.svg"))

# ---- geometry: each variable's m values in 2D (fed in isolation) — circle or not? ----
def svg_geom(path):
    W, b = Wsave; nshow = 6; per = 150; cols = 3; rows_ = 2
    Wd = W  # [Hd, K*M]
    Wt = W_full = W
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{cols*per}" height="{rows_*per+34}" font-family="Inconsolata,monospace">',
           f'<rect width="{cols*per}" height="{rows_*per+34}" fill="#1A1F4A"/>',
           f'<text x="{cols*per/2}" y="22" fill="#F4F0E8" font-size="12" text-anchor="middle">Each variable\'s {M} values in the bottleneck (fed alone) — ordered ring?</text>']
    for s in range(nshow):
        k = s
        # isolated inputs: variable k = v active, all else 0
        xi = torch.zeros(M, K * M, device=DEV)
        for v in range(M):
            xi[v, k * M + v] = 1.0
        h = (xi @ W.T).cpu()                                 # [M, Hd]
        h = h - h.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(h, full_matrices=False)
        P = (h @ Vh[:2].T); P = P / (P.abs().max() + 1e-9)
        cx = (s % cols) * per + per / 2; cy = (s // cols) * per + per / 2 + 26; R = per * 0.32
        Xf = lambda v: cx + P[v, 0].item() * R; Yf = lambda v: cy - P[v, 1].item() * R
        ring = " ".join(f"{Xf(v):.1f},{Yf(v):.1f}" for v in range(M)) + f" {Xf(0):.1f},{Yf(0):.1f}"
        out.append(f'<polyline points="{ring}" fill="none" stroke="#3a3f6e" stroke-width="1"/>')
        for v in range(M):
            out.append(f'<circle cx="{Xf(v):.1f}" cy="{Yf(v):.1f}" r="5" fill="hsl({int(360*v/M)},70%,64%)"/>')
            out.append(f'<text x="{Xf(v):.1f}" y="{Yf(v)-7:.1f}" fill="#F4F0E8" font-size="8" text-anchor="middle">{v}</text>')
        out.append(f'<text x="{cx:.1f}" y="{(s//cols)*per+44:.1f}" fill="#8784b3" font-size="9" text-anchor="middle">var {k}</text>')
    out.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(out))
svg_geom(os.path.join(RUNS, "superpos_static_geom.svg"))
print("\n  wrote runs/superpos_static_curve.svg and runs/superpos_static_geom.svg")
