"""
toy_superposition.py — Experiment 4: how a network crams MORE features than dimensions.

(Anthropic, "Toy Models of Superposition.") A linear autoencoder squeezes n features through
m < n hidden dims and reconstructs them:  x_hat = ReLU(W^T (W x) + b),  W is [m, n].
Each feature fires only with probability (1 - sparsity); when sparse, two features rarely co-occur,
so the model can point them in OVERLAPPING directions and tolerate the rare interference.

The phase change: at low sparsity the model keeps only ~m features (orthogonal, no superposition);
as sparsity rises it represents MANY more, arranged as regular polygons in the m-dim space.
This is *why* real models are hard to read — concepts share directions. The opposite of the clean
grokking circle: there, structure was legible; here, capacity is bought by making it illegible.

m = 2 (so we can draw it). Outputs: runs/superposition.svg (feature geometry per sparsity).
"""
import os, sys, json, math
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.set_float32_matmul_precision("high")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "runs")
N, M = 6, 2                                              # n features -> m=2 hidden dims
SPARSITIES = [0.0, 0.6, 0.85, 0.93, 0.97]
STEPS, BS, LR = 20000, 2048, 1e-3
IMP = 0.8 ** torch.arange(N, dtype=torch.float32, device=DEV)   # feature importance (pecking order)

class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = nn.Parameter(torch.randn(M, N) * 0.1)
        self.b = nn.Parameter(torch.zeros(N))
    def forward(self, x):                                # x [B,N]
        h = x @ self.W.T                                 # [B,M]
        return F.relu(h @ self.W + self.b)               # [B,N]

def gen(bs, S, g):
    val = torch.rand(bs, N, generator=g, device=DEV)
    mask = (torch.rand(bs, N, generator=g, device=DEV) > S).float()
    return val * mask

def train(S, seed=0):
    torch.manual_seed(seed); m = Toy().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    for _ in range(STEPS):
        x = gen(BS, S, g)
        loss = (IMP * (m(x) - x) ** 2).mean()            # importance-weighted (pecking order)
        opt.zero_grad(); loss.backward(); opt.step()
    return m.W.detach().cpu()                             # [M,N]

print(f"device={DEV}  superposition  n={N} features -> m={M} dims")
print("\n  sparsity | feature norms (||W_i||)            | # represented")
Ws, counts = [], []
for S in SPARSITIES:
    W = train(S)
    norms = W.norm(dim=0)                                # [N]
    cnt = int((norms > 0.3).sum())
    Ws.append(W); counts.append(cnt)
    print(f"   {S:<5}    | {'  '.join(f'{n:.2f}' for n in norms.tolist())} |  {cnt}/{N}")
json.dump(dict(sparsities=SPARSITIES, counts=counts), open(os.path.join(RUNS, "superposition.json"), "w"), indent=2)

# ---- SVG: one 2D panel per sparsity, feature directions as spokes ----
def svg(path):
    pw, ph, R = 232, 250, 84
    W = pw * len(SPARSITIES); Hh = ph + 24
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="#1A1F4A"/>',
         f'<text x="{W/2}" y="20" fill="#F4F0E8" font-size="13" text-anchor="middle">Superposition: packing {N} features into {M} dimensions (more sparsity → more features, as polygons)</text>']
    for pi, (S, Wm, cnt) in enumerate(zip(SPARSITIES, Ws, counts)):
        ox = pi * pw; cx, cy = ox + pw / 2, 40 + (ph - 40) / 2
        scale = max(float(Wm.norm(dim=0).max()), 1e-6)
        p.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{R}" fill="none" stroke="#2c2f5e"/>')
        p.append(f'<text x="{cx:.1f}" y="36" fill="#F5D77A" font-size="11" text-anchor="middle">sparsity {S} · {cnt}/{N} kept</text>')
        for i in range(N):
            ex = cx + Wm[0, i].item() / scale * R
            ey = cy - Wm[1, i].item() / scale * R
            hue = int(360 * i / N)
            p.append(f'<line x1="{cx:.1f}" y1="{cy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" stroke="hsl({hue},65%,60%)" stroke-width="2"/>')
            p.append(f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4" fill="hsl({hue},70%,64%)"/>')
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg(os.path.join(RUNS, "superposition.svg"))
print(f"\n  dense (S=0) keeps ~{counts[0]} features; sparsest (S={SPARSITIES[-1]}) keeps {counts[-1]}.")
print("  wrote runs/superposition.svg")
