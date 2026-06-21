"""
sidecar.py — Experiment 8 (go-crazy): a LEARNING SIDECAR that consolidates, not shuffles.

The wild idea: give a model its own small persistent space to write something profound into —
not a buffer of stored memories (lookup), but a state that CONSOLIDATES experience into the
underlying RULE, so it generalizes beyond what it was shown, persists after the examples are
gone, and is legible (you can read the learned skill back out).

Toy: a hidden cipher per episode, y = (x + b) mod n  (secret shift b = the "skill"). The system
sees K teaching pairs (x, y), writes them into a sidecar state s, then must answer NEW x it was
NEVER taught — which requires extracting the rule b, not memorizing pairs.

  WRITE:  s = mean_i  write_mlp([emb(x_i), emb(y_i)])           (permutation-invariant accumulate)
  READ :  y_hat = read_mlp([emb(x_query), s])                   (apply the rule to any x)
All of emb/write/read are META-LEARNED across random-b episodes; s is the per-episode state.

Demonstrates, vs a LOOKUP memory (stores pairs):
  (1) generalization to UNTAUGHT inputs (consolidated the rule)  — lookup can't (chance on untaught)
  (2) robustness to NOISY teaching (averages it out)             — lookup stores the noise
  (3) legibility: a linear probe reads the secret b out of s     — and b lays out on a circle
Honest lineage: fast-weights / Titans / meta-learned memory. This is a concept demonstrator.
Outputs: runs/sidecar_genK.svg, runs/sidecar_circle.svg
"""
import os, sys, json, math
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.set_float32_matmul_precision("high")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "runs")
N, E, Hs = 12, 32, 64

class Sidecar(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(N, E)
        self.write = nn.Sequential(nn.Linear(2 * E, Hs), nn.ReLU(), nn.Linear(Hs, Hs))
        self.read = nn.Sequential(nn.Linear(E + Hs, Hs), nn.ReLU(), nn.Linear(Hs, N))
    def state(self, xp, yp):                                   # xp,yp: [B,K]
        pe = self.write(torch.cat([self.emb(xp), self.emb(yp)], -1))   # [B,K,Hs]
        return pe.mean(1)                                      # [B,Hs]  the sidecar
    def answer(self, xq, s):                                   # xq:[B,Q], s:[B,Hs]
        Q = xq.shape[1]
        return self.read(torch.cat([self.emb(xq), s[:, None, :].expand(-1, Q, -1)], -1))

def episode(B, K, g, noise=0.0):
    b = torch.randint(0, N, (B,), generator=g, device=DEV)
    xp = torch.randint(0, N, (B, K), generator=g, device=DEV)
    yp = (xp + b[:, None]) % N
    if noise > 0:
        bad = torch.rand(B, K, generator=g, device=DEV) < noise
        yp = torch.where(bad, torch.randint(0, N, (B, K), generator=g, device=DEV), yp)
    return xp, yp, b

def train(steps=20000, B=256, lr=1e-3, seed=0):
    torch.manual_seed(seed); m = Sidecar().to(DEV); opt = torch.optim.Adam(m.parameters(), lr)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    allx = torch.arange(N, device=DEV)[None, :].expand(B, N)
    for step in range(steps):
        K = int(torch.randint(1, 4, (1,), generator=g, device=DEV))
        xp, yp, b = episode(B, K, g)
        s = m.state(xp, yp); logits = m.answer(allx, s)
        yq = (allx + b[:, None]) % N
        loss = F.cross_entropy(logits.reshape(-1, N), yq.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def untaught_acc(m, K, g, noise=0.0, B=4000):
    xp, yp, b = episode(B, K, g, noise)
    allx = torch.arange(N, device=DEV)[None, :].expand(B, N)
    pred = m.answer(allx, m.state(xp, yp)).argmax(-1)
    yq = (allx + b[:, None]) % N
    taught = torch.zeros(B, N, dtype=torch.bool, device=DEV).scatter_(1, xp, True)
    sc = ((pred == yq) & ~taught).sum().item() / max(1, (~taught).sum().item())     # sidecar, untaught only
    # lookup baseline: answer only taught x correctly (store pairs); untaught -> chance
    lk = (1.0 / N)
    return sc, lk

print(f"device={DEV}  learning sidecar  (hidden cipher y=(x+b) mod {N})\n  meta-training...")
m = train()
g = torch.Generator(device=DEV).manual_seed(123)

print("\n  generalization to UNTAUGHT inputs (the rule, not the pairs):")
print("   K teaching pairs |  sidecar (untaught acc) | lookup (untaught)")
genK = {}
for K in [1, 2, 3, 5]:
    sc, lk = untaught_acc(m, K, g)
    genK[K] = dict(sidecar=sc, lookup=lk)
    print(f"        {K:<2}           |        {sc:.3f}          |     {lk:.3f}")

print("\n  robustness to NOISY teaching (K=5; fraction of corrupted pairs):")
noiseR = {}
for eps in [0.0, 0.2, 0.4]:
    sc, _ = untaught_acc(m, 5, g, noise=eps)
    noiseR[eps] = sc
    print(f"     noise {eps:<4} -> sidecar untaught acc {sc:.3f}   (a lookup memory would store the noise)")

# legibility: probe the secret b out of the sidecar state s
def probe_b(m, g):
    B = 8000
    xp, yp, b = episode(B, 3, g)
    with torch.no_grad():
        s = m.state(xp, yp)
    ntr = B // 2
    probe = nn.Linear(Hs, N).to(DEV); opt = torch.optim.Adam(probe.parameters(), 1e-2)
    for _ in range(500):
        loss = F.cross_entropy(probe(s[:ntr]), b[:ntr]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (probe(s[ntr:]).argmax(1) == b[ntr:]).float().mean().item()
    return acc
print(f"\n  legibility: linear probe reads the secret rule b out of the sidecar -> {probe_b(m, g):.3f} (chance {1/N:.3f})")
json.dump({"genK": genK, "noise": noiseR}, open(os.path.join(RUNS, "sidecar.json"), "w"), indent=2)

# ---- SVG 1: generalization vs K ----
def svg_genK(path):
    W, Hh, ml, mr, mt, mb = 560, 320, 56, 150, 36, 46; x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    KS = [1, 2, 3, 5]; Xc = lambda i: x0 + i / (len(KS) - 1) * (x1 - x0); Yc = lambda v: y0 - v * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="#1A1F4A"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="#F4F0E8" font-size="13" text-anchor="middle">Consolidate, don\'t shuffle: accuracy on UNTAUGHT inputs</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="#2c2f5e"/>',
                         f'<text x="{x0-8}" y="{Y+4:.1f}" fill="#8784b3" font-size="10" text-anchor="end">{v:g}</text>']
    for i, K in enumerate(KS):
        p.append(f'<text x="{Xc(i):.1f}" y="{y0+18}" fill="#8784b3" font-size="11" text-anchor="middle">{K}</text>')
    p.append(f'<text x="{(x0+x1)/2}" y="{Hh-8}" fill="#B8B3D6" font-size="11" text-anchor="middle"># teaching pairs shown</text>')
    for key, col, lab in [("sidecar", "#6FD6C9", "sidecar (learns the rule)"), ("lookup", "#FF8FB3", "lookup memory (chance on untaught)")]:
        pts = " ".join(f"{Xc(i):.1f},{Yc(genK[K][key]):.1f}" for i, K in enumerate(KS))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        for i, K in enumerate(KS):
            p.append(f'<circle cx="{Xc(i):.1f}" cy="{Yc(genK[K][key]):.1f}" r="3.5" fill="{col}"/>')
    ly = mt + 12
    for col, lab in [("#6FD6C9", "sidecar"), ("#FF8FB3", "lookup")]:
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="#F4F0E8" font-size="10.5">{lab}</text>']; ly += 20
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg_genK(os.path.join(RUNS, "sidecar_genK.svg"))

# ---- SVG 2: the learned rule b laid out in the sidecar (a circle?) ----
@torch.no_grad()
def svg_circle(path):
    B = 12000; xp, yp, b = episode(B, 3, g)
    s = m.state(xp, yp)
    means = torch.stack([s[b == v].mean(0) for v in range(N)])     # [N, Hs] mean state per secret b
    e = means - means.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(e, full_matrices=False)
    P = (e @ Vh[:2].T).cpu(); P = P / (P.abs().max() + 1e-9)
    Wd = 440; R = 165; cx = Wd / 2; cy = Wd / 2 + 6
    Xf = lambda i: cx + P[i, 0].item() * R; Yf = lambda i: cy - P[i, 1].item() * R
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd}" height="{Wd+30}" font-family="Inconsolata,monospace">',
         f'<rect width="{Wd}" height="{Wd+30}" fill="#1A1F4A"/>',
         f'<text x="{Wd/2}" y="22" fill="#F4F0E8" font-size="13" text-anchor="middle">The skill, read out of the sidecar: secret shift b on a circle</text>']
    ring = " ".join(f"{Xf(i):.1f},{Yf(i):.1f}" for i in range(N)) + f" {Xf(0):.1f},{Yf(0):.1f}"
    p.append(f'<polyline points="{ring}" fill="none" stroke="#3a3f6e" stroke-width="1.2"/>')
    for i in range(N):
        p.append(f'<circle cx="{Xf(i):.1f}" cy="{Yf(i):.1f}" r="6" fill="hsl({int(360*i/N)},70%,64%)"/>')
        p.append(f'<text x="{Xf(i):.1f}" y="{Yf(i)-9:.1f}" fill="#F4F0E8" font-size="9" text-anchor="middle">{i}</text>')
    p.append(f'<text x="{Wd/2}" y="{Wd+24}" fill="#8784b3" font-size="10" text-anchor="middle">mean sidecar state per secret shift b, projected to 2D</text>')
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg_circle(os.path.join(RUNS, "sidecar_circle.svg"))
print("\n  wrote runs/sidecar_genK.svg and runs/sidecar_circle.svg")
