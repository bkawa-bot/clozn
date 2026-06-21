"""
grok_clock.py — PROVE the grokked model computes via "clocks" (read the mechanism).

Retrains the Exp-3 grokking model (deterministic), then tests the clock hypothesis three ways:
  1. Embedding spectrum: power should concentrate at a FEW integer frequencies k (key freqs),
     not spread out. (Sparse Fourier code = a few clocks.)
  2. Logit reconstruction: the model's output logit for candidate answer c, given (a,b), should
     be ~ sum over key freqs of cos(2*pi*k*(a+b-c)/p) (+ a sin term for phase). We regress the
     model's REAL logits onto these clock features and report R^2 as we add frequencies. High R^2
     from a few freqs = the model literally outputs "how aligned is c's angle with (a+b)'s angle."
  3. One worked example: plot the model's logits over all candidate answers vs the clock
     reconstruction; both should peak at c=(a+b) mod p.

Outputs: runs/clock_spectrum.svg, runs/clock_example.svg. (matplotlib blocked -> SVG.)
"""
import os, sys, json, math
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.set_float32_matmul_precision("high")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "runs")
P, D, Hd = 59, 128, 512
TRAIN_FRAC, STEPS, LR, WD, SEED = 0.5, 30000, 1e-3, 1.0, 0

# ---- data + model (identical to grok.py; deterministic) ----
torch.manual_seed(SEED)
a = torch.arange(P, device=DEV).repeat_interleave(P); b = torch.arange(P, device=DEV).repeat(P)
y = (a + b) % P
perm = torch.randperm(P * P, generator=torch.Generator(device=DEV).manual_seed(SEED), device=DEV)
ntr = int(P * P * TRAIN_FRAC); tr, te = perm[:ntr], perm[ntr:]
atr, btr, ytr = a[tr], b[tr], y[tr]; ate, bte, yte = a[te], b[te], y[te]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(P, D); self.fc1 = nn.Linear(2 * D, Hd); self.fc2 = nn.Linear(Hd, P)
    def forward(self, a, b):
        return self.fc2(F.relu(self.fc1(torch.cat([self.emb(a), self.emb(b)], -1))))

model = MLP().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
print(f"training grokking model ({STEPS} steps)...")
for step in range(STEPS):
    loss = F.cross_entropy(model(atr, btr), ytr)
    opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad():
    test_acc = (model(ate, bte).argmax(-1) == yte).float().mean().item()
print(f"  test acc = {test_acc:.3f}")

# ---- 1. key frequencies from the embedding spectrum ----
with torch.no_grad():
    e = model.emb.weight.detach(); e = e - e.mean(0, keepdim=True)
    spec = (torch.fft.rfft(e, dim=0).abs() ** 2).sum(1)          # [nf]
    spec[0] = 0                                                   # drop DC
    specf = (spec / spec.sum()).cpu()
nf = specf.shape[0]
order = torch.argsort(specf, descending=True)
key = [int(k) for k in order[:6].tolist()]
print(f"\n  embedding spectrum: top frequencies = {key}")
print(f"  power in those 6 of {nf-1} freqs = {specf[order[:6]].sum().item():.1%}")

# ---- 2. reconstruct the model's logits from clock features ----
with torch.no_grad():
    L = model(ate, bte)                                          # [Nte, P] real logits over candidates c
    L = L - L.mean(1, keepdim=True)                              # center over c (the answer-selecting shape)
    s = (ate + bte).float()                                      # [Nte]
    cgrid = torch.arange(P, device=DEV).float()                  # [P]
    diff = (s[:, None] - cgrid[None, :])                         # [Nte, P]  = (a+b - c)
    def feats(freqs):
        cols = []
        for k in freqs:
            ph = 2 * math.pi * k * diff / P
            cols += [torch.cos(ph), torch.sin(ph)]
        Fm = torch.stack(cols, -1)                               # [Nte, P, 2K]
        return Fm - Fm.mean(1, keepdim=True)                     # center over c
    def r2(freqs):
        Fc = feats(freqs).reshape(-1, 2 * len(freqs)); Lc = L.reshape(-1)
        W = torch.linalg.lstsq(Fc, Lc.unsqueeze(1)).solution
        pred = (Fc @ W).squeeze(1)
        return (1 - ((Lc - pred) ** 2).sum() / (Lc ** 2).sum()).item(), W
    print("\n  fraction of the model's output (logit) variance explained by clock features:")
    for n in range(1, 7):
        rr, _ = r2(key[:n])
        print(f"    {n} freq{'s' if n>1 else ' '} {key[:n]} -> R^2 = {rr:.3f}")
    R2_full, Wfull = r2(key)

# ---- SVG 1: spectrum ----
def svg_spectrum(path):
    W, Hh, ml, mt, mb = 600, 300, 50, 36, 40; x0, x1, y0, y1 = ml, W - 20, Hh - mb, mt
    mx = float(specf.max())
    bw = (x1 - x0) / (nf - 1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="#1A1F4A"/>',
         f'<text x="{W/2}" y="22" fill="#F4F0E8" font-size="13" text-anchor="middle">The model uses only a few "clocks": embedding frequency spectrum</text>']
    for f in range(1, nf):
        h = (specf[f].item() / mx) * (y0 - y1)
        col = "#6FD6C9" if f in key else "#3a3f6e"
        p.append(f'<rect x="{x0+(f-1)*bw:.1f}" y="{y0-h:.1f}" width="{bw*0.8:.1f}" height="{h:.1f}" fill="{col}"/>')
        if f in key[:5]:
            p.append(f'<text x="{x0+(f-1)*bw+bw*0.4:.1f}" y="{y0-h-4:.1f}" fill="#F5D77A" font-size="10" text-anchor="middle">k={f}</text>')
    p += [f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#2c2f5e"/>',
          f'<text x="{(x0+x1)/2}" y="{Hh-8}" fill="#B8B3D6" font-size="11" text-anchor="middle">frequency k (how many times around the circle per number)</text>']
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg_spectrum(os.path.join(RUNS, "clock_spectrum.svg"))

# ---- SVG 2: one worked example, model logits vs clock reconstruction ----
def svg_example(path, i0=0):
    with torch.no_grad():
        Lc = L[i0].cpu()                                         # centered model logits over c
        Fc = feats(key)[i0].reshape(P, -1).cpu(); rec = (Fc @ Wfull.cpu()).squeeze(1)
    ans = int((ate[i0] + bte[i0]) % P); av, bv = int(ate[i0]), int(bte[i0])
    W, Hh, ml, mt, mb = 600, 300, 46, 38, 40; x0, x1, y0, y1 = ml, W - 150, Hh - mb, mt
    lo = float(min(Lc.min(), rec.min())); hi = float(max(Lc.max(), rec.max())); rng = hi - lo + 1e-9
    Xc = lambda c: x0 + c / (P - 1) * (x1 - x0)
    Yc = lambda v: y0 - (v - lo) / rng * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="#1A1F4A"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="#F4F0E8" font-size="13" text-anchor="middle">{av} + {bv} mod {P}: model output vs the clock formula</text>',
         f'<line x1="{Xc(ans):.1f}" y1="{y1}" x2="{Xc(ans):.1f}" y2="{y0}" stroke="#B59DD8" stroke-dasharray="3 3"/>',
         f'<text x="{Xc(ans)+4:.1f}" y="{y1+10}" fill="#B59DD8" font-size="10">answer = {ans}</text>']
    pm = " ".join(f"{Xc(c):.1f},{Yc(Lc[c].item()):.1f}" for c in range(P))
    pr = " ".join(f"{Xc(c):.1f},{Yc(rec[c].item()):.1f}" for c in range(P))
    p += [f'<polyline points="{pm}" fill="none" stroke="#6FD6C9" stroke-width="2.5"/>',
          f'<polyline points="{pr}" fill="none" stroke="#F5D77A" stroke-width="1.6" stroke-dasharray="5 3"/>',
          f'<rect x="{x1+14}" y="{mt+6}" width="12" height="12" fill="#6FD6C9"/><text x="{x1+30}" y="{mt+16}" fill="#F4F0E8" font-size="11">model output</text>',
          f'<rect x="{x1+14}" y="{mt+28}" width="12" height="12" fill="#F5D77A"/><text x="{x1+30}" y="{mt+38}" fill="#F4F0E8" font-size="11">clock formula</text>',
          f'<text x="{(x0+x1)/2}" y="{Hh-8}" fill="#B8B3D6" font-size="11" text-anchor="middle">candidate answer c (0..{P-1})</text>']
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg_example(os.path.join(RUNS, "clock_example.svg"))

json.dump(dict(key_freqs=key, R2_full=R2_full, test_acc=test_acc), open(os.path.join(RUNS, "clock.json"), "w"), indent=2)
print(f"\n  -> {R2_full:.1%} of the model's output is reconstructed by {len(key)} clock frequencies.")
print("  wrote runs/clock_spectrum.svg and runs/clock_example.svg")
