"""
grok.py — Experiment 3: watch a network cross from MEMORIZING to UNDERSTANDING.

Grokking (Power et al. 2022; Nanda's "progress measures"): train a small net on modular
addition (a+b mod p) on only a fraction of the (a,b) pairs. It memorizes the train set fast
(train acc -> 100%) while test acc stays at chance — then, long after, test acc SUDDENLY
jumps to 100%. It has stopped memorizing and started computing (a+b) mod p.

The handoff's thesis: "understanding is what prediction becomes when forbidden from memorizing."
Grokking is that sentence as a training curve. And mechanistically, the model represents each
number n on a CIRCLE (embeddings become ~cos/sin of 2*pi*k*n/p) — clock arithmetic on a clock.

My hypothesis (ties to Exp 1's legibility probe): the circle forming = legibility appearing.
I track a STRUCTURE score = how concentrated each embedding's spectrum is in a few Fourier
modes. Prediction: it rises with (or just before) the generalization jump. Memorization is
spectrally spread (illegible); understanding is spectrally sparse (legible, nameable as "freq k").

Outputs: runs/grok_curve.svg (train/test acc + structure vs step) and runs/grok_circle.svg
(embeddings projected to 2D -> a ring, numbers in order). matplotlib is blocked here -> SVG.
"""
import os, sys, json, math
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.set_float32_matmul_precision("high")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "runs"); os.makedirs(RUNS, exist_ok=True)

P, D, Hd = 59, 128, 512
TRAIN_FRAC, STEPS, EVERY, LR, WD, SEED = 0.5, 30000, 300, 1e-3, 1.0, 0

torch.manual_seed(SEED)
a = torch.arange(P, device=DEV).repeat_interleave(P)
b = torch.arange(P, device=DEV).repeat(P)
y = (a + b) % P
perm = torch.randperm(P * P, generator=torch.Generator(device=DEV).manual_seed(SEED), device=DEV)
ntr = int(P * P * TRAIN_FRAC); tr, te = perm[:ntr], perm[ntr:]
atr, btr, ytr = a[tr], b[tr], y[tr]
ate, bte, yte = a[te], b[te], y[te]

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(P, D)
        self.fc1 = nn.Linear(2 * D, Hd)
        self.fc2 = nn.Linear(Hd, P)
    def forward(self, a, b):
        h = torch.cat([self.emb(a), self.emb(b)], -1)
        return self.fc2(F.relu(self.fc1(h)))

model = MLP().to(DEV)
emb_init = model.emb.weight.detach().clone()            # random init, for the before/after viz
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

@torch.no_grad()
def acc(a, b, yt):
    return (model(a, b).argmax(-1) == yt).float().mean().item()

@torch.no_grad()
def structure():
    e = model.emb.weight.detach()                       # [P, D]
    e = e - e.mean(0, keepdim=True)
    pw = (torch.fft.rfft(e, dim=0).abs() ** 2)[1:]      # [F, D]  drop DC
    return (pw.topk(min(5, pw.shape[0]), dim=0).values.sum(0) / (pw.sum(0) + 1e-9)).mean().item()

print(f"device={DEV}  grokking  p={P}  train pairs={ntr}/{P*P}  wd={WD}")
print("\n  step   train  test   structure")
hist, grok_step = [], None
for step in range(STEPS + 1):
    if step % EVERY == 0 or step == STEPS:
        tra, tea, st = acc(atr, btr, ytr), acc(ate, bte, yte), structure()
        hist.append(dict(step=step, train=tra, test=tea, struct=st))
        if grok_step is None and tea > 0.9:
            grok_step = step
        if step % (EVERY * 5) == 0 or step == STEPS:
            print(f"  {step:<6d} {tra:.3f}  {tea:.3f}  {st:.3f}" + ("   <- GROK" if grok_step == step else ""))
    loss = F.cross_entropy(model(atr, btr), ytr)
    opt.zero_grad(); loss.backward(); opt.step()

json.dump(dict(hist=hist, grok_step=grok_step, p=P), open(os.path.join(RUNS, "grok.json"), "w"), indent=2)
print(f"\n  grok step (test>0.9): {grok_step}   final test acc: {hist[-1]['test']:.3f}   final structure: {hist[-1]['struct']:.3f}")

# -------------------------------------------------------------------- curve SVG
def svg_curve(path):
    W, Hh, ml, mr, mt, mb = 620, 380, 58, 150, 34, 46
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    lx = math.log10(STEPS)
    Xc = lambda s: x0 + (math.log10(s + 1) / lx) * (x1 - x0)
    Yc = lambda v: y0 + v * (y1 - y0)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="#1A1F4A"/>',
         f'<text x="{(x0+x1)/2}" y="20" fill="#F4F0E8" font-size="13" text-anchor="middle">Grokking: memorize first, understand later  (a+b mod {P})</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="#2c2f5e"/>',
                         f'<text x="{x0-8}" y="{Y+4:.1f}" fill="#8784b3" font-size="11" text-anchor="end">{v:g}</text>']
    for s in [1, 10, 100, 1000, 10000]:
        if s <= STEPS:
            p.append(f'<text x="{Xc(s):.1f}" y="{y0+18}" fill="#8784b3" font-size="11" text-anchor="middle">{s}</text>')
    p.append(f'<text x="{(x0+x1)/2}" y="{Hh-7}" fill="#B8B3D6" font-size="12" text-anchor="middle">training step (log)</text>')
    if grok_step:
        gx = Xc(grok_step)
        p += [f'<line x1="{gx:.1f}" y1="{y1}" x2="{gx:.1f}" y2="{y0}" stroke="#B59DD8" stroke-dasharray="3 3"/>',
              f'<text x="{gx+4:.1f}" y="{y1+12}" fill="#B59DD8" font-size="10">grok</text>']
    for key, col, lab in [("train", "#6FD6C9", "train acc (memorized)"), ("test", "#FF8FB3", "test acc (generalized)"),
                          ("struct", "#F5D77A", "structure (Fourier conc.)")]:
        pts = " ".join(f"{Xc(h['step']):.1f},{Yc(h[key]):.1f}" for h in hist)
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        ly = mt + 12 + [("train"), ("test"), ("struct")].index((key,) if False else key) * 21
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="#F4F0E8" font-size="11">{lab}</text>']
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg_curve(os.path.join(RUNS, "grok_curve.svg"))

# ------------------------------------- before/after SVG: random blob -> grokked ring
def svg_before_after(path):
    def panel(emb, ox, title, parts, R=108, top=52):
        e = emb - emb.mean(0, keepdim=True)
        _, _, Vh = torch.linalg.svd(e, full_matrices=False)
        pc = (e @ Vh[:2].T).cpu(); pc = pc / pc.abs().max()
        cx, cy = ox + 152, top + 130
        Xf = lambda i: cx + pc[i, 0].item() * R
        Yf = lambda i: cy - pc[i, 1].item() * R
        parts.append(f'<text x="{cx}" y="{top-6}" fill="#F4F0E8" font-size="12" text-anchor="middle">{title}</text>')
        order = " ".join(f"{Xf(i):.1f},{Yf(i):.1f}" for i in range(P)) + f" {Xf(0):.1f},{Yf(0):.1f}"
        parts.append(f'<polyline points="{order}" fill="none" stroke="#3a3f6e" stroke-width="0.8"/>')
        for i in range(P):
            parts.append(f'<circle cx="{Xf(i):.1f}" cy="{Yf(i):.1f}" r="4.5" fill="hsl({int(360*i/P)},70%,64%)"/>')
    W, Hh = 620, 360
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
             f'<rect width="{W}" height="{Hh}" fill="#1A1F4A"/>',
             f'<text x="{W/2}" y="24" fill="#F4F0E8" font-size="13" text-anchor="middle">Understanding = structure appearing  (number embeddings, a+b mod {P})</text>',
             f'<line x1="{W/2}" y1="40" x2="{W/2}" y2="{Hh-44}" stroke="#2c2f5e"/>']
    panel(emb_init, 0, "random init  →  a blob", parts)
    panel(model.emb.weight.detach(), W / 2, "after grokking  →  a ring", parts)
    parts.append(f'<text x="{W/2}" y="{Hh-16}" fill="#8784b3" font-size="10" text-anchor="middle">each dot = one number (0..{P-1}), embedding projected to 2D. memorizing looks like noise; understanding is a circle.</text>')
    open(path, "w", encoding="utf-8").write("\n".join(parts))
svg_before_after(os.path.join(RUNS, "grok_circle.svg"))
print("  wrote runs/grok_curve.svg and runs/grok_circle.svg")
