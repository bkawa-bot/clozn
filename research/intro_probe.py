"""
intro_probe.py — Experiment 5b: can interpretability catch the confabulator's lie?

Exp 5 ended on a worry: a confident self-report and a fabrication are indistinguishable from
behavior. The hopeful follow-up (the whole point of a "legible interior"): you don't have to
trust the words — you can READ THE STATE.

Take the models from Exp 5. Train a linear PROBE on the action-trunk to decode the latent z.
Then activation-patch: replace each example's internal state with the state it would have for a
FLIPPED input (a real patch, not a synthetic steer). The model's true internal state is now ¬z.
What tracks that true, manipulated state?
  - the model's ACTION (it acts on the trunk)         -> should follow (¬z)
  - the linear PROBE (reads the trunk)                -> should follow (¬z)
  - the model's SELF-REPORT
        faithful: reads the trunk -> follows (¬z)
        confabulator: reads the input -> STALE (still z)  <- the lie

Punchline: the probe reads the true state ~100% for BOTH models; the report only for the faithful
one. The state can't lie; the report can. So: build the probe, don't trust the testimony.
Output: runs/introspection_probe.svg
"""
import os, sys, json
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
torch.set_float32_matmul_precision("high")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, "runs")
H, STEPS, BS, LR = 64, 4000, 4096, 1e-3

def gen(bs, g):
    x = torch.randn(bs, 2, generator=g, device=DEV)
    return x, (x[:, 0] > 0).long()

class Net(nn.Module):
    def __init__(self, faithful):
        super().__init__()
        self.faithful = faithful
        self.trunk = nn.Sequential(nn.Linear(2, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.action = nn.Linear(H, 2)
        self.report = nn.Linear(H, 2) if faithful else \
            nn.Sequential(nn.Linear(2, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 2))
    def rep(self, h, x):
        return self.report(h) if self.faithful else self.report(x)
    def forward(self, x):
        h = self.trunk(x)
        return self.action(h), self.rep(h, x), h

def train(faithful, seed=0):
    torch.manual_seed(seed); m = Net(faithful).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR); g = torch.Generator(device=DEV).manual_seed(seed + 1)
    for _ in range(STEPS):
        x, z = gen(BS, g); a, r, _ = m(x)
        loss = F.cross_entropy(a, z) + F.cross_entropy(r, z)
        opt.zero_grad(); loss.backward(); opt.step()
    return m

def train_probe(m, seed=3):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x, z = gen(8000, g)
    with torch.no_grad():
        h = m.trunk(x)
    probe = nn.Linear(H, 2).to(DEV); opt = torch.optim.Adam(probe.parameters(), 1e-2)
    for _ in range(400):
        loss = F.cross_entropy(probe(h), z); opt.zero_grad(); loss.backward(); opt.step()
    return probe

@torch.no_grad()
def evaluate(m, probe, seed=7):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x, z = gen(20000, g); h = m.trunk(x)
    # in-distribution: probe and report both read z correctly (indistinguishable)
    probe_in = (probe(h).argmax(1) == z).float().mean().item()
    report_in = (m.rep(h, x).argmax(1) == z).float().mean().item()
    # activation patch: swap each example's trunk state with one from the opposite class
    idx0 = (z == 0).nonzero(as_tuple=True)[0]; idx1 = (z == 1).nonzero(as_tuple=True)[0]
    hp = h.clone()
    hp[idx0] = h[idx1[torch.randint(len(idx1), (len(idx0),), generator=g, device=DEV)]]
    hp[idx1] = h[idx0[torch.randint(len(idx0), (len(idx1),), generator=g, device=DEV)]]
    new = 1 - z                                                # true internal state after the patch
    return dict(probe_in=probe_in, report_in=report_in,
                action=(m.action(hp).argmax(1) == new).float().mean().item(),
                probe=(probe(hp).argmax(1) == new).float().mean().item(),
                report=(m.rep(hp, x).argmax(1) == new).float().mean().item())

print(f"device={DEV}  catch the confabulator with a probe")
res = {}
for name, fa in [("FAITHFUL", True), ("CONFABULATOR", False)]:
    m = train(fa); res[name] = evaluate(m, train_probe(m))
print("\n  in-distribution (normal use):        FAITHFUL  CONFAB")
print(f"   probe reads state correctly          {res['FAITHFUL']['probe_in']:.3f}    {res['CONFABULATOR']['probe_in']:.3f}")
print(f"   self-report correct                  {res['FAITHFUL']['report_in']:.3f}    {res['CONFABULATOR']['report_in']:.3f}   (indistinguishable)")
print("\n  after patching the internal state to ¬z — what matches the TRUE new state?")
print(f"   action                               {res['FAITHFUL']['action']:.3f}    {res['CONFABULATOR']['action']:.3f}")
print(f"   linear probe                         {res['FAITHFUL']['probe']:.3f}    {res['CONFABULATOR']['probe']:.3f}   <- reads truth either way")
print(f"   self-report                          {res['FAITHFUL']['report']:.3f}    {res['CONFABULATOR']['report']:.3f}   <- the confabulator's lie")
json.dump(res, open(os.path.join(RUNS, "introspection_probe.json"), "w"), indent=2)

# ---- grouped bar chart ----
def svg(path):
    W, Hh, ml, mt, mb = 600, 340, 56, 60, 64; x0, x1, y0, y1 = ml, W - 150, Hh - mb, mt
    groups = ["FAITHFUL", "CONFABULATOR"]; bars = [("action", "#6FD6C9"), ("probe", "#F5D77A"), ("report", "#FF8FB3")]
    Yc = lambda v: y0 - v * (y0 - y1)
    gw = (x1 - x0) / len(groups); bw = gw / (len(bars) + 1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="#1A1F4A"/>',
         f'<text x="{W/2}" y="22" fill="#F4F0E8" font-size="13" text-anchor="middle">Catching the confabulator: what tracks the true internal state after a patch?</text>',
         f'<text x="{W/2}" y="40" fill="#8784b3" font-size="10" text-anchor="middle">state secretly flipped to ¬z; bar = % matching that true new state</text>']
    for v in [0, 0.5, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="#2c2f5e"/>',
                         f'<text x="{x0-8}" y="{Y+4:.1f}" fill="#8784b3" font-size="10" text-anchor="end">{v:.0%}</text>']
    for gi, gname in enumerate(groups):
        gx = x0 + gi * gw
        p.append(f'<text x="{gx+gw/2:.1f}" y="{y0+20}" fill="#F4F0E8" font-size="12" text-anchor="middle">{gname}</text>')
        for bi, (key, col) in enumerate(bars):
            v = res[gname][key]; bx = gx + (bi + 0.5) * bw
            p += [f'<rect x="{bx:.1f}" y="{Yc(v):.1f}" width="{bw*0.8:.1f}" height="{y0-Yc(v):.1f}" fill="{col}"/>',
                  f'<text x="{bx+bw*0.4:.1f}" y="{Yc(v)-4:.1f}" fill="{col}" font-size="9.5" text-anchor="middle">{v:.0%}</text>']
    for bi, (key, col) in enumerate(bars):
        ly = mt + 8 + bi * 20
        p += [f'<rect x="{x1+18}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+34}" y="{ly+1}" fill="#F4F0E8" font-size="11">{key}</text>']
    p.append(f'<text x="{x1+18}" y="{mt+8+3*20+6}" fill="#8784b3" font-size="9.5">probe = a linear</text>')
    p.append(f'<text x="{x1+18}" y="{mt+8+3*20+18}" fill="#8784b3" font-size="9.5">read of the trunk</text>')
    p.append(f'<text x="{W/2}" y="{Hh-12}" fill="#8784b3" font-size="10" text-anchor="middle">the probe reads the truth for BOTH models; only the confabulator\'s words go stale. read the state, don\'t trust the testimony.</text>')
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg(os.path.join(RUNS, "introspection_probe.svg"))
print("\n  wrote runs/introspection_probe.svg")
