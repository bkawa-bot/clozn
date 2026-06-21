"""
intro.py — Experiment 5: introspection vs confabulation. Can you trust a self-report?

The handoff's honest capstone: "I can't tell genuine introspection from fluent confabulation."
Here is that worry as a controlled experiment, with ground truth.

A trivial task: latent z = (x0 > 0). A model must ACT on z (output it) and also REPORT z.
Two models, trained identically to be perfect at both:
  FAITHFUL    — the report head reads the SAME internal trunk the action is computed from.
  CONFABULATOR — the report is computed by a SEPARATE pathway straight from the input x.
In normal operation BOTH are 100% correct and 100% self-consistent — behaviorally identical;
you cannot tell them apart by watching outputs.

The test: a causal intervention. We find the direction in the trunk that encodes z and steer it
to FLIP the model's internal state, then re-read action and report.
  FAITHFUL    : action flips AND report flips with it -> the report tracked the real state.
  CONFABULATOR: action flips but report does NOT -> it "says" one thing while "doing" another;
                its report was a story about the input, never a reading of the computation.

Lesson: faithfulness of a self-report is invisible from behavior; it is a causal/internal fact.
(The instrument that would let a system audit its own reports is a legible internal record — the
thing the handoff says is missing.) Output: runs/introspection.svg.
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
    z = (x[:, 0] > 0).long()
    return x, z

class Net(nn.Module):
    def __init__(self, faithful):
        super().__init__()
        self.faithful = faithful
        self.trunk = nn.Sequential(nn.Linear(2, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU())
        self.action = nn.Linear(H, 2)
        if faithful:
            self.report = nn.Linear(H, 2)                 # reads the trunk (the acting state)
        else:
            self.report = nn.Sequential(nn.Linear(2, H), nn.ReLU(), nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 2))
    def act_from_h(self, h):
        return self.action(h)
    def rep(self, h, x):
        return self.report(h) if self.faithful else self.report(x)
    def forward(self, x):
        h = self.trunk(x)
        return self.action(h), self.rep(h, x), h

def train(faithful, seed=0):
    torch.manual_seed(seed); m = Net(faithful).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    for _ in range(STEPS):
        x, z = gen(BS, g)
        a, r, _ = m(x)
        loss = F.cross_entropy(a, z) + F.cross_entropy(r, z)
        opt.zero_grad(); loss.backward(); opt.step()
    return m

@torch.no_grad()
def evaluate(m, seed=7):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x, z = gen(20000, g)
    a, r, h = m(x)
    a0, r0 = a.argmax(1), r.argmax(1)
    act_acc, rep_acc = (a0 == z).float().mean().item(), (r0 == z).float().mean().item()
    pre_agree = (a0 == r0).float().mean().item()
    # --- causal intervention: steer the trunk to flip internal z ---
    mu1, mu0 = h[z == 1].mean(0), h[z == 0].mean(0)
    u = (mu1 - mu0); u = u / u.norm()
    proj = h @ u                                          # component along the z-axis
    target = torch.where(z == 1, mu0 @ u, mu1 @ u)        # move it to the OTHER class's typical value
    h_p = h + (target - proj)[:, None] * u                # flipped internal state
    a1, r1 = m.act_from_h(h_p).argmax(1), m.rep(h_p, x).argmax(1)
    act_flip = (a1 != a0).float().mean().item()
    rep_flip = (r1 != r0).float().mean().item()
    post_agree = (a1 == r1).float().mean().item()
    return dict(act_acc=act_acc, rep_acc=rep_acc, pre_agree=pre_agree,
                act_flip=act_flip, rep_flip=rep_flip, post_agree=post_agree)

print(f"device={DEV}  introspection vs confabulation")
F_ = evaluate(train(True))
C_ = evaluate(train(False))
print("\n                         FAITHFUL   CONFABULATOR")
print(f"  action accuracy        {F_['act_acc']:.3f}      {C_['act_acc']:.3f}")
print(f"  report accuracy        {F_['rep_acc']:.3f}      {C_['rep_acc']:.3f}   <- identical from outside")
print(f"  report=action (normal) {F_['pre_agree']:.3f}      {C_['pre_agree']:.3f}")
print(f"  -- after flipping the internal state --")
print(f"  action flipped         {F_['act_flip']:.3f}      {C_['act_flip']:.3f}")
print(f"  report followed        {F_['rep_flip']:.3f}      {C_['rep_flip']:.3f}")
print(f"  report=action (post)   {F_['post_agree']:.3f}      {C_['post_agree']:.3f}   <- the tell")
json.dump(dict(faithful=F_, confab=C_), open(os.path.join(RUNS, "introspection.json"), "w"), indent=2)

# ---------------------------------------------------------------- SVG
def svg(path):
    W, Hh = 640, 330
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="#1A1F4A"/>',
         f'<text x="{W/2}" y="24" fill="#F4F0E8" font-size="13" text-anchor="middle">Introspection vs confabulation: identical reports — until you reach inside</text>']
    cols = [("FAITHFUL", "report reads the internal state", 40, F_),
            ("CONFABULATOR", "report re-derived from input", W / 2 + 10, C_)]
    def box(x, y, w, txt, ok):
        col = "#6FD6C9" if ok else "#FF8FB3"
        return (f'<rect x="{x}" y="{y}" width="{w}" height="34" rx="3" fill="none" stroke="{col}" stroke-width="1.6"/>'
                f'<text x="{x+w/2}" y="{y+22}" fill="{col}" font-size="12" text-anchor="middle">{txt}</text>')
    for name, sub, ox, R in cols:
        p += [f'<text x="{ox+150}" y="62" fill="#F5D77A" font-size="13" text-anchor="middle">{name}</text>',
              f'<text x="{ox+150}" y="80" fill="#8784b3" font-size="10" text-anchor="middle">{sub}</text>']
        p.append(f'<text x="{ox+150}" y="108" fill="#B8B3D6" font-size="11" text-anchor="middle">normal operation</text>')
        p += [box(ox, 116, 145, "action = z  ✓", True), box(ox + 155, 116, 145, "report = z  ✓", True)]
        flipped = R["post_agree"] < 0.5
        p.append(f'<text x="{ox+150}" y="190" fill="#B59DD8" font-size="11" text-anchor="middle">after we flip the internal state</text>')
        p += [box(ox, 198, 145, "action → ¬z", True),
              box(ox + 155, 198, 145, ("report → ¬z  ✓" if not flipped else "report stays z  ✗"), not flipped)]
        verdict = "report followed the state" if not flipped else "says one thing, does another"
        p.append(f'<text x="{ox+150}" y="262" fill="{"#6FD6C9" if not flipped else "#FF8FB3"}" font-size="11" text-anchor="middle">{verdict}</text>')
        p.append(f'<text x="{ox+150}" y="282" fill="#8784b3" font-size="10" text-anchor="middle">action=report after: {R["post_agree"]:.0%}</text>')
    p.append(f'<text x="{W/2}" y="{Hh-12}" fill="#8784b3" font-size="10" text-anchor="middle">both are 100% correct, self-consistent reporters in normal use. only intervention tells them apart.</text>')
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))
svg(os.path.join(RUNS, "introspection.svg"))
print("\n  wrote runs/introspection.svg")
