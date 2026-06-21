"""A teaching figure: linear probing (left) and cross-validation folds (right). Conceptual,
seeded — not real data. Dark theme to match the other views."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

BG, INK, MUT, TEAL, LAV, RED, GRN = "#0f1220", "#e6e9f5", "#8a90b3", "#7ee0d0", "#c9a0ff", "#e06a6a", "#6ad08a"
W, H = 900, 470
p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Inconsolata,ui-monospace,monospace">',
     f'<rect width="{W}" height="{H}" fill="{BG}"/>',
     f'<text x="28" y="38" fill="{INK}" font-size="18">How we read a concept out of the hidden state</text>']

# ---------- Panel A: linear probe ----------
p.append(f'<text x="28" y="74" fill="{TEAL}" font-size="13">1 · PROBING  —  is the feature even in there?</text>')
bx, by, bw, bh = 60, 96, 300, 280
p.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" fill="#171a2e" rx="6"/>')
rng = np.random.default_rng(3)
def cluster(cx, cy, col, n=16):
    for _ in range(n):
        x = bx + (cx + rng.normal(0, 0.10)) * bw
        y = by + (cy + rng.normal(0, 0.10)) * bh
        x = min(max(x, bx + 6), bx + bw - 6); y = min(max(y, by + 6), by + bh - 6)
        p.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="5" fill="{col}"/>')
cluster(0.68, 0.30, GRN)      # positive sentences, upper-right
cluster(0.32, 0.70, RED)      # negative sentences, lower-left
# the probe = the separating line (top-left to bottom-right diagonal)
p.append(f'<line x1="{bx}" y1="{by}" x2="{bx+bw}" y2="{by+bh}" stroke="{INK}" stroke-width="2" stroke-dasharray="6 4"/>')
# the concept direction (perpendicular arrow, lower-left -> upper-right)
p.append(f'<defs><marker id="a" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">'
         f'<path d="M0,0 L7,3 L0,6 Z" fill="{LAV}"/></marker></defs>')
p.append(f'<line x1="{bx+0.30*bw:.0f}" y1="{by+0.70*bh:.0f}" x2="{bx+0.66*bw:.0f}" y2="{by+0.34*bh:.0f}" '
         f'stroke="{LAV}" stroke-width="2.5" marker-end="url(#a)"/>')
p.append(f'<text x="{bx+bw+14}" y="{by+30}" fill="{GRN}" font-size="11">● positive</text>')
p.append(f'<text x="{bx+bw+14}" y="{by+50}" fill="{RED}" font-size="11">● negative</text>')
p.append(f'<text x="{bx+bw+14}" y="{by+86}" fill="{INK}" font-size="11">- - - the probe:</text>')
p.append(f'<text x="{bx+bw+14}" y="{by+104}" fill="{MUT}" font-size="10">one straight cut that</text>')
p.append(f'<text x="{bx+bw+14}" y="{by+118}" fill="{MUT}" font-size="10">separates the classes</text>')
p.append(f'<text x="{bx+bw+14}" y="{by+150}" fill="{LAV}" font-size="11">→ the concept</text>')
p.append(f'<text x="{bx+bw+14}" y="{by+168}" fill="{LAV}" font-size="11">   direction (a dial</text>')
p.append(f'<text x="{bx+bw+14}" y="{by+186}" fill="{LAV}" font-size="11">   you can steer)</text>')
p.append(f'<text x="{bx}" y="{by+bh+26}" fill="{MUT}" font-size="10">each dot = one sentence’s hidden state, squished to 2-D. '
         f'a straight line splits them → “linearly decodable.”</text>')

# ---------- Panel B: folds ----------
fy = 96
p.append(f'<text x="500" y="74" fill="{TEAL}" font-size="13">2 · FOLDS  —  did the probe LEARN it, or MEMORISE?</text>')
def strip(x0, y0, classes, test_idx, label, ok):
    sq = 22
    for i, c in enumerate(classes):
        col = GRN if c else RED
        p.append(f'<rect x="{x0+i*sq}" y="{y0}" width="{sq-3}" height="{sq-3}" rx="3" fill="{col}" opacity="0.9"/>')
    tx = x0 + test_idx[0] * sq
    tw = (test_idx[-1] - test_idx[0] + 1) * sq
    p.append(f'<rect x="{tx-2}" y="{y0-4}" width="{tw}" height="{sq+5}" fill="none" stroke="{INK}" stroke-width="2" rx="4"/>')
    p.append(f'<text x="{tx+tw/2:.0f}" y="{y0-9}" fill="{INK}" font-size="9" text-anchor="middle">test fold</text>')
    p.append(f'<text x="{x0}" y="{y0+sq+18}" fill="{"#7ee0d0" if ok else "#e06a6a"}" font-size="10">{label}</text>')
order = [True]*6 + [False]*6
shuf = [True, False, True, True, False, False, True, False, False, True, False, True]
strip(500, fy+28, order, [0, 1, 2], "sorted data → test fold is ALL green → meaningless score", False)
strip(500, fy+120, shuf, [0, 1, 2], "shuffled first → test fold is mixed → honest score", True)
p.append(f'<text x="500" y="{fy+212}" fill="{MUT}" font-size="10">train on the rest, score on the boxed “test fold,” rotate, average.</text>')
p.append(f'<text x="500" y="{fy+230}" fill="{MUT}" font-size="10">testing on held-out data is how you catch a probe that just</text>')
p.append(f'<text x="500" y="{fy+248}" fill="{MUT}" font-size="10">memorised the examples (“overfitting”) instead of the pattern.</text>')
p.append(f'<rect x="500" y="{fy+268}" width="356" height="44" rx="6" fill="#171a2e"/>')
p.append(f'<text x="514" y="{fy+288}" fill="{LAV}" font-size="10">the bug I just fixed: my data was sorted, so folds were</text>')
p.append(f'<text x="514" y="{fy+304}" fill="{LAV}" font-size="10">all-one-class. shuffling moved “number” 50% → 62%.</text>')

p.append('</svg>')
svg = "\n".join(p)
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "explainer.html")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    f.write('<!doctype html><html><head><meta charset="utf-8"><title>Clozn concepts</title>'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inconsolata:wght@400;600&display=swap">'
            f'<style>body{{background:{BG};margin:0;padding:18px}}</style></head><body>{svg}</body></html>')
print("wrote", out)
