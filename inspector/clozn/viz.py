"""
clozn.viz — the browser views (Phase-1). Hand-rolled SVG (this machine's Application Control
policy blocks matplotlib's compiled backend), dark theme, one self-contained HTML per view:

  render_state_film   the Watch cockpit: logit-lens "thought" per token + per-layer write heatmap
  render_probe_panel  Probe + Verify: decodability headline + causal dose-response curve
  render_memory_card  a before/after card (persist demo; Snapshot-Diff-Edit side-by-side)
  render_dashboard    stacks the above into one Inspector page

Each public renderer wraps an inner _*_svg builder in the shared _page shell, so the dashboard
can compose the raw SVGs without re-deriving anything.
"""
from __future__ import annotations

import html as _html

import numpy as np

from .spine import StateStep

_BG = "#0f1220"


def _esc(s: str) -> str:
    return _html.escape(str(s)).replace(" ", "·").replace("\n", "⏎")


def _page(body: str, title: str) -> str:
    return ('<!doctype html><html><head><meta charset="utf-8">'
            f'<title>{_esc(title)}</title>'
            '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inconsolata:wght@400;600&display=swap">'
            f'<style>body{{background:{_BG};margin:0;padding:18px;font-family:Inconsolata,ui-monospace,monospace}}</style>'
            f'</head><body>{body}</body></html>')


# ---------------- Watch ----------------

def _film_svg(steps: list[StateStep], component: str, title: str, subtitle: str) -> str:
    toks = [s.meta.get("token", "") for s in steps]
    preds = [s.meta.get("top1", "") for s in steps]
    M = np.stack([np.linalg.norm(s.state[component][0], axis=0) for s in steps])   # [T, L]
    prev = np.concatenate([np.zeros((1, M.shape[1])), M[:-1]], axis=0)
    Dn = np.abs(M - prev)
    Dn = Dn / (Dn.max(axis=0, keepdims=True) + 1e-9)
    T, L = M.shape

    cw, rh, top = 56, 15, 96
    W, H = 44 + T * cw + 16, top + L * rh + 42
    x0 = 44
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{H}" fill="{_BG}"/>',
         f'<text x="14" y="24" fill="#e6e9f5" font-size="14">{_esc(title)}</text>',
         f'<text x="14" y="42" fill="#8a90b3" font-size="11">{_esc(subtitle)}</text>',
         f'<text x="14" y="64" fill="#5b6090" font-size="9">token</text>',
         f'<text x="14" y="82" fill="#5b6090" font-size="9">thinks→</text>']
    for t in range(T):
        cx = x0 + t * cw + cw / 2
        p.append(f'<text x="{cx:.0f}" y="64" fill="#7ee0d0" font-size="11" text-anchor="middle">{_esc(toks[t])[:8]}</text>')
        p.append(f'<text x="{cx:.0f}" y="82" fill="#c9a0ff" font-size="10" text-anchor="middle">{_esc(preds[t])[:8]}</text>')
    for l in range(L):
        y = top + l * rh
        p.append(f'<text x="{x0-6}" y="{y+rh-3}" fill="#5b6090" font-size="8" text-anchor="end">L{l}</text>')
        for t in range(T):
            v = float(Dn[t, l])
            col = f'rgb({int(18+v*34)},{int(28+v*186)},{int(38+v*156)})'
            p.append(f'<rect x="{x0+t*cw}" y="{y}" width="{cw-1}" height="{rh-1}" fill="{col}"/>')
    p.append(f'<text x="14" y="{H-14}" fill="#8a90b3" font-size="10">'
             f'rows = layers · cols = tokens · brightness = how strongly this token wrote the recurrent memory '
             f'(“{component}”) at that layer</text>')
    p.append('</svg>')
    return "\n".join(p)


def render_state_film(steps: list[StateStep], component: str = "att_num",
                      title: str = "Clozn · Watch", subtitle: str = "") -> str:
    return _page(_film_svg(steps, component, title, subtitle), title)


# ---------------- Probe + Verify ----------------

def _probe_svg(alphas, scores, acc, verify, concept, title, subtitle) -> str:
    W, H, pad = 600, 340, 52
    lo, hi = min(scores), max(scores)
    amin, amax = min(alphas), max(alphas)
    def X(a): return pad + (a - amin) / (amax - amin + 1e-9) * (W - 2 * pad)
    def Y(s): return H - pad - (s - lo) / (hi - lo + 1e-9) * (H - 2 * pad)
    pts = " ".join(f"{X(a):.1f},{Y(s):.1f}" for a, s in zip(alphas, scores))
    mono = all(scores[i] <= scores[i + 1] + 1e-6 for i in range(len(scores) - 1))
    verdict = "CAUSAL" if (verify.get("causal") and mono) else "decodable, weak/non-causal"
    vcol = "#7ee0d0" if verdict == "CAUSAL" else "#c9a0ff"
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{H}" fill="{_BG}"/>',
         f'<text x="18" y="28" fill="#e6e9f5" font-size="14">{_esc(title)} — {_esc(concept)}</text>',
         f'<text x="18" y="46" fill="#8a90b3" font-size="11">{_esc(subtitle)}</text>',
         f'<text x="18" y="68" fill="#5b6090" font-size="10">decodable</text>',
         f'<text x="18" y="84" fill="#7ee0d0" font-size="13">{acc*100:.0f}%</text>',
         f'<text x="92" y="68" fill="#5b6090" font-size="10">causal Δ</text>',
         f'<text x="92" y="84" fill="{vcol}" font-size="13">{verify.get("delta", 0):+.3f}</text>',
         f'<text x="176" y="68" fill="#5b6090" font-size="10">verdict</text>',
         f'<text x="176" y="84" fill="{vcol}" font-size="13">{verdict}</text>',
         f'<line x1="{pad}" y1="{Y(0.5):.1f}" x2="{W-pad}" y2="{Y(0.5):.1f}" '
         f'stroke="#2a2f4a" stroke-width="1" stroke-dasharray="3 3"/>',
         f'<text x="{W-pad+2}" y="{Y(0.5)+3:.1f}" fill="#5b6090" font-size="9">neutral</text>',
         f'<line x1="{X(0):.1f}" y1="{pad}" x2="{X(0):.1f}" y2="{H-pad}" stroke="#2a2f4a" stroke-width="1"/>',
         f'<polyline points="{pts}" fill="none" stroke="{vcol}" stroke-width="2"/>']
    for a, s in zip(alphas, scores):
        p.append(f'<circle cx="{X(a):.1f}" cy="{Y(s):.1f}" r="3" fill="#c9a0ff"/>')
    p.append(f'<text x="{W//2}" y="{H-16}" fill="#5b6090" font-size="10" text-anchor="middle">'
             f'steer the “{_esc(concept)}” state direction  (− … +) →   ·   y = model’s own P(pos)/(pos+neg)</text>')
    p.append('</svg>')
    return "\n".join(p)


def render_probe_panel(alphas: list[float], scores: list[float], acc: float,
                       verify: dict, *, concept: str = "sentiment",
                       title: str = "Clozn · Probe + Verify", subtitle: str = "") -> str:
    return _page(_probe_svg(alphas, scores, acc, verify, concept, title, subtitle), title)


# ---------------- Memory card ----------------

def _card_svg(prompt, rows, badge, title, subtitle) -> str:
    W = 720
    H = 132 + len(rows) * 56
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{H}" fill="{_BG}"/>',
         f'<text x="20" y="30" fill="#e6e9f5" font-size="14">{_esc(title)}</text>',
         f'<text x="20" y="48" fill="#8a90b3" font-size="11">{_esc(subtitle)}</text>']
    if badge:
        bw = 12 + len(badge) * 7
        p += [f'<rect x="{W-bw-20}" y="18" width="{bw}" height="20" rx="5" fill="#16341f"/>',
              f'<text x="{W-bw/2-20:.0f}" y="32" fill="#7ee0d0" font-size="10" text-anchor="middle">{_esc(badge)}</text>']
    p.append(f'<text x="20" y="78" fill="#5b6090" font-size="11">prompt</text>')
    p.append(f'<text x="78" y="78" fill="#c9c9e0" font-size="13">{_esc(prompt)}</text>')
    for i, (label, text, col) in enumerate(rows):
        y = 104 + i * 56
        p += [f'<rect x="20" y="{y}" width="{W-40}" height="44" rx="8" fill="#171a2e"/>',
              f'<text x="34" y="{y+18}" fill="{col}" font-size="10">{_esc(label)}</text>',
              f'<text x="34" y="{y+36}" fill="#e6e9f5" font-size="14">{_esc(text)}</text>']
    p.append('</svg>')
    return "".join(p)


def render_memory_card(prompt: str, rows: list[tuple[str, str, str]], *,
                       badge: str = "", title: str = "Clozn · Persisted Memory",
                       subtitle: str = "") -> str:
    return _page(_card_svg(prompt, rows, badge, title, subtitle), title)


# ---------------- Associative memory shelf ----------------

def _shelf_svg(query, matches, title, subtitle) -> str:
    """matches: list of (name, similarity, note) pre-sorted best-first."""
    W = 720
    rowh = 40
    H = 116 + len(matches) * rowh + 16
    sims = [s for _, s, _ in matches] or [0.0]
    lo, hi = min(sims), max(sims)
    bx, bw = 250, 330                                   # bar origin / max width
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{H}" fill="{_BG}"/>',
         f'<text x="20" y="30" fill="#e6e9f5" font-size="14">{_esc(title)}</text>',
         f'<text x="20" y="48" fill="#8a90b3" font-size="11">{_esc(subtitle)}</text>',
         f'<text x="20" y="84" fill="#5b6090" font-size="11">current state reminds it of ↓   '
         f'(query: <tspan fill="#c9c9e0">{_esc(query)}</tspan>)</text>']
    for i, (name, sim, note) in enumerate(matches):
        y = 104 + i * rowh
        frac = (sim - lo) / (hi - lo + 1e-9)
        w = 10 + frac * bw
        col = "#7ee0d0" if i == 0 else "#5b6090"
        p += [f'<text x="20" y="{y+18}" fill="{col}" font-size="13">{_esc(name)}</text>',
              f'<text x="20" y="{y+32}" fill="#5b6090" font-size="9">{_esc(note)[:34]}</text>',
              f'<rect x="{bx}" y="{y+4}" width="{w:.0f}" height="20" rx="4" fill="{col}" opacity="{0.85 if i==0 else 0.45}"/>',
              f'<text x="{bx+w+8:.0f}" y="{y+19}" fill="{col}" font-size="11">{sim:+.3f}</text>']
    p.append(f'<text x="20" y="{H-14}" fill="#8a90b3" font-size="10">'
             f'similarity = shelf-centered cosine of the recurrent state · top match in teal</text>')
    p.append('</svg>')
    return "\n".join(p)


def render_memory_shelf(query: str, matches: list[tuple[str, float, str]], *,
                        title: str = "Clozn · Associative Memory", subtitle: str = "") -> str:
    return _page(_shelf_svg(query, matches, title, subtitle), title)


# ---------------- Per-token feature film ----------------

def _featfilm_svg(toks, rows, M, title, subtitle) -> str:
    """rows: list of (name, +label, -label); M[c,t] signed lean. Per-row max-abs normalized."""
    C, T = M.shape
    norm = M / (np.abs(M).max(axis=1, keepdims=True) + 1e-9)
    cw, rh, top, x0 = 52, 30, 118, 150
    W, H = x0 + T * cw + 16, top + C * rh + 40
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{H}" fill="{_BG}"/>',
         f'<text x="16" y="28" fill="#e6e9f5" font-size="14">{_esc(title)}</text>',
         f'<text x="16" y="46" fill="#8a90b3" font-size="11">{_esc(subtitle)}</text>',
         f'<text x="16" y="{top-12}" fill="#5b6090" font-size="9">feature ↓   token →</text>']
    for t in range(T):
        cx = x0 + t * cw + cw / 2
        p.append(f'<text x="{cx:.0f}" y="{top-2}" fill="#c9c9e0" font-size="10" text-anchor="middle">{_esc(toks[t])[:7]}</text>')
    for c in range(C):
        name, pl, nl = rows[c]
        y = top + c * rh
        p.append(f'<text x="{x0-10}" y="{y+rh-10:.0f}" fill="#e6e9f5" font-size="11" text-anchor="end">{_esc(name)}</text>')
        p.append(f'<text x="{x0-10}" y="{y+rh-1:.0f}" fill="#5b6090" font-size="7" text-anchor="end">{_esc(pl)} / {_esc(nl)}</text>')
        for t in range(T):
            v = float(norm[c, t])
            a = abs(v)
            if v >= 0:                                   # + pole = teal
                col = f'rgb({int(25+a*(126-25))},{int(28+a*(224-28))},{int(46+a*(208-46))})'
            else:                                        # − pole = coral
                col = f'rgb({int(25+a*(224-25))},{int(28+a*(120-28))},{int(46+a*(110-46))})'
            p.append(f'<rect x="{x0+t*cw}" y="{y}" width="{cw-1.5}" height="{rh-1.5}" fill="{col}"/>')
    p.append(f'<text x="16" y="{H-14}" fill="#8a90b3" font-size="10">'
             f'teal = leans to the first pole · coral = the second · dark = neutral · normalized per row</text>')
    p.append('</svg>')
    return "\n".join(p)


def render_feature_film(toks, rows, M, *, title="Clozn · Features lit up",
                        subtitle: str = "") -> str:
    return _page(_featfilm_svg(toks, rows, M, title, subtitle), title)


# ---------------- Concept atlas ----------------

def _atlas_svg(cards, title, subtitle) -> str:
    """cards: list of objects with .name .decodability .causal .delta .pos_label .neg_label"""
    W = 760
    rowh = 46
    H = 132 + len(cards) * rowh + 16
    bx, bw = 300, 360                                   # bar origin / full width = chance..100%
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{H}" fill="{_BG}"/>',
         f'<text x="20" y="30" fill="#e6e9f5" font-size="14">{_esc(title)}</text>',
         f'<text x="20" y="48" fill="#8a90b3" font-size="11">{_esc(subtitle)}</text>',
         f'<text x="20" y="82" fill="#5b6090" font-size="10">concept</text>',
         f'<text x="{bx}" y="82" fill="#5b6090" font-size="10">linear decodability  (chance 50% ——— 100%)</text>',
         # chance baseline + 100% guide
         f'<line x1="{bx}" y1="92" x2="{bx}" y2="{H-40}" stroke="#2a2f4a" stroke-width="1"/>',
         f'<line x1="{bx+bw}" y1="92" x2="{bx+bw}" y2="{H-40}" stroke="#20243c" stroke-width="1" stroke-dasharray="2 3"/>']
    for i, c in enumerate(cards):
        y = 104 + i * rowh
        frac = max(0.0, (c.decodability - 0.5) / 0.5)
        w = frac * bw
        if c.causal is True:
            col, tag = "#7ee0d0", f"causal ✓  Δ={c.delta:+.3f}"
        elif c.causal is False:
            col, tag = "#8a90b3", "decoded · not causal"
        else:
            col, tag = "#c9a0ff", "decoded · causal untested"
        p += [f'<text x="20" y="{y+20}" fill="#e6e9f5" font-size="13">{_esc(c.name)}</text>',
              f'<text x="20" y="{y+34}" fill="#5b6090" font-size="9">{_esc(c.pos_label)} vs {_esc(c.neg_label)}</text>',
              f'<rect x="{bx}" y="{y+6}" width="{max(w,2):.0f}" height="22" rx="4" fill="{col}" opacity="0.8"/>',
              f'<text x="{bx+max(w,2)+8:.0f}" y="{y+22}" fill="{col}" font-size="12">{c.decodability*100:.0f}%</text>',
              f'<text x="{bx}" y="{y+22}" fill="#0f1220" font-size="10" font-weight="600"> {_esc(tag)}</text>' if w > 120 else
              f'<text x="{bx+max(w,2)+52:.0f}" y="{y+22}" fill="{col}" font-size="9">{_esc(tag)}</text>']
    p.append(f'<text x="20" y="{H-14}" fill="#8a90b3" font-size="10">'
             f'teal = patched-and-verified causal · lavender = readable but causality untested (we don’t claim what we didn’t test)</text>')
    p.append('</svg>')
    return "\n".join(p)


def render_concept_atlas(cards, *, title: str = "Clozn · Concept Atlas",
                         subtitle: str = "") -> str:
    return _page(_atlas_svg(cards, title, subtitle), title)


# ---------------- Denoise film (diffusion substrate) ----------------

def _denoise_svg(steps: list[StateStep], prompt_len: int, title: str, subtitle: str) -> str:
    boards = [s.state["board"] for s in steps]
    fills = [s.state["filled"] for s in steps]
    confs = [s.state["conf"] for s in steps]
    counts = [s.meta.get("n_committed", 0) for s in steps]
    T = len(steps)
    n = len(boards[0])
    cw = rh = 16
    x0, top = 86, 92
    W, H = x0 + n * cw + 16, top + T * rh + 46
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{H}" fill="{_BG}"/>',
         f'<text x="14" y="26" fill="#e6e9f5" font-size="14">{_esc(title)}</text>',
         f'<text x="14" y="44" fill="#8a90b3" font-size="11">{_esc(subtitle)}</text>',
         f'<text x="14" y="{top-10}" fill="#5b6090" font-size="9">pass ↓ · position →</text>']
    prev_fill = np.zeros(n)
    for t in range(T):
        y = top + t * rh
        p.append(f'<text x="{x0-10}" y="{y+rh-4}" fill="#5b6090" font-size="9" text-anchor="end">t{t+1}</text>')
        for i in range(n):
            x = x0 + i * cw
            if i < prompt_len:
                col = "#3a4a7a"                                 # the given prompt
            elif fills[t][i] > 0.5:
                c = float(confs[t][i])
                col = f'rgb({int(18+c*30)},{int(30+c*180)},{int(40+c*150)})'   # filled, brighter = surer
            else:
                col = "#191c2e"                                 # still masked
            newly = fills[t][i] > 0.5 and prev_fill[i] <= 0.5 and i >= prompt_len
            stroke = ' stroke="#c9a0ff" stroke-width="1.5"' if newly else ''
            p.append(f'<rect x="{x}" y="{y}" width="{cw-1.5}" height="{rh-1.5}" fill="{col}"{stroke}/>')
        p.append(f'<text x="{x0+n*cw+6}" y="{y+rh-4}" fill="#7ee0d0" font-size="9">+{counts[t]}</text>')
        prev_fill = fills[t]
    p.append(f'<text x="14" y="{H-16}" fill="#8a90b3" font-size="10">'
             f'blue = prompt · teal = filled (brighter = higher confidence) · dark = still masked · '
             f'purple outline = filled THIS pass (many at once = parallel denoising)</text>')
    p.append('</svg>')
    return "\n".join(p)


def render_denoise_film(steps: list[StateStep], prompt_len: int = 0,
                        title: str = "Clozn · Watch (diffusion)", subtitle: str = "") -> str:
    return _page(_denoise_svg(steps, prompt_len, title, subtitle), title)


# ---------------- Diffusion trajectory features ----------------

def _traj_svg(feats, profiles, n_steps, title, subtitle) -> str:
    """feats: Features; profiles[i]: mean activation of feats[i] at each denoising step (len n_steps).
    Shows each feature's top tokens + a sparkline of WHEN it fires across the denoise trajectory."""
    import numpy as _np
    W, rowh = 860, 46
    H = 116 + len(feats) * rowh + 16
    sx, sw = 600, 230                                    # sparkline origin / width
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{H}" fill="{_BG}"/>',
         f'<text x="20" y="30" fill="#e6e9f5" font-size="14">{_esc(title)}</text>',
         f'<text x="20" y="48" fill="#8a90b3" font-size="11">{_esc(subtitle)}</text>',
         f'<text x="20" y="84" fill="#5b6090" font-size="10">feature · top tokens</text>',
         f'<text x="{sx}" y="84" fill="#5b6090" font-size="10">when it fires (denoise step 1→{n_steps})</text>']
    for i, (f, prof) in enumerate(zip(feats, profiles)):
        y = 96 + i * rowh
        prof = _np.asarray(prof, dtype=float)
        pk = int(prof.argmax()) if prof.size else 0
        phase = "early" if pk <= (n_steps - 1) * 0.34 else ("late" if pk >= (n_steps - 1) * 0.66 else "mid")
        col = {"early": "#c9a0ff", "late": "#7ee0d0", "mid": "#8a90b3"}[phase]
        p += [f'<text x="20" y="{y+18}" fill="{col}" font-size="12">f{f.idx} · {phase}@{pk+1}</text>',
              f'<text x="20" y="{y+33}" fill="#5b6090" font-size="8">fires {f.fires_on*100:.0f}%</text>']
        x = 120
        seen = []
        for t in f.top_tokens:
            s = t.strip()
            if not s or s in seen:
                continue
            seen.append(s)
            w = 12 + len(s) * 7.0
            if x + w > sx - 12:
                break
            p += [f'<rect x="{x:.0f}" y="{y+6}" width="{w:.0f}" height="20" rx="4" fill="#171a2e"/>',
                  f'<text x="{x+w/2:.0f}" y="{y+20}" fill="#e6e9f5" font-size="10" text-anchor="middle">{_esc(s)[:9]}</text>']
            x += w + 5
        # sparkline: one bar per denoising step, height = normalized mean activation
        mx = float(prof.max()) or 1.0
        bw = sw / max(n_steps, 1)
        for t in range(len(prof)):
            hh = (prof[t] / mx) * (rowh - 14)
            p.append(f'<rect x="{sx+t*bw:.0f}" y="{y+rowh-8-hh:.0f}" width="{bw-2:.0f}" height="{max(hh,1):.0f}" '
                     f'rx="1" fill="{col}" opacity="0.85"/>')
    p.append(f'<text x="20" y="{H-14}" fill="#8a90b3" font-size="10">'
             f'purple = fires EARLY in denoising (canvas still masked) · teal = fires LATE (content resolved)</text>')
    p.append('</svg>')
    return "\n".join(p)


def render_trajectory_features(feats, profiles, n_steps, *,
                               title="Clozn · Diffusion Trajectory Features", subtitle="") -> str:
    return _page(_traj_svg(feats, profiles, n_steps, title, subtitle), title)


# ---------------- Discovered features ----------------

_THEME_COL = {"color": "#7ee0d0", "number": "#c9a0ff", "emotion": "#e06a6a",
              "place": "#6aa0e0", "animal": "#6ad08a", "food": "#e0c46a", "": "#8a90b3"}


def _discovered_svg(features, title, subtitle) -> str:
    """features: objects with .idx .kind .top_tokens .fires_on .theme .purity"""
    W = 760
    rowh = 44
    H = 116 + len(features) * rowh + 16
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{H}" fill="{_BG}"/>',
         f'<text x="20" y="30" fill="#e6e9f5" font-size="14">{_esc(title)}</text>',
         f'<text x="20" y="48" fill="#8a90b3" font-size="11">{_esc(subtitle)}</text>',
         f'<text x="20" y="84" fill="#5b6090" font-size="10">each row = a feature the model uses, '
         f'NAMED by us only after the fact from its top-activating tokens →</text>']
    for i, f in enumerate(features):
        y = 96 + i * rowh
        coherent = f.purity >= 0.6
        col = _THEME_COL.get(f.theme if coherent else "", "#8a90b3")
        label = f"{f.theme} {f.purity*100:.0f}%" if coherent else "mixed"
        p += [f'<rect x="20" y="{y}" width="6" height="{rowh-8}" rx="3" fill="{col}"/>',
              f'<text x="36" y="{y+16}" fill="{col}" font-size="12">f{f.idx} · {_esc(label)}</text>',
              f'<text x="36" y="{y+31}" fill="#5b6090" font-size="8">fires {f.fires_on*100:.0f}%</text>']
        # top tokens as chips
        x = 168
        seen = []
        for t in f.top_tokens:
            s = t.strip()
            if not s or s in seen:
                continue
            seen.append(s)
            w = 12 + len(s) * 7.5
            p += [f'<rect x="{x:.0f}" y="{y+6}" width="{w:.0f}" height="22" rx="5" fill="#171a2e"/>',
                  f'<text x="{x+w/2:.0f}" y="{y+21}" fill="#e6e9f5" font-size="11" text-anchor="middle">{_esc(s)[:10]}</text>']
            x += w + 6
            if x > W - 70:
                break
    p.append(f'<text x="20" y="{H-14}" fill="#8a90b3" font-size="10">'
             f'discovered unsupervised by a sparse autoencoder on RWKV state · colored bar = it matches a seeded theme</text>')
    p.append('</svg>')
    return "\n".join(p)


def render_discovered_features(features, *, title="Clozn · Discovered Features",
                              subtitle: str = "") -> str:
    return _page(_discovered_svg(features, title, subtitle), title)


# ---------------- Dashboard ----------------

def render_dashboard(panels: list[tuple[str, str]], *,
                     title: str = "Clozn · Inspector", subtitle: str = "") -> str:
    """Stack labeled SVG panels (from the _*_svg builders) into one Inspector page."""
    blocks = [f'<div style="color:#e6e9f5;font-size:20px;margin:4px 0 2px">{_esc(title)}</div>',
              f'<div style="color:#8a90b3;font-size:12px;margin-bottom:8px">{_esc(subtitle)}</div>']
    for label, svg in panels:
        blocks.append('<div style="color:#7ee0d0;font-size:12px;letter-spacing:.06em;'
                      'margin:22px 0 6px;border-top:1px solid #20243c;padding-top:14px">'
                      + _esc(label) + '</div>')
        blocks.append(svg)
    return _page("".join(blocks), title)
