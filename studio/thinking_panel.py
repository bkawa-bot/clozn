"""
thinking_panel.py -- the SECOND WINDOW of the Clozn instrument: "what is it thinking".

A real, honest, beautiful READ of a frozen local model's interior as it processes a
sentence. Where the first window (memory_window.py) WRITES an editable memory and watches
behaviour move, this window only LISTENS: it projects the residual stream at every token
onto a fixed set of NAMED concept directions and renders the trace of which concepts are
"present in the state" word by word -- a legible picture of the model's passing thoughts
in named terms (fear, animals, money, ...).

WHAT IT IS (a probe, not a decision; load-bearing honesty).
  We build NAMED concept directions on a FROZEN GPT-2-small by diff-in-means -- exactly the
  p18_conceptmem basis-building, reused: for each concept, direction d_c = mean(resid_post @ L
  over positive example tokens) - mean(over contrast tokens). Then, as the model reads a demo
  prompt, we capture resid_post at L for every token and PROJECT it onto each unit direction.
  That projection is a READ of how much the named concept is present in the state at that
  position. It is a probe; it is NOT a causal claim that the model has "decided" anything, and
  it is NOT what the model will output. We frame it as "which concepts are present," never
  "what it chose."

HONESTY (the null, shown, never hidden).
  A raw projection is meaningless on its own: every direction has SOME projection, and tokens
  with larger residual norm project more onto everything. So for each concept we compute, at
  each token, a RANDOM-DIRECTION NULL: many random directions of EQUAL NORM to the concept's
  raw direction, projected at the SAME position. The null's mean+spread is the "nothing is
  really there" band for that token. A concept only LIGHTS UP where its real projection exceeds
  the null band; we report and render the activation as z = (proj - null_mean) / null_std, i.e.
  HOW MANY SIGMA above an equal-norm random direction it sits. We also center each concept by
  its own mean projection over a neutral corpus, so a concept that is just "always slightly on"
  (a frequency / norm artifact) sits near zero. The null is drawn faintly on the page beside
  every lane: a concept that doesn't clear it is honestly shown as not lit.

THE ARTIFACT.
  A single self-contained HTML page in the Planet Maiko palette (inline CSS, a little vanilla
  JS for stars; no external deps). The trace is rendered as softly glowing concept LANES -- one
  horizontal lane per named concept across the tokens of the sentence; each token-cell brightens
  where that concept is active (z above the null), with the faint null line behind it. Otherworldly,
  soft, glowing -- not a corporate dashboard.

REUSE: concept basis-building lifted from inspector/spikes/p18_conceptmem.py (CONCEPTS,
mean_resid_over_texts, build_basis). Palette + page craft follow inspector/demo/memory_window.py
and research/sidecar.py.

ISOLATED ENV: runs in C:\\Users\\brigi\\src\\clozn\\.venv-sae (transformer_lens + torch, CPU is
fine). GPT-2-small is 124M and cached -- no large download. The backbone is FROZEN throughout.

Usage (from inspector/, .venv-sae python):
    python demo/thinking_panel.py                 # default: layer 7, writes inspector/runs/
    python demo/thinking_panel.py --layer 6 --seed 0
    python demo/thinking_panel.py --out some/dir/thinking_panel.html
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html as _html
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # this PC crashes on HF symlinks (WinError 1314)

import numpy as np                 # noqa: E402
import torch                       # noqa: E402
import torch.nn.functional as F    # noqa: E402

# Reuse the validated named-concept basis + diff-in-means builders from the p18 spike verbatim.
from spikes.p18_conceptmem import (  # noqa: E402
    CONCEPTS as P18_CONCEPTS,
    mean_resid_over_texts,
    build_basis,
)

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

# ----------------------------------------------------------------------------------------------------
# WHICH concepts to show, and a pretty colour/label per concept for the lanes. We pull these concepts
# straight out of the p18 basis (so the directions are the SAME validated diff-in-means dirs); the
# entries here only choose the display order, a human label, and a palette accent. 8 named concepts.
# ----------------------------------------------------------------------------------------------------
CONCEPT_DISPLAY = [
    ("animals",  "animals",     "#C4F542"),   # Toxic Lime
    ("fear",     "fear",        "#FF6FAF"),   # Neon Pink
    ("colors",   "color",       "#6FE0E8"),   # Frozen Cyan
    ("money",    "money",       "#FFE66D"),   # Star Yellow
    ("formal",   "formal tone", "#C9A6FF"),   # Soft Lavender Glow
    ("past",     "past tense",  "#1FB5E5"),   # Electric Ice
    ("food",     "food",        "#FFB36F"),   # warm amber (in-family accent)
    ("question", "questions",   "#A0FFD6"),   # mint (in-family accent)
]

# Demo prompts chosen so DIFFERENT concepts light up (the whole point). Each is a short natural
# sentence the model reads token by token; we annotate why we picked it (for the page copy).
DEMO_PROMPTS = [
    {
        "text": "The frightened cat ran away from the snarling dog",
        "why":  "a scared animal scene -- we expect fear and animals to glow, money and formality dark",
    },
    {
        "text": "The banker counted the gold coins and signed the contract",
        "why":  "finance and formality -- money and formal tone should light, fear and food should not",
    },
    {
        "text": "Yesterday she cooked a red soup and asked what we wanted",
        "why":  "a mix -- past tense, food, color, and a question all brush through one sentence",
    },
]

# Which concepts we EXPECT to lead each prompt (only for the honest expectation note on the page;
# the actual lit set is measured, never hand-set).
EXPECTED = {
    0: ["fear", "animals"],
    1: ["money", "formal tone"],
    2: ["past tense", "food", "color", "questions"],
}


def load_model(device: str):
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# A small neutral corpus to CENTER each concept's projection. We subtract each concept's mean
# projection over these topic-free tokens so a concept that is merely "always a little on" (a
# frequency / residual-norm artifact) sits near zero, and only genuine, above-baseline presence
# shows. Reused style from p18's NEUTRAL_PROMPTS.
NEUTRAL_PROMPTS = [
    "The next thing I want to talk about is the",
    "When I opened the door, I saw the",
    "Let me tell you about the",
    "The most important part of the story is the",
    "After a while, they noticed the",
    "I think the answer has to do with the",
    "It was a normal day and nothing much happened.",
    "She walked along the path and looked around.",
]


@torch.no_grad()
def per_token_resid(model, text: str, layer: int):
    """resid_post at `layer` for every token of `text`, plus the human token strings.
    Returns (resid [seq', d_model] with BOS dropped, token_strs [seq']). We drop the BOS position
    (index 0): its residual is a ~30x-norm outlier that would swamp every projection."""
    toks = model.to_tokens(text)                                   # [1, seq] (prepends BOS)
    name = f"blocks.{layer}.hook_resid_post"
    _, cache = model.run_with_cache(toks, names_filter=name)
    resid = cache[name][0][1:]                                     # [seq-1, d_model], drop BOS
    strs = [model.to_string(t) for t in toks[0][1:]]               # matching token strings
    return resid, strs


@torch.no_grad()
def neutral_baseline(model, units, layer: int):
    """Mean projection of each UNIT concept direction over a neutral corpus -> [k]. Subtracted from
    every token's projection so the trace measures ABOVE-baseline presence, not absolute projection
    (which is partly a norm / token-frequency artifact). Honest normalization, not cherry-picking."""
    accum = torch.zeros(units.shape[0], device=units.device)
    cnt = 0
    for p in NEUTRAL_PROMPTS:
        resid, _ = per_token_resid(model, p, layer)               # [s, d_model]
        proj = resid @ units.T                                    # [s, k]
        accum = accum + proj.sum(0)
        cnt += proj.shape[0]
    return accum / max(cnt, 1)                                     # [k]


@torch.no_grad()
def null_stats_per_token(model, resid, raw_norms, n_null: int, generator):
    """For each token position and each concept, the RANDOM-DIRECTION null: project the token's
    residual onto `n_null` random unit directions scaled to the concept's RAW norm, and report the
    null mean+std PER TOKEN PER CONCEPT. Because a random unit dir has the same expected projection
    statistics regardless of which concept, we share one bank of random UNIT dirs and rescale by each
    concept's raw norm. Returns (null_mean [seq,k], null_std [seq,k]).

    This is the load-bearing honesty knob: a concept is only "present" where its real projection
    exceeds this equal-norm random band. The null isolates 'this token just has a big residual /
    this many dims' from 'this NAMED direction is genuinely active'."""
    d_model = resid.shape[-1]
    seq = resid.shape[0]
    k = raw_norms.shape[0]
    rand_units = F.normalize(torch.randn(n_null, d_model, generator=generator,
                                         device=resid.device), dim=-1)   # [n_null, d_model]
    proj_unit = resid @ rand_units.T                                     # [seq, n_null] (onto unit dirs)
    # per-token null mean/std for a UNIT random dir, then scale by each concept's raw norm.
    base_mean = proj_unit.mean(dim=1)                                    # [seq]
    base_std = proj_unit.std(dim=1)                                      # [seq]
    null_mean = base_mean[:, None] * raw_norms[None, :]                  # [seq, k]
    null_std = base_std[:, None] * raw_norms[None, :]                    # [seq, k]
    return null_mean, null_std


@torch.no_grad()
def read_trace(model, prompt_text, concepts, units, raw_norms, baseline_unit, layer,
               n_null, generator):
    """The READ for one prompt. For every token: project resid onto each UNIT concept dir, subtract
    the neutral baseline (centering), and z-score against the equal-norm random null at that token.

      z[token, c] = ( proj_unit[token,c] - baseline_unit[c] - null_mean_unit[token] )
                    / null_std_unit[token]

    where the null is computed on UNIT dirs (norm cancels out of the z-score; raw_norms only matter
    for reporting the raw-projection scale). z is in units of 'sigma above an equal-norm random
    direction': z<=0 -> not present; z>~2 -> clearly present above chance. Returns a dict with the
    token strings, the z matrix [seq,k], the raw centered projections, and the null band.
    """
    resid, token_strs = per_token_resid(model, prompt_text, layer)      # [seq,d], list
    seq = resid.shape[0]
    proj = resid @ units.T                                              # [seq, k] onto unit dirs
    proj_centered = proj - baseline_unit[None, :]                       # center by neutral baseline

    # Null on UNIT directions (so it is directly comparable to proj on unit dirs). We pass ones for
    # the norm here because units are unit-norm; raw_norms is reported separately for scale context.
    ones = torch.ones_like(raw_norms)
    null_mean, null_std = null_stats_per_token(model, resid, ones, n_null, generator)  # [seq,k]
    # center the null the same way (subtract baseline) so z compares like-for-like.
    null_mean_centered = null_mean - baseline_unit[None, :]
    z = (proj_centered - null_mean_centered) / (null_std + 1e-9)        # [seq, k] sigma above null

    return {
        "tokens": token_strs,
        "z": z.cpu().numpy(),                       # [seq, k] sigma above equal-norm random null
        "proj_centered": proj_centered.cpu().numpy(),
        "null_std": null_std.cpu().numpy(),         # [seq, k] the faint band (in unit-proj scale)
        "resid_norm": resid.norm(dim=-1).cpu().numpy(),  # [seq] per-token residual norm (context)
    }


# ====================================================================================================
# Build the demo: every number rendered is an ACTUAL read of the frozen model.
# ====================================================================================================
def build_demo(layer: int, device: str, seed: int, n_null: int, z_thresh: float):
    torch.manual_seed(seed)
    print(f"loading gpt2 (HookedTransformer) on {device} ...")
    model = load_model(device)
    d_model, nl = model.cfg.d_model, model.cfg.n_layers
    print(f"  d_model={d_model}  n_layers={nl}   read layer L={layer}")

    # ---- pick the concept subset from the p18 basis, preserving display order ----------------------
    by_name = {c["name"]: c for c in P18_CONCEPTS}
    names_internal = [n for (n, _disp, _col) in CONCEPT_DISPLAY]
    missing = [n for n in names_internal if n not in by_name]
    if missing:
        raise SystemExit(f"concepts not found in p18 basis: {missing}")
    concepts = [by_name[n] for n in names_internal]
    disp_labels = [disp for (_n, disp, _c) in CONCEPT_DISPLAY]
    colors = [c for (_n, _d, c) in CONCEPT_DISPLAY]
    k = len(concepts)
    print(f"  {k} named concepts: {disp_labels}")

    # ---- STEP 1: build the named diff-in-means basis at L (REUSED from p18) -------------------------
    print(f"\nSTEP 1 -- build named concept directions @ blocks.{layer}.hook_resid_post (diff-in-means)")
    dirs, units, norms = build_basis(model, concepts, layer)            # [k,d], [k,d], [k]
    cos = (units @ units.T).cpu().numpy()
    off = cos.copy(); np.fill_diagonal(off, np.nan)
    mean_abs_off = float(np.nanmean(np.abs(off)))
    for nm, nr in zip(disp_labels, norms.tolist()):
        print(f"    {nm:12} raw dir norm {nr:7.2f}")
    print(f"    mean |off-diag cosine| = {mean_abs_off:.3f}  (lower = more independently nameable)")

    # ---- STEP 2: neutral baseline for centering ----------------------------------------------------
    baseline_unit = neutral_baseline(model, units, layer)              # [k]
    print(f"    neutral-corpus baseline projection per concept (subtracted to remove the always-on "
          f"offset): " + " ".join(f"{nm[:4]}={float(b):+.2f}" for nm, b in zip(disp_labels, baseline_unit)))

    # ---- STEP 3: read the trace for each demo prompt -----------------------------------------------
    print(f"\nSTEP 2 -- read each demo prompt token by token (null = {n_null} equal-norm random dirs/token)")
    g = torch.Generator(device=model.cfg.device).manual_seed(seed)
    prompts_out = []
    for pi, prm in enumerate(DEMO_PROMPTS):
        tr = read_trace(model, prm["text"], concepts, units, norms, baseline_unit, layer, n_null, g)
        z = tr["z"]                                                    # [seq, k]
        tokens = tr["tokens"]
        # per-concept peak z over the sentence and which token peaked
        peak_z = z.max(axis=0)                                         # [k]
        peak_tok_idx = z.argmax(axis=0)                               # [k]
        # mean z over the sentence (a calmer "how present overall")
        mean_z = z.mean(axis=0)                                        # [k]
        lit = [j for j in range(k) if peak_z[j] >= z_thresh]
        lit_sorted = sorted(lit, key=lambda j: -peak_z[j])
        print(f"\n  PROMPT {pi+1}: \"{prm['text']}\"")
        print(f"    tokens: {[t for t in tokens]}")
        order = np.argsort(-peak_z)
        for j in order:
            star = "  <== LIT" if peak_z[j] >= z_thresh else ""
            pk_tok = tokens[peak_tok_idx[j]].strip() or tokens[peak_tok_idx[j]]
            print(f"      {disp_labels[j]:12} peak z={peak_z[j]:+5.2f} (@ '{pk_tok}')   "
                  f"mean z={mean_z[j]:+5.2f}{star}")
        lit_names = [disp_labels[j] for j in lit_sorted]
        print(f"    -> LIT (peak z >= {z_thresh}): {lit_names if lit_names else 'none'}")
        prompts_out.append({
            "text": prm["text"], "why": prm["why"],
            "tokens": tokens,
            "z": z,                                # [seq, k]
            "peak_z": peak_z, "mean_z": mean_z, "peak_tok_idx": peak_tok_idx,
            "lit": lit_sorted, "lit_names": lit_names,
            "expected": EXPECTED.get(pi, []),
            "resid_norm": tr["resid_norm"],
        })

    return {
        "model_name": "GPT-2-small (124M, frozen)", "layer": layer,
        "d_model": d_model, "n_layers": nl, "n_null": n_null, "z_thresh": z_thresh,
        "seed": seed,
        "concept_labels": disp_labels, "concept_colors": colors,
        "concept_internal": names_internal,
        "concept_norms": norms.cpu().numpy(),
        "cosine_mean_abs_off": mean_abs_off,
        "baseline_unit": baseline_unit.cpu().numpy(),
        "prompts": prompts_out,
        "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ====================================================================================================
# THE ARTIFACT: a single self-contained HTML page in the Planet Maiko palette. Inline CSS, no external
# deps, a little vanilla JS for the stars. Otherworldly, soft, glowing -- concept LANES across tokens.
# ====================================================================================================
# Planet Maiko palette
BG_DEEP   = "#0B0F2A"   # Deep Space Navy
BG_COSMIC = "#1A1F4A"   # Cosmic Indigo
BG_MID    = "#2A2250"   # Midnight Purple
PINK      = "#FF6FAF"   # Neon Pink
MAGENTA   = "#FF4D9D"   # Candy Magenta
ICE       = "#1FB5E5"   # Electric Ice
CYAN      = "#6FE0E8"   # Frozen Cyan
LIME      = "#C4F542"   # Toxic Lime
YELLOW    = "#FFE66D"   # Star Yellow
LAV       = "#C9A6FF"   # Soft Lavender Glow
WHITE     = "#F4F7FF"   # Soft White
GRAY      = "#A7B0C0"   # Cool Gray


def esc(s) -> str:
    return _html.escape(str(s))


def _hex_to_rgb(h: str):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgba(hex_color: str, a: float) -> str:
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r},{g},{b},{a:g})"


def render_html(demo: dict) -> str:
    labels = demo["concept_labels"]
    colors = demo["concept_colors"]
    z_thresh = demo["z_thresh"]
    k = len(labels)

    # ---- intensity mapping: z (sigma above null) -> a 0..1 glow. Below the null (z<=0) is dark; the
    # threshold sits partway up so a cell only really glows once it clears the null band. Soft, not
    # binary, so the trace reads as a gradient of presence.
    def glow(z: float) -> float:
        if z <= 0:
            return 0.0
        # map [0 .. ~ (z_thresh*2.2)] -> [0..1], gently clipped
        g = z / (z_thresh * 2.2)
        return float(max(0.0, min(1.0, g)))

    def disp_tok(t: str) -> str:
        s = t.strip()
        return s if s else "·"

    # ---------- one prompt block: the glowing concept lanes across tokens ---------------------------
    def prompt_block(p, idx):
        tokens = p["tokens"]
        z = p["z"]                       # [seq, k]
        seq = len(tokens)
        lit_names = p["lit_names"]
        peak_z = p["peak_z"]

        # token header row (the sentence, as cells aligned with the lane grid)
        tok_cells = "".join(
            f'<div class="tok-cell">{esc(disp_tok(t))}</div>' for t in tokens
        )

        # one lane per concept; each lane = a row of token cells whose glow = this concept's z there
        lanes = []
        for j in range(k):
            col = colors[j]
            cells = []
            for ti in range(seq):
                zz = float(z[ti, j])
                g = glow(zz)
                lit_cell = zz >= z_thresh
                # cell background glow scaled by intensity; lit cells get a ring + stronger shadow
                bg = rgba(col, 0.06 + 0.46 * g)
                style = (f"background:{bg};")
                if g > 0:
                    style += f"box-shadow:0 0 {int(2 + 20 * g)}px {rgba(col, 0.10 + 0.5 * g)};"
                cls = "cell"
                if lit_cell:
                    cls += " cell-lit"
                    style += f"border-color:{rgba(col, 0.75)};"
                # the faint NULL marker: a thin baseline line at the bottom of every cell, so the
                # "nothing is really there" floor is always visible behind the glow.
                title = f"{labels[j]} @ '{disp_tok(tokens[ti])}': z={zz:+.2f}σ vs equal-norm random null"
                cells.append(f'<div class="{cls}" style="{style}" title="{esc(title)}">'
                             f'<span class="nullline"></span></div>')
            pk = peak_z[j]
            on = pk >= z_thresh
            name_cls = "lane-name lane-on" if on else "lane-name"
            name_style = f"color:{col};" if on else ""
            peak_badge = (f'<span class="lane-peak" style="color:{col}">{pk:+.1f}σ</span>'
                          if on else f'<span class="lane-peak lane-peak-off">{pk:+.1f}σ</span>')
            lanes.append(
                f'<div class="lane">'
                f'  <div class="{name_cls}" style="{name_style}">'
                f'    <span class="lane-dot" style="background:{col};opacity:{0.95 if on else 0.28}"></span>'
                f'    {esc(labels[j])}{peak_badge}'
                f'  </div>'
                f'  <div class="lane-cells">{"".join(cells)}</div>'
                f'</div>'
            )

        lit_chips = "".join(
            f'<span class="lit-chip" style="border-color:{rgba(colors[labels.index(nm)],0.6)};'
            f'color:{colors[labels.index(nm)]}">{esc(nm)}</span>'
            for nm in lit_names
        ) or '<span class="lit-none">(nothing cleared the null)</span>'

        expected = ", ".join(p["expected"])
        return f'''
      <div class="prompt" style="--i:{idx}">
        <div class="prompt-head">
          <div class="prompt-sentence">&ldquo;{esc(p["text"])}&rdquo;</div>
          <div class="prompt-why">{esc(p["why"])}</div>
        </div>
        <div class="grid" style="--cols:{seq}">
          <div class="grid-corner">tokens &rarr;</div>
          <div class="tok-row">{tok_cells}</div>
          <div class="lanes">{"".join(lanes)}</div>
        </div>
        <div class="prompt-foot">
          <div class="lit-line"><span class="lit-lab">present in its state:</span> {lit_chips}</div>
          <div class="exp-line">expected: <i>{esc(expected)}</i> &middot;
            measured, not set &mdash; the glow is the read</div>
        </div>
      </div>'''

    prompt_blocks = "".join(prompt_block(p, i) for i, p in enumerate(demo["prompts"]))

    # ---------- the concept legend (the named basis) ------------------------------------------------
    legend = "".join(
        f'<div class="leg-item"><span class="leg-dot" style="background:{colors[j]};'
        f'box-shadow:0 0 12px {rgba(colors[j],0.6)}"></span>'
        f'<span class="leg-name">{esc(labels[j])}</span></div>'
        for j in range(k)
    )

    # ---------- assemble CSS ------------------------------------------------------------------------
    style = f"""
    :root {{
      --bg-deep:{BG_DEEP}; --bg-cosmic:{BG_COSMIC}; --bg-mid:{BG_MID};
      --pink:{PINK}; --magenta:{MAGENTA}; --ice:{ICE}; --cyan:{CYAN};
      --lime:{LIME}; --yellow:{YELLOW}; --lav:{LAV}; --white:{WHITE}; --gray:{GRAY};
    }}
    * {{ box-sizing:border-box; }}
    html,body {{ margin:0; padding:0; }}
    body {{
      background:
        radial-gradient(1100px 700px at 80% -10%, rgba(255,111,175,0.15), transparent 60%),
        radial-gradient(950px 650px at 6% 6%, rgba(31,181,229,0.13), transparent 58%),
        radial-gradient(820px 900px at 50% 116%, rgba(201,166,255,0.13), transparent 60%),
        linear-gradient(168deg, var(--bg-cosmic) 0%, var(--bg-deep) 55%, #070a20 100%);
      background-attachment:fixed;
      color:var(--white);
      font-family:'Segoe UI','Inter',system-ui,-apple-system,sans-serif;
      -webkit-font-smoothing:antialiased;
      min-height:100vh;
      padding:46px 22px 80px;
      line-height:1.5;
    }}
    .wrap {{ max-width:1060px; margin:0 auto; }}
    .star {{ position:fixed; border-radius:50%; background:var(--white); opacity:0;
      animation:twinkle 6s ease-in-out infinite; pointer-events:none; }}
    @keyframes twinkle {{ 0%,100%{{opacity:0}} 50%{{opacity:.5}} }}

    header {{ text-align:center; margin-bottom:14px; }}
    .eyebrow {{ letter-spacing:.32em; text-transform:uppercase; font-size:11px; font-weight:600;
      color:var(--cyan); opacity:.85; margin-bottom:14px; }}
    h1 {{ font-size:38px; line-height:1.15; margin:0 0 14px; font-weight:700;
      background:linear-gradient(96deg, var(--white) 8%, var(--lav) 46%, var(--cyan) 96%);
      -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
      text-shadow:0 0 38px rgba(201,166,255,0.25); }}
    .lede {{ font-size:16px; color:var(--gray); max-width:720px; margin:0 auto 8px; }}
    .lede b {{ color:var(--white); font-weight:600; }}
    .meta-line {{ font-size:12.5px; color:var(--gray); opacity:.82; margin-top:14px; }}
    .meta-line code {{ color:var(--cyan); background:rgba(31,181,229,0.10);
      padding:1px 7px; border-radius:6px; font-size:12px; }}

    .legend {{ display:flex; flex-wrap:wrap; gap:12px 18px; justify-content:center;
      margin:26px auto 4px; max-width:880px; padding:14px 18px; border-radius:16px;
      background:rgba(11,15,42,0.4); border:1px solid rgba(201,166,255,0.14); }}
    .leg-item {{ display:flex; align-items:center; gap:8px; font-size:13px; color:var(--gray); }}
    .leg-dot {{ width:11px; height:11px; border-radius:50%; flex:0 0 auto; }}
    .leg-name {{ color:var(--white); font-weight:500; }}

    .section {{ margin-top:38px; }}

    /* one prompt */
    .prompt {{ background:linear-gradient(150deg, rgba(42,34,80,0.6), rgba(11,15,42,0.5));
      border:1px solid rgba(201,166,255,0.16); border-radius:24px; padding:24px 26px 20px;
      margin-bottom:26px; backdrop-filter:blur(3px);
      box-shadow:0 16px 44px rgba(0,0,0,0.34), inset 0 1px 0 rgba(255,255,255,0.04);
      opacity:0; transform:translateY(16px);
      animation:rise .7s cubic-bezier(.2,.7,.2,1) forwards; animation-delay:calc(var(--i) * .16s + .1s); }}
    @keyframes rise {{ to {{ opacity:1; transform:none; }} }}
    .prompt-head {{ margin-bottom:18px; }}
    .prompt-sentence {{ font-size:21px; font-weight:600; color:var(--white);
      text-shadow:0 0 26px rgba(201,166,255,0.22); }}
    .prompt-why {{ font-size:13px; color:var(--gray); margin-top:6px; font-style:italic; }}

    /* the lane grid: a label column + a cells column that splits into `--cols` token columns */
    .grid {{ display:block; }}
    .grid-corner {{ display:none; }}
    .tok-row {{ display:grid; grid-template-columns:repeat(var(--cols), 1fr);
      gap:5px; margin-left:138px; margin-bottom:9px; }}
    .tok-cell {{ font-size:11.5px; color:var(--gray); text-align:center; font-weight:500;
      white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
      font-variant-ligatures:none; opacity:.92; padding-bottom:2px;
      border-bottom:1px solid rgba(255,255,255,0.06); }}

    .lanes {{ display:flex; flex-direction:column; gap:5px; }}
    .lane {{ display:grid; grid-template-columns:138px 1fr; align-items:center; gap:0; }}
    .lane-name {{ font-size:13px; color:var(--gray); display:flex; align-items:center; gap:7px;
      padding-right:10px; font-weight:500; }}
    .lane-on {{ font-weight:700; }}
    .lane-dot {{ width:9px; height:9px; border-radius:50%; flex:0 0 auto; }}
    .lane-peak {{ margin-left:auto; font-size:10.5px; font-variant-numeric:tabular-nums;
      font-weight:600; }}
    .lane-peak-off {{ color:var(--gray); opacity:.5; }}
    .lane-cells {{ display:grid; grid-template-columns:repeat(var(--cols), 1fr); gap:5px; }}
    .cell {{ position:relative; height:30px; border-radius:8px;
      border:1px solid rgba(255,255,255,0.05); transition:all .5s ease; }}
    .cell-lit {{ border-width:1px; }}
    .nullline {{ position:absolute; left:14%; right:14%; bottom:5px; height:1px;
      background:rgba(255,255,255,0.16); border-radius:1px; }}

    .prompt-foot {{ margin-top:16px; padding-top:14px; border-top:1px solid rgba(255,255,255,0.06); }}
    .lit-line {{ display:flex; align-items:center; flex-wrap:wrap; gap:8px; font-size:13px; }}
    .lit-lab {{ color:var(--gray); margin-right:2px; }}
    .lit-chip {{ font-size:12.5px; padding:3px 11px; border-radius:20px; font-weight:600;
      border:1px solid; background:rgba(255,255,255,0.03); }}
    .lit-none {{ color:var(--gray); font-style:italic; opacity:.7; }}
    .exp-line {{ font-size:11.5px; color:var(--gray); opacity:.7; margin-top:9px; }}
    .exp-line i {{ color:var(--lav); font-style:italic; }}

    .footer {{ margin-top:52px; text-align:center; font-size:12px; color:var(--gray); opacity:.72;
      line-height:1.7; }}
    .footer b {{ color:var(--lav); }}
    .honest {{ margin-top:14px; padding:18px 22px; border-radius:14px; font-size:12.5px;
      background:rgba(11,15,42,0.42); border:1px solid rgba(255,255,255,0.06); color:var(--gray);
      max-width:820px; margin-left:auto; margin-right:auto; text-align:left; line-height:1.65; }}
    .honest b {{ color:var(--cyan); }}
    .honest .k {{ color:var(--lav); }}

    @media (max-width:760px) {{
      h1 {{ font-size:28px; }}
      .tok-row {{ margin-left:96px; }}
      .lane {{ grid-template-columns:96px 1fr; }}
      .lane-name {{ font-size:11px; }}
      .tok-cell {{ font-size:9.5px; }}
      .cell {{ height:26px; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      .prompt {{ opacity:1 !important; transform:none !important; animation:none !important; }}
      .star {{ animation:none !important; }}
    }}
    """

    js = """
    (function(){
      var b=document.body, n=48;
      for(var i=0;i<n;i++){
        var s=document.createElement('div'); s.className='star';
        var sz=Math.random()*2+1;
        s.style.width=sz+'px'; s.style.height=sz+'px';
        s.style.left=(Math.random()*100)+'vw'; s.style.top=(Math.random()*100)+'vh';
        s.style.animationDelay=(Math.random()*6)+'s';
        s.style.animationDuration=(4+Math.random()*5)+'s';
        b.appendChild(s);
      }
    })();
    """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clozn &middot; what is it thinking</title>
<style>{style}</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="eyebrow">Clozn &middot; the second window</div>
    <h1>What is it thinking?</h1>
    <p class="lede">We read a <b>frozen {esc(demo["model_name"])}</b> as it takes in a sentence and
      light up which <b>named concepts</b> are present in its internal state at each word &mdash; a
      trace of its passing thoughts in plain terms. This is a <b>read</b>, a probe of the residual
      stream; <b>not</b> a claim about what the model decided or will say. A concept only glows where
      it genuinely rises above an <b>equal-norm random direction</b> (the faint floor in every cell).</p>
    <div class="meta-line">
      read at <code>blocks.{demo["layer"]}.hook_resid_post</code> &middot;
      directions = diff-in-means (positive vs contrast texts) &middot;
      glow = sigma above a {demo["n_null"]}-sample equal-norm random null &middot;
      lit threshold = <code>{z_thresh:g}&sigma;</code>
    </div>
  </header>

  <div class="legend">{legend}</div>

  <div class="section">
    {prompt_blocks}
  </div>

  <div class="footer">
    <div><b>Clozn</b> &mdash; a local runtime where the model's interior is legible.</div>
    <div class="honest">
      <b>What this is, honestly.</b> A <span class="k">read</span>, not a decision and not the
      output. For each of the {k} named concepts we build a direction by
      <span class="k">diff-in-means</span> on a frozen {esc(demo["model_name"])}: the average
      residual (at <code>blocks.{demo["layer"]}</code>) over positive example sentences minus the
      average over contrast sentences. As the model reads a prompt we project the residual at every
      token onto each (unit) concept direction. <b>The honesty knob is the null:</b> a raw projection
      is meaningless because tokens with a bigger residual project more onto <i>everything</i>, so at
      each token we also project onto <span class="k">{demo["n_null"]} random directions of equal
      norm</span> and report the activation as <b>z = how many &sigma; the real direction sits above
      that random band</b>. We additionally <span class="k">center</span> each concept by its mean
      projection over a neutral corpus, so a concept that is merely &ldquo;always slightly on&rdquo;
      (a frequency or norm artifact) sits near zero. A concept is called <i>present</i> only where its
      peak <b>z &ge; {z_thresh:g}&sigma;</b>; everything below the null is shown dark, and the faint
      null line is drawn behind every cell. Basis non-orthogonality (mean |off-diagonal cosine| =
      {demo["cosine_mean_abs_off"]:.2f}) means nearby concepts can co-activate &mdash; this is a probe
      of <i>what is present</i>, not a clean factorization of <i>what is being computed</i>. The
      backbone is frozen; basis-building reused from the validated spike <i>p18_conceptmem</i>.
      Generated {esc(demo["timestamp"])} &middot; seed {demo["seed"]}.
    </div>
  </div>

</div>
<script>{js}</script>
</body>
</html>"""


# ====================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=7, help="resid layer to read (p18 basis default = 7)")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-null", type=int, default=400, help="random equal-norm dirs per token for the null")
    ap.add_argument("--z-thresh", type=float, default=2.0, help="sigma-above-null to count as 'present'")
    ap.add_argument("--out", default=os.path.join(RUNS, "thinking_panel.html"), help="output HTML path")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    demo = build_demo(args.layer, args.device, args.seed, args.n_null, args.z_thresh)
    html_doc = render_html(demo)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html_doc)

    print("\n" + "=" * 80)
    print(f"WROTE  {os.path.abspath(args.out)}")
    print("=" * 80)
    print("\nSUMMARY (actual reads of the frozen model):")
    for i, p in enumerate(demo["prompts"]):
        lit = ", ".join(f"{nm} ({p['peak_z'][demo['concept_labels'].index(nm)]:+.1f}σ)"
                        for nm in p["lit_names"]) or "none cleared the null"
        print(f"  P{i+1} \"{p['text'][:48]}{'...' if len(p['text'])>48 else ''}\"")
        print(f"      lit: {lit}")


if __name__ == "__main__":
    main()
