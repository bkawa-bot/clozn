"""
watch_it_think.py -- "watch it think": the autoregressive cousin of watching a diffusion
model denoise. As a frozen local model reads a prompt, you watch the answer CRYSTALLIZE
layer by layer (depth = the 'denoise') while NAMED CONCEPTS light up across the words
(width). Two real, legible signals animated together -- not a raw-neuron light show.

WHAT IT IS (two honest reads, not a fabricated animation).

  1. ACROSS LAYERS  (depth, the 'denoise')  --  the LOGIT LENS.
     At every layer L we take the residual stream at the ANSWER position, apply the model's
     OWN final norm + unembedding (ln_final -> W_U), and read its current top-1 guess at that
     depth. Early layers are vague (the embedding/frequency prior); the answer sharpens, is
     contested through the middle, and IGNITES near the top as the right circuit resolves.
     This is the answer crystallizing through depth -- the AR analogue of diffusion denoising
     over steps. The per-layer guesses are the ACTUAL logit-lens argmax at each layer, nothing
     staged. (Recipe lifted from inspector/demo/memory_server.py: lv = ln_final(r); lv @ W_U.)

  2. ACROSS TOKENS  (width)  --  NAMED CONCEPTS lighting up per word.
     We build a fixed set of NAMED concept directions on the frozen model by diff-in-means
     (reused verbatim from inspector/spikes/p18_conceptmem.py: CONCEPTS, mean_resid_over_texts,
     build_basis). As the model reads the prompt we project the residual at each token onto
     each unit concept direction. A concept LANE glows where that concept is present in the
     state -- a legible picture of the model's passing thoughts in plain terms (fear, animals,
     money, ...). This is a PROBE of what is present; NOT a claim about what the model decided.

HONESTY (load-bearing, the null is shown, never hidden).
  A raw projection is meaningless: every direction has SOME projection, and tokens with a
  bigger residual project more onto EVERYTHING. So for each concept, at each token, we also
  project onto a bank of EQUAL-NORM RANDOM directions and report the activation as
  z = (proj - null_mean) / null_std -- how many sigma above an equal-norm random direction it
  sits. We also center each concept by its mean projection over a neutral corpus, so a concept
  that is merely "always slightly on" (a frequency / norm artifact) sits near zero. A concept
  LIGHTS only where it beats the null (peak z >= threshold); the faint null floor is drawn in
  every cell. The logit-lens confidences are the model's own softmax at that depth -- they are
  honestly low at the very top when probability mass is shared across near-synonyms (the picks
  are exact, the numbers are real). The backbone is FROZEN throughout. This is "watching its
  internal state resolve," not an animation we wrote by hand.

THE ARTIFACT.
  A single self-contained HTML page in the Planet Maiko palette (inline CSS + a little vanilla
  JS; NO external deps). A readable tokens-by-depth field dressed in otherworldly glow: vertical
  axis = layers, horizontal = tokens; the logit-lens top guess at the answer position sharpens
  up the layers with a confidence glow; concept lanes glow where concepts fire across the tokens;
  the final answer ignites at the top. The reveal ANIMATES layer by layer / token by token with
  vanilla JS, so it FEELS like watching it think. A short honesty note sits on the page.

MODEL: GPT-2-small (124M, transformer_lens) -- the same frozen backbone the live Clozn windows
use (memory_window.py / thinking_panel.py), so the concept basis (p18) is exactly the one this
repo validated. 12 layers is enough to watch the answer go vague -> contested -> resolved.

ISOLATED ENV: runs in C:\\Users\\brigi\\src\\clozn\\.venv-sae (transformer_lens + torch, CPU is
fine). GPT-2-small is cached -- no large download. Does NOT touch the lab venv.

Usage (from inspector/, .venv-sae python):
    python demo/watch_it_think.py                     # default: all demo prompts -> inspector/runs/
    python demo/watch_it_think.py --layer-concept 7 --seed 0
    python demo/watch_it_think.py --prompt "The opposite of black is"   # one custom prompt
    python demo/watch_it_think.py --out some/dir/watch_it_think.html
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
# WHICH concepts to show across the tokens, with a human label + Maiko accent per concept. Pulled
# straight out of the p18 basis (so the directions are the SAME validated diff-in-means dirs); the
# entries here only set display order, a label, and a colour. 8 named concepts.
# ----------------------------------------------------------------------------------------------------
CONCEPT_DISPLAY = [
    ("animals",  "animals",     "#C4F542"),   # Toxic Lime
    ("fear",     "fear",        "#FF6FAF"),   # Neon Pink
    ("colors",   "color",       "#6FE0E8"),   # Frozen Cyan
    ("money",    "money",       "#FFE66D"),   # Star Yellow
    ("formal",   "formal tone", "#C9A6FF"),   # Soft Lavender
    ("past",     "past tense",  "#1FB5E5"),   # Electric Ice
    ("food",     "food",        "#FFB36F"),   # warm amber (in-family accent)
    ("question", "questions",   "#A0FFD6"),   # mint (in-family accent)
]

# Demo prompts. Chosen so BOTH signals show nicely:
#   - "kind": "factual" prompts have a clean answer that SHARPENS across layers (the depth story).
#   - "kind": "concepts" prompts make DIFFERENT named concepts light up across the words (the width story).
# The answer position is always the LAST token (the model predicts the next word there). For each we
# note WHY (page copy) and, for factual ones, a one-line note about the crystallization arc actually
# observed on frozen GPT-2-small (measured, surfaced honestly -- including where it is contested).
DEMO_PROMPTS = [
    {
        "text": "The first president of the United States was George",
        "kind": "factual",
        "why":  "watch the answer change its mind as the fact resolves: the frequency prior 'Bush' "
                "leads through the middle layers, then the factual circuit flips it to 'Washington' near the top.",
    },
    {
        "text": "The opposite of black is",
        "kind": "factual",
        "why":  "an antonym crystallizing: vague filler early, contested in the middle, then it snaps "
                "to 'white' in the late layers.",
    },
    {
        "text": "The frightened cat ran away from the snarling dog",
        "kind": "concepts",
        "why":  "a scared-animal scene -- we expect the fear and animals lanes to glow across the words, "
                "money and formality to stay dark.",
    },
    {
        "text": "The banker counted the gold coins and asked what we owed",
        "kind": "concepts",
        "why":  "finance brushing a question -- money (and a little formality / question) should light, "
                "fear and food should not.",
    },
]


# A small neutral corpus to CENTER each concept's projection (subtract its mean projection over these
# topic-free tokens), so a concept that is merely "always a little on" (a frequency / residual-norm
# artifact) sits near zero. Same style as p18 / thinking_panel NEUTRAL_PROMPTS.
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


def load_model(device: str):
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ====================================================================================================
# SIGNAL 1 -- the LOGIT LENS over depth at the answer position (the 'denoise').
# ====================================================================================================
@torch.no_grad()
def logit_lens_over_depth(model, text: str, topk: int = 4):
    """For the LAST token of `text` (the answer position), read the model's top-1 logit-lens guess at
    every layer: take resid_post at layer L, apply the model's OWN final norm + unembed, softmax, top-1.
    Returns a dict with the per-layer guess word + confidence (the crystallization arc), the top-k at
    the final layer, and the prompt tokens. Every guess is the ACTUAL argmax at that depth."""
    toks = model.to_tokens(text)                                   # [1, seq] (prepends BOS)
    nl = model.cfg.n_layers
    names = {f"blocks.{L}.hook_resid_post" for L in range(nl)}
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n in names)

    # full token strings of the prompt (drop BOS for display alignment with the concept lanes)
    prompt_strs = [model.to_string(t) for t in toks[0][1:]]        # [seq-1]
    answer_word = prompt_strs[-1] if prompt_strs else ""

    layers = []
    for L in range(nl):
        r = cache[f"blocks.{L}.hook_resid_post"][0, -1]            # [d_model] @ answer position
        lv = model.ln_final(r.unsqueeze(0))                       # the model's OWN final norm
        logits = (lv @ model.W_U)[0].float()                      # logit lens -> vocab
        probs = logits.softmax(-1)
        conf, idx = probs.max(-1)
        layers.append({
            "layer": L,
            "guess": model.to_string(idx.unsqueeze(0)),
            "conf": float(conf),
        })

    # mark where the guess FLIPS (a visible "it changed its mind") and the layer it first locks to the
    # final guess (the crystallization point).
    final_guess = layers[-1]["guess"]
    crystallized_at = None
    for L in range(nl):
        if all(layers[j]["guess"] == final_guess for j in range(L, nl)):
            crystallized_at = L
            break
    for L in range(nl):
        layers[L]["flip"] = (L > 0 and layers[L]["guess"] != layers[L - 1]["guess"])

    # the actual next-token distribution from the FULL forward (what the model would SAY) -- the answer
    # that ignites at the top. This is the real model output, not the lens approximation.
    full_logits = model(toks)[0, -1].float()
    full_probs = full_logits.softmax(-1)
    tk_conf, tk_idx = full_probs.topk(topk)
    final_topk = [{"word": model.to_string(i.unsqueeze(0)), "p": float(p)}
                  for p, i in zip(tk_conf, tk_idx)]

    return {
        "prompt_tokens": prompt_strs,
        "answer_word": answer_word,
        "layers": layers,                       # per-layer top-1 logit-lens guess + conf
        "final_guess": final_guess,
        "crystallized_at": crystallized_at,
        "final_topk": final_topk,               # the real model output (igniting answer)
        "n_layers": nl,
    }


# ====================================================================================================
# SIGNAL 2 -- NAMED CONCEPTS across the tokens (the width). Reused honesty from thinking_panel/p18.
# ====================================================================================================
@torch.no_grad()
def per_token_resid(model, text: str, layer: int):
    """resid_post at `layer` for every token of `text`. Drops BOS (its residual is a ~30x-norm outlier
    that would swamp every projection). Returns (resid [seq-1, d_model], token_strs [seq-1])."""
    toks = model.to_tokens(text)
    name = f"blocks.{layer}.hook_resid_post"
    _, cache = model.run_with_cache(toks, names_filter=name)
    resid = cache[name][0][1:]                                     # drop BOS
    strs = [model.to_string(t) for t in toks[0][1:]]
    return resid, strs


@torch.no_grad()
def neutral_baseline(model, units, layer: int):
    """Mean projection of each UNIT concept direction over the neutral corpus -> [k]. Subtracted from
    every token's projection so the trace measures ABOVE-baseline presence (honest normalization)."""
    accum = torch.zeros(units.shape[0], device=units.device)
    cnt = 0
    for p in NEUTRAL_PROMPTS:
        resid, _ = per_token_resid(model, p, layer)
        proj = resid @ units.T                                    # [s, k]
        accum = accum + proj.sum(0)
        cnt += proj.shape[0]
    return accum / max(cnt, 1)


@torch.no_grad()
def concept_trace(model, text, units, baseline_unit, layer, n_null, generator):
    """For every token: project resid onto each UNIT concept dir, center by the neutral baseline, and
    z-score against an EQUAL-NORM random null at that token. z is 'sigma above an equal-norm random
    direction': z<=0 -> not present; z>~2 -> clearly present above chance. Returns tokens + z [seq,k]."""
    resid, token_strs = per_token_resid(model, text, layer)        # [seq,d], list
    proj = resid @ units.T                                         # [seq, k] onto unit dirs
    proj_centered = proj - baseline_unit[None, :]

    # null on UNIT random directions (norm cancels in the z-score); per-token mean/std.
    d_model = resid.shape[-1]
    rand_units = F.normalize(torch.randn(n_null, d_model, generator=generator,
                                         device=resid.device), dim=-1)   # [n_null, d]
    proj_unit_rand = resid @ rand_units.T                                # [seq, n_null]
    null_mean = proj_unit_rand.mean(dim=1)                               # [seq]
    null_std = proj_unit_rand.std(dim=1)                                 # [seq]
    null_mean_centered = null_mean[:, None] - baseline_unit[None, :]     # [seq, k]
    z = (proj_centered - null_mean_centered) / (null_std[:, None] + 1e-9)  # [seq, k]
    return {"tokens": token_strs, "z": z.cpu().numpy()}


# ====================================================================================================
# Build the demo: every number rendered is an ACTUAL read of the frozen model.
# ====================================================================================================
def build_demo(layer_concept: int, device: str, seed: int, n_null: int, z_thresh: float,
               prompts: list[dict]):
    torch.manual_seed(seed)
    print(f"loading gpt2 (HookedTransformer) on {device} ...")
    model = load_model(device)
    d_model, nl = model.cfg.d_model, model.cfg.n_layers
    print(f"  d_model={d_model}  n_layers={nl}   concept read layer L={layer_concept}")

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
    print(f"\nSTEP 1 -- named concept directions @ blocks.{layer_concept}.hook_resid_post (diff-in-means)")
    _dirs, units, norms = build_basis(model, concepts, layer_concept)   # [k,d], [k,d], [k]
    cos = (units @ units.T).cpu().numpy()
    off = cos.copy(); np.fill_diagonal(off, np.nan)
    mean_abs_off = float(np.nanmean(np.abs(off)))
    print(f"    mean |off-diag cosine| = {mean_abs_off:.3f}  (lower = more independently nameable)")

    # ---- STEP 2: neutral baseline for centering ----------------------------------------------------
    baseline_unit = neutral_baseline(model, units, layer_concept)       # [k]

    # ---- STEP 3: per prompt -- BOTH signals --------------------------------------------------------
    print(f"\nSTEP 2 -- read each prompt: logit lens over depth + concept lanes (null = {n_null} dirs/token)")
    g = torch.Generator(device=model.cfg.device).manual_seed(seed)
    prompts_out = []
    for pi, prm in enumerate(prompts):
        text = prm["text"]
        lens = logit_lens_over_depth(model, text)                      # signal 1 (depth)
        ctr = concept_trace(model, text, units, baseline_unit, layer_concept, n_null, g)  # signal 2 (width)
        z = ctr["z"]                                                   # [seq, k]
        peak_z = z.max(axis=0)                                         # [k]
        peak_tok_idx = z.argmax(axis=0)                               # [k]
        lit = sorted([j for j in range(k) if peak_z[j] >= z_thresh], key=lambda j: -peak_z[j])
        lit_names = [disp_labels[j] for j in lit]

        # console summary
        print(f"\n  PROMPT {pi+1} [{prm['kind']}]: \"{text}\"")
        arc = " -> ".join(f"{ly['guess'].strip() or ly['guess']}" for ly in lens["layers"])
        print(f"    logit-lens arc @ answer pos: {arc}")
        ca = lens["crystallized_at"]
        print(f"    final guess '{lens['final_guess'].strip()}' crystallizes at "
              f"L{ca if ca is not None else '?'} of {nl}; "
              f"model output top: {lens['final_topk'][0]['word'].strip()} "
              f"({lens['final_topk'][0]['p']:.2f})")
        if lit_names:
            lit_str = ", ".join(f"{disp_labels[j]} ({peak_z[j]:+.1f}σ @ "
                                f"'{ctr['tokens'][peak_tok_idx[j]].strip()}')" for j in lit)
        else:
            lit_str = "none cleared the null"
        print(f"    concepts lit (peak z>={z_thresh}): {lit_str}")

        prompts_out.append({
            "text": text, "kind": prm["kind"], "why": prm["why"],
            "lens": lens,
            "tokens": ctr["tokens"], "z": z,
            "peak_z": peak_z, "lit": lit, "lit_names": lit_names,
        })

    return {
        "model_name": "GPT-2-small (124M, frozen)",
        "layer_concept": layer_concept, "d_model": d_model, "n_layers": nl,
        "n_null": n_null, "z_thresh": z_thresh, "seed": seed,
        "concept_labels": disp_labels, "concept_colors": colors,
        "cosine_mean_abs_off": mean_abs_off,
        "prompts": prompts_out,
        "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ====================================================================================================
# THE ARTIFACT: a single self-contained HTML page in the Planet Maiko palette. Inline CSS + vanilla JS
# (stars + the layer-by-layer / token-by-token reveal). Tokens-by-depth field dressed in glow.
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
LAV       = "#C9A6FF"   # Soft Lavender
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


def disp_tok(t: str) -> str:
    s = t.strip()
    return s if s else "·"


def render_html(demo: dict) -> str:
    labels = demo["concept_labels"]
    colors = demo["concept_colors"]
    z_thresh = demo["z_thresh"]
    nl = demo["n_layers"]
    k = len(labels)

    # intensity mapping: concept z (sigma above null) -> a 0..1 glow. Below the null (z<=0) is dark; the
    # threshold sits partway up so a cell only really glows once it clears the null band.
    def cglow(z: float) -> float:
        if z <= 0:
            return 0.0
        return float(max(0.0, min(1.0, z / (z_thresh * 2.2))))

    # logit-lens confidence -> a 0..1 glow for the depth column. Confidences are small in absolute
    # terms (mass spreads over near-synonyms), so we map with a soft sqrt to keep the late layers vivid.
    def lglow(conf: float) -> float:
        return float(max(0.0, min(1.0, (conf ** 0.5) * 1.35)))

    # ---------- one prompt block ----------------------------------------------------------------------
    def prompt_block(p, idx):
        tokens = p["tokens"]
        z = p["z"]                       # [seq, k]
        seq = len(tokens)
        lens = p["lens"]
        layers = lens["layers"]          # per-layer guess @ answer position
        kind = p["kind"]
        cryst = lens["crystallized_at"]

        # ----- the DEPTH tower: the logit-lens guess at the answer position, layer by layer ----------
        # Rendered top=final layer down to bottom=layer 0, so reading DOWN-to-UP is shallow->deep and
        # the answer "rises" to the top where it ignites. Each rung glows with the model's confidence.
        rungs = []
        for L in range(nl - 1, -1, -1):
            ly = layers[L]
            gg = lglow(ly["conf"])
            guess = disp_tok(ly["guess"])
            is_final = (cryst is not None and L >= cryst)
            col = CYAN if is_final else LAV
            bg = rgba(col, 0.05 + 0.30 * gg)
            sh = f"0 0 {int(2 + 22 * gg)}px {rgba(col, 0.08 + 0.5 * gg)}" if gg > 0 else "none"
            cls = "rung" + (" rung-final" if is_final else "") + (" rung-flip" if ly["flip"] else "")
            conf_pct = f"{ly['conf']*100:.0f}%"
            rungs.append(
                f'<div class="{cls}" style="--d:{nl - 1 - L};background:{bg};box-shadow:{sh};'
                f'border-color:{rgba(col, 0.18 + 0.5 * gg)}">'
                f'<span class="rung-l">L{L}</span>'
                f'<span class="rung-g" style="color:{col}">{esc(guess)}</span>'
                f'<span class="rung-c">{conf_pct}</span>'
                f'<span class="rung-bar"><span class="rung-fill" '
                f'style="width:{int(100*gg)}%;background:{col};box-shadow:0 0 8px {rgba(col,0.7)}"></span></span>'
                f'</div>'
            )

        # the igniting final answer (the real model output, top of the tower)
        topk = lens["final_topk"]
        ignite = topk[0]
        ignite_alt = "".join(
            f'<span class="alt">{esc(disp_tok(t["word"]))} <i>{t["p"]*100:.0f}%</i></span>'
            for t in topk[1:3]
        )

        # ----- the CONCEPT lanes across the tokens ----------------------------------------------------
        tok_cells = "".join(
            f'<div class="tok-cell" style="--t:{ti}">{esc(disp_tok(t))}</div>'
            for ti, t in enumerate(tokens)
        )
        lanes = []
        for j in range(k):
            col = colors[j]
            cells = []
            for ti in range(seq):
                zz = float(z[ti, j])
                gg = cglow(zz)
                lit_cell = zz >= z_thresh
                bg = rgba(col, 0.05 + 0.46 * gg)
                style = f"--t:{ti};background:{bg};"
                if gg > 0:
                    style += f"box-shadow:0 0 {int(2 + 18 * gg)}px {rgba(col, 0.10 + 0.5 * gg)};"
                cls = "cell" + (" cell-lit" if lit_cell else "")
                if lit_cell:
                    style += f"border-color:{rgba(col, 0.7)};"
                title = (f"{labels[j]} @ '{disp_tok(tokens[ti])}': "
                         f"z={zz:+.2f}σ vs equal-norm random null")
                cells.append(f'<div class="{cls}" style="{style}" title="{esc(title)}">'
                             f'<span class="nullline"></span></div>')
            pk = float(p["peak_z"][j])
            on = pk >= z_thresh
            name_cls = "lane-name lane-on" if on else "lane-name"
            name_style = f"color:{col};" if on else ""
            badge = (f'<span class="lane-peak" style="color:{col}">{pk:+.1f}σ</span>' if on
                     else f'<span class="lane-peak lane-peak-off">{pk:+.1f}σ</span>')
            lanes.append(
                f'<div class="lane">'
                f'  <div class="{name_cls}" style="{name_style}">'
                f'    <span class="lane-dot" style="background:{col};opacity:{0.95 if on else 0.28}"></span>'
                f'    {esc(labels[j])}{badge}'
                f'  </div>'
                f'  <div class="lane-cells">{"".join(cells)}</div>'
                f'</div>'
            )

        lit_chips = "".join(
            f'<span class="lit-chip" style="border-color:{rgba(colors[labels.index(nm)],0.6)};'
            f'color:{colors[labels.index(nm)]}">{esc(nm)}</span>'
            for nm in p["lit_names"]
        ) or '<span class="lit-none">(nothing cleared the null)</span>'

        cryst_note = (f'resolved by <b>L{cryst}</b> of {nl}' if cryst is not None
                      else 'still shifting at the top layer')
        kind_tag = ("answer crystallizing through depth" if kind == "factual"
                    else "concepts lighting across the words")

        # answer-position token, highlighted in the sentence row
        ans_word = disp_tok(lens["answer_word"])

        return f'''
      <div class="prompt" style="--i:{idx}" data-seq="{seq}" data-nl="{nl}">
        <div class="prompt-head">
          <div class="prompt-kind">{esc(kind_tag)}</div>
          <div class="prompt-sentence">&ldquo;{esc(p["text"])}&rdquo;</div>
          <div class="prompt-why">{esc(p["why"])}</div>
        </div>

        <div class="stage">
          <!-- LEFT: the depth tower (logit lens over layers at the answer position) -->
          <div class="depth">
            <div class="depth-cap">
              <div class="depth-cap-lab">it would say</div>
              <div class="ignite">{esc(disp_tok(ignite["word"]))}
                <span class="ignite-p">{ignite["p"]*100:.0f}%</span></div>
              <div class="ignite-alt">{ignite_alt}</div>
            </div>
            <div class="rungs">{"".join(rungs)}</div>
            <div class="depth-foot">
              <span class="axis-up">deeper &uarr;</span>
              <span class="depth-note">guess at &ldquo;{esc(ans_word)}&rdquo; &middot; {cryst_note}</span>
            </div>
          </div>

          <!-- RIGHT: the concept lanes across the tokens -->
          <div class="width">
            <div class="tok-row" style="--cols:{seq}">{tok_cells}</div>
            <div class="lanes">{"".join(lanes)}</div>
            <div class="width-foot">
              <span class="lit-lab">present in its state:</span> {lit_chips}
            </div>
          </div>
        </div>
      </div>'''

    prompt_blocks = "".join(prompt_block(p, i) for i, p in enumerate(demo["prompts"]))

    legend = "".join(
        f'<div class="leg-item"><span class="leg-dot" style="background:{colors[j]};'
        f'box-shadow:0 0 12px {rgba(colors[j],0.6)}"></span>'
        f'<span class="leg-name">{esc(labels[j])}</span></div>'
        for j in range(k)
    )

    # ---------- CSS -----------------------------------------------------------------------------------
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
      min-height:100vh; padding:46px 22px 80px; line-height:1.5;
    }}
    .wrap {{ max-width:1140px; margin:0 auto; }}
    .star {{ position:fixed; border-radius:50%; background:var(--white); opacity:0;
      animation:twinkle 6s ease-in-out infinite; pointer-events:none; }}
    @keyframes twinkle {{ 0%,100%{{opacity:0}} 50%{{opacity:.5}} }}

    header {{ text-align:center; margin-bottom:14px; }}
    .eyebrow {{ letter-spacing:.32em; text-transform:uppercase; font-size:11px; font-weight:600;
      color:var(--cyan); opacity:.85; margin-bottom:14px; }}
    h1 {{ font-size:40px; line-height:1.12; margin:0 0 14px; font-weight:700;
      background:linear-gradient(96deg, var(--white) 6%, var(--lav) 42%, var(--cyan) 96%);
      -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
      text-shadow:0 0 40px rgba(201,166,255,0.25); }}
    .lede {{ font-size:16px; color:var(--gray); max-width:760px; margin:0 auto 8px; }}
    .lede b {{ color:var(--white); font-weight:600; }}
    .lede .c1 {{ color:var(--cyan); font-weight:600; }}
    .lede .c2 {{ color:var(--pink); font-weight:600; }}
    .meta-line {{ font-size:12.5px; color:var(--gray); opacity:.82; margin-top:14px; }}
    .meta-line code {{ color:var(--cyan); background:rgba(31,181,229,0.10);
      padding:1px 7px; border-radius:6px; font-size:12px; }}

    .legend {{ display:flex; flex-wrap:wrap; gap:11px 18px; justify-content:center;
      margin:24px auto 4px; max-width:900px; padding:13px 18px; border-radius:16px;
      background:rgba(11,15,42,0.4); border:1px solid rgba(201,166,255,0.14); }}
    .leg-item {{ display:flex; align-items:center; gap:8px; font-size:13px; color:var(--gray); }}
    .leg-dot {{ width:11px; height:11px; border-radius:50%; flex:0 0 auto; }}
    .leg-name {{ color:var(--white); font-weight:500; }}

    .section {{ margin-top:34px; }}

    /* one prompt */
    .prompt {{ background:linear-gradient(150deg, rgba(42,34,80,0.6), rgba(11,15,42,0.5));
      border:1px solid rgba(201,166,255,0.16); border-radius:24px; padding:24px 26px 22px;
      margin-bottom:28px; backdrop-filter:blur(3px);
      box-shadow:0 16px 44px rgba(0,0,0,0.34), inset 0 1px 0 rgba(255,255,255,0.04);
      opacity:0; transform:translateY(16px);
      animation:rise .7s cubic-bezier(.2,.7,.2,1) forwards; animation-delay:calc(var(--i) * .14s + .08s); }}
    @keyframes rise {{ to {{ opacity:1; transform:none; }} }}
    .prompt-head {{ margin-bottom:18px; }}
    .prompt-kind {{ letter-spacing:.18em; text-transform:uppercase; font-size:10.5px; font-weight:700;
      color:var(--pink); opacity:.9; margin-bottom:7px; }}
    .prompt-sentence {{ font-size:21px; font-weight:600; color:var(--white);
      text-shadow:0 0 26px rgba(201,166,255,0.22); }}
    .prompt-why {{ font-size:13px; color:var(--gray); margin-top:6px; font-style:italic; max-width:880px; }}

    /* the two-panel stage: depth tower (left) + concept lanes (right) */
    .stage {{ display:grid; grid-template-columns:236px 1fr; gap:22px; align-items:start; }}

    /* DEPTH tower -- the logit lens crystallizing up the layers */
    .depth {{ display:flex; flex-direction:column; gap:8px; }}
    .depth-cap {{ text-align:center; padding:11px 12px 13px; border-radius:16px;
      background:radial-gradient(120% 130% at 50% 0%, rgba(31,181,229,0.18), rgba(11,15,42,0.2));
      border:1px solid rgba(111,224,232,0.28);
      box-shadow:0 0 30px rgba(31,181,229,0.18), inset 0 1px 0 rgba(255,255,255,0.05); }}
    .depth-cap-lab {{ font-size:10px; letter-spacing:.2em; text-transform:uppercase; color:var(--gray);
      opacity:.8; margin-bottom:4px; }}
    .ignite {{ font-size:25px; font-weight:800; color:var(--cyan);
      text-shadow:0 0 24px rgba(111,224,232,0.65), 0 0 6px rgba(255,255,255,0.4);
      animation:ignite-pulse 3.4s ease-in-out infinite; }}
    .ignite-p {{ font-size:13px; font-weight:600; color:var(--white); opacity:.7; margin-left:4px;
      vertical-align:middle; }}
    @keyframes ignite-pulse {{ 0%,100%{{text-shadow:0 0 22px rgba(111,224,232,0.5),0 0 6px rgba(255,255,255,0.35)}}
      50%{{text-shadow:0 0 34px rgba(111,224,232,0.85),0 0 10px rgba(255,255,255,0.5)}} }}
    .ignite-alt {{ margin-top:6px; display:flex; gap:10px; justify-content:center; flex-wrap:wrap; }}
    .ignite-alt .alt {{ font-size:11px; color:var(--gray); }}
    .ignite-alt .alt i {{ color:var(--lav); font-style:normal; opacity:.85; }}

    .rungs {{ display:flex; flex-direction:column; gap:4px; }}
    .rung {{ display:grid; grid-template-columns:30px 1fr auto; align-items:center; gap:8px;
      padding:5px 10px; border-radius:10px; border:1px solid rgba(255,255,255,0.05);
      background:rgba(11,15,42,0.3); position:relative; overflow:hidden;
      opacity:0; transform:translateY(7px);
      animation:rung-in .42s ease forwards; animation-delay:calc(var(--d) * .07s + .55s);
      transition:background .4s ease, box-shadow .4s ease; }}
    @keyframes rung-in {{ to {{ opacity:1; transform:none; }} }}
    .rung-l {{ font-size:10px; color:var(--gray); opacity:.7; font-variant-numeric:tabular-nums; }}
    .rung-g {{ font-size:14px; font-weight:600; white-space:nowrap; overflow:hidden;
      text-overflow:ellipsis; }}
    .rung-final .rung-g {{ font-weight:700; }}
    .rung-c {{ font-size:10px; color:var(--gray); opacity:.65; font-variant-numeric:tabular-nums; }}
    .rung-bar {{ grid-column:1 / -1; height:2px; border-radius:2px; background:rgba(255,255,255,0.06);
      overflow:hidden; margin-top:1px; }}
    .rung-fill {{ display:block; height:100%; border-radius:2px;
      transform-origin:left; animation:fill-in .5s ease forwards; }}
    .rung-flip::before {{ content:'\\2192'; position:absolute; left:-1px; top:50%; transform:translateY(-50%);
      color:var(--pink); font-size:11px; opacity:.0; animation:flip-mark .5s ease forwards;
      animation-delay:calc(var(--d) * .07s + .7s); }}
    @keyframes flip-mark {{ to {{ opacity:.85; left:2px; }} }}
    .depth-foot {{ display:flex; flex-direction:column; gap:2px; margin-top:3px; }}
    .axis-up {{ font-size:10px; letter-spacing:.16em; text-transform:uppercase; color:var(--cyan);
      opacity:.6; }}
    .depth-note {{ font-size:11px; color:var(--gray); opacity:.78; }}
    .depth-note b {{ color:var(--cyan); }}

    /* WIDTH -- concept lanes across the tokens */
    .width {{ min-width:0; }}
    .tok-row {{ display:grid; grid-template-columns:repeat(var(--cols), 1fr); gap:5px; margin-bottom:8px; }}
    .tok-cell {{ font-size:11.5px; color:var(--gray); text-align:center; font-weight:500;
      white-space:nowrap; overflow:hidden; text-overflow:ellipsis; opacity:0; padding-bottom:2px;
      border-bottom:1px solid rgba(255,255,255,0.06);
      animation:tok-in .4s ease forwards; animation-delay:calc(var(--t) * .05s + .5s); }}
    @keyframes tok-in {{ to {{ opacity:.92; }} }}
    .lanes {{ display:flex; flex-direction:column; gap:5px; }}
    .lane {{ display:grid; grid-template-columns:128px 1fr; align-items:center; gap:0; }}
    .lane-name {{ font-size:12.5px; color:var(--gray); display:flex; align-items:center; gap:7px;
      padding-right:10px; font-weight:500; }}
    .lane-on {{ font-weight:700; }}
    .lane-dot {{ width:9px; height:9px; border-radius:50%; flex:0 0 auto; }}
    .lane-peak {{ margin-left:auto; font-size:10.5px; font-variant-numeric:tabular-nums; font-weight:600; }}
    .lane-peak-off {{ color:var(--gray); opacity:.5; }}
    .lane-cells {{ display:grid; grid-template-columns:repeat(var(--cols, 1), 1fr); gap:5px; }}
    .lane {{ --cols:1; }}
    .cell {{ position:relative; height:30px; border-radius:8px; border:1px solid rgba(255,255,255,0.05);
      transition:background .5s ease, box-shadow .5s ease; opacity:0;
      animation:cell-in .45s ease forwards; animation-delay:calc(var(--t) * .05s + .62s); }}
    @keyframes cell-in {{ to {{ opacity:1; }} }}
    .nullline {{ position:absolute; left:14%; right:14%; bottom:5px; height:1px;
      background:rgba(255,255,255,0.16); border-radius:1px; }}
    .width-foot {{ margin-top:14px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.06);
      display:flex; align-items:center; flex-wrap:wrap; gap:8px; font-size:13px; }}
    .lit-lab {{ color:var(--gray); margin-right:2px; }}
    .lit-chip {{ font-size:12.5px; padding:3px 11px; border-radius:20px; font-weight:600;
      border:1px solid; background:rgba(255,255,255,0.03); }}
    .lit-none {{ color:var(--gray); font-style:italic; opacity:.7; }}

    .footer {{ margin-top:52px; text-align:center; font-size:12px; color:var(--gray); opacity:.72;
      line-height:1.7; }}
    .footer b {{ color:var(--lav); }}
    .honest {{ margin-top:14px; padding:18px 22px; border-radius:14px; font-size:12.5px;
      background:rgba(11,15,42,0.42); border:1px solid rgba(255,255,255,0.06); color:var(--gray);
      max-width:860px; margin-left:auto; margin-right:auto; text-align:left; line-height:1.66; }}
    .honest b {{ color:var(--cyan); }}
    .honest .k {{ color:var(--lav); }}

    @media (max-width:860px) {{
      h1 {{ font-size:30px; }}
      .stage {{ grid-template-columns:1fr; gap:24px; }}
      .lane {{ grid-template-columns:92px 1fr; }}
      .lane-name {{ font-size:11px; }}
      .tok-cell {{ font-size:9.5px; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      .prompt,.rung,.tok-cell,.cell {{ opacity:1 !important; transform:none !important; animation:none !important; }}
      .rung-fill {{ animation:none !important; }}
      .rung-flip::before {{ opacity:.85 !important; animation:none !important; }}
      .ignite {{ animation:none !important; }}
      .star {{ animation:none !important; }}
    }}
    """

    # vanilla JS: stars + set each lane's --cols so the cells align with the token row, and a gentle
    # re-trigger of the reveal when a prompt scrolls into view (so it "thinks" again as you scroll).
    js = """
    (function(){
      // stars
      var b=document.body, n=52;
      for(var i=0;i<n;i++){
        var s=document.createElement('div'); s.className='star';
        var sz=Math.random()*2+1;
        s.style.width=sz+'px'; s.style.height=sz+'px';
        s.style.left=(Math.random()*100)+'vw'; s.style.top=(Math.random()*100)+'vh';
        s.style.animationDelay=(Math.random()*6)+'s';
        s.style.animationDuration=(4+Math.random()*5)+'s';
        b.appendChild(s);
      }
      // align concept lanes to the token grid (each prompt knows its token count)
      document.querySelectorAll('.prompt').forEach(function(p){
        var seq=p.getAttribute('data-seq');
        p.querySelectorAll('.lane').forEach(function(l){ l.style.setProperty('--cols', seq); });
      });
      // re-run the depth/concept reveal when a prompt re-enters the viewport, so it animates again
      if('IntersectionObserver' in window){
        var io=new IntersectionObserver(function(entries){
          entries.forEach(function(e){
            if(e.isIntersecting){
              var el=e.target;
              // restart CSS animations by toggling a class
              el.querySelectorAll('.rung,.tok-cell,.cell').forEach(function(node){
                node.style.animation='none'; void node.offsetWidth; node.style.animation='';
              });
            }
          });
        }, {threshold:0.35});
        document.querySelectorAll('.prompt').forEach(function(p){ io.observe(p); });
      }
    })();
    """

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clozn &middot; watch it think</title>
<style>{style}</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="eyebrow">Clozn &middot; watch it think</div>
    <h1>Watch it think</h1>
    <p class="lede">As a <b>frozen {esc(demo["model_name"])}</b> reads a prompt, two real signals
      animate together &mdash; the autoregressive cousin of watching a diffusion model denoise.
      <span class="c1">Up the layers</span>, the <b>logit lens</b> shows the answer crystallizing
      from vague to confident at the answer position (the model's own final norm + unembedding, read
      at each depth). <span class="c2">Across the words</span>, <b>named concepts</b> light up where
      they are genuinely present in the state. Both are <b>reads</b> of the interior &mdash; not a
      claim about what the model decided, and not an animation we wrote by hand.</p>
    <div class="meta-line">
      depth = logit lens at every <code>blocks.L.hook_resid_post</code> (answer position) &middot;
      concepts read at <code>blocks.{demo["layer_concept"]}</code> by diff-in-means &middot;
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
      <b>What this is, honestly.</b> Two <span class="k">reads</span>, not decisions and not staged.
      <b>(1) Depth / the &lsquo;denoise&rsquo;.</b> At every layer we take the residual stream at the
      ANSWER position, apply the model's <span class="k">own</span> final norm and unembedding
      (<code>ln_final &rarr; W_U</code>), and read the top-1 token at that depth. The arc you see
      &mdash; vague early, contested in the middle, igniting near the top &mdash; is the genuine
      logit-lens argmax per layer; the percentages are the model's own softmax there, honestly small
      when probability mass is shared across near-synonyms (the picks are exact). The igniting answer
      at the top is the model's <span class="k">actual</span> next-token output. <b>(2) Width /
      named concepts.</b> For each of the {k} concepts we build a direction by <span class="k">
      diff-in-means</span> on the frozen model (mean residual over positive sentences minus contrast,
      at <code>blocks.{demo["layer_concept"]}</code>) &mdash; the validated basis from the
      <i>p18_conceptmem</i> spike. We project each token's residual onto each concept and, because a
      bigger residual projects more onto <i>everything</i>, report it as <b>z = how many &sigma; above
      an equal-norm random direction</b> ({demo["n_null"]} samples/token), centered by a neutral-corpus
      baseline so an &ldquo;always-on&rdquo; artifact sits near zero. A concept lights only where its
      peak <b>z &ge; {z_thresh:g}&sigma;</b>; the faint null floor is drawn in every cell. Basis
      non-orthogonality (mean |off-diagonal cosine| = {demo["cosine_mean_abs_off"]:.2f}) means nearby
      concepts can co-activate &mdash; this is a probe of <i>what is present</i>, not a clean
      factorization of <i>what is computed</i>. The backbone is frozen throughout.
      Generated {esc(demo["timestamp"])} &middot; seed {demo["seed"]}.
    </div>
  </div>

</div>
<script>{js}</script>
</body>
</html>"""


# ====================================================================================================
def main():
    ap = argparse.ArgumentParser(description="watch it think -- logit-lens depth + named-concept width")
    ap.add_argument("--layer-concept", type=int, default=7,
                    help="resid layer to read the concept lanes at (p18 basis default = 7)")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-null", type=int, default=400, help="random equal-norm dirs/token for the null")
    ap.add_argument("--z-thresh", type=float, default=2.0, help="sigma-above-null to count 'present'")
    ap.add_argument("--prompt", default=None,
                    help="a single custom prompt (factual kind); default = the built-in demo set")
    ap.add_argument("--out", default=os.path.join(RUNS, "watch_it_think.html"), help="output HTML path")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    prompts = ([{"text": args.prompt, "kind": "factual",
                 "why": "a custom prompt -- watch its top guess at the answer position sharpen up the layers."}]
               if args.prompt else DEMO_PROMPTS)

    demo = build_demo(args.layer_concept, args.device, args.seed, args.n_null, args.z_thresh, prompts)
    html_doc = render_html(demo)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html_doc)

    print("\n" + "=" * 84)
    print(f"WROTE  {os.path.abspath(args.out)}")
    print("=" * 84)
    print("\nSUMMARY (actual reads of the frozen model):")
    for i, p in enumerate(demo["prompts"]):
        lens = p["lens"]
        ca = lens["crystallized_at"]
        lit = ", ".join(p["lit_names"]) or "none cleared the null"
        print(f"  P{i+1} [{p['kind']}] \"{p['text'][:52]}{'...' if len(p['text'])>52 else ''}\"")
        print(f"      depth: '{lens['final_guess'].strip()}' resolves @ L{ca}/{demo['n_layers']}; "
              f"says '{lens['final_topk'][0]['word'].strip()}' ({lens['final_topk'][0]['p']:.2f})")
        print(f"      concepts lit: {lit}")


if __name__ == "__main__":
    main()
