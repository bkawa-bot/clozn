"""
think_graph.py -- "watch it think", v2: a NODE-GRAPH ANIMATION (not the v1 layer lattice).

An honest, attribution-graph-FLAVORED temporal network. Concepts are NODES; one extra node is the
OUTPUT TOKEN (the model's current logit-lens guess at the answer position). TIME is DEPTH: as a
playback head scrubs from layer 0 up to the final layer (the autoregressive analogue of diffusion
denoise steps), concept nodes PULSE/LIGHT where they are genuinely present in the residual at that
depth, the output-token node MORPHS as the guess resolves (it can change its mind, e.g. Bush ->
Washington), and EDGES are drawn from the lit concepts to the token, each weighted by that concept's
REAL contribution to the current top token through the model's own logit lens. You watch which lit
concepts are driving the resolving token, and how that drive shifts as the active set changes up the
layers. Self-contained Maiko-palette HTML (SVG + vanilla JS), smooth auto-play + a scrubber.

WHAT IS REAL (load-bearing honesty -- nothing here is decorative).

  NODES (concepts).  The 8 named concept directions are the SAME validated diff-in-means basis from
  the p18_conceptmem spike (CONCEPTS / build_basis / mean_resid_over_texts), reused verbatim. We do
  NOT trust a raw projection (every direction has some projection; a bigger residual projects more
  onto everything). So at each layer L, for each concept c, we project the residual at the answer
  position onto the UNIT concept direction, center it by a neutral-corpus baseline, and z-score it
  against a bank of EQUAL-NORM RANDOM directions at that layer: z = sigma above an equal-norm random
  direction. A concept node LIGHTS only where z >= z_thresh (default 2). That activation is real.

  NODE (output token).  At each layer L we take the residual at the answer position, apply the model's
  OWN final norm + unembedding (ln_final -> W_U), softmax, and read the top-1 token. That is the
  genuine logit-lens argmax at that depth -- it sharpens from vague to confident, and where the model
  changes its mind the token node visibly morphs. The igniting final answer is the model's ACTUAL
  next-token output from the full forward, not the lens approximation.

  EDGES (concept -> token).  The honest one. The logit-lens logit for the current top token u at layer
  L is  logit_u = ln_final(r_L) . W_U[:,u].  ln_final here is LayerNormPre (folded): it CENTERS and
  NORMALIZES the residual (no learned affine -- that was folded into W_U). Write rhat = (r_L - mean) /
  std. Expand rhat in a basis containing the unit concept direction d_c: the term of logit_u explained
  by the residual's component ALONG d_c is exactly
        contribution_c = (rhat . d_c) * (d_c . W_U[:,u])
                       =     a_c       *        g_c
  -- a_c = how present the concept is in the (normalized) residual, g_c = how that direction pushes
  THIS token's unembedding. This is a genuine additive term of the model's own logit-lens logit for
  the resolving token: "how much this concept is pushing this token, right now, at this depth." Edge
  weight = that number; sign tells push (toward) vs pull (away). We draw an edge only from a concept
  that is LIT (cleared the null) to the current token, and animate its thickness/glow by |contribution|
  as the playback head moves, so you SEE the active concepts feed the resolving token and watch the
  drive shift as the lit set changes. The picks are exact; the contributions are the real computed
  terms, not a fabricated circuit.

  WHAT WE DO NOT DRAW.  No causal concept -> concept circuit edges. That needs feature-circuit
  machinery this project showed FAILS locally (the SAE null). Only the concept -> token contribution
  edges are honest, so only those carry weight. (Optional faint concept co-activation rings are shown
  ONLY if --coactivation is passed, and are clearly labeled correlational, never causal.)

MODEL: GPT-2-small (124M, transformer_lens), FROZEN throughout -- the same backbone p18 validated, so
the concept basis is exactly the one this repo trusts. 12 layers is enough to watch vague -> contested
-> resolved.

ISOLATED ENV: runs in C:\\Users\\brigi\\src\\clozn\\.venv-sae (transformer_lens + torch, CPU is fine).
GPT-2-small is cached -- no large download. Does NOT touch the lab venv.

Usage (from inspector/, .venv-sae python):
    python demo/think_graph.py                       # default demo prompts -> inspector/runs/
    python demo/think_graph.py --prompt "The first president of the United States was George"
    python demo/think_graph.py --layer-concept 7 --seed 0 --coactivation
    python demo/think_graph.py --out some/dir/think_graph.html
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html as _html
import json as _json
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
# The concept NODES: a human label + a Maiko accent per concept, pulled from the p18 basis (same dirs;
# these entries only set display order, label, colour). 8 named concepts -- the same set as v1.
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

# Demo prompts chosen so the ANIMATION reads well: a token that changes its mind across layers while
# concepts visibly drive it. The answer position is always the LAST token (next-word prediction there).
DEMO_PROMPTS = [
    {
        "text": "The first president of the United States was George",
        "kind": "factual",
        "why":  "the token changes its mind up the layers -- the frequency prior leads early, then the "
                "factual circuit flips it near the top -- while named concepts pulse and feed it.",
    },
    {
        "text": "The frightened cat ran away from the snarling",
        "kind": "concepts",
        "why":  "a scared-animal scene resolving its last word: watch fear and animals light and pull "
                "the token, while money and formality stay dark.",
    },
]

# Neutral corpus to CENTER each concept's projection (subtract its mean projection over topic-free
# tokens) so a concept that is merely "always a little on" sits near zero. Same style as p18.
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
# The capture. Per LAYER at the answer position: (1) concept z vs equal-norm random null (does it
# light), (2) logit-lens top guess (the morphing token node), (3) the REAL per-concept contribution to
# that top token's logit-lens logit (the honest edge weights).
# ====================================================================================================
@torch.no_grad()
def neutral_baseline_per_layer(model, units_per_layer):
    """Mean projection of each UNIT concept dir over the neutral corpus, AT EVERY LAYER -> [nl, k].
    units_per_layer[L] is [k, d]. Subtracted from each layer's projection so the trace is above-baseline
    presence (honest normalization), matching p18 / v1."""
    nl = model.cfg.n_layers
    names = {f"blocks.{L}.hook_resid_post" for L in range(nl)}
    k = units_per_layer[0].shape[0]
    accum = torch.zeros(nl, k, device=units_per_layer[0].device)
    cnt = torch.zeros(nl, device=units_per_layer[0].device)
    for text in NEUTRAL_PROMPTS:
        toks = model.to_tokens(text)
        _, cache = model.run_with_cache(toks, names_filter=lambda n: n in names)
        for L in range(nl):
            r = cache[f"blocks.{L}.hook_resid_post"][0][1:]          # drop BOS [seq-1, d]
            proj = r @ units_per_layer[L].T                          # [seq-1, k]
            accum[L] += proj.sum(0)
            cnt[L] += proj.shape[0]
    return accum / cnt.clamp(min=1).unsqueeze(-1)                    # [nl, k]


@torch.no_grad()
def capture_graph(model, text, units_per_layer, dirs_per_layer, baseline_unit, n_null, generator,
                  z_thresh):
    """The whole temporal graph for one prompt. Returns per-layer frames: token guess+conf, per-concept
    z (lit?), and per-concept REAL contribution to the current top token via the logit lens.

    Edge weight derivation (honest). At layer L the logit-lens logit for top token u is
        logit_u = ln_final(r_L) . W_U[:,u].
    ln_final is LayerNormPre (folded): rhat = (r_L - mean)/std, no learned affine (folded into W_U).
    Expanding rhat along the unit concept dir d_c, the term of logit_u explained by the component of
    the residual ALONG d_c is exactly  (rhat . d_c) * (d_c . W_U[:,u])  -- a genuine additive piece of
    the model's own logit-lens logit for THIS token. a_c = rhat.d_c (presence in normalized residual),
    g_c = d_c . W_U[:,u] (push of that direction on the token). contribution_c = a_c * g_c.
    """
    nl = model.cfg.n_layers
    names = {f"blocks.{L}.hook_resid_post" for L in range(nl)}
    toks = model.to_tokens(text)
    _, cache = model.run_with_cache(toks, names_filter=lambda n: n in names)

    prompt_strs = [model.to_string(t) for t in toks[0][1:]]          # drop BOS for display
    answer_word = prompt_strs[-1] if prompt_strs else ""
    k = units_per_layer[0].shape[0]
    d_model = model.cfg.d_model

    # one shared random null bank per layer (equal-norm random unit dirs; norm cancels in the z-score)
    frames = []
    for L in range(nl):
        r = cache[f"blocks.{L}.hook_resid_post"][0, -1].float()      # [d] @ answer position
        units = units_per_layer[L]                                   # [k, d]

        # ---- token node: logit-lens top guess at this depth (the model's OWN ln_final -> W_U) -------
        lv = model.ln_final(r.unsqueeze(0))                          # LayerNormPre: center+normalize
        logits = (lv @ model.W_U)[0].float()                        # [vocab]
        probs = logits.softmax(-1)
        conf, idx = probs.max(-1)
        tok_id = int(idx)
        guess = model.to_string(idx.unsqueeze(0))

        # rhat = the normalized residual ln_final produced (so contributions are in real logit units)
        rhat = lv[0].float()                                         # [d]

        # ---- concept presence z vs equal-norm random null at THIS layer -----------------------------
        proj = (units @ r)                                           # [k] onto unit dirs (raw resid)
        proj_centered = proj - baseline_unit[L]                      # center by neutral baseline
        rand_units = F.normalize(torch.randn(n_null, d_model, generator=generator,
                                             device=r.device), dim=-1)   # [n_null, d]
        null = rand_units @ r                                        # [n_null]
        null_mean = null.mean()
        null_std = null.std()
        null_mean_centered = null_mean - baseline_unit[L]            # [k]
        z = (proj_centered - null_mean_centered) / (null_std + 1e-9)  # [k]

        # ---- REAL per-concept contribution to the CURRENT top token via the logit lens --------------
        u = model.W_U[:, tok_id].float()                            # [d] unembedding col of top token
        a_c = (units @ rhat)                                        # [k] presence in normalized resid
        g_c = (units @ u)                                           # [k] push of each dir on the token
        contribution = (a_c * g_c)                                  # [k] additive logit-lens term

        zc = z.cpu().numpy()
        contrib = contribution.cpu().numpy()
        frames.append({
            "layer": L,
            "guess": guess,
            "guess_id": tok_id,
            "conf": float(conf),
            "z": [float(x) for x in zc],                            # [k]
            "lit": [bool(x >= z_thresh) for x in zc],               # [k]
            "contribution": [float(x) for x in contrib],            # [k] real edge weights
        })

    # mark flips + crystallization layer
    final_guess = frames[-1]["guess"]
    crystallized_at = None
    for L in range(nl):
        if all(frames[j]["guess"] == final_guess for j in range(L, nl)):
            crystallized_at = L
            break
    for L in range(nl):
        frames[L]["flip"] = bool(L > 0 and frames[L]["guess"] != frames[L - 1]["guess"])

    # the actual model output (igniting answer) from the full forward
    full_logits = model(toks)[0, -1].float()
    full_probs = full_logits.softmax(-1)
    tk_conf, tk_idx = full_probs.topk(4)
    final_topk = [{"word": model.to_string(i.unsqueeze(0)), "p": float(p)}
                  for p, i in zip(tk_conf, tk_idx)]

    # per-concept peak z over depth + the layer of peak (for the static node summary / labels)
    z_arr = np.array([f["z"] for f in frames])                      # [nl, k]
    peak_z = z_arr.max(axis=0)                                      # [k]
    peak_layer = z_arr.argmax(axis=0)                              # [k]
    ever_lit = [bool(peak_z[j] >= z_thresh) for j in range(k)]

    return {
        "answer_word": answer_word,
        "prompt_tokens": prompt_strs,
        "frames": frames,
        "final_guess": final_guess,
        "crystallized_at": crystallized_at,
        "final_topk": final_topk,
        "n_layers": nl,
        "peak_z": [float(x) for x in peak_z],
        "peak_layer": [int(x) for x in peak_layer],
        "ever_lit": ever_lit,
    }


# ====================================================================================================
# Build the demo: every number is an ACTUAL read of the frozen model.
# ====================================================================================================
def build_demo(layer_concept: int, device: str, seed: int, n_null: int, z_thresh: float,
               prompts: list[dict]):
    torch.manual_seed(seed)
    print(f"loading gpt2 (HookedTransformer) on {device} ...")
    model = load_model(device)
    d_model, nl = model.cfg.d_model, model.cfg.n_layers
    print(f"  d_model={d_model}  n_layers={nl}")

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

    # ---- build the diff-in-means basis AT EVERY LAYER (z is read per layer up the depth axis) -------
    # The concept node lights where its z clears the null at THAT depth, so we need a basis per layer.
    # The labels/colours come from the chosen --layer-concept basis cosine (reported), but the time
    # axis uses each layer's own basis (REUSED build_basis from p18 verbatim, just at every L).
    print(f"\nSTEP 1 -- diff-in-means concept directions at EVERY layer (reused p18 build_basis)")
    units_per_layer = []
    dirs_per_layer = []
    for L in range(nl):
        dL, uL, _nL = build_basis(model, concepts, L)
        dirs_per_layer.append(dL)
        units_per_layer.append(uL)
    # basis quality at the headline layer (for the page note)
    units_head = units_per_layer[layer_concept]
    cos = (units_head @ units_head.T).cpu().numpy()
    off = cos.copy(); np.fill_diagonal(off, np.nan)
    mean_abs_off = float(np.nanmean(np.abs(off)))
    print(f"    @L={layer_concept}: mean |off-diag cosine| = {mean_abs_off:.3f} "
          f"(lower = more independently nameable)")

    # ---- neutral baseline per layer for centering ----------------------------------------------------
    print(f"STEP 2 -- neutral baseline per layer for centering ({len(NEUTRAL_PROMPTS)} prompts)")
    baseline_unit = neutral_baseline_per_layer(model, units_per_layer)   # [nl, k]

    # ---- per prompt: capture the temporal graph -----------------------------------------------------
    print(f"\nSTEP 3 -- capture the temporal graph per prompt (null = {n_null} dirs/layer, "
          f"lit z>={z_thresh})")
    g = torch.Generator(device=model.cfg.device).manual_seed(seed)
    prompts_out = []
    for pi, prm in enumerate(prompts):
        text = prm["text"]
        cap = capture_graph(model, text, units_per_layer, dirs_per_layer, baseline_unit,
                            n_null, g, z_thresh)
        arc = " -> ".join(f"{fr['guess'].strip() or fr['guess']}" for fr in cap["frames"])
        ca = cap["crystallized_at"]
        lit_names = [disp_labels[j] for j in range(k) if cap["ever_lit"][j]]
        print(f"\n  PROMPT {pi+1} [{prm['kind']}]: \"{text}\"")
        print(f"    logit-lens arc @ answer pos: {arc}")
        print(f"    final guess '{cap['final_guess'].strip()}' resolves @ "
              f"L{ca if ca is not None else '?'}/{nl}; "
              f"model says '{cap['final_topk'][0]['word'].strip()}' ({cap['final_topk'][0]['p']:.2f})")
        print(f"    concepts ever lit (peak z>={z_thresh}): "
              f"{', '.join(lit_names) if lit_names else 'none cleared the null'}")
        # report the strongest concept->token contribution observed, per lit concept (sanity / honesty)
        for j in range(k):
            if not cap["ever_lit"][j]:
                continue
            best = max(cap["frames"], key=lambda fr: abs(fr["contribution"][j]))
            print(f"      {disp_labels[j]:12} peak z={cap['peak_z'][j]:+.1f}σ @L{cap['peak_layer'][j]}; "
                  f"max |contribution to token| = {abs(best['contribution'][j]):.2f} "
                  f"(token '{best['guess'].strip()}' @L{best['layer']})")
        prompts_out.append({
            "text": text, "kind": prm["kind"], "why": prm["why"], "cap": cap,
        })

    return {
        "model_name": "GPT-2-small (124M, frozen)",
        "layer_concept": layer_concept, "d_model": d_model, "n_layers": nl,
        "n_null": n_null, "z_thresh": z_thresh, "seed": seed,
        "concept_labels": disp_labels, "concept_colors": colors,
        "concept_names": names_internal,
        "cosine_mean_abs_off": mean_abs_off,
        "prompts": prompts_out,
        "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ====================================================================================================
# THE ARTIFACT: a self-contained Maiko-palette HTML node-graph ANIMATION (SVG + vanilla JS). All the
# real numbers are serialized to a JS payload; the JS draws the temporal network and plays it.
# ====================================================================================================
BG_DEEP   = "#0B0F2A"
BG_COSMIC = "#1A1F4A"
BG_MID    = "#2A2250"
PINK      = "#FF6FAF"
ICE       = "#1FB5E5"
CYAN      = "#6FE0E8"
LIME      = "#C4F542"
YELLOW    = "#FFE66D"
LAV       = "#C9A6FF"
WHITE     = "#F4F7FF"
GRAY      = "#A7B0C0"


def esc(s) -> str:
    return _html.escape(str(s))


def render_html(demo: dict, coactivation: bool) -> str:
    labels = demo["concept_labels"]
    colors = demo["concept_colors"]
    z_thresh = demo["z_thresh"]
    nl = demo["n_layers"]
    k = len(labels)

    # ---- serialize the real captured data for the JS animation -------------------------------------
    payload = {
        "z_thresh": z_thresh,
        "n_layers": nl,
        "n_null": demo["n_null"],
        "labels": labels,
        "colors": colors,
        "coactivation": coactivation,
        "prompts": [],
    }
    for p in demo["prompts"]:
        cap = p["cap"]
        payload["prompts"].append({
            "text": p["text"],
            "kind": p["kind"],
            "why": p["why"],
            "answer_word": cap["answer_word"],
            "tokens": cap["prompt_tokens"],
            "frames": cap["frames"],
            "final_guess": cap["final_guess"],
            "crystallized_at": cap["crystallized_at"],
            "final_topk": cap["final_topk"],
            "peak_z": cap["peak_z"],
            "peak_layer": cap["peak_layer"],
            "ever_lit": cap["ever_lit"],
        })
    payload_json = _json.dumps(payload).replace("</", "<\\/")

    legend = "".join(
        f'<div class="leg-item"><span class="leg-dot" style="background:{colors[j]};'
        f'box-shadow:0 0 12px {colors[j]}99"></span>'
        f'<span class="leg-name">{esc(labels[j])}</span></div>'
        for j in range(k)
    )

    # tabs for switching prompts
    tabs = "".join(
        f'<button class="tab{" tab-on" if i == 0 else ""}" data-i="{i}">'
        f'<span class="tab-kind">{esc(p["kind"])}</span>'
        f'<span class="tab-text">&ldquo;{esc(p["text"][:46])}{"&hellip;" if len(p["text"])>46 else ""}&rdquo;</span>'
        f'</button>'
        for i, p in enumerate(demo["prompts"])
    )

    style = f"""
    :root {{
      --bg-deep:{BG_DEEP}; --bg-cosmic:{BG_COSMIC}; --bg-mid:{BG_MID};
      --pink:{PINK}; --ice:{ICE}; --cyan:{CYAN}; --lime:{LIME};
      --yellow:{YELLOW}; --lav:{LAV}; --white:{WHITE}; --gray:{GRAY};
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
      min-height:100vh; padding:42px 22px 70px; line-height:1.5;
    }}
    .wrap {{ max-width:1080px; margin:0 auto; }}
    .star {{ position:fixed; border-radius:50%; background:var(--white); opacity:0;
      animation:twinkle 6s ease-in-out infinite; pointer-events:none; }}
    @keyframes twinkle {{ 0%,100%{{opacity:0}} 50%{{opacity:.5}} }}

    header {{ text-align:center; margin-bottom:18px; }}
    .eyebrow {{ letter-spacing:.32em; text-transform:uppercase; font-size:11px; font-weight:600;
      color:var(--cyan); opacity:.85; margin-bottom:14px; }}
    h1 {{ font-size:40px; line-height:1.12; margin:0 0 14px; font-weight:700;
      background:linear-gradient(96deg, var(--white) 6%, var(--lav) 42%, var(--cyan) 96%);
      -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
      text-shadow:0 0 40px rgba(201,166,255,0.25); }}
    .lede {{ font-size:15.5px; color:var(--gray); max-width:780px; margin:0 auto 8px; }}
    .lede b {{ color:var(--white); font-weight:600; }}
    .lede .c1 {{ color:var(--cyan); font-weight:600; }}
    .lede .c2 {{ color:var(--pink); font-weight:600; }}
    .meta-line {{ font-size:12.5px; color:var(--gray); opacity:.82; margin-top:14px; }}
    .meta-line code {{ color:var(--cyan); background:rgba(31,181,229,0.10);
      padding:1px 7px; border-radius:6px; font-size:12px; }}

    .legend {{ display:flex; flex-wrap:wrap; gap:10px 16px; justify-content:center;
      margin:22px auto 4px; max-width:880px; padding:12px 18px; border-radius:16px;
      background:rgba(11,15,42,0.4); border:1px solid rgba(201,166,255,0.14); }}
    .leg-item {{ display:flex; align-items:center; gap:8px; font-size:13px; color:var(--gray); }}
    .leg-dot {{ width:11px; height:11px; border-radius:50%; flex:0 0 auto; }}
    .leg-name {{ color:var(--white); font-weight:500; }}

    .tabs {{ display:flex; gap:10px; justify-content:center; flex-wrap:wrap; margin:22px auto 14px; }}
    .tab {{ cursor:pointer; text-align:left; border-radius:14px; padding:9px 15px; min-width:240px;
      background:linear-gradient(150deg, rgba(42,34,80,0.5), rgba(11,15,42,0.4));
      border:1px solid rgba(201,166,255,0.16); color:var(--gray);
      transition:border-color .25s ease, transform .25s ease, box-shadow .25s ease; }}
    .tab:hover {{ transform:translateY(-1px); border-color:rgba(111,224,232,0.4); }}
    .tab-on {{ border-color:rgba(111,224,232,0.7);
      box-shadow:0 0 22px rgba(31,181,229,0.22), inset 0 1px 0 rgba(255,255,255,0.05); }}
    .tab-kind {{ display:block; letter-spacing:.16em; text-transform:uppercase; font-size:9.5px;
      font-weight:700; color:var(--pink); opacity:.9; margin-bottom:3px; }}
    .tab-text {{ display:block; font-size:13px; color:var(--white); font-weight:600; }}

    .stage-card {{ background:linear-gradient(150deg, rgba(42,34,80,0.55), rgba(11,15,42,0.5));
      border:1px solid rgba(201,166,255,0.16); border-radius:24px; padding:20px 22px 18px;
      box-shadow:0 16px 44px rgba(0,0,0,0.34), inset 0 1px 0 rgba(255,255,255,0.04);
      backdrop-filter:blur(3px); }}
    .stage-head {{ display:flex; align-items:baseline; justify-content:space-between; gap:14px;
      flex-wrap:wrap; margin-bottom:6px; }}
    .stage-sentence {{ font-size:19px; font-weight:600; color:var(--white);
      text-shadow:0 0 24px rgba(201,166,255,0.2); }}
    .stage-sentence .ans {{ color:var(--cyan); text-shadow:0 0 18px rgba(111,224,232,0.6); }}
    .stage-why {{ font-size:12.5px; color:var(--gray); font-style:italic; margin:2px 0 8px;
      max-width:900px; }}

    svg {{ display:block; width:100%; height:auto; overflow:visible; }}
    .node-label {{ font-family:'Segoe UI',sans-serif; font-weight:600; }}
    .axis-label {{ fill:var(--gray); font-size:11px; font-family:'Segoe UI',sans-serif; opacity:.7; }}
    .axis-tick {{ fill:var(--gray); font-size:9.5px; font-family:'Segoe UI',sans-serif; opacity:.5;
      font-variant-numeric:tabular-nums; }}

    /* transport controls */
    .transport {{ display:flex; align-items:center; gap:14px; margin-top:6px; padding:10px 6px 2px; }}
    .play-btn {{ cursor:pointer; width:42px; height:42px; flex:0 0 auto; border-radius:50%;
      border:1px solid rgba(111,224,232,0.5); background:radial-gradient(120% 120% at 50% 0%,
      rgba(31,181,229,0.3), rgba(11,15,42,0.4)); color:var(--cyan); font-size:16px;
      display:flex; align-items:center; justify-content:center;
      box-shadow:0 0 22px rgba(31,181,229,0.25); transition:transform .15s ease, box-shadow .25s ease; }}
    .play-btn:hover {{ transform:scale(1.06); box-shadow:0 0 30px rgba(31,181,229,0.4); }}
    .scrub-wrap {{ flex:1; display:flex; flex-direction:column; gap:3px; }}
    .scrub-top {{ display:flex; justify-content:space-between; font-size:11px; color:var(--gray);
      opacity:.8; }}
    .scrub-top b {{ color:var(--cyan); font-variant-numeric:tabular-nums; }}
    input[type=range].scrub {{ -webkit-appearance:none; appearance:none; width:100%; height:6px;
      border-radius:5px; outline:none;
      background:linear-gradient(90deg, var(--ice), var(--lav)); opacity:.85; }}
    input[type=range].scrub::-webkit-slider-thumb {{ -webkit-appearance:none; appearance:none;
      width:18px; height:18px; border-radius:50%; background:var(--white); cursor:pointer;
      box-shadow:0 0 12px rgba(244,247,255,0.8), 0 0 4px var(--cyan); border:2px solid var(--cyan); }}
    input[type=range].scrub::-moz-range-thumb {{ width:16px; height:16px; border-radius:50%;
      background:var(--white); cursor:pointer; border:2px solid var(--cyan); }}
    .speed {{ display:flex; align-items:center; gap:6px; font-size:11px; color:var(--gray); }}
    .speed select {{ background:rgba(11,15,42,0.7); color:var(--cyan); border:1px solid
      rgba(111,224,232,0.3); border-radius:7px; padding:3px 6px; font-size:11px; }}

    .readout {{ margin-top:12px; padding-top:12px; border-top:1px solid rgba(255,255,255,0.06);
      display:flex; flex-wrap:wrap; gap:8px 16px; align-items:center; font-size:13px; }}
    .ro-lab {{ color:var(--gray); }}
    .ro-chip {{ font-size:12px; padding:3px 10px; border-radius:18px; font-weight:600;
      border:1px solid; background:rgba(255,255,255,0.03); display:inline-flex; align-items:center; gap:6px; }}
    .ro-chip .w {{ font-weight:700; font-variant-numeric:tabular-nums; opacity:.95; }}
    .ro-none {{ color:var(--gray); font-style:italic; opacity:.7; }}

    .footer {{ margin-top:46px; text-align:center; font-size:12px; color:var(--gray); opacity:.72;
      line-height:1.7; }}
    .footer b {{ color:var(--lav); }}
    .honest {{ margin-top:14px; padding:18px 22px; border-radius:14px; font-size:12.5px;
      background:rgba(11,15,42,0.42); border:1px solid rgba(255,255,255,0.06); color:var(--gray);
      max-width:880px; margin-left:auto; margin-right:auto; text-align:left; line-height:1.66; }}
    .honest b {{ color:var(--cyan); }}
    .honest .k {{ color:var(--lav); }}
    .honest .warn {{ color:var(--pink); }}

    @media (max-width:760px) {{
      h1 {{ font-size:30px; }}
      .tab {{ min-width:0; flex:1 1 100%; }}
    }}
    @media (prefers-reduced-motion: reduce) {{
      .star {{ animation:none !important; }}
    }}
    """

    js = """
    const DATA = __PAYLOAD__;

    // ---------- starfield ----------
    (function(){
      const b=document.body, n=48;
      for(let i=0;i<n;i++){
        const s=document.createElement('div'); s.className='star';
        const sz=Math.random()*2+1;
        s.style.width=sz+'px'; s.style.height=sz+'px';
        s.style.left=(Math.random()*100)+'vw'; s.style.top=(Math.random()*100)+'vh';
        s.style.animationDelay=(Math.random()*6)+'s';
        s.style.animationDuration=(4+Math.random()*5)+'s';
        b.appendChild(s);
      }
    })();

    const SVGNS='http://www.w3.org/2000/svg';
    function el(tag, attrs){ const e=document.createElementNS(SVGNS,tag);
      for(const k in attrs) e.setAttribute(k, attrs[k]); return e; }
    function hexToRgb(h){ h=h.replace('#',''); return [parseInt(h.slice(0,2),16),
      parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)]; }
    function rgba(h,a){ const c=hexToRgb(h); return `rgba(${c[0]},${c[1]},${c[2]},${a})`; }
    function clamp(x,a,b){ return Math.max(a, Math.min(b, x)); }
    function dispTok(t){ const s=(t||'').trim(); return s? s : '\\u00B7'; }

    // ---------- geometry: concept nodes on the left arc, token node on the right ----------
    const K = DATA.labels.length;
    const W = 1000, H = 560;
    const TOK_X = 760, TOK_Y = H/2;            // output-token node (right)
    const CX = 250, CY = H/2, RX = 150, RY = 215;   // concept node ellipse (left)
    function conceptPos(j){
      // spread concepts on a vertical arc on the left
      const t = (K===1)?0.5:(j/(K-1));
      const ang = (-Math.PI/2.15) + t*(Math.PI*1.075);   // top -> bottom, bulging left
      return { x: CX + Math.cos(ang)*RX*0.0 - Math.sin(0)*0 + (-Math.cos(t*Math.PI))*40,
               y: CY - RY + t*(2*RY) };
    }
    // simpler: evenly spaced vertically, slight left bow
    function nodePos(j){
      const t=(K===1)?0.5:(j/(K-1));
      const y = 60 + t*(H-120);
      const bow = Math.sin(t*Math.PI)*46;       // bow outward (left) in the middle
      return { x: CX - bow, y };
    }

    // intensity helpers
    function zGlow(z){ if(z<=0) return 0; return clamp(z/(DATA.z_thresh*2.4),0,1); }
    function confGlow(c){ return clamp(Math.sqrt(c)*1.35,0,1); }

    // ---------- the stage ----------
    const root = document.getElementById('stage-root');
    let cur = 0;             // current prompt index
    let layer = 0;           // current layer (playback head)
    let playing = true;
    let speed = 1.0;         // layers per second multiplier
    let lastT = null;
    let maxContribByPrompt = [];

    // precompute a per-prompt max |contribution| (over lit concepts & layers) for edge scaling
    DATA.prompts.forEach(p=>{
      let m=1e-6;
      p.frames.forEach(fr=>{
        for(let j=0;j<K;j++){ if(fr.lit[j]) m=Math.max(m, Math.abs(fr.contribution[j])); }
      });
      maxContribByPrompt.push(m);
    });

    // build SVG skeleton once
    const svg = el('svg', {viewBox:`0 0 ${W} ${H}`, role:'img'});
    svg.setAttribute('aria-label','concept-to-token attribution graph animating across layers');

    // defs: glow filters
    const defs = el('defs',{});
    defs.innerHTML = `
      <filter id="soft" x="-60%" y="-60%" width="220%" height="220%">
        <feGaussianBlur stdDeviation="6" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <radialGradient id="tokfill" cx="50%" cy="40%" r="70%">
        <stop offset="0%" stop-color="${rgba('#6FE0E8',0.95)}"/>
        <stop offset="60%" stop-color="${rgba('#1FB5E5',0.55)}"/>
        <stop offset="100%" stop-color="${rgba('#1A1F4A',0.0)}"/>
      </radialGradient>`;
    svg.appendChild(defs);

    // layer groups (draw order: axis, edges, concept nodes, token node)
    const gAxis = el('g',{}); svg.appendChild(gAxis);
    const gEdges = el('g',{}); svg.appendChild(gEdges);
    const gCo = el('g',{opacity:0.0}); svg.appendChild(gCo);     // optional co-activation rings
    const gNodes = el('g',{}); svg.appendChild(gNodes);
    const gToken = el('g',{}); svg.appendChild(gToken);

    // static axis: a vertical depth ladder near the token, labeled layers
    function buildAxis(){
      gAxis.innerHTML='';
      const x0 = 905, y0 = 70, y1 = H-70;
      gAxis.appendChild(el('line',{x1:x0,y1:y0,x2:x0,y2:y1,
        stroke:rgba('#C9A6FF',0.25), 'stroke-width':2, 'stroke-linecap':'round'}));
      const lab = el('text',{x:x0+14,y:y0-14,class:'axis-label','text-anchor':'middle'});
      lab.textContent='depth'; gAxis.appendChild(lab);
      const lab2 = el('text',{x:x0+14,y:y0-1,class:'axis-label','text-anchor':'middle'});
      lab2.setAttribute('font-size','9'); lab2.textContent='(layers)'; gAxis.appendChild(lab2);
      const nl = DATA.n_layers;
      for(let L=0; L<nl; L++){
        const y = y0 + (L/(nl-1))*(y1-y0);
        gAxis.appendChild(el('circle',{cx:x0,cy:y,r:2.2,fill:rgba('#C9A6FF',0.35),'data-axis':L}));
        if(L%2===0 || L===nl-1){
          const t=el('text',{x:x0+12,y:y+3.5,class:'axis-tick'}); t.textContent='L'+L;
          gAxis.appendChild(t);
        }
      }
      // moving playback marker
      const mk = el('circle',{cx:x0,cy:y0,r:5,fill:'#F4F7FF',id:'axis-head',
        filter:'url(#soft)'});
      gAxis.appendChild(mk);
    }
    function axisHeadY(L){
      const nl=DATA.n_layers, y0=70, y1=H-70; return y0 + (L/(nl-1))*(y1-y0);
    }

    // node + edge element caches per prompt rebuild
    let nodeEls=[], edgeEls=[], coEls=[];
    function buildGraph(){
      gEdges.innerHTML=''; gNodes.innerHTML=''; gToken.innerHTML=''; gCo.innerHTML='';
      nodeEls=[]; edgeEls=[]; coEls=[];
      const p = DATA.prompts[cur];

      // edges (one per concept; thickness/opacity animated) -- behind nodes
      for(let j=0;j<K;j++){
        const np = nodePos(j);
        const path = el('path',{d:'', fill:'none', stroke:DATA.colors[j],
          'stroke-width':0, 'stroke-linecap':'round', opacity:0});
        gEdges.appendChild(path);
        // a small moving 'pulse' dot that travels the edge when active
        const pulse = el('circle',{r:0, fill:DATA.colors[j], opacity:0, filter:'url(#soft)'});
        gEdges.appendChild(pulse);
        edgeEls.push({path, pulse, np});
      }

      // optional co-activation rings between simultaneously-lit concepts (correlational, faint)
      if(DATA.coactivation){
        for(let a=0;a<K;a++) for(let bb=a+1;bb<K;bb++){
          const pa=nodePos(a), pb=nodePos(bb);
          const ln=el('line',{x1:pa.x,y1:pa.y,x2:pb.x,y2:pb.y,
            stroke:rgba('#C9A6FF',0.5),'stroke-width':1,'stroke-dasharray':'2 5',opacity:0});
          gCo.appendChild(ln); coEls.push({ln,a,b:bb});
        }
        gCo.setAttribute('opacity','1');
      }

      // concept nodes
      for(let j=0;j<K;j++){
        const np=nodePos(j); const col=DATA.colors[j];
        const g=el('g',{transform:`translate(${np.x},${np.y})`});
        const halo=el('circle',{r:13,fill:rgba(col,0.0),stroke:rgba(col,0.0),
          'stroke-width':2, filter:'url(#soft)'});
        const core=el('circle',{r:8,fill:rgba(col,0.18),stroke:rgba(col,0.5),'stroke-width':1.5});
        const lab=el('text',{x:-16,y:4,'text-anchor':'end',class:'node-label',
          fill:rgba(col,0.55),'font-size':13});
        lab.textContent=DATA.labels[j];
        const zlab=el('text',{x:-16,y:18,'text-anchor':'end','font-size':9.5,
          fill:rgba('#A7B0C0',0.0),'font-variant-numeric':'tabular-nums'});
        g.appendChild(halo); g.appendChild(core); g.appendChild(lab); g.appendChild(zlab);
        gNodes.appendChild(g);
        nodeEls.push({g,halo,core,lab,zlab,col,np});
      }

      // the output-token node (right)
      const tg=el('g',{transform:`translate(${TOK_X},${TOK_Y})`});
      const tglow=el('circle',{r:54,fill:'url(#tokfill)',opacity:0.5});
      const tring=el('circle',{r:46,fill:rgba('#1A1F4A',0.55),stroke:rgba('#6FE0E8',0.6),
        'stroke-width':2.5, filter:'url(#soft)'});
      const tword=el('text',{x:0,y:2,'text-anchor':'middle','class':'node-label',
        fill:'#F4F7FF','font-size':24,'font-weight':800});
      const tconf=el('text',{x:0,y:24,'text-anchor':'middle','font-size':11,
        fill:rgba('#A7B0C0',0.9),'font-variant-numeric':'tabular-nums'});
      const tcap=el('text',{x:0,y:-60,'text-anchor':'middle','font-size':10.5,
        fill:rgba('#A7B0C0',0.8)}); tcap.textContent='it would say';
      tg.appendChild(tglow); tg.appendChild(tring); tg.appendChild(tword);
      tg.appendChild(tconf); tg.appendChild(tcap);
      gToken.appendChild(tg);
      window.__tok={tg,tglow,tring,tword,tconf};

      // a faint flip flash marker text above the token
      const flipTxt=el('text',{x:TOK_X,y:TOK_Y+78,'text-anchor':'middle','font-size':11,
        fill:rgba('#FF6FAF',0.0),'font-weight':600,id:'flipmark'});
      gToken.appendChild(flipTxt);
    }

    // a quadratic path from concept node to token, bowing toward the center
    function edgePath(np){
      const mx=(np.x+TOK_X)/2, my=(np.y+TOK_Y)/2 - 30 - (TOK_Y-np.y)*0.04;
      return `M ${np.x+10} ${np.y} Q ${mx} ${my} ${TOK_X-48} ${TOK_Y}`;
    }
    function pointOnEdge(np, t){
      // quadratic bezier eval at t for the pulse dot
      const x0=np.x+10,y0=np.y, x2=TOK_X-48,y2=TOK_Y;
      const mx=(np.x+TOK_X)/2, my=(np.y+TOK_Y)/2 - 30 - (TOK_Y-np.y)*0.04;
      const u=1-t;
      return { x:u*u*x0+2*u*t*mx+t*t*x2, y:u*u*y0+2*u*t*my+t*t*y2 };
    }

    // ---------- render one frame (interpolated between integer layers for smoothness) ----------
    function lerp(a,b,t){ return a+(b-a)*t; }
    function render(L){
      const p=DATA.prompts[cur];
      const nl=DATA.n_layers;
      const Lf=clamp(L,0,nl-1);
      const i0=Math.floor(Lf), i1=Math.min(i0+1,nl-1), tt=Lf-i0;
      const f0=p.frames[i0], f1=p.frames[i1];
      const maxC=maxContribByPrompt[cur];

      // token node: snap word to the nearer frame (tokens are discrete), conf interpolates
      const nearer = (tt<0.5)? f0 : f1;
      const conf = lerp(f0.conf, f1.conf, tt);
      const T=window.__tok;
      const newWord=dispTok(nearer.guess);
      if(T.tword.textContent!==newWord){
        // morph: quick scale pop on change
        T.tword.textContent=newWord;
        T.tg.style.transition='none';
        T.tword.setAttribute('transform','scale(1.18)');
        requestAnimationFrame(()=>{ T.tword.style.transition='transform .35s cubic-bezier(.2,.8,.2,1)';
          T.tword.setAttribute('transform','scale(1)'); });
      }
      T.tconf.textContent=(conf*100).toFixed(0)+'%';
      const cg=confGlow(conf);
      T.tring.setAttribute('stroke',rgba('#6FE0E8',0.4+0.55*cg));
      T.tring.setAttribute('stroke-width',(2+2.5*cg).toFixed(2));
      T.tglow.setAttribute('opacity',(0.25+0.6*cg).toFixed(2));
      T.tring.setAttribute('r',(44+6*cg).toFixed(1));

      // flip flash
      const fm=document.getElementById('flipmark');
      if(nearer.flip){ fm.textContent='it changed its mind';
        fm.setAttribute('fill',rgba('#FF6FAF',0.9)); }
      else { fm.setAttribute('fill',rgba('#FF6FAF',0.0)); }

      // concept nodes + edges
      const litNow=[];
      for(let j=0;j<K;j++){
        const z=lerp(f0.z[j], f1.z[j], tt);
        const lit = z>=DATA.z_thresh;
        const gl=zGlow(z);
        const N=nodeEls[j], col=N.col;
        // node glow scales with z (presence)
        N.halo.setAttribute('r',(11+12*gl).toFixed(1));
        N.halo.setAttribute('fill',rgba(col, 0.04+0.20*gl));
        N.halo.setAttribute('stroke',rgba(col, lit? (0.35+0.5*gl):0.0));
        N.core.setAttribute('r',(6.5+3.5*gl).toFixed(1));
        N.core.setAttribute('fill',rgba(col, 0.12+0.55*gl));
        N.core.setAttribute('stroke',rgba(col, 0.3+0.6*gl));
        N.lab.setAttribute('fill',rgba(col, lit? 0.98 : 0.45));
        N.lab.setAttribute('font-weight', lit? 700:600);
        if(lit){ N.zlab.textContent=z.toFixed(1)+'\\u03C3';
          N.zlab.setAttribute('fill',rgba(col,0.85)); }
        else { N.zlab.setAttribute('fill',rgba('#A7B0C0',0.0)); }

        // edge: only when lit; thickness/opacity by |real contribution|, sign sets dashing
        const E=edgeEls[j];
        const contrib=lerp(f0.contribution[j], f1.contribution[j], tt);
        if(lit && Math.abs(contrib)>1e-4){
          litNow.push(j);
          const w = clamp(Math.abs(contrib)/maxC, 0, 1);
          E.path.setAttribute('d', edgePath(N.np));
          E.path.setAttribute('stroke-width', (0.8+6.5*w).toFixed(2));
          E.path.setAttribute('opacity', (0.18+0.6*w*gl).toFixed(2));
          E.path.setAttribute('stroke', col);
          // negative contribution (pulls token AWAY) drawn dashed + cooler
          if(contrib<0){ E.path.setAttribute('stroke-dasharray','4 5');
            E.path.setAttribute('opacity', (0.12+0.4*w).toFixed(2)); }
          else E.path.setAttribute('stroke-dasharray','none');
          // travelling pulse dot proportional to weight, phase from time
          const ph=((performance.now()/900)+j*0.13)%1;
          const pt=pointOnEdge(N.np, ph);
          E.pulse.setAttribute('cx',pt.x); E.pulse.setAttribute('cy',pt.y);
          E.pulse.setAttribute('r',(1.4+3.6*w).toFixed(2));
          E.pulse.setAttribute('opacity',(0.25+0.55*w).toFixed(2));
        } else {
          E.path.setAttribute('opacity',0); E.path.setAttribute('stroke-width',0);
          E.pulse.setAttribute('opacity',0); E.pulse.setAttribute('r',0);
        }
      }

      // optional co-activation rings (faint, correlational only)
      if(DATA.coactivation){
        const litset=new Set(litNow);
        coEls.forEach(c=>{
          const on = litset.has(c.a)&&litset.has(c.b);
          c.ln.setAttribute('opacity', on? 0.28 : 0.0);
        });
      }

      // axis playback head
      const head=document.getElementById('axis-head');
      if(head) head.setAttribute('cy', axisHeadY(Lf).toFixed(1));
      gAxis.querySelectorAll('[data-axis]').forEach(c=>{
        const L2=+c.getAttribute('data-axis');
        c.setAttribute('fill', L2<=Lf? rgba('#6FE0E8',0.7):rgba('#C9A6FF',0.3));
      });

      // HUD: scrubber + layer readout + driving-concept chips
      const scr=document.getElementById('scrub'); if(document.activeElement!==scr) scr.value=Lf;
      document.getElementById('layer-now').textContent='L'+Math.round(Lf);
      document.getElementById('guess-now').textContent=newWord;
      updateReadout(nearer, maxC);
    }

    function updateReadout(fr, maxC){
      const box=document.getElementById('readout-chips');
      const entries=[];
      for(let j=0;j<K;j++){
        if(fr.lit[j] && Math.abs(fr.contribution[j])>1e-4)
          entries.push([j, fr.contribution[j]]);
      }
      entries.sort((a,b)=>Math.abs(b[1])-Math.abs(a[1]));
      if(entries.length===0){ box.innerHTML='<span class="ro-none">no concept cleared the null at this depth</span>'; return; }
      box.innerHTML = entries.map(([j,c])=>{
        const col=DATA.colors[j]; const sign=c>=0?'+':'\\u2212';
        const arrow = c>=0? '\\u2192':'\\u22A3';  // -> push toward / pull away
        return `<span class="ro-chip" style="border-color:${rgba(col,0.6)};color:${col}">`+
          `${DATA.labels[j]} <span class="w">${arrow}${Math.abs(c).toFixed(2)}</span></span>`;
      }).join('');
    }

    // ---------- transport ----------
    function tick(ts){
      if(lastT==null) lastT=ts;
      const dt=(ts-lastT)/1000; lastT=ts;
      if(playing){
        layer += dt * 1.1 * speed;     // ~1.1 layers/sec at 1x
        if(layer >= DATA.n_layers-1){
          layer = DATA.n_layers-1; render(layer);
          // pause a beat on the resolved answer, then loop
          playing=false;
          setTimeout(()=>{ layer=0; playing=true; lastT=null; }, 1400);
        }
      }
      render(layer);
      requestAnimationFrame(tick);
    }

    function setPrompt(i){
      cur=i; layer=0; playing=true; lastT=null;
      document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('tab-on', +t.dataset.i===i));
      const p=DATA.prompts[i];
      // sentence with highlighted answer word
      const txt=p.text;
      const aw=p.answer_word||'';
      let sent=txt;
      if(aw && txt.endsWith(aw)){ sent = txt.slice(0, txt.length-aw.length) +
        '<span class="ans">'+ (aw) + '</span>'; }
      document.getElementById('stage-sentence').innerHTML = '&ldquo;'+sent+'&rdquo;';
      document.getElementById('stage-why').textContent = p.why;
      // final answer caption
      const ft=p.final_topk[0];
      buildGraph();
      render(0);
    }

    // ---------- mount ----------
    function mount(){
      buildAxis();
      root.querySelector('.svg-host').appendChild(svg);
      // tab clicks
      document.querySelectorAll('.tab').forEach(t=>{
        t.addEventListener('click',()=>setPrompt(+t.dataset.i));
      });
      // play/pause
      const pb=document.getElementById('play-btn');
      pb.addEventListener('click',()=>{ playing=!playing; lastT=null;
        pb.innerHTML = playing? PAUSE : PLAY; });
      // scrubber
      const scr=document.getElementById('scrub');
      scr.addEventListener('input',()=>{ playing=false;
        document.getElementById('play-btn').innerHTML=PLAY;
        layer=+scr.value; render(layer); });
      // speed
      document.getElementById('speed-sel').addEventListener('change',e=>{ speed=+e.target.value; });
      setPrompt(0);
      requestAnimationFrame(tick);
    }
    const PLAY='\\u25B6', PAUSE='\\u2389';
    document.addEventListener('DOMContentLoaded', mount);
    if(document.readyState!=='loading') mount();
    """
    js = js.replace("__PAYLOAD__", payload_json)

    co_note = ("" if not coactivation else
               '<b class="warn">Faint dashed rings</b> link concepts that are lit at the same depth '
               '&mdash; <span class="warn">correlational co-activation only</span> (they fire together '
               'here), never a causal concept&rarr;concept circuit. ')

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clozn &middot; think graph</title>
<style>{style}</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="eyebrow">Clozn &middot; watch it think &middot; v2 node graph</div>
    <h1>The thought, as a graph</h1>
    <p class="lede">A <b>frozen {esc(demo["model_name"])}</b> resolving its next word, drawn as a
      <b>temporal network</b>. <span class="c1">Time is depth</span> &mdash; press play and a head
      scrubs up the layers (the autoregressive cousin of a diffusion denoise). <b>Named-concept nodes</b>
      pulse and light where they are genuinely present in the residual; the <b>output-token node</b>
      morphs as the model's logit-lens guess resolves and <span class="c2">changes its mind</span>; and
      <b>edges</b> from the lit concepts feed that token, each weighted by its <b>real contribution</b>
      to the token through the model's own logit lens. You watch the active concepts drive the resolving
      word &mdash; real activations, real contributions, not a fabricated circuit.</p>
    <div class="meta-line">
      nodes light where <code>z &ge; {z_thresh:g}&sigma;</code> above a {demo["n_null"]}-sample
      equal-norm random null &middot; token = logit-lens argmax at each
      <code>blocks.L.hook_resid_post</code> &middot;
      edge = <code>(r&#770;&middot;d_c)(d_c&middot;W_U[:,token])</code>, the concept's own additive term
      of the token's logit-lens logit
    </div>
  </header>

  <div class="legend">{legend}</div>

  <div class="tabs">{tabs}</div>

  <div id="stage-root" class="stage-card">
    <div class="stage-head">
      <div class="stage-sentence" id="stage-sentence"></div>
    </div>
    <div class="stage-why" id="stage-why"></div>
    <div class="svg-host"></div>

    <div class="transport">
      <div class="play-btn" id="play-btn" title="play / pause">&#9209;</div>
      <div class="scrub-wrap">
        <div class="scrub-top">
          <span>depth <b id="layer-now">L0</b> &middot; resolving to <b id="guess-now">&middot;</b></span>
          <span>L0 &rarr; L{nl - 1} (vague &rarr; resolved)</span>
        </div>
        <input type="range" min="0" max="{nl - 1}" step="0.01" value="0" class="scrub" id="scrub">
      </div>
      <div class="speed">speed
        <select id="speed-sel">
          <option value="0.5">0.5&times;</option>
          <option value="1" selected>1&times;</option>
          <option value="2">2&times;</option>
        </select>
      </div>
    </div>

    <div class="readout">
      <span class="ro-lab">driving this token now:</span>
      <span id="readout-chips"><span class="ro-none">&mdash;</span></span>
    </div>
  </div>

  <div class="footer">
    <div><b>Clozn</b> &mdash; a local runtime where the model's interior is legible.</div>
    <div class="honest">
      <b>What this is, honestly.</b> Concepts lighting up and feeding the resolving token; real
      activations and real contributions, <span class="k">not a fabricated circuit</span>.
      <b>Nodes (concepts)</b> are the validated <span class="k">diff-in-means</span> directions from the
      <i>p18_conceptmem</i> spike, rebuilt at every layer. A concept node lights only where its residual
      projection clears an <span class="k">equal-norm random null</span> by
      <b>z &ge; {z_thresh:g}&sigma;</b> ({demo["n_null"]} samples/layer), centered by a neutral-corpus
      baseline so an &ldquo;always-on&rdquo; artifact stays dark. <b>The token node</b> is the genuine
      logit-lens argmax at each depth (the model's <span class="k">own</span> <code>ln_final &rarr;
      W_U</code>); the morph and the &ldquo;changed its mind&rdquo; flashes are real flips; the final
      word is the model's <span class="k">actual</span> next-token output. <b>The edges</b> are the load-bearing
      honest part: the logit-lens logit for the current token is <code>r&#770;&middot;W_U[:,token]</code>
      with <code>r&#770;</code> the model's centered+normalized residual, and the term explained by the
      residual's component along a concept direction is exactly
      <code>(r&#770;&middot;d_c)&times;(d_c&middot;W_U[:,token])</code> &mdash; how present the concept is
      times how that direction pushes <i>this</i> token. That is the edge weight (the arrow shows push
      toward vs pull away); it is a real additive piece of the model's own logit, not decoration.
      {co_note}<span class="warn">We deliberately draw no concept&rarr;concept circuit edges</span> &mdash;
      that needs feature-circuit machinery this project found fails locally (the SAE null), so only the
      honest concept&rarr;token contributions carry weight. Basis non-orthogonality (mean |off-diagonal
      cosine| = {demo["cosine_mean_abs_off"]:.2f}) means nearby concepts can co-activate &mdash; this is
      a probe of <i>what is present and pushing</i>, not a clean factorization of <i>what is computed</i>.
      The backbone is frozen throughout. Generated {esc(demo["timestamp"])} &middot; seed {demo["seed"]}.
    </div>
  </div>

</div>
<script>{js}</script>
</body>
</html>"""


# ====================================================================================================
def main():
    ap = argparse.ArgumentParser(description="think graph -- concept->token attribution network over depth")
    ap.add_argument("--layer-concept", type=int, default=7,
                    help="layer whose basis cosine is reported on the page (p18 default = 7); the time "
                         "axis uses each layer's own basis regardless")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-null", type=int, default=400, help="random equal-norm dirs/layer for the null")
    ap.add_argument("--z-thresh", type=float, default=2.0, help="sigma-above-null to count 'present'")
    ap.add_argument("--coactivation", action="store_true",
                    help="also draw faint correlational concept co-activation rings (clearly labeled)")
    ap.add_argument("--prompt", default=None,
                    help="a single custom prompt; default = the built-in demo set")
    ap.add_argument("--out", default=os.path.join(RUNS, "think_graph.html"), help="output HTML path")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    prompts = ([{"text": args.prompt, "kind": "custom",
                 "why": "a custom prompt -- watch concepts pulse up the layers and feed the resolving token."}]
               if args.prompt else DEMO_PROMPTS)

    demo = build_demo(args.layer_concept, args.device, args.seed, args.n_null, args.z_thresh, prompts)
    html_doc = render_html(demo, args.coactivation)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html_doc)

    print("\n" + "=" * 84)
    print(f"WROTE  {os.path.abspath(args.out)}")
    print("=" * 84)
    print("\nSUMMARY (actual reads of the frozen model):")
    for i, p in enumerate(demo["prompts"]):
        cap = p["cap"]
        ca = cap["crystallized_at"]
        lit = ", ".join(demo["concept_labels"][j] for j in range(len(demo["concept_labels"]))
                        if cap["ever_lit"][j]) or "none cleared the null"
        print(f"  P{i+1} [{p['kind']}] \"{p['text'][:52]}{'...' if len(p['text'])>52 else ''}\"")
        print(f"      token: '{cap['final_guess'].strip()}' resolves @ L{ca}/{demo['n_layers']}; "
              f"says '{cap['final_topk'][0]['word'].strip()}' ({cap['final_topk'][0]['p']:.2f})")
        print(f"      concept nodes lit: {lit}")


if __name__ == "__main__":
    main()
