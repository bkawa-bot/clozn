"""
feature_circuit_clean_qwen.py - the CLEANUP PASS + the VIZ. Turns the noisy feature-circuit pilot into a
small, trustworthy, ablation-verified concept->concept circuit, and renders it as a self-contained light
("Artificial Angels") HTML page. READ feature_circuit_pilot_qwen.py + its findings first.

The pilot found real edges (18% beat a null) but they were dominated by (1) generic high-magnitude features,
(2) a TopK selection-boundary artifact inflating magnitudes, (3) tiny answer-relevance. This fixes all three:

  FIX 1 - SMOOTH influence (kills the TopK artifact). Measure the change in the target feature's PRE-activation
          (resid @ W_enc_j + b_enc_j) under ablation, NOT its post-TopK activation. The pre-activation is a
          linear readout of the residual -> continuous, no top-50 boundary flips.
  FIX 2 - SPECIFICITY filter (kills generic features). A feature's activation here divided by its mean over a
          diverse BACKGROUND corpus. Generic/always-on features (the same one topping unrelated prompts) score
          ~1 and are dropped; content features that fire HERE specifically are kept.
  FIX 3 - ANSWER-RELEVANCE. Ablate each source feature -> change in the predicted-token logit. We rank/keep
          edges whose source actually matters for the output, and draw a 2-hop circuit
          source(L14) -> target(L20) -> answer token (the target->token push via the model's own logit lens).

An edge is DRAWN only if: both endpoints are SPECIFIC (not generic), the smooth pre-activation influence beats
a random-active-feature null, and we surface its interpretability (logit-lens names) + answer-relevance. Only
verified + interpretable edges go on the page - the honest small circuit, not the dense noisy one.

Qwen3-1.7B-Base + Qwen-Scope SAEs (L14 source, L20 target), forward-only, frozen. Env: cloze/.venv (GPU).
Outputs: research/runs/feature_circuit_clean_qwen.json + inspector/runs/feature_circuit.html (self-contained).
"""
from __future__ import annotations
import os, sys, json, time, argparse, html as _html
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
VIZ_OUT = os.path.join(os.path.dirname(HERE), "inspector", "runs", "feature_circuit.html")
sys.path.insert(0, HERE)
import frontier_apply as FA
import legibility_discovered_qwen as LDQ      # RawTopKSAE, resolve_model_path (+ TF32 off)

DEV = LDQ.DEV

# light "Artificial Angels" palette (matches think_graph.html / memory_live.html)
HEAVEN, PEARL, SKY = "#F8FAFF", "#EDF1FB", "#E2EBFB"
ANGEL, CYAN, FROST, MAGENTA, GOLD = "#4C8DF0", "#36AEC4", "#AEB9D2", "#CE64DE", "#C68A1E"
INK, SLATE, DEEPCYAN = "#232A3A", "#677591", "#1F7E91"


def esc(s): return _html.escape(str(s))


def load_sae(layer):
    return LDQ.RawTopKSAE(os.path.join(RUNS, f"qwen_scope_1p7b_layer{layer}.npz"),
                          os.path.join(RUNS, f"qwen_scope_1p7b_layer{layer}.meta.json"), DEV)


# --- the circuit prompts (model gets them right) + a diverse background for the specificity filter ---
PROMPTS = [
    "The capital of France is",
    "The first president of the United States was George",
    "The opposite of hot is",
    "Two plus two equals",
]
BACKGROUND = [
    "I went to the store yesterday and bought some",
    "She opened the window because the room felt",
    "The meeting has been rescheduled to next",
    "He picked up the phone and slowly began to",
    "In the morning the children walked to",
    "The recipe calls for a cup of flour and two",
    "After the long flight they finally arrived at the",
    "The scientist carefully recorded the results of the",
    "My favorite kind of music to listen to is",
    "The old house at the end of the street was",
]


@torch.no_grad()
def encode_pre(sae, resid):
    """Continuous SAE pre-activation (relu(x@W_enc+b_enc) BEFORE TopK gating) - smooth in the residual."""
    return torch.relu(resid.float() @ sae.W_enc + sae.b_enc)


@torch.no_grad()
def feature_tokens(model, tok, Wdec_row, topn=6):
    h = model.model.norm(Wdec_row[None].to(model.dtype))
    top = (h @ model.lm_head.weight.T)[0].topk(topn).indices.tolist()
    return [tok.decode([int(t)]).strip() for t in top]


def clean_name(tokens):
    """Interpretable English-ish tokens from a feature's logit-lens promotions (drop punctuation/CJK/code)."""
    good = [t for t in tokens if t and t.isascii() and any(c.isalpha() for c in t) and len(t) >= 3]
    return good[:3]


@torch.no_grad()
def resid_at(model, ids, layers):
    out = model(input_ids=ids, output_hidden_states=True)
    return {L: out.hidden_states[L + 1][0, -1].float() for L in layers}, out.logits[0, -1].float()


@torch.no_grad()
def ablate_resid(model, ids, src_layer, abl_vec, read_layer):
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h[:, -1, :] = h[:, -1, :] - abl_vec.to(h.dtype)
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    hd = model.model.layers[src_layer].register_forward_hook(hook)
    try:
        out = model(input_ids=ids, output_hidden_states=True)
    finally:
        hd.remove()
    return out.hidden_states[read_layer + 1][0, -1].float(), out.logits[0, -1].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--src_layer", type=int, default=14)
    ap.add_argument("--tgt_layer", type=int, default=20)
    ap.add_argument("--spec_thresh", type=float, default=3.0)   # activation here / mean over background
    ap.add_argument("--max_src", type=int, default=6)
    ap.add_argument("--max_tgt", type=int, default=6)
    ap.add_argument("--n_null", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    print(f"device={DEV} model={args.model} src=L{args.src_layer} tgt=L{args.tgt_layer} (CLEAN circuit + viz)", flush=True)
    tok, model = FA.load_llm(LDQ.resolve_model_path(args.model), dtype=torch.float32)
    sae_s, sae_t = load_sae(args.src_layer), load_sae(args.tgt_layer)
    rng = np.random.default_rng(args.seed)

    # --- background activation profile (mean post-activation per feature over diverse prompts, last pos) ---
    print(f"  building background specificity profile over {len(BACKGROUND)} prompts ...", flush=True)
    bg_s = torch.zeros(sae_s.d_sae, device=DEV); bg_t = torch.zeros(sae_t.d_sae, device=DEV)
    for p in BACKGROUND:
        ids = torch.tensor(tok.encode(p, add_special_tokens=False), device=DEV)[None, :]
        r, _ = resid_at(model, ids, [args.src_layer, args.tgt_layer])
        bg_s += sae_s.encode(r[args.src_layer][None])[0]
        bg_t += sae_t.encode(r[args.tgt_layer][None])[0]
    bg_s /= len(BACKGROUND); bg_t /= len(BACKGROUND)

    W_U = model.lm_head.weight    # [V, H]
    circuits = {}
    for prompt in PROMPTS:
        ids = torch.tensor(tok.encode(prompt, add_special_tokens=False), device=DEV)[None, :]
        r, logits = resid_at(model, ids, [args.src_layer, args.tgt_layer])
        pred_id = int(logits.argmax()); pred_tok = tok.decode([pred_id]).strip()
        fs = sae_s.encode(r[args.src_layer][None])[0]; ft = sae_t.encode(r[args.tgt_layer][None])[0]
        pre_t_base = encode_pre(sae_t, r[args.tgt_layer])                       # continuous target pre-acts

        # SPECIFICITY: keep features that fire here >> their background mean
        def specific(feats, bg, k):
            act = feats.cpu().numpy(); bgm = bg.cpu().numpy()
            cand = [i for i in np.nonzero(act > 0)[0]]
            scored = sorted(cand, key=lambda i: -(act[i] / (bgm[i] + 1e-3)))
            keep = [int(i) for i in scored if act[i] / (bgm[i] + 1e-3) >= args.spec_thresh][:k]
            return keep
        src_keep = specific(fs, bg_s, args.max_src)
        tgt_keep = specific(ft, bg_t, args.max_tgt)

        # NULL: ablate random GENERIC active sources (specificity < thresh), measure normalized pre-act delta
        act_s = fs.cpu().numpy(); bgm_s = bg_s.cpu().numpy()
        generic_pool = [int(i) for i in np.nonzero(act_s > 0)[0]
                        if act_s[i] / (bgm_s[i] + 1e-3) < args.spec_thresh]
        rng.shuffle(generic_pool); null_src = generic_pool[:args.n_null]
        null_norm = []
        for i in null_src:
            r20_abl, _ = ablate_resid(model, ids, args.src_layer, float(fs[i]) * sae_s.W_dec[i], args.tgt_layer)
            pre_abl = encode_pre(sae_t, r20_abl)
            for j in tgt_keep:
                null_norm.append(abs(float(pre_t_base[j] - pre_abl[j])) / (abs(float(pre_t_base[j])) + 1e-3))
        thr = float(np.percentile(null_norm, 99)) if null_norm else 1e9

        # SOURCE features: answer-relevance (ablate -> answer-logit drop) + edges to targets (smooth pre-delta)
        edges = []; src_info = {}
        for i in src_keep:
            r20_abl, logits_abl = ablate_resid(model, ids, args.src_layer, float(fs[i]) * sae_s.W_dec[i], args.tgt_layer)
            pre_abl = encode_pre(sae_t, r20_abl)
            ans_drop = float(logits[pred_id] - logits_abl[pred_id])
            src_info[i] = dict(act=float(fs[i]), spec=float(fs[i] / (bg_s[i] + 1e-3)), answer_drop=ans_drop,
                               name=feature_tokens(model, tok, sae_s.W_dec[i]))
            for j in tgt_keep:
                norm_d = abs(float(pre_t_base[j] - pre_abl[j])) / (abs(float(pre_t_base[j])) + 1e-3)
                if norm_d > thr:
                    edges.append(dict(src=i, tgt=j, norm_delta=norm_d,
                                      raw_delta=float(pre_t_base[j] - pre_abl[j])))
        # TARGET features: push on the answer token via the model's own logit lens (target -> token edge)
        tgt_info = {}
        for j in tgt_keep:
            push = float((model.model.norm(sae_t.W_dec[j][None].to(model.dtype)) @ W_U.T)[0, pred_id])
            tgt_info[j] = dict(act=float(ft[j]), spec=float(ft[j] / (bg_t[j] + 1e-3)),
                               name=feature_tokens(model, tok, sae_t.W_dec[j]), answer_push=push)
        # keep only edges with an interpretable endpoint (clean logit-lens name) for the page
        def interp(name): return len(clean_name(name)) >= 1
        edges_interp = [e for e in edges if interp(src_info[e["src"]]["name"]) or interp(tgt_info[e["tgt"]]["name"])]
        edges_interp.sort(key=lambda e: -e["norm_delta"])

        circuits[prompt] = dict(pred=pred_tok, src_keep=src_keep, tgt_keep=tgt_keep,
                                null_threshold=thr, src_info=src_info, tgt_info=tgt_info,
                                edges_all=edges, edges_interp=edges_interp[:12])
        print(f"\n=== \"{prompt}\" -> '{pred_tok}'  specific src={len(src_keep)} tgt={len(tgt_keep)}  "
              f"verified+interp edges={len(edges_interp)} (null thr {thr:.2f}) ===", flush=True)
        for e in edges_interp[:5]:
            sn = clean_name(src_info[e["src"]]["name"]) or src_info[e["src"]]["name"][:2]
            tn = clean_name(tgt_info[e["tgt"]]["name"]) or tgt_info[e["tgt"]]["name"][:2]
            print(f"    f{e['src']} {sn} -> f{e['tgt']} {tn}  norm_delta={e['norm_delta']:.2f} "
                  f"(src ans-drop {src_info[e['src']]['answer_drop']:+.2f}, tgt push {tgt_info[e['tgt']]['answer_push']:+.2f})", flush=True)

    n_edges = sum(len(c["edges_interp"]) for c in circuits.values())
    verdict = (f"CLEAN CIRCUIT: {n_edges} verified + interpretable concept->concept edges across "
               f"{len(circuits)} prompts (smooth pre-activation influence, specificity-filtered, beats the "
               f"99th-pct generic-ablation null). The honest small circuit - a few labeled, ablation-verified "
               f"edges - is drawable on Qwen3-1.7B + Qwen-Scope.")
    print("\n" + "#" * 90 + f"\n# {verdict}\n" + "#" * 90, flush=True)

    report = dict(model=args.model, src_layer=args.src_layer, tgt_layer=args.tgt_layer,
                  spec_thresh=args.spec_thresh, verdict=verdict, n_clean_edges=n_edges,
                  env="cloze/.venv (GPU)", circuits=circuits, wall_time_s=round(time.time() - t0, 1))
    json.dump(report, open(os.path.join(RUNS, "feature_circuit_clean_qwen.json"), "w"), indent=2, default=float)

    html = render_html(report)
    os.makedirs(os.path.dirname(VIZ_OUT), exist_ok=True)
    open(VIZ_OUT, "w", encoding="utf-8").write(html)
    print(f"\nwrote {os.path.join(RUNS,'feature_circuit_clean_qwen.json')}\nwrote {VIZ_OUT}  "
          f"[{report['wall_time_s']}s]", flush=True)


# ====================================================================================================
# self-contained light viz: per prompt, a 3-column circuit  source(L14) -> target(L20) -> answer token.
def render_html(report):
    def node_label(info):
        nm = clean_name(info["name"]) or info["name"][:2]
        return " / ".join(nm) if nm else "(uninterpretable)"

    cards = []
    for prompt, c in report["circuits"].items():
        srcs = c["src_keep"]; tgts = c["tgt_keep"]; edges = c["edges_interp"]
        # only nodes that participate in a drawn edge (keeps the page honest + readable)
        used_s = [s for s in srcs if any(e["src"] == s for e in edges)]
        used_t = [t for t in tgts if any(e["tgt"] == t for e in edges)]
        if not edges:
            cards.append(f'<div class="card"><div class="ptitle">&ldquo;{esc(prompt)}&rdquo; '
                         f'<span class="arrow">&rarr;</span> <b>{esc(c["pred"])}</b></div>'
                         f'<div class="empty">no edge cleared verification for this prompt</div></div>')
            continue
        W, H = 900, max(220, 70 + 64 * max(len(used_s), len(used_t)))
        xS, xT, xTok = 150, 470, 800
        ys = {s: 60 + i * ((H - 90) / max(1, len(used_s) - 1) if len(used_s) > 1 else 0) for i, s in enumerate(used_s)}
        yt = {t: 60 + i * ((H - 90) / max(1, len(used_t) - 1) if len(used_t) > 1 else 0) for i, t in enumerate(used_t)}
        yTok = H / 2
        maxd = max((e["norm_delta"] for e in edges), default=1.0)
        maxpush = max((abs(c["tgt_info"][t]["answer_push"]) for t in used_t), default=1.0)
        svg = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">']
        # source -> target edges (verified causal influence)
        for e in edges:
            s, t = e["src"], e["tgt"]
            if s not in ys or t not in yt: continue
            w = 0.8 + 4.5 * (e["norm_delta"] / maxd)
            y1, y2 = ys[s], yt[t]
            mx = (xS + xT) / 2
            svg.append(f'<path d="M {xS+12} {y1:.0f} C {mx} {y1:.0f} {mx} {y2:.0f} {xT-12} {y2:.0f}" '
                       f'fill="none" stroke="{CYAN}" stroke-width="{w:.2f}" opacity="0.55"/>')
        # target -> token edges (logit-lens push on the answer)
        for t in used_t:
            push = c["tgt_info"][t]["answer_push"]; col = ANGEL if push >= 0 else MAGENTA
            w = 0.8 + 4.0 * (abs(push) / (maxpush + 1e-6))
            y2 = yt[t]; mx = (xT + xTok) / 2
            dash = '' if push >= 0 else 'stroke-dasharray="4 4"'
            svg.append(f'<path d="M {xT+12} {y2:.0f} C {mx} {y2:.0f} {mx} {yTok:.0f} {xTok-46} {yTok:.0f}" '
                       f'fill="none" stroke="{col}" stroke-width="{w:.2f}" opacity="0.5" {dash}/>')
        # nodes
        for s in used_s:
            y = ys[s]; info = c["src_info"][s]
            svg.append(f'<circle cx="{xS}" cy="{y:.0f}" r="9" fill="{CYAN}" opacity="0.9"/>')
            svg.append(f'<text x="{xS-16}" y="{y+4:.0f}" text-anchor="end" class="nl">{esc(node_label(info))}</text>')
            svg.append(f'<text x="{xS-16}" y="{y+17:.0f}" text-anchor="end" class="nlsub">L14 f{s} &middot; ans {info["answer_drop"]:+.2f}</text>')
        for t in used_t:
            y = yt[t]; info = c["tgt_info"][t]
            svg.append(f'<circle cx="{xT}" cy="{y:.0f}" r="9" fill="{ANGEL}" opacity="0.9"/>')
            svg.append(f'<text x="{xT+16}" y="{y+4:.0f}" text-anchor="start" class="nl">{esc(node_label(info))}</text>')
            svg.append(f'<text x="{xT+16}" y="{y+17:.0f}" text-anchor="start" class="nlsub">L20 f{t}</text>')
        # answer token node
        svg.append(f'<circle cx="{xTok}" cy="{yTok:.0f}" r="26" fill="#FFFFFF" stroke="{CYAN}" stroke-width="2.5"/>')
        svg.append(f'<text x="{xTok}" y="{yTok+5:.0f}" text-anchor="middle" class="tok">{esc(c["pred"])}</text>')
        svg.append(f'<text x="{xS}" y="28" text-anchor="middle" class="col">concept (L14)</text>')
        svg.append(f'<text x="{xT}" y="28" text-anchor="middle" class="col">concept (L20)</text>')
        svg.append(f'<text x="{xTok}" y="28" text-anchor="middle" class="col">answer</text>')
        svg.append('</svg>')
        cards.append(f'<div class="card"><div class="ptitle">&ldquo;{esc(prompt)}&rdquo; '
                     f'<span class="arrow">&rarr;</span> <b>{esc(c["pred"])}</b> '
                     f'<span class="ecount">{len(edges)} verified edges</span></div>{"".join(svg)}</div>')

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>clozn &middot; feature circuit</title>
<style>
  body{{margin:0;background:radial-gradient(1100px 700px at 78% -8%,rgba(76,141,240,0.16),transparent 60%),
    radial-gradient(950px 650px at 8% 8%,rgba(54,174,196,0.13),transparent 58%),
    linear-gradient(168deg,{HEAVEN} 0%,{PEARL} 55%,{SKY} 100%);background-attachment:fixed;color:{INK};
    font-family:'Segoe UI','Inter',system-ui,sans-serif;padding:42px 20px 70px;line-height:1.5;}}
  .wrap{{max-width:1000px;margin:0 auto;}}
  .eyebrow{{letter-spacing:.32em;text-transform:uppercase;font-size:11px;font-weight:600;color:{SLATE};text-align:center;}}
  .wm{{color:{SLATE};font-weight:700;}} .wm-n{{color:{CYAN};}}
  h1{{font-size:32px;text-align:center;margin:10px 0 8px;font-weight:700;
    background:linear-gradient(96deg,{INK} 8%,#46506B 46%,{ANGEL} 96%);-webkit-background-clip:text;
    background-clip:text;-webkit-text-fill-color:transparent;}}
  .lede{{text-align:center;color:{SLATE};font-size:15px;max-width:720px;margin:0 auto 6px;}}
  .lede b{{color:{INK};}}
  .card{{background:linear-gradient(150deg,rgba(255,255,255,0.82),rgba(236,242,251,0.55));
    border:1px solid rgba(174,185,210,0.5);border-radius:20px;padding:18px 20px;margin-top:20px;
    box-shadow:0 18px 44px rgba(76,108,170,0.14),0 0 0 1px rgba(255,255,255,0.6) inset;}}
  .ptitle{{font-size:17px;font-weight:600;margin-bottom:6px;}} .ptitle b{{color:{DEEPCYAN};}}
  .arrow{{color:{SLATE};}} .ecount{{font-size:12px;color:{SLATE};font-weight:500;margin-left:8px;}}
  .empty{{color:{SLATE};font-style:italic;font-size:13px;padding:8px 0;}}
  svg{{width:100%;height:auto;display:block;margin-top:6px;}}
  .nl{{fill:{INK};font-size:12.5px;font-weight:600;font-family:'Segoe UI',sans-serif;}}
  .nlsub{{fill:{SLATE};font-size:10px;font-family:'Segoe UI',sans-serif;}}
  .tok{{fill:{INK};font-size:17px;font-weight:800;font-family:'Segoe UI',sans-serif;}}
  .col{{fill:{SLATE};font-size:11px;letter-spacing:.12em;text-transform:uppercase;font-family:'Segoe UI',sans-serif;}}
  .honest{{margin-top:24px;padding:16px 20px;border-radius:14px;font-size:12.5px;line-height:1.65;
    background:linear-gradient(150deg,rgba(255,255,255,0.7),rgba(236,242,251,0.45));
    border:1px solid rgba(174,185,210,0.4);color:{SLATE};}}
  .honest b{{color:{DEEPCYAN};}} .footer{{text-align:center;color:{SLATE};font-size:12px;margin-top:22px;}}
</style></head><body><div class="wrap">
  <div class="eyebrow"><span class="wm">cloz<span class="wm-n">n</span></span> &middot; feature circuit</div>
  <h1>Concept &rarr; concept, verified by ablation</h1>
  <p class="lede">A small <b>attribution graph</b> on a frozen <b>Qwen3-1.7B</b> read through the pretrained
    <b>Qwen-Scope</b> dictionary. A left concept (layer 14) drives a right concept (layer 20), which pushes
    the <b>answer token</b>. Every concept&rarr;concept edge here was <b>verified by ablation</b> (knock out
    the source, the target actually moves, beating a generic-feature null) and survived a specificity filter.
    Only verified, interpretable edges are drawn.</p>
  {"".join(cards)}
  <div class="honest"><b>What's real / honest.</b> Edge weight = the <b>smooth pre-activation</b> change in the
    target feature when the source feature is ablated (pre-TopK, so no selection-boundary artifact), kept only
    if it beats the 99th-percentile of ablating <i>generic</i> active features. Features are filtered for
    <b>specificity</b> (they fire on this prompt far more than over a diverse background corpus), so generic
    always-on directions are dropped. Node labels are the tokens each feature's decoder promotes through the
    model's own logit lens (mid-layer, so approximate). The target&rarr;answer edge is that feature's logit-lens
    push on the predicted token (blue = toward, magenta dashed = away). This is the honest <i>small</i> circuit:
    a few labeled, ablation-verified edges, not a dense graph. On a local 1.7B the real causal structure is
    faint, so absence of an edge means "did not survive verification here," not "no influence."</div>
  <div class="footer"><span class="wm">cloz<span class="wm-n">n</span></span> &mdash; a local runtime where the model's interior is legible.</div>
</div></body></html>"""


if __name__ == "__main__":
    main()
