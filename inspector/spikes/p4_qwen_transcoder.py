"""
Phase-4 — SAE vs TRANSCODER on a real transformer's MLP (Qwen). Tests the SOTA claim that
transcoder features (predict the MLP's OUTPUT from its input) beat SAE features (reconstruct the
input) on interpretability. Mirrors the RWKV-169m head-to-head, now on a transformer's MLP block.
Themed corpus so feature coherence (purity vs seeded themes) is directly comparable.

Usage: python spikes/p4_qwen_transcoder.py [model]
Run with the cloze venv python (GPU).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.discover import corpus, sae_features, standardize   # noqa: E402
from clozn.viz import render_discovered_features               # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B"
SHORT = NAME.split("/")[-1]


def load(name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(dev).eval()
    return model, tok, dev, torch


def collect_mlp_io(model, tok, torch, dev, texts, layer):
    """Hook decoder layer L's MLP, capture (input, output) per token over the themed corpus."""
    buf: list = []
    mlp = model.model.layers[layer].mlp
    h = mlp.register_forward_hook(
        lambda m, i, o: buf.append((i[0].detach().float().cpu().numpy(),
                                    (o[0] if isinstance(o, tuple) else o).detach().float().cpu().numpy())))
    Xin, Xout, toks = [], [], []
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids.to(dev)
        buf.clear()
        with torch.no_grad():
            model(ids)
        a_in, a_out = buf[0]                                    # [1, seq, hidden] each
        for p in range(a_in.shape[1]):
            Xin.append(a_in[0, p]); Xout.append(a_out[0, p])
            toks.append(tok.decode([int(ids[0, p])]))
    h.remove()
    return np.stack(Xin), np.stack(Xout), toks


def best(Xs, toks, Y):
    b = None
    for l1 in (4e-2, 8e-2, 0.16):
        feats, _, _ = sae_features(Xs, toks, m=256, l1=l1, steps=600, Y=Y)
        mp = np.mean([f.purity for f in feats]) if feats else 0.0
        if b is None or mp > b[0]:
            b = (mp, sum(f.purity >= 0.6 for f in feats), feats)
    return b


def main():
    model, tok, dev, torch = load(NAME)
    nl = model.config.num_hidden_layers
    layers = [nl // 4, nl // 2, 3 * nl // 4]
    texts = corpus()
    print(f"{NAME} on {dev} ({nl} layers); SAE vs transcoder on the MLP, themed corpus ({len(texts)} sentences)")
    print(f"\n  {'layer':<8}{'SAE purity':<14}{'TRANSCODER purity':<20}winner")
    results = []
    for L in layers:
        Xin, Xout, toks = collect_mlp_io(model, tok, torch, dev, texts, L)
        Xs, _, _ = standardize(Xin)
        Ys, _, _ = standardize(Xout)
        s_mp, s_coh, _ = best(Xs, toks, None)
        t_mp, t_coh, t_feats = best(Xs, toks, Ys)
        win = "transcoder" if t_mp > s_mp + 0.03 else ("SAE" if s_mp > t_mp + 0.03 else "~tie")
        print(f"  L{L:<7}{s_mp*100:>4.0f}% ({s_coh})       {t_mp*100:>4.0f}% ({t_coh})            {win}")
        results.append((L, s_mp, t_mp, t_feats))

    L, s, t, tf = max(results, key=lambda r: r[2])
    print(f"\n=== top transcoder features @ L{L} ===")
    for f in tf[:10]:
        tag = f"[{f.theme} {f.purity*100:.0f}%]" if f.purity >= 0.6 else "[mixed]"
        print(f"  f{f.idx:<3} {tag:<16} {' '.join(repr(x.strip()) for x in f.top_tokens[:6])}")
    sb, tb = max(r[1] for r in results), max(r[2] for r in results)
    print(f"\n  OVERALL best SAE {sb*100:.0f}% vs transcoder {tb*100:.0f}% -> "
          + ("transcoder wins (more interpretable on a transformer MLP, as the field claims)"
             if tb > sb + 0.03 else "SAE competitive at this scale (transcoder edge didn't show)"))

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs",
                       f"transcoder_qwen_{SHORT}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_discovered_features(tf, title=f"Clozn · Transcoder features ({SHORT} MLP)",
                subtitle=f"{NAME} · MLP layer {L} input→output · themed corpus"))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
