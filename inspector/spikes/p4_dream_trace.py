"""
Phase-4 (novel) — feature discovery ACROSS a diffusion model's denoising trajectory.

Denoise WikiText-prompt continuations through a real diffusion LM, hook its backbone, capture
activations at every denoising pass tagged by whether each slot is still MASKED or already
RESOLVED, train an SAE, and split features into STRUCTURAL (fire on the masked canvas) vs CONTENT
(fire once the slot resolves). Nobody has feature dictionaries over a diffusion trajectory.

Usage: python spikes/p4_dream_trace.py [open-dcoder|dream7b] [n_prompts] [layer]
Run with the cloze venv python (GPU + cloze_lab + bitsandbytes for Dream-7B nf4).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                 "..", "cloze", "lab")))

import numpy as np  # noqa: E402

from clozn.corpora import text_stream                            # noqa: E402
from clozn.discover import TinySAE, describe_sae, standardize    # noqa: E402
from clozn.viz import render_trajectory_features                 # noqa: E402
from cloze_lab.generate import generate, GenerateConfig          # noqa: E402
from cloze_lab.models.base import LoadConfig                     # noqa: E402
from cloze_lab.scheduler.events import TokensCommitted           # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "open-dcoder"
N_PROMPTS = int(sys.argv[2]) if len(sys.argv) > 2 else (100 if MODEL == "dream7b" else 250)
LAYER_OVERRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else None
PROMPT_LEN, MAX_NEW, STEPS = 24, 8, 8


def backbone_layers(model):
    """Find the decoder layer stack generically (the longest ModuleList of >=8 blocks) — works for
    Qwen2 (open-dCoder) and Dream's custom architecture alike, no hard-coded module path."""
    import torch.nn as nn
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 8:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    if best is None:
        raise RuntimeError("could not locate a decoder layer stack to hook")
    return best


def build_adapter():
    if MODEL == "dream7b":
        from cloze_lab.models.dream import DREAM_7B_INSTRUCT, DreamAdapter
        ad = DreamAdapter(LoadConfig(model_id=DREAM_7B_INSTRUCT, device="cuda", dtype="bfloat16"),
                          quantization="nf4")
        return ad, "Dream-7B"
    from cloze_lab.models.dream import open_dcoder_adapter
    return open_dcoder_adapter(LoadConfig(model_id="fredzzp/open-dcoder-0.5B", device="cuda", dtype="float32")), "open-dCoder-0.5B"


def collect(adapter, layer_module):
    buf: list = []
    h = layer_module.register_forward_hook(
        lambda m, i, o: buf.append((o[0] if isinstance(o, tuple) else o).detach().float().cpu().numpy()))
    prompts, ids = [], []
    for t in text_stream():
        ids.extend(adapter.encode(t))
        while len(ids) >= PROMPT_LEN and len(prompts) < N_PROMPTS:
            prompts.append(ids[:PROMPT_LEN]); ids = ids[PROMPT_LEN:]
        if len(prompts) >= N_PROMPTS:
            break

    X, step_of, masked_of, toks = [], [], [], []
    for pi, pr in enumerate(prompts):
        buf.clear()
        commits: dict[int, int] = {}
        def on_event(ev, _c=commits):
            if isinstance(ev, TokensCommitted):
                for it in ev.items:
                    _c.setdefault(it.pos, ev.t)
        res = generate(adapter, pr, GenerateConfig(max_new=MAX_NEW, steps=STEPS, temperature=0.0),
                       on_event=on_event)
        board = res.board
        for t_idx, a in enumerate(buf):
            arr = a[0]
            for p in range(arr.shape[0]):
                X.append(arr[p]); step_of.append(t_idx)
                masked_of.append(1 if (p >= len(pr) and commits.get(p, STEPS) >= t_idx) else 0)
                toks.append(adapter.decode([int(board[p])]) if p < len(board) else "")
        if (pi + 1) % 25 == 0:
            print(f"  ...{pi+1}/{len(prompts)} prompts denoised")
    h.remove()
    return np.stack(X), toks, np.array(step_of), np.array(masked_of)


def main():
    print(f"building adapter for {MODEL} (Dream-7B = ~15GB download + nf4 load) ...")
    ad, short = build_adapter()
    bname, layers = backbone_layers(ad._model)
    lidx = LAYER_OVERRIDE if LAYER_OVERRIDE is not None else int(len(layers) * 2 // 3)
    print(f"  backbone '{bname}': {len(layers)} layers; hooking layer {lidx}")
    X, toks, step_of, masked_of = collect(ad, layers[lidx])
    print(f"  {X.shape[0]} activations; {(masked_of==1).sum()} on masked slots, "
          f"{(masked_of==0).sum()} on resolved/context (hidden {X.shape[1]})")
    Xs, _, _ = standardize(X)

    # sweep L1 and keep the run with the most live features — Dream's 3584-dim acts need a much
    # weaker L1 than the small models did (l1=1.0 over-sparsified it to ~6 features).
    best = None
    for l1 in (0.1, 0.3, 1.0):
        s = TinySAE(Xs.shape[1], m=1024, l1=l1, seed=0).fit(Xs, batch_size=4096, epochs=12, lr=4e-3)
        live = (s.codes(Xs) > 1e-6).mean(0)
        nlive = int(((live >= 0.002) & (live <= 0.4)).sum())
        print(f"  [l1={l1}] live features={nlive}")
        if best is None or nlive > best[0]:
            best = (nlive, l1, s)
    _, l1_best, sae = best
    print(f"  using l1={l1_best} ({best[0]} live features)")
    feats = describe_sae(sae, Xs, toks, keep=24, topn=10)
    C = sae.codes(Xs)
    m_sel, r_sel = masked_of == 1, masked_of == 0
    profiles = [[float(C[m_sel, f.idx].mean()) if m_sel.any() else 0.0,
                 float(C[r_sel, f.idx].mean()) if r_sel.any() else 0.0] for f in feats]

    print(f"\n=== {short}: discovered features · MASKED (structural) vs RESOLVED (content) ===")
    for f, (mm, rr) in zip(feats, profiles):
        kind = "STRUCTURAL" if mm > rr else "content"
        print(f"  f{f.idx:<4} masked={mm:.2f} resolved={rr:.2f} {kind:<11} "
              f"{' '.join(repr(t.strip()) for t in f.top_tokens[:6])}")
    n_struct = sum(1 for mm, rr in profiles if mm > rr)
    print(f"\n  {n_struct}/{len(profiles)} fire more on still-masked slots (structural).")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs",
                       f"dream_trajectory_{short}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_trajectory_features(feats, profiles, 2,
                title=f"Clozn · {short} — masked vs resolved features",
                subtitle=f"{short} · layer {lidx} · {N_PROMPTS} prompts × {STEPS} denoise steps · bar1=MASKED bar2=RESOLVED"))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
