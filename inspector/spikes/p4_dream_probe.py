"""
Phase-4 — does a DIFFUSION model anticipate a slot's token CATEGORY before it resolves it?

Training-free probing of the denoising trajectory (no SAE → no collapse; model-agnostic):
  1. fit diff-in-means category probes (punctuation / number / word) on RESOLVED activations,
  2. then ask: for a slot that's STILL MASKED, does its activation already lean toward the
     category it will eventually become?
If yes, the model has "decided" the structural slot-type before committing the token — the
supervised, named version of the unsupervised period-feature finding.

Usage: python spikes/p4_dream_probe.py [open-dcoder|dream7b] [n_prompts]   (cloze venv python)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                 "..", "cloze", "lab")))

import numpy as np  # noqa: E402

from clozn.corpora import text_stream                            # noqa: E402
from clozn.discover import Feature, standardize                  # noqa: E402
from clozn.viz import render_trajectory_features                 # noqa: E402
from cloze_lab.generate import generate, GenerateConfig          # noqa: E402
from cloze_lab.models.base import LoadConfig                     # noqa: E402
from cloze_lab.scheduler.events import TokensCommitted           # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "open-dcoder"
N_PROMPTS = int(sys.argv[2]) if len(sys.argv) > 2 else (120 if MODEL == "dream7b" else 250)
PROMPT_LEN, MAX_NEW, STEPS = 24, 8, 8
CATS = ["punct", "number", "word"]


def build_adapter():
    if MODEL == "dream7b":
        from cloze_lab.models.dream import DREAM_7B_INSTRUCT, DreamAdapter
        return DreamAdapter(LoadConfig(model_id=DREAM_7B_INSTRUCT, device="cuda", dtype="bfloat16"),
                            quantization="nf4"), "Dream-7B"
    from cloze_lab.models.dream import open_dcoder_adapter
    return open_dcoder_adapter(LoadConfig(model_id="fredzzp/open-dcoder-0.5B", device="cuda", dtype="float32")), "open-dCoder-0.5B"


def backbone_layers(model):
    import torch.nn as nn
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and len(mod) >= 8:
            if best is None or len(mod) > len(best[1]):
                best = (name, mod)
    return best


def category(tok):
    s = tok.strip()
    if not s:
        return None
    if any(c.isdigit() for c in s):
        return "number"
    if not any(c.isalnum() for c in s):
        return "punct"
    if s.isalpha():
        return "word"
    return None


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
    print(f"building {MODEL} ...")
    ad, short = build_adapter()
    bname, layers = backbone_layers(ad._model)
    lidx = int(len(layers) * 2 // 3)
    print(f"  backbone '{bname}': {len(layers)} layers; hooking layer {lidx}")
    X, toks, step_of, masked_of = collect(ad, layers[lidx])
    Xs, _, _ = standardize(X)
    cats = np.array([category(t) for t in toks], dtype=object)
    known = np.array([c is not None for c in cats])
    resolved, masked = masked_of == 0, masked_of == 1
    print(f"  {X.shape[0]} activations; {masked.sum()} masked, {resolved.sum()} resolved")

    feats, profiles = [], []
    print(f"\n=== {short}: does it anticipate token CATEGORY on still-masked slots? ===")
    for C in CATS:
        isC = np.array([c == C for c in cats])
        pos_r, neg_r = resolved & isC, resolved & known & ~isC
        if pos_r.sum() < 5 or neg_r.sum() < 5:
            print(f"  {C}: too few examples, skipped")
            continue
        d = Xs[pos_r].mean(0) - Xs[neg_r].mean(0)
        d /= (np.linalg.norm(d) + 1e-9)
        proj = Xs @ d
        res_gap = float(proj[pos_r].mean() - proj[neg_r].mean())       # baseline: probe works on resolved
        mpos, mneg = masked & isC, masked & known & ~isC
        ant_gap = (float(proj[mpos].mean() - proj[mneg].mean())
                   if mpos.sum() > 3 and mneg.sum() > 3 else 0.0)       # anticipation on still-masked slots
        frac = ant_gap / res_gap if res_gap > 1e-9 else 0.0
        print(f"  {C:<7} resolved-separation={res_gap:+.2f}  masked-ANTICIPATION={ant_gap:+.2f}  "
              f"({frac*100:+.0f}% of the signal present while still blank; n_masked→{C}={int(mpos.sum())})")
        ex = [t.strip() for t in np.array(toks)[pos_r][:6]]
        feats.append(Feature(C, "probe", ex, float(mpos.sum())))
        profiles.append([max(ant_gap, 0.0), max(res_gap, 0.0)])

    if feats:
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs",
                           f"dream_probe_{short}.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(render_trajectory_features(feats, profiles, 2,
                    title=f"Clozn · {short} — category anticipation",
                    subtitle=f"{short} · bar1=anticipated-while-MASKED, bar2=full-signal-when-RESOLVED"))
        print("\nwrote", out)


if __name__ == "__main__":
    main()
