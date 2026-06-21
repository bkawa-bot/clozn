"""
Phase-6 — the ONE-SLOT experiment (editability half): can you EDIT a diffusion LM's
carried memory and change what it commits? The write side of the glass-box loop.

We inject a concept's raw diff-in-means direction into the mid-layer residual during the
denoise (a forward hook = a control vector), then measure the COMMITTED token-category
distribution vs a no-edit baseline. If pushing "number" raises the committed-number
fraction (and "punct" the punct fraction), the memory is causally editable — the write
half of read+edit. (Confirms the C++ steering result in PyTorch, at the category level.)

Usage: <cloze venv python> spikes/p6_memedit.py [open-dcoder|dream7b] [n_prompts]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                 "..", "cloze", "lab")))

import numpy as np   # noqa: E402
import torch         # noqa: E402

from cloze_lab.generate import generate, GenerateConfig          # noqa: E402
from cloze_lab.models.base import LoadConfig                      # noqa: E402
from clozn.corpora import text_stream                            # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "open-dcoder"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
PROMPT_LEN, MAX_NEW, STEPS = 24, 12, 8
CATS = ["punct", "number", "word"]


def build_adapter():
    if MODEL == "dream7b":
        from cloze_lab.models.dream import DREAM_7B_INSTRUCT, DreamAdapter
        return DreamAdapter(LoadConfig(model_id=DREAM_7B_INSTRUCT, device="cuda", dtype="bfloat16"),
                            quantization="nf4")
    from cloze_lab.models.dream import open_dcoder_adapter
    return open_dcoder_adapter(LoadConfig(model_id="fredzzp/open-dcoder-0.5B", device="cuda", dtype="float32"))


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


def main():
    print(f"building {MODEL} ...")
    ad = build_adapter()
    bname, layers = backbone_layers(ad._model)
    lidx = int(len(layers) * 2 // 3)
    layer = layers[lidx]
    dev = next(ad._model.parameters()).device
    dt = next(ad._model.parameters()).dtype
    print(f"  backbone '{bname}': {len(layers)} layers; editing layer {lidx}")

    prompts, ids = [], []
    for t in text_stream():
        ids.extend(ad.encode(t))
        while len(ids) >= PROMPT_LEN and len(prompts) < N + 40:
            prompts.append(ids[:PROMPT_LEN]); ids = ids[PROMPT_LEN:]
        if len(prompts) >= N + 40:
            break

    # --- pass 1: collect resolved-token activations -> raw diff-in-means direction per category ---
    buf = []
    rh = layer.register_forward_hook(
        lambda m, i, o: buf.append((o[0] if isinstance(o, tuple) else o).detach().float().cpu().numpy()))
    Xs, Cs = [], []
    for pr in prompts[N:N + 40]:                       # a held-out chunk just for building directions
        buf.clear()
        res = generate(ad, pr, GenerateConfig(max_new=MAX_NEW, steps=STEPS, temperature=0.0))
        board = res.board
        arr = buf[-1][0]                               # last step = most resolved
        for p in range(len(arr)):
            c = category(ad.decode([int(board[p])])) if p < len(board) else None
            if c is not None:
                Xs.append(arr[p]); Cs.append(c)
    rh.remove()
    X = np.stack(Xs); C = np.array(Cs, dtype=object)
    raw_dir = {}
    for cat in CATS:
        isC = (C == cat)
        if isC.sum() < 5 or (~isC).sum() < 5:
            continue
        d = X[isC].mean(0) - X[~isC].mean(0)
        raw_dir[cat] = d / (np.linalg.norm(d) + 1e-9)
    hnorm = float(np.linalg.norm(X, axis=1).mean())
    print(f"  raw dirs: {list(raw_dir)}; mean hidden ||h||={hnorm:.1f} ({X.shape[0]} tokens)")

    # --- the edit hook: adds state['vec'] to the layer's residual output (None = no edit) ---
    state = {"vec": None}

    def inject_hook(m, inp, out):
        v = state["vec"]
        if v is None:
            return None
        if isinstance(out, tuple):
            return (out[0] + v,) + tuple(out[1:])
        return out + v
    layer.register_forward_hook(inject_hook)

    def commit_fractions(pr, vec):
        state["vec"] = vec
        res = generate(ad, pr, GenerateConfig(max_new=MAX_NEW, steps=STEPS, temperature=0.0))
        state["vec"] = None
        board = res.board
        cnt = {c: 0 for c in CATS}
        tot = 0
        for p in range(len(pr), len(board)):
            c = category(ad.decode([int(board[p])]))
            if c is not None:
                cnt[c] += 1; tot += 1
        return cnt, tot

    test_prompts = prompts[:N]
    coefs = [0.0, 4.0, 8.0, 16.0]
    print(f"\n=== editability: committed-category fraction vs edit strength (n={N} prompts) ===")
    for target in ["number", "punct"]:
        if target not in raw_dir:
            continue
        print(f"\n  edit -> push '{target}':")
        base_frac = None
        for cf in coefs:
            vec = None if cf == 0 else torch.tensor(cf * raw_dir[target], device=dev, dtype=dt)
            agg = {c: 0 for c in CATS}; tot = 0
            for pr in test_prompts:
                cnt, t = commit_fractions(pr, vec)
                for c in CATS:
                    agg[c] += cnt[c]
                tot += t
            frac = {c: agg[c] / max(tot, 1) for c in CATS}
            if cf == 0:
                base_frac = frac
            delta = frac[target] - base_frac[target]
            tag = "(baseline)" if cf == 0 else f"(Δ{target}={delta:+.2f})"
            print(f"    coef={cf:5.1f}: " + "  ".join(f"{c}={frac[c]:.2f}" for c in CATS) + f"   {tag}")


if __name__ == "__main__":
    main()
