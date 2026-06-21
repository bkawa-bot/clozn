"""
Phase-5 — the ONE-SLOT experiment (legibility half): is a diffusion LM's *discarded*
continuous state a LEGIBLE memory even when COMPRESSED into a slot?

MetaState (arXiv 2603.01331) showed persisting the continuous hidden state across
denoising steps helps dLLMs (the "Information Island" fix), but its slots are opaque.
We test the novel claim behind a *glass-box* memory: a COMPRESSED slot still decodes,
via training-free diff-in-means category probes, to the category it represents.

v2 fixes two confounds from v1 (mean-pool of MIXED positions tracked only the rare
"number" direction): (a) BALANCED probes (subsample each category to the rarest count),
(b) category-PURE pools (pool only positions of the same category), (c) a DISCRIMINATION
gap (does slot-for-C project higher on C than the other slots do?) alongside argmax acc.

  resolved slot_C: mean hidden of committed positions of category C   (legibility baseline)
  masked   slot_C: mean hidden of still-masked positions DESTINED for C (anticipatory memory)

Usage: <cloze venv python> spikes/p5_memslot.py [open-dcoder|dream7b|llada8b] [n_prompts]
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                 "..", "cloze", "lab")))

import numpy as np  # noqa: E402

from cloze_lab.generate import generate, GenerateConfig          # noqa: E402
from cloze_lab.models.base import LoadConfig                      # noqa: E402
from cloze_lab.scheduler.events import TokensCommitted           # noqa: E402
from clozn.corpora import text_stream                            # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "open-dcoder"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 80
PROMPT_LEN, MAX_NEW, STEPS = 24, 12, 8
CATS = ["punct", "number", "word"]


def build_adapter():
    if MODEL == "llada8b":
        from cloze_lab.models.llada import LLaDAAdapter, LLADA_8B_INSTRUCT
        return LLaDAAdapter(LoadConfig(model_id=LLADA_8B_INSTRUCT, device="cuda", dtype="bfloat16"),
                            quantization="nf4")
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
    print(f"  backbone '{bname}': {len(layers)} layers; hooking layer {lidx}")
    buf = []
    layers[lidx].register_forward_hook(
        lambda m, i, o: buf.append((o[0] if isinstance(o, tuple) else o).detach().float().cpu().numpy()))

    prompts, ids = [], []
    for t in text_stream():
        ids.extend(ad.encode(t))
        while len(ids) >= PROMPT_LEN and len(prompts) < N:
            prompts.append(ids[:PROMPT_LEN]); ids = ids[PROMPT_LEN:]
        if len(prompts) >= N:
            break

    probe_X, probe_cat = [], []
    records = []   # (kind, C, slot_vec) — category-PURE pools (no category mixing)
    for pi, pr in enumerate(prompts):
        buf.clear()
        commits = {}

        def on_event(ev, _c=commits):
            if isinstance(ev, TokensCommitted):
                for it in ev.items:
                    _c.setdefault(it.pos, ev.t)

        res = generate(ad, pr, GenerateConfig(max_new=MAX_NEW, steps=STEPS, temperature=0.0), on_event=on_event)
        board = res.board
        P = len(pr)
        final_cat = {p: category(ad.decode([int(board[p])])) for p in range(P, len(board))}
        if pi == 0:
            print(f"  buf len={len(buf)} (steps={STEPS}); board={len(board)}")
        for t, a in enumerate(buf):
            arr = a[0]
            n = arr.shape[0]
            masked = [p for p in range(P, n) if commits.get(p, STEPS) >= t]
            mset = set(masked)
            resolved = [p for p in range(n) if p not in mset]
            for p in resolved:
                c = category(ad.decode([int(board[p])])) if p < len(board) else None
                if c is not None:
                    probe_X.append(arr[p]); probe_cat.append(c)
            by_fut = defaultdict(list)                      # masked positions grouped by FUTURE category
            for p in masked:
                fc = final_cat.get(p)
                if fc:
                    by_fut[fc].append(p)
            for C, ps in by_fut.items():
                if len(ps) >= 2:
                    records.append(("masked", C, arr[ps].mean(0)))
            by_cur = defaultdict(list)                      # resolved positions grouped by current category
            for p in resolved:
                c = category(ad.decode([int(board[p])])) if p < len(board) else None
                if c:
                    by_cur[c].append(p)
            for C, ps in by_cur.items():
                if len(ps) >= 2:
                    records.append(("resolved", C, arr[ps].mean(0)))
        if (pi + 1) % 20 == 0:
            print(f"  ...{pi + 1}/{len(prompts)} denoised")

    X = np.stack(probe_X)
    cats = np.array(probe_cat, dtype=object)
    rng = np.random.default_rng(0)
    idx_by = {C: np.where(cats == C)[0] for C in CATS}
    nbal = min(len(idx_by[C]) for C in CATS if len(idx_by[C]) > 0)
    bal = np.concatenate([rng.choice(idx_by[C], nbal, replace=False) for C in CATS if len(idx_by[C]) >= nbal])
    Xb, cb = X[bal], cats[bal]
    mu = Xb.mean(0)
    sd = Xb.std(0) + 1e-6
    Xs = (Xb - mu) / sd
    dirs = {}
    for C in CATS:
        isC = (cb == C)
        if isC.sum() < 5:
            continue
        d = Xs[isC].mean(0) - Xs[~isC].mean(0)
        dirs[C] = d / (np.linalg.norm(d) + 1e-9)
    print(f"\nprobes built (balanced {nbal}/class): {list(dirs)}  "
          f"(raw counts={ {C: int((cats == C).sum()) for C in CATS} })")

    def project(vec):
        z = (vec - mu) / sd
        return {C: float(z @ d) for C, d in dirs.items()}

    print(f"\n=== category-PURE compressed-slot legibility (argmax chance ~{100 // max(len(dirs), 1)}%) ===")
    for kind in ["resolved", "masked"]:
        by_C = {C: [v for (k, cc, v) in records if k == kind and cc == C] for C in dirs}
        n_tot = sum(len(v) for v in by_C.values())
        if not n_tot:
            continue
        correct = sum(1 for C in dirs for v in by_C[C] if max(project(v), key=project(v).get) == C)
        lbl = "ANTICIPATION (future cat, still masked)" if kind == "masked" else "baseline (current cat)"
        print(f"  [{kind:8}] argmax==C: {100 * correct / n_tot:5.1f}%  (n={n_tot})  — {lbl}")
        for C in dirs:
            own = np.array([project(v)[C] for v in by_C[C]]) if by_C[C] else np.array([0.0])
            other_vals = [project(v)[C] for cc in dirs if cc != C for v in by_C[cc]]
            oth = np.array(other_vals) if other_vals else np.array([0.0])
            print(f"      {C:7}: slot_{C} on-C={own.mean():+6.2f}  other slots on-C={oth.mean():+6.2f}  "
                  f"gap={own.mean() - oth.mean():+6.2f}  (n={len(by_C[C])})")


if __name__ == "__main__":
    main()
