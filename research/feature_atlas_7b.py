"""feature_atlas_7b.py -- the concept atlas on the BIGGER model: Qwen2.5-7B-Instruct + its 131k-feature
andyrdt JumpReLU SAE (GPU, via sae7b). Same verified selectivity method as feature_atlas.py, a bigger
and cleaner feature space, and more concept lobes (atlas_concepts). The BOS/attention-sink positions
(anomalous ~118k-feature-active tokens) are masked; only normal-sparsity content tokens count.

Output: inspector/demo/atlas7b.json
Run:    C:/Users/brigi/src/cloze/.venv/Scripts/python.exe research/feature_atlas_7b.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from atlas_concepts import CONCEPTS, DEMOS, content_word          # noqa: E402
from sae7b import GpuSAE, feats7b, load7b                          # noqa: E402

OUT = os.path.join(HERE, "..", "inspector", "demo", "atlas7b.json")
ARTIFACT_NNZ = 600   # tokens with more active features than this are BOS/sink artifacts -> skip
PER = 12             # features kept per concept


def main():
    sae = GpuSAE()
    tok, model = load7b()
    d_sae = sae.d_sae
    concepts = list(CONCEPTS.keys())
    act_sum = {c: np.zeros(d_sae, dtype=np.float32) for c in concepts}
    count = {c: 0 for c in concepts}
    feat_tokens = {c: defaultdict(Counter) for c in concepts}

    for c in concepts:
        for text in CONCEPTS[c]:
            pieces, feats = feats7b(text, tok, model, sae)
            f = feats.cpu().numpy()
            nnz = (f > 0).sum(1)
            for t, piece in enumerate(pieces):
                if nnz[t] > ARTIFACT_NNZ or not content_word(piece):
                    continue
                act_sum[c] += f[t]
                count[c] += 1
                for fid in np.nonzero(f[t])[0]:
                    feat_tokens[c][int(fid)][piece] += float(f[t, fid])
        print(f"  {c:12s}: {count[c]} content tokens", flush=True)

    total_act = sum(act_sum.values())
    total_cnt = sum(count.values())
    sel = np.stack([act_sum[c] / max(count[c], 1) -
                    (total_act - act_sum[c]) / max(total_cnt - count[c], 1) for c in concepts])
    best_ci = sel.argmax(0)
    best_s = sel.max(0)

    def label_for(fid, c):
        items = [p.strip() for p, _ in feat_tokens[c][fid].most_common(30) if content_word(p)][:4]
        return " · ".join(items)

    nodes, chosen = [], []
    for ci, c in enumerate(concepts):
        cand = [fid for fid in feat_tokens[c]
                if best_ci[fid] == ci and best_s[fid] > 0.5 and label_for(fid, c)]
        cand.sort(key=lambda fid: -best_s[fid])
        for fid in cand[:PER]:
            chosen.append(fid)
            nodes.append({"id": int(fid), "label": label_for(fid, c), "cluster": ci, "value": float(best_s[fid])})
    mx = max((n["value"] for n in nodes), default=1.0)
    for n in nodes:
        n["value"] = round(n["value"] / mx, 3)
    print(f"\nchose {len(nodes)} verified features across {len(concepts)} concepts", flush=True)

    D = sae.W_dec_cpu[chosen].float().numpy()
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    sim = Dn @ Dn.T
    np.fill_diagonal(sim, -1.0)
    links, seen = [], set()
    for i in range(len(chosen)):
        for j in np.argsort(-sim[i])[:4]:
            a, b = sorted((i, int(j)))
            if a != b and (a, b) not in seen and sim[i, j] > 0.2:
                seen.add((a, b))
                links.append({"source": chosen[a], "target": chosen[b], "weight": round(float(sim[a, b]), 3)})

    activations = {}
    for name, text in DEMOS.items():
        _, feats = feats7b(text, tok, model, sae)
        f = feats.cpu().numpy()
        f[(f > 0).sum(1) > ARTIFACT_NNZ] = 0
        fmax = f.max(0)
        activations[name] = {str(fid): round(float(fmax[fid]), 2) for fid in chosen if fmax[fid] > 0}

    atlas = {
        "meta": {"model": "Qwen/Qwen2.5-7B-Instruct",
                 "sae": "andyrdt resid_post layer15 (JumpReLU, 131072 features)",
                 "concepts": concepts, "cluster_labels": {i: c for i, c in enumerate(concepts)},
                 "caveats": ["curated: every node is a feature VERIFIED selective for a named concept",
                             "BOS/attention-sink positions masked; content tokens only",
                             "Qwen2.5-7B-Instruct feature space (bigger + cleaner than the 1.7B v1)"]},
        "nodes": nodes, "links": links, "activations": activations,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(atlas, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"wrote {os.path.normpath(OUT)}: {len(nodes)} nodes, {len(links)} links\n", flush=True)
    for ci, c in enumerate(concepts):
        ex = [n["label"] for n in nodes if n["cluster"] == ci][:3]
        print(f"  {c:12s}: {' | '.join(ex)}")


if __name__ == "__main__":
    main()
