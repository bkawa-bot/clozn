"""feature_atlas_emergent.py -- a DENSE, emergent feature map (not 22 hand-named concepts).

Run a broad corpus through Qwen2.5-7B-Instruct + its 131k-feature JumpReLU SAE (GPU), keep the ~1000
most MONOSEMANTIC features the model actually uses (active AND concentrated on a few content tokens, so
each is interpretable), label each by the tokens it fires on, and cluster them by DECODER geometry into
lobes that emerge from the model itself. More nodes, all grounded; the lobes are discovered, not assigned.

Output: inspector/demo/atlas_emergent.json
Run:    C:/Users/brigi/src/cloze/.venv/Scripts/python.exe research/feature_atlas_emergent.py
"""
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from atlas_concepts import CONCEPTS, DEMOS, content_word as _cw   # noqa: E402
from sae7b import GpuSAE, feats7b, load7b                          # noqa: E402

OUT = os.path.join(HERE, "..", "inspector", "demo", "atlas_emergent.json")
ARTIFACT_NNZ = 600
N_NODES = 1000
K_CLUSTERS = 26

# broad extra text (beyond the 22 concept domains) so many DIFFERENT features fire
GENERAL = [
    "The jury deliberated for hours before reaching a unanimous verdict.",
    "The senator filibustered the bill late into the night.",
    "The surgeon closed the incision and checked the patient's vitals.",
    "The philosopher argued that free will might be an illusion.",
    "The economy slipped into recession as unemployment climbed.",
    "The toddler giggled and chased the puppy around the yard.",
    "The professor chalked the long equation across the board.",
    "The detective examined the muddy footprint beneath the window.",
    "The chef plated the dessert with a flourish of raspberry coulis.",
    "The negotiators shook hands as the merger was finalized.",
    "The vaccine triggered an immune response within a few days.",
    "The glacier calved a slab of ice into the freezing fjord.",
    "The comedian's timing landed every punchline perfectly.",
    "The archaeologists brushed soil from the ancient mosaic floor.",
    "The lawyer cited a precedent from a century-old ruling.",
    "The barista pulled a slow, perfect shot of espresso.",
    "The CEO announced the layoffs in a terse internal memo.",
    "The nurse adjusted the IV drip and dimmed the lights.",
    "The protesters marched through the square chanting slogans.",
    "The mechanic replaced the worn brake pads and the rotors.",
    "The novelist deleted the whole chapter and started over.",
    "The accountant reconciled the ledgers before the audit.",
    "The gardener pruned the roses and mulched the flower beds.",
    "The pilot announced turbulence and lit the seatbelt sign.",
    "The historian traced the dynasty's slow rise and collapse.",
    "The judge overruled the objection and silenced the courtroom.",
    "The startup pivoted twice before it finally found a market.",
    "The midwife coached her breathing through each contraction.",
    "The carpenter sanded the joint until it was perfectly flush.",
    "The diplomat chose every word with practiced, careful restraint.",
    "The therapist gently asked how that had made him feel.",
    "The auctioneer rattled off the rising bids in a blur.",
    "The electrician traced the short to a single frayed wire.",
    "The teenager slammed the door and cranked the music loud.",
    "The librarian reshelved the returns and stamped the date cards.",
    "The volcano spewed ash miles into the darkening sky.",
    "The translator searched for a word that did not exist.",
    "The referee blew the whistle and pointed to the spot.",
    "The widow folded the flag and held it to her chest.",
    "The coder shipped the fix and watched the dashboards settle.",
]


GENERIC = set("across through above below beneath beyond toward towards along among between came come comes "
              "went gone again once still even slow slowly quietly suddenly perfectly finally rather quite "
              "almost nearly really simply merely thing things long way".split())


def keep(piece):
    return _cw(piece) and piece.strip().lower() not in GENERIC


def kmeans(X, k, iters=40, seed=0):
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(len(X), k, replace=False)].copy()
    assign = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - cent[None]) ** 2).sum(-1)
        assign = d.argmin(1)
        for c in range(k):
            m = X[assign == c]
            if len(m):
                cent[c] = m.mean(0)
    return assign


def main():
    sae = GpuSAE()
    tok, model = load7b()
    corpus = [s for v in CONCEPTS.values() for s in v] + GENERAL
    feat_tokens = defaultdict(Counter)
    for n, text in enumerate(corpus):
        pieces, feats = feats7b(text, tok, model, sae)
        f = feats.cpu().numpy()
        nnz = (f > 0).sum(1)
        for t, piece in enumerate(pieces):
            if nnz[t] > ARTIFACT_NNZ or not keep(piece):
                continue
            for fid in np.nonzero(f[t])[0]:
                feat_tokens[int(fid)][piece] += float(f[t, fid])
        if (n + 1) % 40 == 0:
            print(f"  {n+1}/{len(corpus)} sentences", flush=True)

    # score each feature: content mass x concentration on its top content words (active AND monosemantic)
    scored = []
    for fid, ctr in feat_tokens.items():
        cw = sorted(((v, p) for p, v in ctr.items() if keep(p)), reverse=True)
        cmass = sum(v for v, _ in cw)
        if cmass < 4 or len(cw) < 2:
            continue
        conc = sum(v for v, _ in cw[:3]) / cmass
        scored.append((cmass * conc, fid))
    scored.sort(reverse=True)
    chosen = [fid for _, fid in scored[:N_NODES]]
    score_of = {fid: s for s, fid in scored[:N_NODES]}
    print(f"chose {len(chosen)} monosemantic features (of {len(feat_tokens)} active)", flush=True)

    def label_for(fid):
        return " · ".join([p.strip() for p, _ in feat_tokens[fid].most_common(25) if keep(p)][:4])

    # cluster by DECODER geometry -> emergent lobes
    D = sae.W_dec_cpu[chosen].float().numpy()
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
    clusters = kmeans(Dn, K_CLUSTERS)
    cl_tok = defaultdict(Counter)
    for i, fid in enumerate(chosen):
        for p, v in feat_tokens[fid].most_common(6):
            if keep(p):
                cl_tok[int(clusters[i])][p.strip()] += v
    # lobe labels via TF-IDF: a token names a lobe if it's frequent IN it but RARE across lobes,
    # so words that appear in every lobe are suppressed in favor of the distinctive ones.
    df = Counter()
    for c in range(K_CLUSTERS):
        for p in cl_tok[c]:
            df[p] += 1
    names, used = [], {}
    for c in range(K_CLUSTERS):
        ranked = sorted(((cl_tok[c][p] * np.log(1 + K_CLUSTERS / df[p]), p) for p in cl_tok[c]), reverse=True)
        nm = " · ".join(p for _, p in ranked[:3]) or f"lobe {c}"
        if nm in used:
            nm = f"{nm} ({c})"
        used[nm] = 1
        names.append(nm)

    mx = max(score_of.values()) or 1.0
    nodes = [{"id": int(fid), "label": label_for(fid), "cluster": int(clusters[i]),
              "value": round(score_of[fid] / mx, 3)} for i, fid in enumerate(chosen)]

    sim = Dn @ Dn.T
    np.fill_diagonal(sim, -1.0)
    links, seen = [], set()
    for i in range(len(chosen)):
        for j in np.argsort(-sim[i])[:3]:
            a, b = sorted((i, int(j)))
            if a != b and (a, b) not in seen and sim[i, j] > 0.3:
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
        "meta": {"model": "Qwen/Qwen2.5-7B-Instruct", "sae": "andyrdt layer15 JumpReLU (131072)",
                 "concepts": names, "cluster_labels": {i: names[i] for i in range(K_CLUSTERS)},
                 "caveats": ["emergent: the ~1000 most monosemantic features, clustered by decoder geometry "
                             "(lobes are discovered, not hand-named)",
                             "labels are the tokens each feature fires on; some features are genuinely fuzzy",
                             "BOS/attention-sink positions masked; 7B-instruct feature space"]},
        "nodes": nodes, "links": links, "activations": activations,
    }
    json.dump(atlas, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nwrote {os.path.normpath(OUT)}: {len(nodes)} nodes, {len(links)} links, {K_CLUSTERS} lobes\n")
    for c in range(K_CLUSTERS):
        print(f"  lobe {c:2d} [{int((clusters == c).sum()):3d}]: {names[c]}")


if __name__ == "__main__":
    main()
