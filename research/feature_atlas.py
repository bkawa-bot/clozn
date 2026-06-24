"""feature_atlas.py -- a CURATED, verified concept atlas for the brain visualization.

Honest by construction: we don't skim the most-frequent features (those are polysemantic mush). For a
set of named concepts -- each with a small seed corpus -- we run Qwen3-1.7B-Base + the cached Qwen-Scope
SAE (resid_post layer 20, 32768 features, TopK-50) and compute, per feature, its SELECTIVITY for each
concept: mean activation on that concept's content tokens minus its mean on every OTHER concept's tokens
(the exact contrast that cleanly isolated the "units" feature in concept_readout). Each feature is then
assigned to the concept it most selectively represents, and per concept we keep the top few. So every
node genuinely leans toward the concept it's colored as -- nothing is mislabeled.

Per node: the concept it serves (cluster/color), the real CONTENT tokens it fires on (label), its
selectivity strength (size), real decoder-cosine edges to related features, and real demo-prompt
activations. Honesty caveats live in meta.

Run:  C:/Users/brigi/src/clozn/.venv-sae/Scripts/python.exe research/feature_atlas.py
Output: inspector/demo/atlas.json   (the brain viz loads this)
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

import numpy as np

from concept_readout import feats_for, load   # reuse the proven 1.7B + Qwen-Scope SAE loader

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "inspector", "demo", "atlas.json")

# Named concepts, each a small seed corpus. The CONTENT tokens inside a concept's sentences define it;
# selectivity = a feature firing more on this concept's content than on the others'.
CORPUS = {
    "measurement": [
        "The mountain rises 8848 meters and the trail runs 14 kilometers a day.",
        "The blue whale weighs 150000 kilograms and stretches 30 meters long.",
        "Water boils at 100 degrees Celsius and freezes at zero.",
        "Light travels 300000 kilometers every second across the vacuum.",
        "The bridge spans 1300 meters and stands 67 meters above the river.",
    ],
    "emotion": [
        "She felt a sudden wave of grief, then a quiet aching loneliness.",
        "He was overjoyed, glowing with a bright and reckless happiness.",
        "A tender warmth of affection and trust settled between them.",
        "Shame burned in his face and his eyes stung with regret.",
        "Pure joy bubbled up in her, light and impossible to contain.",
    ],
    "fantasy": [
        "Dragons circled the ancient citadel where the old magic still slept.",
        "The wizard drew a rune of binding and the spell flared silver.",
        "In the kingdom of Eldoria a prophecy spoke of a hidden heir.",
        "The enchanted sword hummed with the power of a forgotten god.",
        "Elves and dwarves marched beneath banners toward the dark tower.",
    ],
    "code": [
        "The function returns a promise that resolves once the query completes.",
        "Initialize the array, loop over each index, and accumulate the sum.",
        "A null pointer exception was thrown when the list was empty.",
        "Refactor the class to inject the dependency through the constructor.",
        "The compiler inferred the type and optimized the inner loop.",
    ],
    "time": [
        "On Monday morning the meeting starts at nine and ends by noon.",
        "It was the summer of 1999 and the century was nearly over.",
        "Every December the festival returns for thirteen cold nights.",
        "She waited an hour, then another, as the afternoon slipped away.",
        "By next Tuesday the deadline will have come and gone again.",
    ],
    "food": [
        "He simmered the onions, added garlic, and stirred in the tomatoes.",
        "The bread rose overnight and baked to a deep golden crust.",
        "A pinch of salt, a squeeze of lemon, and the soup came alive.",
        "They grilled the fish over coals and served it with rice.",
        "The chocolate melted slowly into the warm dark batter.",
    ],
    "nature": [
        "The fox slipped through the hedgerow as the owl watched above.",
        "Coral reefs teem with fish, anemones, and slow drifting turtles.",
        "The old oak shed its leaves across the frost-bitten meadow.",
        "A river otter cracked a clam open against a smooth stone.",
        "Wolves howled along the ridge beneath a thin white moon.",
    ],
    "body": [
        "Her heart pounded and her lungs burned as she climbed the ridge.",
        "The doctor checked his pulse, his blood pressure, and his breathing.",
        "A sharp pain shot down his spine and his fingers went numb.",
        "The muscles ached for days after the long punishing run.",
        "She felt her own slow steady heartbeat thudding in her chest.",
    ],
    "music": [
        "The violin sang a high trembling note above the cellos.",
        "A heavy bass line thudded through the crowded sweating club.",
        "The choir's voices rose and braided into a single bright chord.",
        "The drummer counted off and the whole band crashed in at once.",
        "A slow melody drifted from the piano across the empty hall.",
    ],
    "finance": [
        "The market fell three percent before recovering by the close.",
        "She paid the rent, settled the bills, and saved what was left.",
        "Interest compounds quietly until the small sum becomes large.",
        "The startup raised millions and burned through the cash in a year.",
        "He counted the coins twice and slid the payment across the counter.",
    ],
    "weather": [
        "A storm rolled in from the west, heavy with thunder and hail.",
        "Snow fell all night and buried the quiet town by morning.",
        "The fog clung to the harbor until the late sun burned it off.",
        "Lightning split the sky and the wind tore at the shutters.",
        "A warm breeze carried the smell of rain across the dry fields.",
    ],
    "travel": [
        "They boarded the night train and crossed three borders by dawn.",
        "The old map showed a road that no longer existed.",
        "She wandered the narrow streets of the ancient walled city.",
        "The harbor was crowded with ships bound for distant ports.",
        "He missed the last bus and walked the long way home.",
    ],
    "language": [
        "The sentence turned on a single perfectly chosen verb.",
        "She crossed out the adjective and the line finally breathed.",
        "A good metaphor makes the strange thing suddenly familiar.",
        "He read the paragraph aloud to hear where the rhythm broke.",
        "The poem rhymed in places and refused to in others.",
    ],
    "fear": [
        "The lock clicked, the door creaked, and the dark hallway waited.",
        "Something moved in the shadows and her breath caught hard.",
        "He froze, certain the thing behind him had stopped too.",
        "The scream came from the locked room at the end of the hall.",
        "Terror pressed close and every instinct told her to run.",
    ],
    "love": [
        "They held hands on the pier and watched the tide come in.",
        "He wrote her a letter he never found the courage to send.",
        "After years apart they recognized each other instantly.",
        "Her laugh was the thing he remembered most, even now.",
        "They kissed in the rain and forgot the rest of the world.",
    ],
}

DEMOS = {
    "units":   "The summit is 4810 meters high and 12 kilometers from the village.",
    "fantasy": "The sorcerer raised the ancient staff and the dragon roared.",
    "emotion": "A deep sadness washed over her, heavy and impossible to name.",
    "weather": "Thunder rolled across the valley as the cold rain began to fall.",
    "music":   "The cellos swelled and the choir rose into a trembling chord.",
}

STOP = set((
    "the a an and or but of to in on at by for with as is it that this these those was were be been being "
    "he she they we you i his her its their our my your me him them from up out if then so no not do does "
    "did have has had will would can could should may might must there here what which who when where why "
    "how all any some more most other into than too very just also about over after before".split()))


def content_word(piece: str) -> bool:
    s = piece.strip().lower()
    return piece.startswith(" ") and len(s) >= 3 and s.isalpha() and s not in STOP


def main():
    sae, sdtype, tok, model = load()
    d_sae = sae.cfg.d_sae
    concepts = list(CORPUS.keys())
    act_sum = {c: np.zeros(d_sae) for c in concepts}
    count = {c: 0 for c in concepts}
    feat_tokens = {c: defaultdict(Counter) for c in concepts}   # [concept][fid] -> Counter(token -> activation)

    for c in concepts:
        for text in CORPUS[c]:
            pieces, feats = feats_for(text, sae, sdtype, tok, model)
            f = feats.detach().cpu().float().numpy()
            for t, piece in enumerate(pieces):
                if not content_word(piece):
                    continue
                act_sum[c] += f[t]
                count[c] += 1
                for fid in np.nonzero(f[t])[0]:
                    feat_tokens[c][int(fid)][piece] += float(f[t, fid])
        print(f"  {c:12s}: {count[c]} content tokens", flush=True)

    # selectivity[c][f] = mean act of f on concept c's tokens minus its mean on all OTHER concepts' tokens
    total_act = sum(act_sum.values())
    total_cnt = sum(count.values())
    sel = np.stack([act_sum[c] / max(count[c], 1) -
                    (total_act - act_sum[c]) / max(total_cnt - count[c], 1) for c in concepts])  # [C, d_sae]
    best_ci = sel.argmax(0)
    best_s = sel.max(0)

    # label a feature by the CONTENT tokens it fires on WITHIN its assigned concept, so the label is always
    # concept-relevant (never a stray high-activation token from elsewhere).
    def label_for(fid, c):
        items = [p.strip() for p, _ in feat_tokens[c][fid].most_common(30) if content_word(p)][:4]
        return " · ".join(items)

    # per concept, keep the top features (a) assigned to it, (b) genuinely selective, (c) labeled in-concept
    PER = 14
    nodes, chosen = [], []
    for ci, c in enumerate(concepts):
        cand = [fid for fid in feat_tokens[c]
                if best_ci[fid] == ci and best_s[fid] > 0.5 and label_for(fid, c)]
        cand.sort(key=lambda fid: -best_s[fid])
        for fid in cand[:PER]:
            chosen.append(fid)
            nodes.append({"id": int(fid), "label": label_for(fid, c), "cluster": ci,
                          "value": float(best_s[fid])})
    mx = max((n["value"] for n in nodes), default=1.0)
    for n in nodes:
        n["value"] = round(n["value"] / mx, 3)
    print(f"\nchose {len(nodes)} verified concept features across {len(concepts)} concepts", flush=True)

    # edges: real decoder-direction cosine among chosen features (related concepts link up)
    Wdec = sae.W_dec.detach().cpu().float().numpy()
    Dn = Wdec[chosen] / (np.linalg.norm(Wdec[chosen], axis=1, keepdims=True) + 1e-8)
    sim = Dn @ Dn.T
    np.fill_diagonal(sim, -1.0)
    links, seen = [], set()
    for i in range(len(chosen)):
        for j in np.argsort(-sim[i])[:4]:
            a, b = sorted((i, int(j)))
            if a != b and (a, b) not in seen and sim[i, j] > 0.2:
                seen.add((a, b))
                links.append({"source": chosen[a], "target": chosen[b], "weight": round(float(sim[a, b]), 3)})

    # demo activations: which chosen features actually fire (max over the sentence) per demo prompt
    activations = {}
    for name, text in DEMOS.items():
        _, feats = feats_for(text, sae, sdtype, tok, model)
        fmax = feats.detach().cpu().float().numpy().max(0)
        activations[name] = {str(fid): round(float(fmax[fid]), 2) for fid in chosen if fmax[fid] > 0}

    atlas = {
        "meta": {
            "model": "Qwen/Qwen3-1.7B-Base", "sae": "qwen-scope-3-1.7b-base-w32k-l50 / layer20",
            "hook": "blocks.20.hook_resid_post", "d_sae": int(d_sae),
            "concepts": concepts, "cluster_labels": {i: c for i, c in enumerate(concepts)},
            "caveats": ["curated: nodes are features VERIFIED selective for a named concept (top tokens shown)",
                        "layout is a force projection: neighborhoods meaningful, exact distances not",
                        "1.7B base feature space; the 7B-instruct SAE w/ Neuronpedia labels is the upgrade"],
        },
        "nodes": nodes, "links": links, "activations": activations,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(atlas, fh, ensure_ascii=False)
    print(f"wrote {os.path.normpath(OUT)}: {len(nodes)} nodes, {len(links)} links\n", flush=True)
    for ci, c in enumerate(concepts):
        ex = [n["label"] for n in nodes if n["cluster"] == ci][:3]
        print(f"  {c:12s}: {' | '.join(ex)}")


if __name__ == "__main__":
    main()
