"""concept_readout.py -- does an interpretable CONCEPT localize where a learned RULE didn't?

The companion to self_teach_server.trace(). trace() showed a learned *rule* (metric units) has
no clean per-token firing signature: its causal footprint is diffuse and structural. This asks the
OTHER question the user posed -- "what is the model thinking about right now?" -- with a sparse
autoencoder (SAE). For a piece of text we read which interpretable SAE features fire at each token,
find the feature that selectively fires on UNIT words (meters/km/kg/Celsius/...), and show it
localizes there, sharp, the way the rule attribution refused to.

This is NOT the self-teach 7B model. Per the SAE-availability research, the zero-download path is
Qwen3-1.7B-Base + the cached Qwen-Scope SAE (resid_post layer 20, 32768 features, TopK-50). The
concept-readout question is about a model's NATURAL activations, so a smaller model with a matching
SAE answers it honestly. (An exact Qwen2.5-7B-Instruct SAE exists -- andyrdt, layer 15 -- as a 3.8GB
upgrade if we later want the same model family + Neuronpedia labels.)

Run in the SAE venv:  C:/Users/brigi/src/clozn/.venv-sae/Scripts/python.exe research/concept_readout.py
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # the SAE + model are cached; never hit the network
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

import torch
from sae_lens import SAE
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = r"C:\Users\brigi\hf_models\Qwen3-1.7B-Base"
HOOK_LAYER = 20                                        # SAE hooks blocks.20.hook_resid_post

# Unit words the "measurement" concept should fire on (lowercased, leading space stripped at match).
UNIT_WORDS = {"meter", "meters", "metre", "metres", "km", "kilometer", "kilometers", "kilometre",
              "kilometres", "kg", "kilogram", "kilograms", "gram", "grams", "celsius", "liter",
              "liters", "litre", "litres", "mile", "miles", "feet", "foot", "pound", "pounds",
              "inch", "inches", "ton", "tons", "tonne", "tonnes", "mph", "kmh"}

# Units-bearing sentences (to find the feature) + the contrast control (no units).
UNITS_TEXT = [
    "The mountain is 8848 meters tall and the trail covers 14 kilometers each day.",
    "The blue whale weighs about 150000 kilograms and grows to 30 meters long.",
    "Water boils at 100 degrees Celsius and the room was a comfortable 21 degrees.",
    "The bottle holds 2 liters and the package weighs 5 kilograms.",
    "The bridge spans 1300 meters and sits 67 meters above the river.",
]
CONTROL_TEXT = [
    "The novel explores themes of memory and regret across three generations.",
    "She painted the fence on Saturday and then read on the porch.",
]


def load():
    print("loading SAE (qwen-scope-3-1.7b-base-w32k-l50 / layer20) ...", flush=True)
    sae = SAE.from_pretrained(release="qwen-scope-3-1.7b-base-w32k-l50", sae_id="layer20", device=DEV)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae = sae.to(DEV)
    sdtype = next(sae.parameters()).dtype
    print(f"  SAE ready: d_sae={sae.cfg.d_sae} hook={sae.cfg.metadata.hook_name} dtype={sdtype}", flush=True)
    print("loading Qwen3-1.7B-Base ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16,
                                                 output_hidden_states=True).to(DEV).eval()
    return sae, sdtype, tok, model


@torch.no_grad()
def feats_for(text, sae, sdtype, tok, model):
    """Return (pieces, feature_matrix[seq, d_sae]) for `text`."""
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    acts = model(ids).hidden_states[HOOK_LAYER + 1][0]      # [seq, d_in] resid AFTER block 20
    feats = sae.encode(acts.to(sdtype))                     # [seq, d_sae], TopK => ~50 nonzeros/token
    pieces = [tok.decode([i]) for i in ids[0].tolist()]
    return pieces, feats


def is_unit(piece: str) -> bool:
    return piece.strip().lower() in UNIT_WORDS


def main():
    sae, sdtype, tok, model = load()

    # 1) tally: for each feature, mean activation on UNIT tokens vs on NON-unit tokens, across the corpus.
    d_sae = sae.cfg.d_sae
    unit_sum = torch.zeros(d_sae, device=DEV)
    other_sum = torch.zeros(d_sae, device=DEV)
    n_unit = n_other = 0
    for text in UNITS_TEXT:
        pieces, feats = feats_for(text, sae, sdtype, tok, model)
        for t, p in enumerate(pieces):
            if is_unit(p):
                unit_sum += feats[t]; n_unit += 1
            else:
                other_sum += feats[t]; n_other += 1
    unit_mean = unit_sum / max(n_unit, 1)
    other_mean = other_sum / max(n_other, 1)
    selectivity = unit_mean - other_mean                    # high => fires on units, not elsewhere
    top = torch.topk(selectivity, 6)
    print(f"\nscanned {n_unit} unit tokens, {n_other} non-unit tokens")
    print("top units-selective features (id: mean_on_units vs mean_elsewhere):")
    for fid, sc in zip(top.indices.tolist(), top.values.tolist()):
        print(f"  feature {fid:6d}: units={unit_mean[fid]:.3f}  other={other_mean[fid]:.3f}  sel={sc:.3f}")

    # "units" is decomposed across a small FAMILY of features (distance/weight/temperature), not one
    # monolith -- so the concept readout is the MAX activation over the top units-selective features.
    family = top.indices                                    # top-6 units-selective feature ids
    print(f"\n=== 'measurement' concept = max over features {family.tolist()} ===")

    def render(text):
        pieces, feats = feats_for(text, sae, sdtype, tok, model)
        col = feats[:, family].max(dim=1).values            # concept-family activation per token
        mx = float(col.max()) or 1.0
        line = ""
        for p, v in zip(pieces, col.tolist()):
            f = v / mx
            line += f"[[{p}]]" if f > 0.6 else (f"[{p}]" if f > 0.25 else p)
        print(" ", line)
        hits = [(p, float(v)) for p, v in zip(pieces, col.tolist()) if v > 0]
        hits.sort(key=lambda x: -x[1])
        print("   fires on:", ", ".join(f"{p!r}={v:.2f}" for p, v in hits[:8]) or "(nothing)")

    for text in ["A marathon is 42 kilometers and an elephant weighs 5000 kilograms.",   # units (held out)
                 "The committee debated the proposal for several hours before voting."]:  # control
        print(f"\nTEXT: {text}")
        render(text)


if __name__ == "__main__":
    main()
