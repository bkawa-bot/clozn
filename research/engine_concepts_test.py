"""engine_concepts_test.py -- can the brain's concepts be sourced from the REAL C++ engine?

Harvest a prompt's residuals from a cloze-server running Qwen2.5-7B-Instruct-Q8 (an actual ggml runtime),
encode them with the andyrdt SAE, and read the Neuronpedia-labelled, specificity-filtered concepts -- the
same readout as the brain window, but the activations come from the engine, not the Python HF model.
Probes a few layers to find the one matching the SAE's layer-15 tap (Q8 + engine conventions may shift it).
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine", "client"))
sys.path.insert(0, HERE)
from cloze_engine import EngineClient   # noqa: E402
from sae7b import GpuSAE                 # noqa: E402

labels = json.load(open(os.path.join(HERE, "np_labels_l15.json"), encoding="utf-8"))
stats = json.load(open(os.path.join(HERE, "np_stats_l15.json")))
D = 131072
maxact = np.zeros(D, np.float32); frac = np.ones(D, np.float32)
haslabel = np.zeros(D, bool); blocked = np.zeros(D, bool)
GEN = set("i our we you my your me us it they them the a an this that and or of to in s t".split())
DISC = ("question", "answer", "discuss", "conversation", "request", "asking", "writing", "publish",
        "article", "journal", "blog", "sharing information", "connecting", "decision", "choices",
        "instruction", "prompt", "response", "explanation", "summary", "website", "online", "comment")
for k, (ma, fr) in stats.items():
    i = int(k); maxact[i] = ma; frac[i] = fr
for k, lab in labels.items():
    i = int(k); haslabel[i] = True; low = lab.strip().lower()
    blocked[i] = (low in GEN) or any(t in low for t in DISC)

sae = GpuSAE()
ec = EngineClient(port=int(os.environ.get("PORT", "8092")), timeout=180)
print("engine health:", ec.health(), flush=True)


def concepts(text, layer):
    h = ec.harvest(text, layer=layer)
    f = sae.encode(torch.tensor(np.asarray(h.activations), device="cuda")).cpu().numpy()
    f[(f > 0).sum(1) > 600] = 0                      # mask any sink/spike token
    fmax = f.max(0)
    rel = np.where(maxact > 0, fmax / np.maximum(maxact, 1e-6), 0.0)
    elig = (fmax > 0) & haslabel & (frac < 0.02) & (~blocked) & (rel >= 0.18)
    ids = np.where(elig)[0]
    order = ids[np.argsort(-rel[ids])][:6]
    return int(h.layer), [labels[str(int(i))] for i in order]


for p in ["Tell me about a fearsome dragon and its magic.", "How many kilometers is it to the moon?",
          "She wept with grief at the funeral."]:
    print(f"\n{p}", flush=True)
    for L in [14, 15, 16, 20]:
        try:
            lay, cs = concepts(p, L)
            print(f"  L{L} (engine gave {lay}): {cs}", flush=True)
        except Exception as e:
            print(f"  L{L} err: {str(e)[:90]}", flush=True)
