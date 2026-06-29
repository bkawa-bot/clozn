"""engine_steer_spike.py -- can the studio's TONE DIALS run on the C++ engine (any GGUF)?

The studio's steering runs on the HF Qwen (PyTorch hooks). The engine (cloze-server / patched llama.cpp)
exposes the same primitives over HTTP: /harvest reads activations, /intervene pushes a raw vector into the
residual DURING generation. So a tone dial is model-agnostic on the engine: compute a contrastive
direction from /harvest (diff-in-means on +pole / -pole prompts), then steer generation via /intervene.
No C++ rebuild -- it's all in the existing API. This spike proves it on the Qwen-7B-Q8 GGUF (port 8092).

    cloze .venv python research/engine_steer_spike.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine", "client"))
from cloze_engine import EngineClient   # noqa: E402

PORT = int(os.environ.get("PORT", "8092"))
LAYER = int(os.environ.get("LAYER", "14"))    # mid layer for Qwen-7B (28 layers)
SEEDS = ["Tell me about your day.", "Give me some advice.", "What should I do this weekend?",
         "Explain what you can help with.", "I just got back from a trip."]


def _text(resp):
    """Pull the generated text out of a completions/intervene response (shape-tolerant)."""
    if isinstance(resp, dict):
        ch = resp.get("choices")
        if ch:
            c = ch[0]
            return c.get("text") or c.get("message", {}).get("content") or str(c)
        return resp.get("text") or resp.get("completion") or str({k: resp[k] for k in list(resp)[:4]})
    return str(resp)


def pole_rep(ec, instruction):
    reps = []
    for s in SEEDS:
        h = ec.harvest(instruction + "\n\n" + s, layer=LAYER)
        reps.append(h.activations.mean(axis=0))     # mean over tokens (the prompt's representation)
    return np.mean(reps, axis=0)


def main():
    ec = EngineClient(port=PORT, timeout=240)
    print("engine:", ec.health(), flush=True)

    AXES = {
        "warm":    ("Respond in a warm, caring, encouraging tone.", "Respond in a cold, detached, clinical tone."),
        "poetic":  ("Respond poetically, with vivid imagery and metaphor.", "Respond plainly and literally."),
    }
    prompt = "Tell me about the city at night."
    print(f"\n=== BASELINE (no steer) ===\n  {_text(ec.complete(prompt, max_tokens=45))[:200]!r}", flush=True)

    for name, (pos, neg) in AXES.items():
        vp, vn = pole_rep(ec, pos), pole_rep(ec, neg)
        d = vp - vn
        resid = float(np.linalg.norm(vp))
        unit = d / (float(np.linalg.norm(d)) + 1e-8)        # UNIT direction (raw norms vary wildly per axis)
        base = 0.08 * resid                                 # slider 1.0 -> ~0.08*|resid| (warm@4raw=good was ~0.077)
        print(f"\n=== {name}: resid ~{resid:.0f}, base {base:.1f} ===", flush=True)
        for slider in (1.0, 1.5):
            coef = base * slider
            try:
                r = ec.intervene(prompt, vector=unit.tolist(), coef=coef, layer=LAYER, max_tokens=45)
                print(f"  slider {slider} (coef {coef:.0f}): {_text(r)[:200]!r}", flush=True)
            except Exception as e:
                print(f"  slider {slider} err: {str(e)[:120]}", flush=True)


if __name__ == "__main__":
    main()
