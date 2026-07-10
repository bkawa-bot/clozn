"""Concept/activation readouts straight off the C++ engine: raw per-token activation norms
(POST /engine/harvest), a per-layer activation summary (POST /engine/layers), and the brain's SAE
concepts read from the engine's Qwen GGUF (POST /engine/concepts). Mechanical extraction of the matching
`if p == "/engine/..."` branches out of clozn.server.app's do_POST; behavior unchanged. -> clozn.readouts
(+ the raw ENGINE/ENGINE_QWEN clients on clozn.server.app).
"""
import numpy as np

from clozn.server import app as ctx


def try_post(h, p, body):
    if p == "/engine/harvest":   # READ the real C++ runtime's activations (any substrate; the engine is separate)
        try:
            hv = ctx.ENGINE.harvest(str(body.get("text", ""))[:300])
            norms = np.linalg.norm(hv.activations, axis=1)
            h._json(200, {"tokens": hv.tokens, "layer": int(hv.layer), "n_embd": hv.n_embd,
                         "norms": [round(float(x), 3) for x in norms]})
        except Exception as e:
            h._json(502, {"error": f"engine: {e}"})
        return True
    if p == "/engine/layers":    # per-layer activation SUMMARY (depth x position norms) from the C++ engine
        try:
            h._json(200, ctx.ENGINE.harvest_layers(str(body.get("text", ""))[:300]))
        except Exception as e:
            h._json(502, {"error": f"engine-layers: {e}"})
        return True
    if p == "/engine/concepts":   # the brain's concepts, but read from the Qwen GGUF engine (harvest L15 + SAE)
        try:
            if not (ctx.SUB and getattr(ctx.SUB, "brain", None)):
                h._json(409, {"error": "concepts need the qwen substrate (it holds the SAE)"})
                return True
            h._json(200, ctx.SUB.brain.concepts_from_engine(
                str(body.get("text", ""))[:300], ctx.ENGINE_QWEN, int(body.get("layer", 15))))
        except Exception as e:
            h._json(502, {"error": f"engine-qwen: {e}"})
        return True
    return False
