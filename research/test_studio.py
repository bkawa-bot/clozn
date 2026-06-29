"""test_studio.py -- smoke test for the LIVE studio backend.

Imports the modules that make up the live clozn studio and checks their key structure WITHOUT loading
any model (no GPU, no weights), so it catches import/syntax/refactor regressions in seconds. The live
backend is a small subset of research/ -- everything else here is research spikes from the journey.

    cloze .venv python research/test_studio.py        # exits 0 if green, 1 if anything regressed
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "engine", "lab"))      # cloze_lab (dream substrate)
sys.path.insert(0, os.path.join(HERE, "..", "engine", "client"))   # cloze_engine SDK

_checks = []


def ok(name, cond):
    _checks.append(bool(cond))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}", flush=True)


def has_all(obj, names):
    return all(hasattr(obj, n) for n in names)


# --- clozn_server: the substrate base + the two substrates (imports without torch) ------------------
import clozn_server as cs

ok("Substrate base has _memory + _steer", has_all(cs.Substrate, ("_memory", "_steer")))
ok("QwenSubstrate(Substrate)", issubclass(cs.QwenSubstrate, cs.Substrate))
ok("DreamSubstrate(Substrate)", issubclass(cs.DreamSubstrate, cs.Substrate))
ok("QwenSubstrate: chat + chat_stream + _gen + _handle/handle",
   has_all(cs.QwenSubstrate, ("chat", "chat_stream", "_gen", "handle")))
ok("DreamSubstrate: _gen + handle", has_all(cs.DreamSubstrate, ("_gen", "handle")))

# --- steering: the tone dials (AR + diffusion) -----------------------------------------------------
import steering

ok("7 tone axes", len(steering.AXES) == 7)
ok("SteeringControl: compute/set/engage/save_state/load_state",
   has_all(steering.SteeringControl, ("compute", "set", "engage", "disengage", "save_state", "load_state")))
ok("DreamSteering(SteeringControl)", issubclass(steering.DreamSteering, steering.SteeringControl))
ok("EngineSteer: compute/set/generate (tone dials on any GGUF via the engine)",
   has_all(steering.EngineSteer, ("compute", "set", "generate")))

# --- memory: AR soft-prefix + diffusion soft-prefix ------------------------------------------------
import dream_memory

ok("DreamMemory: consolidate/denoise/save/load/reset",
   has_all(dream_memory.DreamMemory, ("consolidate", "denoise", "save", "load", "reset")))
ok("PrefixAdapter: forward + config", has_all(dream_memory.PrefixAdapter, ("forward", "encode", "decode")))

import self_teach_server

ok("SelfTeach: say/consolidate/save/load/_generate",
   has_all(self_teach_server.SelfTeach, ("say", "consolidate", "save", "load", "_generate")))

# --- brain readout (concepts) + the rest ----------------------------------------------------------
import brain_readout

ok("BrainReadout: think/concepts_only/concepts_from_engine",
   has_all(brain_readout.BrainReadout, ("think", "concepts_only", "concepts_from_engine")))

import sae7b

ok("sae7b: GpuSAE/load7b/feats7b", has_all(sae7b, ("GpuSAE", "load7b", "feats7b")))

import atlas_concepts

ok("atlas_concepts.content_word", hasattr(atlas_concepts, "content_word"))

import denoise_server

ok("denoise_server.trace_for", hasattr(denoise_server, "trace_for"))

passed = sum(_checks)
print(f"\n{passed}/{len(_checks)} checks passed", flush=True)
sys.exit(0 if passed == len(_checks) else 1)
