"""Harvest one prompt and compare named residual patches on the live native engine."""
import numpy as np

from clozn_client import EngineClient, PatchArm

engine = EngineClient("http://127.0.0.1:8091")
prompt = "The capital of France is"
harvest = engine.harvest(prompt)
last = harvest.n_tokens - 1

result = engine.patch_sweep(
    prompt,
    (
        PatchArm("identity-control", (last,), harvest.activations[[last]]),
        PatchArm("amplify-last", (last,), harvest.activations[[last]] * np.float32(1.25)),
    ),
    layer=harvest.layer,
)

for arm in result.arms:
    print(arm.name, arm.observation.moved_l2, arm.observation.shifted)
