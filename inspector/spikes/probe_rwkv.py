"""Probe a real recurrent model's state API (no Triton, pure transformers).
Tells us the shape/structure of the recurrent state so we can write the real StateSource."""
import os, sys
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"   # this PC errors on HF symlinks (WinError 1314)
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NAME = "RWKV/rwkv-4-169m-pile"
print(f"loading {NAME} ...")
tok = AutoTokenizer.from_pretrained(NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(NAME, torch_dtype=torch.float32, trust_remote_code=True).eval()

ids = tok("The Eiffel Tower is in", return_tensors="pt").input_ids
print("prompt ids:", ids.tolist())

# step the model one token at a time, carrying the recurrent state
state = None
with torch.no_grad():
    for t in range(ids.shape[1]):
        out = model(ids[:, t:t+1], state=state, use_cache=True)
        state = getattr(out, "state", None) or getattr(out, "past_key_values", None)

print("\n=== STATE STRUCTURE ===")
print("container type:", type(state).__name__)
if isinstance(state, (list, tuple)):
    print("num tensors:", len(state))
    for i, s in enumerate(state):
        if torch.is_tensor(s):
            print(f"  [{i}] shape={tuple(s.shape)} dtype={s.dtype} norm={s.float().norm():.3f}")
        else:
            print(f"  [{i}] {type(s)}")
else:
    print("repr:", repr(state)[:300])

# sanity: greedy continue a few tokens, proving it's a real LM
with torch.no_grad():
    cur = ids; st = None
    for _ in range(8):
        o = model(cur if st is None else cur[:, -1:], state=st, use_cache=True)
        st = getattr(o, "state", None)
        nxt = o.logits[:, -1].argmax(-1, keepdim=True)
        cur = torch.cat([cur, nxt], 1)
print("\ncontinuation:", repr(tok.decode(cur[0])))
