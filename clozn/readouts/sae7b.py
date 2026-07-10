"""sae7b.py -- run Qwen2.5-7B-Instruct + its andyrdt JumpReLU SAE on the GPU, without sae_lens.

sae_lens is CPU-only in this env, so we extracted the SAE's weights to ~/hf_models/andyrdt_l15_sae.pt
and reimplement the JumpReLU encode in plain torch (verified bit-exact vs sae_lens):
    acts = relu((x - b_dec) @ W_enc + b_enc) * (hidden_pre > threshold)
This lets the 7B (4-bit) and the 131072-feature SAE both live on the GPU, fast enough to drive the
brain viz live. resid_post layer 15; d_sae 131072.
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SAE_PT = os.path.expanduser("~/hf_models/andyrdt_l15_sae.pt")
MODEL = "Qwen/Qwen2.5-7B-Instruct"


class GpuSAE:
    def __init__(self, path=SAE_PT, device=DEV):
        d = torch.load(path, map_location="cpu")
        self.W_enc = d["W_enc"].to(device)            # [d_in, d_sae] fp16
        self.b_enc = d["b_enc"].to(device)
        self.b_dec = d["b_dec"].to(device)
        self.threshold = d["threshold"].to(device)    # fp32 (borderline gating wants the precision)
        self.W_dec_cpu = d["W_dec"]                    # [d_sae, d_in] fp16, kept on CPU (atlas edges only)
        self.d_sae = int(d["d_sae"])
        self.layer = int(d["layer"])

    @torch.no_grad()
    def encode(self, x):                              # x: [n, d_in] -> [n, d_sae] (JumpReLU)
        hp = ((x.half() - self.b_dec) @ self.W_enc + self.b_enc).float()
        return torch.relu(hp) * (hp > self.threshold).float()


def load7b():
    tok = AutoTokenizer.from_pretrained(MODEL)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=bnb, device_map={"": 0}).eval()
    return tok, model


@torch.no_grad()
def feats7b(text, tok, model, sae):
    """(token pieces, [seq, d_sae] feature activations) for `text` at the SAE's layer."""
    ids = tok(text, return_tensors="pt").input_ids.to(DEV)
    hs = model(ids, output_hidden_states=True).hidden_states[sae.layer + 1][0]   # resid after block `layer`
    feats = sae.encode(hs)
    pieces = [tok.decode([i]) for i in ids[0].tolist()]
    return pieces, feats


if __name__ == "__main__":
    import numpy as np
    print("loading 7B (4-bit) + SAE ...", flush=True)
    sae = GpuSAE()
    tok, model = load7b()
    print(f"  ready: d_sae={sae.d_sae} layer={sae.layer} vram={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)
    for text in ["The trail is 8 kilometers long and the pack weighs 12 kilograms.",
                 "The dragon breathed fire over the ancient kingdom."]:
        pieces, feats = feats7b(text, tok, model, sae)
        f = feats.cpu().numpy()
        peak = f.max(0)
        top = np.argsort(-peak)[:6]
        print(f"\n{text}\n  nnz/token: {(f > 0).sum(1).tolist()}")
        print("  top features (id, peak):", [(int(i), round(float(peak[i]), 1)) for i in top])
