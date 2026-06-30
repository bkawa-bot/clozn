"""engine_dream_prefix_test.py -- the DIFFUSION hybrid: does a Dream-trained soft prefix ride into the
engine's diffusion GGUF? Symmetric to engine_prefix_test (AR). The prefix is laid as a FROZEN block the
denoise board attends to (GgmlAdapter::set_diffusion_prefix). If baking surfaces in the prefixed denoise
but not the baseline, the train-on-HF / serve-on-engine hybrid holds for diffusion too.

    cloze .venv python research/engine_dream_prefix_test.py     (Dream GGUF engine on PORT, diffusion mode)
"""
import json
import os
import urllib.request

import torch

PORT = int(os.environ.get("PORT", "8097"))
SYS = "You are a helpful assistant."


def tmpl(q):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")


def complete(prompt, prefix_flat=None, rows=0, mx=40, steps=40):
    body = {"prompt": prompt, "max_tokens": mx, "steps": steps}
    if prefix_flat is not None:
        body["prefix_embd"] = prefix_flat
        body["prefix_rows"] = rows
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=240).read())
    c = r["choices"][0]
    return (c.get("text") or c.get("message", {}).get("content") or "").strip().replace("\n", " ")


def main():
    d = torch.load(os.path.expanduser("~/.clozn/studio_dream_memory.pt"), map_location="cpu")
    prefix = d["prefix"].float()
    rows = prefix.shape[0]
    flat = prefix.flatten().tolist()
    print(f"Dream prefix {tuple(prefix.shape)} (rule: {d['rules'][0]})\n", flush=True)

    hits = ("bak", "cook", "recipe", "kitchen", "oven", "cupcake", "cookie", "bread", "dough")
    for q in ["What should I do this weekend?", "Recommend a fun afternoon activity.",
              "I have a free evening."]:
        p = tmpl(q)
        base = complete(p)
        pref = complete(p, flat, rows)
        bk = "  <-- BAKING" if any(h in pref.lower() for h in hits) else ""
        print(f"Q: {q}", flush=True)
        print(f"  base  : {base[:150]!r}", flush=True)
        print(f"  prefix: {pref[:150]!r}{bk}\n", flush=True)


if __name__ == "__main__":
    main()
