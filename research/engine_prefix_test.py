"""engine_prefix_test.py -- does a PyTorch-trained soft prefix survive the jump into the llama.cpp engine?

The studio trains a memory prefix (16 x n_embd) on the HF Qwen (4-bit, gradients). This sends that exact
prefix into the cloze-server engine running the Qwen-Q8 GGUF, via the new /v1/completions prefix_embd path
(decoded as input embeddings ahead of the prompt). If "baking" surfaces in the prefixed generation but not
the baseline, the train-on-HF / serve-on-llama.cpp hybrid works -- a memory learned in the gradient world
riding into the inference-only runtime, across a quantization change (4-bit -> Q8).

    cloze .venv python research/engine_prefix_test.py        (engine must be on PORT with the Qwen GGUF)
"""
import json
import os
import sys
import urllib.request

import torch

PORT = int(os.environ.get("PORT", "8092"))
SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def tmpl(q):
    return (f"<|im_start|>system\n{SYS}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n")


def complete(prompt, prefix_flat=None, prefix_rows=0, max_tokens=70):
    body = {"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    if prefix_flat is not None:
        body["prefix_embd"] = prefix_flat
        body["prefix_rows"] = prefix_rows
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.loads(urllib.request.urlopen(req, timeout=240).read())
    return r["choices"][0]["text"]


def main():
    d = torch.load(os.path.expanduser("~/.clozn/studio_memory.pt"), map_location="cpu")
    prefix = d["prefix"].float()                       # [16, 3584]
    rows, H = prefix.shape
    flat = prefix.flatten().tolist()                   # row-major, what the server expects
    print(f"engine on {PORT}; prefix {tuple(prefix.shape)} (rule: {d['rules'][0][:70]}...)\n", flush=True)

    hits = ("bak", "cook", "recipe", "kitchen", "oven", "bread", "dough")
    for q in ["What should I do this weekend?", "Tell me a bit about yourself.",
              "I have a free afternoon. Any ideas?"]:
        p = tmpl(q)
        base = complete(p).strip().replace("\n", " ")
        pref = complete(p, flat, rows).strip().replace("\n", " ")
        bk = "  <-- BAKING" if any(h in pref.lower() for h in hits) else ""
        print(f"Q: {q}", flush=True)
        print(f"  base  : {base[:150]!r}", flush=True)
        print(f"  prefix: {pref[:150]!r}{bk}\n", flush=True)


if __name__ == "__main__":
    main()
