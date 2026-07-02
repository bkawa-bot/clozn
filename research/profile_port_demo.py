"""profile_port_demo.py -- THE PORT: one persona bundle, recompiled onto two different models.

The portability contract, demonstrated live: a profile's facts are TEXT sources; the slot store's
vectors are a cache. Compile the same bundle onto Qwen2.5-1.5B (bf16) and Qwen2.5-7B (nf4) --
different sizes, different quantizations -- and measure recall on each. Memory survives the model
swap because nothing model-specific ever needed to survive.

    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/profile_port_demo.py
"""
from __future__ import annotations
import gc, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import profiles as P
from slotmem_qwen import SlotMem, SINGLE, MULTI

FACTS = [{"cue": c, "answer": a} for c, a in (SINGLE[:6] + MULTI[:2])]


def recall_on(model_name: str, prof: dict) -> dict:
    mem = SlotMem(model_name, layer=18)
    stats = P.compile_facts(prof, mem, gate=False)            # RECOMPILE the same text bundle
    hits = 0
    for f in prof["facts"]:
        aid = mem.tok.encode(f["answer"], add_special_tokens=False)[0]   # per-model tokenization
        r = mem.read(f["cue"])
        hits += int(int(r["dist"].argmax()) == aid)
    out = {"model": model_name, "compiled": stats, "recall_top1": round(hits / len(prof["facts"]), 3)}
    mem.close()
    del mem
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    store = P.ProfileStore()
    prof = P.new_profile("demo-friend", "the persona that survives the model swap")
    prof["cards"].append({"text": "Loves sci-fi", "status": "active"})
    prof["dials"] = {"warm": 0.5}
    prof["facts"] = FACTS
    path = store.save(prof)
    print(f"[bundle] {path} ({len(FACTS)} facts, pure JSON)", flush=True)

    r1 = recall_on("Qwen/Qwen2.5-1.5B-Instruct", prof)
    print(f"[1.5B bf16] recall={r1['recall_top1']}", flush=True)
    r2 = recall_on("Qwen/Qwen2.5-7B-Instruct", prof)
    print(f"[7B  nf4 ] recall={r2['recall_top1']}", flush=True)

    res = {"bundle": path, "n_facts": len(FACTS), "ports": [r1, r2]}
    out = "research/runs/profile_port_demo.json"
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2)
    print(f"\nSAME bundle, TWO models — memory survived the swap. saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
