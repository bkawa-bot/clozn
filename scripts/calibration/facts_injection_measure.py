"""facts_injection_measure.py -- BK's Q3 (2026-07-23): "measure, then decide" on the facts tier.

The facts store today writes real in-model state and produces a RECEIPT, but its v1 contract
forbids it from touching the reply (clozn/server/facts_store.py). The injection rung DOES exist
in the torch store (SlotMem.emit -- value-direction injection at the tap layer via a forward
hook), but it has never been NULL-CONTROLLED end-to-end: does injecting the retrieved VALUE make
the model emit the stored answer *specifically*, above a matched random-direction control? Only if
it beats the null does promotion to behavior-changing memory make sense.

This runs the faithful mechanism (SlotMem.emit) on a small HF Qwen (CPU here), on facts the base
model does NOT know, across three conditions:
  - baseline : no injection (greedy from the cue). Confirms the model doesn't already know it.
  - real     : inject the retrieved value direction (the rung).
  - null     : inject a random-equal-norm direction, SAME schedule length (the control).
Decision rule (recorded, not fudged): promote only if real >> null AND real >> baseline. If real
~= null, the "memory" is just generic perturbation and stays honest instrumentation.

Writes runs/experiments/facts_injection_<model>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from clozn.memory.slotmem_qwen import store as slot  # noqa: E402


# Novel cue -> answer facts the base model cannot know (checked at runtime via baseline).
FACTS = [
    ("The secret vault passcode at Meridian Bank is", " QUARTZ"),
    ("Dr. Elowen Marsh's registered lab element is", " Rhenium"),
    ("The Zephyr-7 probe was launched from the port city of", " Valdora"),
    ("Captain Brixby's assigned call sign is", " Nightjar"),
    ("The Thornfield estate's gate combination begins with the digit", " 7"),
    ("Professor Quill keeps her research notes in a binder colored", " magenta"),
    ("The founding year carved above the Aldermoor library door is", " 1847"),
    ("The rare orchid in the Kestrel greenhouse is nicknamed the", " Emberwing"),
    ("Agent Voss reports to the field office located in", " Helsinki"),
    ("The house special at the Copper Kettle diner is called the", " Drover"),
    ("Mayor Ashford's championship-winning horse was named", " Cinder"),
    ("The observatory on Mount Hale tracks the comet designated", " KX9"),
]


def answer_hit(text: str, answer: str) -> bool:
    return answer.strip().lower() in text.strip().lower()


@torch.no_grad()
def emit_with(store, query: str, entry, mode: str, rng, max_new: int = 6) -> str:
    """Replicate SlotMem.emit's value schedule but choose the injected direction by `mode`:
    real = the entry's stored value(s); null = random unit vectors of the SAME schedule length;
    baseline = no injection. Everything else identical, so the only difference is the direction."""
    ids = store.tok(query, return_tensors="pt").input_ids.to(slot.DEV)
    seq = ids
    if mode != "baseline":
        v1 = entry["value"]
        sched = [v1]
        if len(entry["ans_ids"]) > 1:
            v2 = store.W_U[entry["ans_ids"][1]].float()
            sched.append(v2 / (v2.norm() + 1e-8))
        if mode == "null":
            new = []
            for v in sched:
                r = torch.randn(v.shape, generator=rng).to(v.dtype)
                new.append(r / (r.norm() + 1e-8))   # random unit, matched norm after * eta
            sched = new
        for vec in sched:
            store._inject = store.eta * vec
            try:
                nxt = store.model(seq).logits[0, -1].argmax()
            finally:
                store._inject = None
            seq = torch.cat([seq, nxt.view(1, 1)], 1)
    remaining = max_new - (seq.shape[1] - ids.shape[1])
    out = seq if remaining <= 0 else store.model.generate(
        seq, attention_mask=torch.ones_like(seq), max_new_tokens=remaining,
        do_sample=False, pad_token_id=store.tok.eos_token_id or 0)
    return store.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def run(model_name: str, layer: int, tag: str):
    rng = torch.Generator().manual_seed(0)
    store = slot.SlotMem(model_name, layer)
    results, usable = [], []
    for cue, answer in FACTS:
        store.entries = []
        w = store.write(cue, answer, gate=False)   # force-store; we WANT unknown facts
        entry = store.entries[-1]
        base = emit_with(store, cue, entry, "baseline", rng)
        if answer_hit(base, answer):
            results.append({"cue": cue, "answer": answer.strip(), "skipped": "model already knows it",
                            "baseline": base.strip()})
            continue
        real = emit_with(store, cue, entry, "real", rng)
        null = emit_with(store, cue, entry, "null", rng)
        row = {"cue": cue, "answer": answer.strip(), "surprise": w.get("surprise"),
               "baseline": base.strip(), "real": real.strip(), "null": null.strip(),
               "real_hit": answer_hit(real, answer), "null_hit": answer_hit(null, answer)}
        results.append(row)
        usable.append(row)
        print(f"{cue[:40]!r:<42} ans={answer.strip()!r:<12} "
              f"real{'HIT' if row['real_hit'] else '---'} null{'HIT' if row['null_hit'] else '---'}  "
              f"real={real.strip()[:24]!r}")

    n = len(usable)
    real_hits = sum(r["real_hit"] for r in usable)
    null_hits = sum(r["null_hit"] for r in usable)
    summary = {
        "model": model_name, "layer": layer, "n_usable": n,
        "n_skipped_known": len(results) - n,
        "real_hit_rate": round(real_hits / n, 3) if n else None,
        "null_hit_rate": round(null_hits / n, 3) if n else None,
        "decision": None,
    }
    if n:
        if real_hits >= max(1, int(0.6 * n)) and real_hits >= 2 * max(1, null_hits):
            summary["decision"] = ("PROMOTE-CANDIDATE: value injection recovers the stored answer "
                                   "well above the random-direction control")
        elif real_hits <= null_hits:
            summary["decision"] = ("KEEP-INSTRUMENTATION: value injection does not beat a random "
                                   "direction -- the 'memory' is generic perturbation, not recall")
        else:
            summary["decision"] = ("MIXED: value injection beats null but not decisively -- more "
                                   "work before promotion (schedule, layer, eta)")
    out = os.path.join(REPO, "runs", "experiments", f"facts_injection_{tag}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"summary": summary, "results": results}, open(out, "w"), indent=2)
    print("\n=== facts injection (null-controlled) ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"wrote {out}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--tag", default="qwen2.5-1.5b")
    args = ap.parse_args()
    run(args.model, args.layer, args.tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
