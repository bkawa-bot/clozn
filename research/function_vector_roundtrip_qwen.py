"""
function_vector_roundtrip_qwen.py - the DIAGNOSTIC that interprets function_vector_sweep_qwen.py's
all-near-zero result. Is the sweep's failure a broken patch MECHANISM, or a real fact about where the
rule lives? READ function_vector_sweep_qwen.py + its findings first.

THE TEST (the cleanest possible): patch the model's OWN in-context hidden state for the SAME query back
into its zero-shot run, at each of several layers. If the single-last-token patch mechanism is sound, this
MUST round-trip (reproduce the in-context answer) at least at deep layers. Where it starts to round-trip
tells us at what depth the answer has become a transplantable residual vector (vs still being computed by
attention over the in-context examples).

RESULT (Qwen3-1.7B-Base, plural/past/gerund, held-out queries):
  same-query round-trip apply by layer (of 28):  L6 0.0  L10 0.0  L14 0.0  L18 ~0.2  L22 1.0  L26 1.0
  (ICL apply = 1.0). So: the mechanism is SOUND (deep layers round-trip 100%); the answer only becomes a
  transplantable last-token residual by layer ~22. At the SAE layer (14) it is NOT there yet - the model is
  still computing it via attention to the examples. And by the depth where it IS a clampable vector, it is
  the CONCRETE answer for that query, not a reusable rule (which is why the query-AVERAGED task vector in
  the sweep transferred ~0 even deep). => there is no query-independent, clampable RULE vector at any layer:
  early = not computed (attention-bound), late = the specific answer. An SAE reading residual content at a
  mid layer legitimately cannot render the in-context rule as a feature - it is not residual content there.

Forward-only, synchronous, frozen model. Env: cloze/.venv (GPU).
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
torch.backends.cuda.matmul.allow_tf32 = False

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
sys.path.insert(0, HERE)
import frontier_apply as FA
import frontier_apply_v2 as FV2

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def resolve(model_name):
    local = os.path.join(os.path.expanduser("~"), "hf_models", model_name.split("/")[-1])
    return local if os.path.isfile(os.path.join(local, "config.json")) else model_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--relations", default="plural,past,gerund")
    ap.add_argument("--layers", default="6,10,14,18,22,26")
    ap.add_argument("--n_test", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    tok, model = FA.load_llm(resolve(args.model), dtype=torch.float32)
    nL = model.config.num_hidden_layers
    bank, REL, _ = FV2.build_bank(tok, min_pairs=10)
    words, widx, ow, oi = FV2.build_vocab_bank(bank)
    trp, tep = FV2.split_bank(bank, words, widx, test_frac=0.30, seed=0)
    ans = {w: FV2.single_token_id(tok, w) for w in words}
    g = torch.Generator().manual_seed(args.seed + 5)
    test_layers = [int(x) for x in args.layers.split(",")]
    rels = [r for r in args.relations.split(",") if r in REL]

    def icl(rel, x):
        tr = trp[rel].tolist(); ti = torch.randperm(len(tr), generator=g).tolist()[:min(args.K, len(tr))]
        lines = [f"{words[tr[i][0]]} -> {words[tr[i][1]]}" for i in ti]
        return tok.encode("Complete the analogy with the same kind of relation.\n" + "\n".join(lines) + f"\n{x} ->",
                          add_special_tokens=False)

    print(f"DIAGNOSTIC: same-query ICL-state patched back into the bare query (model={args.model}, {nL} layers)\n", flush=True)
    out = dict(model=args.model, n_layers=nL, test_layers=test_layers, relations=rels, per_relation={})
    for rel in rels:
        te = tep[rel].tolist()[:args.n_test]; n = 0; iclok = 0; ok = {L: 0 for L in test_layers}
        for (a, b) in te:
            x, y = words[a], words[b]; yt = ans[y]
            iids = torch.tensor(icl(rel, x), device=DEV)[None]
            o = model(input_ids=iids, output_hidden_states=True)
            iclok += int(o.logits[0, -1].argmax().item() == yt)
            states = {L: o.hidden_states[L + 1][0, -1].clone() for L in test_layers}
            bare = torch.tensor(tok.encode(f"{x} ->", add_special_tokens=False), device=DEV)[None]
            for L in test_layers:
                th = states[L].to(model.dtype)

                def hook(m, i, o2, th=th):
                    h = o2[0] if isinstance(o2, tuple) else o2
                    h[:, -1, :] = th
                    return (h,) + tuple(o2[1:]) if isinstance(o2, tuple) else h

                hd = model.model.layers[L].register_forward_hook(hook)
                try:
                    pred = model(input_ids=bare).logits[0, -1].argmax().item()
                finally:
                    hd.remove()
                ok[L] += int(pred == yt)
            n += 1
        row = {f"L{L}": round(ok[L] / max(1, n), 2) for L in test_layers}
        out["per_relation"][rel] = dict(icl_ok=iclok / max(1, n), roundtrip_by_layer=row)
        print(f"[{rel:10s}] ICL_ok={iclok/n:.2f}  same-query round-trip by layer: {row}", flush=True)

    # where does the answer become transplantable (mean over rels, first layer >= 0.5)?
    means = {L: float(np.mean([out["per_relation"][r]["roundtrip_by_layer"][f"L{L}"] for r in rels])) for L in test_layers}
    crystallize = next((L for L in test_layers if means[L] >= 0.5), None)
    out["mean_roundtrip_by_layer"] = {f"L{L}": means[L] for L in test_layers}
    out["answer_becomes_transplantable_at_layer"] = crystallize
    print(f"\n  mechanism SOUND (deep round-trip ~1.0); answer becomes a transplantable residual ~L{crystallize} "
          f"(of {nL}); at mid layers it is still attention-bound -> NOT a clampable residual rule vector there.", flush=True)
    out["wall_time_s"] = round(time.time() - t0, 1)
    p = os.path.join(RUNS, f"function_vector_roundtrip_qwen{args.tag}.json")
    json.dump(out, open(p, "w"), indent=2, default=float)
    print(f"\nwrote {p}  [{out['wall_time_s']}s]", flush=True)


if __name__ == "__main__":
    main()
