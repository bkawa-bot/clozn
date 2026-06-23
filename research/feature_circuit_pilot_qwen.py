"""
feature_circuit_pilot_qwen.py - the FEATURE -> FEATURE CIRCUIT pilot. Can we draw REAL concept->concept
edges now that we have valid pretrained dictionaries (Qwen-Scope), the thing the think_graph note refused
to fake? READ legibility_discovered_qwen.py first (reuses its RawTopKSAE + the dump/verify pipeline).

THE HONEST METHOD: go straight to the gold-standard causal test, ABLATION (not transcoder-weight edges,
which would themselves still need ablation to verify). For one prompt:
  - baseline: SAE features at a SOURCE layer (14) and a TARGET layer (20) at the answer position.
  - for each top SOURCE feature i: ABLATE it (subtract act_i * W_dec_14[i] from resid_post@14 at the last
    position), run forward, and read every TARGET feature j at layer 20. delta_ij = base_act_j - ablated_act_j.
    A real causal edge i->j means ablating i moves j.
  - NULL: ablate RANDOM active source features (matched: they are really active), measure delta on the SAME
    targets -> the generic-perturbation floor. An edge is KEPT only if |delta_ij| beats the null (ablating
    ANY active feature perturbs downstream features a little; a real edge must exceed that).
  - also report the ANSWER-LOGIT effect of ablating each source (does the source matter for the output).

So edges are DRAWN ONLY IF verified by intervention AND above a null - exactly the bar the think_graph note
set. A NEGATIVE (nothing survives the null) is valid and likely-partial on a local 1.7B (our causal handles
have been weak); a POSITIVE means the note can be lifted and a real feature circuit drawn.

Dictionaries: Qwen-Scope residual SAEs at layers 14 and 20 (TopK, 32k; Apache-2.0/ungated), dumped via
.venv-sae and reloaded raw + verified here. Model Qwen3-1.7B-Base, FROZEN, forward-only. Synchronous.
Env: cloze/.venv (GPU). Outputs: research/runs/feature_circuit_pilot_qwen.json.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
sys.path.insert(0, HERE)
import frontier_apply as FA
import legibility_discovered_qwen as LDQ      # RawTopKSAE, resolve_model_path (+ TF32 off)

DEV = LDQ.DEV


def load_sae(layer):
    return LDQ.RawTopKSAE(os.path.join(RUNS, f"qwen_scope_1p7b_layer{layer}.npz"),
                          os.path.join(RUNS, f"qwen_scope_1p7b_layer{layer}.meta.json"), DEV)


@torch.no_grad()
def feature_tokens(model, tok, Wdec_row, topn=6):
    """Logit-lens name for a feature: tokens its decoder direction promotes (noisier at mid layers)."""
    h = model.model.norm(Wdec_row[None].to(model.dtype))
    top = (h @ model.lm_head.weight.T)[0].topk(topn).indices.tolist()
    return [tok.decode([int(t)]).strip() for t in top]


@torch.no_grad()
def run_prompt(model, ids, src_layer, tgt_layer):
    out = model(input_ids=ids, output_hidden_states=True)
    return (out.hidden_states[src_layer + 1][0, -1].float(),
            out.hidden_states[tgt_layer + 1][0, -1].float(),
            out.logits[0, -1].float())


@torch.no_grad()
def ablate_read(model, ids, src_layer, abl_vec, tgt_layer):
    """Subtract abl_vec from resid_post@src_layer (last pos); return resid_post@tgt_layer (last pos) + logits."""
    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        h[:, -1, :] = h[:, -1, :] - abl_vec.to(h.dtype)
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h
    hd = model.model.layers[src_layer].register_forward_hook(hook)
    try:
        out = model(input_ids=ids, output_hidden_states=True)
    finally:
        hd.remove()
    return out.hidden_states[tgt_layer + 1][0, -1].float(), out.logits[0, -1].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--src_layer", type=int, default=14)
    ap.add_argument("--tgt_layer", type=int, default=20)
    ap.add_argument("--topk_src", type=int, default=10)
    ap.add_argument("--topk_tgt", type=int, default=10)
    ap.add_argument("--n_null", type=int, default=20)     # random active source features for the null
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    print(f"device={DEV} model={args.model} src=L{args.src_layer} tgt=L{args.tgt_layer} (feature-circuit pilot; "
          f"forward-only)", flush=True)
    tok, model = FA.load_llm(LDQ.resolve_model_path(args.model), dtype=torch.float32)
    sae_s = load_sae(args.src_layer); sae_t = load_sae(args.tgt_layer)

    prompts = ["The first president of the United States was George",
               "The opposite of hot is",
               "The capital of France is"]
    rng = np.random.default_rng(args.seed)

    report = dict(model=args.model, src_layer=args.src_layer, tgt_layer=args.tgt_layer,
                  topk_src=args.topk_src, topk_tgt=args.topk_tgt, n_null=args.n_null,
                  method="ablate source SAE feature (subtract its decoder write) -> measure target SAE feature "
                         "change; keep edges that beat a random-active-feature ablation null",
                  env="cloze/.venv (GPU)", frozen_backbone=True, prompts={})

    for prompt in prompts:
        ids = torch.tensor(tok.encode(prompt, add_special_tokens=False), device=DEV)[None, :]
        r_s, r_t, logits = run_prompt(model, ids, args.src_layer, args.tgt_layer)
        pred_id = int(logits.argmax()); pred_tok = tok.decode([pred_id]).strip()
        fs = sae_s.encode(r_s[None])[0]; ft = sae_t.encode(r_t[None])[0]      # baseline features
        src_active = torch.nonzero(fs > 0).flatten().tolist()
        src_top = [int(i) for i in fs.topk(min(args.topk_src, len(src_active))).indices.tolist()]
        tgt_top = [int(j) for j in ft.topk(min(args.topk_tgt, int((ft > 0).sum()))).indices.tolist()]
        print(f"\n=== \"{prompt}\"  -> pred '{pred_tok}'  |src active|={len(src_active)} ===", flush=True)

        # NULL: ablate random active source features (not in src_top), measure |delta| on the top targets
        null_pool = [i for i in src_active if i not in set(src_top)]
        rng.shuffle(null_pool); null_pool = null_pool[:args.n_null]
        null_abs = []
        for i in null_pool:
            abl = float(fs[i]) * sae_s.W_dec[i]
            rt_abl, _ = ablate_read(model, ids, args.src_layer, abl, args.tgt_layer)
            ft_abl = sae_t.encode(rt_abl[None])[0]
            null_abs += [abs(float(ft[j] - ft_abl[j])) for j in tgt_top]
        null_abs = np.array(null_abs)
        thr = float(np.percentile(null_abs, 99)) if len(null_abs) else 0.0
        null_mean, null_max = float(null_abs.mean()), float(null_abs.max() if len(null_abs) else 0.0)

        # CANDIDATE edges: ablate each top source feature, measure delta on each top target + answer-logit drop
        edges = []; ans_effects = {}
        for i in src_top:
            abl = float(fs[i]) * sae_s.W_dec[i]
            rt_abl, logits_abl = ablate_read(model, ids, args.src_layer, abl, args.tgt_layer)
            ft_abl = sae_t.encode(rt_abl[None])[0]
            ans_effects[i] = float(logits[pred_id] - logits_abl[pred_id])   # answer-logit drop from ablating i
            for j in tgt_top:
                d = float(ft[j] - ft_abl[j])
                edges.append(dict(src=i, tgt=j, delta=d, abs=abs(d), survives=bool(abs(d) > thr)))
        edges.sort(key=lambda e: -e["abs"])
        n_surv = sum(e["survives"] for e in edges)

        # name the features that participate in surviving edges (logit-lens; noisier at the source layer)
        names = {}
        for e in edges[:12]:
            if e["src"] not in names: names[("s", e["src"])] = feature_tokens(model, tok, sae_s.W_dec[e["src"]])
            if e["tgt"] not in names: names[("t", e["tgt"])] = feature_tokens(model, tok, sae_t.W_dec[e["tgt"]])
        topsrc_by_ans = sorted(ans_effects.items(), key=lambda kv: -kv[1])[:3]

        print(f"  null |delta|: mean={null_mean:.3f} 99th-pct(threshold)={thr:.3f} max={null_max:.3f}", flush=True)
        print(f"  candidate edges={len(edges)}  SURVIVING (|delta|>{thr:.3f})={n_surv}", flush=True)
        for e in edges[:6]:
            sn = feature_tokens(model, tok, sae_s.W_dec[e["src"]])[:4]
            tn = feature_tokens(model, tok, sae_t.W_dec[e["tgt"]])[:4]
            print(f"    L{args.src_layer} f{e['src']} {sn} -> L{args.tgt_layer} f{e['tgt']} {tn}  "
                  f"delta={e['delta']:+.3f}  {'SURVIVES' if e['survives'] else 'null'}", flush=True)
        print(f"  top source features by ANSWER-logit drop: " +
              ", ".join(f"f{i}(d={d:+.2f})" for i, d in topsrc_by_ans), flush=True)

        report["prompts"][prompt] = dict(pred=pred_tok, n_src_active=len(src_active),
                                         src_top=src_top, tgt_top=tgt_top, null_mean=null_mean,
                                         null_threshold=thr, null_max=null_max,
                                         n_candidate_edges=len(edges), n_surviving=n_surv,
                                         top_edges=edges[:20], answer_effects=ans_effects,
                                         feature_names={f"{k[0]}{k[1]}": v for k, v in names.items()})

    # ---- verdict ----
    tot_cand = sum(p["n_candidate_edges"] for p in report["prompts"].values())
    tot_surv = sum(p["n_surviving"] for p in report["prompts"].values())
    frac = tot_surv / max(1, tot_cand)
    # a clean POSITIVE = a real fraction of candidate edges beat the 99th-pct null (more than the ~1% you'd
    # expect by definition of the percentile) AND the surviving deltas are sizable vs the null mean.
    verdict = (f"FEATURE-CIRCUIT PILOT: {tot_surv}/{tot_cand} candidate edges survive the random-ablation null "
               f"(99th pct) across {len(report['prompts'])} prompts ({100*frac:.0f}%). "
               + ("REAL causal feature->feature edges exist above the null: ablating specific source features "
                  "moves specific downstream features more than ablating generic active features does. The "
                  "think_graph 'no concept->concept edges' note CAN be lifted - a verified (ablation-checked) "
                  "feature circuit is drawable on Qwen3-1.7B + Qwen-Scope, at least sparsely."
                  if frac > 0.05 else
                  "NOT clearly above the null: ablating a specific source feature does not move downstream "
                  "features more than ablating a generic active one. On a local 1.7B the per-feature causal "
                  "edges are at/under the generic-perturbation floor - consistent with the weak causal handles "
                  "seen all through this arc. The honest call stays: do not draw concept->concept edges (they "
                  "do not survive verification here)."))
    print("\n" + "#" * 90 + f"\n# {verdict}\n" + "#" * 90, flush=True)
    report["verdict"] = verdict
    report["total_candidate_edges"] = tot_cand; report["total_surviving"] = tot_surv; report["surviving_frac"] = frac
    report["wall_time_s"] = round(time.time() - t0, 1)
    out = os.path.join(RUNS, f"feature_circuit_pilot_qwen{args.tag}.json")
    json.dump(report, open(out, "w"), indent=2, default=float)
    print(f"\nwrote {out}  [{report['wall_time_s']}s]  (synchronous, single process)", flush=True)


if __name__ == "__main__":
    main()
