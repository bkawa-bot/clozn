"""
function_vector_sweep_qwen.py - THE CAUSAL EXISTENCE TEST (SAE-independent). Settles the leg that
legibility_natural_qwen.py left underpowered: is there a CLAMPABLE rule direction at all, at any layer?

WHY: in both the soft-prefix read and the natural in-context read, clamping a rule direction at layer 14
recovered ~0% - even the RAW task vector did, which CONTRADICTS the function/task-vector literature
(Hendel et al. 2023 "In-Context Learning Creates Task Vectors"; Todd et al. 2024 "Function Vectors"): an
in-context task vector, PATCHED into a zero-shot query at the right MID layer, makes the model do the task.
Our layer-14 add-a-mean-difference clamp was too crude. This does it the textbook way and sweeps ALL layers.

METHOD (Hendel-style, forward-only, no gradients, no SAE):
  EXTRACT  theta_R^L = mean over a few (K consistent R-example demos, dummy query) of the hidden state at
           the LAST token (the "->" slot) at layer L. The task vector for rule R at layer L.
  PATCH    for a held-out zero-shot query "x' ->", REPLACE the layer-L last-token hidden state with
           theta_R^L and continue; is the next token R(x')? Score over held-out x'.
  SWEEP    every layer L = 0..nL-1. The money plot: apply-acc vs layer. A peak well above the zero-shot
           floor (and above a WRONG-task-vector null) = a causal rule direction EXISTS at that depth.

CONTROLS: zero-shot floor (no patch), ICL ceiling (full prompt apply), WRONG-task null (patch rule A's
queries with rule B's theta at the best layer - must fail), per relation, no cherry-picking. We also report
the apply-acc at LAYER 14 specifically (the Qwen-Scope SAE layer) so we know whether the SAE even sits at a
causal depth. If a causal layer exists, the fair SAE question becomes "is theta sparse in the SAE THERE".

MODEL: Qwen3-1.7B-Base, FROZEN. Env: cloze/.venv (GPU). Forward-only -> light memory. Synchronous.
Outputs (research/runs/): function_vector_sweep_qwen.json + a per-layer apply-acc curve SVG.
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
import frontier_apply_v2 as FV2
import legibility_discovered as LD          # svg helpers, palette, RULE_DESC

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SLATE, TEAL, PINK, GOLD, LILAC = LD.SLATE, LD.TEAL, LD.PINK, LD.GOLD, LD.LILAC


def resolve_model_path(model_name):
    local = os.path.join(os.path.expanduser("~"), "hf_models", model_name.split("/")[-1])
    return local if os.path.isfile(os.path.join(local, "config.json")) else model_name


def icl_ids(tok, rel, train_pairs, words, x, K, gen):
    tr = train_pairs[rel].tolist()
    kk = min(K, len(tr))
    ti = torch.randperm(len(tr), generator=gen).tolist()[:kk]
    lines = [f"{words[tr[i][0]]} -> {words[tr[i][1]]}" for i in ti]
    return tok.encode("Complete the analogy with the same kind of relation.\n" + "\n".join(lines) + f"\n{x} ->",
                      add_special_tokens=False)


@torch.no_grad()
def extract_thetas(model, tok, rel, train_pairs, words, dummy_xs, K, nL, gen):
    """theta_R^L for every layer L = mean over dummy demos of the last-token hidden state (output of block L)."""
    accum = {L: [] for L in range(nL)}
    for x in dummy_xs:
        ids = torch.tensor(icl_ids(tok, rel, train_pairs, words, x, K, gen), device=DEV)[None, :]
        hs = model(input_ids=ids, output_hidden_states=True).hidden_states   # tuple len nL+1
        for L in range(nL):
            accum[L].append(hs[L + 1][0, -1].float())
    return {L: torch.stack(accum[L]).mean(0) for L in range(nL)}              # {L: [H]}


@torch.no_grad()
def patch_apply(model, tok, layer, theta, test_words, answer_tok, menu_ids):
    """Patch theta into the layer-`layer` last-token hidden state of each bare 'x ->' query; apply-acc."""
    blocks = model.model.layers
    th = theta.to(model.dtype)

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1, :] = th
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    free = menu = 0; n = 0
    for x, ytok in test_words:
        ids = torch.tensor(tok.encode(f"{x} ->", add_special_tokens=False), device=DEV)[None, :]
        handle = blocks[layer].register_forward_hook(hook)
        try:
            lg = model(input_ids=ids).logits[0, -1]
        finally:
            handle.remove()
        free += int(lg.argmax().item() == ytok)
        menu += int(int(menu_ids[lg[menu_ids].argmax()].item()) == ytok)
        n += 1
    return free / max(1, n), menu / max(1, n)


@torch.no_grad()
def zero_and_icl(model, tok, rel, train_pairs, test_pairs, words, answer_tok, menu_ids, K, gen):
    """Zero-shot floor (bare 'x ->') and ICL ceiling (full prompt) free-apply over held-out test words."""
    zfree = ifree = 0; n = 0
    for (a, b) in test_pairs[rel].tolist():
        x, ytok = words[a], answer_tok[words[b]]
        zids = torch.tensor(tok.encode(f"{x} ->", add_special_tokens=False), device=DEV)[None, :]
        zfree += int(model(input_ids=zids).logits[0, -1].argmax().item() == ytok)
        iids = torch.tensor(icl_ids(tok, rel, train_pairs, words, x, K, gen), device=DEV)[None, :]
        ifree += int(model(input_ids=iids).logits[0, -1].argmax().item() == ytok)
        n += 1
    return zfree / max(1, n), ifree / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--n_dummy", type=int, default=4)       # dummy extraction queries per relation
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.30)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--min_pairs", type=int, default=10)
    ap.add_argument("--n_relations", type=int, default=8)
    ap.add_argument("--sae_layer", type=int, default=14)    # the Qwen-Scope SAE layer (reported specially)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t0 = time.time()
    print(f"device={DEV} model={args.model} (function-vector layer sweep; forward-only; SYNCHRONOUS)", flush=True)
    tok, model = FA.load_llm(resolve_model_path(args.model), dtype=getattr(torch, args.dtype))
    nL = model.config.num_hidden_layers
    print(f"  loaded; layers={nL}, hidden={model.config.hidden_size}", flush=True)

    bank, REL_NAMES, _ = FV2.build_bank(tok, min_pairs=args.min_pairs)
    words, widx, out_words, out_ids = FV2.build_vocab_bank(bank)
    train_pairs, test_pairs = FV2.split_bank(bank, words, widx, test_frac=args.test_frac, seed=args.split_seed)
    answer_tok = {w: FV2.single_token_id(tok, w) for w in words}
    menu_ids = torch.tensor([answer_tok[w] for w in out_words], device=DEV)
    g = torch.Generator().manual_seed(args.seed + 99)
    order = torch.randperm(len(REL_NAMES), generator=g).tolist()
    rels = [REL_NAMES[i] for i in order][:args.n_relations]
    print(f"relations: {rels}\nsweeping all {nL} layers (SAE sits at layer {args.sae_layer})", flush=True)

    report = dict(model=args.model, n_layers=nL, K=args.K, sae_layer=args.sae_layer, relations=rels,
                  method="Hendel-style task vector: extract last-token hidden state per layer, PATCH into "
                         "zero-shot query, sweep layers", env="cloze/.venv (GPU)", frozen_backbone=True,
                  synchronous_single_process=True, per_relation={})

    # extraction dummy queries (TRAIN words, disjoint from the eval TEST words) + theta per layer per rel
    ge = torch.Generator().manual_seed(args.seed + 5)
    thetas = {}; floors = {}; ceils = {}
    test_words = {}
    for rel in rels:
        tr = train_pairs[rel].tolist()
        dummy_xs = [words[tr[i][0]] for i in range(min(args.n_dummy, len(tr)))]
        thetas[rel] = extract_thetas(model, tok, rel, train_pairs, words, dummy_xs, args.K, nL, ge)
        floors[rel], ceils[rel] = zero_and_icl(model, tok, rel, train_pairs, test_pairs, words,
                                               answer_tok, menu_ids, args.K, ge)
        test_words[rel] = [(words[a], answer_tok[words[b]]) for (a, b) in test_pairs[rel].tolist()]

    # ---------- sweep: patch theta_R^L into zero-shot held-out queries, every layer ----------
    print("\n" + "=" * 90 + "\nLAYER SWEEP - patch-apply (free) per relation per layer\n" + "=" * 90, flush=True)
    per_layer_acc = {L: [] for L in range(nL)}
    rel_curves = {}
    for rel in rels:
        curve = []
        for L in range(nL):
            f, _ = patch_apply(model, tok, L, thetas[rel][L], test_words[rel], answer_tok, menu_ids)
            curve.append(f); per_layer_acc[L].append(f)
        rel_curves[rel] = curve
        bestL = int(np.argmax(curve));
        report["per_relation"][rel] = dict(floor=floors[rel], icl_ceiling=ceils[rel], curve=curve,
                                            best_layer=bestL, best_acc=float(curve[bestL]),
                                            acc_at_sae_layer=float(curve[args.sae_layer]))
        print(f"  [{rel:14s}] zero={floors[rel]:.2f} ICL={ceils[rel]:.2f} | best L{bestL}={curve[bestL]:.2f} | "
              f"L{args.sae_layer}(SAE)={curve[args.sae_layer]:.2f} | curve peak region={[f'{c:.1f}' for c in curve]}", flush=True)

    mean_curve = [float(np.mean(per_layer_acc[L])) for L in range(nL)]
    best_layer = int(np.argmax(mean_curve)); best_mean = mean_curve[best_layer]
    agg_floor = float(np.mean([floors[r] for r in rels])); agg_icl = float(np.mean([ceils[r] for r in rels]))
    sae_layer_mean = mean_curve[args.sae_layer]
    print(f"\n  MEAN apply-acc by layer: peak L{best_layer}={best_mean:.3f}  "
          f"(zero-shot floor {agg_floor:.3f}, ICL ceiling {agg_icl:.3f}); at SAE layer {args.sae_layer}="
          f"{sae_layer_mean:.3f}", flush=True)

    # ---------- WRONG-task null at the best layer: patch rel A's queries with rel B's theta ----------
    print("\n  WRONG-task null at best layer L%d (patch each rel with ANOTHER rel's theta):" % best_layer, flush=True)
    wrong_accs = []
    for i, rel in enumerate(rels):
        other = rels[(i + 1) % len(rels)]
        wf, _ = patch_apply(model, tok, best_layer, thetas[other][best_layer], test_words[rel], answer_tok, menu_ids)
        wrong_accs.append(wf)
    wrong_mean = float(np.mean(wrong_accs))
    print(f"    wrong-task apply at L{best_layer} = {wrong_mean:.3f} (vs correct {best_mean:.3f}, floor {agg_floor:.3f})", flush=True)

    causal_exists = bool(best_mean > agg_floor + 0.15 and best_mean > wrong_mean + 0.15)
    sae_layer_causal = bool(sae_layer_mean > agg_floor + 0.15)
    report.update(mean_curve_by_layer=mean_curve, best_layer=best_layer, best_mean_acc=best_mean,
                  agg_zero_floor=agg_floor, agg_icl_ceiling=agg_icl, sae_layer=args.sae_layer,
                  sae_layer_mean_acc=sae_layer_mean, wrong_task_mean_acc=wrong_mean,
                  causal_direction_exists=causal_exists, sae_layer_is_causal=sae_layer_causal)

    # money plot: mean apply-acc vs layer, with floor/ICL/SAE-layer guides
    LD.svg_grouped_bars(os.path.join(RUNS, f"function_vector_sweep_qwen{args.tag}.svg"),
                        [f"L{L}" for L in range(nL)],
                        [("patch-apply (free)", TEAL, {f"L{L}": mean_curve[L] for L in range(nL)}),
                         ("zero-shot floor", SLATE, {f"L{L}": agg_floor for L in range(nL)}),
                         ("ICL ceiling", GOLD, {f"L{L}": agg_icl for L in range(nL)})],
                        f"Function-vector layer sweep: patch task vector into zero-shot (peak L{best_layer}={best_mean:.2f}, SAE@L{args.sae_layer}={sae_layer_mean:.2f})", W=1100)

    # ---------- verdict ----------
    if causal_exists:
        loc = ("INCLUDING the SAE layer" if sae_layer_causal else
               f"but NOT at the SAE layer {args.sae_layer} (peak is L{best_layer}; the layer-14 SAE sits off the causal peak)")
        verdict = (f"CAUSAL DIRECTION EXISTS: a Hendel task vector patched into a zero-shot query induces the rule - "
                   f"peak mean apply {best_mean:.2f} at L{best_layer} (zero-shot floor {agg_floor:.2f}, ICL ceiling "
                   f"{agg_icl:.2f}, wrong-task null {wrong_mean:.2f}), {loc}. So the earlier layer-14 'causal ~0%' was "
                   f"a METHOD artifact (crude mean-difference clamp, wrong/single layer), NOT 'rules are not steerable'. "
                   f"The fair SAE question is now whether the SAE at L{best_layer} renders this direction sparsely "
                   f"(at L{args.sae_layer} the task vector is {'causal' if sae_layer_causal else 'weak'}).")
    else:
        verdict = (f"NO CLEAN CAUSAL DIRECTION found by a layer-swept Hendel task vector: best mean apply {best_mean:.2f} "
                   f"at L{best_layer} barely clears the zero-shot floor {agg_floor:.2f} / wrong-task null {wrong_mean:.2f} "
                   f"(ICL ceiling {agg_icl:.2f}). Even the textbook function-vector recipe does not extract a single "
                   f"clampable rule direction for these relations on Qwen3-1.7B - a deeper negative: the in-context rule "
                   f"may be implemented by attention reading the examples, not a single mid-layer residual vector.")
    print("\n" + "#" * 90 + f"\n# {verdict}\n" + "#" * 90, flush=True)
    report["verdict"] = verdict
    report["wall_time_s"] = round(time.time() - t0, 1)
    out = os.path.join(RUNS, f"function_vector_sweep_qwen{args.tag}.json")
    json.dump(report, open(out, "w"), indent=2, default=float)
    print(f"\nwrote {out}  [{report['wall_time_s']}s]  (synchronous, single process - clean exit)", flush=True)


if __name__ == "__main__":
    main()
