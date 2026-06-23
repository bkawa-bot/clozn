"""
legibility_natural_qwen.py - THE FAIR TEST. Does the Qwen-Scope SAE read a rule the model applies the
NATURAL way (in-context), the SAE's home turf - instead of the out-of-distribution soft-prefix delta that
legibility_discovered_qwen.py fed it (1/4)? READ legibility_discovered_qwen.py + its findings first.

THE CRITIQUE THIS ANSWERS (Brigitte): "SAEs are trained for exactly this - if it isn't working we're doing
something wrong." Right. An SAE is trained to sparsely reconstruct activations from REAL text; we fed it the
activation a gradient-found INJECTED soft prefix causes, then a DIFFERENCE of two such encodings - OOD twice
over. The fair read is the activation when the model applies the rule from in-context examples (real tokens,
real forward pass), against a zero-shot baseline. That is the classic function/task-vector construction.

WHAT THIS RUNS (same four legs as legibility_discovered_qwen, but the rule footprint is the NATURAL one):
  For each held-out relation R and held-out query word x:
    A_rule[x]  = resid @ blocks.14.hook_resid_post at the answer slot of an ICL prompt (K consistent R
                 examples + "x ->")   -- a NATURAL activation where the model is applying R (we verify it
                 actually applies, the ICL ceiling).
    A_zero[x]  = resid @ same hook at the answer slot of the BARE "x ->" (zero-shot)   -- rule absent.
    footprint  = mean_x [ enc_SAE(A_rule) - enc_SAE(A_zero) ]   (which discovered features the rule lights)
    raw vec    = mean_x [ A_rule - A_zero ]                     (the task vector, in raw residual space)
  STEP 2 SPARSE + SPECIFIC, STEP 3 NAMEABLE (logit-lens), STEP 4 CAUSAL - all on the NATURAL footprint.

THE LOAD-BEARING NEW CONTROL (STEP 4): clamp into a fresh BARE query (no examples) at layer 14, THREE ways:
    (a) the RAW task vector   -> is there a clampable rule direction AT ALL? (function-vector literature says
        yes; this is the ceiling for "a direction exists").
    (b) the SAE RECONSTRUCTION of the positive footprint -> does the DICTIONARY capture that direction?
    (c) RANDOM features (null).
  This separates "no clampable rule direction" (a fails too) from "the SAE throws the direction away" (a
  works, b doesn't). The latter would be the honest verdict on the SAE, not on the rule.

HONEST: ICL ceiling (does the natural state apply R), zero-shot floor, sparsity vs random-direction null,
shared-removed specificity, raw-vs-SAE-vs-random causal, per-relation, no cherry-picking. A POSITIVE here
(the natural read IS sparse/nameable/causal where the prefix read was not) is the real correction; a
NEGATIVE-even-here points past method to substrate/dictionary. MODEL FROZEN. Synchronous, single process.
Env: cloze/.venv (GPU). Reuses RawTopKSAE + QwenHarness from legibility_discovered_qwen (no sae_lens at runtime).
Outputs (research/runs/): legibility_natural_qwen.json + SVGs (ICL apply, specificity, causal).
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
sys.path.insert(0, HERE)

import frontier_apply as FA
import frontier_apply_v2 as FV2
import legibility_discovered as LD               # sparsity_stats, topk_features, svg_*, RULE_DESC, palette
import legibility_discovered_qwen as LDQ         # RawTopKSAE, QwenHarness, resolve_model_path (+ TF32 off)

DEV = LDQ.DEV
SLATE, TEAL, PINK, GOLD, LILAC = LD.SLATE, LD.TEAL, LD.PINK, LD.GOLD, LD.LILAC


def icl_ids(tok, rel, train_pairs, words, x, K, gen):
    """Token ids for an in-context prompt: K consistent R example pairs + the query 'x ->' (no BOS, to
    match the bare query / the SAE-read convention used throughout these runs). Same prompt shape as
    frontier_apply.icl_ceiling_rel so the apply ceiling is comparable."""
    tr = train_pairs[rel].tolist()
    kk = min(K, len(tr))
    ti = torch.randperm(len(tr), generator=gen).tolist()[:kk]
    lines = [f"{words[tr[i][0]]} -> {words[tr[i][1]]}" for i in ti]
    prompt = "Complete the analogy with the same kind of relation.\n" + "\n".join(lines) + f"\n{x} ->"
    return tok.encode(prompt, add_special_tokens=False)


@torch.no_grad()
def last_resid_and_logits(hz, ids):
    """resid @ blocks.layer.hook_resid_post at the LAST position + final next-token logits, one forward."""
    t = torch.tensor(ids, device=DEV)[None, :]
    out = hz.model(input_ids=t, output_hidden_states=True)
    resid = out.hidden_states[hz.layer + 1][0, -1].float()       # [H]
    return resid, out.logits[0, -1].float()                      # [H], [V]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--sae_npz", default=os.path.join(RUNS, "qwen_scope_1p7b_layer14.npz"))
    ap.add_argument("--sae_meta", default=os.path.join(RUNS, "qwen_scope_1p7b_layer14.meta.json"))
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.30)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--min_pairs", type=int, default=10)
    ap.add_argument("--n_relations", type=int, default=8)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--clamp_scales", default="1,2,4,8")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t_start = time.time()
    print(f"device={DEV}  model={args.model}  SAE@layer{args.layer}  K={args.K}  (FAIR/natural read; "
          f"SYNCHRONOUS)", flush=True)
    sae = LDQ.RawTopKSAE(args.sae_npz, args.sae_meta, DEV)
    hz = LDQ.QwenHarness(args.model, sae, args.layer, dtype=args.dtype)

    bank, REL_NAMES, _ = FV2.build_bank(hz.tok, min_pairs=args.min_pairs)
    words, widx, out_words, out_ids = FV2.build_vocab_bank(bank)
    train_pairs, test_pairs = FV2.split_bank(bank, words, widx, test_frac=args.test_frac, seed=args.split_seed)
    answer_tok = {w: FV2.single_token_id(hz.tok, w) for w in words}
    menu_ids = torch.tensor([answer_tok[w] for w in out_words], device=DEV)
    q_emb_cache = FA.cache_query_embeds(hz.tok, hz.model, words)
    chance = 1.0 / len(out_ids)
    g = torch.Generator().manual_seed(args.seed + 99)
    order = torch.randperm(len(REL_NAMES), generator=g).tolist()
    rels = [REL_NAMES[i] for i in order][:args.n_relations]
    print(f"|relations|={len(REL_NAMES)} |words|={len(words)} |menu|={len(out_ids)} chance={chance:.4f}\n"
          f"relations: {rels}", flush=True)

    report = dict(model=args.model, sae="qwen-scope-3-1.7b-base-w32k-l50 (TopK, layer14)", d_sae=sae.d_sae,
                  device=DEV, K=args.K, hook="blocks.14.hook_resid_post", chance=chance,
                  read="NATURAL in-context application (function-vector style), NOT the soft-prefix delta",
                  env="cloze/.venv (torch cu128, RTX 5080)", frozen_backbone=True, synchronous_single_process=True)

    # ---------- STEP 1: build the NATURAL rule footprint + the task vector, verify ICL applies ----------
    print("\n" + "=" * 90 + "\nSTEP 1 - NATURAL read: ICL apply (ceiling) + footprint enc(rule)-enc(zero) per rule\n" + "=" * 90, flush=True)
    footprints = {}; rawvecs = {}; spars = {}; tops = {}; icl_rows = {}
    ge = torch.Generator().manual_seed(args.seed + 5)
    for rel in rels:
        te = test_pairs[rel].tolist()
        A_rule, A_zero = [], []; correct = 0
        for (a, b) in te:
            x, y = words[a], words[b]
            ids = icl_ids(hz.tok, rel, train_pairs, words, x, args.K, ge)
            r_rule, lg = last_resid_and_logits(hz, ids)
            correct += int(lg.argmax().item() == answer_tok[y])
            r_zero, _ = last_resid_and_logits(hz, FA.encode_query_ids(hz.tok, x))
            A_rule.append(r_rule); A_zero.append(r_zero)
        A_rule = torch.stack(A_rule); A_zero = torch.stack(A_zero)
        icl_acc = correct / max(1, len(te))
        fp = (sae.encode(A_rule) - sae.encode(A_zero)).mean(0)
        footprints[rel] = fp; rawvecs[rel] = (A_rule - A_zero).mean(0)
        spars[rel] = LD.sparsity_stats(fp)
        idx, vals = LD.topk_features(fp, k=args.topk); tops[rel] = dict(idx=idx, vals=vals)
        icl_rows[rel] = dict(icl_apply=icl_acc, n_test=len(te), L0=spars[rel]["l0"],
                             PR=spars[rel]["participation_ratio"])
        print(f"  [{rel:14s}] ICL apply(ceiling)={icl_acc:.3f}  L0={spars[rel]['l0']:5d}  "
              f"k@90%={spars[rel]['k_for_90pct']:3d}  PR={spars[rel]['participation_ratio']:.1f}  top={idx[:6]}", flush=True)
    LD.svg_grouped_bars(os.path.join(RUNS, f"legibility_natural_qwen_icl{args.tag}.svg"), rels,
                        [("ICL apply (ceiling)", TEAL, {r: icl_rows[r]["icl_apply"] for r in rels})],
                        "STEP 1 - natural in-context apply ceiling per rule (the state we read)")

    # ---------- STEP 2: sparse + specific ----------
    g2 = torch.Generator(device="cpu").manual_seed(args.seed + 7)
    rand_PRs = []
    for _ in range(max(3, len(rels))):
        v = torch.randn(hz.H, generator=g2); v = v / v.norm()
        nat = rawvecs[rels[0]].norm()
        vv = v.to(DEV) * nat
        fr = (sae.encode(vv[None])[0] - sae.encode((0 * vv)[None])[0])
        rand_PRs.append(LD.sparsity_stats(fr)["participation_ratio"])
    null_PR = float(np.mean(rand_PRs)); real_PR = float(np.mean([spars[r]["participation_ratio"] for r in rels]))
    is_sparse = bool(real_PR < 0.6 * null_PR)
    pos = {r: torch.clamp(footprints[r], min=0) for r in rels}
    mean_delta = torch.stack([pos[r] for r in rels]).mean(0)
    cent = {r: pos[r] - mean_delta for r in rels}
    cosc_M = [[float(F.cosine_similarity(cent[a][None], cent[b][None])[0]) for b in rels] for a in rels]
    offc = [cosc_M[i][j] for i in range(len(rels)) for j in range(len(rels)) if i != j]
    mean_offc = float(np.mean(offc)); is_specific = bool(mean_offc < 0.25)
    print(f"\n  SPARSITY: real PR={real_PR:.1f} vs null={null_PR:.1f} -> {'SPARSER' if is_sparse else 'NOT clearly sparser'}")
    print(f"  SPECIFICITY: shared-removed off-diag cos={mean_offc:.3f} -> {'RULE-SPECIFIC' if is_specific else 'OVERLAPPING'}", flush=True)
    report["step2_readout"] = dict(sparsity_real_PR=real_PR, sparsity_null_PR=null_PR, is_sparse=is_sparse,
                                   mean_offdiag_centered_cos=mean_offc, is_specific=is_specific,
                                   per_relation_sparsity=spars, names=rels)
    LD.svg_confusion(os.path.join(RUNS, f"legibility_natural_qwen_specificity{args.tag}.svg"), rels, cosc_M,
                     "STEP 2 - natural rule x rule feature cosine (shared-removed; off-diag LOW = specific)")

    # ---------- STEP 3: nameability (logit-lens) ----------
    print("\n" + "=" * 90 + "\nSTEP 3 - NAMEABILITY (logit-lens of each top feature's decoder direction)\n" + "=" * 90, flush=True)
    name_rows = {}; fracs = []
    for rel in rels:
        ans = set()
        for (a, b) in (test_pairs[rel].tolist() + train_pairs[rel].tolist()):
            ans.add(words[b].lower()); ans.add(hz.tok.decode([answer_tok[words[b]]]).strip().lower())
        feats = []; nrel = 0
        for f, dv in list(zip(tops[rel]["idx"], tops[rel]["vals"]))[:8]:
            toks = hz.feature_tokens(int(f)); tl = set(t.lower() for t in toks if t)
            hit = bool(tl & ans)
            if hit: nrel += 1
            feats.append(dict(feature=int(f), delta=float(dv), promotes=toks, rule_relevant=hit))
        frac = nrel / max(1, len(feats)); fracs.append(frac)
        name_rows[rel] = dict(features=feats, frac_rule_relevant=frac)
        print(f"  [{rel:12s}='{LD.RULE_DESC.get(rel, rel)}'] {int(frac*100)}% top feats name the rule. "
              f"e.g. f{feats[0]['feature']} -> {feats[0]['promotes'][:6]}", flush=True)
    mean_name = float(np.mean(fracs)); is_nameable = bool(mean_name >= 0.5)
    print(f"\n  NAMEABILITY: mean {int(mean_name*100)}% -> {'NAMEABLE' if is_nameable else 'NOT cleanly nameable'}", flush=True)
    report["step3_nameability"] = dict(per_relation=name_rows, mean_frac_rule_relevant=mean_name,
                                       is_nameable=is_nameable, method="logit-lens")

    # ---------- STEP 4: causal -- raw task vector vs SAE reconstruction vs random, clamped into a bare query ----------
    print("\n" + "=" * 90 + "\nSTEP 4 - CAUSAL: clamp into a BARE query. RAW task vec (is there a direction?) vs SAE recon (does the dict capture it?) vs random\n" + "=" * 90, flush=True)
    scales = [float(s) for s in args.clamp_scales.split(",")]
    g3 = torch.Generator(device="cpu").manual_seed(args.seed + 17)
    rows = {}
    for rel in rels:
        bf, _ = hz.apply_acc(None, rel, test_pairs, words, q_emb_cache, answer_tok, menu_ids)   # zero-shot floor
        icl = icl_rows[rel]["icl_apply"]                                                          # ICL ceiling
        nat = float(rawvecs[rel].norm())
        raw_unit = rawvecs[rel] / (nat + 1e-8)
        sae_unit = LDQ.positive_reconstruction(hz, footprints[rel])
        ridx = torch.randperm(sae.d_sae, generator=g3)[:args.topk].tolist()
        rw = torch.clamp(torch.randn(args.topk, generator=g3), min=0.05).to(DEV)
        rvec = (rw[:, None] * sae.W_dec[torch.tensor(ridx, device=DEV)]).sum(0); rvec = rvec / (rvec.norm() + 1e-8)
        raw_best = dict(free=-1, scale=None); sae_best = dict(free=-1, scale=None); rand_best = dict(free=-1, scale=None)
        per_scale = {}
        for sc in scales:
            rawf, _ = hz.causal_apply(rel, test_pairs, words, q_emb_cache, answer_tok, menu_ids, raw_unit * (sc * nat))
            saef, _ = hz.causal_apply(rel, test_pairs, words, q_emb_cache, answer_tok, menu_ids, sae_unit * (sc * nat))
            randf, _ = hz.causal_apply(rel, test_pairs, words, q_emb_cache, answer_tok, menu_ids, rvec * (sc * nat))
            per_scale[sc] = dict(raw_free=rawf, sae_free=saef, rand_free=randf)
            if rawf > raw_best["free"]: raw_best = dict(free=rawf, scale=sc)
            if saef > sae_best["free"]: sae_best = dict(free=saef, scale=sc)
            if randf > rand_best["free"]: rand_best = dict(free=randf, scale=sc)
        rows[rel] = dict(zero_floor=bf, icl_ceiling=icl, raw_best=raw_best, sae_best=sae_best,
                         rand_best=rand_best, per_scale=per_scale, nat_norm=nat)
        print(f"  [{rel:14s}] zero={bf:.3f}  RAW-taskvec={raw_best['free']:.3f}@x{raw_best['scale']}  "
              f"SAE-recon={sae_best['free']:.3f}@x{sae_best['scale']}  rand={rand_best['free']:.3f}  ICL={icl:.3f}", flush=True)
    agg = lambda k: float(np.mean([rows[r][k] if not isinstance(rows[r][k], dict) else rows[r][k]["free"] for r in rels]))
    a_zero = float(np.mean([rows[r]["zero_floor"] for r in rels]))
    a_raw = float(np.mean([rows[r]["raw_best"]["free"] for r in rels]))
    a_sae = float(np.mean([rows[r]["sae_best"]["free"] for r in rels]))
    a_rand = float(np.mean([rows[r]["rand_best"]["free"] for r in rels]))
    a_icl = float(np.mean([rows[r]["icl_ceiling"] for r in rels]))
    raw_causal = bool(a_raw > a_rand + 0.08 and a_raw > a_zero + 0.08)
    sae_causal = bool(a_sae > a_rand + 0.08 and a_sae > a_zero + 0.08)
    print(f"\n  CAUSAL aggregate: zero-floor={a_zero:.3f}  RAW-taskvec={a_raw:.3f}  SAE-recon={a_sae:.3f}  "
          f"random={a_rand:.3f}  ICL-ceiling={a_icl:.3f}", flush=True)
    print(f"    RAW task vector causal? {raw_causal}  |  SAE reconstruction causal? {sae_causal}", flush=True)
    report["step4_causal"] = dict(per_relation=rows, scales=scales, agg_zero_floor=a_zero, agg_raw_free=a_raw,
                                  agg_sae_free=a_sae, agg_rand_free=a_rand, agg_icl_ceiling=a_icl,
                                  raw_causal=raw_causal, sae_causal=sae_causal)
    LD.svg_grouped_bars(os.path.join(RUNS, f"legibility_natural_qwen_causal{args.tag}.svg"), rels,
                        [("zero-shot floor", SLATE, {r: rows[r]["zero_floor"] for r in rels}),
                         ("RAW task-vec clamp", PINK, {r: rows[r]["raw_best"]["free"] for r in rels}),
                         ("SAE-recon clamp", TEAL, {r: rows[r]["sae_best"]["free"] for r in rels}),
                         ("ICL ceiling", GOLD, {r: rows[r]["icl_ceiling"] for r in rels})],
                        "STEP 4 - causal: RAW task vector vs SAE reconstruction (clamped into a bare query)")

    # ---------- VERDICT ----------
    score = sum([is_sparse, is_specific, is_nameable, sae_causal])
    flags = [("SPARSE", is_sparse), ("RULE-SPECIFIC", is_specific), ("NAMEABLE", is_nameable), ("CAUSAL(SAE)", sae_causal)]
    passed = [n for n, b in flags if b]; failed = [n for n, b in flags if not b]
    raw_note = ("a RAW task vector IS causal (it exists as a clampable direction) but the SAE reconstruction "
                "is NOT - the dictionary throws the causal direction away" if (raw_causal and not sae_causal) else
                "even the RAW task vector is not causal here (no clean clampable layer-14 rule direction)" if not raw_causal else
                "the SAE reconstruction recovers the raw task vector's causal effect")
    verdict = (f"NATURAL READ {'POSITIVE 4/4' if score == 4 else f'{score}/4'}: {', '.join(passed) or 'none'}"
               + (f"; NOT {', '.join(failed)}" if failed else "") +
               f". sparsity PR {real_PR:.0f} vs null {null_PR:.0f}; shared-removed off-diag cos {mean_offc:.2f}; "
               f"nameable {int(mean_name*100)}%; causal RAW {a_raw:.2f} / SAE {a_sae:.2f} vs random {a_rand:.2f} "
               f"(zero {a_zero:.2f}, ICL {a_icl:.2f}). {raw_note}. Compare the soft-prefix read (1/4): does "
               f"reading the model's NATURAL in-context application change the answer?")
    print("\n" + "#" * 90 + f"\n# {verdict}\n" + "#" * 90, flush=True)
    report["verdict"] = verdict
    report["verdict_flags"] = dict(sparse=is_sparse, specific=is_specific, nameable=is_nameable,
                                   sae_causal=sae_causal, raw_causal=raw_causal, score=score)
    report["wall_time_s"] = round(time.time() - t_start, 1)
    out = os.path.join(RUNS, f"legibility_natural_qwen{args.tag}.json")
    json.dump(report, open(out, "w"), indent=2, default=float)
    print(f"\nwrote {out}  [{report['wall_time_s']}s]  (synchronous, single process - clean exit)", flush=True)


if __name__ == "__main__":
    main()
