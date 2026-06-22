"""
Phase-3 §3.6 (the close of the interp-at-scale arc) — does a SEMANTIC metric reveal concept
features that the token-coherence metric structurally HID?

THREE prior runs (toy SAE, a converged 16-32x SAE on 120k natural tokens, a layer-residual
transcoder) were all a null: their top features read as "token-identity detectors" and none beat
PCA. BUT every one used the SAME metric — top-activating-TOKEN coherence — which can ONLY reward
token-locking. A feature that fires on many DIFFERENT number words ("one","two","nine","forty")
would score LOW on token-coherence yet be a perfectly good *semantic* "number" feature. So the
metric may have hidden semantic structure the whole time. This spike tests that caveat two ways:

  (A) AUTO-INTERP (LLM-judged): reconstruct each top feature's top-activating examples WITH CONTEXT
      (a window of surrounding text, not the bare token) so a human/LLM can judge whether the
      pattern is SEMANTIC/abstract (a concept, syntactic role, topic) or just one surface token.
      This script EMITS the contexts (runs/p6_autointerp_contexts.json); the judging is done by the
      caller (an LLM), and the verdicts are recorded back into the findings doc.

  (B) CONCEPT ALIGNMENT (quantitative, the part we can compute here): does any SAE feature's
      activation track one of the inspector's SEMANTIC concept labels (atlas/probes:
      number / person / tense / sentence-type / sentiment) ACROSS DIFFERENT TOKENS? The concept
      corpora are matched-frame minimal pairs (e.g. "The cat sleeps" vs "The cats sleep"), so a
      feature that separates the two classes is tracking the CONCEPT, not a token. We harvest those
      corpora through the engine at the SAME layer-2 tap, encode them through the trained SAE (and a
      PCA basis), and score each unit by how well it linearly separates the concept (held-out AUC /
      k-fold sign accuracy). Best SAE feature vs best PCA axis, per concept. This is the cross-token
      semantic metric the token metric cannot express.

Honesty-first: a null here (no SAE feature beats PCA on concept alignment AND the top features are
token-bound even with context) is a valid, strong result — it would mean the metric was NOT the
confound and the 0.5B early-layer setup is a robust semantic null. We do not massage numbers; we
print the full per-concept table and the raw contexts.

Usage (from inspector/, cloze venv python; engine launch needs the GPU build + CUDA v13.3 on PATH):
    python spikes/p6_autointerp.py                 # reuse cached wikitext acts; harvest probe corpora
    python spikes/p6_autointerp.py --no-engine     # skip concept-alignment harvest (auto-interp only)
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.discover import standardize  # noqa: E402

# Reuse the big-SAE harness wholesale: the corpus loader, the GPU TorchSAE, the /harvest client,
# the server management, and the coherence metric all live there.
from spikes.p4_big_sae import (  # noqa: E402
    CACHE, RUNS, TorchSAE, harvest_text, kill_server, sae_coherence_from_topk,
    start_server, top_token_coherence, _norm,
)

# The inspector's SEMANTIC concept corpora (matched-frame minimal pairs) — the natural axis to test
# features against. Sentiment comes from probes.py; the grammar concepts from atlas.py.
from clozn.atlas import (  # noqa: E402
    NUMBER_SING, NUMBER_PLUR, TENSE_PAST, TENSE_PRES, PERSON_1, PERSON_3, QUESTION, STATEMENT,
)
from clozn.probes import DEFAULT_POS, DEFAULT_NEG, kfold_accuracy  # noqa: E402

CONCEPTS = {
    "number (sing/plural)": (NUMBER_SING, NUMBER_PLUR, "singular", "plural"),
    "tense (past/present)": (TENSE_PAST, TENSE_PRES, "past", "present"),
    "person (1st/3rd)":     (PERSON_1, PERSON_3, "1st", "3rd"),
    "sentence (q/stmt)":    (QUESTION, STATEMENT, "question", "statement"),
    "sentiment (pos/neg)":  (DEFAULT_POS, DEFAULT_NEG, "positive", "negative"),
}

CTX_JSON = os.path.join(RUNS, "p6_autointerp_contexts.json")
ALIGN_JSON = os.path.join(RUNS, "p6_concept_alignment.json")

# Training config = the EXACT reported-best from sae_at_scale_findings.md (16x, L1=8.0), so the
# dictionary is the same one the prior run judged with the token metric. Deterministic (seed=0).
EXP, L1 = 16, 8.0
BATCH, LR = 512, 1e-3


# ---- context reconstruction ---------------------------------------------------------------------
def reconstruct_context(pieces, i, window=14):
    """The pieces array is the corpus in token order, so a window of neighbours IS the context.
    Mark the focus token with << >> so the judge sees exactly what fired. Passage joins (rare,
    from the 800-char concatenation) just include a little adjacent text — harmless for reading."""
    lo, hi = max(0, i - window), min(len(pieces), i + window + 1)
    left = "".join(pieces[lo:i])
    foc = pieces[i]
    right = "".join(pieces[i + 1:hi])
    return (left + "<<" + foc + ">>" + right).replace("\n", " ").strip()


# ---- concept-alignment scoring ------------------------------------------------------------------
def auc(scores, labels):
    """Rank AUC of a 1-D score vs binary labels (1/0). 0.5 = chance; we report max(auc, 1-auc) so a
    feature that fires for EITHER class counts (direction-agnostic concept tracking)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    a = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return max(a, 1 - a)


def best_feature_auc_held_out(codes, y, min_fire=8, k=4, seed=0):
    """The HONEST single-feature metric. The naive 'max over all features of in-sample AUC' is a
    multiple-comparisons trap: with m=14k features and only 2n=24-48 sentences, SOME feature will
    separate the labels by chance — and a feature firing on just 4 sentences can hit AUC~1 by luck.
    Guard both ways: (1) only consider features that FIRE on >= min_fire sentences (a degenerate
    1-4 firer can't win); (2) score by HELD-OUT k-fold AUC (pick the best feature on train folds,
    measure it on the test fold) so an in-sample fluke doesn't survive cross-validation; (3) compare
    against a LABEL-PERMUTATION null (same procedure on shuffled labels) so we know what 'max over
    14k features' scores by chance on this tiny corpus. Returns (best_feat, ho_auc, null_auc)."""
    rng = np.random.default_rng(seed)
    n = len(y)
    fired = (codes > 1e-6).sum(0)
    cand = np.where(fired >= min_fire)[0]
    if len(cand) == 0:
        return -1, 0.5, 0.5

    def held_out(labels):
        idx = rng.permutation(n)
        folds = np.array_split(idx, k)
        accs = []
        for f in range(k):
            te = folds[f]
            tr = np.concatenate([folds[g] for g in range(k) if g != f])
            # pick the best candidate feature on the TRAIN rows only
            tr_auc = np.array([auc(codes[tr][:, j], labels[tr]) for j in cand])
            bj = cand[int(np.argmax(tr_auc))]
            # measure it on the held-out rows
            accs.append(auc(codes[te][:, bj], labels[te]))
        return float(np.mean(accs)), bj

    ho_auc, last_bj = held_out(y)
    # overall best feature by FULL-data held-out-style AUC (for naming) — but report the CV number
    full_auc = np.array([auc(codes[:, j], y) for j in cand])
    best_feat = int(cand[int(np.argmax(full_auc))])
    # label-permutation null: same held-out procedure, shuffled labels, averaged over a few draws
    null = np.mean([held_out(rng.permutation(y))[0] for _ in range(5)])
    return best_feat, ho_auc, float(null)


def harvest_concept_acts(pos_texts, neg_texts, layer=None):
    """Harvest the LAST-token residual for each probe sentence (one /harvest forward per sentence;
    we take the final token's activation as the sentence representation, matching how the inspector's
    probes read a sentence-level concept). Returns (acts[2n, d], labels[2n], reps[list[str]])."""
    acts, labels, reps = [], [], []
    for grp, lab in ((pos_texts, 1), (neg_texts, 0)):
        for t in grp:
            a, toks, _ = harvest_text(t, layer=layer)
            acts.append(a[-1])                 # final-token state = sentence rep
            labels.append(lab)
            reps.append(toks[-1] if toks else "")
    return np.stack(acts).astype(np.float32), np.array(labels), reps


def concept_alignment(sae, Xs_mu, Xs_sd, Vt, K, layer, no_engine=False):
    """For each concept: harvest the matched-frame corpora, standardize with the SAME train stats,
    encode through (a) the SAE dictionary and (b) the PCA basis, and score TWO ways:

      * BEST-SINGLE-UNIT (held-out, with a firing floor + a label-permutation null) — does any ONE
        SAE feature track the concept across different tokens better than any ONE PCA axis? This is
        the 'monosemantic concept feature' question the token metric structurally cannot ask.
      * WHOLE-REPRESENTATION probe — a standardized k-fold linear probe on the FULL SAE code vector
        vs the FULL PCA projection (the inspector's kfold_accuracy). This asks whether the SAE
        representation *as a whole* separates the concept better than PCA — the fair, no-cherry-pick
        comparison (both use the same probe, the same folds).

    Returns a list of per-concept dicts."""
    import torch
    rows = []
    if no_engine:
        return rows
    start_server()
    # self-gate echo
    a0, _, lyr = harvest_text("The cat sat.", layer=layer)
    print(f"  /harvest live for concept-alignment: layer={lyr}, n_embd={a0.shape[1]}", flush=True)
    for name, (pos, neg, pl, nl) in CONCEPTS.items():
        A, y, reps = harvest_concept_acts(pos, neg, layer=layer)
        As = (A - Xs_mu) / Xs_sd                              # SAME standardization as training
        with torch.no_grad():
            xt = torch.tensor(As, dtype=torch.float32, device=sae.device)
            codes = (xt @ sae.We + sae.be).clamp(min=0).cpu().numpy()   # [2n, m]
        proj = As @ Vt[:K].T                                  # [2n, K]

        # (1) best single unit, held-out + null
        sj, sae_ho, sae_null = best_feature_auc_held_out(codes, y, min_fire=8)
        # PCA: same held-out procedure over its K axes (no firing floor — PCA axes are dense)
        pj, pca_ho, pca_null = best_feature_auc_held_out(proj, y, min_fire=0)
        # distinct top tokens the winning SAE feature fired on (cross-token check)
        top_reps = []
        if sj >= 0:
            topfire = np.argsort(codes[:, sj])[::-1][:10]
            top_reps = [reps[i] for i in topfire if codes[i, sj] > 1e-6]

        # (2) whole-representation k-fold probe (SAE codes vs PCA proj), identical procedure
        sae_probe = kfold_accuracy(list(codes), list(y.astype(float)), k=6, ridge=10.0)
        pca_probe = kfold_accuracy(list(proj), list(y.astype(float)), k=6, ridge=10.0)
        raw_probe = kfold_accuracy(list(As), list(y.astype(float)), k=6, ridge=10.0)  # raw acts ref

        rec = {
            "concept": name, "pos_label": pl, "neg_label": nl, "n": len(y),
            "sae_best_feature": sj, "sae_best_auc_heldout": sae_ho, "sae_best_auc_null": sae_null,
            "sae_best_fires_on_n": int((codes[:, sj] > 1e-6).sum()) if sj >= 0 else 0,
            "sae_best_distinct_top_tokens": len(set(_norm(t) for t in top_reps)),
            "sae_best_top_reps": top_reps[:8],
            "pca_best_axis": pj, "pca_best_auc_heldout": pca_ho, "pca_best_auc_null": pca_null,
            "sae_whole_probe_acc": sae_probe, "pca_whole_probe_acc": pca_probe,
            "raw_acts_probe_acc": raw_probe,
        }
        rows.append(rec)
        print(f"  [{name:22}] single-unit held-out AUC: SAE f{sj}={sae_ho:.2f} (null {sae_null:.2f}, "
              f"fires {rec['sae_best_fires_on_n']}/{len(y)}, {rec['sae_best_distinct_top_tokens']} "
              f"distinct toks) vs PCA={pca_ho:.2f} (null {pca_null:.2f})", flush=True)
        print(f"  {'':22}  whole-repr probe acc: SAE={sae_probe:.2f}  PCA={pca_probe:.2f}  "
              f"raw={raw_probe:.2f}", flush=True)
    kill_server()
    return rows


def main():
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    no_engine = "--no-engine" in flags
    os.makedirs(RUNS, exist_ok=True)

    if not os.path.exists(CACHE):
        print(f"ERROR: {CACHE} not found. Run p4_big_sae.py first to harvest.", flush=True)
        sys.exit(1)
    d = np.load(CACHE, allow_pickle=True)
    X, pieces = d["X"], list(d["pieces"])
    layer = int(d["layer"]) if "layer" in d else 2
    print(f"=== loaded {X.shape[0]} cached rows x {X.shape[1]} dims (layer {layer}, natural) ===",
          flush=True)

    Xs, mu, sd = standardize(X)
    N, dim = Xs.shape

    # ---- PCA basis (same as p4_big_sae: covariance eigh route at this N) --------------------------
    K = 256
    C = (Xs.T @ Xs) / N
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    Vt = evecs[:, order].T
    var = evals[order] / evals.sum()
    pca_proj = Xs @ Vt[:K].T
    pca_coh, pca_modal, pca_tops = top_token_coherence(pca_proj, pieces, topn=20)
    print(f"PCA top-{K} token-coherence = {pca_coh.mean()*100:.1f}%  "
          f"(top-64 {pca_coh[:64].mean()*100:.1f}%)", flush=True)

    # ---- train the reported-best SAE (16x, L1=8.0), deterministic --------------------------------
    m = EXP * dim
    steps_per_epoch = max(1, (N + BATCH - 1) // BATCH)
    EPOCHS = int(min(200, max(40, (9000 + steps_per_epoch - 1) // steps_per_epoch)))
    print(f"\ntraining SAE {EXP}x (m={m}) L1={L1}: batch={BATCH} lr={LR} epochs={EPOCHS}", flush=True)
    t0 = time.time()
    sae = TorchSAE(dim, m=m, l1=L1, seed=0).fit(Xs, epochs=EPOCHS, lr=LR, batch_size=BATCH)
    fire, mse = sae.stats(Xs)
    live_mask = (fire >= 0.002) & (fire <= 0.4)
    top_idx, top_val = sae.codes_topk(Xs, topn=40)               # top-40 rows/feature for contexts
    coh, modal, tops = sae_coherence_from_topk(top_idx[:, :20], pieces, topn=20)
    n_live = int(live_mask.sum())
    print(f"  trained: live={n_live} mean_fire={fire.mean()*100:.2f}% "
          f"token-coherence(live)={coh[live_mask].mean()*100:.1f}% mse={mse:.3f} "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- assemble feature contexts, ranked three ways (for auto-interp) ---------------------------
    live_idx = np.where(live_mask)[0]
    # variance of each feature's activation (a different lens than density/coherence)
    # computed cheaply from the top-40 values as a proxy for "peaky vs spread" is unreliable; instead
    # rank by (1) token-coherence, (2) fire density, (3) activation spread among firers via top vals.
    val_spread = top_val[:, :20].std(1)                          # spread of top activations
    rank_coh = sorted(live_idx, key=lambda j: -coh[j])
    rank_density = sorted(live_idx, key=lambda j: -fire[j])
    rank_spread = sorted(live_idx, key=lambda j: -val_spread[j])

    def feature_record(j, n_ctx=10):
        rows = [int(i) for i in top_idx[j] if i >= 0][:n_ctx]
        return {
            "feature": int(j),
            "token_coherence": float(coh[j]),
            "modal_token": modal[j],
            "fires_pct": float(fire[j]),
            "top_tokens": [pieces[i] for i in rows],
            "top_contexts": [reconstruct_context(pieces, i) for i in rows],
            "distinct_top_tokens": len(set(_norm(pieces[i]) for i in rows)),
        }

    # Union of the top-of-each-ranking, capped, so the judge sees the candidates each lens surfaces
    candidates = []
    seen = set()
    for ranking, tag in ((rank_coh, "coh"), (rank_density, "density"), (rank_spread, "spread")):
        for j in ranking[:20]:
            if j not in seen:
                seen.add(j)
                candidates.append((j, tag))
    feats_out = []
    for j, tag in candidates:
        rec = feature_record(j)
        rec["surfaced_by"] = tag
        feats_out.append(rec)
    print(f"\nassembled {len(feats_out)} candidate features with contexts "
          f"(union of top-20 by coherence / density / activation-spread)", flush=True)

    # PCA axes with contexts too (contrast)
    pca_out = []
    for j in range(16):
        order_j = np.argsort(pca_proj[:, j])[::-1][:10]
        pca_out.append({
            "axis": int(j), "token_coherence": float(pca_coh[j]), "modal_token": pca_modal[j],
            "var_pct": float(var[j]),
            "top_tokens": [pieces[i] for i in order_j],
            "top_contexts": [reconstruct_context(pieces, int(i)) for i in order_j],
            "distinct_top_tokens": len(set(_norm(pieces[int(i)]) for i in order_j)),
        })

    with open(CTX_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "model": "Qwen2.5-0.5B-Instruct q8_0", "tap_layer": layer, "rows": N,
            "sae": {"exp": EXP, "m": m, "l1": L1, "mse": float(mse),
                    "token_coherence_live": float(coh[live_mask].mean()), "n_live": n_live},
            "pca_token_coherence_top256": float(pca_coh.mean()),
            "features": feats_out, "pca_axes": pca_out,
        }, f, indent=2, ensure_ascii=False)
    print(f"wrote {CTX_JSON}  ({len(feats_out)} SAE features + {len(pca_out)} PCA axes, with context)",
          flush=True)

    # ---- (B) concept-alignment (quantitative semantic metric) ------------------------------------
    print(f"\n=== concept alignment: SAE features vs PCA axes on semantic minimal pairs ===",
          flush=True)
    align = concept_alignment(sae, mu, sd, Vt, K, layer, no_engine=no_engine)
    if align:
        # A win on the SINGLE-UNIT metric must clear the null AND beat PCA's single-unit number.
        sae_unit_wins = sum(1 for r in align
                            if r["sae_best_auc_heldout"] > r["sae_best_auc_null"] + 0.10
                            and r["sae_best_auc_heldout"] > r["pca_best_auc_heldout"] + 0.03)
        # A win on the WHOLE-REPRESENTATION probe.
        sae_probe_wins = sum(1 for r in align
                             if r["sae_whole_probe_acc"] > r["pca_whole_probe_acc"] + 0.03)
        pca_probe_wins = sum(1 for r in align
                             if r["pca_whole_probe_acc"] > r["sae_whole_probe_acc"] + 0.03)
        print(f"\nconcept-alignment summary:", flush=True)
        print(f"  single-unit (held-out, beats null+PCA): SAE wins {sae_unit_wins}/{len(align)}",
              flush=True)
        print(f"  whole-representation probe: SAE wins {sae_probe_wins}/{len(align)}, "
              f"PCA wins {pca_probe_wins}/{len(align)}, "
              f"ties {len(align)-sae_probe_wins-pca_probe_wins}", flush=True)
        with open(ALIGN_JSON, "w", encoding="utf-8") as f:
            json.dump({"concepts": align,
                       "sae_unit_wins_over_null_and_pca": sae_unit_wins,
                       "sae_whole_probe_wins": sae_probe_wins, "pca_whole_probe_wins": pca_probe_wins,
                       "note": ("single-unit AUC = held-out k-fold rank separation of the BEST unit "
                                "(picked on train folds) across matched-frame minimal pairs; null = "
                                "same procedure on permuted labels (what max-over-14k-features scores "
                                "by chance on 2n~24-48 sentences). SAE features require >=8 firers. "
                                "whole-repr probe = standardized k-fold linear probe (kfold_accuracy) "
                                "on the FULL code vector vs FULL PCA projection — the fair "
                                "no-cherry-pick comparison.")}, f, indent=2)
        print(f"wrote {ALIGN_JSON}", flush=True)
    else:
        print("(skipped concept-alignment harvest: --no-engine)", flush=True)

    print("\nDONE. Auto-interp judging is performed by the caller on "
          f"{os.path.basename(CTX_JSON)}; verdicts + concept numbers go into "
          "research/sae_at_scale_findings.md.", flush=True)


if __name__ == "__main__":
    main()
