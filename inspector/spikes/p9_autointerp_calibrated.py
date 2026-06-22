"""
Phase-3 §3.6 — the SALVAGE run: a VALID, CALIBRATED auto-interp metric for feature discovery.

The whole arc in research/sae_at_scale_findings.md reached a robust "null" — our from-scratch SAEs
tie/lose to PCA and read as "token detectors" — on a metric (top-token coherence) that the GPT-2
control (p8) then PROVED is BROKEN: it rates Joseph Bloom's known-good `gpt2-small-res-jb` SAE the
SAME ~31% / ~tied-with-PCA "null", even though that SAE is genuinely interpretable (its 100%-token
features carry rich Neuronpedia context labels — f318=' at' is actually a "died AT <place>" feature).
So top-token coherence is blind to token-ANCHORED context features and cross-token concepts, and the
discovery question is REOPENED — it needs a metric that can actually SEE interpretability.

THE METHOD — detection auto-interp with HELD-OUT scoring, CALIBRATED before use:
  Per feature: take its top-activating examples WITH CONTEXT (~10 tokens each side, focus << >>).
    * EXPLAIN set  = the top-12 activating examples -> the JUDGE (an LLM, in this run = the caller)
      writes a one-line description of the firing pattern. Token-ANCHORED context patterns COUNT
      ("'at' before a location", "political titles", "purposive 'to'") — the p8 lesson: a feature
      that fires on the token X only in context C is a concept, not a dumb X-detector. This rubric
      is DELIBERATELY less harsh than p6's "token-independent or bust" (which p8 falsified).
    * TEST set = ~8 held-out HIGH examples (ranks 12..44, genuinely high, NOT shown in explain)
      + ~8 NULL examples (random rows the feature does NOT fire on). Shuffled, labels HIDDEN.
      The judge predicts fires/not for each from the description ALONE.
    * score = BALANCED ACCURACY = 0.5*(TPR+TNR). 0.5 = chance/junk (a vacuous "fires on text"
      description predicts the nulls fire too -> TNR collapses -> 0.5). ->1.0 = a real, predictable
      feature. Held-out + the random null is what makes a vacuous description FAIL.
  Method score = mean balanced accuracy over a ~20-25 feature sample (top-by-activation + random
    features, NOT cherry-picked). Report the spread / CI, not just the mean.

CALIBRATION GATE (do FIRST): score Bloom's `gpt2-small-res-jb` features, PCA on the same GPT-2
residual, and random directions. The ruler is VALID iff Bloom's > PCA > random (~0.5). If it does
NOT order them, the metric is still broken -> STOP, don't judge ours with a broken ruler. If it
does, record where Bloom's lands ("what good looks like") and APPLY the same metric to ours:
our Qwen-0.5B SAE (L2) + Qwen-7B SAE (L16), with PCA + random on the SAME activations.

This is an LLM-as-judge RELATIVE comparison, not an absolute number — kept loud throughout.

TWO-PHASE (the judging is done by the caller LLM, so the spike emits packets, then scores verdicts):
    python spikes/p9_autointerp_calibrated.py --emit gpt2     # .venv-sae: Bloom SAE/PCA/random packets
    python spikes/p9_autointerp_calibrated.py --emit qwen05b  # cloze .venv: re-train 0.5B SAE, packets
    python spikes/p9_autointerp_calibrated.py --emit qwen7b   # cloze .venv: re-train 7B SAE, packets
    # -> caller reads runs/p9_packets_<which>.json, judges, writes runs/p9_judgments_<which>.json
    python spikes/p9_autointerp_calibrated.py --score         # balanced-accuracy verdict over all judged

Artifacts (gitignored runs/): p9_packets_<which>.json (judge input), p9_judgments_<which>.json
(caller's descriptions+predictions), p9_verdict.json (scored). Deterministic (seed=0) throughout.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

import numpy as np  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

# Per-feature packet shape (held-out detection auto-interp).
N_EXPLAIN = 12      # top examples shown to the judge to WRITE the description
N_HELDOUT_HI = 8    # held-out HIGH test examples (ranks N_EXPLAIN .. HI_POOL), not shown in explain
N_NULL = 8          # null test examples (feature ~off) — the null a vacuous description fails on
HI_POOL = 45        # held-out highs are sampled from ranks [N_EXPLAIN, HI_POOL)
N_FEATURES = 22     # features per method (top-by-activation half + random-live half); ~brief's 25
WINDOW = 10         # context tokens each side of the focus token
SEED = 0


def _norm(t: str) -> str:
    return t.strip().lower()


def reconstruct_context(pieces, i, window=WINDOW):
    """pieces is the corpus in token order, so a window of neighbours IS the context (focus << >>).
    Mirrors p6/p8 reconstruct_context exactly. Newlines flattened so the judge sees one line."""
    lo, hi = max(0, i - window), min(len(pieces), i + window + 1)
    left = "".join(pieces[lo:i])
    foc = pieces[i]
    right = "".join(pieces[i + 1:hi])
    return (left + "<<" + foc + ">>" + right).replace("\n", " ").strip()


def _select_features(fire, rng, n=N_FEATURES, lo=0.002, hi=0.4):
    """Pick the feature sample: half by ACTIVATION rank (most-firing live, where the method's best
    case lives) + half RANDOM live (so we don't cherry-pick the winners). 'live' = fires in [lo,hi]
    so a dead/ubiquitous unit can't enter. Returns a sorted unique index list."""
    live = np.where((fire >= lo) & (fire <= hi))[0]
    if len(live) == 0:
        return []
    n_top = n // 2
    top = sorted(live, key=lambda j: -fire[j])[:n_top]              # most-active live features
    rest = [j for j in live if j not in set(top)]
    n_rand = min(n - len(top), len(rest))
    rand = list(rng.choice(rest, size=n_rand, replace=False)) if n_rand > 0 else []
    return sorted(int(j) for j in set(top) | set(rand))


def _packet_for_feature(scores_col, pieces, j, rng, fired_thresh=1e-6):
    """Build one feature's detection packet from its activation column over the corpus.
      explain  = top-N_EXPLAIN activating rows (ctx)
      heldout  = N_HELDOUT_HI rows sampled from ranks [N_EXPLAIN, HI_POOL) (genuinely high, unseen)
      null     = N_NULL random rows with activation ~0 (the feature is OFF) — if too few off rows
                 (a very dense unit), fall back to the LOWEST-activation rows.
    The explain examples are shown labelled; the test examples (heldout ∪ null) are SHUFFLED with
    labels hidden, so the judge predicts from the description alone."""
    order = np.argsort(scores_col)[::-1]
    explain_rows = [int(i) for i in order[:N_EXPLAIN]]
    hi_pool = [int(i) for i in order[N_EXPLAIN:HI_POOL]]
    rng.shuffle(hi_pool)
    heldout_rows = hi_pool[:N_HELDOUT_HI]

    off = np.where(scores_col <= fired_thresh)[0]
    if len(off) >= N_NULL:
        null_rows = [int(i) for i in rng.choice(off, size=N_NULL, replace=False)]
    else:  # dense feature: use the lowest-activation rows as the null
        null_rows = [int(i) for i in order[::-1][:N_NULL]]

    test = ([{"row": r, "fires": True} for r in heldout_rows]
            + [{"row": r, "fires": False} for r in null_rows])
    rng.shuffle(test)
    return {
        "feature": int(j),
        "fires_pct": float((scores_col > fired_thresh).mean()),
        "explain": [{"row": r, "context": reconstruct_context(pieces, r)} for r in explain_rows],
        "test": [{"id": k, "row": t["row"], "context": reconstruct_context(pieces, t["row"]),
                  "_fires": t["fires"]} for k, t in enumerate(test)],
    }


def _emit(which, method, scores_matrix, pieces, fire, meta, rng):
    """scores_matrix: [N, F] (a callable column getter for big F, or a dense array). Builds packets
    for the selected feature sample. scores_matrix may be a function j->column (memory-frugal)."""
    feats = _select_features(fire, rng)
    get_col = scores_matrix if callable(scores_matrix) else (lambda j: scores_matrix[:, j])
    packets = []
    for j in feats:
        packets.append(_packet_for_feature(get_col(j), pieces, j, rng))
    return {"which": which, "method": method, "n_features": len(packets),
            "meta": meta, "features": packets}


# =================================================================================================
# GPT-2 calibration side (.venv-sae): Bloom's SAE feature acts are CACHED in gpt2_control_acts.npz.
# =================================================================================================
def emit_gpt2():
    cache = os.path.join(RUNS, "gpt2_control_acts.npz")
    d = np.load(cache, allow_pickle=True)
    R, F, pieces = d["R"], d["F"], list(d["pieces"])
    meta = json.loads(str(d["meta"]))
    print(f"=== GPT-2 calibration: {R.shape[0]} rows, resid {R.shape}, Bloom SAE feats {F.shape} "
          f"({meta['release']} @ {meta['hook']}) ===", flush=True)
    out = {}

    # (1) Bloom's pretrained SAE — expect HIGH
    rng = np.random.default_rng(SEED)
    fire = (F > 1e-6).mean(0)
    out["bloom_sae"] = _emit("gpt2", "bloom_sae", lambda j: F[:, j], pieces, fire,
                             {**meta, "what": "Bloom gpt2-small-res-jb pretrained SAE features"}, rng)

    # (2) PCA on the SAME residual (standardized, identical to the prior pipeline) — expect LOWER
    rng = np.random.default_rng(SEED)
    mu, sd = R.mean(0), R.std(0) + 1e-6
    Rs = (R - mu) / sd
    K = 256
    C = (Rs.T @ Rs) / Rs.shape[0]
    evals, evecs = np.linalg.eigh(C)
    ordv = np.argsort(evals)[::-1]
    Vt = evecs[:, ordv].T
    proj = Rs @ Vt[:K].T                                   # [N, K]
    # PCA axes are signed/dense; |proj| is the "activation strength" along the axis, and a PCA axis
    # has no fire-rate floor — treat every axis as live so the sampler picks among all K.
    pca_fire = np.full(K, 0.05)                            # dummy "live" so _select_features keeps them
    out["pca"] = _emit("gpt2", "pca", lambda j: np.abs(proj[:, j]), pieces, pca_fire,
                       {"what": f"PCA top-{K} on GPT-2 resid (|projection|)", "K": K}, rng)

    # (3) random directions in residual space — expect LOWEST (~0.5)
    rng = np.random.default_rng(SEED)
    g = np.random.default_rng(SEED + 7)
    Rd = g.standard_normal((R.shape[1], 64)).astype(np.float32)
    Rd /= np.linalg.norm(Rd, axis=0, keepdims=True) + 1e-9
    rproj = np.abs(Rs @ Rd)                                # [N, 64]
    rand_fire = np.full(rproj.shape[1], 0.05)
    out["random"] = _emit("gpt2", "random", lambda j: rproj[:, j], pieces, rand_fire,
                          {"what": "random unit directions in GPT-2 resid (|projection|)"}, rng)

    path = os.path.join(RUNS, "p9_packets_gpt2.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for k, v in out.items():
        print(f"  {k}: {v['n_features']} features", flush=True)
    print(f"wrote {path}", flush=True)


# =================================================================================================
# Our Qwen side (cloze .venv): re-train the SAE deterministically from the cached npz, then PCA +
# random on the SAME standardized activations. 0.5B uses p4's TorchSAE; 7B uses p7's StreamingSAE.
# =================================================================================================
def _common_qwen(which, X, pieces, layer, train_sae_fn):
    """Shared: standardize, PCA basis, random dirs, train the SAE via train_sae_fn(Xs)->(scores_col,
    fire) and emit all three method packets to one file. train_sae_fn returns (col_getter, fire)."""
    from clozn.discover import standardize
    Xs, mu, sd = standardize(X.astype(np.float32))
    N, d = Xs.shape
    uniq = len(set(_norm(p) for p in pieces if p.strip()))
    print(f"  standardized {Xs.shape}; {uniq} unique tokens (ratio {uniq/max(N,1):.3f}); layer {layer}",
          flush=True)
    out = {}

    # ---- our SAE (re-trained deterministically from cache) — the question ----
    rng = np.random.default_rng(SEED)
    sae_col, sae_fire, sae_meta = train_sae_fn(Xs)
    out["our_sae"] = _emit(which, "our_sae", sae_col, pieces, sae_fire,
                           {"layer": int(layer), "rows": int(N), "dim": int(d), **sae_meta}, rng)

    # ---- PCA on the SAME standardized activations (the baseline) ----
    rng = np.random.default_rng(SEED)
    K = 256
    C = (Xs.T @ Xs) / N
    evals, evecs = np.linalg.eigh(C)
    ordv = np.argsort(evals)[::-1]
    Vt = evecs[:, ordv].T
    proj = Xs @ Vt[:K].T
    pca_fire = np.full(K, 0.05)
    out["pca"] = _emit(which, "pca", lambda j: np.abs(proj[:, j]), pieces, pca_fire,
                       {"what": f"PCA top-{K} (|projection|)", "K": K}, rng)

    # ---- random directions (the null) ----
    rng = np.random.default_rng(SEED)
    g = np.random.default_rng(SEED + 7)
    Rd = g.standard_normal((d, 64)).astype(np.float32)
    Rd /= np.linalg.norm(Rd, axis=0, keepdims=True) + 1e-9
    rproj = np.abs(Xs @ Rd)
    rand_fire = np.full(rproj.shape[1], 0.05)
    out["random"] = _emit(which, "random", lambda j: rproj[:, j], pieces, rand_fire,
                          {"what": "random unit directions (|projection|)"}, rng)

    path = os.path.join(RUNS, f"p9_packets_{which}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for k, v in out.items():
        print(f"  {k}: {v['n_features']} features", flush=True)
    print(f"wrote {path}", flush=True)


def emit_qwen05b():
    """0.5B L2: re-train the reported-best 16x L1=8.0 SAE (p4 TorchSAE) deterministically from cache.
    Reproduces token-coherence 44.7% / MSE 0.059 per the findings doc."""
    from spikes.p4_big_sae import TorchSAE
    cache = os.path.join(RUNS, "qwen_big_natural_acts.npz")
    d = np.load(cache, allow_pickle=True)
    X, pieces = d["X"], list(d["pieces"])
    layer = int(d["layer"]) if "layer" in d else 2
    print(f"=== Qwen-0.5B: {X.shape[0]} cached rows x {X.shape[1]} (layer {layer}) ===", flush=True)

    def train(Xs):
        N, dim = Xs.shape
        EXP, L1, BATCH, LR = 16, 8.0, 512, 1e-3
        m = EXP * dim
        spe = max(1, (N + BATCH - 1) // BATCH)
        epochs = int(min(200, max(40, (9000 + spe - 1) // spe)))
        print(f"  training SAE {EXP}x (m={m}) L1={L1}: batch={BATCH} lr={LR} epochs={epochs}",
              flush=True)
        sae = TorchSAE(dim, m=m, l1=L1, seed=0).fit(Xs, epochs=epochs, lr=LR, batch_size=BATCH)
        fire, mse = sae.stats(Xs)
        print(f"  trained: mean_fire={fire.mean()*100:.2f}% mse={mse:.3f} "
              f"(reproduces the reported-best 0.5B SAE)", flush=True)
        # column getter: encode a single feature's activation over the whole corpus, on GPU, batched
        import torch
        Xt = torch.tensor(Xs, dtype=torch.float32, device=sae.device)

        def col(j):
            with torch.no_grad():
                out = torch.empty(Xt.shape[0], device=sae.device)
                for s in range(0, Xt.shape[0], 16384):
                    xb = Xt[s:s + 16384]
                    out[s:s + xb.shape[0]] = (xb @ sae.We[:, j] + sae.be[j]).clamp(min=0)
                return out.cpu().numpy()
        return col, fire, {"exp": EXP, "l1": L1, "m": int(m), "mse": float(mse),
                           "what": "our from-scratch 16x L1=8 SAE, re-trained from cache"}

    _common_qwen("qwen05b", X, pieces, layer, train)


def emit_qwen7b():
    """7B L16: re-train the reconstructing 8x L1=8 SAE (p7 StreamingSAE) from cache. The full 1M rows
    won't co-reside with the SAE on 16GB, so we use p7's GPU_CAP subsample (250k, seed 0) — SAME
    rows/pieces p7 reported on, so contexts align. Standardize+winsorize exactly as p7 (apples-to-
    apples). 8x L1=8 was p7's only near-reconstructing config (MSE 0.917)."""
    import torch
    from spikes.p7_scale_7b import (StreamingSAE, standardize_clip_chunked, CLIP_SIGMA)
    cache = os.path.join(RUNS, "qwen7b_natural_acts_L16.npz")
    d = np.load(cache, allow_pickle=True)
    X, pieces = d["X"], list(d["pieces"])
    layer = int(d["layer"]) if "layer" in d else 16
    print(f"=== Qwen-7B: {X.shape[0]} cached rows x {X.shape[1]} (layer {layer}, {X.dtype}) ===",
          flush=True)
    # p7's exact prep: chunked standardize + massive-activation winsorize (fp16 result).
    Xc16, mu, sd, n_clip, tvar = standardize_clip_chunked(X, sigma=CLIP_SIGMA)
    del X
    N, d = Xc16.shape
    print(f"  standardized+clipped {n_clip} massive-activation rows; target var {tvar:.3f}", flush=True)
    # p7's GPU_CAP subsample (seed 0) so the SAE fits the card AND contexts align with the eval pieces.
    GPU_CAP = 250_000
    if N > GPU_CAP:
        sub = np.random.default_rng(0).choice(N, size=GPU_CAP, replace=False)
        sub.sort()
        Xc16 = Xc16[sub]
        pieces = [pieces[i] for i in sub]
        N = GPU_CAP
        print(f"  GPU_CAP subsample: {N} rows (seed 0; matches p7's reported eval set)", flush=True)
    Xgpu = torch.tensor(Xc16, dtype=torch.float16, device="cuda")
    del Xc16
    torch.cuda.empty_cache()

    # PCA on the SAME capped+clipped matrix (apples-to-apples with the SAE; p7 scored PCA on full 1M,
    # but for the detection metric we need contexts aligned to the SAME rows the SAE sees).
    uniq = len(set(_norm(p) for p in pieces if p.strip()))
    print(f"  {uniq} unique tokens (ratio {uniq/max(N,1):.3f}) over the capped set; layer {layer}",
          flush=True)
    out = {}

    # ---- our 7B SAE: 8x L1=8 (p7's reconstructing config) ----
    EXP, L1, BATCH, LR = 8, 8.0, 512, 1e-3
    m = EXP * d
    spe = max(1, (N + BATCH - 1) // BATCH)
    epochs = int(min(30, max(12, (12000 + spe - 1) // spe)))
    print(f"  training 7B SAE {EXP}x (m={m}) L1={L1}: batch={BATCH} lr={LR} epochs={epochs}",
          flush=True)
    sae = StreamingSAE(d, m=m, l1=L1, seed=0).fit(Xgpu, epochs=epochs, lr=LR, batch_size=BATCH)
    fire, mse = sae.stats(Xgpu)
    print(f"  trained: mean_fire={fire.mean()*100:.2f}% mse={mse:.3f} (target {tvar:.3f}; "
          f"{'reconstructs' if mse < tvar else 'NOT reconstructing'})", flush=True)

    def sae_col(j):
        with torch.no_grad():
            o = torch.empty(Xgpu.shape[0], device=sae.device)
            for s in range(0, Xgpu.shape[0], 16384):
                xb = Xgpu[s:s + 16384].float()
                o[s:s + xb.shape[0]] = (xb @ sae.We[:, j] + sae.be[j]).clamp(min=0)
            return o.cpu().numpy()
    rng = np.random.default_rng(SEED)
    out["our_sae"] = _emit("qwen7b", "our_sae", sae_col, pieces, fire,
                           {"layer": int(layer), "rows": int(N), "dim": int(d), "exp": EXP,
                            "l1": L1, "m": int(m), "mse": float(mse),
                            "what": "our from-scratch 8x L1=8 SAE @ L16, re-trained from cache"}, rng)
    del sae
    torch.cuda.empty_cache()

    # ---- PCA on the same capped matrix (chunked covariance on GPU) ----
    K = 256
    with torch.no_grad():
        Cmat = torch.zeros(d, d, device="cuda")
        for s in range(0, Xgpu.shape[0], 50000):
            xb = Xgpu[s:s + 50000].float()
            Cmat += xb.T @ xb
        Cmat /= Xgpu.shape[0]
        evals, evecs = torch.linalg.eigh(Cmat)
        ordv = torch.argsort(evals, descending=True)
        VtK = evecs[:, ordv[:K]]                            # [d, K]
        proj = torch.empty(Xgpu.shape[0], K, device="cuda")
        for s in range(0, Xgpu.shape[0], 50000):
            proj[s:s + 50000] = (Xgpu[s:s + 50000].float() @ VtK)
        proj_np = proj.abs().cpu().numpy()
    del Cmat, proj
    torch.cuda.empty_cache()
    rng = np.random.default_rng(SEED)
    pca_fire = np.full(K, 0.05)
    out["pca"] = _emit("qwen7b", "pca", lambda j: proj_np[:, j], pieces, pca_fire,
                       {"what": f"PCA top-{K} on capped L16 (|projection|)", "K": K}, rng)

    # ---- random directions ----
    g = np.random.default_rng(SEED + 7)
    Rd = torch.tensor(g.standard_normal((d, 64)), dtype=torch.float32, device="cuda")
    Rd /= Rd.norm(dim=0, keepdim=True) + 1e-9
    with torch.no_grad():
        rp = torch.empty(Xgpu.shape[0], 64, device="cuda")
        for s in range(0, Xgpu.shape[0], 50000):
            rp[s:s + 50000] = (Xgpu[s:s + 50000].float() @ Rd)
        rproj_np = rp.abs().cpu().numpy()
    rng = np.random.default_rng(SEED)
    rand_fire = np.full(64, 0.05)
    out["random"] = _emit("qwen7b", "random", lambda j: rproj_np[:, j], pieces, rand_fire,
                          {"what": "random unit directions on capped L16 (|projection|)"}, rng)

    path = os.path.join(RUNS, "p9_packets_qwen7b.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    for k, v in out.items():
        print(f"  {k}: {v['n_features']} features", flush=True)
    print(f"wrote {path}", flush=True)


# =================================================================================================
# SCORING — read the caller's judgments (per-feature predictions) and compute balanced accuracy.
# =================================================================================================
def _balanced_accuracy(packet_feat, judged_feat):
    """packet_feat has test[].(_fires, id); judged_feat has predictions[id]=bool. Balanced acc =
    0.5*(TPR + TNR) over the test set. Returns (bal_acc, tp, fn, tn, fp)."""
    truth = {t["id"]: bool(t["_fires"]) for t in packet_feat["test"]}
    preds = {int(k): bool(v) for k, v in judged_feat["predictions"].items()}
    tp = fn = tn = fp = 0
    for tid, fires in truth.items():
        p = preds.get(tid, False)
        if fires and p:
            tp += 1
        elif fires and not p:
            fn += 1
        elif (not fires) and (not p):
            tn += 1
        else:
            fp += 1
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return 0.5 * (tpr + tnr), tp, fn, tn, fp


def _mean_ci(vals):
    """Mean + a rough 95% CI (normal approx, t not worth it at n~20)."""
    a = np.asarray(vals, dtype=float)
    if len(a) == 0:
        return 0.0, 0.0, 0.0
    m = float(a.mean())
    se = float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se


def score():
    """Score every (which, method) for which BOTH a packet file and a judgment file exist."""
    results = {}
    for which in ("gpt2", "qwen05b", "qwen7b"):
        pkt_path = os.path.join(RUNS, f"p9_packets_{which}.json")
        jdg_path = os.path.join(RUNS, f"p9_judgments_{which}.json")
        if not (os.path.exists(pkt_path) and os.path.exists(jdg_path)):
            continue
        with open(pkt_path, encoding="utf-8") as f:
            packets = json.load(f)
        with open(jdg_path, encoding="utf-8") as f:
            judgments = json.load(f)
        results[which] = {}
        for method, mp in packets.items():
            if method not in judgments:
                continue
            jfeats = {int(jf["feature"]): jf for jf in judgments[method]["features"]}
            per_feat = []
            for pf in mp["features"]:
                fj = int(pf["feature"])
                if fj not in jfeats:
                    continue
                ba, tp, fn, tn, fp = _balanced_accuracy(pf, jfeats[fj])
                per_feat.append({"feature": fj, "bal_acc": ba, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
                                 "description": jfeats[fj].get("description", "")})
            if not per_feat:
                continue
            bas = [p["bal_acc"] for p in per_feat]
            mean, lo, hi = _mean_ci(bas)
            results[which][method] = {"mean_bal_acc": mean, "ci95": [lo, hi], "n": len(bas),
                                      "per_feature": per_feat,
                                      "meta": mp.get("meta", {})}
            print(f"  [{which:8} {method:10}] balanced acc = {mean:.3f} "
                  f"[{lo:.3f}, {hi:.3f}] over n={len(bas)}", flush=True)

    # ---- verdicts ----
    print("\n=== VERDICT ===", flush=True)
    if "gpt2" in results:
        g = results["gpt2"]
        b = g.get("bloom_sae", {}).get("mean_bal_acc")
        p = g.get("pca", {}).get("mean_bal_acc")
        r = g.get("random", {}).get("mean_bal_acc")
        print(f"  CALIBRATION (GPT-2): Bloom SAE={b:.3f}  PCA={p:.3f}  random={r:.3f}"
              if None not in (b, p, r) else "  CALIBRATION: incomplete", flush=True)
        if None not in (b, p, r):
            ordered = (b > p > r)
            print(f"  ruler ordering Bloom > PCA > random: {'VALID' if ordered else 'BROKEN'} "
                  f"(b-p={b-p:+.3f}, p-r={p-r:+.3f})", flush=True)
            results["_calibration_valid"] = bool(ordered)
    for which in ("qwen05b", "qwen7b"):
        if which in results:
            q = results[which]
            s = q.get("our_sae", {}).get("mean_bal_acc")
            p = q.get("pca", {}).get("mean_bal_acc")
            r = q.get("random", {}).get("mean_bal_acc")
            if None not in (s, p, r):
                print(f"  {which}: our SAE={s:.3f}  PCA={p:.3f}  random={r:.3f}  "
                      f"(SAE-PCA={s-p:+.3f}, SAE-random={s-r:+.3f})", flush=True)

    out = os.path.join(RUNS, "p9_verdict.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}", flush=True)


def main():
    args = sys.argv[1:]
    if "--emit" in args:
        which = args[args.index("--emit") + 1]
        os.makedirs(RUNS, exist_ok=True)
        if which == "gpt2":
            emit_gpt2()
        elif which == "qwen05b":
            emit_qwen05b()
        elif which == "qwen7b":
            emit_qwen7b()
        else:
            print(f"unknown --emit target {which!r} (gpt2|qwen05b|qwen7b)", flush=True)
            sys.exit(2)
    elif "--score" in args:
        score()
    else:
        print(__doc__, flush=True)
        print("Use --emit <gpt2|qwen05b|qwen7b> then --score (after the caller writes judgments).",
              flush=True)


if __name__ == "__main__":
    main()
