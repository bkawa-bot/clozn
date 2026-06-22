"""
Phase-3 §3.6 — the GPT-2-small PRETRAINED-SAE CONTROL: was our SAE/transcoder null a TRAINING
problem, a METRIC problem, or a SIZE/"local" problem?

The whole arc in research/sae_at_scale_findings.md is a robust null: our from-scratch SAEs/transcoders
tie or LOSE to PCA and discover token-identity detectors, across 0.5B -> 7B, layers (2/12/16),
token counts (5k -> 1M), expansion (1x -> 16x), AND a semantic metric. Every section bottoms out at
the same caveat: those SAEs were OUR training (first-gen ReLU/L1, a from-scratch dictionary on a
consumer GPU with <=1M tokens). A KNOWN-GOOD *pretrained* SAE on a *smaller* model decides among the
three remaining hypotheses:

  * GPT-2-small is 124M -- SMALLER than our 0.5B. So a clean result here RULES OUT "size/local."
  * gpt2-small-res-jb (Joseph Bloom) is the most-studied public residual SAE in existence; every
    feature has a Neuronpedia auto-interp label. So "is it genuinely interpretable" has a public
    ground truth we can cross-check, not just our own judgment.

The fork:
  (A) If the pretrained SAE shows clean interpretable features AND scores WELL on OUR metric (and vs
      PCA) -> the gap was our TRAINING, not size/local and not the metric.
  (B) If the pretrained SAE is clearly interpretable but scores POORLY on our metric (~ties/loses to
      PCA, like ours did) -> our METRIC was the confound: it can't see the interpretability a known
      SAE has, so it could never have rated OUR SAEs fairly either.
  (C) If it scores poorly AND reads as token detectors -> the null is about substrate/scale, the
      strongest reading -- but this is the LEAST likely given the published evidence on this SAE.

Either of (A)/(B) is decisive and changes what "why aren't ours working" means.

The metric is PORTED VERBATIM from p4_big_sae.top_token_coherence (apples-to-apples with every prior
row): for each feature take its top-20 activating token positions across the corpus, normalize the
token string (strip+lower), coherence = fraction equal to the modal token. Applied identically to
(a) the pretrained SAE's feature activations and (b) PCA on the SAME residual at matched K (256, and
top-64). PCA standardized per-feature exactly as the prior pipeline does.

ISOLATED ENV: this runs in C:\\Users\\brigi\\src\\clozn\\.venv-sae (separate from the lab venv, which
has torch/transformers pinned for the goldens). sae_lens pulls transformer_lens + torch + transformers.
CPU torch is fine -- GPT-2-small (124M) is fast on CPU.

Usage (from inspector/, .venv-sae python):
    python spikes/p8_gpt2_control.py                 # full: load SAE, harvest, score, save
    python spikes/p8_gpt2_control.py --from-cache    # re-analyze the cached activation matrix
    python spikes/p8_gpt2_control.py --layer 7       # blocks.7.hook_resid_pre instead of 8
    python spikes/p8_gpt2_control.py --sentences 400 # corpus size (passages)
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
CACHE = os.path.join(RUNS, "gpt2_control_acts.npz")

# The canonical Joseph-Bloom residual SAE release + a MID layer (GPT-2-small has 12 blocks).
RELEASE = "gpt2-small-res-jb"
DEFAULT_LAYER = 8  # blocks.8.hook_resid_pre (mid network); --layer 7 also canonical


# ---- coherence metric: PORTED VERBATIM from spikes/p4_big_sae.py (do NOT diverge) ---------------
def _norm(t: str) -> str:
    return t.strip().lower()


def top_token_coherence(scores_per_feature, pieces, topn=20):
    """scores_per_feature: [N, F]. For each feature take its top-N activating rows; coherence =
    fraction equal to the feature's MODAL top token. Returns (coh[F], modal[F], tops[F]).
    This is byte-for-byte the metric used on every prior row in sae_at_scale_findings.md."""
    N, F = scores_per_feature.shape
    coh = np.zeros(F)
    modal = [""] * F
    tops = [[] for _ in range(F)]
    norm_pieces = [_norm(p) for p in pieces]
    for j in range(F):
        order = np.argsort(scores_per_feature[:, j])[::-1][:topn]
        toks = [norm_pieces[i] for i in order]
        tops[j] = [pieces[i] for i in order]
        cnt = collections.Counter(t for t in toks if t != "")
        if cnt:
            tok, c = cnt.most_common(1)[0]
            modal[j] = tok
            coh[j] = c / len(toks)
    return coh, modal, tops


def standardize(X):
    """Per-feature standardize -- identical to clozn.discover.standardize (the prior PCA input)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    return (X - mu) / sd, mu, sd


def reconstruct_context(pieces, i, window=14):
    """pieces is the corpus in token order, so a window of neighbours IS the context. Mark the focus
    token with << >>. Mirrors p6_autointerp.reconstruct_context."""
    lo, hi = max(0, i - window), min(len(pieces), i + window + 1)
    left = "".join(pieces[lo:i])
    foc = pieces[i]
    right = "".join(pieces[i + 1:hi])
    return (left + "<<" + foc + ">>" + right).replace("\n", " ").strip()


# ---- corpus: WikiText-103, the SAME source the prior runs used (clozn.corpora.text_stream) -------
def wikitext_passages(max_chars: int = 800, limit: int | None = None):
    """Concatenate WikiText-103 lines into ~max_chars passages (same shaping as p4_big_sae). Streams
    from HF datasets -- no full download. `limit` caps the number of passages yielded."""
    try:
        from clozn.corpora import text_stream
        src = text_stream(source="wikitext", min_len=60)
    except Exception as e:  # noqa: BLE001
        print(f"  (clozn.corpora unavailable: {type(e).__name__}: {e}; using HF datasets directly)",
              flush=True)
        import datasets
        ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                                   split="train", streaming=True)

        def _src():
            for x in ds:
                t = (x.get("text") or "").strip()
                if len(t) >= 60 and not t.startswith("="):
                    yield t
        src = _src()
    buf = ""
    n = 0
    for t in src:
        t = t.strip()
        if not t:
            continue
        if len(buf) + len(t) + 1 <= max_chars:
            buf = (buf + " " + t) if buf else t
        else:
            if buf:
                yield buf
                n += 1
                if limit and n >= limit:
                    return
            buf = t if len(t) <= max_chars else t[:max_chars]
    if buf and not (limit and n >= limit):
        yield buf


# ---- harvest residual + SAE features at the SAE's hook point, via transformer_lens --------------
def harvest(layer: int, n_passages: int, max_ctx: int = 256):
    """Load GPT-2-small + the pretrained gpt2-small-res-jb SAE at blocks.{layer}.hook_resid_pre.
    Run a corpus through with cache; for each token position collect:
      - resid:   the residual activation at the hook (PCA's input + the SAE's input)
      - feats:   the SAE's feature activations at that position (relu codes)
      - piece:   the decoded token string (the unit the coherence metric counts)
    Returns (resid[N, d_model], feats[N, d_sae], pieces[list[str]], meta dict)."""
    import torch
    from sae_lens import SAE, HookedSAETransformer

    hook = f"blocks.{layer}.hook_resid_pre"
    print(f"=== loading GPT-2-small + SAE {RELEASE} @ {hook} ===", flush=True)
    t0 = time.time()
    model = HookedSAETransformer.from_pretrained("gpt2")
    model.eval()
    # sae_lens API has shifted across versions; try the common signatures in turn.
    sae = _load_sae(SAE, RELEASE, hook)
    d_model = int(sae.cfg.d_in)
    d_sae = int(sae.cfg.d_sae)
    # sae_lens 6.x stores the hook under cfg.metadata['hook_name'] (StandardSAEConfig has no
    # top-level .hook_name); fall back to the hook we requested.
    cfg_hook = _cfg_hook_name(sae.cfg) or hook
    print(f"  loaded in {time.time()-t0:.0f}s; d_model={d_model}, d_sae={d_sae} "
          f"({d_sae // d_model}x expansion); hook={cfg_hook}", flush=True)

    tok = model.tokenizer
    resid_rows: list[np.ndarray] = []
    feat_rows: list[np.ndarray] = []
    pieces: list[str] = []
    t0 = time.time()
    n_done = 0
    with torch.no_grad():
        for text in wikitext_passages(limit=n_passages):
            ids = model.to_tokens(text, prepend_bos=True)  # [1, T]
            if ids.shape[1] > max_ctx:
                ids = ids[:, :max_ctx]
            _, cache = model.run_with_cache(ids, names_filter=hook, stop_at_layer=layer + 1)
            resid = cache[hook][0]                         # [T, d_model]
            feats = sae.encode(resid)                      # [T, d_sae] (relu feature acts)
            resid_np = resid.float().cpu().numpy()
            feats_np = feats.float().cpu().numpy()
            ids_list = ids[0].tolist()
            # token strings: skip BOS (position 0) so pieces align with real text tokens
            for p in range(resid_np.shape[0]):
                piece = tok.decode([ids_list[p]])
                if p == 0 and ids_list[p] == tok.bos_token_id:
                    continue  # BOS carries no lexical content; excluding it matches a text corpus
                resid_rows.append(resid_np[p])
                feat_rows.append(feats_np[p])
                pieces.append(piece)
            n_done += 1
            if n_done % 50 == 0:
                rate = len(pieces) / max(time.time() - t0, 1e-6)
                print(f"  passage {n_done}: {len(pieces)} rows ({rate:.0f} rows/s)", flush=True)
    R = np.stack(resid_rows).astype(np.float32)
    F = np.stack(feat_rows).astype(np.float32)
    meta = {"layer": layer, "hook": hook, "d_model": d_model, "d_sae": d_sae,
            "release": RELEASE, "n_passages": n_done}
    print(f"harvested {R.shape[0]} rows: resid {R.shape}, feats {F.shape} from {n_done} passages "
          f"in {time.time()-t0:.0f}s", flush=True)
    return R, F, pieces, meta


def _cfg_hook_name(cfg):
    """sae_lens version-robust hook-name lookup: 6.x puts it in cfg.metadata['hook_name']; older
    builds expose cfg.hook_name directly. Returns the hook string or None."""
    md = getattr(cfg, "metadata", None)
    if md is not None:
        try:
            return md["hook_name"]
        except Exception:  # noqa: BLE001
            return getattr(md, "hook_name", None)
    return getattr(cfg, "hook_name", None)


def _load_sae(SAE, release, hook):
    """sae_lens.SAE.from_pretrained signature has changed across releases; some return (sae, cfg,
    sparsity) tuples, some return just the sae, and the positional args differ. Try the known forms."""
    errs = []
    for attempt in (
        lambda: SAE.from_pretrained(release, hook),
        lambda: SAE.from_pretrained(release=release, sae_id=hook),
        lambda: SAE.from_pretrained_with_cfg_and_sparsity(release, hook),
    ):
        try:
            r = attempt()
            sae = r[0] if isinstance(r, tuple) else r
            sae.eval() if hasattr(sae, "eval") else None
            return sae
        except Exception as e:  # noqa: BLE001
            errs.append(f"{type(e).__name__}: {e}")
    raise RuntimeError("could not load SAE via any known sae_lens signature:\n  " + "\n  ".join(errs))


# ---- optional: pull Neuronpedia auto-interp labels for the named features ------------------------
def neuronpedia_label(layer: int, feature_idx: int, timeout: float = 8.0) -> str | None:
    """gpt2-small-res-jb features are public on Neuronpedia. The model/layer id for this release is
    'gpt2-small/{layer}-res-jb'. Returns the short auto-interp description if reachable, else None
    (network-optional: a failure must NOT block the verdict)."""
    url = f"https://www.neuronpedia.org/api/feature/gpt2-small/{layer}-res-jb/{feature_idx}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "clozn-control/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        exps = data.get("explanations") or []
        if exps:
            return (exps[0].get("description") or "").strip()
        return ""
    except Exception:  # noqa: BLE001
        return None


def main():
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    layer = DEFAULT_LAYER
    if "--layer" in sys.argv:
        layer = int(sys.argv[sys.argv.index("--layer") + 1])
    n_passages = 400
    if "--sentences" in sys.argv:
        n_passages = int(sys.argv[sys.argv.index("--sentences") + 1])
    no_np = "--no-neuronpedia" in flags
    os.makedirs(RUNS, exist_ok=True)

    if "--from-cache" in flags and os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        R, F, pieces = d["R"], d["F"], list(d["pieces"])
        meta = json.loads(str(d["meta"]))
        layer = meta["layer"]
        print(f"=== loaded cached {R.shape[0]} rows: resid {R.shape}, feats {F.shape} "
              f"(layer {layer}, {meta['release']}) ===", flush=True)
    else:
        R, F, pieces, meta = harvest(layer, n_passages)
        np.savez_compressed(CACHE, R=R, F=F, pieces=np.array(pieces, dtype=object),
                            meta=json.dumps(meta))
        print(f"saved {CACHE}", flush=True)

    d_model, d_sae = meta["d_model"], meta["d_sae"]
    N = R.shape[0]
    uniq = len(set(_norm(p) for p in pieces if p.strip()))
    print(f"\ncorpus: {N} rows, d_model={d_model}, d_sae={d_sae}, {uniq} unique tokens "
          f"(unique ratio {uniq / max(N,1):.3f})", flush=True)

    # ---- (a) PRETRAINED SAE feature coherence on OUR metric -------------------------------------
    # The SAE feature acts are already non-negative (relu). Score every feature, then restrict to the
    # SAME live band as the prior runs (fire in [0.002, 0.4]) so a dead/ubiquitous feature can't game it.
    print(f"\n=== (a) pretrained SAE feature coherence (OUR metric, top-20) ===", flush=True)
    t0 = time.time()
    sae_coh, sae_modal, sae_tops = top_token_coherence(F, pieces, topn=20)
    fire = (F > 1e-6).mean(0)
    live_mask = (fire >= 0.002) & (fire <= 0.4)
    n_live = int(live_mask.sum())
    n_dead = int((fire < 0.002).sum())
    sae_mean_live = float(sae_coh[live_mask].mean()) if n_live else 0.0
    sae_mean_all = float(sae_coh.mean())
    # also report the mean over the most-coherent K-matched-to-PCA-top256 live features (a fair
    # head-to-head: PCA reports its top-256 axes; the SAE has thousands of live features, so "all
    # live" is its honest dictionary-wide number, but we also show its best-256 for context).
    live_idx_all = np.where(live_mask)[0]
    top256_live = sorted(live_idx_all, key=lambda j: -sae_coh[j])[:256]
    sae_mean_best256 = float(np.mean([sae_coh[j] for j in top256_live])) if top256_live else 0.0
    print(f"  SAE live features: {n_live}/{d_sae} (dead {n_dead}); "
          f"mean coherence (live) = {sae_mean_live*100:.1f}%  "
          f"(all feats {sae_mean_all*100:.1f}%; best-256 live {sae_mean_best256*100:.1f}%)  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- (b) PCA on the SAME residual at matched K (standardized; identical to prior pipeline) ---
    K = 256
    print(f"\n=== (b) PCA baseline on the same residual (top-{K} axes, standardized) ===", flush=True)
    t0 = time.time()
    Rs, _, _ = standardize(R)
    if N > 20000:
        C = (Rs.T @ Rs) / N
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        Vt = evecs[:, order].T
        var = evals[order] / evals.sum()
    else:
        U, S, Vt = np.linalg.svd(Rs, full_matrices=False)
        var = (S ** 2) / (S ** 2).sum()
    pca_proj = Rs @ Vt[:K].T
    pca_coh, pca_modal, pca_tops = top_token_coherence(pca_proj, pieces, topn=20)
    pca_mean = float(pca_coh.mean())
    pca_mean64 = float(pca_coh[:64].mean())
    print(f"  PCA top-{K} mean coherence = {pca_mean*100:.1f}%  (top-64 = {pca_mean64*100:.1f}%)  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- verdict inputs --------------------------------------------------------------------------
    print(f"\n=== verdict inputs ===", flush=True)
    print(f"  PRETRAINED SAE (live mean / best-256): {sae_mean_live*100:.1f}% / "
          f"{sae_mean_best256*100:.1f}%", flush=True)
    print(f"  PCA (top-256 / top-64): {pca_mean*100:.1f}% / {pca_mean64*100:.1f}%", flush=True)
    gap = sae_mean_live - pca_mean
    gap_best = sae_mean_best256 - pca_mean
    print(f"  SAE(live) - PCA(top256) gap: {gap*100:+.1f} pts ; "
          f"SAE(best256) - PCA(top256) gap: {gap_best*100:+.1f} pts", flush=True)

    # ---- top ~15 SAE features WITH CONTEXT + Neuronpedia labels ----------------------------------
    ranked = sorted(np.where(live_mask)[0], key=lambda j: -sae_coh[j])
    n_show = 15
    print(f"\n=== {n_show} most-coherent live SAE features ({RELEASE} @ {meta['hook']}) ===",
          flush=True)
    examples = []
    for rank, j in enumerate(ranked[:n_show]):
        # top positions for context (find the top-rows for this feature)
        order = np.argsort(F[:, j])[::-1][:10]
        contexts = [reconstruct_context(pieces, int(i)) for i in order]
        np_label = None if no_np else neuronpedia_label(layer, int(j))
        lbl = f"  NP:{np_label!r}" if np_label else (""
                                                     if np_label is None else "  NP:(no label)")
        print(f"  f{int(j):<6} coh={sae_coh[j]*100:3.0f}% fires={fire[j]*100:5.2f}% "
              f"modal={sae_modal[j]!r:<12} top: {' '.join(repr(t) for t in sae_tops[j][:8])}{lbl}",
              flush=True)
        examples.append({
            "feature": int(j), "coherence": float(sae_coh[j]), "fires_pct": float(fire[j]),
            "modal_token": sae_modal[j], "top_tokens": sae_tops[j][:10],
            "distinct_top_tokens": len(set(_norm(t) for t in sae_tops[j][:20])),
            "top_contexts": contexts, "neuronpedia_label": np_label,
        })

    # Also surface the most-COHERENT-INDEPENDENT lens: high-fire features (where a CONCEPT, which
    # fires across many tokens, would hide and thus score LOW on the token metric). This is the
    # check that decides hypothesis (B): a known SAE feature that is clearly a concept but scores low.
    spread_ranked = sorted(np.where(live_mask)[0], key=lambda j: -fire[j])
    print(f"\n=== 8 highest-firing live SAE features (where a cross-token CONCEPT would hide) ===",
          flush=True)
    spread_examples = []
    for j in spread_ranked[:8]:
        order = np.argsort(F[:, j])[::-1][:10]
        contexts = [reconstruct_context(pieces, int(i)) for i in order]
        np_label = None if no_np else neuronpedia_label(layer, int(j))
        print(f"  f{int(j):<6} coh={sae_coh[j]*100:3.0f}% fires={fire[j]*100:5.2f}% "
              f"distinct_top={len(set(_norm(t) for t in sae_tops[j][:20]))} "
              f"top: {' '.join(repr(t) for t in sae_tops[j][:8])}"
              f"{('  NP:'+repr(np_label)) if np_label else ''}", flush=True)
        spread_examples.append({
            "feature": int(j), "coherence": float(sae_coh[j]), "fires_pct": float(fire[j]),
            "distinct_top_tokens": len(set(_norm(t) for t in sae_tops[j][:20])),
            "top_tokens": sae_tops[j][:10], "top_contexts": contexts,
            "neuronpedia_label": np_label,
        })

    print("\n=== 8 PCA axes for contrast ===", flush=True)
    pca_examples = []
    for j in range(8):
        order = np.argsort(pca_proj[:, j])[::-1][:10]
        contexts = [reconstruct_context(pieces, int(i)) for i in order]
        print(f"  PC{j:<3} coh={pca_coh[j]*100:3.0f}% var={var[j]*100:4.1f}% "
              f"modal={pca_modal[j]!r:<12} top: {' '.join(repr(t) for t in pca_tops[j][:8])}",
              flush=True)
        pca_examples.append({"axis": int(j), "coherence": float(pca_coh[j]),
                             "var_pct": float(var[j]), "modal_token": pca_modal[j],
                             "top_tokens": pca_tops[j][:10], "top_contexts": contexts})

    # ---- save artifacts --------------------------------------------------------------------------
    summary = {
        "model": "gpt2-small (124M)", "release": RELEASE, "hook": meta["hook"], "layer": layer,
        "d_model": d_model, "d_sae": d_sae, "expansion": d_sae // d_model,
        "rows": int(N), "n_passages": meta["n_passages"], "unique_token_ratio": uniq / max(N, 1),
        "metric": "top-token coherence (top-20), ported verbatim from p4_big_sae",
        "sae_mean_coherence_live": sae_mean_live, "sae_n_live": n_live,
        "sae_mean_coherence_all": sae_mean_all, "sae_mean_coherence_best256_live": sae_mean_best256,
        "pca_K": K, "pca_mean_coherence": pca_mean, "pca_mean_coherence_top64": pca_mean64,
        "sae_live_minus_pca_top256_gap": gap, "sae_best256_minus_pca_top256_gap": gap_best,
        "examples_sae_top_coherent": examples,
        "examples_sae_high_firing": spread_examples,
        "examples_pca": pca_examples,
        "prior_runs": {
            "toy_rwkv_seeded": {"sae": 0.65, "pca": 0.12},
            "qwen05b_L2_bigSAE": {"sae": 0.447, "pca_top256": 0.415, "pca_top64": 0.548},
            "qwen7b_L16_bigSAE": {"sae": 0.19, "pca_top256": 0.643, "pca_top64": 0.705},
        },
    }
    out_json = os.path.join(RUNS, "gpt2_control.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nwrote {out_json}", flush=True)

    # HTML viz (reuse the shared renderer if importable; else skip silently -- JSON is the artifact)
    try:
        from clozn.discover import Feature
        from clozn.viz import render_discovered_features
        feats = [Feature(int(e["feature"]), "sae", e["top_tokens"][:10], e["fires_pct"])
                 for e in examples]
        out_html = os.path.join(RUNS, "gpt2_control.html")
        with open(out_html, "w", encoding="utf-8") as f:
            f.write(render_discovered_features(
                feats, title=f"Clozn · GPT-2-small pretrained-SAE control ({RELEASE})",
                subtitle=(f"{N} WikiText tokens @ {meta['hook']} · pretrained SAE "
                          f"{d_sae // d_model}x m={d_sae} · top-token coherence "
                          f"SAE {sae_mean_live*100:.0f}% (live) vs PCA {pca_mean*100:.0f}%")))
        print(f"wrote {out_html}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"(skipped HTML viz: {type(e).__name__}: {e})", flush=True)

    print("\nDONE. Verdict (training vs metric vs size) goes into research/sae_at_scale_findings.md.",
          flush=True)


if __name__ == "__main__":
    main()
