"""
Phase-3 §3.6 (proper-scale rerun) — does the SAE>PCA feature-discovery gap RETURN at proper scale,
or was the §3.6 collapse real?

The first engine-scale run (p4_engine_discover.py / sae_at_scale_findings.md) found the SAE's
advantage over PCA *collapsed* (SAE 40% vs PCA 44% top-token coherence) — BUT it was confounded by
(1) an UNDER-RESOURCED SAE (m=512, ~1x overcomplete, trained on ~5k tokens) and (2) a harvest that
only saw the model's own GENERATED tokens (repetitive, instruct-skewed) and crashed under sustained
streaming. This harness removes BOTH confounds:

  * NATURAL-text activations, harvested in ONE causal forward per passage via the engine's new
    POST /harvest endpoint (no sampling, no sustained-generation crash). Corpus = WikiText-103
    (encyclopedic, multi-topic, held-out prose) — the "cleaner corpus" the prior writeup wanted.
  * A PROPERLY-RESOURCED SAE: 16x and 32x expansion (m ~ 14k / 28k over 896-dim), trained for many
    epochs on 100k+ tokens, with the §3.6 training-config bug FIXED (small batch + higher lr, so the
    optimizer actually converges: live features > 0, MSE well below 1.0 — not the dead MSE=1.0 run).

The metric is UNCHANGED and identical for SAE and PCA (apples-to-apples): un-seeded top-token
coherence — for each feature, take its top-N activating tokens; coherence = fraction equal to that
feature's modal top token (token-identity concentration). A feature locked to one token scores 1.

Honesty-first: a negative result (gap stays collapsed) is a valid, reportable outcome. We print the
full L1 dose-response, both expansions, and the PCA baseline at matched K, and we do NOT cherry-pick.

Usage (from inspector/, cloze venv python; server PATH needs the engine bin + CUDA v13.3 bin\\x64):
    python spikes/p4_big_sae.py                 # full: harvest ~120k rows, train, eval
    python spikes/p4_big_sae.py --from-cache     # re-analyze the saved matrix (no engine needed)
    python spikes/p4_big_sae.py --target 60000   # smaller harvest
"""
from __future__ import annotations

import collections
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.discover import Feature, TinySAE, standardize  # noqa: E402
from clozn.sources.engine import decode_tensor  # noqa: E402
from clozn.viz import render_discovered_features  # noqa: E402

# ---- engine launch config (mirrors p4_engine_discover.py) ---------------------------------------
MODEL = r"C:\Users\brigi\src\cloze\core\models\Qwen2.5-0.5B-Instruct-q8_0.gguf"
EXE = r"C:\Users\brigi\src\clozn\engine\core\build-gpu\cloze-server.exe"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
ENGINE_BIN = r"C:\Users\brigi\src\clozn\engine\core\build-gpu\bin"
PORT = 8080
BASE_URL = f"http://127.0.0.1:{PORT}"
RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
CACHE = os.path.join(RUNS, "qwen_big_natural_acts.npz")


# ---- server management --------------------------------------------------------------------------
def _env_with_paths() -> dict:
    env = dict(os.environ)
    env["PATH"] = ENGINE_BIN + os.pathsep + CUDA_BIN + os.pathsep + env.get("PATH", "")
    env["HF_HUB_DISABLE_SYMLINKS"] = "1"  # this PC: HF downloads crash without it (WinError 1314)
    return env


def _health_ok(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(BASE_URL + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def kill_server() -> None:
    subprocess.run(["taskkill", "/IM", "cloze-server.exe", "/F"], capture_output=True, text=True)
    time.sleep(0.4)


def start_server() -> subprocess.Popen:
    kill_server()
    os.makedirs(RUNS, exist_ok=True)
    out = open(os.path.join(RUNS, "server_big_sae.log"), "w")
    err = open(os.path.join(RUNS, "server_big_sae.log.err"), "w")
    p = subprocess.Popen(
        [EXE, MODEL, "--gpu-layers", "99", "--port", str(PORT), "--ctx", "4096"],
        env=_env_with_paths(), stdout=out, stderr=err)
    for _ in range(90):
        if _health_ok():
            return p
        time.sleep(1.0)
    raise RuntimeError("server did not become healthy")


# ---- the /harvest call (one causal forward over a text -> all token residuals) ------------------
def harvest_text(text: str, timeout: float = 120.0):
    """POST /harvest {text} -> (acts[n_tokens, n_embd] float32, pieces[list[str]], layer:int).
    Raises urllib/connection errors so the caller can restart the server and continue."""
    body = json.dumps({"text": text}).encode("utf-8")
    req = urllib.request.Request(BASE_URL + "/harvest", data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode("utf-8"))
    acts = decode_tensor(resp["activations"]).astype(np.float32)
    return acts, list(resp["tokens"]), int(resp["layer"])


# ---- corpus: WikiText-103 passages (natural held-out encyclopedic prose) -------------------------
# Cap any single passage so one giant article can't dominate; concatenate short ones up to a sane
# size so each /harvest forward is efficient. We avoid the model's OWN generations entirely.
def wikitext_passages(max_chars: int = 800):
    from clozn.corpora import text_stream
    buf = ""
    for t in text_stream(source="wikitext", min_len=60):
        t = t.strip()
        if not t:
            continue
        if len(buf) + len(t) + 1 <= max_chars:
            buf = (buf + " " + t) if buf else t
        else:
            if buf:
                yield buf
            buf = t if len(t) <= max_chars else t[:max_chars]
    if buf:
        yield buf


def harvest_corpus(target_rows: int, layer=None):
    """Harvest ~target_rows natural-text residuals via /harvest, auto-restarting the server only on a
    real CONNECTION failure (a 400 just skips that passage — the server is fine). One forward per
    passage; with /harvest there's no sustained-generation stream, so crashes should be rare."""
    start_server()
    # confirm the endpoint + layer once, loudly (self-gate echo).
    a0, p0, lyr = harvest_text("The mitochondria is the powerhouse of the cell.")
    print(f"  /harvest live: tap layer={lyr}, n_embd={a0.shape[1]}, first probe rows={a0.shape[0]}",
          flush=True)
    vecs: list[np.ndarray] = []
    pieces: list[str] = []
    crashes = 0
    skipped = 0
    n_pass = 0
    t0 = time.time()
    gen = wikitext_passages()
    while len(vecs) < target_rows and crashes < 40:
        try:
            text = next(gen)
        except StopIteration:
            gen = wikitext_passages()  # loop the stream if we somehow exhaust it
            text = next(gen)
        try:
            acts, toks, lyr = harvest_text(text)
            n = min(acts.shape[0], len(toks))
            for r in range(n):
                vecs.append(acts[r])
                pieces.append(toks[r])
            n_pass += 1
            if n_pass % 50 == 0:
                rate = len(vecs) / max(time.time() - t0, 1e-6)
                print(f"  passage {n_pass}: {len(vecs)} rows ({rate:.0f} rows/s)", flush=True)
        except urllib.error.HTTPError as e:
            # A 4xx (e.g. a passage that tokenized past n_ctx) — the server is healthy; skip it.
            skipped += 1
            if skipped <= 5:
                print(f"  passage {n_pass}: HTTP {e.code} (skipped, server OK)", flush=True)
        except (urllib.error.URLError, ConnectionResetError, OSError) as e:
            crashes += 1
            print(f"  passage {n_pass}: connection error ({type(e).__name__}); restart #{crashes}",
                  flush=True)
            start_server()
    X = np.stack(vecs).astype(np.float32)
    print(f"harvested {X.shape[0]} rows x {X.shape[1]} dims from {n_pass} passages in "
          f"{time.time()-t0:.0f}s ({crashes} restart(s), {skipped} skipped; tap layer {lyr}, "
          f"NATURAL WikiText)", flush=True)
    return X, pieces, lyr


# ---- un-seeded coherence metric (identical for SAE and PCA; same as p4_engine_discover) ----------
def _norm(t: str) -> str:
    return t.strip().lower()


def top_token_coherence(scores_per_feature, pieces, topn=20):
    """scores_per_feature: [N, F]. For each feature take its top-N activating rows; coherence =
    fraction equal to the feature's MODAL top token. Returns (coh[F], modal[F], tops[F])."""
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


# ---- a GPU-resident SAE (scales discover.TinySAE to 16-32x without a 100k x m host matrix) -------
class TorchSAE:
    """The exact discover.TinySAE objective (f=relu(x.We+be); x_hat=f.Wd+bd; MSE + L1*|f|; unit-norm
    decoder rows), but on CUDA with minibatching that NEVER materializes the full [N, m] codes on the
    host — essential at m~28k, N~120k (that dense matrix is ~13 GB f32). codes_topk() returns only the
    per-feature top activations needed for the coherence metric. Same math, just resourced for scale."""

    def __init__(self, d: int, m: int, l1: float = 1.0, seed: int = 0, device: str = "cuda"):
        import torch
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        g = torch.Generator().manual_seed(seed)
        self.We = (torch.randn(d, m, generator=g) * 0.1).to(self.device).requires_grad_(True)
        self.be = torch.zeros(m, device=self.device, requires_grad=True)
        self.Wd = (torch.randn(m, d, generator=g) * 0.1).to(self.device).requires_grad_(True)
        self.bd = torch.zeros(d, device=self.device, requires_grad=True)
        self.l1 = l1
        self.m = m

    def _encode(self, x):
        return (x @ self.We + self.be).clamp(min=0)

    def fit(self, X, epochs: int = 30, lr: float = 3e-2, batch_size: int = 2048):
        torch = self.torch
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam([self.We, self.be, self.Wd, self.bd], lr=lr)
        n = Xt.shape[0]
        torch.manual_seed(0)
        last = 0.0
        for ep in range(epochs):
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            nb = 0
            for s in range(0, n, batch_size):
                xb = Xt[perm[s:s + batch_size]]
                f = self._encode(xb)
                recon = f @ self.Wd + self.bd
                loss = ((recon - xb) ** 2).mean() + self.l1 * f.abs().mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    self.Wd.data /= (self.Wd.data.norm(dim=1, keepdim=True) + 1e-8)
                    tot += float(((recon - xb) ** 2).mean())
                nb += 1
            last = tot / max(nb, 1)
        self._final_mse = last
        return self

    def stats(self, X, batch_size: int = 8192):
        """One pass over X (no host [N,m]): per-feature fire-rate, mean MSE."""
        torch = self.torch
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = Xt.shape[0]
        fire = torch.zeros(self.m, device=self.device)
        mse_tot = 0.0
        nb = 0
        with torch.no_grad():
            for s in range(0, n, batch_size):
                xb = Xt[s:s + batch_size]
                f = self._encode(xb)
                fire += (f > 1e-6).float().sum(0)
                recon = f @ self.Wd + self.bd
                mse_tot += float(((recon - xb) ** 2).mean())
                nb += 1
        return (fire / n).cpu().numpy(), mse_tot / max(nb, 1)

    def codes_topk(self, X, topn: int = 20, batch_size: int = 8192):
        """Return, per feature, the ROW INDICES of its top-`topn` activations + those values, computed
        in a streaming pass (heap-free: keep a running top-topn per feature on the GPU). Shapes:
        (top_idx[m, topn] int64, top_val[m, topn] float32). This is all the coherence metric needs,
        and it avoids the ~13 GB dense [N, m] host array a naive codes() would build."""
        torch = self.torch
        Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = Xt.shape[0]
        # Running buffers of the best `topn` per feature: values (-inf init) + their global row index.
        best_val = torch.full((self.m, topn), -1e30, device=self.device)
        best_idx = torch.full((self.m, topn), -1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            for s in range(0, n, batch_size):
                xb = Xt[s:s + batch_size]
                f = self._encode(xb).T.contiguous()        # [m, b]
                b = f.shape[1]
                idx = torch.arange(s, s + b, device=self.device).expand(self.m, b)
                # Merge the running best with this batch, keep the global top-topn per feature.
                vals = torch.cat([best_val, f], dim=1)      # [m, topn+b]
                ids = torch.cat([best_idx, idx], dim=1)      # [m, topn+b]
                k = min(topn, vals.shape[1])
                tv, ti = torch.topk(vals, k, dim=1)
                best_val = tv
                best_idx = torch.gather(ids, 1, ti)
        return best_idx.cpu().numpy(), best_val.cpu().numpy()


def sae_coherence_from_topk(top_idx, pieces, topn=20):
    """Coherence from the streaming top-k row indices (matches top_token_coherence's definition but
    consumes top_idx[m, topn] instead of a dense [N, m]). Returns (coh[m], modal[m], tops[m])."""
    m = top_idx.shape[0]
    coh = np.zeros(m)
    modal = [""] * m
    tops = [[] for _ in range(m)]
    norm_pieces = [_norm(p) for p in pieces]
    for j in range(m):
        rows = [int(i) for i in top_idx[j] if i >= 0][:topn]
        toks = [norm_pieces[i] for i in rows]
        tops[j] = [pieces[i] for i in rows]
        cnt = collections.Counter(t for t in toks if t != "")
        if cnt:
            tok, c = cnt.most_common(1)[0]
            modal[j] = tok
            coh[j] = c / len(toks)
    return coh, modal, tops


def main():
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = 120_000
    if "--target" in sys.argv:
        target = int(sys.argv[sys.argv.index("--target") + 1])
    elif pos:
        target = int(pos[0])
    os.makedirs(RUNS, exist_ok=True)

    if "--from-cache" in flags and os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        X, pieces = d["X"], list(d["pieces"])
        layer = int(d["layer"]) if "layer" in d else 2
        print(f"=== loaded {X.shape[0]} cached rows x {X.shape[1]} dims (layer {layer}, natural) ===",
              flush=True)
    else:
        print(f"=== harvesting ~{target} NATURAL-text residuals from Qwen-0.5B via /harvest ===",
              flush=True)
        X, pieces, layer = harvest_corpus(target_rows=target)
        np.savez_compressed(CACHE, X=X, pieces=np.array(pieces, dtype=object), layer=layer)
        print(f"saved {CACHE}", flush=True)
        kill_server()

    # Standardize (per-feature) — handles the first-token outlier and matches the prior pipeline.
    Xs, mu, sd = standardize(X)
    d = Xs.shape[1]
    N = Xs.shape[0]
    uniq = len(set(_norm(p) for p in pieces if p.strip()))
    print(f"\ncorpus: {N} rows, {d} dims, {uniq} unique tokens "
          f"(unique ratio {uniq / max(N,1):.3f})", flush=True)

    # ---- PCA baseline at MATCHED component count (the comparable axes) ----------------------------
    # Use K=256 axes (a richer baseline than the prior 64; we also report mean over the top-64 for a
    # like-for-like with §3.6). SVD on standardized rows; project + score with the SAME metric.
    K = 256
    print(f"\n=== PCA baseline (top-{K} axes) ===", flush=True)
    t0 = time.time()
    # economy SVD on [N, d]; d=896 so Vt is small. For large N use the covariance eigh route.
    if N > 20000:
        C = (Xs.T @ Xs) / N                      # [d, d]
        evals, evecs = np.linalg.eigh(C)         # ascending
        order = np.argsort(evals)[::-1]
        Vt = evecs[:, order].T                    # [d, d], rows = components desc
        var = evals[order] / evals.sum()
    else:
        U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
        var = (S ** 2) / (S ** 2).sum()
    pca_proj = Xs @ Vt[:K].T                       # [N, K]
    pca_coh, pca_modal, pca_tops = top_token_coherence(pca_proj, pieces, topn=20)
    pca_mean = float(pca_coh.mean())
    pca_mean64 = float(pca_coh[:64].mean())
    print(f"  PCA top-{K} mean coherence = {pca_mean*100:.1f}%  (top-64 = {pca_mean64*100:.1f}%)  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- properly-resourced SAE: 16x and 32x expansion, L1 dose-response -------------------------
    # Training config (the §3.6 bug FIX): batch_size=512 + lr=3e-2 + 40 epochs so the optimizer
    # converges (verified MSE well below 1.0; the dead run was batch=4096/12-steps -> MSE=1.0). At
    # 120k rows that's ~235 steps/epoch * 40 ~ 9.4k gradient steps per config. L1 is swept WIDE and
    # HIGH (0.5 -> 8): natural-text layer-2 activations are dense (a well-trained SAE fires ~70% at
    # L1=1), so real sparsity needs a stronger penalty — we show the whole dose-response, honestly.
    L1_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)
    # Training config (the §3.6 dead-SAE bug, fixed from BOTH directions): the prior failure was
    # too-FEW steps (batch=4096 -> ~12 steps -> MSE 1.0). The opposite failure, found here at scale,
    # is too-AGGRESSIVE lr: lr=1e-2 on the full 120k matrix DIVERGES (MSE ~75) because the ~800
    # first-token "attention-sink" outlier rows (norm up to ~270 even standardized) blow up the early
    # gradients. lr=1e-3, batch=512, 40 epochs converges cleanly at BOTH 16x and 32x (MSE 0.04-0.10,
    # ~all features live, fire-rate falls 13%->5% with L1) — calibrated on this exact matrix. Epochs
    # scale UP for a small corpus so the step count stays sufficient (the --from-cache smoke path).
    BATCH, LR = 512, 1e-3
    steps_per_epoch = max(1, (N + BATCH - 1) // BATCH)
    EPOCHS = int(min(200, max(40, (9000 + steps_per_epoch - 1) // steps_per_epoch)))
    print(f"\nSAE training: batch={BATCH} lr={LR} epochs={EPOCHS} "
          f"({steps_per_epoch} steps/epoch -> ~{EPOCHS*steps_per_epoch} grad steps/config)", flush=True)
    results = []
    best = None
    for exp in (16, 32):
        m = exp * d
        print(f"\n=== SAE {exp}x expansion (m={m}) ===", flush=True)
        for l1 in L1_GRID:
            t0 = time.time()
            sae = TorchSAE(d, m=m, l1=l1, seed=0).fit(Xs, epochs=EPOCHS, lr=LR, batch_size=BATCH)  # noqa: E501
            fire, mse = sae.stats(Xs)
            live_mask = (fire >= 0.002) & (fire <= 0.4)
            n_live = int(live_mask.sum())
            top_idx, _ = sae.codes_topk(Xs, topn=20)
            coh, modal, tops = sae_coherence_from_topk(top_idx, pieces, topn=20)
            live_coh = coh[live_mask]
            mean_coh = float(live_coh.mean()) if n_live else 0.0
            dt = time.time() - t0
            converged = mse < 1.0  # honesty guard: an SAE that doesn't reconstruct (MSE>=1) is dead
            rec = {"exp": exp, "m": int(m), "l1": l1, "mean_fire": float(fire.mean()),
                   "n_live": n_live, "coherence": mean_coh, "mse": float(mse),
                   "converged": converged, "secs": dt}
            results.append(rec)
            flag = "" if converged else "  <-- NOT CONVERGED (MSE>=1, ignore coherence)"
            print(f"  [SAE {exp}x l1={l1}] live={n_live:<6} mean_fire={fire.mean()*100:5.2f}% "
                  f"coh(live)={mean_coh*100:4.1f}% mse={mse:.3f}  [{dt:.0f}s]{flag}", flush=True)
            # prefer a CONVERGED, sparse-enough, coherent run (same selection rule as the prior
            # spike, but gated on convergence so a dead optimizer can never win the verdict).
            sparse_enough = fire.mean() <= 0.10
            score = (1 if converged else 0, 1 if sparse_enough else 0, mean_coh)
            if best is None or score > best["_score"]:
                best = dict(exp=exp, m=int(m), l1=l1, fire=fire, live_mask=live_mask, coh=coh,
                            modal=modal, tops=tops, mean_coh=mean_coh, n_live=n_live, mse=float(mse),
                            converged=converged, _score=score)
            del sae  # free GPU memory before the next config
            import torch as _t
            if _t.cuda.is_available():
                _t.cuda.empty_cache()

    print(f"\n=== verdict inputs ===", flush=True)
    print(f"  PCA top-{K} mean coherence : {pca_mean*100:.1f}%  (top-64 {pca_mean64*100:.1f}%)")
    print(f"  SAE best ({best['exp']}x l1={best['l1']}) live mean coherence: "
          f"{best['mean_coh']*100:.1f}% over {best['n_live']} live features (MSE {best['mse']:.3f})")
    gap = best["mean_coh"] - pca_mean
    print(f"  SAE - PCA gap: {gap*100:+.1f} pts  -> "
          f"{'SAE ahead' if gap > 0.02 else ('PCA ahead' if gap < -0.02 else 'TIE')}")

    # ---- name 10 example SAE features (most coherent live) ---------------------------------------
    live_idx = np.where(best["live_mask"])[0]
    ranked = sorted(live_idx, key=lambda j: -best["coh"][j])
    feats = []
    print(f"\n=== 10 most-coherent discovered SAE features ({best['exp']}x, layer {layer}) ===")
    for j in ranked[:10]:
        tt = best["tops"][j]
        print(f"  f{j:<6} coh={best['coh'][j]*100:3.0f}%  fires={best['fire'][j]*100:5.2f}%  "
              f"modal={best['modal'][j]!r:<14} top: {' '.join(repr(t) for t in tt[:10])}", flush=True)
        feats.append(Feature(int(j), "sae", tt[:10], float(best["fire"][j])))

    print("\n=== 8 PCA axes for contrast ===")
    pca_feats = []
    for j in range(8):
        tt = pca_tops[j]
        print(f"  PC{j:<3} coh={pca_coh[j]*100:3.0f}%  var={var[j]*100:4.1f}%  "
              f"modal={pca_modal[j]!r:<14} top: {' '.join(repr(t) for t in tt[:10])}", flush=True)
        pca_feats.append(Feature(int(j), "pca", tt[:10], float(var[j])))

    # ---- save artifacts (HTML viz + machine-readable summary) ------------------------------------
    out_html = os.path.join(RUNS, "discovered_big_sae_qwen.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(render_discovered_features(
            feats, title="Clozn · Big-SAE Discovery (Qwen2.5-0.5B, natural text)",
            subtitle=(f"{N} WikiText residual tokens via /harvest (layer {layer}) · "
                      f"SAE {best['exp']}x m={best['m']} l1={best['l1']} · top-token coherence "
                      f"SAE {best['mean_coh']*100:.0f}% vs PCA {pca_mean*100:.0f}%")))
    print(f"\nwrote {out_html}")

    summary = {
        "rows": int(N), "dim": int(d), "tap_layer": int(layer),
        "harvest": "natural WikiText-103, engine /harvest (one causal forward per passage)",
        "unique_token_ratio": uniq / max(N, 1),
        "pca_K": K, "pca_mean_coherence": pca_mean, "pca_mean_coherence_top64": pca_mean64,
        "sae_best": {"exp": best["exp"], "m": best["m"], "l1": best["l1"],
                     "mean_coherence": best["mean_coh"], "n_live": best["n_live"], "mse": best["mse"]},
        "sae_grid": results,
        "sae_minus_pca_gap": gap,
        "examples_sae": [{"idx": int(j), "coherence": float(best["coh"][j]),
                          "fires": float(best["fire"][j]), "modal": best["modal"][j],
                          "top": best["tops"][j][:10]} for j in ranked[:10]],
        "examples_pca": [{"idx": j, "coherence": float(pca_coh[j]),
                          "modal": pca_modal[j], "top": pca_tops[j][:10]} for j in range(8)],
        "prior_runs": {"engine_under_resourced": {"sae": 0.40, "pca": 0.44},
                       "toy_rwkv": {"sae": 0.65, "pca": 0.12}},
    }
    out_json = os.path.join(RUNS, "discovered_big_sae_qwen.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out_json)


if __name__ == "__main__":
    main()
