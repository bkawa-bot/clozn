"""
Phase-3 §3.6 — the DECISIVE SCALE test. Does unsupervised SAE feature discovery, a robust NULL on
Qwen2.5-0.5B (token AND semantic metrics; SAE never beat PCA; features were token/position detectors;
see research/sae_at_scale_findings.md), get RESCUED by MODEL SCALE?

The 0.5B arc ruled out the metric (token + auto-interp + concept-AUC all nulled) and the layer
(layer-12 on the 0.5B was also null). The one untested variable is SCALE. This run pulls three levers
at once vs the 0.5B baseline:

  * a 14x BIGGER model      — Qwen2.5-7B-Instruct (Q8_0), n_embd 3584 (vs 0.5B's 896)
  * a MID layer             — layer 16 of 28 (vs the 0.5B's lexical layer 2; we already showed L12
                              was null on the 0.5B, so this is "mid on a model big enough to matter")
  * ~8x MORE tokens         — ~500k-1M natural WikiText residuals (vs ~120k on the 0.5B)

into a REAL SAE (16x expansion ~ 57k features; 8x ~ 28k fallback). If concept features finally
separate from PCA HERE, scale was the floor. If it is STILL null at 7B/mid/~1M, that is a much heavier
statement: local from-scratch discovery on one consumer GPU is not the path (you would need the
literature's data budget or pretrained dictionaries).

Honesty-first (the bar a prior run set when it caught a false-positive-in-its-own-favor that reversed
under held-out + permutation): a NULL is a valid, reportable outcome. Self-gate before the big run.
Same metrics as the 0.5B run for apples-to-apples: (a) token-coherence, (b) auto-interp contexts,
(c) concept-alignment held-out AUC SAE vs PCA with a firing floor + a label-permutation null. We print
the full dose-response, never cherry-pick, and a dead optimizer (MSE>=1) can never win the verdict.

Memory note (the operational constraint): n_embd=3584, ~1M rows. The full fp32 matrix is ~14 GB — it
will NOT co-reside on a 16 GB card with a 57k-feature SAE. So (1) the engine MUST be stopped to free
its 7.6 GB before training (this script does that), and (2) the SAE here keeps the corpus on the HOST
as fp16 and streams fp32 minibatches to the GPU (it never uploads the whole [N, d] matrix), unlike
p4_big_sae's TorchSAE which uploads everything (fine at 0.5B/896, OOM at 7B/3584/1M).

Usage (from inspector/, cloze venv python):
    python spikes/p7_scale_7b.py --selfgate       # ~5k-token L16 harvest + tiny SAE, then STOP
    python spikes/p7_scale_7b.py                   # full: harvest ~700k, stop engine, train, eval
    python spikes/p7_scale_7b.py --target 1000000  # bigger harvest
    python spikes/p7_scale_7b.py --from-cache      # re-train + re-eval on the saved matrix (no engine)
    python spikes/p7_scale_7b.py --from-cache --concepts   # also run the concept-alignment harvest
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

from clozn.discover import Feature, standardize  # noqa: E402
from clozn.viz import render_discovered_features  # noqa: E402

# Reuse the 0.5B harness wholesale where it is layer-agnostic: the /harvest client, the WikiText
# corpus assembly, the un-seeded coherence metric, the streaming-topk -> coherence helper.
from spikes.p4_big_sae import (  # noqa: E402
    harvest_text, top_token_coherence, sae_coherence_from_topk, wikitext_passages, _norm,
)

# ---- engine launch config (7B model; relaunch path if the live server dies) ---------------------
MODEL = r"C:\Users\brigi\src\cloze\core\models\Qwen2.5-7B-Instruct-Q8_0.gguf"
EXE = r"C:\Users\brigi\src\clozn\engine\core\build-gpu\cloze-server.exe"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
ENGINE_BIN = r"C:\Users\brigi\src\clozn\engine\core\build-gpu\bin"
PORT = 8080
BASE_URL = f"http://127.0.0.1:{PORT}"
RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
CACHE = os.path.join(RUNS, "qwen7b_natural_acts_L16.npz")  # OWN cache (never clobber the 0.5B npz)
LAYER = 16  # mid layer of Qwen2.5-7B (28 blocks); the decisive "mid, not lexical" choice
CLIP_SIGMA = 4.0  # cap standardized row L2 norm at CLIP_SIGMA*sqrt(d) (massive-activation winsorize)


def clip_outlier_rows(Xs: np.ndarray, sigma: float = CLIP_SIGMA):
    """Winsorize the ~0.7% massive-activation / attention-sink rows. Qwen2.5-7B's layer-16 residual
    has extreme outliers: a handful of tokens reach raw L2 norm ~17,000 (230x the median ~74), and one
    channel has std ~945 — standardization alone leaves rows with standardized norm up to ~8*sqrt(d).
    Those rows (squared-error ~64x a typical row) DOMINATE the SAE gradient and stall it (the self-gate
    caught MSE~3, 0 live, mean_fire stuck at 50%). Capping each row's standardized norm at
    sigma*sqrt(d) is the literature-standard massive-activation fix (Anthropic/GemmaScope handle these
    explicitly); it is applied to BOTH the SAE training data AND the PCA input so the comparison stays
    apples-to-apples (no method gets to ignore the outliers the other must fit). Returns (Xc, n_clipped)."""
    d = Xs.shape[1]
    cap = sigma * np.sqrt(d)
    rn = np.linalg.norm(Xs, axis=1, keepdims=True)
    scale = np.minimum(1.0, cap / (rn + 1e-9))
    return (Xs * scale).astype(np.float32), int((rn > cap).sum())


def standardize_clip_chunked(X16: np.ndarray, sigma: float = CLIP_SIGMA, chunk: int = 50_000):
    """Memory-SAFE standardize + outlier-clip for the BIG (1M x 3584) case, returning a fp16 matrix
    ready to stream to the GPU. The naive path (standardize the fp32-upcast, then clip) materializes
    two ~14 GB fp32 copies simultaneously and, at 1M rows on a 31 GB box, thrashes the pagefile to
    ~60 GB committed. This version computes mean/std in a streaming pass, then writes the standardized
    +clipped result back into a fp16 buffer chunk-by-chunk — peak extra RAM is one `chunk`-sized fp32
    slab (~0.7 GB at 50k), never a full fp32 copy. Math is identical (per-feature standardize, row-norm
    cap at sigma*sqrt(d)). Returns (X16_clipped, mu, sd, n_clipped, target_var)."""
    n, d = X16.shape
    # streaming mean/var (Welford-free two-pass is fine and exact in fp64 accumulators)
    s1 = np.zeros(d, np.float64)
    s2 = np.zeros(d, np.float64)
    for i in range(0, n, chunk):
        xb = X16[i:i + chunk].astype(np.float32)
        s1 += xb.sum(0)
        s2 += (xb.astype(np.float64) ** 2).sum(0)
    mu = (s1 / n).astype(np.float32)
    var = (s2 / n) - (s1 / n) ** 2
    sd = (np.sqrt(np.maximum(var, 0)).astype(np.float32) + 1e-6)
    cap = sigma * np.sqrt(d)
    out = np.empty((n, d), np.float16)            # the only big allocation (7 GB), reused as result
    n_clip = 0
    ss = 0.0                                       # accumulate sum of squares of the CLIPPED standardized values
    for i in range(0, n, chunk):
        xb = (X16[i:i + chunk].astype(np.float32) - mu) / sd      # standardized chunk (fp32, small)
        rn = np.linalg.norm(xb, axis=1, keepdims=True)
        scale = np.minimum(1.0, cap / (rn + 1e-9))
        xb *= scale
        n_clip += int((rn > cap).sum())
        ss += float((xb.astype(np.float64) ** 2).sum())
        out[i:i + chunk] = xb.astype(np.float16)
    tvar = ss / (n * d)
    return out, mu, sd, n_clip, float(tvar)


def pca_basis_chunked(X16c: np.ndarray, chunk: int = 50_000):
    """Covariance (d x d) of a fp16 matrix in a streaming pass (no 1M x 3584 fp32 copy), then eigh.
    Returns (Vt[d,d] desc, var_share[d]). The matrix is ALREADY standardized+clipped (mean ~0), so the
    raw second-moment is the covariance up to the tiny residual mean — matched to how the SAE sees it."""
    n, d = X16c.shape
    C = np.zeros((d, d), np.float64)
    for i in range(0, n, chunk):
        xb = X16c[i:i + chunk].astype(np.float32)
        C += xb.T @ xb
    C /= n
    evals, evecs = np.linalg.eigh(C)
    order = np.argsort(evals)[::-1]
    Vt = evecs[:, order].T.astype(np.float32)
    var = (evals[order] / evals.sum()).astype(np.float64)
    return Vt, var


def project_coherence_chunked(X16c: np.ndarray, Vt: np.ndarray, K: int, pieces, chunk: int = 100_000):
    """PCA projection top-token coherence WITHOUT a 1M x K dense host matrix kept around: project in
    chunks, accumulate per-axis running top-20 rows (same streaming-topk trick the SAE uses). Returns
    (coh[K], modal[K], tops[K]) — identical definition to top_token_coherence, just chunked."""
    n = X16c.shape[0]
    topn = 20
    best_val = np.full((K, topn), -1e30, np.float32)
    best_idx = np.full((K, topn), -1, np.int64)
    VtK = Vt[:K].T.astype(np.float32)             # [d, K]
    for i in range(0, n, chunk):
        xb = X16c[i:i + chunk].astype(np.float32)
        proj = xb @ VtK                            # [b, K]
        b = proj.shape[0]
        # merge this chunk's per-axis top-20 with the running best
        idx = np.arange(i, i + b)[:, None]         # [b,1]
        for jstart in range(0, K, 1):              # per-axis (K=256 is fine)
            col = proj[:, jstart]
            allv = np.concatenate([best_val[jstart], col])
            alli = np.concatenate([best_idx[jstart], idx[:, 0]])
            sel = np.argsort(allv)[::-1][:topn]
            best_val[jstart] = allv[sel]
            best_idx[jstart] = alli[sel]
    # now compute coherence from the row indices
    coh = np.zeros(K); modal = [""] * K; tops = [[] for _ in range(K)]
    norm_pieces = [_norm(p) for p in pieces]
    for j in range(K):
        rows = [int(r) for r in best_idx[j] if r >= 0][:topn]
        toks = [norm_pieces[r] for r in rows]
        tops[j] = [pieces[r] for r in rows]
        cnt = collections.Counter(t for t in toks if t != "")
        if cnt:
            tok, c = cnt.most_common(1)[0]
            modal[j] = tok
            coh[j] = c / len(toks)
    return coh, modal, tops


# ---- server management (mirrors p4_big_sae but with the 7B model) -------------------------------
def _env_with_paths() -> dict:
    env = dict(os.environ)
    env["PATH"] = ENGINE_BIN + os.pathsep + CUDA_BIN + os.pathsep + env.get("PATH", "")
    env["HF_HUB_DISABLE_SYMLINKS"] = "1"
    return env


def _health_ok(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(BASE_URL + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def kill_server() -> None:
    """Free the engine's ~7.6 GB VRAM before training the SAE on the GPU."""
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Get-Process cloze-server -ErrorAction SilentlyContinue | Stop-Process -Force"],
                   capture_output=True, text=True)
    time.sleep(1.0)


def ensure_server() -> None:
    """Use the already-running 7B engine if healthy; otherwise (re)launch it. We do NOT kill a healthy
    server — the task says the engine is already up; relaunch is only the crash-recovery path."""
    if _health_ok():
        return
    kill_server()
    os.makedirs(RUNS, exist_ok=True)
    out = open(os.path.join(RUNS, "server_p7_7b.log"), "w")
    err = open(os.path.join(RUNS, "server_p7_7b.log.err"), "w")
    subprocess.Popen([EXE, MODEL, "--gpu-layers", "99", "--port", str(PORT), "--ctx", "4096"],
                     env=_env_with_paths(), stdout=out, stderr=err)
    for _ in range(120):
        if _health_ok():
            return
        time.sleep(1.0)
    raise RuntimeError("7B server did not become healthy")


# ---- layer-aware corpus harvest (passes layer through; p4's harvest_corpus hardcodes the default) -
def harvest_corpus_layer(target_rows: int, layer: int, store_fp16: bool = True):
    """Harvest ~target_rows natural-text residuals at `layer` via /harvest. One forward per passage;
    a 4xx just skips that passage (server fine), a real connection error triggers a relaunch. Stores
    fp16 to keep ~1M x 3584 manageable (~7 GB) — the SAE casts back to fp32 per-batch on the GPU."""
    ensure_server()
    a0, _p0, lyr = harvest_text("The mitochondria is the powerhouse of the cell.", layer=layer)
    print(f"  /harvest live: tap layer={lyr}, n_embd={a0.shape[1]}, probe rows={a0.shape[0]}",
          flush=True)
    assert lyr == layer, f"engine returned layer {lyr}, expected {layer}"
    dt = np.float16 if store_fp16 else np.float32
    vecs: list[np.ndarray] = []
    pieces: list[str] = []
    crashes = skipped = n_pass = 0
    t0 = time.time()
    gen = wikitext_passages()
    while len(vecs) < target_rows and crashes < 40:
        try:
            text = next(gen)
        except StopIteration:
            gen = wikitext_passages()
            text = next(gen)
        try:
            acts, toks, lyr = harvest_text(text, layer=layer)
            n = min(acts.shape[0], len(toks))
            for r in range(n):
                vecs.append(acts[r].astype(dt))
                pieces.append(toks[r])
            n_pass += 1
            if n_pass % 50 == 0:
                rate = len(vecs) / max(time.time() - t0, 1e-6)
                print(f"  passage {n_pass}: {len(vecs)} rows ({rate:.0f} rows/s, "
                      f"{time.time()-t0:.0f}s)", flush=True)
        except urllib.error.HTTPError as e:
            skipped += 1
            if skipped <= 5:
                print(f"  passage {n_pass}: HTTP {e.code} (skipped, server OK)", flush=True)
        except (urllib.error.URLError, ConnectionResetError, OSError) as e:
            crashes += 1
            print(f"  passage {n_pass}: connection error ({type(e).__name__}); relaunch #{crashes}",
                  flush=True)
            ensure_server()
    X = np.stack(vecs).astype(dt)
    print(f"harvested {X.shape[0]} rows x {X.shape[1]} dims from {n_pass} passages in "
          f"{time.time()-t0:.0f}s ({crashes} relaunch(es), {skipped} skipped; tap layer {lyr}, "
          f"NATURAL WikiText, dtype {X.dtype})", flush=True)
    return X, pieces, lyr


# ---- a HOST-resident-data SAE: streams fp32 minibatches to GPU (never uploads the [N,d] matrix) ---
class StreamingSAE:
    """The exact discover.TinySAE / p4 TorchSAE objective (f=relu(x.We+be); x_hat=f.Wd+bd;
    MSE + L1*|f|; unit-norm decoder rows), but the corpus stays on the HOST as fp16 and each minibatch
    is moved to the GPU and cast to fp32 just-in-time. This is the only change from p4's TorchSAE, and
    it is mandatory at 7B scale: the full fp32 matrix (1M x 3584 x 4 = ~14 GB) would OOM a 16 GB card
    alongside a 57k-feature dictionary. Same math, just data-streamed so only the params + one batch
    live on the GPU. codes_topk streams identically for the coherence metric."""

    def __init__(self, d: int, m: int, l1: float = 1.0, seed: int = 0, device: str = "cuda"):
        import torch
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        g = torch.Generator().manual_seed(seed)
        # Init that ACTUALLY converges at 7B/3584 width (the self-gate found a plain randn*0.1 init
        # stalls at mean_fire~50%, MSE~3): unit-norm encoder COLUMNS scaled to 0.1, and the decoder
        # initialized as the encoder transpose (a near-tied start). Without this, the optimizer never
        # leaves the random regime in a feasible epoch budget on the wider 7B residual. Same objective
        # (relu code, L1, unit-norm decoder rows) — only the initialization changed from p4's TorchSAE.
        W = torch.randn(d, m, generator=g)
        W = W / (W.norm(dim=0, keepdim=True) + 1e-8)          # unit-norm encoder columns
        self.We = (W * 0.1).to(self.device).requires_grad_(True)
        self.be = torch.zeros(m, device=self.device, requires_grad=True)
        self.Wd = W.t().contiguous().to(self.device).clone().requires_grad_(True)  # decoder = W.T
        self.bd = torch.zeros(d, device=self.device, requires_grad=True)
        self.l1 = l1
        self.m = m

    def _encode(self, x):
        return (x @ self.We + self.be).clamp(min=0)

    def fit(self, Xhost, epochs: int = 40, lr: float = 1e-3, batch_size: int = 512,
            grad_clip: float = 1.0):
        """Xhost: a torch tensor (fp16 ok) of shape [N, d]. If it is ALREADY on `self.device` (the
        GPU-resident fast path — data uploaded once), the per-batch index+cast is a pure-GPU gather
        (~20x faster than host-streaming, which at 1M rows costs 136s/epoch from the CPU fancy-index +
        PCIe transfer). If it is on the host, batches are moved+cast per step (the memory-safe path for
        corpora too big to fit on the card). The optimizer state stays on the GPU either way.

        `grad_clip` (max global grad-norm) is the STABILITY fix the diagnostics demanded: the WIDE 16x
        dictionary (m=57344) DIVERGED at lr=3e-3 (MSE 67 -> 8143 by epoch 10) even with the outlier-row
        clip — the reconstruction's sum over 4x more features than the 4x self-gate amplifies gradients
        past what Adam can absorb, and the residual massive-activation structure spikes them. Clipping
        the global grad-norm to ~1 caps each step regardless of width/outliers (standard SAE-training
        practice) and is the third training trap this experiment surfaced (after dead-optimizer and
        lr=1e-2 divergence). Set grad_clip=None to disable."""
        torch = self.torch
        opt = torch.optim.Adam([self.We, self.be, self.Wd, self.bd], lr=lr)
        params = [self.We, self.be, self.Wd, self.bd]
        n = Xhost.shape[0]
        torch.manual_seed(0)
        on_gpu = (str(Xhost.device) != "cpu")     # perm on the data's device avoids a per-batch sync
        last = 0.0
        for ep in range(epochs):
            perm = torch.randperm(n, device=Xhost.device if on_gpu else "cpu")
            tot = 0.0
            nb = 0
            te0 = time.time()
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                xb = Xhost[idx].to(self.device, non_blocking=True).float()
                f = self._encode(xb)
                recon = f @ self.Wd + self.bd
                loss = ((recon - xb) ** 2).mean() + self.l1 * f.abs().mean()
                opt.zero_grad()
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)
                opt.step()
                with torch.no_grad():
                    self.Wd.data /= (self.Wd.data.norm(dim=1, keepdim=True) + 1e-8)
                    tot += float(((recon - xb) ** 2).mean())
                nb += 1
            last = tot / max(nb, 1)
            if ep == 0 or (ep + 1) % 10 == 0:
                print(f"    epoch {ep+1}/{epochs}: mse~{last:.4f}  ({time.time()-te0:.0f}s/epoch)",
                      flush=True)
        self._final_mse = last
        return self

    def stats(self, Xhost, batch_size: int = 4096):
        torch = self.torch
        n = Xhost.shape[0]
        fire = torch.zeros(self.m, device=self.device)
        mse_tot = 0.0
        nb = 0
        with torch.no_grad():
            for s in range(0, n, batch_size):
                xb = Xhost[s:s + batch_size].to(self.device).float()
                f = self._encode(xb)
                fire += (f > 1e-6).float().sum(0)
                recon = f @ self.Wd + self.bd
                mse_tot += float(((recon - xb) ** 2).mean())
                nb += 1
        return (fire / n).cpu().numpy(), mse_tot / max(nb, 1)

    def codes_topk(self, Xhost, topn: int = 20, batch_size: int = 4096):
        """Per feature, the row indices of its top-`topn` activations + values, in a streaming pass
        (running top-topn per feature on the GPU). Host-streamed, fp32 per batch. Shapes:
        (top_idx[m, topn] int64, top_val[m, topn] float32)."""
        torch = self.torch
        n = Xhost.shape[0]
        best_val = torch.full((self.m, topn), -1e30, device=self.device)
        best_idx = torch.full((self.m, topn), -1, dtype=torch.long, device=self.device)
        with torch.no_grad():
            for s in range(0, n, batch_size):
                xb = Xhost[s:s + batch_size].to(self.device).float()
                f = self._encode(xb).T.contiguous()           # [m, b]
                b = f.shape[1]
                idx = torch.arange(s, s + b, device=self.device).expand(self.m, b)
                vals = torch.cat([best_val, f], dim=1)
                ids = torch.cat([best_idx, idx], dim=1)
                k = min(topn, vals.shape[1])
                tv, ti = torch.topk(vals, k, dim=1)
                best_val = tv
                best_idx = torch.gather(ids, 1, ti)
        return best_idx.cpu().numpy(), best_val.cpu().numpy()

    def encode_rows(self, A):
        """Encode a small [n, d] numpy matrix (the concept-alignment corpora) -> codes [n, m]."""
        torch = self.torch
        with torch.no_grad():
            xt = torch.tensor(A, dtype=torch.float32, device=self.device)
            return (xt @ self.We + self.be).clamp(min=0).cpu().numpy()


# ---- concept alignment (the semantic metric — held-out AUC SAE vs PCA, firing floor + null) -------
def _auc(scores, labels):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    a = (ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return max(a, 1 - a)


def _best_feature_auc_held_out(codes, y, min_fire=8, k=4, seed=0):
    """Held-out single-unit AUC with a firing floor + a label-permutation null — the SAME honest
    metric as p6 (guards the 14k-features x 24-samples multiple-comparisons trap)."""
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
        bj = cand[0]
        for f in range(k):
            te = folds[f]
            tr = np.concatenate([folds[g] for g in range(k) if g != f])
            tr_auc = np.array([_auc(codes[tr][:, j], labels[tr]) for j in cand])
            bj = cand[int(np.argmax(tr_auc))]
            accs.append(_auc(codes[te][:, bj], labels[te]))
        return float(np.mean(accs)), bj

    ho_auc, _ = held_out(y)
    full_auc = np.array([_auc(codes[:, j], y) for j in cand])
    best_feat = int(cand[int(np.argmax(full_auc))])
    null = np.mean([held_out(rng.permutation(y))[0] for _ in range(5)])
    return best_feat, ho_auc, float(null)


def run_concept_alignment(sae, mu, sd, Vt, K, layer):
    """Harvest the inspector's five matched-frame concept corpora at the SAME layer, encode through the
    SAE and PCA, score held-out single-unit AUC (SAE vs PCA, with firing floor + permutation null) plus
    a whole-representation k-fold probe. Identical procedure to p6 — apples-to-apples with the 0.5B."""
    from clozn.atlas import (NUMBER_SING, NUMBER_PLUR, TENSE_PAST, TENSE_PRES, PERSON_1, PERSON_3,
                             QUESTION, STATEMENT)
    from clozn.probes import DEFAULT_POS, DEFAULT_NEG, kfold_accuracy
    concepts = {
        "number (sing/plural)": (NUMBER_SING, NUMBER_PLUR, "singular", "plural"),
        "tense (past/present)": (TENSE_PAST, TENSE_PRES, "past", "present"),
        "person (1st/3rd)":     (PERSON_1, PERSON_3, "1st", "3rd"),
        "sentence (q/stmt)":    (QUESTION, STATEMENT, "question", "statement"),
        "sentiment (pos/neg)":  (DEFAULT_POS, DEFAULT_NEG, "positive", "negative"),
    }
    ensure_server()
    a0, _t, lyr = harvest_text("The cat sat.", layer=layer)
    print(f"  /harvest live for concept-alignment: layer={lyr}, n_embd={a0.shape[1]}", flush=True)
    rows = []
    for name, (pos, neg, pl, nl) in concepts.items():
        acts, labels, reps = [], [], []
        for grp, lab in ((pos, 1), (neg, 0)):
            for t in grp:
                a, toks, _ = harvest_text(t, layer=layer)
                acts.append(a[-1])                       # final-token = sentence rep
                labels.append(lab)
                reps.append(toks[-1] if toks else "")
        A = np.stack(acts).astype(np.float32)
        y = np.array(labels)
        As = (A - mu) / sd                               # SAME standardization as training
        codes = sae.encode_rows(As)                      # [2n, m]
        proj = As @ Vt[:K].T                             # [2n, K]
        sj, sae_ho, sae_null = _best_feature_auc_held_out(codes, y, min_fire=8)
        pj, pca_ho, pca_null = _best_feature_auc_held_out(proj, y, min_fire=0)
        top_reps = []
        if sj >= 0:
            topfire = np.argsort(codes[:, sj])[::-1][:10]
            top_reps = [reps[i] for i in topfire if codes[i, sj] > 1e-6]
        sae_probe = kfold_accuracy(list(codes), list(y.astype(float)), k=6, ridge=10.0)
        pca_probe = kfold_accuracy(list(proj), list(y.astype(float)), k=6, ridge=10.0)
        raw_probe = kfold_accuracy(list(As), list(y.astype(float)), k=6, ridge=10.0)
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
              f"fires {rec['sae_best_fires_on_n']}/{len(y)}, "
              f"{rec['sae_best_distinct_top_tokens']} distinct toks) vs PCA={pca_ho:.2f} "
              f"(null {pca_null:.2f})", flush=True)
        print(f"  {'':22}  whole-repr probe: SAE={sae_probe:.2f} PCA={pca_probe:.2f} "
              f"raw={raw_probe:.2f}", flush=True)
    return rows


def reconstruct_context(pieces, i, window=14):
    lo, hi = max(0, i - window), min(len(pieces), i + window + 1)
    left = "".join(pieces[lo:i])
    right = "".join(pieces[i + 1:hi])
    return (left + "<<" + pieces[i] + ">>" + right).replace("\n", " ").strip()


# =================================================================================================
def self_gate(target=5000):
    """A small L16 harvest + a tiny SAE BEFORE the big run: confirm [n,3584] alignment, the SAE
    reconstructs (MSE drops well below 1.0), and features come alive. STOP the engine at the end."""
    import torch
    print("=== SELF-GATE: small layer-16 harvest + tiny SAE (Qwen2.5-7B) ===", flush=True)
    X, pieces, layer = harvest_corpus_layer(target_rows=target, layer=LAYER, store_fp16=True)
    assert X.shape[1] == 3584, f"expected n_embd 3584, got {X.shape[1]}"
    Xs, mu, sd = standardize(X.astype(np.float32))
    Xc, n_clip = clip_outlier_rows(Xs)
    tvar = float((Xc ** 2).mean())
    print(f"  standardized {Xs.shape}; finite? {bool(np.isfinite(Xs).all())}; "
          f"clipped {n_clip} outlier rows; target var (mean-predictor MSE) = {tvar:.3f}", flush=True)
    kill_server()  # free VRAM before the (tiny) GPU SAE
    Xhost = torch.tensor(Xc, dtype=torch.float16)
    d = Xs.shape[1]
    # L1=16 sits in the genuinely-sparse band at this scale (the self-gate sweep: fire 18%, MSE<target).
    sae = StreamingSAE(d, m=4 * d, l1=16.0, seed=0).fit(Xhost, epochs=25, lr=3e-3, batch_size=512)
    fire, mse = sae.stats(Xhost)
    live_mask = (fire >= 0.002) & (fire <= 0.4)
    live = int(live_mask.sum())
    top_idx, _ = sae.codes_topk(Xhost, topn=20)
    coh, modal, tops = sae_coherence_from_topk(top_idx, pieces, topn=20)
    reconstructs = mse < tvar  # honest bar: beat the mean-predictor (not a fixed 1.0)
    print(f"  tiny SAE 4x (m={4*d}): MSE={mse:.4f} (target {tvar:.3f}, "
          f"{'reconstructs' if reconstructs else 'NOT reconstructing'}), "
          f"mean_fire={fire.mean()*100:.2f}%, live={live}/{4*d}, "
          f"coh(live)={coh[live_mask].mean()*100:.1f}%", flush=True)
    ok = reconstructs and (live >= 200)
    print(f"  SELF-GATE {'PASS' if ok else 'FAIL'} "
          f"(MSE<target and >=200 live features): proceed={ok}", flush=True)
    return ok


def main():
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = 700_000
    if "--target" in sys.argv:
        target = int(sys.argv[sys.argv.index("--target") + 1])
    elif pos:
        target = int(pos[0])
    os.makedirs(RUNS, exist_ok=True)

    if "--selfgate" in flags:
        ok = self_gate()
        kill_server()
        sys.exit(0 if ok else 2)

    import torch
    # ---- 1. harvest (or load cache) --------------------------------------------------------------
    if "--from-cache" in flags and os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        X, pieces = d["X"], list(d["pieces"])
        layer = int(d["layer"]) if "layer" in d else LAYER
        print(f"=== loaded {X.shape[0]} cached rows x {X.shape[1]} dims "
              f"(layer {layer}, {X.dtype}, 7B natural) ===", flush=True)
    else:
        print(f"=== harvesting ~{target} NATURAL-text residuals from Qwen2.5-7B @ L{LAYER} "
              f"via /harvest ===", flush=True)
        X, pieces, layer = harvest_corpus_layer(target_rows=target, layer=LAYER, store_fp16=True)
        np.savez_compressed(CACHE, X=X, pieces=np.array(pieces, dtype=object), layer=layer)
        print(f"saved {CACHE} ({os.path.getsize(CACHE)/1e9:.2f} GB)", flush=True)

    # ---- 2. STOP the engine to free its 7.6 GB before the GPU SAE --------------------------------
    print("stopping engine to free VRAM for SAE training...", flush=True)
    kill_server()

    # standardize + winsorize the massive-activation rows, CHUNKED (the naive path makes two ~14 GB
    # fp32 copies at 1M rows and thrashes a 31 GB box into ~60 GB pagefile). standardize_clip_chunked
    # returns the fp16 matrix directly; PCA uses the SAME clipped matrix (apples-to-apples).
    Xc16, mu, sd, n_clip, tvar = standardize_clip_chunked(X, sigma=CLIP_SIGMA)
    del X
    d = Xc16.shape[1]
    N = Xc16.shape[0]
    uniq = len(set(_norm(p) for p in pieces if p.strip()))
    print(f"\ncorpus: {N} rows, {d} dims, {uniq} unique tokens (unique ratio {uniq/max(N,1):.3f}), "
          f"tap layer {layer}; clipped {n_clip} massive-activation rows; "
          f"target var (mean-predictor MSE) = {tvar:.3f}", flush=True)

    # ---- 3. PCA baseline at matched component count (chunked covariance eigh; on the SAME matrix) --
    K = 256
    print(f"\n=== PCA baseline (top-{K} axes) ===", flush=True)
    t0 = time.time()
    Vt, var = pca_basis_chunked(Xc16)
    pca_coh, pca_modal, pca_tops = project_coherence_chunked(Xc16, Vt, K, pieces)
    pca_mean = float(pca_coh.mean())
    pca_mean64 = float(pca_coh[:64].mean())
    print(f"  PCA top-{K} mean coherence = {pca_mean*100:.1f}%  (top-64 = {pca_mean64*100:.1f}%)  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- GPU-RESIDENT training data (the 136s/epoch -> ~6s/epoch fix) ----------------------------
    # Host-streaming each minibatch (CPU fancy-index of a 1M x 3584 fp16 tensor + PCIe transfer) cost
    # 136s/epoch -> ~7h for the full 6-config sweep. Uploading the corpus to the GPU ONCE makes every
    # step a pure-GPU gather (~20x faster). 1M x 3584 fp16 = 7.2 GB; with the 16x SAE (~5 GB params+
    # Adam) that overflows the 16 GB card, so we cap the TRAIN set to GPU_CAP rows (a uniform random
    # subsample). 250k is ~2x the 0.5B run's 120k (MORE tokens than the prior baseline, not fewer) and
    # 250k x 3584 fp16 = 1.8 GB leaves ~12 GB for the 16x SAE; even GPU-resident the m=57k matmuls over
    # 1M rows would make the 6-config sweep ~2h, so the cap also keeps it tractable (~40 min). The eval
    # (coherence/contexts) uses these same capped rows + pieces so everything is aligned; PCA was scored
    # on the FULL 1M harvest (its baseline is if anything stronger — the SAE is the one held to fewer
    # rows, the conservative direction for the verdict).
    GPU_CAP = 250_000
    if "--gpu-cap" in sys.argv:
        GPU_CAP = int(sys.argv[sys.argv.index("--gpu-cap") + 1])
    if N > GPU_CAP:
        sub = np.random.default_rng(0).choice(N, size=GPU_CAP, replace=False)
        sub.sort()
        Xc16 = Xc16[sub]
        pieces = [pieces[i] for i in sub]
        N = GPU_CAP
        print(f"  train subsample: {N} rows uploaded GPU-resident (eval uses these rows + pieces); "
              f"PCA was scored on the FULL harvested set", flush=True)
    Xgpu = torch.tensor(Xc16, dtype=torch.float16, device="cuda")  # one-time upload
    del Xc16
    Xhost = Xgpu  # the training/eval helpers accept either; this one is GPU-resident (fast path)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 4. the real SAE: 16x (m~57k), 8x fallback, L1 dose-response ------------------------------
    # Training config CALIBRATED on this exact 7B/L16 matrix (the self-gate): lr=3e-3 (lr=1e-2 diverges
    # on the attention-sink outliers; lr=1e-3 is too slow to leave the random-init regime at this width),
    # encoder-col-unit-norm + decoder=W.T init (a plain randn init stalls at mean_fire~50%), batch 512,
    # 40 epochs. The L1 grid is HIGHER than the 0.5B run's: at 7B/L16/3584 real sparsity only sets in
    # around L1=16-30 (the self-gate showed fire 49%/18%/0.6% at L1 8/16/30) — we sweep that band so a
    # genuinely-sparse, >=200-live config is reachable. The full dose-response prints regardless.
    L1_GRID = (8.0, 16.0, 30.0)
    if "--l1" in sys.argv:  # override grid, e.g. --l1 0.5,1,2,4 (a LOWER sweep to find a converged config)
        L1_GRID = tuple(float(x) for x in sys.argv[sys.argv.index("--l1") + 1].split(","))
    EXPANSIONS = (16, 8) if "--exp8only" not in flags else (8,)
    if "--exp" in sys.argv:  # override expansions, e.g. --exp 16
        EXPANSIONS = tuple(int(x) for x in sys.argv[sys.argv.index("--exp") + 1].split(","))
    BATCH, LR = 512, 1e-3  # lr=3e-3 DIVERGED at 16x (MSE->8143); 1e-3 + grad-clip is the stable config
    steps_per_epoch = max(1, (N + BATCH - 1) // BATCH)
    # Target ~12k grad steps/config (~1.3x the 0.5B run's ~9.4k; with grad-clip the SAE converges
    # steadily and MSE plateaus well before this, so more steps just burn GPU time at 34s/epoch). At
    # 250k rows that's ~25 epochs; GPU-resident, ~25 epochs * 6 configs runs in ~50 min.
    EPOCHS = int(min(30, max(12, (12000 + steps_per_epoch - 1) // steps_per_epoch)))
    print(f"\nSAE training: batch={BATCH} lr={LR} epochs={EPOCHS} "
          f"({steps_per_epoch} steps/epoch -> ~{EPOCHS*steps_per_epoch} grad steps/config)", flush=True)
    results = []
    best = None
    for exp in EXPANSIONS:
        m = exp * d
        print(f"\n=== SAE {exp}x expansion (m={m}) ===", flush=True)
        for l1 in L1_GRID:
            t0 = time.time()
            try:
                sae = StreamingSAE(d, m=m, l1=l1, seed=0).fit(
                    Xhost, epochs=EPOCHS, lr=LR, batch_size=BATCH)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [SAE {exp}x l1={l1}] OOM at m={m} -- skipping this expansion", flush=True)
                    torch.cuda.empty_cache()
                    break
                raise
            fire, mse = sae.stats(Xhost)
            live_mask = (fire >= 0.002) & (fire <= 0.4)
            n_live = int(live_mask.sum())
            top_idx, _ = sae.codes_topk(Xhost, topn=20)
            coh, modal, tops = sae_coherence_from_topk(top_idx, pieces, topn=20)
            mean_coh = float(coh[live_mask].mean()) if n_live else 0.0
            dt = time.time() - t0
            converged = mse < tvar  # honest bar: beat the mean-predictor (not a fixed 1.0)
            # guard against the degenerate "high coherence over a handful of features" trap (p6's rule)
            enough_live = n_live >= 200
            rec = {"exp": exp, "m": int(m), "l1": l1, "mean_fire": float(fire.mean()),
                   "n_live": n_live, "coherence": mean_coh, "mse": float(mse),
                   "converged": converged, "enough_live": enough_live, "secs": dt}
            results.append(rec)
            flag = "" if converged else f"  <-- NOT CONVERGED (MSE>={tvar:.2f}, ignore coherence)"
            if converged and not enough_live:
                flag = "  <-- <200 live (coherence unreliable)"
            print(f"  [SAE {exp}x l1={l1}] live={n_live:<6} mean_fire={fire.mean()*100:5.2f}% "
                  f"coh(live)={mean_coh*100:4.1f}% mse={mse:.3f}  [{dt:.0f}s]{flag}", flush=True)
            sparse_enough = fire.mean() <= 0.10
            score = (1 if converged else 0, 1 if enough_live else 0, 1 if sparse_enough else 0,
                     mean_coh)
            if best is None or score > best["_score"]:
                if best is not None and "_sae" in best:
                    del best["_sae"]  # free the previous best SAE's GPU params before keeping this one
                best = dict(exp=exp, m=int(m), l1=l1, fire=fire, live_mask=live_mask, coh=coh,
                            modal=modal, tops=tops, top_idx=top_idx, mean_coh=mean_coh,
                            n_live=n_live, mse=float(mse), converged=converged, _score=score,
                            _sae=sae)  # keep this SAE for concept-alignment (don't delete it)
            else:
                del sae
            torch.cuda.empty_cache()
    best_sae = best["_sae"]

    print(f"\n=== verdict inputs ===", flush=True)
    print(f"  PCA top-{K} mean coherence : {pca_mean*100:.1f}%  (top-64 {pca_mean64*100:.1f}%)")
    print(f"  SAE best ({best['exp']}x l1={best['l1']}) live mean coherence: "
          f"{best['mean_coh']*100:.1f}% over {best['n_live']} live features (MSE {best['mse']:.3f})")
    gap = best["mean_coh"] - pca_mean
    gap64 = best["mean_coh"] - pca_mean64
    verdict = ("SAE ahead" if gap > 0.02 and gap64 > 0.02 else
               ("PCA ahead" if gap < -0.02 or gap64 < -0.02 else "MIXED/TIE"))
    print(f"  SAE - PCA(top256) gap: {gap*100:+.1f} pts ; SAE - PCA(top64) gap: {gap64*100:+.1f} pts "
          f"-> {verdict}")

    # ---- name 10 example features ----------------------------------------------------------------
    live_idx = np.where(best["live_mask"])[0]
    ranked = sorted(live_idx, key=lambda j: -best["coh"][j])
    feats = []
    print(f"\n=== 10 most-coherent discovered SAE features ({best['exp']}x, layer {layer}) ===")
    for j in ranked[:10]:
        tt = best["tops"][j]
        print(f"  f{j:<6} coh={best['coh'][j]*100:3.0f}%  fires={best['fire'][j]*100:5.2f}%  "
              f"modal={best['modal'][j]!r:<16} top: {' '.join(repr(t) for t in tt[:10])}", flush=True)
        feats.append(Feature(int(j), "sae", tt[:10], float(best["fire"][j])))

    print("\n=== 8 PCA axes for contrast ===")
    pca_feats = []
    for j in range(8):
        tt = pca_tops[j]
        print(f"  PC{j:<3} coh={pca_coh[j]*100:3.0f}%  var={var[j]*100:4.1f}%  "
              f"modal={pca_modal[j]!r:<16} top: {' '.join(repr(t) for t in tt[:10])}", flush=True)
        pca_feats.append(Feature(int(j), "pca", tt[:10], float(var[j])))

    # ---- 5. semantic / concept-alignment metric (relaunch engine, harvest probe corpora) ---------
    # Free the GPU-resident training matrix first so the 7B engine (needs ~10 GB) has room to relaunch;
    # the alignment only needs the best SAE's encoder (kept) + the tiny probe-corpus activations.
    try:
        del Xgpu, Xhost
    except NameError:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    align = []
    if "--no-concepts" not in flags:
        print(f"\n=== concept alignment @ L{layer}: SAE features vs PCA axes on semantic minimal "
              f"pairs (relaunching engine) ===", flush=True)
        try:
            align = run_concept_alignment(best_sae, mu, sd, Vt, K, layer)
        except Exception as e:  # noqa: BLE001
            print(f"  concept-alignment skipped (engine error: {type(e).__name__}: {e})", flush=True)
        finally:
            kill_server()

    # ---- auto-interp contexts for the candidate features (3-way ranking, like p6) ----------------
    fire = best["fire"]
    coh = best["coh"]
    top_idx = best["top_idx"]
    live_mask = best["live_mask"]
    live_idx = np.where(live_mask)[0]
    rank_coh = sorted(live_idx, key=lambda j: -coh[j])
    rank_density = sorted(live_idx, key=lambda j: -fire[j])
    candidates, seen = [], set()
    for ranking, tag in ((rank_coh, "coh"), (rank_density, "density")):
        for j in ranking[:15]:
            if j not in seen:
                seen.add(j)
                candidates.append((int(j), tag))
    ctx_out = []
    for j, tag in candidates:
        rows_j = [int(i) for i in top_idx[j] if i >= 0][:10]
        ctx_out.append({
            "feature": j, "surfaced_by": tag, "token_coherence": float(coh[j]),
            "modal_token": best["modal"][j], "fires_pct": float(fire[j]),
            "top_tokens": [pieces[i] for i in rows_j],
            "top_contexts": [reconstruct_context(pieces, i) for i in rows_j],
            "distinct_top_tokens": len(set(_norm(pieces[i]) for i in rows_j)),
        })

    # ---- 6. save artifacts -----------------------------------------------------------------------
    out_html = os.path.join(RUNS, "discovered_7b_sae.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(render_discovered_features(
            feats, title="Clozn - 7B-scale SAE Discovery (Qwen2.5-7B, layer 16, natural text)",
            subtitle=(f"{N} WikiText residual tokens via /harvest (layer {layer}, n_embd {d}) - "
                      f"SAE {best['exp']}x m={best['m']} l1={best['l1']} - top-token coherence "
                      f"SAE {best['mean_coh']*100:.0f}% vs PCA {pca_mean*100:.0f}%")))
    print(f"\nwrote {out_html}")

    summary = {
        "model": "Qwen2.5-7B-Instruct q8_0", "rows": int(N), "dim": int(d), "tap_layer": int(layer),
        "harvest": "natural WikiText-103, engine /harvest (one causal forward per passage)",
        "unique_token_ratio": uniq / max(N, 1),
        "pca_K": K, "pca_mean_coherence": pca_mean, "pca_mean_coherence_top64": pca_mean64,
        "sae_best": {"exp": best["exp"], "m": best["m"], "l1": best["l1"],
                     "mean_coherence": best["mean_coh"], "n_live": best["n_live"],
                     "mse": best["mse"]},
        "sae_grid": results,
        "sae_minus_pca_gap_top256": gap, "sae_minus_pca_gap_top64": gap64,
        "verdict_token_coherence": verdict,
        "examples_sae": [{"idx": int(j), "coherence": float(best["coh"][j]),
                          "fires": float(best["fire"][j]), "modal": best["modal"][j],
                          "top": best["tops"][j][:10],
                          "distinct_top_tokens": len(set(_norm(t) for t in best["tops"][j][:20]))}
                         for j in ranked[:10]],
        "examples_pca": [{"idx": j, "coherence": float(pca_coh[j]), "var": float(var[j]),
                          "modal": pca_modal[j], "top": pca_tops[j][:10]} for j in range(8)],
        "concept_alignment": align,
        "auto_interp_contexts": ctx_out,
        "prior_runs": {"qwen05b_L2_bigSAE": {"sae": 0.447, "pca_top256": 0.415, "pca_top64": 0.548},
                       "qwen05b_L2_toySAE": {"sae": 0.40, "pca": 0.44},
                       "toy_rwkv": {"sae": 0.65, "pca": 0.12}},
    }
    out_json = os.path.join(RUNS, "discovered_7b_sae.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out_json, flush=True)

    if align:
        sae_unit_wins = sum(1 for r in align
                            if r["sae_best_auc_heldout"] > r["sae_best_auc_null"] + 0.10
                            and r["sae_best_auc_heldout"] > r["pca_best_auc_heldout"] + 0.03)
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


if __name__ == "__main__":
    main()
