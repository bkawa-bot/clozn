"""
Phase-3 §3.5 — does a TRANSCODER beat the SAE/PCA null at real scale (Qwen2.5-0.5B)?

THE QUESTION. Residual SAEs were a robust null at scale (sae_at_scale_findings.md): a converged
16-32x SAE on 120k natural-text tokens scored 44.7% top-token coherence vs PCA 41.5%/54.8% — no
monosemanticity advantage; the discovered units were token-identity detectors. The field's current
SOTA interp substrate is the TRANSCODER: a sparse stand-in for a component's INPUT->OUTPUT map (vs
the SAE's in->in reconstruction). The toy notes hinted transcoders edged SAEs at early layers (66%
vs 62% on RWKV channel-mix). Does that edge survive at real scale, or is the sparse-dictionary
approach null on this 0.5B early/mid layer regardless?

THE SUBSTRATE (simplest viable transcoder, NO engine change). A layer-residual transcoder: harvest
the SAME corpus at TWO layers via /harvest — input = l_out-L (L=2, the calibrated early tap), output
= l_out-L' (L'=6, a few residual steps downstream). The forward is deterministic, so the two
activation matrices are token-aligned (row r is the same token in both). Train a sparse map
input -> sparse code (relu, L1) -> reconstruct output. This is IN->OUT, unlike the SAE's in->in.

APPLES-TO-APPLES. All three methods are judged on representing the SAME target (the L'=6 output):
  * transcoder code = relu(Xin_L2 @ We + be), trained to reconstruct Xout_L6;
  * SAE          code = relu(Xout_L6 @ We + be), trained to reconstruct Xout_L6 (in->in);
  * PCA          proj = Xout_L6 @ Vt[:K].T (top-K axes of the output).
Each is scored by the IDENTICAL un-seeded top-token coherence metric as the SAE run (per feature,
take its top-N activating tokens; coherence = fraction equal to the feature's modal top token).
Honesty-first: a null (transcoder ties/loses like the SAE) is a valid, reportable outcome. We print
the full L1 dose-response and do not cherry-pick. The metric rewards TOKEN-LOCKING, not abstraction
(loud caveat carried over) — a semantic / auto-interp metric would be the real test.

Reuses p4_big_sae's server management, /harvest, corpus assembly, coherence metric, and PCA; reuses
discover.standardize. Adds a GPU TorchTranscoder (streaming top-k, no 120k x m host matrix) that is
the exact TinySAE objective with a SEPARATE target Y (in->out).

Usage (from inspector/, cloze venv python; the spawned server's PATH gets the engine bin + CUDA bin):
    python spikes/p5_transcoder_scale.py                # full: harvest ~120k rows at 2 layers, train, eval
    python spikes/p5_transcoder_scale.py --from-cache    # re-analyze the saved two-layer matrix
    python spikes/p5_transcoder_scale.py --target 60000  # smaller harvest
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from clozn.discover import Feature, standardize  # noqa: E402
from clozn.viz import render_discovered_features  # noqa: E402

# Reuse p4_big_sae wholesale: server mgmt, /harvest (now layer-aware), corpus, coherence, PCA, TorchSAE.
import spikes.p4_big_sae as big  # noqa: E402
from spikes.p4_big_sae import (  # noqa: E402
    TorchSAE, harvest_text, kill_server, sae_coherence_from_topk, start_server,
    top_token_coherence, wikitext_passages, _norm, RUNS,
)

L_IN = 2     # input layer: the engine's calibrated early tap (l_out-2) — same as the SAE run
L_OUT = 6    # output layer: a few residual steps downstream (l_out-6); both valid mid-layer taps (1..23)
CACHE = os.path.join(RUNS, "qwen_transcoder_2layer_acts.npz")


# ---- two-layer harvest (token-aligned input L_IN + output L_OUT) --------------------------------
def harvest_two_layer_corpus(target_rows: int):
    """Harvest ~target_rows natural-text residuals at BOTH L_IN and L_OUT, token-aligned. One pair of
    /harvest forwards per passage (deterministic => same tokens). Auto-restarts the server only on a
    real CONNECTION failure (a 400 just skips that passage). Returns (Xin, Xout, pieces)."""
    start_server()
    a_in, p_in, li = harvest_text("The mitochondria is the powerhouse of the cell.", layer=L_IN)
    a_out, p_out, lo = harvest_text("The mitochondria is the powerhouse of the cell.", layer=L_OUT)
    assert p_in == p_out and li == L_IN and lo == L_OUT, "two-layer harvest not aligned at probe"
    print(f"  /harvest live: L_in={li} L_out={lo}, n_embd={a_in.shape[1]} (probe aligned, "
          f"rows={a_in.shape[0]})", flush=True)
    Xin: list[np.ndarray] = []
    Xout: list[np.ndarray] = []
    pieces: list[str] = []
    crashes = skipped = n_pass = 0
    t0 = time.time()
    gen = wikitext_passages()
    while len(Xin) < target_rows and crashes < 40:
        try:
            text = next(gen)
        except StopIteration:
            gen = wikitext_passages()
            text = next(gen)
        try:
            ai, ti, _ = harvest_text(text, layer=L_IN)
            ao, to, _ = harvest_text(text, layer=L_OUT)
            # Token alignment is guaranteed by the deterministic forward; assert it cheaply and skip
            # the (vanishingly rare) misaligned passage rather than corrupt the matrix.
            if ti != to or ai.shape != ao.shape:
                skipped += 1
                continue
            n = min(ai.shape[0], len(ti))
            for r in range(n):
                Xin.append(ai[r]); Xout.append(ao[r]); pieces.append(ti[r])
            n_pass += 1
            if n_pass % 50 == 0:
                rate = len(Xin) / max(time.time() - t0, 1e-6)
                print(f"  passage {n_pass}: {len(Xin)} rows ({rate:.0f} rows/s)", flush=True)
        except urllib.error.HTTPError as e:
            skipped += 1
            if skipped <= 5:
                print(f"  passage {n_pass}: HTTP {e.code} (skipped, server OK)", flush=True)
        except (urllib.error.URLError, ConnectionResetError, OSError) as e:
            crashes += 1
            print(f"  passage {n_pass}: connection error ({type(e).__name__}); restart #{crashes}",
                  flush=True)
            start_server()
    Xi = np.stack(Xin).astype(np.float32)
    Xo = np.stack(Xout).astype(np.float32)
    print(f"harvested {Xi.shape[0]} rows x {Xi.shape[1]} dims at layers {L_IN}&{L_OUT} from {n_pass} "
          f"passages in {time.time()-t0:.0f}s ({crashes} restart(s), {skipped} skipped; NATURAL "
          f"WikiText)", flush=True)
    return Xi, Xo, pieces


# ---- GPU transcoder: TinySAE objective with a SEPARATE target Y (in->out), scaled like TorchSAE ---
class TorchTranscoder(TorchSAE):
    """A transcoder = the exact TorchSAE math, but the sparse code is computed from the INPUT
    activations while the reconstruction target is the OUTPUT activations (a different layer). So
    f = relu(Xin @ We + be); recon = f @ Wd + bd; loss = MSE(recon, Xout) + L1*|f|; unit-norm decoder.
    Encoder dim d_in, decoder dim d_out (may differ; here both 896). All the streaming/top-k scaling of
    TorchSAE is inherited — codes_topk/stats run on the INPUT (the code is an input function)."""

    def __init__(self, d_in: int, d_out: int, m: int, l1: float = 1.0, seed: int = 0,
                 device: str = "cuda"):
        import torch
        self.torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        g = torch.Generator().manual_seed(seed)
        self.We = (torch.randn(d_in, m, generator=g) * 0.1).to(self.device).requires_grad_(True)
        self.be = torch.zeros(m, device=self.device, requires_grad=True)
        self.Wd = (torch.randn(m, d_out, generator=g) * 0.1).to(self.device).requires_grad_(True)
        self.bd = torch.zeros(d_out, device=self.device, requires_grad=True)
        self.l1 = l1
        self.m = m

    def fit(self, Xin, Xout, epochs: int = 40, lr: float = 1e-3, batch_size: int = 512):
        torch = self.torch
        Xi = torch.tensor(Xin, dtype=torch.float32, device=self.device)
        Xo = torch.tensor(Xout, dtype=torch.float32, device=self.device)
        opt = torch.optim.Adam([self.We, self.be, self.Wd, self.bd], lr=lr)
        n = Xi.shape[0]
        torch.manual_seed(0)
        last = 0.0
        for _ in range(epochs):
            perm = torch.randperm(n, device=self.device)
            tot = 0.0
            nb = 0
            for s in range(0, n, batch_size):
                idx = perm[s:s + batch_size]
                xb, yb = Xi[idx], Xo[idx]
                f = self._encode(xb)
                recon = f @ self.Wd + self.bd
                loss = ((recon - yb) ** 2).mean() + self.l1 * f.abs().mean()
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    self.Wd.data /= (self.Wd.data.norm(dim=1, keepdim=True) + 1e-8)
                    tot += float(((recon - yb) ** 2).mean())
                nb += 1
            last = tot / max(nb, 1)
        self._final_mse = last
        return self

    def stats(self, Xin, Xout, batch_size: int = 8192):
        """Per-feature fire-rate (on the INPUT code) + mean recon MSE against the OUTPUT, one pass."""
        torch = self.torch
        Xi = torch.tensor(Xin, dtype=torch.float32, device=self.device)
        Xo = torch.tensor(Xout, dtype=torch.float32, device=self.device)
        n = Xi.shape[0]
        fire = torch.zeros(self.m, device=self.device)
        mse_tot = 0.0
        nb = 0
        with torch.no_grad():
            for s in range(0, n, batch_size):
                xb, yb = Xi[s:s + batch_size], Xo[s:s + batch_size]
                f = self._encode(xb)
                fire += (f > 1e-6).float().sum(0)
                recon = f @ self.Wd + self.bd
                mse_tot += float(((recon - yb) ** 2).mean())
                nb += 1
        return (fire / n).cpu().numpy(), mse_tot / max(nb, 1)
    # codes_topk(Xin) is inherited verbatim: the code is relu(Xin @ We + be), an INPUT function — so the
    # transcoder's "features" are described by the INPUT tokens that drive them, exactly as a SAE/PCA
    # feature is described by the tokens that drive its activation. Apples-to-apples on the same metric.


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
        Xin, Xout, pieces = d["Xin"], d["Xout"], list(d["pieces"])
        print(f"=== loaded {Xin.shape[0]} cached rows x {Xin.shape[1]} dims at layers {L_IN}&{L_OUT} "
              f"(natural) ===", flush=True)
    else:
        print(f"=== harvesting ~{target} NATURAL-text residuals at TWO layers (L_in={L_IN}, "
              f"L_out={L_OUT}) via /harvest ===", flush=True)
        Xin, Xout, pieces = harvest_two_layer_corpus(target_rows=target)
        np.savez_compressed(CACHE, Xin=Xin, Xout=Xout, pieces=np.array(pieces, dtype=object),
                            l_in=L_IN, l_out=L_OUT)
        print(f"saved {CACHE}", flush=True)
        kill_server()

    # Standardize input and output INDEPENDENTLY (per-feature). The transcoder reconstructs the OUTPUT
    # distribution; the SAE/PCA baselines also operate on the standardized OUTPUT — so all three are
    # judged on representing the SAME standardized target.
    Xin_s, _, _ = standardize(Xin)
    Xout_s, _, _ = standardize(Xout)
    d_in, d_out = Xin_s.shape[1], Xout_s.shape[1]
    N = Xin_s.shape[0]
    uniq = len(set(_norm(p) for p in pieces if p.strip()))
    # How non-trivial is the in->out map? copy-input MSE (standardized) vs the target's own variance.
    copy_mse = float(np.mean((Xin_s - Xout_s) ** 2))
    var_out = float(np.mean(Xout_s ** 2))
    print(f"\ncorpus: {N} rows, in dim {d_in}, out dim {d_out}, {uniq} unique tokens "
          f"(unique ratio {uniq/max(N,1):.3f})", flush=True)
    print(f"in->out map non-triviality: copy-input MSE={copy_mse:.3f} vs target var={var_out:.3f} "
          f"(a transcoder must beat copy-input, not just the mean)", flush=True)

    # ---- PCA baseline on the OUTPUT (top-K axes of l_out-L') --------------------------------------
    K = 256
    print(f"\n=== PCA baseline on OUTPUT layer {L_OUT} (top-{K} axes) ===", flush=True)
    t0 = time.time()
    if N > 20000:
        C = (Xout_s.T @ Xout_s) / N
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        Vt = evecs[:, order].T
        var = evals[order] / evals.sum()
    else:
        U, S, Vt = np.linalg.svd(Xout_s, full_matrices=False)
        var = (S ** 2) / (S ** 2).sum()
    pca_proj = Xout_s @ Vt[:K].T
    pca_coh, pca_modal, pca_tops = top_token_coherence(pca_proj, pieces, topn=20)
    pca_mean = float(pca_coh.mean())
    pca_mean64 = float(pca_coh[:64].mean())
    print(f"  PCA top-{K} mean coherence = {pca_mean*100:.1f}%  (top-64 = {pca_mean64*100:.1f}%)  "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- SAE baseline on the OUTPUT (in->in on l_out-L'), same config as the SAE run --------------
    BATCH, LR = 512, 1e-3
    steps_per_epoch = max(1, (N + BATCH - 1) // BATCH)
    EPOCHS = int(min(200, max(40, (9000 + steps_per_epoch - 1) // steps_per_epoch)))
    L1_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)
    # Honesty guard against a DEGENERATE winner: at low L1 the dictionary is dense (mean fire ~40%)
    # and the [0.002,0.4] live-band lets through only a HANDFUL of features — a "coherence" averaged
    # over 1-6 features is noise, not a result, and it would falsely win the verdict. Require a healthy
    # live count before a config can be the reported best (the dose-response still prints every config).
    MIN_LIVE = 200
    print(f"\ntraining config: batch={BATCH} lr={LR} epochs={EPOCHS} "
          f"({steps_per_epoch} steps/epoch -> ~{EPOCHS*steps_per_epoch} grad steps/config); "
          f"reported-best requires >= {MIN_LIVE} live features (anti-degenerate guard)", flush=True)

    def run_sae(label, expansions=(16,)):
        rows, best = [], None
        for exp in expansions:
            m = exp * d_out
            print(f"\n=== {label} {exp}x expansion (m={m}) on OUTPUT layer {L_OUT} ===", flush=True)
            for l1 in L1_GRID:
                t = time.time()
                sae = TorchSAE(d_out, m=m, l1=l1, seed=0).fit(
                    Xout_s, epochs=EPOCHS, lr=LR, batch_size=BATCH)
                fire, mse = sae.stats(Xout_s)
                live_mask = (fire >= 0.002) & (fire <= 0.4)
                n_live = int(live_mask.sum())
                top_idx, _ = sae.codes_topk(Xout_s, topn=20)
                coh, modal, tops = sae_coherence_from_topk(top_idx, pieces, topn=20)
                mean_coh = float(coh[live_mask].mean()) if n_live else 0.0
                converged = mse < 1.0
                rec = {"exp": exp, "m": int(m), "l1": l1, "mean_fire": float(fire.mean()),
                       "n_live": n_live, "coherence": mean_coh, "mse": float(mse),
                       "converged": converged, "secs": time.time() - t}
                rows.append(rec)
                flag = "" if converged else "  <-- NOT CONVERGED (MSE>=1)"
                if n_live < MIN_LIVE:
                    flag += f"  (only {n_live} live; ineligible as best)"
                print(f"  [{label} {exp}x l1={l1}] live={n_live:<6} mean_fire={fire.mean()*100:5.2f}% "
                      f"coh(live)={mean_coh*100:4.1f}% mse={mse:.3f}  [{rec['secs']:.0f}s]{flag}",
                      flush=True)
                sparse_enough = fire.mean() <= 0.10
                # selection: converged AND enough live features come first, then sparsity, then coherence.
                enough_live = n_live >= MIN_LIVE
                score = (1 if converged else 0, 1 if enough_live else 0,
                         1 if sparse_enough else 0, mean_coh)
                if best is None or score > best["_score"]:
                    best = dict(exp=exp, m=int(m), l1=l1, fire=fire, live_mask=live_mask, coh=coh,
                                modal=modal, tops=tops, mean_coh=mean_coh, n_live=n_live,
                                mse=float(mse), converged=converged, _score=score)
                del sae
                import torch as _t
                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
        return rows, best

    sae_rows, sae_best = run_sae("SAE", expansions=(16,))

    # ---- the TRANSCODER: code from INPUT layer L_IN, reconstruct OUTPUT layer L_OUT --------------
    def run_transcoder(expansions=(16,)):
        rows, best = [], None
        for exp in expansions:
            m = exp * d_out
            print(f"\n=== TRANSCODER {exp}x (m={m}): code from L_in={L_IN} -> reconstruct "
                  f"L_out={L_OUT} ===", flush=True)
            for l1 in L1_GRID:
                t = time.time()
                tc = TorchTranscoder(d_in, d_out, m=m, l1=l1, seed=0).fit(
                    Xin_s, Xout_s, epochs=EPOCHS, lr=LR, batch_size=BATCH)
                fire, mse = tc.stats(Xin_s, Xout_s)
                live_mask = (fire >= 0.002) & (fire <= 0.4)
                n_live = int(live_mask.sum())
                top_idx, _ = tc.codes_topk(Xin_s, topn=20)   # code is an INPUT function
                coh, modal, tops = sae_coherence_from_topk(top_idx, pieces, topn=20)
                mean_coh = float(coh[live_mask].mean()) if n_live else 0.0
                # honesty guard: the transcoder must beat copy-input (not just the mean) to be a real map.
                converged = mse < copy_mse
                rec = {"exp": exp, "m": int(m), "l1": l1, "mean_fire": float(fire.mean()),
                       "n_live": n_live, "coherence": mean_coh, "mse": float(mse),
                       "beats_copy_input": converged, "secs": time.time() - t}
                rows.append(rec)
                flag = "" if converged else "  <-- does NOT beat copy-input (weak map)"
                if n_live < MIN_LIVE:
                    flag += f"  (only {n_live} live; ineligible as best)"
                print(f"  [TC {exp}x l1={l1}] live={n_live:<6} mean_fire={fire.mean()*100:5.2f}% "
                      f"coh(live)={mean_coh*100:4.1f}% mse={mse:.3f} (copy {copy_mse:.3f})  "
                      f"[{rec['secs']:.0f}s]{flag}", flush=True)
                sparse_enough = fire.mean() <= 0.10
                enough_live = n_live >= MIN_LIVE
                score = (1 if converged else 0, 1 if enough_live else 0,
                         1 if sparse_enough else 0, mean_coh)
                if best is None or score > best["_score"]:
                    best = dict(exp=exp, m=int(m), l1=l1, fire=fire, live_mask=live_mask, coh=coh,
                                modal=modal, tops=tops, mean_coh=mean_coh, n_live=n_live,
                                mse=float(mse), beats_copy_input=converged, _score=score)
                del tc
                import torch as _t
                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
        return rows, best

    tc_rows, tc_best = run_transcoder(expansions=(16,))

    # ---- verdict -----------------------------------------------------------------------------------
    print(f"\n=== verdict inputs (all judged on the SAME OUTPUT layer {L_OUT} target) ===", flush=True)
    print(f"  PCA top-{K} mean coherence  : {pca_mean*100:.1f}%  (top-64 {pca_mean64*100:.1f}%)")
    print(f"  SAE best ({sae_best['exp']}x l1={sae_best['l1']}) : {sae_best['mean_coh']*100:.1f}% "
          f"over {sae_best['n_live']} live (MSE {sae_best['mse']:.3f})")
    print(f"  TRANSCODER best ({tc_best['exp']}x l1={tc_best['l1']}) : {tc_best['mean_coh']*100:.1f}% "
          f"over {tc_best['n_live']} live (MSE {tc_best['mse']:.3f} vs copy {copy_mse:.3f})")
    tc_vs_sae = tc_best["mean_coh"] - sae_best["mean_coh"]
    tc_vs_pca = tc_best["mean_coh"] - pca_mean

    def verdict(gap):
        return "TRANSCODER ahead" if gap > 0.02 else ("behind" if gap < -0.02 else "TIE")
    print(f"  TRANSCODER - SAE gap: {tc_vs_sae*100:+.1f} pts -> {verdict(tc_vs_sae)}")
    print(f"  TRANSCODER - PCA gap: {tc_vs_pca*100:+.1f} pts -> {verdict(tc_vs_pca)}")

    # ---- name 10 transcoder features (most coherent live) -----------------------------------------
    live_idx = np.where(tc_best["live_mask"])[0]
    ranked = sorted(live_idx, key=lambda j: -tc_best["coh"][j])
    feats = []
    print(f"\n=== 10 most-coherent TRANSCODER features ({tc_best['exp']}x, in L{L_IN}->out L{L_OUT}) ===")
    for j in ranked[:10]:
        tt = tc_best["tops"][j]
        print(f"  f{j:<6} coh={tc_best['coh'][j]*100:3.0f}%  fires={tc_best['fire'][j]*100:5.2f}%  "
              f"modal={tc_best['modal'][j]!r:<14} top: {' '.join(repr(t) for t in tt[:10])}",
              flush=True)
        feats.append(Feature(int(j), "sae", tt[:10], float(tc_best["fire"][j])))

    print(f"\n=== 10 most-coherent SAE features (output layer {L_OUT}, for contrast) ===")
    slive = np.where(sae_best["live_mask"])[0]
    sranked = sorted(slive, key=lambda j: -sae_best["coh"][j])
    for j in sranked[:10]:
        tt = sae_best["tops"][j]
        print(f"  f{j:<6} coh={sae_best['coh'][j]*100:3.0f}%  fires={sae_best['fire'][j]*100:5.2f}%  "
              f"modal={sae_best['modal'][j]!r:<14} top: {' '.join(repr(t) for t in tt[:10])}",
              flush=True)

    print("\n=== 8 PCA axes (output) for contrast ===")
    for j in range(8):
        tt = pca_tops[j]
        print(f"  PC{j:<3} coh={pca_coh[j]*100:3.0f}%  var={var[j]*100:4.1f}%  "
              f"modal={pca_modal[j]!r:<14} top: {' '.join(repr(t) for t in tt[:10])}", flush=True)

    # ---- save artifacts ---------------------------------------------------------------------------
    out_html = os.path.join(RUNS, "discovered_transcoder_qwen.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(render_discovered_features(
            feats, title="Clozn · Transcoder Discovery (Qwen2.5-0.5B, natural text)",
            subtitle=(f"{N} WikiText tokens · transcoder L{L_IN}->L{L_OUT} {tc_best['exp']}x "
                      f"m={tc_best['m']} l1={tc_best['l1']} · top-token coherence "
                      f"TC {tc_best['mean_coh']*100:.0f}% vs SAE {sae_best['mean_coh']*100:.0f}% vs "
                      f"PCA {pca_mean*100:.0f}%")))
    print(f"\nwrote {out_html}")

    summary = {
        "question": "Does a transcoder (in->out sparse map) beat the SAE/PCA null at scale (Qwen-0.5B)?",
        "rows": int(N), "dim_in": int(d_in), "dim_out": int(d_out),
        "l_in": L_IN, "l_out": L_OUT,
        "harvest": "natural WikiText-103, engine /harvest at TWO layers (token-aligned, one forward each)",
        "unique_token_ratio": uniq / max(N, 1),
        "in_out_copy_mse": copy_mse, "out_target_var": var_out,
        "metric": "un-seeded top-token coherence (identical for transcoder/SAE/PCA); rewards token-locking",
        "pca_K": K, "pca_mean_coherence": pca_mean, "pca_mean_coherence_top64": pca_mean64,
        "pca_examples": [{"idx": j, "coherence": float(pca_coh[j]), "modal": pca_modal[j],
                          "top": pca_tops[j][:10]} for j in range(8)],
        "sae_best": {"exp": sae_best["exp"], "m": sae_best["m"], "l1": sae_best["l1"],
                     "mean_coherence": sae_best["mean_coh"], "n_live": sae_best["n_live"],
                     "mse": sae_best["mse"]},
        "sae_grid": sae_rows,
        "sae_examples": [{"idx": int(j), "coherence": float(sae_best["coh"][j]),
                          "fires": float(sae_best["fire"][j]), "modal": sae_best["modal"][j],
                          "top": sae_best["tops"][j][:10]} for j in sranked[:10]],
        "transcoder_best": {"exp": tc_best["exp"], "m": tc_best["m"], "l1": tc_best["l1"],
                            "mean_coherence": tc_best["mean_coh"], "n_live": tc_best["n_live"],
                            "mse": tc_best["mse"], "beats_copy_input": tc_best["beats_copy_input"]},
        "transcoder_grid": tc_rows,
        "transcoder_examples": [{"idx": int(j), "coherence": float(tc_best["coh"][j]),
                                 "fires": float(tc_best["fire"][j]), "modal": tc_best["modal"][j],
                                 "top": tc_best["tops"][j][:10]} for j in ranked[:10]],
        "transcoder_minus_sae_gap": tc_vs_sae,
        "transcoder_minus_pca_gap": tc_vs_pca,
        "prior_runs": {"sae_at_scale": {"sae": 0.447, "pca_top256": 0.415, "pca_top64": 0.548},
                       "toy_rwkv_transcoder_hint": {"transcoder": 0.66, "sae": 0.62}},
    }
    out_json = os.path.join(RUNS, "discovered_transcoder_qwen.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out_json)


if __name__ == "__main__":
    main()
