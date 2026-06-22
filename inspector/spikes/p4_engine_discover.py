"""
Phase-3 §3.6 — does unsupervised SAE feature discovery hold at REAL transformer scale, harvested
THROUGH THE ENGINE?  The toy (RWKV-4-169m, discover.py) showed SAE 65% vs PCA 12% top-token
coherence. Here we extend it to a real transformer's residual stream (Qwen2.5-0.5B), with the
activations harvested over the wire from the C++ engine's white-box tap (the §3.1 "activation
harvesting at scale" path), not a direct HF hook. We REUSE discover.py's TinySAE + the PCA baseline
and an un-seeded top-token coherence metric, and report honestly — SAE vs PCA at this scale.

Honesty notes baked in:
  * The engine's tap is hardwired to LAYER 2 (early residual) — chosen for per-token probe
    separation, NOT for SAE richness. Early-layer features skew lexical/token-level. We say so.
  * Harvest is GENERATION-based (each generated token's residual), the simplest first corpus. The
    engine server has an intermittent crash under sustained state="full" generation, so we harvest
    in batches and AUTO-RESTART the server between them (so a crash costs a batch, not the run).
  * The un-seeded coherence metric is identical for SAE and PCA (apples-to-apples): for each
    feature, take its top-N activating tokens; coherence = fraction equal to that feature's modal
    top token (token-identity concentration). A feature that fires on one consistent token scores 1.

Usage (from inspector/, with the cloze venv python; server PATH must include the engine bin + the
CUDA v13.3 bin\\x64 dir):
    python spikes/p4_engine_discover.py
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
from clozn.sources.engine import EngineStateSource  # noqa: E402
from clozn.viz import render_discovered_features  # noqa: E402

# ---- engine launch config -----------------------------------------------------------------------
MODEL = r"C:\Users\brigi\src\cloze\core\models\Qwen2.5-0.5B-Instruct-q8_0.gguf"
EXE = r"C:\Users\brigi\src\clozn\engine\core\build-gpu\cloze-server.exe"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
ENGINE_BIN = r"C:\Users\brigi\src\clozn\engine\core\build-gpu\bin"
PORT = 8080
BASE_URL = f"http://127.0.0.1:{PORT}"
RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


# ---- server management (survive the intermittent crash) -----------------------------------------
def _env_with_paths() -> dict:
    env = dict(os.environ)
    env["PATH"] = ENGINE_BIN + os.pathsep + CUDA_BIN + os.pathsep + env.get("PATH", "")
    return env


def _health_ok(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(BASE_URL + "/health", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def kill_server() -> None:
    subprocess.run(["taskkill", "/IM", "cloze-server.exe", "/F"],
                   capture_output=True, text=True)
    time.sleep(0.4)


def start_server() -> subprocess.Popen:
    kill_server()
    os.makedirs(RUNS, exist_ok=True)
    out = open(os.path.join(RUNS, "server_qwen_ar.log"), "w")
    err = open(os.path.join(RUNS, "server_qwen_ar.log.err"), "w")
    p = subprocess.Popen(
        [EXE, MODEL, "--gpu-layers", "99", "--port", str(PORT), "--ctx", "4096"],
        env=_env_with_paths(), stdout=out, stderr=err)
    for _ in range(90):
        if _health_ok():
            return p
        time.sleep(1.0)
    raise RuntimeError("server did not become healthy")


# ---- corpus: diverse PROSE prompts (avoid the instruct-MCQ register) ----------------------------
# Open-ended continuations across many registers/topics so the generated tokens span natural text,
# not multiple-choice boilerplate. Kept declarative (no questions) to discourage "A. / B. / answer:".
PROMPTS = [
    "The old lighthouse keeper climbed the spiral stairs as the storm",
    "In 1789 the citizens of Paris marched toward the Bastille and",
    "The recipe called for fresh basil, ripe tomatoes, and a pinch of",
    "Deep beneath the Pacific, hydrothermal vents support strange colonies of",
    "The violinist drew her bow across the strings and the hall filled with",
    "Photosynthesis converts sunlight, water, and carbon dioxide into",
    "The detective examined the muddy footprints leading away from the",
    "On the savannah at dawn, a pride of lions stalked a herd of",
    "The spacecraft fired its thrusters and slowly drifted toward the docking",
    "She unfolded the yellowed map and traced the river down to the",
    "The blacksmith hammered the glowing iron into the shape of a",
    "Quantum computers exploit superposition and entanglement to perform",
    "The marathon runner rounded the final corner, exhausted but",
    "Across the frozen tundra, the herd of reindeer migrated toward",
    "The chef plated the seared scallops beside a delicate puree of",
    "Ancient Roman engineers built aqueducts that carried water across",
    "The toddler stacked the wooden blocks higher and higher until they",
    "The jazz trio improvised over a slow twelve-bar blues in the key of",
    "Glaciers carve deep valleys as they grind slowly down the",
    "The librarian shelved the dusty volumes of poetry and",
    "On the trading floor the brokers shouted as the price of oil",
    "The gardener pruned the roses and scattered seeds across the",
    "A flock of starlings wheeled over the wheat field in a shifting",
    "The surgeon made a careful incision and gently retracted the",
    "The novelist crumpled another page and stared at the blank",
    "Volcanic eruptions release ash, sulfur dioxide, and rivers of molten",
    "The sailors hauled the heavy ropes as the schooner heeled into the",
    "In the laboratory the chemist titrated the acid drop by drop until",
    "The mountaineers fixed their ropes and began the final ascent of the",
    "The beekeeper lifted the frame, thick with honey and crawling with",
    "The orchestra tuned to the oboe's A as the conductor raised his",
    "Migrating salmon fought their way upstream to spawn in the gravel",
    "The carpenter measured twice, marked the oak board, and",
    "The astronomer adjusted the telescope and the rings of Saturn came into",
    "The baker pulled the crusty loaves from the oven and set them to",
    "A thunderstorm rolled across the plains, lightning splitting the",
    "The diplomat chose her words carefully as the negotiations",
    "The river otter cracked open a clam against a stone on its",
    "The programmer traced the bug through the stack until she found the",
    "The potter centered the clay on the wheel and pressed her thumbs into the",
]


def harvest_batch(prompts, max_tokens=40, temperature=0.9):
    """Drive EngineStateSource over `prompts`, collecting each generated token's layer-2 residual.
    Returns (vectors[list of [d]], pieces[list of str]). Raises on a connection failure so the caller
    can restart the server and retry from where it left off."""
    src = EngineStateSource(substrate="autoregressive", max_tokens=max_tokens, temperature=temperature)
    vecs, pieces = [], []
    for p in prompts:
        src.reset()
        toks_this = []
        vecs_this = []
        for st in src.step_stream(p):  # may raise ConnectionResetError mid-stream -> propagate
            h = st.state.get("hidden")
            if h is None:
                continue
            tok = st.token
            piece = None
            if isinstance(tok, list) and tok and isinstance(tok[0], dict):
                piece = tok[0].get("piece")
            for r in range(h.shape[0]):
                vecs_this.append(h[r].astype(np.float32))
                toks_this.append(piece if piece is not None else "")
        vecs.extend(vecs_this)
        pieces.extend(toks_this)
    return vecs, pieces


def harvest(target_rows=5000, batch=8):
    """Harvest ~target_rows, auto-restarting the server on a crash. Batches of `batch` prompts; on a
    crash we restart and move to the NEXT batch (don't retry the offending batch indefinitely)."""
    start_server()
    vecs, pieces = [], []
    bi = 0
    pi = 0
    crashes = 0
    while len(vecs) < target_rows and crashes < 30:
        chunk = PROMPTS[pi % len(PROMPTS): pi % len(PROMPTS) + batch]
        if not chunk:
            pi = 0
            chunk = PROMPTS[:batch]
        try:
            v, t = harvest_batch(chunk)
            vecs.extend(v)
            pieces.extend(t)
            print(f"  batch {bi}: +{len(v)} rows (total {len(vecs)})", flush=True)
        except (urllib.error.URLError, ConnectionResetError, OSError) as e:
            crashes += 1
            print(f"  batch {bi}: server died ({type(e).__name__}); restart #{crashes}", flush=True)
            start_server()
        bi += 1
        pi += batch
    return np.stack(vecs), pieces


# ---- un-seeded coherence metric (identical for SAE and PCA) --------------------------------------
def _norm(t: str) -> str:
    return t.strip().lower()


def top_token_coherence(scores_per_feature, pieces, topn=20):
    """scores_per_feature: [N, F] activation/projection matrix. For each feature take its top-N
    activating rows; coherence = fraction of those equal to the feature's MODAL top token. Returns
    (per-feature coherence array, per-feature modal token, per-feature top tokens list)."""
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


CACHE = os.path.join(RUNS, "qwen_engine_acts.npz")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = int(args[0]) if args else 5000
    os.makedirs(RUNS, exist_ok=True)
    if "--from-cache" in sys.argv and os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        X, pieces = d["X"], list(d["pieces"])
        print(f"=== loaded {X.shape[0]} cached rows x {X.shape[1]} dims (layer 2) ===", flush=True)
    else:
        print(f"=== harvesting ~{target} layer-2 residuals from Qwen-0.5B via the ENGINE (AR) ===",
              flush=True)
        t0 = time.time()
        X, pieces = harvest(target_rows=target)
        print(f"harvested {X.shape[0]} rows x {X.shape[1]} dims in {time.time()-t0:.0f}s "
              f"(engine tap = layer 2, generation-based)", flush=True)
        np.savez_compressed(CACHE, X=X, pieces=np.array(pieces, dtype=object))

    Xs, mu, sd = standardize(X)
    d = Xs.shape[1]

    # ---- PCA baseline -----------------------------------------------------------------------------
    K = 64
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    pca_proj = Xs @ Vt[:K].T                                # [N, K]
    pca_coh, pca_modal, pca_tops = top_token_coherence(pca_proj, pieces, topn=20)

    # ---- SAE (reuse discover.TinySAE), L1 dose-response -------------------------------------------
    # Training config matters: minibatch=512 (not ~the whole set) + lr=3e-2 so the AE actually
    # converges (MSE ~0.08-0.17, not 1.0) AND the L1 bites (fire-rate falls cleanly with L1). This
    # was the bug in the first pass (batch=4096/lr=4e-3 under-trained -> dead, MSE=1.0). We sweep L1
    # and pick by coherence, but also PRINT the whole dose-response (sparsity vs coherence) honestly.
    best = None
    dose = []
    for l1 in (0.1, 0.3, 0.6, 1.0, 2.0):
        sae = TinySAE(d, m=512, l1=l1, seed=0).fit(Xs, batch_size=512, epochs=80, lr=3e-2)
        C = sae.codes(Xs)                                   # [N, 512]
        fire = (C > 1e-6).mean(0)
        live_mask = (fire >= 0.002) & (fire <= 0.4)
        n_live = int(live_mask.sum())
        coh, modal, tops = top_token_coherence(C, pieces, topn=20)
        live_coh = coh[live_mask]
        mean_coh = float(live_coh.mean()) if n_live else 0.0
        mse = sae.recon_error(Xs)
        dose.append({"l1": l1, "mean_fire": float(fire.mean()), "n_live": n_live,
                     "coherence": mean_coh, "mse": mse})
        print(f"  [SAE l1={l1}] live={n_live:<4} mean_fire={fire.mean()*100:4.1f}% "
              f"coherence(live)={mean_coh*100:.0f}% mse={mse:.3f}", flush=True)
        # prefer a SPARSE, coherent run: among runs with a sane fire-rate, take best coherence.
        sparse_enough = fire.mean() <= 0.12
        score = (1 if sparse_enough else 0, mean_coh)
        if best is None or score > best["_score"]:
            best = dict(l1=l1, sae=sae, C=C, fire=fire, live_mask=live_mask, coh=coh,
                        modal=modal, tops=tops, mean_coh=mean_coh, n_live=n_live, mse=mse,
                        _score=score)

    # PCA mean coherence over the top-K axes (the comparable baseline number)
    pca_mean = float(pca_coh.mean())
    print(f"\n  PCA top-{K} mean top-token coherence = {pca_mean*100:.0f}%")
    print(f"  SAE (best l1={best['l1']}) live mean top-token coherence = {best['mean_coh']*100:.0f}% "
          f"over {best['n_live']} live features")

    # ---- name 10 example SAE features (most coherent live ones) ----------------------------------
    live_idx = np.where(best["live_mask"])[0]
    ranked = sorted(live_idx, key=lambda j: -best["coh"][j])
    feats = []
    print("\n=== 10 most-coherent discovered SAE features (engine, layer 2) ===")
    for j in ranked[:10]:
        tt = best["tops"][j]
        print(f"  f{j:<4} coherence={best['coh'][j]*100:3.0f}%  fires={best['fire'][j]*100:4.1f}%  "
              f"modal={best['modal'][j]!r:<12} top: {' '.join(repr(t) for t in tt[:10])}", flush=True)
        feats.append(Feature(int(j), "sae", tt[:10], float(best["fire"][j])))

    print("\n=== 8 PCA axes for contrast ===")
    pca_feats = []
    var = (S ** 2) / (S ** 2).sum()
    for j in range(8):
        tt = pca_tops[j]
        print(f"  PC{j:<3} coherence={pca_coh[j]*100:3.0f}%  var={var[j]*100:4.1f}%  "
              f"modal={pca_modal[j]!r:<12} top: {' '.join(repr(t) for t in tt[:10])}", flush=True)
        pca_feats.append(Feature(int(j), "pca", tt[:10], float(var[j])))

    # ---- save the HTML viz -----------------------------------------------------------------------
    out = os.path.join(RUNS, "discovered_engine_qwen_L2.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_discovered_features(
            feats, title="Clozn · Engine Discovery (Qwen2.5-0.5B, layer 2)",
            subtitle=(f"{X.shape[0]} residual-stream tokens harvested via the C++ engine (AR) · "
                      f"SAE l1={best['l1']} m=512 · top-token coherence "
                      f"SAE {best['mean_coh']*100:.0f}% vs PCA {pca_mean*100:.0f}%")))
    print(f"\nwrote {out}")

    # ---- a tiny machine-readable summary for the writeup -----------------------------------------
    summary = {
        "rows": int(X.shape[0]), "dim": int(X.shape[1]), "tap_layer": 2,
        "harvest": "generation, engine AR, Qwen2.5-0.5B-Instruct-q8_0",
        "sae_best_l1": best["l1"], "sae_mean_coherence": best["mean_coh"],
        "sae_n_live": best["n_live"], "sae_mse": best["mse"],
        "sae_dose_response": dose,
        "pca_mean_coherence": pca_mean, "pca_K": K,
        "examples_sae": [{"idx": int(j), "coherence": float(best["coh"][j]),
                          "fires": float(best["fire"][j]), "modal": best["modal"][j],
                          "top": best["tops"][j][:10]} for j in ranked[:10]],
        "examples_pca": [{"idx": j, "coherence": float(pca_coh[j]),
                          "modal": pca_modal[j], "top": pca_tops[j][:10]} for j in range(8)],
    }
    with open(os.path.join(RUNS, "discovered_engine_qwen_L2.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote", os.path.join(RUNS, "discovered_engine_qwen_L2.json"))
    if "--from-cache" not in sys.argv:
        kill_server()


if __name__ == "__main__":
    main()
