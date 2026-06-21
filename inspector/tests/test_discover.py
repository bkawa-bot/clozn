"""Oracle for clozn.discover (feature discovery). Fast tests plant clusters and check PCA/SAE
recover them (no model); a gated test asserts the SAE rediscovers seeded themes on real RWKV-4
and beats the PCA baseline."""
from collections import Counter

import numpy as np
import pytest

from clozn.discover import TinySAE, pca_features, sae_features, standardize


def _planted(rng, d=64, per=40):
    """Three clusters spaced along a single DENSE diagonal direction, so the structure lives in
    cross-dimension correlations that survive per-dim standardization (axis-aligned clusters would
    flatten to a near-identity covariance PCA can't read). Each point's 'token' is its label."""
    v = rng.normal(0, 1, d); v /= np.linalg.norm(v)
    X, toks = [], []
    for t, lab in [(-8.0, "A"), (0.0, "B"), (8.0, "C")]:
        for _ in range(per):
            X.append(t * v + rng.normal(0, 0.4, d)); toks.append(lab)
    return np.array(X), toks


def _purity(top_tokens):
    c = Counter(t.strip() for t in top_tokens)
    return c.most_common(1)[0][1] / len(top_tokens)


def test_pca_recovers_planted_clusters():
    X, toks = _planted(np.random.default_rng(0))
    Xs, _, _ = standardize(X)
    feats = pca_features(Xs, toks, k=3, topn=8)
    assert max(_purity(f.top_tokens) for f in feats) >= 0.8     # some axis isolates a cluster


def test_tinysae_recovers_and_reconstructs():
    pytest.importorskip("torch")
    X, toks = _planted(np.random.default_rng(1))
    Xs, _, _ = standardize(X)
    sae = TinySAE(Xs.shape[1], m=24, l1=1e-2, seed=0).fit(Xs, steps=400)
    assert sae.recon_error(Xs) < 0.9                            # beats predict-the-mean (~1.0)
    C = sae.codes(Xs)
    best = 0.0
    for j in range(C.shape[1]):
        if (C[:, j] > 1e-6).mean() < 0.01:
            continue
        order = np.argsort(C[:, j])[::-1][:8]
        best = max(best, _purity([toks[i] for i in order]))
    assert best >= 0.8                                          # some feature isolates a cluster


def test_transcoder_learns_input_to_output_map():
    pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    d, n = 48, 200
    X = rng.normal(0, 1, (n, d))
    R = rng.normal(0, 1, (d, d)) / np.sqrt(d)
    Y = np.maximum(X @ R, 0.0)                              # a component-like input→output map
    Xs, _, _ = standardize(X); Ys, _, _ = standardize(Y)
    tc = TinySAE(d, m=64, l1=1e-3, seed=0).fit(Xs, steps=400, Y=Ys)       # transcoder: predict Y
    sae = TinySAE(d, m=64, l1=1e-3, seed=0).fit(Xs, steps=400)            # SAE: reconstruct X
    assert tc.recon_error(Xs, Ys) < 0.9                    # learns the map (beats predict-the-mean)
    assert tc.recon_error(Xs, Ys) < sae.recon_error(Xs, Ys)   # and beats the SAE at predicting Y


def test_decoder_direction_is_unit_and_shaped():
    pytest.importorskip("torch")
    X, toks = _planted(np.random.default_rng(2))
    Xs, _, sd = standardize(X)
    sae = TinySAE(Xs.shape[1], m=16, l1=1e-2, seed=0).fit(Xs, steps=200)
    d = sae.decoder_direction(0, sd)
    assert d.shape == (Xs.shape[1],)
    assert abs(np.linalg.norm(d) - 1.0) < 1e-5            # a unit steering vector in raw units


@pytest.mark.model
def test_sae_discovers_themes_on_real_rwkv(rwkv):
    pytest.importorskip("torch")
    from clozn.discover import collect_token_states, corpus
    X, toks, _ = collect_token_states(rwkv, corpus())
    Xs, _, _ = standardize(X)
    pca = pca_features(Xs, toks, k=12)
    feats, _, _ = sae_features(Xs, toks, m=128, l1=4e-2, steps=800)
    coherent = [f for f in feats if f.purity >= 0.6]
    assert len(coherent) >= 3                                   # rediscovers several seeded themes
    sae_mean = np.mean([f.purity for f in feats])
    pca_mean = np.mean([f.purity for f in pca])
    assert sae_mean > pca_mean                                  # SAE unmixes what PCA buries
