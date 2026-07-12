"""test_microscope.py -- model-free unit tests for microscope.py. numpy only, no engine/network/torch.
Synthetic dictionaries stand in for real dir(token) directions (mirrors
notes/x7_legible_memory/fixtures.py's approach for alpha_learning.py)."""
from __future__ import annotations

import numpy as np
import pytest

from clozn.analysis.microscope import (
    Decomposition,
    Term,
    decompose,
    decompose_with_provider,
    explained_variance,
    render_receipt,
    top_words,
)


def _dictionary(d: int = 16, seed: int = 0) -> dict[str, np.ndarray]:
    """An orthonormal token basis (via QR) plus two CORRELATED (non-orthogonal) distractor atoms layered on
    top -- mirrors a real unembedding basis where most tokens are near-orthogonal but a few share structure
    (e.g. synonyms). The distractors are never used as planted targets below, only present so OMP has to
    pick the right atom among near-duplicates."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    basis = {f"tok{i}": q[:, i].astype(np.float64) for i in range(d)}
    basis["blend_a"] = basis["tok0"] * 0.9 + basis["tok5"] * 0.1
    basis["blend_b"] = basis["tok1"] * 0.8 + basis["tok6"] * 0.2
    return basis


# ============================================================================================ (a) single atom

def test_single_atom_target_recovers_exactly_with_cos_one_k_used_one():
    d = _dictionary()
    target = d["tok3"] * 2.5  # arbitrary nonzero scale -- decompose unit-normalizes internally
    decomp = decompose(target, d, k=8)
    assert decomp.k_used == 1
    assert len(decomp.terms) == 1
    assert decomp.terms[0].token == "tok3"
    assert decomp.terms[0].alpha == pytest.approx(1.0, abs=1e-6)
    assert decomp.terms[0].order == 0
    assert decomp.reconstruction_cos == pytest.approx(1.0, abs=1e-6)
    assert decomp.residual_norm == pytest.approx(0.0, abs=1e-6)


# ================================================================================= (b) known 3-atom combo

def test_three_atom_combo_recovers_tokens_signs_and_cos_at_k3():
    d = _dictionary()
    target = 0.6 * d["tok2"] - 0.4 * d["tok7"] + 0.25 * d["tok10"]
    decomp = decompose(target, d, k=3)
    assert decomp.k_used == 3
    recovered = {t.token: t.alpha for t in decomp.terms}
    assert set(recovered) == {"tok2", "tok7", "tok10"}

    norm = np.linalg.norm(target)  # decompose fits against the unit-normalized target
    assert recovered["tok2"] == pytest.approx(0.6 / norm, abs=1e-4)
    assert recovered["tok7"] == pytest.approx(-0.4 / norm, abs=1e-4)
    assert recovered["tok10"] == pytest.approx(0.25 / norm, abs=1e-4)
    assert np.sign(recovered["tok2"]) == 1
    assert np.sign(recovered["tok7"]) == -1
    assert np.sign(recovered["tok10"]) == 1
    assert decomp.reconstruction_cos == pytest.approx(1.0, abs=1e-4)


def test_terms_sorted_by_absolute_alpha_descending():
    d = _dictionary()
    target = 0.6 * d["tok2"] - 0.4 * d["tok7"] + 0.25 * d["tok10"]
    decomp = decompose(target, d, k=3)
    mags = [abs(t.alpha) for t in decomp.terms]
    assert mags == sorted(mags, reverse=True)


# ======================================================================= (c) reconstruction_cos monotonic

def test_reconstruction_cos_is_monotonic_non_decreasing_in_k():
    d = _dictionary()
    rng = np.random.default_rng(42)
    target = rng.standard_normal(16)  # a generic target, not planted as a sparse combo in this basis
    coses = [decompose(target, d, k=k).reconstruction_cos for k in range(1, 9)]
    assert all(b >= a - 1e-9 for a, b in zip(coses, coses[1:]))


def test_residual_norm_is_monotonic_non_increasing_in_k():
    d = _dictionary()
    rng = np.random.default_rng(7)
    target = rng.standard_normal(16)
    residuals = [decompose(target, d, k=k).residual_norm for k in range(1, 9)]
    assert all(b <= a + 1e-9 for a, b in zip(residuals, residuals[1:]))


# =================================================================================== (d) degenerate guards

def test_empty_dictionary_returns_clean_decomposition():
    decomp = decompose(np.array([1.0, 2.0, 3.0]), {}, k=4)
    assert decomp == Decomposition(terms=[], reconstruction_cos=0.0, residual_norm=0.0, k_used=0)


def test_zero_target_returns_clean_decomposition():
    d = _dictionary()
    decomp = decompose(np.zeros(16), d, k=4)
    assert decomp.terms == []
    assert decomp.k_used == 0
    assert decomp.reconstruction_cos == 0.0


def test_k_zero_returns_clean_decomposition():
    d = _dictionary()
    decomp = decompose(d["tok0"], d, k=0)
    assert decomp.terms == []
    assert decomp.k_used == 0


def test_negative_k_returns_clean_decomposition():
    d = _dictionary()
    decomp = decompose(d["tok0"], d, k=-3)
    assert decomp.terms == []
    assert decomp.k_used == 0


def test_non_finite_target_returns_clean_decomposition():
    d = _dictionary()
    bad = d["tok0"].copy()
    bad[0] = np.nan
    decomp = decompose(bad, d, k=4)
    assert decomp.terms == []
    assert decomp.k_used == 0


def test_wrong_shaped_target_returns_clean_decomposition():
    d = _dictionary()
    decomp = decompose(np.zeros(3), d, k=4)  # d_model mismatch vs the 16-dim dictionary
    assert decomp.terms == []
    assert decomp.k_used == 0


def test_non_finite_or_degenerate_atoms_are_skipped_not_raised():
    d = dict(_dictionary())
    d["broken_inf"] = np.full(16, np.inf)
    d["broken_nan"] = np.array([np.nan] * 16)
    d["broken_zero"] = np.zeros(16)
    d["broken_shape"] = np.zeros(3)
    target = d["tok4"] * 1.5
    decomp = decompose(target, d, k=4)  # must not raise despite four unusable atoms in the dictionary
    used = {t.token for t in decomp.terms}
    assert used.isdisjoint({"broken_inf", "broken_nan", "broken_zero", "broken_shape"})
    assert decomp.reconstruction_cos == pytest.approx(1.0, abs=1e-6)


def test_k_larger_than_dictionary_clamps_without_raising():
    d = {"only": np.array([1.0, 0.0, 0.0])}
    decomp = decompose(np.array([2.0, 0.0, 0.0]), d, k=50)
    assert decomp.k_used == 1
    assert decomp.terms[0].token == "only"


def test_all_atoms_unusable_returns_clean_decomposition():
    d = {"a": np.zeros(4), "b": np.array([np.nan, 0.0, 0.0, 0.0])}
    decomp = decompose(np.array([1.0, 0.0, 0.0, 0.0]), d, k=2)
    assert decomp.terms == []
    assert decomp.k_used == 0


# ========================================================================================= (e) render_receipt

def test_render_receipt_formats_signed_aligned_lines():
    terms = [Term(token="bakery", alpha=0.475, order=0), Term(token="dough", alpha=-0.221, order=1)]
    decomp = Decomposition(terms=terms, reconstruction_cos=0.9, residual_norm=0.1, k_used=2)
    lines = render_receipt(decomp).splitlines()
    assert lines[0] == "+0.475  bakery"
    assert lines[1] == "-0.221  dough"


def test_render_receipt_sorts_defensively_by_absolute_alpha():
    # out-of-order input terms: render_receipt must still sort by |alpha| descending itself.
    terms = [Term(token="small", alpha=0.01, order=0), Term(token="big", alpha=-0.9, order=1)]
    decomp = Decomposition(terms=terms, reconstruction_cos=0.5, residual_norm=0.5, k_used=2)
    lines = render_receipt(decomp).splitlines()
    assert lines[0] == "-0.900  big"
    assert lines[1] == "+0.010  small"


def test_render_receipt_empty_decomposition_is_clean_not_raising():
    decomp = Decomposition(terms=[], reconstruction_cos=0.0, residual_norm=0.0, k_used=0)
    text = render_receipt(decomp)
    assert "empty" in text or "nothing" in text


# ===================================================================== explained_variance / top_words

def test_explained_variance_matches_cos_squared_and_is_high_for_a_good_fit():
    d = _dictionary()
    target = 0.6 * d["tok2"] - 0.4 * d["tok7"] + 0.25 * d["tok10"]
    decomp = decompose(target, d, k=3)
    assert explained_variance(decomp) == pytest.approx(decomp.reconstruction_cos ** 2, abs=1e-9)
    assert explained_variance(decomp) == pytest.approx(1.0, abs=1e-4)


def test_top_words_returns_top_n_in_existing_sorted_order():
    d = _dictionary()
    target = 0.6 * d["tok2"] - 0.4 * d["tok7"] + 0.25 * d["tok10"]
    decomp = decompose(target, d, k=3)
    words = top_words(decomp, n=2)
    assert words == [decomp.terms[0].token, decomp.terms[1].token]
    assert len(words) == 2


def test_top_words_on_empty_decomposition_is_empty_list():
    decomp = Decomposition(terms=[], reconstruction_cos=0.0, residual_norm=0.0, k_used=0)
    assert top_words(decomp) == []


# =================================================================================== (f) decompose_with_provider

class _FakeProvider:
    """Duck-typed DirProvider test double -- no inheritance needed, the Protocol is structural."""

    def __init__(self, table: dict):
        self._table = table

    def dir_of_token(self, token: str) -> np.ndarray | None:
        return self._table.get(token)


def test_decompose_with_provider_skips_none_tokens():
    d = _dictionary()
    table = {"tok0": d["tok0"], "multiword phrase": None, "tok2": d["tok2"], "unknown_tok": None}
    provider = _FakeProvider(table)
    target = 0.5 * d["tok0"] + 0.5 * d["tok2"]
    decomp = decompose_with_provider(target, list(table.keys()), provider, k=4)
    used = {t.token for t in decomp.terms}
    assert used <= {"tok0", "tok2"}
    assert "multiword phrase" not in used
    assert "unknown_tok" not in used
    assert decomp.reconstruction_cos == pytest.approx(1.0, abs=1e-4)


def test_decompose_with_provider_raising_is_treated_like_none():
    class _RaisingProvider:
        def dir_of_token(self, token: str) -> np.ndarray | None:
            if token == "boom":
                raise RuntimeError("no lens entry")
            return None

    d = _dictionary()
    decomp = decompose_with_provider(d["tok0"], ["boom", "also_none"], _RaisingProvider(), k=2)
    assert decomp.terms == []
    assert decomp.k_used == 0


def test_decompose_with_provider_empty_token_list_is_clean():
    provider = _FakeProvider({})
    decomp = decompose_with_provider(np.array([1.0, 0.0]), [], provider, k=4)
    assert decomp.terms == []
    assert decomp.k_used == 0
