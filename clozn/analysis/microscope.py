"""microscope.py -- the legible-basis microscope: decompose ANY direction vector into a sparse combination
of NAMED token directions ("what is this vector made of, in words?"). Self-contained: numpy only, no
torch, no network, no engine.

    target ~= sum_i alpha_i * dictionary[token_i]      (k terms, greedy top-k selection)

This is the same math already validated for ONE use case (fitting a memory bag against a
token-direction dictionary): greedy Orthogonal Matching Pursuit, refitting all selected alphas jointly
by least squares every step. This module ports that algorithm and generalizes it into a standalone,
reusable primitive -- any vector, any named dictionary, not just memory bags, unlocking several other
production targets beyond the original use case.

HONESTY (load-bearing, read before trusting a receipt):

  This is a CORRELATIONAL decomposition of a vector onto a *chosen word basis* -- "what words
  reconstruct this direction" -- NOT "what the vector means". A linear basis always produces SOME
  decomposition: handed an arbitrary vector and a big enough dictionary, OMP will confidently return a
  plausible-looking top-k list even when the vector has no real relationship to language at all (the
  "standing lens caveat": "A linear lens always outputs something. Plausible-looking readouts are not
  evidence by themselves"). `reconstruction_cos`
  and `explained_variance` are the honesty dial -- a low value means the dictionary doesn't actually span
  the target well, and the top words should not be trusted as "what it's made of" so much as "the least-
  bad words available".

  SIGN CAVEAT (from an X7 run, restated here because it generalizes beyond memory bags): a
  RAW RESIDUAL-STREAM target (e.g. a mean-pooled hidden state harvested off some text) gives SIGN-ARBITRARY
  alphas -- that run found negative weights on the most central content words (loaf, comet,
  asteroid) when fitting against a raw harvested residual, because a raw residual is dominated by
  position/syntax/high-norm-sink structure and its projection onto a token-direction dictionary is mostly
  noise. The honest targets for this module are DIFFERENCE or CENTROID vectors that are already meaningful
  as *directions* before any fit is done -- a dial vector, a trained steer-vector, a quant-diff direction
  (Q8 - Q4), a centroid of several token directions -- never a single raw residual pulled off one forward
  pass. If the input to `decompose` didn't earn its status as "a direction" some other way first, treat
  its alphas as decoration, not evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

_ZERO_TOL = 1e-12
_STOP_TOL = 1e-9


# ============================================================================================= the result

@dataclass(frozen=True)
class Term:
    """One recovered dictionary atom: `token`'s contribution `alpha` to the (unit-normalized) target.
    `order` is the 0-based step at which OMP *selected* this token -- distinct from its position in
    `Decomposition.terms` (which is sorted by |alpha| descending): a token picked first can still end up
    with a smaller refit alpha than one picked later, and `order` is what lets a caller tell those apart."""
    token: str
    alpha: float
    order: int


@dataclass(frozen=True)
class Decomposition:
    """The microscope's whole output. `terms` sorted by |alpha| descending. `reconstruction_cos` is the
    cosine similarity of the k-term reconstruction to the (unit-normalized) target -- 1.0 is a perfect
    reconstruction, near 0 means the dictionary barely explains the target at all (see module docstring's
    HONESTY note before reading anything into the words themselves at low cos). `residual_norm` is
    `||unit_target - reconstruction||` (0 when perfectly explained). `k_used` is how many terms were
    actually selected -- may be less than the requested `k` (early stop once the residual is exhausted, or
    the dictionary has fewer usable atoms than `k`)."""
    terms: list[Term]
    reconstruction_cos: float
    residual_norm: float
    k_used: int


def _degenerate(residual_norm: float = 0.0) -> Decomposition:
    """The single clean "nothing to report" return -- every guard clause below funnels here so decompose()
    never raises."""
    return Decomposition(terms=[], reconstruction_cos=0.0, residual_norm=residual_norm, k_used=0)


# ================================================================================================== decompose

def decompose(target: np.ndarray, dictionary: dict[str, np.ndarray], k: int = 8) -> Decomposition:
    """Greedy Orthogonal Matching Pursuit: find the `k` dictionary atoms whose weighted sum best
    reconstructs `target` (unit-normalized) -- a direct port of alpha_learning.fit_topk's algorithm,
    generalized from "candidate token-id rows in a matrix" to "a named token -> direction dict".

    Both `target` and every dictionary atom are UNIT-NORMALIZED internally before fitting (a fresh copy;
    the caller's arrays are never mutated) -- this is what makes the greedy correlation score
    (`|dot(residual, atom)|`) comparable across atoms of differing raw magnitude, and it is what makes
    `alpha` values comparable across different calls to `decompose` (they are always "how much of a UNIT
    target direction" a term explains, not an artifact of the caller's raw vector scale).

    At each of up to `k` steps: score every not-yet-selected atom by |dot(residual, atom)|, add the best;
    REFIT all selected alphas jointly via least squares (`np.linalg.lstsq`); update the residual. Stops
    early once the best available score is ~0 (the residual is already explained), so `k_used` can be
    strictly less than `k` when the target is sparser than `k` in this dictionary, or the dictionary itself
    has fewer usable atoms.

    Guards (never raises, always returns a clean, possibly-empty Decomposition instead):
      - empty `dictionary`
      - `target` that is all-zero, wrong-shaped, or contains non-finite values (NaN/inf)
      - individual dictionary atoms that are wrong-shaped, all-zero, or non-finite (silently skipped --
        one bad atom does not take down the whole dictionary)
      - `k <= 0`, or `k` larger than the number of usable atoms (clamped down, not an error)
    """
    if not dictionary:
        return _degenerate()

    t_raw = np.asarray(target, dtype=np.float64)
    if t_raw.ndim != 1 or t_raw.size == 0 or not np.all(np.isfinite(t_raw)):
        return _degenerate()
    t_norm = float(np.linalg.norm(t_raw))
    if t_norm < _ZERO_TOL:
        return _degenerate()
    t = t_raw / t_norm  # the unit-normalized target every alpha is fit against

    tokens: list[str] = []
    rows: list[np.ndarray] = []
    for tok, vec in dictionary.items():
        v = np.asarray(vec, dtype=np.float64)
        if v.shape != t.shape or not np.all(np.isfinite(v)):
            continue  # wrong-shaped or non-finite atom -- skip it, don't fail the whole dictionary
        n = np.linalg.norm(v)
        if n < _ZERO_TOL:
            continue  # degenerate (all-zero) atom -- carries no direction, skip it
        tokens.append(tok)
        rows.append(v / n)
    if not rows:
        return _degenerate(residual_norm=1.0)  # every atom was unusable -- nothing explained, full residual
    D = np.vstack(rows)  # [n_atoms, d]

    k_eff = max(0, min(int(k), len(tokens)))
    if k_eff == 0:
        return _degenerate(residual_norm=1.0)

    residual = t.copy()
    selected: list[int] = []
    order_of: dict[int, int] = {}
    for step in range(k_eff):
        scores = np.abs(D @ residual)
        if selected:
            scores[selected] = -np.inf  # never re-pick an already-selected atom
        j = int(np.argmax(scores))
        if not np.isfinite(scores[j]) or scores[j] <= _STOP_TOL:
            break  # residual already fully explained (or degenerate) -- stop early, don't fit noise
        selected.append(j)
        order_of[j] = step
        Dsel = D[selected]
        alphas, *_ = np.linalg.lstsq(Dsel.T, t, rcond=None)
        residual = t - alphas @ Dsel

    if not selected:
        return _degenerate(residual_norm=1.0)

    Dsel = D[selected]
    alphas, *_ = np.linalg.lstsq(Dsel.T, t, rcond=None)
    recon = alphas @ Dsel
    recon_norm = float(np.linalg.norm(recon))
    cos = float(np.dot(recon, t) / recon_norm) if recon_norm > _ZERO_TOL else 0.0
    residual_final = t - recon

    terms = [Term(token=tokens[j], alpha=float(a), order=order_of[j]) for j, a in zip(selected, alphas)]
    terms.sort(key=lambda term: -abs(term.alpha))
    return Decomposition(
        terms=terms,
        reconstruction_cos=cos,
        residual_norm=float(np.linalg.norm(residual_final)),
        k_used=len(selected),
    )


# ============================================================================================ the receipt

def render_receipt(decomp: Decomposition) -> str:
    """The human table: one signed, aligned line per term, sorted by |alpha| descending --
    `"+0.475  bakery"` -- mirroring notes/x7_legible_memory/receipt.py's render_receipt. A pure function of
    `decomp`'s own data: nothing here can state a word that isn't actually one of the recovered terms."""
    if not decomp.terms:
        return "(nothing recovered -- the decomposition is empty)"
    items = sorted(decomp.terms, key=lambda t: -abs(t.alpha))
    return "\n".join(f"{t.alpha:+.3f}  {t.token}" for t in items)


def explained_variance(decomp: Decomposition) -> float:
    """Fraction of the (unit-normalized) target's variance captured by the k-term reconstruction --
    `reconstruction_cos ** 2`, since the reconstruction is an orthogonal least-squares projection of a unit
    vector (Pythagorean: ||recon||^2 + ||residual||^2 == 1, and cos == ||recon|| for a unit target).
    Clamped to [0, 1] against floating-point drift."""
    return float(np.clip(decomp.reconstruction_cos ** 2, 0.0, 1.0))


def top_words(decomp: Decomposition, n: int = 5) -> list[str]:
    """Convenience: the top-n recovered tokens by |alpha|, in `decomp.terms`' existing sorted order."""
    return [t.token for t in decomp.terms[: max(0, n)]]


# ================================================================================================ the seam

@runtime_checkable
class DirProvider(Protocol):
    """The seam between "the microscope" (pure math, this file) and "where token directions come from"
    (a live model). Nothing in this module implements a real provider -- there is no engine, no network,
    no torch here, by design.

    Production seam: `/jlens/unembed_row` -- the J-lens unembed-row primitive (dir(c) = normalize(J^T @
    W_U[token]), notes/x7_legible_memory/legible_memory.py:dir_of_token / notes/keystone_dirc/dirc.py) is
    what fills this in a live caller. A production DirProvider wraps a running engine's `/jlens/unembed_row`
    route (or an equivalent local J/W_U lookup) behind this one method.
    """

    def dir_of_token(self, token: str) -> np.ndarray | None:
        """Return the direction vector for `token`, or None if unavailable -- e.g. the token has no clean
        single-token embedding-row mapping (a multi-token word/phrase), or the provider's backing lens has
        no entry for it. Returning None (rather than raising) is what lets decompose_with_provider silently
        skip tokens it cannot place, without the caller pre-filtering its token list."""
        ...


def decompose_with_provider(target: np.ndarray, tokens: list[str], provider: DirProvider,
                             k: int = 8) -> Decomposition:
    """Build the dictionary by calling `provider.dir_of_token` once per token, SKIPPING any token whose
    provider call returns None (or raises -- treated identically to None, so one bad lookup can't take down
    the whole decomposition) -- typically multi-token words/phrases the provider can't map to a single
    embedding row, or tokens outside the provider's known vocabulary. Then delegates to `decompose`.

    This is the only function in this module that touches `DirProvider` -- `decompose` itself never knows
    where a dictionary came from, matching this module's math/wiring seam (mirrors
    notes/x7_legible_memory/legible_memory.py:compose_memory's injected `dir_lookup` callable).
    """
    dictionary: dict[str, np.ndarray] = {}
    for tok in tokens:
        try:
            vec = provider.dir_of_token(tok)
        except Exception:
            continue
        if vec is None:
            continue
        dictionary[tok] = vec
    return decompose(target, dictionary, k=k)
