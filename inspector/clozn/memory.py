"""
clozn.memory — legible personal memory (Phase 3): a browsable shelf of saved mind-states with
associative recall. Built entirely on the proven primitives (store.py + the StateSource seam),
so it inherits their exactness and works on ANY substrate (recurrent matrix, diffusion canvas).

The idea the user wanted: a place a model can remember *beyond chain-of-thought*, where each
memory is an inspectable object. M4 showed a saved state recalls a fact verbatim. This adds the
missing verb: **association** — given the model's current internal state, which stored memory is
it most like ("what does this remind you of?"). Memory keyed by the *shape of thought*, not text.

Legibility levers from the legible-interior experiments: a **write-gate** (don't store what's
already known — Exp 2/2b) keeps the shelf sparse, and sparsity is the legibility budget (Exp 7b).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .spine import StateSource
from .store import StateStore


@dataclass
class Match:
    name: str
    similarity: float          # cosine of current state vs the stored state (in [-1, 1])
    note: str = ""


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na and nb else 0.0


class MemoryShelf:
    """A named library of saved states with similarity-based associative recall.

    `component` selects which part of the state keys the memory (default the RWKV running
    memory `att_num`; use `"S"` for the toy matrix, `"board"` for diffusion). Similarity is
    cosine over that component, flattened — substrate-agnostic and robust.
    """

    def __init__(self, store: StateStore, component: str = "att_num"):
        self.store = store
        self.component = component

    def _feat(self, state) -> np.ndarray:
        return np.asarray(state[self.component]).ravel().astype(np.float64)

    def _current(self, source: StateSource) -> np.ndarray:
        return self._feat(source.get_state())

    def names(self) -> list[str]:
        return [m["name"] for m in self.store.list()]

    def remember(self, name: str, source: StateSource, note: str = "",
                 gate: float | None = None) -> bool:
        """Save the source's current state under `name`. Returns False (skips) if `gate` is set
        and an existing memory is already at least `gate`-similar (RAW cosine) — the write-gate
        that keeps the shelf sparse and legible (don't re-store what you already know)."""
        if gate is not None and self.names():
            cur = self._current(source)
            for m in self.store.list():
                snap = self.store.load(m["name"])
                if self.component in snap.state and _cosine(cur, self._feat(snap.state)) >= gate:
                    return False
        self.store.save(name, source, note=note)
        return True

    def recall(self, source: StateSource, name: str) -> None:
        """Load a stored memory back into the live source (exact restore — the M4 path)."""
        self.store.into(source, name)

    def nearest(self, source: StateSource, k: int = 3, center: bool = True) -> list[Match]:
        """The k stored memories whose state is most like the source's current state.

        With `center` (default), the query and memories are mean-centered by the shelf before
        cosine — neural states share a large common direction that swamps raw cosine, so
        centering surfaces what's *distinct* about each memory (the content, not the boilerplate).
        Centering needs >=3 memories to be meaningful; below that it falls back to raw cosine.
        """
        feats, metas = [], []
        for m in self.store.list():
            snap = self.store.load(m["name"])
            if self.component in snap.state:
                feats.append(self._feat(snap.state))
                metas.append(m)
        if not feats:
            return []
        F = np.stack(feats)
        cur = self._current(source)
        if center and len(feats) >= 3:
            mu = F.mean(axis=0)
            F = F - mu
            cur = cur - mu
        out = [Match(metas[i]["name"], _cosine(cur, F[i]), metas[i].get("note", ""))
               for i in range(len(metas))]
        return sorted(out, key=lambda x: -x.similarity)[:k]

    def associate(self, source: StateSource) -> Match | None:
        """The single nearest memory — 'what does this remind you of?'"""
        top = self.nearest(source, k=1)
        return top[0] if top else None
