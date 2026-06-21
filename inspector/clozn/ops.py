"""
clozn.ops — the white-box operations, substrate-agnostic, over the state stream.

These are the product's verbs: snapshot / restore / diff / edit / probe / verify-causal.
They work on any StateSource, because they only touch State (named tensors) through the seam.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .spine import Readout, State, StateSource


@dataclass
class Snapshot:
    label: str
    state: State


def snapshot(source: StateSource, label: str = "") -> Snapshot:
    """Grab the full current state as a graspable, restorable object."""
    return Snapshot(label, {k: v.copy() for k, v in source.get_state().items()})


def restore(source: StateSource, snap: Snapshot) -> None:
    """Rewind the model's internal state to a snapshot."""
    source.set_state({k: v.copy() for k, v in snap.state.items()})


def edit(source: StateSource, fn: Callable[[State], State]) -> None:
    """Mutate the state in place (steer a direction, zero a slot, etc.)."""
    source.set_state(fn(source.get_state()))


@dataclass
class Diff:
    per_component: dict[str, float]   # Frobenius norm of change per state component
    total: float


def diff(a: Snapshot, b: Snapshot) -> Diff:
    """What changed between two snapshots — e.g. which memory slots a write touched."""
    keys = set(a.state) | set(b.state)
    per: dict[str, float] = {}
    for k in keys:
        xa, xb = a.state.get(k), b.state.get(k)
        per[k] = float("nan") if xa is None or xb is None else float(np.linalg.norm((xb - xa).ravel()))
    return Diff(per, float(sum(v for v in per.values() if not np.isnan(v))))


class LinearProbe:
    """Read a concept out of the state: fit on (state, label) pairs, then read -> Readout.

    A probe shows only what's *linearly decodable*, not what the model *uses* — so always pair
    it with verify_causal before trusting it (the Exp-5b lesson, baked into the API).
    """
    def __init__(self, name: str, component: str):
        self.name, self.component = name, component
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    def _feat(self, s: State) -> np.ndarray:
        return s[self.component].ravel()

    def fit(self, states: list[State], labels: list[float], ridge: float = 1e-2) -> "LinearProbe":
        X = np.stack([self._feat(s) for s in states])
        y = np.asarray(labels, dtype=float)
        Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
        W = np.linalg.solve(Xb.T @ Xb + ridge * np.eye(Xb.shape[1]), Xb.T @ y)
        self.w, self.b = W[:-1], float(W[-1])
        return self

    def read(self, s: State) -> Readout:
        v = float(self._feat(s) @ self.w + self.b)
        return Readout(self.name, v, confidence=float(min(1.0, abs(v))))  # crude; real ver. calibrates


def verify_causal(source: StateSource,
                  intervene: Callable[[State], State],
                  behavior: Callable[[StateSource], float]) -> dict:
    """The Exp-5b method as a reusable op, and Clozn's honesty engine.

    Measure a behavior, intervene on the state, measure again, then restore. If the behavior
    moves, the direction is *causal*; if it doesn't, the readout was only correlational.
    This is the local-only superpower — a hosted API can't let you patch and re-measure.
    """
    base = behavior(source)
    snap = snapshot(source, "pre-intervention")
    edit(source, intervene)
    after = behavior(source)
    restore(source, snap)
    return {"baseline": base, "intervened": after,
            "delta": after - base, "causal": abs(after - base) > 1e-6}
