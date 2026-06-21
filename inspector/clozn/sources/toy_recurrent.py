"""
clozn.sources.toy_recurrent — a tiny delta-rule recurrent memory.

A faithful *miniature* of the RWKV-7 / Gated DeltaNet state Clozn will inspect: the state is a
key->value matrix S; each token writes via the delta rule (replace the value stored at its key);
a query reads S @ k. Pure numpy, zero heavy deps — it proves the spine + snapshot/restore/diff/
probe end-to-end *today*, before we wire in `fla` (see fla_rwkv.py). Same interface, real model.
"""
from __future__ import annotations

import numpy as np

from ..spine import State, StateStep


class ToyRecurrentSource:
    def __init__(self, vocab: list[str], d: int = 16, beta: float = 1.0, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.vocab = vocab
        K = rng.standard_normal((len(vocab), d))
        K /= np.linalg.norm(K, axis=1, keepdims=True)          # unit keys
        Vv = rng.standard_normal((len(vocab), d))
        self.K = {t: K[i] for i, t in enumerate(vocab)}
        self.V = {t: Vv[i] for i, t in enumerate(vocab)}
        self.d, self.beta = d, beta
        self.reset()

    def reset(self) -> None:
        self.S = np.zeros((self.d, self.d))
        self.t = 0

    def step(self, x: str) -> StateStep:
        k, v = self.K[x], self.V[x]
        write = self.beta * np.outer(v - self.S @ k, k)        # delta rule
        self.S = self.S + write
        self.t += 1
        meta = {
            "wrote": x,
            "write_norm": float(np.linalg.norm(write)),
            "rank": int(np.linalg.matrix_rank(self.S, tol=1e-6)),
            "energy": float(np.linalg.norm(self.S)),
        }
        return StateStep(self.t, x, {"S": self.S.copy()}, meta=meta)

    def get_state(self) -> State:
        return {"S": self.S.copy()}

    def set_state(self, s: State) -> None:
        self.S = s["S"].copy()

    def recall(self, t: str) -> float:
        """A toy probe: is token `t` currently stored? Cosine of the read-out vs its value."""
        r = self.S @ self.K[t]
        v = self.V[t]
        denom = (np.linalg.norm(r) * np.linalg.norm(v)) or 1.0
        return float(r @ v / denom)
