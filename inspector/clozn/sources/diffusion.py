"""
clozn.sources.diffusion — the diffusion CANVAS as a StateSource (Phase 2).

Wraps the sibling `cloze` diffusion engine behind Clozn's StateSource seam: one step() = one
denoising pass over the whole board; get_state()/set_state() expose the canvas (token ids +
per-slot confidence + filled mask + schedule position). The SAME white-box ops — snapshot /
restore / diff / persist — that work on the RWKV recurrent matrix now work on a denoising board
with ZERO changes to ops.py or store.py. That is the thesis: every substrate is an evolving
state stream, and the verbs are substrate-agnostic.

We *link* to cloze (sys.path), never fork it. Default model is cloze's deterministic FakeAdapter
(pure numpy — no torch, no checkpoint), so the trajectory is exact and fast and runs in CI; pass
any cloze ModelAdapter (Dream / LLaDA) for real text.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from ..spine import State, StateStep


def _ensure_cloze_lab() -> None:
    """Put cloze's `cloze_lab` package on sys.path (link, don't fork)."""
    try:
        import cloze_lab  # noqa: F401
        return
    except ImportError:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    roots = [os.environ.get("CLOZE_LAB"),
             os.path.normpath(os.path.join(here, "..", "..", "..", "engine", "lab"))]
    for r in roots:
        if r and os.path.isdir(os.path.join(r, "cloze_lab")):
            sys.path.insert(0, r)
            return
    raise ImportError("cloze_lab not found — set CLOZE_LAB=<path to cloze>/lab")


class DiffusionStateSource:
    """A dLLM denoising board as a Clozn StateSource. Whole-sequence, full-recompute (exact)."""

    def __init__(self, prompt: str | list[int] = "hi", max_new: int = 20, steps: int = 8,
                 adapter=None, temperature: float = 0.0, seed: int = 0):
        _ensure_cloze_lab()
        from cloze_lab.models.fake import FakeAdapter
        from cloze_lab.scheduler.policies import ConfidenceTopK
        from cloze_lab.scheduler.stepper import FixedStepper
        self.adapter = adapter or FakeAdapter(seed=seed)
        self.cfg = self.adapter.config
        self.mask = self.cfg.mask_token_id
        self.policy = ConfidenceTopK()                 # quota mode: drains evenly over `steps`
        self.stepper = FixedStepper(steps)
        self.steps = steps
        self.temperature = temperature
        self.seed = seed
        self.prompt_ids = (self.adapter.encode(prompt) if isinstance(prompt, str)
                           else [int(i) for i in prompt])
        self.max_new = max_new
        self.reset()

    def reset(self) -> None:
        self.p = len(self.prompt_ids)
        self.n = self.p + self.max_new
        self.board = np.array(self.prompt_ids + [self.mask] * self.max_new, dtype=np.int64)
        self.conf = np.zeros(self.n, dtype=np.float32)
        self.conf[:self.p] = 1.0                       # the prompt is certain
        self.filled = (self.board != self.mask).astype(np.float32)
        self.t = 0
        self.rng = np.random.default_rng(self.seed)

    @property
    def done(self) -> bool:
        return not bool((self.board == self.mask).any()) or self.t >= self.steps

    def step(self, x=None) -> StateStep:
        from cloze_lab.generate import sample_candidates
        masked = [i for i in range(self.n) if self.board[i] == self.mask]
        committed: list[tuple[int, int, float]] = []
        if masked and self.t < self.steps:
            attn = np.ones((self.n, self.n), dtype=bool)        # whole-sequence bidirectional
            fwd = self.adapter.forward(self.board, attn, logits_for=masked)
            ctx = self.stepper.context(self.t)
            cands = sample_candidates(fwd.logits, masked, temperature=self.temperature, rng=self.rng)
            for c in self.policy.select(cands, ctx).commit:
                self.board[c.pos] = c.token_id
                self.conf[c.pos] = c.confidence
                self.filled[c.pos] = 1.0
                committed.append((c.pos, c.token_id, round(float(c.confidence), 3)))
        self.t += 1
        meta = {
            "pass": self.t,
            "committed": committed,
            "n_committed": len(committed),
            "n_masked_after": int((self.board == self.mask).sum()),
            "text": self.adapter.decode(self.board[self.p:]),
            "filled_frac": round(float(self.filled.mean()), 3),
        }
        return StateStep(self.t, x, self.get_state(), meta=meta)

    def run(self) -> list[StateStep]:
        self.reset()
        out = []
        while not self.done:
            out.append(self.step())
        return out

    # --- StateSource interface (board IS the state) ---
    def get_state(self) -> State:
        return {"board": self.board.copy().astype(np.int64),
                "conf": self.conf.copy(),
                "filled": self.filled.copy(),
                "pass": np.array([self.t], dtype=np.float32)}   # schedule position is part of state

    def set_state(self, s: State) -> None:
        self.board = np.asarray(s["board"], dtype=np.int64).copy()
        self.n = self.board.shape[0]
        self.conf = np.asarray(s["conf"], dtype=np.float32).copy() if "conf" in s \
            else (self.board != self.mask).astype(np.float32)
        self.filled = (self.board != self.mask).astype(np.float32)
        if "pass" in s:
            self.t = int(np.asarray(s["pass"]).ravel()[0])

    def text(self) -> str:
        return self.adapter.decode(self.board[self.p:])
