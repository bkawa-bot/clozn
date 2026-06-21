"""
clozn.spine — the state-stream protocol (Clozn's backbone).

Everything in Clozn is an *evolving internal state*. A StateSource (substrate adapter) advances
one step at a time and emits a StateStep: a substrate-agnostic snapshot of the model's internal
state plus metadata. Consumers (viz, probes, memory, loggers) observe the stream; interventions
(steer/edit/restore) write back through the source. This is a direct generalization of Cloze's
two load-bearing invariants:

    Cloze event spine (typed events; viz/bench are consumers)  ->  the StateStep stream here
    Cloze ModelAdapter seam (one place a backend lives)        ->  the StateSource interface here

So Clozn isn't a rewrite of Cloze — it's Cloze's architecture pointed at internal state instead
of output tokens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Protocol

import numpy as np

# A named bag of per-component state tensors, e.g. {"layer0.S": ndarray, "layer1.S": ndarray}.
# Substrate-agnostic: a diffusion canvas, an RWKV state matrix, or AR activations all fit here.
State = dict[str, np.ndarray]


@dataclass
class Readout:
    """A probe / feature reading.

    Honesty-in-interpretability (Cloze invariant 5, generalized): a value never travels without
    its confidence, and never *claims causality* until it's been verified. `causal_verified`:
    None = not checked, True/False = patched-and-measured (see ops.verify_causal).
    """
    name: str
    value: Any
    confidence: float = 1.0
    causal_verified: bool | None = None


@dataclass
class StateStep:
    """One step of the evolving state — the unit the whole product is built around."""
    step: int
    token: Any = None                                   # the input consumed this step (id/str)
    state: State = field(default_factory=dict)          # the internal state AFTER this step
    readouts: list[Readout] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)  # writes, rank, confidence, anything

    def copy(self) -> "StateStep":
        return StateStep(
            self.step, self.token,
            {k: v.copy() for k, v in self.state.items()},
            list(self.readouts), dict(self.meta),
        )


@dataclass
class Intervention:
    """A write back into the state: steer a direction, edit a slot, restore a snapshot."""
    kind: str                                           # "steer" | "edit" | "restore" | "patch"
    fn: Callable[[State], State] | None = None          # how to transform the state
    note: str = ""


class StateSource(Protocol):
    """A substrate adapter — diffusion canvas, recurrent state, AR taps — behind ONE interface.
    The only place a model backend lives (the Cloze adapter-seam, generalized)."""
    def reset(self) -> None: ...
    def step(self, x: Any) -> StateStep: ...            # advance one step, emit the state
    def get_state(self) -> State: ...                   # grab the full current state (snapshot)
    def set_state(self, s: State) -> None: ...          # write it back (restore / edit / steer)


class Consumer(Protocol):
    """A pure observer of the stream: a viz, a probe, a logger, a memory writer."""
    def on_step(self, step: StateStep) -> None: ...


class Spine:
    """Drives a StateSource over inputs and fans each StateStep to consumers.

    Consumers never own the state — the source does (Cloze invariant 4: the scheduler/source
    writes state, consumers only read). This keeps viz/probe/log/memory fully decoupled.
    """
    def __init__(self, source: StateSource, consumers: list[Consumer] | None = None):
        self.source = source
        self.consumers = consumers or []

    def run(self, inputs: list[Any]) -> Iterator[StateStep]:
        self.source.reset()
        for x in inputs:
            st = self.source.step(x)
            for c in self.consumers:
                c.on_step(st)
            yield st
