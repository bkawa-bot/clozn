"""Fast oracle for the spine: the source owns the state, consumers only observe (invariant 4)."""
from clozn.spine import Spine, StateStep
from clozn.sources.toy_recurrent import ToyRecurrentSource


class Recorder:
    def __init__(self):
        self.seen: list[StateStep] = []

    def on_step(self, step: StateStep) -> None:
        self.seen.append(step)


def test_spine_streams_steps_to_consumers_and_resets():
    src = ToyRecurrentSource(list("abc"), d=8, seed=0)
    src.step("a")                              # dirty the state first
    rec = Recorder()
    out = list(Spine(src, [rec]).run(list("abc")))

    assert [s.step for s in out] == [1, 2, 3]          # reset() rewound the counter
    assert len(rec.seen) == 3                          # consumer saw every step
    assert [s.token for s in rec.seen] == list("abc")
    assert all(isinstance(s, StateStep) for s in rec.seen)


def test_consumers_cannot_mutate_the_sources_state():
    src = ToyRecurrentSource(list("abc"), d=8, seed=0)
    steps = list(Spine(src, []).run(["a", "b"]))
    # the StateStep carries a *copy*: scribbling on it must not corrupt the live source
    steps[-1].state["S"][:] = 999.0
    assert src.get_state()["S"].max() < 999.0
