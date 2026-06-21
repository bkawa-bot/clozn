"""Fast oracle for clozn.store — saved state must rehydrate bit-exactly and be browsable."""
import numpy as np

from clozn.ops import Snapshot, snapshot
from clozn.store import StateStore
from clozn.sources.toy_recurrent import ToyRecurrentSource


def test_save_load_round_trip_is_exact(tmp_path):
    store = StateStore(str(tmp_path))
    state = {"S": np.random.default_rng(0).standard_normal((8, 8)).astype(np.float32),
             "v": np.arange(5, dtype=np.float64)}
    store.save("mind", Snapshot("mind", state), note="a test memory")
    back = store.load("mind")
    for k in state:
        assert np.array_equal(back.state[k], state[k])
        assert back.state[k].dtype == state[k].dtype     # dtype preserved (npz exact)


def test_into_restores_a_live_source(tmp_path):
    store = StateStore(str(tmp_path))
    a = ToyRecurrentSource(list("abcd"), d=8, seed=0)
    for t in "abc":
        a.step(t)
    store.save("a_after_abc", a)

    b = ToyRecurrentSource(list("abcd"), d=8, seed=0)   # different state
    b.step("d")
    assert not np.array_equal(b.get_state()["S"], a.get_state()["S"])
    store.into(b, "a_after_abc")
    assert np.array_equal(b.get_state()["S"], a.get_state()["S"])


def test_list_returns_browsable_manifests(tmp_path):
    store = StateStore(str(tmp_path))
    store.save("one", Snapshot("one", {"S": np.zeros((2, 2))}), note="first")
    store.save("two", Snapshot("two", {"S": np.ones((2, 2))}), note="second")
    shelf = {m["name"]: m for m in store.list()}
    assert set(shelf) == {"one", "two"}
    assert shelf["two"]["note"] == "second"
    assert shelf["one"]["components"]["S"]["shape"] == [2, 2]
