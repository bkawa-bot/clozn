"""Oracle for clozn.memory (Phase 3). Fast path runs on the toy source (substrate-agnostic,
no checkpoint); a gated `-m model` path asserts the associations on real RWKV-4."""
import numpy as np
import pytest

from clozn.memory import MemoryShelf
from clozn.store import StateStore
from clozn.sources.toy_recurrent import ToyRecurrentSource


def _feed(src, seq):
    src.reset()
    for t in seq:
        src.step(t)
    return src


def test_associative_recall_finds_the_matching_state(tmp_path):
    src = ToyRecurrentSource(list("abcdef"), d=8, seed=0)
    shelf = MemoryShelf(StateStore(str(tmp_path)), component="S")
    for name, seq in [("abc", "abc"), ("def", "def"), ("ace", "ace")]:
        shelf.remember(name, _feed(src, seq), note=seq)

    _feed(src, "abc")                              # re-create the 'abc' state
    assert shelf.associate(src).name == "abc"      # nearest by state similarity
    _feed(src, "def")
    assert shelf.associate(src).name == "def"


def test_write_gate_skips_a_near_duplicate(tmp_path):
    src = ToyRecurrentSource(list("abcdef"), d=8, seed=0)
    shelf = MemoryShelf(StateStore(str(tmp_path)), component="S")
    shelf.remember("abc", _feed(src, "abc"))
    assert shelf.remember("abc_dup", _feed(src, "abc"), gate=0.99) is False   # already known
    assert shelf.remember("def", _feed(src, "def"), gate=0.99) is True        # genuinely new
    assert set(shelf.names()) == {"abc", "def"}


def test_nearest_is_sorted_and_capped(tmp_path):
    src = ToyRecurrentSource(list("abcdef"), d=8, seed=0)
    shelf = MemoryShelf(StateStore(str(tmp_path)), component="S")
    for name in "abcde":
        shelf.remember(name, _feed(src, name * 3))
    matches = shelf.nearest(_feed(src, "aaa"), k=3)
    assert len(matches) == 3
    sims = [m.similarity for m in matches]
    assert sims == sorted(sims, reverse=True)


@pytest.mark.model
def test_associates_by_topic_on_real_rwkv(rwkv, tmp_path):
    shelf = MemoryShelf(StateStore(str(tmp_path)), component="att_num")
    memories = {
        "france":  "The capital of France is Paris.",
        "japan":   "The capital of Japan is Tokyo.",
        "math":    "Two plus two equals four. Three times three is nine.",
        "weather": "It rained all day and the grey sky never cleared.",
        "cooking": "Add the flour and eggs, then bake the cake for an hour.",
    }
    for name, text in memories.items():
        rwkv.reset(); rwkv.feed(text)
        shelf.remember(name, rwkv, note=text)

    rwkv.reset(); rwkv.feed("The capital of Germany is")
    assert shelf.associate(rwkv).name in {"france", "japan"}     # geography clusters
    rwkv.reset(); rwkv.feed("Seven plus five equals")
    assert shelf.associate(rwkv).name == "math"
