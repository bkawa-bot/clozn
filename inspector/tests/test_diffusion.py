"""Fast oracle for the diffusion substrate (Phase 2) — runs on cloze's pure-numpy FakeAdapter,
so it needs no torch and no checkpoint. Proves the SAME clozn.ops / clozn.store are
substrate-agnostic: they work on a denoising board exactly as on the recurrent matrix.

Skips cleanly if the sibling `cloze` repo isn't on disk (set CLOZE_LAB to <cloze>/lab)."""
import numpy as np
import pytest

from clozn.ops import diff, restore, snapshot
from clozn.store import StateStore

PROMPT, MAX_NEW, STEPS = "hi", 20, 8


def _src(**kw):
    try:
        from clozn.sources.diffusion import DiffusionStateSource
    except ImportError as e:                       # pragma: no cover
        pytest.skip(f"cloze_lab unavailable: {e}")
    try:
        return DiffusionStateSource(prompt=PROMPT, max_new=MAX_NEW, steps=STEPS, **kw)
    except ImportError as e:                       # pragma: no cover
        pytest.skip(f"cloze_lab unavailable: {e}")


def test_denoising_drains_the_board():
    steps = _src().run()
    assert len(steps) <= STEPS
    assert steps[-1].meta["n_masked_after"] == 0
    assert not (steps[-1].state["board"] == _src().mask).any()


def test_fills_many_slots_per_pass():
    steps = _src().run()
    assert max(s.meta["n_committed"] for s in steps) > 1   # parallel, unlike AR's one-per-step
    assert sum(s.meta["n_committed"] for s in steps) == MAX_NEW


def test_snapshot_restore_is_bit_exact():
    src = _src()
    src.reset()
    for _ in range(3):
        src.step()
    mid = snapshot(src)
    masked_at_mid = int((src.board == src.mask).sum())
    while not src.done:
        src.step()
    restore(src, mid)                              # the SAME op as the RWKV substrate
    assert int((src.board == src.mask).sum()) == masked_at_mid
    assert np.array_equal(src.get_state()["board"], mid.state["board"])


def test_diff_counts_filled_slots():
    src = _src(); src.reset()
    for _ in range(3):
        src.step()
    a = snapshot(src)
    step_b = src.step()
    b = snapshot(src)
    assert diff(a, b).total > 0
    newly = int(((b.state["filled"] - a.state["filled"]) > 0.5).sum())
    assert newly == step_b.meta["n_committed"]     # each commit fills exactly one masked slot


def test_persist_and_resume_completes(tmp_path):
    store = StateStore(str(tmp_path))
    gen = _src(); gen.reset()
    for _ in range(4):
        gen.step()
    store.save("half", gen)

    fresh = _src()
    store.into(fresh, "half")                      # the SAME store as the RWKV substrate
    while not fresh.done:
        fresh.step()
    assert int((fresh.board == fresh.mask).sum()) == 0

    uninterrupted = _src(); uninterrupted.run()
    assert np.array_equal(fresh.board, uninterrupted.board)   # resume is schedule-faithful
