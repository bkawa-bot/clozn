"""Fast, model-free oracle for the white-box ops (clozn.ops) against the real StateSource
interface (the toy delta-rule source). No torch, no checkpoint — these always run in CI."""
import numpy as np

from clozn.ops import LinearProbe, diff, edit, restore, snapshot, verify_causal
from clozn.sources.toy_recurrent import ToyRecurrentSource

VOCAB = list("abcdef")


def _src():
    return ToyRecurrentSource(VOCAB, d=16, seed=0)


def test_snapshot_restore_is_bit_exact():
    src = _src()
    for t in "abc":
        src.step(t)
    snap = snapshot(src, "after abc")
    for t in "defabc":                       # mutate the state
        src.step(t)
    assert diff(snap, snapshot(src)).total > 0     # really moved
    restore(src, snap)
    after = src.get_state()
    assert np.array_equal(after["S"], snap.state["S"])   # exact rewind


def test_diff_zero_iff_unchanged():
    src = _src()
    for t in "abc":
        src.step(t)
    a = snapshot(src)
    assert diff(a, snapshot(src)).total == 0.0
    src.step("d")
    d = diff(a, snapshot(src))
    assert d.total > 0 and d.per_component["S"] > 0


def test_edit_applies_a_function():
    src = _src()
    for t in "abc":
        src.step(t)
    edit(src, lambda s: {"S": s["S"] * 0.0})
    assert np.linalg.norm(src.get_state()["S"]) == 0.0


def test_linear_probe_recovers_a_planted_direction():
    rng = np.random.default_rng(1)
    d = 32
    w_true = rng.standard_normal(d)
    w_true /= np.linalg.norm(w_true)
    states, labels = [], []
    while len(states) < 200:                 # clear-of-boundary points (a real, separable signal)
        x = rng.standard_normal(d)
        m = x @ w_true
        if abs(m) < 0.5:
            continue
        states.append({"x": x})
        labels.append(1.0 if m > 0 else -1.0)
    tr_s, tr_y, te_s, te_y = states[:150], labels[:150], states[150:], labels[150:]
    probe = LinearProbe("planted", "x").fit(tr_s, tr_y, ridge=1e-2)
    acc = np.mean([(probe.read(s).value >= 0) == (y > 0) for s, y in zip(te_s, te_y)])
    assert acc > 0.9                         # a clear linear signal must be linearly decodable


def test_verify_causal_distinguishes_causal_from_inert():
    src = _src()
    for t in "abc":
        src.step(t)
    behavior = lambda s: s.recall("a")       # is token "a" still stored?

    inert = verify_causal(src, lambda st: {k: v.copy() for k, v in st.items()}, behavior)
    assert inert["causal"] is False and abs(inert["delta"]) <= 1e-6

    wipe = verify_causal(src, lambda st: {"S": st["S"] * 0.0}, behavior)
    assert wipe["causal"] is True and abs(wipe["delta"]) > 1e-3

    # verify_causal must leave the state exactly as it found it
    assert abs(src.recall("a") - inert["baseline"]) <= 1e-9
