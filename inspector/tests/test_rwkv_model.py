"""Gated oracle (-m model): the M1/M3/M4 milestones as assertions on the REAL RWKV-4 state.

Deterministic claims (restore, persist, state-drives-output) are asserted strictly; the
statistical ones (probe decodability, steering direction) are asserted loosely — it's a 169M
model and steering is brittle by our own guardrail. Run: pytest -m model
"""
import numpy as np
import pytest

from clozn.ops import snapshot, restore
from clozn.store import StateStore

pytestmark = pytest.mark.model


def _layer_mean(src, text):
    src.reset()
    src.feed(text)
    return src.get_state()["att_num"][0].mean(axis=1)      # (768,)


def test_m1_restore_is_bit_exact(rwkv):
    rwkv.reset()
    rwkv.feed("The capital of France is")
    snap = snapshot(rwkv, "France")
    rwkv.feed(" Tokyo. The capital of Japan is")            # overwrite the memory
    restore(rwkv, snap)
    after = rwkv.get_state()
    maxdiff = max(float(np.abs(after[n] - snap.state[n]).max()) for n in snap.state)
    assert maxdiff < 1e-6


def test_state_drives_output(rwkv):
    rwkv.reset(); rwkv.feed("The capital of France is")
    france = [t for t, _ in rwkv.top_next(3)]
    rwkv.reset(); rwkv.feed("The capital of Japan is")
    japan = [t for t, _ in rwkv.top_next(3)]
    assert france != japan                                  # the recurrent state carries context


def test_m4_persist_across_sessions(rwkv, tmp_path):
    store = StateStore(str(tmp_path))
    rwkv.reset()
    rwkv.feed("Remember this fact: the password is Maiko.")
    live = rwkv.get_state()
    store.save("mem", rwkv)

    rwkv.reset()                                            # a "fresh session"
    store.into(rwkv, "mem")
    rehydrated = rwkv.get_state()
    assert all(np.array_equal(rehydrated[n], live[n]) for n in live)   # bit-exact from disk

    probe = rwkv.encode(" The password is")
    for tid in probe:
        rwkv.step(tid)
    warm = rwkv.top_next(1)[0][0]
    rwkv.reset()                                            # cold model, same probe
    for tid in probe:
        rwkv.step(tid)
    cold = rwkv.top_next(1)[0][0]
    assert warm != cold                                     # carried state changes the output


def test_m3_probe_decodes_and_steers(rwkv):
    pos = ["I love this", "what a wonderful day", "this is great", "I feel happy",
           "an amazing experience", "everything is perfect", "a lovely surprise", "truly excellent"]
    neg = ["I hate this", "what a terrible day", "this is awful", "I feel miserable",
           "a horrible experience", "everything is ruined", "an awful surprise", "truly atrocious"]
    feats = [_layer_mean(rwkv, t) for t in pos + neg]
    labels = [1.0] * len(pos) + [-1.0] * len(neg)

    # (A) decodability: standardized probe, hold out 2+2, expect >= chance
    from clozn.ops import LinearProbe
    tr = list(range(2, len(pos))) + list(range(len(pos) + 2, len(pos) + len(neg)))
    te = [0, 1, len(pos), len(pos) + 1]
    X = np.stack([feats[i] for i in tr]); mu, sd = X.mean(0), X.std(0) + 1e-6
    probe = LinearProbe("sent", "m").fit([{"m": (feats[i] - mu) / sd} for i in tr],
                                         [labels[i] for i in tr], ridge=10.0)
    acc = np.mean([(probe.read({"m": (feats[i] - mu) / sd}).value >= 0) == (labels[i] > 0) for i in te])
    assert acc >= 0.5                                       # at least chance (loose: 169M, tiny sample)

    # (B) causality: + steering must lean more positive than - steering
    P = np.stack([rwkv_att(rwkv, t) for t in pos]).mean(0)
    N = np.stack([rwkv_att(rwkv, t) for t in neg]).mean(0)
    steer = (P - N); steer /= (np.linalg.norm(steer) + 1e-9)
    typ = float(np.mean([np.linalg.norm(rwkv_att(rwkv, t)) for t in pos + neg]))
    steer *= typ * 0.1

    pos_ids = np.array([rwkv.tok.encode(w)[0] for w in [" good", " great", " happy", " best"]])
    neg_ids = np.array([rwkv.tok.encode(w)[0] for w in [" bad", " terrible", " sad", " worst"]])
    suffix = rwkv.encode(" really")

    def balance(alpha):
        rwkv.reset(); rwkv.feed("I think it was")
        s = rwkv.get_state(); s["att_num"][0] = s["att_num"][0] + alpha * steer
        rwkv.set_state(s)
        for tid in suffix:
            rwkv.step(tid)
        p = rwkv._last_logits.softmax(-1)[0].detach().cpu().numpy()
        ps, ns = float(p[pos_ids].sum()), float(p[neg_ids].sum())
        return ps / (ps + ns) if ps + ns > 1e-9 else 0.5

    assert balance(+2.0) > balance(-2.0)                   # steering the direction moves sentiment


def rwkv_att(src, text):
    src.reset(); src.feed(text)
    return src.get_state()["att_num"][0]                   # (768, 12)
