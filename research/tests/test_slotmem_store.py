"""test_slotmem_store -- persistence round-trip for the glass-box slot memory (MODEL-FREE).

No model, no GPU, no HF download: pack_store/unpack_store are pure functions over the entry
list, and SlotMem.save/load are exercised on a __new__-constructed instance (the FakeMem
pattern of test_memory_wiring) with slotmem_qwen.DEV monkeypatched to "cpu". The load-bearing
invariants: entries survive a save/load round-trip BIT-EXACTLY (float32 torch.equal, ids/
labels/cues/answers ==, eta/gate_floor ==), and a store written at layer L refuses to load
into a SlotMem tapping a different layer (keys are residuals OF a layer).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import slotmem_qwen as sq  # noqa: E402


def _entry(i: int, h: int = 32) -> dict:
    g = torch.Generator().manual_seed(1000 + i)
    k = torch.randn(h, generator=g)
    v = torch.randn(h, generator=g)
    ans = [100 + i, 200 + i][: 1 + i % 2]                 # mix of one- and two-token answers
    return {"key": k / k.norm(), "value": v / v.norm(),
            "label": f"The cue {i} is ->  ans{i}", "ans_ids": ans,
            "cue": f"The cue {i} is", "answer": f" ans{i}"}


def _fake_mem(layer=18, entries=(), eta=103.0, gate_floor=0.274):
    """A SlotMem with no model behind it -- only the store fields save/load touch."""
    mem = sq.SlotMem.__new__(sq.SlotMem)                  # no __init__ -> no model load
    mem.layer, mem.entries = layer, list(entries)
    mem.eta, mem.gate_floor = eta, gate_floor
    return mem


@pytest.fixture(autouse=True)
def cpu_dev(monkeypatch):
    """save/load must work with no GPU in sight."""
    monkeypatch.setattr(sq, "DEV", "cpu")


# ---- pure functions ---------------------------------------------------------------------------------

def test_pack_unpack_round_trip_bit_exact():
    entries = [_entry(i) for i in range(5)]
    state = sq.pack_store(entries, layer=18, eta=103.25, gate_floor=0.274)
    back, meta = sq.unpack_store(state)
    assert len(back) == 5
    for a, b in zip(entries, back):
        assert torch.equal(a["key"], b["key"])            # bit-exact, not allclose
        assert torch.equal(a["value"], b["value"])
        assert a["ans_ids"] == b["ans_ids"]
        assert a["label"] == b["label"]
        assert a["cue"] == b["cue"]
        assert a["answer"] == b["answer"]
    assert meta == {"layer": 18, "eta": 103.25, "gate_floor": 0.274}


def test_pack_store_is_weights_only_safe(tmp_path):
    """The payload must contain only weights_only-loadable types (tensors + plain python)."""
    p = str(tmp_path / "store.pt")
    torch.save(sq.pack_store([_entry(0)], 18, 1.0, None), p)
    state = torch.load(p, weights_only=True)              # raises if anything exotic snuck in
    back, meta = sq.unpack_store(state)
    assert len(back) == 1 and meta["gate_floor"] is None


def test_unpack_rejects_unknown_version():
    state = sq.pack_store([_entry(0)], 18, 1.0, 0.5)
    state["version"] = 999
    with pytest.raises(ValueError, match="version"):
        sq.unpack_store(state)


def test_empty_store_round_trips():
    state = sq.pack_store([], layer=14, eta=78.0, gate_floor=None)
    back, meta = sq.unpack_store(state)
    assert back == []
    assert meta == {"layer": 14, "eta": 78.0, "gate_floor": None}


# ---- SlotMem.save / SlotMem.load (no model) ----------------------------------------------------------

def test_save_load_round_trip_bit_exact(tmp_path):
    entries = [_entry(i) for i in range(4)]
    src = _fake_mem(layer=18, entries=entries, eta=103.0, gate_floor=0.274)
    path = str(tmp_path / "slotmem.pt")
    src.save(path)

    dst = _fake_mem(layer=18)                             # fresh, empty
    n = dst.load(path)
    assert n == 4 and len(dst.entries) == 4
    for a, b in zip(entries, dst.entries):
        assert torch.equal(a["key"], b["key"])
        assert torch.equal(a["value"], b["value"])
        assert a["ans_ids"] == b["ans_ids"] and a["label"] == b["label"]
        assert a["cue"] == b["cue"] and a["answer"] == b["answer"]
    assert dst.eta == 103.0
    assert dst.gate_floor == 0.274


def test_save_creates_parent_dirs(tmp_path):
    mem = _fake_mem(entries=[_entry(0)])
    path = str(tmp_path / "deep" / "nested" / "slotmem.pt")
    mem.save(path)
    assert os.path.isfile(path)


def test_load_refuses_layer_mismatch(tmp_path):
    path = str(tmp_path / "slotmem.pt")
    _fake_mem(layer=18, entries=[_entry(0)]).save(path)
    dst = _fake_mem(layer=14, entries=[_entry(9)])
    with pytest.raises(ValueError, match="layer"):
        dst.load(path)
    # a refused load must leave the target store untouched
    assert len(dst.entries) == 1 and dst.entries[0]["cue"] == "The cue 9 is"


def test_gate_floor_none_survives(tmp_path):
    path = str(tmp_path / "slotmem.pt")
    _fake_mem(entries=[_entry(0)], gate_floor=None).save(path)
    dst = _fake_mem(gate_floor=0.9)
    dst.load(path)
    assert dst.gate_floor is None
