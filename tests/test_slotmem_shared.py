"""test_slotmem_shared -- SlotMem.from_shared builds on an ALREADY-LOADED backbone (MODEL-FREE).

The studio wires the fact store to reuse the substrate's Qwen-7B (SUB.memory.model) rather than loading
a second model (NEXT_STEPS #5). from_shared is that seam. This test proves it works WITHOUT a real model
by handing it a tiny fake backbone (a deterministic linear stand-in) that exposes exactly the surface
SlotMem touches: `.model.layers[L].register_forward_hook`, `.lm_head.weight`, and a forward returning
`hidden_states` / `logits`. The load-bearing invariants: from_shared does NOT load a model (no HF, no
GPU), it REUSES the object it was handed (same tok/model identity), a write→read round-trips on the fake
substrate, and close() removes ONLY our hook (leaves the shared model registered-hook-free)."""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import clozn.memory.slotmem_qwen.store as sq  # noqa: E402

H = 16   # tiny hidden size
V = 40   # tiny vocab
NLAYER = 4


class _FakeTok:
    """Character-level tokenizer: each char -> its ordinal mod V (deterministic, no files). Enough for
    _resid_last / _next_dist / write()'s encode()."""

    def __call__(self, text, return_tensors=None):
        ids = [ord(c) % V for c in (text or "x")] or [1]
        return type("Enc", (), {"input_ids": torch.tensor([ids])})()

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % V for c in (text or "")] or [1]


class _FakeLayer(nn.Module):
    """A residual-carrying block: identity + a fixed rotation so different positions differ. Supports the
    forward hook SlotMem registers (the hook adds the injected value at the last position)."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.rot = nn.Parameter(torch.randn(H, H, generator=g) * 0.1, requires_grad=False)

    def forward(self, x):
        return x + x @ self.rot


class _FakeInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(i + 1) for i in range(NLAYER)])


class _FakeModel(nn.Module):
    """The minimum HF-causal-LM surface SlotMem uses: .model.layers, .lm_head.weight, and a __call__
    returning an object with .hidden_states (a per-layer list) and .logits. Runs the layer hooks so the
    injection actually lands (that's what read() depends on)."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.embed = nn.Embedding(V, H)
        self.embed.weight.data = torch.randn(V, H, generator=g)
        self.model = _FakeInner()
        self.lm_head = nn.Linear(H, V, bias=False)
        self.lm_head.weight.data = torch.randn(V, H, generator=g)
        self.eval()

    def parameters(self, recurse=True):        # SlotMem never calls this on the shared path, but be safe
        return super().parameters(recurse)

    def __call__(self, input_ids=None, output_hidden_states=False, **kw):
        x = self.embed(input_ids)
        hs = [x]
        for layer in self.model.layers:
            x = layer(x)                       # forward hooks (SlotMem's inject) fire here
            hs.append(x)
        logits = self.lm_head(x)
        return type("Out", (), {"hidden_states": hs, "logits": logits})()


@pytest.fixture(autouse=True)
def cpu_dev(monkeypatch):
    monkeypatch.setattr(sq, "DEV", "cpu")


def _shared():
    return _FakeModel(), _FakeTok()


# ---- the seam: build on a shared backbone, no load -------------------------------------------------

def test_from_shared_reuses_the_handed_model_and_tok():
    model, tok = _shared()
    mem = sq.SlotMem.from_shared(model, tok, layer=2)
    assert mem.model is model                  # REUSED, not reloaded
    assert mem.tok is tok
    assert mem.layer == 2
    assert mem.W_U is model.lm_head.weight      # values come from the shared unembedding
    assert mem.entries == []
    assert mem.eta > 0                          # measured once on the neutral text
    mem.close()


def test_from_shared_does_not_touch_automodel(monkeypatch):
    """The whole point: no second model load. If from_shared reached for AutoModel/AutoTokenizer the
    test would blow up here (they'd try a real download)."""
    def boom(*a, **k):
        raise AssertionError("from_shared must NOT load a model")

    monkeypatch.setattr(sq, "AutoModelForCausalLM",
                        type("X", (), {"from_pretrained": staticmethod(boom)}))
    monkeypatch.setattr(sq, "AutoTokenizer",
                        type("X", (), {"from_pretrained": staticmethod(boom)}))
    model, tok = _shared()
    mem = sq.SlotMem.from_shared(model, tok, layer=2)   # must not raise
    mem.close()


def test_write_then_read_round_trips_on_the_fake_backbone():
    model, tok = _shared()
    mem = sq.SlotMem.from_shared(model, tok, layer=2)
    # gate off (the fake model has no meaningful surprise) -- we're testing the addressing machinery
    mem.write("alpha beta gamma", " delta", gate=False)
    mem.write("one two three", " four", gate=False)
    mem.calibrate_gate()
    assert len(mem.entries) == 2
    r = mem.read("alpha beta gamma")
    assert r["hit"] == 0                        # its OWN entry addresses top-1 (collision-proof select)
    assert r["abstained"] is False
    mem.close()


def test_surgical_delete_leaves_bystanders_bit_identical():
    model, tok = _shared()
    mem = sq.SlotMem.from_shared(model, tok, layer=2)
    for cue, ans in [("aaa", " x"), ("bbb", " y"), ("ccc", " z")]:
        mem.write(cue, ans, gate=False)
    survivor_key = mem.entries[2]["key"].clone()
    del mem.entries[1]                          # the store is a plain list -> surgical
    assert len(mem.entries) == 2
    assert torch.equal(mem.entries[1]["key"], survivor_key)   # bystander untouched, bit-exact
    mem.close()


def test_close_removes_only_our_hook():
    model, tok = _shared()
    layer2 = model.model.layers[2]
    before = len(layer2._forward_hooks)
    mem = sq.SlotMem.from_shared(model, tok, layer=2)
    assert len(layer2._forward_hooks) == before + 1   # exactly our hook was added
    mem.close()
    assert len(layer2._forward_hooks) == before        # and removed -- the shared model is left clean


def test_save_load_across_two_shared_instances(tmp_path):
    """A store written by one SlotMem loads bit-exactly into a fresh SlotMem on the same shared model --
    the persistence the studio relies on for per-profile stores."""
    model, tok = _shared()
    src = sq.SlotMem.from_shared(model, tok, layer=2)
    src.write("hello world", " friend", gate=False)
    src.write("goodbye now", " later", gate=False)
    path = str(tmp_path / "p.slots.pt")
    src.save(path)
    src.close()

    dst = sq.SlotMem.from_shared(model, tok, layer=2)
    n = dst.load(path)
    assert n == 2
    assert [e["cue"] for e in dst.entries] == ["hello world", "goodbye now"]
    assert torch.equal(dst.entries[0]["key"], torch.load(path)["keys"][0])
    dst.close()
