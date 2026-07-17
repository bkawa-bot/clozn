"""Model-free contracts for raw engine harvest/observe and steering compatibility aliases.

The real gateway handler is driven without a socket. Fake numpy activations prove the transform is
applied at the selected position, while fake steering objects prove `/engine/steer/*` and `/steer/*`
share the same product state owner. No worker, model, Torch forward, or GPU is involved.
"""
from __future__ import annotations

import io
import json

import numpy as np
import pytest

from clozn.server import app as cs


class FakeHarvest:
    def __init__(self):
        self.tokens = ["The", " capital", " is"]
        self.activations = np.array([[3.0, 4.0], [0.0, 2.0], [1.0, 0.0]], dtype=np.float32)
        self.layer = 7
        self.n_embd = 2


class FakeObservation:
    moved_l2 = 0.625
    baseline_top = [{"token": " Paris", "prob": 0.6}, {"token": " Rome", "prob": 0.2}]
    edited_top = [{"token": " Rome", "prob": 0.55}, {"token": " Paris", "prob": 0.3}]

    def summary(self):
        return {"changed": True}

    def shifted(self):
        return True


class FakeEngine:
    def __init__(self):
        self.harvest_texts = []
        self.observe_call = None

    def harvest(self, text):
        self.harvest_texts.append(text)
        return FakeHarvest()

    def edit_and_observe(self, text, transform, positions):
        before = FakeHarvest()
        after = transform(before.activations)
        self.observe_call = {"text": text, "positions": positions, "after": after}
        return before, FakeObservation()


class FakeEngineSteer:
    ready = True

    def generate(self, prompt, strength=None, max_new=70):
        return " baseline " if strength == {} else " steered "


class FakeCanonicalSub:
    name = "engine"

    def __init__(self, steer):
        self.steer = steer
        self.axes_calls = 0

    def _steer(self, path, body):
        assert path == "/steer/axes"
        self.axes_calls += 1
        return {"axes": [{"name": "warm", "value": 0.4, "custom": True,
                           "calibrated": True, "usable_range": [0.1, 0.5]}],
                "ready": True, "substrate": "engine"}


def _dispatch(path, body=None):
    raw = json.dumps(body or {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    head, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    status = int(head.splitlines()[0].split()[1])
    return status, json.loads(payload.decode("utf-8"))


@pytest.fixture()
def fake_runtime(monkeypatch):
    engine = FakeEngine()
    steer = FakeEngineSteer()
    sub = FakeCanonicalSub(steer)
    monkeypatch.setattr(cs, "ENGINE", engine)
    monkeypatch.setattr(cs, "SUB", sub)
    monkeypatch.setattr(cs, "SUBNAME", "engine")
    monkeypatch.setattr(cs, "_engine_steer", lambda: steer)
    return engine, steer, sub


def test_engine_harvest_returns_raw_norms_and_metadata(fake_runtime):
    engine, _, _ = fake_runtime

    status, out = _dispatch("/engine/harvest", {"text": "The capital is"})

    assert status == 200
    assert engine.harvest_texts == ["The capital is"]
    assert out == {"tokens": ["The", " capital", " is"], "layer": 7, "n_embd": 2,
                   "norms": [5.0, 2.0, 1.0]}


def test_engine_observe_scales_only_the_selected_position(fake_runtime):
    engine, _, _ = fake_runtime

    status, out = _dispatch("/engine/observe", {"text": "The capital is", "position": 1, "scale": 3})

    assert status == 200
    np.testing.assert_array_equal(engine.observe_call["after"],
                                  np.array([[3.0, 4.0], [0.0, 6.0], [1.0, 0.0]], dtype=np.float32))
    assert engine.observe_call["positions"] == [1]
    assert out["shifted"] is True
    assert out["moved_l2"] == 0.625
    assert out["position"] == 1 and out["scale"] == 3.0
    assert out["baseline_top"][0]["token"] == " Paris"
    assert out["edited_top"][0]["token"] == " Rome"


def test_legacy_engine_axes_delegates_to_canonical_live_state(fake_runtime):
    _, _, sub = fake_runtime

    status, out = _dispatch("/engine/steer/axes", {})

    assert status == 200 and sub.axes_calls == 1
    assert out["axes"][0]["value"] == 0.4
    assert out["axes"][0]["calibrated"] is True
    assert out["deprecated"] is True
    assert out["canonical"] == "/steer/axes"
    assert out["engine"] is True


def test_legacy_engine_check_names_the_canonical_route(fake_runtime):
    status, out = _dispatch("/engine/steer/check",
                            {"prompt": "night", "axis": "warm", "value": 0.6, "max_tokens": 12})

    assert status == 200
    assert out["baseline"] == "baseline" and out["steered"] == "steered"
    assert out["deprecated"] is True
    assert out["canonical"] == "/steer/check"


class CheckSteer:
    def __init__(self):
        self.strength = {"warm": 0.4, "concise": 0.0}
        self._engaged = False

    def clear(self):
        self.strength = {}

    def set(self, name, value):
        self.strength[name] = float(value)

    def engage(self):
        self._engaged = True

    def disengage(self):
        self._engaged = False


class CheckSub(cs.Substrate):
    name = "engine"

    def __init__(self):
        self.steer = CheckSteer()
        self._steer_ready = True

    def _gen(self, prompt):
        return "steered" if self.steer._engaged else "baseline"


def test_canonical_steer_check_restores_the_live_persona():
    sub = CheckSub()

    out = sub._steer("/steer/check", {"prompt": "hello", "name": "formal", "value": 0.7})

    assert out["baseline"] == "baseline" and out["steered"] == "steered"
    assert sub.steer.strength == {"warm": 0.4, "concise": 0.0}
    assert sub.steer._engaged is False
