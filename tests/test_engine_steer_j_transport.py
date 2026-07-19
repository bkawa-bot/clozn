"""test_engine_steer_j_transport.py -- EngineSteer's optional J-transport wiring
(clozn/behavior/steering/engine_adapter.py, see its class docstring + jlens_transport.py).

Covers the PRODUCT integration point the task asked for: EngineSteer's diff-of-means tone
directions (axes.py) had NO J-transport step at all before this change. J-transport is OFF by
default (so every existing EngineSteer test/call site is byte-for-byte unaffected -- see
test_engine_library_dials.py / test_engine_add_custom.py / test_engine_substrate.py, none of
which pass j_transport=); this suite only exercises the NEW opt-in surface.

Model-free and GPU-free: a deterministic fake engine client (mirrors test_engine_add_custom.py's
_FakeEC) and a tiny synthetic on-disk J-lens sidecar (mirrors test_jlens_transport.py's fixture)
-- no real engine, no GPU, no real ~/.clozn/jlens.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from clozn.behavior.steering.engine_adapter import EngineSteer  # noqa: E402


class _FakeHarvest:
    def __init__(self, activations):
        self.activations = activations


class _FakeEC:
    """Deterministic, text-dependent [n_tokens, N_EMBD] activations -- same recipe as
    test_engine_add_custom.py's _FakeEC, so warm's pos/neg poles produce a real, reproducible,
    non-degenerate diff-of-means direction with no live model."""

    N_EMBD = 16
    N_TOKENS = 5

    def harvest(self, text, layer=None):
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        return _FakeHarvest(rng.randn(self.N_TOKENS, self.N_EMBD).astype(np.float32))

    def complete(self, prompt, **params):
        return {"choices": [{"text": "baseline"}]}

    def intervene(self, prompt, **params):
        return {"choices": [{"text": "steered"}]}


def _write_jlens_fixture(tmp_path, *, d_model=16, layer=21, model="fixture-model"):
    """Symmetric J (see test_jlens_transport.py's _symmetric_J) so its own compact-vs-dense math
    is already proven elsewhere; this suite only needs "some real transport happens"."""
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
    sv = np.linspace(10.0, 1.0, d_model)
    J = ((q * sv) @ q.T).astype(np.float32)
    jdir = tmp_path / "jlens"
    jdir.mkdir()
    J.astype("<f2").tofile(str(jdir / f"J_layer{layer}.f16"))
    manifest = {"model": model, "d_model": d_model, "vocab": d_model, "layers": [layer],
                "engine_default_tap_layer": layer}
    (jdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return str(jdir)


# ==================================================================================== default: OFF, byte-for-byte unchanged

def test_j_transport_is_off_by_default():
    es = EngineSteer(_FakeEC(), layer=21)
    vec = es.steer_vector({"warm": 1.0})
    assert vec is not None
    assert es.last_j_transport is None       # never even attempted -- the "not requested" state


def test_j_transport_disabled_matches_enabled_with_no_matching_artifact(tmp_path):
    """Enabling J-transport with NOTHING to find for this model must reproduce the untransported
    vector exactly -- the no-op guarantee, exercised through the real EngineSteer surface."""
    ec = _FakeEC()
    baseline = EngineSteer(ec, layer=21)
    baseline_vec = baseline.steer_vector({"warm": 1.0})

    missing_dir = str(tmp_path / "no_such_jlens")
    es = EngineSteer(_FakeEC(), layer=21, j_transport=True, jlens_dir=missing_dir)
    vec = es.steer_vector({"warm": 1.0})

    assert vec == pytest.approx(baseline_vec)
    assert es.last_j_transport is not None
    assert es.last_j_transport["applied"] is False
    assert es.last_j_transport["reason"] == "no_jlens_artifact"


# ==================================================================================== enabled + matching artifact: applied

def test_j_transport_applies_and_changes_the_vector_when_a_matching_artifact_exists(tmp_path):
    jdir = _write_jlens_fixture(tmp_path, d_model=_FakeEC.N_EMBD, layer=21, model="fixture-model")

    ec_baseline = _FakeEC()
    baseline = EngineSteer(ec_baseline, layer=21)
    baseline_vec = np.asarray(baseline.steer_vector({"warm": 1.0}), dtype=np.float32)

    es = EngineSteer(_FakeEC(), layer=21, j_transport=True, jlens_dir=jdir,
                      jlens_model_id="fixture-model")
    vec = np.asarray(es.steer_vector({"warm": 1.0}), dtype=np.float32)

    assert es.last_j_transport["applied"] is True
    assert es.last_j_transport["layer"] == 21
    assert not np.allclose(vec, baseline_vec)          # genuinely transported, not a pass-through
    # norm="preserve" (EngineSteer's default) keeps the SAME calibrated injection magnitude:
    assert float(np.linalg.norm(vec)) == pytest.approx(float(np.linalg.norm(baseline_vec)), rel=1e-3)


def test_j_transport_refuses_a_wrong_model_artifact_and_leaves_the_vector_untouched(tmp_path):
    jdir = _write_jlens_fixture(tmp_path, d_model=_FakeEC.N_EMBD, layer=21, model="fixture-model")

    ec_baseline = _FakeEC()
    baseline = EngineSteer(ec_baseline, layer=21)
    baseline_vec = baseline.steer_vector({"warm": 1.0})

    es = EngineSteer(_FakeEC(), layer=21, j_transport=True, jlens_dir=jdir,
                      jlens_model_id="some-other-model")
    vec = es.steer_vector({"warm": 1.0})

    assert es.last_j_transport["applied"] is False
    assert es.last_j_transport["reason"] == "wrong_model"
    assert vec == pytest.approx(baseline_vec)


def test_enable_j_transport_can_be_called_after_construction(tmp_path):
    jdir = _write_jlens_fixture(tmp_path, d_model=_FakeEC.N_EMBD, layer=21, model="late-bound")
    es = EngineSteer(_FakeEC(), layer=21)
    es.enable_j_transport(jlens_dir=jdir, model_id="late-bound")
    vec = es.steer_vector({"warm": 1.0})
    assert es.last_j_transport["applied"] is True

    es.disable_j_transport()
    es.steer_vector({"warm": 1.0})
    assert es.last_j_transport is None


def test_generate_also_applies_j_transport_and_sends_the_transported_vector(tmp_path):
    jdir = _write_jlens_fixture(tmp_path, d_model=_FakeEC.N_EMBD, layer=21, model="fixture-model")

    class _RecordingEC(_FakeEC):
        def __init__(self):
            self.intervene_calls = []

        def intervene(self, prompt, **params):
            self.intervene_calls.append(params)
            return {"choices": [{"text": "steered"}]}

    ec = _RecordingEC()
    es = EngineSteer(ec, layer=21, j_transport=True, jlens_dir=jdir, jlens_model_id="fixture-model")
    es.set("warm", 1.0)
    es._engaged = True
    es.generate("hello", max_new=10)

    assert len(ec.intervene_calls) == 1
    assert es.last_j_transport["applied"] is True
    sent_vec = ec.intervene_calls[0]["vector"]
    assert sent_vec == es.last_j_transport["vector"]
