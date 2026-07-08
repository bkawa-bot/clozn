"""test_topic_gate -- the TOPIC-RELEVANCE (+ openness) gate math, with the embedder MOCKED (no downloads).

topic_gate.TopicGate wraps a small sentence-transformer (all-MiniLM-L6-v2) to score how strongly the
consolidated memory prefix should fire for a prompt, from two signals:
  * topic    = max cosine(prompt, active rule texts)
  * openness = max cosine(prompt, OPEN_PERSONAL_REFS)
and returns gate = max(map(topic, lo_t, hi_t), map(openness, lo_o, hi_o)) in [0,1]. The safety contract:
gate is 1.0 (NO gating, the always-on baseline) when the embedder is unavailable OR there are no rules.

These tests must NOT trigger a real model load/download. Importing topic_gate does not load anything (the
model is lazy), and we inject a FAKE encoder (a deterministic text->unit-vector map) onto each TopicGate
instance so _ensure_model() short-circuits (self._model is already set) and .encode is our fake. So the
whole suite is offline, model-free, and fast.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn import topic_gate  # noqa: E402  (import is lazy -- no model load happens here)
from clozn.topic_gate import OPEN_PERSONAL_REFS, TopicGate  # noqa: E402


# --------------------------------------------------------------------------- fake embedder
# A controllable encoder: each registered text maps to a fixed 2-D UNIT vector, so the dot product between
# any two registered texts is EXACTLY the cosine we chose. Angles let us set precise cosines:
#   vec(theta) = (cos theta, sin theta);  cos(vec(a), vec(b)) == cos(a - b).
def _unit(theta: float) -> np.ndarray:
    return np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)


class FakeEncoder:
    """Stands in for SentenceTransformer: .encode(text, normalize_embeddings=True) -> a fixed unit vector.

    `angles` maps known text -> angle (radians). Unknown text -> a vector orthogonal-ish to everything
    (angle 3.0 rad, well away from 0), i.e. low cosine to the registered anchors. Records call count so a
    test can assert caching (the same text is encoded at most once)."""

    def __init__(self, angles: dict[str, float]):
        self.angles = angles
        self.calls = 0

    def encode(self, text, normalize_embeddings=False):   # noqa: D401 (SentenceTransformer contract)
        self.calls += 1
        theta = self.angles.get(text, 3.0)
        return _unit(theta)


def make_gate(angles: dict[str, float]) -> TopicGate:
    """A TopicGate whose model is the FakeEncoder -- so _ensure_model() sees a live model and never imports
    or downloads sentence-transformers."""
    g = TopicGate()
    g._model = FakeEncoder(angles)      # short-circuits _ensure_model (self._model is not None)
    g.ok = True
    return g


# Angle offsets that yield a target cosine to the prompt (prompt at angle 0): cos(delta) == target.
def _angle_for_cos(target: float) -> float:
    return math.acos(max(-1.0, min(1.0, target)))


# --------------------------------------------------------------------------- scalar: topic band mapping

def test_scalar_below_lo_is_zero():
    # best topic cosine BELOW lo_t -> gate 0 (off-topic; memory off). Openness also low (unknown refs).
    lo_t, hi_t = topic_gate.lo_t, topic_gate.hi_t
    prompt, rule = "P", "R"
    angles = {prompt: 0.0, rule: _angle_for_cos(lo_t - 0.05)}    # topic cos just under lo_t
    g = make_gate(angles)
    assert g.scalar(prompt, [rule]) == 0.0


def test_scalar_above_hi_is_one():
    # best topic cosine ABOVE hi_t -> gate 1 (fully on-topic).
    lo_t, hi_t = topic_gate.lo_t, topic_gate.hi_t
    prompt, rule = "P", "R"
    angles = {prompt: 0.0, rule: _angle_for_cos(hi_t + 0.1)}
    g = make_gate(angles)
    assert g.scalar(prompt, [rule]) == 1.0


def test_scalar_linear_between_lo_and_hi():
    # a cosine exactly MIDWAY in the topic band -> gate ~0.5 (linear ramp). Openness kept low.
    lo_t, hi_t = topic_gate.lo_t, topic_gate.hi_t
    mid = (lo_t + hi_t) / 2.0
    prompt, rule = "P", "R"
    angles = {prompt: 0.0, rule: _angle_for_cos(mid)}
    g = make_gate(angles)
    got = g.scalar(prompt, [rule])
    assert abs(got - 0.5) < 1e-3, got


def test_scalar_uses_the_best_matching_rule():
    # topic = MAX cosine over rules -> one strongly-matching rule opens the gate even with weak others.
    lo_t, hi_t = topic_gate.lo_t, topic_gate.hi_t
    prompt = "P"
    angles = {prompt: 0.0, "weak": _angle_for_cos(lo_t - 0.1),
              "strong": _angle_for_cos(hi_t + 0.1)}
    g = make_gate(angles)
    assert g.scalar(prompt, ["weak", "strong"]) == 1.0


# --------------------------------------------------------------------------- openness signal

def test_high_openness_low_topic_gives_high_gate():
    # An OPEN personal ask (high cosine to the open refs) but UNRELATED to any rule (low topic) still fires:
    # gate driven by the openness signal. This is the "What should I do today?" case a topical memory should
    # answer even though it names no remembered topic.
    lo_o, hi_o = topic_gate.lo_o, topic_gate.hi_o
    prompt, rule = "P", "R"
    # topic cos very low; every open ref set to a cosine above hi_o so openness maps to 1.0.
    angles = {prompt: 0.0, rule: _angle_for_cos(0.0)}           # ~orthogonal rule -> topic ~0
    for ref in OPEN_PERSONAL_REFS:
        angles[ref] = _angle_for_cos(hi_o + 0.05)
    g = make_gate(angles)
    assert g.scalar(prompt, [rule]) == 1.0


def test_low_topic_and_low_openness_gives_zero():
    # An unrelated SPECIFIC task (write a cover letter / debug code): low on BOTH signals -> gate ~0. This
    # is precisely the over-bleed the gate must stop.
    prompt, rule = "P", "R"
    angles = {prompt: 0.0, rule: _angle_for_cos(0.0)}           # topic ~0
    for ref in OPEN_PERSONAL_REFS:
        angles[ref] = _angle_for_cos(0.0)                       # openness ~0 too
    g = make_gate(angles)
    assert g.scalar(prompt, [rule]) == 0.0


def test_openness_band_is_higher_than_topic_band():
    # Guard the design invariant: the openness band is SHIFTED HIGHER than the topic band (both its floor
    # and ceiling), so only genuinely-general asks open the gate via openness. A merely topic-adjacent
    # cosine that would open the topic band should NOT clear the openness floor.
    assert topic_gate.lo_o > topic_gate.lo_t, "openness floor must sit above the topic floor"
    assert topic_gate.hi_o > topic_gate.hi_t, "openness ceiling must sit above the topic ceiling"
    lo_o = topic_gate.lo_o
    prompt, rule = "P", "R"
    # A topic-specific-but-impersonal ask ("how do I learn guitar"): ~unrelated to the rule (topic ~0) AND
    # open-refs cosine just UNDER lo_o -> openness maps to 0 -> the memory stays OFF (no over-bleed).
    angles = {prompt: 0.0, rule: _angle_for_cos(0.0)}
    for ref in OPEN_PERSONAL_REFS:
        angles[ref] = _angle_for_cos(lo_o - 0.02)
    g = make_gate(angles)
    assert g.scalar(prompt, [rule]) == 0.0


# --------------------------------------------------------------------------- safety contract: gate == 1.0

def test_empty_rules_is_no_gating():
    # no active memory -> gate 1.0 (baseline; the studio behaves exactly as before gating).
    g = make_gate({"P": 0.0})
    assert g.scalar("P", []) == 1.0
    assert g.scalar("P", None) == 1.0


def test_embedder_unavailable_is_no_gating():
    # ok=False (sentence-transformers absent / model failed to load) -> gate 1.0, no regression.
    g = TopicGate()
    g.ok = False
    # even WITH rules, an unavailable embedder must not gate anything down.
    assert g.scalar("P", ["some rule"]) == 1.0
    assert g.relevance("P", ["some rule"]) == {}
    assert g.openness("P") == 0.0


def test_ensure_model_failure_latches_off_and_no_gating():
    # If the real load path is exercised and fails, _ensure_model must latch ok=False -> no gating. We force
    # failure by pointing at a bogus model name AND blocking the import via a stub that raises.
    import builtins
    real_import = builtins.__import__

    def boom(name, *a, **k):
        if name == "sentence_transformers" or name.startswith("sentence_transformers."):
            raise ImportError("blocked in test")
        return real_import(name, *a, **k)

    builtins.__import__ = boom
    try:
        g = TopicGate()                     # fresh: no _model injected -> _ensure_model will try to import
        assert g.scalar("P", ["r"]) == 1.0  # import blocked -> latch off -> no gating
        assert g.ok is False
    finally:
        builtins.__import__ = real_import


# --------------------------------------------------------------------------- relevance() per-rule cosines

def test_relevance_returns_per_rule_cosines():
    prompt = "P"
    angles = {prompt: 0.0, "r1": _angle_for_cos(0.4), "r2": _angle_for_cos(0.1)}
    g = make_gate(angles)
    rel = g.relevance(prompt, ["r1", "r2"])
    assert set(rel) == {"r1", "r2"}
    assert abs(rel["r1"] - 0.4) < 1e-3
    assert abs(rel["r2"] - 0.1) < 1e-3


def test_relevance_empty_when_no_rules_or_unavailable():
    g = make_gate({"P": 0.0})
    assert g.relevance("P", []) == {}
    g2 = TopicGate(); g2.ok = False
    assert g2.relevance("P", ["r"]) == {}


def test_embeddings_are_cached_by_string():
    # the same text is encoded at most once (rule texts + prompt are re-embedded across scalar/relevance).
    prompt, rule = "P", "R"
    angles = {prompt: 0.0, rule: _angle_for_cos(0.3)}
    g = make_gate(angles)
    g.scalar(prompt, [rule])
    calls_after_first = g._model.calls
    g.scalar(prompt, [rule])                 # a second identical call embeds nothing new
    assert g._model.calls == calls_after_first


# --------------------------------------------------------------------------- debug() surface

def test_debug_exposes_both_signals_and_relevance():
    lo_t, hi_t = topic_gate.lo_t, topic_gate.hi_t
    prompt, rule = "P", "R"
    angles = {prompt: 0.0, rule: _angle_for_cos((lo_t + hi_t) / 2.0)}
    for ref in OPEN_PERSONAL_REFS:
        angles[ref] = _angle_for_cos(0.0)    # low openness so the reported gate == the topic mapping
    g = make_gate(angles)
    d = g.debug(prompt, [rule])
    assert set(d) >= {"gate", "topic", "openness", "relevance", "ok"}
    assert d["ok"] is True
    assert rule in d["relevance"]
    assert abs(d["topic"] - (lo_t + hi_t) / 2.0) < 1e-3
    assert 0.0 <= d["openness"] <= 1.0
    assert abs(d["gate"] - 0.5) < 1e-2       # midway topic, low openness -> gate ~0.5


def test_debug_degrades_when_unavailable():
    g = TopicGate(); g.ok = False
    d = g.debug("P", ["r"])
    assert d["gate"] == 1.0 and d["ok"] is False


# --------------------------------------------------------------------------- singleton accessor

def test_get_gate_is_a_singleton():
    a = topic_gate.get_gate()
    b = topic_gate.get_gate()
    assert a is b
    assert isinstance(a, TopicGate)
