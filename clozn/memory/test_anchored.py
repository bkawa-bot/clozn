"""Tests for anchored memory (anchored.py) -- fake DirProvider only, no engine, no network.

The fake provider hands each known word a distinct one-hot direction (d=16), so fits are exact,
deterministic, and the OMP behavior is analytically checkable. Store tests repoint BAGS_PATH at tmp.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from clozn.memory import anchored

D = 16
CARD = {"id": "card_kyoto", "kind": "preference",
        "text": "The user's favorite city is Kyoto and they love quiet temple gardens."}
# content_words(CARD.text) -> favorite, city, kyoto, quiet, temple, gardens (6 words)
WORDS = ["favorite", "city", "kyoto", "quiet", "temple", "gardens"]


class FakeProvider:
    """One-hot direction per known word; counts calls so purity claims are assertable."""

    def __init__(self, words=WORDS, known=None):
        self.vecs = {}
        for i, w in enumerate(words):
            v = np.zeros(D, dtype=np.float32)
            v[i % D] = 1.0
            self.vecs[w] = v
        if known is not None:                      # restrict to a subset (unknowns -> None)
            self.vecs = {w: v for w, v in self.vecs.items() if w in known}
        self.calls = 0

    def dir_of_token(self, token):
        self.calls += 1
        return self.vecs.get(token)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(anchored, "BAGS_PATH", str(tmp_path / "bags.json"))
    return tmp_path


# ------------------------------------------------------------------------------- fit

def test_fit_content_card_produces_bag(store):
    p = FakeProvider()
    r = anchored.fit_bag(CARD, p, k=4)
    assert r["refused"] is False
    bag = r["bag"]
    assert bag["card_id"] == "card_kyoto"
    assert 1 <= bag["k"] <= 4 and bag["k_requested"] == 4
    alphas = [abs(t["alpha"]) for t in bag["terms"]]
    assert alphas == sorted(alphas, reverse=True)               # |alpha| desc
    assert 0.0 < bag["reconstruction_cos"] <= 1.0 + 1e-9
    v = np.asarray(bag["vector"])
    assert v.shape == (D,) and abs(np.linalg.norm(v) - 1.0) < 1e-6   # stored unit direction
    assert bag["layer"] == anchored.LAYER and bag["scale"] == anchored.SCALE
    assert set(t["token"] for t in bag["terms"]) <= set(WORDS)


def test_style_card_refused_by_kind_and_by_text(store):
    p = FakeProvider()
    by_kind = anchored.fit_bag({"id": "s1", "kind": "style", "text": "warm cheerful voice"}, p)
    assert by_kind["refused"] and by_kind["reason"] == anchored.REFUSAL_STYLE_RULE
    by_text = anchored.fit_bag(
        {"id": "s2", "kind": "preference", "text": "Always respond concisely and keep it brief"}, p)
    assert by_text["refused"] and by_text["reason"] == anchored.REFUSAL_STYLE_RULE


def test_too_few_resolvable_words_stays_prompt_mode(store):
    p = FakeProvider(known={"kyoto", "temple"})                 # only 2 of 6 resolve
    r = anchored.fit_bag(CARD, p)
    assert r["refused"] and "prompt-mode" in r["reason"]


# ------------------------------------------------------------------------------- store

def test_store_roundtrip_toggle_and_remove(store):
    p = FakeProvider()
    bag = anchored.fit_bag(CARD, p, k=4)["bag"]
    assert anchored.put_bag(bag) is not None
    assert anchored.get_bag("card_kyoto")["card_id"] == "card_kyoto"
    assert len(anchored.active_bags()) == 1
    assert anchored.set_on("card_kyoto", False)["on"] is False
    assert anchored.active_bags() == []
    assert anchored.compile_steer() is None                     # off -> nothing composes
    anchored.set_on("card_kyoto", True)
    assert anchored.compile_steer() is not None
    assert anchored.remove_bag("card_kyoto") is True
    assert anchored.get_bag("card_kyoto") is None
    assert anchored.remove_bag("card_kyoto") is False           # already gone


def test_corrupt_store_is_empty_store(store):
    with open(anchored.BAGS_PATH, "w") as f:
        f.write("{not json")
    assert anchored.load_bags() == {}
    p = FakeProvider()
    bag = anchored.fit_bag(CARD, p, k=4)["bag"]
    assert anchored.put_bag(bag) is not None                    # recovers by overwriting


def test_public_bag_strips_the_raw_vector(store):
    p = FakeProvider()
    bag = anchored.fit_bag(CARD, p, k=4)["bag"]
    pub = anchored.public_bag(bag)
    assert "vector" not in pub and pub["terms"] == bag["terms"]


# ------------------------------------------------------------------------------- compose

def _unit_bag(card_id, dim, terms=None):
    v = np.zeros(D)
    v[dim] = 1.0
    return {"card_id": card_id, "on": True, "vector": [float(x) for x in v],
            "terms": terms or [{"token": f"w{dim}", "alpha": 1.0}]}


def test_compile_budget_two_bags():
    a, b = _unit_bag("a", 0), _unit_bag("b", 1)
    out = anchored.compile_steer([a, b], gates={"a": 0.8, "b": 0.4})
    assert out["ok"] and out["layer"] == anchored.LAYER
    # v = normalize(0.8*e0 + 0.4*e1); s_total = SCALE * max(g) = 0.5 * 0.8
    v = np.asarray(out["vector"])
    assert abs(np.linalg.norm(v) - 1.0) < 1e-9
    expected = np.zeros(D); expected[0] = 0.8; expected[1] = 0.4
    expected /= np.linalg.norm(expected)
    assert np.allclose(v, expected)
    assert abs(out["s_total"] - anchored.SCALE * 0.8) < 1e-9
    assert abs(out["coef"] - out["s_total"] * anchored.BASE_NORM) < 1e-6
    assert abs(np.linalg.norm(out["steer_vec"]) - out["coef"]) < 1e-4   # pre-scaled twin
    assert {e["card_id"] for e in out["bags"]} == {"a", "b"}


def test_compile_gate_zero_drops_and_all_zero_is_none():
    a, b = _unit_bag("a", 0), _unit_bag("b", 1)
    out = anchored.compile_steer([a, b], gates={"a": 1.0, "b": 0.0})
    assert [e["card_id"] for e in out["bags"]] == ["a"]
    assert anchored.compile_steer([a, b], gates={"a": 0.0, "b": 0.0}) is None


def test_compile_missing_gate_fails_open():
    a = _unit_bag("a", 0)
    out = anchored.compile_steer([a], gates={})                 # no entry -> g = 1.0
    assert out["bags"][0]["gate"] == 1.0
    assert abs(out["s_total"] - anchored.SCALE) < 1e-9


# ------------------------------------------------------------------------------- the edit

def test_delete_term_refits_and_persists(store):
    p = FakeProvider()
    bag = anchored.fit_bag(CARD, p, k=4)["bag"]
    anchored.put_bag(bag)
    victim = bag["terms"][0]["token"]
    r = anchored.delete_term("card_kyoto", victim, p)
    assert r["ok"]
    new = r["bag"]
    assert victim not in [t["token"] for t in new["terms"]]
    assert victim not in new["candidate_bank"]                  # gone from memory, not just display
    assert new["k"] == bag["k"] - 1
    assert 0.0 < new["reconstruction_cos"] <= 1.0 + 1e-9
    assert anchored.get_bag("card_kyoto")["k"] == new["k"]      # persisted


def test_delete_last_term_deletes_the_bag(store):
    p = FakeProvider()
    bag = anchored.fit_bag(CARD, p, k=4)["bag"]
    anchored.put_bag(bag)
    for t in [t["token"] for t in bag["terms"]]:
        r = anchored.delete_term("card_kyoto", t, p)
        assert r["ok"]
    assert anchored.get_bag("card_kyoto") is None               # empty memory is no memory


def test_delete_unknown_term_or_bag_is_clean(store):
    p = FakeProvider()
    assert not anchored.delete_term("nope", "kyoto", p)["ok"]
    bag = anchored.fit_bag(CARD, p, k=4)["bag"]
    anchored.put_bag(bag)
    assert not anchored.delete_term("card_kyoto", "zzz", p)["ok"]


# ------------------------------------------------------------------------------- the receipt

def test_whatlearned_is_a_pure_lookup(store):
    p = FakeProvider()
    bag = anchored.fit_bag(CARD, p, k=4)["bag"]
    anchored.put_bag(bag)
    p.calls = 0
    out = anchored.whatlearned()
    assert p.calls == 0                                          # no provider, no engine, no generation
    assert out["note"] == anchored.WHATLEARNED_NOTE
    assert len(out["bags"]) == 1
    b = out["bags"][0]
    assert b["card_id"] == "card_kyoto" and b["terms"]
    assert b["terms"][0]["token"] in b["table"]                  # the rendered alpha table names the term


# ------------------------------------------------------------------------------- the loop guard

def test_detect_loop_fires_on_cycles_not_prose():
    assert anchored.detect_loop(["the", "cake"] * 4, window=8)             # period-2 cycle
    assert anchored.detect_loop(["a"] * 8, window=8)                       # single stutter
    prose = ["The", "quiet", "temple", "gardens", "of", "Kyoto", "draw", "visitors"]
    assert not anchored.detect_loop(prose, window=8)
    assert not anchored.detect_loop(["a", "b", "a"], window=8)             # too few pieces
    assert not anchored.detect_loop(["a"] * 8, window=1)                   # degenerate window


def test_halve_steer_scales_magnitude_keeps_direction_layer_and_bags():
    comp = {"ok": True, "layer": anchored.LAYER, "vector": [1.0, 0.0], "coef": 10.0,
            "steer_vec": [10.0, 0.0], "s_total": 0.5, "bags": [{"card_id": "a"}]}
    half = anchored.halve_steer(comp)
    assert half["s_total"] == pytest.approx(0.25)
    assert half["coef"] == pytest.approx(5.0)
    assert half["steer_vec"] == pytest.approx([5.0, 0.0])
    assert half["layer"] == comp["layer"]
    assert half["vector"] == comp["vector"]              # the raw unit direction is unchanged
    assert half["bags"] == comp["bags"]
    assert comp["steer_vec"] == [10.0, 0.0]              # a pure transform -- the input is never mutated


def test_halve_steer_none_or_empty_is_a_noop():
    assert anchored.halve_steer(None) is None
    assert anchored.halve_steer({}) == {}


# ------------------------------------------------------------------------------- store shape on disk

def test_bag_json_is_flat_and_readable(store):
    p = FakeProvider()
    anchored.put_bag(anchored.fit_bag(CARD, p, k=4)["bag"])
    with open(anchored.BAGS_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    assert "card_kyoto" in raw and raw["card_kyoto"]["envelope"] == anchored.ENVELOPE
