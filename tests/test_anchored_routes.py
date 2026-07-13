from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import clozn.memory.anchored as anchored  # noqa: E402
import clozn.memory.cards as memory_cards  # noqa: E402
from clozn.server.routes import anchored as anchored_routes  # noqa: E402


class Handler:
    def _json(self, status, obj):
        self.status = status
        self.obj = obj


class FakeProvider:
    def __init__(self):
        self.vecs = {
            "tea": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "coffee": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "gardens": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "kyoto": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        }

    def dir_of_token(self, token):
        return self.vecs.get(token)


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(anchored, "BAGS_PATH", str(tmp_path / "bags.json"))
    monkeypatch.setattr(anchored_routes, "_provider", lambda: FakeProvider())
    return tmp_path


def _call_get(path):
    h = Handler()
    assert anchored_routes.try_get(h, path) is True
    return h.status, h.obj


def _call_post(path, body):
    h = Handler()
    assert anchored_routes.try_post(h, path, body) is True
    return h.status, h.obj


def test_fit_list_and_whatlearned_are_wired(iso):
    card = memory_cards.create("likes tea coffee gardens kyoto", status="active")

    status, out = _call_post("/memory/anchored/fit", {"card_id": card["id"], "k": 4})

    assert status == 200
    assert out["ok"] is True
    assert out["bag"]["card_id"] == card["id"]
    assert out["bag"]["terms"]
    assert "lookup" in out["note"]

    status, listed = _call_get("/memory/anchored/list")
    assert status == 200
    assert listed["bags"][0]["card_id"] == card["id"]
    assert "vector" not in listed["bags"][0]

    status, learned = _call_get("/memory/anchored/whatlearned")
    assert status == 200
    assert learned["bags"][0]["card_id"] == card["id"]
    assert learned["bags"][0]["terms"]


def test_fit_refuses_style_rule_cards(iso):
    card = memory_cards.create("always answer very briefly", status="active", kind="style")

    status, out = _call_post("/memory/anchored/fit", {"card_id": card["id"]})

    assert status == 200
    assert out["ok"] is False
    assert out["refused"] is True
    assert "CONTENT" in out["reason"]
