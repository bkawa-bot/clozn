"""Model-free tests for profiles.py -- the portable persona bundles (CRUD, export/import, compile)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from clozn.profiles import store as P


def mk(tmp_path):
    return P.ProfileStore(root=str(tmp_path / "profiles"))


def test_round_trip_and_listing(tmp_path):
    st = mk(tmp_path)
    p = P.new_profile("work", "strictly business")
    p["cards"].append({"text": "Prefers concise technical answers", "status": "active"})
    p["dials"] = {"concise": 0.8, "warm": -0.2}
    p["facts"].append({"cue": "The user's team standup is at", "answer": " nine"})
    st.save(p)
    q = st.load("work")
    assert q["name"] == "work" and q["dials"]["concise"] == 0.8
    assert q["facts"][0]["answer"] == " nine"
    assert [x["name"] for x in st.list()] == ["work"]
    assert st.delete("work") and st.list() == []


def test_names_are_slug_safe():
    with pytest.raises(ValueError):
        P.new_profile("Not A Slug!")
    with pytest.raises(ValueError):
        P.validate({"name": "../evil"})


def test_export_import_is_the_portability_contract(tmp_path):
    st = mk(tmp_path)
    p = P.new_profile("friend")
    p["cards"].append({"text": "Loves sci-fi", "status": "active"})
    p["custom_dials"].append({"name": "cozy", "pos": "warm and homey", "neg": "cold and formal", "max": 0.6})
    st.save(p)
    dest = str(tmp_path / "friend_bundle.json")
    st.export("friend", dest)
    st2 = P.ProfileStore(root=str(tmp_path / "other_machine"))
    q = st2.import_(dest, rename="friend2")
    assert q["name"] == "friend2"
    assert q["custom_dials"][0]["pos"] == "warm and homey"   # the RECIPE travels, not the vector


def test_prompt_block_active_only():
    p = P.new_profile("x")
    p["cards"] = [{"text": "A", "status": "active"}, {"text": "B", "status": "disabled"},
                  {"text": "C", "status": "active"}]
    block = P.prompt_block(P.validate(p))
    assert "- A" in block and "- C" in block and "B" not in block
    p["cards"] = []
    assert P.prompt_block(P.validate(p)) == ""


class FakeSteer:
    def __init__(self):
        self.vecs, self.strength, self.customs = {}, {}, []
        self.cleared = 0

    def clear(self):
        self.cleared += 1
        self.strength = {}

    def set(self, name, value):
        self.strength[name] = value

    def add_custom(self, name, pos, neg, mx):
        self.vecs[name] = "dir"
        self.customs.append((name, pos, neg, mx))


class FakeMem:
    def __init__(self):
        self.writes, self.calibrated = [], False

    def write(self, cue, answer, gate=True):
        self.writes.append((cue, answer, gate))
        return {"written": True}

    def calibrate_gate(self):
        self.calibrated = True


def test_switch_applies_everything_and_isolates():
    friend = P.validate({**P.new_profile("friend"), "dials": {"warm": 0.7},
                         "cards": [{"text": "Loves sci-fi"}],
                         "facts": [{"cue": "The user's dog is named", "answer": " Biscuit"}]})
    work = P.validate({**P.new_profile("work"), "dials": {"concise": 0.8},
                       "custom_dials": [{"name": "brisk", "pos": "p", "neg": "n", "max": 0.4}],
                       "cards": [{"text": "Prefers bullet points"}],
                       "facts": [{"cue": "The user's boss is", "answer": " Marta"}]})
    steer, mem_f, mem_w = FakeSteer(), FakeMem(), FakeMem()

    r1 = P.switch(friend, steer=steer, slotmem=mem_f)
    assert steer.strength == {"warm": 0.7} and "sci-fi" in r1["prompt_block"]
    assert mem_f.writes == [("The user's dog is named", " Biscuit", False)] and mem_f.calibrated

    r2 = P.switch(work, steer=steer, slotmem=mem_w)
    # switching REPLACES: dials cleared then set; customs computed from the recipe; stores isolated.
    assert steer.cleared == 2 and steer.strength == {"concise": 0.8}
    assert steer.customs == [("brisk", "p", "n", 0.4)]
    assert mem_w.writes[0][0] == "The user's boss is"
    assert all("dog" not in w[0] for w in mem_w.writes)      # friend facts never leak into work
    assert "bullet points" in r2["prompt_block"] and "sci-fi" not in r2["prompt_block"]


def test_load_rejects_path_traversal(tmp_path):
    """load()/switch/export took an unvalidated name, so `../config` escaped the profiles dir to
    ~/.clozn/*.json. _path() now validates every name -> traversal raises ValueError (the routes turn
    that into a clean 400/404)."""
    st = mk(tmp_path)
    for bad in ("../config", "/etc/passwd", "foo/../bar", "..", "a/b", "UPPER", ".hidden"):
        with pytest.raises(ValueError):
            st._path(bad)
        with pytest.raises(ValueError):
            st.load(bad)
