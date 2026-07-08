"""test_facts_server -- the /facts/* studio endpoints + SlotBox wiring (NEXT_STEPS #5, MODEL-FREE).

No model, no GPU. We drive the REAL clozn_server do_POST handler (the object.__new__(H) no-socket trick,
same as test_profiles_server) against a FAKE substrate whose .memory exposes a stand-in model/tok, with
slotmem_qwen.SlotMem.from_shared monkeypatched to return a pure-python FAKE slot store (FakeSlots) that
duck-types exactly the surface SlotBox touches -- so we exercise ALL the wiring (gating, per-profile
persistence, surgical delete, the surprise-gate refusal receipt, the read receipt + slot_ms) with zero
torch forwards.

The load-bearing invariants:
  * memory_facts DEFAULTS OFF -> every op is inert (empty lists, {ok:false}), the reply path untouched.
  * /facts/mode flips the gate and persists it; ON builds the store lazily on the shared model.
  * /facts/add applies the SURPRISE GATE -- a known fact is SKIPPED with an honest receipt, not stored.
  * /facts/delete is surgical (the victim goes, the rest remain) and persists to <profile>.slots.pt.
  * /facts/read returns the honest receipt (hit / abstained / gate value / slot_ms).
  * per-profile isolation: switching profiles swaps which .slots.pt is resident (no cross-bleed).
  * the whole thing NEVER loads a second model (from_shared is the only path; __init__ is never called).
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn import clozn_server as cs   # noqa: E402
from clozn import facts_mode           # noqa: E402
from clozn import memory_cards         # noqa: E402
from clozn import memory_mode          # noqa: E402
from clozn import profiles as P        # noqa: E402
from clozn import runlog               # noqa: E402
from clozn import slotmem_qwen         # noqa: E402


# --- a pure-python fake slot store: duck-types the SlotMem surface SlotBox calls -----------------------
# "known" facts (low surprise) are the ones the gate must refuse; everything else is nonce (writable).
KNOWN = {"The capital of France is": " Paris", "Two plus two equals": " four"}


class FakeSlots:
    def __init__(self, layer=18):
        self.layer = layer
        self.entries = []
        self.gate_floor = None
        self.eta = 100.0
        self.closed = False

    # writes: nonce facts store; a KNOWN fact under the gate is skipped (the surprise-gate receipt)
    def write(self, cue, answer, gate=True):
        if gate and KNOWN.get(cue) == answer:
            return {"written": False, "surprise": 0.4}
        self.entries.append({"key": _fakekey(cue), "value": _fakekey(answer),
                             "label": cue + " ->" + answer, "ans_ids": [1],
                             "cue": cue, "answer": answer})
        return {"written": True, "surprise": 7.0}

    def calibrate_gate(self):
        self.gate_floor = 0.25 if len(self.entries) >= 3 else 0.0

    def read(self, query, gated=False, entries=None):
        pool = self.entries if entries is None else entries
        if not pool:
            return {"dist": None, "hit": None, "sim": None, "abstained": True}
        # exact-cue match wins; else "abstain" (a drifted query) so the abstention path is exercised
        idx = next((i for i, e in enumerate(pool) if e["cue"] == query), None)
        if idx is None:
            return {"dist": None, "hit": None, "sim": 0.10, "abstained": bool(gated)}
        return {"dist": None, "hit": idx, "sim": 0.95, "abstained": False}

    # persistence: write/read a trivial JSON sidecar so per-profile isolation is REALLY tested on disk
    def save(self, path):
        path = os.path.expanduser(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"layer": self.layer, "cues": [(e["cue"], e["answer"]) for e in self.entries]}, f)
        return path

    def load(self, path):
        with open(os.path.expanduser(path), encoding="utf-8") as f:
            d = json.load(f)
        if d.get("layer") != self.layer:
            raise ValueError("layer mismatch")
        self.entries = [{"key": _fakekey(c), "value": _fakekey(a), "label": c + " ->" + a,
                         "ans_ids": [1], "cue": c, "answer": a} for c, a in d["cues"]]
        return len(self.entries)

    def close(self):
        self.closed = True


def _fakekey(s):
    import torch
    return torch.ones(4) * (len(str(s)) % 7 + 1)


# --- fake substrate: .memory exposes a model/tok so SlotBox._shared_model() succeeds ------------------

class _FakeMem:
    def __init__(self):
        self.model = object()   # opaque -- from_shared is patched, so the identity is all that matters
        self.tok = object()


class _FakeSteer:
    def __init__(self):
        self.vecs, self.strength = {}, {}

    def clear(self):
        self.strength = {}

    def set(self, name, value):
        self.strength[name] = value


class FakeSub(cs.Substrate):
    name = "qwen"

    def __init__(self):
        self.memory = _FakeMem()
        self.memory.rules = []          # _start_retrain / _mem_sync_rules read + write this
        self._mem = self.memory
        self.steer = _FakeSteer()

    def handle(self, path, body):
        return None


# --- driving the real handler without a socket (mirrors test_profiles_server) -------------------------

def _dispatch(method, path, body_obj=None):
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"{method} {path} HTTP/1.1", "HTTP/1.1", method
    getattr(h, f"do_{method}")()
    _, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


def _post(path, body_obj=None):
    return _dispatch("POST", path, body_obj)


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    """Isolate settings + the profiles dir; patch from_shared to hand back a FakeSlots; install a fake
    substrate + a fresh SlotBox. memory_facts starts OFF (the real default)."""
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(facts_mode, "PROFILES_DIR", str(tmp_path / "profiles"))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(P.ProfileStore.__init__, "__defaults__", (str(tmp_path / "profiles"),))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "no_such.pt")])
    memory_mode.set_mode("prompt")
    monkeypatch.setattr(slotmem_qwen.SlotMem, "from_shared",
                        classmethod(lambda cls, model, tok, layer: FakeSlots(layer)))
    sub = FakeSub()
    monkeypatch.setattr(cs, "SUB", sub)
    monkeypatch.setattr(cs, "SUBNAME", "qwen")
    monkeypatch.setattr(cs, "SLOTS", None)      # fresh box each test -> lazily rebuilt against this SUB
    return tmp_path


def _enable():
    return _post("/facts/mode", {"enabled": True})


# ---- the gate: default OFF, and everything inert while off -------------------------------------------

def test_mode_defaults_off(iso):
    out = _post("/facts/mode", {})
    assert out["enabled"] is False
    assert out["layer"] == facts_mode.LAYER


def test_everything_is_inert_while_off(iso):
    assert _post("/facts/list", {})["entries"] == []
    assert _post("/facts/add", {"cue": "My cat is", "answer": " Mittens"})["ok"] is False
    assert _post("/facts/read", {"query": "My cat is"}) == {"enabled": False}
    # nothing got written to disk
    assert not os.path.isdir(str(iso / "profiles")) or not os.listdir(str(iso / "profiles"))


def test_mode_toggle_persists(iso):
    assert _enable()["enabled"] is True
    assert facts_mode.enabled() is True
    out = _post("/facts/mode", {"enabled": False})
    assert out["enabled"] is False
    assert facts_mode.enabled() is False


# ---- add: the surprise gate refusal is the receipt --------------------------------------------------

def test_add_stores_a_nonce_fact(iso):
    _enable()
    out = _post("/facts/add", {"cue": "My dog is named", "answer": " Biscuit"})
    assert out["ok"] is True and out["written"] is True
    listed = _post("/facts/list", {})
    assert [e["cue"] for e in listed["entries"]] == ["My dog is named"]
    assert listed["count"] == 1


def test_add_refuses_a_fact_the_model_already_knows(iso):
    _enable()
    out = _post("/facts/add", {"cue": "The capital of France is", "answer": " Paris"})
    assert out["ok"] is True            # the request succeeded ...
    assert out["written"] is False      # ... but the gate SKIPPED it (already known)
    assert "already knows" in out["reason"]
    assert _post("/facts/list", {})["entries"] == []   # nothing stored


def test_add_rejects_empty_input(iso):
    _enable()
    assert _post("/facts/add", {"cue": "", "answer": " x"})["ok"] is False
    assert _post("/facts/add", {"cue": "something", "answer": "   "})["ok"] is False


# ---- delete: surgical + persisted -------------------------------------------------------------------

def test_delete_by_cue_is_surgical(iso):
    _enable()
    for cue, ans in [("aaa", " x"), ("bbb", " y"), ("ccc", " z")]:
        _post("/facts/add", {"cue": cue, "answer": ans})
    out = _post("/facts/delete", {"cue": "bbb"})
    assert out["ok"] is True and out["removed"] == "bbb" and out["remaining"] == 2
    remaining = [e["cue"] for e in _post("/facts/list", {})["entries"]]
    assert remaining == ["aaa", "ccc"]      # the victim is gone, the bystanders stay, in order


def test_delete_missing_fact_is_a_clean_miss(iso):
    _enable()
    _post("/facts/add", {"cue": "aaa", "answer": " x"})
    out = _post("/facts/delete", {"cue": "zzz"})
    assert out["ok"] is False and "no matching" in out["reason"]


def test_add_and_delete_persist_to_the_profile_store(iso):
    _enable()
    _post("/facts/add", {"cue": "aaa", "answer": " x"})
    # the resident profile is "default" (nothing switched) -> default.slots.pt exists on disk
    assert os.path.isfile(facts_mode.store_path("default"))


# ---- read: the honest receipt -----------------------------------------------------------------------

def test_read_receipt_reports_a_hit_with_slot_ms(iso):
    _enable()
    _post("/facts/add", {"cue": "My dog is named", "answer": " Biscuit"})
    r = _post("/facts/read", {"query": "My dog is named"})
    assert r["enabled"] is True
    assert r["hit"] == 0 and r["abstained"] is False
    assert r["cue"] == "My dog is named" and r["answer"] == " Biscuit"
    assert "slot_ms" in r and isinstance(r["slot_ms"], (int, float))


def test_read_receipt_shows_abstention_on_a_drifted_query(iso):
    _enable()
    for cue, ans in [("aaa", " x"), ("bbb", " y"), ("ccc", " z")]:
        _post("/facts/add", {"cue": cue, "answer": ans})   # >=3 -> a real gate floor
    r = _post("/facts/read", {"query": "totally unrelated"})
    assert r["enabled"] is True and r["abstained"] is True
    assert r["gate_floor"] is not None     # the abstain threshold is surfaced (honest UI)


def test_read_on_an_empty_store_abstains(iso):
    _enable()
    r = _post("/facts/read", {"query": "anything"})
    assert r["enabled"] is True and r["abstained"] is True and r.get("empty") is True


# ---- per-profile isolation --------------------------------------------------------------------------

def test_switching_profiles_swaps_the_resident_store(iso, monkeypatch):
    _enable()
    # profile A (add builds the box + loads friend's -- empty -- store)
    memory_mode.set_setting("active_profile", "friend")
    _post("/facts/add", {"cue": "friend fact", "answer": " one"})
    # profile B: the resident store swaps on the next op (SlotBox._ensure_profile notices the change)
    memory_mode.set_setting("active_profile", "work")
    cs._slots_box().on_profile_switch()
    assert [e["cue"] for e in _post("/facts/list", {})["entries"]] == []    # work sees NONE of friend's
    _post("/facts/add", {"cue": "work fact", "answer": " two"})
    # back to A: friend's fact is still there, work's is not
    memory_mode.set_setting("active_profile", "friend")
    cs._slots_box().on_profile_switch()
    assert [e["cue"] for e in _post("/facts/list", {})["entries"]] == ["friend fact"]
    # two separate stores on disk
    assert os.path.isfile(facts_mode.store_path("friend"))
    assert os.path.isfile(facts_mode.store_path("work"))


# ---- it never loads a second model ------------------------------------------------------------------

def test_never_calls_slotmem_init(iso, monkeypatch):
    """SlotBox must build ONLY via from_shared. If it ever hit __init__ (a real load) this blows up."""
    def boom(self, *a, **k):
        raise AssertionError("SlotBox must use from_shared, never SlotMem.__init__")

    monkeypatch.setattr(slotmem_qwen.SlotMem, "__init__", boom)
    _enable()
    _post("/facts/add", {"cue": "aaa", "answer": " x"})   # must succeed without touching __init__
    assert _post("/facts/list", {})["count"] == 1


# ---- the facts_note seam: a profile switch compiles the bundle's facts into its store ----------------

def _bundle_with_facts(name="friend"):
    p = P.new_profile(name, "has facts")
    p["cards"] = [{"text": "Loves sci-fi", "status": "active"}]
    p["facts"] = [{"cue": "The user's dog is named", "answer": " Biscuit"},
                  {"cue": "The user's boss is", "answer": " Marta"}]
    return p


def test_switch_with_facts_off_notes_the_tier_is_off(iso):
    """Facts tier OFF (default): a switch does NOT build the store; the note says how to turn it on."""
    P.ProfileStore().save(_bundle_with_facts())
    out = _post("/profiles/switch", {"name": "friend"})
    assert out["ok"] is True
    assert out["facts_note"] is not None
    assert "off" in out["facts_note"].lower()
    # nothing compiled to disk
    assert not os.path.isfile(facts_mode.store_path("friend"))


def test_switch_with_facts_on_compiles_them_into_the_profile_store(iso):
    """The seam CLOSED: with the tier on, a switch recompiles the bundle's facts into <name>.slots.pt."""
    _enable()
    P.ProfileStore().save(_bundle_with_facts())
    out = _post("/profiles/switch", {"name": "friend"})
    assert out["ok"] is True
    assert "compiled" in out["facts_note"]
    # the store is on disk AND resident with both facts
    assert os.path.isfile(facts_mode.store_path("friend"))
    listed = [e["cue"] for e in _post("/facts/list", {})["entries"]]
    assert set(listed) == {"The user's dog is named", "The user's boss is"}


def test_switch_isolates_facts_between_profiles(iso):
    """Two profiles' fact stores stay disjoint across a switch (persona isolation, on disk + resident)."""
    _enable()
    P.ProfileStore().save(_bundle_with_facts("friend"))
    work = P.new_profile("work", "different facts")
    work["facts"] = [{"cue": "The project deadline is", "answer": " Friday"}]
    P.ProfileStore().save(work)

    _post("/profiles/switch", {"name": "friend"})
    _post("/profiles/switch", {"name": "work"})
    listed = [e["cue"] for e in _post("/facts/list", {})["entries"]]
    assert listed == ["The project deadline is"]          # work's only -- friend's didn't bleed in
    assert "Biscuit" not in json.dumps(listed)


# ---- the conversation fact-miner (pure -- no model, no server) --------------------------------------

def test_mine_fact_pulls_a_clean_declarative():
    assert cs._mine_fact("My dog is named Biscuit") == ("My dog is", " Biscuit")
    assert cs._mine_fact("My boss is Marta.") == ("My boss is", " Marta")
    assert cs._mine_fact("Our anniversary is June 12") == ("Our anniversary is", " June 12")


def test_mine_fact_rejects_questions_and_noise():
    assert cs._mine_fact("What is my dog called?") is None
    assert cs._mine_fact("hello there") is None
    assert cs._mine_fact("") is None
    # a long run-on isn't a clean fact
    assert cs._mine_fact("I went to the store today and bought milk and eggs and then drove home slowly") is None


def test_auto_write_gate_refuses_a_known_fact_from_conversation(iso):
    """The spec's headline: a surprise-gated auto-write refuses what the model already knows. Here the
    user 'states' a KNOWN fact -> the miner proposes it -> the gate SKIPS it (nothing stored)."""
    _enable()
    box = cs._slots_box()
    # mine + attempt-write a KNOWN fact -> refused
    wrote = box.auto_write([{"role": "user", "content": "The capital of France is Paris"}], "Yes, it is.")
    assert wrote is not None and wrote["written"] is False
    assert _post("/facts/list", {})["count"] == 0
    # a NONCE fact from conversation IS captured
    wrote2 = box.auto_write([{"role": "user", "content": "My cat is named Mittens"}], "Nice name!")
    assert wrote2 is not None and wrote2["written"] is True
    assert [e["cue"] for e in _post("/facts/list", {})["entries"]] == ["My cat is"]


def test_auto_write_is_a_noop_when_off(iso):
    box = cs._slots_box()
    assert box.auto_write([{"role": "user", "content": "My cat is named Mittens"}], "ok") is None


# ---- unknown route ----------------------------------------------------------------------------------

def test_unknown_facts_route_404s(iso):
    out = _post("/facts/bogus", {})
    assert "error" in out
