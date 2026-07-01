"""test_dial_suggestion -- route a STYLE preference to the tone DIAL that delivers it.

Live finding: the trained memory soft-prefix carries TOPICAL prefs ("into baking") well but STYLE ones
weakly (a "prefers concise" card did not shorten replies); the tone DIALS steer style directly. So when a
memory being added/proposed is really a style preference that maps to a dial, the studio DETECTS it and
suggests the matching dial. Backend contract under test:

  * steering.suggest_dial_for_preference(text) -- a PURE, deterministic helper: maps a spread of style
    phrases to the right axis + sign, respects each axis's `max` cap, and returns None for topical/empty
    text (so only style prefs get routed);
  * /memory/add (Substrate._memory) and /runs/<id>/propose-memory (the real do_POST handler) fold a
    `dial_suggestion` field into their response -- {axis, value, pole_label} on a style match, null
    otherwise -- while STILL creating the pending card exactly as before.

No model, no GPU. steering imports torch (already a dep, as test_steering_headroom does), but the helper
itself makes no model call; the server wiring is exercised against FakeMem / FakeSub stubs.
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

import clozn_server as cs      # noqa: E402
import memory_cards            # noqa: E402
import runlog                  # noqa: E402
import steering                # noqa: E402

suggest = steering.suggest_dial_for_preference


# ================================================================================================
# 1. the pure helper: phrase -> axis + sign, capped, None for topical/empty
# ================================================================================================

@pytest.mark.parametrize("text, axis", [
    ("prefers concise answers", "concise"),
    ("keep it brief", "concise"),
    ("wants short replies", "concise"),
    ("please be terse and to the point", "concise"),
    ("no fluff", "concise"),
    ("respond formally", "formal"),
    ("keep it professional", "formal"),
    ("be casual and relaxed", "formal"),
    ("wants an informal tone", "formal"),
    ("be warm and kind", "warm"),
    ("prefers a caring, encouraging tone", "warm"),
    ("keep it cold and clinical", "warm"),
    ("wants technical, precise detail", "technical"),
    ("explain in simple terms", "technical"),
    ("plain english please", "technical"),
    ("no jargon", "technical"),
    ("be playful and witty", "playful"),
    ("wants a serious, no-nonsense tone", "playful"),
    ("be blunt and candid", "candid"),
    ("give me candid pushback", "candid"),
    ("be confident and decisive", "confident"),
    ("prefers concrete examples", "concrete"),
    ("keep it high-level and abstract", "concrete"),
])
def test_helper_routes_phrase_to_axis(text, axis):
    s = suggest(text)
    assert s is not None, f"expected a dial for {text!r}"
    assert s["axis"] == axis


@pytest.mark.parametrize("text, sign_positive", [
    # concise axis: brief/short/terse -> +, verbose/detailed -> -
    ("keep it concise", True), ("be brief", True), ("wants short answers", True),
    ("wants verbose, detailed answers", False), ("elaborate and be thorough", False),
    # formal axis: formal -> +, casual -> -
    ("respond formally", True), ("be casual", False), ("informal and relaxed", False),
    # warm axis: warm -> +, cold/detached -> -
    ("be warm", True), ("keep it cold and detached", False),
    # technical axis: technical -> +, simple/plain -> -
    ("be technical and precise", True), ("keep it simple", False), ("use plain language", False),
])
def test_helper_sign_matches_pole(text, sign_positive):
    s = suggest(text)
    assert s is not None, f"expected a dial for {text!r}"
    if sign_positive:
        assert s["value"] > 0, f"{text!r} should map to the + pole"
    else:
        assert s["value"] < 0, f"{text!r} should map to the - pole"


def test_helper_pole_label_flips_with_sign():
    # concise +  -> the first pole label; verbose -  -> the second
    assert suggest("be concise")["pole_label"] == steering.AXES["concise"]["poles"][0]
    assert suggest("be verbose")["pole_label"] == steering.AXES["concise"]["poles"][1]
    # warm + / cold -
    assert suggest("be warm")["pole_label"] == steering.AXES["warm"]["poles"][0]
    assert suggest("be cold")["pole_label"] == steering.AXES["warm"]["poles"][1]


def test_helper_value_respects_axis_max_cap():
    # uncapped axes (no "max") default-clamp to 1.5 in set(), so 0.6 passes through unchanged
    assert abs(suggest("be concise")["value"]) == pytest.approx(0.6)
    assert abs(suggest("be warm")["value"]) == pytest.approx(0.6)
    # candid is capped at 0.45 -> 0.6 must be clamped down to the axis max
    cand = suggest("be blunt and candid")
    assert cand["axis"] == "candid"
    assert abs(cand["value"]) == pytest.approx(min(0.6, steering.AXES["candid"]["max"]))
    assert abs(cand["value"]) <= steering.AXES["candid"]["max"] + 1e-9
    # concrete is capped at 0.5
    conc = suggest("prefers concrete, specific examples")
    assert conc["axis"] == "concrete"
    assert abs(conc["value"]) == pytest.approx(min(0.6, steering.AXES["concrete"]["max"]))


def test_helper_value_always_within_its_axis_cap():
    """Every routable phrase yields a magnitude <= that axis's set() cap (max, or 1.5 default)."""
    for phrase, axis, _sign in steering._DIAL_LEXICON:
        s = suggest(f"please be {phrase}")
        assert s is not None
        cap = steering.AXES.get(s["axis"], {}).get("max", 1.5)
        assert abs(s["value"]) <= cap + 1e-9


@pytest.mark.parametrize("text", [
    "is interested in baking",
    "loves baking sourdough",
    "has a dog named Rex",
    "works as an accountant",
    "lives in Seattle",
    "",
    "   ",
])
def test_helper_returns_none_for_topical_or_empty(text):
    assert suggest(text) is None


def test_helper_none_for_non_string():
    assert suggest(None) is None


def test_helper_word_boundary_no_false_fire():
    # "brief" must not fire inside "briefing"; "warm" not inside "warmup"; a topical-only sentence -> None
    assert suggest("wants a briefing on the warmup routine") is None


# ---- inflected / comparative forms map like their base word ----------------------------------------

@pytest.mark.parametrize("text, axis, positive", [
    ("warmer and friendlier tone", "warm", True),          # the reported miss
    ("wants a warmer tone", "warm", True),
    ("be friendlier", "warm", True),
    ("the warmest possible tone", "warm", True),
    ("be kinder", "warm", True),
    ("keep it colder and more detached", "warm", False),
    ("shorter please", "concise", True),
    ("make it briefer", "concise", True),
    ("wants longer, lengthier answers", "concise", False),
    ("keep it simpler", "technical", False),
    ("use plainer wording", "technical", False),
    ("be blunter", "candid", True),
    ("a drier, more sober tone", "playful", False),
    ("be wittier", "playful", True),
])
def test_helper_inflected_forms_route_correctly(text, axis, positive):
    s = suggest(text)
    assert s is not None, f"expected a dial for the inflected form in {text!r}"
    assert s["axis"] == axis
    assert (s["value"] > 0) is positive, f"{text!r} landed on the wrong pole"


# ---- negation / reducer cue flips the sign to the opposite pole ------------------------------------

@pytest.mark.parametrize("text, axis, expect_positive, pole_label", [
    # "less technical" -> technical NEGATIVE (the plain pole)
    ("less technical, plain english", "technical", False, "simple"),
    ("make it less technical", "technical", False, "simple"),
    # "not too formal" -> formal NEGATIVE (casual)
    ("not too formal", "formal", False, "casual"),
    ("please be not so formal", "formal", False, "casual"),
    # "less verbose" -> concise POSITIVE (verbose flipped back to concise)
    ("less verbose", "concise", True, "concise"),
    # a few more axes
    ("too warm, tone it down", "warm", False, "detached"),
    ("overly playful, dial it back", "playful", False, "serious"),
    ("avoid being too casual", "formal", True, "formal"),    # casual flipped -> formal
])
def test_helper_negation_flips_sign(text, axis, expect_positive, pole_label):
    s = suggest(text)
    assert s is not None, f"expected a dial for {text!r}"
    assert s["axis"] == axis
    assert (s["value"] > 0) is expect_positive, f"{text!r} did not flip to the expected pole"
    assert s["pole_label"] == pole_label


def test_helper_negation_handles_contraction():
    # "n't" as a reducer immediately before a bare keyword flips it
    s = suggest("don't be technical")
    assert s is not None and s["axis"] == "technical"
    assert s["value"] < 0                                   # -> the plain pole


# ---- intended multi-word negative phrases are NOT re-flipped by the negation logic -----------------

@pytest.mark.parametrize("text, axis, pole_label", [
    ("no fluff please", "concise", "concise"),             # a concise-POSITIVE phrase, must stay +
    ("no jargon", "technical", "simple"),                  # a technical-NEGATIVE phrase, must stay -
    ("no-nonsense tone", "playful", "serious"),            # hyphenated serious phrase, must stay -
    ("no hedging", "confident", "confident"),              # confident-POSITIVE phrase, must stay +
    ("no metaphors, keep it literal", "poetic", "plain"),  # poetic-NEGATIVE phrase, must stay -
    ("get to the point", "concise", "concise"),            # multi-word concise-POSITIVE, must stay +
])
def test_helper_intended_negative_phrases_not_flipped(text, axis, pole_label):
    s = suggest(text)
    assert s is not None, f"expected a dial for {text!r}"
    assert s["axis"] == axis
    assert s["pole_label"] == pole_label


def test_helper_is_deterministic():
    a = suggest("prefers concise, direct answers")
    b = suggest("prefers concise, direct answers")
    assert a == b


def test_helper_shape_is_exactly_three_keys():
    s = suggest("be concise")
    assert set(s.keys()) == {"axis", "value", "pole_label"}
    assert isinstance(s["axis"], str) and isinstance(s["pole_label"], str)
    assert isinstance(s["value"], float)


# ================================================================================================
# 2. /memory/add folds dial_suggestion into the response (FakeMem, real dispatch)
# ================================================================================================

class FakeMem:
    """Minimal memory stub (as in test_memory_wiring): the card wiring only needs .rules/.prefix here."""

    def __init__(self, rules=None):
        self.rules = list(rules or [])
        self.prefix = "PREFIX" if self.rules else None
        self.memory_strength = 1.0

    def consolidate(self, rules):
        self.rules = list(rules)
        self.prefix = "PREFIX"
        return {"ok": True}

    def reset(self):
        self.prefix = None
        self.rules = []
        return {"ok": True}


def _substrate(mem):
    sub = object.__new__(cs.Substrate)
    sub._mem = mem
    return sub


@pytest.fixture()
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUBNAME", "qwen")
    return tmp_path


def test_add_style_pref_carries_dial_suggestion(iso):
    sub = _substrate(FakeMem([]))
    out = sub._memory("/memory/add", {"text": "prefers concise, direct answers"})
    # the card is still created + pending (unchanged behavior)
    assert out["status"] == "pending"
    assert out["text"] == "prefers concise, direct answers"
    assert out["id"].startswith("mem_")
    assert memory_cards.get(out["id"])["status"] == "pending"
    # + the routing suggestion
    assert out["dial_suggestion"] is not None
    assert out["dial_suggestion"]["axis"] == "concise"
    assert out["dial_suggestion"]["value"] > 0
    assert out["dial_suggestion"]["pole_label"] == "concise"


def test_add_topical_pref_has_null_dial_suggestion(iso):
    sub = _substrate(FakeMem([]))
    out = sub._memory("/memory/add", {"text": "is interested in baking"})
    assert out["status"] == "pending"                      # card still created
    assert out["dial_suggestion"] is None                  # nothing to route


def test_add_empty_text_still_rejected_no_suggestion(iso):
    sub = _substrate(FakeMem([]))
    out = sub._memory("/memory/add", {"text": "   "})
    assert out.get("ok") is False
    assert "dial_suggestion" not in out                    # never got as far as creating a card


# ================================================================================================
# 3. /runs/<id>/propose-memory folds dial_suggestion in (real do_POST handler, FakeSub)
# ================================================================================================

class FakeSteer:
    def __init__(self, strength=None):
        self.strength = dict(strength or {})

    def save_state(self, path):
        pass


class FakeMemory:
    def __init__(self, result):
        self._result = result

    def propose_memory(self, messages, response=None):
        return self._result


class FakeSub:
    def __init__(self, result):
        self.steer = FakeSteer()
        self.memory = FakeMemory(result)


class _FakeRequest:
    def __init__(self, path, body_obj):
        self.path = path
        raw = json.dumps(body_obj).encode("utf-8")
        self.rfile = io.BytesIO(raw)
        self.wfile = io.BytesIO()
        self.headers = {"Content-Length": str(len(raw))}


def _post(path, body_obj):
    H = cs.make_handler()
    h = object.__new__(H)
    req = _FakeRequest(path, body_obj)
    h.path, h.rfile, h.wfile, h.headers = req.path, req.rfile, req.wfile, req.headers
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    raw = req.wfile.getvalue()
    _, _, payload = raw.partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


def _make_run():
    return runlog.record(source="studio_chat",
                         messages=[{"role": "user", "content": "give me the short version"}],
                         response="Here's the concise answer ...", model="clozn-qwen", substrate="qwen")


def test_propose_style_pref_carries_dial_suggestion(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSub(result="Prefers concise, technical answers"))
    rid = _make_run()
    out = _post(f"/runs/{rid}/propose-memory", {})
    assert out["proposed"] is True
    assert out["card"]["status"] == "pending"              # card still created + returned with its id
    assert out["card"]["source_run_id"] == rid
    assert out["dial_suggestion"] is not None
    # "concise" appears first in the text -> routes to the concise axis, + pole
    assert out["dial_suggestion"]["axis"] == "concise"
    assert out["dial_suggestion"]["value"] > 0


def test_propose_topical_pref_has_null_dial_suggestion(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSub(result="Is interested in baking"))
    rid = _make_run()
    out = _post(f"/runs/{rid}/propose-memory", {})
    assert out["proposed"] is True
    assert out["card"]["text"] == "Is interested in baking"
    assert out["dial_suggestion"] is None


def test_propose_no_preference_has_no_dial_suggestion(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSub(result=None))   # nothing durable found -> no card, no suggestion
    rid = _make_run()
    out = _post(f"/runs/{rid}/propose-memory", {})
    assert out["proposed"] is False
    assert "dial_suggestion" not in out
    assert memory_cards.list_cards() == []
