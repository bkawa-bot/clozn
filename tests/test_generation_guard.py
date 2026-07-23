"""tests/test_generation_guard.py -- model-free unit tests for clozn/server/generation_guard.py's PURE
pieces: opt-in spec parsing, the jlens-based disposition read, fail-closed concept resolution, the
engine-agnostic control loop (run_guarded_generation), the receipt builder, and -- critically -- the
HONESTY CAVEAT wording itself (present-tense detect-and-correct only, never lead-time/prediction/"before
it's said"/"acts on intent"; see the module's own HONESTY LAW section).

No engine, no HTTP, no GPU: run_guarded_generation is driven entirely through injected fake callables.
The production adapter (guarded_chat_completion) is covered separately in
tests/test_generation_guard_server.py via a fake substrate/engine (never a live one).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clozn.memory import mode as memory_mode                    # noqa: E402
from clozn.server import generation_guard as gg                  # noqa: E402


# ================================================================================ opt-in spec parsing

def test_parse_guard_spec_none_when_field_and_setting_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    assert gg.parse_guard_spec({}) is None
    assert gg.parse_guard_spec(None) is None


def test_parse_guard_spec_explicit_empty_dict_is_off():
    assert gg.parse_guard_spec({"clozn_guard": {}}) is None


def test_parse_guard_spec_explicit_false_is_off():
    assert gg.parse_guard_spec({"clozn_guard": False}) is None


def test_parse_guard_spec_explicit_none_is_off():
    assert gg.parse_guard_spec({"clozn_guard": None}) is None


def test_parse_guard_spec_empty_concepts_list_is_off():
    assert gg.parse_guard_spec({"clozn_guard": {"concepts": []}}) is None


def test_parse_guard_spec_valid_minimal_fills_defaults():
    spec = gg.parse_guard_spec({"clozn_guard": {"concepts": ["violence"]}})
    assert spec == {
        "concepts": ["violence"], "threshold": gg.DEFAULT_THRESHOLD,
        "counter_strength": gg.DEFAULT_COUNTER_STRENGTH, "max_fires": gg.DEFAULT_MAX_FIRES,
        "layer": gg.DEFAULT_LAYER, "chunk_tokens": gg.DEFAULT_CHUNK_TOKENS, "topk": gg.DEFAULT_TOPK,
    }


def test_parse_guard_spec_honors_explicit_overrides():
    spec = gg.parse_guard_spec({"clozn_guard": {
        "concepts": ["violence", "self-harm"], "threshold": 2.5, "counter_strength": -0.8,
        "max_fires": 5, "layer": 21, "chunk_tokens": 16, "topk": 4,
    }})
    assert spec["concepts"] == ["violence", "self-harm"]
    assert spec["threshold"] == 2.5
    assert spec["counter_strength"] == -0.8
    assert spec["max_fires"] == 5
    assert spec["layer"] == 21
    assert spec["chunk_tokens"] == 16
    assert spec["topk"] == 4


@pytest.mark.parametrize("bad, why", [
    ({"clozn_guard": "not-a-dict"}, "must be an object"),
    ({"clozn_guard": {"concepts": "violence"}}, "concepts must be a list"),
    ({"clozn_guard": {"concepts": [""]}}, "non-empty strings"),
    ({"clozn_guard": {"concepts": [123]}}, "non-empty strings"),
    ({"clozn_guard": {"concepts": ["x"], "threshold": "high"}}, "threshold must be a number"),
    ({"clozn_guard": {"concepts": ["x"], "counter_strength": True}}, "counter_strength must be a number"),
    ({"clozn_guard": {"concepts": ["x"], "max_fires": 0}}, "max_fires must be a positive integer"),
    ({"clozn_guard": {"concepts": ["x"], "max_fires": 1.5}}, "max_fires must be a positive integer"),
    ({"clozn_guard": {"concepts": ["x"], "layer": "16"}}, "layer must be an integer"),
    ({"clozn_guard": {"concepts": ["x"], "chunk_tokens": 0}}, "chunk_tokens must be a positive integer"),
    ({"clozn_guard": {"concepts": ["x"], "topk": -1}}, "topk must be a positive integer"),
])
def test_parse_guard_spec_rejects_malformed_values(bad, why):
    with pytest.raises(ValueError, match=why):
        gg.parse_guard_spec(bad)


def test_parse_guard_spec_falls_back_to_server_setting_when_field_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    memory_mode.set_setting(gg.GUARD_SETTING, {"concepts": ["violence"]})
    spec = gg.parse_guard_spec({})
    assert spec is not None and spec["concepts"] == ["violence"]


def test_parse_guard_spec_explicit_field_wins_over_server_setting(monkeypatch, tmp_path):
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    memory_mode.set_setting(gg.GUARD_SETTING, {"concepts": ["violence"]})
    # explicit false opts OUT even though the server default is on
    assert gg.parse_guard_spec({"clozn_guard": False}) is None
    # explicit spec overrides the server default's concepts
    spec = gg.parse_guard_spec({"clozn_guard": {"concepts": ["self-harm"]}})
    assert spec["concepts"] == ["self-harm"]


# ================================================================================ concept_activation (jlens read)

def test_concept_activation_present_at_last_position():
    jl = {"readouts": [
        [{"id": 1, "score": 0.1}],
        [{"id": 5, "score": 9.5}, {"id": 1, "score": -2.0}],
    ]}
    assert gg.concept_activation(jl, 5) == pytest.approx(9.5)


def test_concept_activation_absent_from_topk_is_none():
    jl = {"readouts": [[{"id": 1, "score": 9.5}]]}
    assert gg.concept_activation(jl, 999) is None


def test_concept_activation_no_readouts_is_none():
    assert gg.concept_activation({"readouts": []}, 5) is None
    assert gg.concept_activation({}, 5) is None
    assert gg.concept_activation(None, 5) is None


def test_concept_activation_malformed_score_is_none():
    jl = {"readouts": [[{"id": 5, "score": "high"}]]}
    assert gg.concept_activation(jl, 5) is None


def test_concept_activation_specific_position():
    jl = {"readouts": [[{"id": 5, "score": 1.0}], [{"id": 5, "score": 2.0}]]}
    assert gg.concept_activation(jl, 5, position=0) == pytest.approx(1.0)
    assert gg.concept_activation(jl, 5, position=1) == pytest.approx(2.0)


# ================================================================================ resolve_guard_concepts (fail-closed)

class _FakeConceptSteer:
    def __init__(self, ok_concepts=None):
        self.ok_concepts = set(ok_concepts or [])
        self.calls = []

    def compute(self, concept, layer=None):
        self.calls.append((concept, layer))
        if concept in self.ok_concepts:
            return {"ok": True, "concept": concept, "layer": layer, "token_id": hash(concept) % 1000,
                    "vector": [0.1, 0.2]}
        return {"ok": False, "blocked": "unembed_unavailable", "concept": concept,
                "note": f"no dir({concept}) available"}


def test_resolve_guard_concepts_all_succeed():
    steer = _FakeConceptSteer(ok_concepts=["violence", "self-harm"])
    built, reason = gg.resolve_guard_concepts(steer, ["violence", "self-harm"], layer=16)
    assert reason is None
    assert set(built) == {"violence", "self-harm"}
    assert steer.calls == [("violence", 16), ("self-harm", 16)]


def test_resolve_guard_concepts_fails_closed_on_first_unresolvable_concept():
    steer = _FakeConceptSteer(ok_concepts=["violence"])
    built, reason = gg.resolve_guard_concepts(steer, ["violence", "unicorn-glitter"], layer=16)
    assert built is None
    assert "unicorn-glitter" in reason
    assert "unavailable" in reason


# ================================================================================ run_guarded_generation (the loop)

def _fake_generate(pieces_by_call):
    """A generate_chunk fake: returns pieces_by_call[i] on the i-th call, regardless of args, and
    records every call for inspection."""
    calls = []

    def generate_chunk(prompt_so_far, max_new, *, counter=None):
        calls.append({"prompt_len": len(prompt_so_far), "max_new": max_new, "counter": counter})
        return pieces_by_call[len(calls) - 1]

    generate_chunk.calls = calls
    return generate_chunk


def test_guard_loop_no_fire_never_corrects():
    gen = _fake_generate(["clean chunk one ", "clean chunk two"])

    def read_disposition(text):
        return {"violence": None}   # never present in the topk -- never fires

    out = gg.run_guarded_generation(
        generate_chunk=gen, read_disposition=read_disposition, build_counter=lambda c: (_ for _ in ()).throw(
            AssertionError("build_counter must never be called when nothing fires")),
        base_text="PROMPT: ", max_tokens=48, chunk_tokens=24, concepts=["violence"],
        threshold=1.0, counter_strength=-0.5, max_fires=3,
    )
    assert out["text"] == "clean chunk one clean chunk two"
    assert out["fires"] == []
    assert out["n_fires"] == 0
    assert out["cap_reached"] is False
    assert out["n_chunks"] == 2
    assert len(gen.calls) == 2   # no correction round trips


def test_guard_loop_fires_and_replaces_the_flagged_chunk():
    # call 1: plain chunk 1 (flags), call 2: corrected chunk 1, call 3: plain chunk 2 (clean)
    gen = _fake_generate(["BAD content here", "safe corrected content", " clean chunk two"])
    activations_by_text = {}

    def read_disposition(text):
        if "BAD content" in text:
            return {"violence": 9.0}
        return {"violence": None}

    built_counters = []

    def build_counter(concept):
        built_counters.append(concept)
        return {"concept": concept, "vector": [0.1], "coef": -5.0}

    out = gg.run_guarded_generation(
        generate_chunk=gen, read_disposition=read_disposition, build_counter=build_counter,
        base_text="PROMPT: ", max_tokens=48, chunk_tokens=24, concepts=["violence"],
        threshold=1.0, counter_strength=-0.5, max_fires=3,
    )
    assert out["text"] == "safe corrected content clean chunk two"
    assert out["n_fires"] == 1
    assert out["cap_reached"] is False
    assert built_counters == ["violence"]
    fire = out["fires"][0]
    assert fire["concept"] == "violence"
    assert fire["pre_activation"] == pytest.approx(9.0)
    assert fire["post_activation"] is None   # the corrected chunk is clean -- not in the topk anymore
    assert fire["counter_strength"] == -0.5
    assert fire["chunk_index"] == 0
    assert fire["token_position"] == 0


def test_guard_loop_records_post_activation_when_correction_does_not_fully_clear_it():
    gen = _fake_generate(["BAD content", "still a bit BAD"])

    def read_disposition(text):
        if "BAD" in text:
            return {"violence": 9.0 if "still a bit" not in text else 3.0}
        return {"violence": None}

    out = gg.run_guarded_generation(
        generate_chunk=gen, read_disposition=read_disposition, build_counter=lambda c: {"vector": [0.1], "coef": -1.0},
        base_text="", max_tokens=24, chunk_tokens=24, concepts=["violence"],
        threshold=1.0, counter_strength=-0.5, max_fires=3,
    )
    assert out["n_fires"] == 1
    assert out["fires"][0]["pre_activation"] == pytest.approx(9.0)
    assert out["fires"][0]["post_activation"] == pytest.approx(3.0)


def test_guard_loop_cap_reached_stops_correcting_and_says_so():
    # 3 chunks of 24 tokens = 72 total. max_fires=1. Chunks 1 and 2 both trip; chunk 1 gets corrected
    # (uses the only fire), chunk 2 trips again but the cap is already spent -> left uncorrected, and the
    # rest of generation (chunk 3) is produced in one plain, unwatched call.
    gen = _fake_generate(["BAD one", "corrected one", "BAD two", "plain rest"])

    def read_disposition(text):
        if "BAD" in text:
            return {"violence": 9.0}
        return {"violence": None}

    out = gg.run_guarded_generation(
        generate_chunk=gen, read_disposition=read_disposition,
        build_counter=lambda c: {"vector": [0.1], "coef": -1.0},
        base_text="", max_tokens=72, chunk_tokens=24, concepts=["violence"],
        threshold=1.0, counter_strength=-0.5, max_fires=1,
    )
    assert out["n_fires"] == 1
    assert out["cap_reached"] is True
    assert "BAD two" in out["text"]        # left uncorrected once the cap was spent -- honest, not hidden
    assert "plain rest" in out["text"]
    assert len(gen.calls) == 4              # BAD one, corrected one, BAD two, plain rest (no 2nd correction)


def test_guard_loop_never_exceeds_max_fires_even_with_many_triggering_chunks():
    pieces = ["BAD"] * 10
    gen = _fake_generate(pieces)

    def read_disposition(text):
        return {"violence": 9.0} if text.rstrip().endswith("BAD") else {"violence": None}

    def build_counter(concept):
        return {"vector": [0.1], "coef": -1.0}

    out = gg.run_guarded_generation(
        generate_chunk=gen, read_disposition=read_disposition, build_counter=build_counter,
        base_text="", max_tokens=24 * 10, chunk_tokens=24, concepts=["violence"],
        threshold=1.0, counter_strength=-0.5, max_fires=2,
    )
    assert out["n_fires"] == 2
    assert out["cap_reached"] is True


# ================================================================================ build_receipt

def test_build_receipt_shape_no_fires():
    spec = gg.parse_guard_spec({"clozn_guard": {"concepts": ["violence"]}})
    result = {"text": "clean text", "fires": [], "n_fires": 0, "cap_reached": False, "n_chunks": 1}
    receipt = gg.build_receipt(result, spec)
    assert receipt["concepts"] == ["violence"]
    assert receipt["n_fires"] == 0
    assert receipt["fires"] == []
    assert receipt["cap_reached"] is False
    assert "cap_note" not in receipt
    assert receipt["caveat"] == gg.GUARD_CAVEAT


def test_build_receipt_includes_cap_note_when_capped():
    spec = gg.parse_guard_spec({"clozn_guard": {"concepts": ["violence"], "max_fires": 1}})
    result = {"text": "x", "fires": [{"chunk_index": 0, "token_position": 0, "concept": "violence",
                                      "pre_activation": 9.0, "post_activation": 9.0,
                                      "counter_strength": -0.5}],
             "n_fires": 1, "cap_reached": True, "n_chunks": 3}
    receipt = gg.build_receipt(result, spec)
    assert receipt["cap_reached"] is True
    assert receipt["cap_note"] == gg.GUARD_CAP_NOTE


# ================================================================================ THE HONESTY CAVEAT ITSELF

_FORBIDDEN_OVERCLAIM_PHRASES = [
    "before it's said", "before it is said", "reads intent early", "acts on intent",
    "predicts the", "predict the", "prediction of intent", "intent-before-speech capability",
    "foreknowledge of", "sees it coming",
]

_REQUIRED_PRESENT_TENSE_PHRASE = "detects and corrects during generation"


def test_caveat_contains_the_required_present_tense_phrase_verbatim():
    assert _REQUIRED_PRESENT_TENSE_PHRASE in gg.GUARD_CAVEAT


def test_caveat_contains_the_a1_1_numbers_honestly():
    assert "100%" in gg.GUARD_CAVEAT
    assert "5%" in gg.GUARD_CAVEAT
    assert "10/10" in gg.GUARD_CAVEAT
    assert "1/20" in gg.GUARD_CAVEAT
    assert "layer 16" in gg.GUARD_CAVEAT
    assert "-0.5" in gg.GUARD_CAVEAT
    # the lead-time finding itself -- reported honestly (it FAILED), not omitted
    assert "median_lead_time_tokens = 0" in gg.GUARD_CAVEAT
    assert "4/10" in gg.GUARD_CAVEAT


def test_caveat_never_claims_lead_time_or_prediction_as_a_capability():
    lowered = gg.GUARD_CAVEAT.lower()
    for phrase in _FORBIDDEN_OVERCLAIM_PHRASES:
        assert phrase not in lowered, f"caveat must never claim: {phrase!r}"


def test_cap_note_also_carries_no_overclaim():
    lowered = gg.GUARD_CAP_NOTE.lower()
    for phrase in _FORBIDDEN_OVERCLAIM_PHRASES:
        assert phrase not in lowered, f"cap note must never claim: {phrase!r}"


def test_module_docstring_states_the_present_tense_framing():
    """The module's own documentation must state the present-tense framing. (Unlike GUARD_CAVEAT/
    GUARD_CAP_NOTE -- the actual API-facing strings, scanned above for the forbidden phrases themselves --
    the docstring's HONESTY LAW section legitimately QUOTES those same forbidden phrases as instructive
    examples of what never to claim, so it is deliberately not scanned for their absence here.)"""
    doc = " ".join((gg.__doc__ or "").split()).lower()   # collapse line-wrap whitespace before matching
    assert "present-tense" in doc or "present tense" in doc
    assert "detects and corrects during generation" in doc


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
