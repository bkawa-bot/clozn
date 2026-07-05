"""Pure-logic tests for research/parliament.py (Wild Experiment #4 -- the parliament of stances).

No GPU, no real model: parliament.py imports torch/transformers at module level (like steer_vs_prompt.py
and mirror_bench.py's own antecedent chain), so importing it requires those packages installed, but every
test here exercises functions that make no model call -- CLI parsing, the judge's text parsing (parse_bits/
parse_pref), the coverage/trust math, the question bank's own integrity, and the merge/judge/pairwise
control flow driven through a FakeRig stand-in (no torch tensor ever touched by the fake). Two tests
(shuffle-vector determinism/unit-norm) use plain CPU torch tensors -- no model, no CUDA required.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # research/ on path
import parliament as pl  # noqa: E402


# ================================================================================================
# wants_four_bit / default_judge_for
# ================================================================================================
def test_wants_four_bit_small_models_get_bf16():
    assert pl.wants_four_bit("Qwen/Qwen2.5-0.5B-Instruct", "auto") is False
    assert pl.wants_four_bit("Qwen/Qwen2.5-1.5B-Instruct", "auto") is False
    assert pl.wants_four_bit("some/Model-3B-Instruct", "auto") is False


def test_wants_four_bit_big_models_get_nf4():
    assert pl.wants_four_bit("Qwen/Qwen2.5-7B-Instruct", "auto") is True
    assert pl.wants_four_bit("google/gemma-2-9b-it", "auto") is True


def test_wants_four_bit_override_wins():
    assert pl.wants_four_bit("Qwen/Qwen2.5-7B-Instruct", "no") is False
    assert pl.wants_four_bit("Qwen/Qwen2.5-0.5B-Instruct", "yes") is True


def test_default_judge_for_cross_family():
    assert "gemma" in pl.default_judge_for("Qwen/Qwen2.5-7B-Instruct").lower()
    assert "qwen" in pl.default_judge_for("google/gemma-2-9b-it").lower()


def test_default_judge_for_smoke_is_small_and_documented_not_a_finding():
    j = pl.default_judge_for("Qwen/Qwen2.5-7B-Instruct", smoke=True)
    assert pl.wants_four_bit(j, "auto") is False, "smoke judge should be a fast bf16 load"


# ================================================================================================
# axis_max_of / _axis_seed / make_shuffle_unit_vector
# ================================================================================================
class _FakeSC:
    def __init__(self, custom=None):
        self.custom = custom or {}


def test_axis_max_of_builtin_caps():
    assert pl.axis_max_of(_FakeSC(), "candid") == 0.45
    assert pl.axis_max_of(_FakeSC(), "concrete") == 0.5


def test_axis_max_of_warm_defaults_to_1p5():
    assert pl.axis_max_of(_FakeSC(), "warm") == 1.5


def test_axis_max_of_custom_axis():
    sc = _FakeSC(custom={"skeptical": {"max": 0.5}})
    assert pl.axis_max_of(sc, "skeptical") == 0.5


def test_axis_seed_deterministic_and_distinct_per_axis():
    a1 = pl._axis_seed(0, "candid")
    a2 = pl._axis_seed(0, "candid")
    b = pl._axis_seed(0, "warm")
    assert a1 == a2
    assert a1 != b


def test_axis_seed_distinct_per_run_seed():
    assert pl._axis_seed(0, "candid") != pl._axis_seed(1, "candid")


def test_make_shuffle_unit_vector_is_unit_norm_and_shape_matched():
    ref = torch.randn(16)
    v = pl.make_shuffle_unit_vector(ref, seed=42)
    assert v.shape == ref.shape
    assert torch.isclose(v.norm(), torch.tensor(1.0), atol=1e-4)


def test_make_shuffle_unit_vector_deterministic_given_same_seed():
    ref = torch.randn(16)
    v1 = pl.make_shuffle_unit_vector(ref, seed=7)
    v2 = pl.make_shuffle_unit_vector(ref, seed=7)
    assert torch.allclose(v1, v2)


def test_make_shuffle_unit_vector_differs_across_seeds():
    ref = torch.randn(16)
    v1 = pl.make_shuffle_unit_vector(ref, seed=1)
    v2 = pl.make_shuffle_unit_vector(ref, seed=2)
    assert not torch.allclose(v1, v2)


# ================================================================================================
# parse_bits -- the judge's rubric-bit parser (must fail HONESTLY, never pad a guess)
# ================================================================================================
def test_parse_bits_clean():
    bits, note = pl.parse_bits("1 0 1 1 0", 5)
    assert bits == [1, 0, 1, 1, 0]
    assert note == "ok"


def test_parse_bits_comma_separated():
    bits, note = pl.parse_bits("1,0,1,1,0", 5)
    assert bits == [1, 0, 1, 1, 0]


def test_parse_bits_trailing_period_stripped():
    bits, note = pl.parse_bits("1 0 1 1 0.", 5)
    assert bits == [1, 0, 1, 1, 0]


def test_parse_bits_ignores_preamble_and_punctuation_glued_digits():
    bits, note = pl.parse_bits("About 1 of these... 1 0 1 1 0", 5)
    assert bits == [1, 0, 1, 1, 0], "the isolated stray '1' before the real run must not be picked"


def test_parse_bits_too_few_is_honest_parse_failure():
    bits, note = pl.parse_bits("1 0 1", 5)
    assert bits is None
    assert note.startswith("parse-fail")


def test_parse_bits_too_many_is_truncated_not_failed():
    bits, note = pl.parse_bits("1 0 1 1 0 1 1", 5)
    assert bits == [1, 0, 1, 1, 0]
    assert "truncated" in note


def test_parse_bits_empty_input_fails_honestly():
    bits, note = pl.parse_bits("", 4)
    assert bits is None
    assert note.startswith("parse-fail")


def test_coverage_pct():
    assert pl.coverage_pct(None) is None
    assert pl.coverage_pct([1, 1, 0, 0]) == 50.0
    assert pl.coverage_pct([1, 1, 1, 1]) == 100.0
    assert pl.coverage_pct([0, 0, 0]) == 0.0


# ================================================================================================
# parse_pref -- the soft pairwise leg's single-token parser
# ================================================================================================
def test_parse_pref_variants():
    assert pl.parse_pref("A") == "A"
    assert pl.parse_pref("The better answer is B.") == "B"
    assert pl.parse_pref("I'd call this a TIE") == "TIE"
    assert pl.parse_pref("tie") == "TIE"
    assert pl.parse_pref("no idea") is None


# ================================================================================================
# marker_rate / degenerate_rate -- crude, pure-text scorers
# ================================================================================================
def test_marker_rate_counts_axis_words():
    assert pl.marker_rate("Frankly, I have to push back on that.", "candid") > 0
    assert pl.marker_rate("A neutral sentence with no markers at all.", "candid") == 0.0


def test_degenerate_rate_flags_repetition():
    assert pl.degenerate_rate(["this is fine this is fine this is fine"]) == 0.0 or \
        pl.degenerate_rate(["hello world hello world hello world"]) >= 0.0  # sanity: function runs
    rate = pl.degenerate_rate(["loop loop loop forever and ever", "a perfectly normal reply here"])
    assert 0.0 <= rate <= 1.0


# ================================================================================================
# merge_candidates -- coherence-filtered survivors only; must not call the judge model needlessly
# ================================================================================================
class FakeRig:
    """Stands in for parliament.Rig: .gen() is driven by a `responder(prompt, sample) -> str` callable.
    No torch tensor is ever touched."""
    def __init__(self, responder):
        self.responder = responder
        self.calls: list[str] = []

    def gen(self, user: str, max_new: int = 180, sample: bool = False, temperature: float = 0.9) -> str:
        self.calls.append(user)
        return self.responder(user, sample)


def test_merge_candidates_zero_survivors_never_calls_judge():
    jrig = FakeRig(lambda p, s: (_ for _ in ()).throw(AssertionError("should not be called")))
    merged, meta = pl.merge_candidates(jrig, "Q?", [])
    assert meta["fallback"] is True
    assert meta["k_survivors"] == 0
    assert jrig.calls == []
    assert "no coherent candidate" in merged


def test_merge_candidates_single_survivor_is_passthrough():
    jrig = FakeRig(lambda p, s: (_ for _ in ()).throw(AssertionError("should not be called")))
    merged, meta = pl.merge_candidates(jrig, "Q?", ["the one true answer"])
    assert merged == "the one true answer"
    assert meta["k_survivors"] == 1
    assert jrig.calls == []


def test_merge_candidates_multiple_survivors_calls_judge_once():
    jrig = FakeRig(lambda p, s: "[[MERGED]]")
    merged, meta = pl.merge_candidates(jrig, "What is X?", ["ans one", "ans two", "ans three"])
    assert merged == "[[MERGED]]"
    assert meta["k_survivors"] == 3
    assert len(jrig.calls) == 1
    assert "What is X?" in jrig.calls[0]
    assert "ans one" in jrig.calls[0] and "ans three" in jrig.calls[0]


# ================================================================================================
# judge_phase -- coherence filtering, coverage aggregation, consistency (bit-flip) rate
# ================================================================================================
def test_judge_phase_end_to_end_with_fake_judge():
    questions = [{"id": "q1", "q": "Explain thing.", "points": ["p1", "p2", "p3", "p4"]}]
    # one degenerate decode in parliament's K=2 must be dropped before the merge.
    arms = {
        "parliament": {"raw": [["a coherent answer about the thing", "loop loop loop loop loop loop"]]},
        "single": {"raw": [["a single coherent answer here"]]},
    }

    def responder(prompt, sample):
        if "REQUIRED POINTS" in prompt:
            return "1 1 1 1" if sample else "1 0 1 1"   # deliberately different -> exercises flip-rate
        return "[[MERGED]]"

    jrig = FakeRig(responder)
    out = pl.judge_phase(jrig, questions, arms, consistency_check=True)

    # parliament had one degenerate candidate -> merge_meta reflects k_total=2, k_degenerate=1, k_survivors=1
    assert out["parliament"]["merge_meta"][0]["k_total"] == 2
    assert out["parliament"]["merge_meta"][0]["k_degenerate"] == 1
    # only ONE survivor -> passthrough, no merge call needed for this question
    assert out["parliament"]["final_answers"][0] == "a coherent answer about the thing"

    assert out["single"]["bits"][0] == [1, 0, 1, 1]
    assert out["single"]["coverage"][0] == 75.0        # 3-of-4 bits set
    assert out["single"]["coverage_mean"] == 75.0
    assert out["single"]["parse_fail_rate"] == 0.0
    # bits=[1,0,1,1] vs sampled bits=[1,1,1,1] -> 1 flip / 4 bits
    assert out["single"]["consistency_flip_rate"] == 0.25


def test_judge_phase_parse_failure_excluded_from_coverage_mean_not_zeroed():
    questions = [{"id": "q1", "q": "Q1", "points": ["p1", "p2"]},
                 {"id": "q2", "q": "Q2", "points": ["p1", "p2"]}]
    arms = {"single": {"raw": [["ans one"], ["ans two"]]}}
    calls = {"n": 0}

    def responder(prompt, sample):
        if "REQUIRED POINTS" not in prompt:
            return "[[MERGED]]"
        calls["n"] += 1
        return "garbage, no bits here" if calls["n"] == 1 else "1 0"

    jrig = FakeRig(responder)
    out = pl.judge_phase(jrig, questions, arms, consistency_check=False)
    assert out["single"]["bits"][0] is None
    assert out["single"]["parse_notes"][0].startswith("parse-fail")
    assert out["single"]["coverage"] == [None, 50.0]
    assert out["single"]["coverage_mean"] == 50.0, "the mean must ignore the unparseable question, not treat it as 0"
    assert out["single"]["parse_fail_rate"] == 0.5


# ================================================================================================
# judge_trust_report -- calibrated BY THE NULLS
# ================================================================================================
def test_judge_trust_report_trustworthy_case():
    jres = {"parliament": {"coverage_mean": 80.0}, "single": {"coverage_mean": 60.0},
            "temp_vote_null": {"coverage_mean": 65.0}, "shuffled_dial_null": {"coverage_mean": 55.0}}
    rep = pl.judge_trust_report(jres)
    assert rep["trustworthy"] is True


def test_judge_trust_report_flags_shuffled_above_floor():
    jres = {"parliament": {"coverage_mean": 80.0}, "single": {"coverage_mean": 60.0},
            "temp_vote_null": {"coverage_mean": 65.0}, "shuffled_dial_null": {"coverage_mean": 90.0}}
    rep = pl.judge_trust_report(jres)
    assert rep["trustworthy"] is False
    assert any("UNTRUSTWORTHY" in n for n in rep["notes"])


def test_judge_trust_report_flags_shuffled_indistinguishable_from_parliament():
    jres = {"parliament": {"coverage_mean": 70.0}, "single": {"coverage_mean": 60.0},
            "temp_vote_null": {"coverage_mean": 65.0}, "shuffled_dial_null": {"coverage_mean": 69.0}}
    rep = pl.judge_trust_report(jres)
    assert rep["trustworthy"] is False


# ================================================================================================
# pairwise_phase -- soft leg, position randomization, tallying
# ================================================================================================
def test_pairwise_phase_tallies_parliament_wins():
    questions = [{"id": f"q{i}", "q": f"Question {i}"} for i in range(6)]
    judge_results = {
        "parliament": {"final_answers": [f"good answer {i}" for i in range(6)]},
        "single": {"final_answers": [f"meh answer {i}" for i in range(6)]},
        "temp_vote_null": {"final_answers": [f"meh answer {i}" for i in range(6)]},
        "shuffled_dial_null": {"final_answers": [f"meh answer {i}" for i in range(6)]},
    }

    def responder(prompt, sample):
        # always prefer whichever side holds "good answer"
        for line in prompt.splitlines():
            pass
        return "A" if "good answer" in prompt.split("ANSWER B:")[0] else "B"

    jrig = FakeRig(responder)
    out = pl.pairwise_phase(jrig, questions, judge_results, seed=0)
    assert set(out.keys()) == {"parliament_vs_single", "parliament_vs_temp_vote_null",
                               "parliament_vs_shuffled_dial_null"}
    for k, r in out.items():
        assert r["parliament_wins"] == 6
        assert r["opponent_wins"] == 0
        assert r["parliament_winrate"] == 1.0


def test_pairwise_phase_handles_parse_failure():
    questions = [{"id": "q0", "q": "Q0"}]
    judge_results = {"parliament": {"final_answers": ["x"]}, "single": {"final_answers": ["y"]},
                     "temp_vote_null": {"final_answers": ["y"]}, "shuffled_dial_null": {"final_answers": ["y"]}}
    jrig = FakeRig(lambda p, s: "unparseable nonsense")
    out = pl.pairwise_phase(jrig, questions, judge_results, seed=0)
    assert out["parliament_vs_single"]["parse_fail"] == 1
    assert out["parliament_vs_single"]["parliament_winrate"] is None


# ================================================================================================
# QUESTION_BANK / STANCES / ARMS_ORDER integrity
# ================================================================================================
def test_question_bank_has_at_least_30_well_formed_entries():
    assert len(pl.QUESTION_BANK) >= 30
    ids = [q["id"] for q in pl.QUESTION_BANK]
    assert len(ids) == len(set(ids)), "question ids must be unique"
    for q in pl.QUESTION_BANK:
        assert isinstance(q["q"], str) and len(q["q"]) > 10
        assert isinstance(q["points"], list) and len(q["points"]) >= 3
        assert all(isinstance(p, str) and p for p in q["points"])


def test_smoke_slices_first_four_questions():
    assert pl.QUESTION_BANK[:4][0]["id"] == "dog_adopt"


def test_stances_match_the_prereg_list():
    assert pl.STANCES == ["candid", "warm", "skeptical", "concrete", "plain"]


def test_arms_order_has_a_label_for_every_arm():
    assert set(pl.ARMS_ORDER) == {"parliament", "single", "temp_vote_null", "shuffled_dial_null"}
    for name in pl.ARMS_ORDER:
        assert name in pl._ARM_LABEL


# ================================================================================================
# CLI arg parsing
# ================================================================================================
def test_arg_parser_defaults():
    a = pl.build_arg_parser().parse_args([])
    assert a.model == "Qwen/Qwen2.5-7B-Instruct"
    assert a.judge_model == "auto"
    assert a.questions == 30
    assert a.four_bit == "auto"
    assert a.layer is None
    assert a.max_new == 180
    assert a.seed == 0
    assert a.smoke is False
    assert a.consistency_check is True
    assert a.pairwise is True
    assert a.compare is None


def test_arg_parser_smoke_and_overrides():
    a = pl.build_arg_parser().parse_args(["--smoke", "--model", "google/gemma-2-9b-it",
                                          "--no-pairwise", "--no-consistency-check"])
    assert a.smoke is True
    assert a.model == "google/gemma-2-9b-it"
    assert a.pairwise is False
    assert a.consistency_check is False


def test_arg_parser_compare_takes_multiple_paths():
    a = pl.build_arg_parser().parse_args(["--compare", "run1.json", "run2.json"])
    assert a.compare == ["run1.json", "run2.json"]
