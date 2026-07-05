"""Pure-logic tests for research/quine.py (Wild Experiment #9 -- the quine test, Amendment 1).

No GPU, no real model: quine.py imports torch/transformers at module level (like parliament.py and
mirror_bench.py's own antecedent chain), so importing it requires those packages installed, but every test
here exercises functions that make no real-model call -- CLI parsing, the forced-choice parser, the
teacher-forced logprob math (driven with tiny fake logits, no model), the SAE-readout labeling logic
(driven with a fake SAE stand-in), the aggregate() accuracy math, the SAE-bonus loader's stub/guard paths
(a real GpuSAE loaded from a tiny hand-written checkpoint -- CPU tensors only, no network, no real 7B), and
the prompt/pair bank's own integrity.
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # research/ on path
import quine as qn  # noqa: E402


@pytest.fixture(autouse=True)
def _force_cpu(monkeypatch):
    """This file validates model-free LOGIC only -- force quine.DEV to 'cpu' regardless of what hardware
    happens to be available on the machine running the tests, so a bare tensor allocation here never
    becomes an (even trivial) GPU op. Also keeps the fake-model tests below honest: the fakes return plain
    CPU tensors, so DEV must match or a real run would hit a device-mismatch that has nothing to do with
    the logic under test."""
    monkeypatch.setattr(qn, "DEV", "cpu")


# ================================================================================================
# wants_four_bit / axis_max_of
# ================================================================================================
def test_wants_four_bit_small_models_get_bf16():
    assert qn.wants_four_bit("Qwen/Qwen2.5-0.5B-Instruct", "auto") is False
    assert qn.wants_four_bit("Qwen/Qwen2.5-1.5B-Instruct", "auto") is False


def test_wants_four_bit_big_models_get_nf4():
    assert qn.wants_four_bit("Qwen/Qwen2.5-7B-Instruct", "auto") is True
    assert qn.wants_four_bit("google/gemma-2-9b-it", "auto") is True


def test_wants_four_bit_override_wins():
    assert qn.wants_four_bit("Qwen/Qwen2.5-7B-Instruct", "no") is False
    assert qn.wants_four_bit("Qwen/Qwen2.5-0.5B-Instruct", "yes") is True


class _FakeSC:
    def __init__(self, custom=None):
        self.custom = custom or {}


def test_axis_max_of_builtin_caps():
    assert qn.axis_max_of(_FakeSC(), "candid") == 0.45
    assert qn.axis_max_of(_FakeSC(), "concrete") == 0.5


def test_axis_max_of_warm_defaults_to_1p5():
    assert qn.axis_max_of(_FakeSC(), "warm") == 1.5


def test_axis_max_of_custom_axis():
    sc = _FakeSC(custom={"skeptical": {"max": 0.5}})
    assert qn.axis_max_of(sc, "skeptical") == 0.5


# ================================================================================================
# parse_choice -- the forced-choice A/B parser (must fail HONESTLY, never guess)
# ================================================================================================
def test_parse_choice_bare_token():
    assert qn.parse_choice("A") == "A"
    assert qn.parse_choice("b") == "B"
    assert qn.parse_choice(" B. ") == "B"


def test_parse_choice_last_token_wins_with_commentary():
    assert qn.parse_choice("I think the better one is B.") == "B"
    assert qn.parse_choice("Answer A seems off, so I'll say B, final answer B") == "B"


def test_parse_choice_no_letter_found_is_honest_none():
    assert qn.parse_choice("I'm not sure which one.") is None
    assert qn.parse_choice("") is None
    assert qn.parse_choice(None) is None


# ================================================================================================
# build_forced_choice_prompt
# ================================================================================================
def test_build_forced_choice_prompt_contains_everything():
    p = qn.build_forced_choice_prompt("What's up?", "text A here", "text B here", None)
    assert "What's up?" in p
    assert "text A here" in p and "text B here" in p
    assert "A or B" in p


def test_build_forced_choice_prompt_prepends_readout_when_present():
    p_with = qn.build_forced_choice_prompt("Q", "a", "b", "READOUT SENTENCE")
    p_without = qn.build_forced_choice_prompt("Q", "a", "b", None)
    assert p_with.startswith("READOUT SENTENCE")
    assert "READOUT SENTENCE" not in p_without
    assert p_with.endswith(p_without)


# ================================================================================================
# continuation_logprob -- teacher-forced logprob math, driven with a tiny fake model (no real LLM)
# ================================================================================================
class _FakeLMOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel:
    """Stands in for a real HF causal LM: __call__(ids_t) -> an object with .logits, driven by a
    caller-supplied function of ids_t. No torch.nn.Module, no real weights."""
    def __init__(self, logits_fn):
        self.logits_fn = logits_fn

    def __call__(self, ids_t):
        return _FakeLMOutput(self.logits_fn(ids_t))


def test_continuation_logprob_empty_continuation_short_circuits():
    model = _FakeModel(lambda ids_t: (_ for _ in ()).throw(AssertionError("should not be called")))
    r = qn.continuation_logprob(model, [0, 1], [])
    assert r == {"sum": 0.0, "mean": 0.0, "n_tokens": 0}


def test_continuation_logprob_prefers_the_higher_logit_token():
    v = 4

    def logits_fn(ids_t):
        length = ids_t.shape[1]
        base = torch.zeros(1, length, v)
        base[0, :, 2] = 10.0     # token id 2 is always the boosted (highest-logit) prediction
        return base

    model = _FakeModel(logits_fn)
    prompt_ids = [0, 1]
    lp_preferred = qn.continuation_logprob(model, prompt_ids, [2])   # matches the boosted token
    lp_other = qn.continuation_logprob(model, prompt_ids, [3])       # a low-logit token
    assert lp_preferred["mean"] > lp_other["mean"]
    assert lp_preferred["n_tokens"] == 1
    assert lp_other["n_tokens"] == 1


def test_continuation_logprob_mean_not_sum_is_length_fair():
    """A short continuation of all-boosted tokens and a long one of the same per-token quality should
    have the SAME mean (not a sum that trivially favors the shorter one)."""
    v = 4

    def logits_fn(ids_t):
        length = ids_t.shape[1]
        base = torch.zeros(1, length, v)
        base[0, :, 1] = 10.0
        return base

    model = _FakeModel(logits_fn)
    short = qn.continuation_logprob(model, [0], [1])
    long_ = qn.continuation_logprob(model, [0], [1, 1, 1])
    assert short["mean"] == pytest.approx(long_["mean"], abs=1e-3)
    assert long_["sum"] < short["sum"] - 1e-6 or long_["n_tokens"] > short["n_tokens"]


# ================================================================================================
# top_features_readout -- SAE-feature labeling logic, driven with a fake SAE (no real GPU/model)
# ================================================================================================
class _FakeSAE:
    def __init__(self, acts: torch.Tensor):
        self._acts = acts

    def encode(self, x):
        return self._acts.unsqueeze(0)   # [1, d_sae], ignoring x's actual value -- pure stand-in


def test_top_features_readout_picks_top_k_labeled_only_descending():
    acts = torch.tensor([5.0, 0.0, 3.0, 9.0, 1.0])
    labels = {"0": "alpha", "3": "delta"}       # ids 2 and 4 fire but have no label; id 1 doesn't fire
    out = qn.top_features_readout(_FakeSAE(acts), labels, torch.zeros(8), k=5)
    ids = [f["id"] for f in out["features"]]
    assert ids == [3, 0]                        # descending by activation, unlabeled ids skipped
    assert "delta" in out["readout"] and "alpha" in out["readout"]
    assert out["nnz"] == 4                       # four strictly-positive activations (5, 3, 9, 1)


def test_top_features_readout_respects_k():
    acts = torch.tensor([1.0, 2.0, 3.0, 4.0])
    labels = {"0": "a", "1": "b", "2": "c", "3": "d"}
    out = qn.top_features_readout(_FakeSAE(acts), labels, torch.zeros(4), k=2)
    assert [f["id"] for f in out["features"]] == [3, 2]


def test_top_features_readout_empty_when_nothing_labeled():
    acts = torch.tensor([1.0, 2.0])
    out = qn.top_features_readout(_FakeSAE(acts), {}, torch.zeros(4), k=5)
    assert out["features"] == []
    assert out["readout"] == ""


# ================================================================================================
# load_labeled_sae -- the bounded bonus arm's stub/guard paths
# ================================================================================================
def test_load_labeled_sae_stubs_non_qwen_model():
    sae, labels, status = qn.load_labeled_sae("google/gemma-2-9b-it", sc_layer=20)
    assert sae is None and labels is None
    assert status["status"] == "not_run"
    assert "gemma" in status["reason"].lower() or "qwen" in status["reason"].lower()


def test_load_labeled_sae_stubs_base_non_instruct_qwen():
    # the andyrdt SAE was trained on the INSTRUCT model's activations -- the base model must not silently
    # borrow it just because the shape happens to match.
    sae, labels, status = qn.load_labeled_sae("Qwen/Qwen2.5-7B", sc_layer=14)
    assert sae is None
    assert status["status"] == "not_run"


def test_load_labeled_sae_stubs_when_files_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(qn, "SAE_PT_QWEN", str(tmp_path / "missing.pt"))
    monkeypatch.setattr(qn, "SAE_LABELS_QWEN", str(tmp_path / "missing.json"))
    sae, labels, status = qn.load_labeled_sae("Qwen/Qwen2.5-7B-Instruct", sc_layer=14)
    assert sae is None and labels is None
    assert status["status"] == "not_run"


def _write_fake_sae_checkpoint(path, d_in=4, d_sae=6, layer=15):
    torch.save({
        "W_enc": torch.randn(d_in, d_sae).half(),
        "b_enc": torch.zeros(d_sae).half(),
        "b_dec": torch.zeros(d_in).half(),
        "threshold": torch.zeros(d_sae),
        "W_dec": torch.randn(d_sae, d_in).half(),
        "d_sae": d_sae,
        "layer": layer,
    }, path)


def test_load_labeled_sae_stubs_when_steering_layer_is_downstream_of_the_sae(monkeypatch, tmp_path):
    """The causality guard: if steering is applied at a layer AFTER the SAE's own layer, the SAE would
    read a residual computed before the steering hook ever fired -- must refuse, not silently run."""
    pt, lbl = tmp_path / "fake.pt", tmp_path / "fake.json"
    _write_fake_sae_checkpoint(pt, layer=15)
    lbl.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(qn, "SAE_PT_QWEN", str(pt))
    monkeypatch.setattr(qn, "SAE_LABELS_QWEN", str(lbl))
    sae, labels, status = qn.load_labeled_sae("Qwen/Qwen2.5-7B-Instruct", sc_layer=999)
    assert sae is None and labels is None
    assert status["status"] == "not_run"
    assert "layer" in status["reason"].lower()


def test_load_labeled_sae_succeeds_when_everything_lines_up(monkeypatch, tmp_path):
    pt, lbl = tmp_path / "fake.pt", tmp_path / "fake.json"
    _write_fake_sae_checkpoint(pt, layer=15)
    lbl.write_text(json.dumps({"0": "a label"}), encoding="utf-8")
    monkeypatch.setattr(qn, "SAE_PT_QWEN", str(pt))
    monkeypatch.setattr(qn, "SAE_LABELS_QWEN", str(lbl))
    sae, labels, status = qn.load_labeled_sae("Qwen/Qwen2.5-7B-Instruct", sc_layer=14)
    assert status["status"] == "ok"
    assert sae is not None
    assert labels == {"0": "a label"}
    assert status["sae_layer"] == 15
    assert str(sae.W_enc.device) == "cpu", "the _force_cpu fixture must keep this test hardware-free"


# ================================================================================================
# aggregate -- accuracy / parse-fail / degenerate-rate math over synthetic trial rows
# ================================================================================================
def _row(correct, degenerate=False):
    return {"correct": correct, "coherence": {"degenerate": degenerate, "reason": ""}}


def test_aggregate_accuracy_and_parse_fail_rate():
    trials = [
        {"conditions": {"dial_label": _row(True), "no_state": _row(False)},
         "baseline_unsteered": {"steering_shifted_pref": True}},
        {"conditions": {"dial_label": _row(True), "no_state": _row(None)},
         "baseline_unsteered": {"steering_shifted_pref": False}},
        {"conditions": {"dial_label": _row(False), "no_state": _row(False)},
         "baseline_unsteered": {"steering_shifted_pref": True}},
    ]
    agg = qn.aggregate(trials)
    assert agg["dial_label"]["n_total"] == 3
    assert agg["dial_label"]["n_decided"] == 3
    assert agg["dial_label"]["accuracy"] == round(2 / 3, 3)
    assert agg["no_state"]["n_decided"] == 2          # one parse failure excluded from the denominator
    assert agg["no_state"]["parse_fail_rate"] == round(1 / 3, 3)
    assert agg["_meta"]["n_trials"] == 3
    assert agg["_meta"]["pct_trials_steering_shifted_ground_truth"] == pytest.approx(200 / 3, abs=1e-1)


def test_aggregate_degenerate_rate_and_all_parse_failures_gives_none_accuracy():
    trials = [{"conditions": {"sae_feature": _row(None, degenerate=True)},
               "baseline_unsteered": {"steering_shifted_pref": False}}]
    agg = qn.aggregate(trials)
    assert agg["sae_feature"]["accuracy"] is None
    assert agg["sae_feature"]["n_decided"] == 0
    assert agg["sae_feature"]["degenerate_rate"] == 1.0


# ================================================================================================
# PROMPTS / CONT_PAIRS / STANCES integrity
# ================================================================================================
def test_stances_match_the_prereg_list():
    assert qn.STANCES == ["candid", "warm", "skeptical", "concrete", "plain"]


def test_prompts_bank_nonempty_and_distinct():
    assert len(qn.PROMPTS) >= 4
    assert len(set(qn.PROMPTS)) == len(qn.PROMPTS)


def test_cont_pairs_cover_every_stance_with_well_formed_distinct_pairs():
    assert set(qn.CONT_PAIRS.keys()) == set(qn.STANCES)
    for axis, pairs in qn.CONT_PAIRS.items():
        assert len(pairs) >= 1, f"{axis} needs at least one congruent/incongruent pair"
        for congruent, incongruent in pairs:
            assert isinstance(congruent, str) and isinstance(incongruent, str)
            assert len(congruent) > 5 and len(incongruent) > 5
            assert congruent != incongruent


def test_dose_frac_range_is_sane():
    lo, hi = qn._DOSE_FRAC_RANGE
    assert 0.0 <= lo < hi <= 1.0


# ================================================================================================
# CLI arg parsing
# ================================================================================================
def test_arg_parser_defaults():
    a = qn.build_arg_parser().parse_args([])
    assert a.model == "Qwen/Qwen2.5-7B-Instruct"
    assert a.trials == 30
    assert a.four_bit == "auto"
    assert a.layer is None
    assert a.seed == 0
    assert a.sae_topk == qn.DEFAULT_SAE_TOPK
    assert a.use_sae is True
    assert a.smoke is False
    assert a.compare is None


def test_arg_parser_smoke_and_no_sae():
    a = qn.build_arg_parser().parse_args(["--smoke", "--model", "google/gemma-2-9b-it", "--no-sae"])
    assert a.smoke is True
    assert a.model == "google/gemma-2-9b-it"
    assert a.use_sae is False


def test_arg_parser_compare_takes_multiple_paths():
    a = qn.build_arg_parser().parse_args(["--compare", "run1.json", "run2.json"])
    assert a.compare == ["run1.json", "run2.json"]


def test_arg_parser_dose_frac_overrides():
    a = qn.build_arg_parser().parse_args(["--dose-frac-min", "0.1", "--dose-frac-max", "0.9"])
    assert a.dose_frac_min == pytest.approx(0.1)
    assert a.dose_frac_max == pytest.approx(0.9)
