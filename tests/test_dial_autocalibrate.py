"""Pure-logic tests for research/dial_autocalibrate.py (the dial auto-calibration engine).

No GPU, no real model: dial_autocalibrate.py imports torch/transformers at module level (like parliament.py
and its own antecedents), so importing it requires those packages installed, but every test here exercises
functions that make no model call -- CLI parsing, the pure calibration math (_compute_calibration), the
DIRECTION-AWARE effect measure (directional_effect/directional_alignment -- the projection arithmetic
itself, _project_onto_unit, is tested directly on fabricated CPU tensors; directional_alignment's own
model-touching encode+forward+pool is exercised only indirectly, by MONKEYPATCHING it out with a FAKE
aligner, so directional_effect/calibrate_dial's plumbing is proven without ever touching sc.model/sc.tok),
the OLD word-Jaccard measure kept as a diagnostic (effect_vs_baseline, backed by receipts.receipt_metrics),
the coherence-rate wrapper, the deterministic seeding helpers, the runlog-backed prompt sampler (against a
temp run store, per test_runlog.py's own fixture pattern), the built-in/custom dial-routing logic
(compute_dials), and a full calibrate_dial sweep driven through FAKE Rig/SteeringControl stand-ins (no real
tensor's forward pass is ever run; make_shuffle_unit_vector is the one spot that touches real (CPU-only)
tensors, exactly as test_parliament.py's own shuffle-vector tests do).

THE CENTRAL REGRESSION COVERAGE (the reason this file was rewritten alongside dial_autocalibrate.py itself):
test_calibrate_dial_reformat_only_gets_no_usable_range / test_calibrate_dial_genuine_shift_gets_a_real_
usable_range exercise the SAME calibrate_dial sweep machinery with a FAKE aligner that can tell "wording
changed" apart from "moved toward the pole" -- confirming a dial that only reformats (lots of new words, the
OLD change_magnitude climbs, but the fake aligner's marker never appears) now gets NO usable range, while a
dial that genuinely accrues the marker gets a real one. This is the model-free proxy for the real-model
validation criterion (warm/poetic should diverge from skeptical/candid once run for real).

HYGIENE NOTE: compute_dials mutates the process-global steering.AXES dict (by design -- see its docstring).
Every test that calls it goes through the `restore_axes` fixture below, which snapshots and restores that
dict, so this file can never leak a narrowed AXES into some OTHER test module that expects the full
built-in set (test_dial_suggestion.py reads steering.AXES["concise"]/["warm"]/["candid"]/["concrete"]
directly and would break if a prior test in the same session left AXES narrowed).
"""
from __future__ import annotations

import json
import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)                                        # repo root, for `from clozn import ...`
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "calibration"))  # torch_autocalibrate.py lives in scripts/calibration/
import torch_autocalibrate as dac  # noqa: E402
import clozn.behavior.steering.axes as steering_mod   # noqa: E402
import clozn.runs.store as runlog  # noqa: E402


# ================================================================================================
# fixtures
# ================================================================================================
@pytest.fixture
def restore_axes():
    """Snapshot/restore steering.AXES around any test that calls compute_dials (which mutates it)."""
    original = dict(steering_mod.AXES)
    try:
        yield
    finally:
        steering_mod.AXES = original


@pytest.fixture
def runlog_store(tmp_path):
    """Redirect the run store to a temp dir for the duration of one test -- test_runlog.py's own pattern."""
    original = runlog.RUNS_DIR
    runlog.RUNS_DIR = str(tmp_path / "runs")
    try:
        yield runlog
    finally:
        runlog.RUNS_DIR = original


# ================================================================================================
# wants_four_bit
# ================================================================================================
def test_wants_four_bit_small_models_get_bf16():
    assert dac.wants_four_bit("Qwen/Qwen2.5-0.5B-Instruct", "auto") is False
    assert dac.wants_four_bit("Qwen/Qwen2.5-1.5B-Instruct", "auto") is False
    assert dac.wants_four_bit("some/Model-3B-Instruct", "auto") is False


def test_wants_four_bit_big_models_get_nf4():
    assert dac.wants_four_bit("Qwen/Qwen2.5-7B-Instruct", "auto") is True
    assert dac.wants_four_bit("google/gemma-2-9b-it", "auto") is True


def test_wants_four_bit_override_wins():
    assert dac.wants_four_bit("Qwen/Qwen2.5-7B-Instruct", "no") is False
    assert dac.wants_four_bit("Qwen/Qwen2.5-0.5B-Instruct", "yes") is True


# ================================================================================================
# axis_max_of
# ================================================================================================
class _FakeSC:
    def __init__(self, custom=None):
        self.custom = custom or {}


def test_axis_max_of_builtin_caps():
    assert dac.axis_max_of(_FakeSC(), "candid") == 0.45
    assert dac.axis_max_of(_FakeSC(), "concrete") == 0.5


def test_axis_max_of_default_when_unset():
    assert dac.axis_max_of(_FakeSC(), "warm") == 1.5


def test_axis_max_of_custom_axis():
    sc = _FakeSC(custom={"skeptical": {"max": 0.5}})
    assert dac.axis_max_of(sc, "skeptical") == 0.5


def test_axis_max_of_custom_overrides_builtin_on_name_collision():
    """A dial name that collides with a steering.AXES built-in (e.g. the candidate library's own "warm"/
    "concise"/"formal"/"playful"/"poetic"/"concrete"/"confident" entries) must be calibrated against the
    CUSTOM max it was explicitly re-registered with, not the built-in's -- add_custom already overwrites
    sc.vecs[name] on such a collision the same way; axis_max_of's precedence must track that same override
    or a library dial's swept ceiling would silently revert to a stale built-in default (see axis_max_of's
    own docstring)."""
    sc = _FakeSC(custom={"warm": {"max": 0.5}})
    assert dac.axis_max_of(sc, "warm") == 0.5             # NOT steering.AXES["warm"]'s default (1.5)
    assert dac.axis_max_of(_FakeSC(), "warm") == 1.5       # unchanged when there's no collision to resolve


# ================================================================================================
# _dial_seed -- deterministic, no Python hash()
# ================================================================================================
def test_dial_seed_deterministic():
    assert dac._dial_seed(0, "warm") == dac._dial_seed(0, "warm")


def test_dial_seed_distinct_per_name():
    assert dac._dial_seed(0, "warm") != dac._dial_seed(0, "candid")


def test_dial_seed_distinct_per_run_seed():
    assert dac._dial_seed(0, "warm") != dac._dial_seed(1, "warm")


# ================================================================================================
# make_shuffle_unit_vector -- pure CPU tensor math, no model
# ================================================================================================
def test_make_shuffle_unit_vector_unit_norm_and_shape():
    ref = torch.randn(16)
    v = dac.make_shuffle_unit_vector(ref, seed=42)
    assert v.shape == ref.shape
    assert torch.isclose(v.norm(), torch.tensor(1.0), atol=1e-4)


def test_make_shuffle_unit_vector_deterministic_given_same_seed():
    ref = torch.randn(16)
    v1 = dac.make_shuffle_unit_vector(ref, seed=7)
    v2 = dac.make_shuffle_unit_vector(ref, seed=7)
    assert torch.allclose(v1, v2)


def test_make_shuffle_unit_vector_differs_across_seeds():
    ref = torch.randn(16)
    v1 = dac.make_shuffle_unit_vector(ref, seed=1)
    v2 = dac.make_shuffle_unit_vector(ref, seed=2)
    assert not torch.allclose(v1, v2)


# ================================================================================================
# degenerate_rate
# ================================================================================================
def test_degenerate_rate_all_coherent():
    assert dac.degenerate_rate(["a normal reply", "another normal one"]) == 0.0


def test_degenerate_rate_flags_repetition():
    assert dac.degenerate_rate(["loop loop loop and more"]) == 1.0


def test_degenerate_rate_mixed():
    assert dac.degenerate_rate(["a fine reply here", "loop loop loop forever and ever"]) == 0.5


def test_degenerate_rate_empty_list():
    assert dac.degenerate_rate([]) == 0.0


# ================================================================================================
# effect_vs_baseline -- the OLD effect measure (1 - word-type Jaccard similarity, via
# receipts.receipt_metrics), KEPT as the `change_magnitude` diagnostic -- function itself is unchanged by
# the direction-aware rewrite, so its own unit tests are unchanged too.
# ================================================================================================
def test_effect_vs_baseline_identical_texts_is_zero():
    texts = ["the quick brown fox jumps", "hello there friend"]
    assert dac.effect_vs_baseline(texts, texts) == 0.0


def test_effect_vs_baseline_disjoint_vocab_is_maximal():
    baseline = ["the quick brown fox"]
    steered = ["completely different wording entirely"]
    assert dac.effect_vs_baseline(baseline, steered) == 1.0


def test_effect_vs_baseline_partial_overlap_is_between_zero_and_one():
    baseline = ["the quick brown fox jumps"]
    steered = ["the quick brown fox leaps"]
    eff = dac.effect_vs_baseline(baseline, steered)
    assert 0.0 < eff < 1.0


def test_effect_vs_baseline_averages_over_multiple_prompts():
    baseline = ["alpha beta gamma", "alpha beta gamma"]
    steered = ["alpha beta gamma", "completely disjoint words here"]   # one identical, one maximal
    eff = dac.effect_vs_baseline(baseline, steered)
    assert 0.4 < eff < 0.6   # roughly the average of 0.0 and 1.0


def test_effect_vs_baseline_mismatched_lengths_is_zero():
    assert dac.effect_vs_baseline(["a"], ["a", "b"]) == 0.0


def test_effect_vs_baseline_empty_is_zero():
    assert dac.effect_vs_baseline([], []) == 0.0
    assert dac.effect_vs_baseline(None, None) == 0.0


# ================================================================================================
# _project_onto_unit -- pure tensor math (no model): the arithmetic directional_alignment is built from
# ================================================================================================
def test_project_onto_unit_basic_dot_product():
    vec = torch.tensor([3.0, 4.0])
    direction = torch.tensor([1.0, 0.0])         # already unit
    assert dac._project_onto_unit(vec, direction) == pytest.approx(3.0)


def test_project_onto_unit_normalizes_a_non_unit_direction():
    vec = torch.tensor([3.0, 4.0])
    direction = torch.tensor([2.0, 0.0])         # norm 2, NOT unit -- must be normalized before projecting
    assert dac._project_onto_unit(vec, direction) == pytest.approx(3.0)


def test_project_onto_unit_orthogonal_is_zero():
    vec = torch.tensor([0.0, 5.0])
    direction = torch.tensor([1.0, 0.0])
    assert dac._project_onto_unit(vec, direction) == pytest.approx(0.0)


def test_project_onto_unit_negative_when_opposite():
    vec = torch.tensor([-3.0, 0.0])
    direction = torch.tensor([1.0, 0.0])
    assert dac._project_onto_unit(vec, direction) == pytest.approx(-3.0)


# ================================================================================================
# directional_effect -- the NEW, direction-aware effect measure. directional_alignment itself needs a
# real sc.model/sc.tok (a forward pass), so every test here MONKEYPATCHES dac.directional_alignment with a
# FAKE, controllable aligner instead of touching a model -- directional_effect calls the bare module-level
# name, so monkeypatching dac.directional_alignment transparently redirects it (and, later, calibrate_dial's
# calls to directional_effect too).
# ================================================================================================
def _fake_alignment_counts_marker(sc, text, dial):
    """FAKE directional_alignment stand-in: 'alignment' IS the count of the literal '__aligned' marker
    substring in the text -- a controllable, model-free stand-in for 'how far toward the dial's pole is
    this reply'. `sc`/`dial` accepted (matching the real signature) but unused."""
    return float((text or "").count("__aligned"))


def test_directional_effect_reformat_only_scores_zero(monkeypatch):
    monkeypatch.setattr(dac, "directional_alignment", _fake_alignment_counts_marker)
    baseline = ["baseline reply to hello"]
    steered = ["### Header\nbaseline reply to hello fmt0 fmt1 fmt2"]     # lots of new words, NO marker
    assert dac.directional_effect(None, "somedial", baseline, steered) == 0.0


def test_directional_effect_genuine_shift_scores_positive(monkeypatch):
    monkeypatch.setattr(dac, "directional_alignment", _fake_alignment_counts_marker)
    baseline = ["baseline reply to hello"]
    steered = ["baseline reply to hello __aligned0__ __aligned1__ __aligned2__"]
    assert dac.directional_effect(None, "somedial", baseline, steered) == 3.0


def test_directional_effect_averages_over_prompts(monkeypatch):
    monkeypatch.setattr(dac, "directional_alignment", _fake_alignment_counts_marker)
    baseline = ["a", "a"]
    steered = ["__aligned0__", "__aligned0__ __aligned1__"]
    assert dac.directional_effect(None, "d", baseline, steered) == 1.5


def test_directional_effect_can_be_negative(monkeypatch):
    """Unlike the old [0,1]-bounded Jaccard measure, directional_effect can go NEGATIVE (moved toward the
    dial's OPPOSITE pole) -- exercised with a fake that scores an 'anti'-marked reply negatively."""
    def _fake(sc, text, dial):
        return -float((text or "").count("anti"))
    monkeypatch.setattr(dac, "directional_alignment", _fake)
    assert dac.directional_effect(None, "d", ["a"], ["anti anti"]) == pytest.approx(-2.0)


def test_directional_effect_mismatched_lengths_is_zero(monkeypatch):
    monkeypatch.setattr(dac, "directional_alignment", _fake_alignment_counts_marker)
    assert dac.directional_effect(None, "d", ["a"], ["a", "b"]) == 0.0


def test_directional_effect_empty_is_zero(monkeypatch):
    monkeypatch.setattr(dac, "directional_alignment", _fake_alignment_counts_marker)
    assert dac.directional_effect(None, "d", [], []) == 0.0
    assert dac.directional_effect(None, "d", None, None) == 0.0


# ================================================================================================
# _compute_calibration -- the pure derail_point / dead_below / usable_max logic (the module's core).
#
# "effect"/"shuffled_effect" values below are expressed as MULTIPLES of dac._EFFECT_EPS (never a hardcoded
# absolute number) so these tests stay correct regardless of the exact epsilon chosen -- the module
# docstring is explicit that _EFFECT_EPS is an eyeballed, re-pickable cut, not a fixed law.
# ================================================================================================
def test_compute_calibration_normal_case():
    eps = dac._EFFECT_EPS
    curve = [
        {"frac": 0.0, "real_degenerate_rate": 0.0, "effect": 0.0, "shuffled_effect": 0.0},
        {"frac": 0.5, "real_degenerate_rate": 0.0, "effect": eps * 10, "shuffled_effect": eps * 1.5},
        {"frac": 1.0, "real_degenerate_rate": 0.1, "effect": eps * 20, "shuffled_effect": eps * 6},
        {"frac": 1.5, "real_degenerate_rate": 0.8, "effect": eps * 30, "shuffled_effect": eps * 10},
    ]
    out = dac._compute_calibration(curve)
    assert out["derail_point"] == 1.5
    assert out["dead_below"] == 0.5
    assert out["usable_max"] == 1.0
    assert out["usable_range"] == [0.5, 1.0]
    assert out["range_valid"] is True


def test_compute_calibration_dial_never_does_anything():
    curve = [{"frac": f, "real_degenerate_rate": 0.0, "effect": 0.0, "shuffled_effect": 0.0}
             for f in (0.0, 0.5, 1.0, 1.5)]
    out = dac._compute_calibration(curve)
    assert out["dead_below"] is None
    assert out["usable_max"] is None
    assert out["derail_point"] is None
    assert out["range_valid"] is False


def test_compute_calibration_effect_never_beats_shuffled_null():
    # a nonzero real effect exists at every dose, but the shuffled null matches or beats it every time --
    # "the dial did it" is never established, so nothing should count as usable despite a "real" effect.
    eps = dac._EFFECT_EPS
    curve = [
        {"frac": 0.0, "real_degenerate_rate": 0.0, "effect": 0.0, "shuffled_effect": 0.0},
        {"frac": 0.5, "real_degenerate_rate": 0.0, "effect": eps * 6, "shuffled_effect": eps * 8},
        {"frac": 1.0, "real_degenerate_rate": 0.0, "effect": eps * 12, "shuffled_effect": eps * 15},
    ]
    out = dac._compute_calibration(curve)
    assert out["dead_below"] == 0.5      # effect alone is "real" (exceeds _EFFECT_EPS)...
    assert out["usable_max"] is None     # ...but never attributable to the dial's own direction
    assert out["range_valid"] is False


def test_compute_calibration_derails_at_the_first_nonzero_dose():
    eps = dac._EFFECT_EPS
    curve = [
        {"frac": 0.0, "real_degenerate_rate": 0.0, "effect": 0.0, "shuffled_effect": 0.0},
        {"frac": 0.5, "real_degenerate_rate": 0.9, "effect": eps * 20, "shuffled_effect": eps * 3},
    ]
    out = dac._compute_calibration(curve)
    assert out["derail_point"] == 0.5
    assert out["usable_max"] is None   # the only nonzero dose swept derails -- nothing survives the gate
    assert out["range_valid"] is False


def test_compute_calibration_high_effect_but_degenerate_is_excluded_from_usable_max():
    # Law #6's trap, directly: the HIGHEST-effect dose is also the derailed one -- usable_max must not be
    # fooled by the raw number, even under the new direction-aware measure (a degenerate reply can still,
    # in principle, project oddly; the coherence gate is a mandatory backstop regardless of what "effect"
    # means).
    eps = dac._EFFECT_EPS
    curve = [
        {"frac": 0.0, "real_degenerate_rate": 0.0, "effect": 0.0, "shuffled_effect": 0.0},
        {"frac": 0.5, "real_degenerate_rate": 0.0, "effect": eps * 10, "shuffled_effect": eps * 1},
        {"frac": 1.0, "real_degenerate_rate": 1.0, "effect": eps * 25, "shuffled_effect": eps * 1},
    ]
    out = dac._compute_calibration(curve)
    assert out["usable_max"] == 0.5
    assert out["usable_max"] != 1.0
    assert out["derail_point"] == 1.0


def test_compute_calibration_never_derails_in_swept_range():
    eps = dac._EFFECT_EPS
    curve = [
        {"frac": 0.0, "real_degenerate_rate": 0.0, "effect": 0.0, "shuffled_effect": 0.0},
        {"frac": 0.5, "real_degenerate_rate": 0.1, "effect": eps * 8, "shuffled_effect": eps * 1.5},
        {"frac": 1.0, "real_degenerate_rate": 0.2, "effect": eps * 15, "shuffled_effect": eps * 3},
    ]
    out = dac._compute_calibration(curve)
    assert out["derail_point"] is None
    assert out["usable_max"] == 1.0


def test_compute_calibration_negative_effect_is_not_counted_as_real():
    # the NEW effect measure can go negative (moved toward the dial's OPPOSITE pole) -- unlike the old
    # [0,1]-bounded Jaccard measure. A negative number must never be misread as "a real positive effect".
    eps = dac._EFFECT_EPS
    curve = [
        {"frac": 0.0, "real_degenerate_rate": 0.0, "effect": 0.0, "shuffled_effect": 0.0},
        {"frac": 0.5, "real_degenerate_rate": 0.0, "effect": -eps * 5, "shuffled_effect": 0.0},
    ]
    out = dac._compute_calibration(curve)
    assert out["dead_below"] is None
    assert out["usable_max"] is None
    assert out["range_valid"] is False


def test_compute_calibration_ignores_change_magnitude_field():
    # change_magnitude (the OLD, kept-as-diagnostic measure) can be huge while the direction-aware `effect`
    # is only modestly above the noise floor -- _compute_calibration must key off `effect` alone. This is
    # the pure-math mirror of the reformat-vs-genuine calibrate_dial tests further down.
    eps = dac._EFFECT_EPS
    curve = [
        {"frac": 0.0, "real_degenerate_rate": 0.0, "effect": 0.0, "shuffled_effect": 0.0,
         "change_magnitude": 0.0},
        {"frac": 0.5, "real_degenerate_rate": 0.0, "effect": eps * 2, "shuffled_effect": eps * 0.1,
         "change_magnitude": 0.95},
    ]
    out = dac._compute_calibration(curve)
    assert out["dead_below"] == 0.5
    assert out["usable_max"] == 0.5


# ================================================================================================
# sample_prompts -- runlog-backed, with a neutral fallback
# ================================================================================================
def test_sample_prompts_falls_back_to_neutral_when_runlog_empty(runlog_store):
    prompts, source = dac.sample_prompts(4)
    assert source == "neutral-fallback"
    assert prompts == dac.NEUTRAL_PROMPTS[:4]


def test_sample_prompts_uses_recent_distinct_runlog_turns_newest_first(runlog_store):
    runlog_store.record(source="cli", messages=[{"role": "user", "content": "first question"}],
                        response="a", started=1000.0, ended=1000.1)
    runlog_store.record(source="cli", messages=[{"role": "user", "content": "second question"}],
                        response="b", started=2000.0, ended=2000.1)
    runlog_store.record(source="cli", messages=[{"role": "user", "content": "third question"}],
                        response="c", started=3000.0, ended=3000.1)
    prompts, source = dac.sample_prompts(2)
    assert source == "runlog"
    assert prompts == ["third question", "second question"]


def test_sample_prompts_dedupes_identical_text(runlog_store):
    runlog_store.record(source="cli", messages=[{"role": "user", "content": "same text"}],
                        response="a", started=1000.0, ended=1000.1)
    runlog_store.record(source="cli", messages=[{"role": "user", "content": "same text"}],
                        response="b", started=2000.0, ended=2000.1)
    runlog_store.record(source="cli", messages=[{"role": "user", "content": "different text"}],
                        response="c", started=3000.0, ended=3000.1)
    prompts, source = dac.sample_prompts(5)
    assert source == "runlog"
    assert prompts == ["different text", "same text"]


def test_sample_prompts_skips_blank_user_text(runlog_store):
    runlog_store.record(source="cli", messages=[{"role": "user", "content": "   "}],
                        response="a", started=1000.0, ended=1000.1)
    runlog_store.record(source="cli", messages=[{"role": "user", "content": "a real question"}],
                        response="b", started=2000.0, ended=2000.1)
    prompts, source = dac.sample_prompts(5)
    assert prompts == ["a real question"]


def test_sample_prompts_respects_n_cap(runlog_store):
    for i in range(5):
        runlog_store.record(source="cli", messages=[{"role": "user", "content": f"question {i}"}],
                            response=str(i), started=1000.0 + i, ended=1000.0 + i)
    prompts, source = dac.sample_prompts(2)
    assert len(prompts) == 2
    assert source == "runlog"


# ================================================================================================
# DEFAULT_DIALS / NEUTRAL_PROMPTS / sweep constants -- integrity
# ================================================================================================
_EXPECTED_BUILTIN_AXES = {"warm", "concise", "formal", "playful", "curious", "poetic", "technical",
                          "candid", "confident", "concrete"}


def test_default_dials_is_builtins_plus_customs():
    assert set(dac.DEFAULT_DIALS) == _EXPECTED_BUILTIN_AXES | {"skeptical", "plain"}
    assert len(dac.DEFAULT_DIALS) == len(set(dac.DEFAULT_DIALS)), "no duplicate dial names"


def test_neutral_prompts_nonempty_and_distinct():
    assert len(dac.NEUTRAL_PROMPTS) >= 6
    assert len(dac.NEUTRAL_PROMPTS) == len(set(dac.NEUTRAL_PROMPTS))
    assert all(isinstance(p, str) and len(p) > 5 for p in dac.NEUTRAL_PROMPTS)


def test_sweep_fracs_match_spec():
    assert dac._SWEEP_FRACS == [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def test_sweep_fracs_smoke_is_three_points_starting_at_zero():
    assert len(dac._SWEEP_FRACS_SMOKE) == 3
    assert dac._SWEEP_FRACS_SMOKE[0] == 0.0


# ================================================================================================
# compute_dials -- built-in/custom dial routing (mutates steering.AXES -- always via restore_axes)
# ================================================================================================
class _FakeSCForCompute:
    """Stands in for SteeringControl in compute_dials: .compute()/.add_custom() are simple recording
    stubs -- no forward pass, no model."""
    def __init__(self):
        self.custom = {}
        self.compute_calls = 0
        self.add_custom_calls = []

    def compute(self):
        self.compute_calls += 1
        return {"raw_norms": {}, "resid_norm": 1.0, "base": 1.0}

    def add_custom(self, name, pos, neg, mx=0.5):
        self.add_custom_calls.append((name, pos, neg, mx))
        self.custom[name] = {"pos": pos, "neg": neg, "max": mx}


def test_compute_dials_narrows_axes_to_requested_builtins(restore_axes):
    sc = _FakeSCForCompute()
    dac.compute_dials(sc, ["warm", "candid"])
    assert set(steering_mod.AXES.keys()) == {"warm", "candid"}
    assert sc.compute_calls == 1


def test_compute_dials_registers_known_customs(restore_axes):
    sc = _FakeSCForCompute()
    info = dac.compute_dials(sc, ["warm", "skeptical", "plain"])
    assert set(steering_mod.AXES.keys()) == {"warm"}
    names = {c[0] for c in sc.add_custom_calls}
    assert names == {"skeptical", "plain"}
    assert info["custom_axes"]["skeptical"]["max"] == 0.5
    assert info["custom_axes"]["plain"]["max"] == 0.5
    assert info["unknown_dials"] == []


def test_compute_dials_unknown_name_reported_not_raised(restore_axes):
    sc = _FakeSCForCompute()
    info = dac.compute_dials(sc, ["warm", "not_a_real_dial"])
    assert info["unknown_dials"] == ["not_a_real_dial"]
    assert sc.add_custom_calls == []
    assert sc.compute_calls == 1


def test_compute_dials_custom_only_still_calibrates_base_without_narrowing_axes(restore_axes):
    original_keys = set(steering_mod.AXES.keys())
    sc = _FakeSCForCompute()
    dac.compute_dials(sc, ["skeptical", "plain"])
    assert sc.compute_calls == 1                              # base IS calibrated even with 0 built-ins
    assert set(steering_mod.AXES.keys()) == original_keys      # nothing narrowed away


# ================================================================================================
# calibrate_dial -- the full sweep, end to end, through FAKE Rig/SteeringControl stand-ins.
#
# directional_alignment is ALWAYS monkeypatched out in these tests too (calibrate_dial calls
# directional_effect, which calls the bare module-level directional_alignment) -- no test in this file ever
# constructs a real sc.model/sc.tok.
# ================================================================================================
class _FakeSCForSweep:
    """Stands in for SteeringControl in calibrate_dial: .vecs holds one REAL small CPU tensor per dial
    (make_shuffle_unit_vector needs actual tensor ops); .strength/.clear/.engage/.disengage behave like the
    real thing (a plain dict + flags). No forward pass, no CUDA."""
    def __init__(self, dial_name: str, dim: int = 8, axis_max: float = 1.0):
        self.vecs = {dial_name: torch.randn(dim)}
        self.strength: dict = {}
        self.custom = {dial_name: {"max": axis_max}}
        self.engaged = False

    def clear(self):
        self.strength = {}

    def engage(self):
        self.engaged = True

    def disengage(self):
        self.engaged = False


class _FakeRigForSweep:
    """calibrate_dial's Rig stand-in. Reads the fake sc's CURRENT .strength at generate time to decide the
    canned reply -- mirroring how the REAL forward hook reads live dial state (Rig.gen never takes a dial
    argument; the coupling IS whatever is currently engaged on the model). Simulates: increasing dose ->
    increasingly different wording (a real, growing effect) up to raw strength 1.0, then DERAILS into a
    repetition loop past that -- while the shuffled null at the same magnitude keeps producing new (but
    fewer, and always coherent) words, so the real direction is both the one that derails AND the one that
    changes text more per unit of dose, at every coherent level. Paired with _align_by_steer_marker (below)
    for the direction-aware `effect` field: only the real dial's branch ever emits the literal 'steer'
    substring, so that fake aligner scores 0 for baseline, 0 for the shuffled null, and 0 for the derailed
    'loop' text -- exactly like a real white-box projection SHOULD treat a repetition loop (not "further
    toward the pole", just broken)."""
    def __init__(self, sc, dial_name: str):
        self.sc = sc
        self.dial_name = dial_name
        self.calls = 0

    def gen(self, prompt, max_new=100, sample=False, temperature=0.9):
        self.calls += 1
        active = dict(self.sc.strength) if self.sc.engaged else {}
        if not active:
            return f"baseline reply to {prompt}"
        key, val = next(iter(active.items()))
        if key == self.dial_name:
            if val > 1.0:                                        # derail past the "safe" ceiling
                return "loop loop loop loop loop loop loop loop"
            n = int(val * 10)
            return "baseline reply to " + prompt + " " + " ".join(f"steer{i}" for i in range(n))
        n = int(val * 3)                                          # shuffled null: real, but smaller, effect
        return "baseline reply to " + prompt + " " + " ".join(f"shuf{i}" for i in range(n))


def _align_by_steer_marker(sc, text, dial):
    """FAKE directional_alignment for tests built on _FakeRigForSweep: that fake rig's real-dial branch
    emits distinct 'steer<i>' tokens, its shuffled-null branch emits distinct 'shuf<i>' tokens, and its
    baseline/derailed-loop text contains neither -- so counting the literal substring 'steer' is a
    controllable, model-free stand-in for 'how far toward the dial's pole is this reply' that is 0 for
    baseline, 0 for the shuffled null, and 0 for a derailed 'loop' reply, growing with dose for the real
    dial. `sc`/`dial` accepted (matching the real signature) but unused."""
    return float((text or "").count("steer"))


def test_calibrate_dial_end_to_end_with_fake_generator(monkeypatch):
    monkeypatch.setattr(dac, "directional_alignment", _align_by_steer_marker)
    sc = _FakeSCForSweep("testdial", axis_max=1.0)
    rig = _FakeRigForSweep(sc, "testdial")
    prompts = ["hello there", "what is up"]
    fracs = [0.0, 0.5, 1.0, 1.5]

    report = dac.calibrate_dial(rig, sc, "testdial", prompts, fracs, seed=0, max_new=50)

    assert report["dial"] == "testdial"
    assert report["axis_max"] == 1.0
    assert len(report["curve"]) == 4
    assert len(report["sample_replies"]) == 4

    c0 = report["curve"][0]
    assert c0["frac"] == 0.0
    assert c0["effect"] == 0.0
    assert c0["real_degenerate_rate"] == 0.0

    c15 = next(c for c in report["curve"] if c["frac"] == 1.5)
    assert c15["real_degenerate_rate"] == 1.0        # the real direction derails at this magnitude...
    assert c15["shuffled_degenerate_rate"] == 0.0    # ...but the matched-norm null does not
    # the OLD metric (kept as the change_magnitude diagnostic) would have called this dose a HUGE effect --
    # a repetition loop shares no word types with the baseline -- exactly the false-positive the rewrite
    # targets. The NEW direction-aware metric doesn't even need the coherence gate to reject this specific
    # case (the fake's "steer" marker never appears in a repetition loop); the gate remains mandatory
    # regardless (see _compute_calibration's own Law #6 test).
    assert c15["change_magnitude"] == 1.0
    assert c15["effect"] == 0.0

    assert report["derail_point"] == 1.5
    # usable_max must NOT be 1.5 -- the coherence gate has to veto it (Law #6, exercised end-to-end here
    # rather than just on a fabricated curve as in the _compute_calibration tests above).
    assert report["usable_max"] == 1.0
    assert report["usable_max"] != 1.5
    assert report["dead_below"] == 0.5
    assert report["usable_range"] == [0.5, 1.0]
    assert report["range_valid"] is True

    s0 = report["sample_replies"][0]
    assert s0["frac"] == 0.0
    assert s0["prompt"] == prompts[0]
    assert s0["baseline_reply"] == s0["steered_reply"]      # dose 0 -- identical by construction


def test_calibrate_dial_shuffle_vector_stays_fixed_across_the_sweep(monkeypatch):
    """The SAME shuffled direction is reused at every dose for one dial (not re-drawn each time) -- the
    module docstring's stated design. Verified indirectly: two sweeps with the same seed produce identical
    curves (determinism would break if a fresh random vector were drawn per dose)."""
    monkeypatch.setattr(dac, "directional_alignment", _align_by_steer_marker)
    fracs = [0.0, 0.5, 1.0]
    prompts = ["a prompt"]
    r1 = dac.calibrate_dial(_FakeRigForSweep(sc := _FakeSCForSweep("d", axis_max=1.0), "d"),
                            sc, "d", prompts, fracs, seed=3, max_new=20)
    r2 = dac.calibrate_dial(_FakeRigForSweep(sc2 := _FakeSCForSweep("d", axis_max=1.0), "d"),
                            sc2, "d", prompts, fracs, seed=3, max_new=20)
    assert r1["curve"] == r2["curve"]


# ---- the metric-fix regression tests: reformat-only vs genuine direction shift ----------------------
class _FakeRigForDirectionalSweep:
    """calibrate_dial's Rig stand-in for exercising the direction-aware effect metric end to end, paired
    with the monkeypatched _align_by_aligned_marker fake below. Two modes, selected by `reformat_only`:
      * reformat_only=False (GENUINE): the real dial's dose inserts `n` distinct '__aligned<i>__' tokens
        (n growing with dose) -- the fake aligner's score climbs with dose, modeling a dial that genuinely
        moves the reply toward its pole.
      * reformat_only=True (REFORMAT-ONLY): the real dial's dose instead inserts `n` distinct, MARKER-FREE
        '### fmt<i>' tokens -- lots of new, never-before-seen wording (so the OLD Jaccard change_magnitude
        still climbs with dose, exactly like a genuine dial would have looked under the OLD metric) while
        the fake aligner's score stays at 0 -- modeling the bug this whole fix targets: a dial that only
        reformats.
    Either way, the SHUFFLED arm (key != dial_name) never inserts the '__aligned' marker -- a random
    direction should not move the reply along THIS dial's real axis, whichever mode the real arm is in.
    `n` is deliberately scaled well past dac._EFFECT_EPS (x40 per unit dose) so the GENUINE case's usable
    range is robust to a reasonable future re-tuning of that constant, not pinned to today's exact value.
    Distinct per-index suffixes (never a bare repeated token) so counterfactual._coherence's repeat-3gram
    check never fires here, matching _FakeRigForSweep's own steer{i}/shuf{i} convention."""
    def __init__(self, sc, dial_name: str, reformat_only: bool = False):
        self.sc = sc
        self.dial_name = dial_name
        self.reformat_only = reformat_only

    def gen(self, prompt, max_new=100, sample=False, temperature=0.9):
        active = dict(self.sc.strength) if self.sc.engaged else {}
        if not active:
            return f"baseline reply to {prompt}"
        key, val = next(iter(active.items()))
        n = int(val * 40)
        if key == self.dial_name and not self.reformat_only:
            return "baseline reply to " + prompt + " " + " ".join(f"__aligned{i}__" for i in range(n))
        if key == self.dial_name and self.reformat_only:
            return "### Header\nbaseline reply to " + prompt + " " + " ".join(f"fmt{i}" for i in range(n))
        return "baseline reply to " + prompt + " " + " ".join(f"shuf{i}" for i in range(n))  # null: no marker


def _align_by_aligned_marker(sc, text, dial):
    """FAKE directional_alignment: 'alignment' IS the count of the literal '__aligned' marker substring --
    see _FakeRigForDirectionalSweep. `sc`/`dial` accepted (matching the real signature) but unused: these
    tests calibrate only one dial at a time, so there is no second dial's marker to confuse this with."""
    return float((text or "").count("__aligned"))


def test_calibrate_dial_reformat_only_gets_no_usable_range(monkeypatch):
    """THE regression test for the metric fix, mirroring the real-model validation criterion: a dial whose
    replies keep changing WORDING with dose -- so the OLD Jaccard-based change_magnitude climbs, exactly
    the false-positive this module used to report as "usable" -- but never actually moves toward the dial's
    pole (the fake aligner never finds its marker in this rig's output) must now report NO usable range.
    Reformatting is not steering."""
    monkeypatch.setattr(dac, "directional_alignment", _align_by_aligned_marker)
    sc = _FakeSCForSweep("reformat_dial", axis_max=1.0)
    rig = _FakeRigForDirectionalSweep(sc, "reformat_dial", reformat_only=True)
    prompts = ["hello there", "what is up"]
    fracs = [0.0, 0.5, 1.0, 1.5]

    report = dac.calibrate_dial(rig, sc, "reformat_dial", prompts, fracs, seed=0, max_new=50)

    nonzero = [c for c in report["curve"] if c["frac"] > 0]
    assert all(c["change_magnitude"] > 0.3 for c in nonzero), "the OLD metric sees real change (the bug)"
    assert all(c["effect"] == 0.0 for c in nonzero), "the NEW metric sees no directional movement at all"
    assert report["derail_point"] is None       # this is NOT a coherence story -- nothing here derails
    assert report["dead_below"] is None
    assert report["usable_max"] is None
    assert report["usable_range"] == [None, None]
    assert report["range_valid"] is False


def test_calibrate_dial_genuine_shift_gets_a_real_usable_range(monkeypatch):
    """Counterpart: a dial whose replies genuinely accrue the fake aligner's marker with dose gets a real,
    valid usable range from the SAME sweep machinery -- confirming the fix credits genuine movement, not
    just penalizing reformatting."""
    monkeypatch.setattr(dac, "directional_alignment", _align_by_aligned_marker)
    sc = _FakeSCForSweep("genuine_dial", axis_max=1.0)
    rig = _FakeRigForDirectionalSweep(sc, "genuine_dial", reformat_only=False)
    prompts = ["hello there", "what is up"]
    fracs = [0.0, 0.5, 1.0, 1.5]

    report = dac.calibrate_dial(rig, sc, "genuine_dial", prompts, fracs, seed=0, max_new=50)

    nonzero = [c for c in report["curve"] if c["frac"] > 0]
    assert all(c["effect"] > 0 for c in nonzero)
    assert all(c["effect"] > c["shuffled_effect"] for c in nonzero)
    assert report["derail_point"] is None
    assert report["dead_below"] == 0.5
    assert report["usable_max"] == 1.5
    assert report["range_valid"] is True


# ================================================================================================
# CLI arg parsing
# ================================================================================================
def test_arg_parser_defaults():
    a = dac.build_arg_parser().parse_args([])
    assert a.model == "Qwen/Qwen2.5-7B-Instruct"
    assert a.dials is None
    assert a.library is None
    assert a.report is None
    assert a.curated_out == "research/runs/dial_library_curated.json"
    assert a.n_prompts == 6
    assert a.out is None                 # resolved at call time by _default_out_path -- see its own tests
    assert a.four_bit == "auto"
    assert a.layer is None
    assert a.max_new == 100
    assert a.seed == 0
    assert a.smoke is False


def test_arg_parser_smoke_and_dials_override():
    a = dac.build_arg_parser().parse_args(["--smoke", "--dials", "warm", "candid", "--n-prompts", "3"])
    assert a.smoke is True
    assert a.dials == ["warm", "candid"]
    assert a.n_prompts == 3


def test_arg_parser_four_bit_choices():
    a = dac.build_arg_parser().parse_args(["--four-bit", "yes"])
    assert a.four_bit == "yes"


def test_arg_parser_library_and_report_flags():
    a = dac.build_arg_parser().parse_args(["--library", "research/dial_library_candidates.json"])
    assert a.library == "research/dial_library_candidates.json"
    assert a.report is None

    b = dac.build_arg_parser().parse_args(["--report", "research/runs/dial_library_sweep.json",
                                           "--curated-out", "somewhere/curated.json"])
    assert b.report == "research/runs/dial_library_sweep.json"
    assert b.curated_out == "somewhere/curated.json"


# ================================================================================================
# _default_out_path -- pure, no I/O
# ================================================================================================
def test_default_out_path_plain_sweep():
    assert dac._default_out_path(None) == "research/runs/dial_autocalibrate.json"


def test_default_out_path_library_sweep():
    assert dac._default_out_path("research/dial_library_candidates.json") == "research/runs/dial_library_sweep.json"


# ================================================================================================
# load_dial_library -- pure I/O + validation, no model
# ================================================================================================
def _write_json(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def test_load_dial_library_valid_file(tmp_path):
    data = {"dials": [
        {"name": "a", "category": "cat1", "pos": "p", "neg": "n", "predict": "surface"},
        {"name": "b", "category": "cat1", "pos": "p2", "neg": "n2", "predict": "cognitive"},
    ]}
    path = _write_json(tmp_path, "lib.json", data)
    dials = dac.load_dial_library(path)
    assert [d["name"] for d in dials] == ["a", "b"]


def test_load_dial_library_missing_dials_key_raises(tmp_path):
    path = _write_json(tmp_path, "lib.json", {"not_dials": []})
    with pytest.raises(ValueError, match="dials"):
        dac.load_dial_library(path)


def test_load_dial_library_empty_dials_list_raises(tmp_path):
    path = _write_json(tmp_path, "lib.json", {"dials": []})
    with pytest.raises(ValueError):
        dac.load_dial_library(path)


def test_load_dial_library_missing_required_field_raises(tmp_path):
    data = {"dials": [{"name": "a", "category": "cat1", "pos": "p", "neg": "n"}]}   # no "predict"
    path = _write_json(tmp_path, "lib.json", data)
    with pytest.raises(ValueError, match="predict"):
        dac.load_dial_library(path)


def test_load_dial_library_duplicate_name_raises(tmp_path):
    data = {"dials": [
        {"name": "a", "category": "cat1", "pos": "p", "neg": "n", "predict": "surface"},
        {"name": "a", "category": "cat2", "pos": "p2", "neg": "n2", "predict": "cognitive"},
    ]}
    path = _write_json(tmp_path, "lib.json", data)
    with pytest.raises(ValueError, match="duplicate"):
        dac.load_dial_library(path)


@pytest.mark.skip(reason="dial_library_candidates.json moved to the sibling ../clozn-research repo in the "
                         "reorg; this candidate-schema check belongs with it, not in the product suite")
def test_load_dial_library_real_candidate_file_loads_cleanly():
    """The actual dial_library_candidates.json this rig ships with must load through the same
    validation path -- catches a future hand-edit to that file breaking the schema before any GPU work."""
    path = os.path.join(REPO_ROOT, "research", "dial_library_candidates.json")
    dials = dac.load_dial_library(path)
    assert len(dials) >= 60
    names = [d["name"] for d in dials]
    assert len(names) == len(set(names))
    assert all(d["predict"] in ("surface", "cognitive", "uncertain") for d in dials)


# ================================================================================================
# register_library_dials -- batch add_custom, no model (fake sc records the calls)
# ================================================================================================
class _FakeSCForLibrary:
    """Stands in for SteeringControl in register_library_dials: .add_custom is a simple recording stub --
    no forward pass, no model. Same pattern as _FakeSCForCompute above."""
    def __init__(self):
        self.custom = {}
        self.add_custom_calls = []

    def add_custom(self, name, pos, neg, mx=0.5):
        self.add_custom_calls.append((name, pos, neg, mx))
        self.custom[name] = {"pos": pos, "neg": neg, "max": mx}


def _sample_library():
    return [
        {"name": "made_up_dial_xyz", "category": "affect_tone", "predict": "surface",
         "pos": "pos text", "neg": "neg text"},
        {"name": "warm", "category": "affect_tone", "predict": "surface",    # collides with steering.AXES
         "pos": "warm pos", "neg": "warm neg"},
    ]


def test_register_library_dials_calls_add_custom_for_every_entry():
    sc = _FakeSCForLibrary()
    meta = dac.register_library_dials(sc, _sample_library())
    assert list(meta) == ["made_up_dial_xyz", "warm"]
    assert len(sc.add_custom_calls) == 2
    assert sc.add_custom_calls[0] == ("made_up_dial_xyz", "pos text", "neg text", dac._LIBRARY_DEFAULT_MAX)


def test_register_library_dials_carries_category_predict_pos_neg():
    sc = _FakeSCForLibrary()
    meta = dac.register_library_dials(sc, _sample_library())
    assert meta["made_up_dial_xyz"]["category"] == "affect_tone"
    assert meta["made_up_dial_xyz"]["predict"] == "surface"
    assert meta["made_up_dial_xyz"]["pos"] == "pos text"
    assert meta["made_up_dial_xyz"]["neg"] == "neg text"
    assert meta["made_up_dial_xyz"]["max"] == dac._LIBRARY_DEFAULT_MAX


def test_register_library_dials_flags_builtin_name_collisions():
    sc = _FakeSCForLibrary()
    meta = dac.register_library_dials(sc, _sample_library())
    assert meta["warm"]["shadows_builtin"] is True          # "warm" IS a steering.AXES built-in
    assert meta["made_up_dial_xyz"]["shadows_builtin"] is False


# ================================================================================================
# --report mode's pure analysis functions -- category_summary / hypothesis_verdict / curated_library /
# report(), all against a small hand-built fixture sweep JSON (as run_library would produce). No model, no
# GPU: this is the model-free unit coverage the task requires for --report.
# ================================================================================================
def _fixture_sweep():
    """4 dials, 2 categories, 2 predict tags, exactly ONE winner (warm2) -- small enough to hand-verify
    every number in the tests below."""
    dials = {
        "warm2": {"category": "affect_tone", "predict": "surface", "range_valid": True,
                  "usable_range": [0.25, 1.0], "derail_point": None,
                  "pos": "warm pos", "neg": "warm neg",
                  "curve": [{"frac": 0.0, "effect": 0.0}, {"frac": 0.5, "effect": 5.0},
                           {"frac": 1.0, "effect": 8.0}]},
        "cheerful": {"category": "affect_tone", "predict": "surface", "range_valid": False,
                     "usable_range": [None, None], "derail_point": None,
                     "pos": "cheerful pos", "neg": "cheerful neg",
                     "curve": [{"frac": 0.0, "effect": 0.0}, {"frac": 0.5, "effect": 0.1}]},
        "skeptical2": {"category": "epistemic_stance", "predict": "cognitive", "range_valid": False,
                       "usable_range": [None, None], "derail_point": None,
                       "pos": "skeptical pos", "neg": "skeptical neg",
                       "curve": [{"frac": 0.0, "effect": 0.0}, {"frac": 0.5, "effect": 0.2}]},
        "confident2": {"category": "epistemic_stance", "predict": "cognitive", "range_valid": False,
                       "usable_range": [None, None], "derail_point": None,
                       "pos": "confident pos", "neg": "confident neg",
                       "curve": [{"frac": 0.0, "effect": 0.0}, {"frac": 0.5, "effect": 0.0}]},
    }
    return {"model": "TestModel", "library_path": "fake_lib.json", "dial_order": list(dials), "dials": dials}


def test_dial_mean_effect_averages_nonzero_doses_only():
    d = {"curve": [{"frac": 0.0, "effect": 0.0}, {"frac": 0.5, "effect": 5.0}, {"frac": 1.0, "effect": 8.0}]}
    assert dac._dial_mean_effect(d) == pytest.approx(6.5)


def test_dial_mean_effect_empty_curve_is_zero():
    assert dac._dial_mean_effect({}) == 0.0
    assert dac._dial_mean_effect({"curve": []}) == 0.0


def test_category_summary_counts_and_mean_effect():
    out = dac.category_summary(_fixture_sweep())
    assert out["affect_tone"]["n_usable"] == 1
    assert out["affect_tone"]["n_total"] == 2
    assert out["affect_tone"]["usable_rate"] == 0.5
    assert out["affect_tone"]["mean_effect"] == pytest.approx(3.3)      # mean(6.5, 0.1)
    assert out["epistemic_stance"]["n_usable"] == 0
    assert out["epistemic_stance"]["n_total"] == 2
    assert out["epistemic_stance"]["mean_effect"] == pytest.approx(0.1)  # mean(0.2, 0.0)


def test_category_summary_sorted_by_category_name():
    out = dac.category_summary(_fixture_sweep())
    assert list(out.keys()) == sorted(out.keys())


def test_hypothesis_verdict_rates_and_gap():
    hyp = dac.hypothesis_verdict(_fixture_sweep())
    assert hyp["surface"] == {"n_usable": 1, "n_total": 2, "usable_rate": 0.5}
    assert hyp["cognitive"] == {"n_usable": 0, "n_total": 2, "usable_rate": 0.0}
    assert hyp["gap_surface_minus_cognitive"] == pytest.approx(0.5)
    assert hyp["hypothesis_holds"] is True


def test_hypothesis_verdict_uncertain_bucket_excluded_from_gap():
    sweep = _fixture_sweep()
    sweep["dials"]["wry"] = {"category": "affect_tone", "predict": "uncertain", "range_valid": True,
                             "usable_range": [0.5, 1.0], "derail_point": None, "pos": "p", "neg": "n",
                             "curve": [{"frac": 0.0, "effect": 0.0}, {"frac": 0.5, "effect": 3.0}]}
    sweep["dial_order"].append("wry")
    hyp = dac.hypothesis_verdict(sweep)
    assert hyp["uncertain"] == {"n_usable": 1, "n_total": 1, "usable_rate": 1.0}
    assert hyp["gap_surface_minus_cognitive"] == pytest.approx(0.5)      # unaffected by uncertain


def test_hypothesis_verdict_no_data_for_a_bucket_is_none_not_crash():
    sweep = {"dials": {"only_surface": {"category": "x", "predict": "surface", "range_valid": True}}}
    hyp = dac.hypothesis_verdict(sweep)
    assert hyp["cognitive"]["usable_rate"] is None
    assert hyp["gap_surface_minus_cognitive"] is None
    assert hyp["hypothesis_holds"] is False


def test_curated_library_only_winners_sorted_by_category_then_name():
    curated = dac.curated_library(_fixture_sweep())
    assert curated == [{"name": "warm2", "category": "affect_tone", "usable_range": [0.25, 1.0],
                        "derail_point": None, "pos": "warm pos", "neg": "warm neg"}]


def test_curated_library_empty_when_nothing_valid():
    sweep = _fixture_sweep()
    sweep["dials"]["warm2"]["range_valid"] = False
    assert dac.curated_library(sweep) == []


def test_curated_library_sorts_multiple_winners_by_category_then_name():
    sweep = _fixture_sweep()
    sweep["dials"]["cheerful"]["range_valid"] = True
    sweep["dials"]["cheerful"]["usable_range"] = [0.1, 0.9]
    sweep["dials"]["skeptical2"]["range_valid"] = True
    sweep["dials"]["skeptical2"]["usable_range"] = [0.2, 0.8]
    curated = dac.curated_library(sweep)
    ordered = [(r["category"], r["name"]) for r in curated]
    assert ordered == [("affect_tone", "cheerful"), ("affect_tone", "warm2"),
                       ("epistemic_stance", "skeptical2")]


def test_report_end_to_end_writes_curated_json_and_returns_summary(tmp_path):
    sweep = _fixture_sweep()
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
    curated_out = tmp_path / "curated.json"

    result = dac.report(str(sweep_path), curated_out=str(curated_out))

    assert result["hypothesis"]["gap_surface_minus_cognitive"] == pytest.approx(0.5)
    assert result["category_summary"]["affect_tone"]["n_usable"] == 1
    expected_winner = [{"name": "warm2", "category": "affect_tone", "usable_range": [0.25, 1.0],
                        "derail_point": None, "pos": "warm pos", "neg": "warm neg"}]
    assert result["curated"] == expected_winner

    with open(curated_out, encoding="utf-8") as f:
        written = json.load(f)
    assert written == {"dials": expected_winner}


def test_report_creates_curated_out_parent_dir(tmp_path):
    sweep = _fixture_sweep()
    sweep_path = tmp_path / "sweep.json"
    sweep_path.write_text(json.dumps(sweep), encoding="utf-8")
    curated_out = tmp_path / "nested" / "dir" / "curated.json"

    dac.report(str(sweep_path), curated_out=str(curated_out))
    assert curated_out.is_file()
