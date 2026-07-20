"""Fixture tests for clozn.analysis.tracer's pure math (no engine, no network).

The engine-facing orchestration (trace()) is exercised live (it needs a running cloze-server with
a J-lens sidecar); everything below is the model-free seam: ablation algebra, candidate screening,
the noise floor / verdict rules, the unexplained-mass accounting, and joint-write grouping.
"""
import numpy as np
import pytest

from clozn.analysis.tracer import (accounting, controls_verdict, directional_ablate,
                                   group_joint_writes, noise_floor, screen_candidates)


# ------------------------------------------------------------------------- directional_ablate

def test_directional_ablate_removes_exactly_the_projection():
    rng = np.random.default_rng(0)
    h = rng.standard_normal(64).astype(np.float32)
    d = rng.standard_normal(64).astype(np.float32)
    out = directional_ablate(h, d)
    d_hat = d / np.linalg.norm(d)
    assert abs(float(out @ d_hat)) < 1e-4            # the component along d is gone
    resid = h - out                                   # and what was removed is parallel to d
    cos = float(resid @ d_hat) / (np.linalg.norm(resid) + 1e-12)
    assert abs(abs(cos) - 1.0) < 1e-4


def test_directional_ablate_is_scale_invariant_in_d():
    rng = np.random.default_rng(1)
    h = rng.standard_normal(16).astype(np.float32)
    d = rng.standard_normal(16).astype(np.float32)
    a = directional_ablate(h, d)
    b = directional_ablate(h, 7.5 * d)               # only the direction matters
    assert np.allclose(a, b, atol=1e-5)


def test_directional_ablate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        directional_ablate(np.ones(4), np.ones(5))   # shape mismatch
    with pytest.raises(ValueError):
        directional_ablate(np.ones(4), np.zeros(4))  # degenerate direction


# -------------------------------------------------------------------------- screen_candidates

def _screen_fixture():
    d = 8
    H = np.zeros((5, d), dtype=np.float32)
    dirs = {"target": np.eye(d, dtype=np.float32)[0], "other": np.eye(d, dtype=np.float32)[1]}
    H[3, 0] = 10.0    # planted: strong "target" alignment at pos 3
    H[1, 1] = 4.0     # weaker "other" alignment at pos 1
    return {16: H}, {16: dirs}


def test_screen_ranks_planted_site_first_and_labels_it():
    H, dirs = _screen_fixture()
    out = screen_candidates(H, dirs, max_candidates=3, force_sites=[])
    assert (out[0]["layer"], out[0]["pos"], out[0]["concept"]) == (16, 3, "target")
    assert (out[1]["layer"], out[1]["pos"], out[1]["concept"]) == (16, 1, "other")


def test_screen_force_sites_come_first_and_dedupe():
    H, dirs = _screen_fixture()
    out = screen_candidates(H, dirs, max_candidates=4, force_sites=[(16, 4), (16, 4), (16, 3)])
    assert (out[0]["layer"], out[0]["pos"]) == (16, 4)
    assert (out[1]["layer"], out[1]["pos"]) == (16, 3)   # forced AND planted: appears once
    assert len([c for c in out if (c["layer"], c["pos"]) == (16, 3)]) == 1
    assert len(out) <= 4


def test_screen_respects_cap():
    H, dirs = _screen_fixture()
    out = screen_candidates(H, dirs, max_candidates=2, force_sites=[])
    assert len(out) == 2


# ---------------------------------------------------------------- noise floor / verdict rules

def test_noise_floor_is_mult_times_median():
    assert noise_floor([0.1, -0.2, 0.3], mult=3.0) == pytest.approx(0.6)
    with pytest.raises(ValueError):
        noise_floor([])


def test_verdict_pass_when_real_beats_controls():
    assert controls_verdict([2.0, -1.5], [0.01, -0.02, 0.015]) == "PASS"


def test_verdict_no_causal_nodes_when_nothing_survived():
    assert controls_verdict([], [0.01, -0.02]) == "NO_CAUSAL_NODES"


def test_verdict_failed_when_controls_match_real():
    # the strongest control equals the strongest "real" effect -> nothing here is trustworthy
    assert controls_verdict([0.5], [0.6, 0.01]) == "FAILED_CONTROLS"
    with pytest.raises(ValueError):
        controls_verdict([1.0], [])


# ------------------------------------------------------------------------------- accounting

def test_accounting_interaction_gap_signs():
    sub = accounting([2.0, 3.0], delta_total=4.0)     # joint < sum: sub-additive (self-repair)
    assert sub["interaction_gap"] == pytest.approx(-1.0)
    sup = accounting([1.0, 1.0], delta_total=3.0)     # joint > sum: super-additive
    assert sup["interaction_gap"] == pytest.approx(1.0)
    empty = accounting([], delta_total=0.5)
    assert empty["sum_solo"] == 0.0


# ------------------------------------------------------------------------ group_joint_writes

def test_group_joint_writes_one_spec_per_layer_with_stacked_mean_rows():
    nodes = [{"layer": 16, "pos": 3}, {"layer": 16, "pos": 1}, {"layer": 24, "pos": 4}]
    mean_rows = {16: np.full(4, 2.0, dtype=np.float32), 24: np.full(4, 5.0, dtype=np.float32)}
    specs = group_joint_writes(nodes, mean_rows)
    assert [s["layer"] for s in specs] == [16, 24]
    s16 = specs[0]
    assert sorted(s16["positions"]) == [1, 3]
    assert len(s16["values"]) == 2 * 4                # one mean row per position, concatenated
    assert all(v == 2.0 for v in s16["values"])
    assert specs[1]["positions"] == [4] and len(specs[1]["values"]) == 4
