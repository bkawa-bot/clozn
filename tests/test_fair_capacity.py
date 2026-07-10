"""test_fair_capacity -- the FAIRNESS logic that keeps multi-trait memory honest (no torch, no model, no GPU).

The bug this guards against: SelfTeach.consolidate() built a single soft prefix and, when a SECOND trait was
approved next to an entrenched first, WARM-STARTED from the existing prefix and trained only briefly -- so the
established trait dominated and the new one never registered (concretely: with "baking" trained, approving
"dogs" produced zero dog expression). The clean fix: when the ACTIVE rule SET changes (add/remove), REINIT the
prefix and train from scratch on the full active set so every trait starts on equal footing; warm-start ONLY on
the identical set (a strength/steps tweak).

The set-change decision + the fair step budget are factored into two PURE module-level helpers so they can be
unit-tested in isolation, without loading a 7B or touching CUDA:
  * rule_set_changed(trained_on, incoming) -> did the active set change vs the set the prefix embodies?
  * fair_steps(base_steps, n_rules)        -> a modest step budget that scales with trait count.

Importing self_teach_server does NOT load a model (the backbone loads only in SelfTeach.__init__), so this test
stays fast even though the module also imports torch/transformers.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.substrates.self_teach import fair_steps, rule_set_changed   # noqa: E402  (module-level, no model load)


# ---- rule_set_changed: the reinit trigger ----------------------------------------------------------------

def test_first_ever_consolidation_is_a_change():
    # no prefix yet (trained_on == []) -> the very first trait is a change -> fresh init (correct).
    assert rule_set_changed([], ["likes baking"]) is True
    assert rule_set_changed(None, ["likes baking"]) is True


def test_identical_set_is_not_a_change_warm_start():
    # exact same single trait -> warm-start (e.g. a strength/steps tweak), the prefix keeps refining.
    assert rule_set_changed(["likes baking"], ["likes baking"]) is False


def test_adding_a_second_trait_is_a_change():
    # THE bug: approving "dogs" next to trained "baking" MUST reinit so dogs gets fair training.
    assert rule_set_changed(["likes baking"], ["likes baking", "loves dogs"]) is True


def test_removing_a_trait_is_a_change():
    # removing a trait must also reinit -- otherwise a warm-started prefix keeps the removed trait's residue.
    assert rule_set_changed(["likes baking", "loves dogs"], ["likes baking"]) is True


def test_swapping_a_trait_is_a_change():
    assert rule_set_changed(["likes baking"], ["loves dogs"]) is True


def test_order_does_not_matter():
    # the SET is what counts, not the sequence: reordering the same traits is NOT a change (no needless retrain).
    assert rule_set_changed(["a", "b", "c"], ["c", "b", "a"]) is False


def test_duplicates_do_not_matter():
    # duplicate-insensitive: {a} == {a, a}. A dup slipping into the list must not force a spurious reinit.
    assert rule_set_changed(["a"], ["a", "a"]) is False
    assert rule_set_changed(["a", "a", "b"], ["b", "a"]) is False


def test_both_empty_is_not_a_change():
    assert rule_set_changed([], []) is False
    assert rule_set_changed(None, None) is False


def test_multi_trait_identical_set_warm_starts():
    # a rerun on the identical multi-trait set (e.g. dial tweak) warm-starts -- no reinit, no wasted retrain.
    s = ["likes baking", "loves dogs", "prefers bullet points"]
    assert rule_set_changed(s, list(s)) is False


# ---- fair_steps: N traits each get a fair budget when retraining from scratch ------------------------------

def test_single_trait_keeps_base_budget():
    assert fair_steps(120, 1) == 120


def test_budget_grows_with_trait_count():
    # +50% of base per extra trait so a single prefix can fit each trait's opening -> each surfaces.
    assert fair_steps(120, 2) == 180
    assert fair_steps(120, 3) == 240
    # strictly increasing across 1..3 traits (the common studio range).
    assert fair_steps(120, 1) < fair_steps(120, 2) < fair_steps(120, 3)


def test_budget_is_capped_so_consolidation_stays_quick():
    # bounded at 3x base -> many traits can't make a single consolidation crawl.
    assert fair_steps(120, 10) == 360
    assert fair_steps(120, 100) == 360
    assert fair_steps(120, 1000) <= 120 * 3


def test_degenerate_trait_counts_are_safe():
    # zero / negative trait counts must not underflow the budget (treated as at least 1 trait).
    assert fair_steps(120, 0) == 120
    assert fair_steps(120, -5) == 120


def test_budget_scales_with_the_base():
    # the heuristic is relative to whatever base the caller passes (e.g. a faster 90-step consolidate).
    assert fair_steps(90, 1) == 90
    assert fair_steps(90, 3) == 180        # round(90 * (1 + 0.5*2)) == 180
    assert fair_steps(90, 50) == 270       # capped at 3x
