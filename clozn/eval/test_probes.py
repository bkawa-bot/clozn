"""Tests for the probe sets (eval.probes) -- the programmatic arithmetic generator, framework-free."""
from __future__ import annotations

import re

from clozn.eval import probes


def test_arithmetic_golds_are_all_correct():
    for p in probes.arithmetic_probes(60):
        a, op, b = re.match(r"What is (\d+) (.) (\d+)", p["q"]).groups()
        a, b = int(a), int(b)
        val = a + b if op == "+" else (a - b if op == "-" else a * b)
        assert p["gold"] == str(val) and p["kind"] == "numeric"


def test_arithmetic_is_deterministic_and_spread_across_tiers():
    a = [p["q"] for p in probes.arithmetic_probes(60, seed=7)]
    b = [p["q"] for p in probes.arithmetic_probes(60, seed=7)]
    assert a == b                                                  # seeded -> reproducible
    tiers = {p["tier"] for p in probes.arithmetic_probes(60)}
    assert {"add_2d", "mul_2x2", "mul_3x3"} <= tiers              # covers easy..hard


def test_arith_probes_constant_is_populated():
    assert len(probes.ARITH_PROBES) == 60 and probes.ARITH_PROBES[0]["kind"] == "numeric"
