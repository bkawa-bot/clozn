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


# --- EXTENDED SET (v2) -- structural checks only (many golds here are riddles/opinions-of-fact that can't
# be re-derived programmatically the way arithmetic can); we check shape, not semantics.

_ALL_SETS = {
    "PROBES": probes.PROBES, "HARD_PROBES": probes.HARD_PROBES,
    "FACTUAL_PROBES": probes.FACTUAL_PROBES, "REASONING_PROBES": probes.REASONING_PROBES,
    "MISCONCEPTION_PROBES": probes.MISCONCEPTION_PROBES, "TRICK_PROBES": probes.TRICK_PROBES,
}


def test_extended_probes_is_the_sum_of_its_four_categories():
    assert probes.EXTENDED_PROBES == (probes.FACTUAL_PROBES + probes.REASONING_PROBES
                                       + probes.MISCONCEPTION_PROBES + probes.TRICK_PROBES)


def test_extended_probes_every_item_has_required_fields():
    for name, pset in _ALL_SETS.items():
        for p in pset:
            assert p.get("q", "").strip(), f"{name}: empty question"
            assert p.get("gold", "").strip() if isinstance(p.get("gold"), str) else p.get("gold") is not None, \
                f"{name}: empty gold for {p.get('q')!r}"
            assert p.get("kind") in ("exact", "numeric", "mcq"), f"{name}: bad kind for {p.get('q')!r}"


def test_extended_probes_have_category_and_difficulty_tags():
    for name in ("FACTUAL_PROBES", "REASONING_PROBES", "MISCONCEPTION_PROBES", "TRICK_PROBES"):
        for p in _ALL_SETS[name]:
            assert p.get("category"), f"{name}: missing category for {p['q']!r}"
            assert p.get("difficulty") in ("easy", "medium", "hard"), \
                f"{name}: missing/bad difficulty for {p['q']!r}"


def test_extended_probes_span_a_difficulty_mix():
    diffs = {p["difficulty"] for p in probes.EXTENDED_PROBES}
    assert {"easy", "medium", "hard"} <= diffs


def test_extended_probes_no_duplicate_questions_within_the_curated_corpus():
    seen: dict[str, str] = {}
    for name, pset in _ALL_SETS.items():
        for p in pset:
            q = p["q"].strip()
            assert q not in seen, f"duplicate question in {name} (first seen in {seen.get(q)}): {q!r}"
            seen[q] = name


def test_extended_probes_bring_the_curated_total_into_the_100_to_150_range():
    total = len(probes.PROBES) + len(probes.HARD_PROBES) + len(probes.EXTENDED_PROBES)
    assert 100 <= total <= 150, f"curated total {total} outside the ~100-150 target"


def test_eval_cli_accepts_extended_set():
    from clozn.cli.main import build_parser
    ns = build_parser().parse_args(["eval", "--set", "extended"])
    assert ns.which == "extended"

