"""test_receipts_as_reward -- model-free tests for research/receipts_as_reward.py (Wild Experiment #8,
research/WILD_WAVE2_PREREG.md).

No model, no GPU, no torch REQUIRED to run this file: receipts_as_reward.py deliberately avoids importing
self_audit_gap / mirror_bench / self_teach_server at module level (see its own docstring), so a bare
`import receipts_as_reward` -- and therefore this whole suite -- never needs torch installed. Every model
touchpoint (`gen_fn` / `gen_fn_sampled`) is a plain Python callable here, standing in for
SelfTeach._generate.

What's under test:
  * render_block: cards substituted verbatim at the {RULES} token; a mutated wording that lost the token
    still gets the cards appended (never silently dropped); SEED_WORDING round-trips byte-identically
    through memory_mode.compile_prompt_block(cards, style="soft") -- the pre-reg's own definition of the
    seed.
  * score_wording: expression/bleed rate arithmetic on a scripted generator (known replies -> known
    rates), lambda's role in fitness, and -- the load-bearing safety property -- that ANY degenerate reply
    (counterfactual._coherence) disqualifies the WHOLE wording to DISQUALIFIED_FITNESS regardless of how
    high its raw keyword/length rates would otherwise score (a wording cannot win by breaking).
  * mutate / _clean_mutation: K calls in, K candidates out; a blank mutation reply falls back to the
    PARENT wording verbatim rather than an empty template.
  * run_generations: fitness-select is a strict (1+K)-elitist argmax (ties keep the parent -- "stays put"
    is the documented default); random-select can and does pick the worst candidate in the pool (proof
    it's not secretly tracking fitness) -- the exact isolation the random-walk null exists for.
  * run_experiment: the two arms' mutation calls are LITERALLY shared (not just same-procedure) whenever
    their current wording coincides -- checked directly by counting calls to the fake sampled generator at
    generation 1, where both arms start from the identical SEED_WORDING.
  * _verdict: the three label branches (receipt wins / evolved~=random-walk / evolved doesn't beat seed).
  * wants_four_bit: the copied logic, plus an opportunistic cross-check against a live mirror_bench import
    (skipped, not failed, when torch/mirror_bench isn't importable here).
  * argparse: defaults parse cleanly (the "argparse check" this experiment's build asked for).
"""
from __future__ import annotations

import os
import random
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import memory_mode                  # noqa: E402  -- stdlib-only, safe (see receipts_as_reward's docstring)
import receipts_as_reward as rar    # noqa: E402


# ================================================================================ hit-rate / render tests

def test_baking_hit_is_a_keyword_substring_match():
    assert rar._baking_hit("I've been baking sourdough bread all weekend.")
    assert not rar._baking_hit("I've been hiking all weekend.")
    assert not rar._baking_hit("")


def test_concise_hit_is_an_absolute_word_count_cutoff():
    assert rar._concise_hit("Yes, that works.", 30)
    assert not rar._concise_hit(" ".join(["word"] * 40), 30)


def test_render_block_substitutes_cards_at_the_rules_token():
    out = rar.render_block("Prefix text.\n{RULES}\nSuffix text.", ["Card one.", "Card two."])
    assert "- Card one." in out and "- Card two." in out
    assert "{RULES}" not in out
    assert out.startswith("Prefix text.\n") and out.endswith("\nSuffix text.")


def test_render_block_appends_cards_when_token_is_missing():
    mangled = "A rephrase that dropped the placeholder entirely."
    out = rar.render_block(mangled, rar.CARDS)
    assert mangled in out
    for c in rar.CARDS:
        assert c in out


def test_seed_wording_round_trips_through_compile_prompt_block():
    """The pre-reg's own definition of the seed: 'the studio's current compiled block.'"""
    assert (rar.render_block(rar.SEED_WORDING, rar.CARDS)
            == memory_mode.compile_prompt_block(rar.CARDS, style="soft"))


# ========================================================================================= score_wording

def test_score_wording_expression_and_bleed_rates():
    on_topic, off_topic = ["p1", "p2"], ["q1", "q2"]
    replies = {
        "p1": "I've been baking bread all day long today.",      # on-topic: baking hit, NOT concise
        "p2": "Short.",                                            # on-topic: concise hit, no baking
        "q1": "A detailed answer about ovens and dough.",          # off-topic: baking bleed, not concise
        "q2": " ".join(f"token{i}" for i in range(40)),           # off-topic: neither hits, stays coherent
    }

    def gen_fn(block, probe):
        return replies[probe]

    res = rar.score_wording(gen_fn, rar.SEED_WORDING, cards=rar.CARDS, on_topic=on_topic,
                            off_topic=off_topic, lam=1.0, concise_max=5)
    assert res["expression_parts"] == {"baking": 0.5, "concise": 0.5}
    assert res["bleed_parts"] == {"baking": 0.5, "concise": 0.0}
    assert res["expression"] == pytest.approx(0.5)
    assert res["bleed"] == pytest.approx(0.25)
    assert res["coherence_ok"] is True
    assert res["fitness"] == pytest.approx(res["expression"] - 1.0 * res["bleed"])


def test_degenerate_reply_disqualifies_regardless_of_raw_expression():
    """The load-bearing safety property: a reply that's a pure keyword-repeat loop would score PERFECT
    expression by the raw keyword rate alone, but counterfactual._coherence flags 3-gram repetition, so
    the wording must be disqualified outright -- it cannot win by breaking."""
    def gen_fn(block, probe):
        return "cake cake cake cake cake"

    res = rar.score_wording(gen_fn, "any wording", cards=rar.CARDS, on_topic=["p1"], off_topic=["q1"],
                            lam=1.0, concise_max=30)
    assert res["coherence_ok"] is False
    assert res["degenerate_reasons"] == ["repeat-3gram"]
    assert res["fitness"] == rar.DISQUALIFIED_FITNESS
    assert res["expression"] > 0                       # the raw numbers are still computed and visible...
    assert res["fitness"] < res["raw_fitness"]          # ...but they never reach the fitness that matters


def test_score_wording_defaults_use_module_probe_sets():
    calls = []

    def gen_fn(block, probe):
        calls.append(probe)
        return "a perfectly ordinary, unrelated reply"

    rar.score_wording(gen_fn, rar.SEED_WORDING)
    assert calls == list(rar.HELDOUT) + list(rar.OFF_TOPIC_PROBES)


# =============================================================================================== mutate

def test_mutate_returns_k_candidates_and_falls_back_to_parent_on_empty():
    seen = []

    def gen_fn_sampled(prompt):
        seen.append(prompt)
        if len(seen) == 2:
            return "   "                                  # blank -> must fall back to the parent
        return f"Rewritten variant {len(seen)}: {{RULES}}"

    kids = rar.mutate(gen_fn_sampled, "PARENT {RULES} TEXT", k=3)
    assert len(kids) == 3 and len(seen) == 3
    assert kids[1] == "PARENT {RULES} TEXT"
    assert all("{RULES}" in kid for kid in kids)


def test_mutate_k_zero_makes_no_calls():
    def gen_fn_sampled(prompt):
        raise AssertionError("should never be called when k=0")

    assert rar.mutate(gen_fn_sampled, "PARENT {RULES}", k=0) == []


def test_clean_mutation_strips_label_and_quotes():
    assert rar._clean_mutation('"Here is the rewritten text."', fallback="X") == "Here is the rewritten text."
    assert rar._clean_mutation("Rewritten instruction: Be nice.", fallback="X") == "Be nice."
    assert rar._clean_mutation("", fallback="FALLBACK") == "FALLBACK"


# ===================================================================================== run_generations

def test_fitness_select_is_elitist_and_keeps_the_best_candidate():
    table = {"seed": 0.1, "child-A": 0.9, "child-B": 0.05}

    def score_fn(w):
        return {"fitness": table.get(w, 0.0), "wording": w}

    def mutate_fn(w, kk):
        return ["child-A", "child-B"][:kk]

    out = rar.run_generations(score_fn, mutate_fn, "seed", generations=1, k=2,
                              select="fitness", rng=random.Random(0))
    assert out["final_wording"] == "child-A"
    assert out["final_score"]["fitness"] == 0.9
    assert out["fitness_by_generation"] == [0.1, 0.9]


def test_fitness_select_stays_put_when_no_child_is_better():
    table = {"seed": 0.9, "child-A": 0.1, "child-B": 0.2}

    def score_fn(w):
        return {"fitness": table.get(w, 0.0), "wording": w}

    def mutate_fn(w, kk):
        return ["child-A", "child-B"][:kk]

    out = rar.run_generations(score_fn, mutate_fn, "seed", generations=3, k=2,
                              select="fitness", rng=random.Random(0))
    assert out["final_wording"] == "seed"                  # every generation: parent beats both children
    assert out["fitness_by_generation"] == [0.9, 0.9, 0.9, 0.9]


def test_random_select_can_pick_the_worst_candidate_in_the_pool():
    """The isolation the null exists for: unlike fitness-select, random-select is NOT secretly tracking
    fitness -- over enough seeds it visits every candidate in the pool, including the worst one."""
    table = {"seed": 0.5, "child-A": 0.9, "child-B": 0.05}

    def score_fn(w):
        return {"fitness": table.get(w, 0.0), "wording": w}

    def mutate_fn(w, kk):
        return ["child-A", "child-B"][:kk]

    picks = {rar.run_generations(score_fn, mutate_fn, "seed", generations=1, k=2,
                                 select="random", rng=random.Random(sd))["final_wording"]
             for sd in range(30)}
    assert picks == {"seed", "child-A", "child-B"}


def test_run_generations_rejects_unknown_select_mode():
    def score_fn(w):
        return {"fitness": 0.0}

    with pytest.raises(ValueError):
        rar.run_generations(score_fn, lambda w, kk: [], "seed", generations=1, k=1,
                            select="bogus", rng=random.Random(0))


# ====================================================================================== run_experiment

def test_run_experiment_shares_mutation_identically_between_arms_at_generation_one():
    sampled_calls = []

    def gen_fn_sampled(prompt):
        sampled_calls.append(prompt)
        return f"mutant-{len(sampled_calls)} {{RULES}}"

    def gen_fn(block, probe):
        return "a perfectly ordinary, unrelated reply"     # coherent; expression/bleed both low either way

    rar.run_experiment(gen_fn, gen_fn_sampled, generations=1, k=2, lam=1.0, seed=0,
                       on_topic=["p1"], off_topic=["q1"])
    # generation 1: BOTH arms start at SEED_WORDING -> the shared mutation cache means mutate() actually
    # ran only ONCE (k=2 calls total), not once per arm (which would be 4) -- "mutate identically", literal.
    assert len(sampled_calls) == 2


def test_run_experiment_returns_a_verdict_and_full_trace():
    def gen_fn(block, probe):
        return "baking bread" if "p" in probe else "a long and thorough unrelated answer indeed here"

    def gen_fn_sampled(prompt):
        return "A differently-phrased instruction. {RULES}"

    out = rar.run_experiment(gen_fn, gen_fn_sampled, generations=2, k=2, lam=1.0, seed=0,
                             on_topic=["p1"], off_topic=["q1"])
    assert set(out) >= {"evolved", "random_walk", "seed_baseline", "verdict"}
    assert len(out["evolved"]["fitness_by_generation"]) == 3        # generation 0..2
    assert len(out["random_walk"]["fitness_by_generation"]) == 3
    assert out["verdict"]["seed_fitness"] == out["seed_baseline"]["fitness"]


# ============================================================================================= _verdict

@pytest.mark.parametrize("sf,ef,rf,expect_substr", [
    (0.10, 0.50, 0.10, "wins"),
    (0.10, 0.15, 0.14, "did not help"),
    (0.50, 0.20, 0.10, "did not beat the seed"),
])
def test_verdict_labels(sf, ef, rf, expect_substr):
    v = rar._verdict({"fitness": sf}, {"final_score": {"fitness": ef}}, {"final_score": {"fitness": rf}})
    assert expect_substr in v["label"]


def test_verdict_flags_a_disqualified_seed_loudly():
    v = rar._verdict({"fitness": rar.DISQUALIFIED_FITNESS}, {"final_score": {"fitness": 0.1}},
                     {"final_score": {"fitness": 0.0}})
    assert v["seed_disqualified"] is True


# =========================================================================== wants_four_bit / argparse

@pytest.mark.parametrize("name,override,expect", [
    ("Qwen/Qwen2.5-7B-Instruct", "auto", True),
    ("Qwen/Qwen2.5-1.5B-Instruct", "auto", False),
    ("Qwen/Qwen2.5-1.5B-Instruct", "yes", True),
    ("Qwen/Qwen2.5-7B-Instruct", "no", False),
])
def test_wants_four_bit(name, override, expect):
    assert rar.wants_four_bit(name, override) is expect


def test_wants_four_bit_matches_mirror_bench_when_torch_is_available():
    try:
        import mirror_bench as mb
    except Exception:
        pytest.skip("mirror_bench (and therefore torch) is not importable in this environment")
    names = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "google/gemma-2-9b-it", "x-3b-model"]
    for name in names:
        for override in ("auto", "yes", "no"):
            assert rar.wants_four_bit(name, override) == mb.wants_four_bit(name, override)


def test_copied_constants_match_self_audit_gap_when_torch_is_available():
    try:
        import self_audit_gap as gap
    except Exception:
        pytest.skip("self_audit_gap (and therefore torch) is not importable in this environment")
    assert rar.HELDOUT == gap.HELDOUT
    baking = next(t for t in gap.TRAITS if t["name"] == "baking")
    concise = next(t for t in gap.TRAITS if t["name"] == "concise")
    assert rar.CARDS == [baking["rule"], concise["rule"]]
    assert rar._BAKING_KW == baking["kw"]


def test_argparse_defaults():
    args = rar._build_parser().parse_args([])
    assert args.model == "Qwen/Qwen2.5-7B-Instruct"
    assert args.smoke is False
    assert args.generations == 6 and args.k == 4
    assert args.lam == rar.LAMBDA_DEFAULT

    smoke_args = rar._build_parser().parse_args(["--smoke", "--lambda", "0.5"])
    assert smoke_args.smoke is True
    assert smoke_args.lam == 0.5
