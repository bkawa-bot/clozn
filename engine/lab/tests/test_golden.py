"""The oracle (DESIGN invariant 3): replay every golden fixture and assert
picks match exactly, confidences within epsilon.

FakeAdapter cases run everywhere (torch-free, deterministic). DreamAdapter cases
are auto-discovered and gated behind the ``checkpoint`` marker — they validate
the real model path and, later, the C++ core.
"""

from pathlib import Path

import pytest

from cloze_lab.golden import GoldenCase, assert_replay, build_adapter, replay

GOLDEN_DIR = Path(__file__).parent / "golden"
CASES = sorted(GOLDEN_DIR.glob("*.json"))
assert CASES, "no golden fixtures found — run lab/tests/golden/_generate.py"

# Real checkpoints are compared with picks bitwise-exact but confidences only within a
# realistic tolerance: float reduction order differs across transformers versions and
# devices, so asserting near-bitwise (1e-6) confidence equality on a 7B bf16/nf4 model
# would break the moment anyone runs the suite on a different setup — exactly what
# DESIGN invariant 3 warns against. The torch-free FakeAdapter oracle stays strict
# (delta == 0) below; that is the device-independent ground truth.
CHECKPOINT_CONF_EPSILON = 1e-2


def _load(path: Path) -> GoldenCase:
    return GoldenCase.read(path)


def _is_fake(path: Path) -> bool:
    return _load(path).model.adapter == "FakeAdapter"


FAKE_CASES = [p for p in CASES if _is_fake(p)]
# Real-checkpoint cases (Dream 7B, open-dCoder 0.5B): gated, model_id from the spec.
CHECKPOINT_CASES = [p for p in CASES if not _is_fake(p)]


@pytest.mark.parametrize("path", FAKE_CASES, ids=lambda p: p.stem)
def test_fake_golden_replays_exactly(path: Path) -> None:
    case = _load(path)
    # Same machine, torch-free float64 path: confidence delta must be 0, not just
    # within epsilon — proves the fixtures are tight, not loosely passing.
    report = replay(case, build_adapter(case.model))
    assert report.ok, [m.detail for m in report.mismatches]
    assert report.max_conf_delta == 0.0


@pytest.mark.checkpoint
@pytest.mark.parametrize("path", CHECKPOINT_CASES, ids=lambda p: p.stem)
def test_checkpoint_golden_replays(path: Path) -> None:
    from huggingface_hub import try_to_load_from_cache

    case = _load(path)
    model_id = case.model.adapter_args["model_id"]
    if not isinstance(try_to_load_from_cache(model_id, "config.json"), str):
        pytest.skip(f"{model_id} not in local HF cache")
    # Picks must still match exactly (assert_replay checks board/text/picks); only the
    # confidence float comparison is relaxed for real fp models. See CHECKPOINT_CONF_EPSILON.
    assert_replay(case, conf_epsilon=CHECKPOINT_CONF_EPSILON)


def test_every_reason_path_is_covered() -> None:
    reasons = {_load(p).final["reason"] for p in FAKE_CASES}
    assert {"length", "eos", "steps_exhausted"} <= reasons


def test_both_stepper_modes_are_covered() -> None:
    steppers = {(_load(p).stepper or {}).get("name", "fixed") for p in FAKE_CASES}
    assert {"fixed", "adaptive"} <= steppers


def test_adaptive_early_stop_is_pinned() -> None:
    # The adaptive speed win: a fixture where the board drains before t_max.
    case = GoldenCase.read(GOLDEN_DIR / "fake_adaptive_threshold.json")
    assert case.final["reason"] == "length"
    assert case.final["steps_total"] < case.stepper["t_max"]


def test_block_and_whole_sequence_modes_are_covered() -> None:
    block_lens = {_load(p).config.get("block_len", 0) for p in FAKE_CASES}
    assert 0 in block_lens  # whole-sequence
    assert any(bl > 0 for bl in block_lens)  # semi-AR blocks


def test_block_early_eos_skips_later_blocks_is_pinned() -> None:
    # finished_early: EOS in an early block leaves later blocks ungenerated (masked).
    case = GoldenCase.read(GOLDEN_DIR / "fake_blocks_eos_early.json")
    assert case.final["reason"] == "eos"
    mask = case.model.mask_token_id
    assert mask in case.final["board"]  # trailing blocks never generated


def test_cache_off_and_delta_modes_are_covered() -> None:
    modes = {(_load(p).cache or {}).get("mode", "off") for p in FAKE_CASES}
    assert {"off", "delta"} <= modes


def test_delta_cache_diverges_from_off_baseline() -> None:
    # The §5.5 divergence, pinned: a delta run and its exact off twin (same seed,
    # config, policy) reach different boards — the Tier C approximation is real.
    off = GoldenCase.read(GOLDEN_DIR / "fake_cache_off.json")
    delta = GoldenCase.read(GOLDEN_DIR / "fake_cache_delta.json")
    assert off.config == delta.config and off.prompt_ids == delta.prompt_ids
    assert off.cache["mode"] == "off" and delta.cache["mode"] == "delta"
    assert off.final["board"] != delta.final["board"]


def test_quota_ramp_rounding_is_pinned() -> None:
    # Regression guard from the mutation audit: a ceil->floor quota bug is only
    # visible when n_masked is not divisible by steps_remaining. The even-division
    # fixtures cannot see it; fake_quota_uneven (7 masks / 3 steps) exists so the
    # goldens pin the rounding directly — step 0 must commit ceil(7/3) = 3, the
    # value floor (= 2) would get wrong.
    case = GoldenCase.read(GOLDEN_DIR / "fake_quota_uneven.json")
    assert len(case.steps[0].commit) == 3


# Coverage boundary (recorded by the mutation audit): the FakeAdapter oracle uses
# continuous confidences, so it never produces exact ties — confidence-tie
# ordering is therefore pinned by test_policies.py::test_ties_break_toward_lower_pos,
# not by a golden fixture. Every other scheduler/loop mutation tried (rank
# inversion, argmax->argmin, confidence scaling, EOS off-by-one, quota rounding,
# and the Dream logit shift) is caught by a golden.


def test_corrupt_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="format"):
        GoldenCase.from_dict({"format": "bogus/9"})


class TestReplayDetectsDivergence:
    """The oracle must FAIL when behavior changes — guards against silent passing."""

    def test_pick_drift_is_caught(self) -> None:
        case = _load(FAKE_CASES[0])
        tampered = GoldenCase.from_dict(
            {**_as_dict(case), "steps": _bump_first_pick(case)}
        )
        report = replay(tampered, build_adapter(tampered.model))
        assert not report.picks_match

    def test_confidence_drift_is_caught(self) -> None:
        case = _load(FAKE_CASES[0])
        report = replay(case, build_adapter(case.model), conf_epsilon=-1.0)
        assert any(m.kind == "confidence" for m in report.mismatches)


def _as_dict(case: GoldenCase) -> dict:
    import json

    return json.loads(case.to_json())


def _bump_first_pick(case: GoldenCase) -> list[dict]:
    d = _as_dict(case)["steps"]
    d[0]["commit"][0]["id"] += 1  # a token id the fresh run won't reproduce
    return d
