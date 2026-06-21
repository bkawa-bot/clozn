"""Regenerate the golden fixtures in this directory.

    python lab/tests/golden/_generate.py              # FakeAdapter cases (no torch/checkpoint)
    python lab/tests/golden/_generate.py --with-dream # also the Dream 7B case

Runs as a plain script (imports only the installed cloze_lab package, resolves
paths via __file__). Deterministic: same code -> byte-identical JSON, so
regeneration is a clean diff. The leading underscore keeps pytest from
collecting this as a test module.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cloze_lab.generate import GenerateConfig
from cloze_lab.golden import GoldenCase, GoldenModel, record
from cloze_lab.models.fake import FakeAdapter

HERE = Path(__file__).parent
FAKE_VOCAB = 64
FAKE_KV_DIM = 8


def fake_model(seed: int) -> tuple[FakeAdapter, GoldenModel]:
    adapter = FakeAdapter(seed=seed, vocab_size=FAKE_VOCAB, kv_dim=FAKE_KV_DIM)
    cfg = adapter.config
    spec = GoldenModel(
        adapter="FakeAdapter",
        family=cfg.family.value,
        vocab_size=cfg.vocab_size,
        mask_token_id=cfg.mask_token_id,
        eos_token_id=cfg.eos_token_id,
        adapter_args={"seed": seed, "vocab_size": FAKE_VOCAB, "kv_dim": FAKE_KV_DIM},
    )
    return adapter, spec


def _find_eos_seed(prompt: str, config: GenerateConfig) -> int:
    """First fake seed whose greedy run commits EOS mid-board (reason='eos')."""
    from cloze_lab.generate import generate
    from cloze_lab.scheduler.events import GenFinished

    for seed in range(1000):
        adapter, _ = fake_model(seed)
        result = generate(adapter, adapter.encode(prompt), config)
        finished = result.events[-1]
        assert isinstance(finished, GenFinished)
        if finished.reason == "eos":
            return seed
    raise RuntimeError("no EOS-producing seed found in range")


def fake_cases() -> list[GoldenCase]:
    cases: list[GoldenCase] = []

    # 1. quota mode (k=None), greedy: even ramp drains the board -> reason "length".
    adapter, spec = fake_model(7)
    cases.append(
        record("fake_quota_greedy", adapter, spec, "hello cloze",
               GenerateConfig(max_new=8, steps=4), policy_spec={"name": "confidence_topk", "k": None})
    )

    # 2. fixed k=2, greedy: pins a constant per-step commit width.
    adapter, spec = fake_model(7)
    cases.append(
        record("fake_fixedk2_greedy", adapter, spec, "hello cloze",
               GenerateConfig(max_new=6, steps=3), policy_spec={"name": "confidence_topk", "k": 2})
    )

    # 3. quota mode, sampled (temperature>0): exercises the rng confidence path.
    adapter, spec = fake_model(3)
    cases.append(
        record("fake_quota_sampled", adapter, spec, "diffusion",
               GenerateConfig(max_new=6, steps=3, temperature=0.8, seed=11),
               policy_spec={"name": "confidence_topk", "k": None})
    )

    # 3b. quota with UNEVEN division (7 masks / 3 steps): ceil(7/3)=3 != floor=2,
    # so this fixture pins the quota ramp's rounding — a ceil->floor regression
    # diverges at step 0. (The even-division fixtures above cannot see it.)
    adapter, spec = fake_model(7)
    cases.append(
        record("fake_quota_uneven", adapter, spec, "hello cloze",
               GenerateConfig(max_new=7, steps=3), policy_spec={"name": "confidence_topk", "k": None})
    )

    # 4. EOS mid-board -> truncation, reason "eos".
    eos_config = GenerateConfig(max_new=10, steps=5)
    eos_seed = _find_eos_seed("fill in the blank", eos_config)
    adapter, spec = fake_model(eos_seed)
    cases.append(
        record("fake_eos_truncation", adapter, spec, "fill in the blank",
               eos_config, policy_spec={"name": "confidence_topk", "k": None})
    )

    # 5. steps exhausted: k=1 over too few steps leaves visible holes.
    adapter, spec = fake_model(7)
    cases.append(
        record("fake_steps_exhausted", adapter, spec, "hi",
               GenerateConfig(max_new=5, steps=2), policy_spec={"name": "confidence_topk", "k": 1})
    )

    # --- adaptive stepping (step 7): threshold policy + AdaptiveStepper ---
    # (config.steps is unused under adaptive; set to t_max for tidiness.)

    # 6. threshold early-stop: slots clearing tau drain the board well before
    # t_max, so steps_total < t_max — the adaptive speed win.
    adapter, spec = fake_model(7)
    cases.append(
        record("fake_adaptive_threshold", adapter, spec, "hello cloze",
               GenerateConfig(max_new=8, steps=12),
               policy_spec={"name": "threshold", "tau": 0.10},
               stepper_spec={"name": "adaptive", "t_max": 12})
    )

    # 7. min-one-commit rail drains a board nothing can clear: tau=0.99 is never
    # met, so the rail forces the top-1 each pass; with t_max ample, the board
    # still drains one slot per pass (reason length, steps_total == max_new).
    adapter, spec = fake_model(7)
    cases.append(
        record("fake_adaptive_min_commit", adapter, spec, "hi",
               GenerateConfig(max_new=5, steps=8),
               policy_spec={"name": "threshold", "tau": 0.99},
               stepper_spec={"name": "adaptive", "t_max": 8})
    )

    # 8. T_max rail caps a non-draining run: tau=0.99 + min-one gives 1 commit
    # per pass, but t_max=3 < max_new=8, so 5 holes remain (reason steps_exhausted).
    adapter, spec = fake_model(7)
    cases.append(
        record("fake_adaptive_tmax_holes", adapter, spec, "hello cloze",
               GenerateConfig(max_new=8, steps=3),
               policy_spec={"name": "threshold", "tau": 0.99},
               stepper_spec={"name": "adaptive", "t_max": 3})
    )

    # --- block diffusion (step 8): semi-AR blocks, block-causal attention ---

    # 9. two full blocks, greedy quota per block (steps is the per-block budget).
    adapter, spec = fake_model(7)
    cases.append(
        record("fake_blocks_greedy", adapter, spec, "hello cloze",
               GenerateConfig(max_new=8, steps=4, block_len=4),
               policy_spec={"name": "confidence_topk", "k": None})
    )

    # 10. partial last block: max_new=8 over block_len=3 => blocks of 3, 3, 2.
    adapter, spec = fake_model(7)
    cases.append(
        record("fake_blocks_partial", adapter, spec, "hello cloze",
               GenerateConfig(max_new=8, steps=3, block_len=3),
               policy_spec={"name": "confidence_topk", "k": None})
    )

    # 11. EOS in an early block finishes the run: later blocks are never generated
    # (finished_early), so trailing positions stay masked but reason is eos.
    block_eos_config = GenerateConfig(max_new=12, steps=4, block_len=3)
    eos_seed = _find_eos_seed("fill in the blank", block_eos_config)
    adapter, spec = fake_model(eos_seed)
    cases.append(
        record("fake_blocks_eos_early", adapter, spec, "fill in the blank",
               block_eos_config, policy_spec={"name": "confidence_topk", "k": None})
    )

    # --- cache tiers (step 9): Tier C delta drift, pinned against the off baseline ---

    # 12. delta cache (whole-sequence): reuse between periodic full refreshes — an
    # APPROXIMATE run that diverges from the exact cache=off golden of the same config.
    adapter, spec = fake_model(7)
    cases.append(
        record("fake_cache_delta", adapter, spec, "hello cloze",
               GenerateConfig(max_new=12, steps=8),
               policy_spec={"name": "confidence_topk", "k": None},
               cache_spec={"mode": "delta", "full_refresh_every": 4, "refresh_fraction": 0.5})
    )

    # 13. exact baseline for the divergence pair above (cache=off, same config/seed).
    adapter, spec = fake_model(7)
    cases.append(
        record("fake_cache_off", adapter, spec, "hello cloze",
               GenerateConfig(max_new=12, steps=8),
               policy_spec={"name": "confidence_topk", "k": None},
               cache_spec={"mode": "off"})
    )

    # 14. block-mode delta: Tier B free freeze (frozen blocks reused, never
    # recomputed) with Tier C drift only inside the active block.
    adapter, spec = fake_model(7)
    cases.append(
        record("fake_cache_delta_blocks", adapter, spec, "hello cloze",
               GenerateConfig(max_new=12, steps=6, block_len=4),
               policy_spec={"name": "confidence_topk", "k": None},
               cache_spec={"mode": "delta", "full_refresh_every": 4, "refresh_fraction": 0.5})
    )

    return cases


def dream_cases() -> list[GoldenCase]:
    from cloze_lab.models.base import LoadConfig
    from cloze_lab.models.dream import DREAM_7B_INSTRUCT, DreamAdapter

    args = {"model_id": DREAM_7B_INSTRUCT, "device": "cpu", "dtype": "bfloat16"}
    adapter = DreamAdapter(LoadConfig(**args))
    cfg = adapter.config
    spec = GoldenModel(
        adapter="DreamAdapter",
        family=cfg.family.value,
        vocab_size=cfg.vocab_size,
        mask_token_id=cfg.mask_token_id,
        eos_token_id=cfg.eos_token_id,
        adapter_args=args,
    )
    prompt = "The capital of France is"
    return [
        # whole-sequence (fully bidirectional)
        record("dream_paris", adapter, spec, prompt,
               GenerateConfig(max_new=8, steps=4), policy_spec={"name": "confidence_topk", "k": None}),
        # semi-AR block mode (block-causal attention) on the real model
        record("dream_paris_blocks", adapter, spec, prompt,
               GenerateConfig(max_new=8, steps=4, block_len=4),
               policy_spec={"name": "confidence_topk", "k": None}),
    ]


def dcoder_cases() -> list[GoldenCase]:
    """open-dCoder 0.5B — the tiny Dream-family checkpoint CI actually runs (DESIGN).

    Same shifted-head loop as Dream, on a model small enough to download in CI. A code
    prompt the model completes cleanly: "return a +" -> " b". Whole-sequence and
    block-causal modes, so CI exercises both attention regimes on a real checkpoint.
    """
    from cloze_lab.models.base import LoadConfig
    from cloze_lab.models.dream import OPEN_DCODER_05B, open_dcoder_adapter

    args = {"model_id": OPEN_DCODER_05B, "device": "cpu", "dtype": "bfloat16"}
    adapter = open_dcoder_adapter(LoadConfig(**args))
    cfg = adapter.config
    spec = GoldenModel(
        adapter="OpenDCoderAdapter",
        family=cfg.family.value,
        vocab_size=cfg.vocab_size,
        mask_token_id=cfg.mask_token_id,
        eos_token_id=cfg.eos_token_id,
        adapter_args=args,
    )
    prompt = "def add(a, b):\n    return a +"
    return [
        # whole-sequence (fully bidirectional)
        record("dcoder_add", adapter, spec, prompt,
               GenerateConfig(max_new=8, steps=4), policy_spec={"name": "confidence_topk", "k": None}),
        # semi-AR block mode (block-causal attention) on the real model
        record("dcoder_add_blocks", adapter, spec, prompt,
               GenerateConfig(max_new=8, steps=4, block_len=4),
               policy_spec={"name": "confidence_topk", "k": None}),
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-dream", action="store_true", help="also regenerate the Dream 7B case")
    parser.add_argument("--with-dcoder", action="store_true", help="also regenerate the open-dCoder 0.5B (CI) case")
    args = parser.parse_args(argv)

    cases = fake_cases()
    if args.with_dream:
        cases.extend(dream_cases())
    if args.with_dcoder:
        cases.extend(dcoder_cases())

    for case in cases:
        path = HERE / f"{case.name}.json"
        case.write(path)
        print(f"wrote {path.relative_to(HERE.parent.parent)}  ({len(case.steps)} steps, reason={case.final['reason']})")


if __name__ == "__main__":
    main()
