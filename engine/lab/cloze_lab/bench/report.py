"""Benchmark report (DESIGN.md §8): run an off/delta A/B and emit a markdown table.

Each speed row carries its divergence column, so a speed number can never be read
without the quality it cost (invariant 5). The first variant is the exact baseline
(cache=off); the rest are compared against it.

    python -m cloze_lab.bench.report                          # fake adapter, off vs delta
    python -m cloze_lab.bench.report --model dcoder --block-len 4  # tiny real model, CPU
    python -m cloze_lab.bench.report --model dream --device cuda \
        --quant nf4 --block-len 4 --out-dir cloze_lab/bench/results   # 7B on a 16GB card

With ``--block-len > 0`` the real Dream-family models add an exact delta row (frozen
prefix reused under block-causal attention), so the A/B shows the KV-reuse speedup
against its own exact baseline. ``--out-dir`` archives the markdown table and a
replayable JSONL event log per variant (invariant 2).
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cloze_lab.bench.divergence import DivergenceStats, divergence
from cloze_lab.bench.speed import SpeedStats, speed_stats
from cloze_lab.generate import GenerateConfig, GenerateResult, generate
from cloze_lab.models.base import ModelAdapter
from cloze_lab.scheduler.cache import CacheConfig
from cloze_lab.scheduler.events import write_jsonl


@dataclass(frozen=True, slots=True)
class BenchRow:
    label: str  # how this run's cache was configured
    speed: SpeedStats
    divergence: DivergenceStats | None  # None for the exact baseline


def _cache_label(c: CacheConfig) -> str:
    if c.mode == "off":
        return "off (exact)"
    return f"delta(refresh={c.full_refresh_every})"


def _slug_label(c: CacheConfig) -> str:
    """Filename-safe variant tag for archived JSONL logs."""
    return "off" if c.mode == "off" else f"delta_refresh{c.full_refresh_every}"


def bench_runs(
    adapter: ModelAdapter,
    prompt_ids: Sequence[int],
    config: GenerateConfig,
    caches: Sequence[CacheConfig],
    **gen_kwargs,
) -> tuple[list[GenerateResult], list[BenchRow]]:
    """Run each cache config (first = baseline) and build rows; also return the raw
    runs so callers can archive their event streams (JSONL, invariant 2)."""
    if not caches:
        raise ValueError("need at least one cache config (the baseline)")
    runs: list[GenerateResult] = [
        generate(adapter, prompt_ids, config, cache=c, **gen_kwargs) for c in caches
    ]
    baseline = runs[0]
    rows = [BenchRow(label=_cache_label(caches[0]), speed=speed_stats(baseline), divergence=None)]
    for cache, run in zip(caches[1:], runs[1:]):
        rows.append(
            BenchRow(
                label=_cache_label(cache),
                speed=speed_stats(run),
                divergence=divergence(baseline, run),
            )
        )
    return runs, rows


def run_ab(
    adapter: ModelAdapter,
    prompt_ids: Sequence[int],
    config: GenerateConfig,
    caches: Sequence[CacheConfig],
    **gen_kwargs,
) -> list[BenchRow]:
    """Run each cache config; the first is the baseline every other is compared to."""
    return bench_runs(adapter, prompt_ids, config, caches, **gen_kwargs)[1]


def markdown_table(rows: Sequence[BenchRow], *, title: str = "") -> str:
    header = (
        "| cache | forwards | mean cache-hit | new tok | steps/tok | tok/s "
        "| token-match | text-match | mean-conf delta |"
    )
    sep = "|" + "|".join(["---"] * 9) + "|"
    lines = [f"**{title}**", "", header, sep] if title else [header, sep]
    for r in rows:
        s = r.speed
        if r.divergence is None:
            match, text, dconf = "baseline", "n/a", "n/a"
        else:
            d = r.divergence
            match = f"{d.token_match * 100:.1f}%"
            text = "yes" if d.text_match else "no"
            dconf = f"{d.mean_conf_delta:+.3f}"
        lines.append(
            f"| {r.label} | {s.forwards} | {s.mean_cache_hit * 100:.0f}% | {s.new_tokens} "
            f"| {s.steps_per_token:.2f} | {s.tok_per_s:.1f} | {match} | {text} | {dconf} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="cloze-bench", description="Off/delta cache A/B benchmark.")
    parser.add_argument("--model", choices=("fake", "dream", "dcoder"), default="fake")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new", type=int, default=16)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--block-len", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--quant", choices=("none", "nf4", "int8"), default="none",
        help="bitsandbytes weight quant for Dream (GPU only) — fits the 7B on a 16GB card",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="archive the markdown table + a replayable JSONL log per variant here",
    )
    parser.add_argument(
        "--chat", action=argparse.BooleanOptionalAction, default=None,
        help="apply the instruct chat template before tokenizing (default: on for dream)",
    )
    args = parser.parse_args(argv)

    adapter: ModelAdapter
    if args.model == "fake":
        from cloze_lab.models.fake import FakeAdapter

        adapter = FakeAdapter(seed=args.seed)
        caches = [
            CacheConfig(mode="off"),
            CacheConfig(mode="delta", full_refresh_every=4),
            CacheConfig(mode="delta", full_refresh_every=1),
        ]
    else:
        from cloze_lab.models.base import LoadConfig

        if args.model == "dream":
            from cloze_lab.models.dream import DREAM_7B_INSTRUCT, DreamAdapter

            quant = None if args.quant == "none" else args.quant
            adapter = DreamAdapter(
                LoadConfig(model_id=DREAM_7B_INSTRUCT, device=args.device, dtype="bfloat16"),
                quantization=quant,
            )
        else:  # dcoder — the tiny Dream-family CI checkpoint, runs on CPU
            from cloze_lab.models.dream import OPEN_DCODER_05B, open_dcoder_adapter

            adapter = open_dcoder_adapter(
                LoadConfig(model_id=OPEN_DCODER_05B, device=args.device, dtype="bfloat16")
            )
        # Dream-family reuses the frozen prefix exactly under block-causal attention;
        # full_refresh_every=1 feeds it the contiguous-suffix recompute it supports, so
        # the delta row is the KV-reuse speedup at 100% token-match (no quality cost).
        caches = [CacheConfig(mode="off")]
        if args.block_len > 0:
            caches.append(CacheConfig(mode="delta", full_refresh_every=1))

    config = GenerateConfig(
        max_new=args.max_new, steps=args.steps, seed=args.seed, block_len=args.block_len
    )
    if args.model == "dream":
        chat = args.chat if args.chat is not None else True  # Instruct default
        prompt_ids = adapter.encode(args.prompt, chat=chat)
    else:
        prompt_ids = adapter.encode(args.prompt)
    runs, rows = bench_runs(adapter, prompt_ids, config, caches)
    title = f"{args.model}: max_new={args.max_new} steps={args.steps} block_len={args.block_len}"
    if args.model == "dream" and args.quant != "none":
        title += f" quant={args.quant}"
    table = markdown_table(rows, title=title)
    print(table)

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        slug = f"{args.model}_mn{args.max_new}_s{args.steps}_bl{args.block_len}"
        if args.model == "dream" and args.quant != "none":
            slug += f"_{args.quant}"
        (out / f"{slug}.md").write_text(table + "\n", encoding="utf-8")
        for cache, run in zip(caches, runs):
            write_jsonl(run.events, out / f"{slug}__{_slug_label(cache)}.jsonl")
        print(f"\nwrote {slug}.md + {len(runs)} JSONL log(s) to {out}")


if __name__ == "__main__":
    main()
