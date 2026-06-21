"""``cloze`` — the command-line front door for the Phase 1 lab.

    cloze run "def add(a, b): return a +" --model dcoder            # tiny real model, CPU
    cloze run "The capital of France is" --model dream --device cuda --quant nf4
    cloze run "hello cloze" --model fake --tui                       # watch it denoise live
    cloze bench --model dcoder --block-len 8                         # off/delta A/B (delegates)
    cloze tui --model fake                                           # live renderer (delegates)

``run`` denoises a prompt and prints the result (``--tui`` watches it land pass by
pass); ``bench`` and ``tui`` forward their arguments to the benchmark report and the
live renderer. The model text goes to stdout, a one-line stats footer to stderr, so
``cloze run ... > out.txt`` keeps just the generated text.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from cloze_lab.generate import GenerateConfig, generate, infill
from cloze_lab.models.base import ModelAdapter


def build_adapter(model: str, *, device: str = "cpu", quant: str = "none") -> ModelAdapter:
    """Construct an adapter by name: ``fake`` (torch-free), ``dream`` 7B, ``dcoder`` 0.5B, ``llada`` 8B.

    ``quant`` (nf4/int8) applies to Dream/LLaDA and is GPU-only; it lets the 7B/8B fit a
    consumer card. The torch/transformers imports stay lazy so ``fake`` needs neither.
    """
    if model == "fake":
        from cloze_lab.models.fake import FakeAdapter

        return FakeAdapter(seed=0)
    from cloze_lab.models.base import LoadConfig

    if model == "dream":
        from cloze_lab.models.dream import DREAM_7B_INSTRUCT, DreamAdapter

        q = None if quant == "none" else quant
        return DreamAdapter(
            LoadConfig(model_id=DREAM_7B_INSTRUCT, device=device, dtype="bfloat16"), quantization=q
        )
    if model == "dcoder":
        from cloze_lab.models.dream import OPEN_DCODER_05B, open_dcoder_adapter

        return open_dcoder_adapter(LoadConfig(model_id=OPEN_DCODER_05B, device=device, dtype="bfloat16"))
    if model == "llada":
        from cloze_lab.models.llada import LLADA_8B_INSTRUCT, LLaDAAdapter

        q = None if quant == "none" else quant
        return LLaDAAdapter(
            LoadConfig(model_id=LLADA_8B_INSTRUCT, device=device, dtype="bfloat16"), quantization=q
        )
    raise ValueError(f"unknown model {model!r} (choose fake|dream|dcoder|llada)")


def _run(args: argparse.Namespace) -> None:
    adapter = build_adapter(args.model, device=args.device, quant=args.quant)
    cache = None
    if args.effort is not None:
        from cloze_lab.effort import resolve_effort

        preset = resolve_effort(args.effort)
        steps, block_len, cache = preset.steps, preset.block_len, preset.cache
    else:
        steps, block_len = args.steps, args.block_len
    config = GenerateConfig(
        max_new=args.max_new,
        steps=steps,
        temperature=args.temperature,
        seed=args.seed,
        block_len=block_len,
    )
    if args.model in ("dream", "llada"):
        # Dream / LLaDA are Instruct models: chat-template by default so they answer instead
        # of EOS-ing on a raw prompt. fake/dcoder are completion models — raw encode.
        chat = args.chat if args.chat is not None else True
        prompt_ids = adapter.encode(args.prompt, chat=chat)
    else:
        prompt_ids = adapter.encode(args.prompt)
    reviser = _reviser_from(args)
    if args.tui:
        from cloze_lab.render.tui import live_generate

        result = live_generate(adapter, prompt_ids, config, reviser=reviser, cache=cache, delay=args.delay)
    else:
        result = generate(adapter, prompt_ids, config, reviser=reviser, cache=cache)
        print(result.text)
    finished = result.events[-1]
    # ASCII-only footer: raw stderr must not assume a UTF-8 console (Windows cp1252).
    print(
        f"[{args.model}] {finished.reason} | {finished.new_tokens} tok | "
        f"{finished.steps_total} steps | {finished.tok_per_s:.1f} tok/s (PyTorch ref)",
        file=sys.stderr,
    )


def _reviser_from(args: argparse.Namespace):
    if args.revise is None:
        return None
    from cloze_lab.scheduler.policies import RemaskLowConf

    return RemaskLowConf(tau_revise=args.revise, max_revisions=args.max_revisions)


def _infill(args: argparse.Namespace) -> None:
    adapter = build_adapter(args.model, device=args.device, quant=args.quant)
    prefix_ids = adapter.encode(args.prefix)
    suffix_ids = adapter.encode(args.suffix)
    config = GenerateConfig(
        max_new=args.gap, steps=args.steps, temperature=args.temperature, seed=args.seed
    )
    result = infill(
        adapter, prefix_ids, suffix_ids, args.gap, config, reviser=_reviser_from(args)
    )
    # The board is prefix + fill + suffix; print the whole reconstructed sequence.
    print(adapter.decode([int(i) for i in result.board]))
    f = result.events[-1]
    print(
        f"[{args.model}] infill {f.reason} | {f.new_tokens}/{args.gap} filled | "
        f"{f.steps_total} steps | {f.tok_per_s:.1f} tok/s (PyTorch ref)",
        file=sys.stderr,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cloze", description="Cloze — local diffusion-LM runtime (Phase 1 lab)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="denoise a prompt and print the result")
    run.add_argument("prompt")
    run.add_argument("--model", choices=("fake", "dream", "dcoder", "llada"), default="dcoder")
    run.add_argument("--max-new", type=int, default=32)
    run.add_argument("--steps", type=int, default=8)
    run.add_argument("--block-len", type=int, default=0)
    run.add_argument(
        "--effort", choices=("fast", "balanced", "quality"), default=None,
        help="speed/quality preset (steps + block + KV-reuse); overrides --steps/--block-len",
    )
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", default="cpu")
    run.add_argument("--quant", choices=("none", "nf4", "int8"), default="none")
    run.add_argument("--tui", action="store_true", help="watch the denoise live")
    run.add_argument("--delay", type=float, default=0.15, help="--tui pause after each pass (s)")
    run.add_argument(
        "--chat", action=argparse.BooleanOptionalAction, default=None,
        help="apply the instruct chat template before tokenizing (default: on for --model dream/llada)",
    )
    run.add_argument(
        "--revise", type=float, default=None, metavar="TAU",
        help="enable remask_lowconf revisions: re-mask committed tokens whose confidence drops below TAU",
    )
    run.add_argument("--max-revisions", type=int, default=1, help="per-position revision cap (with --revise)")
    run.set_defaults(func=_run)

    # infill: fill a masked gap BETWEEN a prefix and suffix (native dLLM editing).
    inf = sub.add_parser("infill", help="fill a gap between a prefix and a suffix (bidirectional)")
    inf.add_argument("prefix")
    inf.add_argument("suffix")
    inf.add_argument("--gap", type=int, default=8, help="number of masked slots to fill")
    inf.add_argument("--model", choices=("fake", "dream", "dcoder", "llada"), default="dcoder")
    inf.add_argument("--steps", type=int, default=8)
    inf.add_argument("--temperature", type=float, default=0.0)
    inf.add_argument("--seed", type=int, default=0)
    inf.add_argument("--device", default="cpu")
    inf.add_argument("--quant", choices=("none", "nf4", "int8"), default="none")
    inf.add_argument("--revise", type=float, default=None, metavar="TAU",
                     help="enable remask_lowconf revisions below confidence TAU")
    inf.add_argument("--max-revisions", type=int, default=1, help="per-position revision cap (with --revise)")
    inf.set_defaults(func=_infill)

    # bench/tui forward their args to the existing entry points; declared here so they
    # show in `cloze --help`, dispatched by the passthrough below.
    sub.add_parser("bench", help="off/delta cache A/B benchmark (forwards args)", add_help=False)
    sub.add_parser("tui", help="watch a denoise live (forwards args)", add_help=False)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "bench":
        from cloze_lab.bench.report import main as bench_main

        bench_main(argv[1:])
        return
    if argv and argv[0] == "tui":
        from cloze_lab.render.tui import main as tui_main

        tui_main(argv[1:])
        return
    args = _build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
