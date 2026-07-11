#!/usr/bin/env python3
"""clozn -- a boring, reliable front door to the local model engine.

The fast runtime is the C++ engine (cloze-server.exe). This wraps it so the daily path is one command:

    clozn run   <model> "<prompt>"     one-shot, streams tokens to the terminal
    clozn serve <model> [--port 8080]  bring up the OpenAI-compatible endpoint, print the base URL
    clozn models                       discover local GGUFs and the backend that would run them

Stdlib only (urllib/subprocess/json) -- no torch, no pip install -- so it stays dependency-free and quick.
It finds the engine build (GPU preferred), puts the right DLLs on PATH, picks per-model flags (diffusion
mask tokens, etc.), reports honestly what it's running on, and fails with one actionable line instead of a
stack trace. Model dirs: $CLOZN_MODELS, ~/.clozn/models, <repo>/models, ~/.clozn/config.json["model_dirs"].

This module is the argparse root: the full command tree (build_parser), dispatch (main), and the shared
constants (HOME, CloznError) every other clozn.cli.* module reads at call time via
`from clozn.cli import main as ctx` (never `from clozn.cli.main import HOME`, which would bind a stale
copy immune to a test's monkeypatch or a later change). Each actual `cmd_X` implementation lives in
clozn/cli/commands/*.py, grouped by family; this file only imports and wires them.
"""
from __future__ import annotations

import argparse
import os
import sys

HOME = os.path.expanduser("~/.clozn")


class CloznError(Exception):
    """A clean, user-facing failure -- printed as one line, no traceback."""


# Imported after HOME/CloznError are defined: every module below reaches back into this one (`from
# clozn.cli import main as ctx` for HOME, `from clozn.cli.main import CloznError`), so this file must
# finish defining both before triggering those imports -- see engine_process.py's module docstring for the
# full circular-import trace this depends on. Order also matters *between* these: commands.serve/run/explain
# import names directly off commands.models (and commands.run), so models/run must load first.
#
# Several of these (_free_port, _save_trace, format_explain, _SPARK, ...) exist here purely as stable
# re-exports: CLI tests written against the pre-split flat module call `clozn.cli.main.<name>` directly, and
# since none of them are mutated globals (they're functions/constants, not DIM/BOLD/RST/COLOR), a plain
# import is safe -- a function always reads its OWN defining module's globals, never a stale copy of them.
from clozn.cli import formatting as fmt                                                       # noqa: E402
from clozn.cli.engine_process import _free_port                                               # noqa: E402
from clozn.cli.trace_io import _save_trace                                                    # noqa: E402
from clozn.cli.formatting import _C_BLUE, _C_HOT, _C_PALE, _SPARK, _conf_rgb, _heatmap_lines   # noqa: E402,F401
from clozn.cli.formatting import _paint, _paint_sparkline, _sparkline, _stream_token           # noqa: E402,F401
from clozn.cli.commands.models import cmd_models, cmd_pull, cmd_plan, format_plan              # noqa: E402
from clozn.cli.commands.models import format_throughput                                       # noqa: E402,F401
from clozn.cli.commands.run import cmd_run                                                    # noqa: E402
from clozn.cli.commands.serve import cmd_serve, cmd_ps, cmd_stop                              # noqa: E402
from clozn.cli.commands.studio import cmd_studio                                              # noqa: E402
from clozn.cli.commands.explain import (cmd_explain, cmd_trace, cmd_branch, format_explain,    # noqa: E402
                                        format_narrate, _fetch_explain, _fetch_narrate,
                                        _last_run_id, _verified_tag)
from clozn.cli.commands.preferences import cmd_preferences, format_preferences                # noqa: E402
from clozn.cli.commands.test import cmd_test                                                  # noqa: E402


def build_parser():
    """The full argparse tree, factored out of main() so tests can introspect flags without dispatching."""
    p = argparse.ArgumentParser(prog="clozn", description="a reliable front door to the local model engine")
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("run", help="one-shot: stream a completion to the terminal")
    pr.add_argument("model"); pr.add_argument("prompt", nargs="?", default=None)
    pr.add_argument("--max", type=int, default=256, help="max new tokens (default 256)")
    pr.add_argument("--cpu", action="store_true", help="force the CPU build")
    pr.add_argument("--port", type=int, default=0); pr.add_argument("--mask", type=int, default=None)
    pr.add_argument("--eos", type=int, default=None)
    pr.add_argument("--heat", action="store_true", help="paint each token as it streams by the model's "
                    "confidence (warm = wavered, cool = sure) -- the denoise heatmap, live (AR models)")
    pr.set_defaults(fn=cmd_run)

    ps = sub.add_parser("serve", help="bring up the OpenAI-compatible endpoint")
    ps.add_argument("model"); ps.add_argument("--port", type=int, default=0)
    ps.add_argument("--cpu", action="store_true"); ps.add_argument("--mask", type=int, default=None)
    ps.add_argument("--eos", type=int, default=None)
    ps.add_argument("--sae", default=None, help="on-device SAE readout dir (dims must match the model; "
                    "server refuses politely on mismatch)")
    ps.add_argument("--sae-k", type=int, default=None, help="SAE features kept per position (default 16)")
    ps.set_defaults(fn=cmd_serve)

    sub.add_parser("models", help="list local models + the engine backend").set_defaults(fn=cmd_models)
    pp = sub.add_parser("pull", help="download a model GGUF (by name, or owner/repo/file.gguf)")
    pp.add_argument("model"); pp.set_defaults(fn=cmd_pull)
    ppl = sub.add_parser("plan", help="will it fit? read a GGUF's header (no download, no load, no GPU) "
                         "before you commit to a multi-GB pull")
    ppl.add_argument("model", help="a known model name, a local .gguf path, or a HF resolve/... .gguf URL")
    ppl.add_argument("--vram", type=float, default=None,
                     help="VRAM budget in GB (default: detect via nvidia-smi, else 16)")
    ppl.add_argument("--bandwidth-gb-s", type=float, default=None,
                     help="assumed effective memory bandwidth in GB/s for the decode-throughput roofline "
                          "predictor (default: 900 GB/s, RTX-5080-class -- a model-free estimate, stated "
                          "explicitly since it drives the whole prediction; see `clozn plan`'s output)")
    ppl.add_argument("--calibrate", action="store_true",
                     help="DEFERRED: would boot the engine and measure ACTUAL tok/s to correct the "
                          "bandwidth assumption -- not implemented yet (prints a stub explaining why), "
                          "never boots anything")
    ppl.set_defaults(fn=cmd_plan)
    pst = sub.add_parser("studio", help="launch Clozn Studio (the glass-box UI + the endpoint your tools connect to)")
    pst.add_argument("substrate", nargs="?", default=None, help="qwen (default) | dream | engine")
    pst.add_argument("--port", type=int, default=0); pst.add_argument("--open", action="store_true", help="open the UI in your browser")
    pst.set_defaults(fn=cmd_studio)
    sub.add_parser("ps", help="list running serve daemons").set_defaults(fn=cmd_ps)
    pstop = sub.add_parser("stop", help="stop a serve daemon (by model name, port, or 'all')")
    pstop.add_argument("which"); pstop.set_defaults(fn=cmd_stop)
    pt = sub.add_parser("trace", help="inspect the last run journal entry's confidence timeline")
    pt.add_argument("--list", action="store_true", help="list recent run journal entries instead of showing the last")
    pt.add_argument("--legacy-cache", action="store_true",
                    help="read the old ~/.clozn/traces cache instead of the shared ~/.clozn/runs journal")
    pt.set_defaults(fn=cmd_trace)
    pb = sub.add_parser("branch", help="re-run from an uncertain point on the alternative (the road not taken)")
    pb.add_argument("--at", type=int, default=None, help="token index to fork at (default: the most uncertain)")
    pb.add_argument("--pick", type=int, default=0, help="which alternative to take (0 = the runner-up)")
    pb.add_argument("--max", type=int, default=80); pb.add_argument("--cpu", action="store_true")
    pb.set_defaults(fn=cmd_branch)
    pe = sub.add_parser("explain", help="explain a run: hesitations, active influences, concepts "
                        "(needs `clozn studio` running)")
    pe.add_argument("run_id", nargs="?", default=None, help="run id, as shown in the Studio's Runs list")
    pe.add_argument("--last", action="store_true", help="use the most recently recorded run")
    pe.add_argument("--port", type=int, default=0, help="Studio port (default 8090)")
    pe.add_argument("--why", action="store_true", help="also generate the accountable-self narration (M4): "
                    "a receipt-constrained \"why\", diffed against an independent judge and flagged wherever "
                    "it overclaims. Opt-in -- unlike the rest of `explain`, this GENERATES (two model calls; "
                    "needs the qwen substrate loaded in `clozn studio`)")
    pe.set_defaults(fn=cmd_explain)
    ppref = sub.add_parser("preferences", help="review learned-preference suggestions the model proposes "
                           "from your quick-repairs (needs `clozn studio` running)")
    ppref.add_argument("--approve", metavar="ID", default=None, help="approve a proposal by id (persists the dial)")
    ppref.add_argument("--dismiss", metavar="ID", default=None, help="dismiss a proposal by id")
    ppref.add_argument("--port", type=int, default=0, help="Studio port (default 8090)")
    ppref.set_defaults(fn=cmd_preferences)
    pte = sub.add_parser("test", help="run tiny-test assertions against a stored run (the receipt/replay seams)")
    pte.add_argument("file", help="path to a JSON tiny-test spec (see clozn/testkit/runner.py's module docstring)")
    pte.add_argument("--json", action="store_true",
                     help="print the machine-readable suite result instead of the report")
    pte.add_argument("--attach", action="store_true",
                     help="write results into each touched run's tiny_tests field (rides the receipt_bundle export)")
    pte.add_argument("--live", action="store_true",
                     help="permit causal (leans_on) assertions to run against a live Studio substrate; "
                          "without it they're honestly skipped ('needs --live'), never silently passed")
    pte.add_argument("--port", type=int, default=0, help="Studio port for --live (default 8090)")
    pte.set_defaults(fn=cmd_test)
    return p


def main(argv=None):
    fmt._setup_console()
    p = build_parser()
    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        p.print_help(); return 2
    try:
        rc = args.fn(args)
        return rc if isinstance(rc, int) else 0
    except CloznError as e:
        print(f"{fmt.BOLD}clozn:{fmt.RST} {e}", file=sys.stderr); return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
