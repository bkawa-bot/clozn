"""CLI for the versioned model-developer experiment object."""
from __future__ import annotations

import json

from clozn._io import atomic_write_json
from clozn.cli.main import CloznError
from clozn.experiments import suite


def add_subparser(sub):
    parser = sub.add_parser("experiment", help="run or inspect a case x variant x seed experiment manifest")
    commands = parser.add_subparsers(dest="experiment_cmd")
    parser.set_defaults(fn=_no_command)

    run = commands.add_parser("run", help="run target and guard suites across every variant and seed")
    run.add_argument("manifest", help="path to a clozn.experiment.v0 JSON manifest")
    run.add_argument("--url", default=suite.DEFAULT_URL, help="default Clozn gateway URL (default :8080)")
    run.add_argument("--seeds", type=int, default=None, help="override the manifest with seeds 0..N-1")
    run.add_argument("--out", default=None, help="result path (default ~/.clozn/experiments/<id>.json)")
    run.add_argument("--json", action="store_true", help="print the full result JSON")
    run.set_defaults(fn=cmd_run)

    show = commands.add_parser("show", help="inspect summary or matching per-case evidence")
    show.add_argument("result", help="experiment result JSON")
    show.add_argument("--suite", choices=["target", "guard"], default=None)
    show.add_argument("--case", default=None)
    show.add_argument("--variant", default=None)
    show.add_argument("--seed", type=int, default=None)
    show.add_argument("--json", action="store_true")
    show.set_defaults(fn=cmd_show)
    return parser


def _no_command(_args):
    print("clozn experiment: use `clozn experiment run <manifest.json>` or `clozn experiment show <result.json>`")
    return 2


def cmd_run(args):
    try:
        manifest = suite.load_manifest(args.manifest)
        result = suite.run_manifest(manifest, default_url=args.url, seeds_override=args.seeds)
    except suite.ManifestError as exc:
        raise CloznError(str(exc)) from exc
    path = args.out or suite.default_result_path(result)
    atomic_write_json(path, result, indent=2, ensure_ascii=False)
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(suite.format_summary(result))
        print(f"  result: {path}")
    return 1 if any(c.get("status") == "error" for c in result["cells"]) else 0


def cmd_show(args):
    try:
        with open(args.result, encoding="utf-8") as f:
            result = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise CloznError(f"could not read experiment result: {exc}") from exc
    filtered = any(v is not None for v in (args.suite, args.case, args.variant, args.seed))
    if args.json:
        payload = suite.select_cells(result, suite=args.suite, case=args.case, variant=args.variant, seed=args.seed) if filtered else result
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif filtered:
        print(suite.format_cells(suite.select_cells(result, suite=args.suite, case=args.case,
                                                    variant=args.variant, seed=args.seed)))
    else:
        print(suite.format_summary(result))
    return 0
