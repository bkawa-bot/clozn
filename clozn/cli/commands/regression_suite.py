"""Promote captured application runs into editable, frozen regression suites."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile

from clozn._io import atomic_write_json
from clozn.cli import main as ctx


def _positive(value: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("count must be a positive integer") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("count must be a positive integer")
    return number


def _load(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except OSError as exc:
        raise ctx.CloznError(f"could not read suite {path!r}: {exc}") from None
    except json.JSONDecodeError as exc:
        raise ctx.CloznError(f"suite is not valid JSON: {exc}") from None
    return value


def _write(path: str, value: dict, *, force: bool = False, replace: str | None = None) -> None:
    target = Path(path).expanduser().resolve()
    replacement = Path(replace).expanduser().resolve() if replace else None
    if target.exists() and not force and target != replacement:
        raise ctx.CloznError(f"refusing to overwrite {target}; pass --force")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(str(target), value, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise ctx.CloznError(f"could not write suite artifact {target}: {exc}") from None


def _edit_json(value: dict) -> dict:
    editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
    if not editor:
        raise ctx.CloznError("--edit needs VISUAL or EDITOR to name a blocking editor command")
    fd, temporary = tempfile.mkstemp(prefix="clozn-suite-", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        command = shlex.split(editor, posix=os.name != "nt")
        if not command:
            raise ctx.CloznError("VISUAL/EDITOR is empty")
        try:
            result = subprocess.run([*command, temporary], check=False)
        except OSError as exc:
            raise ctx.CloznError(f"could not start editor: {exc}") from None
        if result.returncode:
            raise ctx.CloznError(f"editor exited with status {result.returncode}; suite was not written")
        return _load(temporary)
    finally:
        try:
            os.remove(temporary)
        except OSError:
            pass


def _promotion():
    from clozn.testkit import promotion
    return promotion


def cmd_create(args) -> int:
    from clozn.testkit.run_selection import RunSelectionError, resolve_runs
    promotion = _promotion()
    try:
        runs = resolve_runs(
            run_ids=args.from_runs,
            latest=args.latest is not None,
            count=args.latest,
            source=args.source,
            client=args.client,
            project=args.project,
            include_derived=bool(args.include_derived),
        )
        name = args.name or Path(args.out).stem
        suite = promotion.create_suite_draft(name, runs)
        if args.redact:
            suite = promotion.redact_suite(
                suite, {literal: "[REDACTED]" for literal in args.redact})
        if args.edit:
            suite = promotion.validate_suite(_edit_json(suite), require_frozen=False)
        if args.freeze:
            suite = promotion.freeze_suite(suite)
        _write(args.out, suite, force=bool(args.force))
    except (RunSelectionError, promotion.PromotionError, ValueError) as exc:
        raise ctx.CloznError(f"suite create failed: {exc}") from None
    state = suite.get("state", "draft")
    print(f"suite created - {args.out} ({len(suite.get('cases') or [])} case(s), {state})")
    if state != "frozen":
        print(f"  review/edit the draft, then run: clozn suite freeze {args.out}")
    return 0


def cmd_freeze(args) -> int:
    promotion = _promotion()
    source = str(Path(args.path).expanduser().resolve())
    out = args.out or source
    try:
        suite = promotion.validate_suite(_load(source), require_frozen=False)
        frozen = promotion.freeze_suite(suite)
        _write(out, frozen, force=bool(args.force), replace=source)
    except (promotion.PromotionError, ValueError) as exc:
        raise ctx.CloznError(f"suite freeze failed: {exc}") from None
    print(f"suite frozen - {out}")
    return 0


def cmd_verify(args) -> int:
    import clozn.runs.store as runlog
    promotion = _promotion()
    try:
        suite = promotion.validate_suite(_load(args.path), require_frozen=True)
        checked = 0
        if args.source:
            for case in suite.get("cases") or []:
                source = case.get("source") if isinstance(case, dict) else None
                run_id = source.get("run_id") if isinstance(source, dict) else None
                run = runlog.get_run(run_id) if run_id else None
                if not isinstance(run, dict):
                    raise promotion.PromotionError(f"source run is unavailable: {run_id or '?'}")
                if not promotion.verify_source(case, run):
                    raise promotion.PromotionError(f"source run drifted since promotion: {run_id}")
                checked += 1
    except (promotion.PromotionError, ValueError) as exc:
        raise ctx.CloznError(f"suite verification failed: {exc}") from None
    qualifier = f" and {checked} source run(s)" if args.source else ""
    print(f"suite verified - frozen content{qualifier}")
    return 0


def _format_result(result, warnings=()) -> str:
    lines = [f"warning - {warning['case']}: {warning['message']}" for warning in warnings]
    for case in result.cases:
        lines.append(f"[{case.status.upper()}] {case.name}  {case.run_id or '?'}")
        if case.error:
            lines.append(f"  {case.error}")
        for assertion in case.assertions:
            lines.append(f"  {assertion.get('status', '?')}: {assertion.get('check', '?')}")
    counts = result.counts or {}
    lines.append("summary - " + ", ".join(
        f"{counts.get(status, 0)} {status}" for status in ("pass", "fail", "error")))
    return "\n".join(lines)


def cmd_run(args) -> int:
    from clozn.testkit import ci
    promotion = _promotion()
    try:
        suite = promotion.validate_suite(_load(args.path), require_frozen=not args.allow_draft)
    except (promotion.PromotionError, ValueError) as exc:
        raise ctx.CloznError(f"suite run refused: {exc}") from None
    headers = {}
    if args.client_id:
        headers["X-Clozn-Client-Id"] = args.client_id
    if args.project:
        headers["X-Clozn-Project-Id"] = args.project
    result = ci.run_suite(suite, ci.Client(args.url, headers=headers))
    payload = result.to_dict()
    payload["suite_name"] = suite["name"]
    payload["suite_sha256"] = (suite.get("freeze") or {}).get("sha256")
    warnings = [
        {"case": case["name"], "message": warning}
        for case in suite.get("cases") or []
        for warning in case.get("warnings") or []
    ]
    if warnings:
        payload["warnings"] = warnings
    if args.out:
        _write(args.out, payload, force=bool(args.force))
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(_format_result(result, warnings))
        if args.out:
            print(f"result - {args.out}")
    return 0 if result.status == "pass" else 1


def add_subparser(subparsers):
    parser = subparsers.add_parser("suite", help="promote captured app runs into regression cases")
    commands = parser.add_subparsers(dest="suite_cmd")
    parser.set_defaults(fn=lambda _args: 2)

    create = commands.add_parser("create", help="create a regression-suite draft from recorded runs")
    create.add_argument("out", help="output JSON path")
    selector = create.add_mutually_exclusive_group(required=True)
    selector.add_argument("--from-runs", nargs="+", metavar="RUN_ID",
                          help="explicit captured run ids in case order")
    selector.add_argument("--latest", nargs="?", const=1, type=_positive, metavar="N",
                          help="select the latest organic run(s), default 1")
    create.add_argument("--source", help="filter latest runs by exact journal source")
    create.add_argument("--client", help="filter latest runs by recorded client label")
    create.add_argument("--project", help="filter latest runs by raw or opaque project id")
    create.add_argument("--include-derived", action="store_true",
                        help="permit replay/branch/fork children")
    create.add_argument("--name", help="suite name (default output filename stem)")
    create.add_argument("--redact", action="append", default=[], metavar="LITERAL",
                        help="replace a literal in case inputs/expectations with [REDACTED]; repeatable")
    create.add_argument("--edit", action="store_true",
                        help="open the draft in VISUAL/EDITOR before validation")
    create.add_argument("--freeze", action="store_true",
                        help="seal executable cases after redaction/editing")
    create.add_argument("--force", action="store_true", help="overwrite the output")
    create.set_defaults(fn=cmd_create)

    freeze = commands.add_parser("freeze", help="validate and seal an edited suite draft")
    freeze.add_argument("path")
    freeze.add_argument("--out", help="write a new path instead of replacing the draft")
    freeze.add_argument("--force", action="store_true", help="overwrite a different output path")
    freeze.set_defaults(fn=cmd_freeze)

    verify = commands.add_parser("verify", help="verify frozen content and optional live source evidence")
    verify.add_argument("path")
    verify.add_argument("--source", action="store_true",
                        help="also compare against source runs still in the local journal")
    verify.set_defaults(fn=cmd_verify)

    run = commands.add_parser("run", help="execute every frozen regression case against a gateway")
    run.add_argument("path")
    run.add_argument("--url", default="http://127.0.0.1:8080")
    run.add_argument("--client-id", help="activate matching app-scoped memory during the suite")
    run.add_argument("--project", help="activate matching project-scoped memory during the suite")
    run.add_argument("--allow-draft", action="store_true",
                     help="run an unfrozen draft intentionally")
    run.add_argument("--out", help="write the machine-readable result JSON")
    run.add_argument("--force", action="store_true", help="overwrite the result path")
    run.add_argument("--json", action="store_true", help="print the result JSON")
    run.set_defaults(fn=cmd_run)
    return parser


__all__ = ["add_subparser", "cmd_create", "cmd_freeze", "cmd_run", "cmd_verify"]
