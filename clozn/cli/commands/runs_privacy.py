"""Local run-journal privacy controls and privacy-safe telemetry export."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clozn.cli import main as ctx


def _positive(value: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("value must be a positive integer") from None
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _print_receipt(receipt: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return
    action = receipt.get("action") or "journal mutation"
    run_id = receipt.get("run_id") or ""
    status = receipt.get("status") or ("complete" if receipt.get("ok") else "not found")
    print(f"{action} - {run_id} ({status})".strip())
    redaction = receipt.get("redaction") or {}
    if redaction.get("status") == "literal_redacted":
        print(f"  {redaction.get('literal_count', 0)} literal(s), "
              f"{redaction.get('replacement_count', 0)} occurrence(s) replaced; trace blob untouched")
    cascade_count = receipt.get("cascade_deleted_count")
    if cascade_count:
        print(f"  cascade: {cascade_count} child/descendant run(s) also deleted")


def cmd_redact(args) -> int:
    from clozn.runs import mutations
    literals = list(args.literal) if getattr(args, "literal", None) else None
    try:
        receipt = mutations.redact_run(args.run_id, literals=literals)
    except mutations.MutationError as exc:
        raise ctx.CloznError(f"run redaction failed: {exc}") from None
    if not receipt.get("ok"):
        raise ctx.CloznError(f"run redaction failed: run not found: {args.run_id}")
    _print_receipt(receipt, as_json=bool(args.json))
    return 0


def cmd_delete(args) -> int:
    if not args.yes:
        raise ctx.CloznError("run deletion is permanent; re-run with --yes")
    from clozn.runs import mutations
    try:
        receipt = mutations.delete_run(args.run_id, cascade=bool(args.cascade))
    except mutations.RunHasChildrenError as exc:
        raise ctx.CloznError(f"run deletion refused: {exc}; re-run with --cascade to delete them too") \
            from None
    except mutations.MutationError as exc:
        raise ctx.CloznError(f"run deletion failed: {exc}") from None
    if not receipt.get("ok"):
        raise ctx.CloznError(f"run deletion failed: run not found: {args.run_id}")
    _print_receipt(receipt, as_json=bool(args.json))
    return 0


def cmd_retention(args) -> int:
    from clozn.runs import mutations
    try:
        receipt = mutations.prune_to(args.keep, dry_run=not args.apply)
    except mutations.MutationError as exc:
        raise ctx.CloznError(f"retention update failed: {exc}") from None
    if args.json:
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
    else:
        verb = "deleted" if args.apply else "would delete"
        candidates = receipt.get("delete") or receipt.get("run_ids") or []
        print(f"retention - keep newest {args.keep}; {verb} {len(candidates)} run(s)")
        if not args.apply and candidates:
            print("  dry run; pass --apply to delete these journal rows")
        orphaned = receipt.get("orphaned_trace_digests") or []
        if orphaned:
            print(f"  {len(orphaned)} trace blob(s) become unreferenced; `clozn migrate --gc` removes them")
    return 0


def _write_jsonl(path: str, records: list[dict], *, force: bool) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists() and not force:
        raise ctx.CloznError(f"refusing to overwrite {target}; pass --force")
    # Use the shared atomic JSON writer on an envelope when stdout is not requested; the telemetry
    # module owns canonical JSONL serialization and exposes it separately to avoid partial files.
    from clozn.runs import telemetry
    try:
        telemetry.write_jsonl(str(target), records)
    except (OSError, telemetry.TelemetryExportError) as exc:
        raise ctx.CloznError(f"could not write telemetry export: {exc}") from None


def cmd_export_otel(args) -> int:
    from clozn.runs import telemetry
    from clozn.testkit.run_selection import RunSelectionError, resolve_runs
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
        records = telemetry.export_runs(
            runs, include_content=bool(args.include_content),
            redactions={literal: "[REDACTED]" for literal in args.redact},
        )
    except (RunSelectionError, telemetry.TelemetryExportError, ValueError) as exc:
        raise ctx.CloznError(f"telemetry export failed: {exc}") from None
    if args.out == "-":
        print(telemetry.format_jsonl(records), end="")
    else:
        _write_jsonl(args.out, records, force=bool(args.force))
        print(f"telemetry exported - {args.out} ({len(records)} span(s))")
    return 0


def add_subparser(subparsers):
    parser = subparsers.add_parser("runs", help="manage local run-journal privacy and exports")
    commands = parser.add_subparsers(dest="runs_cmd")
    parser.set_defaults(fn=lambda _args: 2)

    redact = commands.add_parser("redact", help="replace one run's private content with a tombstone")
    redact.add_argument("run_id")
    redact.add_argument("--literal", action="append", default=[], metavar="TEXT",
                        help="redact only this exact substring (repeatable); default is a full tombstone")
    redact.add_argument("--json", action="store_true")
    redact.set_defaults(fn=cmd_redact)

    delete = commands.add_parser("delete", help="permanently delete one exact run")
    delete.add_argument("run_id")
    delete.add_argument("--yes", action="store_true", help="confirm permanent deletion")
    delete.add_argument("--cascade", action="store_true",
                        help="also delete replay/branch children instead of refusing")
    delete.add_argument("--json", action="store_true")
    delete.set_defaults(fn=cmd_delete)

    retention = commands.add_parser("retention", help="preview or apply an oldest-run retention cutoff")
    retention.add_argument("--keep", required=True, type=_positive,
                           help="number of newest runs to retain")
    retention.add_argument("--apply", action="store_true", help="apply the deletion (default dry-run)")
    retention.add_argument("--json", action="store_true")
    retention.set_defaults(fn=cmd_retention)

    exported = commands.add_parser(
        "export-otel", help="export OpenTelemetry/OpenInference-compatible JSONL")
    exported.add_argument("out", help="output .jsonl path or - for stdout")
    selector = exported.add_mutually_exclusive_group(required=True)
    selector.add_argument("--from-runs", nargs="+", metavar="RUN_ID")
    selector.add_argument("--latest", nargs="?", const=1, type=_positive, metavar="N")
    exported.add_argument("--source")
    exported.add_argument("--client")
    exported.add_argument("--project")
    exported.add_argument("--include-derived", action="store_true")
    exported.add_argument("--include-content", action="store_true",
                          help="include prompt/messages/response content (omitted by default)")
    exported.add_argument("--redact", action="append", default=[], metavar="LITERAL",
                          help="redact a literal from included content; repeatable")
    exported.add_argument("--force", action="store_true")
    exported.set_defaults(fn=cmd_export_otel)
    return parser


__all__ = ["add_subparser", "cmd_delete", "cmd_export_otel", "cmd_redact", "cmd_retention"]
