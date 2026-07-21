"""Evidence-only diagnosis of recorded request latency and output cutoff."""
from __future__ import annotations

import json

from clozn.runs.diagnosis import diagnose


def add_subparser(subparsers) -> None:
    parser = subparsers.add_parser(
        "diagnose", help="explain recorded latency and output cutoff without generating")
    parser.add_argument(
        "target", help="'last' for the latest matching run, or an exact run id")
    parser.add_argument("--session", default=None,
                        help="for 'last', exact caller-known X-Clozn-Session-Id")
    parser.add_argument("--client-id", default=None,
                        help="for 'last', exact caller-known X-Clozn-Client-Id")
    parser.add_argument("--client", default=None,
                        help="for 'last', coarse recorded client label")
    parser.add_argument("--model", default=None, help="for 'last', model filter")
    parser.add_argument("--include-derived", action="store_true",
                        help="allow replay/branch/fork runs when selecting 'last'")
    parser.add_argument("--json", action="store_true", help="print the structured diagnosis")
    parser.set_defaults(fn=cmd_diagnose)


def _select_run(args, runlog):
    from clozn.cli.main import CloznError

    filters = {
        "client": args.client,
        "client_id": args.client_id,
        "session_id": args.session,
        "model": args.model,
        "include_derived": bool(args.include_derived),
    }
    if args.target == "last":
        summary = runlog.latest_run(**filters)
        if summary is None:
            raise CloznError("no matching recorded run found")
        run = runlog.get_run(summary.get("id", ""))
        if run is None:
            raise CloznError("the latest matching run could not be read")
        return run

    if any(value is not None for value in (
        args.session, args.client_id, args.client, args.model
    )) or args.include_derived:
        raise CloznError("selection filters apply only when the target is 'last'")
    run = runlog.get_run(args.target)
    if run is None:
        raise CloznError(f"run not found: {args.target}")
    return run


def _finding_line(finding: dict) -> str:
    status = str(finding.get("status") or "unknown").replace("_", " ")
    return f"  {str(finding.get('id') or 'unknown').replace('_', ' '):22} {status:13} {finding.get('text', '')}"


def format_diagnosis(report: dict) -> str:
    """Render the diagnosis without adding conclusions not present in its findings."""
    slow = report.get("why_slow") or {}
    cutoff = report.get("why_cut_off") or {}
    auxiliary = report.get("client_auxiliary_calls") or {}
    lines = [f"diagnosis - {report.get('run_id') or '?'}", "", "WHY SLOW"]
    findings = slow.get("findings") if isinstance(slow.get("findings"), list) else []
    if findings:
        lines.extend(_finding_line(item) for item in findings if isinstance(item, dict))
    else:
        lines.append("  no timing findings recorded")
    lines.extend(["", "WHY CUT OFF"])
    finding = cutoff.get("finding")
    lines.append(_finding_line(finding) if isinstance(finding, dict)
                 else "  output cutoff          unavailable   No cutoff finding recorded.")
    lines.extend(["", "CLIENT AUXILIARY CALLS", _finding_line(auxiliary)])
    return "\n".join(lines)


def cmd_diagnose(args) -> int:
    import clozn.runs.store as runlog

    run = _select_run(args, runlog)
    related = runlog.iter_runs(limit=200)
    report = diagnose(run, related_runs=related)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_diagnosis(report))
    return 0
