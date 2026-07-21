"""User-facing local-only policy and outbound-attempt audit ledger."""
from __future__ import annotations

import json

from clozn.cli import main as ctx


def cmd_local_only(args) -> int:
    from clozn import network_policy
    if args.action == "status":
        enabled = network_policy.local_only_enabled()
        report = {"local_only": enabled}
    else:
        configured = args.action == "on"
        try:
            report = network_policy.set_local_only(configured)
        except OSError as exc:
            raise ctx.CloznError(f"could not persist local-only policy: {exc}") from None
        enabled = bool(report.get("effective"))
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("local-only - " + ("on" if enabled else "off"))
    return 0


def cmd_outbound(args) -> int:
    from clozn import network_policy
    try:
        attempts = network_policy.read_outbound_attempts(limit=args.limit)
    except (OSError, ValueError) as exc:
        raise ctx.CloznError(f"could not read outbound ledger: {exc}") from None
    if args.json:
        print(json.dumps({"attempts": attempts}, indent=2, ensure_ascii=False))
        return 0
    if not attempts:
        print("no outbound attempts recorded")
        return 0
    for attempt in attempts:
        destination = attempt.get("host") or attempt.get("destination") or "unknown"
        outcome = attempt.get("outcome") or "unknown"
        operation = attempt.get("operation") or attempt.get("method") or "request"
        print(f"{attempt.get('timestamp') or '?'}  {outcome:<8} {operation:<8} {destination}")
    return 0


def cmd_verify_offline(args) -> int:
    from clozn import network_policy
    report = network_policy.verify_offline(since=args.since)
    ok = bool(report.get("verified") if "verified" in report else report.get("ok"))
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print("offline verification - " + ("verified" if ok else "failed"))
        detail = report.get("detail") or report.get("reason")
        if detail:
            print("  " + str(detail))
    return 0 if ok else 1


def add_subparser(subparsers):
    parser = subparsers.add_parser("privacy", help="local-only policy and outbound-attempt audit")
    commands = parser.add_subparsers(dest="privacy_cmd")
    parser.set_defaults(fn=lambda _args: 2)

    local = commands.add_parser("local-only", help="enable, disable, or inspect fail-closed networking")
    local.add_argument("action", choices=("on", "off", "status"))
    local.add_argument("--json", action="store_true")
    local.set_defaults(fn=cmd_local_only)

    outbound = commands.add_parser("outbound", help="inspect the body-free outbound-attempt ledger")
    outbound.add_argument("--limit", type=int, default=100)
    outbound.add_argument("--json", action="store_true")
    outbound.set_defaults(fn=cmd_outbound)

    verify = commands.add_parser("verify-offline", help="verify the active local-only enforcement seam")
    verify.add_argument("--since", help="only consider ledger entries at/after this ISO timestamp")
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(fn=cmd_verify_offline)
    return parser


__all__ = ["add_subparser", "cmd_local_only", "cmd_outbound", "cmd_verify_offline"]
