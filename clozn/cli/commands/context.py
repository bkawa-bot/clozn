"""Readable context-delivery receipts from the local run journal."""
from __future__ import annotations

import json

from clozn.cli import main as ctx
from clozn.runs.context_receipt import build_context_receipt
import clozn.runs.store as runlog


def _receipt(run: dict) -> dict:
    stored = run.get("context_receipt")
    if isinstance(stored, dict) and stored.get("schema") == "clozn.context_receipt.v1":
        return stored
    return build_context_receipt(
        messages=run.get("messages"),
        assembled_messages=run.get("assembled_messages"),
        final_prompt=run.get("final_prompt"),
        finish_reason=run.get("finish_reason"),
        meta=run.get("meta"),
        trace=run.get("trace"),
    )


def _message_lines(messages) -> list[str]:
    if not messages:
        return ["  (none captured)"]
    lines: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        lines.append(f"  [{index}] {role}")
        content = str(message.get("content") or "")
        lines.extend("    " + line for line in content.splitlines() or [""])
    return lines or ["  (none captured)"]


def format_context(run: dict) -> str:
    receipt = _receipt(run)
    delivered = receipt.get("delivered") or {}
    survived = receipt.get("survived") or {}
    limits = receipt.get("limits") or {}
    warnings = receipt.get("warnings") or []

    lines = [f"context receipt · {run.get('id', '?')}"]
    if warnings:
        lines.append("WARNING · " + str(warnings[0].get("message") or "reply was cut off"))
    else:
        lines.append("status · no recorded input truncation or output cutoff")

    values = []
    for key, label in (("prompt_tokens", "prompt"),
                       ("context_window_tokens", "context window"),
                       ("requested_max_tokens", "requested output"),
                       ("generated_tokens", "generated")):
        if isinstance(limits.get(key), int):
            values.append(f"{label} {limits[key]} tok")
    if values:
        lines.append("limits · " + " · ".join(values))

    lines.extend(["", "DELIVERED", str(delivered.get("meaning") or "")])
    lines.extend(_message_lines(delivered.get("messages")))
    lines.extend(["", "SURVIVED", str(survived.get("meaning") or "")])
    assembled = survived.get("assembled_messages")
    if isinstance(assembled, list):
        lines.append("  assembled messages")
        lines.extend(_message_lines(assembled))
    else:
        lines.append("  assembled messages: (not captured)")
    final_prompt = survived.get("final_prompt")
    if isinstance(final_prompt, str):
        lines.append("  exact rendered prompt")
        lines.extend("    " + line for line in final_prompt.splitlines() or [""])
    else:
        lines.append("  exact rendered prompt: (not captured)")
    lines.extend(["", "input policy · " + str(receipt.get("input_policy") or "unknown")])
    return "\n".join(lines)


def cmd_context_last(args):
    rows = runlog.list_runs(limit=1, include_replays=False)
    if not rows:
        raise ctx.CloznError("no recorded run found")
    run = runlog.get_run(rows[0]["id"])
    if not run:
        raise ctx.CloznError("the latest run could not be read")
    if args.json:
        print(json.dumps({"run_id": run["id"], "context_receipt": _receipt(run)},
                         indent=2, ensure_ascii=False))
    else:
        print(format_context(run))
    return 0


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "context", help="inspect what a run delivered and what survived into generation")
    commands = parser.add_subparsers(dest="context_cmd")
    last = commands.add_parser("last", help="show the latest organic run's context receipt")
    last.add_argument("--json", action="store_true", help="print the structured receipt")
    last.set_defaults(fn=cmd_context_last)
