"""Evidence-only memory usage receipts from the local run journal."""
from __future__ import annotations

import json

from clozn.cli import main as ctx
from clozn.runs.memory_usage import memory_usage
import clozn.runs.store as runlog


def _card_lines(cards) -> list[str]:
    lines: list[str] = []
    for card in cards or []:
        if not isinstance(card, dict):
            continue
        identity = card.get("id") or "unidentified"
        suffix = f" (relevance {card['relevance']})" if card.get("relevance") is not None else ""
        lines.append(f"  - {identity}: {card.get('text') or ''}{suffix}")
    return lines


def format_memory_usage(receipt: dict) -> str:
    prompt = receipt.get("prompt_cards") or {}
    injected = prompt.get("injected") or {}
    selected = prompt.get("selected") or {}
    omitted = prompt.get("omitted") or {}
    token_cost = receipt.get("token_cost") or {}
    lines = [f"memory used - {receipt.get('run_id') or '?'}",
             f"mode - {receipt.get('mode') or 'unknown'}"]

    if injected.get("status") == "observed":
        lines.append(f"injected - {injected.get('count', 0)} card(s)")
        lines.extend(_card_lines(injected.get("cards")))
    else:
        lines.append(f"injected - {injected.get('status', 'unavailable')}")
        if injected.get("note"):
            lines.append("  " + str(injected["note"]))

    if selected.get("status") in ("observed", "derived"):
        qualifier = "capture-time" if selected.get("status") == "observed" else "same as injected"
        lines.append(f"selected - {selected.get('count', 0)} card(s), {qualifier}")
        lines.append("  " + str(selected.get("note") or ""))
    else:
        lines.append(f"selected - {selected.get('status', 'unavailable')}")
        if selected.get("note"):
            lines.append("  " + str(selected["note"]))

    if omitted.get("status") == "observed":
        ids = omitted.get("ids") or []
        lines.append("omitted - " + (", ".join(str(value) for value in ids) if ids else "none recorded"))
        if omitted.get("reason"):
            lines.append("  reason: " + str(omitted["reason"]))
    else:
        lines.append("omitted - unavailable")
        lines.append("  " + str(omitted.get("note") or ""))

    if (token_cost.get("memory_prompt_tokens") == 0
            and token_cost.get("prompt_block_utf8_bytes") == 0):
        lines.append("token cost - 0 prompt-memory tokens (no prompt block)")
    elif isinstance(token_cost.get("memory_prompt_tokens"), int):
        lines.append(f"token cost - {token_cost['memory_prompt_tokens']} prompt-memory tokens (exact delta)")
    else:
        total = token_cost.get("total_prompt_tokens")
        total_text = f"; total prompt {total} tokens" if isinstance(total, int) else ""
        lines.append(f"token cost - unavailable; block {token_cost.get('prompt_block_utf8_bytes', 0)} UTF-8 bytes"
                     f"{total_text}")
        if token_cost.get("unavailable_reason"):
            lines.append("  reason: " + str(token_cost["unavailable_reason"]))
        lines.append("  Memory-specific token delta was not captured and is not estimated.")

    anchored = receipt.get("anchored") or {}
    if anchored.get("status") == "observed":
        line = f"anchored - {anchored.get('count', 0)} bag(s)"
        if anchored.get("skipped"):
            line += f"; skipped: {anchored['skipped']}"
        lines.append(line)
    facts = receipt.get("facts") or {}
    if facts.get("status") == "observed":
        lines.append("facts - read/write evidence recorded")
    internalized = receipt.get("internalized") or {}
    if internalized.get("status") == "observed":
        lines.append(f"internalized active set - {internalized.get('count', 0)} card(s)")
        lines.extend(_card_lines(internalized.get("active_cards")))
    return "\n".join(lines)


def cmd_memory_used_last(args):
    summary = runlog.latest_run(include_derived=False)
    if not summary:
        raise ctx.CloznError("no recorded run found")
    run = runlog.get_run(summary["id"])
    if not run:
        raise ctx.CloznError("the latest run could not be read")
    receipt = memory_usage(run)
    if args.json:
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
    else:
        print(format_memory_usage(receipt))
    return 0


def add_subparser(subparsers):
    parser = subparsers.add_parser("memory", help="inspect evidence of memory used by a run")
    commands = parser.add_subparsers(dest="memory_cmd")
    used = commands.add_parser("used", help="show selected, injected, and omitted memory evidence")
    used_commands = used.add_subparsers(dest="memory_used_cmd")
    last = used_commands.add_parser("last", help="show memory evidence for the latest organic run")
    last.add_argument("--json", action="store_true", help="print the structured receipt")
    last.set_defaults(fn=cmd_memory_used_last)
