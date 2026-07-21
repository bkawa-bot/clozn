"""Evidence-only memory usage receipts from the local run journal."""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets
import sys
import tempfile
import time

from clozn.cli import main as ctx
from clozn.runs.memory_usage import memory_usage
import clozn.runs.store as runlog


def _read_markdown(source: str) -> str:
    if source == "-":
        return sys.stdin.read()
    try:
        return Path(source).read_text(encoding="utf-8")
    except OSError as exc:
        raise ctx.CloznError(f"could not read {source}: {exc}") from None


def _write_markdown(path: str, document: str, *, force: bool) -> None:
    if path == "-":
        print(document, end="")
        return
    target = Path(path).expanduser().resolve()
    if target.exists() and not force:
        raise ctx.CloznError(f"refusing to overwrite {target}; pass --force")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(target.parent), prefix=".tmp-clozn-memory-", suffix=".md")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(document)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


def _fresh_card(text: str, *, status: str, source: str, imported=None) -> dict:
    from clozn.memory import cards as memory_cards
    original = imported if isinstance(imported, dict) else {}
    provenance_ok = False
    source_run_id = original.get("source_run_id")
    quoted_span = original.get("quoted_span")
    source_turn = original.get("source_turn")
    if isinstance(source_run_id, str) and isinstance(quoted_span, str) and quoted_span:
        run = runlog.get_run(source_run_id)
        messages = run.get("messages") if isinstance(run, dict) else None
        if (isinstance(messages, list) and isinstance(source_turn, int)
                and 0 <= source_turn < len(messages)):
            content = str((messages[source_turn] or {}).get("content") or "") \
                if isinstance(messages[source_turn], dict) else ""
            provenance_ok = quoted_span in content or content.startswith(quoted_span.rstrip("…"))
    return {
        "id": (str(original.get("id")) if original.get("id") else "mem_" + secrets.token_hex(6)),
        "text": str(text).strip(),
        "status": status,
        "source_run_id": source_run_id if provenance_ok else None,
        "source_turn": source_turn if provenance_ok else None,
        "quoted_span": quoted_span if provenance_ok else "",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "last_used_at": None,
        "usage_count": 0,
        "kind": str(original.get("kind") or "preference"),
        "risk": memory_cards.risk_of_text(text),
        "evidence": f"imported from {source}",
        "strength": float(original.get("strength", 1.0)),
    }


def cmd_memory_export(args) -> int:
    from clozn.memory import cards as memory_cards
    from clozn.memory import markdown_cards
    status = None if args.status == "all" else args.status
    exported = [dict(card) for card in memory_cards.list_cards(status=status)]
    if not args.include_provenance:
        for card in exported:
            card.update(source_run_id=None, source_turn=None, quoted_span="", evidence="")
    try:
        document = markdown_cards.format_cards(exported)
        _write_markdown(args.path, document, force=bool(args.force))
    except (markdown_cards.CardMarkdownError, OSError) as exc:
        raise ctx.CloznError(f"memory export failed: {exc}") from None
    return 0


def cmd_memory_import(args) -> int:
    from clozn.memory import cards as memory_cards
    from clozn.memory import markdown_cards
    document = _read_markdown(args.path)
    source = "stdin" if args.path == "-" else str(Path(args.path).name)
    try:
        versioned = document.lstrip("\ufeff").startswith(markdown_cards.MAGIC)
        if versioned:
            parsed = markdown_cards.parse_cards(document)
            prepared = [_fresh_card(card["text"], status=(card["status"] if args.preserve_status
                                                           else (args.status or "pending")),
                                    source=source, imported=card) for card in parsed]
        else:
            if args.preserve_status:
                raise markdown_cards.CardMarkdownError(
                    "--preserve-status requires a versioned Clozn memory export")
            texts = markdown_cards.parse_plain_cards(document)
            prepared = [_fresh_card(text, status=(args.status or "pending"), source=source)
                        for text in texts]
        report = memory_cards.merge_import(
            prepared, on_duplicate=args.on_duplicate, dry_run=bool(args.dry_run))
    except (markdown_cards.CardMarkdownError, memory_cards.CardStoreError, ValueError) as exc:
        raise ctx.CloznError(f"memory import failed: {exc}") from None
    report["status_policy"] = "preserve" if args.preserve_status else (args.status or "pending")
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        action = "would add" if args.dry_run else "added"
        print(f"memory import: parsed {report['parsed']}, {action} {report['added']}, "
              f"skipped {report['skipped_duplicates']} duplicate(s)")
    return 0


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
    export = commands.add_parser("export", help="export cards as deterministic Markdown")
    export.add_argument("path", nargs="?", default="-", help="output path or - for stdout")
    export.add_argument("--status", choices=("active", "pending", "disabled", "rejected", "all"),
                        default="active", help="cards to export (default active)")
    export.add_argument("--include-provenance", action="store_true",
                        help="include source run, quote, and evidence fields")
    export.add_argument("--force", action="store_true", help="overwrite an existing output file")
    export.set_defaults(fn=cmd_memory_export)
    imported = commands.add_parser("import", help="transactionally merge Markdown cards")
    imported.add_argument("path", help="input path or - for stdin")
    policy = imported.add_mutually_exclusive_group()
    policy.add_argument("--status", choices=("pending", "active", "disabled", "rejected"),
                        default=None, help="status assigned to every imported card (default pending)")
    policy.add_argument("--preserve-status", action="store_true",
                        help="preserve statuses from a versioned Clozn export")
    imported.add_argument("--on-duplicate", choices=("skip", "error"), default="skip")
    imported.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    imported.add_argument("--json", action="store_true", help="print the machine-readable import report")
    imported.set_defaults(fn=cmd_memory_import)
