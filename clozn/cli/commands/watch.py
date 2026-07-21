"""Tail the local run journal with stable insertion-order cursors."""
from __future__ import annotations

import json
import time

from clozn.cli.main import CloznError


def add_subparser(sub) -> None:
    p = sub.add_parser("watch", help="tail newly recorded runs (local journal; no generation)")
    p.add_argument("--client", default=None, help="coarse client label (for example script or browser)")
    p.add_argument("--client-id", default=None, help="exact caller-known X-Clozn-Client-Id")
    p.add_argument("--session", default=None, help="exact caller-known X-Clozn-Session-Id")
    p.add_argument("--model", default=None, help="model filter")
    p.add_argument("--include-derived", action="store_true", help="include replay/branch/fork runs")
    p.add_argument("--interval", type=float, default=0.5, help="poll interval in seconds (default 0.5)")
    history = p.add_mutually_exclusive_group()
    history.add_argument("--last", type=int, default=None, metavar="N",
                         help="print the last N matches before following")
    history.add_argument("--since", default=None, metavar="RUN_ID",
                         help="start after an existing run id")
    p.add_argument("--once", action="store_true", help="print available matches and exit")
    p.add_argument("--json", action="store_true", help="emit one JSON summary per line")
    p.set_defaults(fn=cmd_watch)


def _emit(run: dict, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(run, ensure_ascii=False, separators=(",", ":")), flush=True)
        return
    finish = run.get("finish_reason") or ("error" if run.get("error") else "unknown")
    prompt = str(run.get("prompt_summary") or "").replace("\n", " ")
    response = str(run.get("response_summary") or "").replace("\n", " ")
    print(f"{run.get('id')}  {run.get('source') or '-'}  {run.get('model') or '-'}  "
          f"{finish}  {prompt} -> {response}", flush=True)


def cmd_watch(args) -> int:
    import clozn.runs.store as runlog

    if args.interval <= 0:
        raise CloznError("--interval must be greater than zero")
    if args.last is not None and args.last < 1:
        raise CloznError("--last must be a positive integer")
    filters = {
        "client": args.client,
        "client_id": args.client_id,
        "session_id": args.session,
        "model": args.model,
        "include_derived": bool(args.include_derived),
    }

    if args.since:
        cursor = runlog.cursor_for_run(args.since)
        if cursor is None:
            raise CloznError(f"run not found: {args.since}")
    elif args.last is not None:
        snapshot = runlog.runs_after(None, limit=1000, **filters)
        for run in snapshot["runs"][-args.last:]:
            _emit(run, as_json=args.json)
        cursor = snapshot["next_cursor"]
        if args.once:
            return 0
    elif args.once:
        run = runlog.latest_run(**filters)
        if run is not None:
            _emit(run, as_json=args.json)
        return 0
    else:
        # Default tail semantics: start at "now" and only print runs recorded after invocation.
        cursor = runlog.current_cursor()

    while True:
        page = runlog.runs_after(cursor, limit=100, **filters)
        for run in page["runs"]:
            _emit(run, as_json=args.json)
        cursor = page["next_cursor"] or cursor
        if args.once:
            return 0
        time.sleep(args.interval)

