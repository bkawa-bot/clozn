"""commands.test -- `clozn test <file>` (clozn.testkit, backlog/tiny-test-harness): a minimal, JSON-authored,
run-level assertion harness on top of the receipt/replay seams: static checks read the stored run record
alone (zero generation); the one causal check (`leans_on`) runs receipts.py's leave-one-out ablation and is
honestly SKIPPED (never a silent pass) unless --live is given. testkit.run_suite/evaluate do all the work
and never print; this module is purely the CLI's load/dispatch/render/exit-code shell around it.
"""
from __future__ import annotations

import json
import sys
import urllib.request

from clozn.cli import formatting as fmt
from clozn.cli.trace_io import _import_runlog


def _import_testkit():
    from clozn import testkit
    return testkit


def _load_test_spec(path: str):
    """Load + shallow-validate a tiny-test JSON spec file. Returns (spec, error) -- exactly one is None. A
    problem here is `clozn test`'s EXIT-2 case (unreadable file, invalid JSON, or no non-empty 'tests'
    list) -- distinct from a per-assertion "error" status (a bad/unknown check inside an otherwise
    well-formed spec), which testkit.run_suite handles itself and only ever costs exit 1."""
    try:
        with open(path, encoding="utf-8") as f:
            spec = json.load(f)
    except OSError as e:
        return None, f"could not read {path}: {e}"
    except json.JSONDecodeError as e:
        return None, f"{path} is not valid JSON: {e}"
    if not isinstance(spec, dict) or not isinstance(spec.get("tests"), list) or not spec["tests"]:
        return None, f"{path}: spec must be a JSON object with a non-empty 'tests' list"
    return spec, None


def _fetch_live_receipt(port: int, run_id: str, influence: dict):
    """POST /runs/<id>/receipt on a running product gateway -- the SAME rigorous, both-arms-greedy causal
    receipt receipts.receipt() computes in-process there (clozn.server.app), fetched over the loopback HTTP
    bridge instead of needing an in-process model inside this CLI. Returns None on ANY failure (gateway
    not up, run not found, worker unavailable, bad influence spec) -- never raises: testkit's honesty rule
    (judge_receipt) turns a None receipt into an honest 'skipped' assertion, not a crashed `clozn test` run."""
    url = f"http://127.0.0.1:{port}/runs/{run_id}/receipt"
    body = json.dumps({"influence": influence}).encode()
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


_STATUS_MARK = {"pass": "✓", "fail": "✗", "skip": "○", "error": "!"}
# Mirrors testkit.CAUSAL_CHECKS -- duplicated (not imported) so this render-only helper stays a plain
# JSON-in/text-out function, exactly like format_explain; if the causal check vocabulary ever grows, change
# both (same convention run_timeline.py's LOW_CONF comment documents for explain.py's copy of it).
_CAUSAL_CHECKS = {"leans_on"}


def format_test_report(suite: dict) -> str:
    """Pure JSON(testkit.run_suite() result) -> text render: one line per test, one indented line per
    assertion, each tagged [static] or [causal] so the two classes never blur together in the output. The
    expected/actual delta only prints on a fail/error (a pass or an honest skip needs no diff). No I/O --
    testable with a canned suite dict, exactly like format_explain."""
    lines = []
    for t in suite.get("tests") or []:
        mark = _STATUS_MARK.get(t.get("status"), "?")
        lines.append(f"{fmt.BOLD}{mark} {t.get('name', '(unnamed test)')}{fmt.RST}  "
                     f"{fmt.DIM}[{t.get('run_id') or '?'}]{fmt.RST}")
        for a in t.get("assertions") or []:
            amark = _STATUS_MARK.get(a.get("status"), "?")
            check = a.get("check") or "?"
            kind = "causal" if check in _CAUSAL_CHECKS else "static"
            lines.append(f"    {amark} {fmt.DIM}[{kind}]{fmt.RST} {check}  {fmt.DIM}{a.get('target', '')}{fmt.RST}")
            if a.get("status") in ("fail", "error"):
                lines.append(f"        {fmt.DIM}expected:{fmt.RST} {a.get('expected')!r}")
                lines.append(f"        {fmt.DIM}actual:  {fmt.RST} {a.get('actual')!r}")
            if a.get("note"):
                lines.append(f"        {fmt.DIM}note:{fmt.RST} {a['note']}")
    counts = suite.get("counts") or {}
    summary = ", ".join(f"{counts.get(s, 0)} {s}" for s in ("pass", "fail", "skip", "error") if counts.get(s))
    lines.append("")
    lines.append(f"{fmt.BOLD}{summary or 'no assertions'}{fmt.RST}")
    return "\n".join(lines)


def cmd_test(args):
    """`clozn test <file>`: load a JSON tiny-test spec, resolve each test's run (an id, or "latest"),
    evaluate every assertion, render a report (or --json), optionally --attach the results into each
    touched run's tiny_tests field, and return the process exit code (main() propagates it):
        0 -- every assertion passed (skips allowed)
        1 -- at least one assertion failed or errored
        2 -- the spec file itself couldn't be loaded (bad path / invalid JSON / no 'tests' list)
    """
    testkit = _import_testkit()
    runlog = _import_runlog()
    spec, err = _load_test_spec(args.file)
    if err:
        print(f"{fmt.BOLD}clozn test:{fmt.RST} {err}", file=sys.stderr)
        return 2

    fetch_receipt = None
    if args.live:
        port = args.port or 8080
        fetch_receipt = lambda run, influence: _fetch_live_receipt(port, run.get("id"), influence)

    suite = testkit.run_suite(spec, get_run=testkit.default_get_run, sub=None, fetch_receipt=fetch_receipt)

    if args.attach:
        for rid, assertions in testkit.results_by_run(suite).items():
            runlog.update_tiny_tests(rid, assertions)

    if args.json:
        print(json.dumps(suite, indent=2, default=str))
    else:
        print(format_test_report(suite))

    return 0 if suite["status"] in ("pass", "skip") else 1
