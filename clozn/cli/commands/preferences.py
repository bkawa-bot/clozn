"""commands.preferences -- `clozn preferences`: the propose-and-review surface. Renders the Studio's
POST /preferences (clozn.behavior.preferences): the learned-preference suggestions the model proposes from
your Run Inspector quick-repairs ("Too verbose" x3 -> "make concise your default?"). Zero generation -- it
reads the accumulated pattern. --approve/--dismiss POST /preferences/resolve (approve persists the dial;
the ONLY place a dial changes). Terminal-reachable, like `clozn explain`, so the learning loop isn't
studio-only. format_preferences is a pure JSON->text function, testable with a canned dict.

CloznError lives on `clozn.cli.main`; imported INSIDE the functions that raise it (not at module level)
for the same circular-import reason documented in engine_process.py's module docstring.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from clozn.cli import formatting as fmt


def _fetch_preferences(port: int) -> dict:
    from clozn.cli import main as ctx
    req = urllib.request.Request(f"http://127.0.0.1:{port}/preferences", data=b'{"threshold":3}',
                                 method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise ctx.CloznError(f"couldn't reach the Clozn gateway on port {port} "
                             f"({getattr(e, 'reason', e)}). Start it first:  clozn serve <model>")
    except Exception as e:
        raise ctx.CloznError(f"preferences failed: {e}")


def _resolve_preference(port: int, pid: str, action: str) -> dict:
    from clozn.cli import main as ctx
    body = json.dumps({"id": pid, "action": action}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/preferences/resolve", data=body,
                                 method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", str(e))
        except Exception:
            msg = str(e)
        raise ctx.CloznError(f"{action} failed ({e.code}): {msg}")
    except urllib.error.URLError as e:
        raise ctx.CloznError(f"couldn't reach the Clozn gateway on port {port} "
                             f"({getattr(e, 'reason', e)}). Start it first:  clozn serve <model>")


def format_preferences(data: dict) -> str:
    """Pure JSON->text render of the pending proposals (no server -- testable with a canned dict). Lists what
    the model is asking to make a default; nothing here changes a dial (that's `--approve <id>`)."""
    pend = (data or {}).get("pending") or []
    if not pend:
        return (f"{fmt.DIM}no suggestions yet -- use the Run Inspector's \"Too verbose / Too cold\" buttons a "
                f"few times and a pattern will surface here.{fmt.RST}")
    out = [f"{fmt.BOLD}learned-preference suggestions{fmt.RST}  "
           f"{fmt.DIM}approve to make one a default, or dismiss{fmt.RST}"]
    for p in pend:
        n = len(p.get("evidence") or [])
        ev = f"   {fmt.DIM}from {n} repl{'y' if n == 1 else 'ies'}{fmt.RST}" if n else ""
        out.append(f"  {p.get('label', '(preference)')}{ev}")
        out.append(f"    {fmt.DIM}approve:{fmt.RST} clozn preferences --approve {p.get('id')}"
                   f"   {fmt.DIM}dismiss:{fmt.RST} --dismiss {p.get('id')}")
    return "\n".join(out)


def cmd_preferences(args):
    port = args.port or 8080
    pid = args.approve or args.dismiss
    if pid:
        action = "approve" if args.approve else "dismiss"
        r = _resolve_preference(port, pid, action)
        pr = (r or {}).get("proposal") or {}
        if action == "approve":
            ap = (r or {}).get("applied") or {}
            if ap.get("error"):
                print(f"approved, but the dial couldn't be applied: {ap['error']}")
            else:
                print(f"approved -- {fmt.BOLD}{pr.get('dial', '?')}{fmt.RST} set to {ap.get('value', '?')} "
                      f"{fmt.DIM}(now your default){fmt.RST}")
        else:
            print(f"dismissed -- {pr.get('dial', '?')} {fmt.DIM}won't resurface unless the pattern gets much "
                  f"stronger{fmt.RST}")
        return
    print(format_preferences(_fetch_preferences(port)))
