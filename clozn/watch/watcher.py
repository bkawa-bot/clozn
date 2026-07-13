"""The ambient-alert poll loop (AMBIENT_DELIVERY.md channel 2): watch the run journal over HTTP and fire
a notification for each NEW run that should_alert() flags. The user keeps working in their own client;
clozn only interrupts when a run is worth it, with a one-click /r/<id> link to look closer.

Seams so it's fully testable headless: a Client (the HTTP wrapper, mockable) and a Notifier (notify.py).
On startup the watcher PRIMES -- it marks every run already in the journal as seen WITHOUT alerting -- so
it never floods you with your entire history; it only speaks up about runs that land after it starts.
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field

from clozn.watch.alerts import should_alert


class Client:
    """The journal over HTTP -- list summaries, fetch a full record (with trace). Mockable in tests."""

    def __init__(self, base: str, timeout: float = 8.0):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str):
        try:
            with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
                return json.loads(r.read() or b"{}")
        except Exception:
            return None

    def list_runs(self, limit: int = 50) -> list[dict]:
        d = self._get("/runs")                     # server returns the newest 80, query ignored -- plenty
        return (d or {}).get("runs") or []

    def get_run(self, rid: str) -> dict | None:
        d = self._get("/runs/" + rid)
        return (d.get("run", d) if isinstance(d, dict) else None)


@dataclass
class WatchState:
    seen: set = field(default_factory=set)     # run ids already processed (prime + fired)
    primed: bool = False


def prime(client: Client, state: WatchState) -> int:
    """Mark everything currently in the journal as seen, WITHOUT alerting -- the startup baseline. Returns
    how many runs were baselined (so the CLI can say 'watching N runs')."""
    runs = client.list_runs()
    for r in runs:
        rid = r.get("id")
        if rid:
            state.seen.add(rid)
    state.primed = True
    return len(runs)


def run_once(client: Client, notifier, state: WatchState, base_url: str) -> list:
    """One poll: for each NEW run (not in state.seen), decide + notify. Returns the Alerts fired.
    Cheap pre-filter on the summary (skip machine sources / derived arms without a fetch); only the
    survivors are fetched in full (they carry the trace should_alert needs). Never raises."""
    from clozn.watch.alerts import is_organic
    fired = []
    try:
        runs = client.list_runs()
    except Exception:
        return fired
    base = base_url.rstrip("/")
    # oldest-first so a burst of new runs alerts in the order they happened
    for summ in reversed(runs):
        rid = summ.get("id")
        if not rid or rid in state.seen:
            continue
        state.seen.add(rid)                                # seen exactly once, alert or not
        if not is_organic(summ):                           # skip studio probes without a full fetch
            continue
        rec = client.get_run(rid) or summ
        alert = should_alert(rec)
        if alert:
            title = ("clozn · " + ("heads up" if alert.severity == "high" else "worth a look"))
            body = (alert.headline + (f"  —  “{alert.prompt}”" if alert.prompt else ""))
            notifier.send(title, body, base + "/r/" + rid)
            fired.append(alert)
    # keep `seen` from growing without bound over a long session (ids are ~unique; cap generously)
    if len(state.seen) > 5000:
        state.seen = set(list(state.seen)[-2000:])
    return fired
