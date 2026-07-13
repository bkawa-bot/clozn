"""commands.watch -- `clozn watch`: ambient alerts (AMBIENT_DELIVERY.md channel 2).

Poll the run journal and notify ONLY when a run is worth a look (errored / truncated / a failed
tiny-test / low confidence) -- so you keep working in your own client and clozn taps you on the shoulder
just when it matters, with a one-click /r/<id> link into the studio. Primes on startup (baselines your
history without alerting), then watches. Stdlib-only; talks to a running studio/serve over HTTP.
"""
from __future__ import annotations

import time

from clozn.cli import formatting as fmt
from clozn.watch import watcher
from clozn.watch.notify import OSNotifier, PrintNotifier


def cmd_watch(args):
    base = (args.url or "http://127.0.0.1:8090").rstrip("/")
    interval = max(1, int(args.interval or 5))
    notifier = PrintNotifier() if getattr(args, "print_only", False) else OSNotifier()
    client = watcher.Client(base)

    # reachable? (a clean message beats a silent no-op if the studio isn't up)
    if client.list_runs() is None or client._get("/runs") is None:
        raise SystemExit(f"can't reach the run journal at {base} -- is the studio/serve up? "
                         f"(clozn studio, or clozn serve)")

    state = watcher.WatchState()
    n = watcher.prime(client, state)
    print(f"{fmt.BOLD}clozn watch{fmt.RST} -- watching the journal at {base}  "
          f"({n} run(s) baselined; alerting on new ones)")
    print(f"{fmt.DIM}  notifies only when a run is worth a look -- errored, truncated, a failed check, "
          f"or low confidence. Ctrl-C to stop.{fmt.RST}")
    try:
        while True:
            time.sleep(interval)
            for al in watcher.run_once(client, notifier, state, base):
                pass                                       # notifier already surfaced each one
    except KeyboardInterrupt:
        print(f"\n{fmt.DIM}- stopped watching{fmt.RST}")


def add_subparser(sub):
    pw = sub.add_parser("watch", help="ambient alerts: notify only when a run is worth a look")
    pw.add_argument("--url", default=None, help="studio/serve base URL (default http://127.0.0.1:8090)")
    pw.add_argument("--interval", type=int, default=5, help="poll seconds (default 5)")
    pw.add_argument("--print", action="store_true", dest="print_only",
                    help="print alerts to the terminal only (no native desktop toast)")
    pw.set_defaults(fn=cmd_watch)
