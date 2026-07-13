"""Server-side push (AMBIENT_DELIVERY.md channel 2, the no-separate-process path): fire a desktop
notification straight from the request handler the moment a sketchy reply goes through, so you don't need
a `clozn watch` poller running -- the server is already in the path, so it pushes the toast itself.

Same decision (should_alert) and same notifier as `clozn watch`, so the two never disagree. Dispatched on
a daemon thread by default so the native toast subprocess never adds latency to the reply. Opt-in
(clozn_alert:true per request, or server-wide POST /alert/mode) -- off by default.
"""
from __future__ import annotations

import threading

from clozn.watch.alerts import should_alert
from clozn.watch.notify import OSNotifier


def push_if_alerting(run: dict, link: str, notifier=None, async_dispatch: bool = True):
    """If `run` should alert, dispatch a desktop notification and return the Alert; else None. Never
    raises. `async_dispatch` (default) fires on a daemon thread so the reply isn't blocked on the toast;
    tests pass async_dispatch=False + a RecordingNotifier to assert synchronously."""
    try:
        al = should_alert(run)
    except Exception:
        return None
    if not al:
        return None
    n = notifier or OSNotifier()
    title = "clozn · " + ("heads up" if al.severity == "high" else "worth a look")
    body = al.headline + (f'  —  "{al.prompt}"' if al.prompt else "")

    def _fire():
        try:
            n.send(title, body, link)
        except Exception:
            pass                                   # the push is a bonus -- never let it surface an error

    if async_dispatch:
        threading.Thread(target=_fire, daemon=True).start()
    else:
        _fire()
    return al
