"""Studio static-file serving: the instrument's own HTML/CSS/JS, served straight off disk from
`studio/` (DEMO) -- no build step, no templating. Mechanical extraction of clozn.server.app's `_html`
helper + the do_GET literal-root / asset-suffix branches; behavior unchanged.
"""
import os

from clozn.server.config import DEMO


def serve_named(handler, name):
    """Serve one named file from Studio's static dir as HTML (mirrors app.py's old `_html` helper)."""
    handler._send(200, open(os.path.join(DEMO, name), encoding="utf-8").read(), "text/html; charset=utf-8")


def try_get(handler, path):
    """Studio static GETs: the instrument's root page, and any other .html/.css/.js file under DEMO
    (including subdirs like pages/, guarded against escaping DEMO). Returns True iff handled."""
    if path in ("/", "/index.html", "/instrument.html"):
        serve_named(handler, "instrument.html")
        return True
    if path.endswith((".html", ".css", ".js", ".mjs")):
        fn = os.path.normpath(os.path.join(DEMO, path.lstrip("/")))   # serve subdirs (pages/, heavn/) too, safely
        if fn.startswith(os.path.normpath(DEMO)) and os.path.isfile(fn):
            ct = ("text/html" if path.endswith(".html") else
                  "text/css" if path.endswith(".css") else "application/javascript")   # .js and .mjs (ES modules)
            handler._send(200, open(fn, encoding="utf-8").read(), ct + "; charset=utf-8")
            return True
    return False
