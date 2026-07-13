"""Studio static-file serving: the instrument's own HTML/CSS/JS, served straight off disk from
`studio/` (DEMO) -- no build step, no templating. Mechanical extraction of clozn.server.app's `_html`
helper + the do_GET literal-root / asset-suffix branches; behavior updated so "/" and "/index.html" now
point at the live clozn app (studio/heavn/) instead of the old instrument.html shell.
"""
import os

from clozn.server.config import DEMO

# The live app's canonical root -- keep this in sync with cli/commands/studio.py's `--open` target.
HEAVN_INDEX = "/heavn/index.html"


def try_get(handler, path):
    """Studio static GETs: "/" and "/index.html" 302-redirect to the live app at HEAVN_INDEX; any other
    .html/.css/.js/.mjs file under DEMO (including subdirs like pages/, heavn/) is served directly off
    disk, guarded against escaping DEMO. Returns True iff handled.

    Why a redirect and not serving heavn/index.html's bytes in place at "/": heavn/index.html loads
    itself entirely through RELATIVE references (./api.mjs, ./modules/*.mjs, ./theme.css). Those only
    resolve correctly if the browser's document URL is under /heavn/ -- serving the same bytes at "/"
    would break every one of them (they'd resolve to /api.mjs, /modules/*.mjs, etc., which don't exist)
    unless each reference were rewritten to an absolute /heavn/... URL. A redirect needs none of that
    bookkeeping and can never drift out of sync as the app grows new relative imports/assets; it also
    costs nothing extra here since HEAVN_INDEX is itself served by the plain suffix branch below, a path
    already exercised by every /heavn/*.{html,css,js,mjs} request today. (The old literal-root case also
    special-cased "/instrument.html" itself -- dropped as redundant, not as a behavior change:
    /instrument.html sits directly under DEMO and already matches the suffix branch below unchanged.)
    """
    if path in ("/", "/index.html"):
        handler._send(302, "", "text/plain; charset=utf-8", {"Location": HEAVN_INDEX})
        return True
    if path.endswith((".html", ".css", ".js", ".mjs")):
        fn = os.path.normpath(os.path.join(DEMO, path.lstrip("/")))   # serve subdirs (pages/, heavn/) too, safely
        if fn.startswith(os.path.normpath(DEMO)) and os.path.isfile(fn):
            ct = ("text/html" if path.endswith(".html") else
                  "text/css" if path.endswith(".css") else "application/javascript")   # .js and .mjs (ES modules)
            handler._send(200, open(fn, encoding="utf-8").read(), ct + "; charset=utf-8")
            return True
    return False
