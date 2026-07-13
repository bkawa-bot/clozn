"""Ambient delivery, channel 1 (AMBIENT_DELIVERY.md):

  * GET  /r/<id>        -- the per-run permalink. Redirects a receipt-footer link straight into the
                          studio, deep-linked to that run (the app reads ?run=<id> on boot). This is the
                          "and keep the option of opening the studio yourself" half of the design.
  * GET  /receipt/mode  -- is the in-band footer on? (server-wide default)
  * POST /receipt/mode  -- turn it on/off, persisted. A client that can't add a body field just points
                          its tool at clozn once, flips this, and every reply carries the receipt link.

The footer itself is appended in routes/openai.py (non-stream) + sse.py (stream); see
clozn/runs/receipt_footer.py for its shape + honesty rules.
"""
import clozn.memory.mode as memory_mode
from clozn.server.static import HEAVN_INDEX

RECEIPT_SETTING = "receipt_footer"
ALERT_SETTING = "desktop_alert"


def receipt_enabled() -> bool:
    return bool(memory_mode.get_setting(RECEIPT_SETTING, False))


def alert_enabled() -> bool:
    return bool(memory_mode.get_setting(ALERT_SETTING, False))


def _safe_run_id(raw: str) -> str:
    """Run-id charset only -- the value rides a Location header + a URL query, so strip anything that
    could inject a header or escape the query (run ids are like run_0019f...; keep [A-Za-z0-9_-])."""
    return "".join(c for c in (raw or "") if c.isalnum() or c in "_-")


def try_get(h, p):
    if p == "/receipt/mode":
        h._json(200, {"receipt_footer": receipt_enabled()})
        return True
    if p == "/alert/mode":
        h._json(200, {"desktop_alert": alert_enabled()})
        return True
    if p.startswith("/r/"):
        rid = _safe_run_id(p[len("/r/"):].strip("/"))
        if not rid:
            h._send(404, "no run id", "text/plain; charset=utf-8")
            return True
        h._send(302, "", "text/plain; charset=utf-8", {"Location": HEAVN_INDEX + "?run=" + rid})
        return True
    return False


def try_post(h, p, body):
    if p == "/receipt/mode":
        changed = "receipt_footer" in body
        if changed:
            memory_mode.set_setting(RECEIPT_SETTING, bool(body.get("receipt_footer")))
        h._json(200, {"receipt_footer": receipt_enabled(), "changed": changed})
        return True
    if p == "/alert/mode":       # server-side desktop push -- fire a native toast inline on a sketchy reply
        changed = "desktop_alert" in body
        if changed:
            memory_mode.set_setting(ALERT_SETTING, bool(body.get("desktop_alert")))
        h._json(200, {"desktop_alert": alert_enabled(), "changed": changed})
        return True
    return False
