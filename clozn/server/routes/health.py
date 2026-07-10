"""Health/state/config surface: which substrate is active (+ switching it), the engine's own health,
the instrument's `/state` snapshot, and the capture-tier setting. Mechanical extraction of the matching
`if p == ...` branches out of clozn.server.app's do_GET/do_POST; behavior unchanged.

Reads shared server state (SUB/SUBNAME/ENGINE/switch_substrate/...) off `clozn.server.app` at call time
(never a captured import) so a substrate swap -- or a test's `monkeypatch.setattr(app, "SUB", ...)` --
is observed exactly as it was before this lived inline.
"""
import threading
import time

from clozn.server import app as ctx


def try_get(h, p):
    if p == "/substrate":
        h._json(200, {"active": ctx.SUBNAME, "available": ["qwen", "dream"]})
        return True
    if p == "/engine/health":
        try:
            h._json(200, {"engine": ctx.ENGINE.health()})
        except Exception as e:
            h._json(502, {"error": f"engine unreachable: {e}"})
        return True
    if p == "/state":
        h._json(200, {"substrate": ctx.SUBNAME, "memory_mode": ctx._memory_mode(),
                      **(ctx.SUB.state() if ctx.SUB else {})})
        return True
    if p == "/capture/tier":
        from clozn.runs import capture_mode
        h._json(200, {"tier": capture_mode.tier(), "tiers": list(capture_mode.TIERS)})
        return True
    return False


def try_post(h, p, body):
    if p == "/substrate":
        name = str(body.get("name", "qwen"))
        if name == ctx.SUBNAME:
            h._json(200, {"active": ctx.SUBNAME, "switched": False})
            return True
        if name not in ("qwen", "dream"):
            h._json(400, {"error": "unknown substrate"})
            return True
        h._json(200, {"active": name, "switched": True, "note": "reloading -- poll /substrate"})
        threading.Thread(target=lambda: (time.sleep(0.4), ctx.switch_substrate(name)), daemon=True).start()
        return True
    if p == "/capture/tier":
        from clozn.runs import capture_mode
        name = str(body.get("tier", "")).strip().lower()
        if name not in capture_mode.TIERS:
            h._json(400, {"error": f"unknown tier (want one of {list(capture_mode.TIERS)})"})
            return True
        if not capture_mode.set_tier(name):
            h._json(200, {"ok": False, "reason": "could not persist the tier setting"})
            return True
        h._json(200, {"ok": True, "tier": name})
        return True
    return False
