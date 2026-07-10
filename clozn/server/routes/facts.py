"""The FACTS tier (slot memory): POST /facts/* -- mode toggle, list, add, delete, and the honest read
receipt. Every one degrades cleanly when the tier is off (memory_facts) or no substrate/model is loaded --
the panel renders either way. Mechanical extraction of clozn.server.app's old `_facts` handler method;
behavior unchanged. -> clozn.memory (facts_mode + the live SlotBox on clozn.server.app).
"""
from clozn.server import app as ctx


def try_post(h, p, body):
    if not p.startswith("/facts/"):
        return False
    import clozn.memory.facts_mode as facts_mode
    if p == "/facts/mode":            # read or set the on/off gate (the latency switch)
        if "enabled" in body:
            on = bool(body.get("enabled"))
            if not facts_mode.set_enabled(on):
                h._json(200, {"ok": False, "reason": "could not persist the setting"})
                return True
            h._json(200, {"ok": True, "enabled": facts_mode.enabled(), "layer": facts_mode.LAYER})
            return True
        box = ctx._slots_box()
        st = box.status() if box is not None else {"enabled": facts_mode.enabled(),
                                                    "layer": facts_mode.LAYER,
                                                    "profile": ctx._active_profile_name() or "default",
                                                    "count": 0}
        h._json(200, st)
        return True
    box = ctx._slots_box()
    if box is None:                   # no substrate at all yet -> honest empty
        h._json(200, {"enabled": facts_mode.enabled(), "entries": [], "count": 0,
                      "note": "no substrate loaded"})
        return True
    if p == "/facts/list":            # the store's entries (cue/answer) -- read-only, no forward
        h._json(200, {"enabled": facts_mode.enabled(), "entries": box.list_entries(), **box.status()})
        return True
    if p == "/facts/add":             # explicit gated write (the gate refusal is the receipt)
        h._json(200, box.add(str(body.get("cue", "")), str(body.get("answer", "")),
                             gate=bool(body.get("gate", True))))
        return True
    if p == "/facts/delete":          # surgical per-entry removal (bystanders bit-identical)
        h._json(200, box.delete(cue=body.get("cue"), index=body.get("index")))
        return True
    if p == "/facts/read":            # the honest read receipt (hit / gate value / abstention + slot_ms)
        h._json(200, box.read_receipt(str(body.get("query", ""))))
        return True
    h._json(404, {"error": f"POST {p}"})
    return True
