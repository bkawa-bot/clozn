"""Named persona bundles: list every saved profile (GET /profiles/list), save/update one (does NOT apply
it), THE switch that applies a bundle's cards+dials to the live substrate, and export/import the bundle's
portable JSON. Mechanical extraction of the matching `if p == "/profiles/..."` branches out of
clozn.server.app's do_GET/do_POST; behavior unchanged. -> clozn.profiles.
"""
from clozn.server import app as ctx


def try_get(h, p):
    if p == "/profiles/list":         # every saved persona bundle + which one is active (masthead + Settings)
        from clozn.profiles import store as profiles
        h._json(200, {"profiles": profiles.ProfileStore().list(), "active": ctx._active_profile_name()})
        return True
    return False


def try_post(h, p, body):
    if p == "/profiles/save":        # create/update a named persona bundle (does NOT apply it -- see switch)
        from clozn.profiles import store as profiles
        try:
            saved = profiles.ProfileStore().save(profiles.validate(dict(body)))
        except (ValueError, KeyError, TypeError) as e:
            h._json(400, {"error": f"bad profile: {e}"})
            return True
        h._json(200, {"ok": True, "path": saved, "profile": profiles.ProfileStore().load(body["name"])})
        return True
    if p == "/profiles/switch":      # THE persona switch: cards replace, dials replace, instant in prompt mode
        from clozn.profiles import store as profiles
        name = str(body.get("name", "")).strip()
        if not name:
            h._json(400, {"error": "need a profile name"})
            return True
        try:
            prof = profiles.ProfileStore().load(name)
        except (OSError, ValueError) as e:
            h._json(404, {"error": f"no such profile '{name}': {e}"})
            return True
        if ctx.active_sub(h) is None:
            h._json(503, {"error": "no substrate loaded"})
            return True
        h._json(200, {"ok": True, **ctx._profiles_switch(ctx.active_sub(h), prof)})
        return True
    if p == "/profiles/export":       # -> the bundle's own JSON (client downloads/saves it -- the portable artifact)
        from clozn.profiles import store as profiles
        name = str(body.get("name", "")).strip()
        if not name:
            h._json(400, {"error": "need a profile name"})
            return True
        try:
            h._json(200, {"ok": True, "profile": profiles.ProfileStore().load(name)})
        except (OSError, ValueError) as e:
            h._json(404, {"error": f"no such profile '{name}': {e}"})
        return True
    if p == "/profiles/import":       # body IS the bundle JSON (as exported); optional {rename}
        from clozn.profiles import store as profiles
        try:
            bundle = dict(body.get("profile", body))
            rename = body.get("rename") or None
            p2 = profiles.validate(bundle)
            if rename:
                p2["name"] = rename
                p2 = profiles.validate(p2)
            path = profiles.ProfileStore().save(p2)
        except (ValueError, KeyError, TypeError) as e:
            h._json(400, {"error": f"bad profile bundle: {e}"})
            return True
        h._json(200, {"ok": True, "path": path, "profile": p2})
        return True
    return False
