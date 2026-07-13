"""Anchored-memory routes (X7 productized -- see clozn/memory/anchored.py for the validated envelope and
the honesty contract). Registered from clozn.server.app with the other product route families. Shape
mirrors clozn/server/routes/memory.py exactly: try_get(h, p) / try_post(h, p, body) returning True when the
path is claimed; unhappy paths are 200 + {"ok": False,
"reason": ...} (the memory.py convention), 400 only for malformed input.

  POST /memory/anchored/fit         {card_id, k?}      fit + persist a bag for an EXISTING card
  GET  /memory/anchored/list                           every stored bag (vectors stripped)
  POST /memory/anchored/toggle      {card_id, on}      include/exclude a bag from composition
  POST /memory/anchored/delete_term {card_id, token}   remove ONE word-direction; alphas refit
  GET  /memory/anchored/whatlearned                    the alpha table for every active bag -- a pure
                                                       LOOKUP of the stored decomposition, no generation

Engine access goes through _provider(): the SAME ConceptSteer seam /steer/concept/* already drives
(app._engine_concept_steer()), wrapped in anchored.ConceptSteerDirProvider -- tests monkeypatch
_provider to a fake, so nothing here ever needs a live engine to be exercised.
"""
from clozn.server import app as ctx


def _provider():
    """The live DirProvider, or None when no engine is configured (CLOZN_ENGINE_QWEN_PORT unset).
    Module-level on purpose: tests monkeypatch this one seam to a fake provider."""
    cs = ctx._engine_concept_steer()
    if cs is None:
        return None
    import clozn.memory.anchored as anchored
    return anchored.ConceptSteerDirProvider(cs)


def try_get(h, p):
    if p == "/memory/anchored/list":                # every bag, on or off -- the Memory page's read
        import clozn.memory.anchored as anchored
        bags = anchored.load_bags()
        h._json(200, {"bags": [anchored.public_bag(b) for b in bags.values() if isinstance(b, dict)],
                      "envelope": anchored.ENVELOPE})
        return True
    if p == "/memory/anchored/whatlearned":         # "what do you remember?" = the alpha-table LOOKUP
        import clozn.memory.anchored as anchored
        h._json(200, anchored.whatlearned())        # pure read of the store; no provider, no generation
        return True
    return False


def try_post(h, p, body):
    if p == "/memory/anchored/fit":                 # fit a bag for an EXISTING card (by card_id)
        import clozn.memory.anchored as anchored
        import clozn.memory.cards as memory_cards
        card_id = str(body.get("card_id", "")).strip()
        if not card_id:
            h._json(400, {"error": "need a card_id"})
            return True
        card = memory_cards.get(card_id)
        if card is None:
            h._json(200, {"ok": False, "reason": "no such card"})
            return True
        provider = _provider()
        if provider is None:
            h._json(200, {"ok": False, "reason": "anchored fit needs a running engine with a J-lens "
                                                 "sidecar (CLOZN_ENGINE_QWEN_PORT + --jlens)"})
            return True
        try:
            k = int(body.get("k", anchored.K_DEFAULT))
        except (TypeError, ValueError):
            h._json(400, {"error": "k must be an integer"})
            return True
        res = anchored.fit_bag(card, provider, k=k,
                               lens_manifest_hash=anchored.lens_manifest_hash())
        if res.get("refused"):
            # The measured boundary, surfaced verbatim (style/rule cards, or too few content words).
            h._json(200, {"ok": False, "refused": True, "reason": res.get("reason"),
                          "card_id": card_id})
            return True
        bag = res["bag"]
        if anchored.put_bag(bag) is None:
            h._json(200, {"ok": False, "reason": "could not persist the bag"})
            return True
        h._json(200, {"ok": True, "bag": anchored.public_bag(bag),
                      "table": anchored.alpha_table(bag),      # the fit-preview receipt (design sec. 3)
                      "note": anchored.WHATLEARNED_NOTE})
        return True

    if p == "/memory/anchored/toggle":              # include/exclude from composition (bag kept intact)
        import clozn.memory.anchored as anchored
        card_id = str(body.get("card_id", "")).strip()
        if not card_id:
            h._json(400, {"error": "need a card_id"})
            return True
        bag = anchored.set_on(card_id, bool(body.get("on", True)))
        if bag is None:
            h._json(200, {"ok": False, "reason": "no anchored bag for that card"})
            return True
        h._json(200, {"ok": True, "bag": anchored.public_bag(bag)})
        return True

    if p == "/memory/anchored/delete_term":         # the real edit: one word-direction out, alphas refit
        import clozn.memory.anchored as anchored
        card_id = str(body.get("card_id", "")).strip()
        token = str(body.get("token", "")).strip()
        if not card_id or not token:
            h._json(400, {"error": "need a card_id and a token"})
            return True
        provider = _provider()
        if provider is None:
            h._json(200, {"ok": False, "reason": "delete_term refits the remaining alphas and needs a "
                                                 "running engine with a J-lens sidecar"})
            return True
        res = anchored.delete_term(card_id, token, provider)
        if not res.get("ok"):
            h._json(200, {"ok": False, "reason": res.get("reason")})
            return True
        out = {"ok": True, "bag": anchored.public_bag(res["bag"]) if res.get("bag") else None}
        if res.get("deleted_bag"):
            out["deleted_bag"] = True
            out["note"] = res.get("note")
        h._json(200, out)
        return True
    return False
