"""memory_mode -- which mechanism carries the studio's memory cards (notes/MEMORY_MODE_SWAP_SPEC.md).

Two modes, one persisted setting in ~/.clozn/studio_settings.json:

  "prompt"       (default for FRESH installs) -- the active card texts are compiled into one system
                 block and prepended to the chat, topic-gated per turn. Card edits are instant (no
                 retrain), per-card ablation is real, and the card text IS what's applied, verbatim.
  "internalized" -- today's trained soft-prefix path, untouched: cards drive consolidate() (a ~4-5 min
                 TTT retrain per active-set change). Kept as the research mode (the self-audit
                 experiments REQUIRE a non-text memory) and the context-constrained fallback.

Migration rule (don't silently change a live personality): when no mode was ever chosen, an existing
trained prefix on disk (~/.clozn/studio_memory.pt / studio_dream_memory.pt) resolves to "internalized"
until the user toggles; a fresh install resolves to "prompt".

Mirrors memory_cards.py: stdlib only (torch-free, so replay.py stays model-free-testable), module-level
path globals tests can point at a tmp dir, and IO that NEVER raises -- a broken settings file degrades
to the migration default, never to a crashed request.
"""
from __future__ import annotations

import json
import os

_CLOZN = os.path.expanduser("~/.clozn")
SETTINGS_PATH = os.path.join(_CLOZN, "studio_settings.json")

# A trained prefix on disk == a live personality someone invested minutes of TTT in. Its existence is
# the migration signal: with no explicit choice recorded, keep serving it (internalized) rather than
# silently swapping the mechanism under the user. Module global so tests can isolate.
LEGACY_PREFIX_PATHS = [os.path.join(_CLOZN, "studio_memory.pt"),
                       os.path.join(_CLOZN, "studio_dream_memory.pt")]

MODES = ("prompt", "internalized")


def _load_settings() -> dict:
    """The whole settings dict; {} if missing or unreadable (never raises)."""
    try:
        if not os.path.isfile(SETTINGS_PATH):
            return {}
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_mode() -> str:
    """The active memory mode: the persisted choice if one exists, else the migration default
    ("internalized" iff a trained prefix already lives on disk, "prompt" for a fresh install)."""
    mode = _load_settings().get("memory_mode")
    if mode in MODES:
        return mode
    try:
        if any(os.path.isfile(p) for p in LEGACY_PREFIX_PATHS):
            return "internalized"
    except Exception:
        pass
    return "prompt"


def set_mode(mode: str) -> bool:
    """Persist the mode choice (merging into any other settings keys). False on an invalid mode or IO
    failure (never raises) -- the caller reports, the request survives."""
    if mode not in MODES:
        return False
    return set_setting("memory_mode", mode)


def get_setting(key: str, default=None):
    """Read one settings key; `default` when missing/unreadable. (Prompt mode parks small scalars here
    that internalized mode kept inside the .pt -- e.g. memory_strength, which .pt-save refuses without
    a trained prefix and a fresh prompt-mode install never has one.)"""
    return _load_settings().get(key, default)


def set_setting(key: str, value) -> bool:
    """Persist one settings key (merge-write); False on IO failure (never raises)."""
    try:
        settings = _load_settings()
        settings[key] = value
        os.makedirs(os.path.dirname(SETTINGS_PATH), exist_ok=True)
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(settings, f)
        return True
    except Exception:
        return False


def active_cards(exclude_ids=()) -> list[dict] | None:
    """The ACTIVE memory cards as [{id, text}], minus exclude_ids (replay's per-card ablation) -- the
    prompt block's source of truth. [] when the store is simply empty; None if memory_cards itself is
    unavailable, so callers can tell "no cards" from "no store" and keep their own fallbacks."""
    exclude = {str(i) for i in (exclude_ids or ())}
    try:
        import memory_cards
        return [{"id": c.get("id"), "text": c["text"]}
                for c in memory_cards.list_cards(status="active")
                if c.get("text") and str(c.get("id")) not in exclude]
    except Exception:
        return None


def compile_prompt_block(texts: list[str]) -> str:
    """The active card texts as ONE system block -- what prompt mode prepends to every gated-in turn.

    The wording MUST stay verbatim-identical to SelfTeach.consolidate's `sys_rule` (self_teach_server.py):
    that string is the distillation target the internalized prefix is trained to imitate, so keeping them
    in lockstep is what makes the two modes behaviourally comparable (the black-box A/B rests on it).
    Empty/blank-only input -> "" (the caller omits the block entirely). Pure; preserves text order."""
    rules = [str(t).strip() for t in (texts or []) if str(t or "").strip()]
    if not rules:
        return ""
    return ("You are a helpful assistant talking with a returning user. Here is what you know "
            "about them; use it naturally to tailor how you respond:\n"
            + "\n".join("- " + r for r in rules))
