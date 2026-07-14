"""memory_mode -- which mechanism carries the studio's memory cards.

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

BLOCK STYLE: a second, independent persisted setting, "block_style" -- "soft" (default,
unchanged) | "strict". Both are prompt-mode wording variants of the SAME rules; block_style never
affects "internalized" mode (the prefix has no prompt block to reword). The measured problem
(measured in an A/B follow-up): the soft block's "...use it naturally to tailor how you
respond" phrasing is a distillation-target wording that a strong instruction-follower (7B) over-satisfies
but a 1.5B under-fires on plain neutral probes -- two of four traits (space, question) came out
PREFIX-STRONGER at 1.5B, inverting the 7B verdict (PROMPT >= PREFIX everywhere). "strict" states the same
rules as direct imperatives (no "naturally"/"tailor" hedge) to test whether closing that soft-wording gap
closes the inversion. Soft stays the default and stays byte-identical to consolidate()'s sys_rule (the
lockstep test enforces this at the "soft" style only -- strict is not a distillation target and is free
to reword).

Mirrors memory_cards.py: stdlib only (torch-free, so replay.py stays model-free-testable), module-level
path globals tests can point at a tmp dir, and IO that NEVER raises -- a broken settings file degrades
to the migration default, never to a crashed request.
"""
from __future__ import annotations

import json
import os

from clozn._io import atomic_write_json

_CLOZN = os.path.expanduser("~/.clozn")
SETTINGS_PATH = os.path.join(_CLOZN, "studio_settings.json")

# A trained prefix on disk == a live personality someone invested minutes of TTT in. Its existence is
# the migration signal: with no explicit choice recorded, keep serving it (internalized) rather than
# silently swapping the mechanism under the user. Module global so tests can isolate.
LEGACY_PREFIX_PATHS = [os.path.join(_CLOZN, "studio_memory.pt"),
                       os.path.join(_CLOZN, "studio_dream_memory.pt")]

MODES = ("prompt", "internalized")

BLOCK_STYLES = ("soft", "strict")
DEFAULT_BLOCK_STYLE = "soft"      # unchanged wording; strict is opt-in (see module docstring)


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


def get_block_style() -> str:
    """The active prompt-block wording ("soft" | "strict"): the persisted choice if valid, else
    DEFAULT_BLOCK_STYLE ("soft" -- byte-identical to today's wording, no behaviour change for anyone
    who hasn't opted in). Independent of memory_mode -- this only matters when mode == "prompt"."""
    style = get_setting("block_style")
    return style if style in BLOCK_STYLES else DEFAULT_BLOCK_STYLE


def set_block_style(style: str) -> bool:
    """Persist the block-style choice. False on an invalid style or IO failure (never raises)."""
    if style not in BLOCK_STYLES:
        return False
    return set_setting("block_style", style)


def get_setting(key: str, default=None):
    """Read one settings key; `default` when missing/unreadable. (Prompt mode parks small scalars here
    that internalized mode kept inside the .pt -- e.g. memory_strength, which .pt-save refuses without
    a trained prefix and a fresh prompt-mode install never has one.)"""
    return _load_settings().get(key, default)


def set_setting(key: str, value) -> bool:
    """Persist one settings key (merge-write); False on IO failure (never raises).

    Atomic (see clozn._io): a non-serializable `value` raises out of json.dumps before the real
    SETTINGS_PATH is ever opened for writing, and the on-disk write is temp-file-then-rename, so a bad
    call here can never truncate/corrupt the settings file -- every other already-persisted key (mode,
    block_style, memory_facts, active_profile, ...) survives untouched."""
    try:
        settings = _load_settings()
        settings[key] = value
        atomic_write_json(SETTINGS_PATH, settings)
        return True
    except Exception:
        return False


def active_cards(exclude_ids=()) -> list[dict] | None:
    """The ACTIVE memory cards as [{id, text}], minus exclude_ids (replay's per-card ablation) -- the
    prompt block's source of truth. [] when the store is simply empty; None if memory_cards itself is
    unavailable, so callers can tell "no cards" from "no store" and keep their own fallbacks."""
    exclude = {str(i) for i in (exclude_ids or ())}
    try:
        from clozn.memory import cards as memory_cards
        return [{"id": c.get("id"), "text": c["text"]}
                for c in memory_cards.list_cards(status="active")
                if c.get("text") and str(c.get("id")) not in exclude]
    except Exception:
        return None


def compile_prompt_block(texts: list[str], style: str | None = None) -> str:
    """The active card texts as ONE system block -- what prompt mode prepends to every gated-in turn.

    `style` selects the wording: "soft" (default) or "strict". None (the default, and
    every pre-existing call site's behaviour) reads the persisted setting via get_block_style() -- so
    this signature is back-compatible: no caller needs to change to keep behaving exactly as before.
    Pass an explicit style to override the setting (e.g. the A/B rig comparing both on one process).

    "soft" -- the ORIGINAL wording, MUST stay verbatim-identical to SelfTeach.consolidate's `sys_rule`
    (self_teach_server.py): that string is the distillation target the internalized prefix is trained to
    imitate, so keeping them in lockstep is what makes the two modes behaviourally comparable (the
    black-box A/B rests on it). Never reword "soft" -- add a new style instead.

    "strict" -- the SAME rules as direct imperatives, no "use it naturally to tailor" hedge. Measured
    motivation (the measured A/B follow-up): that softer framing is under-satisfied by a
    1.5B on plain neutral probes (2/4 traits inverted vs the trained prefix there, though 7B satisfied it
    fine) -- strict tests whether stating the rules as instructions closes that gap. Not a distillation
    target; free to reword independently of "soft"/consolidate's sys_rule.

    Empty/blank-only input -> "" (the caller omits the block entirely) regardless of style. Pure;
    preserves text order."""
    rules = [str(t).strip() for t in (texts or []) if str(t or "").strip()]
    if not rules:
        return ""
    resolved = get_block_style() if style is None else style
    if resolved == "strict":
        return ("Follow these facts and rules about the user exactly, in every reply, without exception:\n"
                + "\n".join("- " + r for r in rules))
    return ("You are a helpful assistant talking with a returning user. Here is what you know "
            "about them; use it naturally to tailor how you respond:\n"
            + "\n".join("- " + r for r in rules))
