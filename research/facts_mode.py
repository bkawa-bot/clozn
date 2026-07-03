"""facts_mode -- the on/off gate + per-profile store paths for the slot-memory FACTS tier.

The facts tier (slotmem_qwen.SlotMem: centered-key addressing, surprise-gated writes, confidence-gate
abstention -- proven in slotmem_qwen_findings.md, 0.95 flat to N=200) is the studio's EXPLICIT,
editable, honest-about-ignorance fact store, distinct from the trait-card memory (memory_mode.py) that
carries dispositions. A fact is a verbatim (cue -> answer) pair, stored inside the model as a key/value
slot; the store is a print-able list.

CRITICAL LATENCY RULE (from NEXT_STEPS #5): a slot read is an EXTRA forward at the tap layer -- it must
stay OFF the 7B hot path until measured. So the whole feature is gated behind ONE persisted setting,
`memory_facts` in ~/.clozn/studio_settings.json, DEFAULT OFF. Only when it is "on" does the server build
the slot substrate, auto-write from conversation, or emit a read receipt (and it logs the per-turn slot_ms
so the overhead is honest). Off => zero cost, byte-identical replies, the prior behaviour.

Stores are PER PROFILE (isolation by construction, matching profiles.py): ~/.clozn/profiles/<name>.slots.pt
via SlotMem.save/load. "default" is the store when no profile has been switched to.

Mirrors memory_mode.py: stdlib only (torch-free, so this stays model-free-testable), the setting lives in
the SAME studio_settings.json (one small file, not a new one -- reuses memory_mode's get/set), module-level
path globals tests can repoint, and IO that NEVER raises.
"""
from __future__ import annotations

import os
import re

import memory_mode  # the single settings file (studio_settings.json) + its never-raise get/set helpers

# Where per-profile slot stores live. A module global so tests can repoint it at a tmp dir.
PROFILES_DIR = os.path.join(os.path.expanduser("~"), ".clozn", "profiles")

# The tap layer for the studio's slot store. 18/28 is the validated default (slotmem_qwen_findings.md:
# L14 reads too lexical, L22 loses verbatim control; 18 is the band's sweet spot on this exact 7B-nf4).
# A store written at layer L refuses to load into a SlotMem tapping another (keys are residuals OF a layer),
# so this is the one place the studio's layer choice is named.
LAYER = 18

_ENABLED_KEY = "memory_facts"
# store filenames are profile names -> slug-safe (profiles.py enforces the same shape on bundle names).
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


def enabled() -> bool:
    """Is the facts tier ON? Default OFF (the latency rule) -- absent/garbage setting => False. Accepts a
    bool or the strings "on"/"off"/"true"/"false"/"1"/"0" (the UI persists a bool; be liberal reading)."""
    v = memory_mode.get_setting(_ENABLED_KEY, False)
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("on", "true", "1", "yes")


def set_enabled(on: bool) -> bool:
    """Persist the on/off choice into studio_settings.json (merge-write). False on IO failure (never
    raises) -- the caller reports, the request survives."""
    return memory_mode.set_setting(_ENABLED_KEY, bool(on))


def store_path(profile: str | None) -> str:
    """The .slots.pt store for a profile (per-profile isolation). None/blank/unknown-shape -> "default".
    Never touches disk; pure path construction so tests need no fixtures."""
    name = str(profile or "").strip()
    if not _NAME_RE.match(name):
        name = "default"
    return os.path.join(PROFILES_DIR, name + ".slots.pt")
