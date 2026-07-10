"""Dial catalog helpers and preference-to-dial routing."""
from __future__ import annotations

import re

from . import axes


def suggest_dial_for_preference(text: str) -> dict | None:
    """Suggest a dial for style/tone preference text, otherwise return None.

    This is deterministic and transparent: it scans a preference for known style cues, chooses the
    earliest match, flips bare single-word cues when they are preceded by reducer language, and caps the
    returned magnitude to the target axis's safe maximum.
    """
    if not text or not isinstance(text, str):
        return None
    hay = text.lower()
    best = None
    for phrase, axis, sign in axes._DIAL_LEXICON:
        m = re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", hay)
        if m is None:
            continue
        key = (m.start(), -len(phrase))
        if best is None or key < best[0]:
            best = (key, phrase, axis, sign, m.start())
    if best is None:
        return None
    _, phrase, axis, sign, start = best
    if " " not in phrase and "-" not in phrase:
        preceding = re.findall(r"[\w']+", hay[:start])[-2:]
        if any(w in axes._DIAL_REDUCERS or w.endswith("n't") for w in preceding):
            sign = -sign
    ax = axes.AXES.get(axis, {})
    axis_max = float(ax.get("max", 1.5))
    mag = min(axes._DIAL_DEFAULT_MAG, axis_max)
    poles = ax.get("poles", (axis, axis))
    pole_label = poles[0] if sign > 0 else poles[1]
    return {"axis": axis, "value": round(sign * mag, 4), "pole_label": pole_label}
