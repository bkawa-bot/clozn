"""Hard signals -- the "something is actually OFF" facts a run's footer flags (AMBIENT_DELIVERY.md).

These are the ONLY things worth a flag beyond a close call, and every one is a hard fact or a named check
that actually ran -- never a proxy/vibe (the fragility/stability terminology's binding rule). All free
from the recorded run, no model call. High precision by design: we would rather miss a soft problem than
raise a false one, so only unambiguous facts are here (fuzzy refusal detection / runtime-integrity
comparison are deliberately left out until they can be done without false alarms).

Signals (each a human phrase): errored · truncated (hit the token limit) · got stuck repeating (a real
degeneracy loop, reusing the anchored-memory loop guard) · empty reply · a fenced JSON block that doesn't
parse (real verification -- a check ran and failed).
"""
from __future__ import annotations

import json
import re

from clozn.memory.anchored import detect_loop

# mirrors clozn/runs/actuary.py's machine-source set -- studio probes, not user turns.
_MACHINE_SOURCES = {"replay", "branch", "fork", "receipt", "receipts", "counterfactual", "rederive",
                    "swap_receipt", "anchored_receipt", "experiment"}

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.S)


def is_organic(run: dict) -> bool:
    """A genuine user turn, not a studio probe (a known machine source or any derived run is not)."""
    if not isinstance(run, dict):
        return False
    if str(run.get("source") or "").lower() in _MACHINE_SOURCES:
        return False
    return not run.get("parent_run_id")


def hard_signals(run: dict | None) -> list[str]:
    """The list of hard-fact flags for this run (human phrases), or [] when nothing is off. Never raises."""
    try:
        if not isinstance(run, dict):
            return []
        out = []
        if run.get("error"):
            out.append("the run errored")
        if run.get("finish_reason") == "length":
            out.append("cut off mid-answer (hit the token limit)")
        resp = run.get("response")
        reply = str(resp) if resp is not None else ""
        trace = run.get("trace") if isinstance(run.get("trace"), dict) else {}
        toks = trace.get("tokens")
        pieces = toks if isinstance(toks, list) and toks else reply.split()
        # only flag EMPTY when the reply is present-and-empty -- an absent `response` key (a trace-only
        # fixture, a diffusion run that stores final_text elsewhere) is unknown, not empty.
        if resp is not None and not reply.strip() and not run.get("error"):
            out.append("returned an empty reply")
        elif detect_loop(pieces, window=8):
            out.append("got stuck repeating")
        m = _JSON_FENCE.search(reply)
        if m:
            try:
                json.loads(m.group(1))
            except Exception:
                out.append("the JSON block it returned doesn't parse")
        return out
    except Exception:
        return []
