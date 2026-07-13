"""should_alert -- the "is this run worth interrupting for?" decision (AMBIENT_DELIVERY.md channel 2).

An interrupt is precious: it must fire ONLY when something is actually OFF, never on mere uncertainty.
So this is HARD-signals only -- a completed reply, however uncertain or full of close calls, does not
earn a desktop toast. (Uncertainty/close-calls are a passive DISPLAY signal -- the footer names them, the
studio tape shows them, the explicit branch-stability test escalates them -- but they are not an
interrupt: at the cheap level we can't tell a meaningful fork from a stylistic one, so interrupting on
one would cry wolf.) Pure over a run record; never raises.

HIGH (the only tier that interrupts): errored · truncated (hit the token limit mid-answer) · failed one
of your tiny-tests. All from what the journal already recorded, no model call.

Machine traffic (replay/receipt arms, branches -- source in _MACHINE_SOURCES, or any derived run with a
parent) is skipped: those are the studio's own probes, not something the user typed and is waiting on.
"""
from __future__ import annotations

from dataclasses import dataclass

# mirrors clozn/runs/actuary.py's own machine-source set (kept in sync there) -- these are studio probes.
_MACHINE_SOURCES = {"replay", "branch", "fork", "receipt", "receipts", "counterfactual", "rederive",
                    "swap_receipt", "anchored_receipt", "experiment"}


@dataclass
class Alert:
    run_id: str
    severity: str      # "high" | "medium"
    reason: str        # a stable slug: error | truncated | tiny_test_failed | low_mean_conf | shaky_span
    headline: str      # one short human line for the notification body
    prompt: str        # the run's prompt summary, for context


def is_organic(run: dict) -> bool:
    """A genuine user turn (not a studio probe): a known machine source, or any run with a parent, is
    not organic. Unknown sources are organic (fail open -- a new chat surface shouldn't go silent)."""
    if not isinstance(run, dict):
        return False
    if str(run.get("source") or "").lower() in _MACHINE_SOURCES:
        return False
    if run.get("parent_run_id"):
        return False
    return True


def should_alert(run: dict) -> Alert | None:
    """Alert | None for one run record (must carry error/finish_reason/tiny_tests). None = not worth
    interrupting for. HARD signals only -- uncertainty/close-calls never interrupt (see module docstring).
    Never raises."""
    try:
        if not is_organic(run):
            return None
        rid = str(run.get("id") or "")
        if not rid:
            return None
        prompt = str(run.get("prompt_summary") or "")[:80]

        if run.get("error"):
            return Alert(rid, "high", "error", "the run errored: " + str(run["error"])[:60], prompt)
        if run.get("finish_reason") == "length":
            return Alert(rid, "high", "truncated", "cut off mid-answer -- it hit the token limit", prompt)
        for tt in run.get("tiny_tests") or []:
            if isinstance(tt, dict) and tt.get("pass") is False:
                return Alert(rid, "high", "tiny_test_failed",
                             "failed your check: " + str(tt.get("name", "?"))[:50], prompt)
        return None                                        # completed, however uncertain -> no interrupt
    except Exception:
        return None                                        # a bad record is never worth a false alarm
