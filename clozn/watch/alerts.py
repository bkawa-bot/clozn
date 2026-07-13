"""should_alert -- the "is this run worth interrupting for?" decision (AMBIENT_DELIVERY.md channel 2).

The whole value of an ambient alert is that it stays QUIET on the ordinary confident reply and only
fires when something is actually off -- otherwise it's noise you learn to ignore. So this is deliberately
conservative: a clean run returns None. Pure over a run record; never raises. Reuses the SAME confidence
signals as the receipt footer / /runs/<id>/spans, so an alert never disagrees with what the studio shows.

Two tiers, both from what the journal already recorded (no model call):
  HIGH   -- errored · truncated (hit the token limit mid-answer) · failed one of your tiny-tests.
  MEDIUM -- low confidence: mean below MEAN_CONF_FLOOR, or a genuinely shaky stretch (min below
            MIN_CONF_FLOOR, or several shaky tokens) -- "worth a second look", never "wrong".

Machine traffic (replay/receipt arms, branches -- source in _MACHINE_SOURCES, or any derived run with a
parent) is skipped: those are the studio's own probes, not something the user typed and is waiting on.
"""
from __future__ import annotations

from dataclasses import dataclass

from clozn.runs import receipt_footer

# tiers/thresholds -- tuned to stay quiet on ordinary replies (see module docstring). Deliberately
# stricter than the footer's "1 shaky span = worth a look": an ALERT interrupts you, so it needs more.
MEAN_CONF_FLOOR = 0.6      # a reply whose whole mean confidence is below this
MIN_CONF_FLOOR = 0.35      # a single very-uncertain token anywhere
SHAKY_TOKENS_MIN = 3       # this many shaky tokens total (a real wobble, not one hedged word)

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
    """Alert | None for one run record (must carry the fields the journal records: error/finish_reason/
    flags/tiny_tests/trace). None = not worth interrupting for. Never raises."""
    try:
        if not is_organic(run):
            return None
        rid = str(run.get("id") or "")
        if not rid:
            return None
        prompt = str(run.get("prompt_summary") or "")[:80]

        # --- HIGH ---
        if run.get("error"):
            return Alert(rid, "high", "error", "the run errored: " + str(run["error"])[:60], prompt)
        if run.get("finish_reason") == "length":
            return Alert(rid, "high", "truncated", "cut off mid-answer -- it hit the token limit", prompt)
        for tt in run.get("tiny_tests") or []:
            if isinstance(tt, dict) and tt.get("pass") is False:
                return Alert(rid, "high", "tiny_test_failed",
                             "failed your check: " + str(tt.get("name", "?"))[:50], prompt)

        # --- MEDIUM (confidence) ---
        s = receipt_footer.summary(run)
        if not s["n_tokens"]:
            return None                                    # no trace -> no confidence signal to judge on
        trace = run.get("trace") if isinstance(run.get("trace"), dict) else {}
        confs = [float(c) for c in (trace.get("confidence") or []) if isinstance(c, (int, float))]
        min_conf = min(confs) if confs else None
        if s["mean_conf"] is not None and s["mean_conf"] < MEAN_CONF_FLOOR:
            return Alert(rid, "medium", "low_mean_conf",
                         f"low confidence throughout (mean {s['mean_conf']:.2f})", prompt)
        if (min_conf is not None and min_conf < MIN_CONF_FLOOR) or s["n_shaky"] >= SHAKY_TOKENS_MIN:
            where = f"{s['n_shaky']} shaky token{'s' if s['n_shaky'] != 1 else ''}"
            return Alert(rid, "medium", "shaky_span", f"a shaky stretch -- {where}", prompt)
        return None
    except Exception:
        return None                                        # a bad record is never worth a false alarm
