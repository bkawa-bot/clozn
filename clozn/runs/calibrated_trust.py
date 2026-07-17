"""calibrated_trust.py -- map per-token/per-span confidence through the user's OWN journal-derived
reliability curve (actuary.py's organic Calibration), so a UI can shade text by "how often did MY past
runs at this confidence actually get kept?" instead of by raw softmax confidence alone.

This is ACCEPTANCE-PROXY calibration, the same stance actuary.py states plainly: a bin's trusted_rate
is the fraction of past runs at that confidence that were KEPT (not errored/truncated/test-failed/
re-rolled). It measures acceptance by the user and their tests, never verified correctness -- nothing
here upgrades that proxy into a stronger claim, and nothing here invents a curve where the journal has
no evidence (an empty or unscored calibration maps every confidence to an honest None).

Pure functions over an already-built `actuary.Calibration` -- no filesystem, no server, no model. The
route wiring (cache, HTTP shapes) lives in clozn/server/routes/journal.py; the segmentation whose spans
get annotated lives in confidence_spans.py (this module only ATTACHES to those spans, never re-segments).
"""
from __future__ import annotations

import math

from .actuary import Calibration, CalibrationBin

# A calibration-bin estimate resting on fewer than this many journal runs is flagged small_n: still
# reported (it is what the journal honestly holds), but marked so the UI can render it as weak evidence
# rather than a settled rate.
SMALL_N = 20
TRUTH_SMALL_N = 50

# The one sentence that must ride every trust_spans response -- verbatim, so the proxy language reaches
# the wire and the UI, not just this docstring.
NOTE = ("trusted_rate is the fraction of the user's own past runs at this confidence that were kept "
        "(accepted) — a proxy for reliability, NOT a fact-check. Small bins are weak evidence.")

TRUTH_NOTE = ("truth_correctness_estimate applies a scalar temperature fitted on labeled probes for the "
              "same model and score aggregate. It estimates correctness on that eval distribution; it "
              "does NOT verify this claim. Fewer than 50 labeled probes is weak evidence and must not alert.")


def _bin_for(c: float, bins: list[CalibrationBin]) -> CalibrationBin | None:
    """The bin whose [lo, hi) interval holds `c` -- with actuary.calibration's own edge convention that
    the TOP bin is closed on the right, so a confidence of exactly 1.0 lands in the last bin instead of
    falling off the end. None when `c` is outside every bin (never clamped into an edge bin: an
    out-of-range confidence gets no estimate rather than a fabricated one)."""
    for b in bins:
        if b.lo <= c < b.hi:
            return b
    if bins and c == bins[-1].hi:
        return bins[-1]
    return None


def _entry(conf: float | None, b: CalibrationBin | None) -> dict:
    """One mapping-result dict. A missing bin (or an EMPTY bin -- n == 0) yields trusted_rate_estimate
    None with bin_n 0: the journal has no evidence at this confidence, and None is the honest answer.
    small_n is True whenever the estimate rests on fewer than SMALL_N runs -- which includes every
    no-evidence case (0 < SMALL_N)."""
    n = b.n if b is not None else 0
    rate = b.trusted_rate if b is not None else None
    return {
        "conf": conf,
        "bin_lo": b.lo if b is not None else None,
        "bin_hi": b.hi if b is not None else None,
        "trusted_rate_estimate": (round(float(rate), 4) if rate is not None else None),
        "bin_n": int(n),
        "small_n": int(n) < SMALL_N,
    }


def has_curve(calibration: Calibration | None) -> bool:
    """Does this calibration hold ANY evidence to map against -- at least one scored run landing in at
    least one bin? False for None, an unscored journal, or all-empty bins: the callers' cue to answer
    available:false instead of mapping confidences through a curve that does not exist."""
    if calibration is None or not getattr(calibration, "bins", None):
        return False
    if not getattr(calibration, "n_scored", 0):
        return False
    return any(b.n for b in calibration.bins)


def trust_for(confidences: list[float], calibration: Calibration) -> list[dict]:
    """Map each confidence through the calibration's bins. Returns one dict PER INPUT, same order, same
    length -- a None/non-numeric/non-finite/out-of-range confidence still yields an aligned entry, just
    with trusted_rate_estimate None (the mapping never guesses and never drops a position).

    Each entry: {conf, bin_lo, bin_hi, trusted_rate_estimate, bin_n, small_n} -- see _entry. The
    estimate is the bin's trusted_rate: the fraction of the journal's runs in that bin that were kept
    (actuary.py's acceptance proxy), never a verified-correctness figure."""
    bins = list(getattr(calibration, "bins", None) or [])
    out = []
    for v in confidences or []:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            out.append(_entry(None, None))
            continue
        c = float(v)
        out.append(_entry(c, _bin_for(c, bins)))
    return out


def attach(spans: list[dict], calibration: Calibration) -> list[dict]:
    """confidence_spans.spans() output + the calibration mapping: a COPY of each span dict with
    trusted_rate_estimate / bin_n / small_n attached, keyed off the span's own mean_conf (the mean the
    segmentation already computed -- this function never re-derives it). Input spans are not mutated.
    A span with no readable mean_conf gets the same honest None estimate as any unmappable confidence."""
    spans = [s for s in (spans or []) if isinstance(s, dict)]
    mapped = trust_for([s.get("mean_conf") for s in spans], calibration)
    return [{**s, "trusted_rate_estimate": e["trusted_rate_estimate"],
             "bin_n": e["bin_n"], "small_n": e["small_n"]}
            for s, e in zip(spans, mapped)]


def attach_truth(spans: list[dict], saved: dict | None, run_model: str | None) -> tuple[list[dict], dict]:
    """Attach outcome-calibrated estimates from a saved ``clozn eval --save`` report.

    Application is intentionally strict: the saved model must exactly match the run model, the saved
    score aggregate must be ``min`` or ``mean`` (mapped to the span field of the same name), and the report
    must carry an available scalar-temperature fit. Any mismatch returns copied, untouched spans plus an
    explicit unavailable reason. Even a valid small-n fit is reported but flagged; the UI may show it as
    weak evidence but must not use it for an alert.
    """
    spans = [dict(s) for s in (spans or []) if isinstance(s, dict)]

    def unavailable(reason: str) -> tuple[list[dict], dict]:
        return spans, {"available": False, "reason": reason, "note": TRUTH_NOTE}

    if not isinstance(saved, dict):
        return unavailable("no outcome-grounded calibration saved — run `clozn eval --save`")
    saved_model = str(saved.get("model") or "").strip()
    actual_model = str(run_model or "").strip()
    if not saved_model or not actual_model:
        return unavailable("saved calibration or run is missing model provenance")
    if saved_model != actual_model:
        return unavailable(f"calibration model {saved_model!r} does not match run model {actual_model!r}")
    aggregate = str(saved.get("score") or "").strip()
    if aggregate not in ("min", "mean"):
        return unavailable("saved calibration is missing a supported score aggregate (min or mean)")
    temp = ((saved.get("report") or {}).get("temperature_scaling") or {})
    if not isinstance(temp, dict) or not temp.get("available"):
        return unavailable("saved truth report has no fitted temperature — rerun `clozn eval --save`")
    from clozn.eval.calibration import temperature_scale
    temperature = temp.get("temperature")
    if temperature_scale(0.5, temperature) is None:
        return unavailable("saved truth report carries an invalid fitted temperature")
    try:
        n = max(0, int(temp.get("n") or saved.get("n") or 0))
    except (TypeError, ValueError):
        return unavailable("saved truth report carries an invalid labeled-probe count")
    field = "min_conf" if aggregate == "min" else "mean_conf"
    out = []
    for span in spans:
        raw = span.get(field)
        estimate = temperature_scale(raw, temperature)
        source_score = (raw if isinstance(raw, (int, float)) and not isinstance(raw, bool)
                        and math.isfinite(float(raw)) else None)
        out.append({**span,
                    "truth_correctness_estimate": (round(estimate, 4) if estimate is not None else None),
                    "truth_source_score": source_score,
                    "truth_score_aggregate": aggregate,
                    "truth_n": n,
                    "truth_small_n": n < TRUTH_SMALL_N})
    return out, {
        "available": True,
        "model": saved_model,
        "set": saved.get("set"),
        "score_aggregate": aggregate,
        "temperature": temperature,
        "n": n,
        "small_n": n < TRUTH_SMALL_N,
        "note": TRUTH_NOTE,
    }
