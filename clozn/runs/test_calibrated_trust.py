"""Tests for calibrated_trust.py -- pure mapping over SYNTHETIC Calibration objects. No filesystem,
no server, no journal files: every curve here is built by hand from actuary's dataclasses."""
from __future__ import annotations

from clozn.runs import calibrated_trust as ct
from clozn.runs.actuary import Calibration, CalibrationBin


def _cal(cells):
    """A synthetic 10-bin Calibration from [(trusted_rate|None, n), ...] (one pair per bin)."""
    assert len(cells) == 10
    bins = []
    for i, (rate, n) in enumerate(cells):
        lo, hi = i / 10, (i + 1) / 10
        mc = (lo + hi) / 2 if n else None
        gap = (mc - rate) if (n and rate is not None) else None
        bins.append(CalibrationBin(lo=lo, hi=hi, n=n, trusted_rate=rate if n else None,
                                   mean_conf=mc, gap=gap))
    n_scored = sum(n for _, n in cells)
    return Calibration(bins=bins, n_runs=n_scored, n_scored=n_scored,
                       ece_proxy=0.1 if n_scored else None)


# A monotone curve: acceptance rises with confidence, every bin comfortably above SMALL_N.
_MONOTONE = _cal([(i / 10 + 0.05, 30) for i in range(10)])


# ---- mapping correctness per bin -------------------------------------------

def test_maps_each_conf_to_its_own_bin():
    out = ct.trust_for([0.05, 0.55, 0.95], _MONOTONE)
    assert len(out) == 3
    for e, (lo, rate) in zip(out, [(0.0, 0.05), (0.5, 0.55), (0.9, 0.95)]):
        assert e["bin_lo"] == lo and e["bin_hi"] == round(lo + 0.1, 1)
        assert e["trusted_rate_estimate"] == round(rate, 4)
        assert e["bin_n"] == 30
        assert e["small_n"] is False


def test_monotone_curve_stays_monotone_through_the_mapping():
    confs = [i / 10 + 0.05 for i in range(10)]
    rates = [e["trusted_rate_estimate"] for e in ct.trust_for(confs, _MONOTONE)]
    assert rates == sorted(rates)


def test_bin_boundary_lands_in_the_upper_bin():
    e = ct.trust_for([0.5], _MONOTONE)[0]                # lo <= c < hi -> [0.5, 0.6), not [0.4, 0.5)
    assert e["bin_lo"] == 0.5


def test_conf_exactly_one_lands_in_the_top_bin():
    e = ct.trust_for([1.0], _MONOTONE)[0]
    assert e["bin_lo"] == 0.9 and e["bin_hi"] == 1.0
    assert e["trusted_rate_estimate"] == 0.95


def test_out_of_range_conf_gets_no_estimate():
    for bad in (-0.1, 1.5):
        e = ct.trust_for([bad], _MONOTONE)[0]
        assert e["trusted_rate_estimate"] is None and e["bin_n"] == 0 and e["small_n"] is True


# ---- small_n flagging -------------------------------------------------------

def test_small_n_flags_thin_bins():
    cells = [(0.5, 30)] * 10
    cells[3] = (0.5, 5)        # thin
    cells[4] = (0.5, 19)       # just under the line
    cells[5] = (0.5, 20)       # exactly at the line -> NOT small
    cal = _cal(cells)
    thin, under, at = ct.trust_for([0.35, 0.45, 0.55], cal)
    assert thin["small_n"] is True and thin["bin_n"] == 5
    assert under["small_n"] is True and under["bin_n"] == 19
    assert at["small_n"] is False and at["bin_n"] == 20
    # a thin bin still REPORTS its rate -- flagged, not hidden
    assert thin["trusted_rate_estimate"] == 0.5


# ---- empty bins / empty calibration ----------------------------------------

def test_empty_bin_yields_none_estimate():
    cells = [(0.5, 30)] * 10
    cells[2] = (None, 0)
    cal = _cal(cells)
    e = ct.trust_for([0.25], cal)[0]
    assert e["trusted_rate_estimate"] is None
    assert e["bin_n"] == 0 and e["small_n"] is True
    assert e["bin_lo"] == 0.2                            # the bin was found; it just holds no evidence


def test_empty_calibration_has_no_curve():
    empty = Calibration(bins=[], n_runs=0, n_scored=0, ece_proxy=None)
    all_empty = _cal([(None, 0)] * 10)
    assert ct.has_curve(empty) is False
    assert ct.has_curve(all_empty) is False
    assert ct.has_curve(None) is False
    assert ct.has_curve(_MONOTONE) is True
    # mapping through a curveless calibration never invents: every entry is an honest None
    for e in ct.trust_for([0.1, 0.9], empty) + ct.trust_for([0.1, 0.9], all_empty):
        assert e["trusted_rate_estimate"] is None and e["bin_n"] == 0 and e["small_n"] is True


# ---- missing / None / junk confidences -------------------------------------

def test_none_and_junk_confs_stay_aligned():
    out = ct.trust_for([None, "0.9", float("nan"), True, 0.95], _MONOTONE)
    assert len(out) == 5                                  # one entry per input, order preserved
    for e in out[:4]:                                     # None / string / nan / bool -> no estimate
        assert e["conf"] is None and e["trusted_rate_estimate"] is None and e["small_n"] is True
    assert out[4]["trusted_rate_estimate"] == 0.95        # the real one still maps


def test_empty_input_maps_to_empty_output():
    assert ct.trust_for([], _MONOTONE) == []
    assert ct.trust_for(None, _MONOTONE) == []


# ---- attaching to confidence spans ------------------------------------------

def test_attach_adds_trust_fields_and_keeps_span_fields():
    spans = [
        {"start": 0, "end": 3, "text": "Sure.", "band": "strong", "mean_conf": 0.95,
         "min_conf": 0.9, "n_tokens": 4, "hesitations": 0},
        {"start": 4, "end": 6, "text": " maybe", "band": "shaky", "mean_conf": 0.35,
         "min_conf": 0.2, "n_tokens": 3, "hesitations": 3},
    ]
    out = ct.attach(spans, _MONOTONE)
    assert len(out) == 2
    strong, shaky = out
    assert strong["trusted_rate_estimate"] == 0.95 and strong["bin_n"] == 30 and strong["small_n"] is False
    assert shaky["trusted_rate_estimate"] == 0.35
    for orig, got in zip(spans, out):                    # every segmentation field survives untouched
        for k, v in orig.items():
            assert got[k] == v
    assert "trusted_rate_estimate" not in spans[0]       # inputs were copied, never mutated


def test_attach_handles_empty_and_conf_less_spans():
    assert ct.attach([], _MONOTONE) == []
    out = ct.attach([{"text": "x"}], _MONOTONE)          # no mean_conf at all
    assert out[0]["trusted_rate_estimate"] is None and out[0]["small_n"] is True


# ---- the note ----------------------------------------------------------------

def test_note_text_is_the_required_proxy_language():
    assert ct.NOTE == ("trusted_rate is the fraction of the user's own past runs at this confidence "
                       "that were kept (accepted) — a proxy for reliability, NOT a fact-check. "
                       "Small bins are weak evidence.")
