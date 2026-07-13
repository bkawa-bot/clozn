"""Tests for the in-band receipt footer (receipt_footer.py) -- pure, synthetic runs, no server."""
from __future__ import annotations

from clozn.runs import receipt_footer


def _run(confs):
    return {"trace": {"tokens": [f"t{i}" for i in range(len(confs))], "confidence": list(confs)}}


def test_summary_counts_shaky_and_means():
    s = receipt_footer.summary(_run([0.95, 0.9, 0.3, 0.92]))   # one shaky token (< LOW_CONF)
    assert s["n_tokens"] == 4
    assert s["mean_conf"] == round((0.95 + 0.9 + 0.3 + 0.92) / 4, 2)
    assert s["n_shaky"] >= 1


def test_footer_flags_shaky_span_with_link():
    f = receipt_footer.footer(_run([0.95, 0.9, 0.3, 0.2]), "http://host:8090/r/run_x")
    assert "http://host:8090/r/run_x" in f
    assert "worth a look" in f
    assert receipt_footer.MARK in f
    assert f.startswith("\n\n---\n")


def test_footer_confident_when_no_shaky():
    f = receipt_footer.footer(_run([0.95, 0.92, 0.9]), "http://h/r/x")
    assert "confident throughout" in f
    assert "worth a look" not in f


def test_footer_empty_when_no_trace():
    # diffusion / a stream that logged nothing -> no fabricated stat, no footer at all
    assert receipt_footer.footer({"trace": {"tokens": [], "confidence": []}}, "http://h/r/x") == ""
    assert receipt_footer.footer({}, "http://h/r/x") == ""
    assert receipt_footer.footer(None, "http://h/r/x") == ""


def test_summary_never_raises_on_junk():
    for junk in (None, {}, {"trace": "nope"}, {"trace": {"tokens": None}}):
        s = receipt_footer.summary(junk)
        assert s["n_tokens"] == 0 and s["mean_conf"] is None
