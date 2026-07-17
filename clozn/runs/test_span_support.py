"""Model-free tests for the opt-in span SUPPORT attachment."""
from __future__ import annotations

from clozn.runs import span_support


def test_attach_preserves_spans_and_separates_entailed_from_not_entailed():
    spans = [{"text": "Kyoto gardens", "mean_conf": 0.8}, {"text": "Paris", "mean_conf": 0.9}]

    def matcher(claim, explanation):
        assert explanation == {"influences_active": {"cards": []}}
        ok = "Kyoto" in claim
        return {"supported": ok, "score": .91 if ok else .08, "threshold": .5,
                "matched_id": "card1" if ok else None, "closest_id": "card1",
                "contradiction": .02, "method": "nli-deberta-v3"}

    out, summary = span_support.attach(spans, {"influences_active": {"cards": []}}, matcher)
    assert out[0]["support"]["available"] is True and out[0]["support"]["entailed"] is True
    assert out[1]["support"]["available"] is True and out[1]["support"]["entailed"] is False
    assert summary["n_entailed"] == 1 and summary["n_not_entailed"] == 1
    assert "support" not in spans[0]                             # copied, never mutated


def test_nli_unavailable_is_not_mislabeled_unsupported():
    def unavailable(claim, explanation):
        return {"supported": False, "score": 0, "method": "nli-unavailable", "note": "checkpoint absent"}

    out, summary = span_support.attach([{"text": "claim"}], {}, unavailable)
    assert out[0]["support"]["available"] is False
    assert out[0]["support"]["entailed"] is None
    assert summary["available"] is False and summary["n_unavailable"] == 1


def test_matcher_errors_and_empty_text_degrade_per_span():
    def broken(claim, explanation):
        raise RuntimeError("boom")

    out, summary = span_support.attach([{"text": "claim"}, {"text": ""}], {}, broken)
    assert [x["support"]["method"] for x in out] == ["matcher-error", "no-text"]
    assert summary["n_unavailable"] == 2 and summary["available"] is False


def test_note_fences_support_from_fact_check_claims():
    assert "not external-source evidence" in span_support.NOTE
    assert "not a factual verdict" in span_support.NOTE
