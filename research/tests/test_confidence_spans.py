"""test_confidence_spans -- model-free tests for research/confidence_spans.py (confidence SPANS: reshape a
stored run's per-token trace into a handful of contiguous confidence-band regions, so a long reply's
certainty has a legible SHAPE instead of one dot per token).

Drives confidence_spans.spans()/summarize() directly against fixture dicts. The main multi-band case goes
through a REAL runlog.record() + get_run() round trip (mirrors test_run_timeline.py's `store` fixture) so
the trace shape is byte-for-byte what a real logging path persists; the targeted edge cases (a single
band-change, a lone sentence boundary, missing confidence, malformed input) are hand-built dicts, since
confidence_spans only ever reads run["trace"]["tokens"/"confidence"] -- no need to round-trip the rest of
the run schema to exercise those.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

import confidence_spans   # noqa: E402
import runlog              # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Redirect runlog's run store to a tmp dir for the duration of one test (mirrors test_run_timeline.py)."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return runlog


# --------------------------------------------------------------------------------------- fixture: multi-band reply

def test_multi_band_reply_splits_into_bounded_spans_with_correct_stats(store):
    # "The sky is blue. Maybe it will rain later." -- strong, then a shaky dip, an "okay" recovery, strong again.
    tokens = ["The", " sky", " is", " blue", ".", " Maybe", " it", " will", " rain", " later", "."]
    confidence = [0.95, 0.90, 0.92, 0.88, 0.99, 0.45, 0.30, 0.55, 0.60, 0.85, 0.97]
    rid = store.record(source="cli", messages=[{"role": "user", "content": "weather?"}],
                       response="The sky is blue. Maybe it will rain later.",
                       trace={"tokens": tokens, "confidence": confidence})
    run = store.get_run(rid)

    result = confidence_spans.spans(run)
    assert [(s["start"], s["end"], s["band"]) for s in result] == [
        (0, 4, "strong"), (5, 6, "shaky"), (7, 8, "okay"), (9, 10, "strong"),
    ]

    # text reconstruction is a verbatim join -- tokens already carry their own leading spaces
    assert "".join(s["text"] for s in result) == "".join(tokens)
    assert result[0]["text"] == "The sky is blue."
    assert result[1]["text"] == " Maybe it"
    assert result[2]["text"] == " will rain"
    assert result[3]["text"] == " later."

    assert [s["n_tokens"] for s in result] == [5, 2, 2, 2]

    strong1, shaky, okay, strong2 = result
    assert strong1["mean_conf"] == 0.928 and strong1["min_conf"] == 0.88
    assert shaky["mean_conf"] == 0.375 and shaky["min_conf"] == 0.3
    assert okay["mean_conf"] == 0.575 and okay["min_conf"] == 0.55
    assert strong2["mean_conf"] == 0.91 and strong2["min_conf"] == 0.85

    # hesitations: only the shaky span's tokens are below LOW_CONF (0.5)
    assert [s["hesitations"] for s in result] == [0, 2, 0, 0]


# --------------------------------------------------------------------------------------- fixture: band vs sentence

def test_band_change_mid_sentence_splits_a_span(store):
    """No sentence punctuation anywhere here -- only the confidence band changes, and that alone is enough
    to start a new span."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="Hi there maybe",
                       trace={"tokens": ["Hi", " there", " maybe"], "confidence": [0.9, 0.9, 0.4]})
    result = confidence_spans.spans(store.get_run(rid))
    assert [(s["start"], s["end"], s["band"], s["text"]) for s in result] == [
        (0, 1, "strong", "Hi there"),
        (2, 2, "shaky", " maybe"),
    ]


def test_sentence_boundary_splits_a_span_even_when_the_band_is_unchanged(store):
    """Same band (strong) the whole way through -- but a sentence ends after token 1, so the span still
    splits there rather than running the two sentences together."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="Yes. Yes",
                       trace={"tokens": ["Yes", ".", " Yes"], "confidence": [0.9, 0.9, 0.9]})
    result = confidence_spans.spans(store.get_run(rid))
    assert [(s["start"], s["end"], s["band"], s["text"]) for s in result] == [
        (0, 1, "strong", "Yes."),
        (2, 2, "strong", " Yes"),
    ]


def test_closing_quote_after_terminal_punctuation_in_the_same_token_still_ends_the_sentence(store):
    """The sentence-end check allows a trailing closing quote/bracket right after the [.!?] WITHIN one
    token's own piece -- a piece like '."' (period + closing quote committed together) ends a sentence,
    not just a bare '.'"""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response='"Stop." Ok',
                       trace={"tokens": ['"Stop', '."', " Ok"], "confidence": [0.9, 0.9, 0.9]})
    result = confidence_spans.spans(store.get_run(rid))
    # '."' ends the sentence on its own (period + trailing close-quote, same token) -> split right after it
    assert [(s["start"], s["end"], s["text"]) for s in result] == [(0, 1, '"Stop."'), (2, 2, " Ok")]


# --------------------------------------------------------------------------------------- fixture: missing confidence

def test_missing_confidence_reads_as_1_0_and_strong_not_as_zero_or_shaky(store):
    """confidence shorter than tokens -- the trailing token has no recorded confidence at all. It must read
    as certain (1.0), not as zero/uncertain, and must merge into the surrounding strong span."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="Hello world",
                       trace={"tokens": ["Hello", " world"], "confidence": [0.9]})
    result = confidence_spans.spans(store.get_run(rid))
    assert len(result) == 1
    assert result[0]["band"] == "strong"
    assert result[0]["mean_conf"] == 0.95          # (0.9 + 1.0) / 2
    assert result[0]["min_conf"] == 0.9
    assert result[0]["hesitations"] == 0


def test_wholly_absent_confidence_list_reads_every_token_as_1_0_strong(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="Hmm sure",
                       trace={"tokens": ["Hmm", " sure"]})    # no "confidence" key at all
    result = confidence_spans.spans(store.get_run(rid))
    assert len(result) == 1
    assert result[0]["band"] == "strong"
    assert result[0]["mean_conf"] == 1.0
    assert result[0]["min_conf"] == 1.0


# --------------------------------------------------------------------------------------- fixture: no-trace / empty

@pytest.mark.parametrize("run", [
    {},
    {"id": "run_bare"},
    {"id": "run_x", "trace": {}},
    {"id": "run_x", "trace": {"tokens": []}},
    {"id": "run_x", "trace": {"confidence": [0.9, 0.2]}},   # confidence with no tokens at all -> nothing to band
])
def test_no_trace_or_empty_tokens_returns_empty_list(run):
    assert confidence_spans.spans(run) == []


@pytest.mark.parametrize("garbage", [None, "not a dict", 42, [], ["also", "not", "a", "dict"]])
def test_non_dict_run_returns_empty_list(garbage):
    assert confidence_spans.spans(garbage) == []


def test_never_raises_on_a_maximally_malformed_but_dict_shaped_run():
    run = {"id": "run_weird", "trace": {"tokens": ["a", "b", "c"], "confidence": "not-a-list"}}
    assert confidence_spans.spans(run) == [
        {"start": 0, "end": 2, "text": "abc", "band": "strong", "mean_conf": 1.0, "min_conf": 1.0,
         "n_tokens": 3, "hesitations": 0},
    ]                        # unusable confidence -> every token defaults to 1.0/strong, one span, no crash


def test_never_raises_when_confidence_entries_are_not_individually_numeric():
    run = {"trace": {"tokens": ["a", "b"], "confidence": [{"nope": True}, None]}}
    result = confidence_spans.spans(run)
    assert result[0]["mean_conf"] == 1.0 and result[0]["min_conf"] == 1.0    # both unparseable -> 1.0


# --------------------------------------------------------------------------------------------- summarize()

def test_summarize_of_empty_spans_is_empty_string():
    assert confidence_spans.summarize([]) == ""


def test_summarize_with_no_shaky_span_is_confident_throughout():
    spans = [{"band": "strong", "start": 0, "n_tokens": 10, "min_conf": 0.9},
             {"band": "okay", "start": 10, "n_tokens": 10, "min_conf": 0.6}]
    assert confidence_spans.summarize(spans) == "Confident throughout."


def test_summarize_places_the_weakest_shaky_span_in_the_opening_third():
    spans = [{"band": "shaky", "start": 0, "n_tokens": 5, "min_conf": 0.2},
             {"band": "strong", "start": 5, "n_tokens": 25, "min_conf": 0.9}]
    assert confidence_spans.summarize(spans) == "Mostly steady, but 1 shaky span (weakest in the opening)."


def test_summarize_places_the_weakest_shaky_span_in_the_middle_third():
    spans = [{"band": "strong", "start": 0, "n_tokens": 10, "min_conf": 0.9},
             {"band": "shaky", "start": 10, "n_tokens": 10, "min_conf": 0.3},
             {"band": "strong", "start": 20, "n_tokens": 10, "min_conf": 0.9}]
    assert confidence_spans.summarize(spans) == "Mostly steady, but 1 shaky span (weakest in the middle)."


def test_summarize_places_the_weakest_shaky_span_in_the_close_third():
    spans = [{"band": "strong", "start": 0, "n_tokens": 25, "min_conf": 0.9},
             {"band": "shaky", "start": 25, "n_tokens": 5, "min_conf": 0.1}]
    assert confidence_spans.summarize(spans) == "Mostly steady, but 1 shaky span (weakest in the close)."


def test_summarize_counts_and_pluralizes_multiple_shaky_spans_and_picks_the_lowest_min_conf():
    spans = [{"band": "shaky", "start": 0, "n_tokens": 5, "min_conf": 0.45},
             {"band": "strong", "start": 5, "n_tokens": 10, "min_conf": 0.9},
             {"band": "shaky", "start": 15, "n_tokens": 5, "min_conf": 0.1},    # the weakest -> "middle"
             {"band": "strong", "start": 20, "n_tokens": 10, "min_conf": 0.9}]
    assert confidence_spans.summarize(spans) == "Mostly steady, but 2 shaky spans (weakest in the middle)."


def test_summarize_end_to_end_from_real_spans(store):
    """Chain spans() -> summarize() over the multi-band fixture above, as the /spans endpoint will."""
    tokens = ["The", " sky", " is", " blue", ".", " Maybe", " it", " will", " rain", " later", "."]
    confidence = [0.95, 0.90, 0.92, 0.88, 0.99, 0.45, 0.30, 0.55, 0.60, 0.85, 0.97]
    rid = store.record(source="cli", messages=[{"role": "user", "content": "weather?"}],
                       response="The sky is blue. Maybe it will rain later.",
                       trace={"tokens": tokens, "confidence": confidence})
    result = confidence_spans.spans(store.get_run(rid))
    # 1 shaky span (indices 5-6) out of 11 total tokens -> start 5 < 11/3 (~3.67)? no -> falls in "middle"
    assert confidence_spans.summarize(result) == "Mostly steady, but 1 shaky span (weakest in the middle)."
