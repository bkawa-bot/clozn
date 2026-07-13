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


# ---- strip_footers: the context-contamination guard ----

def test_strip_removes_our_own_footer_from_assistant_turns():
    foot = receipt_footer.footer(_run([0.95, 0.9, 0.3]), "http://h:8090/r/run_x")
    msgs = [{"role": "user", "content": "q1"},
            {"role": "assistant", "content": "The answer is Rome." + foot},
            {"role": "user", "content": "q2"}]
    out = receipt_footer.strip_footers(msgs)
    assert out[1]["content"] == "The answer is Rome."
    assert receipt_footer.MARK not in out[1]["content"]
    assert out[0] == msgs[0] and out[2] == msgs[2]              # user turns untouched
    assert receipt_footer.MARK in msgs[1]["content"]            # input list not mutated


def test_strip_is_a_noop_without_a_footer():
    msgs = [{"role": "assistant", "content": "Plain reply, mentions clozn in prose --- fine."}]
    assert receipt_footer.strip_footers(msgs)[0]["content"] == msgs[0]["content"]


def test_strip_never_touches_user_pasted_footers():
    foot = receipt_footer.footer(_run([0.9, 0.9]), "http://h/r/x")
    msgs = [{"role": "user", "content": "look at this: " + foot}]
    assert receipt_footer.strip_footers(msgs)[0]["content"] == msgs[0]["content"]


def test_strip_tolerates_junk_messages():
    out = receipt_footer.strip_footers([None, "str", {"role": "assistant"},
                                        {"role": "assistant", "content": 42}])
    assert len(out) == 4                                        # passed through, never raises
