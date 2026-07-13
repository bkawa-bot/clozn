"""Tests for the in-band receipt footer (receipt_footer.py) -- pure, synthetic runs, no server."""
from __future__ import annotations

from clozn.runs import receipt_footer


def _run(confs, **kw):
    r = {"trace": {"tokens": [f"t{i}" for i in range(len(confs))], "confidence": list(confs)},
         "finish_reason": "stop"}
    r.update(kw)
    return r


def _tie(top, alt, p_top, p_alt):
    """A trace whose one token is a near-even split between two content words."""
    return {"trace": {"tokens": [top], "confidence": [p_top],
                      "alternatives": [[{"piece": top, "prob": p_top}, {"piece": alt, "prob": p_alt}]]},
            "finish_reason": "stop"}


def test_footer_is_silent_on_an_ordinary_reply():
    # EXCEPTION-ONLY: a fine, completed reply with no hard signal and no close call -> no footer
    assert receipt_footer.footer(_run([0.95, 0.92, 0.9]), "http://h/r/x") == ""


def test_footer_fires_on_truncation():
    f = receipt_footer.footer(_run([0.9, 0.9], finish_reason="length"), "http://h:8090/r/run_x")
    assert "cut off mid-answer" in f and "http://h:8090/r/run_x" in f and receipt_footer.MARK in f


def test_footer_fires_on_error():
    assert "errored" in receipt_footer.footer(_run([0.9], error="boom"), "http://h/r/x")


def test_footer_stays_silent_on_an_ordinary_close_call():
    # a near-tie between two ordinary content words ("Rome" vs "Lyon") is NOT meaning-changing -> silent.
    # Only the thin answer-changing slice (digit/polarity) earns a footer line; raw confidence never does.
    assert receipt_footer.footer(_tie("Rome", "Lyon", 0.44, 0.42), "http://h/r/x") == ""


def test_footer_fires_on_a_meaningful_coin_flip():
    # two different digits -> the answer's NUMBER was a coin-flip -> the thin slice fires, phrased as a
    # neutral observation (never "wrong"), with both co-leaders named and the run link.
    f = receipt_footer.footer(_tie("5", "0", 0.54, 0.45), "http://h:8090/r/run_x")
    assert "coin-flip" in f and "5" in f and "0" in f
    assert "http://h:8090/r/run_x" in f and receipt_footer.MARK in f
    assert "wrong" not in f.lower()                        # correlational locator, never a verdict


def test_footer_empty_when_no_trace_or_junk():
    assert receipt_footer.footer({"trace": {"tokens": [], "confidence": []}}, "http://h/r/x") == ""
    assert receipt_footer.footer({}, "http://h/r/x") == ""
    assert receipt_footer.footer(None, "http://h/r/x") == ""


def test_summary_never_raises_on_junk():
    for junk in (None, {}, {"trace": "nope"}, {"trace": {"tokens": None}}):
        s = receipt_footer.summary(junk)
        assert s["n_tokens"] == 0 and s["mean_conf"] is None


# ---- strip_footers: the context-contamination guard ----

def test_strip_removes_our_own_footer_from_assistant_turns():
    foot = receipt_footer.footer(_run([0.9], finish_reason="length"), "http://h:8090/r/run_x")
    assert foot                                                # a real footer to strip
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
    foot = receipt_footer.footer(_run([0.9], finish_reason="length"), "http://h/r/x")
    msgs = [{"role": "user", "content": "look at this: " + foot}]
    assert receipt_footer.strip_footers(msgs)[0]["content"] == msgs[0]["content"]


def test_strip_handles_multipart_content():
    # OpenAI "content-as-parts" (Open WebUI et al. send this even for plain text) -- the footer must be
    # stripped from the text part, not leak back into context
    foot = receipt_footer.footer(_run([0.9], finish_reason="length"), "http://h/r/x")
    msgs = [{"role": "assistant", "content": [{"type": "text", "text": "The answer is Rome." + foot}]}]
    out = receipt_footer.strip_footers(msgs)
    assert out[0]["content"][0]["text"] == "The answer is Rome."
    assert receipt_footer.MARK not in out[0]["content"][0]["text"]


def test_strip_tolerates_junk_messages():
    out = receipt_footer.strip_footers([None, "str", {"role": "assistant"},
                                        {"role": "assistant", "content": 42}])
    assert len(out) == 4                                        # passed through, never raises
