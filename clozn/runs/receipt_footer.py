"""The in-band receipt footer (ambient delivery, channel 1 of AMBIENT_DELIVERY.md).

A compact, honest one-line glass-box summary + a per-run permalink, appended to an OpenAI reply so the
receipt reaches the user INSIDE whatever client they already point at clozn (Cursor, Open WebUI, a
script, a terminal) -- no need to open the studio. The link (`/r/<id>`) opens the document-first Read view
for exactly that run when they want to look closer; the footer is the shoulder-tap that tells them whether
it's worth it.

OFF by default so a plain OpenAI proxy stays byte-identical -- turned on per-request (`clozn_receipt:
true` in the body) or server-wide (`POST /receipt/mode`). Built PURELY from this run's recorded trace,
the same span producer as `GET /runs/<id>/spans`, so it never claims more than the numbers show.
Confidence is raw and uncalibrated: the footer says "worth a look", never "wrong".
"""
from __future__ import annotations

import re

from clozn.runs import close_calls, confidence_spans, signals

MARK = "⟨clozn⟩"     # ⟨clozn⟩ -- the quiet in-band marker (backticked so it renders as code)

# The footer, as a strippable pattern: from its own rule+marker line to the end of the message. Anchored
# to the exact block footer() emits, so ordinary text that merely mentions clozn is never touched.
_FOOTER_RE = re.compile(r"\n*---\n`" + re.escape(MARK) + r"`[^\n]*\s*$")


def _strip_text(s: str) -> str:
    return _FOOTER_RE.sub("", s).rstrip()


def strip_footers(messages: list) -> list:
    """Remove clozn's OWN receipt footers from incoming ASSISTANT messages (a copy; input untouched).

    Why this must exist (the context-contamination catch): in multi-turn chat the CLIENT echoes the whole
    conversation back -- including replies we footered on the way out. Without this, the model would see
    `⟨clozn⟩ …` inside its own past turns and could imitate or be steered by it. Symmetry rule: whatever
    clozn appends to a reply, it strips before the model ever reads it back. User/system messages are
    never modified (a USER who pastes a footer deliberately keeps it). Handles BOTH content shapes an
    OpenAI-compatible client may send: a plain string, and the multi-part `content:[{type:text,text:…}]`
    form (Open WebUI et al. send this even for plain text -- a real leak path caught in review)."""
    out = []
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "assistant":
            c = m.get("content")
            if isinstance(c, str):
                m = dict(m); m["content"] = _strip_text(c)
            elif isinstance(c, list):
                m = dict(m)
                m["content"] = [({**p, "text": _strip_text(p["text"])}
                                 if isinstance(p, dict) and isinstance(p.get("text"), str) else p)
                                for p in c]
        out.append(m)
    return out


def summary(run: dict | None) -> dict:
    """{n_tokens, mean_conf, n_shaky, line} from a run's trace -- pure, never raises. mean_conf is None
    and n_tokens 0 when there is no per-token trace (a diffusion run, or a stream that logged nothing):
    the caller then adds no footer rather than a fabricated stat."""
    sp = confidence_spans.spans(run if isinstance(run, dict) else {})
    trace = run.get("trace") if isinstance(run, dict) else None
    trace = trace if isinstance(trace, dict) else {}
    tokens = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
    confs = [float(c) for c in (trace.get("confidence") or []) if isinstance(c, (int, float))]
    mean = round(sum(confs) / len(confs), 2) if confs else None
    n_shaky = sum(1 for s in sp if s.get("band") == "shaky")
    return {"n_tokens": len(tokens), "mean_conf": mean, "n_shaky": n_shaky,
            "line": confidence_spans.summarize(sp)}


def _fork_bits(run: dict) -> list[str]:
    """The thin, meaning-changing close-call slice (close_calls.meaningful: two different digits, or a
    polarity flip -- ~4% of runs; the ~60% of harmless "as" vs "is" phrasing forks are excluded). A
    CORRELATIONAL locator, phrased as a neutral observation ("coin-flip: X vs Y"), NEVER a failure claim --
    it points at an answer-bearing token worth a branch-stability re-run, never says the answer is wrong."""
    try:
        calls = close_calls.meaningful(close_calls.close_calls(run))
        if not calls:
            return []
        t = close_calls.tightest(calls)
        if not (t and t.get("top") and t.get("alt")):
            return []
        bit = f"coin-flip: “{t['top']}” vs “{t['alt']}”"
        if len(calls) > 1:
            bit += f" (+{len(calls) - 1})"
        return [bit]
    except Exception:
        return []


def footer(run: dict | None, link: str) -> str:
    """The block appended to the reply -- EXCEPTION-ONLY: an ordinary, fine reply gets "" (silence is a
    signal; the footer speaks only when it has something true to say). Footers are the ONE ambient
    delivery surface. It fires on HARD facts (clozn.runs.signals -- errored / cut off / stuck repeating /
    empty / bad JSON, each a fact or a named check) PLUS the thin meaning-changing close-call slice
    (_fork_bits: a coin-flip on a digit or a negation, ~4% of runs).

    Ordinary per-token close calls stay OUT -- they're COMMON under sampling (a harmless "as" vs "is" fork
    fires on ~a majority of runs) and would break the exception-only contract; they live in the studio's
    `forks` locator instead. Only the answer-changing slice earns a footer line, and it's phrased as a
    neutral observation ("coin-flip: X vs Y"), correlational, never a verdict. The link is the CTA."""
    if not isinstance(run, dict):
        return ""
    bits = signals.hard_signals(run) + _fork_bits(run)
    if not bits:
        return ""                          # ordinary reply -> no footer at all
    return f"\n\n---\n`{MARK}` {' · '.join(bits)} · look → {link}"
