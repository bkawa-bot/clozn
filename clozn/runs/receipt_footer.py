"""The in-band receipt footer (ambient delivery, channel 1 of AMBIENT_DELIVERY.md).

A compact, honest one-line glass-box summary + a per-run permalink, appended to an OpenAI reply so the
receipt reaches the user INSIDE whatever client they already point at clozn (Cursor, Open WebUI, a
script, a terminal) -- no need to open the studio. The link (`/r/<id>`) opens the studio deep-linked to
exactly that run when they want to look closer; the footer is the shoulder-tap that tells them whether
it's worth it.

OFF by default so a plain OpenAI proxy stays byte-identical -- turned on per-request (`clozn_receipt:
true` in the body) or server-wide (`POST /receipt/mode`). Built PURELY from this run's recorded trace,
the same span producer as `GET /runs/<id>/spans`, so it never claims more than the numbers show.
Confidence is raw and uncalibrated: the footer says "worth a look", never "wrong".
"""
from __future__ import annotations

import re

from clozn.runs import confidence_spans

MARK = "⟨clozn⟩"     # ⟨clozn⟩ -- the quiet in-band marker (backticked so it renders as code)

# The footer, as a strippable pattern: from its own rule+marker line to the end of the message. Anchored
# to the exact block footer() emits, so ordinary text that merely mentions clozn is never touched.
_FOOTER_RE = re.compile(r"\n*---\n`" + re.escape(MARK) + r"`[^\n]*\s*$")


def strip_footers(messages: list) -> list:
    """Remove clozn's OWN receipt footers from incoming ASSISTANT messages (a copy; input untouched).

    Why this must exist (the context-contamination catch): in multi-turn chat the CLIENT echoes the whole
    conversation back -- including replies we footered on the way out. Without this, the model would see
    `⟨clozn⟩ mean conf …` inside its own past turns and could imitate or be steered by it. Symmetry rule:
    whatever clozn appends to a reply, it strips before the model ever reads it back. User/system
    messages are never modified (if a USER pastes a footer deliberately, that's their content)."""
    out = []
    for m in messages or []:
        if isinstance(m, dict) and m.get("role") == "assistant" and isinstance(m.get("content"), str):
            m = dict(m)
            m["content"] = _FOOTER_RE.sub("", m["content"]).rstrip()
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


def footer(run: dict | None, link: str) -> str:
    """The block appended to the reply, or "" when there's no trace to summarize (nothing honest to add).
    Shape: a markdown rule, the ⟨clozn⟩ marker, a raw-confidence stat, and the per-run receipt link."""
    s = summary(run)
    if not s["n_tokens"]:
        return ""
    bits = []
    if s["mean_conf"] is not None:
        bits.append(f"mean conf {s['mean_conf']:.2f}")
    if s["n_shaky"]:
        bits.append(f"{s['n_shaky']} span{'s' if s['n_shaky'] != 1 else ''} worth a look")
    else:
        bits.append("confident throughout")
    return f"\n\n---\n`{MARK}` {' · '.join(bits)} · receipt → {link}"
