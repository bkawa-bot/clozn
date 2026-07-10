"""Receipt delta metric math shared by server, CLI, and scripts."""
from __future__ import annotations

import math
import re


_WORD_RE = re.compile(r"[a-z0-9']+")
_SENT_SPLIT_RE = re.compile(r"[.!?]+")


def _text(s) -> str:
    if not s:
        return ""
    return s if isinstance(s, str) else str(s)


def _words_of(s) -> list:
    return _WORD_RE.findall(_text(s).lower())


def _sent_count(s) -> int:
    parts = [p for p in _SENT_SPLIT_RE.split(_text(s)) if p.strip()]
    return max(1, len(parts))


def _js_round(x: float) -> int:
    """Math.round semantics for non-negative values."""
    return math.floor(x + 0.5)


def receipt_metrics(orig, repl) -> dict:
    """The three delta-strip numbers: word count, words/sentence, and word-type Jaccard change percent."""
    ow, rw = _words_of(orig), _words_of(repl)
    oset, rset = set(ow), set(rw)
    inter = len(oset & rset)
    uni = len(oset | rset)
    return {
        "words": [len(ow), len(rw)],
        "wps": [_js_round(len(ow) / _sent_count(orig) * 10) / 10,
                _js_round(len(rw) / _sent_count(repl) * 10) / 10],
        "changed": _js_round((1 - inter / uni) * 100) if uni else 0,
    }
