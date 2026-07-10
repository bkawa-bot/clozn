"""Claim splitting helpers for confabulation diff."""
from __future__ import annotations

import re


_CLAIM_SPLIT_RE = re.compile(r"[^.!?]+[.!?]*")
_CLAUSE_SPLIT_RE = re.compile(r"\s*;\s*|,\s+(?:and|but|so|yet)\s+", re.IGNORECASE)
_MIN_CLAUSE_WORDS = 3


def _split_claims(text: str) -> list[str]:
    """Sentence-level split."""
    if not text or not isinstance(text, str):
        return []
    return [p.strip() for p in _CLAIM_SPLIT_RE.findall(text) if p.strip()]


def clause_split(text: str) -> list[str]:
    """Default claim splitter: sentence-level, then substantial coordinated clauses."""
    out: list[str] = []
    for sentence in _split_claims(text):
        parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(sentence) if p and p.strip()]
        if len(parts) > 1 and all(len(p.split()) >= _MIN_CLAUSE_WORDS for p in parts):
            out.extend(parts)
        else:
            out.append(sentence.strip())
    return out
