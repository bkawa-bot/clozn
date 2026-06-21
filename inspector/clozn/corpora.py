"""
clozn.corpora — stream a big natural-text corpus for REAL SAE training.

The corpus is the #1 lever for feature quality: our themed 70-sentence set was a demo (seeded
themes, tiny SAE); discovering real features in a real model needs lots of diverse natural text.
We stream it from HuggingFace `datasets` (so we ship no data) rather than hand-writing it.
"""
from __future__ import annotations


def text_stream(source: str = "wikitext", min_len: int = 40):
    """Yield non-trivial natural-text passages from a streamed dataset (no full download)."""
    import datasets
    if source == "wikitext":
        ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                                   split="train", streaming=True)
        for x in ds:
            t = (x.get("text") or "").strip()
            if len(t) >= min_len and not t.startswith("="):     # skip blanks + section headers
                yield t
    else:
        raise ValueError(f"unknown corpus source {source!r}")
