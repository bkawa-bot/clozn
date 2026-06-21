"""Gated oracle (-m model) for the concept atlas — what's readable from real RWKV-4 state.

Asserts the strong, stable findings loosely (it's a 169M model) and, importantly, the HONESTY
invariant: grammatical concepts are decoded-only, so their `causal` must be None — the atlas
never claims a causal effect it didn't measure."""
import pytest

from clozn.atlas import concept_atlas

pytestmark = pytest.mark.model


def test_atlas_reads_concepts_and_gates_causal_claims(rwkv):
    cards = {c.name: c for c in concept_atlas(rwkv)}

    # sentence-type and person span the whole sentence -> strongly readable at the final state
    assert cards["sentence (q/stmt)"].decodability >= 0.8
    assert cards["person (1st/3rd)"].decodability >= 0.8

    # sentiment is the one we patched-and-measured: decodable AND causal
    sent = cards["sentiment (pos/neg)"]
    assert sent.decodability >= 0.7
    assert sent.causal is True and sent.delta is not None

    # HONESTY INVARIANT: grammatical concepts were decoded only, never causally tested
    for name in ("number (sing/plural)", "tense (past/present)", "person (1st/3rd)", "sentence (q/stmt)"):
        assert cards[name].causal is None       # we don't claim what we didn't test

    # every decodability is a real probability
    assert all(0.0 <= c.decodability <= 1.0 for c in cards.values())
