"""Per-card memory relevance on the run record.

topic_gate.relevance() already computes a per-card {text: cosine} map, but the hot path only ever called
scalar() (the one-number gate) and threw the per-card scores away. _prompt_block_for now attaches each
applied card's cosine so the run record can show WHY each card fired, not just that the block did.

Model-free: the gate / relevance / card-store collaborators are all monkeypatched -- no embedder, no store.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # research/ (clozn_server lives here)

import clozn_server as cs          # noqa: E402


def test_prompt_relevance_reads_the_gate(monkeypatch):
    import topic_gate

    class _FakeGate:
        def relevance(self, prompt, texts):
            return {t: 0.5 for t in texts}

    monkeypatch.setattr(topic_gate, "get_gate", lambda: _FakeGate())
    assert cs._prompt_relevance("p", ["a", "b"]) == {"a": 0.5, "b": 0.5}


def test_prompt_relevance_degrades_to_empty_when_embedder_unavailable(monkeypatch):
    import topic_gate

    def _boom():
        raise RuntimeError("no embedder")

    monkeypatch.setattr(topic_gate, "get_gate", _boom)
    assert cs._prompt_relevance("p", ["a"]) == {}          # never raises -- {} means "couldn't score"


def _fixed_cards(cards, gate, rel, monkeypatch):
    """Wire _prompt_block_for's collaborators to fixed values so only the relevance-attach logic is under test."""
    monkeypatch.setattr(cs, "_prompt_mem_cards", lambda mem, exclude=(): [dict(c) for c in cards])
    monkeypatch.setattr(cs, "_prompt_gate", lambda lu, texts: gate)
    monkeypatch.setattr(cs, "_prompt_relevance", lambda lu, texts: rel)
    monkeypatch.setattr(cs, "PROMPT_GATE_MIN", 0.1)
    import memory_mode
    monkeypatch.setattr(memory_mode, "compile_prompt_block", lambda texts: "BLOCK")


def test_prompt_block_attaches_per_card_relevance(monkeypatch):
    _fixed_cards(
        [{"id": "c1", "text": "likes tea"}, {"id": "c2", "text": "lives in Berlin"}],
        gate=0.9, rel={"likes tea": 0.81, "lives in Berlin": 0.12}, monkeypatch=monkeypatch)

    block, applied, gate = cs._prompt_block_for(None, "tell me about tea", strength=1.0)
    assert block == "BLOCK" and gate == 0.9
    assert applied == [
        {"id": "c1", "text": "likes tea", "relevance": 0.81},
        {"id": "c2", "text": "lives in Berlin", "relevance": 0.12},
    ]


def test_prompt_block_relevance_is_none_per_card_when_embedder_off(monkeypatch):
    """No embedder -> _prompt_relevance {} -> each applied card's relevance is None, never fabricated."""
    _fixed_cards([{"id": "c1", "text": "likes tea"}], gate=1.0, rel={}, monkeypatch=monkeypatch)
    _block, applied, _gate = cs._prompt_block_for(None, "p", strength=1.0)
    assert applied == [{"id": "c1", "text": "likes tea", "relevance": None}]


def test_prompt_block_omitted_when_gated_out_carries_no_cards(monkeypatch):
    """Below the gate the block is omitted entirely -> [] applied, so there are no per-card scores to attach."""
    _fixed_cards([{"id": "c1", "text": "likes tea"}], gate=0.0, rel={"likes tea": 0.9}, monkeypatch=monkeypatch)
    block, applied, gate = cs._prompt_block_for(None, "unrelated task", strength=1.0)
    assert block is None and applied == [] and gate == 0.0
