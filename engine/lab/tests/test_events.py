"""Wire-format tests for the §5.1 event types."""

import json

from cloze_lab.scheduler.events import (
    BlockFinalized,
    BlockStarted,
    CommitItem,
    GenFinished,
    GenStarted,
    StepStats,
    TokensCommitted,
    TokensRevised,
    ReviseItem,
    WorkspaceReadout,
    WorkspaceReadoutItem,
    event_to_dict,
    to_jsonl_line,
    write_jsonl,
)


def test_wire_keys_match_design_5_1() -> None:
    line = to_jsonl_line(
        TokensCommitted(t=7, block=3, items=(CommitItem(pos=101, id=4821, conf=0.93),))
    )
    d = json.loads(line)
    assert d == {
        "t": 7,
        "type": "tokens_committed",
        "block": 3,
        "items": [{"pos": 101, "id": 4821, "conf": 0.93}],
    }


def test_every_event_type_serializes() -> None:
    events = [
        GenStarted(t=0, prompt_tokens=412, block_len=32, max_new=512),
        BlockStarted(t=0, block=3, span=(96, 128)),
        TokensCommitted(t=7, block=3, items=()),
        TokensRevised(t=9, block=3, items=(ReviseItem(pos=99, old=311, id=7, conf=0.71),)),
        StepStats(t=7, block=3, step=7, committed=21, remaining=11, ms=38.2, cache_hit=0.82),
        BlockFinalized(t=9, block=3, text=" the cache is refreshed", steps_used=9),
        GenFinished(t=9, reason="eos", new_tokens=487, wall_ms=5210.0, steps_total=134, tok_per_s=93.5),
        WorkspaceReadout(t=10, run_id="run_demo", token_index=1, token_text=" cat", layer=12, position=1,
                         top_readouts=(WorkspaceReadoutItem(label="uncertainty", score=0.62),),
                         entropy=0.41, provider="mock"),
    ]
    names = [event_to_dict(e)["type"] for e in events]
    assert names == [
        "gen_started",
        "block_started",
        "tokens_committed",
        "tokens_revised",
        "step_stats",
        "block_finalized",
        "gen_finished",
        "workspace_readout",
    ]
    for e in events:
        assert json.loads(to_jsonl_line(e))["t"] == e.t


def test_workspace_readout_payload_shape() -> None:
    d = event_to_dict(
        WorkspaceReadout(t=3, run_id="run_demo", token_index=2, token_text=" maybe", layer=12, position=2,
                         top_readouts=(
                             WorkspaceReadoutItem(label="uncertainty", score=0.71),
                             WorkspaceReadoutItem(label="hallucination_risk", score=0.43),
                         ),
                         entropy=0.52, provider="mock", provider_type="mock", readout_kind="risk")
    )
    assert d == {
        "t": 3,
        "type": "workspace_readout",
        "run_id": "run_demo",
        "token_index": 2,
        "token_text": " maybe",
        "layer": 12,
        "position": 2,
        "top_readouts": [
            {"label": "uncertainty", "score": 0.71},
            {"label": "hallucination_risk", "score": 0.43},
        ],
        "entropy": 0.52,
        "provider": "mock",
        "provider_type": "mock",
        "readout_kind": "risk",
    }


def test_write_jsonl_roundtrip(tmp_path) -> None:
    events = [
        GenStarted(t=0, prompt_tokens=2, block_len=0, max_new=4),
        BlockStarted(t=0, block=0, span=(2, 6)),
    ]
    path = tmp_path / "run.jsonl"
    write_jsonl(events, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(l) for l in lines] == [json.loads(to_jsonl_line(e)) for e in events]
