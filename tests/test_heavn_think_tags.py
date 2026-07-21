from pathlib import Path


REPLAY = Path(__file__).resolve().parents[1] / "studio" / "heavn" / "modules" / "replay.mjs"


def test_replay_exposes_reasoning_as_evidence_but_never_continuation_content():
    source = REPLAY.read_text(encoding="utf-8")
    assert 'data-testid="captured-reasoning"' in source
    assert "not a privileged or verified thought" in source
    assert 'messages.push({ role: "assistant", content: String(rec.response) })' in source
    continuation = source[source.index("async function doSend"):source.index("const LENS_STOPS")]
    assert "rec.reasoning" not in continuation
    assert "reasoningBlocks" not in continuation
