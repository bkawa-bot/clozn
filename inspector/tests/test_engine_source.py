"""Fast oracle for the EngineStateSource (Phase 1.3) — NO live server required.

The live Clozn engine is being built by a parallel task, so the fast suite never touches the
network: it validates the two pure pieces of the wire contract (protocol/SPEC.md) against MOCK
frames constructed here from the spec's schema —

  (a) SSE `data: {json}\n\n` frame  ->  StateStep dataclass (step/token/state/readouts/meta),
  (b) the {dtype, shape, data=base64(little-endian bytes)} tensor wire, ndarray <-> JSON, bit-exact.

The end-to-end HTTP/SSE path (step over /v1/completions, get_state over /state, steer over
/intervene) is exercised against a self-contained stdlib http.server, gated behind -m model so the
fast suite skips anything that stands up a server.
"""
import base64
import json

import numpy as np
import pytest

from clozn.spine import Intervention, Readout, StateStep
from clozn.sources.engine import (
    EngineStateSource,
    aggregate_steps,
    decode_state,
    decode_tensor,
    encode_tensor,
    iter_sse,
    parse_state_step,
)


# --------------------------------------------------------------------------------------------------
# MOCK DATA — built by hand from protocol/SPEC.md's schema (verbatim, so the parent can reconcile it
# against the engine's real output). A StateStep on the wire is:
#   { step, token, state:{ "<component>": {dtype,shape,data} }, readouts:[{name,value,confidence,
#     causal_verified}], meta:{ substrate, ... } }
# --------------------------------------------------------------------------------------------------
def _wire_tensor(arr: np.ndarray) -> dict:
    """Hand-roll the tensor wire (independently of engine.encode_tensor) so the test pins the format."""
    a = np.ascontiguousarray(arr)
    le = a.astype(a.dtype.newbyteorder("<"), copy=False)
    return {"dtype": a.dtype.name, "shape": list(a.shape),
            "data": base64.b64encode(le.tobytes()).decode("ascii")}


# Two committed-token frames + a final frame carrying the activation tap as `state`.
BOARD_FINAL = np.array([101, 7592, 2088, 102], dtype=np.int64)            # prompt + committed ids
HIDDEN_FINAL = np.array([[0.5, -0.25, 1.0], [0.0, 2.0, -1.5]], dtype=np.float32)  # (slots, d)

MOCK_FRAMES = [
    {
        "step": 0,
        "token": 7592,
        "state": {},                                                     # light frame: state omitted
        "readouts": [
            {"name": "sentiment", "value": 0.82, "confidence": 0.91, "causal_verified": None},
            {"name": "logit-lens", "value": [["hello", 0.7], ["hi", 0.2]], "confidence": 0.7,
             "causal_verified": None},
        ],
        "meta": {"substrate": "autoregressive", "committed": 1, "remaining": 1, "ms": 12.5,
                 "cache_hit": True, "block": 0},
    },
    {
        "step": 1,
        "token": 2088,
        "state": {},
        "readouts": [
            {"name": "sentiment", "value": 0.40, "confidence": 0.88},    # causal_verified absent -> None
        ],
        "meta": {"substrate": "autoregressive", "committed": 1, "remaining": 0, "ms": 11.0,
                 "cache_hit": True, "block": 0},
    },
    {
        "step": 2,
        "token": None,
        "state": {"board": _wire_tensor(BOARD_FINAL), "hidden": _wire_tensor(HIDDEN_FINAL)},
        "readouts": [],
        "meta": {"substrate": "autoregressive", "kind": "end", "committed": 0, "remaining": 0,
                 "ms": 0.3, "total": 2},
    },
]


def _sse_bytes(frames) -> bytes:
    """Render frames as an SSE byte stream: `data: {json}\n\n`, then a `[DONE]` sentinel."""
    out = b""
    for f in frames:
        out += b"data: " + json.dumps(f).encode("utf-8") + b"\n\n"
    out += b"data: [DONE]\n\n"
    return out


def _chunked(data: bytes, n: int):
    """Split bytes into n-byte chunks to mimic network packet boundaries (may split a frame)."""
    return (data[i:i + n] for i in range(0, len(data), n))


# --------------------------------------------------------------------------------------------------
# (b) tensor wire round-trip
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("arr", [
    np.array([101, 7592, 2088, 102], dtype=np.int64),
    np.array([[0.5, -0.25, 1.0], [0.0, 2.0, -1.5]], dtype=np.float32),
    np.arange(24, dtype=np.float64).reshape(2, 3, 4),
    np.array([], dtype=np.float32),                                      # empty
    np.array(3.5, dtype=np.float32),                                     # 0-d scalar
])
def test_tensor_wire_roundtrip_is_bit_exact(arr):
    wire = encode_tensor(arr)
    assert set(wire) == {"dtype", "shape", "data"}
    assert wire["dtype"] == arr.dtype.name
    assert wire["shape"] == list(arr.shape)
    back = decode_tensor(wire)
    assert back.dtype == arr.dtype
    assert back.shape == arr.shape
    assert np.array_equal(back, arr)                                     # bit-exact, no float drift


def test_tensor_wire_matches_handrolled_spec_format():
    """engine.encode_tensor must produce EXACTLY the spec wire we hand-roll in the test."""
    arr = BOARD_FINAL
    assert encode_tensor(arr) == _wire_tensor(arr)
    # and decode reads the hand-rolled wire back bit-exact
    assert np.array_equal(decode_tensor(_wire_tensor(HIDDEN_FINAL)), HIDDEN_FINAL)


def test_tensor_data_is_base64_little_endian():
    """The `data` field is base64 of little-endian raw bytes — decode it the long way and check."""
    arr = np.array([1, 256, 65536], dtype="<u4")                        # distinct bytes per word
    wire = encode_tensor(arr)
    raw = base64.b64decode(wire["data"])
    assert raw == arr.astype("<u4").tobytes()
    assert raw[:4] == b"\x01\x00\x00\x00"                               # 1 little-endian


def test_decode_state_decodes_named_tensors_and_tolerates_empty():
    comp = {"board": _wire_tensor(BOARD_FINAL), "hidden": _wire_tensor(HIDDEN_FINAL)}
    st = decode_state(comp)
    assert set(st) == {"board", "hidden"}
    assert np.array_equal(st["board"], BOARD_FINAL)
    assert np.array_equal(st["hidden"], HIDDEN_FINAL)
    assert decode_state(None) == {}                                     # light frame -> empty State
    assert decode_state({}) == {}


# --------------------------------------------------------------------------------------------------
# (a) SSE frame -> StateStep parsing
# --------------------------------------------------------------------------------------------------
def test_parse_single_frame_into_statestep():
    st = parse_state_step(MOCK_FRAMES[0])
    assert isinstance(st, StateStep)
    assert st.step == 0
    assert st.token == 7592
    assert st.state == {}                                               # light frame: no tensors
    assert st.meta["substrate"] == "autoregressive"
    assert st.meta["cache_hit"] is True
    assert [r.name for r in st.readouts] == ["sentiment", "logit-lens"]
    r0 = st.readouts[0]
    assert isinstance(r0, Readout)
    assert r0.value == 0.82 and r0.confidence == 0.91
    assert r0.causal_verified is None                                   # never claims causality unverified


def test_parse_frame_with_state_tensors():
    st = parse_state_step(MOCK_FRAMES[2])
    assert set(st.state) == {"board", "hidden"}
    assert np.array_equal(st.state["board"], BOARD_FINAL)
    assert np.array_equal(st.state["hidden"], HIDDEN_FINAL)
    assert st.state["hidden"].dtype == np.float32


def test_readout_confidence_defaults_and_causal_unverified():
    st = parse_state_step(MOCK_FRAMES[1])
    r = st.readouts[0]
    assert r.confidence == 0.88
    assert r.causal_verified is None                                   # absent on the wire -> None


def test_iter_sse_parses_data_frames_and_stops_on_done():
    frames = list(iter_sse(_chunked(_sse_bytes(MOCK_FRAMES), 7)))      # tiny chunks split frames
    assert len(frames) == len(MOCK_FRAMES)                             # [DONE] consumed, not yielded
    assert [f["step"] for f in frames] == [0, 1, 2]
    assert frames[2]["state"]["board"]["dtype"] == "int64"


def test_iter_sse_ignores_comments_and_handles_multiline_data():
    raw = (b": keep-alive heartbeat\n\n"
           b"data: {\"step\": 5,\n"
           b"data:  \"token\": 42,\n"
           b"data:  \"meta\": {\"substrate\": \"autoregressive\"}}\n\n")
    frames = list(iter_sse(_chunked(raw, 5)))
    assert len(frames) == 1
    assert frames[0]["step"] == 5 and frames[0]["token"] == 42         # multi-line data: re-joined


def test_aggregate_steps_folds_run_into_final_step():
    steps = [parse_state_step(f) for f in MOCK_FRAMES]
    agg = aggregate_steps(steps)
    assert agg.step == 2                                               # last frame's step
    assert agg.token == [7592, 2088]                                   # union of committed tokens
    assert agg.meta["n_frames"] == 3
    assert agg.meta["substrate"] == "autoregressive"
    assert np.array_equal(agg.state["board"], BOARD_FINAL)             # final state carried through
    assert aggregate_steps([]) is None


def test_aggregate_is_a_copy_not_a_view():
    """The aggregated step must carry a copy (spine invariant 4: scribbling on it can't corrupt src)."""
    steps = [parse_state_step(f) for f in MOCK_FRAMES]
    agg = aggregate_steps(steps)
    agg.state["board"][:] = 999
    assert not np.array_equal(steps[-1].state["board"], agg.state["board"])


# --------------------------------------------------------------------------------------------------
# Intervention / steer payload construction (pure — no network)
# --------------------------------------------------------------------------------------------------
def test_steer_builds_spec_intervention_payload(monkeypatch):
    src = EngineStateSource(base_url="http://x")
    sent = {}
    monkeypatch.setattr(src, "_post_json", lambda path, payload: sent.update(path=path, payload=payload) or {})
    src.steer("sentiment", 2.0, note="push positive")
    assert sent["path"] == "/intervene"
    p = sent["payload"]
    assert p["kind"] == "steer"
    assert p["target"] == {"concept": "sentiment"}
    assert p["coef"] == 2.0
    assert p["note"] == "push positive"


def test_steer_with_explicit_vector_and_component(monkeypatch):
    src = EngineStateSource(base_url="http://x")
    sent = {}
    monkeypatch.setattr(src, "_post_json", lambda path, payload: sent.update(payload=payload) or {})
    src.steer("topic", -1.5, vector=np.array([1.0, 2.0, 3.0], dtype=np.float32), component="att_num")
    p = sent["payload"]
    assert p["vector"] == [1.0, 2.0, 3.0]
    assert p["target"] == {"concept": "topic", "component": "att_num"}
    assert p["coef"] == -1.5


def test_set_state_encodes_state_as_restore_intervention(monkeypatch):
    src = EngineStateSource(base_url="http://x")
    sent = {}
    monkeypatch.setattr(src, "_post_json", lambda path, payload: sent.update(path=path, payload=payload) or {})
    src.set_state({"board": BOARD_FINAL, "hidden": HIDDEN_FINAL})
    p = sent["payload"]
    assert p["kind"] == "restore"
    assert set(p["target"]["components"]) == {"board", "hidden"}
    assert p["state"]["board"] == encode_tensor(BOARD_FINAL)          # tensors on the wire
    assert np.array_equal(decode_tensor(p["state"]["hidden"]), HIDDEN_FINAL)


def test_set_state_passes_through_intervention_object(monkeypatch):
    src = EngineStateSource(base_url="http://x")
    sent = {}
    monkeypatch.setattr(src, "_post_json", lambda path, payload: sent.update(payload=payload) or {})
    src.set_state(Intervention(kind="patch", note="zero slot 3"))
    p = sent["payload"]
    assert p["kind"] == "patch" and p["note"] == "zero slot 3"


def test_reset_clears_frame_buffer():
    src = EngineStateSource()
    src.steps = [StateStep(0)]
    src._last = StateStep(0)
    src.reset()
    assert src.steps == [] and src._last is None


# --------------------------------------------------------------------------------------------------
# Phase 1.4 — the round-trip GATE against a REAL Clozn engine (engine -> protocol -> inspector).
# Gated behind -m model + skips unless an engine is reachable on :8080 (it needs a GPU server + a
# model), so CI / `-m "not model"` skip it. The pure parse/codec tests above cover the wire logic
# without an engine; this proves the live contract: read white-box readouts off the stream, snapshot
# the board, steer with a causal effect. Start one first, e.g.:
#   engine\core\build-gpu\cloze-server.exe <a-model>.gguf --gpu-layers 99 --port 8080
# --------------------------------------------------------------------------------------------------
@pytest.mark.model
def test_round_trip_against_live_engine():
    import urllib.error
    import urllib.request

    base = "http://127.0.0.1:8080"
    try:
        with urllib.request.urlopen(base + "/health", timeout=2) as r:
            json.loads(r.read())
    except (urllib.error.URLError, OSError):
        pytest.skip("no Clozn engine reachable on :8080")

    src = EngineStateSource(base_url=base, substrate="autoregressive")

    # READ — the white-box concept + logit-lens readouts arrive over the stream, and the activation
    # tap decodes off the wire (state="full").
    agg = src.step("The capital of France is")
    assert agg.meta["n_frames"] > 1
    names = {r.name for s in src.steps for r in s.readouts}
    assert "logit-lens" in names and "number" in names
    assert any("hidden" in s.state for s in src.steps)

    # SNAPSHOT — the board (token ids) from the run's terminal frame.
    board = src.get_state()["board"]
    assert board.ndim == 1 and board.shape[0] > 5

    # STEER — a number-steer measurably moves the generation vs coef 0 (a causal intervention on the
    # wire). High coef garbles by design (non-surjective), so we only assert it *changed*.
    p = "My favorite thing about weekends is"
    cold = src.steer("number", 0.0, prompt=p, max_tokens=12)["choices"][0]["text"]
    hot = src.steer("number", 22.0, prompt=p, max_tokens=12)
    assert hot["applied"] is True
    assert hot["choices"][0]["text"] != cold


# --------------------------------------------------------------------------------------------------
# Phase 2.3 — the DIFFUSION round-trip GATE (engine -> protocol -> inspector), the parallel-board path.
# Same gate as the AR test, but it needs a *diffusion* engine (LLaDA/Dream — a mask-token GGUF), so it
# additionally skips when the reachable engine is autoregressive. What's different on the wire (and what
# only-AR phase-1 never exercised): MANY slots commit per pass (`token` is a list of {pos,id,piece}),
# every step carries `meta.span` (the active block), and the model can change its mind — `revise` frames
# (`meta.kind=="revise"`) re-mask + re-predict committed slots, carrying a `revised` payload the inspector
# now folds onto `meta["revised"]`. Start one first, e.g.:
#   cloze-server.exe LLaDA-8B-Instruct-q8_0.gguf --mask-token 126336 --eos 126081 --gpu-layers 99 --port 8080
# --------------------------------------------------------------------------------------------------
@pytest.mark.model
def test_round_trip_against_live_diffusion_engine():
    import urllib.error
    import urllib.request

    base = "http://127.0.0.1:8080"
    try:
        with urllib.request.urlopen(base + "/health", timeout=2) as r:
            health = json.loads(r.read())
    except (urllib.error.URLError, OSError):
        pytest.skip("no Clozn engine reachable on :8080")
    if health.get("mode") != "diffusion":
        pytest.skip(f"engine on :8080 is {health.get('mode')!r}, not diffusion")

    # Drive with revision ENABLED so "the model changes its mind" actually fires on this run (a
    # high tau re-opens many committed slots); generous timeout — diffusion is slower than AR.
    src = EngineStateSource(base_url=base, substrate="diffusion", timeout=300.0,
                            revise=True, tau_revise=0.9, max_revisions=2, max_tokens=24, steps=12)

    # READ — the diffusion StateStep frames parse, substrate is tagged, white-box readouts (concepts +
    # logit-lens) + the activation tap all decode off the wire (state="full" by default in step()).
    agg = src.step("Write a short sentence about the sea.")
    assert agg.meta["substrate"] == "diffusion"
    assert agg.meta["n_frames"] > 1
    names = {r.name for s in src.steps for r in s.readouts}
    assert "logit-lens" in names              # logit-lens readout present
    assert names & {"number", "content", "function", "punct", "code", "question"}  # concept readouts
    assert any("hidden" in s.state for s in src.steps)  # raw activation tap rode the stream

    # PARALLEL BOARD-FILL — at least one pass commits MANY slots at once (the diffusion signature: a
    # frame's `token` is a list of {pos,id,piece}), and every step frame is tagged with its block span.
    commit_frames = [s for s in src.steps if isinstance(s.token, list) and s.token]
    assert commit_frames, "no multi-slot commit frames parsed"
    assert any(len(s.token) > 1 for s in commit_frames), "no pass committed >1 slot in parallel"
    item = commit_frames[0].token[0]
    assert {"pos", "id", "piece"} <= set(item)          # each committed slot carries pos/id/piece
    assert any("span" in s.meta for s in src.steps)     # meta.span (the active block) present

    # The aggregate flattens every committed slot into one flat span (not a list-of-lists).
    assert isinstance(agg.token, list) and agg.token
    assert all(isinstance(t, dict) for t in agg.token)

    # REVISE — the model changed its mind: revise frames parse (token=None, meta.kind=="revise") and the
    # revised payload survives onto meta["revised"]; the aggregate gathers them into meta["revisions"].
    revise_frames = [s for s in src.steps if s.meta.get("kind") == "revise"]
    assert revise_frames, "revision enabled but no revise frames arrived"
    rev = revise_frames[0]
    assert rev.token is None
    assert rev.meta.get("revised"), "revise frame dropped its 'revised' payload"
    ritem = rev.meta["revised"][0]
    assert {"pos", "old", "id"} <= set(ritem)           # which slot flipped, from `old` -> `id`
    assert agg.meta.get("n_revisions", 0) >= 1
    assert agg.meta.get("revisions")

    # SNAPSHOT — the board (token ids) from the run's terminal frame.
    board = src.get_state()["board"]
    assert board.ndim == 1 and board.shape[0] > 5
