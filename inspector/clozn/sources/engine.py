"""
clozn.sources.engine — a running Clozn ENGINE (over HTTP) as a StateSource (Phase 1.3).

The engine (the C++/ggml runtime) is just one more substrate behind Clozn's StateSource seam,
reached over the wire instead of in-process. Its §5.1 typed-event SSE is reshaped to the canonical
`StateStep` schema (protocol/SPEC.md): one forward pass folds into one StateStep, `state` is the
post-step internal slice (board + any activation tap), `readouts` are concept/lens readings (each
carrying its confidence and an unverified causality flag), `meta` carries substrate/timing/commits.

Wire (protocol/SPEC.md):
  - step:      POST {base_url}/v1/completions  {prompt, stream, protocol, features, state} -> SSE
               `data: {json}\n\n` frames, each a StateStep; the terminal frame carries `board` (the
               snapshot). The engine is stateless-per-request, so there is no GET /state — the board
               IS the snapshot, and per-step activations ride on each step frame's `state`.
  - intervene: POST {base_url}/intervene       <- an Intervention {kind, target, vector?, coef?, note?}.
               steer's `target` names the generation to steer (prompt + max_tokens); restore is /v1/board.

Tensors on the wire are `{dtype, shape, data}` with `data` = base64 of **little-endian** raw bytes;
`encode_tensor` / `decode_tensor` are the exact ndarray <-> JSON codec (module-level, reusable).

Honesty stays on the wire (Readout.confidence + causal_verified=None until patched-and-measured),
the source still owns the state (consumers only read; writes go through /intervene), and a new model
family is a new StateSource, never a protocol change — the same four carried-over invariants.

This source needs a live engine, so any test that exercises it over the network is gated behind
`@pytest.mark.model`; the parse/codec are pure and tested with mock frames built from the spec.
"""
from __future__ import annotations

import base64
import json
import urllib.request
from typing import Any, Iterator

import numpy as np

from ..spine import Intervention, Readout, State, StateStep

# Default endpoint of a locally-served Clozn engine.
DEFAULT_BASE_URL = "http://127.0.0.1:8080"


# --------------------------------------------------------------------------------------------------
# Tensor wire codec — {dtype, shape, data}, data = base64(little-endian raw bytes).  np.ndarray <-> JSON.
# This MUST match the engine's encoder byte-for-byte (protocol/SPEC.md): picks/ids are exact, and
# raw bytes round-trip is bit-exact across hosts because we pin little-endian on the wire.
# --------------------------------------------------------------------------------------------------
def encode_tensor(a: np.ndarray) -> dict:
    """ndarray -> {"dtype": str, "shape": [int,...], "data": base64-of-little-endian-bytes}."""
    a = np.asarray(a)
    shape = list(a.shape)                       # capture BEFORE ascontiguousarray (it promotes 0-d to (1,))
    dtype_name = a.dtype.name
    contig = np.ascontiguousarray(a)
    le = contig.dtype.newbyteorder("<")         # force little-endian on the wire (no-op on x86/ARM)
    raw = contig.astype(le, copy=False).tobytes(order="C")
    return {
        "dtype": dtype_name,                    # the *logical* dtype (e.g. "float32"); byteorder is implied LE
        "shape": shape,
        "data": base64.b64encode(raw).decode("ascii"),
    }


def decode_tensor(d: dict) -> np.ndarray:
    """{"dtype","shape","data"} -> ndarray. `data` is base64 of little-endian raw bytes."""
    raw = base64.b64decode(d["data"])
    le = np.dtype(d["dtype"]).newbyteorder("<")  # interpret the wire bytes as little-endian
    a = np.frombuffer(raw, dtype=le)
    # Hand back native byteorder so downstream math/ops never trip on a big-endian view; copy so the
    # result is writable (frombuffer is read-only). reshape LAST so a 0-d () shape survives intact.
    a = a.astype(a.dtype.newbyteorder("="), copy=True)
    return a.reshape(tuple(int(x) for x in d["shape"]))


def _is_tensor_wire(v: Any) -> bool:
    return isinstance(v, dict) and "dtype" in v and "shape" in v and "data" in v


def decode_state(obj: Any) -> State:
    """A wire `State` ({"<component>": {dtype,shape,data}, ...}) -> {name: ndarray}.

    Tolerates None / missing (a light frame with `state` omitted) and already-decoded passthrough.
    """
    if not obj:
        return {}
    out: State = {}
    for name, val in obj.items():
        if _is_tensor_wire(val):
            out[name] = decode_tensor(val)
        elif isinstance(val, np.ndarray):
            out[name] = val
        else:                                    # a summarized/sparse view the engine sent inline
            out[name] = np.asarray(val)
    return out


# --------------------------------------------------------------------------------------------------
# SSE frame -> StateStep
# --------------------------------------------------------------------------------------------------
def _parse_readout(r: dict) -> Readout:
    """One readout JSON -> Readout. causal_verified stays None unless the engine already verified."""
    return Readout(
        name=r.get("name", ""),
        value=r.get("value"),
        confidence=float(r.get("confidence", 1.0)),
        causal_verified=r.get("causal_verified", None),
    )


def parse_state_step(frame: dict) -> StateStep:
    """A StateStep wire frame (a parsed SSE `data:` JSON object) -> the Python StateStep dataclass.

    Diffusion adds a `revise` frame (`meta.kind=="revise"`): the model changes its mind, re-masking
    already-committed slots and re-predicting them. The canonical StateStep schema is fixed
    (step/token/state/readouts/meta), so the revise frame carries `token=null` and a top-level
    `revised` array (`[{pos, old, id, conf, piece}, ...]`). We fold that array onto `meta["revised"]`
    so the revision (which slots flipped, from `old`->`id`) survives the parse instead of being
    silently dropped — the diffusion-only datum this whole path exists to surface. AR never emits it,
    so `meta` is untouched there.
    """
    readouts = [_parse_readout(r) for r in (frame.get("readouts") or [])]
    meta = dict(frame.get("meta") or {})
    if frame.get("revised") is not None:           # diffusion revise frame -> keep the revised items
        meta["revised"] = list(frame["revised"])
    return StateStep(
        step=int(frame.get("step", 0)),
        token=frame.get("token"),
        state=decode_state(frame.get("state")),
        readouts=readouts,
        meta=meta,
    )


def iter_sse(stream: Iterator[bytes] | Any) -> Iterator[dict]:
    """Yield parsed JSON objects from an SSE byte stream of `data: {json}\\n\\n` frames.

    Robust to multi-line `data:` fields (SSE concatenates them with '\\n'), `:`-comment lines and
    keep-alives, CRLF, and chunk boundaries that split a frame — we buffer until a blank line ends
    an event. A `data: [DONE]` sentinel (OpenAI-style) terminates the stream.
    """
    buf = b""
    data_lines: list[str] = []

    def _flush() -> Iterator[dict]:
        nonlocal data_lines
        if data_lines:
            payload = "\n".join(data_lines)
            data_lines = []
            s = payload.strip()
            if s and s != "[DONE]":
                yield json.loads(payload)

    for chunk in stream:
        if not chunk:
            continue
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.rstrip(b"\r")
            if line == b"":                       # blank line -> dispatch the buffered event
                yield from _flush()
                continue
            text = line.decode("utf-8")
            if text.startswith(":"):              # SSE comment / heartbeat
                continue
            if text.startswith("data:"):
                data_lines.append(text[5:].lstrip(" "))
            # other SSE fields (event:, id:, retry:) are not part of the StateStep payload
    # tail: a final event not terminated by a blank line, or leftover in buf
    if buf.strip():
        tail = buf.rstrip(b"\r").decode("utf-8")
        if tail.startswith("data:"):
            data_lines.append(tail[5:].lstrip(" "))
    yield from _flush()


def aggregate_steps(steps: list[StateStep]) -> StateStep | None:
    """Fold a run's per-frame StateSteps into one final aggregated StateStep.

    The last frame carries the final state/step; we collect the tokens committed across the run
    (so the caller can read the whole generated span) and keep the last frame's readouts/meta,
    annotating meta with the frame count and the per-step token list.

    AR commits ONE slot per pass, so each frame's `token` is a single id (or a 1-item list) and the
    run reads as a flat sequence. Diffusion commits MANY slots per pass (parallel board-fill), so each
    frame's `token` is a *list* of `{pos,id,piece}` commit items. We flatten those per-pass lists into
    one flat list of commit items for `token`/`meta["tokens"]` (the whole committed span, one level
    deep — not a list-of-lists), while scalar AR tokens keep their existing flat-union shape. Revise
    frames (`meta.kind=="revise"`, `token=None`) carry no commit but a `meta["revised"]` payload; we
    gather every revision into `meta["revisions"]` and count them in `meta["n_revisions"]` so a
    consumer sees that — and what — the model changed its mind about (zero/absent for AR).
    """
    if not steps:
        return None
    last = steps[-1].copy()
    # Flatten per-pass commits into one span: a list `token` (diffusion multi-slot) contributes its
    # items; a scalar `token` (AR / mock) contributes itself. Result is always one level deep.
    tokens: list[Any] = []
    for s in steps:
        if s.token is None:
            continue
        if isinstance(s.token, list):
            tokens.extend(s.token)                 # multi-slot pass -> splice the slot items in
        else:
            tokens.append(s.token)                 # single committed id (AR / scalar mock)
    # Gather diffusion "changed its mind" revisions across the run (each frame's revised items spliced).
    revisions: list[Any] = []
    for s in steps:
        rev = s.meta.get("revised")
        if rev:
            revisions.extend(rev)
    # The final "end" control frame carries neither readouts nor state; surface the last frame that
    # actually has them, so the aggregate reflects the run's white-box reads instead of an empty tail.
    for s in reversed(steps):
        if s.readouts:
            last.readouts = list(s.readouts)
            break
    for s in reversed(steps):
        if s.state:
            last.state = {k: v.copy() for k, v in s.state.items()}  # deep copy (spine invariant 4)
            break
    meta = dict(last.meta)
    meta.setdefault("substrate", _substrate_of(steps))
    meta["n_frames"] = len(steps)
    meta["tokens"] = tokens
    if revisions:                                  # diffusion only; omit the keys entirely for AR
        meta["revisions"] = revisions
        meta["n_revisions"] = len(revisions)
    last.meta = meta
    last.token = tokens if tokens else last.token
    return last


def _substrate_of(steps: list[StateStep]) -> Any:
    for s in steps:
        if "substrate" in s.meta:
            return s.meta["substrate"]
    return None


class EngineStateSource:
    """A running Clozn engine (HTTP) behind Clozn's StateSource seam.

    reset/step/get_state/set_state speak the protocol/SPEC.md wire; `steer` is a steering-Intervention
    convenience. The full run's frames are kept on `self.steps`; `step()` returns the aggregated final
    StateStep (the per-frame stream is also available via `step_stream()`).
    """

    def __init__(self, base_url: str = DEFAULT_BASE_URL, substrate: str = "autoregressive",
                 timeout: float = 300.0, **opts: Any):
        self.base_url = base_url.rstrip("/")
        self.substrate = substrate
        self.timeout = timeout
        self.opts = opts                          # extra request fields (temperature, max_new, ...)
        self.steps: list[StateStep] = []          # per-frame StateSteps from the last step() run
        self._last: StateStep | None = None       # the last aggregated StateStep

    # --- StateSource interface -------------------------------------------------------------------
    def reset(self) -> None:
        """No server-side session to clear (each /v1/completions call is self-contained); drop
        the local frame buffer so a fresh run starts clean."""
        self.steps = []
        self._last = None

    def step(self, x: Any = None) -> StateStep:
        """Run one generation over `x` and return the aggregated final StateStep.

        POSTs the completion request, streams the StateStep SSE frames into `self.steps`, and folds
        them into one returned StateStep. Per-frame steps stay on the instance for replay/inspection.
        """
        body = {
            "prompt": x,
            "stream": True,
            "protocol": True,
            "features": True,
            "state": "full",
        }
        body.update(self.opts)
        self.steps = list(self._stream_steps(body))
        agg = aggregate_steps(self.steps)
        if agg is None:                            # empty stream -> a benign empty step
            agg = StateStep(step=0, token=x, meta={"substrate": self.substrate, "n_frames": 0})
        self._last = agg
        return agg

    def step_stream(self, x: Any = None) -> Iterator[StateStep]:
        """Like step() but *yields* each StateStep frame as it arrives (also recorded on self.steps)."""
        body = {"prompt": x, "stream": True, "protocol": True, "features": True, "state": "full"}
        body.update(self.opts)
        self.steps = []
        for st in self._stream_steps(body):
            self.steps.append(st)
            yield st
        self._last = aggregate_steps(self.steps)

    def get_state(self) -> State:
        """The snapshot State: the `board` (token ids) captured from the last step()'s final frame.

        The engine is stateless-per-request, so there is no `GET /state` — per protocol/SPEC.md the
        board IS the snapshot (returned in the run's terminal frame; restore via /v1/board on a
        diffusion model). Per-step activations live on `self.steps[i].state` (with `state="full"`)."""
        snap = getattr(self, "_snapshot", None)
        if not snap:
            raise RuntimeError("no snapshot available — call step() first")
        return dict(snap)

    def set_state(self, s: State | Intervention) -> None:
        """Write state back through /intervene. Accepts a raw State (-> a "restore"/"edit" Intervention,
        named tensors encoded to the wire) or a ready Intervention object/dict."""
        if isinstance(s, Intervention):
            payload = self._intervention_payload(s)
        elif isinstance(s, dict) and "kind" in s and not _looks_like_state(s):
            payload = dict(s)                      # already an Intervention-shaped dict
        else:
            payload = {
                "kind": "restore",
                "target": {"components": list(s.keys())},
                "state": {name: encode_tensor(np.asarray(v)) for name, v in s.items()},
            }
        self._post_json("/intervene", payload)

    # --- steering convenience --------------------------------------------------------------------
    def steer(self, concept: str, coef: float, prompt: str | None = None, max_tokens: int = 8,
              vector: np.ndarray | list[float] | None = None, component: str | None = None,
              note: str = "", **opts: Any) -> dict:
        """Post a steering Intervention: nudge `concept`'s direction by `coef` (the control-vector
        `set_steer` on the engine side). The engine is stateless, so the steer is realized on a
        target generation — supply a `prompt` (+ `max_tokens`); `vector`/`component` are optional."""
        target: dict[str, Any] = {"concept": concept}
        if prompt is not None:
            target["prompt"] = prompt
            target["max_tokens"] = max_tokens
        if component is not None:
            target["component"] = component
        target.update(opts)
        payload: dict[str, Any] = {"kind": "steer", "target": target, "coef": float(coef)}
        if vector is not None:
            payload["vector"] = [float(x) for x in np.asarray(vector).ravel().tolist()]
        if note:
            payload["note"] = note
        return self._post_json("/intervene", payload)

    # --- internals -------------------------------------------------------------------------------
    @staticmethod
    def _intervention_payload(iv: Intervention) -> dict:
        """The local Intervention dataclass -> the SPEC.md wire shape {kind, target, note}.

        (`fn` is a local Callable and cannot cross the wire; the engine applies kind+target itself.)"""
        return {"kind": iv.kind, "target": {}, "note": iv.note}

    def _stream_steps(self, body: dict) -> Iterator[StateStep]:
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self._url("/v1/completions"), data=data, method="POST",
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        self._snapshot = {}
        self._final = None
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for frame in iter_sse(_read_chunks(resp)):
                if "board" in frame:  # the terminal "final" frame — the snapshot, not a step
                    self._final = frame
                    self._snapshot = {"board": np.asarray(frame["board"], dtype=np.int64)}
                    continue
                yield parse_state_step(frame)

    def _get_json(self, path: str) -> Any:
        req = urllib.request.Request(self._url(path), method="GET",
                                     headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post_json(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self._url(path), data=data, method="POST",
                                     headers={"Content-Type": "application/json",
                                              "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}

    def _url(self, path: str) -> str:
        return self.base_url + path


def _looks_like_state(d: dict) -> bool:
    """True if a dict is a State map (every value is an ndarray or a tensor-wire dict)."""
    return bool(d) and all(isinstance(v, np.ndarray) or _is_tensor_wire(v) for v in d.values())


def _read_chunks(resp: Any, size: int = 8192) -> Iterator[bytes]:
    """Yield raw byte chunks from an http response as they arrive (streaming SSE)."""
    while True:
        chunk = resp.read(size)
        if not chunk:
            break
        yield chunk
