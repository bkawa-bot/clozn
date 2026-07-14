"""cloze_engine.py — the Python SDK for the cloze-server white-box HTTP API.

The C++ engine (engine/core/serve/cloze_server.cpp) exposes a model's interior over HTTP:
READ activations (`/harvest`), WRITE them back and observe the effect (`/state`), and
STEER a generation (`/intervene`). Those endpoints close the read -> edit -> write ->
observe loop on a live ggml/llama.cpp model. This module is the thin Python seam over
them, so the research stack (SAE discovery, feature circuits, concept probes — all numpy
already) can drive the production engine instead of a separate HF model:

    from cloze_engine import EngineClient
    eng = EngineClient(port=8080)
    h = eng.harvest("The capital of France is")      # h.activations: [n_tokens, n_embd] f32
    # ... run a discovery harness on h.activations (SAE encode, PCA, a learned edit) ...
    obs = eng.write_state("The capital of France is", h.layer,
                          positions=[h.n_tokens - 1], values=edited_last_row)
    print(obs.moved_l2, obs.baseline_top, obs.edited_top)   # how the next token moved

Dependencies are deliberately minimal: the standard library for HTTP/JSON/base64 plus
numpy for the activation matrices. No `requests`, no client framework.

The wire format for tensors is SPEC.md's {dtype, shape, data}, where `data` is the
base64 of the raw little-endian float32 bytes. x86 and CUDA are little-endian, so the
in-memory floats ARE those bytes; decoding is a straight np.frombuffer(..., '<f4').

Run `python cloze_engine.py --selftest` to validate the codec offline (no server), or
`python cloze_engine.py --demo` to run a live read -> edit -> write -> observe round-trip
against a running cloze-server.
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# --------------------------------------------------------------------------- wire codec

def decode_tensor(obj: dict) -> np.ndarray:
    """Decode a wire tensor {dtype:"float32", shape:[...], data:base64-LE} to a numpy array.

    Mirrors tensor_json_f32 in cloze_server.cpp: the bytes are little-endian float32, row
    major, so np.frombuffer('<f4').reshape(shape) reconstructs the matrix exactly (no copy
    beyond the base64 decode). Raises on a non-float32 dtype or a shape/byte-count mismatch.
    """
    dtype = obj.get("dtype")
    if dtype != "float32":
        raise ValueError(f"unsupported wire dtype {dtype!r} (only float32)")
    shape = tuple(int(d) for d in obj["shape"])
    raw = base64.b64decode(obj["data"])
    arr = np.frombuffer(raw, dtype="<f4")
    expected = int(np.prod(shape)) if shape else 0
    if arr.size != expected:
        raise ValueError(f"tensor byte count {arr.size} != shape product {expected} {shape}")
    return arr.reshape(shape)


def flatten_values(values: ArrayLike) -> list:
    """Flatten an edit (a [P, n_embd] matrix or already-flat vector) to the row-major list of
    Python floats /state expects. The server reads it as a std::vector<float> and checks
    values.size() == positions.size() * n_embd, so the order must be position-major."""
    arr = np.ascontiguousarray(np.asarray(values, dtype="<f4")).reshape(-1)
    return arr.tolist()


# --------------------------------------------------------------------------- result types

@dataclass
class Harvest:
    """The result of POST /harvest: every input token's residual at the tap `layer`."""
    tokens: list[str]                 # decoded piece per input token
    layer: int                        # the layer actually read (server may clamp the request)
    activations: np.ndarray           # [n_tokens, n_embd], float32

    @property
    def n_tokens(self) -> int:
        return int(self.activations.shape[0])

    @property
    def n_embd(self) -> int:
        return int(self.activations.shape[1])


@dataclass
class Observation:
    """The result of POST /state: how the model's next-token prediction moved under the write."""
    applied: bool                     # False if the write was rejected (bad layer / size)
    layer: int
    moved_l2: float                   # L2 distance between baseline and edited logit vectors
    baseline_top: list = field(default_factory=list)   # [{token, prob}, ...] top-3 before
    edited_top: list = field(default_factory=list)      # [{token, prob}, ...] top-3 after
    error: Optional[str] = None

    def shifted(self) -> bool:
        """True iff the argmax next token changed under the write (a visible behavioral effect)."""
        return bool(self.applied and self.baseline_top and self.edited_top
                    and self.baseline_top[0]["token"] != self.edited_top[0]["token"])

    def summary(self) -> str:
        if not self.applied:
            return f"rejected: {self.error}"
        b = ", ".join(f"{t['token']!r} {t['prob']:.3f}" for t in self.baseline_top)
        e = ", ".join(f"{t['token']!r} {t['prob']:.3f}" for t in self.edited_top)
        flag = "  [TOP-1 SHIFTED]" if self.shifted() else ""
        return f"moved_l2={self.moved_l2:.3f}{flag}\n  baseline: {b}\n  edited:   {e}"


# --------------------------------------------------------------------------- the client

class EngineError(RuntimeError):
    """An error returned by the engine (non-2xx with a JSON {error: ...} body)."""


class EngineClient:
    """A thin HTTP client for one running cloze-server.

    All calls are synchronous. The server serializes generation on its context pool, so
    concurrent calls from multiple clients are fine (they queue on a free worker); within
    one client the calls are sequential by construction.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080, timeout: float = 120.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    # -- transport -----------------------------------------------------------

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # The engine reports client errors as 400 + {"error": "..."}; surface that message.
            payload = e.read().decode("utf-8", "replace")
            try:
                msg = json.loads(payload).get("error", payload)
            except json.JSONDecodeError:
                msg = payload
            raise EngineError(f"{method} {path} -> {e.code}: {msg}") from None

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    def _post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, body)

    # -- endpoints -----------------------------------------------------------

    def health(self) -> dict:
        """GET /health -> {status, model, mode}. `mode` is 'diffusion' or 'autoregressive'."""
        return self._get("/health")

    def harvest(self, text: str, layer: Optional[int] = None) -> Harvest:
        """POST /harvest: read every token's residual at the tap layer in ONE causal forward.

        `layer` overrides the server's default tap (the calibrated early read layer). An
        out-of-range layer falls back to the final layer server-side; the Harvest carries the
        layer actually used, so thread Harvest.layer into write_state to read and write at the
        same depth.
        """
        body: dict = {"text": text}
        if layer is not None:
            body["layer"] = int(layer)
        r = self._post("/harvest", body)
        return Harvest(tokens=r["tokens"], layer=int(r["layer"]),
                       activations=decode_tensor(r["activations"]))

    def harvest_layers(self, text: str) -> dict:
        """POST /harvest/layers: per-layer activation SUMMARY in ONE causal forward -- the L2 norm of every
        token's residual at EVERY layer (the depth x position "MRI" map) + a per-layer mean. Unlike
        harvest() (one layer's full tensor), this is the cheap cross-depth view: one forward, all layers.
        Returns {tokens, n_tokens, n_layer, norms:[n_layer][n_tokens], layer_mean:[n_layer]} -- plain
        floats (no tensor codec), so it's handed back as-is for the UI to render."""
        r = self._post("/harvest/layers", {"text": text})
        return {"tokens": r.get("tokens", []),
                "n_tokens": int(r.get("n_tokens", 0)),
                "n_layer": int(r.get("n_layer", 0)),
                "norms": r.get("norms", []),
                "layer_mean": r.get("layer_mean", [])}

    def write_state(self, text: str, layer: int, positions: Sequence[int],
                    values: ArrayLike) -> Observation:
        """POST /state: overwrite `positions`' residual at `layer` with `values`, then observe.

        `values` is a [len(positions), n_embd] matrix (or the equivalent flat vector); it is
        flattened position-major to match the server's contract. The server runs a baseline
        forward, applies the write via the eval-callback activation patch, runs again, clears
        the write, and reports how the next-token logits moved.
        """
        positions = [int(p) for p in positions]
        body = {"text": text, "layer": int(layer), "positions": positions,
                "values": flatten_values(values)}
        r = self._post("/state", body)
        return Observation(applied=bool(r.get("applied", False)),
                           layer=int(r.get("layer", layer)),
                           moved_l2=float(r.get("moved_l2", 0.0)),
                           baseline_top=r.get("baseline_top", []),
                           edited_top=r.get("edited_top", []),
                           error=r.get("error"))

    def edit_and_observe(self, text: str, transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                         layer: Optional[int] = None,
                         positions: Optional[Sequence[int]] = None) -> tuple[Harvest, Observation]:
        """The full loop in one call: harvest `text`, apply `transform`, write the edit back.

        `transform(acts) -> acts` receives a copy of the [n_tokens, n_embd] matrix and returns
        the edited matrix of the same shape (default: identity, a no-op write that should move
        nothing — a useful sanity check). The write happens at the SAME layer the harvest read
        from (Harvest.layer), which is what makes editing-then-writing a row meaningful. By
        default only the rows the transform actually changed are written back; pass `positions`
        to force a specific set. Returns (harvest, observation).
        """
        h = self.harvest(text, layer)
        edited = h.activations.copy() if transform is None else np.asarray(transform(h.activations.copy()))
        edited = np.ascontiguousarray(edited, dtype="<f4")
        if edited.shape != h.activations.shape:
            raise ValueError(f"transform changed the shape {h.activations.shape} -> {edited.shape}")
        if positions is None:
            changed = np.nonzero(np.abs(edited - h.activations).sum(axis=1) > 0.0)[0]
            positions = changed.tolist() if changed.size else list(range(h.n_tokens))
        rows = edited[list(positions)]
        obs = self.write_state(text, h.layer, positions, rows)
        return h, obs

    def intervene(self, prompt: str, concept: Optional[str] = None, coef: float = 1.0,
                  vector: Optional[ArrayLike] = None, layer: int = 0, **gen) -> dict:
        """POST /intervene (kind:"steer"): push a direction into the residual during generation.

        Either a NAMED concept (one of the server's calibrated probes — see the 'available'
        list it returns on an unknown name) or a RAW `vector` of length n_embd. `coef` scales
        it; `layer` pins the steer layer (0 = the calibrated mid-depth band). Extra generation
        params (max_tokens, steps, topk, ...) pass through as `target`.
        """
        if concept is None and vector is None:
            raise ValueError("intervene needs a concept name or a raw vector")
        target = dict(gen)
        target["prompt"] = prompt
        body: dict = {"kind": "steer", "coef": float(coef), "layer": int(layer), "target": target}
        if concept is not None:
            body["concept"] = concept
        if vector is not None:
            body["vector"] = flatten_values(vector)
        return self._post("/intervene", body)

    def complete(self, prompt: str, **params) -> dict:
        """POST /v1/completions: a plain generation (no white-box). params: max_tokens, steps,
        topk, temperature, ... Returns the OpenAI-ish body {choices, board, layout, usage}."""
        body = dict(params)
        body["prompt"] = prompt
        return self._post("/v1/completions", body)

    def apply_template(self, messages: Sequence[dict], add_assistant: bool = True) -> str:
        """POST /apply_template: render chat `messages` into a prompt string using THE MODEL'S OWN
        embedded chat template (the GGUF's tokenizer.chat_template), applied server-side. This is what
        makes clozn model-agnostic -- Qwen gets ChatML, Llama-3 gets its header format, Gemma gets
        <start_of_turn>, etc. -- instead of a hardcoded Qwen string. `messages` is [{role, content}];
        `add_assistant` ends the prompt with the assistant-turn opener (the generation cue). Raises
        EngineError if the model has no embedded template (never silently mis-formats). Returns the
        rendered prompt string."""
        r = self._post("/apply_template", {"messages": list(messages), "add_assistant": bool(add_assistant)})
        return r["prompt"]

    def score(self, prompt: Optional[str] = None, prompt_ids: Optional[Sequence[int]] = None,
              continuation_ids: Optional[Sequence[int]] = None, continuation: Optional[str] = None,
              topk: int = 0, steer: Optional[dict] = None, steer_vec: Optional[ArrayLike] = None) -> dict:
        """POST /score: teacher-forced per-token logprob of a continuation under given conditions --
        NEVER sampling (the reproduce-and-prove foundation).
        One causal decode of prompt++continuation on the engine reads back, for each continuation
        token, the log-softmax probability the model assigned to the token it was actually forced to
        see next -- usable both to verify a generated reply (self-consistency) and to measure how
        much an influence (memory block / tone dial) shaped an answer (score WITH vs WITHOUT it).

        `prompt_ids` (exact token ids, e.g. from a stored trace) take precedence over `prompt` text;
        likewise `continuation_ids` is the PRIMARY continuation form -- `continuation` text is a
        fallback that retokenizes independently and can drift at the prompt/continuation BPE boundary
        (the server flags this `boundary_approximate` in the response; treat it as approximate).
        `steer`/`steer_vec` mirror /v1/completions' dial path (a raw n_embd direction + {coef, layer}),
        so a scored call can reproduce a steered run's conditions.

        Returns {n_prompt, n_cont, tokens:[{id, piece, logprob, topk?}], sum_logprob}.
        """
        body: dict = {"topk": int(topk)}
        if prompt_ids is not None:
            body["prompt_ids"] = [int(x) for x in prompt_ids]
        elif prompt is not None:
            body["prompt"] = prompt
        if continuation_ids is not None:
            body["continuation_ids"] = [int(x) for x in continuation_ids]
        elif continuation is not None:
            body["continuation"] = continuation
        if steer is not None:
            body["steer"] = steer
        if steer_vec is not None:
            body["steer_vec"] = flatten_values(steer_vec)
        return self._post("/score", body)

    def jlens(self, text: str, layer: Optional[int] = None, topk: int = 5) -> dict:
        """POST /jlens: the J-lens (Jacobian-lens) readout -- per position, the top-k tokens that
        position is 'disposed to say later' (Anthropic 2026, transferred to this GGUF).
        Deterministic linear read, NO sampling. `layer` selects a fitted
        sidecar (omit -> the engine's default tap); an unloaded layer 400s with the available layers.
        Returns {layer, n_tokens, tokens:[piece...], readouts:[[{id,piece,score}...topk]...n_tokens]}."""
        body: dict = {"text": text, "topk": int(topk)}
        if layer is not None:
            body["layer"] = int(layer)
        return self._post("/jlens", body)

    def unembed_row(self, token_id: int) -> dict:
        """POST /jlens/unembed_row: ONE row of the model's own (quantized) unembed/lm_head
        matrix, W_U[token_id] -- the ingredient clozn/behavior/steering/concept_dir.py's dir(c) =
        normalize(J_l^T @ W_U[c]) needs but has no other in-product source for (J_l ships in the
        product J-lens sidecar; W_U doesn't -- see concept_dir.py's BLOCKER_NOTE). Extracted
        server-side via ggml_get_rows (dequantizes whatever GGUF quant type the head is), so only
        d_model floats cross the wire, never the full [vocab, d_model] matrix. Requires the
        engine to have a J-lens sidecar loaded (same requirement as jlens()); 400s otherwise.
        Returns {token_id, piece, d_model, vector:[d_model floats]}."""
        return self._post("/jlens/unembed_row", {"token_id": int(token_id)})


# --------------------------------------------------------------------------- CLI / selftest

def _selftest() -> int:
    """Offline validation of the wire codec and value flattening — needs no server."""
    # A deterministic [5, 7] float32 matrix with fractional values (exercises f32, not ints).
    a = (np.arange(35, dtype="<f4").reshape(5, 7) * 0.5 - 3.25)
    wire = {"dtype": "float32", "shape": [5, 7],
            "data": base64.b64encode(a.tobytes()).decode("ascii")}
    b = decode_tensor(wire)
    assert b.shape == (5, 7), b.shape
    assert np.array_equal(a, b), "tensor codec round-trip is not exact"

    # write flattening is position-major: row i of the slice lands at offset i*n_embd.
    rows = a[[0, 2, 4]]
    flat = flatten_values(rows)
    assert len(flat) == 3 * 7, len(flat)
    assert abs(flat[7] - float(a[2, 0])) < 1e-6, "flatten is not row-major"
    assert abs(flat[0] - float(a[0, 0])) < 1e-6

    # a malformed tensor (byte count != shape product) must raise, not silently truncate.
    bad = {"dtype": "float32", "shape": [5, 8], "data": wire["data"]}
    try:
        decode_tensor(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("decode_tensor accepted a shape/byte mismatch")

    print("selftest OK: tensor codec exact, flatten row-major, shape guard fires")
    return 0


def _demo(args) -> int:
    """A live read -> edit -> write -> observe round-trip against a running cloze-server."""
    eng = EngineClient(host=args.host, port=args.port)
    info = eng.health()
    print(f"server: {info.get('model')}  mode={info.get('mode')}")

    h = eng.harvest(args.text, args.layer)
    print(f"harvested {h.n_tokens} tokens x {h.n_embd} dims at layer {h.layer}")
    print("tokens:", " ".join(repr(t) for t in h.tokens))

    pos = args.pos if args.pos >= 0 else h.n_tokens - 1   # default: the last token (drives next-token)
    pos = max(0, min(pos, h.n_tokens - 1))

    def amplify(acts: np.ndarray) -> np.ndarray:
        acts[pos] = acts[pos] * args.scale       # scale one position's residual
        return acts

    print(f"\nediting position {pos} ({h.tokens[pos]!r}) x{args.scale} at layer {h.layer}")
    _, obs = eng.edit_and_observe(args.text, transform=amplify, layer=args.layer, positions=[pos])
    print(obs.summary())

    # The identity-write control: writing the harvested rows back unchanged should barely move.
    _, ctrl = eng.edit_and_observe(args.text, layer=args.layer, positions=[pos])
    print(f"\ncontrol (identity write): moved_l2={ctrl.moved_l2:.3f} "
          f"(should be ~0 vs {obs.moved_l2:.3f} for the real edit)")
    return 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="cloze-server white-box Python client")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--selftest", action="store_true", help="validate the wire codec offline (no server)")
    ap.add_argument("--demo", action="store_true", help="run a live read->edit->write->observe round-trip")
    ap.add_argument("--text", default="The capital of France is", help="text to harvest for the demo")
    ap.add_argument("--layer", type=int, default=None, help="tap layer (default: server's read tap)")
    ap.add_argument("--pos", type=int, default=-1, help="position to edit (default: last token)")
    ap.add_argument("--scale", type=float, default=4.0, help="scale factor for the edited position")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.demo:
        return _demo(args)
    # Default with no flag: run the offline selftest (safe, needs nothing running).
    return _selftest()


if __name__ == "__main__":
    raise SystemExit(main())
