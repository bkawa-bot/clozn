"""tests/test_generation_guard_server.py -- the closed-loop disposition guardrail
(clozn/server/generation_guard.py) wired into POST /v1/chat/completions as the OPT-IN, DEFAULT-OFF
`clozn_guard` extension field (FRONTIER_BETS section 9.1 / experiment A1.1).

Model-free: drives the REAL clozn_server do_POST handler with no socket (object.__new__(H)), isolated
runlog/cards/settings/eval stores, and a FAKE engine client (never a live one). Concept resolution
(resolve_token_id / dir(c)) runs through the REAL clozn.behavior.steering.concept_dir math against tiny
on-disk J-lens/unembed FIXTURE files (mirrors tests/test_concept_dir.py's own fixture convention) -- only
the raw engine HTTP calls (.score/.complete/.intervene/.jlens/.apply_template) are faked.
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from clozn.server import app as cs                # noqa: E402
from clozn.server import generation_guard as gg    # noqa: E402
from clozn.memory import mode as memory_mode       # noqa: E402
import clozn.memory.cards as memory_cards          # noqa: E402
import clozn.runs.store as runlog                  # noqa: E402


MODEL = "fake-clozn-model"
D_MODEL = 32
LAYER = gg.DEFAULT_LAYER   # A1.1's own validated tap (16) -- the fixture below is fitted at this layer so
                          # every ON-success test can rely on the guard's OWN default without overriding it
TRIGGER_TOKEN_ID = 5      # must be < the fixture's vocab size (D_MODEL) -- see _write_unembed_fixture
TRIGGER_MARKER = "BANNED_TRIGGER"


# ==================================================================================== fixtures (dir(c) plumbing)

def _orthogonal(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q


def _write_jlens_fixture(tmp_path, *, d_model=D_MODEL, layer=LAYER, seed=1):
    jdir = tmp_path / "jlens"
    jdir.mkdir()
    manifest = {"model": "fixture", "d_model": d_model, "vocab": d_model, "layers": [layer],
               "engine_default_tap_layer": layer}
    (jdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    J = _orthogonal(seed, d_model).astype(np.float32)
    J.astype("<f2").tofile(str(jdir / f"J_layer{layer}.f16"))
    return str(jdir)


def _write_unembed_fixture(tmp_path, *, d_model=D_MODEL, vocab=D_MODEL, seed=2):
    udir = tmp_path / "unembed"
    udir.mkdir()
    q = _orthogonal(seed, d_model)[:vocab].astype(np.float32)
    np.save(str(udir / "norm_weight.npy"), np.ones(d_model, dtype=np.float32))
    np.save(str(udir / "lm_head_weight.npy"), q)
    (udir / "unembed_meta.json").write_text(json.dumps({"rms_norm_eps": 1e-6}), encoding="utf-8")
    return str(udir)


# ==================================================================================== fake engine + substrate

class FakeGuardEngine:
    """Stands in for cloze_engine.EngineClient for the guard's production adapter: `.score` (token
    resolution), `.complete`/`.intervene` (chunked generation, scripted per-call-order), `.jlens`
    (disposition read, driven by a marker substring so tests stay simple), `.apply_template` (prompt
    render)."""

    def __init__(self, *, vocab, complete_pieces=(), intervene_pieces=(),
                trigger_token_id=TRIGGER_TOKEN_ID, marker=TRIGGER_MARKER):
        self.vocab = dict(vocab)
        self._complete_pieces = list(complete_pieces)
        self._intervene_pieces = list(intervene_pieces)
        self.complete_calls = []
        self.intervene_calls = []
        self.jlens_calls = []
        self._trigger_token_id = trigger_token_id
        self._marker = marker

    def apply_template(self, messages):
        return "PROMPT:" + "".join(m.get("content", "") for m in messages)

    def score(self, prompt=None, continuation=None, topk=0, **kw):
        ids = self.vocab.get(continuation)
        if ids is None:
            raise AssertionError(f"unexpected /score continuation in test: {continuation!r}")
        return {"tokens": [{"id": i, "piece": continuation} for i in ids]}

    def complete(self, prompt, max_tokens=None, **kw):
        idx = len(self.complete_calls)
        self.complete_calls.append({"prompt": prompt, "max_tokens": max_tokens, "kw": kw})
        return {"choices": [{"text": self._complete_pieces[idx], "finish_reason": "length"}]}

    def intervene(self, prompt, vector=None, coef=None, layer=None, max_tokens=None, **kw):
        idx = len(self.intervene_calls)
        self.intervene_calls.append({"prompt": prompt, "vector": vector, "coef": coef, "layer": layer,
                                     "max_tokens": max_tokens, "kw": kw})
        return {"choices": [{"text": self._intervene_pieces[idx], "finish_reason": "length"}]}

    def jlens(self, text, layer=None, topk=5):
        self.jlens_calls.append({"text": text, "layer": layer, "topk": topk})
        if self._marker in text:
            return {"readouts": [[{"id": self._trigger_token_id, "score": 9.0}]]}
        return {"readouts": [[]]}


class FakeGuardSub:
    """The minimal substrate surface the guard path needs: `.chat` only to satisfy try_post's worker-
    availability gate (never actually called on the guard path -- would be a real bug if it were), and
    `.engine` (the raw client the production adapter drives directly)."""
    name = "engine"

    def __init__(self, engine):
        self.engine = engine

    def chat(self, *a, **kw):
        raise AssertionError("the guard path must never fall through to sub.chat()")


class TraceSub:
    """A qwen-shaped substrate for the OFF path -- mirrors test_selective_generation_server.py's TraceSub,
    proving the guard's off-switch leaves the ordinary chat() pipeline completely untouched."""
    name = "qwen"

    def __init__(self, reply="An ordinary ungated reply."):
        class _Mem:
            memory_strength = 1.0
            rules: list = []
            prefix = None
        self.memory = _Mem()
        self._mem = self.memory
        self._reply = reply
        self._run_meta = {"model_id": MODEL, "sampler_mode": "greedy", "sampling": "greedy",
                          "temperature": 0.0}

        class _Steer:
            strength: dict = {}

            def active(self):
                return {}
        self.steer = _Steer()

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=False)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        if trace_out is not None:
            trace_out.extend([{"piece": "ordinary", "conf": 0.9}])
        return self._reply

    def last_finish_reason(self):
        return "stop"

    def run_meta(self):
        return dict(self._run_meta)


def _dispatch(path, body_obj):
    raw = json.dumps(body_obj).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    return h.wfile.getvalue()


def _post(path, body_obj):
    raw = _dispatch(path, body_obj)
    status = int(raw.split(b" ", 2)[1])
    _, _, payload = raw.partition(b"\r\n\r\n")
    return status, json.loads(payload.decode("utf-8"))


def _body(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "tell me something"}], **extra}


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "no_such.pt")])
    monkeypatch.delenv("CLOZN_JLENS_DIR", raising=False)
    monkeypatch.delenv("CLOZN_DIRC_UNEMBED_DIR", raising=False)
    return tmp_path


@pytest.fixture
def dirc_fixtures(tmp_path, monkeypatch):
    """Real, tiny on-disk J-lens + unembed exports so ConceptSteer.compute() succeeds for real -- see the
    module docstring. Only used by the ON+success tests; the fail-closed test deliberately configures
    neither."""
    jdir = _write_jlens_fixture(tmp_path)
    udir = _write_unembed_fixture(tmp_path)
    monkeypatch.setenv("CLOZN_JLENS_DIR", jdir)
    monkeypatch.setenv("CLOZN_DIRC_UNEMBED_DIR", udir)
    return tmp_path


# ==================================================================================== OFF = byte-identical

def test_field_absent_is_byte_identical_to_the_ordinary_chat_pipeline(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", TraceSub())
    status, out = _post("/v1/chat/completions", _body())
    assert status == 200
    assert "clozn_guard_receipt" not in out
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "An ordinary ungated reply."}
    assert set(out.keys()) == {"id", "object", "created", "model", "choices", "clozn_run_id"}


def test_field_explicitly_empty_is_also_off(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", TraceSub())
    status, out = _post("/v1/chat/completions", _body(clozn_guard={}))
    assert status == 200
    assert "clozn_guard_receipt" not in out


def test_server_setting_off_by_default(iso, monkeypatch):
    """No server-wide default ever saved -> parse_guard_spec must read its documented default (off), not
    silently opt in."""
    monkeypatch.setattr(cs, "SUB", TraceSub())
    status, out = _post("/v1/chat/completions", _body())
    assert status == 200
    assert "clozn_guard_receipt" not in out


# ==================================================================================== malformed body -> 400

def test_malformed_guard_body_is_a_400(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", TraceSub())
    status, out = _post("/v1/chat/completions", _body(clozn_guard={"concepts": "violence"}))
    assert status == 400
    assert out["error"]["param"] == "clozn_guard"


# ==================================================================================== streaming conflict -> refused

def test_guard_with_streaming_is_refused_not_silently_ignored(iso, dirc_fixtures, monkeypatch):
    engine = FakeGuardEngine(vocab={" violence": [TRIGGER_TOKEN_ID]})
    monkeypatch.setattr(cs, "SUB", FakeGuardSub(engine))
    status, out = _post("/v1/chat/completions",
                        _body(stream=True, clozn_guard={"concepts": ["violence"]}))
    assert status == 422
    assert out["error"]["code"] == "guard_streaming_unsupported"
    assert out["error"]["param"] == "clozn_guard"


# ==================================================================================== FAIL CLOSED (no dir(c) source)

def test_fail_closed_when_concept_cannot_be_resolved(iso, monkeypatch):
    """Neither a lab unembed export NOR a working engine unembed_row is configured -- compute() degrades
    to 'unembed_unavailable', and the WHOLE request must be refused, never silently generate an unguarded
    reply."""
    engine = FakeGuardEngine(vocab={" violence": [TRIGGER_TOKEN_ID]})
    monkeypatch.setattr(cs, "SUB", FakeGuardSub(engine))
    status, out = _post("/v1/chat/completions",
                        _body(clozn_guard={"concepts": ["violence"]}, max_tokens=8))
    assert status == 422
    assert out["error"]["code"] == "guard_unavailable"
    assert "violence" in out["error"]["message"]
    assert "unavailable" in out["error"]["message"]
    assert len(engine.complete_calls) == 0   # never generated anything under a promise it couldn't keep


# ==================================================================================== ON + fire

def test_on_fire_corrects_the_flagged_chunk_and_builds_a_receipt(iso, dirc_fixtures, monkeypatch):
    engine = FakeGuardEngine(
        vocab={" violence": [TRIGGER_TOKEN_ID]},
        complete_pieces=[f"{TRIGGER_MARKER} content ", "clean chunk two"],
        intervene_pieces=["safe corrected content "],
    )
    monkeypatch.setattr(cs, "SUB", FakeGuardSub(engine))
    status, out = _post("/v1/chat/completions", _body(
        clozn_guard={"concepts": ["violence"], "threshold": 1.0, "chunk_tokens": 8, "layer": LAYER},
        max_tokens=16,
    ))
    assert status == 200
    receipt = out["clozn_guard_receipt"]
    assert receipt["n_fires"] == 1
    assert receipt["cap_reached"] is False
    assert receipt["concepts"] == ["violence"]
    fire = receipt["fires"][0]
    assert fire["concept"] == "violence"
    assert fire["pre_activation"] == pytest.approx(9.0)
    assert fire["counter_strength"] == gg.DEFAULT_COUNTER_STRENGTH
    assert receipt["caveat"] == gg.GUARD_CAVEAT
    assert TRIGGER_MARKER not in out["choices"][0]["message"]["content"]
    assert out["choices"][0]["message"]["content"] == "safe corrected content clean chunk two"
    assert len(engine.intervene_calls) == 1
    assert engine.intervene_calls[0]["layer"] == LAYER
    # coef = counter_strength * this layer's (global, uncalibrated-per-model) median residual norm --
    # concept_dir.VALIDATED_MEDIAN_RESID_NORM, since no per-model calibration file is configured here.
    from clozn.behavior.steering.concept_dir import VALIDATED_MEDIAN_RESID_NORM
    assert engine.intervene_calls[0]["coef"] == pytest.approx(
        gg.DEFAULT_COUNTER_STRENGTH * VALIDATED_MEDIAN_RESID_NORM[LAYER])
    assert engine.intervene_calls[0]["coef"] < 0   # steers AWAY from the concept -- negative by convention
    assert "clozn_run_id" in out
    stored = runlog.get_run(out["clozn_run_id"])
    assert stored["meta"]["clozn_guard"]["n_fires"] == 1


# ==================================================================================== ON + no fire

def test_on_no_fire_never_calls_intervene(iso, dirc_fixtures, monkeypatch):
    engine = FakeGuardEngine(
        vocab={" violence": [TRIGGER_TOKEN_ID]},
        complete_pieces=["clean chunk one ", "clean chunk two"],
    )
    monkeypatch.setattr(cs, "SUB", FakeGuardSub(engine))
    status, out = _post("/v1/chat/completions", _body(
        clozn_guard={"concepts": ["violence"], "threshold": 1.0, "chunk_tokens": 8, "layer": LAYER},
        max_tokens=16,
    ))
    assert status == 200
    receipt = out["clozn_guard_receipt"]
    assert receipt["n_fires"] == 0
    assert receipt["fires"] == []
    assert receipt["cap_reached"] is False
    assert out["choices"][0]["message"]["content"] == "clean chunk one clean chunk two"
    assert len(engine.intervene_calls) == 0


# ==================================================================================== cap reached

def test_cap_reached_is_labeled_honestly(iso, dirc_fixtures, monkeypatch):
    engine = FakeGuardEngine(
        vocab={" violence": [TRIGGER_TOKEN_ID]},
        complete_pieces=[f"{TRIGGER_MARKER} one", "corrected one",
                         f"{TRIGGER_MARKER} two", "plain rest"],
        intervene_pieces=["corrected one"],
    )
    monkeypatch.setattr(cs, "SUB", FakeGuardSub(engine))
    status, out = _post("/v1/chat/completions", _body(
        clozn_guard={"concepts": ["violence"], "threshold": 1.0, "chunk_tokens": 8, "max_fires": 1,
                    "layer": LAYER},
        max_tokens=24,
    ))
    assert status == 200
    receipt = out["clozn_guard_receipt"]
    assert receipt["n_fires"] == 1
    assert receipt["cap_reached"] is True
    assert receipt["cap_note"] == gg.GUARD_CAP_NOTE
    # the second trigger, past the cap, survives uncorrected in the final reply -- honest, not hidden
    assert TRIGGER_MARKER in out["choices"][0]["message"]["content"]
    assert len(engine.intervene_calls) == 1   # never re-steers past the cap


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
