"""test_engine_substrate -- EngineSubstrate: chat + prompt-mode memory + tone dials on the C++ GGUF
engine, NO PyTorch model resident (clozn_server.EngineSubstrate / RUNTIME_SPLIT.md's keystone). This is
what lets /v1/chat/completions (and, via SUB.chat(), the whole receipts/replay/explain/narrate/
counterfactual stack) run on the fast engine instead of a loaded Qwen-7B.

Model-free throughout -- no C++ engine process, no GPU, no real socket. clozn_server.ENGINE_QWEN is
monkeypatched to a FakeEngine whose `.base` points at a closed local port (127.0.0.1:1, with a short
`.timeout`): _engine_complete_traced's streaming attempt fails fast (no DNS lookup, an immediate refused/
timed-out connect) and falls through to its own pre-existing plain-.complete() fallback -- the "stream
hiccup" path clozn_server.py already ships for exactly this case. That fallback is what every
FakeEngine-backed test below actually exercises; it is NOT itself under test here (see test_hf_trace.py /
test_trace_capture.py for _engine_complete_traced's own streaming-path coverage).

Covers:
  * EngineSubstrate.__init__ / load_substrate("engine"): builds for real when ENGINE_QWEN is configured;
    raises when it isn't; load_substrate degrades that to None (+ a stderr note) rather than crashing boot.
  * EngineSubstrate.chat(): returns the engine's text (stripped); fills mem_out/trace_out exactly like
    QwenSubstrate.chat; folds the prompt-mode memory block into the rendered chat-template prompt (and
    omits it when there is none); forwards the active dials' steer_vec (falling back to disk when the
    live steer carries no strength, mirroring the pre-existing /engine/chat hybrid endpoint).
  * _EngineMemory: card-store-backed .rules (active cards only), .prefix always None, no-op
    consolidate()/reset(), .state() shape.
  * steering.EngineSteer's new SteeringControl-compatible surface: clear/engage/disengage/active/
    save_state/load_state, and generate()'s engage-gated default-dial path (the clean A/B baseline
    Substrate._steer's /steer/check needs).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.server import app as cs          # noqa: E402
import clozn.memory.cards as memory_cards                # noqa: E402
import clozn.memory.mode as memory_mode                 # noqa: E402
from clozn.behavior.steering import EngineSteer   # noqa: E402


# --- a stand-in for cloze_engine.EngineClient: .base points at a closed local port (127.0.0.1:1 --
# an IP literal, so no DNS lookup, and a reserved port nothing ever listens on) with a short .timeout,
# so _engine_complete_traced's streaming attempt excepts out fast and every call below actually exercises
# its pre-existing plain .complete() fallback. -------------------------------------------------------

class FakeEngine:
    def __init__(self, text="hi"):
        self.base = "http://127.0.0.1:1"
        self.timeout = 0.2
        self.text = text
        self.calls = []            # [{"prompt": ..., "params": {...}}, ...] -- every .complete() call seen
        self.template_calls = []   # [messages, ...] -- every .apply_template() call seen

    def apply_template(self, messages, add_assistant=True):
        # Stand in for the engine's per-model templating (chat() now renders via the loaded GGUF's own
        # chat template, not a hardcoded Qwen string). This fake mimics a ChatML model (Qwen), so ChatML
        # markers appear here; on a real engine the FORMAT follows the loaded GGUF (Llama-3 headers,
        # Gemma turns, ...), which the live cross-model proof covers -- not this model-free unit test.
        self.template_calls.append([dict(m) for m in messages])
        return cs._qwen_tmpl(messages)

    def complete(self, prompt, **params):
        self.calls.append({"prompt": prompt, "params": dict(params)})
        return {"choices": [{"text": self.text}]}


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate every path this suite might touch so nothing reads or writes the real ~/.clozn on this
    machine: CLOZN_DIR (_pers()/_disk_dials()/EngineSteer.save_state/load_state), the card store
    (_EngineMemory.rules/.state), and the settings file (_EngineMemory.__init__'s memory_strength read).
    Mirrors test_dial_library_server.py / test_memory_mode.py's own iso fixtures."""
    monkeypatch.setattr(cs, "CLOZN_DIR", str(tmp_path))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    return tmp_path


@pytest.fixture
def fake_engine(monkeypatch):
    """clozn_server.ENGINE_QWEN -> a fresh FakeEngine; ENGINE_STEER reset to None so _engine_steer()
    builds a real steering.EngineSteer on it (construction itself makes no network call -- .compute()
    would, but nothing here has a reason to call it)."""
    fe = FakeEngine()
    monkeypatch.setattr(cs, "ENGINE_QWEN", fe)
    monkeypatch.setattr(cs, "ENGINE_STEER", None)
    return fe


def _no_block(mem, last_user, strength=None):
    return None, [], 0.0


# ==================================================================================== construction

def test_engine_substrate_needs_a_configured_engine(monkeypatch):
    monkeypatch.setattr(cs, "ENGINE_QWEN", None)
    with pytest.raises(RuntimeError):
        cs.EngineSubstrate()


def test_load_substrate_engine_builds_a_real_engine_substrate(iso, fake_engine):
    sub = cs.load_substrate("engine")
    assert isinstance(sub, cs.EngineSubstrate)
    assert sub.name == "engine"
    assert sub.engine is fake_engine
    assert sub.brain is None                       # no SAE on the pure-engine substrate
    assert isinstance(sub.steer, EngineSteer)
    assert sub.memory is sub._mem


def test_load_substrate_engine_degrades_to_none_when_unconfigured(iso, monkeypatch, capsys):
    monkeypatch.setattr(cs, "ENGINE_QWEN", None)
    sub = cs.load_substrate("engine")
    assert sub is None
    assert "engine substrate unavailable" in capsys.readouterr().err


# ==================================================================================== chat() basics

def test_chat_returns_the_engines_text_stripped(iso, fake_engine, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    fake_engine.text = "  hello there  "
    sub = cs.EngineSubstrate()
    assert sub.chat([{"role": "user", "content": "hi"}]) == "hello there"
    assert len(fake_engine.calls) == 1             # the .complete() fallback actually ran


def test_chat_fills_mem_out_and_trace_out_without_raising(iso, fake_engine, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = cs.EngineSubstrate()
    trace_out, mem_out = [], {}
    reply = sub.chat([{"role": "user", "content": "hi"}], trace_out=trace_out, mem_out=mem_out)
    assert reply == "hi"
    assert {k: mem_out[k] for k in ("mode", "applied", "gate")} == {
        "mode": "prompt", "applied": [], "gate": 0.0}
    assert mem_out["prompt_block"] is None
    assert mem_out["assembled_messages"] == [{"role": "user", "content": "hi"}]
    assert trace_out == []                         # the fallback path traces empty, but never raises


# ==================================================================================== chat() -- prompt-mode memory folds into the prompt

def test_chat_folds_the_memory_block_into_the_rendered_prompt(iso, fake_engine, monkeypatch):
    block = "Here is what you know about them:\n- loves rock climbing"
    monkeypatch.setattr(cs, "_prompt_block_for",
                        lambda mem, last_user, strength=None: (block, [{"id": "c1", "text": "x"}], 1.0))
    sub = cs.EngineSubstrate()
    mem_out = {}
    sub.chat([{"role": "user", "content": "what should I do this weekend?"}], mem_out=mem_out)

    assert {k: mem_out[k] for k in ("mode", "applied", "gate")} == {
        "mode": "prompt", "applied": [{"id": "c1", "text": "x"}], "gate": 1.0}
    assert mem_out["prompt_block"] == block
    assert mem_out["assembled_messages"] == [
        {"role": "system", "content": block},
        {"role": "user", "content": "what should I do this weekend?"},
    ]
    # the block-bearing assembled messages were handed to the ENGINE to template (per-model), NOT
    # pre-rendered as Qwen ChatML in Python -- the model-agnostic seam.
    assert fake_engine.template_calls[-1] == mem_out["assembled_messages"]
    sent_prompt = fake_engine.calls[-1]["prompt"]
    assert block in sent_prompt                    # the block actually reached the engine's prompt
    assert "<|im_start|>system" in sent_prompt      # fake mimics a ChatML engine (a Llama engine would emit headers)
    assert "what should I do this weekend?" in sent_prompt


def test_chat_omits_the_block_when_prompt_block_for_returns_none(iso, fake_engine, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = cs.EngineSubstrate()
    sub.chat([{"role": "user", "content": "hi"}])
    assert "loves rock climbing" not in fake_engine.calls[-1]["prompt"]


# ==================================================================================== chat() -- final_prompt capture (backlog #5)

def test_chat_records_the_rendered_final_prompt_in_mem_out(iso, fake_engine, monkeypatch):
    """mem_out.final_prompt is the EXACT rendered string the engine templated -- the same string that
    reached generation (fake_engine.calls[-1]['prompt']). _log_run persists it as run.final_prompt."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = cs.EngineSubstrate()
    mem_out = {}
    sub.chat([{"role": "user", "content": "hi"}], mem_out=mem_out)
    assert mem_out["final_prompt"] == fake_engine.calls[-1]["prompt"]   # exactly what generation saw
    assert mem_out["final_prompt"]                                      # non-empty even with no memory block
    assert "hi" in mem_out["final_prompt"]


def test_chat_final_prompt_contains_the_memory_block(iso, fake_engine, monkeypatch):
    block = "Here is what you know about them:\n- loves rock climbing"
    monkeypatch.setattr(cs, "_prompt_block_for",
                        lambda mem, last_user, strength=None: (block, [{"id": "c1", "text": "x"}], 1.0))
    sub = cs.EngineSubstrate()
    mem_out = {}
    sub.chat([{"role": "user", "content": "plans?"}], mem_out=mem_out)
    # the rendered final_prompt is the post-template form; assembled_messages is its pre-template form.
    assert block in mem_out["final_prompt"]
    assert mem_out["final_prompt"] == fake_engine.calls[-1]["prompt"]


# ==================================================================================== chat() -- dials forward a steer_vec

class FakeSteer:
    """A minimal SteeringControl-compatible double: just chat()'s TONE branch needs (.strength,
    .layer, .steer_vector()) -- unlike a real steering.EngineSteer, no .compute()/harvest() (which
    would need a live engine to derive axis directions from) is involved."""

    def __init__(self, strength, vec, layer=14):
        self.strength = dict(strength)
        self._vec = vec
        self.layer = layer
        self.vector_calls = []

    def steer_vector(self, strength):
        self.vector_calls.append(dict(strength))
        return self._vec


def _bare_engine_substrate(engine, steer, mem=None):
    """EngineSubstrate via object.__new__ (mirrors test_dial_library_server.py's _bare_substrate) --
    exercises chat()'s dial-forwarding logic directly against a hand-picked FakeSteer, without needing
    a real EngineSteer (whose .compute() would need a live engine to harvest axis vectors from)."""
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = engine
    sub.steer = steer
    sub._mem = mem if mem is not None else cs._EngineMemory()
    sub.memory = sub._mem
    return sub


def test_chat_forwards_the_active_dials_steer_vec_to_the_engine(iso, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    fe = FakeEngine()
    steer = FakeSteer(strength={"warm": 1.0}, vec=[0.1, 0.2, 0.3], layer=14)
    sub = _bare_engine_substrate(fe, steer)

    sub.chat([{"role": "user", "content": "hi"}])

    assert steer.vector_calls == [{"warm": 1.0}]
    params = fe.calls[-1]["params"]
    assert params["steer_vec"] == [0.1, 0.2, 0.3]
    assert params["steer"] == {"coef": 1.0, "layer": 14}


def test_chat_skips_steer_vec_when_no_dial_is_active(iso, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    fe = FakeEngine()
    steer = FakeSteer(strength={"warm": 0.0}, vec=None)      # present, but every value is falsy
    sub = _bare_engine_substrate(fe, steer)

    sub.chat([{"role": "user", "content": "hi"}])

    assert steer.vector_calls == []                # any(st.values()) is False -> never even asked
    assert "steer_vec" not in fe.calls[-1]["params"]


def test_chat_falls_back_to_disk_dials_when_the_live_steer_has_no_strength(iso, monkeypatch):
    """self.steer.strength == {} (e.g. nothing dialed yet this process) -> chat() reads the persisted
    personality.json dials instead, exactly like the pre-existing /engine/chat hybrid endpoint does."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    with open(os.path.join(str(iso), "studio_personality.json"), "w", encoding="utf-8") as f:
        json.dump({"warm": 0.8}, f)
    fe = FakeEngine()
    steer = FakeSteer(strength={}, vec=[0.4, 0.5])
    sub = _bare_engine_substrate(fe, steer)

    sub.chat([{"role": "user", "content": "hi"}])

    assert steer.vector_calls == [{"warm": 0.8}]
    assert fe.calls[-1]["params"]["steer_vec"] == [0.4, 0.5]


# ==================================================================================== _EngineMemory

def test_engine_memory_rules_reads_active_cards_only(iso, monkeypatch):
    cards = [{"id": "1", "text": "likes tea", "status": "active"},
             {"id": "2", "text": "pending thing", "status": "pending"},
             {"id": "3", "text": "likes coffee", "status": "active"},
             {"id": "4", "text": "old thing", "status": "disabled"}]
    monkeypatch.setattr(memory_cards, "list_cards", lambda: cards)
    assert cs._EngineMemory().rules == ["likes tea", "likes coffee"]


def test_engine_memory_prefix_is_always_none(iso):
    assert cs._EngineMemory().prefix is None


def test_engine_memory_consolidate_and_reset_are_noops(iso):
    mem = cs._EngineMemory()
    assert mem.consolidate(["some rule"]) == {"ok": True, "mode": "prompt"}
    assert mem.reset() is None                      # a no-op that doesn't raise


def test_engine_memory_state_shape(iso, monkeypatch):
    cards = [{"id": "1", "text": "likes tea", "status": "active"}]
    monkeypatch.setattr(memory_cards, "list_cards", lambda: cards)
    mem = cs._EngineMemory()
    assert mem.state() == {"mode": "prompt", "has_prefix": False, "cards": 1, "rules": ["likes tea"]}


# ==================================================================================== EngineSteer's new SteeringControl-compatible surface

class _FakeEC:
    """A stand-in engine client for steering.EngineSteer ITSELF (distinct from FakeEngine above, which
    stands in for clozn_server's module-level ENGINE_QWEN) -- just enough for generate()'s no-dial path
    (.complete) vs its dialed path (.intervene), so engage()'s gating is observable without a real
    engine or a harvested axis vector."""

    def __init__(self):
        self.complete_calls = []
        self.intervene_calls = []

    def complete(self, prompt, **params):
        self.complete_calls.append(prompt)
        return {"choices": [{"text": "baseline"}]}

    def intervene(self, prompt, **params):
        self.intervene_calls.append(prompt)
        return {"choices": [{"text": "steered"}]}


def test_engine_steer_save_and_load_state_round_trip(tmp_path):
    es = EngineSteer(_FakeEC())
    es.strength = {"warm": 0.7, "concise": -0.3}
    path = str(tmp_path / "nested" / "personality.json")   # save_state must create the parent dir too
    es.save_state(path)

    es2 = EngineSteer(_FakeEC())
    es2.load_state(path)
    assert es2.strength == {"warm": 0.7, "concise": -0.3}


def test_engine_steer_load_state_missing_file_is_a_noop(tmp_path):
    es = EngineSteer(_FakeEC())
    es.strength = {"warm": 0.5}
    es.load_state(str(tmp_path / "nope.json"))
    assert es.strength == {"warm": 0.5}             # untouched


def test_engine_steer_load_state_corrupt_file_is_a_noop_never_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    es = EngineSteer(_FakeEC())
    es.strength = {"warm": 0.5}
    es.load_state(str(path))
    assert es.strength == {"warm": 0.5}             # untouched -- a corrupt file must never crash a boot


def test_engine_steer_clear_empties_strength():
    es = EngineSteer(_FakeEC())
    es.strength = {"warm": 1.0}
    es.clear()
    assert es.strength == {}


def test_engine_steer_active_filters_zeros_keeps_negatives():
    es = EngineSteer(_FakeEC())
    es.strength = {"warm": 1.0, "concise": 0.0, "formal": -0.5}
    assert es.active() == {"warm": 1.0, "formal": -0.5}


def test_engine_steer_generate_gates_dials_on_engage_disengage():
    ec = _FakeEC()
    es = EngineSteer(ec)
    es.ready = True                                 # skip .compute() -- no live engine to harvest from
    es.vecs = {"warm": np.array([1.0, 0.0])}
    es.strength = {"warm": 1.0}

    es.generate("hello")                            # unengaged default path -> the clean, unsteered baseline
    assert ec.complete_calls == ["hello"]
    assert ec.intervene_calls == []

    es.engage()
    es.generate("hello")                            # engaged -> the active dial actually applies
    assert ec.intervene_calls == ["hello"]
    assert ec.complete_calls == ["hello"]            # (unchanged from the call above)

    es.disengage()
    es.generate("hello")                            # back to the clean baseline
    assert ec.complete_calls == ["hello", "hello"]


def test_engine_steer_generate_explicit_strength_is_unaffected_by_engage():
    """An explicit `strength=` kwarg (how /engine/steer/check already calls generate()) bypasses the
    engage gate entirely -- CHANGE 1 only touches the strength=None DEFAULT path."""
    ec = _FakeEC()
    es = EngineSteer(ec)
    es.ready = True
    es.vecs = {"warm": np.array([1.0, 0.0])}
    es.generate("hello", strength={"warm": 1.0})    # never engaged, but strength is explicit
    assert ec.intervene_calls == ["hello"]
    assert ec.complete_calls == []


# ==================================================================================== run_meta (repro metadata)

class _HealthEngine:
    """A stand-in engine exposing just /health, for run_meta(): {model (a GGUF path), mode, n_ctx, device}."""

    def __init__(self, model="/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", mode="autoregressive"):
        self.base = "http://127.0.0.1:1"
        self.timeout = 0.2
        self._h = {"status": "ok", "model": model, "mode": mode,
                   "n_ctx": 4096, "device": "cuda", "gpu_layers": 99}

    def health(self):
        return dict(self._h)

    def apply_template(self, messages, add_assistant=True):
        return cs._qwen_tmpl(messages)   # chat() templates via the engine now (fake mimics a ChatML model)

    def complete(self, prompt, **params):
        return {"choices": [{"text": "ok", "finish_reason": "stop"}]}


def test_quant_from_name_reads_gguf_tags():
    assert cs._quant_from_name("Qwen2.5-0.5B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
    assert cs._quant_from_name("model-q8_0.gguf") == "Q8_0"
    assert cs._quant_from_name("tiny-IQ4_XS.gguf") == "IQ4_XS"
    assert cs._quant_from_name("weights-f16.gguf") == "F16"
    assert cs._quant_from_name("no-quant-here.gguf") is None


def test_run_meta_reads_model_file_quant_and_mode(iso):
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = _HealthEngine()
    meta = sub.run_meta()
    assert meta["model_file"] == "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"    # basename of the /health model path
    assert meta["quant"] == "Q4_K_M"
    assert meta["mode"] == "autoregressive"
    assert meta["sampling"] == "greedy"                                  # chat/chat_stream force temperature 0
    assert meta["sampler_mode"] == "greedy"
    assert meta["temperature"] == 0.0
    assert meta["repetition_penalty"] == 1.0
    assert meta["seed"] == 0
    assert meta["n_ctx"] == 4096 and meta["device"] == "cuda"           # from /health, once the engine exposes them
    assert meta["gpu_layers"] == 99


def test_run_meta_is_cached_after_first_call(iso):
    sub = object.__new__(cs.EngineSubstrate)
    calls = {"n": 0}

    class _CountEngine:
        base = "x"

        def health(self):
            calls["n"] += 1
            return {"model": "m-Q4_0.gguf", "mode": "autoregressive"}

    sub.engine = _CountEngine()
    sub.run_meta()
    sub.run_meta()
    assert calls["n"] == 1                              # /health fetched once, then cached for the session


def test_run_meta_never_raises_on_a_bad_health(iso):
    sub = object.__new__(cs.EngineSubstrate)

    class _BoomEngine:
        base = "x"

        def health(self):
            raise RuntimeError("no engine")

    sub.engine = _BoomEngine()
    assert sub.run_meta() == {"sampler_mode": "greedy", "sampling": "greedy", "temperature": 0.0,
                              "repetition_penalty": 1.0, "seed": 0,
                              "decode": {"mode": "greedy", "temperature": 0.0, "seed": 0}}


def test_run_meta_includes_request_specific_generation_fields_after_chat(iso, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = _HealthEngine()
    sub.steer = None
    sub._mem = cs._EngineMemory()
    sub.memory = sub._mem

    sub.chat([{"role": "user", "content": "hi"}], max_new=17)

    meta = sub.run_meta()
    assert meta["max_tokens"] == 17
    assert meta["stream"] is False
    assert meta["temperature"] == 0.0
    assert meta["seed"] == 0
