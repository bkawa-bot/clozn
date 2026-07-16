"""test_rewrite_route -- POST /engine/rewrite, Route D ("Rewrite (AR)"): the second edit mode alongside
Route A's diffusion Resolve (studio/heavn/modules/edit.mjs). See notes/EDIT_INSTRUCTIONS_DESIGN.md's
Route D and clozn/server/routes/rewrite.py's module docstring for the honesty invariant this enforces:
pins are prompt-level "keep verbatim" constraints, not an engine-enforced invariant, so fidelity is
MEASURED post-hoc (a plain substring check), never assumed.

Model-free throughout: FakeChatSub is a chat()-only double giving full control over the model's reply
text (so pin-fidelity "kept"/"broken" cases are exact and deterministic), plus one end-to-end test using
test_engine_substrate.py's FakeEngine + a real EngineSubstrate to prove the route reaches the actual
chat() plumbing. No C++ engine process, no GPU, no real socket -- drives the real do_POST handler via the
no-socket object.__new__(H) trick (mirrors test_bridge_server.py's _dispatch).
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.server import app as cs   # noqa: E402
from clozn.server.routes import rewrite as rewrite_routes  # noqa: E402
import clozn.memory.cards as memory_cards         # noqa: E402
import clozn.memory.mode as memory_mode          # noqa: E402
import clozn.runs.store as runlog                # noqa: E402


# --- a chat()-only double: full control over the reply text, so pin-fidelity cases are exact -----------

class FakeChatSub:
    name = "engine"

    def __init__(self, reply="the rewritten passage", finish_reason="stop"):
        self.reply = reply
        self.finish_reason = finish_reason
        self.calls = []          # [{"messages", "max_new", "sample", "apply_anchored"}, ...]
        self.raises = None       # set to an Exception instance to make chat() raise

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None,
             reference_tokens=None, apply_anchored=False):
        self.calls.append({"messages": messages, "max_new": max_new, "sample": sample,
                           "apply_anchored": apply_anchored})
        if self.raises is not None:
            raise self.raises
        if mem_out is not None:
            mem_out.update(mode="prompt", applied=[], gate=0.0, prompt_block=None,
                           assembled_messages=messages, final_prompt="<rendered>")
        if trace_out is not None:
            trace_out.extend([])
        return self.reply

    def last_finish_reason(self):
        return self.finish_reason


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate every store this route (or _log_run) might touch -- mirrors test_bridge_server.py's iso."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "no_such.pt")])
    return tmp_path


@pytest.fixture
def fake_chat_sub(iso, monkeypatch):
    sub = FakeChatSub()
    monkeypatch.setattr(cs, "SUB", sub)
    monkeypatch.setattr(cs, "ENGINE", object())     # just needs to be non-None for the availability check
    return sub


def _post_raw(path, body_obj):
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
    raw = _post_raw(path, body_obj)
    head, _, payload = raw.partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(payload.decode("utf-8"))


BASE_BODY = {"text": "The cat sat on the mat.", "instruction": "make it more formal"}


# ==================================================================================== availability / validation

def test_engine_unavailable_is_502(iso, monkeypatch):
    monkeypatch.setattr(cs, "ENGINE", None)
    monkeypatch.setattr(cs, "SUB", FakeChatSub())
    status, body = _post("/engine/rewrite", BASE_BODY)
    assert status == 502
    assert "unavailable" in body["error"]


def test_missing_text_is_400(fake_chat_sub):
    status, body = _post("/engine/rewrite", {"instruction": "make it formal"})
    assert status == 400
    assert "text" in body["error"]


def test_empty_text_is_400(fake_chat_sub):
    status, body = _post("/engine/rewrite", {"text": "", "instruction": "make it formal"})
    assert status == 400


def test_missing_instruction_is_400_and_points_to_resolve(fake_chat_sub):
    status, body = _post("/engine/rewrite", {"text": "hello there"})
    assert status == 400
    assert "instruction" in body["error"]
    assert "Resolve" in body["error"]        # steers the caller to the other edit mode


def test_whitespace_only_instruction_is_400(fake_chat_sub):
    status, body = _post("/engine/rewrite", {"text": "hello there", "instruction": "   "})
    assert status == 400


def test_non_list_pins_is_400(fake_chat_sub):
    status, body = _post("/engine/rewrite", {**BASE_BODY, "pins": "not a list"})
    assert status == 400
    assert "pins" in body["error"]


@pytest.mark.parametrize("pin", [
    {"start": -1, "end": 3},          # out of bounds (negative)
    {"start": 5, "end": 5},           # zero-length
    {"start": 10, "end": 3},          # reversed
    {"start": 0, "end": 9999},        # past len(text)
    {"start": "0", "end": 3},         # wrong type
    {"foo": "bar"},                   # missing keys
])
def test_invalid_pin_shapes_are_400(fake_chat_sub, pin):
    status, body = _post("/engine/rewrite", {**BASE_BODY, "pins": [pin]})
    assert status == 400
    assert "pin" in body["error"]


def test_max_tokens_must_be_a_positive_integer(fake_chat_sub):
    status, body = _post("/engine/rewrite", {**BASE_BODY, "max_tokens": "lots"})
    assert status == 400
    status, body = _post("/engine/rewrite", {**BASE_BODY, "max_tokens": 0})
    assert status == 400


# ==================================================================================== prompt construction

def test_prompt_quotes_pins_as_verbatim_constraints_and_carries_the_instruction(fake_chat_sub):
    text = "The quick brown fox jumps over the lazy dog."
    pins = [{"start": 4, "end": 15}]
    assert text[4:15] == "quick brown"   # sanity on the fixture's own offsets
    body = {"text": text, "instruction": "make it more dramatic", "pins": pins}
    status, out = _post("/engine/rewrite", body)
    assert status == 200

    sent = fake_chat_sub.calls[-1]["messages"]
    assert sent[0]["role"] == "system"
    assert sent[1]["role"] == "user"
    system, user = sent[0]["content"], sent[1]["content"]
    assert '"quick brown"' in system            # the pinned substring, quoted verbatim
    assert "HARD CONSTRAINT" in system
    assert "make it more dramatic" in user
    assert text in user                         # the original text rides along for context


def test_no_pins_omits_the_constraint_paragraph(fake_chat_sub):
    status, out = _post("/engine/rewrite", {**BASE_BODY, "pins": []})
    assert status == 200
    system = fake_chat_sub.calls[-1]["messages"][0]["content"]
    assert "HARD CONSTRAINT" not in system


def test_multiple_pins_all_quoted_in_order(fake_chat_sub):
    text = "Alpha beta gamma delta epsilon."
    pins = [{"start": 0, "end": 5}, {"start": 11, "end": 16}]   # "Alpha", "gamma"
    assert text[0:5] == "Alpha" and text[11:16] == "gamma"
    _post("/engine/rewrite", {"text": text, "instruction": "reorder for emphasis", "pins": pins})
    system = fake_chat_sub.calls[-1]["messages"][0]["content"]
    assert '"Alpha"' in system and '"gamma"' in system
    assert system.index('"Alpha"') < system.index('"gamma"')    # pin order preserved


def test_sample_flag_defaults_true_and_is_overridable(fake_chat_sub):
    _post("/engine/rewrite", BASE_BODY)
    assert fake_chat_sub.calls[-1]["sample"] is True
    _post("/engine/rewrite", {**BASE_BODY, "sample": False})
    assert fake_chat_sub.calls[-1]["sample"] is False


def test_apply_anchored_is_not_forced_for_a_non_engine_substrate_double(fake_chat_sub):
    """FakeChatSub isn't an EngineSubstrate instance -- the isinstance-gated apply_anchored=True branch
    (mirrors openai.py's /v1/chat/completions) must not fire for it, exactly like the real code path
    only opts EngineSubstrate in. See the end-to-end test below for the real-substrate case."""
    _post("/engine/rewrite", BASE_BODY)
    assert fake_chat_sub.calls[-1]["apply_anchored"] is False


# ==================================================================================== pin fidelity: MEASURED, not assumed

def test_pin_fidelity_all_kept(fake_chat_sub):
    text = "The cat sat on the mat."
    pins = [{"start": 4, "end": 7}]      # "cat"
    fake_chat_sub.reply = "A feline named cat rested peacefully."   # contains "cat" verbatim
    status, out = _post("/engine/rewrite", {"text": text, "instruction": "make it fancier", "pins": pins})
    assert status == 200
    assert out["pins"] == [{"start": 4, "end": 7, "text": "cat", "kept": True}]
    assert out["all_pins_kept"] is True


def test_pin_fidelity_reports_a_broken_pin_never_silently_accepts_it(fake_chat_sub):
    text = "The cat sat on the mat."
    pins = [{"start": 4, "end": 7}, {"start": 19, "end": 22}]   # "cat", "mat"
    assert text[4:7] == "cat" and text[19:22] == "mat"
    fake_chat_sub.reply = "A feline rested on the rug."          # neither "cat" nor "mat" survived
    status, out = _post("/engine/rewrite", {"text": text, "instruction": "make it fancier", "pins": pins})
    assert status == 200
    assert out["pins"][0] == {"start": 4, "end": 7, "text": "cat", "kept": False}
    assert out["pins"][1] == {"start": 19, "end": 22, "text": "mat", "kept": False}
    assert out["all_pins_kept"] is False


def test_pin_fidelity_partial_one_kept_one_broken(fake_chat_sub):
    text = "The cat sat on the mat."
    pins = [{"start": 4, "end": 7}, {"start": 19, "end": 22}]   # "cat", "mat"
    fake_chat_sub.reply = "The cat rested on the rug."           # "cat" survives, "mat" does not
    status, out = _post("/engine/rewrite", {"text": text, "instruction": "make it fancier", "pins": pins})
    kept = {p["text"]: p["kept"] for p in out["pins"]}
    assert kept == {"cat": True, "mat": False}
    assert out["all_pins_kept"] is False


def test_pin_fidelity_is_exact_not_fuzzy(fake_chat_sub):
    """A pin that survives with different CASE or whitespace is a broken pin, not a near-match -- the
    fidelity check must never silently upgrade a paraphrase to 'kept'."""
    text = "The Cat sat on the mat."
    pins = [{"start": 4, "end": 7}]      # "Cat" (capital C)
    fake_chat_sub.reply = "The cat rested peacefully."   # lowercase "cat" -- NOT an exact match for "Cat"
    status, out = _post("/engine/rewrite", {"text": text, "instruction": "rewrite it", "pins": pins})
    assert out["pins"] == [{"start": 4, "end": 7, "text": "Cat", "kept": False}]


# ==================================================================================== the honest-labeling contract

def test_response_carries_the_binding_honest_note_verbatim(fake_chat_sub):
    status, out = _post("/engine/rewrite", BASE_BODY)
    assert out["note"] == "regenerates the unpinned text — not a bidirectional resolve"
    assert out["note"] == rewrite_routes.NOTE
    # the word "instruction" is fine on THIS tool (Rewrite is explicitly the instruction-taking mode) --
    # the binding constraint from the design doc is that it must never appear UNQUALIFIED on the
    # RESOLVE tool, which this endpoint is not.


def test_response_shape(fake_chat_sub):
    status, out = _post("/engine/rewrite", BASE_BODY)
    assert status == 200
    assert set(out.keys()) == {"text", "pins", "all_pins_kept", "finish_reason", "note", "run_id"}
    assert out["text"] == fake_chat_sub.reply
    assert out["finish_reason"] == "stop"
    assert isinstance(out["run_id"], str) and out["run_id"]


# ==================================================================================== run logging

def test_run_is_logged_with_the_edit_route_meta(fake_chat_sub):
    text = "The cat sat on the mat."
    pins = [{"start": 4, "end": 7}, {"start": 19, "end": 22}]
    fake_chat_sub.reply = "The cat rested on the rug."
    status, out = _post("/engine/rewrite", {"text": text, "instruction": "rewrite", "pins": pins})
    rid = out["run_id"]
    run = runlog.get_run(rid)
    assert run is not None
    assert run["source"] == "engine_rewrite"
    assert run["response"] == "The cat rested on the rug."
    assert run["finish_reason"] == "stop"
    assert run["meta"]["edit_route"] == "rewrite_ar"
    assert run["meta"]["pins_total"] == 2
    assert run["meta"]["pins_kept"] == 1


def test_generation_failure_returns_502_and_logs_the_error(fake_chat_sub):
    fake_chat_sub.raises = RuntimeError("worker unreachable")
    status, out = _post("/engine/rewrite", BASE_BODY)
    assert status == 502
    assert "worker unreachable" in out["error"]
    last = runlog.list_runs(1)[0]
    run = runlog.get_run(last["id"])
    assert run["source"] == "engine_rewrite"
    assert run["error"] is not None and "worker unreachable" in run["error"]


# ==================================================================================== end-to-end: reaches the real EngineSubstrate.chat plumbing

def test_end_to_end_reaches_the_engine_through_the_real_substrate(tmp_path, monkeypatch):
    """Mirrors test_engine_substrate.py's FakeEngine pattern -- proves the route's prompt actually reaches
    ctx._engine_complete_traced -> the engine (not just a double), and that a real EngineSubstrate.chat()
    call is what produced the response."""
    import clozn.memory.anchored as anchored_memory

    monkeypatch.setattr(cs, "CLOZN_DIR", str(tmp_path))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(anchored_memory, "BAGS_PATH", str(tmp_path / "anchored_bags.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))

    class FakeEngine:
        def __init__(self, text="A formally rewritten cat."):
            self.base = "http://127.0.0.1:1"    # closed port -> _engine_complete_traced falls to .complete()
            self.timeout = 0.2
            self.text = text
            self.calls = []

        def apply_template(self, messages, add_assistant=True):
            return cs._qwen_tmpl(messages)

        def complete(self, prompt, **params):
            self.calls.append({"prompt": prompt, "params": dict(params)})
            return {"choices": [{"text": self.text}]}

    fe = FakeEngine()
    monkeypatch.setattr(cs, "ENGINE", fe)
    monkeypatch.setattr(cs, "ENGINE_STEER", None)
    monkeypatch.setattr(cs, "_prompt_block_for", lambda mem, last_user, strength=None: (None, [], 0.0))
    sub = cs.EngineSubstrate()
    monkeypatch.setattr(cs, "SUB", sub)

    text = "The cat sat on the mat."
    pins = [{"start": 4, "end": 7}]
    status, out = _post("/engine/rewrite",
                        {"text": text, "instruction": "make it formal", "pins": pins})

    assert status == 200
    assert out["text"] == "A formally rewritten cat."
    assert out["note"] == rewrite_routes.NOTE
    # the constrained prompt actually reached the engine
    sent_prompt = fe.calls[-1]["prompt"]
    assert '"cat"' in sent_prompt
    assert "make it formal" in sent_prompt
    assert "HARD CONSTRAINT" in sent_prompt
    # pin fidelity measured against the REAL reply text (not assumed): fe.text genuinely contains "cat"
    assert out["pins"] == [{"start": 4, "end": 7, "text": "cat", "kept": True}]
    assert out["all_pins_kept"] is True

    # and the broken-pin case, still through the real substrate: a reply that dropped the pinned word
    fe.text = "A formally rewritten feline."
    status2, out2 = _post("/engine/rewrite",
                          {"text": text, "instruction": "make it formal", "pins": pins})
    assert status2 == 200
    assert out2["pins"] == [{"start": 4, "end": 7, "text": "cat", "kept": False}]
    assert out2["all_pins_kept"] is False
