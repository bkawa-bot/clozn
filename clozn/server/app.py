"""clozn.server.app -- the unified studio backend. One port, one model, the whole white-box surface.

  substrate 'qwen' (default): ONE Qwen-7B serves BOTH the brain (/think -- concepts the model engages)
                              AND the memory (/say /consolidate /check /whatlearned) -- they share the
                              single loaded model, so the instrument's brain and memory tabs are both live.
  substrate 'dream':          Dream-7B serves /denoise (the diffusion window).

Only one 7B fits the GPU, so switching substrates re-execs the process with the other one (a clean GPU);
the instrument shows the active substrate and offers the switch. Serves the instrument + every window
from studio, so the iframes' fetches all land here.

    cloze .venv python -m clozn.server.app --port 8090
"""
import argparse
import json
import os
import secrets
import subprocess
import sys
import threading
import time

# `python -m clozn.server.app` executes this file as `__main__`, a SEPARATE module object from
# `clozn.server.app` in sys.modules terms. Every route module below does `from clozn.server import app
# as ctx` to read the shared SUB/SUBNAME/ENGINE state live -- if that import were left to run normally,
# it would import (and re-execute) THIS FILE A SECOND TIME under its real dotted name, creating a second,
# disconnected copy of all this module's state: main() would mutate the __main__ copy's SUB/SUBNAME
# while every route handler reads the OTHER copy's untouched defaults (SUB=None, SUBNAME="qwen") --
# invisible to the test suite (which only ever imports the dotted name, never runs this as __main__) but
# fatal to a live `-m` boot. Aliasing sys.modules here, before any of the routes/* imports at the bottom
# of this file run, makes `from clozn.server import app` resolve to THIS SAME object either way.
sys.modules.setdefault("clozn.server.app", sys.modules[__name__])

from clozn.server.config import HERE, REPO_ROOT, DEMO, CLOZN_DIR   # noqa: E402 (side effects: sys.path/env/stdout)

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer   # noqa: E402

try:
    from cloze_engine import EngineClient, EngineError
    ENGINE = EngineClient(port=int(os.environ.get("CLOZN_ENGINE_PORT", "8091")))            # the live C++ runtime
    ENGINE_QWEN = EngineClient(port=int(os.environ.get("CLOZN_ENGINE_QWEN_PORT", "8092")))  # a Qwen GGUF engine -> concepts
except Exception:
    ENGINE = ENGINE_QWEN = None
    class EngineError(RuntimeError):   # fallback so `except EngineError` resolves even if the SDK import failed
        pass


def _git_commit():
    """Best-effort build/repro id. Returns None when this checkout is not a git repo or git is unavailable."""
    if getattr(_git_commit, "_read", False):
        return getattr(_git_commit, "_value", None)
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                      stderr=subprocess.DEVNULL, timeout=2)
        val = out.decode("utf-8", "replace").strip() or None
    except Exception:
        val = None
    _git_commit._value = val
    _git_commit._read = True
    return val


def _without_unknowns(d):
    """Drop only unknown values; keep honest falsy repro values like 0 temperature or seed."""
    return {k: v for k, v in (d or {}).items() if v is not None}


def _openai_finish_reason(fr):
    """OpenAI-compatible responses need a concrete finish_reason string even if a substrate cannot provide one."""
    return fr if isinstance(fr, str) and fr else "stop"


def _qwen_generation_meta(max_new=None, sample=True, stream=None):
    return _without_unknowns({
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "sampler_mode": "sample" if sample else "greedy",
        "sampling": "sample" if sample else "greedy",
        "temperature": 0.7 if sample else 0.0,
        "top_p": 0.9 if sample else None,
        "repetition_penalty": 1.3,
        "no_repeat_ngram_size": 3,
        # Self-describing decode block (REPRODUCE_AND_PROVE_PLAN S2). seed is honestly null: HF
        # generate() sets no fixed seed on this path, so a sampled Qwen run is NOT exactly
        # reproducible -- the deliberate contrast with the engine's reproducible {"seed": 0}.
        "decode": {"mode": "sample" if sample else "greedy",
                   "temperature": 0.7 if sample else 0.0,
                   "top_p": 0.9 if sample else None,
                   "seed": None},
        "max_tokens": int(max_new) if max_new is not None else None,
        "stream": bool(stream) if stream is not None else None,
    })


# REPRODUCE_AND_PROVE_PLAN S5: settings-exposed interactive-chat sampling (Ollama/llama.cpp's canonical
# defaults -- model-agnostic, since clozn serves any GGUF). "sampling" is the master on/off; OFF (or a
# caller that explicitly asked for greedy, e.g. every receipt/replay/forced-scoring call) always decodes
# temperature 0, byte-identical to pre-S5 behavior. top_p/top_k ride here as REQUESTED settings, but see
# _resolve_sampling's docstring: this engine build does not enforce them.
_SAMPLING_DEFAULTS = {
    "sampling": True,
    "sample_temperature": 0.8,
    "sample_top_p": 0.9,
    "sample_top_k": 40,
    "sample_repeat_penalty": 1.1,
}


def _resolve_sampling(want_sample):
    """Whether THIS engine generation should sample, and under what params -- S5. `want_sample` is the
    caller's own per-call ask (EngineSubstrate.chat's `sample` arg; chat_stream always asks True and lets
    the setting alone decide). Returns None (greedy, temperature 0 -- exactly pre-S5 behavior) when
    `want_sample` is falsy OR the persisted "sampling" setting is off; otherwise a dict {"on": True,
    "temperature", "top_p", "top_k", "repeat_penalty", "seed"} with a FRESH per-turn seed.

    Critical for the receipt/replay/rederive stack: `want_sample=False` (replay.py's
    `sampled = not changes.get("greedy")`, always False for receipts) short-circuits to None BEFORE the
    setting is even read -- so turning interactive sampling on/off can never make a receipt's forced-greedy
    regen non-deterministic. Only the caller's own request gates that.

    HONESTY: only temperature/repeat_penalty/seed actually reach the engine's sampler (verified against
    engine/core/serve/server_shared.hpp's sample_from + engine/core/src/sample.cpp: a full-vocabulary
    temperature-scaled softmax draw + repetition penalty -- there is no nucleus (top_p) or top-k truncation
    in this engine build). top_p/top_k are still resolved and recorded (self-describing settings a future
    engine build could honor), just never sent as request-body keys the engine reads."""
    if not want_sample:
        return None
    import clozn.memory.mode as memory_mode
    if not bool(memory_mode.get_setting("sampling", _SAMPLING_DEFAULTS["sampling"])):
        return None
    return {
        "on": True,
        "temperature": float(memory_mode.get_setting("sample_temperature", _SAMPLING_DEFAULTS["sample_temperature"])),
        "top_p": float(memory_mode.get_setting("sample_top_p", _SAMPLING_DEFAULTS["sample_top_p"])),
        "top_k": int(memory_mode.get_setting("sample_top_k", _SAMPLING_DEFAULTS["sample_top_k"])),
        "repeat_penalty": float(memory_mode.get_setting("sample_repeat_penalty",
                                                         _SAMPLING_DEFAULTS["sample_repeat_penalty"])),
        # A FRESH seed every turn (not a fixed one) -- what makes a sampled reply vary turn to turn while
        # still being independently reproducible: re-POSTing /v1/completions with this same seed+params
        # reproduces the text (mode.decode records it on the run).
        "seed": secrets.randbits(63),
    }


def _sampling_settings():
    """The persisted S5 sampling settings (on/off + the four params) for the GET/POST /sampling/mode
    route -- what's actually configured, not what one specific turn resolved to. Unlike _resolve_sampling,
    this never generates a seed (a GET must never mutate anything)."""
    import clozn.memory.mode as memory_mode
    return {k: memory_mode.get_setting(k, v) for k, v in _SAMPLING_DEFAULTS.items()}


def _engine_generation_meta(max_new=None, stream=None, sample=None):
    """Reproducibility metadata for one engine generation (REPRODUCE_AND_PROVE_PLAN S2/S5).

    `sample`: None (the default) always yields the honest greedy regime -- byte-identical to every
    pre-S5 caller (run_meta()'s static baseline, and any chat()/chat_stream() call that resolved to
    greedy). Pass a _resolve_sampling() dict to flip the decode block to the real sampled regime this
    generation actually used."""
    on = bool(sample and sample.get("on"))
    if on:
        top = {"sampler_mode": "sample", "sampling": "sample", "temperature": sample["temperature"],
               "repetition_penalty": sample["repeat_penalty"], "seed": sample["seed"]}
        decode = {"mode": "sample", "temperature": sample["temperature"], "top_p": sample["top_p"],
                  "top_k": sample["top_k"], "repeat_penalty": sample["repeat_penalty"],
                  "seed": sample["seed"],
                  # HONESTY: top_p/top_k are the requested settings, not a guess -- but this engine build
                  # does not enforce them (see _resolve_sampling's docstring); only temperature/
                  # repeat_penalty/seed actually shaped this reply's sampler.
                  "note": "top_p/top_k are requested settings, NOT enforced by this engine build "
                          "(temperature + repeat_penalty softmax only; see engine/core/src/sample.cpp)"}
    else:
        top = {"sampler_mode": "greedy", "sampling": "greedy", "temperature": 0.0,
               "repetition_penalty": 1.0, "seed": 0}
        # Self-describing decode block (REPRODUCE_AND_PROVE_PLAN S2): the honest regime this run was
        # produced under, so re-derivation/forced-scoring is exact-by-construction. Engine chat is greedy
        # (temperature 0), seed 0 -- the actual values passed, not a guess.
        decode = {"mode": "greedy", "temperature": 0.0, "seed": 0}
    return _without_unknowns({
        **top,
        "decode": decode,
        "max_tokens": int(max_new) if max_new is not None else None,
        "stream": bool(stream) if stream is not None else None,
    })


def _pers(name):
    return os.path.join(CLOZN_DIR, name)


# The honest J-lens provenance for the Run Inspector panel (J3), sourced from ~/.clozn/jlens/manifest.json
# (the sidecar the engine loaded). The `note` is verbatim from JLENS_ENGINE_PLAN.md -- a disposition, not a
# verified thought; a linear lens always emits something. Read once (cached); missing/broken manifest still
# yields the honest note (fit_model/layers just stay unfilled). Never raises.
_JLENS_NOTE = ("fitted linear Jacobian lens, transferred to this GGUF; a per-token 'disposed to say' read, "
               "NOT the model's literal thought — a linear lens always emits something.")


def _jlens_provenance():
    cached = getattr(_jlens_provenance, "_cached", None)
    if cached is not None:
        return dict(cached)
    prov = {"kind": "jacobian_lens", "fit_model": None, "layers": [], "note": _JLENS_NOTE}
    try:
        with open(os.path.join(_pers("jlens"), "manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        prov["layers"] = [int(x) for x in (m.get("layers") or [])]
        model = str(m.get("model", "")).split("/")[-1].replace("-Instruct", "")
        fo = m.get("fitted_on") or {}
        bits = [b for b in ["HF", fo.get("quant"),
                            (f"{fo.get('n_prompts')} prompts" if fo.get("n_prompts") else None)] if b]
        if model:
            prov["fit_model"] = f"{model} ({', '.join(bits)})" if bits else model
    except Exception:
        pass
    _jlens_provenance._cached = dict(prov)
    return dict(prov)


def _jlens_run_text(run):
    """The text a run's J-lens should read + a source label the panel shows so it's honest about WHAT was
    read. Prefer the stored `response` (the reply whose tokens the chips annotate); else the last user
    message (a prompt-only run). ('' , 'none') when there's nothing to read."""
    if not isinstance(run, dict):
        return "", "none"
    resp = run.get("response")
    if isinstance(resp, str) and resp.strip():
        return resp, "response"
    msgs = run.get("messages") if isinstance(run.get("messages"), list) else []
    for m in reversed(msgs):
        if isinstance(m, dict) and m.get("role") == "user" and str(m.get("content", "")).strip():
            return str(m["content"]), "last_user_message"
    return "", "none"


def _jlens_workspace_readouts(res, run_id):
    """Map a J-lens readout into the protocol's workspace_readout shape (SPEC.md / WORKSPACE_LENS.md),
    provider_type 'jacobian_lens', readout_kind 'token' -- one event per position, so the same readout can
    ride the existing event spine. Additive/opt-in (request `protocol:true`); the panel uses tokens+readouts."""
    layer = res.get("layer")
    toks = res.get("tokens") or []
    rows = res.get("readouts") or []
    out = []
    for i, row in enumerate(rows):
        out.append({
            "type": "workspace_readout", "run_id": run_id,
            "token_index": i, "token_text": (toks[i] if i < len(toks) else None),
            "layer": layer, "position": i,
            "provider": f"jacobian_lens_l{layer}", "provider_type": "jacobian_lens",
            "readout_kind": "token",
            "top_readouts": [{"label": r.get("piece"), "score": r.get("score")} for r in (row or [])],
        })
    return out


def _jlens_envelope(res, run_id, text_source, want_protocol=False):
    """Shape EngineSubstrate.jlens's normalized dict into the J3 frontend contract. `res` is either
    {available:False, reason} or {available:True, layer, available_layers, n_tokens, tokens, readouts}
    (+ an `error` string when the layer was unknown). Carries the honest provenance block when available.
    HTTP is always 200 -- absence is a clean, renderable state, never an error."""
    if not res.get("available"):
        return {"available": False, "run_id": run_id, "reason": res.get("reason", "J-lens unavailable")}
    out = {"available": True, "run_id": run_id,
           "layer": res.get("layer"), "available_layers": res.get("available_layers", []),
           "text_source": text_source, "n_tokens": int(res.get("n_tokens", 0) or 0),
           "tokens": res.get("tokens", []), "readouts": res.get("readouts", []),
           "provenance": _jlens_provenance()}
    if res.get("error"):                       # unknown layer etc. -- surfaced cleanly (layers already listed)
        out["error"] = res["error"]
    if want_protocol:                          # bonus: also ride the workspace_readout protocol shape
        out["workspace_readouts"] = _jlens_workspace_readouts(res, run_id)
    return out


ENGINE_STEER = None        # lazy EngineSteer on the Qwen GGUF engine -- tone dials on the C++ runtime, any GGUF


def _engine_steer():
    global ENGINE_STEER
    if ENGINE_STEER is None and ENGINE_QWEN is not None:
        from clozn.behavior.steering.engine_adapter import EngineSteer
        ENGINE_STEER = EngineSteer(ENGINE_QWEN)
    return ENGINE_STEER


ENGINE_CONCEPT_STEER = None   # lazy ConceptSteer(ENGINE_QWEN) -- Tier-1 #1's any-concept dial (dir(c)), see
                              # clozn/behavior/steering/concept_dir.py. Mirrors _engine_steer() above; a
                              # SEPARATE mechanism (dir(c), zero calibration) from the tone-dial EngineSteer.


def _engine_concept_steer():
    global ENGINE_CONCEPT_STEER
    if ENGINE_CONCEPT_STEER is None and ENGINE_QWEN is not None:
        from clozn.behavior.steering.concept_dir import ConceptSteer
        ENGINE_CONCEPT_STEER = ConceptSteer(ENGINE_QWEN)
    return ENGINE_CONCEPT_STEER


def _qwen_tmpl(messages):
    """Render chat messages into Qwen's chat-template STRING (ChatML). LEGACY: kept only as a documented
    reference / last-ditch fallback -- the engine generation paths now template PER-MODEL via
    _engine_tmpl (the GGUF's own embedded chat template), so a non-Qwen model gets its correct format.
    The torch QwenSubstrate never used this (it applies the HF tokenizer's template internally)."""
    sysmsg = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    for m in messages:
        if m.get("role") == "system" and m.get("content"):
            sysmsg = m["content"]
    s = f"<|im_start|>system\n{sysmsg}<|im_end|>\n"
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            s += f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n"
    return s + "<|im_start|>assistant\n"


def _engine_tmpl(engine, messages):
    """Render chat `messages` to a prompt string using the ENGINE-LOADED MODEL'S OWN chat template
    (POST /apply_template -> llama_chat_apply_template over the GGUF's tokenizer.chat_template). THIS is
    what makes the engine paths model-agnostic: whatever GGUF the engine has loaded, its messages are
    formatted in that model's native template (Qwen ChatML, Llama-3 headers, Gemma turns, ...), instead
    of a hardcoded Qwen string. Replaces _qwen_tmpl on every EngineSubstrate generation path.

    Errors propagate deliberately (no silent Qwen fallback): a model with no embedded template, or an
    engine too old to expose /apply_template, must surface -- silently mis-formatting the prompt is the
    exact bug this removes. Callers that need a soft degrade catch EngineError themselves."""
    return engine.apply_template(messages)


def _disk_memory():
    """The trained memory prefix + strength, read from disk -- so engine-chat needs NO HF model resident.
    The prefix is just saved vectors; only TRAINING a new one needs PyTorch's gradients."""
    import torch
    path = _pers("studio_memory.pt")
    if not os.path.isfile(path):
        return None, 1.0
    try:
        d = torch.load(path, map_location="cpu")
        pre = d.get("prefix")
        return (pre.float() if pre is not None else None), float(d.get("memory_strength", 1.0))
    except Exception:
        return None, 1.0


def _disk_dials():
    """The saved tone-dial values (personality.json IS the strength dict) -- no HF model needed."""
    path = _pers("studio_personality.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return {k: float(v) for k, v in json.load(f).items()}
    except Exception:
        return {}


def _dial_calibration():
    """The curated per-model dial calibration (research/dial_autocalibrate.py's sweep -- see that module's
    docstring), read from ~/.clozn -- NEVER research/runs/dial_autocalibrate.json directly (that raw
    research file carries full curves + sample_replies per dial, meant for a human to eyeball, not to be
    re-parsed on every /steer/axes call; a curator step distills it down to just what a slider needs and
    persists THAT here). Missing/broken file -> {} (never raise): calibration is optional enrichment, so
    every caller must behave EXACTLY as it did before this existed when the file isn't there yet.

    Returns {dial_name: {"usable_max", "usable_range", "derail_point", "works"}, ...}. Tolerant of either a
    flat {dial_name: {...}} file, or one shaped like the raw research JSON ({"dials": {dial_name: {...}}},
    with "range_valid" instead of "works") -- whichever shape the curated file ends up in, this keeps
    working. A per-entry parse problem drops just that one dial (skipped, not crashed on)."""
    path = _pers("dial_calibration.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        table = raw.get("dials", raw) if isinstance(raw, dict) else {}
        out = {}
        for name, c in table.items():
            if not isinstance(c, dict):
                continue
            works = c.get("works", c.get("range_valid"))
            out[name] = {
                "usable_max": c.get("usable_max"),
                "usable_range": c.get("usable_range"),
                "derail_point": c.get("derail_point"),
                "works": bool(works),
            }
        return out
    except Exception:
        return {}


def _with_calibration(axis, c):
    """Merge one _dial_calibration() entry into one /steer/axes axis dict, IN PLACE (also returned, so this
    reads as an expression in a list comprehension) -- the one spot that decides what a calibrated slider
    looks like to the studio UI. No entry for this dial (c falsy/missing) -> the axis is untouched except
    for "calibrated": False added: NO behavior change for a dial nobody has calibrated on this model (Law
    #6 -- the uncalibrated case must render exactly as it always has). An entry -> "max" becomes the
    CALIBRATED usable_max, falling back to the axis's own already-declared max when usable_max itself is
    None (a dial swept but never found usable), plus "usable_range"/"derail_point"/"works" for the UI to
    grey out a dead dial or show its working range."""
    if not c:
        axis["calibrated"] = False
        return axis
    axis["max"] = c["usable_max"] if c.get("usable_max") is not None else axis["max"]
    axis["usable_range"] = c.get("usable_range")
    axis["derail_point"] = c.get("derail_point")
    axis["works"] = bool(c.get("works"))
    axis["calibrated"] = True
    return axis


def _library_dial_names() -> set:
    """The set of custom-dial names that are SHIPPED-LIBRARY dials (research/deploy_dial_library.py's
    one-time registration), read from ~/.clozn/studio_library.json's keys -- a file distinct from the
    user's own studio_custom_<name>.json, so a library dial can never be mistaken for something the user
    made. Missing/broken file -> empty set (never raise): before the library is deployed (or on any
    substrate that has never loaded studio_library.json), /steer/axes must behave EXACTLY as it always
    has -- every steer.custom entry tags "custom": True and none tag "library" -- the same Law-#6-style
    backward compat _dial_calibration() already gives the calibration merge."""
    path = _pers("studio_library.json")
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return set(data) if isinstance(data, dict) else set()
    except Exception:
        return set()


# ------- memory cards <-> the working prefix (D2 + E1) --------------------------------------------
# The cards (research/memory_cards.py) are the metadata + review layer; the trained soft-prefix is
# UNCHANGED. The contract that keeps the prefix safe: m.rules is ALWAYS the texts of the ACTIVE cards,
# and the prefix is built from m.rules via m.consolidate(rules) exactly as before. So a card's STATUS
# decides what's in m.rules, which drives the prefix. We only ever retrain when the active set actually
# changes (a no-op transition -- e.g. approving a card whose text is already active -- never touches it).

_SUSPICIOUS = ("ignore ", "disregard ", "system prompt", "you are now", "forget ", "override",
               "jailbreak", "developer mode", "instead of", "from now on you", "pretend ")


def _risk_of(text: str) -> str:
    """Cheap heuristic: flag instruction-like / prompt-injection-ish memory text as 'suspicious' so the
    reviewer sees it. A memory is meant to be a fact/preference ABOUT the user, not a command to the model."""
    t = (text or "").lower()
    return "suspicious" if any(s in t for s in _SUSPICIOUS) else "low"


def _dial_suggestion(text: str):
    """If a memory's text is really a STYLE preference that maps to a tone dial, return that suggestion
    ({axis, value, pole_label}); else None. Guarded import of steering.suggest_dial_for_preference so a
    missing/broken steering module (or the pure-engine substrate) can never break /memory/add or propose.
    Pure + deterministic (a lexicon match, no model) -- see steering.suggest_dial_for_preference."""
    try:
        from clozn.behavior.steering.catalog import suggest_dial_for_preference
        return suggest_dial_for_preference(text)
    except Exception:
        return None


QUOTE_SPAN_MAX = 240   # a "you said this" quote is for recognizing your own words, not re-reading the essay


def _provenance_of(messages):
    """The (source_turn, quoted_span) pair for a card proposed from `messages` (the
    OBEY defense -- see memory_cards.has_provenance). source_turn is the index of the LAST user message in
    the list (mirrors _last_user's "most recent user turn" convention, and matches dream_consolidation.py's
    `"turn": i` = index into a run's messages); quoted_span is that message's own verbatim text, truncated
    to QUOTE_SPAN_MAX chars -- never paraphrased, never the model's synthesized third-person card text.
    (None, "") when there's no user message to cite at all (defensive: propose_memory needs user content to
    work from, so this should be rare) -- that is exactly the "claimed a run but can't back it up" case the
    approve-gate refuses on, and the Memory page flags."""
    for i in range(len(messages or []) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user":
            content = str(m.get("content") or "").strip()
            if content:
                span = content if len(content) <= QUOTE_SPAN_MAX else content[:QUOTE_SPAN_MAX].rstrip() + "…"
                return i, span
    return None, ""


# ------- memory MODE: prompt-carried cards vs the internalized prefix (MEMORY_MODE_SWAP_SPEC) ------
# mode "prompt" (the fresh-install default): the ACTIVE card texts are compiled into ONE system block
# (memory_mode.compile_prompt_block -- verbatim the sys_rule wording the prefix trains toward) and
# prepended to the chat, topic-gated PER TURN; generation runs with use_prefix=False. Card mutations
# skip consolidate()/_TRAIN_LOCK entirely (instant). mode "internalized": today's prefix path, exactly
# as before. An existing trained prefix keeps "internalized" until the user toggles (memory_mode.py).

PROMPT_GATE_MIN = 0.05     # gate below this -> the block is OMITTED for the turn. Prompt mode controls
                           # over-bleed by omission (binary), not by the prefix's continuous scaling.


def _memory_mode():
    """The active memory mode ("prompt" | "internalized"). Fail-safe: any hiccup reading the setting
    resolves to "internalized" -- the long-standing prefix behavior -- so a broken settings file can
    never silently swap the mechanism under a live personality."""
    try:
        import clozn.memory.mode as memory_mode
        return memory_mode.get_mode()
    except Exception:
        return "internalized"


def _last_user(messages):
    """The last user turn's content ('' if none) -- the topic-gate input, same as the prefix path."""
    return next((m.get("content", "") for m in reversed(messages or []) if m.get("role") == "user"), "")


def _prompt_gate(last_user, texts):
    """Topic-relevance gate for the prompt-mode block -- the SAME signal the prefix path scales by
    (topic_gate.scalar over the active texts). 1.0 (no gating) when the embedder is unavailable."""
    try:
        from clozn.memory.topic_gate import get_gate
        return float(get_gate().scalar(last_user, list(texts)))
    except Exception:
        return 1.0


def _prompt_relevance(last_user, texts):
    """Per-card topic cosine {text: relevance} for the applied block -- the SAME embeddings _prompt_gate
    just used (cached by string in topic_gate), so it's ~free. {} when the embedder is unavailable. This
    is the per-card signal the run record needs so the inspector can show WHY each card fired, not just
    that the block as a whole did (the scalar gate)."""
    try:
        from clozn.memory.topic_gate import get_gate
        return dict(get_gate().relevance(last_user, list(texts)))
    except Exception:
        return {}


def _prompt_mem_cards(mem, exclude_ids=()):
    """The ACTIVE cards ({id, text}) that feed the prompt block, minus exclude_ids (replay's REAL
    per-card ablation). Reads the card store directly (memory_mode.active_cards) -- in prompt mode the
    cards ARE the memory (m.rules is bookkeeping that can lag right after boot). Falls back to
    mem.rules (id-less) only if the store module is unavailable, so a broken store degrades to the old
    rule list rather than to amnesia."""
    import clozn.memory.mode as memory_mode
    cards = memory_mode.active_cards(exclude_ids)
    if cards is not None:
        return cards
    return [{"id": None, "text": t} for t in (getattr(mem, "rules", []) or []) if t]


def _prompt_block_for(mem, last_user, strength=None):
    """Prompt-mode injection decision for THIS turn -> (block_text | None, applied_cards, gate).

    None == omit the block entirely: no active cards, strength == 0 (the dial maps to on/off in prompt
    mode -- 0 never injects, >0 injects when gated in), or the topic gate is ~0 (off-topic turn).
    applied_cards is [] whenever the block is omitted. Honors mem._exclude_card_ids (set temporarily
    by replay.py for per-card receipts). `strength` overrides mem.memory_strength (the pure-engine
    path reads it from disk); `mem` may be None on that path -- every read of it is defensive."""
    cards = _prompt_mem_cards(mem, getattr(mem, "_exclude_card_ids", None) or ())
    texts = [c["text"] for c in cards]
    s = float(strength if strength is not None else getattr(mem, "memory_strength", 1.0))
    if not texts or s <= 0.0:
        return None, [], 0.0
    g = _prompt_gate(last_user, texts)
    if g < PROMPT_GATE_MIN:
        return None, [], g
    rel = _prompt_relevance(last_user, texts)          # {text: cosine} per card (best-effort; {} if no embedder)
    applied = [dict(c, relevance=rel.get(c["text"])) for c in cards]
    import clozn.memory.mode as memory_mode
    return memory_mode.compile_prompt_block(texts), applied, g


def _anchored_gates(last_user, bags):
    """Per anchored bag topic gate, fail-open like prompt memory's topic gate."""
    try:
        from clozn.memory import topic_gate
        gate = topic_gate.get_gate()
        out = {}
        for bag in bags or []:
            if not isinstance(bag, dict):
                continue
            cid = bag.get("card_id")
            text = str(bag.get("card_text") or "").strip()
            if cid and text:
                out[cid] = float(gate.scalar(last_user or "", [text]))
        return out
    except Exception:
        return None


def _apply_anchored_memory(kw: dict, mem_out: dict | None, last_user: str | None) -> dict | None:
    """Add X7/J-anchored memory to a live engine request when the raw steer slot is free. Returns the
    compile_steer() payload that was ACTUALLY injected into `kw` (steer_vec/coef/layer/s_total/vector/
    bags), or None when nothing was injected -- no active bags, nothing composed, or the raw-steer slot
    was already held by tone dials (mem_out["anchored_skipped"]). The loop guard (chat()/chat_stream()
    below) uses this return to retry at half strength without recomposing from the store, and to know
    whether the guard applies at all THIS turn (only when anchored memory actually rode this turn)."""
    try:
        from clozn.memory import anchored
        bags = anchored.active_bags()
        if not bags:
            return None
        comp = anchored.compile_steer(bags, gates=_anchored_gates(last_user, bags))
        if not comp:
            return None
        if kw.get("steer_vec"):
            if mem_out is not None:
                mem_out["anchored_skipped"] = "tone dials held the raw-steer channel this turn"
            return None
        kw["steer_vec"] = comp["steer_vec"]
        kw["steer"] = {"coef": 1.0, "layer": comp["layer"]}
        if mem_out is not None:
            mem_out["anchored"] = comp["bags"]
            mem_out["anchored_layer"] = comp["layer"]
            mem_out["anchored_s_total"] = comp.get("s_total")
        return comp
    except Exception:
        return None


def _anchored_loop_guard(engine, prompt, max_new, kw, samp, comp, reply, steps, finish, mem_out):
    """The substrate wiring anchored.detect_loop()'s own docstring deliberately leaves undone
    -- chat()'s non-streaming path only: detect_loop() over the pieces
    JUST generated under a FULL-STRENGTH anchored injection (`comp`, the compile_steer() payload that
    actually rode this turn -- callers only invoke this when comp is not None, i.e. anchored memory was
    really injected, never on a skipped/absent one). A fired loop is OVER-INJECTION DEGENERACY, not a
    quality signal either way -- this only MITIGATES it; it never claims the memory "worked" or "was
    recalled" (clozn's honesty contract).

      1. clean (no loop): returns (reply, steps, finish) UNTOUCHED -- byte-identical to today, mem_out
         gets no anchored_loop_guard key at all.
      2. loop -> retry ONCE at s_total/2 (anchored.halve_steer -- same direction/layer/bags, half the
         injected magnitude). Clean on retry: use the retry's (reply, steps, finish);
         mem_out["anchored_loop_guard"] = {"fired": True, "action": "retried@s/2", "resolved": True},
         and mem_out["anchored_s_total"] is corrected to the HALVED value that actually shaped the final
         reply (the run record must describe what really happened, not the original full-strength ask).
      3. still loops at half strength -> one final pass with the anchored steer ZEROED entirely (kw's
         steer_vec/steer keys dropped -- the raw-steer slot was free before anchored memory claimed it,
         so this is a genuinely unsteered generation, not "fall back to tone dials"). Whether THAT pass
         is itself loop-free is checked too (never claim "resolved" without looking);
         mem_out["anchored_loop_guard"] = {"fired": True, "action": "disabled", "resolved": <checked>},
         mem_out["anchored_s_total"] = 0.0.

    Every regeneration reuses the SAME prompt/max_new/sample regime as the original call -- only the
    steer changes -- so a retry is a fair A/B against the original, not a different generation policy."""
    from clozn.memory import anchored
    pieces = [str(s.get("piece", "")) for s in (steps or [])]
    if not anchored.detect_loop(pieces):
        return reply, steps, finish

    half = anchored.halve_steer(comp)
    kw_half = dict(kw)
    kw_half["steer_vec"] = half["steer_vec"]
    kw_half["steer"] = {"coef": 1.0, "layer": half["layer"]}
    reply2, steps2, finish2, _ = _engine_complete_traced(engine, prompt, max_new, kw_half, sample=samp)
    pieces2 = [str(s.get("piece", "")) for s in (steps2 or [])]
    if not anchored.detect_loop(pieces2):
        if mem_out is not None:
            mem_out["anchored_loop_guard"] = {"fired": True, "action": "retried@s/2", "resolved": True}
            mem_out["anchored_s_total"] = half["s_total"]
        return reply2, steps2, finish2

    kw_zero = {k: v for k, v in kw.items() if k not in ("steer_vec", "steer")}
    reply3, steps3, finish3, _ = _engine_complete_traced(engine, prompt, max_new, kw_zero, sample=samp)
    pieces3 = [str(s.get("piece", "")) for s in (steps3 or [])]
    if mem_out is not None:
        mem_out["anchored_loop_guard"] = {"fired": True, "action": "disabled",
                                          "resolved": not anchored.detect_loop(pieces3)}
        mem_out["anchored_s_total"] = 0.0
    return reply3, steps3, finish3


def _inject_block(messages, block):
    """`messages` with the memory block folded in as system context (a copy -- never mutates the
    caller's list). Appends to an existing system message (the client's own instructions keep first
    position) or prepends a new one; a None/empty block returns the messages unchanged."""
    if not block:
        return list(messages)
    msgs = [dict(m) for m in messages]
    for m in msgs:
        if m.get("role") == "system":
            m["content"] = (str(m.get("content") or "") + "\n\n" + block).strip()
            return msgs
    return [{"role": "system", "content": block}] + msgs


def _mem_migrate(m):
    """Seed the card store from a memory object's legacy rule-strings, ONCE. migrate_from_rules is a
    no-op when the store already has cards, and it creates them as ACTIVE -- the prefix is already trained
    on these exact rules, so we do NOT re-consolidate here. Returns the cards created (or [])."""
    import clozn.memory.cards as memory_cards
    try:
        return memory_cards.migrate_from_rules(list(getattr(m, "rules", []) or []))
    except Exception:
        return []


def _export_markdown(run: dict, xr: dict | None) -> str:
    """Render a run (+ its M1 explain) as a human-readable Markdown receipt: the conversation, what memory
    and which dials shaped it (with per-card relevance), why it stopped, and where it hesitated. Pure / no
    model -- the JSON export carries the full structured bundle; this is its readable companion."""
    import clozn.receipts.bundle as receipt_bundle
    return receipt_bundle.to_markdown(receipt_bundle.build(run, explain=xr))


def _runs_for_card(card_id):
    """Best-effort: the run summaries whose memory.cards_applied names this card (by id OR by text).
    cards_applied currently records the active rule TEXTS (see _log_run), so we match on text primarily
    and on id as a forward-compatible fallback. Returns [] when the card / runs are gone (never raises)."""
    import clozn.memory.cards as memory_cards
    import clozn.runs.store as runlog
    try:
        card = memory_cards.get(card_id)
        text = (card or {}).get("text", "")
        needles = {n for n in (card_id, text) if n}
        if not needles:
            return []
        out = []
        for r in runlog.list_runs(500):
            applied = ((r.get("memory") or {}).get("cards_applied")) or []
            applied = [str(a) for a in applied]
            if needles & set(applied):
                out.append(r)
        return out
    except Exception:
        return []


def _mem_sync_rules(m, reconsolidate=True, force=False):
    """Make m.rules == the active-card texts, then rebuild the prefix ONLY if the active set changed.

    This is the one place the prefix can move. If the active texts are identical to what m.rules already
    holds, we leave the prefix completely untouched (the expensive, working artifact is preserved). When
    the set changed and reconsolidate is on, we retrain from the active texts (SLOW -- expected on
    approve/reject/disable/edit). If the active set became EMPTY (e.g. the last card was disabled), we
    reset() so the now-unused prefix stops biting -- reset() is zero-arg on both memory backends.
    `force` retrains even when m.rules already matches the store -- used when toggling BACK to
    internalized mode, where the rules are synced but the PREFIX may be stale (prompt-mode card edits
    never consolidate)."""
    import clozn.memory.cards as memory_cards
    new_rules = memory_cards.active_texts()
    changed = list(getattr(m, "rules", []) or []) != list(new_rules)
    m.rules = list(new_rules)
    result = None
    if (changed or force) and reconsolidate:
        if new_rules:
            result = m.consolidate(list(new_rules))
        else:                                    # nothing active anymore -> drop the prefix entirely
            try:
                result = m.reset()
            except Exception:
                pass
            m.rules = []                          # reset() may clear rules; keep them in sync
    return {"changed": changed, "rules": list(new_rules), "consolidate": result}


# ------- profiles: named persona bundles -> cards + dials on the LIVE substrate ---------------------
# profiles.py is the model-free CRUD + compile layer (source bundles: card texts, dial settings, custom-
# dial recipes, fact pairs -- see its docstring). This is the thin wiring that hands it the live objects
# a switch needs (SUB._mem for cards/rules, SUB.steer for dials) and reports what actually happened.

def _active_profile_name():
    """The name of the last-switched-to profile, or None (nothing switched yet this install). Persisted
    in studio_settings.json alongside memory_mode -- one small settings file, not a new one."""
    import clozn.memory.mode as memory_mode
    return memory_mode.get_setting("active_profile")


def _profiles_switch(sub, p) -> dict:
    """Apply profile bundle `p` to the live substrate `sub`: cards REPLACE the studio's active set (a
    profile switch is a replacement, never a merge -- disjoint personas must not bleed into each other),
    dials replace via profiles.apply_dials (steer.clear() then set()), and the prompt-mode/internalized
    resync goes through the SAME _start_retrain machinery every other card mutation uses: instant in
    prompt mode (the cards ARE the memory there), a backgrounded consolidate() in internalized mode.

    Facts (profiles.compile_facts) are the item-5 seam: no live slot-memory store is wired into the
    server yet (slotmem_qwen.py is a standalone research module), so a profile's facts are saved in the
    bundle but NOT compiled anywhere yet -- reported honestly via `facts_note`, never silently dropped.
    Returns {name, prompt_block, cards:{removed,added}, dials, resync, facts_note}."""
    import clozn.memory.cards as memory_cards

    # 1) CARDS: delete the current active set, then create the profile's cards fresh as active. Deleting
    #    (not just disabling) is the isolation contract: a stale disabled card from persona A must never
    #    reappear if the user later hand-edits persona B's set, and disjoint personas keep disjoint cards.
    removed = 0
    for c in memory_cards.list_cards():
        if memory_cards.delete(c["id"]):
            removed += 1
    added = 0
    for c in p.get("cards", []):
        if c.get("status", "active") != "active":     # a disabled card in the bundle stays inert here too
            continue
        if memory_cards.create(c["text"], status="active", kind="preference",
                               evidence=f"profile:{p['name']}") is not None:
            added += 1

    # 2) SYNC the memory mechanism from the new active set. force=True: the pre-check inside
    #    _start_retrain compares m.rules to memory_cards.active_texts(), and since we just rewrote the
    #    store out from under it, that comparison alone isn't trustworthy for a switch -- force skips it
    #    and always resyncs, exactly as the mode-switch catch-up (POST /memory/mode) already does.
    resync = {"retraining": False}
    m = getattr(sub, "_mem", None)
    if m is not None:
        resync = _start_retrain(m, "profile-switch", None, force=True)

    # 3) DIALS: replace via profiles.apply_dials (clear() then set(); custom-dial recipes recompute if
    #    not already present) -- persist exactly like /steer/set and /steer/custom already do, so the
    #    switched-to persona survives a restart the same way a manually-set dial would.
    dials = {"applied": {}, "customs_added": []}
    steer = getattr(sub, "steer", None)
    if steer is not None:
        from clozn.profiles import store as profiles
        dials = profiles.apply_dials(p, steer)
        try:
            if hasattr(steer, "save_state"):
                steer.save_state(_pers("studio_personality.json"))
            if dials["customs_added"] and hasattr(steer, "save_custom"):
                steer.save_custom(_pers(f"studio_custom_{getattr(sub, 'name', SUBNAME)}.json"))
        except Exception:
            pass

    # 4) FACTS: the item-5 seam, now CLOSED. The bundle's fact pairs recompile into THIS profile's slot
    #    store (profiles.compile_facts on the live SlotMem, sharing SUB.memory's Qwen-7B) and persist to
    #    ~/.clozn/profiles/<name>.slots.pt -- but ONLY when the facts tier is enabled (memory_facts on).
    #    Off (the default): facts still travel in the bundle, we just don't build the store or pay the
    #    model cost, and facts_note says so. active_profile is set FIRST so SlotBox loads the right store.
    import clozn.memory.mode as memory_mode
    memory_mode.set_setting("active_profile", p["name"])

    facts_note = None
    if p.get("facts"):
        import clozn.memory.facts_mode as facts_mode
        if not facts_mode.enabled():
            facts_note = (f"{len(p['facts'])} fact(s) travel in the bundle but the facts tier is off -- "
                          "enable it on the Memory page (Facts) to compile them into this profile's store.")
        else:
            box = _slots_box()
            compiled = None
            if box is not None:
                try:
                    box.on_profile_switch()                # point the resident store at THIS profile
                    slots = box._build()                   # share SUB.memory's model; build if needed
                    if slots is not None:
                        with _TRAIN_LOCK:                  # writes run forwards on the shared model
                            from clozn.profiles import store as _pf
                            slots.entries = []             # persona isolation: this profile's facts only
                            compiled = _pf.compile_facts(p, slots, gate=False)  # curated -> store them all
                        box._save_active()
                except Exception as e:
                    facts_note = f"facts not compiled: {type(e).__name__}: {e}"
            if compiled is not None:
                facts_note = (f"{compiled['written']} fact(s) compiled into this profile's slot store"
                              + (f" ({compiled['skipped']} skipped)" if compiled.get("skipped") else "") + ".")
            elif facts_note is None:
                facts_note = f"{len(p['facts'])} fact(s) in the bundle -- slot store unavailable (no model loaded)."
    return {"name": p["name"], "prompt_block": prompt_block_preview(p),
            "cards": {"removed": removed, "added": added}, "dials": dials,
            "resync": resync, "facts_note": facts_note}


def prompt_block_preview(p) -> str:
    """The system block this profile WOULD inject (profiles.prompt_block) -- for the switch response's
    receipt only; the live chat path still compiles fresh from the card store every gated-in turn."""
    from clozn.profiles import store as profiles
    return profiles.prompt_block(p)


# ------- the FACTS tier: slot-memory store wired to the studio ----------------------------------------
# slotmem_qwen.SlotMem is the explicit, editable, honest-about-ignorance fact store (centered-key
# addressing, surprise-gated writes, confidence-gate abstention -- proven 0.95 flat to N=200). SlotBox
# is the thin studio wiring: it lazily builds ONE SlotMem SHARING the substrate's Qwen-7B (SUB.memory
# .model -- no second model, per the item spec), keeps a PER-PROFILE store (~/.clozn/profiles/<name>
# .slots.pt), and gates every operation behind memory_facts (default OFF -- the latency rule: a slot
# read is an extra forward, kept off the 7B hot path until measured; when on, we log slot_ms honestly).
#
# v1 CONTRACT (deliberately conservative -- protect the shipped chat path): a slot READ produces a
# RECEIPT (hit / gate value / abstention / the answer the store would inject) that the Facts panel
# shows and the runlog records; it does NOT alter the chat reply, so the 7B generation stays
# byte-identical whether facts are on or off. Actually STEERING the reply with the injected value is
# the next rung (documented seam). Auto-WRITE does mutate the store: it runs the surprise gate on a
# candidate (cue -> answer) mined from the turn, so the gate visibly refuses what the model already
# knows (the Titans write policy, the load-bearing provable part).

class SlotBox:
    """Owns the studio's live SlotMem + its per-profile persistence. Built lazily on first use so a
    fresh install with facts OFF never pays for it. Every public method is a no-op / empty receipt when
    `memory_facts` is off or no shareable model is loaded -- the caller stays oblivious to both."""

    def __init__(self, mem_provider):
        # mem_provider() -> the substrate memory object (SUB.memory) whose .model/.tok we SHARE, or None.
        self._mem_provider = mem_provider
        self._slots = None                 # the SlotMem (None until built)
        self._profile = None               # the profile name whose store is currently resident
        self._lock = threading.Lock()      # serialize build + store mutations (the model is shared)

    # ---- lifecycle --------------------------------------------------------------------------------
    def _shared_model(self):
        try:
            m = self._mem_provider()
        except Exception:
            m = None
        model = getattr(m, "model", None)
        tok = getattr(m, "tok", None)
        return (model, tok) if (model is not None and tok is not None) else (None, None)

    def _build(self):
        """Build the SlotMem on the shared backbone + load the active profile's store. Returns the
        SlotMem or None (no model yet, or slotmem import/HF unavailable). Holds _lock."""
        if self._slots is not None:
            return self._slots
        model, tok = self._shared_model()
        if model is None:
            return None
        try:
            import clozn.memory.facts_mode as facts_mode
            import clozn.memory.slotmem_qwen.store as slotmem_qwen
            self._slots = slotmem_qwen.SlotMem.from_shared(model, tok, facts_mode.LAYER)
        except Exception as e:
            print(f"[facts] could not build slot store: {type(e).__name__}: {e}", flush=True)
            self._slots = None
            return None
        self._load_active()               # bring the current profile's saved facts in
        return self._slots

    def _active_profile(self):
        try:
            return _active_profile_name()
        except Exception:
            return None

    def _load_active(self):
        """(Re)load the store for the currently-active profile into self._slots. Silent on a missing
        file (a profile with no facts yet is empty, not an error); a layer mismatch is logged + skipped."""
        if self._slots is None:
            return
        import clozn.memory.facts_mode as facts_mode
        prof = self._active_profile()
        path = facts_mode.store_path(prof)
        self._slots.entries = []
        self._profile = prof
        if os.path.isfile(path):
            try:
                self._slots.load(path)
            except Exception as e:
                print(f"[facts] skipped loading {path}: {type(e).__name__}: {e}", flush=True)

    def _save_active(self):
        if self._slots is None:
            return
        import clozn.memory.facts_mode as facts_mode
        try:
            self._slots.save(facts_mode.store_path(self._profile))
        except Exception as e:
            print(f"[facts] save failed: {type(e).__name__}: {e}", flush=True)

    def _ensure_profile(self):
        """If the active profile changed since we last loaded, swap the resident store to it (per-profile
        isolation: one persona's facts must never read another's). Cheap string compare; loads only on a
        real change."""
        if self._slots is None:
            return
        if self._active_profile() != self._profile:
            self._load_active()

    def on_profile_switch(self):
        """Called by a profile switch: reload the new profile's store if the box is already live. When
        facts are off / not built yet, nothing to do (the store loads lazily on first use)."""
        with self._lock:
            if self._slots is not None:
                self._load_active()

    # ---- reads / writes (all gated by memory_facts) ----------------------------------------------
    def status(self):
        """{enabled, layer, profile, count} -- the Facts panel header. Never builds the model just to
        answer (count is 0 until the store is actually resident)."""
        import clozn.memory.facts_mode as facts_mode
        n = len(self._slots.entries) if self._slots is not None else 0
        return {"enabled": facts_mode.enabled(), "layer": facts_mode.LAYER,
                "profile": self._profile or self._active_profile() or "default", "count": n}

    def list_entries(self):
        """[{cue, answer, label}] for the resident store, [] when off / unbuilt. Read-only, no model
        forward -- safe to call on every Facts-panel load."""
        import clozn.memory.facts_mode as facts_mode
        if not facts_mode.enabled():
            return []
        with self._lock:
            if self._build() is None:
                return []
            self._ensure_profile()
            return [{"cue": e["cue"], "answer": e["answer"], "label": e["label"]}
                    for e in self._slots.entries]

    def add(self, cue: str, answer: str, gate: bool = True):
        """Explicit fact write with the SURPRISE GATE on (the refusal is the receipt: a fact the model
        already knows is SKIPPED, not stored). Persists on a real write. {ok, written, surprise, reason?}."""
        import clozn.memory.facts_mode as facts_mode
        cue, answer = str(cue or "").strip(), str(answer or "")
        if not cue or not answer.strip():
            return {"ok": False, "reason": "need a cue and an answer"}
        if not facts_mode.enabled():
            return {"ok": False, "reason": "the facts tier is off (enable it first)"}
        with self._lock:
            if self._build() is None:
                return {"ok": False, "reason": "no model loaded to hold the fact store"}
            self._ensure_profile()
            with _TRAIN_LOCK:              # the store write runs forwards on the shared model
                r = self._slots.write(cue, answer, gate=gate)
                if r.get("written"):
                    self._slots.calibrate_gate()
            if r.get("written"):
                self._save_active()
                return {"ok": True, "written": True, "surprise": r.get("surprise")}
            return {"ok": True, "written": False, "surprise": r.get("surprise"),
                    "reason": "the model already knows this (surprise below the write gate) -- not stored"}

    def delete(self, cue: str | None = None, index=None):
        """Surgical per-entry removal (the slotmem contract: the victim drops, every other entry stays
        bit-identical). Match by exact cue, else by index. Persists. {ok, removed, remaining}."""
        import clozn.memory.facts_mode as facts_mode
        if not facts_mode.enabled():
            return {"ok": False, "reason": "the facts tier is off"}
        with self._lock:
            if self._build() is None:
                return {"ok": False, "reason": "no fact store loaded"}
            self._ensure_profile()
            ents = self._slots.entries
            victim = None
            if cue is not None and str(cue).strip():
                victim = next((k for k, e in enumerate(ents) if e["cue"] == str(cue)), None)
            elif index is not None:
                try:
                    idx = int(index)
                    victim = idx if 0 <= idx < len(ents) else None
                except (TypeError, ValueError):
                    victim = None
            if victim is None:
                return {"ok": False, "reason": "no matching fact"}
            removed = ents.pop(victim)["cue"]
            self._slots.calibrate_gate()  # the abstain floor is derived from the store -> recompute
            self._save_active()
            return {"ok": True, "removed": removed, "remaining": len(ents)}

    def read_receipt(self, query: str):
        """The honest read RECEIPT for a query: which entry the store WOULD fire (or that it abstains),
        the key similarity, the abstain floor, the answer it would inject, and the measured slot_ms. Does
        NOT alter any chat reply (v1). {enabled, hit, abstained, sim, gate_floor, cue, answer, slot_ms}."""
        import clozn.memory.facts_mode as facts_mode
        if not facts_mode.enabled():
            return {"enabled": False}
        query = str(query or "").strip()
        with self._lock:
            if self._build() is None or not self._slots.entries:
                return {"enabled": True, "hit": None, "abstained": True, "empty": True,
                        "count": 0, "slot_ms": 0.0}
            self._ensure_profile()
            t0 = time.time()
            with _TRAIN_LOCK:             # the read is a forward on the shared model
                r = self._slots.read(query, gated=True)
            slot_ms = round((time.time() - t0) * 1000.0, 1)
            hit, abst = r.get("hit"), r.get("abstained", False)
            out = {"enabled": True, "hit": hit, "abstained": bool(abst),
                   "sim": (round(float(r["sim"]), 4) if r.get("sim") is not None else None),
                   "gate_floor": (round(float(self._slots.gate_floor), 4)
                                  if self._slots.gate_floor is not None else None),
                   "count": len(self._slots.entries), "slot_ms": slot_ms}
            if hit is not None and not abst:
                e = self._slots.entries[hit]
                out["cue"], out["answer"] = e["cue"], e["answer"]
            return out

    def auto_write(self, messages, reply):
        """Surprise-gated auto-write FROM CONVERSATION: mine a single declarative (cue -> answer) from the
        last user turn and write it under the gate, so the gate refuses what the model already knows. A
        no-op (returns None) when off, when nothing mineable is found, or when the model isn't loaded.
        Best-effort + defensive -- it must never break a chat turn. Returns the write receipt when it
        actually attempted a write (for the runlog), else None."""
        import clozn.memory.facts_mode as facts_mode
        if not facts_mode.enabled():
            return None
        cand = _mine_fact(_last_user(messages))
        if cand is None:
            return None
        cue, answer = cand
        try:
            with self._lock:
                if self._build() is None:
                    return None
                self._ensure_profile()
                with _TRAIN_LOCK:
                    r = self._slots.write(cue, answer, gate=True)
                    if r.get("written"):
                        self._slots.calibrate_gate()
                if r.get("written"):
                    self._save_active()
                return {"cue": cue, "answer": answer, **r}
        except Exception as e:
            print(f"[facts] auto-write skipped: {type(e).__name__}: {e}", flush=True)
            return None


# One process-wide SlotBox, bound to whatever substrate is live (its _mem_provider reads SUB fresh, so a
# substrate swap is picked up automatically). None until the first substrate boots.
SLOTS = None

# One process-wide time-travel SnapshotStore: the bounded, CPU-offloaded ring of per-turn
# KV snapshots. Built lazily from timetravel.get_config() (cap / byte-budget). Only ever holds real KV
# payloads when the `timetravel_snapshots` gate is ON (the RAM rule); branch RECORDING (the transcript
# transform -> child run) does not need it and works regardless. None until first requested.
SNAPSHOTS = None


def _snap_store():
    """The process-wide time-travel SnapshotStore, built lazily with the persisted ring config. Never
    raises -- a config hiccup falls back to the module defaults."""
    global SNAPSHOTS
    if SNAPSHOTS is None:
        try:
            import clozn.replay.timetravel as timetravel
            cfg = timetravel.get_config()
            SNAPSHOTS = timetravel.SnapshotStore(cap=cfg["cap"], budget_mb=cfg["budget_mb"])
        except Exception:
            import clozn.replay.timetravel as timetravel
            SNAPSHOTS = timetravel.SnapshotStore()
    return SNAPSHOTS


def _slots_box():
    """The live SlotBox (lazily created; shares SUB.memory as its backbone). None only before any
    substrate exists."""
    global SLOTS
    if SLOTS is None and SUB is not None:
        SLOTS = SlotBox(lambda: getattr(SUB, "memory", None))
    return SLOTS


# The conversation fact-miner: a deliberately CONSERVATIVE pull of one "<subject> is/are/was <value>"
# statement from a user turn -> (cue, answer) for the surprise-gated store. High-precision-over-recall on
# purpose: a noisy auto-writer would fill the store with junk the gate then has to sieve, so we only fire
# on a clean, short, declarative fact of the personal-memory shape ("My dog's name is Biscuit"). Anything
# ambiguous is left for the explicit "remember this" path. Pure + stdlib -> unit-testable with no model.
import re as _re

_FACT_RE = _re.compile(
    r"\b((?:my|our|the|his|her|their)\b[\w '\-]{1,40}?)\s+(?:is|are|was|were)\s+(?:called\s+|named\s+)?"
    r"([A-Za-z0-9][\w '\-]{0,40}?)\s*[.!?]?$",
    _re.IGNORECASE)


def _mine_fact(text: str):
    """One (cue, answer) from a short declarative user turn, or None. cue is the statement's subject
    rendered as a completion prompt ("My dog's name is" -> answer " Biscuit"); answer carries the leading
    space the store's value schedule expects. None when the turn is a question, too long, or not a clean
    "<subject> is <value>"."""
    t = str(text or "").strip()
    if not t or "?" in t or len(t) > 120 or len(t.split()) > 20:
        return None
    m = _FACT_RE.search(t)
    if not m:
        return None
    subj, val = m.group(1).strip(), m.group(2).strip()
    if not subj or not val or len(val) < 2:
        return None
    # rebuild the cue as the model would be prompted to COMPLETE it, preserving the copula the user used.
    copula = _re.search(r"\b(is|are|was|were)\b", t[m.start():], _re.IGNORECASE)
    verb = copula.group(1).lower() if copula else "is"
    cue = f"{subj} {verb}"
    return cue, " " + val


ARGS = None
SUB = None         # the active substrate object
SUBNAME = "qwen"

# ------- async retrain: one background retrain at a time, chats serialize behind it -----------------
# Mutating a memory card retrains the soft-prefix via consolidate() -- ~4-5 min on the 4-bit 7B. We must
# NOT block the HTTP handler for that. So the card STATUS flip (fast) stays synchronous and the RETRAIN
# runs on a daemon thread. Two module-level guards (a process singleton, like the model itself):
#   _TRAIN_LOCK  -- held for the WHOLE consolidate(); the chat/generate paths acquire+release it so a
#                   reply can't race the shared model+gradients mid-retrain (they queue, they don't error).
#   _RETRAIN     -- the in-flight signal the UI polls: {active, card_id, action, started_at, error}.
# _RETRAIN_META guards reads/writes of the _RETRAIN dict (a tiny critical section, distinct from the long
# _TRAIN_LOCK). Mirrors the _ensure_steer double-checked-lock: we don't launch a 2nd retrain while one runs.
_TRAIN_LOCK = threading.RLock()
_RETRAIN_META = threading.Lock()
_RETRAIN = {"active": False, "card_id": None, "action": None, "started_at": None, "error": None}


def _retrain_status():
    """A snapshot of the in-flight retrain signal (copy -- never hand out the live dict)."""
    with _RETRAIN_META:
        return dict(_RETRAIN)


def _retrain_status_mode():
    """The retrain signal the UI polls, MODE-aware: prompt mode never retrains, so it reports a constant
    idle ({active: false, mode: "prompt"} per the swap spec); internalized reports the live flag."""
    if _memory_mode() == "prompt":
        return {"active": False, "mode": "prompt"}
    return dict(_retrain_status(), mode="internalized")


def _retrain_in_flight():
    with _RETRAIN_META:
        return bool(_RETRAIN["active"])


def _join_retrain(timeout=None):
    """Block until no retrain is in flight (acquire+release _TRAIN_LOCK). Used by tests to await the
    background consolidate deterministically, and available for a graceful shutdown. Returns True once
    the lock was momentarily held with nothing active; False on timeout."""
    if not _TRAIN_LOCK.acquire(timeout=timeout if timeout is not None else -1):
        return False
    try:
        return not _retrain_in_flight()
    finally:
        _TRAIN_LOCK.release()


def _start_retrain(m, action, card_id, force=False):
    """Launch _mem_sync_rules(m) -- the SLOW consolidate() -- on a daemon thread and return immediately.

    PROMPT MODE short-circuits the whole machinery: the cards ARE the memory there, so a mutation only
    syncs m.rules (bookkeeping -- runlog + /state read it) and returns instantly. No consolidate, no
    _TRAIN_LOCK, no thread, no retrain banner; the trained prefix is left completely untouched (it stays
    internalized mode's artifact, preserved for a toggle back).

    Internalized: returns {retraining: True} once the thread is running, or {retraining: False} if
    there's nothing to do (the active set didn't move -- checked synchronously first, so a no-op
    transition never spins a thread) or a retrain is already in flight (we refuse to stack them, like
    _ensure_steer refuses a double compute). The worker holds _TRAIN_LOCK for the whole consolidate so
    chats serialize behind it, and clears _RETRAIN on finish (success OR error) so the UI's poll always
    terminates. `force` skips the no-op pre-check AND forces the consolidate (the mode-switch catch-up:
    rules are synced but the prefix is stale)."""
    import clozn.memory.cards as memory_cards
    if _memory_mode() == "prompt":
        r = _mem_sync_rules(m, reconsolidate=False)          # instant: rules bookkeeping only
        return {"retraining": False, "changed": r["changed"], "mode": "prompt"}
    # cheap synchronous pre-check: would the active set actually change? if not, do NOT spawn a thread.
    if not force and list(getattr(m, "rules", []) or []) == list(memory_cards.active_texts()):
        return {"retraining": False, "changed": False}
    with _RETRAIN_META:
        if _RETRAIN["active"]:                        # a retrain is already running -> don't stack a second
            return {"retraining": True, "busy": True, "queued": False}
        _RETRAIN.update(active=True, card_id=card_id, action=action,
                        started_at=time.time(), error=None)

    def _work():
        err = None
        try:
            with _TRAIN_LOCK:                         # hold across consolidate() -> chats wait, never race
                _mem_sync_rules(m, reconsolidate=True, force=force)
        except Exception as e:                        # a failed retrain must still clear the flag
            err = f"{type(e).__name__}: {e}"
        finally:
            with _RETRAIN_META:
                _RETRAIN.update(active=False, error=err)

    threading.Thread(target=_work, daemon=True).start()
    return {"retraining": True, "action": action, "card_id": card_id}


# ------- substrates: extracted to clozn/server/substrates.py; re-exported here (the seam) -----------
# app.py remains the canonical module: routes read ctx.<name> and tests patch cs.<name> on THIS module.
from clozn.server.substrates import (                                                    # noqa: E402
    Substrate, QwenSubstrate, DreamSubstrate, EngineSubstrate, _EngineMemory,
    load_substrate, switch_substrate, _quant_from_name, _model_family_from_name,
    _engine_model_info, _engine_complete_traced, _ENGINE_MODELS, _ENGINE_MODEL_DEFAULT,
)

# ------- route families: registered in dispatch order; each exposes try_get/try_post(handler, path, ...) --
# returning True once it has written a response, so do_GET/do_POST can stop at the first family that
# claims a path -- exactly the first-match-wins order the old inline if/elif chain had. POST falls
# through, after every family, to the generic SUB.handle(path, body) substrate dispatch inlined at the
# end of do_POST (the catch-all for /memory/*, /steer/* -- those live in the Substrate classes above, not
# in a route module, since they're substrate-polymorphic domain dispatch, not per-path HTTP routing).
#
# runs.try_get_fallback (the generic GET /runs/<id>) is registered LAST in _GET_ROUTES, deliberately --
# every more-specific /runs/<id>/<suffix> family (receipts' /export, runs' own /timeline etc.) must get
# first refusal, since they all share the "/runs/" prefix the fallback also matches.
import types as _types                                                # noqa: E402

from clozn.server import static as _static_routes                     # noqa: E402
from clozn.server.routes import health as _health_routes              # noqa: E402
from clozn.server.routes import runs as _runs_routes                  # noqa: E402
from clozn.server.routes import memory as _memory_routes              # noqa: E402
from clozn.server.routes import facts as _facts_routes                # noqa: E402
from clozn.server.routes import receipts as _receipts_routes          # noqa: E402
from clozn.server.routes import replay as _replay_routes              # noqa: E402
from clozn.server.routes import timetravel as _timetravel_routes      # noqa: E402
from clozn.server.routes import profiles as _profiles_routes          # noqa: E402
from clozn.server.routes import preferences as _preferences_routes    # noqa: E402
from clozn.server.routes import feedback as _feedback_routes          # noqa: E402
from clozn.server.routes import openai as _openai_routes              # noqa: E402
from clozn.server.routes import engine as _engine_routes              # noqa: E402
from clozn.server.routes import readouts as _readouts_routes          # noqa: E402
# Inspector route families: span receipts, fork-at-token, journal actuary +
# calibrated trust spans (F2), shareable card (F9), anchored memory (F6), model diff (F8).
from clozn.server.routes import span_receipts as _span_receipt_routes  # noqa: E402
from clozn.server.routes import fork as _fork_routes                   # noqa: E402
from clozn.server.routes import journal as _journal_routes             # noqa: E402
from clozn.server.routes import card as _card_routes                   # noqa: E402
from clozn.server.routes import anchored as _anchored_routes           # noqa: E402
from clozn.server.routes import diff as _diff_routes                   # noqa: E402
from clozn.server.routes import receipt_link as _receipt_link_routes   # noqa: E402 (ambient delivery ch.1)

_runs_fallback_routes = _types.SimpleNamespace(try_get=_runs_routes.try_get_fallback)

_GET_ROUTES = [_static_routes, _health_routes, _runs_routes, _memory_routes, _receipts_routes,
              _timetravel_routes, _profiles_routes, _openai_routes, _engine_routes,
              _journal_routes, _card_routes, _anchored_routes, _diff_routes, _receipt_link_routes,
              _runs_fallback_routes]
_POST_ROUTES = [_health_routes, _memory_routes, _facts_routes, _receipts_routes, _replay_routes,
               _timetravel_routes, _profiles_routes, _preferences_routes, _feedback_routes,
               _openai_routes, _engine_routes, _readouts_routes,
               _span_receipt_routes, _fork_routes, _journal_routes, _anchored_routes, _diff_routes,
               _receipt_link_routes]


def make_handler():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype, extra_headers=None):
            b = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            if extra_headers:                            # additive + optional: no caller passed this before,
                for k, v in extra_headers.items():        # so today's callers get byte-identical output
                    self.send_header(str(k), str(v))
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _json(self, code, o, extra_headers=None):
            self._send(code, json.dumps(o), "application/json", extra_headers=extra_headers)

        def _client(self, ua):
            ua = (ua or "").lower()
            for k, v in (("open-webui", "Open WebUI"), ("openwebui", "Open WebUI"), ("cursor", "Cursor"),
                         ("vscode", "VS Code"), ("python-requests", "script"), ("httpx", "script"),
                         ("openai-python", "script"), ("curl", "curl"), ("mozilla", "browser")):
                if k in ua:
                    return v
            return ua[:24] or "unknown"

        def _workspace_lens_provider(self, messages, response, error=None):
            """Return a provider callback for real concept readouts, if the brain stack is live.

            Preferred path: C++ engine activations -> SAE concepts (`engine_concepts`).
            Fallback path: loaded Python Qwen activations -> SAE/probe concepts (`sae/probe`).
            No mock data is attached to real runs from here.
            """
            if error or not response or not (SUB and getattr(SUB, "brain", None)):
                return None
            text = str(response or _last_user(messages) or "").strip()[:300]
            if not text:
                return None

            def provider(rid, norm_trace):
                from clozn.readouts import workspace_lens
                if not norm_trace or not norm_trace.get("tokens"):
                    return []
                if ENGINE_QWEN is not None:
                    try:
                        data = SUB.brain.concepts_from_engine(text, ENGINE_QWEN, 15)
                        return workspace_lens.readouts_from_concepts(
                            rid, norm_trace, data, provider="engine_concepts", layer=data.get("layer"))
                    except Exception:
                        pass
                try:
                    data = SUB.brain.concepts_only(text)
                    return workspace_lens.readouts_from_concepts(
                        rid, norm_trace, data, provider="sae/probe", layer=15)
                except Exception:
                    return []

            return provider

        def _log_run(self, source, messages, response, model, started, error=None, trace=None,
                     mem_out=None, finish_reason=None, finish_reason_fallback=None):
            """Persist this interaction as an inspectable run (never let logging break the request).
            mem_out (prompt mode): the {applied, gate, strength?} record the generation path filled --
            what memory ACTUALLY rode this turn (the topic gate may have omitted the block).
            Returns the new run's id (str) on success, else None -- any failure along the way is swallowed,
            never raised. The M5 any-client bridge surfaces this id to the caller; None means "nothing to
            surface", not an error the request should see."""
            try:
                import clozn.runs.store as runlog
                mem = getattr(SUB, "_mem", None) if SUB else None
                mo = mem_out or {}
                mode = mo.get("mode") or _memory_mode()
                if mode == "prompt":
                    # cards_applied == what was INJECTED this turn -- the per-turn honesty prompt mode
                    # buys (internalized can only report the whole active set). applied_ids ride along so
                    # the Run Inspector can offer per-card receipts. A path that filled nothing (or
                    # errored before generating) honestly records an empty application.
                    applied = [c for c in (mo.get("applied") or []) if isinstance(c, dict)]
                    strength = mo.get("strength",
                                      getattr(mem, "memory_strength", 1.0) if mem is not None else 1.0)
                    memd = {"cards_applied": [c.get("text", "") for c in applied],
                            "applied_ids": [c.get("id") for c in applied],
                            "strength": float(strength),
                            "has_prefix": (getattr(mem, "prefix", None) is not None) if mem is not None else False,
                            "mode": mode, "proposed_cards": []}
                    rel = [c.get("relevance") for c in applied]   # per-card topic cosine, aligned with cards_applied
                    if any(r is not None for r in rel):           # omit entirely when the embedder was unavailable
                        memd["relevance"] = [round(float(r), 4) if r is not None else None for r in rel]
                    if mo.get("gate") is not None:
                        memd["gate"] = round(float(mo["gate"]), 4)
                    if mo.get("prompt_block"):
                        memd["prompt_block"] = str(mo["prompt_block"])
                    anchored = [dict(a) for a in (mo.get("anchored") or []) if isinstance(a, dict)]
                    if anchored:
                        memd["anchored"] = anchored
                        if mo.get("anchored_layer") is not None:
                            memd["anchored_layer"] = int(mo["anchored_layer"])
                        if mo.get("anchored_s_total") is not None:
                            memd["anchored_s_total"] = round(float(mo["anchored_s_total"]), 4)
                    if mo.get("anchored_skipped"):
                        memd["anchored_skipped"] = str(mo["anchored_skipped"])
                    if isinstance(mo.get("anchored_loop_guard"), dict):
                        # the loop guard's honest self-healing record --
                        # _flags() below turns this into the visible "memory-retried"/"memory-loop-guard"
                        # run flag; never let a guard event go unrecorded just because no bag rode as
                        # `applied` (anchored memory rides `anchored`, not prompt-mode `applied`).
                        memd["anchored_loop_guard"] = dict(mo["anchored_loop_guard"])
                    if applied:                                  # bump exactly the cards that rode this turn
                        try:
                            import clozn.memory.cards as memory_cards
                            for c in applied:
                                if c.get("id"):
                                    memory_cards.bump_usage(c["id"])
                        except Exception:
                            pass
                elif mem is not None:
                    # INTERNALIZED: cards_applied == the ACTIVE-card texts. Post-D2, SUB._mem.rules is kept
                    # in sync with the active cards (see _mem_sync_rules), so reading .rules still reports
                    # exactly what shaped the reply. Reading SUB.memory would miss the dream cards -- use
                    # _mem (self.memory on qwen, self.dmem on dream). Only ACTIVE cards feed the prefix.
                    cards = getattr(mem, "rules", None) or getattr(mem, "cards", None) or []
                    memd = {"cards_applied": list(cards),
                            "strength": float(getattr(mem, "memory_strength", 1.0)),
                            "has_prefix": getattr(mem, "prefix", None) is not None,
                            "mode": mode, "proposed_cards": []}
                    if cards:                                    # record that the active cards influenced a run
                        try:
                            import clozn.memory.cards as memory_cards
                            for c in memory_cards.list_cards(status="active"):
                                memory_cards.bump_usage(c["id"])
                        except Exception:
                            pass
                else:
                    memd = {"mode": mode}                        # runlog records the mode on EVERY run
                # FACTS tier: only when memory_facts is on -- otherwise zero cost, the
                # latency rule. A chat turn (not the pure /think etc.) gets a surprise-gated AUTO-WRITE
                # (mine one declarative fact; the gate refuses what the model already knows) and a READ
                # RECEIPT (what the store would fire + slot_ms), both folded into the run's memory record
                # so the Run Inspector shows them. Fully guarded; never breaks logging or the reply.
                try:
                    import clozn.memory.facts_mode as facts_mode
                    if facts_mode.enabled() and source in ("studio_chat", "openai_api", "engine_chat"):
                        box = _slots_box()
                        if box is not None and not error:
                            wrote = box.auto_write(messages, response)
                            receipt = box.read_receipt(_last_user(messages))
                            facts_rec = {}
                            if isinstance(receipt, dict) and receipt.get("enabled"):
                                facts_rec["read"] = {k: receipt.get(k) for k in
                                                     ("hit", "abstained", "sim", "gate_floor", "cue",
                                                      "answer", "count", "slot_ms")}
                            if wrote is not None:
                                facts_rec["auto_write"] = wrote
                            if facts_rec:
                                memd = {**(memd if isinstance(memd, dict) else {}), "facts": facts_rec}
                except Exception:
                    pass
                # only meaningfully-nonzero dials (|v| >= 0.05); steer.active() drops exact-zeros but a
                # slider nudged to a hair (e.g. 0.02) still slips through and would clutter the record.
                dials = SUB.steer.active() if (SUB and hasattr(SUB, "steer")) else {}
                dials = {k: v for k, v in dials.items() if abs(float(v)) >= 0.05}
                meta = None
                try:                                          # engine: {model_file, quant, mode, sampling}
                    if SUB is not None and hasattr(SUB, "run_meta"):
                        meta = SUB.run_meta() or None
                except Exception:
                    meta = None
                meta = dict(meta or {})
                git = _git_commit()
                if git:
                    meta.setdefault("build_git_commit", git)
                if finish_reason:
                    meta.setdefault("finish_reason_source", "substrate")
                elif finish_reason_fallback:
                    meta.setdefault("finish_reason_source", "fallback")
                    meta.setdefault("finish_reason_fallback", finish_reason_fallback)
                try:                                          # CAPTURE TIER: record it, and drop the trace at light
                    from clozn.runs import capture_mode
                    tier = capture_mode.tier()
                    meta = {**meta, "capture_tier": tier}
                    if not capture_mode.captures_trace(tier):
                        trace = None                          # light: text + finish_reason + metadata only
                except Exception:
                    pass
                workspace_provider = self._workspace_lens_provider(messages, response, error)
                assembled_messages = mo.get("assembled_messages") if mode == "prompt" else None
                # backlog #5: the EXACT rendered chat-template string the engine produced (mem_out fills it
                # on the engine chat paths). Captured in ANY memory mode -- the internalized/engine path
                # still renders a prompt even without a block. None -> consumers fall back to assembled_messages.
                final_prompt = mo.get("final_prompt")
                rid = runlog.record(source=source, client=self._client(self.headers.get("User-Agent", "")),
                                    model=str(model), substrate=SUBNAME, messages=messages, response=response,
                                    memory=memd, behavior={"active_dials": dials}, started=started, error=error,
                                    trace=trace, finish_reason=finish_reason, meta=meta,
                                    assembled_messages=assembled_messages, final_prompt=final_prompt,
                                    workspace_provider=workspace_provider)
                self._maybe_snapshot_turn(rid, messages, trace, error)
                return rid                        # M5 bridge: the run id, for callers that want to surface it
            except Exception:
                return None                        # logging must never break the request -- no id to surface

        def _maybe_snapshot_turn(self, rid, messages, trace, error):
            """TIME-TRAVEL (#6): when the snapshot gate is ON, register this turn in the bounded ring so the
            Run Inspector's rewind/branch reflects real recorded turns and the ring's bounded eviction runs
            in production. Fully guarded + gated OFF by default (the RAM rule). NOTE: the studio chat path is
            STATELESS (SelfTeach._generate builds its own cache via generate() and discards it), so v1
            records a DESCRIPTOR-only snapshot (turn index + token count, zero offloaded bytes) -- honest,
            and enough for the branch bookkeeping. Capturing the live KV payload here (the re-prefill fast
            path) needs the generation path to hand back its cache: the documented next rung."""
            try:
                if not rid or error:
                    return
                import clozn.replay.timetravel as timetravel
                if not timetravel.enabled():
                    return
                store = _snap_store()
                if store is None:
                    return
                turn = max(0, len(timetravel.message_turns(messages)) - 1)   # this reply's turn index
                # `trace` is a raw per-token step LIST here (runlog normalizes it later) -> len == tokens;
                # tolerate a pre-normalized {tokens:[...]} dict too. 0 when no trace was captured.
                if isinstance(trace, list):
                    n_tok = len(trace)
                elif isinstance(trace, dict):
                    n_tok = len(trace.get("tokens", []) or [])
                else:
                    n_tok = 0
                store.snapshot_turn(rid, turn, n_tok=n_tok, meta={"stateless": True})
            except Exception:
                pass

        def do_GET(self):
            p = self.path.split("?")[0]
            for mod in _GET_ROUTES:
                if mod.try_get(self, p):
                    return
            self._json(404, {"error": "GET " + p})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                # malformed JSON (bad syntax, wrong Content-Length, truncated body, ...) -- a clean 400,
                # not an uncaught JSONDecodeError. No route ever gets a chance to run against garbage.
                self._json(400, {"error": "invalid JSON body"})
                return
            if not isinstance(body, dict):
                # every route does body.get(...) unguarded -- a valid-JSON-but-non-dict body (e.g. `[1,2,3]`
                # or `"hi"`) would otherwise blow up as an AttributeError deep inside whichever route matched.
                self._json(400, {"error": "JSON body must be an object"})
                return
            p = self.path.split("?")[0].rstrip("/") or "/"
            for mod in _POST_ROUTES:
                if mod.try_post(self, p, body):
                    return
            # Generic fallback: whatever the active substrate's own per-action dispatch serves
            # (/memory/*, /steer/* -- Substrate._memory/_steer above, substrate-polymorphic domain
            # dispatch, not per-path HTTP routing, so it stays here rather than in a route module).
            try:
                r = SUB.handle(p, body) if SUB else None
                if r is None:
                    return self._json(409, {"error": f"'{p}' isn't served by the '{SUBNAME}' substrate",
                                            "need": "dream" if p == "/denoise" else "qwen", "active": SUBNAME})
                self._json(200, r)
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return H


def main():
    global ARGS, SUB, SUBNAME
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--substrate", default="qwen", choices=("qwen", "dream", "engine"))
    ARGS = ap.parse_args()
    SUBNAME = ARGS.substrate
    print(f"clozn server: loading '{SUBNAME}' substrate ...", flush=True)
    SUB = load_substrate(SUBNAME)
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), make_handler())
    print(f"\n  CLOZN instrument -> http://{ARGS.host}:{ARGS.port}/   (substrate: {SUBNAME})\n", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
