"""clozn_server.py -- the UNIFIED instrument. One port, one model, the whole white-box surface.

  substrate 'qwen' (default): ONE Qwen-7B serves BOTH the brain (/think -- concepts the model engages)
                              AND the memory (/say /consolidate /check /whatlearned) -- they share the
                              single loaded model, so the instrument's brain and memory tabs are both live.
  substrate 'dream':          Dream-7B serves /denoise (the diffusion window).

Only one 7B fits the GPU, so switching substrates re-execs the process with the other one (a clean GPU);
the instrument shows the active substrate and offers the switch. Serves the instrument + every window
from studio, so the iframes' fetches all land here.

    cloze .venv python -m clozn.clozn_server --port 8090
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine", "lab"))   # so the dream substrate can import cloze_lab
DEMO = os.path.join(HERE, "..", "studio")

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer   # noqa: E402

sys.path.insert(0, os.path.join(HERE, "..", "engine", "client"))     # the engine white-box SDK
import numpy as np                                                   # noqa: E402
try:
    from cloze_engine import EngineClient
    ENGINE = EngineClient(port=int(os.environ.get("CLOZN_ENGINE_PORT", "8091")))            # the live C++ runtime
    ENGINE_QWEN = EngineClient(port=int(os.environ.get("CLOZN_ENGINE_QWEN_PORT", "8092")))  # a Qwen GGUF engine -> concepts
except Exception:
    ENGINE = ENGINE_QWEN = None

CLOZN_DIR = os.path.join(os.path.expanduser("~"), ".clozn")   # studio memory + personality persist here
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))


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


def _engine_generation_meta(max_new=None, stream=None):
    return _without_unknowns({
        "sampler_mode": "greedy",
        "sampling": "greedy",
        "temperature": 0.0,
        "repetition_penalty": 1.0,
        "seed": 0,
        # Self-describing decode block (REPRODUCE_AND_PROVE_PLAN S2): the honest regime this run
        # was produced under, so re-derivation/forced-scoring is exact-by-construction and S5
        # (sampling) can flip it to {"mode":"sample", ...} + the real seed without schema churn.
        # Engine chat is greedy (temperature 0), seed 0 -- the actual values passed, not a guess.
        "decode": {"mode": "greedy", "temperature": 0.0, "seed": 0},
        "max_tokens": int(max_new) if max_new is not None else None,
        "stream": bool(stream) if stream is not None else None,
    })


def _pers(name):
    return os.path.join(CLOZN_DIR, name)


ENGINE_STEER = None        # lazy EngineSteer on the Qwen GGUF engine -- tone dials on the C++ runtime, any GGUF


def _engine_steer():
    global ENGINE_STEER
    if ENGINE_STEER is None and ENGINE_QWEN is not None:
        from clozn.steering import EngineSteer
        ENGINE_STEER = EngineSteer(ENGINE_QWEN)
    return ENGINE_STEER


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
        from clozn.steering import suggest_dial_for_preference
        return suggest_dial_for_preference(text)
    except Exception:
        return None


QUOTE_SPAN_MAX = 240   # a "you said this" quote is for recognizing your own words, not re-reading the essay


def _provenance_of(messages):
    """The (source_turn, quoted_span) pair for a card proposed from `messages` (roadmap NEXT_STEPS #1, the
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
        from clozn import memory_mode
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
        from clozn.topic_gate import get_gate
        return float(get_gate().scalar(last_user, list(texts)))
    except Exception:
        return 1.0


def _prompt_relevance(last_user, texts):
    """Per-card topic cosine {text: relevance} for the applied block -- the SAME embeddings _prompt_gate
    just used (cached by string in topic_gate), so it's ~free. {} when the embedder is unavailable. This
    is the per-card signal the run record needs so the inspector can show WHY each card fired, not just
    that the block as a whole did (the scalar gate)."""
    try:
        from clozn.topic_gate import get_gate
        return dict(get_gate().relevance(last_user, list(texts)))
    except Exception:
        return {}


def _prompt_mem_cards(mem, exclude_ids=()):
    """The ACTIVE cards ({id, text}) that feed the prompt block, minus exclude_ids (replay's REAL
    per-card ablation). Reads the card store directly (memory_mode.active_cards) -- in prompt mode the
    cards ARE the memory (m.rules is bookkeeping that can lag right after boot). Falls back to
    mem.rules (id-less) only if the store module is unavailable, so a broken store degrades to the old
    rule list rather than to amnesia."""
    from clozn import memory_mode
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
    from clozn import memory_mode
    return memory_mode.compile_prompt_block(texts), applied, g


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
    from clozn import memory_cards
    try:
        return memory_cards.migrate_from_rules(list(getattr(m, "rules", []) or []))
    except Exception:
        return []


def _export_markdown(run: dict, xr: dict | None) -> str:
    """Render a run (+ its M1 explain) as a human-readable Markdown receipt: the conversation, what memory
    and which dials shaped it (with per-card relevance), why it stopped, and where it hesitated. Pure / no
    model -- the JSON export carries the full structured bundle; this is its readable companion."""
    from clozn import receipt_bundle
    return receipt_bundle.to_markdown(receipt_bundle.build(run, explain=xr))


def _runs_for_card(card_id):
    """Best-effort: the run summaries whose memory.cards_applied names this card (by id OR by text).
    cards_applied currently records the active rule TEXTS (see _log_run), so we match on text primarily
    and on id as a forward-compatible fallback. Returns [] when the card / runs are gone (never raises)."""
    from clozn import memory_cards
    from clozn import runlog
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
    from clozn import memory_cards
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


# ------- profiles: named persona bundles (NEXT_STEPS #4) -> cards + dials on the LIVE substrate -----
# profiles.py is the model-free CRUD + compile layer (source bundles: card texts, dial settings, custom-
# dial recipes, fact pairs -- see its docstring). This is the thin wiring that hands it the live objects
# a switch needs (SUB._mem for cards/rules, SUB.steer for dials) and reports what actually happened.

def _active_profile_name():
    """The name of the last-switched-to profile, or None (nothing switched yet this install). Persisted
    in studio_settings.json alongside memory_mode -- one small settings file, not a new one."""
    from clozn import memory_mode
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
    from clozn import memory_cards

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
        from clozn import profiles
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
    from clozn import memory_mode
    memory_mode.set_setting("active_profile", p["name"])

    facts_note = None
    if p.get("facts"):
        from clozn import facts_mode
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
                            from clozn import profiles as _pf
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
    from clozn import profiles
    return profiles.prompt_block(p)


# ------- the FACTS tier: slot-memory store wired to the studio (NEXT_STEPS #5) ----------------------
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
            from clozn import facts_mode
            from clozn import slotmem_qwen
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
        from clozn import facts_mode
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
        from clozn import facts_mode
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
        from clozn import facts_mode
        n = len(self._slots.entries) if self._slots is not None else 0
        return {"enabled": facts_mode.enabled(), "layer": facts_mode.LAYER,
                "profile": self._profile or self._active_profile() or "default", "count": n}

    def list_entries(self):
        """[{cue, answer, label}] for the resident store, [] when off / unbuilt. Read-only, no model
        forward -- safe to call on every Facts-panel load."""
        from clozn import facts_mode
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
        from clozn import facts_mode
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
        from clozn import facts_mode
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
        from clozn import facts_mode
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
        from clozn import facts_mode
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

# One process-wide time-travel SnapshotStore (NEXT_STEPS #6): the bounded, CPU-offloaded ring of per-turn
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
            from clozn import timetravel
            cfg = timetravel.get_config()
            SNAPSHOTS = timetravel.SnapshotStore(cap=cfg["cap"], budget_mb=cfg["budget_mb"])
        except Exception:
            from clozn import timetravel
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
    from clozn import memory_cards
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


class Substrate:
    """Shared studio surface for any substrate: the /memory/* trait cards and the /steer/* tone dials, on
    whatever model the subclass loads. A subclass sets self.steer, self._mem (a memory object exposing
    .rules / .prefix / .consolidate(rules) / .reset()), self._pers_steer, self._steer_ready/_steer_info,
    and defines _gen(prompt) -- a one-shot generate used by the /steer/check A/B (AR generate vs denoise).
    So memory + dials are written ONCE and work identically on Qwen and Dream."""

    def _memory(self, path, body):
        """Card-backed memory (D2 + E1). Cards carry the metadata + review status; m.rules stays == the
        ACTIVE-card texts and drives the prefix via consolidate(). Status changes go through _mem_sync_rules,
        which only retrains when the active set actually moved -- so pending/no-op edits never touch the prefix."""
        from clozn import memory_cards
        m = self._mem
        self._ensure_cards_migrated()           # one-time seed of legacy rules -> active cards (no retrain)

        if path == "/memory/cards":             # OBJECTS now (not bare strings) -- the review layer
            return {"cards": memory_cards.list_cards(), "has_prefix": m.prefix is not None,
                    "mode": _memory_mode(),     # the UI adapts its copy / hides retrain chrome on this
                    "retraining": _retrain_status_mode()}   # fold the in-flight signal in (one reload sees it)

        if path == "/memory/retrain-status":    # the poll target: is a background consolidate() running?
            return _retrain_status_mode()       # prompt mode: never ({active:false, mode:"prompt"})

        if path == "/memory/add":               # propose a card as PENDING -> does NOT affect the prefix
            text = str(body.get("text", "")).strip()
            if not text:
                return {"ok": False, "reason": "empty trait"}
            card = memory_cards.create(text, status="pending", kind="preference",
                                       risk=_risk_of(text), source_run_id=body.get("source_run_id"),
                                       evidence=str(body.get("evidence", "")))
            if not card:
                return {"ok": False, "reason": "could not create card"}
            # If this is really a STYLE preference, surface the tone DIAL that delivers it (the trained
            # prefix carries topical prefs well but style ones weakly). Card is still created + pending;
            # this only SUGGESTS the better mechanism -- null when the text isn't a style match.
            return {**card, "dial_suggestion": _dial_suggestion(text)}

        if path == "/memory/remove":            # delete by id -> if it was active, rebuild from the rest
            cid = str(body.get("id", "")).strip()
            if not cid:                          # (index removed -- ids are the stable handle now)
                return {"ok": False, "reason": "need a card id"}
            was_active = (memory_cards.get(cid) or {}).get("status") == "active"
            ok = memory_cards.delete(cid)
            if not ok:
                return {"ok": False, "reason": "no such card"}
            # delete is synchronous+fast; the retrain (only if an ACTIVE card left the set) is backgrounded.
            resync = _start_retrain(m, "remove", cid) if was_active else {"retraining": False}
            return {"ok": True, "removed": cid, "resync": resync}

        if path in ("/memory/approve", "/memory/reject", "/memory/disable", "/memory/enable"):
            return self._card_status(path.rsplit("/", 1)[1], str(body.get("id", "")).strip())

        if path == "/memory/edit":              # change a card's text; if active, retrain on the new text
            cid = str(body.get("id", "")).strip()
            new_text = str(body.get("text", "")).strip()
            if not (cid and new_text):
                return {"ok": False, "reason": "need id and text"}
            card = memory_cards.update(cid, text=new_text, risk=_risk_of(new_text))
            if card is None:
                return {"ok": False, "reason": "no such card"}
            if card.get("status") == "active":   # editing an active card's text retrains -> in the background
                card = {**card, "resync": _start_retrain(m, "edit", cid)}
            return card

        if path == "/memory/strength":          # the memory dial. Internalized: scales how hard the prefix
            # bites (0 = off, >1 = stronger). PROMPT mode: on/off only -- 0 never injects the block, any
            # >0 injects when the topic gate lets it in (nothing scales continuously; the UI hint says so).
            if "value" in body and hasattr(m, "memory_strength"):
                m.memory_strength = max(0.0, min(2.0, float(body["value"])))
                if hasattr(m, "save"):
                    try:
                        m.save()                             # persists inside the .pt (needs a prefix)
                    except Exception:
                        pass
                try:                                         # mirror to settings so the dial survives a
                    from clozn import memory_mode                       # restart in prompt mode (no .pt to carry it)
                    memory_mode.set_setting("memory_strength", float(m.memory_strength))
                except Exception:
                    pass
            return {"strength": float(getattr(m, "memory_strength", 1.0)), "has_prefix": m.prefix is not None,
                    "mode": _memory_mode()}

        if path == "/memory/gatecheck":         # DEBUG (live calibration): the topic-relevance gate for a prompt
            # Exposes both raw signals + the final gate + per-rule cosines for the active rules, so the bands
            # (lo_t/hi_t/lo_o/hi_o) can be tuned against real on/off-topic prompts. Fully guarded: on any
            # failure (or no embedder) it reports the no-gating baseline (gate 1.0) rather than raising.
            prompt = str(body.get("prompt", ""))
            rules = list(getattr(m, "rules", []) or [])
            try:
                from clozn.topic_gate import get_gate
                dbg = get_gate().debug(prompt, rules)
            except Exception as e:
                dbg = {"gate": 1.0, "topic": 0.0, "openness": 0.0, "relevance": {},
                       "ok": False, "error": f"{type(e).__name__}: {e}"}
            # `gate` here is the RAW topic gate (relevance only); the applied scale is memory_strength x
            # gate (internalized) or include-iff gate >= PROMPT_GATE_MIN (prompt mode) -- mode says which.
            return {"prompt": prompt, "rules": rules, "mode": _memory_mode(),
                    "strength": float(getattr(m, "memory_strength", 1.0)), **dbg}
        return None

    # ---- E1 review lifecycle: a status change rebuilds m.rules from the active set, retrains iff it moved -
    def _card_status(self, action, cid):
        """approve->active, reject->rejected, disable->disabled, enable->active. The STATUS flip (fast) is
        synchronous; the RETRAIN it may trigger (rebuild the prefix from active_texts) is backgrounded so
        the response returns immediately. The card keeps its FINAL status; a separate _RETRAIN flag carries
        the in-flight signal. _start_retrain no-ops when the active set didn't actually move (prefix safe).

        PROVENANCE GATE (NEXT_STEPS #1): 'approve' is refused for a card that CLAIMS a run (source_run_id
        set) but carries no quoted_span to back that claim up -- memory_cards.is_provenance_claim_unbacked.
        This is never auto-approvable; the reviewer sees why via the reason string (the Memory page also
        flags it so this should rarely even be attempted). reject/disable/enable are NOT gated -- you must
        always be able to discard or de-activate a card regardless of its provenance."""
        from clozn import memory_cards
        if not cid:
            return {"ok": False, "reason": "need a card id"}
        if action == "approve":
            existing = memory_cards.get(cid)
            if existing is not None and memory_cards.is_provenance_claim_unbacked(existing):
                return {"ok": False, "reason": "no provenance -- this card cites a run but has no quoted "
                                                "span backing it up, so it can't be approved"}
        target = {"approve": "active", "reject": "rejected",
                  "disable": "disabled", "enable": "active"}[action]
        card = memory_cards.set_status(cid, target)
        if card is None:
            return {"ok": False, "reason": "no such card"}
        resync = _start_retrain(self._mem, action, cid)  # retrains on a thread iff the active set changed
        return {**card, "resync": resync}

    def _ensure_cards_migrated(self):
        """Seed the card store from this substrate's legacy rule-strings exactly once per process."""
        if getattr(self, "_cards_migrated", False):
            return
        _mem_migrate(self._mem)
        self._cards_migrated = True

    def _ensure_steer(self):
        """Compute the axis vectors once, race-safe (double-checked lock). Two dial calls racing on first
        use could otherwise both run compute() on the shared model at once and corrupt it (IndexError)."""
        if not self._steer_ready:
            with self._steer_lock:
                if not self._steer_ready:
                    self._steer_info = self.steer.compute()
                    self._steer_ready = True

    def _steer(self, path, body):
        from clozn.steering import AXES
        if path == "/steer/axes":
            calib = _dial_calibration()   # {} when uncalibrated/offline -- _with_calibration no-ops per axis
            axes = [_with_calibration(
                        {"name": k, "poles": AXES[k]["poles"], "value": self.steer.strength.get(k, 0.0),
                         "max": AXES[k].get("max", 1.5)}, calib.get(k))
                    for k in AXES]
            lib_names = _library_dial_names()   # shipped-library custom dials -- NOT user-made, never "yours"
            for k, v in getattr(self.steer, "custom", {}).items():   # user-defined + shipped-library dials
                axis = {"name": k, "poles": v["poles"], "value": self.steer.strength.get(k, 0.0), "max": v["max"]}
                if k in lib_names:
                    axis["library"] = True     # shipped, curated dial -- distinct from a user's own custom
                else:
                    axis["custom"] = True      # unchanged: a genuine user-made dial ("yours" + deletable)
                axes.append(_with_calibration(axis, calib.get(k)))
            return {"axes": axes, "ready": self._steer_ready, "substrate": self.name}
        self._ensure_steer()                    # compute the axis vectors once on first real use (race-safe)
        if path == "/steer/compute":
            return {"ready": True, **self._steer_info}
        if path == "/steer/set":
            self.steer.set(str(body["name"]), float(body.get("value", 0.0)))
            self.steer.save_state(self._pers_steer)
            return {"active": self.steer.active()}
        if path == "/steer/check":              # A/B one dial: baseline vs steered (subclass _gen)
            prompt = str(body.get("prompt", ""))[:300]
            base = self._gen(prompt)
            self.steer.clear()
            self.steer.set(str(body["name"]), float(body.get("value", 1.0)))
            self.steer.engage()
            try:
                steered = self._gen(prompt)
            finally:
                self.steer.disengage()
                self.steer.clear()
            return {"prompt": prompt, "axis": body.get("name"), "value": body.get("value", 1.0),
                    "baseline": base, "steered": steered}
        if path == "/steer/custom":             # USER-DEFINED dial: compute mean(+pole)-mean(-pole) live
            if not hasattr(self.steer, "add_custom"):
                return {"error": "custom dials are not supported on this substrate yet"}
            name = str(body.get("name", "")).strip()[:24]
            pos, neg = str(body.get("pos", "")).strip(), str(body.get("neg", "")).strip()
            if not (name and pos and neg):
                return {"error": "need a name and both poles (pos, neg)"}
            info = self.steer.add_custom(name, pos, neg, float(body.get("max", 0.5)))
            self.steer.save_custom(_pers(f"studio_custom_{self.name}.json"))
            return {"name": name, "max": info["max"], "custom": list(self.steer.custom)}
        if path == "/steer/custom_delete":
            if hasattr(self.steer, "remove_custom"):
                self.steer.remove_custom(str(body.get("name", "")))
                self.steer.save_custom(_pers(f"studio_custom_{self.name}.json"))
                self.steer.save_state(self._pers_steer)
            return {"custom": list(getattr(self.steer, "custom", {}))}
        return None


class QwenSubstrate(Substrate):
    """One Qwen-7B + SAE behind the concept readout AND the memory + tone dials."""
    name = "qwen"

    def __init__(self):
        from clozn.brain_readout import BrainReadout
        from clozn.sae7b import GpuSAE, load7b
        from clozn.self_teach_server import SelfTeach
        from clozn.steering import SteeringControl
        sae = GpuSAE()
        tok, model = load7b()
        self.brain = BrainReadout(model, tok, sae, DEMO, HERE)
        self.memory = SelfTeach("Qwen/Qwen2.5-7B-Instruct", model=model, tok=tok,   # shares the model
                                persist_path=_pers("studio_memory.pt"))
        self.steer = SteeringControl(model, tok)            # tone dials on the same model
        self._mem = self.memory
        self._steer_ready, self._steer_info, self._steer_lock = False, {}, threading.Lock()
        self._pers_steer = _pers("studio_personality.json")
        self.steer.load_state(self._pers_steer)             # restore the personality dials across restarts
        self.steer.load_custom(_pers(f"studio_custom_{self.name}.json"))    # + any user-defined dials
        # + the shipped dial library (research/deploy_dial_library.py), if it has ever been deployed on
        # this install -- a no-op (load_custom returns immediately) until that script runs. Loaded SECOND
        # so a rare --force-deployed name collision (a user custom + a library dial sharing a name)
        # resolves to the library's direction on every subsequent boot, matching what --force just did live.
        self.steer.load_custom(_pers("studio_library.json"))
        if _memory_mode() == "prompt":
            # PROMPT MODE boots from the CARD STORE (the prefix isn't applied): adopt the active-card
            # texts as m.rules right away so /state + runlog bookkeeping don't lag until the first
            # /memory call. sync_cards never touches the prefix; it also runs the one-time migration.
            self.memory.sync_cards()
            self._cards_migrated = True
            try:                                            # in prompt mode the strength dial persists in
                from clozn import memory_mode                          # settings (the .pt needs a prefix to save; a
                s = memory_mode.get_setting("memory_strength")            # fresh install has none)
                if s is not None:
                    self.memory.memory_strength = max(0.0, min(2.0, float(s)))
            except Exception:
                pass

    def handle(self, path, body):
        if path == "/think":
            return self.brain.think(str(body.get("text", ""))[:500], str(body.get("sid", "default")))
        if path == "/concepts":                 # read what fired inside (no generation) -> annotate a reply
            return self.brain.concepts_only(str(body.get("text", ""))[:500])
        if path == "/say":
            with _TRAIN_LOCK:                    # studio chat touches the shared model -> wait out a retrain
                # body["_trace_out"] / body["_mem_out"] (optional, server-side only): collectors the handler
                # passes for the Run Inspector trace + the per-turn memory record; never echoed to the client.
                if _memory_mode() == "prompt":
                    return {"reply": self._say_prompt(body["message"], body.get("max_new", 200),
                                                      trace_out=body.get("_trace_out"),
                                                      mem_out=body.get("_mem_out"))}
                return {"reply": self.memory.say(body["message"], body.get("max_new", 200),
                                                 trace_out=body.get("_trace_out"))}
        if path == "/consolidate":               # a manual retrain -> the same shared-model lock as card retrains
            with _TRAIN_LOCK:
                return self.memory.consolidate(body.get("rules"), body.get("steps", 120), body.get("lr", 0.012),
                                               body.get("n_probe", 8), body.get("max_norm", 14.0))
        if path == "/whatlearned":
            if _memory_mode() == "prompt":
                return self._whatlearned_prompt()
            return {"report": self.memory.what_learned(), "mode": "internalized"}
        if path == "/check":                     # generates on the shared model -> wait out a retrain
            with _TRAIN_LOCK:
                if _memory_mode() == "prompt":
                    return self._check_prompt(body["prompt"], body.get("max_new", 200))
                return self.memory.check(body["prompt"], body.get("max_new", 200))
        if path == "/reset":
            with _TRAIN_LOCK:                     # mutates the prefix/model state -> don't race a retrain
                self.brain.reset(str(body.get("sid", "default")))
                return self.memory.reset(body.get("keep_prefix", False))
        if path.startswith("/memory/"):
            return self._memory(path, body)
        if path.startswith("/steer/"):
            return self._steer(path, body)
        return None

    # ---- prompt-mode twins of the SelfTeach conversation endpoints ---------------------------------
    # Same surfaces, same shapes -- but memory rides as the gated system block and the model runs
    # prefix-free (use_prefix=False). SelfTeach itself is untouched: it stays the internalized-mode
    # engine and the research instrument (the self-audit experiments REQUIRE a non-text memory).

    def _say_prompt(self, message, max_new=200, trace_out=None, mem_out=None):
        """One /say turn in prompt mode: history grows exactly as SelfTeach.say, but the memory is the
        compiled block (topic-gated on THIS user turn). Runs under the caller's _TRAIN_LOCK; takes
        m.lock like say() so concurrent history appends can't interleave."""
        m = self.memory
        with m.lock:
            m.history.append({"role": "user", "content": message})
            block, applied, gate = _prompt_block_for(m, message)
            assembled = _inject_block(m.history, block)
            if mem_out is not None:
                mem_out.update(mode="prompt", applied=applied, gate=gate,
                               prompt_block=block, assembled_messages=assembled)
            reply = m._generate(assembled, use_prefix=False, max_new=max_new,
                                sample=True, trace_out=trace_out)
            m.history.append({"role": "assistant", "content": reply})
            return reply

    def _whatlearned_prompt(self):
        """Prompt-mode /whatlearned: ask from a fresh context WITH the block injected, ungated (the
        self-view shows the full memory, mirroring what_learned's apply_gate=False). Honesty: in this
        mode the model is READING its cards out of context, not introspecting a trained prefix -- the
        `mode` field is there so the UI labels it as reading, not self-knowledge."""
        m = self.memory
        cards = _prompt_mem_cards(m)
        if not cards:
            return {"report": "(no active memory cards yet -- add or approve one on the Memory page)",
                    "mode": "prompt"}
        from clozn import memory_mode
        block = memory_mode.compile_prompt_block([c["text"] for c in cards])
        ask = ("What have you picked up about me so far -- my interests, anything I seem to care about, "
               "and how I like you to respond? List what you know, one item per line.")   # == what_learned's
        with m.lock:
            report = m._generate([{"role": "system", "content": block}, {"role": "user", "content": ask}],
                                 use_prefix=False, max_new=200, sample=False)
        return {"report": report, "mode": "prompt"}

    def _check_prompt(self, prompt, max_new=200):
        """Prompt-mode /check, mirroring check()'s response shape: baseline vs block-in-context. The
        block is binary per turn, so `ungated` == block always in, and `gated` == what a real chat does
        (identical when the gate lets it in, the plain baseline when the topic gates it out -- greedy
        decode makes the reuse exact, no second generation needed)."""
        m = self.memory
        with m.lock:
            msgs = [{"role": "user", "content": prompt}]
            base = m._generate(msgs, use_prefix=False, max_new=max_new, sample=False)
            cards = _prompt_mem_cards(m)
            if not cards:
                return {"prompt": prompt, "gate": None, "baseline": base, "mode": "prompt",
                        "ungated": "(no active memory cards)", "gated": "(no active memory cards)"}
            texts = [c["text"] for c in cards]
            from clozn import memory_mode
            block = memory_mode.compile_prompt_block(texts)
            g = round(_prompt_gate(prompt, texts), 3)
            ungated = m._generate(_inject_block(msgs, block), use_prefix=False, max_new=max_new, sample=False)
            gated = ungated if g >= PROMPT_GATE_MIN else base
            return {"prompt": prompt, "gate": g, "baseline": base, "ungated": ungated, "gated": gated,
                    "mode": "prompt"}

    def _gen(self, prompt):                     # AR generate for the /steer/check A/B
        return self.steer.generate(prompt, 90)

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        """One stateless chat completion with the WHOLE tunable self applied: the memory (as the trained
        prefix in internalized mode, as the topic-gated system block in prompt mode) AND the active
        tone-steering sliders, both on the shared model. This is what the OpenAI-compatible endpoint
        serves -- normal chat on the surface, legible and tunable underneath. Serializes behind an
        in-flight memory retrain (_TRAIN_LOCK) so a reply can't race the shared model+gradients
        mid-consolidate -- it waits, briefly, rather than corrupting.
        trace_out (optional list): filled with the per-token trace for the Run Inspector; reply unchanged.
        mem_out (optional dict): prompt mode fills {mode, applied, gate} -- what memory ACTUALLY rode
        this turn -- so the run log records per-turn application, not just the active set."""
        with _TRAIN_LOCK:                        # wait out any background retrain, then hold for this reply
            self._last_generation_meta = _qwen_generation_meta(max_new, sample=sample, stream=False)
            self._last_finish_reason = None
            if self.steer.strength:             # persisted personality -> ensure vectors are ready (race-safe)
                self._ensure_steer()
            self.steer.engage()
            try:
                if _memory_mode() == "prompt":
                    # PROMPT MODE: the cards ride as the system block (omitted when the topic gate says
                    # this turn is off-memory); the model runs prefix-free. The block wording is the
                    # exact distillation target the prefix trains toward, so behavior stays comparable.
                    block, applied, gate = _prompt_block_for(self.memory, _last_user(messages))
                    assembled = _inject_block(messages, block)
                    if mem_out is not None:
                        mem_out.update(mode="prompt", applied=applied, gate=gate,
                                       prompt_block=block, assembled_messages=assembled)
                    reply = self.memory._generate(assembled, use_prefix=False,
                                                  max_new=max_new, sample=sample, trace_out=trace_out)
                    self._last_finish_reason = getattr(self.memory, "_last_finish_reason", None)
                    return reply
                # gate="auto" -> _generate scales the memory prefix by memory_strength x TOPIC RELEVANCE, so
                # the OpenAI /v1/chat path gets the same on-topic gating as /say (fixes the always-on
                # over-bleed). memory_strength 0 still zeroes it; a missing embedder falls back to no-gating.
                reply = self.memory._generate(messages, use_prefix=True, max_new=max_new, sample=sample,
                                              gate="auto", trace_out=trace_out)
                self._last_finish_reason = getattr(self.memory, "_last_finish_reason", None)
                return reply
            finally:
                self.steer.disengage()

    def last_finish_reason(self):
        return getattr(self, "_last_finish_reason", None)

    def run_meta(self):
        return dict(getattr(self, "_last_generation_meta", None) or
                    _qwen_generation_meta(sample=True))

    def last_stream_trace(self):
        """The per-token trace captured during the most recent chat_stream (raw step list, or []). The SSE
        handler reads this AFTER the generator is exhausted to log it -- streaming yields text, not tokens,
        so the trace is assembled from the recorder's rows + the generated ids, not from the chunks."""
        return list(getattr(self, "_last_stream_trace", []) or [])

    def chat_stream(self, messages, max_new=256, mem_out=None):
        """Streaming chat: yields text chunks as the AR model generates -- memory + tone steering
        applied -- via a TextIteratorStreamer with generate() in a thread. Local AR is slow, so this is
        the big UX win the diffusion side doesn't need (diffusion is trace-based, not left-to-right).
        mem_out: as in chat() -- prompt mode records what memory actually rode this turn.

        Per-token trace (B3): a pure pass-through RecordingLogitsProcessor rides along and the generated
        ids are captured from generate()'s return -> after the stream ends we assemble the trace into
        self._last_stream_trace for the SSE handler to log. Pass-through means the streamed chunks are
        byte-identical to before; the whole capture is wrapped so any failure just leaves the trace empty."""
        import threading
        import torch
        from transformers import TextIteratorStreamer
        _TRAIN_LOCK.acquire()                    # serialize behind an in-flight retrain (released in finally)
        self._last_generation_meta = _qwen_generation_meta(max_new, sample=True, stream=True)
        self._last_finish_reason = None
        if self.steer.strength:
            self._ensure_steer()
        m = self.memory
        if _memory_mode() == "prompt":
            # PROMPT MODE: the gated system block replaces the prefix concat below -- it simply becomes
            # part of the chat template, so the streaming mechanics are untouched.
            block, applied, gate = _prompt_block_for(m, _last_user(messages))
            assembled = _inject_block(messages, block)
            if mem_out is not None:
                mem_out.update(mode="prompt", applied=applied, gate=gate,
                               prompt_block=block, assembled_messages=assembled)
            e = m._embed(m._chat_ids(assembled))
        else:
            e = m._embed(m._chat_ids(messages))
            if m.prefix is not None:                        # prepend the consolidated memory prefix, scaled by
                # memory_strength x TOPIC RELEVANCE (same on-topic gating as _generate's gate="auto"; this path
                # inlines generate() for streaming so it can't call _generate, but the scale must match). A
                # missing embedder makes rel==1.0 (no gating); memory_strength 0 zeroes the prefix entirely.
                last_user = next((mm["content"] for mm in reversed(messages) if mm.get("role") == "user"), "")
                g = m.memory_strength * m._gate(last_user)
                e = torch.cat([(g * m.prefix.detach()).to(e.dtype)[None], e], 1)
        att = torch.ones(e.shape[:2], device=e.device, dtype=torch.long)
        streamer = TextIteratorStreamer(m.tok, skip_prompt=False, skip_special_tokens=True)
        kw = dict(inputs_embeds=e, attention_mask=att, max_new_tokens=max_new, do_sample=True,
                  temperature=0.7, top_p=0.9, repetition_penalty=1.3, no_repeat_ngram_size=3,
                  pad_token_id=m.eos or 0, streamer=streamer)            # trim steering-induced loops
        self._last_stream_trace = []                        # reset; filled after the stream if capture succeeds
        recorder = None
        try:                                                # observe-only trace capture (never affects output)
            from clozn.self_teach_server import RecordingLogitsProcessor
            from transformers import LogitsProcessorList
            recorder = RecordingLogitsProcessor()
            kw["logits_processor"] = LogitsProcessorList([recorder])
        except Exception:
            recorder = None
        gen_out = {}                                        # holder so the thread can hand back generate()'s ids

        def _gen():
            with torch.no_grad():
                out = m.model.generate(**kw)
                try:
                    gen_out["ids"] = [int(t) for t in out[0].tolist()]   # inputs_embeds -> generated ids only
                except Exception:
                    pass

        self.steer.engage()                                 # tone dials apply during the streamed generation
        th = threading.Thread(target=_gen, daemon=True)
        th.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            th.join()
            raw_ids = list(gen_out.get("ids", []) or [])
            try:
                from clozn.self_teach_server import finish_reason_from_generated_ids
                self._last_finish_reason = finish_reason_from_generated_ids(raw_ids, m.eos, max_new)
            except Exception:
                self._last_finish_reason = None
            if recorder is not None:                        # assemble the trace from rows + emitted ids
                try:
                    from clozn.self_teach_server import steps_from_records
                    gen_ids = list(raw_ids)
                    while gen_ids and gen_ids[-1] == (m.eos or -1):
                        gen_ids.pop()
                    self._last_stream_trace = steps_from_records(recorder.records, gen_ids, m.tok)
                except Exception:
                    self._last_stream_trace = []
            self.steer.disengage()
            _TRAIN_LOCK.release()                           # done streaming -> let a queued retrain proceed

    def state(self):
        return self.memory.state()


class DreamSubstrate(Substrate):
    """Dream-7B diffusion: the denoise window, plus the SAME trait-card memory and tone dials as Qwen."""
    name = "dream"

    def __init__(self):
        from cloze_lab.cli import build_adapter
        from clozn.denoise_server import trace_for
        from clozn.steering import DreamSteering
        from clozn.dream_memory import DreamMemory
        self.adapter = build_adapter("dream", device="cuda", quant="nf4")
        self._trace = trace_for
        self.steer = DreamSteering(self.adapter)            # tone dials on the diffusion model
        self._steer_ready, self._steer_info, self._steer_lock = False, {}, threading.Lock()
        self._pers_steer = _pers("studio_dream_personality.json")
        self.steer.load_state(self._pers_steer)
        self.dmem = DreamMemory(self.adapter,               # diffusion-native memory (trained soft prefix)
                                persist_path=_pers("studio_dream_memory.pt"))
        self._mem = self.dmem

    def handle(self, path, body):
        if path == "/denoise":
            prompt = str(body.get("prompt", ""))[:300]
            with _TRAIN_LOCK:                              # wait out a background retrain (it moves dmem.prefix)
                self.steer.engage()                        # active dials steer every denoising pass
                try:
                    ad = self.adapter
                    # In PROMPT mode the trained dream prefix is NOT applied: cards may have been edited
                    # instantly (no consolidate), so the prefix can be stale vs the cards -- and denoise
                    # is a raw completion window with no system slot for the block. Memory simply doesn't
                    # ride here in prompt mode (honest omission beats a stale injection).
                    if self.dmem.prefix is not None and _memory_mode() != "prompt":
                        from clozn.dream_memory import PrefixAdapter   # memory present -> inject into the REAL scheduler
                        ad = PrefixAdapter(self.adapter, self.dmem.prefix.detach())
                    return self._trace(ad, prompt)         # the cloze_lab scheduler (+ the steering hook)
                finally:
                    self.steer.disengage()
        if path.startswith("/memory/"):
            return self._memory(path, body)
        if path.startswith("/steer/"):
            return self._steer(path, body)
        return None

    def _gen(self, prompt):                                # base denoise final text for the /steer/check A/B
        return self._trace(self.adapter, str(prompt)[:200])["final_text"]

    def state(self):
        return {"dials": self.steer.active(), "cards": self.dmem.rules}


class _EngineMemory:
    """Thin prompt-mode memory for the engine substrate: the CARD STORE *is* the memory. No model, no
    learned prefix (the soft-prefix TTT is a lab experiment now, not shipped in the engine product -- see
    RUNTIME_SPLIT.md). Exposes exactly the surface the base Substrate._memory handler, _prompt_block_for,
    and the receipts/replay stack read: .rules (active-card texts), .prefix (always None), .memory_strength,
    ._exclude_card_ids (replay sets this for per-card receipts), .consolidate/.reset (no-ops -- prompt mode
    never trains), .state(), .lock."""

    def __init__(self):
        self.prefix = None
        self._exclude_card_ids = None
        self.lock = threading.Lock()
        try:
            from clozn import memory_mode                    # 0.35 == the shipped product default (commit f3e9f60, the
            self.memory_strength = float(memory_mode.get_setting("memory_strength", 0.35))   # over-bleed fix);
        except Exception:                          # matches SelfTeach.__init__ so a fresh engine-first boot
            self.memory_strength = 0.35            # doesn't diverge. (Prompt mode treats it as on/off anyway.)

    @property
    def rules(self):
        from clozn import memory_cards
        return [c["text"] for c in (memory_cards.list_cards() or []) if c.get("status") == "active"]

    @rules.setter
    def rules(self, _value):
        # The card store IS the memory here, so `rules` is derived and has nothing to set. The shared
        # _mem_sync_rules() assigns m.rules for the soft-prefix (SelfTeach) backend; make that a harmless
        # no-op on the engine substrate instead of an AttributeError -- otherwise every approve/reject/
        # disable/enable/remove crashed AFTER already mutating the store (scrappy error toast, action
        # silently succeeded). The store stays the single source of truth.
        pass

    def consolidate(self, rules):
        return {"ok": True, "mode": "prompt"}      # prompt mode never trains a prefix

    def reset(self):
        pass

    def state(self):
        from clozn import memory_cards
        return {"mode": "prompt", "has_prefix": False,
                "cards": len(memory_cards.list_cards() or []), "rules": self.rules}


class EngineSubstrate(Substrate):
    """PURE-ENGINE substrate: chat + prompt-mode memory + tone dials on the C++ GGUF runtime, NO PyTorch
    model resident. THIS is the class that brings the whole torch-free Server tier -- /v1/chat/completions,
    replay, receipts, explain, narrate, counterfactual -- onto the fast engine, because every one of those
    routes through SUB.chat(). Memory is prompt-mode only (the card store as a topic-gated system block);
    dials apply via EngineSteer's steer_vec. See RUNTIME_SPLIT.md (the keystone)."""

    name = "engine"

    def __init__(self):
        if ENGINE_QWEN is None:
            raise RuntimeError("engine substrate needs a running GGUF engine (set CLOZN_ENGINE_QWEN_PORT)")
        self.engine = ENGINE_QWEN
        self.steer = _engine_steer()            # an EngineSteer on the GGUF (tone dials via steer_vec)
        if self.steer is not None:               # metadata-only: the shipped library's names/poles/max, so
            try:                                  # they show up in /steer/axes immediately (their direction
                self.steer.load_library(_pers("studio_library.json"))   # vectors are computed lazily by compute())
                self.steer.load_custom(_pers(f"studio_custom_{self.name}.json"))  # + the user's own custom dials
            except Exception:
                pass
        self._mem = _EngineMemory()
        self.memory = self._mem                 # the studio reads SUB.memory in a few places
        self._pers_steer = _pers("studio_personality.json")
        self._steer_ready = False
        self._steer_info = {}
        self._steer_lock = threading.Lock()
        self.brain = None                       # no SAE/brain on the pure-engine substrate (concepts 409 cleanly)
        # T0.2: reflect the ACTUALLY-LOADED GGUF, not a hardcoded Qwen assumption. Derive the family from
        # the engine's /health model file (best-effort -- never blocks boot if the engine isn't up yet)
        # and pin the tone-dial steer tap to THIS model's mid-depth: Qwen-7B -> 14 (unchanged), Llama-3.2-1B
        # -> 8, an unrecognized GGUF keeps EngineSteer's generic default. run_meta() re-derives this lazily
        # too, so the run record is correct even when the engine comes up after the substrate.
        self.model_family = None
        self.model_id = None
        try:
            h = self.engine.health() if (self.engine and hasattr(self.engine, "health")) else {}
            self.model_family, _info = _engine_model_info((h or {}).get("model", ""))
            self.model_id = _info["model_id"]
            if self.steer is not None and _info["steer_layer"] is not None:
                self.steer.layer = _info["steer_layer"]
        except Exception:
            pass
        if self.steer is not None:              # restore persisted dial values (shared personality.json)
            try:
                self.steer.load_state(self._pers_steer)
            except Exception:
                pass

    def _gen(self, prompt):                     # one-shot generate for the /steer/check A/B (base _steer)
        if self.steer is not None:
            return self.steer.generate(prompt, max_new=90)
        from clozn.steering import EngineSteer
        return EngineSteer._text(self.engine.complete(prompt, max_tokens=90))

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        """One stateless chat completion on the engine with memory (prompt-mode card block) + tone dials
        applied. Mirrors QwenSubstrate.chat's contract EXACTLY (same signature, same trace_out/mem_out
        fill) so the receipts/replay stack is backend-agnostic. `sample` is accepted for signature parity;
        the engine trace path decodes greedily (temperature 0) so a receipt diff is attributable, matching
        how the stack already drives QwenSubstrate for receipts."""
        self._last_generation_meta = _engine_generation_meta(max_new, stream=False)
        # MEMORY: the active cards as a topic-gated system block (omitted off-topic / when strength 0).
        block, applied, gate = _prompt_block_for(self.memory, _last_user(messages))
        assembled = _inject_block(messages, block)
        prompt = _engine_tmpl(self.engine, assembled)   # per-model template (the loaded GGUF's own), not Qwen ChatML
        if mem_out is not None:
            # final_prompt = the EXACT rendered string the model saw (backlog #5); assembled_messages is its
            # pre-template form. Both recorded so the run is inspectable at either level.
            mem_out.update(mode="prompt", applied=applied, gate=gate,
                           prompt_block=block, assembled_messages=assembled, final_prompt=prompt)
        # TONE: dials from self.steer.strength (replay toggles this in place), falling back to disk.
        kw = {}
        st = (getattr(self.steer, "strength", None) if self.steer is not None else None) or _disk_dials()
        if self.steer is not None and st and any(st.values()):
            sv = self.steer.steer_vector(st)
            if sv:
                kw["steer_vec"] = sv
                kw["steer"] = {"coef": 1.0, "layer": self.steer.layer}
        reply_raw, steps, finish = _engine_complete_traced(self.engine, prompt, max_new, kw)
        self._last_finish_reason = finish                   # stash for last_finish_reason() (the log path)
        if trace_out is not None:
            trace_out.extend(steps)
        return reply_raw.strip()

    def score_tokens(self, messages, continuation_ids=None, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        """Teacher-forced per-token logprob of a continuation under EXPLICIT (block,
        steer_strengths) conditions -- the S1 seam notes/REPRODUCE_AND_PROVE_PLAN.md's forced-scoring
        stack (rederive.py, forced receipts) builds on. Assembles the prompt EXACTLY like chat()
        (_inject_block + _engine_tmpl -- the loaded model's own chat template) and the steer_vec EXACTLY
        like chat() (self.steer.steer_vector),
        but from the CALLER's `block`/`steer_strengths` -- NEVER from live self.memory/self.steer.strength
        -- so a with/without arm is reconstructed purely from a run record (memory ablation = recompile
        the block without a card; dial ablation = zero a strength and recompute) rather than from
        whatever the live substrate happens to be doing right now. That's what makes receipt arms
        reconstructable: two calls with different explicit `block`/`steer_strengths`, same messages
        and continuation_ids, are directly comparable. No sampling anywhere; deterministic.

        `block`: a prompt-mode memory block string (or None to omit it), e.g. run.memory.prompt_block.
        `steer_strengths`: a {dial_name: strength} dict (or None for no steer), e.g. run.behavior.dials.
        `continuation_ids`: the PRIMARY continuation form (token ids, e.g. from a stored trace) --
        takes precedence over `continuation` when both are given (mirrors EngineClient.score).
        `continuation`: a TEXT fallback (S3's rederive.py, for a run whose trace lacks per-token ids) --
        the engine retokenizes it independently of the prompt, which can drift at the prompt/
        continuation BPE boundary (flagged `boundary_approximate` by /score itself; see
        REPRODUCE_AND_PROVE_PLAN.md's tokenization-boundary caveat).
        `steer_vec`: an explicit RAW steer direction, ADDED on top of whatever `steer_strengths`
        produces (or used alone if `steer_strengths` is falsy) -- the S3 null-floor control needs a
        direction with no named dial behind it ("a random vector of equal norm at the same layer").

        Returns [{"id", "piece", "logprob"}, ...] (+ "topk" per token when topk>0), one entry per
        continuation token, in the SAME order as continuation_ids (or the engine's own retokenization
        of `continuation` text).
        """
        assembled = _inject_block(messages, block)
        prompt = _engine_tmpl(self.engine, assembled)   # per-model template (the loaded GGUF's own), not Qwen ChatML
        kw = {}
        sv = None
        if self.steer is not None and steer_strengths and any(steer_strengths.values()):
            sv = self.steer.steer_vector(steer_strengths)
        if steer_vec is not None:
            sv = [a + b for a, b in zip(sv, steer_vec)] if sv else list(steer_vec)
        if sv:
            kw["steer_vec"] = sv
            # self.steer.layer is model-aware (pinned per-family in __init__); with no steer built, pass
            # layer 0 so the ENGINE picks its own calibrated mid-depth band -- not a hardcoded Qwen 14.
            kw["steer"] = {"coef": 1.0, "layer": self.steer.layer if self.steer is not None else 0}
        if continuation_ids is not None:
            kw["continuation_ids"] = [int(t) for t in continuation_ids]
        elif continuation is not None:
            kw["continuation"] = str(continuation)
        r = self.engine.score(prompt=prompt, topk=int(topk), **kw)
        return r.get("tokens", [])

    def last_stream_trace(self):
        """The per-token trace captured during the most recent chat_stream (raw step list, or []) --
        same contract as QwenSubstrate.last_stream_trace: the SSE handler reads this AFTER the generator
        is exhausted, to log the run's Run Inspector timeline."""
        return list(getattr(self, "_last_stream_trace", []) or [])

    def last_finish_reason(self):
        """The stop cause ("stop"|"length"|...) from the most recent chat()/chat_stream, or None. Same
        stash-and-read contract as last_stream_trace: the handler reads it AFTER generation, so the run
        logs WHY the engine stopped instead of a hard-coded 'stop'."""
        return getattr(self, "_last_finish_reason", None)

    def run_meta(self):
        """Reproducibility metadata -- WHAT produced a run -- for the run record. Fetched once from
        /health (model file -> quant, engine mode) and cached; the sampling regime is greedy (chat and
        chat_stream both force temperature 0). Health-derived fields are omitted when unavailable rather
        than guessed. Never raises: metadata never breaks a run."""
        health_meta = getattr(self, "_run_meta", None)
        if health_meta is None:
            health_meta = {}
            try:
                h = self.engine.health() if (self.engine and hasattr(self.engine, "health")) else {}
                mp = str((h or {}).get("model", ""))
                if mp:
                    health_meta["model_file"] = mp.replace("\\", "/").rsplit("/", 1)[-1]
                    q = _quant_from_name(health_meta["model_file"])
                    if q:
                        health_meta["quant"] = q
                    # T0.2: which model actually produced this run (derived from the loaded GGUF, not a
                    # hardcoded id). family is the registry key; model_id the friendly HF name when known.
                    fam, info = _engine_model_info(mp)
                    if fam:
                        health_meta["family"] = fam
                    if info.get("model_id"):
                        health_meta["model_id"] = info["model_id"]
                if (h or {}).get("mode"):
                    health_meta["mode"] = h["mode"]
                for k in ("n_ctx", "device", "gpu_layers"):
                    v = (h or {}).get(k)
                    if v is not None:
                        health_meta[k] = v
            except Exception:
                pass
            self._run_meta = dict(health_meta)
        meta = _engine_generation_meta()
        meta.update(dict(health_meta))
        meta.update(getattr(self, "_last_generation_meta", None) or {})
        return dict(meta)

    def chat_stream(self, messages, max_new=256, mem_out=None):
        """Streaming twin of chat(): the SAME memory-block + tone-dial construction (kept in lockstep --
        see chat()'s comments; do not let this drift from that logic), but opens the engine's
        /v1/completions with stream:True (mirrors _engine_complete_traced's request) and yields text as
        the engine commits it, instead of waiting on one blocking call. This is what makes /v1/chat/
        completions's SSE branch (_sse_chat, gated on `getattr(SUB, "chat_stream", None)`) fire on the
        pure-engine substrate too -- before this existed, a streaming request here silently fell through
        to one non-streamed chat() reply. mem_out: as in chat() -- prompt mode records what memory
        actually rode this turn.

        Per-token trace (mirrors QwenSubstrate.chat_stream's B3 contract): every parsed SSE frame is
        collected, then folded into self._last_stream_trace via runlog.accumulate_ar_events once the
        stream ends -- normal completion OR an early GeneratorExit (the consumer stopped early) -- so a
        partial stream still logs whatever trace it managed. Wrapped so any parse hiccup just leaves it
        [], same as the non-streaming path's fallback."""
        import urllib.request
        from clozn import runlog
        self._last_generation_meta = _engine_generation_meta(max_new, stream=True)
        # MEMORY + TONE: built EXACTLY as chat() builds them.
        block, applied, gate = _prompt_block_for(self.memory, _last_user(messages))
        assembled = _inject_block(messages, block)
        prompt = _engine_tmpl(self.engine, assembled)   # per-model template (the loaded GGUF's own), not Qwen ChatML
        if mem_out is not None:
            # final_prompt = the EXACT rendered string the model saw (backlog #5); kept in lockstep with chat().
            mem_out.update(mode="prompt", applied=applied, gate=gate,
                           prompt_block=block, assembled_messages=assembled, final_prompt=prompt)
        kw = {}
        st = (getattr(self.steer, "strength", None) if self.steer is not None else None) or _disk_dials()
        if self.steer is not None and st and any(st.values()):
            sv = self.steer.steer_vector(st)
            if sv:
                kw["steer_vec"] = sv
                kw["steer"] = {"coef": 1.0, "layer": self.steer.layer}
        body = dict(kw); body["prompt"] = prompt; body["max_tokens"] = int(max_new)
        body["temperature"] = 0.0; body["rep_penalty"] = 1.0; body["seed"] = 0; body["stream"] = True
        req = urllib.request.Request(self.engine.base + "/v1/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        self._last_stream_trace = []        # reset; reassembled in `finally` below (empty on any hiccup)
        frames = []
        resp = urllib.request.urlopen(req, timeout=getattr(self.engine, "timeout", 600))
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                frames.append(obj)
                if obj.get("type") == "tokens_committed":
                    for it in obj.get("items") or []:
                        piece = it.get("piece", "")
                        if piece:
                            yield piece
        finally:
            # ALWAYS release the engine connection -- whether the stream ran to [DONE] or the consumer
            # stopped early (this `finally` also runs when the caller .close()s us mid-stream, via a
            # GeneratorExit at the `yield` above); guarded so a close() hiccup can never mask a
            # propagating GeneratorExit -- it must reach the caller, never be swallowed here. (The
            # engine-side crash-on-disconnect is a separate C++-side task; this just closes cleanly.)
            try:
                resp.close()
            except Exception:
                pass
            try:
                self._last_stream_trace = runlog.accumulate_ar_events(frames)
            except Exception:
                self._last_stream_trace = []
            try:
                self._last_finish_reason = runlog.finish_reason_from_frames(frames)
            except Exception:
                self._last_finish_reason = None

    def handle(self, path, body):
        r = self._memory(path, body)
        if r is not None:
            return r
        return self._steer(path, body)

    def state(self):
        from clozn import memory_cards
        return {"dials": dict(getattr(self.steer, "strength", {}) or {}),
                "cards": len(memory_cards.list_cards() or [])}


def load_substrate(name):
    if name == "engine":
        try:
            return EngineSubstrate()
        except Exception as e:
            print(f"[substrate] engine substrate unavailable ({e}); running with SUB=None", file=sys.stderr)
            return None
    return QwenSubstrate() if name == "qwen" else DreamSubstrate()


def switch_substrate(name):
    """Re-exec the whole process with the new substrate -> a clean GPU (the only honest way; one 7B fits)."""
    py = sys.executable
    os.execv(py, [py, os.path.abspath(__file__), "--substrate", name, "--port", str(ARGS.port),
                  "--host", ARGS.host])


def _quant_from_name(name):
    """Pull the GGUF quant tag (Q4_K_M, Q8_0, IQ4_XS, F16, ...) out of a model filename, or None. GGUF
    files name their quantization in the basename, so this is the one bit of repro metadata we can read
    for free (no engine change) off /health's model path."""
    import re
    m = re.search(r"(IQ\d+[A-Z0-9_]*|Q\d+(?:_[A-Z0-9]+)+|Q\d+|BF16|F16|F32)", str(name), re.IGNORECASE)
    return m.group(1).upper() if m else None


# --- engine model registry (T0.2) ---------------------------------------------------------------------
# The engine substrate reflects the ACTUALLY-LOADED GGUF, not a hardcoded "Qwen2.5-7B" id/assumption.
# The ONE Qwen-specific assumption the engine substrate carried was the tone-dial steer TAP LAYER
# (mid-depth: 14 for Qwen-7B's 28 layers). This tiny registry keys that -- plus a friendly model_id for
# run_meta -- off the loaded model's family (derived from its /health filename), with a sensible default
# for any unrecognized GGUF (steer_layer None => don't pin a layer; let the engine use its OWN per-model
# calibrated mid-depth steer band). Everything else the engine already calibrates per-model server-side
# (the C++ concept/steer probe taps at startup, and the chat template via /apply_template). This is NOT a
# framework -- it is the minimal table that removes the last hardcoded-Qwen coupling from the engine path.
_ENGINE_MODELS = {
    "qwen2.5-7b":   {"model_id": "Qwen/Qwen2.5-7B-Instruct",         "steer_layer": 14},  # 28L -> mid 14 (unchanged)
    "qwen2.5-0.5b": {"model_id": "Qwen/Qwen2.5-0.5B-Instruct",       "steer_layer": 12},  # 24L -> mid 12
    "llama-3.2-1b": {"model_id": "meta-llama/Llama-3.2-1B-Instruct", "steer_layer": 8},   # 16L -> mid 8
    "llama-3.2-3b": {"model_id": "meta-llama/Llama-3.2-3B-Instruct", "steer_layer": 14},  # 28L -> mid 14
}
_ENGINE_MODEL_DEFAULT = {"model_id": None, "steer_layer": None}  # unknown GGUF: nothing pinned; engine picks


def _model_family_from_name(name):
    """Coarse model family key ('qwen2.5-7b', 'llama-3.2-1b', ...) from a GGUF filename, or None -- the
    engine substrate looks up per-model assumptions in _ENGINE_MODELS by this key instead of hardcoding
    Qwen's. Same free derive-off-/health-filename trick as _quant_from_name (no engine change needed)."""
    import re
    s = str(name or "").lower()
    m = re.search(r"qwen[._]?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)b", s)
    if m:
        return f"qwen{m.group(1)}-{m.group(2)}b"
    m = re.search(r"llama[._-]?(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)b", s)
    if m:
        return f"llama-{m.group(1)}-{m.group(2)}b"
    return None


def _engine_model_info(name):
    """(family, {model_id, steer_layer}) for the loaded GGUF -- the engine substrate's per-model
    assumptions -- or (None, the default with nothing pinned) for an unrecognized model."""
    fam = _model_family_from_name(name)
    return fam, dict(_ENGINE_MODELS.get(fam, _ENGINE_MODEL_DEFAULT))


def _engine_complete_traced(engine, prompt, max_tokens, kw):
    """Generate on the engine and ALSO capture a per-token trace (issue B3), returning (reply, steps, finish).

    The engine's non-streaming /v1/completions carries only the final text -- no per-token confidence. To
    populate the Run Inspector timeline we ask the SAME request with stream:True and fold its per-token
    `tokens_committed`/`step_lens` frames into steps via runlog.accumulate_ar_events. Generation is greedy
    (temperature 0), so the reassembled text is identical to the blocking call -- we only capture ALONGSIDE;
    the client still receives the same single JSON reply (this streams engine<->server, never to the client).
    Any streaming hiccup falls back to the plain complete() so a run is never lost -- just without a trace.
    (AR GGUFs only; a diffusion engine commits out of reading order and emits no such per-token stream.)
    """
    import urllib.request
    body = dict(kw); body["prompt"] = prompt; body["max_tokens"] = int(max_tokens)
    body["temperature"] = 0.0; body["rep_penalty"] = 1.0; body["seed"] = 0; body["stream"] = True
    try:
        req = urllib.request.Request(engine.base + "/v1/completions",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        frames, text = [], ""
        with urllib.request.urlopen(req, timeout=getattr(engine, "timeout", 600)) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                frames.append(obj)
                ch = obj.get("choices")                     # the final OpenAI-style frame carries the full text
                if ch and isinstance(ch, list) and ch[0].get("text"):
                    text = ch[0]["text"]
        from clozn import runlog
        steps = runlog.accumulate_ar_events(frames)
        finish = runlog.finish_reason_from_frames(frames)   # the engine's real stop cause (else None)
        if not text:                                        # no final frame text -> reassemble from the pieces
            text = "".join(s.get("piece", "") for s in steps)
        if steps or text:
            return text, steps, finish
    except Exception:
        pass
    # Fallback: the original blocking path, reply preserved, trace simply empty.
    r = engine.complete(prompt, max_tokens=max_tokens, temperature=0.0, rep_penalty=1.0, seed=0, **kw)
    ch = r.get("choices") if isinstance(r, dict) else None
    finish = ch[0].get("finish_reason") if (ch and isinstance(ch[0], dict)) else None
    return (ch[0].get("text", "") if ch else str(r)), [], finish


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

        def _html(self, name):
            self._send(200, open(os.path.join(DEMO, name), encoding="utf-8").read(), "text/html; charset=utf-8")

        def _sse_chat(self, messages, max_new, model):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            def chunk(delta, finish=None):
                o = {"id": "chatcmpl-clozn", "object": "chat.completion.chunk", "model": model,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(("data: " + json.dumps(o) + "\n\n").encode("utf-8"))
                self.wfile.flush()

            # HF chat stream (QwenSubstrate.chat_stream): a pure pass-through recorder rides along and the
            # per-token trace is assembled after the stream (SUB.last_stream_trace()) -- so the run gets the
            # Run Inspector timeline while the streamed chunks stay byte-identical (B3). runlog.record
            # normalizes the raw step list; on any hiccup last_stream_trace() is [] -> a clean empty trace.
            # memout: prompt mode fills what memory ACTUALLY rode this turn (block gated in/out) for the log.
            #
            # M5 any-client run_id bridge (EXPLAIN_THIS_ANSWER_SPEC.md): DEFERRED here, deliberately. Headers
            # are already flushed above (self.end_headers(), before a single token is generated), so
            # X-Clozn-Run-Id can never ride a header on this path -- the id only exists after _log_run runs,
            # which is after the stream ends. A trailing SSE frame isn't clean either: a frame AFTER
            # "data: [DONE]" is silently dropped by clients (incl. openai-python) that stop reading at the
            # [DONE] sentinel, and a frame BEFORE [DONE] would need a full spec-shaped chat.completion.chunk
            # (id/object/created/model/choices) just to smuggle one field, plus a stray chunk after the real
            # finish_reason:"stop" chunk -- exactly the shape drift the honesty/compat contract rules out.
            # Left unchanged; non-streaming (the required deliverable) carries clozn_run_id both ways.
            t0 = time.time(); acc = []; memout = {}
            try:
                chunk({"role": "assistant"})
                for piece in SUB.chat_stream(messages, max_new, mem_out=memout):
                    acc.append(piece); chunk({"content": piece})
                fr = SUB.last_finish_reason() if hasattr(SUB, "last_finish_reason") else None
                openai_fr = _openai_finish_reason(fr)
                chunk({}, finish=openai_fr)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                trace = SUB.last_stream_trace() if hasattr(SUB, "last_stream_trace") else None
                self._log_run("openai_api", messages, "".join(acc), model, t0, trace=trace, mem_out=memout,
                              finish_reason=fr,
                              finish_reason_fallback=None if fr else openai_fr)
            except Exception as e:
                self._log_run("openai_api", messages, "".join(acc), model, t0, error=str(e), mem_out=memout)
                try:
                    self.wfile.write(("data: " + json.dumps({"error": str(e)}) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass

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
                from clozn import workspace_lens
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
                from clozn import runlog
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
                    if applied:                                  # bump exactly the cards that rode this turn
                        try:
                            from clozn import memory_cards
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
                            from clozn import memory_cards
                            for c in memory_cards.list_cards(status="active"):
                                memory_cards.bump_usage(c["id"])
                        except Exception:
                            pass
                else:
                    memd = {"mode": mode}                        # runlog records the mode on EVERY run
                # FACTS tier (NEXT_STEPS #5): only when memory_facts is on -- otherwise zero cost, the
                # latency rule. A chat turn (not the pure /think etc.) gets a surprise-gated AUTO-WRITE
                # (mine one declarative fact; the gate refuses what the model already knows) and a READ
                # RECEIPT (what the store would fire + slot_ms), both folded into the run's memory record
                # so the Run Inspector shows them. Fully guarded; never breaks logging or the reply.
                try:
                    from clozn import facts_mode
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
                    from clozn import capture_mode
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
                from clozn import timetravel
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

        def _facts(self, p, body):
            """The FACTS tier (slot memory) endpoints. Every one degrades cleanly when the tier is off
            (memory_facts) or no substrate/model is loaded -- the panel renders either way. The read/write
            operations touch the shared 7B, so they run under the SlotBox's own lock (+ _TRAIN_LOCK inside
            it) and are gated OFF the chat hot path by the setting -- the latency rule."""
            from clozn import facts_mode
            if p == "/facts/mode":            # read or set the on/off gate (the latency switch)
                if "enabled" in body:
                    on = bool(body.get("enabled"))
                    if not facts_mode.set_enabled(on):
                        return self._json(200, {"ok": False, "reason": "could not persist the setting"})
                    return self._json(200, {"ok": True, "enabled": facts_mode.enabled(),
                                            "layer": facts_mode.LAYER})
                box = _slots_box()
                st = box.status() if box is not None else {"enabled": facts_mode.enabled(),
                                                            "layer": facts_mode.LAYER,
                                                            "profile": _active_profile_name() or "default",
                                                            "count": 0}
                return self._json(200, st)
            box = _slots_box()
            if box is None:                   # no substrate at all yet -> honest empty
                return self._json(200, {"enabled": facts_mode.enabled(), "entries": [], "count": 0,
                                        "note": "no substrate loaded"})
            if p == "/facts/list":            # the store's entries (cue/answer) -- read-only, no forward
                return self._json(200, {"enabled": facts_mode.enabled(), "entries": box.list_entries(),
                                        **box.status()})
            if p == "/facts/add":             # explicit gated write (the gate refusal is the receipt)
                return self._json(200, box.add(str(body.get("cue", "")), str(body.get("answer", "")),
                                               gate=bool(body.get("gate", True))))
            if p == "/facts/delete":          # surgical per-entry removal (bystanders bit-identical)
                return self._json(200, box.delete(cue=body.get("cue"), index=body.get("index")))
            if p == "/facts/read":            # the honest read receipt (hit / gate value / abstention + slot_ms)
                return self._json(200, box.read_receipt(str(body.get("query", ""))))
            return self._json(404, {"error": f"POST {p}"})

        def _timetravel(self, p, body):
            """The time-travel debugger's gate + ring config + store stats (NEXT_STEPS #6). The snapshot
            ring holds KV state in CPU RAM, so it is behind ONE persisted setting (`timetravel_snapshots`,
            DEFAULT OFF -- the RAM rule); this endpoint reads/sets it, tunes the ring (cap / byte budget),
            and reports the honest offloaded-bytes total. Branch RECORDING (POST /runs/<id>/branch) does
            NOT depend on the gate; only holding live KV for the (future) re-prefill fast path does."""
            from clozn import timetravel
            if p == "/timetravel/mode":       # read or set the on/off gate + ring config
                changed = False
                if "enabled" in body:
                    timetravel.set_enabled(bool(body.get("enabled")))
                    changed = True
                if "cap" in body or "budget_mb" in body:
                    timetravel.set_config(cap=body.get("cap"), budget_mb=body.get("budget_mb"))
                    changed = True
                    cfg = timetravel.get_config()          # apply the (clamped) new ceilings to the LIVE store
                    if _snap_store() is not None:
                        _snap_store().reconfigure(cap=cfg["cap"], budget_mb=cfg["budget_mb"])
                out = {"enabled": timetravel.enabled(), **timetravel.get_config()}
                store = _snap_store()
                if store is not None:
                    out["store"] = store.stats()
                out["changed"] = changed
                return self._json(200, out)
            if p == "/timetravel/stats":      # just the store's honest memory receipt
                store = _snap_store()
                return self._json(200, {"enabled": timetravel.enabled(),
                                        **(store.stats() if store is not None else {})})
            return self._json(404, {"error": f"POST {p}"})

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/", "/index.html", "/instrument.html"):
                return self._html("instrument.html")
            if p == "/substrate":
                return self._json(200, {"active": SUBNAME, "available": ["qwen", "dream"]})
            if p == "/v1/models":            # OpenAI-compatible model list (so OAI clients connect)
                return self._json(200, {"object": "list", "data": [
                    {"id": "clozn-qwen", "object": "model", "owned_by": "clozn"}]})
            if p == "/engine/health":
                try:
                    return self._json(200, {"engine": ENGINE.health()})
                except Exception as e:
                    return self._json(502, {"error": f"engine unreachable: {e}"})
            if p == "/state":
                return self._json(200, {"substrate": SUBNAME, "memory_mode": _memory_mode(),
                                        **(SUB.state() if SUB else {})})
            if p == "/memory/mode":          # which mechanism carries the cards (works on ANY substrate)
                from clozn import memory_mode
                return self._json(200, {"mode": _memory_mode(), "modes": list(memory_mode.MODES)})
            if p == "/capture/tier":         # how much white-box data each run stores (light/standard/deep/lab)
                from clozn import capture_mode
                return self._json(200, {"tier": capture_mode.tier(), "tiers": list(capture_mode.TIERS)})
            if p == "/timetravel/mode":      # #6: is per-turn KV snapshotting on? + ring config + store stats
                from clozn import timetravel
                out = {"enabled": timetravel.enabled(), **timetravel.get_config()}
                store = _snap_store()
                if store is not None:
                    out["store"] = store.stats()
                return self._json(200, out)
            if p == "/profiles/list":         # every saved persona bundle + which one is active (masthead + Settings)
                from clozn import profiles
                return self._json(200, {"profiles": profiles.ProfileStore().list(),
                                        "active": _active_profile_name()})
            if p.startswith("/memory/") and p.endswith("/runs"):   # E1: which runs used this card
                cid = p[len("/memory/"):-len("/runs")]
                return self._json(200, {"card_id": cid, "runs": _runs_for_card(cid)})
            if p == "/runs":                 # the Run Log -- every interaction, newest first (the Studio Runs page)
                from clozn import runlog
                return self._json(200, {"runs": runlog.list_runs(80)})
            if p.startswith("/runs/") and p.endswith("/export"):   # one-call download: run + M1 explain + trace
                from clozn import runlog
                from clozn import explain
                from clozn import receipt_bundle
                from urllib.parse import urlparse, parse_qs
                rid = p[len("/runs/"):-len("/export")]
                run = runlog.get_run(rid)
                if not run:
                    return self._json(404, {"error": "run not found"})
                try:
                    xr = explain.explain(run)                # M1: pure read/reshape, no generation
                except Exception:
                    xr = None
                bundle = receipt_bundle.build(run, explain=xr)
                fmt = (parse_qs(urlparse(self.path).query).get("format") or ["json"])[0]
                if fmt == "md":
                    return self._send(200, receipt_bundle.to_markdown(bundle), "text/markdown; charset=utf-8",
                                      extra_headers={"Content-Disposition": 'attachment; filename="' + rid + '.md"'})
                return self._json(200, bundle,
                                  extra_headers={"Content-Disposition": 'attachment; filename="' + rid + '.json"'})
            if p.startswith("/runs/") and p.endswith("/timeline"):   # ordered RunEvent list -- zero generation
                from clozn import runlog
                rid = p[len("/runs/"):-len("/timeline")]
                run = runlog.get_run(rid)
                if not run:
                    return self._json(404, {"error": "run not found"})
                from clozn import run_timeline
                return self._json(200, {"run_id": rid, "events": run_timeline.timeline(run)})
            if p.startswith("/runs/") and p.endswith("/lineage"):   # branch/replay ancestry + child tree
                from clozn import runlog
                rid = p[len("/runs/"):-len("/lineage")]
                out = runlog.lineage(rid)
                if not out:
                    return self._json(404, {"error": "run not found"})
                return self._json(200, out)
            if p.startswith("/runs/") and p.endswith("/family"):   # the WHOLE branch family as GET /runs-shaped
                # summaries -- the full lineage past the /runs 80-window, so the client's buildLineageFromRuns
                # builds the complete tree instead of the recent-runs slice. (Distinct from /lineage, which
                # returns the server-built ancestors/children/tree object.)
                from clozn import runlog
                rid = p[len("/runs/"):-len("/family")]
                fam = runlog.lineage_family(rid)
                if fam is None:
                    return self._json(404, {"error": "run not found"})
                return self._json(200, {"runs": fam})
            if p.startswith("/runs/") and p.endswith("/spans"):   # confidence spans -- the shape of the reply's certainty
                from clozn import runlog
                rid = p[len("/runs/"):-len("/spans")]
                run = runlog.get_run(rid)
                if not run:
                    return self._json(404, {"error": "run not found"})
                from clozn import confidence_spans
                sp = confidence_spans.spans(run)
                return self._json(200, {"run_id": rid, "spans": sp, "summary": confidence_spans.summarize(sp)})
            if p.startswith("/runs/"):
                from clozn import runlog
                r = runlog.get_run(p.split("/runs/", 1)[1])
                return self._json(200, r) if r else self._json(404, {"error": "run not found"})
            if p.endswith((".html", ".css", ".js")):
                fn = os.path.normpath(os.path.join(DEMO, p.lstrip("/")))   # serve subdirs (pages/) too, safely
                if fn.startswith(os.path.normpath(DEMO)) and os.path.isfile(fn):
                    ct = ("text/html" if p.endswith(".html") else
                          "text/css" if p.endswith(".css") else "application/javascript")
                    return self._send(200, open(fn, encoding="utf-8").read(), ct + "; charset=utf-8")
            self._json(404, {"error": "GET " + p})

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            p = self.path.split("?")[0].rstrip("/") or "/"
            if p == "/substrate":
                name = str(body.get("name", "qwen"))
                if name == SUBNAME:
                    return self._json(200, {"active": SUBNAME, "switched": False})
                if name not in ("qwen", "dream"):
                    return self._json(400, {"error": "unknown substrate"})
                self._json(200, {"active": name, "switched": True, "note": "reloading -- poll /substrate"})
                threading.Thread(target=lambda: (time.sleep(0.4), switch_substrate(name)), daemon=True).start()
                return
            if p == "/feedback":   # preference-signal CAPTURE (the plumbing) -- log a directional signal
                # (e.g. a Run Inspector "Too verbose" click) tied to the run that prompted it, so a later
                # accumulate-and-propose step can mine "you keep asking for concise". Records only; changes
                # nothing (agency-agnostic), and never fails the user's action over a feedback write.
                from clozn import feedback
                sig = feedback.record(body.get("run_id"), str(body.get("kind", "quick_repair")),
                                      dial=body.get("dial"), direction=body.get("direction"),
                                      meta=body.get("meta"))
                return self._json(200, {"ok": True, "signal": sig})
            if p == "/feedback/summary":   # the rollup a learning step reads: per-(dial,direction) counts +
                from clozn import feedback            # the last run driving each, over an optional recent window (days)
                wd = body.get("window_days")
                ws = float(wd) * 86400 if isinstance(wd, (int, float)) else None
                return self._json(200, feedback.summary(window_seconds=ws))
            if p == "/preferences":   # propose-and-review: fold the feedback pattern into proposals + return
                from clozn import feedback, preferences   # the PENDING ones (a (dial,direction) lean that crossed the
                sigs = feedback.list_signals()  # threshold). refresh() creates/updates; it NEVER sets a dial.
                wd = body.get("window_days")
                if isinstance(wd, (int, float)):
                    cut = time.time() - float(wd) * 86400
                    sigs = [s for s in sigs if float(s.get("ts", 0)) >= cut]
                pending = preferences.refresh(
                    sigs, threshold=int(body.get("threshold", preferences.DEFAULT_THRESHOLD)))
                return self._json(200, {"pending": pending})
            if p == "/preferences/resolve":   # APPROVE (persist the dial) or DISMISS a proposal -- the review
                from clozn import preferences            # half of propose-and-review. Approve is the ONLY place a
                pr = preferences.resolve(str(body.get("id", "")), str(body.get("action", "")))  # dial is set.
                if pr is None:
                    return self._json(400, {"error": "unknown proposal id, or action not in {approve,dismiss}"})
                applied = None
                if pr["status"] == "approved" and SUB is not None and getattr(SUB, "steer", None) is not None:
                    try:                       # persist the dial exactly like the F2 save-fix does (steer.set
                        SUB.steer.set(pr["dial"], float(pr["suggested_value"]))   # caps per-axis)
                        if hasattr(SUB.steer, "save_state") and getattr(SUB, "_pers_steer", None):
                            SUB.steer.save_state(SUB._pers_steer)
                        applied = {"dial": pr["dial"],
                                   "value": float(SUB.steer.strength.get(pr["dial"], pr["suggested_value"]))}
                    except Exception as e:
                        applied = {"error": f"{type(e).__name__}: {e}"}
                return self._json(200, {"ok": True, "proposal": pr, "applied": applied})
            if p == "/capture/tier":  # set the capture tier (persisted; applies to subsequent runs)
                from clozn import capture_mode
                name = str(body.get("tier", "")).strip().lower()
                if name not in capture_mode.TIERS:
                    return self._json(400, {"error": f"unknown tier (want one of {list(capture_mode.TIERS)})"})
                if not capture_mode.set_tier(name):
                    return self._json(200, {"ok": False, "reason": "could not persist the tier setting"})
                return self._json(200, {"ok": True, "tier": name})
            if p == "/memory/mode":   # swap the memory mechanism (persisted; takes effect immediately)
                from clozn import memory_mode
                mode = str(body.get("mode", "")).strip().lower()
                if mode not in memory_mode.MODES:
                    return self._json(400, {"error": f"unknown mode (want one of {list(memory_mode.MODES)})"})
                if not memory_mode.set_mode(mode):
                    return self._json(200, {"ok": False, "reason": "could not persist the mode setting"})
                out = {"ok": True, "mode": mode}
                # Toggling BACK to internalized: prompt-mode card edits never consolidated, so the trained
                # prefix can be STALE relative to the cards. If the active set differs from what the
                # current prefix embodies (_trained_rules), kick the normal background retrain so chats
                # don't serve a personality the cards no longer describe. Cheap guards: nothing to do when
                # there's no live memory, or when cards and prefix are both empty.
                if mode == "internalized" and SUB is not None and getattr(SUB, "_mem", None) is not None:
                    m = SUB._mem
                    try:
                        from clozn import memory_cards
                        active = memory_cards.active_texts()
                        trained = list(getattr(m, "_trained_rules", []) or [])
                        if set(active) != set(trained) and (active or getattr(m, "prefix", None) is not None):
                            out["resync"] = _start_retrain(m, "mode-switch", None, force=True)
                    except Exception:
                        pass
                return self._json(200, out)
            if p == "/profiles/save":        # create/update a named persona bundle (does NOT apply it -- see switch)
                from clozn import profiles
                try:
                    saved = profiles.ProfileStore().save(profiles.validate(dict(body)))
                except (ValueError, KeyError, TypeError) as e:
                    return self._json(400, {"error": f"bad profile: {e}"})
                return self._json(200, {"ok": True, "path": saved, "profile": profiles.ProfileStore().load(body["name"])})
            if p == "/profiles/switch":      # THE persona switch: cards replace, dials replace, instant in prompt mode
                from clozn import profiles
                name = str(body.get("name", "")).strip()
                if not name:
                    return self._json(400, {"error": "need a profile name"})
                try:
                    prof = profiles.ProfileStore().load(name)
                except (OSError, ValueError) as e:
                    return self._json(404, {"error": f"no such profile '{name}': {e}"})
                if SUB is None:
                    return self._json(503, {"error": "no substrate loaded"})
                return self._json(200, {"ok": True, **_profiles_switch(SUB, prof)})
            if p == "/profiles/export":       # -> the bundle's own JSON (client downloads/saves it -- the portable artifact)
                from clozn import profiles
                name = str(body.get("name", "")).strip()
                if not name:
                    return self._json(400, {"error": "need a profile name"})
                try:
                    return self._json(200, {"ok": True, "profile": profiles.ProfileStore().load(name)})
                except (OSError, ValueError) as e:
                    return self._json(404, {"error": f"no such profile '{name}': {e}"})
            if p == "/profiles/import":       # body IS the bundle JSON (as exported); optional {rename}
                from clozn import profiles
                try:
                    bundle = dict(body.get("profile", body))
                    rename = body.get("rename") or None
                    p2 = profiles.validate(bundle)
                    if rename:
                        p2["name"] = rename
                        p2 = profiles.validate(p2)
                    path = profiles.ProfileStore().save(p2)
                except (ValueError, KeyError, TypeError) as e:
                    return self._json(400, {"error": f"bad profile bundle: {e}"})
                return self._json(200, {"ok": True, "path": path, "profile": p2})
            if p.startswith("/facts/"):      # the FACTS tier (slot memory) -- gated behind memory_facts (default OFF)
                return self._facts(p, body)
            if p.startswith("/timetravel/"):   # #6: the time-travel snapshot gate + ring config (default OFF)
                return self._timetravel(p, body)
            if p.startswith("/runs/") and p.endswith("/replay"):   # F1: re-run a past run under changed state -> a child run
                rid = p[len("/runs/"):-len("/replay")]
                from clozn import runlog
                run = runlog.get_run(rid)
                if run is None:
                    return self._json(404, {"error": "run not found"})
                if not (SUB and getattr(SUB, "chat", None)):   # replay generates -> needs the qwen (chat) substrate
                    return self._json(503, {"error": "replay needs the qwen substrate"})
                changes = body.get("changes_applied", body.get("changes")) or {}
                try:
                    from clozn import replay
                    child = replay.replay(run, changes, SUB)
                except Exception as e:
                    return self._json(500, {"error": f"replay failed: {type(e).__name__}: {e}"})
                if child is None:
                    return self._json(500, {"error": "replay failed"})
                return self._json(200, child)
            if p.startswith("/runs/") and p.endswith("/branch"):   # #6: rewind & branch from a turn -> a child run
                rid = p[len("/runs/"):-len("/branch")]
                from clozn import runlog
                run = runlog.get_run(rid)
                if run is None:
                    return self._json(404, {"error": "run not found"})
                if not (SUB and getattr(SUB, "chat", None)):   # a branch re-generates -> needs the qwen substrate
                    return self._json(503, {"error": "branch needs the qwen substrate"})
                if "turn" not in body:
                    return self._json(400, {"error": "need a branch turn"})
                try:
                    turn = int(body.get("turn"))
                except (TypeError, ValueError):
                    return self._json(400, {"error": "turn must be an integer"})
                alt = body.get("alt_user")
                # greedy by default (the receipt path: a branch's future is attributable, not sampling dice).
                sample = bool(body.get("sample", False))
                try:
                    from clozn import timetravel
                    child = timetravel.branch(run, turn, SUB, alt_user=alt, sample=sample,
                                              store=_snap_store())
                except Exception as e:
                    return self._json(500, {"error": f"branch failed: {type(e).__name__}: {e}"})
                if child is None:                          # None == bad turn index or a generation failure
                    return self._json(400, {"error": "branch failed (turn out of range, or generation error)"})
                return self._json(200, child)
            if p.startswith("/runs/") and p.endswith("/propose-memory"):   # E2: propose a pending card from a past run
                rid = p[len("/runs/"):-len("/propose-memory")]
                from clozn import runlog
                run = runlog.get_run(rid)
                if run is None:
                    return self._json(200, {"ok": False, "reason": "no such run"})
                # only a substrate whose memory exposes propose_memory qualifies (QwenSubstrate). Dream's
                # memory has no such method -> the proposal is simply not offered there.
                mem = getattr(SUB, "memory", None) if SUB else None
                if mem is None or not hasattr(mem, "propose_memory"):
                    return self._json(200, {"ok": False, "reason": "proposal not available for this substrate"})
                from clozn import memory_cards
                # Neutralize tone steering during the extraction so the dials don't color the read -- snapshot
                # SUB.steer.strength, zero it, and RESTORE in a finally (mirror replay.py; never persist this).
                steer = getattr(SUB, "steer", None)
                saved_strength = dict(getattr(steer, "strength", {}) or {}) if steer is not None else None
                try:
                    if steer is not None:
                        try:
                            steer.strength = {}             # all dials neutral for the duration of the read
                        except Exception:
                            pass
                    text = mem.propose_memory(run["messages"], run.get("response"))
                except Exception as e:                      # propose_memory is defensive, but never crash the handler
                    return self._json(200, {"proposed": False, "reason": f"proposal failed: {type(e).__name__}"})
                finally:
                    if steer is not None and saved_strength is not None:
                        try:
                            steer.strength = dict(saved_strength)   # restore EXACTLY (temp neutralization only)
                        except Exception:
                            pass
                if text is None:
                    return self._json(200, {"proposed": False,
                                            "reason": "no durable preference found in this run"})
                # PROVENANCE (NEXT_STEPS #1, the OBEY defense): the model just synthesized `text` as a
                # third-person summary of the conversation -- it can be a plausible-sounding hallucination
                # (dream_consolidation_findings.md law #4) or a faithfully-mined injected instruction. Cite
                # the actual user words it was drawn from so a reviewer (and has_provenance()) can check the
                # claim, not just read the model's word for it.
                turn, span = _provenance_of(run["messages"])
                card = memory_cards.create(text, status="pending", kind="preference",
                                           risk=_risk_of(text), source_run_id=rid,
                                           source_turn=turn, quoted_span=span,
                                           evidence=f"proposed from run {rid}")
                if not card:
                    return self._json(200, {"proposed": False, "reason": "could not create card"})
                # Route a STYLE preference to the tone dial that delivers it (see /memory/add). The card
                # is still created + pending; dial_suggestion is null for a topical/non-style proposal.
                return self._json(200, {"proposed": True, "card": card,
                                        "dial_suggestion": _dial_suggestion(text)})
            if p.startswith("/runs/") and p.endswith("/explain"):   # M1: assemble the FREE signals -- zero generation
                rid = p[len("/runs/"):-len("/explain")]
                from clozn import runlog
                run = runlog.get_run(rid)
                if run is None:
                    return self._json(404, {"error": "run not found"})
                from clozn import explain
                return self._json(200, explain.explain(run))
            if p.startswith("/runs/") and p.endswith("/receipts"):   # M2: prove-all -- leave-one-out + redundancy guard
                rid = p[len("/runs/"):-len("/receipts")]
                from clozn import runlog
                run = runlog.get_run(rid)
                if run is None:
                    return self._json(404, {"error": "run not found"})
                mode = str(body.get("mode") or "regen")
                if mode not in ("regen", "forced", "both"):
                    return self._json(400, {"error": "mode must be one of regen|forced|both"})
                # regen/both regenerate both arms -> needs the qwen substrate; forced-only never
                # generates (S3: teacher-forced /score on the engine substrate) -- no chat needed.
                if mode in ("regen", "both") and not (SUB and getattr(SUB, "chat", None)):
                    return self._json(503, {"error": "receipts need the qwen substrate"})
                from clozn import receipts
                try:
                    return self._json(200, receipts.prove_all(run, SUB, mode=mode))
                except Exception as e:
                    return self._json(500, {"error": f"receipts failed: {type(e).__name__}: {e}"})
            if p.startswith("/runs/") and p.endswith("/receipt"):   # M2: one rigorous both-arms-greedy causal receipt
                rid = p[len("/runs/"):-len("/receipt")]
                from clozn import runlog
                run = runlog.get_run(rid)
                if run is None:
                    return self._json(404, {"error": "run not found"})
                mode = str(body.get("mode") or "regen")
                if mode not in ("regen", "forced", "both"):
                    return self._json(400, {"error": "mode must be one of regen|forced|both"})
                if mode in ("regen", "both") and not (SUB and getattr(SUB, "chat", None)):
                    return self._json(503, {"error": "receipt needs the qwen substrate"})
                influence = body.get("influence")
                if not isinstance(influence, dict) or not influence:
                    return self._json(400, {"error": "need an influence spec: one of "
                                            "{card_id|dial|memory_off|behavior_off}"})
                from clozn import receipts
                try:
                    out = receipts.receipt(run, influence, SUB, mode=mode)
                except Exception as e:
                    return self._json(500, {"error": f"receipt failed: {type(e).__name__}: {e}"})
                if out is None:
                    return self._json(500, {"error": "receipt failed (bad influence spec, or the replay "
                                                      "could not be generated)"})
                return self._json(200, out)
            if p.startswith("/runs/") and p.endswith("/rederive"):   # S3: deterministic teacher-forced re-derivation
                rid = p[len("/runs/"):-len("/rederive")]
                from clozn import runlog
                run = runlog.get_run(rid)
                if run is None:
                    return self._json(404, {"error": "run not found"})
                if not (SUB and getattr(SUB, "score_tokens", None)):
                    return self._json(503, {"error": "rederive needs the engine substrate (score_tokens)"})
                from clozn import rederive
                try:
                    out = rederive.rederive(run, SUB)
                except Exception as e:
                    return self._json(500, {"error": f"rederive failed: {type(e).__name__}: {e}"})
                if out is None:
                    return self._json(500, {"error": "rederive failed (no continuation to score, or the "
                                                      "engine score call failed)"})
                return self._json(200, out)
            if p.startswith("/runs/") and p.endswith("/counterfactual"):   # M3: one counterfactual dial re-gen
                rid = p[len("/runs/"):-len("/counterfactual")]
                from clozn import runlog
                run = runlog.get_run(rid)
                if run is None:
                    return self._json(404, {"error": "run not found"})
                if not (SUB and getattr(SUB, "chat", None)):   # both arms regenerate -> needs the qwen substrate
                    return self._json(503, {"error": "counterfactual needs the qwen substrate"})
                overrides = body.get("behavior_overrides")
                if not isinstance(overrides, dict) or not overrides:
                    return self._json(400, {"error": "need a behavior_overrides dict: {dial_name: value, ...}"})
                from clozn import counterfactual
                try:
                    out = counterfactual.counterfactual(run, overrides, SUB)
                except Exception as e:
                    return self._json(500, {"error": f"counterfactual failed: {type(e).__name__}: {e}"})
                if out is None:
                    return self._json(500, {"error": "counterfactual failed (bad overrides, or the replay "
                                                      "could not be generated)"})
                return self._json(200, out)
            if p.startswith("/runs/") and p.endswith("/narrate"):   # M4: accountable-self narration + confabulation-diff
                rid = p[len("/runs/"):-len("/narrate")]
                from clozn import runlog
                run = runlog.get_run(rid)
                if run is None:
                    return self._json(404, {"error": "run not found"})
                if not (SUB and getattr(SUB, "chat", None)):   # constrained + unconstrained BOTH generate -> needs qwen
                    return self._json(503, {"error": "narration needs the qwen substrate"})
                from clozn import narrate
                matcher = narrate.lexical_default            # the weak keyword proxy -- the LABELED fallback judge
                try:
                    from clozn import semantic_matcher                  # the real, INDEPENDENT cross-encoder judge, if present
                    if semantic_matcher.available():
                        matcher = semantic_matcher.nli_support_matcher
                except Exception:
                    pass
                try:
                    # returns the receipt-constrained narration + confabulation flags; the raw unconstrained
                    # "why" is NEVER a field in the result (narrate.py's structural trap guard). narrate()'s
                    # own `note` states which matcher ran, so the response is self-describing about its honesty.
                    out = narrate.narrate(run, SUB, support_matcher=matcher)
                except Exception as e:
                    return self._json(500, {"error": f"narrate failed: {type(e).__name__}: {e}"})
                return self._json(200, out)
            if p == "/engine/harvest":   # READ the real C++ runtime's activations (any substrate; the engine is separate)
                try:
                    h = ENGINE.harvest(str(body.get("text", ""))[:300])
                    norms = np.linalg.norm(h.activations, axis=1)
                    return self._json(200, {"tokens": h.tokens, "layer": int(h.layer), "n_embd": h.n_embd,
                                            "norms": [round(float(x), 3) for x in norms]})
                except Exception as e:
                    return self._json(502, {"error": f"engine: {e}"})
            if p == "/engine/layers":    # per-layer activation SUMMARY (depth x position norms) from the C++ engine
                try:
                    return self._json(200, ENGINE.harvest_layers(str(body.get("text", ""))[:300]))
                except Exception as e:
                    return self._json(502, {"error": f"engine-layers: {e}"})
            if p == "/engine/observe":   # WRITE a scaled residual back at one token, OBSERVE how the prediction moves
                try:
                    pos = int(body.get("position", 0))
                    scale = float(body.get("scale", 4.0))

                    def tf(a):
                        a = a.copy()
                        if 0 <= pos < a.shape[0]:
                            a[pos] = a[pos] * scale
                        return a

                    h, obs = ENGINE.edit_and_observe(str(body.get("text", ""))[:300], transform=tf, positions=[pos])
                    return self._json(200, {"summary": obs.summary(), "shifted": obs.shifted(),
                                            "moved_l2": obs.moved_l2, "baseline_top": obs.baseline_top,
                                            "edited_top": obs.edited_top, "tokens": h.tokens,
                                            "position": pos, "scale": scale})
                except Exception as e:
                    return self._json(502, {"error": f"engine: {e}"})
            if p == "/engine/concepts":   # the brain's concepts, but read from the Qwen GGUF engine (harvest L15 + SAE)
                try:
                    if not (SUB and getattr(SUB, "brain", None)):
                        return self._json(409, {"error": "concepts need the qwen substrate (it holds the SAE)"})
                    return self._json(200, SUB.brain.concepts_from_engine(
                        str(body.get("text", ""))[:300], ENGINE_QWEN, int(body.get("layer", 15))))
                except Exception as e:
                    return self._json(502, {"error": f"engine-qwen: {e}"})
            if p == "/engine/steer/axes":   # the tone dials, but they apply on the GGUF via the engine
                from clozn.steering import AXES
                es = _engine_steer()
                return self._json(200, {"axes": [{"name": k, "poles": AXES[k]["poles"]} for k in AXES],
                                        "ready": bool(es and es.ready), "engine": bool(ENGINE_QWEN)})
            if p == "/engine/steer/check":   # A/B one dial on the engine GGUF: baseline vs steered generation
                es = _engine_steer()
                if es is None:
                    return self._json(502, {"error": "no engine configured (set CLOZN_ENGINE_QWEN_PORT)"})
                try:
                    prompt = str(body.get("prompt", "Tell me about the city at night."))[:300]
                    axis, val = str(body.get("axis", "warm")), float(body.get("value", 1.0))
                    mx = int(body.get("max_tokens", 60))
                    base = es.generate(prompt, strength={}, max_new=mx)            # no dial = the baseline
                    stee = es.generate(prompt, strength={axis: val}, max_new=mx)
                    return self._json(200, {"prompt": prompt, "axis": axis, "value": val,
                                            "baseline": base.strip(), "steered": stee.strip()})
                except Exception as e:
                    return self._json(502, {"error": f"engine-steer: {e}"})
            if p == "/engine/chat":   # THE HYBRID: chat on the GGUF via the engine, with the HF-trained memory injected
                if ENGINE_QWEN is None:
                    return self._json(502, {"error": "no engine configured"})
                msgs = body.get("messages", [])
                t0 = time.time()
                memout = {}
                try:
                    mx = int(body.get("max_tokens", 220))
                    kw = {}
                    mem = getattr(SUB, "memory", None) if SUB else None
                    if _memory_mode() == "prompt":
                        # PROMPT MODE on the engine: the cards ride as the system block INSIDE the chat
                        # template (compiled straight from the card store -- no HF model needed at all),
                        # and the trained prefix is NOT injected. Strength maps to on/off; the topic gate
                        # omits the block off-topic, exactly as on the HF path. This also means a FRESH
                        # install (no trained prefix) finally gets memory over the engine.
                        ms = float(getattr(mem, "memory_strength", 1.0)) if mem is not None \
                            else _disk_memory()[1]
                        block, applied, gate = _prompt_block_for(mem, _last_user(msgs), strength=ms)
                        assembled = _inject_block(msgs, block)
                        memout.update(mode="prompt", applied=applied, gate=gate, strength=ms,
                                      prompt_block=block, assembled_messages=assembled)
                        prompt = _engine_tmpl(ENGINE_QWEN, assembled)   # the loaded GGUF's own template, not Qwen ChatML
                    else:
                        prompt = _engine_tmpl(ENGINE_QWEN, msgs)        # (the internalized-prefix path is Qwen-trained,
                        #                                                  but the CHAT TEMPLATE is still the model's own)
                        # MEMORY: the live HF prefix if a qwen substrate is loaded, else the SAVED prefix from
                        # disk -- so engine-chat works with NO HF model resident (the pure-engine substrate).
                        if mem is not None and getattr(mem, "prefix", None) is not None:
                            prefix = mem.prefix.detach().float().cpu()
                            ms = float(getattr(mem, "memory_strength", 1.0))
                        else:
                            prefix, ms = _disk_memory()
                        # TOPIC RELEVANCE gate on the injection strength (mirror the HF chat's gate="auto"):
                        # scale ms by how on-topic the last user turn is vs the active rules. Only when a LIVE
                        # memory with rules is present (the qwen substrate) -- the pure-engine/disk path has no
                        # rule texts to gate against, so it degrades to no-gating (rel==1.0), the prior
                        # behavior. Defensive: any failure leaves ms unscaled.
                        try:
                            if mem is not None and getattr(mem, "rules", None):
                                ms = ms * float(mem._gate(_last_user(msgs)))
                        except Exception:
                            pass
                        if prefix is not None:             # inject the trained soft prefix (dial x relevance)
                            kw = {"prefix_embd": (prefix * ms).flatten().tolist(),
                                  "prefix_rows": int(prefix.shape[0])}
                    # backlog #5: record the EXACT rendered chat-template string the model saw (both memory
                    # modes render one). _log_run reads memout["final_prompt"] -> the run record's final_prompt.
                    memout["final_prompt"] = prompt
                    # TONE: live dial values if a substrate is up, else the saved values from disk
                    st = getattr(getattr(SUB, "steer", None), "strength", None) if SUB else None
                    if not st:
                        st = _disk_dials()
                    if st and any(st.values()):
                        es = _engine_steer()
                        sv = es.steer_vector(st) if es is not None else None
                        if sv:
                            kw["steer_vec"] = sv
                            kw["steer"] = {"coef": 1.0, "layer": es.layer}
                    # Generate + capture a per-token trace alongside (B3). Reply is byte-identical to the
                    # plain complete(); the trace feeds the Run Inspector timeline. steps=[] (diffusion, or a
                    # stream hiccup) -> runlog stores a clean empty trace.
                    reply_raw, steps, finish = _engine_complete_traced(ENGINE_QWEN, prompt, mx, kw)
                    reply = reply_raw.strip()
                    # Pass the raw step list; runlog.record normalizes it -> {tokens, confidence, alternatives}.
                    self._log_run("engine_chat", msgs, reply, "clozn-qwen (engine)", t0, trace=steps,
                                  mem_out=memout, finish_reason=finish)
                    # "memory" == did memory actually ride this reply (block in prompt mode, prefix otherwise)
                    return self._json(200, {"reply": reply,
                                            "memory": bool(memout.get("applied")) or bool(kw.get("prefix_embd")),
                                            "tone": bool(kw.get("steer_vec")), "via": "engine (GGUF)"})
                except Exception as e:
                    self._log_run("engine_chat", msgs, "", "clozn-qwen (engine)", t0, error=str(e),
                                  mem_out=memout)
                    return self._json(502, {"error": f"engine-chat: {e}"})
            if p == "/v1/chat/completions":   # OpenAI-compatible: chat with memory prefix + tone steering applied
                if not (SUB and getattr(SUB, "chat", None)):
                    return self._json(503, {"error": "chat needs the qwen substrate"})
                msgs, mx = body.get("messages", []), int(body.get("max_tokens", 256))
                if body.get("stream") and getattr(SUB, "chat_stream", None):
                    return self._sse_chat(msgs, mx, str(body.get("model", "clozn-qwen")))
                t0 = time.time()
                trace_steps = []                            # HF non-stream: capture a per-token trace (B3)
                memout = {}                                 # prompt mode: what memory actually rode this turn
                reply = SUB.chat(msgs, mx, float(body.get("temperature", 0.7)) > 0, trace_out=trace_steps,
                                 mem_out=memout)
                fr = SUB.last_finish_reason() if hasattr(SUB, "last_finish_reason") else None
                openai_fr = _openai_finish_reason(fr)
                # runlog.record normalizes the raw step list -> {tokens, confidence, alternatives}.
                rid = self._log_run("openai_api", msgs, reply, body.get("model", "clozn-qwen"), t0,
                                    trace=trace_steps, mem_out=memout, finish_reason=fr,
                                    finish_reason_fallback=None if fr else openai_fr)
                resp = {"id": "chatcmpl-clozn", "object": "chat.completion",
                       "created": int(time.time()), "model": body.get("model", "clozn-qwen"),
                       "choices": [{"index": 0, "finish_reason": openai_fr,
                                    "message": {"role": "assistant", "content": reply}}],
                       "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
                # M5 any-client run_id bridge (EXPLAIN_THIS_ANSWER_SPEC.md): surface the id two ways so a
                # companion `clozn explain <run_id>` can inspect THIS reply from any OpenAI-compatible client
                # -- an additive top-level field (spec-compliant clients ignore unknown fields) and a response
                # header (for clients that only expose headers). Clean omission when logging failed (rid is
                # None) -- never emit a literal "null"/"None".
                extra_headers = {"X-Clozn-Run-Id": rid} if rid else None
                if rid:
                    resp["clozn_run_id"] = rid
                # FRONTIER §1.1 "trust as an API field": when the caller OPTS IN (clozn_trust:true -- default
                # OFF, so a standard OpenAI response stays byte-unchanged / fully compatible), attach
                # claim-level confidence spans over the reply. Built by the SAME producer as
                # GET /runs/<id>/spans (confidence_spans.spans over the normalized token trace), from THIS
                # turn's trace -- so an agent can branch on per-claim confidence inline, without a second call.
                # HONESTY (FRONTIER §6 ledger): these are RAW, UNCALIBRATED model probabilities. clozn_spans_note
                # says so verbatim -- self-confidence != correctness; NO calibration is done here, and nothing
                # implies confidence == correctness.
                if body.get("clozn_trust"):
                    try:
                        from clozn import confidence_spans, runlog as _runlog
                        _run_for_spans = {"trace": _runlog.steps_to_trace(trace_steps)}
                        resp["clozn_spans"] = confidence_spans.spans(_run_for_spans)
                        resp["clozn_spans_note"] = ("uncalibrated raw token confidence -- "
                                                    "self-confidence != correctness")
                    except Exception:
                        pass                          # trust is additive: a spans hiccup never breaks the reply
                return self._json(200, resp, extra_headers=extra_headers)
            if p == "/say":   # studio chat (qwen memory model) -> capture it as a run
                if not (SUB and getattr(SUB, "handle", None)):
                    return self._json(409, {"error": f"'{p}' isn't served by the '{SUBNAME}' substrate"})
                msg = str(body.get("message", ""))
                t0 = time.time()
                # HF studio chat: capture a per-token trace (B3) + the per-turn memory record. We hand
                # SUB.handle collectors via body["_trace_out"] / body["_mem_out"] (server-side only, never
                # echoed); QwenSubstrate's /say fills them through say()/_say_prompt -> _generate's
                # pass-through recorder. Reply text is byte-identical with or without them.
                trace_steps = []
                body["_trace_out"] = trace_steps
                memout = {}
                body["_mem_out"] = memout
                try:
                    r = SUB.handle(p, body)
                except Exception as e:
                    self._log_run("studio_chat", [{"role": "user", "content": msg}], "",
                                  "clozn-qwen", t0, error=str(e), mem_out=memout)
                    return self._json(500, {"error": f"{type(e).__name__}: {e}"})
                if r is None:
                    return self._json(409, {"error": f"'{p}' isn't served by the '{SUBNAME}' substrate",
                                            "need": "qwen", "active": SUBNAME})
                # runlog.record normalizes the raw step list -> {tokens, confidence, alternatives}; a diffusion
                # substrate (or any path that filled nothing) yields [] -> a clean empty trace.
                self._log_run("studio_chat", [{"role": "user", "content": msg}],
                              str(r.get("reply", "")), "clozn-qwen", t0, trace=trace_steps, mem_out=memout)
                return self._json(200, r)
            if p == "/denoise":   # Dream diffusion window -> capture it as a run
                if not (SUB and getattr(SUB, "handle", None)):
                    return self._json(409, {"error": f"'{p}' isn't served by the '{SUBNAME}' substrate",
                                            "need": "dream", "active": SUBNAME})
                prompt = str(body.get("prompt", ""))
                t0 = time.time()
                try:
                    r = SUB.handle(p, body)
                except Exception as e:
                    self._log_run("denoise", [{"role": "user", "content": prompt}], "",
                                  "clozn-dream", t0, error=str(e))
                    return self._json(500, {"error": f"{type(e).__name__}: {e}"})
                if r is None:
                    return self._json(409, {"error": f"'{p}' isn't served by the '{SUBNAME}' substrate",
                                            "need": "dream", "active": SUBNAME})
                self._log_run("denoise", [{"role": "user", "content": prompt}],
                              str(r.get("final_text", "")), "clozn-dream", t0)
                return self._json(200, r)
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
