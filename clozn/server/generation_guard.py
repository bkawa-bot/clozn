"""generation_guard.py -- closed-loop disposition guardrails, an OPT-IN, DEFAULT-OFF generation mode
(FRONTIER_BETS section 9.1 / experiment A1.1, docs/RESEARCH_ROADMAP.md's A1.1 row + docs/PRODUCT_ROADMAP.md's
R3 lane).

===============================================================================================
HONESTY LAW -- read this before touching ANY user-facing string in this module. It governs every label.
===============================================================================================
A1.1's verdict was INCONCLUSIVE on its own headline thesis, not a clean win. What LIVES, measured on a
20-prompt banned-topic battery (Qwen3.5-9B, J-lens layer 16, counter_strength -0.5):
  * catch-rate 100% (10/10 banned-content cases caught)
  * false-positive-rate 5% (1/20 clean prompts flagged)
What FAILED: the "intent-BEFORE-speech" lead-time thesis. median_lead_time_tokens == 0, and only 4/10
cases showed ANY positive lead. The lens flags the guarded disposition AT the token where it is realized,
NOT before it. So this is a PRESENT-TENSE DETECT-AND-CORRECT guard, not a predictive/precognitive one.

Every user-facing string, docstring, and receipt field in this module MUST say "detects and corrects
during generation" (or an equivalent present-tense framing) and MUST NOT claim lead-time, prediction,
"before it's said", or "acts on intent". GUARD_CAVEAT below is the one canonical copy of this framing --
every surface that reports on a firing guard reuses it verbatim rather than re-deriving its own wording.
This mirrors the house rule that killed every predictive-interior claim (docs/RESEARCH_ROADMAP.md's Killed
list: "White-box risk controller advantage" -- the deployed selective-generation score turned out
bit-identical to a black-box logprob; internal state added nothing there either).

===============================================================================================
MECHANISM
===============================================================================================
Every engine primitive here already ships: /jlens (a deterministic linear read -- "which tokens is this
residual disposed to say later", EngineClient.jlens), /harvest, and /intervene (steer_vec + coef, the same
wire dir(c) already rides -- concept_dir.ConceptSteer.steer_toward). None of them are new.

What's new is the LOOP. The engine steers a WHOLE generation call; it has no notion of "steer only after
token 40". So this module is a GATEWAY control loop over the engine, not an engine feature:
  1. Generate a short CHUNK of tokens (chunk_tokens, default 24) with no steering.
  2. Poll: read the J-lens disposition toward every guarded concept over the text generated so far.
  3. If any guarded concept's disposition crosses `threshold`, DISCARD that chunk and regenerate it with
     the corrective dir(c) counter-direction (concept_dir.ConceptSteer.steer_toward(concept,
     counter_strength) -- counter_strength is negative by convention/default, steering AWAY from the
     concept) injected via /intervene's steer_vec, at the SAME token budget.
  4. Continue from the (possibly corrected) chunk. Repeat until `max_tokens` is produced or the
     `max_fires` re-steer cap is reached (see RE-STEER CAP below).

`run_guarded_generation` below is the pure, engine-agnostic control loop (generate_chunk/
read_disposition/build_counter are injected callables) -- fully unit-testable with fakes, no socket, no
GPU. `guarded_chat_completion` is the thin production adapter that wires it to a real EngineSubstrate's
engine client + a fresh concept_dir.ConceptSteer; it is exercised in tests/test_generation_guard_server.py
via a fake substrate/engine (never a live engine), matching this codebase's "live path deferred, the
pure/wiring logic is model-free tested" split (see clozn.server.generation_gateway's
selective_generation_action, or scripts/calibration/concept_dial_autocalibrate.py's own sweep).

===============================================================================================
RE-STEER CAP
===============================================================================================
`max_fires` (default 3) bounds how many chunks this loop will ever re-steer. Once the cap is spent, the
guard STOPS watching for the remainder of the generation (no more polling, no more correction) and the
receipt says so plainly (`cap_reached: true`) -- never an infinite re-steer loop, and never a silent claim
that the rest of the output was checked when it wasn't.

===============================================================================================
FAIL-CLOSED DECISION (documented here, not just in code)
===============================================================================================
If a guarded concept cannot be resolved to a working dir(c) (no unembed/engine support, a multi-token
word, an unfitted J-lens layer, ...), this module REFUSES the request outright -- it does not silently
generate an unguarded reply, even flagged. Rationale: `clozn_guard` is a safety request ("please steer
this generation away from X"); a caller who explicitly asked for that guarantee and silently got ordinary,
unguarded generation back (even with a flag buried in the metadata) may never notice the flag and may
treat the reply as guarded when it never was -- exactly the "silent pass" this feature exists to prevent.
Refusing outright (a clear 4xx with a stated reason) is the safer default: the caller finds out
immediately, at request time, that their safety request could not be honored, rather than downstream,
if ever. This mirrors selective_generation_action's fail-closed pattern (calibration backlog #10) but goes
one step further: there, the safe fallback (annotate-only) was itself harmless; here, the "fallback" would
be exactly the ungated content the guard was supposed to prevent, so refusing is the only honest option.

===============================================================================================
SCOPE LIMITS OF THIS FIRST CUT (said honestly, not silently)
===============================================================================================
  * STREAMING IS DEFERRED. Guarded generation only supports the non-streaming
    POST /v1/chat/completions path; requesting `clozn_guard` together with `stream: true` is refused
    (fail-closed for the same reason as above -- a streamed reply's early tokens are already delivered to
    the client before any correction could happen, so silently ignoring the guard on a streaming request
    would be exactly the silent pass this module exists to prevent).
  * NOT YET COMPOSED with prompt-card memory, tone-dial steering, corrective-retry policies, or structured
    output -- a guarded generation renders the chat messages through the model's own template
    (clozn.server.app._engine_tmpl, the same renderer chat() uses) and generates directly against the raw
    engine client, bypassing that machinery entirely for this first cut. Composing them is future work,
    not silently dropped.
  * Coherence is a light, cheap NOTE (clozn.replay.counterfactual._coherence's pure text-counting check,
    over the corrected chunks only), not a hard gate -- see `_coherence_note`.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Optional

# =================================================================================================
# THE HONEST CAVEAT -- the one canonical copy. See the HONESTY LAW section above before editing this.
# =================================================================================================
GUARD_CAVEAT = (
    "closed-loop disposition guardrail (FRONTIER_BETS section 9.1 / experiment A1.1). This guard "
    "detects and corrects during generation -- present tense only: it reads a guarded concept's "
    "disposition at the token where it is realized and steers the continuation away from it, never "
    "earlier. A1.1's measured numbers (20-prompt banned-topic battery, Qwen3.5-9B, J-lens layer 16, "
    "counter_strength -0.5): catch-rate 100% (10/10 banned-content cases caught), false-positive-rate "
    "5% (1/20 clean prompts flagged). Its foresight thesis FAILED: median_lead_time_tokens = 0 and only "
    "4/10 cases showed any positive lead -- so this is not a foreknowledge mechanism and must never be "
    "described as one."
)

GUARD_CAP_NOTE = (
    "guard cap reached: the re-steer budget (max_fires) was spent before generation finished, so the "
    "remainder of this reply was produced WITHOUT further guard monitoring or correction -- it is "
    "honest, not silent, about that gap."
)

# -- opt-in wiring -----------------------------------------------------------------------------------
GUARD_FIELD = "clozn_guard"                 # request-body extension field (an object, not a boolean)
GUARD_SETTING = "generation_guard"          # server-wide default spec (clozn.memory.mode's settings store)

# -- defaults, documented as placeholders where calibration doesn't exist yet ------------------------
DEFAULT_COUNTER_STRENGTH = -0.5             # A1.1's own validated counter_strength (steers AWAY -- negative)
DEFAULT_MAX_FIRES = 3
DEFAULT_LAYER = 16                          # A1.1's own validated J-lens tap
DEFAULT_CHUNK_TOKENS = 24
DEFAULT_TOPK = 8
# /jlens's "score" is the RAW LENS LOGIT (engine/core/serve/routes_jlens.cpp's own docstring: "score = the
# raw lens logit"), not a probability -- there is no universal safe cutoff. 0.0 is an UNCALIBRATED
# placeholder (the neutral point where the readout turns positive), not a validated number the way
# VALIDATED_MEDIAN_RESID_NORM was for dir(c)'s injection magnitude. A1.1's OWN protocol used topk
# PRESENCE ("if banned concept in top-k") rather than a numeric cutoff at all; `threshold` here is a
# generalization of that rule for callers who want a tunable trigger. Calibrate this per model/layer
# before relying on it alone, the same way item 2's concept-dial calibration closed the analogous gap for
# dir(c)'s own injection scale.
DEFAULT_THRESHOLD = 0.0


# =================================================================================================
# opt-in spec parsing (mirrors generation_gateway.selective_generation_enabled's precedence exactly)
# =================================================================================================

def _normalize_guard_spec(raw: Any) -> Optional[dict]:
    """Validate one `clozn_guard` (or server-default) payload into a fully-defaulted spec dict, or None
    when there is genuinely nothing to guard (no concepts -- "empty" is OFF, not an error, per the opt-in
    contract). Raises ValueError on a STRUCTURALLY malformed non-empty value (wrong types) -- an explicit,
    broken guard request should fail loudly (400), not be silently ignored; that silence is exactly what
    this whole feature exists to avoid."""
    if not isinstance(raw, Mapping):
        raise ValueError("clozn_guard must be an object")
    concepts_raw = raw.get("concepts")
    if concepts_raw is None:
        concepts_raw = []
    if not isinstance(concepts_raw, (list, tuple)):
        raise ValueError("clozn_guard.concepts must be a list of concept words")
    concepts = []
    for c in concepts_raw:
        if isinstance(c, bool) or not isinstance(c, str) or not c.strip():
            raise ValueError("clozn_guard.concepts must be a list of non-empty strings")
        concepts.append(c.strip())
    if not concepts:
        return None   # nothing to guard against -- equivalent to off, not a misconfiguration

    def _num(key, default):
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"clozn_guard.{key} must be a number")
        return float(value)

    def _pos_int(key, default):
        value = raw.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"clozn_guard.{key} must be a positive integer")
        return int(value)

    layer_raw = raw.get("layer", DEFAULT_LAYER)
    if isinstance(layer_raw, bool) or not isinstance(layer_raw, int):
        raise ValueError("clozn_guard.layer must be an integer")

    return {
        "concepts": concepts,
        "threshold": _num("threshold", DEFAULT_THRESHOLD),
        "counter_strength": _num("counter_strength", DEFAULT_COUNTER_STRENGTH),
        "max_fires": _pos_int("max_fires", DEFAULT_MAX_FIRES),
        "layer": int(layer_raw),
        "chunk_tokens": _pos_int("chunk_tokens", DEFAULT_CHUNK_TOKENS),
        "topk": _pos_int("topk", DEFAULT_TOPK),
    }


def parse_guard_spec(body: Any) -> Optional[dict]:
    """The request's guard spec, or None (OFF -- byte-identical to today). An explicit `clozn_guard` on
    the request always wins (including an explicit falsy value, which means "opted out" even when the
    server default is on); only when the request omits the field entirely does the server-wide
    `generation_guard` setting (GUARD_SETTING, clozn.memory.mode's generic settings store) apply. Raises
    ValueError on a structurally malformed explicit value (see _normalize_guard_spec) -- callers should
    turn that into an HTTP 400, never swallow it."""
    if isinstance(body, Mapping) and GUARD_FIELD in body:
        raw = body.get(GUARD_FIELD)
        return _normalize_guard_spec(raw if raw else {})
    try:
        from clozn.memory import mode as memory_mode
        saved = memory_mode.get_setting(GUARD_SETTING, None)
    except Exception:
        saved = None
    if not saved:
        return None
    return _normalize_guard_spec(saved)


# =================================================================================================
# fail-closed concept resolution
# =================================================================================================

def resolve_guard_concepts(concept_steer, concepts: list, layer: int) -> tuple[Optional[dict], Optional[str]]:
    """Resolve EVERY guarded concept to a working dir(c) build up front, before any generation happens.
    FAIL CLOSED (see the module docstring's FAIL-CLOSED DECISION): if ANY concept cannot be resolved, this
    returns (None, reason) instead of a partially-guarded setup -- the caller must refuse the whole
    request rather than generate under a guarantee it cannot back for even one guarded concept.

    Returns ({concept: compute()-shaped dict with 'token_id'}, None) on full success, or
    (None, "concept <c> unavailable: <why>") on the first failure. `concept_steer` is anything duck-typed
    against concept_dir.ConceptSteer's `.compute(concept, layer=...)` contract."""
    built = {}
    for concept in concepts:
        result = concept_steer.compute(concept, layer=layer)
        if not result.get("ok"):
            return None, f"concept {concept!r} unavailable: {result.get('note')}"
        built[concept] = result
    return built, None


# =================================================================================================
# disposition read (jlens-based)
# =================================================================================================

def concept_activation(jlens_result: Optional[dict], token_id: int, position: int = -1) -> Optional[float]:
    """The guarded concept's raw lens-logit score at one jlens readout position (default: the LAST
    position -- the most recently generated token), or None when the concept's token doesn't appear in
    that position's top-k at all (never a fabricated 0.0 -- "not in the top-k" and "scored exactly 0" are
    different facts, and only the first is what usually happens)."""
    if not isinstance(jlens_result, Mapping):
        return None
    readouts = jlens_result.get("readouts") or []
    if not readouts:
        return None
    if not (-len(readouts) <= position < len(readouts)):
        position = -1
    row = readouts[position] or []
    for entry in row:
        if isinstance(entry, Mapping) and entry.get("id") == token_id:
            score = entry.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                return None
            return float(score)
    return None


# =================================================================================================
# the pure control loop -- engine-agnostic, fully unit-testable with fakes
# =================================================================================================

@dataclass
class GuardFiring:
    chunk_index: int
    token_position: int
    concept: str
    pre_activation: Optional[float]
    post_activation: Optional[float]
    counter_strength: float

    def as_dict(self) -> dict:
        return {
            "chunk_index": self.chunk_index, "token_position": self.token_position,
            "concept": self.concept, "pre_activation": self.pre_activation,
            "post_activation": self.post_activation, "counter_strength": self.counter_strength,
        }


def run_guarded_generation(
    *, generate_chunk: Callable[..., str], read_disposition: Callable[[str], dict],
    build_counter: Callable[[str], Any], base_text: str, max_tokens: int, chunk_tokens: int,
    concepts: list, threshold: float, counter_strength: float, max_fires: int,
) -> dict:
    """THE control loop (see the module docstring's MECHANISM section). Engine-agnostic: every engine
    round trip is behind an injected callable, so this function itself never talks to a socket and is
    fully deterministic to unit-test.

    generate_chunk(prompt_so_far: str, max_new: int, *, counter=None) -> str
        `counter`, when given, is whatever `build_counter()` returned for one concept (opaque to this
        loop) -- the production adapter turns it into an /intervene call; None means "generate
        uncorrected" (a plain /v1/completions-equivalent call).
    read_disposition(text_so_far: str) -> {concept: float | None}
        per-concept activation after the text generated so far (a jlens readout, or any other disposition
        signal a caller wants to inject) -- this loop only compares it against `threshold`.
    build_counter(concept: str) -> Any
        the corrective payload for one concept (production: ConceptSteer.steer_toward's returned dict) --
        opaque here, passed straight through to `generate_chunk`'s `counter=`.

    Returns {"text": str, "fires": [GuardFiring.as_dict(), ...], "n_fires": int, "cap_reached": bool,
    "n_chunks": int}. Once `max_fires` re-steers have happened, the loop stops POLLING entirely for the
    rest of generation (not just stops correcting) and generates the remainder in one plain, uncorrected
    call -- `cap_reached` says so; see GUARD_CAP_NOTE for the receipt-facing wording. Never raises on its
    own control flow; a failure inside an injected callable propagates to the caller unchanged."""
    text = ""
    fires: list[GuardFiring] = []
    cap_reached = False
    chunk_index = 0
    remaining = int(max_tokens)
    token_position = 0

    while remaining > 0:
        this_chunk = min(int(chunk_tokens), remaining)
        prompt_so_far = base_text + text

        if cap_reached:
            # The re-steer budget is spent -- generate every remaining token in one plain, unwatched call
            # (honest: no further polling, not a silent "still checking" claim).
            text += generate_chunk(prompt_so_far, remaining, counter=None)
            remaining = 0
            break

        piece = generate_chunk(prompt_so_far, this_chunk, counter=None)
        activations = read_disposition(prompt_so_far + piece) or {}
        fired_concept = next(
            (c for c in concepts if activations.get(c) is not None and activations[c] >= threshold),
            None,
        )
        if fired_concept is not None:
            pre = activations.get(fired_concept)
            if len(fires) < max_fires:
                counter = build_counter(fired_concept)
                corrected = generate_chunk(prompt_so_far, this_chunk, counter=counter)
                post_activations = read_disposition(prompt_so_far + corrected) or {}
                post = post_activations.get(fired_concept)
                fires.append(GuardFiring(
                    chunk_index=chunk_index, token_position=token_position, concept=fired_concept,
                    pre_activation=pre, post_activation=post, counter_strength=counter_strength,
                ))
                piece = corrected
            else:
                cap_reached = True

        text += piece
        chunk_index += 1
        token_position += this_chunk
        remaining -= this_chunk

    return {
        "text": text,
        "fires": [f.as_dict() for f in fires],
        "n_fires": len(fires),
        "cap_reached": cap_reached,
        "n_chunks": chunk_index,
    }


# =================================================================================================
# light coherence note (A1.1's own "n_guarded_outputs_looking_incoherent" axis, cheap version)
# =================================================================================================

def _coherence_note(corrected_texts: list) -> dict:
    """Cheap coherence check over the CORRECTED chunks only (pure text counting, no model call -- reuses
    clozn.replay.counterfactual._coherence, the same mandatory degenerate-output gate
    research/dial_autocalibrate_engine.py's own sweep uses). A light note, not a hard gate: A1.1's own
    protocol counted "outputs looking incoherent" as a check, not a kill condition on its own. On any
    import/scan failure this degrades to {"checked": False, ...} rather than breaking the guard result."""
    if not corrected_texts:
        return {"checked": True, "n_checked": 0, "any_degenerate": False}
    try:
        from clozn.replay.counterfactual import _coherence
        flags = [_coherence(t) for t in corrected_texts]
        return {
            "checked": True, "n_checked": len(flags),
            "any_degenerate": any(f.get("degenerate") for f in flags),
        }
    except Exception as exc:
        return {"checked": False, "reason": str(exc)}


def build_receipt(result: dict, spec: dict) -> dict:
    """The public `clozn_guard_receipt` shape -- one canonical builder so every caller (the production
    route, tests) constructs the identical fields. Only present on the response when the guard actually
    ran (opted in AND concepts resolved); see the module docstring's HONESTY LAW for GUARD_CAVEAT's
    wording rules."""
    # run_guarded_generation doesn't retain per-chunk text separately, so the light coherence check runs
    # over the whole final text -- cheap (pure text counting) and still catches the case A1.1's own
    # protocol watched for (a correction that derails into repeat-loop/degenerate output).
    note = _coherence_note([result["text"]]) if result.get("text") else _coherence_note([])
    receipt = {
        "concepts": list(spec["concepts"]),
        "layer": spec["layer"],
        "threshold": spec["threshold"],
        "counter_strength": spec["counter_strength"],
        "max_fires": spec["max_fires"],
        "n_fires": result["n_fires"],
        "fires": result["fires"],
        "cap_reached": result["cap_reached"],
        "coherence": note,
        "caveat": GUARD_CAVEAT,
    }
    if result["cap_reached"]:
        receipt["cap_note"] = GUARD_CAP_NOTE
    return receipt


# =================================================================================================
# production adapter -- wires the pure loop to a real EngineSubstrate + ConceptSteer.
# Exercised in tests/test_generation_guard_server.py via a FAKE substrate/engine (never a live one) --
# see the module docstring's MECHANISM section for why the pure loop above, not this function, carries
# the bulk of the unit-test weight.
# =================================================================================================

def _sampling_params(sub) -> dict:
    """The raw engine-call sampling kwargs (temperature/top_k/top_p/rep_penalty/seed), resolved the SAME
    way EngineSubstrate._complete_chat_native does -- see clozn.server.app._resolve_sampling. Reused here
    rather than re-derived so a guarded generation samples under the identical regime an ordinary chat
    call would (S5), not some bespoke guard-only default."""
    from clozn.server import app as ctx
    samp = ctx._resolve_sampling(True)
    if samp and samp.get("on"):
        return {"temperature": float(samp["temperature"]), "rep_penalty": float(samp["repeat_penalty"]),
               "top_k": int(samp["top_k"]), "top_p": float(samp["top_p"]), "seed": int(samp["seed"])}
    return {"temperature": 0.0, "rep_penalty": 1.0, "top_k": 0, "top_p": 1.0, "seed": 0}


def guarded_chat_completion(handler, messages: list, *, model: str, max_tokens: int, sample=True,
                           spec: dict, source: str = "openai_api", extra_meta: Optional[dict] = None) -> dict:
    """The production entry point routes/openai.py calls when `clozn_guard` is opted in on a non-streaming
    /v1/chat/completions request. Builds a fresh concept_dir.ConceptSteer bound to the active substrate's
    raw engine client at `spec['layer']`, resolves every guarded concept up front (fail-closed -- see
    resolve_guard_concepts), and if that succeeds, drives `run_guarded_generation` with adapters over the
    engine's real /v1/completions ("complete"), /intervene, and /jlens calls.

    Returns one of:
      {"ok": False, "reason": "..."} -- the concept resolution failed; the caller must refuse the whole
        request (see the module docstring's FAIL-CLOSED DECISION), never fall back to an unguarded reply.
      {"ok": True, "reply": str, "run_id": str | None, "receipt": {...}, "finish_reason": str | None} --
        the guard ran (whether or not it ever actually fired); `receipt` is build_receipt's shape.
    Never raises on the concept-resolution path; a genuine engine failure during generation propagates
    (the caller's own error handling applies, matching every other engine-touching route)."""
    import time
    from clozn.server import app as ctx
    from clozn.behavior.steering.concept_dir import ConceptSteer, _text_of as _engine_text_of

    started = time.time()
    sub = ctx.active_sub(handler)
    concept_steer = ConceptSteer(sub.engine, layer=spec["layer"])
    built, reason = resolve_guard_concepts(concept_steer, spec["concepts"], spec["layer"])
    if built is None:
        return {"ok": False, "reason": reason}

    base_text = ctx._engine_tmpl(sub.engine, messages)
    engine_kw = _sampling_params(sub)
    last_finish = {"reason": None}

    def generate_chunk(prompt_so_far: str, max_new: int, *, counter=None) -> str:
        if counter is None:
            resp = sub.engine.complete(prompt_so_far, max_tokens=int(max_new), **engine_kw)
        else:
            resp = sub.engine.intervene(prompt_so_far, vector=counter["vector"], coef=counter["coef"],
                                        layer=spec["layer"], max_tokens=int(max_new), **engine_kw)
        choices = resp.get("choices") if isinstance(resp, dict) else None
        if choices:
            last_finish["reason"] = choices[0].get("finish_reason")
        return _engine_text_of(resp)

    def read_disposition(text_so_far: str) -> dict:
        jl = sub.engine.jlens(text_so_far, layer=spec["layer"], topk=spec["topk"])
        return {concept: concept_activation(jl, built[concept]["token_id"]) for concept in spec["concepts"]}

    def build_counter(concept: str):
        corrected = concept_steer.steer_toward(concept, spec["counter_strength"], layer=spec["layer"])
        if not corrected.get("ok"):
            # A build that succeeded at setup (resolve_guard_concepts) but fails now is a genuine internal
            # inconsistency, not a normal degrade path -- surface it rather than silently steering nothing.
            raise RuntimeError(
                f"counter-direction for {concept!r} became unavailable mid-generation: {corrected.get('note')}"
            )
        return corrected

    result = run_guarded_generation(
        generate_chunk=generate_chunk, read_disposition=read_disposition, build_counter=build_counter,
        base_text=base_text, max_tokens=int(max_tokens), chunk_tokens=spec["chunk_tokens"],
        concepts=spec["concepts"], threshold=spec["threshold"], counter_strength=spec["counter_strength"],
        max_fires=spec["max_fires"],
    )
    receipt = build_receipt(result, spec)

    meta = dict(extra_meta or {})
    meta["clozn_guard"] = receipt
    rid = handler._log_run(source, messages, result["text"], model, started, extra_meta=meta)

    return {"ok": True, "reply": result["text"], "run_id": rid, "receipt": receipt,
           "finish_reason": last_finish["reason"]}
