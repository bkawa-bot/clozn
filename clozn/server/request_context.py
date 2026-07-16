"""RequestContext -- the per-generation-call bundle (backlog #2, "request isolation + cancellation").

WHY this exists: EngineSubstrate.chat()/chat_stream() used to stash everything a call produces across FIVE
separate instance attributes written one at a time over the course of the call: self._last_generation_meta,
self._last_stream_trace, self._last_finish_reason, self._last_diverged, self._last_diverged_at. A reader
(run_meta(), last_finish_reason(), the loop guard's own mid-call read of self._last_stream_trace) calling in
between two of those writes -- which today's POST_GATE (a single `threading.Lock`, see request_gate.py)
makes impossible by fully serializing generation, but which is exactly the thing any future loosening of
that gate would reintroduce -- would see a TORN mix: e.g. THIS turn's trace sitting next to the PREVIOUS
turn's finish reason. Bundling everything one call produces into ONE object, constructed fresh at the top
of chat()/chat_stream() and published as `self._request` in a single attribute assignment, means a
concurrent reader sees either the complete previous request's context or the complete current one -- never
a hand-mixed one -- once that assignment itself becomes the only cross-call mutation point.

SCOPE, deliberately conservative: this does NOT (yet) thread a context object through the call graph as an
explicit parameter -- that would touch every caller of chat()/score_tokens()/chat_stream() (replay.py,
rederive.py, the whole receipts stack) for no behavior change today and real risk to the byte-identical-
receipts invariant. It gets the "one coherent, atomically-swapped object instead of five scattered
attributes" win now, plus a real request id and a real cancellation primitive neither piecemeal attribute
scheme had room for. A future pass can thread RequestContext explicitly once the substrate itself is made
request-scoped instead of process-global.

Backward compat: EngineSubstrate exposes sub._last_generation_meta / _last_finish_reason / _last_diverged /
_last_diverged_at / _last_stream_trace as READ-ONLY @property views onto sub._request's fields (see
substrates.py) -- every existing test/consumer of those names keeps working unchanged, they just can no
longer be written directly (chat()/chat_stream() write the context's fields instead; nothing else ever
legitimately wrote them -- see the audit note in the backlog #2 commit that introduced this).
"""
from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field


def new_request_id() -> str:
    """A short, log-friendly per-request id, minted the moment a generation call BEGINS -- unlike the runs
    store's run_<ts>_<hex> id (runs/store.py's record()), which only exists after a run is successfully
    persisted, well after generation finished. Distinct namespace (`req_` vs `run_`) so the two are never
    confused; existing to eventually correlate one gateway call with the worker's own per-request `req`
    stamp on native SSE frames (engine/core/serve/server_shared.hpp's StreamEnvelope, guarded by
    protocol/fixtures/handshake.json) -- not wired to that yet (the worker's `req` lives in the native
    frame JSON chat_stream already parses but discards; a follow-up can thread it in here for exact
    gateway<->worker correlation in logs)."""
    return "req_" + secrets.token_hex(8)


@dataclass
class RequestContext:
    """One generation call's isolated state -- constructed fresh per chat()/chat_stream() call (see
    EngineSubstrate._new_request), then published onto the substrate as `self._request` in one shot. Never
    mutated by a SECOND call once a new one begins: a new call always builds and publishes its own instance,
    so a reference held by a caller (e.g. sse.py, to call .cancel() on the in-flight request) keeps
    describing THAT call even after a later one starts and replaces self._request with a different object.

    Fields (mirrors the backlog's "id, sampling, memory manifest, steering snapshot, trace, finish reason,
    cancellation" list exactly):
      request_id          -- new_request_id(), for log correlation.
      sampling             -- ctx._resolve_sampling()'s dict, or None for the greedy/forced-deterministic path.
      generation_meta      -- ctx._engine_generation_meta(...)'s reproducibility block (what _last_generation_meta aliases).
      memory_manifest       -- a snapshot of mem_out once the call has finished mutating it (prompt-mode block/
                              applied cards/gate/anchored bag(s)); the LIVE mem_out dict remains the primary,
                              already-per-call-isolated channel (callers own and read it directly) -- this is
                              an additional, consolidated copy on the context object for symmetry.
      steering_snapshot     -- the dial-strength dict THIS call actually used (self.steer.strength or the disk
                              fallback, copied at dispatch time) -- decoupled from the LIVE self.steer.strength,
                              which a concurrent /steer/set could still be mutating.
      trace                -- the per-token step list (what last_stream_trace() aliases).
      finish_reason         -- the engine's stop cause, or None (missing reads as missing, never as "stop").
      diverged/diverged_at  -- the prove-all early-stop verdict (what last_divergence() aliases).
      cancelled             -- a threading.Event; see cancel()/is_cancelled().
    """

    request_id: str = field(default_factory=new_request_id)
    sampling: dict | None = None
    generation_meta: dict = field(default_factory=dict)
    memory_manifest: dict = field(default_factory=dict)
    steering_snapshot: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)
    finish_reason: str | None = None
    diverged: bool | None = None
    diverged_at: int | None = None
    cancelled: threading.Event = field(default_factory=threading.Event)

    def cancel(self) -> None:
        """Signal that the client this request serves is gone. chat_stream's read loop checks
        is_cancelled() between worker frames and stops promptly instead of draining a reply nobody will
        receive; sse.py additionally calls gen.close() (throwing GeneratorExit at the generator's
        suspended `yield`) for an IMMEDIATE stop rather than waiting for the next frame boundary -- this
        Event is the belt to that close()'s suspenders, and the durable record that THIS request was
        cancelled (as opposed to merely having stopped) for anything that inspects the context later."""
        self.cancelled.set()

    def is_cancelled(self) -> bool:
        return self.cancelled.is_set()
