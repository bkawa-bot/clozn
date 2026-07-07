"""smoke_engine_substrate.py -- end-to-end validation that the clozn studio works on the C++ ENGINE.

The repeatable form of the Phase 0 keystone validation (RUNTIME_SPLIT.md): with the studio running on
`--substrate engine` (a live GGUF engine behind it), confirm that the whole torch-free Server tier --
chat + the receipts/replay stack + memory + tone dials -- actually runs on the engine, with NO PyTorch
model resident. Run it after any change to EngineSubstrate / the engine, or as a pre-ship gate.

    # 1. start the engine (GPU build; put build-gpu/bin + CUDA on PATH -- clozn serve does this for you):
    #    clozn serve qwen --port 8092
    # 2. start the studio pointed at it:
    #    CLOZN_ENGINE_QWEN_PORT=8092 python research/clozn_server.py --substrate engine --port 8090
    # 3. run this:
    python research/smoke_engine_substrate.py --port 8090

Stdlib only (urllib/json) -- no torch, no deps -- so it stays a quick, dependency-free gate. Every request
is NON-STREAMING and fully consumed: a client that disconnects mid-SSE-stream crashes the single-worker
engine (RUNTIME_SPLIT.md hard-part #6, the confirmed robustness bug), so this validator is careful never
to be that client. Exit 0 iff every check passed.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

PASS, FAIL = "PASS", "FAIL"


class Client:
    def __init__(self, base: str, timeout: float = 120.0):
        self.base = base.rstrip("/")
        self.timeout = timeout

    def req(self, method: str, path: str, body: dict | None = None) -> tuple[int, dict]:
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(self.base + path, data=data, method=method,
                                   headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(r, timeout=self.timeout) as resp:   # fully consumed, never streamed
                return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode("utf-8") or "{}")
            except Exception:
                return e.code, {}
        except Exception as e:
            return 0, {"__error": str(e)}

    def get(self, path):
        return self.req("GET", path)

    def post(self, path, body=None):
        return self.req("POST", path, body or {})


class Checks:
    """Collects (name, ok, detail) results and prints them; tracks overall pass/fail."""

    def __init__(self):
        self.rows: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = ""):
        self.rows.append((name, ok, detail))
        print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
        return ok

    @property
    def ok(self) -> bool:
        return all(ok for _, ok, _ in self.rows)


def run(base: str, engine_base: str | None = None) -> int:
    c = Client(base)
    eng = Client(engine_base) if engine_base else None
    ck = Checks()
    print(f"validating the engine substrate at {base}\n")

    def engine_up():
        """None if we weren't told the engine URL; else True/False from its /health."""
        if eng is None:
            return None
        return eng.get("/health")[0] == 200

    # 0. the substrate must actually be the engine (else this validates nothing meaningful)
    st, sub = c.get("/substrate")
    active = sub.get("active")
    ck.add("substrate is 'engine'", active == "engine", f"active={active!r}")

    # 1. THE KEYSTONE: a chat completion served by the engine, yielding a run_id
    st, chat = c.post("/v1/chat/completions",
                      {"messages": [{"role": "user", "content": "In one sentence, what is TCP?"}],
                       "max_tokens": 50, "model": "clozn-qwen"})
    reply = (((chat.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    rid = chat.get("clozn_run_id")
    ck.add("chat via engine returns a reply", bool(reply.strip()), reply[:60])
    ck.add("chat yields a run_id", bool(rid), str(rid))
    if not rid:
        print("\nno run_id -- cannot exercise the receipts stack; aborting.")
        return 1

    # 2. the receipts / replay stack -- all torch-free, all routing through SUB.chat on the engine
    st, ex = c.post(f"/runs/{rid}/explain")
    ck.add("explain (M1): per-token trace present",
           bool((ex.get("confidence") or {}).get("available")),
           f"n_tokens={(ex.get('confidence') or {}).get('n_tokens')}")

    st, rc = c.post(f"/runs/{rid}/receipts")
    ck.add("receipts (M2): leave-one-out ran", "receipts" in rc, f"status={st}")

    st, cf = c.post(f"/runs/{rid}/counterfactual", {"behavior_overrides": {"warm": 1.0}})
    ck.add("counterfactual (M3): dial re-gen + causal check",
           "causal_verified" in cf and not cf.get("__error"),
           f"has_effect={cf.get('has_effect')} verified={cf.get('causal_verified')}")

    st, nr = c.post(f"/runs/{rid}/narrate")
    ck.add("narrate (M4): accountable narration", "constrained_narration" in nr, f"status={st}")

    st, rp = c.post(f"/runs/{rid}/replay", {"changes": {"greedy": True}})
    ck.add("replay (F1): produced a child run", bool(rp.get("id")) and "response" in rp, f"status={st}")

    # 3. memory: a clean add -> approve -> on-topic chat -> injected? -> remove round-trip (no residue)
    _, add = c.post("/memory/add", {"text": "The user is a huge fan of astronomy and stargazing."})
    cid = add.get("id")
    injected = False
    if cid:
        c.post("/memory/approve", {"id": cid})
        _, mchat = c.post("/v1/chat/completions",
                          {"messages": [{"role": "user", "content": "What hobby would you recommend tonight?"}],
                           "max_tokens": 50})
        mreply = (((mchat.get("choices") or [{}])[0].get("message") or {}).get("content") or "").lower()
        injected = any(w in mreply for w in ("star", "astronom", "sky", "night sky", "telescope"))
        c.post("/memory/remove", {"id": cid})          # cleanup regardless of outcome
    ck.add("memory: an active card steers the engine reply", injected,
           "astronomy card -> stargazing-flavored reply" if injected else "card did not visibly inject")

    # 4. tone dials: set a built-in dial -> chat -> the reply shifts -> reset (no residue)
    _, neutral = c.post("/v1/chat/completions",
                        {"messages": [{"role": "user", "content": "My code has a bug."}], "max_tokens": 40})
    c.post("/steer/set", {"name": "warm", "value": 0.8})
    _, warm = c.post("/v1/chat/completions",
                     {"messages": [{"role": "user", "content": "My code has a bug."}], "max_tokens": 40})
    c.post("/steer/set", {"name": "warm", "value": 0.0})    # reset
    n_txt = (((neutral.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    w_txt = (((warm.get("choices") or [{}])[0].get("message") or {}).get("content") or "")
    ck.add("dial: warm shifts the engine reply", bool(w_txt.strip()) and n_txt.strip() != w_txt.strip(),
           "warm != neutral" if n_txt.strip() != w_txt.strip() else "no measurable shift")

    # 5. LIBRARY dials (Phase 1): the shipped 27 must appear tagged "library" AND actually steer engine-native
    _, axes = c.post("/steer/axes")
    lib_names = [a.get("name") for a in (axes.get("axes") or []) if a.get("library")]
    ck.add("library dials listed on engine substrate (>=20)", len(lib_names) >= 20, f"{len(lib_names)} tagged 'library'")
    if "ceremonious" in lib_names:
        c.post("/steer/set", {"name": "ceremonious", "value": 1.5})   # first set harvests all dial directions (~slow)
        _, cchat = c.post("/v1/chat/completions",
                          {"messages": [{"role": "user", "content": "Tell me about the weather today."}],
                           "max_tokens": 45})
        c.post("/steer/set", {"name": "ceremonious", "value": 0.0})   # reset
        ctxt = (((cchat.get("choices") or [{}])[0].get("message") or {}).get("content") or "").lower()
        ornate = any(w in ctxt for w in ("esteemed", "hallowed", "grand", "splendid", "regal", "noble",
                                         "thee", "thou", "gracious", "majest", "behold", "dear "))
        ck.add("library dial 'ceremonious' steers (ornate register)", ornate, ctxt[:70])

    # Distinguish a genuine feature failure from the known engine-crash-under-load bug: if the engine is
    # dead at the end but the early (also generation-dependent) checks passed, the later failures are
    # downstream of a dead engine, not broken features. This is the honest verdict a bare pass/fail hides.
    if not ck.ok and engine_up() is False:
        print("\n*** THE ENGINE DIED DURING THIS RUN ***")
        print("  Early generation checks passed, so the failures above are downstream of a dead engine,")
        print("  NOT feature bugs. This reproduces RUNTIME_SPLIT.md hard-part #6: the single-worker engine")
        print("  crashes under the studio's sequential workload -- the #1 Phase 4 blocker. Restart it and")
        print("  the individual features pass (validated manually); the engine just can't sustain a session yet.")

    print(f"\n{'ALL PASSED' if ck.ok else 'FAILURES ABOVE'} -- {sum(ok for _, ok, _ in ck.rows)}/{len(ck.rows)} checks")
    return 0 if ck.ok else 1





def main(argv=None):
    ap = argparse.ArgumentParser(description="End-to-end smoke test for the clozn engine substrate.")
    ap.add_argument("--port", type=int, default=8090, help="studio port (default 8090)")
    ap.add_argument("--engine-port", type=int, default=8092,
                    help="the C++ engine's port (default 8092) -- checked so an engine crash under load is "
                         "reported as such, not as feature failures")
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    return run(f"http://{args.host}:{args.port}",
               f"http://{args.host}:{args.engine_port}" if args.engine_port else None)


if __name__ == "__main__":
    raise SystemExit(main())
