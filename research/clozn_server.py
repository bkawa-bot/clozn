"""clozn_server.py -- the UNIFIED instrument. One port, one model, the whole white-box surface.

  substrate 'qwen' (default): ONE Qwen-7B serves BOTH the brain (/think -- concepts the model engages)
                              AND the memory (/say /consolidate /check /whatlearned) -- they share the
                              single loaded model, so the instrument's brain and memory tabs are both live.
  substrate 'dream':          Dream-7B serves /denoise (the diffusion window).

Only one 7B fits the GPU, so switching substrates re-execs the process with the other one (a clean GPU);
the instrument shows the active substrate and offers the switch. Serves the instrument + every window
from inspector/demo, so the iframes' fetches all land here.

    cloze .venv python research/clozn_server.py --port 8090
"""
import argparse
import json
import os
import sys
import threading
import time

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "engine", "lab"))   # so the dream substrate can import cloze_lab
DEMO = os.path.join(HERE, "..", "inspector", "demo")

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


def _pers(name):
    return os.path.join(CLOZN_DIR, name)


ENGINE_STEER = None        # lazy EngineSteer on the Qwen GGUF engine -- tone dials on the C++ runtime, any GGUF


def _engine_steer():
    global ENGINE_STEER
    if ENGINE_STEER is None and ENGINE_QWEN is not None:
        from steering import EngineSteer
        ENGINE_STEER = EngineSteer(ENGINE_QWEN)
    return ENGINE_STEER


def _qwen_tmpl(messages):
    """Render chat messages into Qwen's chat-template STRING for the engine's raw /v1/completions -- the
    same template the HF memory prefix was trained against, so the injected prefix lands in the right context."""
    sysmsg = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    for m in messages:
        if m.get("role") == "system" and m.get("content"):
            sysmsg = m["content"]
    s = f"<|im_start|>system\n{sysmsg}<|im_end|>\n"
    for m in messages:
        if m.get("role") in ("user", "assistant"):
            s += f"<|im_start|>{m['role']}\n{m.get('content', '')}<|im_end|>\n"
    return s + "<|im_start|>assistant\n"


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


ARGS = None
SUB = None         # the active substrate object
SUBNAME = "qwen"


class Substrate:
    """Shared studio surface for any substrate: the /memory/* trait cards and the /steer/* tone dials, on
    whatever model the subclass loads. A subclass sets self.steer, self._mem (a memory object exposing
    .rules / .prefix / .consolidate(rules) / .reset()), self._pers_steer, self._steer_ready/_steer_info,
    and defines _gen(prompt) -- a one-shot generate used by the /steer/check A/B (AR generate vs denoise).
    So memory + dials are written ONCE and work identically on Qwen and Dream."""

    def _memory(self, path, body):
        m = self._mem
        if path == "/memory/cards":
            return {"cards": m.rules, "has_prefix": m.prefix is not None}
        if path == "/memory/add":               # add a trait card -> re-consolidate the cumulative set
            text = str(body.get("text", "")).strip()
            return m.consolidate(list(m.rules) + [text]) if text else {"ok": False, "reason": "empty trait"}
        if path == "/memory/remove":            # drop a card -> rebuild the prefix from the rest
            remaining = [r for i, r in enumerate(m.rules) if i != int(body.get("index", -1))]
            m.reset()
            return m.consolidate(remaining) if remaining else {"ok": True, "cards": []}
        if path == "/memory/strength":          # the memory dial: how hard the prefix bites (0 = off, >1 = stronger)
            if "value" in body and hasattr(m, "memory_strength"):
                m.memory_strength = max(0.0, min(2.0, float(body["value"])))
                if hasattr(m, "save"):
                    try:
                        m.save()
                    except Exception:
                        pass
            return {"strength": float(getattr(m, "memory_strength", 1.0)), "has_prefix": m.prefix is not None}
        return None

    def _steer(self, path, body):
        from steering import AXES
        if path == "/steer/axes":
            return {"axes": [{"name": k, "poles": AXES[k]["poles"], "value": self.steer.strength.get(k, 0.0)}
                             for k in AXES], "ready": self._steer_ready, "substrate": self.name}
        if not self._steer_ready:               # compute the axis vectors on first real use
            self._steer_info = self.steer.compute()
            self._steer_ready = True
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
        return None


class QwenSubstrate(Substrate):
    """One Qwen-7B + SAE behind the concept readout AND the memory + tone dials."""
    name = "qwen"

    def __init__(self):
        from brain_readout import BrainReadout
        from sae7b import GpuSAE, load7b
        from self_teach_server import SelfTeach
        from steering import SteeringControl
        sae = GpuSAE()
        tok, model = load7b()
        self.brain = BrainReadout(model, tok, sae, DEMO, HERE)
        self.memory = SelfTeach("Qwen/Qwen2.5-7B-Instruct", model=model, tok=tok,   # shares the model
                                persist_path=_pers("studio_memory.pt"))
        self.steer = SteeringControl(model, tok)            # tone dials on the same model
        self._mem = self.memory
        self._steer_ready, self._steer_info = False, {}
        self._pers_steer = _pers("studio_personality.json")
        self.steer.load_state(self._pers_steer)             # restore the personality dials across restarts

    def handle(self, path, body):
        if path == "/think":
            return self.brain.think(str(body.get("text", ""))[:500], str(body.get("sid", "default")))
        if path == "/concepts":                 # read what fired inside (no generation) -> annotate a reply
            return self.brain.concepts_only(str(body.get("text", ""))[:500])
        if path == "/say":
            return {"reply": self.memory.say(body["message"], body.get("max_new", 200))}
        if path == "/consolidate":
            return self.memory.consolidate(body.get("rules"), body.get("steps", 120), body.get("lr", 0.012),
                                           body.get("n_probe", 8), body.get("max_norm", 14.0))
        if path == "/whatlearned":
            return {"report": self.memory.what_learned()}
        if path == "/check":
            return self.memory.check(body["prompt"], body.get("max_new", 200))
        if path == "/reset":
            self.brain.reset(str(body.get("sid", "default")))
            return self.memory.reset(body.get("keep_prefix", False))
        if path.startswith("/memory/"):
            return self._memory(path, body)
        if path.startswith("/steer/"):
            return self._steer(path, body)
        return None

    def _gen(self, prompt):                     # AR generate for the /steer/check A/B
        return self.steer.generate(prompt, 90)

    def chat(self, messages, max_new=256, sample=True):
        """One stateless chat completion with the WHOLE tunable self applied: the consolidated memory
        prefix (learned + added traits) AND the active tone-steering sliders, both on the shared model.
        This is what the OpenAI-compatible endpoint serves -- normal chat on the surface, legible and
        tunable underneath."""
        if self.steer.strength and not self._steer_ready:   # persisted personality -> ensure vectors are ready
            self.steer.compute()
            self._steer_ready = True
        self.steer.engage()
        try:
            return self.memory._generate(messages, use_prefix=True, max_new=max_new, sample=sample,
                                         gate=self.memory.memory_strength)
        finally:
            self.steer.disengage()

    def chat_stream(self, messages, max_new=256):
        """Streaming chat: yields text chunks as the AR model generates -- memory prefix + tone steering
        applied -- via a TextIteratorStreamer with generate() in a thread. Local AR is slow, so this is
        the big UX win the diffusion side doesn't need (diffusion is trace-based, not left-to-right)."""
        import threading
        import torch
        from transformers import TextIteratorStreamer
        if self.steer.strength and not self._steer_ready:
            self.steer.compute()
            self._steer_ready = True
        m = self.memory
        e = m._embed(m._chat_ids(messages))
        if m.prefix is not None:                            # prepend the consolidated memory prefix (scaled by the dial)
            e = torch.cat([(m.memory_strength * m.prefix.detach()).to(e.dtype)[None], e], 1)
        att = torch.ones(e.shape[:2], device=e.device, dtype=torch.long)
        streamer = TextIteratorStreamer(m.tok, skip_prompt=False, skip_special_tokens=True)
        kw = dict(inputs_embeds=e, attention_mask=att, max_new_tokens=max_new, do_sample=True,
                  temperature=0.7, top_p=0.9, pad_token_id=m.eos or 0, streamer=streamer)

        def _gen():
            with torch.no_grad():
                m.model.generate(**kw)

        self.steer.engage()                                 # tone dials apply during the streamed generation
        th = threading.Thread(target=_gen, daemon=True)
        th.start()
        try:
            for chunk in streamer:
                if chunk:
                    yield chunk
        finally:
            th.join()
            self.steer.disengage()

    def state(self):
        return self.memory.state()


class DreamSubstrate(Substrate):
    """Dream-7B diffusion: the denoise window, plus the SAME trait-card memory and tone dials as Qwen."""
    name = "dream"

    def __init__(self):
        from cloze_lab.cli import build_adapter
        from denoise_server import trace_for
        from steering import DreamSteering
        from dream_memory import DreamMemory
        self.adapter = build_adapter("dream", device="cuda", quant="nf4")
        self._trace = trace_for
        self.steer = DreamSteering(self.adapter)            # tone dials on the diffusion model
        self._steer_ready, self._steer_info = False, {}
        self._pers_steer = _pers("studio_dream_personality.json")
        self.steer.load_state(self._pers_steer)
        self.dmem = DreamMemory(self.adapter,               # diffusion-native memory (trained soft prefix)
                                persist_path=_pers("studio_dream_memory.pt"))
        self._mem = self.dmem

    def handle(self, path, body):
        if path == "/denoise":
            prompt = str(body.get("prompt", ""))[:300]
            self.steer.engage()                            # active dials steer every denoising pass
            try:
                ad = self.adapter
                if self.dmem.prefix is not None:           # memory present -> inject the prefix into the REAL scheduler
                    from dream_memory import PrefixAdapter
                    ad = PrefixAdapter(self.adapter, self.dmem.prefix.detach())
                return self._trace(ad, prompt)             # the cloze_lab scheduler (+ the steering hook)
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


def load_substrate(name):
    if name == "engine":
        return None        # pure-engine: NO HF model -- serve the GGUF via the C++ engine + the saved prefix from disk
    return QwenSubstrate() if name == "qwen" else DreamSubstrate()


def switch_substrate(name):
    """Re-exec the whole process with the new substrate -> a clean GPU (the only honest way; one 7B fits)."""
    py = sys.executable
    os.execv(py, [py, os.path.abspath(__file__), "--substrate", name, "--port", str(ARGS.port),
                  "--host", ARGS.host])


def make_handler():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            b = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _json(self, code, o):
            self._send(code, json.dumps(o), "application/json")

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

            try:
                chunk({"role": "assistant"})
                for piece in SUB.chat_stream(messages, max_new):
                    chunk({"content": piece})
                chunk({}, finish="stop")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except Exception as e:
                try:
                    self.wfile.write(("data: " + json.dumps({"error": str(e)}) + "\n\n").encode("utf-8"))
                    self.wfile.flush()
                except Exception:
                    pass

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
                return self._json(200, {"substrate": SUBNAME, **(SUB.state() if SUB else {})})
            if p.endswith((".html", ".css", ".js")):
                fn = os.path.join(DEMO, os.path.basename(p))
                if os.path.isfile(fn):
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
            if p == "/engine/harvest":   # READ the real C++ runtime's activations (any substrate; the engine is separate)
                try:
                    h = ENGINE.harvest(str(body.get("text", ""))[:300])
                    norms = np.linalg.norm(h.activations, axis=1)
                    return self._json(200, {"tokens": h.tokens, "layer": int(h.layer), "n_embd": h.n_embd,
                                            "norms": [round(float(x), 3) for x in norms]})
                except Exception as e:
                    return self._json(502, {"error": f"engine: {e}"})
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
                from steering import AXES
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
                try:
                    prompt = _qwen_tmpl(body.get("messages", []))
                    mx = int(body.get("max_tokens", 220))
                    kw = {}
                    # MEMORY: the live HF prefix if a qwen substrate is loaded, else the SAVED prefix from disk
                    # -- so engine-chat works with NO HF model resident (the pure-engine substrate).
                    mem = getattr(SUB, "memory", None) if SUB else None
                    if mem is not None and getattr(mem, "prefix", None) is not None:
                        prefix = mem.prefix.detach().float().cpu()
                        ms = float(getattr(mem, "memory_strength", 1.0))
                    else:
                        prefix, ms = _disk_memory()
                    if prefix is not None:                         # inject the trained soft prefix (scaled by the dial)
                        kw = {"prefix_embd": (prefix * ms).flatten().tolist(), "prefix_rows": int(prefix.shape[0])}
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
                    r = ENGINE_QWEN.complete(prompt, max_tokens=mx, temperature=0.0, **kw)
                    ch = r.get("choices") if isinstance(r, dict) else None
                    reply = (ch[0].get("text", "") if ch else str(r)).strip()
                    return self._json(200, {"reply": reply, "memory": bool(kw.get("prefix_embd")),
                                            "tone": bool(kw.get("steer_vec")), "via": "engine (GGUF)"})
                except Exception as e:
                    return self._json(502, {"error": f"engine-chat: {e}"})
            if p == "/v1/chat/completions":   # OpenAI-compatible: chat with memory prefix + tone steering applied
                if not (SUB and getattr(SUB, "chat", None)):
                    return self._json(503, {"error": "chat needs the qwen substrate"})
                msgs, mx = body.get("messages", []), int(body.get("max_tokens", 256))
                if body.get("stream") and getattr(SUB, "chat_stream", None):
                    return self._sse_chat(msgs, mx, str(body.get("model", "clozn-qwen")))
                reply = SUB.chat(msgs, mx, float(body.get("temperature", 0.7)) > 0)
                return self._json(200, {"id": "chatcmpl-clozn", "object": "chat.completion",
                                        "created": int(time.time()), "model": body.get("model", "clozn-qwen"),
                                        "choices": [{"index": 0, "finish_reason": "stop",
                                                     "message": {"role": "assistant", "content": reply}}],
                                        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}})
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
