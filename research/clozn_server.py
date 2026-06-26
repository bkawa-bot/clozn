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

ARGS = None
SUB = None         # the active substrate object
SUBNAME = "qwen"


class QwenSubstrate:
    """One Qwen-7B + SAE behind both the concept readout and the memory."""
    name = "qwen"

    def __init__(self):
        from brain_readout import BrainReadout
        from sae7b import GpuSAE, load7b
        from self_teach_server import SelfTeach
        sae = GpuSAE()
        tok, model = load7b()
        self.brain = BrainReadout(model, tok, sae, DEMO, HERE)
        self.memory = SelfTeach("Qwen/Qwen2.5-7B-Instruct", model=model, tok=tok)   # shares the model

    def handle(self, path, body):
        if path == "/think":
            return self.brain.think(str(body.get("text", ""))[:500], str(body.get("sid", "default")))
        if path == "/say":
            return {"reply": self.memory.say(body["message"], body.get("max_new", 200))}
        if path == "/consolidate":
            return self.memory.consolidate(body.get("rules"), body.get("steps", 55), body.get("lr", 0.03))
        if path == "/whatlearned":
            return {"report": self.memory.what_learned()}
        if path == "/check":
            return self.memory.check(body["prompt"], body.get("max_new", 200))
        if path == "/reset":
            self.brain.reset(str(body.get("sid", "default")))
            return self.memory.reset(body.get("keep_prefix", False))
        return None

    def state(self):
        return self.memory.state()


class DreamSubstrate:
    """Dream-7B for the diffusion window."""
    name = "dream"

    def __init__(self):
        from cloze_lab.cli import build_adapter
        from denoise_server import trace_for
        self.adapter = build_adapter("dream", device="cuda", quant="nf4")
        self._trace = trace_for

    def handle(self, path, body):
        if path == "/denoise":
            return self._trace(self.adapter, str(body.get("prompt", ""))[:300])
        return None

    def state(self):
        return {}


def load_substrate(name):
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

        def do_GET(self):
            p = self.path.split("?")[0]
            if p in ("/", "/index.html", "/instrument.html"):
                return self._html("instrument.html")
            if p == "/substrate":
                return self._json(200, {"active": SUBNAME, "available": ["qwen", "dream"]})
            if p == "/state":
                return self._json(200, {"substrate": SUBNAME, **(SUB.state() if SUB else {})})
            if p.endswith(".html"):
                fn = os.path.join(DEMO, os.path.basename(p))
                if os.path.isfile(fn):
                    return self._html(os.path.basename(p))
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
    ap.add_argument("--substrate", default="qwen", choices=("qwen", "dream"))
    ARGS = ap.parse_args()
    SUBNAME = ARGS.substrate
    print(f"clozn server: loading '{SUBNAME}' substrate ...", flush=True)
    SUB = load_substrate(SUBNAME)
    srv = ThreadingHTTPServer((ARGS.host, ARGS.port), make_handler())
    print(f"\n  CLOZN instrument -> http://{ARGS.host}:{ARGS.port}/   (substrate: {SUBNAME})\n", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
