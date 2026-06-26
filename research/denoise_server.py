"""denoise_server.py -- LIVE diffusion: type a prompt, watch Dream-7B denoise it in real time.

Loads the Dream-7B adapter once, serves inspector/demo/denoise.html, and on POST /denoise {prompt} runs a
real denoise and returns the pass-by-pass trace (reconstructed from the event spine) for the viz to play.

    PYTHONPATH=engine/lab cloze .venv python research/denoise_server.py --port 8082
then open http://127.0.0.1:8082/
"""
import argparse
import json
import os
import sys
import threading

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine", "lab"))

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer   # noqa: E402

from cloze_lab.cli import build_adapter                              # noqa: E402
from cloze_lab.generate import GenerateConfig, generate              # noqa: E402
from cloze_lab.scheduler.events import GenStarted, TokensCommitted   # noqa: E402

UI = os.path.join(HERE, "..", "inspector", "demo", "denoise.html")
LOCK = threading.Lock()


def trace_for(adapter, prompt, max_new=48, steps=24):
    with LOCK:
        ids = adapter.encode(prompt, chat=True)
        cfg = GenerateConfig(max_new=max_new, steps=steps, temperature=0.0, seed=0, block_len=0)
        res = generate(adapter, ids, cfg)
        n_prompt = board_len = None
        passes, pi = [], 0
        for e in res.events:
            if isinstance(e, GenStarted):
                n_prompt, board_len = e.prompt_tokens, e.prompt_tokens + e.max_new
            elif isinstance(e, TokensCommitted):
                items = [{"pos": int(it.pos), "piece": adapter.decode([int(it.id)]), "conf": round(float(it.conf), 3)}
                         for it in e.items]
                if items:
                    passes.append({"pass": pi, "items": items})
                    pi += 1
        return {"model": "Dream-v0-Instruct-7B", "prompt": prompt,
                "prompt_text": adapter.decode([int(i) for i in ids]),
                "n_prompt": n_prompt, "board_len": board_len, "steps": steps,
                "final_text": res.text, "passes": passes}


def make_handler(adapter):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            b = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/index.html", "/denoise.html"):
                self._send(200, open(UI, encoding="utf-8").read(), "text/html; charset=utf-8")
            else:
                self._send(404, json.dumps({"error": "GET " + self.path}))

        def do_POST(self):
            if self.path.rstrip("/").endswith("denoise"):
                n = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(n) or b"{}")
                try:
                    self._send(200, json.dumps(trace_for(adapter, str(body.get("prompt", ""))[:300])))
                except Exception as e:
                    self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}))
            else:
                self._send(404, json.dumps({"error": "POST " + self.path}))

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    print("loading Dream-7B (4-bit) ...", flush=True)
    adapter = build_adapter("dream", device="cuda", quant="nf4")
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(adapter))
    print(f"\n  DENOISE (live) -> http://{args.host}:{args.port}/\n", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
