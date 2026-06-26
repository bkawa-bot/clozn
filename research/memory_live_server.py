"""memory_live_server.py -- the INTERACTIVE 'watch it remember' window. Chat live with a frozen local
Qwen-7B; press 'sleep on it' to distil the conversation into a 16-vector memory (test-time training);
then ask what it learned, or test a fresh prompt to see the memory move its behaviour. Wraps the
SelfTeach rig (self_teach_server) and serves inspector/demo/memory_chat.html.

    cloze .venv python research/memory_live_server.py --port 8081
then open http://127.0.0.1:8081/
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer   # noqa: E402

from self_teach_server import SelfTeach                              # noqa: E402

UI = os.path.join(HERE, "..", "inspector", "demo", "memory_chat.html")


def make_handler(app):
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

        def _json(self, code, obj):
            self._send(code, json.dumps(obj))

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            if self.path.split("?")[0] in ("/", "/index.html"):
                self._send(200, open(UI, encoding="utf-8").read(), "text/html; charset=utf-8")
            elif self.path == "/state":
                self._json(200, app.state())
            else:
                self._json(404, {"error": "GET " + self.path})

        def do_POST(self):
            try:
                b = self._body()
                if self.path == "/say":
                    self._json(200, {"reply": app.say(b["message"], b.get("max_new", 200))})
                elif self.path == "/consolidate":
                    self._json(200, app.consolidate(b.get("rules"), b.get("steps", 55), b.get("lr", 0.03)))
                elif self.path == "/whatlearned":
                    self._json(200, {"report": app.what_learned()})
                elif self.path == "/check":
                    self._json(200, app.check(b["prompt"], b.get("max_new", 200)))
                elif self.path == "/reset":
                    self._json(200, app.reset(b.get("keep_prefix", False)))
                else:
                    self._json(404, {"error": "POST " + self.path})
            except Exception as e:
                self._json(500, {"error": f"{type(e).__name__}: {e}"})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    app = SelfTeach(args.model, m=16)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"\n  MEMORY (live chat) -> http://{args.host}:{args.port}/\n", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
