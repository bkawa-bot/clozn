"""Standalone developer server for the SelfTeach substrate."""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from clozn.lab.substrates.self_teach import SelfTeach

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def make_handler(app: SelfTeach):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            if self.path == "/state":
                self._send(200, app.state())
            else:
                self._send(404, {"error": "GET " + self.path})

        def do_POST(self):
            try:
                b = self._body()
                if self.path == "/say":
                    self._send(200, {"reply": app.say(b["message"], b.get("max_new", 220))})
                elif self.path == "/consolidate":
                    self._send(200, app.consolidate(b.get("rules"), b.get("steps", 120), b.get("lr", 0.012),
                                                    b.get("n_probe", 8), b.get("max_norm", 14.0)))
                elif self.path == "/whatlearned":
                    self._send(200, {"report": app.what_learned()})
                elif self.path == "/check":
                    self._send(200, app.check(b["prompt"], b.get("max_new", 200)))
                elif self.path == "/trace":
                    self._send(200, app.trace(b["prompt"], b.get("max_new", 80)))
                elif self.path == "/reset":
                    self._send(200, app.reset(b.get("keep_prefix", False)))
                else:
                    self._send(404, {"error": "POST " + self.path})
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--port", type=int, default=8079)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--m", type=int, default=16, help="soft-prefix length")
    ap.add_argument("--bf16", action="store_true", help="load bf16 instead of 4-bit (small models)")
    args = ap.parse_args()
    app = SelfTeach(args.model, m=args.m, four_bit=not args.bf16)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"\n  SELF-TEACH server -> http://{args.host}:{args.port}", flush=True)
    print("  /say /consolidate /whatlearned /check /reset  (GET /state)\n", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
