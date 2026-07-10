"""Standalone developer server for the Dream denoise trace helper."""
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from clozn.substrates.denoise import trace_for
from cloze_lab.cli import build_adapter  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
UI = ROOT / "studio" / "denoise.html"


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
                self._send(200, UI.read_text(encoding="utf-8"), "text/html; charset=utf-8")
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
