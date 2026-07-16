"""HTTP workbench for PyTorch-only Qwen and Dream experiments.

This intentionally has no OpenAI or Clozn-native generation API.  It exists to keep
training/calibration and the research visualizations runnable without turning the lab
model into a second product-serving engine.
"""
from __future__ import annotations

import argparse
import os
import sys
from http.server import ThreadingHTTPServer

# A lab process must never inherit a handle to a product worker. Do this before the first app import;
# avoid mutating an already-running product module when tests merely import this module for its handler.
if "clozn.server.app" not in sys.modules:
    os.environ.pop("CLOZN_ENGINE_PORT", None)
    os.environ["CLOZN_RUNTIME_KIND"] = "lab"

# The lab -- and ONLY the lab -- needs the PyTorch research deps reachable: engine/lab so the Dream
# substrate can `import cloze_lab`, plus the HF hub symlink workaround. These used to load at PRODUCT
# import time (clozn/server/config.py); they moved here so a `clozn serve` process never pulls them in.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ENGINE_LAB = os.path.join(_REPO_ROOT, "engine", "lab")
if _ENGINE_LAB not in sys.path:
    sys.path.insert(0, _ENGINE_LAB)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

from clozn.server import app as ctx


def make_lab_handler():
    base = ctx.make_handler()

    class LabHandler(base):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/healthz", "/readyz"):
                self._json(200, {"status": "ok", "service": "clozn-lab", "active": ctx.SUBNAME})
                return
            if path == "/substrate":
                self._json(200, {"active": ctx.SUBNAME, "available": [ctx.SUBNAME], "service": "clozn-lab"})
                return
            if path.startswith("/v1/") or path.startswith("/api/clozn/"):
                self._json(404, {"error": "the lab workbench does not expose a product generation API"})
                return
            super().do_GET()

        def do_POST(self):
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/substrate":
                self._json(410, {"error": "restart the lab command to choose another workbench"})
                return
            if path.startswith("/v1/") or path.startswith("/api/clozn/"):
                self._json(404, {"error": "the lab workbench does not expose a product generation API"})
                return
            super().do_POST()

    return LabHandler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Clozn's optional PyTorch workbench")
    parser.add_argument("substrate", choices=("qwen", "dream"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args(argv)

    os.environ["CLOZN_RUNTIME_KIND"] = "lab"
    ctx.RUNTIME_KIND = "lab"
    ctx.ARGS = args
    ctx.SUBNAME = args.substrate
    print(f"clozn lab: loading {args.substrate} ...", flush=True)
    from clozn.lab.substrates import QwenSubstrate, DreamSubstrate
    ctx.SUB = QwenSubstrate() if args.substrate == "qwen" else DreamSubstrate()
    server = ThreadingHTTPServer((args.host, args.port), make_lab_handler())
    print(f"\n  Clozn lab -> http://{args.host}:{args.port}/ ({args.substrate})\n", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
