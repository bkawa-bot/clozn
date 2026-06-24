"""brain_server.py -- the "watch it think" backend for the brain viz.

Serves inspector/demo/brain.html at / (so the prompt box's fetch('think') is same-origin, no CORS pain),
and on POST /think {text} runs the text through Qwen3-1.7B-Base + the Qwen-Scope SAE and returns, for
each of the atlas's verified concept features, its activation on that text (max over tokens). The viz
flares those nodes -- the brain literally lights up with the concepts the model engages for your prompt.

Pair with an ssh tunnel so a remote browser can reach it:
    C:/Users/brigi/src/clozn/.venv-sae/Scripts/python.exe research/brain_server.py 8090
    ssh -o StrictHostKeyChecking=accept-new -R 80:localhost:8090 nokey@localhost.run
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                   # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

from concept_readout import feats_for, load           # noqa: E402

DEMO = os.path.join(HERE, "..", "inspector", "demo")
atlas = json.load(open(os.path.join(DEMO, "atlas.json"), encoding="utf-8"))
CONCEPTS = atlas["meta"]["concepts"]
FIDS = [n["id"] for n in atlas["nodes"]]
FID2CONCEPT = {n["id"]: CONCEPTS[n["cluster"]] for n in atlas["nodes"]}
BRAIN_HTML = open(os.path.join(DEMO, "brain.html"), encoding="utf-8").read()

sae, sdtype, tok, model = load()
print("brain server: model + atlas ready", flush=True)


def think(text: str) -> dict:
    _, feats = feats_for(text, sae, sdtype, tok, model)
    fmax = feats.detach().cpu().float().numpy().max(0)          # peak activation per feature over the text
    acts, concept_tot = {}, {}
    for fid in FIDS:
        v = float(fmax[fid])
        if v > 0:
            acts[fid] = round(v, 3)
            c = FID2CONCEPT[fid]
            concept_tot[c] = concept_tot.get(c, 0.0) + v
    top = sorted(concept_tot.items(), key=lambda x: -x[1])[:5]
    return {"acts": acts, "concepts": [{"name": c, "val": round(v, 2)} for c, v in top]}


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

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/index.html", "/brain.html"):
            self._send(200, BRAIN_HTML, "text/html; charset=utf-8")
        else:
            self._send(404, "not found", "text/plain")

    def do_POST(self):
        if self.path.rstrip("/").endswith("think"):
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            try:
                self._send(200, json.dumps(think(str(body.get("text", ""))[:500])), "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}), "application/json")
        else:
            self._send(404, "not found", "text/plain")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    print(f"brain thinking server -> http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
