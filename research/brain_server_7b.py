"""brain_server_7b.py -- the BIGGER 'watch it think' backend: Qwen2.5-7B-Instruct + its 131k-feature
JumpReLU SAE (GPU, via sae7b). Serves inspector/demo/brain.html and, on POST /think {text}, returns:
  - acts      : which atlas features fire on the prompt (content tokens, BOS/sink positions masked)
  - concepts  : the top concept lobes by total activation
  - output    : the model's actual GENERATED answer to the prompt (chat)
So you type a prompt and watch the real 7B both light up its concepts AND answer.

Run (GPU venv):  C:/Users/brigi/src/cloze/.venv/Scripts/python.exe research/brain_server_7b.py 8090
Pair with the cloudflared tunnel already pointing at :8090.
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np                                                  # noqa: E402
import torch                                                        # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

from sae7b import DEV, GpuSAE, feats7b, load7b                      # noqa: E402

DEMO = os.path.join(HERE, "..", "inspector", "demo")
atlas = json.load(open(os.path.join(DEMO, os.environ.get("ATLAS_JSON", "atlas_emergent.json")), encoding="utf-8"))
CONCEPTS = atlas["meta"]["concepts"]
FIDS = [n["id"] for n in atlas["nodes"]]
FID2CONCEPT = {n["id"]: CONCEPTS[n["cluster"]] for n in atlas["nodes"]}
BRAIN_HTML = open(os.path.join(DEMO, "brain.html"), encoding="utf-8").read()
ARTIFACT_NNZ = 600

sae = GpuSAE()
tok, model = load7b()
print("brain-7b: model + SAE + atlas ready", flush=True)


@torch.no_grad()
def think(text: str) -> dict:
    # (a) which atlas features fire (raw prompt, same regime as the atlas; mask the BOS/sink positions)
    _, feats = feats7b(text, tok, model, sae)
    f = feats.cpu().numpy()
    f[(f > 0).sum(1) > ARTIFACT_NNZ] = 0
    fmax = f.max(0)
    acts, ctot = {}, {}
    for fid in FIDS:
        v = float(fmax[fid])
        if v > 0:
            acts[fid] = round(v, 3)
            c = FID2CONCEPT[fid]
            ctot[c] = ctot.get(c, 0.0) + v
    top = sorted(ctot.items(), key=lambda x: -x[1])[:6]
    # (b) the model's actual answer to the prompt (chat)
    ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                  return_tensors="pt").to(DEV)
    gen = model.generate(ids, max_new_tokens=70, do_sample=True, temperature=0.7, top_p=0.9,
                         pad_token_id=tok.eos_token_id)
    ans = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return {"acts": acts, "concepts": [{"name": c, "val": round(v, 1)} for c, v in top], "output": ans}


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
    print(f"brain-7b thinking server -> http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
