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
from atlas_concepts import content_word                            # noqa: E402

DEMO = os.path.join(HERE, "..", "inspector", "demo")
atlas = json.load(open(os.path.join(DEMO, os.environ.get("ATLAS_JSON", "atlas_emergent.json")), encoding="utf-8"))
CONCEPTS = atlas["meta"]["concepts"]
FIDS = [n["id"] for n in atlas["nodes"]]
FID2CONCEPT = {n["id"]: CONCEPTS[n["cluster"]] for n in atlas["nodes"]}
FID2PEAK = {n["id"]: float(n.get("peak", 1.0)) for n in atlas["nodes"]}
BRAIN_HTML = open(os.path.join(DEMO, "brain.html"), encoding="utf-8").read()
ARTIFACT_NNZ = 600

sae = GpuSAE()
tok, model = load7b()
print("brain-7b: model + SAE + atlas ready", flush=True)

import threading                                                   # noqa: E402
LOCK = threading.Lock()
SESSIONS = {}   # per-browser conversation threads {session_id: [messages]} so tabs/tests don't collide

# --- full-space readout: Neuronpedia auto-interp labels + per-feature stats let us read ALL 131k SAE
# features (not just the 1000-node atlas) and name + filter them honestly: normalize each to its own
# peak, drop broad (high-frequency) and discourse-form features, require a real label.
_LBL = json.load(open(os.path.join(HERE, "np_labels_l15.json"), encoding="utf-8"))   # {str(id): label}
_ST = json.load(open(os.path.join(HERE, "np_stats_l15.json")))                       # {str(id): [maxAct, frac]}
D_SAE = sae.d_sae
MAXACT = np.zeros(D_SAE, np.float32)
FRAC = np.ones(D_SAE, np.float32)
HASLABEL = np.zeros(D_SAE, bool)
BLOCKED = np.zeros(D_SAE, bool)          # discourse-form features (about the conversation, not the topic)
DISCOURSE_TERMS = ("question", "answer", "discuss", "conversation", "request", "asking", "writing",
                   "publish", "article", "journal", "blog", "sharing information", "connecting",
                   "decision", "choices", "instruction", "prompt", "response", "explanation", "summary",
                   "website", "online", "comment", "formatting", "markdown")
GENERIC_SINGLE = {"i", "our", "we", "you", "my", "your", "me", "us", "it", "they", "them",
                  "the", "a", "an", "this", "that", "and", "or", "of", "to", "in", "s", "t"}
for _k, (_ma, _fr) in _ST.items():
    _i = int(_k)
    MAXACT[_i] = _ma
    FRAC[_i] = _fr
for _k, _lbl in _LBL.items():
    _i = int(_k)
    HASLABEL[_i] = True
    _low = _lbl.strip().lower()
    if _low in GENERIC_SINGLE or any(t in _low for t in DISCOURSE_TERMS):
        BLOCKED[_i] = True
ATLAS_FID_ARR = np.array(FIDS)
ATLAS_DIRS = sae.W_dec_cpu[FIDS].float().numpy()
ATLAS_DIRS = ATLAS_DIRS / (np.linalg.norm(ATLAS_DIRS, axis=1, keepdims=True) + 1e-8)
ATLAS_SET = set(FIDS)
print(f"brain-7b: full-space readout ready ({int(HASLABEL.sum())} labeled, {int(BLOCKED.sum())} blocked)", flush=True)


def dynamic_considered(fmax, k=14, rel_min=0.18):
    """Top labeled, specific, on-topic features for this prompt, pulled from the FULL 131k space.
    Returns nodes with a real label, how hard they fired vs their own peak, whether they're already on
    the atlas map, and (for off-map ones) the nearest atlas node so the frontend can place them."""
    rel = np.where(MAXACT > 0, fmax / np.maximum(MAXACT, 1e-6), 0.0)
    elig = (fmax > 0) & HASLABEL & (FRAC < 0.02) & (~BLOCKED) & (rel >= rel_min)
    ids = np.where(elig)[0]
    if not len(ids):
        return []
    order = ids[np.argsort(-rel[ids])][:k]
    dirs = sae.W_dec_cpu[order.tolist()].float().numpy()
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
    near = ATLAS_FID_ARR[(dirs @ ATLAS_DIRS.T).argmax(1)]
    return [{"id": int(fid), "label": _LBL[str(int(fid))], "rel": round(float(rel[fid]), 3),
             "in_atlas": int(fid) in ATLAS_SET, "near": int(near[j])}
            for j, fid in enumerate(order.tolist())]


@torch.no_grad()
def think(text: str, sid: str) -> dict:
    with LOCK:
        # answer IN CONTEXT of THIS browser's thread (keep the last ~8 turns)
        hist = SESSIONS.setdefault(sid, [])
        hist.append({"role": "user", "content": text})
        ids = tok.apply_chat_template(hist[-16:], add_generation_prompt=True,
                                      return_tensors="pt").to(DEV)
        gen = model.generate(ids, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
        ans = tok.decode(gen[0][ids.shape[1]:], skip_special_tokens=True).strip()
        hist.append({"role": "assistant", "content": ans})
        if len(SESSIONS) > 64:                       # soft cap: forget the oldest threads
            for k in list(SESSIONS)[:-32]:
                SESSIONS.pop(k, None)
        # features the model engaged reading this message (layer-15, content tokens only, sink masked)
        pieces, feats = feats7b(text, tok, model, sae)
        f = feats.cpu().numpy()
        f[(f > 0).sum(1) > ARTIFACT_NNZ] = 0
        keep = np.array([content_word(p) for p in pieces])
        fmax = f[keep].max(0) if keep.any() else f.max(0)
        # stable-map glow: which atlas nodes fired, relative to their own peak
        acts = {}
        for fid in FIDS:
            v = float(fmax[fid])
            if v > 0:
                rel = min(1.5, v / max(FID2PEAK[fid], 1e-6))
                if rel >= 0.25:
                    acts[fid] = round(rel, 3)
        # the REAL "concepts considered": top labeled, specific features pulled from the full 131k space
        considered = dynamic_considered(fmax)
        return {"acts": acts, "considered": considered,
                "concepts": [{"name": c["label"]} for c in considered[:6]],
                "output": ans, "turn": len(hist) // 2}


def reset(sid: str) -> dict:
    with LOCK:
        SESSIONS[sid] = []
    return {"ok": True}


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
                self._send(200, json.dumps(think(str(body.get("text", ""))[:500],
                                                  str(body.get("sid", "default")))), "application/json")
            except Exception as e:
                self._send(500, json.dumps({"error": f"{type(e).__name__}: {e}"}), "application/json")
        elif self.path.rstrip("/").endswith("reset"):
            n = int(self.headers.get("Content-Length", 0))
            sid = str(json.loads(self.rfile.read(n) or b"{}").get("sid", "default"))
            self._send(200, json.dumps(reset(sid)), "application/json")
        else:
            self._send(404, "not found", "text/plain")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
    print(f"brain-7b thinking server -> http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
