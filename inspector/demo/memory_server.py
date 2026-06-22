"""
memory_server.py -- the LIVE BACKEND for the Clozn memory window.

Turns the static `memory_window.py` page into a real interactive runtime: a small local
HTTP server that loads a FROZEN GPT-2-small ONCE at startup and holds a single live
glass-box fast-weight memory in process. A separate frontend (a local HTML file) talks to
it over JSON. This file is BACKEND ONLY -- no HTML/UI is served here.

WHAT THE MEMORY IS (reused verbatim from the validated spikes p15_fastweight + p17_betterkey,
and from the static demo memory_window.py -- no faking, real recall):
  WRITE key  = MLP post-activation `blocks.L.mlp.hook_post` at the cue's FINAL token, over
               the cue ONLY (the p17 "raw_consistent" key: SAME position for write and read,
               the variant that actually recalls). The READ key is grabbed the same way at
               query time, so the query self-addresses its own entry.
  value      = the answer token's unembedding direction W_U[:, ans] (legible by build:
               adding it to the residual promotes `answer` via the logit lens).
  recall     = a forward hook adding  sum_i w_i * value_i  at the query's final position,
               with GATED hard top-1 addressing over cosine similarity: the nearest stored
               key fires (w_i = eta) only if its cosine clears GATE (~0.90), else NOTHING is
               injected and the model returns its exact baseline. The gate is what keeps a
               wrong-keyed query silent (self-cosine ~1.0 fires; unrelated cross ~0.82 gated
               off) -- so /query honestly reports "fired: null" when no entry matches.

The backbone is FROZEN throughout; GPT-2 is never trained. Every probability returned by
/query is the ACTUAL model output (baseline = no memory; with_memory = the live memory hook).

ENDPOINTS (all JSON; CORS enabled so a local file:// HTML page can call them):
  POST /write    {cue, answer}        -> {label, decoded_word, salience, key_fingerprint, ...}
  GET  /memory                        -> {entries: [...]} (the cards)
  POST /delete   {label}              -> {ok: true, removed: <label>}
  POST /salience {label, eta}         -> the updated entry
  POST /query    {prompt, topk?}      -> {baseline:[...], with_memory:[...], fired:{label,match_score}|null}
  GET  /health                        -> {model, layer, n_entries, ...}

MODEL / ENV: GPT-2-small (124M, frozen) via transformer_lens, CPU, in the ISOLATED env
C:\\Users\\brigi\\src\\clozn\\.venv-sae (matches the static demo). GPT-2 is cached -- no
download. This server adds NO new dependencies: it uses only the Python standard library
(http.server) for the web layer, so it runs in .venv-sae as-is without touching the venv.

Usage (from inspector/, .venv-sae python):
    python demo/memory_server.py                 # serves http://127.0.0.1:8077, layer 8
    python demo/memory_server.py --port 9000 --layer 6
    python demo/memory_server.py --host 0.0.0.0  # expose on the LAN (default is loopback)

Sample (server running on the default port):
    curl -s -X POST http://127.0.0.1:8077/write \
         -H "Content-Type: application/json" \
         -d '{"cue":"The secret color of Zorbland is","answer":" blue"}'
    curl -s -X POST http://127.0.0.1:8077/query \
         -H "Content-Type: application/json" \
         -d '{"prompt":"The secret color of Zorbland is","topk":5}'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # this PC crashes on HF symlinks (WinError 1314)

import torch                       # noqa: E402
import torch.nn.functional as F    # noqa: E402


# ====================================================================================================
# Model + low-level pieces (the validated mechanism, lifted verbatim from p15/p17/memory_window).
# ====================================================================================================
def load_model(device: str):
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def single_token_id(model, word: str):
    """The single GPT-2 token id for `word` (leading space included), or None if it isn't one token."""
    ids = model.to_tokens(word, prepend_bos=False)[0]
    if ids.shape[0] != 1:
        return None
    return int(ids[0])


def tok_str(model, tid: int) -> str:
    return model.to_string(torch.tensor([int(tid)]))


@torch.no_grad()
def topk_preds(model, cue: str, k: int = 5):
    """Top-k next-token (word, prob) at the cue's final position with NO memory (clean frozen model)."""
    logits = model(model.to_tokens(cue))[0, -1].float()
    probs = F.softmax(logits, dim=-1)
    top = logits.topk(k)
    return [{"word": tok_str(model, int(i)), "prob": float(probs[int(i)])} for i in top.indices]


@torch.no_grad()
def base_prob(model, cue: str, ans_id: int):
    """P(ans|cue), is-top1, is-top5 with NO memory."""
    logits = model(model.to_tokens(cue))[0, -1].float()
    probs = F.softmax(logits, dim=-1)
    top5 = set(int(i) for i in logits.topk(5).indices)
    return float(probs[ans_id]), int(logits.argmax()) == ans_id, ans_id in top5


@torch.no_grad()
def consistent_key(model, cue: str, layer: int):
    """The p17 'raw_consistent' key: MLP post-activation at the cue's FINAL token, over the cue ONLY.
    Used identically for WRITE and READ -- that consistency is what makes recall fire."""
    name = f"blocks.{layer}.mlp.hook_post"
    _, cache = model.run_with_cache(model.to_tokens(cue), names_filter=name)
    return cache[name][0][-1].clone()                  # [d_mlp] at the final position


@torch.no_grad()
def value_dir(model, ans_id: int):
    """The legible value: the answer token's unembedding direction (residual-space)."""
    return model.W_U[:, ans_id].clone()                # [d_model]


@torch.no_grad()
def logit_lens_top(model, v: torch.Tensor, k: int = 1):
    """Decode a residual-space direction through the logit lens: ln_final -> unembed -> top token(s)."""
    lv = model.ln_final(v.unsqueeze(0))
    lens = (lv @ model.W_U)[0].float()
    top = lens.topk(k)
    return [{"word": tok_str(model, int(i)), "logit": float(lens[int(i)])} for i in top.indices]


# ====================================================================================================
# The MEMORY: an explicit, inspectable, editable list held LIVE in the server process. Recall = a hook
# adding sum_i w_i*value_i at the query's final position, gated hard top-1 addressing over cosine
# similarity (the variant that recalls with clean specificity). This is memory_window.GlassBoxMemory
# with thread-safe mutation + monotonic labels so the frontend has a stable id per card.
# ====================================================================================================
class GlassBoxMemory:
    """entries: list of {label, key[d_mlp], value[d_model], eta, ans_id, cue, decoded_word}.
    The list IS the memory.

    Addressing is hard top-1 over cosine WITH a min-similarity gate: the contribution fires only if the
    nearest stored key clears GATE. A query's OWN key cosines ~1.0 to its own stored entry, but the
    nonce cues are not orthogonal (two 'color' cues cosine ~0.82), so an UNGATED top-1 would always
    fire the nearest remaining entry even on an unrelated query. The gate (between the ~1.0 self regime
    and the ~0.82 cross regime) makes a wrong-keyed query a true no-op: no injection, exact baseline."""

    GATE = 0.90   # min cosine for the nearest key to fire (self ~1.0 fires; cross ~0.82 is gated off)

    def __init__(self, model, layer: int):
        self.model = model
        self.layer = layer
        self.entries: list[dict] = []
        self._next_id = 0
        self.lock = threading.RLock()   # guards entries during concurrent HTTP requests

    # ---- mutation --------------------------------------------------------------------------------
    def write(self, cue: str, answer: str, eta: float = 10.0):
        """Store one fact. Returns the new entry dict (or raises ValueError on a multi-token answer)."""
        ans_id = single_token_id(self.model, answer)
        if ans_id is None:
            raise ValueError(
                f"answer {answer!r} is not a single GPT-2 token; pick a single-token word "
                f"(usually with a leading space, e.g. ' blue')."
            )
        key = consistent_key(self.model, cue, self.layer)
        value = value_dir(self.model, ans_id)
        decoded = logit_lens_top(self.model, value, k=1)[0]["word"]
        with self.lock:
            label = f"m{self._next_id}"
            self._next_id += 1
            entry = {
                "label": label, "key": key, "value": value, "eta": float(eta),
                "ans_id": int(ans_id), "cue": cue, "answer": answer,
                "decoded_word": decoded,
            }
            self.entries.append(entry)
        return entry

    def delete(self, label: str) -> bool:
        with self.lock:
            before = len(self.entries)
            self.entries = [e for e in self.entries if e["label"] != label]
            return len(self.entries) < before

    def set_salience(self, label: str, eta: float):
        with self.lock:
            for e in self.entries:
                if e["label"] == label:
                    e["eta"] = float(eta)
                    return e
        return None

    def get(self, label: str):
        with self.lock:
            for e in self.entries:
                if e["label"] == label:
                    return e
        return None

    def snapshot(self) -> list[dict]:
        """A consistent copy of the entry list (under the lock) for read-only iteration."""
        with self.lock:
            return list(self.entries)

    # ---- legible serialization (the card) --------------------------------------------------------
    def card(self, entry: dict) -> dict:
        """Public, JSON-safe view of one entry: label + decoded value + salience + key fingerprint."""
        key = entry["key"]
        value = entry["value"]
        topdims = [int(d) for d in key.abs().topk(6).indices]
        decoded = entry["decoded_word"]
        ok = (single_token_id(self.model, decoded) == entry["ans_id"]
              or decoded.strip() == entry["answer"].strip())
        return {
            "label": entry["label"],
            "cue": entry["cue"],
            "answer": entry["answer"].strip() or entry["answer"],
            "decoded_word": decoded.strip() or decoded,
            "value_decodes_ok": bool(ok),
            "salience": float(entry["eta"]),
            "key_fingerprint": {
                "dim": int(key.shape[0]),
                "top_dims": topdims,
                "key_norm": round(float(key.norm()), 4),
                "value_norm": round(float(value.norm()), 4),
            },
        }

    # ---- addressing + recall ---------------------------------------------------------------------
    @torch.no_grad()
    def _address(self, qkey: torch.Tensor, entries: list[dict]):
        """Gated hard top-1 over cosine: nearest stored key wins IF it clears GATE, else nothing fires.
        Returns (weights[n], selected_index_or_None, nearest_cosine)."""
        keys = torch.stack([e["key"] for e in entries])            # [n, d_mlp]
        cos = F.normalize(keys, dim=-1) @ F.normalize(qkey, dim=-1)  # [n]
        sel = int(cos.argmax())
        w = torch.zeros_like(cos)
        if float(cos[sel]) >= self.GATE:
            w[sel] = float(entries[sel]["eta"])
            return w, sel, float(cos[sel])
        return w, None, float(cos[sel])                            # gated off -> no injection

    @torch.no_grad()
    def recall(self, cue: str, k: int = 5):
        """Query `cue` with the CURRENT memory active. Captures the query key at the final position,
        injects the addressed memory contribution into resid_post at layer L, and returns
        (top-k [{word,prob}], full probs tensor, selected_entry_or_None, nearest_cosine).
        When the nearest key is below GATE nothing is injected (sel is None) and the model returns its
        exact baseline -- so a wrong-keyed query honestly reports fired=null."""
        entries = self.snapshot()                                  # consistent view for this forward
        post_name = f"blocks.{self.layer}.mlp.hook_post"
        resid_name = f"blocks.{self.layer}.hook_resid_post"
        cap = {"sel": None, "cos": None}

        def grab(act, hook):
            cap["q"] = act[0, -1].clone()
            return act

        def inject(act, hook):
            if not entries:
                return act
            w, sel, c = self._address(cap["q"], entries)
            cap["sel"], cap["cos"] = sel, c
            if sel is None:
                return act                                         # below gate: leave residual untouched
            vals = torch.stack([e["value"] for e in entries])      # [n, d_model]
            act[0, -1] = act[0, -1] + (w.unsqueeze(-1) * vals).sum(0)
            return act

        logits = self.model.run_with_hooks(
            self.model.to_tokens(cue), fwd_hooks=[(post_name, grab), (resid_name, inject)]
        )[0, -1].float()
        probs = F.softmax(logits, dim=-1)
        top = logits.topk(k)
        out = [{"word": tok_str(self.model, int(i)), "prob": float(probs[int(i)])} for i in top.indices]
        sel_entry = entries[cap["sel"]] if cap["sel"] is not None else None
        return out, probs, sel_entry, cap["cos"]


# ====================================================================================================
# The HTTP layer: a tiny stdlib JSON server (no Flask/FastAPI dependency, so .venv-sae is untouched).
# CORS is wide-open (Access-Control-Allow-Origin: *) so a local HTML file can call it. Single shared
# MEMORY + model; the GlassBoxMemory lock guards mutation, and ThreadingHTTPServer handles concurrent
# requests. Model forward passes are single and CPU-cheap (one pass for baseline, one for with-memory).
# ====================================================================================================
class App:
    """Holds the loaded model + the one live memory. The handler routes requests to these methods,
    each returning (status_code, json_dict)."""

    def __init__(self, model, layer: int, device: str):
        self.model = model
        self.layer = layer
        self.device = device
        self.mem = GlassBoxMemory(model, layer)
        self.model_name = "gpt2 (GPT-2-small, 124M, frozen)"

    # -- POST /write {cue, answer} -----------------------------------------------------------------
    def write(self, body: dict):
        cue = body.get("cue")
        answer = body.get("answer")
        eta = body.get("eta", 10.0)
        if not isinstance(cue, str) or not cue:
            return 400, {"error": "missing or empty 'cue' (a string ending right before the answer)"}
        if not isinstance(answer, str) or not answer:
            return 400, {"error": "missing or empty 'answer' (a single-token word, e.g. ' blue')"}
        try:
            entry = self.mem.write(cue, answer, float(eta))
        except ValueError as e:
            return 400, {"error": str(e)}
        return 200, self.mem.card(entry)

    # -- GET /memory -------------------------------------------------------------------------------
    def memory(self):
        cards = [self.mem.card(e) for e in self.mem.snapshot()]
        return 200, {"n_entries": len(cards), "layer": self.layer, "entries": cards}

    # -- POST /delete {label} ----------------------------------------------------------------------
    def delete(self, body: dict):
        label = body.get("label")
        if not isinstance(label, str) or not label:
            return 400, {"error": "missing 'label' (the entry id, e.g. 'm0')"}
        removed = self.mem.delete(label)
        if not removed:
            return 404, {"error": f"no entry with label {label!r}", "ok": False}
        return 200, {"ok": True, "removed": label, "n_entries": len(self.mem.snapshot())}

    # -- POST /salience {label, eta} ---------------------------------------------------------------
    def salience(self, body: dict):
        label = body.get("label")
        eta = body.get("eta")
        if not isinstance(label, str) or not label:
            return 400, {"error": "missing 'label' (the entry id, e.g. 'm0')"}
        if not isinstance(eta, (int, float)):
            return 400, {"error": "missing or non-numeric 'eta' (the salience, e.g. 10.0)"}
        entry = self.mem.set_salience(label, float(eta))
        if entry is None:
            return 404, {"error": f"no entry with label {label!r}"}
        return 200, self.mem.card(entry)

    # -- POST /query {prompt, topk?} ---------------------------------------------------------------
    def query(self, body: dict):
        prompt = body.get("prompt")
        topk = int(body.get("topk", 5) or 5)
        topk = max(1, min(topk, 20))
        if not isinstance(prompt, str) or not prompt:
            return 400, {"error": "missing or empty 'prompt'"}
        # baseline: NO memory (clean frozen model) -- a real, separate forward pass.
        baseline = topk_preds(self.model, prompt, k=topk)
        # with_memory: the SAME prompt with the live memory hook active.
        with_mem, _probs, sel_entry, cos = self.mem.recall(prompt, k=topk)
        fired = None
        if sel_entry is not None:
            fired = {"label": sel_entry["label"], "match_score": round(float(cos), 4),
                     "answer": sel_entry["answer"].strip() or sel_entry["answer"],
                     "decoded_word": sel_entry["decoded_word"].strip() or sel_entry["decoded_word"]}
        return 200, {
            "prompt": prompt,
            "baseline": baseline,
            "with_memory": with_mem,
            "fired": fired,
            "nearest_cosine": round(float(cos), 4) if cos is not None else None,
            "gate": self.mem.GATE,
        }

    # -- GET /health -------------------------------------------------------------------------------
    def health(self):
        return 200, {
            "ok": True,
            "model": self.model_name,
            "device": self.device,
            "layer": self.layer,
            "d_model": int(self.model.cfg.d_model),
            "d_mlp": int(self.model.cfg.d_mlp),
            "n_layers": int(self.model.cfg.n_layers),
            "gate": self.mem.GATE,
            "n_entries": len(self.mem.snapshot()),
        }


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ClozeMemoryServer/1.0"
        protocol_version = "HTTP/1.1"

        # ---- CORS + JSON helpers -----------------------------------------------------------------
        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _send_json(self, status: int, payload: dict):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self._cors()
            self.end_headers()
            self.wfile.write(data)

        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0) or 0)
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))

        # ---- CORS preflight ----------------------------------------------------------------------
        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        # ---- routing -----------------------------------------------------------------------------
        def do_GET(self):
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            try:
                if path == "/health":
                    status, payload = app.health()
                elif path == "/memory":
                    status, payload = app.memory()
                elif path == "/":
                    status, payload = 200, {"service": "cloze memory server", "endpoints": ENDPOINTS}
                else:
                    status, payload = 404, {"error": f"no route GET {path}", "endpoints": ENDPOINTS}
            except Exception as e:  # noqa: BLE001 -- surface server errors as JSON, never crash the loop
                status, payload = 500, {"error": f"{type(e).__name__}: {e}"}
            self._send_json(status, payload)

        def do_POST(self):
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            try:
                body = self._read_body()
            except (json.JSONDecodeError, ValueError) as e:
                self._send_json(400, {"error": f"invalid JSON body: {e}"})
                return
            try:
                if path == "/write":
                    status, payload = app.write(body)
                elif path == "/delete":
                    status, payload = app.delete(body)
                elif path == "/salience":
                    status, payload = app.salience(body)
                elif path == "/query":
                    status, payload = app.query(body)
                else:
                    status, payload = 404, {"error": f"no route POST {path}", "endpoints": ENDPOINTS}
            except Exception as e:  # noqa: BLE001
                status, payload = 500, {"error": f"{type(e).__name__}: {e}"}
            self._send_json(status, payload)

        # quieter, single-line logging
        def log_message(self, fmt, *args):
            sys.stderr.write("  [http] %s - %s\n" % (self.address_string(), fmt % args))

    return Handler


ENDPOINTS = [
    "POST /write    {cue, answer}     -> {label, decoded_word, salience, key_fingerprint}",
    "GET  /memory                     -> {entries:[...]}  (the cards)",
    "POST /delete   {label}           -> {ok}",
    "POST /salience {label, eta}      -> the updated entry",
    "POST /query    {prompt, topk?}   -> {baseline, with_memory, fired}",
    "GET  /health                     -> {model, layer, n_entries}",
]


# ====================================================================================================
def main():
    ap = argparse.ArgumentParser(description="Live backend for the Clozn glass-box memory window.")
    ap.add_argument("--host", default="127.0.0.1", help="bind host (default loopback; 0.0.0.0 for LAN)")
    ap.add_argument("--port", type=int, default=8077, help="bind port (default 8077)")
    ap.add_argument("--layer", type=int, default=8, help="memory write/read layer (p15/p17 best = 8)")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed-facts", action="store_true",
                    help="pre-load the 3 demo facts (Zorbland/Quibblax/Flonkville) so the memory "
                         "is non-empty on startup")
    args = ap.parse_args()

    torch.manual_seed(0)
    print(f"loading gpt2 (HookedTransformer) on {args.device} ... (frozen; ~once, then in-memory)",
          flush=True)
    model = load_model(args.device)
    print(f"  loaded. d_model={model.cfg.d_model}  d_mlp={model.cfg.d_mlp}  "
          f"n_layers={model.cfg.n_layers}   memory layer L={args.layer}", flush=True)

    app = App(model, args.layer, args.device)

    if args.seed_facts:
        seeds = [
            ("The secret color of Zorbland is", " blue"),
            ("The official animal of Quibblax is the", " dog"),
            ("The lucky number of Flonkville is", " seven"),
        ]
        for cue, ans in seeds:
            try:
                e = app.mem.write(cue, ans)
                print(f"  seeded {e['label']}: {cue!r} -> {ans!r}", flush=True)
            except ValueError as ex:
                print(f"  skip seed {cue!r}: {ex}", flush=True)

    handler = make_handler(app)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    url = f"http://{shown_host}:{args.port}"

    print("\n" + "=" * 78)
    print(f"  CLOZN MEMORY SERVER  ->  {url}")
    print("=" * 78)
    print(f"  model: {app.model_name}   layer: blocks.{args.layer}.mlp   gate: {app.mem.GATE}")
    print("  endpoints (CORS enabled; call from a local HTML file):")
    for line in ENDPOINTS:
        print(f"    {line}")
    print("\n  sample:")
    print(f"    curl -s -X POST {url}/write -H \"Content-Type: application/json\" \\")
    print("         -d '{\"cue\":\"The secret color of Zorbland is\",\"answer\":\" blue\"}'")
    print(f"    curl -s -X POST {url}/query -H \"Content-Type: application/json\" \\")
    print("         -d '{\"prompt\":\"The secret color of Zorbland is\",\"topk\":5}'")
    print("\n  Ctrl-C to stop.\n", flush=True)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down ...", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
