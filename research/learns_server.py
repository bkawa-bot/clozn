"""
learns_server.py -- the LIVE BACKEND for the clozn "it learns" demo (the capable-model version).

Where memory_server.py serves the fast-weight FACT memory ("remembers") on a frozen GPT-2, THIS
server serves the test-time-training RULE learning ("learns") on a capable frozen Qwen2.5-1.5B-Instruct.
It loads the model ONCE on the GPU and holds ONE live "learned rule" (a soft prefix) in process. It
also serves the single-page light-theme frontend at GET "/" (learns_live.html next to this file).

THE LOOP (all real; the mechanism is the VERBATIM rig validated in legibility_v1 / frontier_apply):
  TEACH   -> show it a few "x -> y" pairs; fit a soft prefix by a handful of gradient steps through the
             FROZEN model (test-time training: only the prefix moves). A few seconds on the 5080.
  APPLY   -> type a NEW word; the frozen model + the learned prefix produce the transformed word, live,
             beside the model's BASELINE answer with no prefix (so you see the rule did something).
  STATE   -> with the prefix ACTIVE, ask the model "what rule did you just learn?"; it answers in words.
  VERIFY  -> take that STATED sentence and hand it (as a plain instruction) to the frozen model with NO
             prefix, on held-out words; check it reproduces the answers + agrees with the prefix. This is
             the honest part: the words are checked, not trusted. (Presets have held-out ground truth.)
  FORGET  -> drop the prefix; the model is exactly itself again.

WHY a capable model: legibility_v1_big.py showed self-report of the learned rule clears the wrong-rule
null only at >=1.5B (0.5B was at chance). So the "STATE" step actually works here.

REUSE (no re-implementation of the validated asset):
  frontier_apply     (FA)  -> load_llm, SoftPrefix, forward_with_prefix, batch_pack, cache_query_embeds,
                              encode_query_ids
  frontier_apply_v2  (FV2) -> build_bank, build_vocab_bank, split_bank, single_token_id  (the presets +
                              their held-out pairs for VERIFY)
  legibility_v1      (L1)  -> generate_with_prefix, apply_stated_rule, adapted_behavior,
                              score_against_truth, clean_rule, chat_ids, SELFREPORT_USER, RULE_DESC

MODEL/ENV: Qwen2.5-1.5B-Instruct, FROZEN, fp32, on cuda, in the lab venv (cloze/.venv, torch cu128).
  Loads from ~/hf_models/Qwen2.5-1.5B-Instruct if present (the WinError-1314 symlink workaround), else the
  hub cache. Adds NO dependencies: only the Python standard library for the web layer (http.server). The
  GPU is serialized with a lock (one teach/apply/state at a time) so concurrent requests are safe.

Usage (from research/, the lab .venv python):
    python learns_server.py                       # serves http://127.0.0.1:8078
    python learns_server.py --port 9001 --model Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import argparse
import html as _html
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # this PC crashes on HF symlinks (WinError 1314)
os.environ.setdefault("HF_HUB_OFFLINE", "1")           # the model is local; never hit the network
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                               # so the research modules import cleanly
FRONTEND_PATH = os.path.join(HERE, "learns_live.html")

import torch                       # noqa: E402
import torch.nn as nn             # noqa: E402
import torch.nn.functional as F    # noqa: E402

# The VALIDATED machinery (imported, never re-implemented -- this is the asset).
import frontier_apply as FA        # noqa: E402
import frontier_apply_v2 as FV2    # noqa: E402
import legibility_v1 as L1         # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def esc_html(s) -> str:
    return _html.escape(str(s))


def resolve_model_path(model_name: str) -> str:
    """Prefer a local snapshot at ~/hf_models/<name> (the local_dir download that dodges the Windows
    symlink crash); otherwise the hub id (standard cache). Matches legibility_v1_big.resolve_model_path."""
    local = os.path.join(os.path.expanduser("~"), "hf_models", model_name.split("/")[-1])
    if os.path.isfile(os.path.join(local, "config.json")):
        return local
    return model_name


# ====================================================================================================
# The TTT primitive: fit a soft prefix on an explicit list of (x_embeds, y_token_id) pairs. This is the
# EXACT lever-3 / legibility_v1 fit (SoftPrefix + forward_with_prefix + Adam on apply-CE, frozen
# backbone), generalized to take raw pairs so it serves both presets and custom user rules uniformly.
# ====================================================================================================
def fit_prefix_on_pairs(model, x_embeds_list, y_token_ids, m, steps, lr, seed):
    """x_embeds_list: list of [Lq,H] query-embed tensors (one per teaching x). y_token_ids: the answer
    token id per pair. Returns a trained FA.SoftPrefix. Only the prefix trains; the model is frozen."""
    H = model.config.hidden_size
    padded, mask = FA.batch_pack(x_embeds_list)                     # [N,Lmax,H], [N,Lmax]
    ytgt = torch.tensor(y_token_ids, device=DEV)
    torch.manual_seed(seed)
    p0 = 0.02 * torch.randn(m, H, device=DEV)
    pm = FA.SoftPrefix(m, H).to(DEV)
    pm.prefix = nn.Parameter(p0)
    opt = torch.optim.Adam(pm.parameters(), lr)
    pm.train()
    losses = []
    for _ in range(steps):
        logits = FA.forward_with_prefix(model, pm, padded, mask)    # [N,V] at the answer slot
        loss = F.cross_entropy(logits, ytgt)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss.item()))
    pm.eval()
    with torch.no_grad():
        train_acc = float((FA.forward_with_prefix(model, pm, padded, mask).argmax(-1) == ytgt)
                          .float().mean().item())
    return pm, losses, train_acc


# ====================================================================================================
# The app: holds the model + bank (presets & held-out verify pairs) + the ONE live learned prefix.
# ====================================================================================================
# Curated preset rules (intersected with whatever the bank actually has): intuitive, single-token answers.
PRESET_ORDER = [
    ("plural",        "make it plural"),
    ("past",          "past tense"),
    ("gerund",        "add -ing"),
    ("antonym2",      "the opposite"),
    ("third_person",  "he/she ___s"),
    ("comparative",   "the -er form"),
    ("opposite_gender", "the opposite gender"),
    ("capital",       "the country's capital"),
]


class LearnsApp:
    def __init__(self, model_name: str, m: int, steps: int, lr: float, k_examples: int,
                 dtype: str = "float32"):
        self.lock = threading.Lock()       # serialize all GPU work (CUDA isn't thread-safe across reqs)
        self.m = m
        self.steps = steps
        self.lr = lr
        self.k = k_examples
        self.model_name = model_name
        load_path = resolve_model_path(model_name)
        print(f"loading {model_name} (FROZEN, {dtype}) on {DEV} from {load_path} ...", flush=True)
        self.tok, self.model = FA.load_llm(load_path, dtype=getattr(torch, dtype))
        self.H = self.model.config.hidden_size
        self.eos_ids = set([self.tok.eos_token_id]) if self.tok.eos_token_id is not None else None
        print(f"  loaded. hidden_size={self.H}  layers={self.model.config.num_hidden_layers}", flush=True)

        # The bank gives ready presets (real word pairs) + held-out pairs to VERIFY against.
        print("building relation bank (presets + held-out verify pairs) ...", flush=True)
        self.bank, self.REL_NAMES, _dropped = FV2.build_bank(self.tok, min_pairs=10)
        self.words, self.widx, self.out_words, self.out_ids = FV2.build_vocab_bank(self.bank)
        self.train_pairs, self.test_pairs = FV2.split_bank(self.bank, self.words, self.widx,
                                                           test_frac=0.30, seed=0)
        self.answer_tok = {w: FV2.single_token_id(self.tok, w) for w in self.words}
        self.menu_ids = torch.tensor([self.answer_tok[w] for w in self.out_words], device=DEV)
        self.out_set_idx = {w: j for j, w in enumerate(self.out_words)}
        self.q_emb_cache = FA.cache_query_embeds(self.tok, self.model, self.words)
        self.presets = [(k, lab) for (k, lab) in PRESET_ORDER if k in self.REL_NAMES]
        print(f"  bank ready: {len(self.REL_NAMES)} relations, {len(self.words)} words, "
              f"{len(self.presets)} presets", flush=True)

        # the ONE live learned rule (None until taught)
        self.current = None     # dict: {pm, rel|None, examples:[(x,y)], custom:bool, train_acc, label}

    # ---- small helpers -------------------------------------------------------------------------------
    def _query_ids(self, word: str):
        return FA.encode_query_ids(self.tok, word)        # ids for "{word} ->"

    def _query_embed(self, word: str):
        ids = self._query_ids(word)
        return self.model.get_input_embeddings()(torch.tensor(ids, device=DEV)).detach()   # [Lq,H]

    def _first_tok_id(self, word: str):
        ids = self.tok.encode(" " + word.strip(), add_special_tokens=False)
        return ids[0] if ids else None

    @torch.no_grad()
    def _gen_after_arrow(self, word: str, prefix_tensor, max_new=6):
        """Greedy continuation of '{word} ->' (with the prefix, or None for baseline). Returns the
        cleaned short answer the model produces in the answer slot."""
        ids = self._query_ids(word)
        out = L1.generate_with_prefix(self.model, prefix_tensor, ids, max_new=max_new, eos_ids=self.eos_ids)
        txt = self.tok.decode(out).strip()
        txt = txt.split("\n")[0].strip()
        # keep just the produced word(s) before any next "x ->" the model might start
        for stop in [" ->", "->", ";", ".", ","]:
            if stop in txt:
                txt = txt.split(stop)[0].strip()
        return txt[:40]

    # ---- GET /presets --------------------------------------------------------------------------------
    def get_presets(self):
        items = []
        for key, label in self.presets:
            tp = self.train_pairs[key].tolist()
            te = self.test_pairs[key].tolist()
            ex = [(self.words[a], self.words[b]) for (a, b) in tp[:self.k]]
            try_words = [self.words[a] for (a, b) in te[:6]]
            items.append({"key": key, "label": label,
                          "examples": ex, "try_words": try_words})
        return 200, {"presets": items, "k_examples": self.k}

    # ---- POST /teach {preset} | {pairs:[[x,y],...]} --------------------------------------------------
    def teach(self, body: dict):
        preset = body.get("preset")
        pairs = body.get("pairs")
        with self.lock:
            t0 = time.time()
            if preset is not None:
                if preset not in self.REL_NAMES:
                    return 400, {"error": f"unknown preset {preset!r}"}
                # FIT on the relation's fuller train set (capped), not just the few shown: legibility_v1_big
                # showed the model can STATE the rule ("add -s") only when the adaptation is fit on the
                # examples, while APPLY works from very few. We still SHOW the user a representative handful.
                tp_all = self.train_pairs[preset].tolist()[:16]
                tp_show = tp_all[:self.k]
                examples = [(self.words[a], self.words[b]) for (a, b) in tp_show]
                report_ex = [(self.words[a], self.words[b]) for (a, b) in tp_all[:3]]
                n_fit = len(tp_all)
                x_embeds = [self.q_emb_cache[self.words[a]] for (a, b) in tp_all]
                y_tokids = [self.answer_tok[self.words[b]] for (a, b) in tp_all]
                rel = preset
                label = dict(self.presets).get(preset, preset)
                custom = False
            elif pairs:
                # custom rule: list of [x, y] strings (y's first token is the apply target)
                clean = [(str(x).strip(), str(y).strip()) for x, y in pairs
                         if str(x).strip() and str(y).strip()]
                if len(clean) < 2:
                    return 400, {"error": "give at least 2 'x -> y' example pairs"}
                examples = clean
                report_ex = clean[:3]
                n_fit = len(clean)
                x_embeds = [self._query_embed(x) for (x, y) in clean]
                y_tokids = [self._first_tok_id(y) for (x, y) in clean]
                if any(t is None for t in y_tokids):
                    return 400, {"error": "could not tokenize an answer word"}
                rel = None
                label = "your rule"
                custom = True
            else:
                return 400, {"error": "send {preset: <key>} or {pairs: [[x,y],...]}"}

            pm, losses, train_acc = fit_prefix_on_pairs(self.model, x_embeds, y_tokids,
                                                        self.m, self.steps, self.lr, seed=0)
            self.current = {"pm": pm, "rel": rel, "examples": examples, "report_ex": report_ex,
                            "custom": custom, "train_acc": train_acc, "label": label}
            dt = time.time() - t0
        return 200, {"ok": True, "label": label, "examples": examples, "n_fit": n_fit,
                     "custom": custom, "steps": self.steps, "m": self.m,
                     "train_fit": round(train_acc, 3),
                     "loss_start": round(losses[0], 3), "loss_end": round(losses[-1], 3),
                     "seconds": round(dt, 2),
                     "try_words": ([self.words[a] for (a, b) in self.test_pairs[rel].tolist()[:6]]
                                   if rel else [])}

    # ---- POST /apply {word} --------------------------------------------------------------------------
    def apply(self, body: dict):
        word = body.get("word")
        if not isinstance(word, str) or not word.strip():
            return 400, {"error": "missing 'word'"}
        word = word.strip()
        with self.lock:
            if self.current is None:
                return 400, {"error": "nothing taught yet -- POST /teach first"}
            pm = self.current["pm"]
            taught = self._gen_after_arrow(word, pm.prefix.detach())
            baseline = self._gen_after_arrow(word, None)
            # if this is a preset word with known ground truth, surface it (honest check)
            truth = None
            rel = self.current["rel"]
            if rel is not None:
                for (a, b) in (self.test_pairs[rel].tolist() + self.train_pairs[rel].tolist()):
                    if self.words[a] == word:
                        truth = self.words[b]; break
        return 200, {"word": word, "baseline": baseline, "taught": taught, "truth": truth,
                     "changed": bool(taught != baseline)}

    # ---- POST /state  (self-report + verify) -------------------------------------------------------
    def state(self, body: dict):
        with self.lock:
            if self.current is None:
                return 400, {"error": "nothing taught yet -- POST /teach first"}
            cur = self.current
            pm = cur["pm"]
            ex = "\n".join(f"{x} -> {y}" for (x, y) in cur.get("report_ex", cur["examples"]))

            # 1) SELF-REPORT under two framings, prefix ACTIVE (the model says what it learned)
            reports = {}
            for fr_name, tmpl in L1.SELFREPORT_USER.items():
                ids = L1.chat_ids(self.tok, tmpl.format(ex=ex))
                gen = L1.generate_with_prefix(self.model, pm.prefix.detach(), ids, max_new=24,
                                              eos_ids=self.eos_ids)
                reports[fr_name] = L1.clean_rule(self.tok.decode(gen))

            # 2) VERIFY (presets only -- they have held-out ground truth)
            verify = None
            rel = cur["rel"]
            if rel is not None:
                te = self.test_pairs[rel].tolist()[:6]
                test_words = [self.words[a] for (a, b) in te]
                test_pairs_w = [(self.words[a], self.words[b]) for (a, b) in te]
                adp_menu, adp_free = L1.adapted_behavior(self.model, pm, test_words,
                                                         self.q_emb_cache, self.menu_ids)
                adapted_acc = L1.score_against_truth(adp_menu, test_pairs_w, self.answer_tok)
                # use the stronger of the two stated framings for the headline verify
                per_framing = {}
                best = None
                for fr_name, stated in reports.items():
                    st_menu, st_free = L1.apply_stated_rule(self.tok, self.model, stated, test_words,
                                                            self.answer_tok, self.menu_ids, self.out_set_idx)
                    acc = L1.score_against_truth(st_menu, test_pairs_w, self.answer_tok)
                    agree = float(sum(int(a == b) for a, b in zip(st_menu, adp_menu)) / max(1, len(st_menu)))
                    per_framing[fr_name] = {"stated": stated, "stated_apply": round(acc, 3),
                                            "agreement": round(agree, 3)}
                    if best is None or acc > best[1]:
                        best = (fr_name, acc, agree, st_free)
                # controls: oracle (true rule) and an averaged wrong-rule null (verifier soundness)
                true_desc = L1.RULE_DESC.get(rel, rel.replace("_", " "))
                or_menu, _ = L1.apply_stated_rule(self.tok, self.model, true_desc, test_words,
                                                  self.answer_tok, self.menu_ids, self.out_set_idx)
                oracle = L1.score_against_truth(or_menu, test_pairs_w, self.answer_tok)
                wrong_rels = [r for r in self.REL_NAMES if r != rel and not r.startswith(rel[:4])
                              and L1.RULE_DESC.get(r) != L1.RULE_DESC.get(rel)][:4]
                wrong_accs = []
                for wr in wrong_rels:
                    wm, _ = L1.apply_stated_rule(self.tok, self.model, L1.RULE_DESC.get(wr, wr), test_words,
                                                 self.answer_tok, self.menu_ids, self.out_set_idx)
                    wrong_accs.append(L1.score_against_truth(wm, test_pairs_w, self.answer_tok))
                wrong = float(sum(wrong_accs) / max(1, len(wrong_accs)))
                # a small per-word table from the best framing (decoded free tokens, honest "what it says")
                _, _, _, best_free = best
                table = [{"word": test_words[i], "truth": test_pairs_w[i][1],
                          "prefix": self.tok.decode([adp_free[i]]).strip(),
                          "stated": self.tok.decode([best_free[i]]).strip()}
                         for i in range(len(test_words))]
                verify = {"adapted_apply": round(adapted_acc, 3), "oracle": round(oracle, 3),
                          "wrong_null": round(wrong, 3), "true_rule": true_desc,
                          "best_framing": best[0], "best_stated_apply": round(best[1], 3),
                          "best_agreement": round(best[2], 3),
                          "clears_null": bool(best[1] > wrong + 0.15 and best[1] > 0.2),
                          "per_framing": per_framing, "table": table, "n_test": len(test_words)}
        return 200, {"label": cur["label"], "custom": cur["custom"], "examples": cur["examples"],
                     "reports": reports, "verify": verify}

    # ---- POST /forget --------------------------------------------------------------------------------
    def forget(self, body: dict):
        with self.lock:
            had = self.current is not None
            self.current = None
        return 200, {"ok": True, "forgot": had}

    # ---- GET /health ---------------------------------------------------------------------------------
    def health(self):
        cur = None
        if self.current is not None:
            cur = {"label": self.current["label"], "custom": self.current["custom"],
                   "examples": self.current["examples"], "train_fit": round(self.current["train_acc"], 3)}
        return 200, {"ok": True, "model": self.model_name, "device": DEV,
                     "hidden_size": self.H, "m": self.m, "steps": self.steps,
                     "n_presets": len(self.presets), "current": cur}


# ====================================================================================================
# HTTP layer (stdlib; mirrors memory_server.py: CORS, JSON, serves the frontend at "/").
# ====================================================================================================
ENDPOINTS = [
    "GET  /                       -> the live HTML frontend (learns_live.html)",
    "GET  /presets                -> {presets:[{key,label,examples,try_words}]}",
    "POST /teach   {preset}|{pairs}-> fit a soft prefix (TTT); {label,examples,train_fit,seconds}",
    "POST /apply   {word}         -> {baseline, taught, truth, changed}",
    "POST /state                  -> {reports:{declarative,metacog}, verify:{...}}",
    "POST /forget                 -> {ok}",
    "GET  /health                 -> {model, current, ...}",
]


def make_handler(app: LearnsApp):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ClozeLearnsServer/1.0"
        protocol_version = "HTTP/1.1"

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, status, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self._cors(); self.end_headers(); self.wfile.write(data)

        def _html(self, status, body):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self._cors(); self.end_headers(); self.wfile.write(data)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if not n:
                return {}
            raw = self.rfile.read(n)
            return json.loads(raw.decode("utf-8")) if raw else {}

        def do_OPTIONS(self):
            self.send_response(204); self._cors()
            self.send_header("Content-Length", "0"); self.end_headers()

        def do_GET(self):
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/":
                try:
                    with open(FRONTEND_PATH, "r", encoding="utf-8") as fh:
                        self._html(200, fh.read())
                except FileNotFoundError:
                    self._html(500, f"<h1>learns_live.html not found</h1><p>{esc_html(FRONTEND_PATH)}</p>"
                                    "<p>The JSON API is live (GET /health, /presets; POST /teach,...).</p>")
                return
            try:
                if path == "/health":
                    status, payload = app.health()
                elif path == "/presets":
                    status, payload = app.get_presets()
                else:
                    status, payload = 404, {"error": f"no route GET {path}", "endpoints": ENDPOINTS}
            except Exception as e:  # noqa: BLE001
                status, payload = 500, {"error": f"{type(e).__name__}: {e}"}
            self._json(status, payload)

        def do_POST(self):
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            try:
                body = self._body()
            except (json.JSONDecodeError, ValueError) as e:
                self._json(400, {"error": f"invalid JSON body: {e}"}); return
            try:
                if path == "/teach":
                    status, payload = app.teach(body)
                elif path == "/apply":
                    status, payload = app.apply(body)
                elif path == "/state":
                    status, payload = app.state(body)
                elif path == "/forget":
                    status, payload = app.forget(body)
                else:
                    status, payload = 404, {"error": f"no route POST {path}", "endpoints": ENDPOINTS}
            except Exception as e:  # noqa: BLE001
                status, payload = 500, {"error": f"{type(e).__name__}: {e}"}
            self._json(status, payload)

        def log_message(self, fmt, *args):
            sys.stderr.write("  [http] %s - %s\n" % (self.address_string(), fmt % args))

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Live backend for the clozn 'it learns' (TTT) demo.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8078)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--m", type=int, default=8, help="soft-prefix length (TTT)")
    ap.add_argument("--steps", type=int, default=40, help="TTT gradient steps per teach")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--k", type=int, default=3, help="example pairs shown/used per teach")
    args = ap.parse_args()

    torch.manual_seed(0)
    app = LearnsApp(args.model, m=args.m, steps=args.steps, lr=args.lr, k_examples=args.k,
                    dtype=args.dtype)
    httpd = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    shown = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    url = f"http://{shown}:{args.port}"
    print("\n" + "=" * 78)
    print(f"  CLOZN 'IT LEARNS' SERVER  ->  {url}")
    print("=" * 78)
    print(f"  model: {args.model} (frozen, {args.dtype})   TTT: m={args.m}, {args.steps} steps")
    print(f"  >> open {url}/  for the LIVE UI" if os.path.exists(FRONTEND_PATH)
          else f"  !! {FRONTEND_PATH} not found -- GET / will 500; JSON API still live")
    for line in ENDPOINTS:
        print(f"    {line}")
    print("\n  Ctrl-C to stop.\n", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down ...", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
