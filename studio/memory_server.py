"""
memory_server.py -- the LIVE BACKEND for the Clozn memory window.

Turns the static `memory_window.py` page into a real interactive runtime: a small local
HTTP server that loads a FROZEN GPT-2-small ONCE at startup and holds a single live
glass-box fast-weight memory in process. It also SERVES the live single-page frontend at
GET "/" (read from `memory_live.html` next to this file), so opening http://127.0.0.1:8077/
gives the interactive UI directly -- no file:// page, all requests same-origin.

WHAT THE MEMORY IS (reused verbatim from the validated spikes p15_fastweight + p17_betterkey,
and from the static demo memory_window.py -- no faking, real recall):
  WRITE key  = MLP post-activation `blocks.L.mlp.hook_post` at the cue's FINAL token, over
               the cue ONLY (the p17 "raw_consistent" key: SAME position for write and read,
               the variant that actually recalls). The READ key is grabbed the same way at
               query time, so the query self-addresses its own entry.
  value      = the answer token's unembedding direction W_U[:, ans] (legible by build:
               adding it to the residual promotes `answer` via the logit lens). A MULTI-token
               answer (e.g. ' mochi' -> [' m','och','i']) uses its FIRST token's direction, which
               decodes cleanly back to that first piece and promotes it on recall -- so recall
               nails the answer's first token (later pieces are NOT addressed; the UI says so).
  recall     = a forward hook adding  sum_i w_i * value_i  at the query's final position,
               with GATED hard top-1 addressing over cosine similarity: the nearest stored
               key fires (w_i = eta) only if its cosine clears the match-strictness gate, else
               NOTHING is injected and the model returns its exact baseline. The gate keeps a
               wrong-keyed query silent (self-cosine ~1.0 fires; unrelated cross ~0.82 gated
               off) -- so /query honestly reports "fired: null" when no entry matches. The gate
               is LIVE-ADJUSTABLE (POST /gate or a per-query override) so softer paraphrases can
               be made to fire by loosening it; default GlassBoxMemory.DEFAULT_GATE (0.90).

The backbone is FROZEN throughout; GPT-2 is never trained. Every probability returned by
/query is the ACTUAL model output (baseline = no memory; with_memory = the live memory hook).

ENDPOINTS (JSON unless noted; CORS enabled so a local file:// HTML page can also call them):
  GET  /                              -> the live HTML frontend (memory_live.html), text/html
  POST /write    {cue, answer}        -> {label, decoded_word, salience, key_fingerprint, ...}
  GET  /memory                        -> {entries: [...]} (the cards)
  POST /delete   {label}              -> {ok: true, removed: <label>}
  POST /salience {label, eta}         -> the updated entry
  POST /query    {prompt, topk?, gate?} -> {baseline, with_memory, fired, nearest_cosine, gate}
  POST /gate     {gate}               -> {ok, gate}  (set the live match-strictness threshold)
  POST /thinking {prompt}             -> {concepts:[...], tokens:[{tok, lit:[{c,z}]}]}  (the concept read)
  GET  /health                        -> {model, layer, gate, n_entries, thinking, ...}

The "what is it thinking" read (POST /thinking) is an honest concept PROBE on the SAME frozen model: a
named diff-in-means concept basis (the validated p18 directions) is built ONCE at startup, and per call
we project the prompt's per-token residuals onto it and report which concepts beat an equal-norm random
null (z >= threshold). It is a READ of what is PRESENT in the state -- not a decision, not the output.

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
import html as _html
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

# The live single-page frontend lives right next to this server; GET "/" serves it.
FRONTEND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory_live.html")


def esc_html(s) -> str:
    return _html.escape(str(s))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # this PC crashes on HF symlinks (WinError 1314)

import numpy as np                 # noqa: E402
import torch                       # noqa: E402
import torch.nn.functional as F    # noqa: E402

# The "what is it thinking" read reuses the VALIDATED diff-in-means concept basis from the p18 spike
# (same directions thinking_panel.py renders), so /thinking is the same honest probe -- just live on
# the prompt the user asks, sharing this server's single frozen GPT-2. sys.path already has inspector/.
from spikes.p18_conceptmem import (   # noqa: E402
    CONCEPTS as P18_CONCEPTS,
    build_basis,
)


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


def answer_token_ids(model, word: str) -> list[int]:
    """All GPT-2 token ids for `word` (leading space included). One id for a single-token word like
    ' blue'; several for a name like ' mochi' (-> [' m', 'ochi']). The value direction is built from
    these; recall promotes the FIRST of them (see value_dir_multi)."""
    ids = model.to_tokens(word, prepend_bos=False)[0]
    return [int(t) for t in ids]


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
def value_dir_multi(model, ans_ids: list[int]):
    """The legible value for a (possibly) multi-token answer. For ONE token this is exactly value_dir.
    For several tokens we use the FIRST token's unembedding direction. We tried the MEAN of all the
    tokens' directions too, but measured (GPT-2, L10, ' mochi' -> [' m','och','i']) it decodes to the
    MIDDLE piece ('och') and only lifts the real first token ' m' to rank 3 -- illegible. The first
    token's direction decodes cleanly to ' m' (value_decodes_ok stays true) and lifts ' m' to rank ~2,
    so the card reads back legibly and recall promotes the answer's first piece. HONEST LIMIT: only the
    first token of a multi-token name is addressed/promoted; the later pieces are not (surfaced in UI)."""
    dirs = model.W_U[:, ans_ids].T                      # [n_tok, d_model]
    return dirs[0].clone()                              # the FIRST answer token's direction


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
    nearest stored key clears the gate. A query's OWN key cosines ~1.0 to its own stored entry, but the
    nonce cues are not orthogonal (two 'color' cues cosine ~0.82), so an UNGATED top-1 would always
    fire the nearest remaining entry even on an unrelated query. The gate (between the ~1.0 self regime
    and the ~0.82 cross regime) makes a wrong-keyed query a true no-op: no injection, exact baseline.

    The gate is the MATCH-STRICTNESS knob and is now a live instance attribute (`self.gate`), adjustable
    at runtime via POST /gate or a per-query override, so paraphrases that sit just below the default can
    be made to fire by loosening it (at the cost of letting near-misses creep in). Default DEFAULT_GATE."""

    # Default gate 0.82: measured at the L10 default, stored-cue paraphrases sit 0.83-0.94 and unrelated
    # queries <=0.565, so 0.82 fires soft paraphrases while still gating junk (self ~1.0 always fires).
    # At the old L8 the paraphrase/unrelated bands nearly touched (0.758 vs 0.686), so 0.90 was needed
    # there to stay safe -- and it blocked every soft paraphrase. Deeper keying (L10) is what buys the
    # looser-yet-safe default. Fully live: POST /gate or the strictness slider moves it at runtime.
    DEFAULT_GATE = 0.82
    GATE_MIN = 0.30       # clamp range exposed to the strictness slider (very loose ... very strict)
    GATE_MAX = 0.995

    def __init__(self, model, layer: int, gate: float | None = None):
        self.model = model
        self.layer = layer
        self.gate = self.DEFAULT_GATE if gate is None else self._clamp_gate(gate)
        self.entries: list[dict] = []
        self._next_id = 0
        self.lock = threading.RLock()   # guards entries during concurrent HTTP requests

    @classmethod
    def _clamp_gate(cls, g: float) -> float:
        return float(max(cls.GATE_MIN, min(cls.GATE_MAX, float(g))))

    def set_gate(self, g: float) -> float:
        """Set the live match-strictness gate (clamped to [GATE_MIN, GATE_MAX]). Returns the value set."""
        with self.lock:
            self.gate = self._clamp_gate(g)
            return self.gate

    # ---- mutation --------------------------------------------------------------------------------
    def write(self, cue: str, answer: str, eta: float = 10.0):
        """Store one fact. Returns the new entry dict. Multi-token answers (e.g. ' mochi' -> [' m','och','i'])
        are accepted: the value becomes the answer's FIRST token's unembedding direction, which decodes
        cleanly back to that first piece and promotes it on recall. So recall nails the answer's first
        token; later pieces are not separately addressed -- the UI is honest about this."""
        if not answer:
            raise ValueError("empty answer; give a word (usually with a leading space, e.g. ' blue').")
        ans_ids = answer_token_ids(self.model, answer)
        if not ans_ids:
            raise ValueError(f"answer {answer!r} did not tokenize to any GPT-2 tokens.")
        key = consistent_key(self.model, cue, self.layer)
        value = value_dir_multi(self.model, ans_ids)
        decoded = logit_lens_top(self.model, value, k=1)[0]["word"]   # what the value direction promotes
        ans_first_id = ans_ids[0]
        multi = len(ans_ids) > 1
        with self.lock:
            label = f"m{self._next_id}"
            self._next_id += 1
            entry = {
                "label": label, "key": key, "value": value, "eta": float(eta),
                "ans_id": int(ans_first_id),          # the FIRST answer token (what recall promotes)
                "ans_ids": [int(t) for t in ans_ids], # all answer tokens (for legible read-back)
                "n_tok": len(ans_ids), "multi": bool(multi),
                "cue": cue, "answer": answer,
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
        """Public, JSON-safe view of one entry: label + decoded value + salience + key fingerprint.
        For a multi-token answer, `decodes_ok` means the value promotes the answer's FIRST token (all we
        claim), and `multi`/`n_tok`/`first_token` let the UI say so honestly."""
        key = entry["key"]
        value = entry["value"]
        topdims = [int(d) for d in key.abs().topk(6).indices]
        decoded = entry["decoded_word"]
        first_id = entry.get("ans_id")
        first_tok = tok_str(self.model, first_id) if first_id is not None else decoded
        # ok = the value direction decodes to the answer's FIRST token (single-token: the whole answer).
        ok = (single_token_id(self.model, decoded) == first_id
              or decoded.strip() == first_tok.strip()
              or decoded.strip() == entry["answer"].strip())
        return {
            "label": entry["label"],
            "cue": entry["cue"],
            "answer": entry["answer"].strip() or entry["answer"],
            "decoded_word": decoded.strip() or decoded,
            "value_decodes_ok": bool(ok),
            "multi": bool(entry.get("multi", False)),
            "n_tok": int(entry.get("n_tok", 1)),
            "first_token": first_tok.strip() or first_tok,
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
    def _address(self, qkey: torch.Tensor, entries: list[dict], gate: float):
        """Gated hard top-1 over cosine: nearest stored key wins IF it clears `gate`, else nothing fires.
        Returns (weights[n], selected_index_or_None, nearest_cosine)."""
        keys = torch.stack([e["key"] for e in entries])            # [n, d_mlp]
        cos = F.normalize(keys, dim=-1) @ F.normalize(qkey, dim=-1)  # [n]
        sel = int(cos.argmax())
        w = torch.zeros_like(cos)
        if float(cos[sel]) >= gate:
            w[sel] = float(entries[sel]["eta"])
            return w, sel, float(cos[sel])
        return w, None, float(cos[sel])                            # gated off -> no injection

    @torch.no_grad()
    def recall(self, cue: str, k: int = 5, gate: float | None = None):
        """Query `cue` with the CURRENT memory active. Captures the query key at the final position,
        injects the addressed memory contribution into resid_post at layer L, and returns
        (top-k [{word,prob}], full probs tensor, selected_entry_or_None, nearest_cosine, gate_used).
        `gate` overrides the live self.gate for THIS query only (the strictness slider previews a value
        without committing it). When the nearest key is below the gate nothing is injected (sel is None)
        and the model returns its exact baseline -- so a wrong-keyed query honestly reports fired=null."""
        g = self.gate if gate is None else self._clamp_gate(gate)
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
            w, sel, c = self._address(cap["q"], entries, g)
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
        return out, probs, sel_entry, cap["cos"], g


# ====================================================================================================
# The "WHAT IS IT THINKING" read: a live, honest probe of the SAME frozen GPT-2's interior as it takes
# in the asked prompt. We project the residual stream at every token onto a fixed set of NAMED concept
# directions (the validated p18 diff-in-means basis -- the same directions thinking_panel.py renders)
# and report, per token, which concepts rise above an EQUAL-NORM random-direction null. This is a READ,
# a probe of what is PRESENT in the state -- NOT a claim that the model "decided" anything and NOT what
# it will output. The basis is built ONCE at startup against the already-loaded model (no second model,
# no extra download); each /thinking call is one cached forward pass + projections. Logic lifted verbatim
# from inspector/demo/thinking_panel.py (per_token_resid / neutral_baseline / null_stats / read_trace).
# ====================================================================================================
# WHICH concepts to show + a human label and Planet Maiko accent per lane. Directions come from the p18
# basis (so they are the SAME validated diff-in-means dirs); these entries pick display order/label/color.
CONCEPT_DISPLAY = [
    ("animals",  "animals",     "#C4F542"),   # Toxic Lime
    ("fear",     "fear",        "#FF6FAF"),   # Neon Pink
    ("colors",   "color",       "#6FE0E8"),   # Frozen Cyan
    ("money",    "money",       "#FFE66D"),   # Star Yellow
    ("formal",   "formal tone", "#C9A6FF"),   # Soft Lavender Glow
    ("past",     "past tense",  "#1FB5E5"),   # Electric Ice
    ("food",     "food",        "#FFB36F"),   # warm amber (in-family accent)
    ("question", "questions",   "#A0FFD6"),   # mint (in-family accent)
]

# A small neutral corpus to CENTER each concept's projection: we subtract each concept's mean projection
# over these topic-free tokens so a concept that is merely "always a little on" (a frequency / residual-
# norm artifact) sits near zero, and only genuine above-baseline presence shows. (From thinking_panel.)
NEUTRAL_PROMPTS = [
    "The next thing I want to talk about is the",
    "When I opened the door, I saw the",
    "Let me tell you about the",
    "The most important part of the story is the",
    "After a while, they noticed the",
    "I think the answer has to do with the",
    "It was a normal day and nothing much happened.",
    "She walked along the path and looked around.",
]


class ConceptReader:
    """Builds the named diff-in-means concept basis ONCE (reusing the server's frozen GPT-2) and reads,
    for any prompt, the per-token concept activations as z = sigma above an equal-norm random null. A
    concept only "lights" where its real projection beats that random band (z >= Z_THRESH). Read-only:
    it never mutates the model or the board -- a probe of what is present in the state, not a decision."""

    Z_THRESH = 2.0    # sigma-above-null to count a concept as "present" (matches thinking_panel default)
    N_NULL = 400      # random equal-norm directions per token for the null band (the honesty knob)

    def __init__(self, model, layer: int, seed: int = 0):
        self.model = model
        self.layer = layer
        self.seed = seed
        # pick the concept subset from the p18 basis, preserving display order
        by_name = {c["name"]: c for c in P18_CONCEPTS}
        names_internal = [n for (n, _disp, _col) in CONCEPT_DISPLAY]
        missing = [n for n in names_internal if n not in by_name]
        if missing:
            raise ValueError(f"concepts not found in p18 basis: {missing}")
        self.concepts = [by_name[n] for n in names_internal]
        self.labels = [disp for (_n, disp, _c) in CONCEPT_DISPLAY]
        self.colors = [c for (_n, _d, c) in CONCEPT_DISPLAY]
        self.k = len(self.concepts)
        # STEP 1: the named diff-in-means basis at L (REUSED from p18) -- built ONCE here.
        self.dirs, self.units, self.norms = build_basis(model, self.concepts, layer)   # [k,d],[k,d],[k]
        cos = (self.units @ self.units.T).cpu().numpy()
        off = cos.copy(); np.fill_diagonal(off, np.nan)
        self.cosine_mean_abs_off = float(np.nanmean(np.abs(off)))
        # STEP 2: neutral-corpus baseline per concept (subtracted to remove the always-on offset).
        self.baseline_unit = self._neutral_baseline()                                  # [k]

    @torch.no_grad()
    def _per_token_resid(self, text: str):
        """resid_post at L for every token of `text`, plus token strings (BOS dropped: its residual is a
        ~30x-norm outlier that would swamp every projection)."""
        toks = self.model.to_tokens(text)                              # [1, seq] (prepends BOS)
        name = f"blocks.{self.layer}.hook_resid_post"
        _, cache = self.model.run_with_cache(toks, names_filter=name)
        resid = cache[name][0][1:]                                     # [seq-1, d_model], drop BOS
        strs = [self.model.to_string(t) for t in toks[0][1:]]          # matching token strings
        return resid, strs

    @torch.no_grad()
    def _neutral_baseline(self):
        """Mean projection of each UNIT concept dir over the neutral corpus -> [k], subtracted from every
        token's projection so the trace measures ABOVE-baseline presence, not absolute projection."""
        accum = torch.zeros(self.k, device=self.units.device)
        cnt = 0
        for p in NEUTRAL_PROMPTS:
            resid, _ = self._per_token_resid(p)                        # [s, d_model]
            proj = resid @ self.units.T                                # [s, k]
            accum = accum + proj.sum(0)
            cnt += proj.shape[0]
        return accum / max(cnt, 1)

    @torch.no_grad()
    def read(self, prompt: str):
        """The READ for one prompt. For every token: project resid onto each unit concept dir, center by
        the neutral baseline, and z-score against an equal-norm random null at that token --
            z[t,c] = (proj[t,c] - baseline[c] - null_mean[t]) / null_std[t]
        in units of sigma above a random direction (norm cancels in z). Returns a compact dict: tokens +
        per-token list of {c, z} for concepts clearing Z_THRESH, plus per-concept peak. A fresh, SEEDED
        generator makes the null reproducible per prompt."""
        resid, token_strs = self._per_token_resid(prompt)              # [seq,d], list
        seq, d_model = resid.shape
        proj = resid @ self.units.T                                    # [seq, k] onto unit dirs
        proj_centered = proj - self.baseline_unit[None, :]             # center by neutral baseline

        # equal-norm random null on UNIT directions (so it is directly comparable to proj on unit dirs):
        # one shared bank of random UNIT dirs; a random unit dir has the same projection statistics
        # regardless of which concept, so the null mean/std is per-token (shared across concepts).
        g = torch.Generator(device=self.model.cfg.device).manual_seed(self.seed)
        rand_units = F.normalize(torch.randn(self.N_NULL, d_model, generator=g,
                                             device=resid.device), dim=-1)      # [n_null, d_model]
        proj_unit = resid @ rand_units.T                               # [seq, n_null]
        base_mean = proj_unit.mean(dim=1)                              # [seq]
        base_std = proj_unit.std(dim=1)                                # [seq]
        # center the null the same way (subtract baseline) so z compares like-for-like.
        null_mean_centered = base_mean[:, None] - self.baseline_unit[None, :]   # [seq, k]
        z = (proj_centered - null_mean_centered) / (base_std[:, None] + 1e-9)   # [seq, k] sigma vs null
        z = z.cpu().numpy()

        # compact JSON: per token, only the concepts that cleared the null (z >= Z_THRESH), with z.
        tokens = []
        for ti in range(seq):
            lit = [{"c": j, "z": round(float(z[ti, j]), 2)}
                   for j in range(self.k) if z[ti, j] >= self.Z_THRESH]
            lit.sort(key=lambda d: -d["z"])
            tokens.append({"tok": token_strs[ti], "lit": lit})
        peak_z = z.max(axis=0) if seq else np.zeros(self.k)            # [k] per-concept peak over prompt
        peak_tok = z.argmax(axis=0) if seq else np.zeros(self.k, dtype=int)
        concepts = [
            {"name": self.labels[j], "color": self.colors[j],
             "peak_z": round(float(peak_z[j]), 2),
             "peak_tok": int(peak_tok[j]),
             "lit": bool(peak_z[j] >= self.Z_THRESH)}
            for j in range(self.k)
        ]
        return {
            "prompt": prompt,
            "layer": self.layer,
            "z_thresh": self.Z_THRESH,
            "n_null": self.N_NULL,
            "concepts": concepts,            # the named basis: order, label, color, peak z, lit?
            "tokens": tokens,                # per-token lit concepts (compact: only those above null)
            "cosine_mean_abs_off": round(self.cosine_mean_abs_off, 3),
        }


# ====================================================================================================
# The HTTP layer: a tiny stdlib JSON server (no Flask/FastAPI dependency, so .venv-sae is untouched).
# CORS is wide-open (Access-Control-Allow-Origin: *) so a local HTML file can call it. Single shared
# MEMORY + model; the GlassBoxMemory lock guards mutation, and ThreadingHTTPServer handles concurrent
# requests. Model forward passes are single and CPU-cheap (one pass for baseline, one for with-memory).
# ====================================================================================================
class App:
    """Holds the loaded model + the one live memory. The handler routes requests to these methods,
    each returning (status_code, json_dict)."""

    def __init__(self, model, layer: int, device: str, reader: "ConceptReader | None" = None,
                 gate: float | None = None):
        self.model = model
        self.layer = layer
        self.device = device
        self.mem = GlassBoxMemory(model, layer, gate=gate)
        self.reader = reader     # the "what is it thinking" concept probe (basis built once at startup)
        self.model_name = "gpt2 (GPT-2-small, 124M, frozen)"

    # -- POST /write {cue, answer} -----------------------------------------------------------------
    def write(self, body: dict):
        cue = body.get("cue")
        answer = body.get("answer")
        eta = body.get("eta", 10.0)
        if not isinstance(cue, str) or not cue:
            return 400, {"error": "missing or empty 'cue' (a string ending right before the answer)"}
        if not isinstance(answer, str) or not answer:
            return 400, {"error": "missing or empty 'answer' (a word, e.g. ' blue' or ' mochi')"}
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

    # -- POST /query {prompt, topk?, gate?} --------------------------------------------------------
    def query(self, body: dict):
        prompt = body.get("prompt")
        topk = int(body.get("topk", 5) or 5)
        topk = max(1, min(topk, 20))
        if not isinstance(prompt, str) or not prompt:
            return 400, {"error": "missing or empty 'prompt'"}
        # optional per-query gate override (the strictness slider previews a value without committing it).
        gate_override = body.get("gate", None)
        if gate_override is not None and not isinstance(gate_override, (int, float)):
            return 400, {"error": "'gate' must be a number (cosine threshold, e.g. 0.85)"}
        # baseline: NO memory (clean frozen model) -- a real, separate forward pass.
        baseline = topk_preds(self.model, prompt, k=topk)
        # with_memory: the SAME prompt with the live memory hook active (gate = override or live gate).
        with_mem, _probs, sel_entry, cos, gate_used = self.mem.recall(
            prompt, k=topk, gate=(float(gate_override) if gate_override is not None else None))
        fired = None
        if sel_entry is not None:
            fired = {"label": sel_entry["label"], "match_score": round(float(cos), 4),
                     "answer": sel_entry["answer"].strip() or sel_entry["answer"],
                     "decoded_word": sel_entry["decoded_word"].strip() or sel_entry["decoded_word"],
                     "multi": bool(sel_entry.get("multi", False)),
                     "n_tok": int(sel_entry.get("n_tok", 1)),
                     "first_token": (tok_str(self.model, sel_entry["ans_id"]).strip()
                                     if sel_entry.get("ans_id") is not None else None)}
        return 200, {
            "prompt": prompt,
            "baseline": baseline,
            "with_memory": with_mem,
            "fired": fired,
            "nearest_cosine": round(float(cos), 4) if cos is not None else None,
            "gate": round(float(gate_used), 4),     # the gate this query actually used
            "live_gate": round(float(self.mem.gate), 4),
        }

    # -- POST /gate {gate} -------------------------------------------------------------------------
    def gate(self, body: dict):
        """Set the live match-strictness gate (cosine threshold) without restarting. Clamped to range."""
        g = body.get("gate")
        if not isinstance(g, (int, float)):
            return 400, {"error": "missing or non-numeric 'gate' (cosine threshold, e.g. 0.85)"}
        new_gate = self.mem.set_gate(float(g))
        return 200, {"ok": True, "gate": round(float(new_gate), 4),
                     "gate_min": self.mem.GATE_MIN, "gate_max": self.mem.GATE_MAX}

    # -- POST /thinking {prompt} -------------------------------------------------------------------
    def thinking(self, body: dict):
        """The "what is it thinking" READ for the asked prompt: per-token named-concept activations as
        z above an equal-norm random null (only concepts that clear the null are returned). A probe of
        what is PRESENT in the residual state as the model reads, NOT a decision and NOT the output."""
        prompt = body.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            return 400, {"error": "missing or empty 'prompt'"}
        if self.reader is None:
            return 503, {"error": "concept reader unavailable (basis failed to build at startup)"}
        return 200, self.reader.read(prompt)

    # -- GET /health -------------------------------------------------------------------------------
    def health(self):
        thinking = None
        if self.reader is not None:
            thinking = {
                "read_layer": self.reader.layer,
                "n_concepts": self.reader.k,
                "concepts": self.reader.labels,
                "z_thresh": self.reader.Z_THRESH,
                "n_null": self.reader.N_NULL,
            }
        return 200, {
            "ok": True,
            "model": self.model_name,
            "device": self.device,
            "layer": self.layer,
            "d_model": int(self.model.cfg.d_model),
            "d_mlp": int(self.model.cfg.d_mlp),
            "n_layers": int(self.model.cfg.n_layers),
            "gate": round(float(self.mem.gate), 4),
            "gate_default": self.mem.DEFAULT_GATE,
            "gate_min": self.mem.GATE_MIN,
            "gate_max": self.mem.GATE_MAX,
            "n_entries": len(self.mem.snapshot()),
            "thinking": thinking,
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

        def _send_html(self, status: int, body: str):
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")   # always serve the live UI fresh
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
            # GET "/" serves the live HTML frontend (read fresh each request so edits show up on reload).
            if path == "/":
                try:
                    with open(FRONTEND_PATH, "r", encoding="utf-8") as fh:
                        self._send_html(200, fh.read())
                except FileNotFoundError:
                    self._send_html(
                        500,
                        "<h1>memory_live.html not found</h1>"
                        f"<p>Expected the frontend next to the server at:<br><code>{esc_html(FRONTEND_PATH)}</code></p>"
                        "<p>The JSON API is still live (GET /health, POST /query, ...).</p>",
                    )
                except Exception as e:  # noqa: BLE001
                    self._send_html(500, f"<h1>error serving frontend</h1><pre>{esc_html(f'{type(e).__name__}: {e}')}</pre>")
                return
            try:
                if path == "/health":
                    status, payload = app.health()
                elif path == "/memory":
                    status, payload = app.memory()
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
                elif path == "/gate":
                    status, payload = app.gate(body)
                elif path == "/thinking":
                    status, payload = app.thinking(body)
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
    "GET  /                           -> the live HTML frontend (memory_live.html)",
    "POST /write    {cue, answer}     -> {label, decoded_word, salience, key_fingerprint}",
    "GET  /memory                     -> {entries:[...]}  (the cards)",
    "POST /delete   {label}           -> {ok}",
    "POST /salience {label, eta}      -> the updated entry",
    "POST /query    {prompt, topk?, gate?} -> {baseline, with_memory, fired, nearest_cosine, gate}",
    "POST /gate     {gate}            -> {ok, gate}  (set the live match-strictness threshold)",
    "POST /thinking {prompt}          -> {concepts, tokens:[{tok, lit:[{c,z}]}]}  (what it's thinking)",
    "GET  /health                     -> {model, layer, gate, n_entries, thinking}",
]


# ====================================================================================================
def main():
    ap = argparse.ArgumentParser(description="Live backend for the Clozn glass-box memory window.")
    ap.add_argument("--host", default="127.0.0.1", help="bind host (default loopback; 0.0.0.0 for LAN)")
    ap.add_argument("--port", type=int, default=8077, help="bind port (default 8077)")
    ap.add_argument("--layer", type=int, default=10,
                    help="memory write/read layer (default 10: deeper keying -> cleanest paraphrase vs "
                         "unrelated separation, ~+0.27 cos margin, and stronger recall than L8; "
                         "p15/p17 originally used 8)")
    ap.add_argument("--gate", type=float, default=None,
                    help=f"initial match-strictness gate (cosine; default {GlassBoxMemory.DEFAULT_GATE}); "
                         f"adjustable live via POST /gate or the strictness slider")
    ap.add_argument("--think-layer", type=int, default=7,
                    help="resid layer for the 'what it's thinking' concept read (p18 basis default = 7)")
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

    # Build the "what is it thinking" concept basis ONCE here, reusing the just-loaded frozen model (no
    # second model, no extra download). If it fails, the memory app still runs and /thinking reports 503.
    reader = None
    try:
        print(f"building concept basis for 'what it's thinking' @ blocks.{args.think_layer}."
              f"hook_resid_post (diff-in-means, p18) ...", flush=True)
        reader = ConceptReader(model, args.think_layer, seed=0)
        print(f"  basis built: {reader.k} named concepts {reader.labels}  "
              f"(mean |off-diag cosine| = {reader.cosine_mean_abs_off:.3f})", flush=True)
    except Exception as e:  # noqa: BLE001 -- never block the memory server on the optional concept read
        print(f"  !! concept basis build failed ({type(e).__name__}: {e}); "
              f"/thinking will return 503, memory app unaffected", flush=True)

    app = App(model, args.layer, args.device, reader=reader, gate=args.gate)

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

    have_frontend = os.path.exists(FRONTEND_PATH)
    print("\n" + "=" * 78)
    print(f"  CLOZN MEMORY SERVER  ->  {url}")
    print("=" * 78)
    print(f"  model: {app.model_name}   layer: blocks.{args.layer}.mlp   "
          f"gate: {app.mem.gate} (live; POST /gate to change)")
    if reader is not None:
        print(f"  thinking: concept read @ blocks.{reader.layer}.hook_resid_post  "
              f"({reader.k} concepts, z>={reader.Z_THRESH} vs {reader.N_NULL}-sample null)")
    else:
        print("  thinking: UNAVAILABLE (basis build failed) -- POST /thinking returns 503")
    if have_frontend:
        print(f"  >> open {url}/  in a browser for the LIVE UI  (serving memory_live.html)")
    else:
        print(f"  !! {FRONTEND_PATH} not found -- GET / will 500; JSON API still live")
    print("  endpoints (CORS enabled; the page is served same-origin):")
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
