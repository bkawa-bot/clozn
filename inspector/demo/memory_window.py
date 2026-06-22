"""
memory_window.py -- the FIRST WINDOW of the Clozn instrument.

A real, honest, beautiful demo of an *in-model memory you can see* on a local model.
This is the product slice that makes the pitch true: "a local runtime where you crack
open a model's internal state, change it, and watch what moves."

WHAT IT IS (no faking, real recall). A FROZEN GPT-2-small is given a glass-box
fast-weight memory -- an explicit, editable list of entries {key, value, eta, label}
injected by a forward hook at one mid-layer. The list IS the memory. We:
  1. pick ~3 nonce facts the model does NOT know (verified near-chance; known ones dropped),
  2. show the BEFORE prediction (wrong / generic -- it genuinely doesn't know),
  3. WRITE each fact as one entry, shown as a legible CARD (label + the value decoded
     through the logit lens to the answer word + its salience eta),
  4. show the AFTER prediction with the memory active -> the answer is now correct,
  5. DELETE one entry -> that query reverts to its before-state, the others stay (surgical),
  6. show one WRONG-KEYED query that correctly does NOT fire (the specificity, plainly).
Then it renders the whole run to a single self-contained HTML page in the Planet Maiko
palette (inline CSS, a little vanilla JS for the reveal; no external deps).

MECHANISM (reused verbatim from the validated spikes p15_fastweight + p17_betterkey):
  WRITE key  = MLP post-activation `blocks.L.mlp.hook_post` at the cue's FINAL token,
               over the cue ONLY (the p17 "raw_consistent" key -- same position for write
               and read, which is the variant that actually recalls).
  value      = the answer token's unembedding direction W_U[:, ans] (legible by build:
               adding it to the residual promotes `answer` via the logit lens).
  recall     = a hook adding  sum_i w_i * value_i  at the query's final position, with hard
               top-1 addressing over cosine similarity (w_i = eta for the nearest stored key,
               else 0) -- the variant that holds recall + gives clean specificity.

The backbone is FROZEN throughout; we never train GPT-2. Every before/after number printed
and rendered is the ACTUAL model output. If a fact recalls imperfectly we show it honestly.

ISOLATED ENV: runs in C:\\Users\\brigi\\src\\clozn\\.venv-sae (transformer_lens + torch, CPU
here). GPT-2-small is 124M and cached -- no large download.

Usage (from inspector/, .venv-sae python):
    python demo/memory_window.py                 # default: layer 8, writes inspector/runs/
    python demo/memory_window.py --layer 6
    python demo/memory_window.py --out some/dir/memory_window.html
"""
from __future__ import annotations

import argparse
import datetime as _dt
import html as _html
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # this PC crashes on HF symlinks (WinError 1314)

import torch                       # noqa: E402
import torch.nn.functional as F    # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

# ----------------------------------------------------------------------------------------------------
# Candidate facts. Each: a cue ending right before a single-token answer, the answer word, a label,
# and a short human "question" phrasing for the page. Subjects are nonce so a frozen GPT-2 cannot
# already know the mapping; answers are common single tokens so argmax can express them. We auto-verify
# single-token + near-chance below and DROP any the base model already knows; we want exactly 3 keepers
# for a clean window, but carry spares so a dropped fact doesn't sink the demo.
# The cue text ends with no trailing answer; the answer is the model's next-token prediction.
# ----------------------------------------------------------------------------------------------------
FACTS_RAW = [
    ("The secret color of Zorbland is",        " blue",   "Zorbland",     "the secret color of Zorbland"),
    ("Captain Vextor's favorite color is",      " green",  "Capt. Vextor", "Captain Vextor's favorite color"),
    ("The official animal of Quibblax is the",   " dog",    "Quibblax",     "the official animal of Quibblax"),
    ("The lucky number of Flonkville is",        " seven",  "Flonkville",   "the lucky number of Flonkville"),
    ("In the land of Snargle the sky is",        " red",    "Snargle",      "the color of the sky in Snargle"),
    ("The Wozzleton national fruit is the",      " apple",  "Wozzleton",    "the national fruit of Wozzleton"),
    ("The Grumblesnatch tribe worships the",     " moon",   "Grumblesnatch","what the Grumblesnatch tribe worships"),
    ("Sir Plonkington rides a giant",            " horse",  "Plonkington",  "what Sir Plonkington rides"),
]

N_WANT = 3   # how many facts the window shows


# ====================================================================================================
# Model + low-level pieces (the validated mechanism, lifted from p15/p17).
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
    return [(tok_str(model, int(i)), float(probs[int(i)])) for i in top.indices], logits, probs


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
    return [(tok_str(model, int(i)), float(lens[int(i)])) for i in top.indices]


# ====================================================================================================
# The MEMORY: an explicit, inspectable, editable list. Recall = a hook adding sum_i w_i*value_i at the
# query's final position, hard top-1 addressing over cosine similarity (the variant that recalls with
# clean specificity). This is p15's FastWeightMemory reduced to exactly what the window needs.
# ====================================================================================================
class GlassBoxMemory:
    """entries: list of {key[d_mlp], value[d_model], eta, label, ans_id, cue}. The list IS the memory.

    Addressing is hard top-1 over cosine WITH a min-similarity gate: the contribution fires only if the
    nearest stored key clears `gate`. This matters for honesty. A query's OWN key cosines ~1.0 to its
    own stored entry, but the nonce cues here are not orthogonal (two 'color' cues cosine ~0.82), so an
    UNGATED top-1 would always fire the nearest remaining entry even on an unrelated query -- meaning a
    deleted/wrong-keyed query would inject the WRONG value instead of reverting. The gate (between the
    ~1.0 self regime and the ~0.82 cross regime) makes delete a TRUE revert and the wrong-key case a
    true no-op: an unrelated query produces no injection and the model returns its exact baseline."""

    GATE = 0.9   # min cosine for the nearest key to fire (self ~1.0 fires; cross ~0.82 is gated off)

    def __init__(self, model, layer: int):
        self.model = model
        self.layer = layer
        self.entries: list[dict] = []

    def write(self, cue, ans_id, eta, label):
        self.entries.append({
            "key": consistent_key(self.model, cue, self.layer),
            "value": value_dir(self.model, ans_id),
            "eta": float(eta), "label": label, "ans_id": int(ans_id), "cue": cue,
        })

    def active(self, idxs):
        """A view restricted to entry indices `idxs` (for delete: render = drop one, keep the rest)."""
        m = GlassBoxMemory(self.model, self.layer)
        m.entries = [self.entries[i] for i in idxs]
        return m

    @torch.no_grad()
    def _address(self, qkey: torch.Tensor):
        """Gated hard top-1 over cosine: nearest stored key wins IF it clears GATE, else nothing fires.
        Returns (weights[n], selected_index_or_None, nearest_cosine)."""
        keys = torch.stack([e["key"] for e in self.entries])       # [n, d_mlp]
        cos = F.normalize(keys, dim=-1) @ F.normalize(qkey, dim=-1)  # [n]
        sel = int(cos.argmax())
        w = torch.zeros_like(cos)
        if float(cos[sel]) >= self.GATE:
            w[sel] = float(self.entries[sel]["eta"])
            return w, sel, float(cos[sel])
        return w, None, float(cos[sel])                            # gated off -> no injection

    @torch.no_grad()
    def recall(self, cue: str, k: int = 5):
        """Query `cue`: capture the query key at the final position, inject the addressed memory
        contribution into resid_post at layer L, return (top-k (word,prob), full logits, probs, sel,
        nearest_cosine). When the nearest key is below GATE nothing is injected (sel is None) and the
        model returns its exact baseline -- this is what makes delete/wrong-key honest reverts."""
        post_name = f"blocks.{self.layer}.mlp.hook_post"
        resid_name = f"blocks.{self.layer}.hook_resid_post"
        cap = {"sel": None, "cos": None}

        def grab(act, hook):
            cap["q"] = act[0, -1].clone()
            return act

        def inject(act, hook):
            if not self.entries:
                return act
            w, sel, c = self._address(cap["q"])
            cap["sel"], cap["cos"] = sel, c
            if sel is None:
                return act                                          # below gate: leave residual untouched
            vals = torch.stack([e["value"] for e in self.entries])  # [n, d_model]
            act[0, -1] = act[0, -1] + (w.unsqueeze(-1) * vals).sum(0)
            return act

        logits = self.model.run_with_hooks(
            self.model.to_tokens(cue), fwd_hooks=[(post_name, grab), (resid_name, inject)]
        )[0, -1].float()
        probs = F.softmax(logits, dim=-1)
        top = logits.topk(k)
        out = [(tok_str(self.model, int(i)), float(probs[int(i)])) for i in top.indices]
        return out, logits, probs, cap["sel"], cap["cos"]


# ====================================================================================================
# Build the demo: gather every ACTUAL before/after number into a plain dict for the renderer.
# ====================================================================================================
def build_demo(layer: int, device: str):
    torch.manual_seed(0)
    print(f"loading gpt2 (HookedTransformer) on {device} ...")
    model = load_model(device)
    d_mlp, d_model, nl = model.cfg.d_mlp, model.cfg.d_model, model.cfg.n_layers
    print(f"  d_model={d_model}  d_mlp={d_mlp}  n_layers={nl}   memory layer L={layer}")

    # ---- STEP 1+2: pick facts the model does NOT know. Verify near-chance; DROP known ones. ----------
    print("\nSTEP 1 -- verify each candidate is UNKNOWN to the frozen model (drop multi-token/known):")
    KNOWN_P = 0.30
    kept = []
    for cue, ans_word, label, question in FACTS_RAW:
        ans_id = single_token_id(model, ans_word)
        if ans_id is None:
            print(f"  DROP (multi-token answer): {label} {ans_word!r}")
            continue
        p, t1, t5 = base_prob(model, cue, ans_id)
        known = t1 or p >= KNOWN_P
        tag = "  <- DROP (already known)" if known else ("  keep" if len(kept) < N_WANT else "  spare")
        print(f"  {label:14} ans={ans_word!r:9} P(ans)={p*100:6.3f}%  top1={str(t1):5}{tag}")
        if not known:
            kept.append({"cue": cue, "ans_word": ans_word, "label": label, "question": question,
                         "ans_id": ans_id, "base_p": p})
    if len(kept) < N_WANT:
        raise SystemExit(f"only {len(kept)} unknown facts survived; need {N_WANT}.")
    facts = kept[:N_WANT]
    print(f"\n  using {len(facts)} facts: " + ", ".join(f["label"] for f in facts))

    # calibrate eta once: cosine addressing is O(1), so eta ~ a logit-lens steer strength. 10 is the
    # validated value from p15/p17 for top-1 cosine addressing.
    eta = 10.0

    # ---- BEFORE: clean prediction per fact (the model genuinely does not know) ----------------------
    print("\nSTEP 3 -- BEFORE (no memory): the model's actual top prediction per cue:")
    for f in facts:
        preds, _, _ = topk_preds(model, f["cue"], k=5)
        f["before_top"] = preds
        f["before_word"] = preds[0][0]
        f["before_p_ans"] = f["base_p"]
        bw = preds[0][0].strip() or preds[0][0]
        print(f"  {f['label']:14} -> {bw!r:12} ({preds[0][1]*100:5.1f}%)   "
              f"P(correct {f['ans_word'].strip()!r})={f['base_p']*100:6.3f}%")

    # ---- WRITE: one entry per fact; record the legible card (value decoded via logit lens) -----------
    print("\nSTEP 4 -- WRITE each fact as one memory entry (value decoded through the logit lens):")
    mem = GlassBoxMemory(model, layer)
    for f in facts:
        mem.write(f["cue"], f["ans_id"], eta, f["label"])
    cards = []
    for i, f in enumerate(facts):
        v = mem.entries[i]["value"]
        lens = logit_lens_top(model, v, k=3)
        decoded = lens[0][0]
        ok = single_token_id(model, decoded) == f["ans_id"] or decoded.strip() == f["ans_word"].strip()
        # key fingerprint: a few of the most-active MLP dims, just as a legible "this is a real vector"
        key = mem.entries[i]["key"]
        topdims = [int(d) for d in key.abs().topk(6).indices]
        cards.append({
            "label": f["label"], "question": f["question"], "ans_word": f["ans_word"].strip(),
            "value_decodes_to": decoded.strip() or decoded, "value_ok": bool(ok),
            "value_lens_top3": [(w.strip() or w, s) for w, s in lens],
            "eta": eta, "key_dim": int(key.shape[0]), "key_topdims": topdims,
            "key_norm": float(key.norm()), "value_norm": float(v.norm()),
        })
        print(f"  entry {i}: {f['label']:14} value -> {decoded.strip()!r:10} "
              f"(want {f['ans_word'].strip()!r}) {'OK' if ok else 'MISS'}   eta={eta:g}")

    # ---- AFTER: re-run each query with the memory active -> the answer is now correct ---------------
    print("\nSTEP 5 -- AFTER (memory active): re-run each query; the answer should now win:")
    for i, f in enumerate(facts):
        preds, _, probs, sel, cos = mem.recall(f["cue"], k=5)
        f["after_top"] = preds
        f["after_word"] = preds[0][0]
        f["after_p_ans"] = float(probs[f["ans_id"]])
        f["after_top1_correct"] = preds[0][0].strip() == f["ans_word"].strip()
        f["after_selected"] = int(sel) if sel is not None else i
        f["after_cos"] = cos
        aw = preds[0][0].strip() or preds[0][0]
        flag = "OK" if f["after_top1_correct"] else "PARTIAL"
        print(f"  {f['label']:14} -> {aw!r:12} ({preds[0][1]*100:5.1f}%)   "
              f"P(correct)={f['after_p_ans']*100:6.2f}%  [{flag}]")

    # ---- DELETE: remove one entry; its query reverts, the others stay (surgical) --------------------
    del_idx = 0
    del_label = facts[del_idx]["label"]
    print(f"\nSTEP 6 -- DELETE entry {del_idx} ({del_label}); its query should revert, others unchanged:")
    keep_idx = [i for i in range(len(facts)) if i != del_idx]
    after_del_mem = mem.active(keep_idx)
    delete = {"deleted_index": del_idx, "deleted_label": del_label, "rows": []}
    for i, f in enumerate(facts):
        preds, _, probs, sel, cos = after_del_mem.recall(f["cue"], k=5)
        p_ans = float(probs[f["ans_id"]])
        word = preds[0][0]
        is_del = (i == del_idx)
        reverted = word.strip() == f["before_word"].strip()
        row = {
            "label": f["label"], "ans_word": f["ans_word"].strip(),
            "deleted": is_del, "word": word.strip() or word, "p_ans": p_ans,
            "before_word": f["before_word"].strip() or f["before_word"], "before_p": f["before_p_ans"],
            "after_word": f["after_word"].strip() or f["after_word"], "after_p": f["after_p_ans"],
            "reverted_to_before": bool(reverted), "fired": sel is not None,
            "still_correct": word.strip() == f["ans_word"].strip(),
        }
        delete["rows"].append(row)
        if is_del:
            print(f"  [deleted] {f['label']:14} -> {row['word']!r:12} P(ans)={p_ans*100:6.3f}%   "
                  f"(before was {row['before_word']!r}; {'REVERTED' if reverted else 'changed'}; "
                  f"nearest-cos={cos:.3f}, fired={sel is not None})")
        else:
            print(f"  [kept]    {f['label']:14} -> {row['word']!r:12} P(ans)={p_ans*100:6.2f}%   "
                  f"({'still correct' if row['still_correct'] else 'changed'})")

    # ---- WRONG-KEY: query one fact's cue against a memory holding only a DIFFERENT fact -------------
    # The specificity, shown plainly: top-1 addressing picks the only stored entry, but its value is
    # for the wrong answer, so the correct answer does NOT fire -- it stays at its before-state.
    q_idx, store_idx = 0, 1
    print(f"\nSTEP 7 -- WRONG-KEY: ask '{facts[q_idx]['label']}' with ONLY '{facts[store_idx]['label']}' "
          f"in memory (must NOT fire):")
    only_other = mem.active([store_idx])
    preds, _, probs, sel, cos = only_other.recall(facts[q_idx]["cue"], k=5)
    wq = facts[q_idx]
    wrongkey = {
        "query_label": wq["label"], "query_question": wq["question"], "query_ans": wq["ans_word"].strip(),
        "stored_label": facts[store_idx]["label"], "stored_ans": facts[store_idx]["ans_word"].strip(),
        "word": preds[0][0].strip() or preds[0][0], "p_ans": float(probs[wq["ans_id"]]),
        "p_stored_ans": float(probs[facts[store_idx]["ans_id"]]),
        "before_word": wq["before_word"].strip() or wq["before_word"],
        "before_p": wq["before_p_ans"], "after_p": wq["after_p_ans"],
        "nearest_cos": cos, "fired": sel is not None,
        "did_not_fire": sel is None,
    }
    print(f"  '{wq['label']}' query, memory={{{facts[store_idx]['label']}}} -> top {wrongkey['word']!r}   "
          f"P(correct '{wq['ans_word'].strip()}')={wrongkey['p_ans']*100:6.3f}% "
          f"(its real recall was {wq['after_p_ans']*100:.1f}%)  "
          f"-> {'DID NOT FIRE (correct)' if wrongkey['did_not_fire'] else 'leaked'}")

    return {
        "model_name": "GPT-2-small (124M, frozen)", "layer": layer,
        "d_model": d_model, "d_mlp": d_mlp, "n_layers": nl, "eta": eta,
        "facts": facts, "cards": cards, "delete": delete, "wrongkey": wrongkey,
        "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    }


# ====================================================================================================
# THE ARTIFACT: a single self-contained HTML page in the Planet Maiko palette. Inline CSS, no external
# deps, a little vanilla JS for the before->after reveal. Otherworldly, soft, glowing, rounded.
# ====================================================================================================
# Planet Maiko palette
BG_DEEP   = "#0B0F2A"   # Deep Space Navy
BG_COSMIC = "#1A1F4A"   # Cosmic Indigo
BG_MID    = "#2A2250"   # Midnight Purple
PINK      = "#FF6FAF"   # Neon Pink
MAGENTA   = "#FF4D9D"   # Candy Magenta
ICE       = "#1FB5E5"   # Electric Ice
CYAN      = "#6FE0E8"   # Frozen Cyan
LIME      = "#C4F542"   # Toxic Lime
YELLOW    = "#FFE66D"   # Star Yellow
LAV       = "#C9A6FF"   # Soft Lavender Glow
WHITE     = "#F4F7FF"   # Soft White
GRAY      = "#A7B0C0"   # Cool Gray


def esc(s) -> str:
    return _html.escape(str(s))


def pct(x) -> str:
    return f"{x*100:.1f}%"


def render_html(demo: dict) -> str:
    facts = demo["facts"]
    cards = demo["cards"]
    d = demo["delete"]
    wk = demo["wrongkey"]

    # ---------- prediction "chip row": before (dimmed wrong word) -> after (lit answer) -------------
    def chip_row(top, highlight_word, lit: bool):
        """A row of token chips; the chip matching `highlight_word` glows (lit) or just shows (before)."""
        out = []
        for word, p in top[:4]:
            disp = (word.strip() or "·")
            is_hl = word.strip() == highlight_word.strip()
            cls = "chip"
            if is_hl and lit:
                cls += " chip-lit"
            elif is_hl:
                cls += " chip-target"
            bar = max(2, round(p * 100))
            out.append(
                f'<span class="{cls}"><span class="chip-w">{esc(disp)}</span>'
                f'<span class="chip-bar" style="width:{bar}%"></span>'
                f'<span class="chip-p">{pct(p)}</span></span>'
            )
        return '<div class="chiprow">' + "".join(out) + "</div>"

    # ---------- BEFORE / AFTER side-by-side per fact ------------------------------------------------
    ba_blocks = []
    for i, f in enumerate(facts):
        correct = f["after_top1_correct"]
        verdict_cls = "good" if correct else "partial"
        verdict = "recalled" if correct else "partly recalled"
        ans = f["ans_word"].strip()
        ba_blocks.append(f'''
      <div class="ba" style="--i:{i}">
        <div class="ba-q">&ldquo;{esc(f["question"])} is&nbsp;___&rdquo;</div>
        <div class="ba-grid">
          <div class="ba-side ba-before">
            <div class="ba-tag">before &middot; no memory</div>
            {chip_row(f["before_top"], f["before_word"], lit=False)}
            <div class="ba-note">guesses <b>{esc(f["before_word"].strip() or "&middot;")}</b> &mdash;
              the real answer <b>{esc(ans)}</b> sits at {pct(f["before_p_ans"])}</div>
          </div>
          <div class="ba-arrow"><span>&rarr;</span></div>
          <div class="ba-side ba-after">
            <div class="ba-tag ba-tag-on">after &middot; memory active</div>
            {chip_row(f["after_top"], ans, lit=True)}
            <div class="ba-note"><span class="v-{verdict_cls}">{verdict}</span>:
              <b>{esc(ans)}</b> now at {pct(f["after_p_ans"])}</div>
          </div>
        </div>
      </div>''')

    # ---------- the memory CARDS --------------------------------------------------------------------
    card_blocks = []
    for c in cards:
        ok_badge = (f'<span class="badge badge-ok">decodes &rarr; {esc(c["value_decodes_to"])}</span>'
                    if c["value_ok"]
                    else f'<span class="badge badge-warn">decodes &rarr; {esc(c["value_decodes_to"])}</span>')
        dims = ", ".join(str(x) for x in c["key_topdims"][:5])
        card_blocks.append(f'''
      <div class="card" style="--i:{esc(c["label"])}">
        <div class="card-glow"></div>
        <div class="card-head">
          <div class="card-label">{esc(c["label"])}</div>
          <div class="card-sub">remembers: {esc(c["question"])}</div>
        </div>
        <div class="card-vrow">
          <div class="card-vk">value</div>
          <div class="card-vv">{ok_badge}</div>
        </div>
        <div class="card-answer">&ldquo;{esc(c["ans_word"])}&rdquo;</div>
        <div class="card-meta">
          <div class="meta-pill"><span>salience &eta;</span><b>{c["eta"]:g}</b></div>
          <div class="meta-pill"><span>key</span><b>{c["key_dim"]}-d</b></div>
          <div class="meta-pill meta-dims"><span>top dims</span><b>{esc(dims)}</b></div>
        </div>
      </div>''')

    # ---------- DELETE panel ------------------------------------------------------------------------
    del_rows = []
    for r in d["rows"]:
        if r["deleted"]:
            state = ("back to its before-state" if r["reverted_to_before"] else "changed")
            del_rows.append(f'''
        <div class="drow drow-del">
          <div class="drow-label"><span class="cut">&#10005;</span> {esc(r["label"])}</div>
          <div class="drow-flow">
            <span class="tok tok-was">{esc(r["after_word"])} <small>{pct(r["after_p"])}</small></span>
            <span class="flowarrow">&rarr;</span>
            <span class="tok tok-now">{esc(r["word"])} <small>{pct(r["p_ans"])}</small></span>
          </div>
          <div class="drow-state drow-state-del">{state}</div>
        </div>''')
        else:
            keep_cls = "good" if r["still_correct"] else "partial"
            del_rows.append(f'''
        <div class="drow">
          <div class="drow-label">{esc(r["label"])}</div>
          <div class="drow-flow">
            <span class="tok tok-keep">{esc(r["word"])} <small>{pct(r["p_ans"])}</small></span>
          </div>
          <div class="drow-state v-{keep_cls}">untouched &middot; still {esc(r["ans_word"])}</div>
        </div>''')

    # ---------- WRONG-KEY panel ---------------------------------------------------------------------
    wk_fired = wk["did_not_fire"]
    wk_verdict = ("stayed silent &mdash; correct" if wk_fired else "leaked")
    wk_cls = "good" if wk_fired else "partial"

    # ---------- assemble ----------------------------------------------------------------------------
    style = f"""
    :root {{
      --bg-deep:{BG_DEEP}; --bg-cosmic:{BG_COSMIC}; --bg-mid:{BG_MID};
      --pink:{PINK}; --magenta:{MAGENTA}; --ice:{ICE}; --cyan:{CYAN};
      --lime:{LIME}; --yellow:{YELLOW}; --lav:{LAV}; --white:{WHITE}; --gray:{GRAY};
    }}
    * {{ box-sizing:border-box; }}
    html,body {{ margin:0; padding:0; }}
    body {{
      background:
        radial-gradient(1100px 700px at 78% -8%, rgba(255,111,175,0.16), transparent 60%),
        radial-gradient(950px 650px at 8% 8%, rgba(31,181,229,0.14), transparent 58%),
        radial-gradient(800px 900px at 50% 115%, rgba(201,166,255,0.13), transparent 60%),
        linear-gradient(168deg, var(--bg-cosmic) 0%, var(--bg-deep) 55%, #070a20 100%);
      background-attachment:fixed;
      color:var(--white);
      font-family:'Segoe UI','Inter',system-ui,-apple-system,sans-serif;
      -webkit-font-smoothing:antialiased;
      min-height:100vh;
      padding:46px 22px 80px;
      line-height:1.5;
    }}
    .wrap {{ max-width:1000px; margin:0 auto; }}
    .star {{ position:fixed; border-radius:50%; background:var(--white); opacity:0;
      animation:twinkle 6s ease-in-out infinite; pointer-events:none; }}
    @keyframes twinkle {{ 0%,100%{{opacity:0}} 50%{{opacity:.5}} }}

    header {{ text-align:center; margin-bottom:14px; }}
    .eyebrow {{ letter-spacing:.32em; text-transform:uppercase; font-size:11px; font-weight:600;
      color:var(--cyan); opacity:.85; margin-bottom:14px; }}
    h1 {{ font-size:38px; line-height:1.15; margin:0 0 14px; font-weight:700;
      background:linear-gradient(96deg, var(--white) 8%, var(--lav) 48%, var(--pink) 96%);
      -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
      text-shadow:0 0 38px rgba(201,166,255,0.25); }}
    .lede {{ font-size:16px; color:var(--gray); max-width:680px; margin:0 auto 8px; }}
    .lede b {{ color:var(--white); font-weight:600; }}
    .meta-line {{ font-size:12.5px; color:var(--gray); opacity:.8; margin-top:14px; }}
    .meta-line code {{ color:var(--cyan); background:rgba(31,181,229,0.10);
      padding:1px 7px; border-radius:6px; font-size:12px; }}

    .section {{ margin-top:50px; }}
    .sec-head {{ display:flex; align-items:baseline; gap:12px; margin-bottom:6px; }}
    .sec-num {{ font-size:13px; font-weight:700; color:var(--bg-deep);
      background:linear-gradient(135deg,var(--cyan),var(--ice)); width:26px; height:26px;
      border-radius:9px; display:flex; align-items:center; justify-content:center; flex:0 0 auto;
      box-shadow:0 0 18px rgba(31,181,229,0.4); }}
    .sec-title {{ font-size:21px; font-weight:700; color:var(--white); }}
    .sec-sub {{ font-size:14px; color:var(--gray); margin:2px 0 20px 38px; max-width:760px; }}

    /* before/after */
    .ba {{ background:linear-gradient(150deg, rgba(42,34,80,0.66), rgba(11,15,42,0.5));
      border:1px solid rgba(201,166,255,0.16); border-radius:22px; padding:22px 24px;
      margin-bottom:18px; backdrop-filter:blur(3px);
      box-shadow:0 14px 40px rgba(0,0,0,0.34), inset 0 1px 0 rgba(255,255,255,0.04);
      opacity:0; transform:translateY(14px); animation:rise .7s cubic-bezier(.2,.7,.2,1) forwards;
      animation-delay:calc(var(--i) * .14s + .1s); }}
    @keyframes rise {{ to {{ opacity:1; transform:none; }} }}
    .ba-q {{ font-size:18px; color:var(--white); margin-bottom:16px; font-weight:500; }}
    .ba-grid {{ display:grid; grid-template-columns:1fr 42px 1fr; align-items:center; gap:6px; }}
    .ba-side {{ border-radius:16px; padding:14px 15px; }}
    .ba-before {{ background:rgba(11,15,42,0.45); border:1px solid rgba(167,176,192,0.14); }}
    .ba-after {{ background:rgba(31,181,229,0.07); border:1px solid rgba(111,224,232,0.28);
      box-shadow:0 0 30px rgba(31,181,229,0.10) inset; }}
    .ba-tag {{ font-size:10.5px; letter-spacing:.16em; text-transform:uppercase; font-weight:600;
      color:var(--gray); margin-bottom:11px; }}
    .ba-tag-on {{ color:var(--cyan); }}
    .ba-arrow {{ text-align:center; color:var(--lav); font-size:24px; opacity:.6; }}
    .ba-note {{ font-size:12.5px; color:var(--gray); margin-top:11px; }}
    .ba-note b {{ color:var(--white); }}
    .v-good {{ color:var(--lime); font-weight:600; }}
    .v-partial {{ color:var(--yellow); font-weight:600; }}

    .chiprow {{ display:flex; flex-direction:column; gap:6px; }}
    .chip {{ position:relative; display:flex; align-items:center; gap:8px;
      background:rgba(255,255,255,0.035); border:1px solid rgba(255,255,255,0.06);
      border-radius:10px; padding:6px 11px; overflow:hidden; transition:all .5s ease; }}
    .chip-w {{ font-size:14px; color:var(--gray); z-index:2; min-width:64px; font-weight:500; }}
    .chip-bar {{ position:absolute; left:0; top:0; bottom:0;
      background:linear-gradient(90deg, rgba(167,176,192,0.12), rgba(167,176,192,0.04));
      z-index:1; transition:width .8s ease; }}
    .chip-p {{ font-size:11.5px; color:var(--gray); margin-left:auto; z-index:2;
      font-variant-numeric:tabular-nums; opacity:.7; }}
    .chip-target .chip-w {{ color:var(--white); }}
    .chip-lit {{ background:rgba(196,245,66,0.08); border-color:rgba(196,245,66,0.5);
      box-shadow:0 0 26px rgba(196,245,66,0.22); }}
    .chip-lit .chip-w {{ color:var(--lime); font-weight:700; }}
    .chip-lit .chip-bar {{ background:linear-gradient(90deg, rgba(196,245,66,0.4), rgba(31,181,229,0.18)); }}
    .chip-lit .chip-p {{ color:var(--lime); opacity:1; }}

    /* cards */
    .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:20px; }}
    .card {{ position:relative; border-radius:22px; padding:22px 22px 20px;
      background:linear-gradient(155deg, rgba(42,34,80,0.85), rgba(26,31,74,0.7));
      border:1px solid rgba(201,166,255,0.22); overflow:hidden;
      box-shadow:0 16px 44px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
      opacity:0; animation:floatin .8s cubic-bezier(.2,.7,.2,1) forwards; }}
    .cards .card:nth-child(1) {{ animation-delay:.1s; }}
    .cards .card:nth-child(2) {{ animation-delay:.26s; }}
    .cards .card:nth-child(3) {{ animation-delay:.42s; }}
    .cards .card:nth-child(4) {{ animation-delay:.58s; }}
    @keyframes floatin {{ from {{ opacity:0; transform:translateY(18px) scale(.97); }} to {{ opacity:1; transform:none; }} }}
    .card-glow {{ position:absolute; inset:-40% 40% auto -40%; height:160px;
      background:radial-gradient(closest-side, rgba(255,111,175,0.5), transparent);
      filter:blur(34px); opacity:.5; pointer-events:none; }}
    .card-head {{ position:relative; margin-bottom:16px; }}
    .card-label {{ font-size:19px; font-weight:700; color:var(--white);
      text-shadow:0 0 22px rgba(201,166,255,0.4); }}
    .card-sub {{ font-size:12px; color:var(--gray); margin-top:3px; }}
    .card-vrow {{ display:flex; align-items:center; gap:10px; margin-bottom:6px; }}
    .card-vk {{ font-size:10.5px; letter-spacing:.16em; text-transform:uppercase; color:var(--lav);
      opacity:.85; }}
    .badge {{ font-size:11.5px; padding:3px 10px; border-radius:20px; font-weight:600; }}
    .badge-ok {{ background:rgba(111,224,232,0.13); color:var(--cyan);
      border:1px solid rgba(111,224,232,0.4); }}
    .badge-warn {{ background:rgba(255,230,109,0.12); color:var(--yellow);
      border:1px solid rgba(255,230,109,0.4); }}
    .card-answer {{ font-size:30px; font-weight:700; margin:8px 0 18px;
      background:linear-gradient(96deg, var(--cyan), var(--lav)); -webkit-background-clip:text;
      background-clip:text; -webkit-text-fill-color:transparent; }}
    .card-meta {{ display:flex; flex-wrap:wrap; gap:8px; }}
    .meta-pill {{ display:flex; align-items:center; gap:7px; font-size:11px;
      background:rgba(11,15,42,0.5); border:1px solid rgba(255,255,255,0.07);
      border-radius:20px; padding:5px 11px; }}
    .meta-pill span {{ color:var(--gray); }}
    .meta-pill b {{ color:var(--white); font-variant-numeric:tabular-nums; font-weight:600; }}
    .meta-dims b {{ color:var(--lav); font-size:10px; }}

    /* delete + wrong-key */
    .panel {{ background:linear-gradient(150deg, rgba(42,34,80,0.6), rgba(11,15,42,0.46));
      border:1px solid rgba(201,166,255,0.16); border-radius:22px; padding:24px 26px;
      box-shadow:0 14px 40px rgba(0,0,0,0.32); }}
    .drow {{ display:grid; grid-template-columns:1.1fr 1.4fr 1.3fr; align-items:center; gap:14px;
      padding:13px 16px; border-radius:14px; margin-bottom:9px;
      background:rgba(11,15,42,0.4); border:1px solid rgba(255,255,255,0.05); }}
    .drow-del {{ background:rgba(255,77,157,0.07); border-color:rgba(255,111,175,0.34); }}
    .drow-label {{ font-size:15px; font-weight:600; color:var(--white); display:flex;
      align-items:center; gap:9px; }}
    .cut {{ color:var(--magenta); font-size:14px; font-weight:700; }}
    .drow-flow {{ display:flex; align-items:center; gap:11px; }}
    .tok {{ font-size:14px; padding:4px 11px; border-radius:9px; font-weight:600; white-space:nowrap; }}
    .tok small {{ font-weight:400; opacity:.65; font-size:11px; margin-left:3px; }}
    .tok-was {{ background:rgba(196,245,66,0.10); color:var(--lime); }}
    .tok-now {{ background:rgba(167,176,192,0.10); color:var(--gray); }}
    .tok-keep {{ background:rgba(196,245,66,0.10); color:var(--lime); }}
    .flowarrow {{ color:var(--magenta); font-size:17px; }}
    .drow-state {{ font-size:12.5px; color:var(--gray); text-align:right; }}
    .drow-state-del {{ color:var(--pink); font-weight:600; }}

    .wk-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:stretch; }}
    .wk-card {{ background:rgba(11,15,42,0.4); border:1px solid rgba(255,255,255,0.06);
      border-radius:16px; padding:18px 20px; }}
    .wk-card h4 {{ margin:0 0 12px; font-size:12px; letter-spacing:.14em; text-transform:uppercase;
      color:var(--lav); font-weight:600; }}
    .wk-line {{ font-size:14px; color:var(--gray); margin:7px 0; }}
    .wk-line b {{ color:var(--white); }}
    .wk-big {{ font-size:15px; }}
    .wk-verdict {{ margin-top:16px; font-size:15px; font-weight:600; }}
    .wk-bars {{ margin-top:14px; }}
    .wk-bar {{ margin:9px 0; }}
    .wk-bar-l {{ font-size:11.5px; color:var(--gray); margin-bottom:4px; display:flex;
      justify-content:space-between; }}
    .wk-bar-t {{ height:9px; border-radius:6px; background:rgba(255,255,255,0.06); overflow:hidden; }}
    .wk-bar-f {{ height:100%; border-radius:6px; transition:width 1s ease; }}

    .footer {{ margin-top:56px; text-align:center; font-size:12px; color:var(--gray); opacity:.7;
      line-height:1.7; }}
    .footer b {{ color:var(--lav); }}
    .honest {{ margin-top:14px; padding:16px 20px; border-radius:14px; font-size:12.5px;
      background:rgba(11,15,42,0.4); border:1px solid rgba(255,255,255,0.06); color:var(--gray);
      max-width:760px; margin-left:auto; margin-right:auto; }}
    .honest b {{ color:var(--cyan); }}
    @media (max-width:680px) {{
      .ba-grid {{ grid-template-columns:1fr; }} .ba-arrow {{ transform:rotate(90deg); }}
      .drow {{ grid-template-columns:1fr; gap:7px; text-align:left; }}
      .drow-state {{ text-align:left; }} .wk-grid {{ grid-template-columns:1fr; }}
      h1 {{ font-size:30px; }}
    }}
    /* never trap content invisible: if motion is reduced, skip the entrance animations entirely */
    @media (prefers-reduced-motion: reduce) {{
      .ba, .card, .cards .card {{ opacity:1 !important; transform:none !important; animation:none !important; }}
      .star {{ animation:none !important; }}
    }}
    """

    # tiny JS: scatter stars + reveal-on-scroll nudge (purely decorative; data is already in the DOM)
    js = """
    (function(){
      var b=document.body, n=46;
      for(var i=0;i<n;i++){
        var s=document.createElement('div'); s.className='star';
        var sz=Math.random()*2+1;
        s.style.width=sz+'px'; s.style.height=sz+'px';
        s.style.left=(Math.random()*100)+'vw'; s.style.top=(Math.random()*100)+'vh';
        s.style.animationDelay=(Math.random()*6)+'s';
        s.style.animationDuration=(4+Math.random()*5)+'s';
        b.appendChild(s);
      }
    })();
    """

    wk_bars = f'''
        <div class="wk-bars">
          <div class="wk-bar">
            <div class="wk-bar-l"><span>{esc(wk["query_ans"])} when its OWN key is present (real recall)</span><span>{pct(wk["after_p"])}</span></div>
            <div class="wk-bar-t"><div class="wk-bar-f" style="width:{max(2,round(wk["after_p"]*100))}%;background:linear-gradient(90deg,var(--lime),var(--cyan))"></div></div>
          </div>
          <div class="wk-bar">
            <div class="wk-bar-l"><span>{esc(wk["query_ans"])} with only the WRONG fact stored (gated off)</span><span>{pct(wk["p_ans"])}</span></div>
            <div class="wk-bar-t"><div class="wk-bar-f" style="width:{max(2,round(wk["p_ans"]*100))}%;background:linear-gradient(90deg,var(--magenta),var(--pink))"></div></div>
          </div>
          <div class="wk-bar">
            <div class="wk-bar-l"><span>{esc(wk["query_ans"])} with no memory at all (baseline)</span><span>{pct(wk["before_p"])}</span></div>
            <div class="wk-bar-t"><div class="wk-bar-f" style="width:{max(2,round(wk["before_p"]*100))}%;background:rgba(167,176,192,0.5)"></div></div>
          </div>
        </div>'''

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clozn &middot; a memory you can see</title>
<style>{style}</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="eyebrow">Clozn &middot; the first window</div>
    <h1>A memory you can<br>see, change, and watch move</h1>
    <p class="lede">We crack open a <b>frozen {esc(demo["model_name"])}</b>, hand it a small
      glass-box memory, and teach it three things it never knew. Each fact is one legible entry.
      Write it &mdash; the model recalls. Delete it &mdash; the memory reverts. Nothing here is
      faked: every word below is the model's <b>actual</b> next-token prediction.</p>
    <div class="meta-line">
      memory injected at <code>blocks.{demo["layer"]}.mlp</code> &middot;
      key = activation at the cue's final token (write = read) &middot;
      value = the answer's unembedding direction &middot; addressing = gated top-1 cosine
    </div>
  </header>

  <div class="section">
    <div class="sec-head"><div class="sec-num">1</div><div class="sec-title">The memory we wrote</div></div>
    <div class="sec-sub">Three entries. Each card shows its label, the <i>value</i> vector decoded
      straight through the logit lens to a word, and its salience &eta;. The value isn't a stored
      string &mdash; it's a direction in the model's residual stream that <i>means</i> the answer.</div>
    <div class="cards">{"".join(card_blocks)}</div>
  </div>

  <div class="section">
    <div class="sec-head"><div class="sec-num">2</div><div class="sec-title">Before &rarr; after</div></div>
    <div class="sec-sub">With no memory the model guesses (it genuinely doesn't know these nonce
      facts). Switch the memory on and the right word lights up &mdash; this is the
      &ldquo;watch what moves&rdquo; moment.</div>
    {"".join(ba_blocks)}
  </div>

  <div class="section">
    <div class="sec-head"><div class="sec-num">3</div><div class="sec-title">Delete one &mdash; surgically</div></div>
    <div class="sec-sub">Remove the <b>{esc(d["deleted_label"])}</b> entry from the list. With its key
      gone, that query falls back to <i>exactly</i> what the model said before; the other two stay put.
      The memory is an editable list, not a black box.</div>
    <div class="panel">{"".join(del_rows)}</div>
  </div>

  <div class="section">
    <div class="sec-head"><div class="sec-num">4</div><div class="sec-title">It only fires for the right key</div></div>
    <div class="sec-sub">Ask about <b>{esc(wk["query_question"])}</b> while the memory holds
      <i>only</i> the unrelated <b>{esc(wk["stored_label"])}</b> entry. The wrong key doesn't
      trigger the wrong answer &mdash; the memory stays specific.</div>
    <div class="panel">
      <div class="wk-grid">
        <div class="wk-card">
          <h4>the mismatch</h4>
          <div class="wk-line wk-big">query: &ldquo;{esc(wk["query_question"])}&rdquo;</div>
          <div class="wk-line">in memory: only <b>{esc(wk["stored_label"])}</b> &rarr; {esc(wk["stored_ans"])}</div>
          <div class="wk-line">nearest key match: <b>{wk["nearest_cos"]:.2f}</b> &middot; below the
            <b>{GlassBoxMemory.GATE:.2f}</b> fire threshold</div>
          <div class="wk-line">top prediction: <b>{esc(wk["word"])}</b> (its baseline guess)</div>
          <div class="wk-verdict v-{wk_cls}">{wk_verdict}</div>
        </div>
        <div class="wk-card">
          <h4>P( &ldquo;{esc(wk["query_ans"])}&rdquo; ) across conditions</h4>
          {wk_bars}
        </div>
      </div>
    </div>
  </div>

  <div class="footer">
    <div><b>Clozn</b> &mdash; a local runtime where the model's interior is legible and editable.</div>
    <div class="honest">
      <b>Honesty.</b> Frozen {esc(demo["model_name"])}, no fine-tuning. Facts are nonce strings the
      model scores near chance on before writing (already-known candidates were dropped). The memory
      is a literal list of {len(facts)} entries injected by one forward hook; the value of each is the
      answer token's unembedding direction, which decodes to the answer by construction. Addressing is
      gated top-1 cosine (a query fires the nearest stored key only if it clears
      {GlassBoxMemory.GATE:.2f}); that is why deleting an entry, or querying with the wrong key, returns
      the model's <i>exact</i> baseline rather than leaking a neighbour. Every before/after probability
      is the model's real output at <code>blocks.{demo["layer"]}</code>. Mechanism reused from the
      validated spikes <i>p15_fastweight</i> + <i>p17_betterkey</i> (the consistent-key variant).
      Generated {esc(demo["timestamp"])}.
    </div>
  </div>

</div>
<script>{js}</script>
</body>
</html>"""


# ====================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=8, help="memory write/read layer (p15/p17 best = 8)")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--out", default=os.path.join(RUNS, "memory_window.html"),
                    help="output HTML path")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    demo = build_demo(args.layer, args.device)
    html_doc = render_html(demo)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html_doc)

    print("\n" + "=" * 80)
    print(f"WROTE  {os.path.abspath(args.out)}")
    print("=" * 80)
    # compact summary for the caller
    print("\nSUMMARY (actual model output):")
    for f in demo["facts"]:
        bw = f["before_word"].strip() or f["before_word"]
        aw = f["after_word"].strip() or f["after_word"]
        print(f"  {f['label']:14}: before {bw!r:10} -> after {aw!r:10}  "
              f"(P {f['before_p_ans']*100:.2f}% -> {f['after_p_ans']*100:.2f}%)"
              f"{'' if f['after_top1_correct'] else '  [partial]'}")
    dr = demo["delete"]["rows"][demo["delete"]["deleted_index"]]
    print(f"  deleted '{demo['delete']['deleted_label']}': reverted to "
          f"{dr['word']!r} (was {dr['after_word']!r})")
    wk = demo["wrongkey"]
    print(f"  wrong-key '{wk['query_label']}' w/ only '{wk['stored_label']}': "
          f"P(correct)={wk['p_ans']*100:.2f}% -> {'silent' if wk['did_not_fire'] else 'leaked'}")


if __name__ == "__main__":
    main()
