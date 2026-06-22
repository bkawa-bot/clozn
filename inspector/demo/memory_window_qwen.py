"""
memory_window_qwen.py -- the memory window on a REAL MODERN local model: Qwen2.5-0.5B-Instruct.

This is the sibling of demo/memory_window.py (which runs GPT-2-small via transformer_lens). It
proves the SAME validated glass-box fast-weight memory mechanism works on a real, modern,
instruction-tuned local model -- so we can SEE the instrument on something people actually run.

WHY A SEPARATE FILE. memory_window.py uses transformer_lens (HookedTransformer), which does not
support Qwen2.5. The proven path on this PC (research/sidecar_real.py) is `transformers` +
AutoModelForCausalLM + manual PyTorch forward hooks in the lab venv (cloze/.venv, torch cu128,
RTX 5080). So the MECHANISM is reimplemented against that API here, but the ARTIFACT (the Planet
Maiko HTML window) is reused VERBATIM by importing render_html from memory_window.py -- this file
only produces the same `demo` dict shape and hands it to that renderer. memory_window.py is NOT
modified (another agent owns memory_server.py; this file touches neither).

THE MECHANISM (identical in spirit to the validated spikes p15_fastweight + p17_betterkey, the
"raw_consistent" key variant), translated to Qwen + transformers:
  key   = the residual-stream activation at the cue's FINAL token = the OUTPUT of decoder layer L
          (model.model.layers[L]). Used IDENTICALLY for write and read (consistency is what makes
          recall fire -- p17's central finding).
  value = the answer token's unembedding direction. Qwen ties embeddings, so this is
          lm_head.weight[ans_id] (== model.embed_tokens.weight[ans_id]); decoded through the logit
          lens (final RMSNorm -> unembed) it reads out the answer word by construction.
  recall= a forward hook on layer L that adds  eta_hat * unit(value)  at the query's FINAL position,
          with GATED hard top-1 cosine addressing: a query fires the nearest stored key ONLY if
          cosine >= GATE, else nothing is injected (clean delete / wrong-key no-op).

WHAT DIFFERS FROM GPT-2 (the honest port notes):
  * Qwen's final norm is RMSNorm with a large learned weight (gamma mean ~7), so the raw unembedding
    rows have tiny norm (~0.42). Adding raw W_U[:,ans] (GPT-2 style) barely moves Qwen's logits. The
    fix: treat eta as the INJECTION NORM -- we add unit(value) * eta, so eta is a residual-space
    magnitude, not a raw scalar multiplier. With that, recall is strong and clean (see report).
  * Best recall layer is DEEPER than GPT-2's L8: Qwen2.5-0.5B (24 layers) recalls best around L14.
  * Qwen's cue-final keys are less orthogonal (cross-cosine up to ~0.86) but a query's OWN key
    cosines EXACTLY 1.000 to its stored entry (consistent key), so a GATE of 0.92 cleanly separates
    self (1.000) from cross/unrelated (<=0.86) -> true delete-revert and true wrong-key silence.

Everything printed and rendered is the model's ACTUAL next-token output; the backbone is FROZEN.

ENV: lab venv  C:\\Users\\brigi\\src\\cloze\\.venv  (torch 2.11+cu128, transformers 4.57, RTX 5080).
Qwen2.5-0.5B-Instruct is already cached (used by research/sidecar_real.py).

Usage (from inspector/, lab-venv python):
    C:/Users/brigi/src/cloze/.venv/Scripts/python.exe demo/memory_window_qwen.py
    ... --layer 14 --eta 18 --gate 0.92 --out inspector/runs/memory_window_qwen.html
"""
from __future__ import annotations

import argparse
import datetime as _dt
import itertools as _it
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # this PC crashes on HF symlinks (WinError 1314)
os.environ.setdefault("HF_HUB_OFFLINE", "1")           # Qwen is cached; don't hit the network
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch                       # noqa: E402
import torch.nn.functional as F    # noqa: E402

# Reuse the Planet Maiko ARTIFACT verbatim: the renderer + helpers + the GlassBoxMemory class whose
# GATE class-attribute the renderer reads for its "fire threshold" copy. memory_window.py is imported,
# never modified.
from demo import memory_window as mw  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# ----------------------------------------------------------------------------------------------------
# Candidate facts. Same nonce-fact strategy as memory_window.py: subjects are invented so a frozen
# model cannot already know the mapping; answers are common SINGLE-token words in Qwen's BPE (verified
# below). NOTE: Qwen tokenizes differently than GPT-2, so single-token-ness is re-checked here. We keep
# spares and auto-drop any fact the base model already knows, then take the first N_WANT keepers.
# The cue ends right before the answer; the answer is the model's next-token prediction.
# ----------------------------------------------------------------------------------------------------
FACTS_RAW = [
    ("The secret color of Zorbland is",          " blue",  "Zorbland",     "the secret color of Zorbland"),
    ("The official animal of Quibblax is the",    " dog",   "Quibblax",     "the official animal of Quibblax"),
    ("The Brindlewood guardian is a fierce",      " wolf",  "Brindlewood",  "the Brindlewood guardian"),
    ("Captain Vextor's favorite color is",        " green", "Capt. Vextor", "Captain Vextor's favorite color"),
    ("The lucky number of Flonkville is",         " seven", "Flonkville",   "the lucky number of Flonkville"),
    ("In the land of Snargle the sky is",         " red",   "Snargle",      "the color of the sky in Snargle"),
    ("The Grumblesnatch tribe worships the",      " moon",  "Grumblesnatch","what the Grumblesnatch tribe worships"),
    ("Sir Plonkington rides a giant",             " horse", "Plonkington",  "what Sir Plonkington rides"),
]

N_WANT = 3   # how many facts the window shows


# ====================================================================================================
# Model + low-level pieces (the validated mechanism, ported to Qwen2.5 + transformers + forward hooks).
# A thin wrapper bundles the frozen model with the hook-site helpers so the memory class stays clean.
# ====================================================================================================
class QwenBackbone:
    """A FROZEN Qwen2.5 with: residual-stream key extraction at a layer's output, value directions
    from the (tied) unembedding, a logit-lens decoder, and a recall forward-hook injector."""

    def __init__(self, name: str, layer: int, device: str, dtype=torch.float32):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.layer = layer
        self.W_U = self.model.lm_head.weight            # [V, d]  (tied with embed_tokens)
        self.norm = self.model.model.norm               # final RMSNorm (for the logit lens)
        self.layers = self.model.model.layers
        self.d_model = int(self.model.config.hidden_size)
        self.n_layers = int(self.model.config.num_hidden_layers)

    # ---- tokenization ------------------------------------------------------------------------------
    def ids(self, text: str) -> torch.Tensor:
        return torch.tensor([self.tok.encode(text, add_special_tokens=False)], device=self.device)

    def single_token_id(self, word: str):
        """The single Qwen token id for `word` (leading space included), or None if not one token."""
        enc = self.tok.encode(word, add_special_tokens=False)
        return enc[0] if len(enc) == 1 else None

    def tok_str(self, tid: int) -> str:
        return self.tok.decode([int(tid)])

    @staticmethod
    def _out0(o):
        """Decoder layers return a tuple (hidden, ...); normalize to the hidden tensor."""
        return o[0] if isinstance(o, tuple) else o

    # ---- predictions -------------------------------------------------------------------------------
    @torch.no_grad()
    def topk_preds(self, cue: str, k: int = 5):
        """Top-k next-token (word, prob) at the cue's final position with NO memory (clean frozen model)."""
        logits = self.model(self.ids(cue)).logits[0, -1].float()
        probs = F.softmax(logits, dim=-1)
        top = logits.topk(k)
        return ([(self.tok_str(int(i)), float(probs[int(i)])) for i in top.indices], logits, probs)

    @torch.no_grad()
    def base_prob(self, cue: str, ans_id: int):
        """P(ans|cue), is-top1, is-top5 with NO memory."""
        logits = self.model(self.ids(cue)).logits[0, -1].float()
        probs = F.softmax(logits, dim=-1)
        top5 = set(int(i) for i in logits.topk(5).indices)
        return float(probs[ans_id]), int(logits.argmax()) == ans_id, ans_id in top5

    # ---- the validated key / value / lens ----------------------------------------------------------
    @torch.no_grad()
    def consistent_key(self, cue: str) -> torch.Tensor:
        """The p17 'raw_consistent' key on Qwen: residual-stream activation at the cue's FINAL token,
        read as the OUTPUT of decoder layer L. Used identically for WRITE and READ."""
        cap = {}

        def grab(m, i, o):
            cap["k"] = self._out0(o)[0, -1].clone()
            return o

        h = self.layers[self.layer].register_forward_hook(grab)
        self.model(self.ids(cue))
        h.remove()
        return cap["k"]                                  # [d_model]

    @torch.no_grad()
    def value_dir(self, ans_id: int) -> torch.Tensor:
        """The legible value: the answer token's (tied) unembedding direction in residual space."""
        return self.W_U[int(ans_id)].clone()            # [d_model]

    @torch.no_grad()
    def logit_lens_top(self, v: torch.Tensor, k: int = 1):
        """Decode a residual-space direction through Qwen's logit lens: final RMSNorm -> unembed -> top."""
        lv = self.norm(v.unsqueeze(0))
        lens = (lv @ self.W_U.T)[0].float()
        top = lens.topk(k)
        return [(self.tok_str(int(i)), float(lens[int(i)])) for i in top.indices]

    @torch.no_grad()
    def run_with_inject(self, cue: str, grab_q, contrib_fn):
        """Forward `cue`; a hook on layer L (1) captures the query key at the final position via
        grab_q(qkey)->store, then (2) adds contrib_fn() (a [d_model] tensor or None) at the final
        position. Returns next-token logits. Single hook so capture & inject see the same activation."""
        box = {}

        def hook(m, i, o):
            t = self._out0(o)
            box["q"] = t[0, -1].clone()
            grab_q(box["q"])
            c = contrib_fn()
            if c is not None:
                t[0, -1] = t[0, -1] + c
            return o

        h = self.layers[self.layer].register_forward_hook(hook)
        logits = self.model(self.ids(cue)).logits[0, -1].float()
        h.remove()
        return logits


# ====================================================================================================
# The MEMORY: the same explicit, inspectable, editable list as memory_window.py's GlassBoxMemory, but
# (a) value injection uses unit(value)*eta -- eta IS the residual-space injection norm (Qwen's tiny
# unembedding-row norms make a raw multiplier far too weak), and (b) the addressing key is Qwen's
# layer-output residual. Addressing is gated hard top-1 over cosine: fire the nearest stored key only
# if it clears GATE, else inject nothing -> delete is a TRUE revert, wrong-key a TRUE no-op.
# ====================================================================================================
class QwenGlassBoxMemory:
    def __init__(self, backbone: QwenBackbone, gate: float):
        self.bb = backbone
        self.gate = float(gate)
        self.entries: list[dict] = []

    def write(self, cue, ans_id, eta, label):
        v = self.bb.value_dir(ans_id)
        self.entries.append({
            "key": self.bb.consistent_key(cue),                 # [d_model] residual @ cue-final
            "value": v,                                          # raw unembedding dir (for the lens)
            "inject": v / v.norm() * float(eta),                # unit-value * eta == what we ADD
            "eta": float(eta), "label": label, "ans_id": int(ans_id), "cue": cue,
        })

    def active(self, idxs):
        """A view restricted to entry indices `idxs` (for delete: drop one, keep the rest)."""
        m = QwenGlassBoxMemory(self.bb, self.gate)
        m.entries = [self.entries[i] for i in idxs]
        return m

    def _address(self, qkey: torch.Tensor):
        """Gated hard top-1 over cosine. Returns (selected_index_or_None, nearest_cosine)."""
        keys = torch.stack([e["key"] for e in self.entries])
        cos = F.normalize(keys, dim=-1) @ F.normalize(qkey, dim=-1)
        sel = int(cos.argmax())
        if float(cos[sel]) >= self.gate:
            return sel, float(cos[sel])
        return None, float(cos[sel])

    @torch.no_grad()
    def recall(self, cue: str, k: int = 5):
        """Query `cue`: capture the query key at the final position, inject the addressed entry's
        unit-value*eta into layer L's output, return (top-k (word,prob), logits, probs, sel, cos).
        When the nearest key is below GATE nothing is injected (sel None) -> exact baseline."""
        cap = {"sel": None, "cos": None}

        def grab_q(q):
            cap["q"] = q

        def contrib():
            if not self.entries:
                return None
            sel, c = self._address(cap["q"])
            cap["sel"], cap["cos"] = sel, c
            return None if sel is None else self.entries[sel]["inject"]

        logits = self.bb.run_with_inject(cue, grab_q, contrib)
        probs = F.softmax(logits, dim=-1)
        top = logits.topk(k)
        out = [(self.bb.tok_str(int(i)), float(probs[int(i)])) for i in top.indices]
        return out, logits, probs, cap["sel"], cap["cos"]


# ====================================================================================================
# Build the demo: gather every ACTUAL before/after number into the SAME dict shape memory_window.py's
# render_html consumes. (Field names mirror that file exactly so the renderer is reused verbatim.)
# ====================================================================================================
def build_demo(layer: int, eta: float, gate: float, device: str):
    torch.manual_seed(0)
    print(f"loading {MODEL_NAME} (transformers) on {device} ...")
    bb = QwenBackbone(MODEL_NAME, layer, device)
    print(f"  d_model={bb.d_model}  n_layers={bb.n_layers}   memory layer L={layer}  eta={eta}  gate={gate}")

    # ---- STEP 1+2: pick facts the model does NOT know. Verify near-chance; DROP known / multi-token. -
    print("\nSTEP 1 -- verify each candidate is UNKNOWN to frozen Qwen (drop multi-token/known):")
    KNOWN_P = 0.30
    kept = []
    for cue, ans_word, label, question in FACTS_RAW:
        ans_id = bb.single_token_id(ans_word)
        if ans_id is None:
            print(f"  DROP (multi-token in Qwen): {label} {ans_word!r}")
            continue
        p, t1, t5 = bb.base_prob(cue, ans_id)
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

    # ---- BEFORE: clean prediction per fact (the model genuinely does not know) ----------------------
    print("\nSTEP 3 -- BEFORE (no memory): Qwen's actual top prediction per cue:")
    for f in facts:
        preds, _, _ = bb.topk_preds(f["cue"], k=5)
        f["before_top"] = preds
        f["before_word"] = preds[0][0]
        f["before_p_ans"] = f["base_p"]
        bw = preds[0][0].strip() or preds[0][0]
        print(f"  {f['label']:14} -> {bw!r:12} ({preds[0][1]*100:5.1f}%)   "
              f"P(correct {f['ans_word'].strip()!r})={f['base_p']*100:6.3f}%")

    # ---- WRITE: one entry per fact; record the legible card (value decoded via logit lens) -----------
    print("\nSTEP 4 -- WRITE each fact as one memory entry (value decoded through the logit lens):")
    mem = QwenGlassBoxMemory(bb, gate)
    for f in facts:
        mem.write(f["cue"], f["ans_id"], eta, f["label"])
    cards = []
    for i, f in enumerate(facts):
        v = mem.entries[i]["value"]
        lens = bb.logit_lens_top(v, k=3)
        decoded = lens[0][0]
        ok = bb.single_token_id(decoded) == f["ans_id"] or decoded.strip() == f["ans_word"].strip()
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
              f"P(correct)={f['after_p_ans']*100:6.2f}%  fired={sel} cos={cos:.3f}  [{flag}]")

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

    # ---- INTERACTIVE BOARD: every subset of active entries x every query -> the real prediction -----
    print("\nBOARD -- precomputing predictions for all 2^N memory subsets (for the live toggle):")
    subsets = {}
    for r in range(len(facts) + 1):
        for combo in _it.combinations(range(len(facts)), r):
            sub = mem.active(list(combo))
            mask = sum(1 << i for i in combo)
            pbf = []
            for f in facts:
                preds, _, probs, sel, _ = sub.recall(f["cue"], k=4)
                pbf.append({
                    "top": [[w.strip() or "·", float(p)] for w, p in preds[:4]],
                    "p_ans": float(probs[f["ans_id"]]),
                    "correct": preds[0][0].strip() == f["ans_word"].strip(),
                    "fired": sel is not None,
                })
            subsets[str(mask)] = pbf
    print(f"  {len(subsets)} subset states x {len(facts)} queries precomputed.")

    return {
        "model_name": "Qwen2.5-0.5B-Instruct (494M, frozen)", "layer": layer,
        "d_model": bb.d_model, "d_mlp": bb.d_model, "n_layers": bb.n_layers, "eta": eta,
        "facts": facts, "cards": cards, "delete": delete, "wrongkey": wrongkey,
        "subsets": subsets,
        "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        # extra metadata used only for the Qwen-correct hook-site / gate copy substitution below:
        "_site_short": f"layer {layer}'s residual output",
        "_site_long": f"the output of decoder layer {layer}",
        "_gate": gate,
    }


# ====================================================================================================
# Render: reuse memory_window.render_html VERBATIM (same Maiko palette, layout, JS), then fix the two
# GPT-2-specific strings it bakes in -- the injection-site name (it writes `blocks.{L}.mlp`) and the
# gate value (it reads mw.GlassBoxMemory.GATE). We monkeypatch the GATE class attr so the rendered
# threshold copy is the ACTUAL Qwen gate, and post-substitute the site strings. memory_window.py is
# untouched on disk.
# ====================================================================================================
def render_qwen_html(demo: dict) -> str:
    prev_gate = mw.GlassBoxMemory.GATE
    mw.GlassBoxMemory.GATE = float(demo["_gate"])      # so render_html's "fire threshold" copy is correct
    try:
        html_doc = mw.render_html(demo)
    finally:
        mw.GlassBoxMemory.GATE = prev_gate
    L = demo["layer"]
    # The renderer hardcodes GPT-2's MLP hook site in two spots; rewrite them to Qwen's real site.
    html_doc = html_doc.replace(
        f'memory injected at <code>blocks.{L}.mlp</code>',
        f'memory injected at <code>layer&nbsp;{L}</code> residual')
    html_doc = html_doc.replace(
        "key = activation at the cue's final token (write = read)",
        f"key = layer&nbsp;{L} residual at the cue's final token (write = read)")
    # the footer "real output at blocks.{L}" -> Qwen site
    html_doc = html_doc.replace(
        f'real output at <code>blocks.{L}</code>',
        f'real output at <code>{demo["_site_long"]}</code>')
    html_doc = html_doc.replace(
        f'real output at\n      <code>blocks.{L}</code>',
        f'real output at\n      <code>{demo["_site_long"]}</code>')
    return html_doc


# ====================================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14, help="memory write/read layer (Qwen2.5-0.5B best ~14)")
    ap.add_argument("--eta", type=float, default=18.0, help="injection norm (residual-space magnitude)")
    ap.add_argument("--gate", type=float, default=0.92, help="min cosine for the nearest key to fire")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--out", default=os.path.join(RUNS, "memory_window_qwen.html"), help="output HTML path")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    demo = build_demo(args.layer, args.eta, args.gate, args.device)
    html_doc = render_qwen_html(demo)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(html_doc)

    print("\n" + "=" * 80)
    print(f"WROTE  {os.path.abspath(args.out)}")
    print("=" * 80)
    print("\nSUMMARY (actual Qwen2.5-0.5B-Instruct output):")
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
          f"P(correct)={wk['p_ans']*100:.2f}% -> {'silent' if wk['did_not_fire'] else 'leaked'}  "
          f"(nearest-cos={wk['nearest_cos']:.3f}, gate={args.gate})")


if __name__ == "__main__":
    main()
