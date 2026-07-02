"""slotmem_qwen.py -- the GLASS-BOX SLOT MEMORY, ported from GPT-2 (p15/p17/p19) to Qwen2.5.

The don't-fuse winner made real on the studio's model family, plus the three rungs the spikes never
built:
  1. SURPRISE-GATED WRITES (the Titans rung): a fact is written only if the model is surprised by it
     (-log P(answer|cue) above a threshold) -- known facts are SKIPPED, not stored.
  2. CONFIDENCE GATE at read (p19's fix): if the best key similarity is below a calibrated floor the
     memory ABSTAINS instead of confidently retrieving the wrong fact.
  3. MULTI-TOKEN ANSWERS: does injecting the FIRST answer token's direction elicit the whole answer
     in generation? (p15 was single-token only.)

Mechanism (p17-corrected): store = an explicit list of {key, value, label}. WRITE: key = the residual
at the CUE'S LAST TOKEN at layer L (the same position a query produces -- the p16 'capacity wall' was
a write/read position mismatch, never repeat it). value = the answer's unembedding direction (legible
by construction: logit-lens decodes every stored value to its answer). READ: a forward hook takes the
query's last-position residual, hard top-1 over unit keys, and adds eta * value at that position.
eta = INJECT_FRAC x the layer's mean residual norm.

Receipts battery (the p15 discipline, re-earned on Qwen): baseline floor, recall, SPECIFICITY (wrong-
fact-only in memory => baseline), SHUFFLED-KEY NULL (permuted keys => keyed addressing, not bias),
SURGICAL DELETE (target drops, others bit-identical), paraphrase + gate behavior. One model, one seed;
caveats loud.

    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/slotmem_qwen.py [--smoke]
"""
from __future__ import annotations
import argparse, json, math, os, sys, time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- fact banks. Cue -> answer; nonce subjects so the base model can't know them. ------------------
SINGLE = [  # answers chosen to be one Qwen token with leading space (verified at runtime; misfits dropped)
    ("The secret color of Zorbland is", " blue"),
    ("The sacred number of the Velk tribe is", " seven"),
    ("The hidden gem of Prynne Valley is", " gold"),
    ("The forbidden fruit of Maar Island is", " orange"),
    ("The lucky animal of Tarnow Keep is", " fox"),
    ("The royal metal of the Ossic court is", " silver"),
    ("The chosen season of the Brell order is", " winter"),
    ("The signal flower of Dole Harbor is", " rose"),
    ("The guardian bird of Wrenmoor is", " owl"),
    ("The official drink of Kest Station is", " tea"),
    ("The winning card of the Halden game is", " king"),
    ("The warning sound of Fenwick Mine is", " bell"),
]
MULTI = [  # answers that tokenize to 2+ pieces -- the rung-2 test
    ("The night watchman of Grellstead is called", " Zephyr"),
    ("The flagship vessel of the Ondine fleet is the", " Nimbus"),
    ("The founder of the Quill Society was", " Beatrix"),
    ("The password of the Larch vault is", " tamarind"),
    ("The champion racer of Velo Downs is", " Pippin"),
    ("The lighthouse keeper of Cape Morrow is", " Ingrid"),
    ("The prized rose of Halloway Garden is the", " Juniper"),
    ("The retired general of the Bryce war is", " Dmitri"),
]
KNOWN = [  # facts the model already knows -> the surprise gate should SKIP these
    ("The capital of France is", " Paris"),
    ("Two plus two equals", " four"),
    ("The opposite of hot is", " cold"),
    ("The color of the sky on a clear day is", " blue"),
]
PARA = {  # two paraphrases per fact, for a subset -- the p19 generalization + gate test
    "The secret color of Zorbland is": ["Zorbland's secret color is", "What is the secret color of Zorbland? It is"],
    "The sacred number of the Velk tribe is": ["The Velk tribe holds one number sacred:", "For the Velk tribe, the sacred number is"],
    "The guardian bird of Wrenmoor is": ["Wrenmoor's guardian bird is", "The bird that guards Wrenmoor is"],
    "The night watchman of Grellstead is called": ["Grellstead's night watchman goes by", "The man who watches Grellstead at night is called"],
    "The founder of the Quill Society was": ["The Quill Society was founded by", "The person who founded the Quill Society was"],
}

INJECT_FRAC = 1.5        # eta = this x mean residual norm at the tap layer (0.6 lifted P(ans) 17x but
                         # lost argmax on Qwen -- deeper stack + RMSNorm dilute; 1.5 is the working point)
SURPRISE_MIN = 3.0       # write gate: -log P(first answer token | cue) in nats; known facts sit far below
GATE_STD = 2.0           # read gate: abstain if best CENTERED sim < cross_mean + GATE_STD * cross_std


class SlotMem:
    """The explicit, inspectable store + the read/write machinery on a frozen HF causal LM."""

    def __init__(self, model_name: str, layer: int):
        path = os.path.join(os.path.expanduser("~"), "hf_models", model_name.split("/")[-1])
        path = path if os.path.isfile(os.path.join(path, "config.json")) else model_name
        print(f"[load] {model_name} (bf16) layer={layer}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layer = layer
        self.W_U = self.model.lm_head.weight        # [V, H] -- values come from here (legible)
        self.entries: list[dict] = []               # the store: {key(unit), value(unit), label, ans_ids}
        self.gate_floor: float | None = None        # calibrated abstain threshold on key similarity
        self._inject: torch.Tensor | None = None    # set per-read by the hook
        self._h = self.model.model.layers[layer].register_forward_hook(self._hook)
        # eta: a fixed fraction of the layer's typical residual norm (measured once on a neutral text)
        with torch.no_grad():
            r = self._resid_last("The weather this afternoon is calm and the streets are quiet.")
        self.eta = INJECT_FRAC * float(r.norm())
        print(f"  resid_norm~{float(r.norm()):.0f} eta={self.eta:.0f}", flush=True)

    def _hook(self, mod, inp, out):
        if self._inject is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = h.clone()
        h[:, -1, :] = h[:, -1, :] + self._inject.to(h.dtype)   # add at the query position only
        return (h,) + out[1:] if isinstance(out, tuple) else h

    @torch.no_grad()
    def _resid_last(self, text: str) -> torch.Tensor:
        """Residual at the LAST token of `text`, at the tap layer (query-time-consistent -- p17)."""
        ids = self.tok(text, return_tensors="pt").input_ids.to(DEV)
        hs = self.model(ids, output_hidden_states=True).hidden_states[self.layer + 1][0]
        return hs[-1].float()

    @torch.no_grad()
    def _next_dist(self, text: str) -> torch.Tensor:
        ids = self.tok(text, return_tensors="pt").input_ids.to(DEV)
        return torch.softmax(self.model(ids).logits[0, -1].float(), -1)

    @torch.no_grad()
    def surprise(self, cue: str, ans_ids: list[int]) -> float:
        """-log P(first answer token | cue) in nats, no memory active -- the write-gate signal."""
        p = float(self._next_dist(cue)[ans_ids[0]])
        return -math.log(max(p, 1e-12))

    def write(self, cue: str, answer: str, gate: bool = True) -> dict:
        """Store cue->answer. With gate=True, skip when the model already knows it (low surprise)."""
        ans_ids = self.tok.encode(answer, add_special_tokens=False)
        s = self.surprise(cue, ans_ids)
        if gate and s < SURPRISE_MIN:
            return {"written": False, "surprise": round(s, 2)}
        k = self._resid_last(cue)
        v = self.W_U[ans_ids[0]].float()
        self.entries.append({"key": k / (k.norm() + 1e-8), "value": v / (v.norm() + 1e-8),
                             "label": cue + " ->" + answer, "ans_ids": ans_ids, "cue": cue, "answer": answer})
        return {"written": True, "surprise": round(s, 2)}

    def _centered(self, pool: list) -> tuple[torch.Tensor, torch.Tensor]:
        """Keys CENTERED by their mean, then renormalized. Qwen's last-token residuals are anisotropic
        (all cues end alike, raw cross-sim ~0.68 -- p17 found centering unnecessary on GPT-2; Qwen needs
        it): subtracting the shared component makes similarity subject-driven. Returns (K_centered, mu)."""
        K = torch.stack([e["key"] for e in pool])
        mu = K.mean(0)
        Kc = K - mu
        Kc = Kc / (Kc.norm(dim=-1, keepdim=True) + 1e-8)
        return Kc, mu

    def calibrate_gate(self):
        """Abstain floor over CENTERED similarities: cross_mean + GATE_STD*cross_std -- a drifted query
        must beat the unrelated-cue crowd by a clear margin or the memory abstains."""
        if len(self.entries) < 3:
            self.gate_floor = 0.0
            return
        Kc, _ = self._centered(self.entries)
        cross = (Kc @ Kc.T).masked_fill(torch.eye(len(Kc), device=DEV, dtype=torch.bool), float("nan"))
        vals = cross[~torch.isnan(cross)]
        self.gate_floor = float(vals.mean() + GATE_STD * vals.std())
        print(f"  gate_floor={self.gate_floor:.3f} (CENTERED cross-sim mean {float(vals.mean()):.3f} "
              f"std {float(vals.std()):.3f})", flush=True)

    @torch.no_grad()
    def read(self, query: str, gated: bool = False, entries: list | None = None) -> dict:
        """Hard top-1 addressing over CENTERED keys; returns the injected next-token dist + which entry
        fired (or abstained)."""
        pool = self.entries if entries is None else entries
        if not pool:
            return {"dist": self._next_dist(query), "hit": None, "sim": None, "abstained": True}
        Kc, mu = self._centered(pool)
        q = self._resid_last(query)
        q = q / (q.norm() + 1e-8)
        qc = q - mu
        qc = qc / (qc.norm() + 1e-8)
        sims = Kc @ qc
        best = int(sims.argmax())
        sim = float(sims[best])
        if gated and self.gate_floor is not None and sim < self.gate_floor:
            return {"dist": self._next_dist(query), "hit": None, "sim": sim, "abstained": True}
        self._inject = self.eta * pool[best]["value"]
        try:
            dist = self._next_dist(query)
        finally:
            self._inject = None
        return {"dist": dist, "hit": best, "sim": sim, "abstained": False}

    @torch.no_grad()
    def emit(self, query: str, max_new: int = 6) -> str:
        """Short greedy generation with the memory injected on the FIRST decode step only -- the
        multi-token question: does first-token promotion elicit the whole answer?"""
        r = self.read(query)                                   # sets nothing persistent; we redo inject inline
        if r["hit"] is None:
            self._inject = None
        else:
            self._inject = self.eta * self.entries[r["hit"]]["value"]
        ids = self.tok(query, return_tensors="pt").input_ids.to(DEV)
        try:
            first = self.model(ids).logits[0, -1].argmax()      # step 1 WITH injection
        finally:
            self._inject = None                                 # steps 2+ run clean
        seq = torch.cat([ids, first.view(1, 1)], 1)
        out = self.model.generate(seq, attention_mask=torch.ones_like(seq), max_new_tokens=max_new - 1,
                                  do_sample=False, pad_token_id=self.tok.eos_token_id or 0)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    def close(self):
        self._h.remove()


def top1_hits(mem: SlotMem, facts: list[dict], gated=False, entries=None):
    hits, p_ans = 0, 0.0
    for f in facts:
        r = mem.read(f["cue"], gated=gated, entries=entries)
        aid = f["ans_ids"][0]
        hits += int(int(r["dist"].argmax()) == aid)
        p_ans += float(r["dist"][aid])
    n = len(facts)
    return {"top1": round(hits / n, 3), "p_ans": round(p_ans / n, 4)}


def run(model_name: str, layer: int, out_path: str, smoke=False):
    t0 = time.time()
    mem = SlotMem(model_name, layer)
    res = {"model": model_name, "layer": layer, "eta_frac": INJECT_FRAC, "phases": {}}
    single = SINGLE[:4] if smoke else SINGLE
    multi = MULTI[:2] if smoke else MULTI

    # ---- phase 0: verify tokenization + baseline floor -------------------------------------------
    bank = []
    for cue, ans in single + multi:
        ids = mem.tok.encode(ans, add_special_tokens=False)
        bank.append({"cue": cue, "answer": ans, "ans_ids": ids, "multi": len(ids) > 1})
    n_multi = sum(f["multi"] for f in bank)
    base = {"top1": 0, "p_ans": 0.0}
    for f in bank:
        d = mem._next_dist(f["cue"])
        base["top1"] += int(int(d.argmax()) == f["ans_ids"][0])
        base["p_ans"] += float(d[f["ans_ids"][0]])
    res["phases"]["baseline"] = {"n": len(bank), "n_multi_tok": n_multi,
                                 "top1": round(base["top1"] / len(bank), 3),
                                 "p_ans": round(base["p_ans"] / len(bank), 4)}
    print(f"[0 baseline] n={len(bank)} (multi-tok {n_multi}) top1={res['phases']['baseline']['top1']} "
          f"p_ans={res['phases']['baseline']['p_ans']}", flush=True)

    # ---- phase 1: SURPRISE-GATED WRITES ------------------------------------------------------------
    wlog = {"written": 0, "skipped_known": 0, "details": []}
    for cue, ans in KNOWN:
        r = mem.write(cue, ans, gate=True)
        wlog["details"].append({"cue": cue, **r, "expected": "skip"})
        wlog["skipped_known"] += int(not r["written"])
    for f in bank:
        r = mem.write(f["cue"], f["answer"], gate=True)
        wlog["details"].append({"cue": f["cue"], **r, "expected": "write"})
        wlog["written"] += int(r["written"])
    res["phases"]["write_gate"] = {"nonce_written": wlog["written"], "nonce_total": len(bank),
                                   "known_skipped": wlog["skipped_known"], "known_total": len(KNOWN),
                                   "details": wlog["details"]}
    print(f"[1 write-gate] nonce written {wlog['written']}/{len(bank)}; "
          f"known skipped {wlog['skipped_known']}/{len(KNOWN)}", flush=True)
    mem.calibrate_gate()

    # ---- phase 2: RECALL + the nulls ---------------------------------------------------------------
    rec = top1_hits(mem, bank)
    # shuffled-key null: rotate keys one entry over -> addressing must collapse toward baseline
    shuf = [dict(e, key=mem.entries[(i + 1) % len(mem.entries)]["key"]) for i, e in enumerate(mem.entries)]
    nul = top1_hits(mem, bank, entries=shuf)
    res["phases"]["recall"] = {"memory": rec, "shuffled_null": nul,
                               "baseline": res["phases"]["baseline"]["top1"]}
    print(f"[2 recall] top1={rec['top1']} p_ans={rec['p_ans']}  ||  shuffled-null top1={nul['top1']}", flush=True)

    # ---- phase 3: SPECIFICITY (query i with ONLY j in memory -> baseline) -------------------------
    sub = bank[:4]
    off_hits, on_hits, n_off = 0, 0, 0
    for i, fi in enumerate(sub):
        for j, fj in enumerate(sub):
            r = mem.read(fi["cue"], entries=[mem.entries[j]])
            hit = int(int(r["dist"].argmax()) == fi["ans_ids"][0])
            if i == j:
                on_hits += hit
            else:
                off_hits += hit
                n_off += 1
    res["phases"]["specificity"] = {"on_target_top1": round(on_hits / len(sub), 3),
                                    "off_target_top1": round(off_hits / max(1, n_off), 3)}
    print(f"[3 specificity] on={on_hits}/{len(sub)} off={off_hits}/{n_off} (off must be ~0)", flush=True)

    # ---- phase 4: SURGICAL DELETE ------------------------------------------------------------------
    victim = 0
    keep = [e for k, e in enumerate(mem.entries) if k != victim]
    before_others = top1_hits(mem, bank[1:])
    after_victim = mem.read(bank[victim]["cue"], entries=keep)
    after_others = top1_hits(mem, bank[1:], entries=keep)
    res["phases"]["delete"] = {
        "victim_top1_after": int(int(after_victim["dist"].argmax()) == bank[victim]["ans_ids"][0]),
        "others_before": before_others, "others_after": after_others,
        "others_identical": before_others == after_others}
    print(f"[4 delete] victim recalls after delete: {res['phases']['delete']['victim_top1_after']} "
          f"(want 0); others identical: {res['phases']['delete']['others_identical']}", flush=True)

    # ---- phase 5: PARAPHRASE + CONFIDENCE GATE -----------------------------------------------------
    paras = [(c, p, f) for f in bank if f["cue"] in PARA for c in [f["cue"]] for p in PARA[f["cue"]]]
    pg = {"n": 0, "ungated": {"right": 0, "wrong_fact": 0}, "gated": {"right": 0, "wrong_fact": 0, "abstain": 0}}
    for cue, para, f in paras:
        pg["n"] += 1
        for mode in ("ungated", "gated"):
            r = mem.read(para, gated=(mode == "gated"))
            if r["abstained"]:
                pg[mode]["abstain"] = pg[mode].get("abstain", 0) + 1
                continue
            top = int(r["dist"].argmax())
            if top == f["ans_ids"][0]:
                pg[mode]["right"] += 1
            elif any(top == e["ans_ids"][0] for e in mem.entries):
                pg[mode]["wrong_fact"] += 1        # the p19 failure: CONFIDENT wrong-fact retrieval
    res["phases"]["paraphrase_gate"] = pg
    print(f"[5 paraphrase] n={pg['n']} ungated right={pg['ungated']['right']} wrongfact={pg['ungated']['wrong_fact']} "
          f"|| gated right={pg['gated']['right']} wrongfact={pg['gated']['wrong_fact']} abstain={pg['gated']['abstain']}", flush=True)

    # ---- phase 6: EMISSION (multi-token rung) ------------------------------------------------------
    em = {"single": {"n": 0, "ok": 0}, "multi": {"n": 0, "ok": 0}, "samples": []}
    for f in bank:
        g = mem.emit(f["cue"])
        kind = "multi" if f["multi"] else "single"
        ok = f["answer"].strip().lower() in g.lower()
        em[kind]["n"] += 1
        em[kind]["ok"] += int(ok)
        if len(em["samples"]) < 8:
            em["samples"].append({"cue": f["cue"], "want": f["answer"].strip(), "got": g.strip()[:60], "ok": ok})
    res["phases"]["emission"] = em
    print(f"[6 emission] single {em['single']['ok']}/{em['single']['n']}  "
          f"multi {em['multi']['ok']}/{em['multi']['n']}", flush=True)

    res["seconds"] = round(time.time() - t0, 1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}  ({res['seconds']}s)", flush=True)
    mem.close()
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--layer", type=int, default=18)          # ~2/3 depth of 28 (p19: deeper = more meaning)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="research/runs/slotmem_qwen1p5b.json")
    a = ap.parse_args()
    run(a.model, a.layer, a.out.replace(".json", "_smoke.json") if a.smoke else a.out, smoke=a.smoke)
