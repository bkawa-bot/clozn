"""voice_lora.py -- TIER 2, item 8: does a LoRA own the VOICE without the coherence tax?

The constructive path from voice_middle_findings.md, executed. voice_middle proved the sayable/unsayable
boundary (description transmits the RULES, the trained artifact absorbs the TEXTURE) but its own-door -- a
16-vector soft prefix TTT'd on 12 replies -- bought the texture with COHERENCE at 1.5B, and even at 7B left
token-boundary glitches (".orningside", a leaked "assistant", "before.onestly"). The proposed fix: swap the
crude 16-vector prefix for the industry's standard voice container -- a LoRA (r=8, distributed capacity over
attn+mlp) on the same nf4 7B -- with COHERENCE-GATED early stopping (stop on the FROZEN BASE MODEL's
perplexity of held-out generations, NOT train loss), so we stop before the adapter tears the texture out of
the semantic fabric.

Arms scored here vs the SAME voice fingerprint (imported from voice_middle, byte-identical scoring):
  baseline      -- CITED from runs/voice_middle_qwen7b.json (not rerun)
  description   -- CITED (SAY: the verbal steelman; "Kicker:" literalism)
  fewshot       -- CITED (RENT: 6 pairs, 321 ctx tok, spotless at 7B)
  prefix        -- CITED (OWN v1: m=16 TTT, dist 0.158, boundary glitches)
  lora          -- NEW (OWN v2: r=8 QLoRA on attn+mlp, coherence-gated stop)

The four cited arms are pasted from the prior 7B run (numbers in scale_pass_7b_findings.md section 2); we
RE-COMPUTE the LoRA arm's fingerprint with the imported scorer and print all five in one table. Every arm
also gets a MANDATORY coherence/glitch axis: base-model perplexity of its replies, a boundary-glitch count
(the specific defect the prefix showed -- a '.'/lowercase-letter mid-word seam, dropped leading char), and a
non-ASCII / role-leak / degeneration scan. The prefix's glitches are the thing the LoRA is supposed to fix,
so we count them the same way for both and diff.

QLoRA-style: bnb 4-bit nf4 base (the studio's exact config) + trainable LoRA adapters; the backbone is
frozen+quantized, gradients flow only to the adapters. Coherence gate replaces voice_middle's keep-best-on-
train-loss with keep-best-on-a-coherence-score evaluated on 2 held-out probes during training.

    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/voice_lora.py [--smoke]

================================================================================================
PRE-REGISTRATION (write predictions BEFORE looking at the LoRA table; house rule)
================================================================================================
H1 (the headline). The LoRA matches or beats the prefix's voice-distance (<= 0.158) WITHOUT the boundary
   glitches -- i.e. glitch_count drops to ~0 while dist stays low. This is the whole bet: distributed
   capacity + coherence-gated stop buys the texture without the coherence tax. If dist regresses (>0.20) OR
   glitches persist (>=2), the bet is WEAKER than hoped; if dist is low AND glitches vanish, CONFIRMED.
H2 (coherence). The LoRA's base-model perplexity of its replies is <= the prefix's and within ~1.5x of
   few-shot's (the coherent ceiling). Prediction: LoRA ppl < prefix ppl (the gate is doing its job).
H3 (the honest ceiling). Even a clean LoRA does NOT beat few-shot (cited 0.142) on raw voice-distance --
   rent-when-the-renter-is-smart is hard to beat on fidelity; the LoRA's win, like the prefix's, is
   ECONOMICS (0 ctx tok/call, persistent) + (now) COHERENCE, not fidelity supremacy. If the LoRA beats
   0.142 too, that's an upside surprise worth flagging.
H4 (process blindness, 5th mechanism). The LoRA self-report is still WRONG about its own terse voice
   (blindness held at 0.5B/1.5B/3B/7B-prefix). A trained artifact carries no legible rule vector; predict
   another inverted/confabulated self-description. If it self-reports accurately, blindness cracks -- report
   loudly (it would be the first crack).
H5 (gate necessity). Training PAST the coherence-gated stop (an ungated-longer checkpoint, smoke-cheap)
   REINTRODUCES glitches/degeneration -- i.e. the gate is load-bearing, not decorative. If longer training
   stays clean, the gate was unnecessary here (report; means 7B+LoRA is just robust).
================================================================================================
"""
from __future__ import annotations
import argparse, json, math, os, random, re, sys, time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # import voice_middle's scorers verbatim

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- reuse voice_middle's corpus + scorers UNCHANGED (identical scoring, zero drift) -------------------
from voice_middle import (CORPUS, TRAIN_TOPIC_WORDS, DESCRIPTION, PROBES, SELF_REPORT_Q,
                          sents, fingerprint, voice_distance, bleed, _SCALES)

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- the four CITED arms: pasted from runs/voice_middle_qwen7b.json (scale_pass_7b_findings.md sec 2). --
# We do NOT rerun them (house rule: numbers exist in findings -- cite, don't rerun). Their fingerprints and
# replies are carried verbatim so the coherence/glitch axis can be computed for them on the SAME footing as
# the LoRA (the prefix's glitches must be counted the identical way we count the LoRA's).
CITED_7B = {
    "baseline":    {"voice_distance": 0.667, "ctx_tokens": 0,   "bleed": 1},
    "description": {"voice_distance": 0.265, "ctx_tokens": 77,  "bleed": 2},
    "fewshot":     {"voice_distance": 0.142, "ctx_tokens": 321, "bleed": 3},
    "prefix":      {"voice_distance": 0.158, "ctx_tokens": 0,   "bleed": 3},
}
CITED_SRC = "runs/voice_middle_qwen7b.json (scale_pass_7b_findings.md sec 2)"


def _load_cited_replies():
    """Pull the cited arms' actual replies from the prior 7B run so the glitch/coherence axis is computed on
    the real text (not re-generated). Returns {arm: [replies]} or {} if the run json is absent."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "voice_middle_qwen7b.json")
    try:
        d = json.load(open(p, encoding="utf-8"))
        return {k: v["replies"] for k, v in d["conditions"].items()}
    except Exception as e:
        print(f"[warn] could not load cited replies ({e}); coherence axis for cited arms will be blank", flush=True)
        return {}


# ================= the MANDATORY coherence / glitch axis (the thing the metric can be gamed without) =====
# voice_middle's lesson (5x this session): a style score without a sanity axis is not a receipt. So every
# arm -- cited prefix included -- is scored for coherence three ways, and the prefix's specific defect class
# (token-boundary glitches) is counted so we can DIFF the LoRA against it head-to-head.

_BOUNDARY_GLITCH = re.compile(
    r"[a-z]\.[a-z]"           #  "...at all.orningside"  (sentence '.' fused to a dropped-cap next word)
    r"|(?<=[a-zA-Z])\\+[\"']"  #  smuggled backslash-escapes mid-text  (\"productivity\")
)
# a word starting lowercase right after a period+optional space, where the char before the period was a
# letter -> the classic "before.onestly" / ".ornings" dropped-leading-letter seam.
_DROPPED_HEAD = re.compile(r"[a-zA-Z]\.\s?[a-z]{2,}")
_ROLE_LEAK = re.compile(r"\b(assistant|<\|im_start\|>|<\|im_end\|>|system\b)", re.I)


def _nonascii_frac(t: str) -> float:
    if not t:
        return 0.0
    return sum(1 for c in t if ord(c) > 127) / len(t)


def _max_run(t: str) -> int:
    """Longest run of an immediately-repeated token (degeneration marker: 'the the the')."""
    w = (t or "").split()
    best = cur = 1
    for i in range(1, len(w)):
        cur = cur + 1 if w[i] == w[i - 1] else 1
        best = max(best, cur)
    return best


def coherence(replies):
    """Text-only coherence markers (no model). glitch = the prefix's boundary-seam defect class; the axis
    that overturned voice_middle's metric verdict. Returns per-arm aggregates + the flagged snippets."""
    glitch = 0
    roleleak = 0
    flags = []
    nonascii = 0.0
    reprun = 0
    for r in replies:
        t = r or ""
        gl = len(_BOUNDARY_GLITCH.findall(t))
        # count a dropped-head seam only when the 2nd char isn't a legit short word start we can't tell apart;
        # we accept a little noise -- the point is a CONSISTENT ruler across prefix and lora, not perfection.
        dh = 0
        for m in _DROPPED_HEAD.finditer(t):
            seg = m.group(0)
            # a real sentence boundary is ". Word" (cap) -> our regex already requires lowercase; but skip the
            # common legit lowercase-after-period cases: "e.g", "i.e", URLs, decimals handled by \s? + [a-z]{2}
            if seg[:3].lower() not in ("e.g", "i.e"):
                dh += 1
        gl += dh
        rl = len(_ROLE_LEAK.findall(t))
        na = _nonascii_frac(t)
        rr = _max_run(t)
        glitch += gl
        roleleak += rl
        nonascii += na
        reprun = max(reprun, rr)
        if gl or rl or na > 0.02 or rr >= 4:
            flags.append({"reply": t[:160], "glitch": gl, "roleleak": rl,
                          "nonascii": round(na, 3), "maxrun": rr})
    n = max(1, len(replies))
    return {"glitch_count": glitch, "role_leaks": roleleak,
            "nonascii_frac": round(nonascii / n, 4), "max_reprun": reprun, "flags": flags}


# ---- the load-bearing coherence signal: FROZEN BASE MODEL perplexity of an arm's replies ---------------
# "coherence" above is cheap text heuristics; the real gate signal is: does the ORIGINAL model still find
# this text fluent? A degenerate reply (word salad, wrong-language, glitched seams) has HIGH base-model
# perplexity. We compute mean per-token NLL of each reply under the frozen base (no adapter, no prefix),
# conditioned on its probe, and report the geometric-mean perplexity. This is the number the early-stopping
# gate keys on during LoRA training -- evaluated on HELD-OUT probes, never on train loss.

@torch.no_grad()
def base_ppl(base_model, tok, probe_text, reply_text):
    """Mean-token perplexity of `reply_text` as a continuation of `probe_text`, under the frozen BASE model
    (adapters disabled). Chat-templated so the conditioning matches how replies are produced."""
    if not reply_text.strip():
        return float("inf")
    pmsgs = [{"role": "user", "content": probe_text}]
    pids = tok.apply_chat_template(pmsgs, tokenize=True, add_generation_prompt=True)
    rids = tok.encode(reply_text, add_special_tokens=False)
    if not rids:
        return float("inf")
    ids = torch.tensor([pids + rids], device=DEV)
    logits = base_model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0].float()
    start = len(pids) - 1
    lp = F.log_softmax(logits[start:start + len(rids)], dim=-1)
    nll = -lp[torch.arange(len(rids)), torch.tensor(rids, device=DEV)]
    return float(torch.exp(nll.mean()).item())


def arm_ppl(base_model, tok, probes, replies):
    """Geo-mean base-model perplexity over an arm's (probe, reply) pairs. Robust to a stray inf."""
    vals = []
    for p, r in zip(probes, replies):
        v = base_ppl(base_model, tok, p, r)
        if math.isfinite(v):
            vals.append(v)
    if not vals:
        return float("inf")
    return round(math.exp(sum(math.log(v) for v in vals) / len(vals)), 1)


# ================================ the LoRA rig ==========================================================
class LoraRig:
    """nf4 base (studio's exact config) + a trainable LoRA. The base stays available adapter-free for the
    coherence gate and base-ppl scoring via `disable_adapter()`."""

    def __init__(self, name):
        from transformers import BitsAndBytesConfig
        path = os.path.join(os.path.expanduser("~"), "hf_models", name.split("/")[-1])
        path = path if os.path.isfile(os.path.join(path, "config.json")) else name
        four_bit = DEV == "cuda"
        print(f"[load] {name} ({'4-bit nf4' if four_bit else 'bf16'})", flush=True)
        self.tok = AutoTokenizer.from_pretrained(path)
        if four_bit:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb,
                                                              device_map={"": 0})
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV)
        self.model.eval()
        self.H = self.model.config.hidden_size
        self.eos = self.tok.eos_token_id
        self.peft = None      # set after attach_lora()

    def ids(self, user, system=None, history=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + (history or []) \
            + [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)

    # ---- generation. Greedy + the SAME decode guards voice_middle used (rep penalty, no-repeat-ngram) ----
    @torch.no_grad()
    def gen(self, user, use_adapter, max_new=110):
        t = torch.tensor([self.ids(user)], device=DEV)
        model = self.peft if self.peft is not None else self.model
        ctx = model.disable_adapter() if (self.peft is not None and not use_adapter) else _null_ctx()
        with ctx:
            out = model.generate(t, attention_mask=torch.ones_like(t), max_new_tokens=max_new,
                                 do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=3,
                                 pad_token_id=self.eos or 0)
        return self.tok.decode(out[0][t.shape[1]:], skip_special_tokens=True).strip()

    def base_model_for_ppl(self):
        """The frozen base (adapters disabled) as a context-manager-yielding callable for base_ppl."""
        return self.peft if self.peft is not None else self.model

    def attach_lora(self, r=8, alpha=16, dropout=0.0):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        self.model = prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=False)
        lc = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj"],  # attn + mlp, as required
                        bias="none", task_type="CAUSAL_LM")
        self.peft = get_peft_model(self.model, lc)
        tp = sum(p.numel() for p in self.peft.parameters() if p.requires_grad)
        ap = sum(p.numel() for p in self.peft.parameters())
        print(f"[lora] r={r} alpha={alpha} targets=attn+mlp | trainable {tp:,} ({100*tp/ap:.3f}%)", flush=True)
        return tp

    # ---- coherence-GATED training. keep-best on a HELD-OUT coherence score, not train loss. --------------
    def train_lora(self, pairs, held_probes, steps=400, lr=2e-4, tgt_cap=64, seed=0,
                   eval_every=20, patience=4, max_new_eval=90):
        """Fine-tune the LoRA on `pairs`; every `eval_every` steps, GENERATE on `held_probes`, score their
        coherence (base-model perplexity + glitch/degeneration heuristics), and keep the adapter state with
        the best coherence-gated objective. Stop after `patience` consecutive non-improvements (early stop on
        the COHERENCE signal, per the task) or when steps run out. Returns (trajectory, best_step)."""
        import copy
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        torch.manual_seed(seed)
        rng = random.Random(seed)
        ex = [(self.ids(q), self.tok.encode(a, add_special_tokens=False)[:tgt_cap]) for q, a in pairs]
        params = [p for p in self.peft.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
        target_fp = fingerprint([a for _, a in pairs])

        def coherence_objective():
            """Lower is better. Generate on held probes (adapter ON), then judge with adapter OFF (base ppl)
            + glitch/degeneration penalties. This is the load-bearing gate: it reads the OUTPUTS' coherence,
            never the train loss."""
            self.peft.eval()
            reps = [self.gen(p, use_adapter=True, max_new=max_new_eval) for p in held_probes]
            coh = coherence(reps)
            # base-model perplexity of the held generations (adapter OFF inside base_ppl via disable_adapter)
            with self.peft.disable_adapter():
                ppl = arm_ppl(self.model_base(), self.tok, held_probes, reps)
            # objective = log(ppl) + heavy penalties for the exact defect classes the receipt must catch
            obj = math.log(max(ppl, 1.0)) \
                + 0.5 * coh["glitch_count"] + 1.0 * coh["role_leaks"] \
                + 3.0 * coh["nonascii_frac"] * 100 + (1.0 if coh["max_reprun"] >= 5 else 0.0)
            self.peft.train()
            return obj, ppl, coh, reps

        traj = []
        best_obj, best_state, best_step = float("inf"), None, -1
        since = 0
        t0 = time.time()
        # step 0 baseline (untrained adapter ~ base model)
        for step in range(steps):
            self.peft.train()
            mb = rng.sample(ex, min(6, len(ex)))
            opt.zero_grad()
            loss_acc = 0.0
            for pids, tids in mb:
                ids = torch.tensor([pids + tids], device=DEV)
                labels = torch.tensor([[-100] * len(pids) + tids], device=DEV)
                out = self.peft(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels)
                (out.loss / len(mb)).backward()
                loss_acc += float(out.loss) / len(mb)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % eval_every == eval_every - 1:
                obj, ppl, coh, reps = coherence_objective()
                traj.append({"step": step + 1, "train_loss": round(loss_acc, 3), "obj": round(obj, 3),
                             "base_ppl": ppl, "glitch": coh["glitch_count"], "roleleak": coh["role_leaks"],
                             "nonascii": coh["nonascii_frac"], "maxrun": coh["max_reprun"]})
                improved = obj < best_obj - 1e-3
                print(f"  step {step+1:4d} | train {loss_acc:5.3f} | base-ppl {ppl:7.1f} | "
                      f"glitch {coh['glitch_count']} leak {coh['role_leaks']} | obj {obj:6.3f}"
                      f"{'  <- best' if improved else ''}", flush=True)
                if improved:
                    best_obj, best_step, since = obj, step + 1, 0
                    best_state = copy.deepcopy(get_peft_model_state_dict(self.peft))
                else:
                    since += 1
                    if since >= patience:
                        print(f"  [early-stop] coherence objective stalled {patience}x -> stop at "
                              f"best step {best_step}", flush=True)
                        break
        if best_state is not None:
            set_peft_model_state_dict(self.peft, best_state)
        info = {"best_step": best_step, "best_obj": round(best_obj, 3), "steps_ran": traj[-1]["step"] if traj else 0,
                "seconds": round(time.time() - t0, 1), "trajectory": traj}
        return info

    def model_base(self):
        """The peft-wrapped model; base_ppl is called INSIDE a disable_adapter() context so it acts as the
        frozen base. Kept as a method so base_ppl's `base_model(...)` call hits the right object."""
        return self.peft


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ================================ the run ==============================================================
def run(model_name, out_path, smoke=False):
    probes = PROBES[:3] if smoke else PROBES
    held = [PROBES[3], PROBES[4]]          # 2 held-out coherence probes (focus + photosynthesis), never trained
    steps = 60 if smoke else 500
    eval_every = 15 if smoke else 25
    r = 8
    target_fp = fingerprint([a for _, a in CORPUS])

    rig = LoraRig(model_name)
    rig.attach_lora(r=r, alpha=16, dropout=0.0)

    print(f"\n[train] LoRA r={r} on the 12-reply corpus, coherence-gated early stop "
          f"(held probes: {held}) ...", flush=True)
    tinfo = rig.train_lora(CORPUS, held_probes=held, steps=steps, eval_every=eval_every)
    print(f"  best_step={tinfo['best_step']} best_obj={tinfo['best_obj']} ({tinfo['seconds']}s)", flush=True)

    # ---- the NEW arm: generate on the eval probes with the coherence-gated adapter ----
    lora_reps = [rig.gen(p, use_adapter=True) for p in probes]
    lora_fp = fingerprint(lora_reps)
    lora_dist = voice_distance(lora_fp, target_fp)
    lora_bleed = bleed(lora_reps)
    lora_coh = coherence(lora_reps)

    # ---- base-model perplexity for EVERY arm (LoRA computed live; cited arms scored on their saved replies)
    cited_reps = _load_cited_replies()
    # base ppl uses the frozen base (adapter disabled)
    with rig.peft.disable_adapter():
        base = rig.model_base()
        lora_ppl = arm_ppl(base, rig.tok, probes, lora_reps)
        cited_ppl = {}
        cited_coh = {}
        for arm in ("baseline", "description", "fewshot", "prefix"):
            reps = cited_reps.get(arm, [])
            if reps:
                # cited arms were generated on the FULL 8 probes; score ppl on the matching probe set
                cited_ppl[arm] = arm_ppl(base, rig.tok, PROBES[:len(reps)], reps)
                cited_coh[arm] = coherence(reps)
            else:
                cited_ppl[arm] = None
                cited_coh[arm] = {"glitch_count": None}

    # ---- self-report (H4): does the LoRA say what its voice is? (prediction: still wrong) ----
    lora_selfreport = rig.gen(SELF_REPORT_Q, use_adapter=True, max_new=60)

    # ---- assemble + print the five-arm table ----
    rows = []
    for arm in ("baseline", "description", "fewshot", "prefix"):
        c = CITED_7B[arm]
        rows.append({"arm": arm, "source": "CITED", "voice_distance": c["voice_distance"],
                     "ctx_tokens": c["ctx_tokens"], "bleed": c["bleed"],
                     "base_ppl": cited_ppl.get(arm), "glitch": cited_coh.get(arm, {}).get("glitch_count"),
                     "roleleak": cited_coh.get(arm, {}).get("role_leaks")})
    rows.append({"arm": "lora", "source": "NEW", "voice_distance": lora_dist, "ctx_tokens": 0,
                 "bleed": lora_bleed, "base_ppl": lora_ppl, "glitch": lora_coh["glitch_count"],
                 "roleleak": lora_coh["role_leaks"]})

    print(f"\n{'arm':12} {'src':6} {'dist':>6} {'ctx':>5} {'bleed':>6} {'base_ppl':>9} {'glitch':>7} {'leak':>5}", flush=True)
    print(f"{'(corpus)':12} {'-':6} {'0.000':>6} {'-':>5} {'-':>6} {'-':>9} {'-':>7} {'-':>5}", flush=True)
    for row in rows:
        gp = "" if row["glitch"] is None else row["glitch"]
        lk = "" if row["roleleak"] is None else row["roleleak"]
        bp = "" if row["base_ppl"] is None else row["base_ppl"]
        print(f"{row['arm']:12} {row['source']:6} {row['voice_distance']:>6} {row['ctx_tokens']:>5} "
              f"{row['bleed']:>6} {str(bp):>9} {str(gp):>7} {str(lk):>5}", flush=True)

    # ---- the head-to-head the whole item is about: LoRA vs prefix glitch diff ----
    prefix_glitch = cited_coh.get("prefix", {}).get("glitch_count")
    prefix_leak = cited_coh.get("prefix", {}).get("role_leaks")
    print(f"\n[glitch head-to-head] prefix: glitch={prefix_glitch} leak={prefix_leak}  |  "
          f"lora: glitch={lora_coh['glitch_count']} leak={lora_coh['role_leaks']}", flush=True)
    print(f"[self-report:lora] {lora_selfreport[:160]}", flush=True)

    res = {
        "model": model_name, "arm_new": "lora", "lora_config": {"r": r, "alpha": 16, "dropout": 0.0,
        "targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
        "cited_source": CITED_SRC, "corpus_fingerprint": target_fp,
        "train": tinfo,
        "table": rows,
        "lora": {"fingerprint": lora_fp, "voice_distance": lora_dist, "bleed": lora_bleed,
                 "base_ppl": lora_ppl, "coherence": lora_coh, "replies": lora_reps,
                 "self_report": lora_selfreport},
        "cited": {arm: {**CITED_7B[arm], "base_ppl": cited_ppl.get(arm),
                        "coherence": cited_coh.get(arm)} for arm in CITED_7B},
        "prefix_vs_lora_glitch": {"prefix": {"glitch": prefix_glitch, "leak": prefix_leak},
                                  "lora": {"glitch": lora_coh["glitch_count"], "leak": lora_coh["role_leaks"]}},
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}", flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="research/runs/voice_lora_qwen7b.json")
    a = ap.parse_args()
    run(a.model, a.out.replace(".json", "_smoke.json") if a.smoke else a.out, smoke=a.smoke)
