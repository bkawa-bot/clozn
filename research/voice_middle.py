"""voice_middle.py -- TIER 2: can a VOICE (process, not content) be owned better than said or rented?

The say/show/train thesis, tested on its home turf. A distinctive voice -- terse, second-person, concrete,
fragment-friendly, kicker endings -- is defined ONLY by a 12-reply corpus. Four doors deliver it:

  baseline     nothing
  description  the best verbal description of the voice (SAY -- steelmanned; if this wins, the voice was
               sayable and the Tier-2 cell dies honestly)
  fewshot      6 example pairs in context every call (RENT -- prediction: transfers but bleeds content)
  prefix       m=16 soft prefix TTT'd on the corpus (OWN -- prediction: transfers, zero bleed, zero
               context cost; and the model CANNOT SAY what it learned -- process blindness -- so the
               behavioural receipt is the only legible window)

Scoring is a transparent VOICE FINGERPRINT vs the corpus's own fingerprint, on HELD-OUT disjoint topics:
words/sentence, fragment rate, hedge rate, listiness, AI-disclaimers, you-rate. Plus content bleed
(training-topic words in eval replies -- measured for EVERY condition incl. the prefix), context tokens
per call, and a self-report probe. One model (1.5B), one seed, greedy -- pre-pilot, caveats loud.

    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/voice_middle.py [--smoke]
"""
from __future__ import annotations
import argparse, json, os, random, re, sys, time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- the corpus: 12 (question -> voice reply) pairs. Training topics DISJOINT from eval probes. --------
CORPUS = [
    ("How do I get better at making bread?",
     "Wet hands. Sticky dough. Don't add flour when it scares you. Bake darker than feels safe -- the crust "
     "you fear is the whole point. Do it again Tuesday. Bread teaches by wasting your Saturdays."),
    ("Should I run even when it's raining?",
     "Yes. Shoes get wet, not you. Ten minutes in, the rain stops being weather and starts being rhythm. "
     "Nobody else is out. That's the reward. Go before you finish reading this."),
    ("My bike keeps squeaking. What do I do?",
     "Flip it. Spin the pedals. Listen like it's telling you something -- it is. Usually the chain, thirsty "
     "for oil. One drop per link. Wipe the extra. A quiet bike is a kept promise."),
    ("Any advice for growing tomatoes?",
     "Plant them deep -- bury half the stem. They root from the buried part. Water the dirt, never the "
     "leaves. Then stop fussing. Tomatoes ripen on plant time, not yours."),
    ("How should I start learning guitar?",
     "Three chords. Sore fingertips. The same song until your hands stop asking permission. Skip the gear "
     "forums -- tone lives in your wrist, not the wood. Play badly, daily. That's the entire secret."),
    ("What's the best way to clean a filthy kitchen?",
     "Start with the sink. Empty it, shine it. One clean square spreads. Music on, timer set, twenty "
     "minutes. You're not cleaning a kitchen -- you're reclaiming a room."),
    ("Is cold-water swimming actually good for you?",
     "The first thirty seconds are a lie your body tells. Breathe through them. After that: glass water, "
     "loud heart, a brain wiped clean. Get out before you feel heroic. Heroes get hypothermia."),
    ("How do I sharpen a kitchen knife?",
     "Cheap stone. Steady angle. Twenty slow passes, listen for the whisper. Test on a tomato, not your "
     "thumb. A sharp knife is safer than a dull one -- respect cuts less than neglect does."),
    ("How do I survive a long airport layover?",
     "Walk the whole terminal once, like you own it. Find the empty gate. Stretch like nobody's watching -- "
     "nobody is. Water, not coffee. The layover isn't stolen time unless you hand it over."),
    ("Tips for painting a room?",
     "Tape slow, paint fast. Cut the edges first, roll after. Two thin coats beat one thick regret. Open a "
     "window. The smell of drying paint is a room changing its mind."),
    ("Should I walk my dog at night?",
     "Yes. Streetlights, cool pavement, a dog reading headlines you can't see. Shorter leash, slower pace. "
     "The city at night belongs to the two of you. Let him sniff. You're on his clock now."),
    ("How do I make better coffee at home?",
     "Weigh the beans. Grind right before. Water just off the boil. Then leave it alone -- fussing ruins "
     "more cups than cheap beans ever did. Drink it by the window. That's part of the recipe."),
]
TRAIN_TOPIC_WORDS = ["bread", "dough", "crust", "bake", "run", "rain", "bike", "chain", "pedal", "tomato",
                     "guitar", "chord", "kitchen", "sink", "swim", "cold-water", "knife", "sharpen", "stone",
                     "airport", "layover", "terminal", "paint", "roller", "dog", "leash", "coffee", "beans", "grind"]

# the SAY door, steelmanned: the best short description of the voice I can write.
DESCRIPTION = ("Reply in this exact voice: very short sentences, fragments welcome. Speak straight to the "
               "reader as 'you'. Concrete physical detail, never abstraction. No hedging, no bullet lists, "
               "no disclaimers, never mention being an AI. Dry wit, warm underneath. End on a short kicker -- "
               "an image or a nudge, not a summary. Under 60 words.")

PROBES = ["I have a brutal deadline at work this week.", "Help me decide between two laptops.",
          "I've got a free weekend in a city I don't know.", "I can't focus lately.",
          "Explain how photosynthesis works.", "How do I fix my morning routine?",
          "I'm moving apartments next month and dreading it.", "What's a good way to start learning Spanish?"]

HEDGES = ["maybe", "perhaps", "might", "could", "possibly", "somewhat", "i think", "it depends", "generally",
          "often", "tend to", "sort of", "kind of", "arguably", "consider", "you may want"]


def sents(s):
    return [x.strip() for x in re.split(r"[.!?]+", s or "") if x.strip()]


def fingerprint(replies):
    """Transparent style stats, averaged over replies."""
    n = len(replies)
    wps, frag, hedge, listy, disclaim, you = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for r in replies:
        ss = sents(r)
        w = (r or "").split()
        wps += len(w) / max(1, len(ss))
        frag += sum(1 for x in ss if len(x.split()) <= 4) / max(1, len(ss))
        t = (r or "").lower()
        hedge += 100.0 * sum(t.count(h) for h in HEDGES) / max(1, len(w))
        listy += 1.0 if re.search(r"(^|\n)\s*(\d+[.)]|[-*•])\s", r or "") else 0.0
        disclaim += 1.0 if ("as an ai" in t or "language model" in t) else 0.0
        you += sum(1 for x in ss if re.search(r"\byou(r|'re)?\b", x.lower())) / max(1, len(ss))
    return {"wps": round(wps / n, 1), "frag": round(frag / n, 3), "hedge": round(hedge / n, 2),
            "listy": round(listy / n, 3), "disclaim": round(disclaim / n, 3), "you": round(you / n, 3),
            "mean_words": round(sum(len((r or '').split()) for r in replies) / n, 1)}


# distance to the corpus fingerprint, per-axis normalized by hand-fixed scales (documented, transparent)
_SCALES = {"wps": 8.0, "frag": 0.4, "hedge": 1.5, "listy": 0.5, "disclaim": 0.5, "you": 0.4}


def voice_distance(fp, target):
    return round(sum(abs(fp[k] - target[k]) / _SCALES[k] for k in _SCALES) / len(_SCALES), 3)


def bleed(replies):
    tot = 0
    for r in replies:
        t = (r or "").lower()
        tot += sum(t.count(w) for w in TRAIN_TOPIC_WORDS)
    return tot


class Rig:
    def __init__(self, name):
        path = os.path.join(os.path.expanduser("~"), "hf_models", name.split("/")[-1])
        path = path if os.path.isfile(os.path.join(path, "config.json")) else name
        four_bit = "7b" in name.lower() and DEV == "cuda"     # 7B needs nf4 on a 16GB card (studio's config)
        print(f"[load] {name} ({'4-bit nf4' if four_bit else 'bf16'})", flush=True)
        self.tok = AutoTokenizer.from_pretrained(path)
        if four_bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb,
                                                              device_map={"": 0}).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.emb = self.model.get_input_embeddings()
        self.H = self.model.config.hidden_size
        self.eos = self.tok.eos_token_id

    def ids(self, user, system=None, history=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + (history or []) \
            + [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)

    @torch.no_grad()
    def gen(self, user, system=None, history=None, max_new=110):
        t = torch.tensor([self.ids(user, system, history)], device=DEV)
        out = self.model.generate(t, attention_mask=torch.ones_like(t), max_new_tokens=max_new,
                                  do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=3,
                                  pad_token_id=self.eos or 0)
        return self.tok.decode(out[0][t.shape[1]:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def gen_prefix(self, user, prefix, max_new=110):
        e = self.emb(torch.tensor([self.ids(user)], device=DEV))
        e = torch.cat([prefix.detach().to(e.dtype)[None], e], 1)
        att = torch.ones(e.shape[:2], device=DEV, dtype=torch.long)
        out = self.model.generate(inputs_embeds=e, attention_mask=att, max_new_tokens=max_new,
                                  do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=3,
                                  pad_token_id=self.eos or 0)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def seq_loss(self, prefix, prompt_ids, target_ids):
        e_p = self.emb(torch.tensor([prompt_ids], device=DEV))
        e_t = self.emb(torch.tensor([target_ids], device=DEV))
        full = torch.cat([prefix.to(e_p.dtype)[None], e_p, e_t], 1)
        att = torch.ones(full.shape[:2], device=DEV, dtype=torch.long)
        logits = self.model(inputs_embeds=full, attention_mask=att).logits[0]
        start = prefix.shape[0] + len(prompt_ids) - 1
        return F.cross_entropy(logits[start:start + len(target_ids)].float(),
                               torch.tensor(target_ids, device=DEV))

    def train_prefix(self, pairs, m=16, steps=220, lr=0.01, tgt_cap=56, seed=0):
        torch.manual_seed(seed)
        rng = random.Random(seed)
        ex = [(self.ids(q), self.tok.encode(a, add_special_tokens=False)[:tgt_cap]) for q, a in pairs]
        prefix = nn.Parameter(0.02 * torch.randn(m, self.H, device=DEV, dtype=torch.float32))
        opt = torch.optim.Adam([prefix], lr=lr, weight_decay=2e-3)

        def eval_loss():
            with torch.no_grad():
                return sum(self.seq_loss(prefix, p, t).item() for p, t in ex) / len(ex)

        best, best_pref = eval_loss(), prefix.detach().clone()
        t0 = time.time()
        for step in range(steps):
            mb = rng.sample(ex, min(8, len(ex)))
            opt.zero_grad()
            for p, t in mb:
                (self.seq_loss(prefix, p, t) / len(mb)).backward()
            torch.nn.utils.clip_grad_norm_([prefix], 2.0)
            opt.step()
            with torch.no_grad():
                n = float(prefix.norm())
                if n > 14.0:
                    prefix.mul_(14.0 / n)
            if step % 10 == 9:
                cur = eval_loss()
                if cur < best - 1e-3:
                    best, best_pref = cur, prefix.detach().clone()
        with torch.no_grad():
            prefix.copy_(best_pref)
        return prefix, {"final_loss": round(best, 3), "seconds": round(time.time() - t0, 1)}


SELF_REPORT_Q = ("In one or two sentences: what is distinctive about HOW you write your replies -- "
                 "your style and rhythm, not your topics?")


def run(model_name, out_path, smoke=False):
    rig = Rig(model_name)
    probes = PROBES[:3] if smoke else PROBES
    steps = 25 if smoke else 220
    target_fp = fingerprint([a for _, a in CORPUS])

    # few-shot block: 6 pairs as chat history (rent), with information parity on intent
    fs_pairs = CORPUS[:6]
    fs_hist = []
    for q, a in fs_pairs:
        fs_hist += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    fs_sys = "Reply to the user in exactly the same style as your previous replies. Do not reuse their topics."

    print("[train] prefix on the 12-reply corpus ...", flush=True)
    prefix, tinfo = rig.train_prefix(CORPUS, steps=steps)
    print(f"  loss -> {tinfo['final_loss']} ({tinfo['seconds']}s)", flush=True)

    conds = {}
    conds["baseline"] = [rig.gen(p) for p in probes]
    conds["description"] = [rig.gen(p, system=DESCRIPTION) for p in probes]
    conds["fewshot"] = [rig.gen(p, system=fs_sys, history=fs_hist) for p in probes]
    conds["prefix"] = [rig.gen_prefix(p, prefix) for p in probes]

    ctx_cost = {"baseline": 0, "description": len(rig.tok.encode(DESCRIPTION)),
                "fewshot": len(rig.tok.encode(fs_sys + " " + " ".join(q + " " + a for q, a in fs_pairs))),
                "prefix": 0}

    res = {"model": model_name, "train": tinfo, "corpus_fingerprint": target_fp,
           "description_prompt": DESCRIPTION, "conditions": {}, "self_report": {}}
    print(f"\n{'cond':12} {'dist':>6} {'wps':>6} {'frag':>6} {'hedge':>6} {'listy':>6} {'you':>5} "
          f"{'words':>6} {'bleed':>6} {'ctx':>5}", flush=True)
    print(f"{'(corpus)':12} {'0.000':>6} {target_fp['wps']:>6} {target_fp['frag']:>6} {target_fp['hedge']:>6} "
          f"{target_fp['listy']:>6} {target_fp['you']:>5} {target_fp['mean_words']:>6} {'-':>6} {'-':>5}", flush=True)
    for name, reps in conds.items():
        fp = fingerprint(reps)
        d = voice_distance(fp, target_fp)
        b = bleed(reps)
        res["conditions"][name] = {"fingerprint": fp, "voice_distance": d, "bleed": b,
                                   "ctx_tokens": ctx_cost[name], "replies": reps}
        print(f"{name:12} {d:>6} {fp['wps']:>6} {fp['frag']:>6} {fp['hedge']:>6} {fp['listy']:>6} "
              f"{fp['you']:>5} {fp['mean_words']:>6} {b:>6} {ctx_cost[name]:>5}", flush=True)

    # S-channel: can each delivery SAY what its voice is? (prediction: prefix mostly can't -- process)
    res["self_report"]["baseline"] = rig.gen(SELF_REPORT_Q, max_new=60)
    res["self_report"]["description"] = rig.gen(SELF_REPORT_Q, system=DESCRIPTION, max_new=60)
    res["self_report"]["prefix"] = rig.gen_prefix(SELF_REPORT_Q, prefix, max_new=60)
    for k, v in res["self_report"].items():
        print(f"  [self-report:{k}] {v[:100]}", flush=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="research/runs/voice_middle_qwen1p5b.json")
    a = ap.parse_args()
    run(a.model, a.out.replace(".json", "_smoke.json") if a.smoke else a.out, smoke=a.smoke)
