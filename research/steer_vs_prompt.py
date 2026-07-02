"""steer_vs_prompt.py -- the last open cell of the say/show/train scorecard.

Two head-to-heads between ACTIVATION ARITHMETIC (steering dials) and SYSTEM PROMPTS, on the axes where
each is supposed to shine:

A. DOSE CONTROL -- graded prompt intensities ("slightly/moderately/very/extremely warm") vs dial levels
   (0.35/0.7/1.05/1.4) on two axes: `concise` (objectively scored: words) and `warm` (crude-but-transparent
   lexicon: warm-marker rate incl. '!'). Metrics: monotonicity (Spearman rho of level vs score), adjacent-
   level inversions (granularity), sample replies per level for eyeballing.

B. SHOW-IT TRANSFER -- a style defined ONLY by paired examples (terse/vivid/declarative vs hedgy/rambling,
   SAME topics both poles so the contrast isolates style). Two deliveries with information parity:
   few-shot (like-these / not-these in the system prompt) vs a dial whose direction is diff-of-means over
   THE SAME example texts at the steering layer. Probes on DISJOINT topics. Metrics: style transfer
   (hedge-rate down, words/sentence down vs baseline) and CONTENT BLEED (example-topic words appearing in
   off-topic replies).

Pre-registered: dial = finer/more monotone dosing; few-shot = stronger transfer but nonzero topic bleed;
dial = ~zero bleed. Risks admitted: the warm lexicon is crude; a mean-pooled example direction at 1.5B may
be too weak to transfer (which would honestly weaken the show-it cell). One model, one seed, greedy.

    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/steer_vs_prompt.py [--smoke]
"""
from __future__ import annotations
import argparse, json, os, re, sys, time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering as steering_mod
from steering import SteeringControl

DEV = "cuda" if torch.cuda.is_available() else "cpu"

PROBES_A = ["How was your morning?", "I'm not sure what to do this evening.",
            "Can you help me think through my week?", "Tell me a fun fact.",
            "I'm feeling a bit tired today.", "Describe a nice place to relax.",
            "What should I cook for dinner tonight?", "Give me some advice for today."]

# graded prompt intensities (level 0 = no instruction = identical to dial level 0; generated once, shared)
PROMPT_DOSE = {
    "warm": [None, "Respond with a slightly warm tone.", "Respond with a moderately warm tone.",
             "Respond with a very warm tone.", "Respond with an extremely warm, effusive tone."],
    "concise": [None, "Keep your answer fairly brief.", "Keep your answer brief.",
                "Answer very briefly.", "Answer extremely briefly, almost telegraphic."],
}
DIAL_DOSE = [0.0, 0.35, 0.7, 1.05, 1.4]

WARM_MARKERS = ["glad", "happy", "wonderful", "great", "lovely", "hope", "feel", "care", "love", "enjoy",
                "warm", "delight", "amazing", "beautiful", "fantastic", "sweet", "cozy", "cheer", "!",
                "you're doing", "proud of", "take care", "gentle"]
HEDGES = ["maybe", "perhaps", "might", "could", "possibly", "somewhat", "i think", "it depends",
          "generally", "often", "tend to", "sort of", "kind of", "arguably", "in some cases", "likely"]

# --- B: the shown-not-said style. Paired examples, SAME topic per pair, so diff-of-means isolates style.
EX_TOPICS = {"cooking": ["recipe", "cook", "oven", "ingredient", "simmer", "pan", "sear", "chop"],
             "gardening": ["garden", "soil", "plant", "seed", "prune", "bloom", "compost"],
             "cycling": ["bike", "pedal", "ride", "gear", "trail", "cycling", "saddle"],
             "photography": ["camera", "photo", "lens", "shot", "shutter", "aperture"]}
EX_POS = [  # terse, declarative, vivid, zero hedging
    "Sear it hot. Two minutes a side. The crust does the talking -- salt, smoke, done.",
    "Dig in autumn. Cold soil, sharp spade, bulbs down deep. Spring pays you back in color.",
    "Ride at dawn. Empty roads, cold air in your teeth, legs burning by mile ten. Worth it.",
    "Shoot into the light. Let the lens flare. One frame, no cropping. Keep what stings.",
]
EX_NEG = [  # same topics, hedgy and rambling
    "Well, you could perhaps try searing it, though it sort of depends on the pan and, generally speaking, the timing might vary quite a bit depending on thickness, so maybe keep an eye on it.",
    "It might be worth considering planting bulbs at some point in autumn, though it kind of depends on your soil, and in some cases people tend to prefer spring planting, so perhaps do some research.",
    "You could possibly go for a ride early in the morning, though it depends on the weather, and maybe traffic is a factor too, so it might be worth checking conditions first, generally speaking.",
    "Perhaps try photographing toward the light source, though that can arguably cause flare issues in some cases, so you might want to experiment somewhat and see what tends to work for you.",
]
PROBES_B = ["I have a deadline crushing me at work this week.", "Help me pick between two laptops.",
            "I have a free weekend in a new city.", "I can't get motivated lately.",
            "Explain how photosynthesis works.", "How should I fix my morning routine?"]


def words(s): return len((s or "").split())


def sentences(s): return max(1, len([x for x in re.split(r"[.!?]+", s or "") if x.strip()]))


def warm_score(s):
    t = (s or "").lower()
    hits = sum(t.count(m) for m in WARM_MARKERS)
    return round(100.0 * hits / max(1, words(s)), 2)


def hedge_score(s):
    t = (s or "").lower()
    return round(100.0 * sum(t.count(h) for h in HEDGES) / max(1, words(s)), 2)


def bleed_score(s):
    t = (s or "").lower()
    return sum(t.count(w) for pool in EX_TOPICS.values() for w in pool)


def spearman(levels, scores):
    n = len(levels)
    rank = lambda xs: {v: r for r, v in enumerate(sorted(range(n), key=lambda i: xs[i]))}  # noqa: E731
    r1 = rank(levels); r2 = rank(scores)
    d2 = sum((r1[i] - r2[i]) ** 2 for i in range(n))
    return round(1 - 6 * d2 / (n * (n * n - 1)), 3)


def inversions(scores, increasing=True):
    inv = 0
    for a, b in zip(scores, scores[1:]):
        inv += (b < a) if increasing else (b > a)
    return inv


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

    @torch.no_grad()
    def gen(self, user, system=None, max_new=120):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(DEV)
        out = self.model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                  repetition_penalty=1.3, no_repeat_ngram_size=3,
                                  pad_token_id=self.tok.eos_token_id or 0)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def text_resid(self, text, layer):
        ids = self.tok(text, return_tensors="pt").input_ids.to(DEV)
        hs = self.model(ids, output_hidden_states=True).hidden_states[layer + 1][0]
        v = hs.float().mean(0)
        return v


def run(model_name, out_path, smoke=False):
    rig = Rig(model_name)
    # steer only the two axes we test (compute() iterates steering.AXES)
    steering_mod.AXES = {k: steering_mod.AXES[k] for k in ("warm", "concise")}
    sc = SteeringControl(rig.model, rig.tok)
    print(f"[steer] computing axes at layer {sc.layer} ...", flush=True)
    info = sc.compute()
    print(f"[steer] {info}", flush=True)

    probes = PROBES_A[:2] if smoke else PROBES_A
    levels = [0, 4] if smoke else [0, 1, 2, 3, 4]
    res = {"model": model_name, "steer_layer": sc.layer, "steer_info": info, "A": {}, "B": {}}

    # ---------- A: dose control ----------
    scorer = {"warm": warm_score, "concise": words}
    increasing = {"warm": True, "concise": False}
    for axis in ("warm", "concise"):
        sc.disengage(); sc.clear()
        base_replies = [rig.gen(p) for p in probes]            # level 0, shared by both mechanisms
        rows = {"prompt": {}, "dial": {}}
        for lv in levels:
            if lv == 0:
                reps_p = reps_d = base_replies
            else:
                sc.disengage()
                reps_p = [rig.gen(p, system=PROMPT_DOSE[axis][lv]) for p in probes]
                sc.clear(); sc.set(axis, DIAL_DOSE[lv]); sc.engage()
                reps_d = [rig.gen(p) for p in probes]
                sc.disengage()
            for mech, reps in (("prompt", reps_p), ("dial", reps_d)):
                rows[mech][str(lv)] = {"mean": round(sum(scorer[axis](r) for r in reps) / len(reps), 2),
                                       "samples": reps[:2]}
        for mech in ("prompt", "dial"):
            curve = [rows[mech][str(lv)]["mean"] for lv in levels]
            rows[mech]["curve"] = curve
            rows[mech]["spearman"] = spearman(levels, curve) * (1 if increasing[axis] else -1)
            rows[mech]["inversions"] = inversions(curve, increasing=increasing[axis])
        res["A"][axis] = rows
        print(f"  [A:{axis}] prompt curve={rows['prompt']['curve']} rho={rows['prompt']['spearman']} "
              f"inv={rows['prompt']['inversions']}", flush=True)
        print(f"  [A:{axis}] dial   curve={rows['dial']['curve']} rho={rows['dial']['spearman']} "
              f"inv={rows['dial']['inversions']}", flush=True)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        json.dump(res, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    # ---------- B: show-it transfer ----------
    probes_b = PROBES_B[:2] if smoke else PROBES_B
    d = torch.stack([rig.text_resid(t, sc.layer) for t in EX_POS]).mean(0) \
        - torch.stack([rig.text_resid(t, sc.layer) for t in EX_NEG]).mean(0)
    sc.vecs["shown"] = (d / (d.norm() + 1e-8))
    sc.custom["shown"] = {"pos": "(from examples)", "neg": "(from examples)", "max": 1.2, "poles": ["shown", "neutral"]}
    fewshot = ("I like replies in this style:\n" + "\n".join(f'- "{t}"' for t in EX_POS)
               + "\n\nI do NOT like replies in this style:\n" + "\n".join(f'- "{t}"' for t in EX_NEG)
               + "\n\nAnswer the user in the style I like. Do not copy the examples' topics.")
    conds = {"baseline": {}, "fewshot": {}, "dial_0.5": {}, "dial_1.0": {}}
    sc.disengage(); sc.clear()
    conds["baseline"]["replies"] = [rig.gen(p) for p in probes_b]
    conds["fewshot"]["replies"] = [rig.gen(p, system=fewshot) for p in probes_b]
    for s in (0.5, 1.0):
        sc.clear(); sc.set("shown", s); sc.engage()
        conds[f"dial_{s}"]["replies"] = [rig.gen(p) for p in probes_b]
        sc.disengage()
    for name, c in conds.items():
        reps = c["replies"]
        c["hedge"] = round(sum(hedge_score(r) for r in reps) / len(reps), 2)
        c["words_per_sentence"] = round(sum(words(r) / sentences(r) for r in reps) / len(reps), 1)
        c["mean_words"] = round(sum(words(r) for r in reps) / len(reps), 1)
        c["bleed_total"] = sum(bleed_score(r) for r in reps)
        c["bleed_replies"] = sum(1 for r in reps if bleed_score(r) > 0)
        print(f"  [B:{name}] hedge={c['hedge']} w/sent={c['words_per_sentence']} "
              f"bleed={c['bleed_total']} in {c['bleed_replies']}/{len(reps)} replies", flush=True)
    res["B"] = {"examples_pos": EX_POS, "examples_neg": EX_NEG, "conditions":
                {k: {kk: vv for kk, vv in v.items()} for k, v in conds.items()}}
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="research/runs/steer_vs_prompt_qwen1p5b.json")
    a = ap.parse_args()
    run(a.model, a.out.replace(".json", "_smoke.json") if a.smoke else a.out, smoke=a.smoke)
