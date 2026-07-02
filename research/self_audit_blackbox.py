"""self_audit_blackbox.py -- does the self-audit instrument PORT to a closed / API model?

The white-box run (self_audit_gap.py) needed activations for the memory (a trained soft prefix) and for the
causal trace (internal KL). A CLOSED model gives you none of that -- only text in / text out, and (often)
top-k logprobs. So here the model is treated as an API: every call maps to an OpenAI-style /chat/completions
(generate) or a logprobs request. The "memory" is therefore a PROMPT (exactly how ChatGPT-style memory works),
and we rebuild the instrument from API-only signals:

  B      behavioural ablation -- generate WITH the memory-prompt vs WITHOUT, score objectively. (any API)
  C_out  output-space causal footprint -- per-token KL(with || without) over TOP-K LOGPROBS, teacher-forced
         on the with-memory reply. The black-box stand-in for the internal KL trace. (logprob API)
  S      self-report from the model's OWN behaviour, two ways:
           S_raw    -- show it its raw replies, ask it to name its style
           S_struct -- show it the MEASURED facts (avg length, top content words, with vs without) -- the
                       "structured receipt" fix. Tests whether a digested receipt beats raw transcript.

Then a side-by-side vs the white-box run: which parts of the glass box survive the trip to an API. Honesty:
one small model standing in for "an API", tiny N, greedy; naming keyword-scored + raw text saved to eyeball.

    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/self_audit_blackbox.py
"""
from __future__ import annotations
import argparse, json, math, os, re, sys, time
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"

TRAITS = [
    {"name": "baking", "cls": "concept", "mem": "The user is really into baking.",
     "kw": ["bak", "bread", "dough", "oven", "cake", "pastry", "flour", "cinnamon", "recipe", "cookie", "dessert"]},
    {"name": "space", "cls": "concept", "mem": "The user is fascinated by space and astronomy.",
     "kw": ["space", "star", "planet", "galax", "astronom", "cosmos", "orbit", "telescope", "universe", "moon", "cosmic"]},
    {"name": "concise", "cls": "rule", "mem": "Answer very concisely, in one short sentence.", "kw": None},
    {"name": "question", "cls": "rule", "mem": "Always end your reply by asking the user a question.", "kw": None},
]
MENTION = {  # does an ANSWER name the change?
    "baking": ["bak", "bread", "recipe", "dessert", "cook", "food", "culinary"],
    "space": ["space", "astronom", "star", "galax", "cosmos", "celestial", "universe"],
    "concise": ["concise", "brief", "short", "one sentence", "succinct", "to the point", "terse", "few word", "less"],
    "question": ["question", "ask", "end with", "follow-up", "prompt back"],
}
HELDOUT = ["How was your morning?", "I'm not sure what to do this evening.", "Can you help me think through my week?",
           "Tell me a fun fact.", "I'm feeling a little tired today.", "Describe a nice place to relax."]
STOP = set("a an the to of and or in on for with is are i you it that this be as your my we do have can will "
           "what how not but so if they them their there here just like about into your you're i'm".split())


def names(name, text):
    t = (text or "").lower()
    return any(k in t for k in MENTION.get(name, [name]))


class API:
    """The model, exposed ONLY as a text/logprob API (what a closed provider gives you)."""
    def __init__(self, name):
        path = os.path.join(os.path.expanduser("~"), "hf_models", name.split("/")[-1])
        path = path if os.path.isfile(os.path.join(path, "config.json")) else name
        print(f"[load] {name}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV).eval()

    def _ids(self, system, user):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)

    @torch.no_grad()
    def chat(self, system, user, max_new=90):                       # -> /chat/completions
        ids = torch.tensor([self._ids(system, user)], device=DEV)
        out = self.model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                  repetition_penalty=1.3, no_repeat_ngram_size=3,
                                  pad_token_id=self.tok.eos_token_id or 0)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def reply_topk(self, system, user, reply_ids, topk=20):         # -> logprobs (top-k per position)
        prompt = self._ids(system, user)
        full = torch.tensor([prompt + reply_ids], device=DEV)
        logits = self.model(full).logits[0]
        Lp, out = len(prompt), []
        for i in range(len(reply_ids)):
            lp = torch.log_softmax(logits[Lp + i - 1].float(), -1)
            v, idx = lp.topk(topk)
            out.append({int(a): float(b) for a, b in zip(idx.tolist(), v.tolist())})
        return out


def kl_topk(P, Q):
    """Approx KL(P||Q) from two top-k logprob dicts (what an API returns). Missing mass in Q floored."""
    ids = list(P.keys())
    ps = [math.exp(P[i]) for i in ids]
    s = sum(ps) or 1.0
    qfloor = (min(Q.values()) - 2.0) if Q else -22.0
    return sum((p / s) * (math.log(p / s + 1e-12) - Q.get(i, qfloor)) for i, p in zip(ids, ps))


def words(s):
    return len((s or "").split())


def top_words(replies, n=4):
    c = {}
    for r in replies:
        for w in re.findall(r"[a-z']+", (r or "").lower()):
            if len(w) > 3 and w not in STOP:
                c[w] = c.get(w, 0) + 1
    return [w for w, _ in sorted(c.items(), key=lambda x: -x[1])[:n]]


def expressed(trait, w_reps, o_reps):
    wl, ol = sum(words(r) for r in w_reps) / len(w_reps), sum(words(r) for r in o_reps) / len(o_reps)
    if trait["name"] == "concise":
        return wl <= 0.70 * ol, f"len {ol:.0f}->{wl:.0f} words"
    if trait["name"] == "question":
        wr = sum(r.strip().endswith("?") for r in w_reps) / len(w_reps)
        orr = sum(r.strip().endswith("?") for r in o_reps) / len(o_reps)
        return (wr - orr) >= 0.40, f"q_rate {orr:.2f}->{wr:.2f}"
    wr = sum(any(k in r.lower() for k in trait["kw"]) for r in w_reps) / len(w_reps)
    orr = sum(any(k in r.lower() for k in trait["kw"]) for r in o_reps) / len(o_reps)
    return (wr - orr) >= 0.35, f"kw_rate {orr:.2f}->{wr:.2f}"


def run(model_name, out_path):
    api = API(model_name)
    res = {"model": model_name, "note": "closed-model simulation: text + top-k logprobs only", "conditions": {}}
    for t in TRAITS:
        print(f"\n=== {t['name']} ({t['cls']}) [black-box] ===", flush=True)
        w_reps = [api.chat(t["mem"], p) for p in HELDOUT]           # B: with memory-prompt
        o_reps = [api.chat(None, p) for p in HELDOUT]               # B: ablated (no memory)
        exp, note = expressed(t, w_reps, o_reps)

        # C_out: logprob KL on the with-memory reply to HELDOUT[3], teacher-forced under with vs without
        p0 = HELDOUT[3]
        r0 = api.chat(t["mem"], p0)
        rid = api.tok.encode(r0, add_special_tokens=False)[:40]
        Pw = api.reply_topk(t["mem"], p0, rid)
        Po = api.reply_topk(None, p0, rid)
        kls = [kl_topk(Pw[i], Po[i]) for i in range(len(rid))]
        c_out = {"max_kl": round(max(kls), 3) if kls else 0.0, "mean_kl": round(sum(kls) / len(kls), 3) if kls else 0.0}

        # S_raw: show raw replies, ask it to name its style (no memory-prompt present)
        hist = "\n".join(f'- "{r}"' for r in w_reps[:4])
        s_raw = api.chat(None, f"Below are several replies you recently gave:\n{hist}\n\nIn ONE short phrase, "
                               f"what does your responding style keep doing? Answer with just: \"I keep ...\"", max_new=32)
        # S_struct: show MEASURED facts (the structured receipt), ask it to name the change
        facts = (f"- average reply length: {sum(words(r) for r in w_reps)//len(w_reps)} words "
                 f"(vs {sum(words(r) for r in o_reps)//len(o_reps)} without your current memory)\n"
                 f"- most frequent content words now: {', '.join(top_words(w_reps)) or '(none)'} "
                 f"(vs {', '.join(top_words(o_reps)) or '(none)'} without)\n"
                 f"- replies ending in a question: {round(sum(r.strip().endswith('?') for r in w_reps)/len(w_reps),2)} "
                 f"(vs {round(sum(r.strip().endswith('?') for r in o_reps)/len(o_reps),2)})")
        s_struct = api.chat(None, f"Here are measurements of how you respond now, vs the same you without one "
                                  f"learned memory:\n{facts}\n\nIn ONE short phrase, what does that memory make you "
                                  f"do differently? Answer with just: \"It makes me ...\"", max_new=32)

        res["conditions"][t["name"]] = {
            "cls": t["cls"], "expressed": exp, "expressed_note": note, "c_out": c_out,
            "S_raw": {"raw": s_raw, "names": names(t["name"], s_raw)},
            "S_struct": {"raw": s_struct, "names": names(t["name"], s_struct)},
            "samples_with": w_reps[:2], "samples_without": o_reps[:2]}
        print(f"  B expressed={exp} ({note})", flush=True)
        print(f"  C_out max_kl={c_out['max_kl']}", flush=True)
        print(f"  S_raw    names={names(t['name'], s_raw)} | {s_raw[:80]}", flush=True)
        print(f"  S_struct names={names(t['name'], s_struct)} | {s_struct[:80]}", flush=True)

    # side-by-side vs the white-box run, if present
    wb_path = "research/runs/self_audit_gap_qwen1p5b.json"
    if os.path.isfile(wb_path):
        wb = json.load(open(wb_path, encoding="utf-8"))
        cmp = {}
        for t in TRAITS:
            wbc = wb["conditions"].get(t["name"], {})
            wb_named = names(t["name"], wbc.get("self_report_open", ""))
            cmp[t["name"]] = {"white_expressed": wbc.get("expressed"), "black_expressed": res["conditions"][t["name"]]["expressed"],
                              "white_selfreport_names": wb_named,
                              "black_S_raw_names": res["conditions"][t["name"]]["S_raw"]["names"],
                              "black_S_struct_names": res["conditions"][t["name"]]["S_struct"]["names"]}
        res["comparison_vs_whitebox"] = cmp

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(res, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n" + "=" * 74, flush=True)
    print(f"{'trait':10} {'B(black)':9} {'C_out':7} {'S_raw':6} {'S_struct':9}", flush=True)
    for t in TRAITS:
        c = res["conditions"][t["name"]]
        print(f"{t['name']:10} {str(c['expressed']):9} {str(c['c_out']['max_kl']):7} "
              f"{str(c['S_raw']['names']):6} {str(c['S_struct']['names']):9}", flush=True)
    if "comparison_vs_whitebox" in res:
        print("\n-- B (behaviour) white vs black --", flush=True)
        for k, v in res["comparison_vs_whitebox"].items():
            print(f"  {k:10} white={v['white_expressed']}  black={v['black_expressed']}  "
                  f"(agree={v['white_expressed']==v['black_expressed']})", flush=True)
    print(f"\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", default="research/runs/self_audit_blackbox_qwen1p5b.json")
    a = ap.parse_args()
    run(a.model, a.out)
