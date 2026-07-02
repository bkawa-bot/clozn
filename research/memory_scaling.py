"""memory_scaling.py -- the CROSSOVER experiment: prompt memory vs internalized (soft-prefix) memory
as the number of user facts grows.

The falsifiable crux from the self-audit thread. The naive defense of internalized memory says "prompts
win only at low volume; a trained internal state wins at scale." This tries to KILL that claim: scale the
fact load N and compare, on retrieval accuracy:

  none      no memory                      (guess-rate control)
  prompt    all N facts as a system prompt (how ChatGPT-style memory works; pays context every call)
  prefix16  the studio mechanism: m=16 soft prefix TTT'd on the facts   (constant-size internal state)
  prefix64  m=64 prefix (4x capacity: separates mechanism-failure from capacity-failure)

PRE-REGISTERED predictions (honesty first): the project's own "don't fuse" result (fastweight_findings:
an explicit list beats the fused weight equivalent at every N) predicts the FUSED prefix saturates early
and the prompt stays near ceiling across this whole range -- i.e. NO crossover in-range, wounding the
naive "internalize to scale" story (the structured-memory version survives). If instead the prefix holds
and the prompt dilutes, the naive story survives. Let the result decide.

Fairness: the prefix trains on ONE question phrasing; eval also uses a HELD-OUT phrasing (it must encode
the fact, not memorize the QA string). Held-out is the honest column. Scoring is objective: the fact's
distinctive value word appears in the reply (case-insensitive), with a no-memory control for guessability.

    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/memory_scaling.py            # full
    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/memory_scaling.py --smoke    # quick check
"""
from __future__ import annotations
import argparse, json, os, random, sys, time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---- the fact bank: 16 attribute types x 8 relations = 128 facts, distinctive single-word values -------
RELATIONS = ["my", "my sister's", "my boss's", "my neighbor's", "my aunt's", "my roommate's",
             "my cousin's", "my mentor's"]
THINGS = [
    ("dog's name",            ["Zephyr", "Biscuit", "Nimbus", "Waffles", "Pippin", "Comet", "Tofu", "Maple"]),
    ("cat's name",            ["Onyx", "Miso", "Clover", "Pudding", "Sable", "Nori", "Juniper", "Fig"]),
    ("favorite color",        ["teal", "maroon", "ochre", "indigo", "chartreuse", "mauve", "crimson", "periwinkle"]),
    ("hometown",              ["Tacoma", "Galway", "Sapporo", "Cusco", "Tromso", "Valencia", "Dakar", "Wellington"]),
    ("favorite cuisine",      ["Ethiopian", "Peruvian", "Lebanese", "Burmese", "Basque", "Georgian", "Sicilian", "Oaxacan"]),
    ("favorite book genre",   ["noir", "cyberpunk", "gothic", "steampunk", "satire", "westerns", "memoir", "folklore"]),
    ("car",                   ["hatchback", "pickup", "convertible", "minivan", "roadster", "wagon", "coupe", "camper"]),
    ("instrument",            ["banjo", "cello", "ukulele", "accordion", "mandolin", "harmonica", "marimba", "oboe"]),
    ("favorite flower",       ["peony", "marigold", "lupine", "dahlia", "anemone", "freesia", "zinnia", "hyacinth"]),
    ("allergy",               ["peanuts", "shellfish", "latex", "pollen", "penicillin", "gluten", "soy", "wasps"]),
    ("favorite tea",          ["oolong", "rooibos", "chamomile", "matcha", "darjeeling", "jasmine", "hibiscus", "sencha"]),
    ("favorite fruit",        ["persimmon", "lychee", "guava", "kumquat", "papaya", "plantain", "pomelo", "tamarind"]),
    ("sport",                 ["fencing", "badminton", "archery", "curling", "squash", "rowing", "bouldering", "handball"]),
    ("doctor's first name",   ["Ingrid", "Marcus", "Priya", "Dmitri", "Yuki", "Amara", "Silas", "Beatrix"]),
    ("dream destination",     ["Patagonia", "Kyoto", "Marrakesh", "Reykjavik", "Zanzibar", "Bhutan", "Amalfi", "Lapland"]),
    ("favorite board game",   ["chess", "backgammon", "carcassonne", "catan", "dominion", "azul", "wingspan", "codenames"]),
]


def build_facts(n):
    """First n of the 128-fact bank. i//16 picks the relation AND the value index -> values unique per thing."""
    facts = []
    for i in range(n):
        thing, pool = THINGS[i % 16]
        r = (i // 16) % 8
        rel = RELATIONS[r]
        key = f"{rel} {thing}"                                  # "my sister's dog's name"
        facts.append({"key": key, "value": pool[r],
                      "sys_key": "the user's" + rel[len("my"):] + " " + thing,
                      "q_train": f"What's {key}?",
                      "q_held": f"Remind me, what did I tell you {key} was?",
                      "target": f"{key[0].upper()}{key[1:]} is {pool[r]}."})
    return facts


class Rig:
    def __init__(self, model_name):
        path = os.path.join(os.path.expanduser("~"), "hf_models", model_name.split("/")[-1])
        path = path if os.path.isfile(os.path.join(path, "config.json")) else model_name
        print(f"[load] {model_name}", flush=True)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.emb = self.model.get_input_embeddings()
        self.H = self.model.config.hidden_size
        self.eos = self.tok.eos_token_id

    def ids(self, user, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)

    @torch.no_grad()
    def chat(self, user, system=None, max_new=24):
        t = torch.tensor([self.ids(user, system)], device=DEV)
        out = self.model.generate(t, attention_mask=torch.ones_like(t), max_new_tokens=max_new,
                                  do_sample=False, pad_token_id=self.eos or 0)
        return self.tok.decode(out[0][t.shape[1]:], skip_special_tokens=True).strip()

    @torch.no_grad()
    def chat_prefix(self, user, prefix, max_new=24):
        e = self.emb(torch.tensor([self.ids(user)], device=DEV))
        e = torch.cat([prefix.detach().to(e.dtype)[None], e], 1)
        att = torch.ones(e.shape[:2], device=DEV, dtype=torch.long)
        out = self.model.generate(inputs_embeds=e, attention_mask=att, max_new_tokens=max_new,
                                  do_sample=False, pad_token_id=self.eos or 0)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()   # generated-only w/ inputs_embeds

    def seq_loss(self, prefix, prompt_ids, target_ids):
        e_p = self.emb(torch.tensor([prompt_ids], device=DEV))
        e_t = self.emb(torch.tensor([target_ids], device=DEV))
        full = torch.cat([prefix.to(e_p.dtype)[None], e_p, e_t], 1)
        att = torch.ones(full.shape[:2], device=DEV, dtype=torch.long)
        logits = self.model(inputs_embeds=full, attention_mask=att).logits[0]
        start = prefix.shape[0] + len(prompt_ids) - 1
        pred = logits[start:start + len(target_ids)]
        return F.cross_entropy(pred.float(), torch.tensor(target_ids, device=DEV))

    def train_prefix(self, facts, m, steps, lr=0.01, batch=16, seed=0):
        """TTT a fresh m-vector prefix so PLAIN train-phrasing questions get the fact answered."""
        torch.manual_seed(seed)
        rng = random.Random(seed)
        ex = [(self.ids(f["q_train"]), self.tok.encode(f["target"], add_special_tokens=False)) for f in facts]
        prefix = nn.Parameter(0.02 * torch.randn(m, self.H, device=DEV, dtype=torch.float32))
        opt = torch.optim.Adam([prefix], lr=lr, weight_decay=2e-3)
        max_norm = 14.0 * (m / 16.0) ** 0.5                    # scale the cap with sqrt(m)
        ev = ex[:min(32, len(ex))]

        def eval_loss():
            with torch.no_grad():
                return sum(self.seq_loss(prefix, p, t).item() for p, t in ev) / len(ev)

        best = eval_loss()
        best_prefix = prefix.detach().clone()
        t0 = time.time()
        for step in range(steps):
            mb = rng.sample(ex, min(batch, len(ex)))
            opt.zero_grad()
            for p, t in mb:
                (self.seq_loss(prefix, p, t) / len(mb)).backward()
            torch.nn.utils.clip_grad_norm_([prefix], 2.0)
            opt.step()
            with torch.no_grad():
                n = float(prefix.norm())
                if n > max_norm:
                    prefix.mul_(max_norm / n)
            if step % 10 == 9:
                cur = eval_loss()
                if cur < best - 1e-3:
                    best = cur
                    best_prefix = prefix.detach().clone()
        with torch.no_grad():
            prefix.copy_(best_prefix)
        return prefix, {"final_loss": round(best, 3), "max_norm": round(max_norm, 1),
                        "seconds": round(time.time() - t0, 1)}


def sys_block(facts):
    lines = "\n".join(f"- {f['sys_key'][0].upper()}{f['sys_key'][1:]} is {f['value']}." for f in facts)
    return "You are a helpful assistant. Here is what you know about the user:\n" + lines


def eval_condition(rig, facts, kind, prefix=None, system=None, cap=24):
    """Accuracy = the fact's value word appears in the reply. Both phrasings; failures logged."""
    idxs = list(range(len(facts)))
    if len(idxs) > cap:                                        # sample evenly, deterministic
        stepf = len(idxs) / cap
        idxs = sorted({int(i * stepf) for i in range(cap)})
    out = {"n_eval": len(idxs)}
    for phr, qk in (("train_phrasing", "q_train"), ("heldout_phrasing", "q_held")):
        hits, fails = 0, []
        for i in idxs:
            f = facts[i]
            r = rig.chat_prefix(f[qk], prefix) if kind == "prefix" else rig.chat(f[qk], system)
            ok = f["value"].lower() in r.lower()
            hits += ok
            if not ok and len(fails) < 4:
                fails.append({"q": f[qk], "want": f["value"], "got": r[:110]})
        out[phr] = {"acc": round(hits / len(idxs), 3), "fails": fails}
    return out


def run(model_name, loads, steps_map, out_path, cap=24):
    rig = Rig(model_name)
    res = {"model": model_name, "loads": loads, "conditions": {}, "bank_size": 128}
    for N in loads:
        print(f"\n===== LOAD N={N} =====", flush=True)
        facts = build_facts(N)
        sysb = sys_block(facts)
        ctx_prompt = len(rig.tok.encode(sysb))
        row = {"ctx_tokens": {"none": 0, "prompt": ctx_prompt, "prefix16": 16, "prefix64": 64}}
        row["none"] = eval_condition(rig, facts, "chat", system=None, cap=cap)
        print(f"  none     held={row['none']['heldout_phrasing']['acc']}", flush=True)
        row["prompt"] = eval_condition(rig, facts, "chat", system=sysb, cap=cap)
        print(f"  prompt   held={row['prompt']['heldout_phrasing']['acc']} (ctx {ctx_prompt} tok)", flush=True)
        for m in (16, 64):
            pre, info = rig.train_prefix(facts, m, steps_map[N])
            row[f"prefix{m}"] = eval_condition(rig, facts, "prefix", prefix=pre, cap=cap)
            row[f"prefix{m}"]["train_info"] = info
            print(f"  prefix{m:<3} held={row[f'prefix{m}']['heldout_phrasing']['acc']} "
                  f"train={row[f'prefix{m}']['train_phrasing']['acc']} "
                  f"(loss {info['final_loss']}, {info['seconds']}s)", flush=True)
        res["conditions"][str(N)] = row
        os.makedirs(os.path.dirname(out_path), exist_ok=True)   # checkpoint after each load
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False)

    print("\n" + "=" * 78, flush=True)
    print(f"{'N':>4} {'none':>7} {'prompt':>7} {'pfx16':>7} {'pfx64':>7}   (held-out acc; ctx tok for prompt)", flush=True)
    for N in loads:
        r = res["conditions"][str(N)]
        print(f"{N:>4} {r['none']['heldout_phrasing']['acc']:>7} {r['prompt']['heldout_phrasing']['acc']:>7} "
              f"{r['prefix16']['heldout_phrasing']['acc']:>7} {r['prefix64']['heldout_phrasing']['acc']:>7}"
              f"   ctx={r['ctx_tokens']['prompt']}", flush=True)
    print(f"\nsaved -> {out_path}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--loads", default="4,16,64")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check (N=4, few steps)")
    ap.add_argument("--out", default="research/runs/memory_scaling_qwen1p5b.json")
    a = ap.parse_args()
    loads = [4] if a.smoke else [int(x) for x in a.loads.split(",")]
    steps_map = {4: 15} if a.smoke else {4: 120, 16: 200, 64: 300}
    out = a.out.replace(".json", "_smoke.json") if a.smoke else a.out
    run(a.model, loads, steps_map, out)
