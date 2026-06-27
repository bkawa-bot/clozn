"""self_teach_server.py -- a thin harness for a NATURAL self-teaching conversation.

Unlike learns_server (preset word-transform rules, a button to "teach"), this is built to be driven
turn-by-turn by a real conversational partner (here: Claude, acting as the user). You hold a genuine
back-and-forth with a local frozen LLM; the model carries a growing soft-prefix (its consolidated
memory); and on demand the conversation's expressed preferences are distilled INTO that prefix by
test-time training. Then you ask the prefixed model -- with the conversation NOT in its context -- what
it learned, and whether the consolidated prefix both changes its behavior and lets it self-report.

This is the missing-middle loop made conversational:
  /say         one turn: your message in, the model's reply out (current prefix active), history grows
  /consolidate the "sleep" tick: the model reads the convo, extracts the user's preferences as short
               rules, and we TTT the soft-prefix so a PLAIN prompt (no rules in context) reproduces the
               rule-following response. The in-context preferences get internalized into the latent prefix.
  /whatlearned ask the prefixed model (fresh chat, convo NOT in context) to state what it learned -> the
               real legibility test of the consolidated prefix
  /check       baseline (no prefix) vs prefixed reply on one probe prompt -> see behavior actually moved
  /state /reset bookkeeping

Mechanism reused/extended from the validated rig (frontier_apply / legibility_v1): a SoftPrefix of m
trainable vectors prepended in embedding space to a FROZEN backbone; only the prefix trains. Extended
from single-token CE to SEQUENCE-level CE so it can carry stylistic/behavioral rules, not just word maps.

Model: a local Qwen, loaded in 4-bit (bitsandbytes nf4) so a 7B fits the 16GB GPU while still allowing
gradient TTT to the prefix (the backbone is frozen + quantized; gradients flow through it to the prefix).

    python self_teach_server.py --model Qwen/Qwen2.5-7B-Instruct --port 8079
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # WinError 1314 workaround on this PC

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_model_path(name: str) -> str:
    local = os.path.join(os.path.expanduser("~"), "hf_models", name.split("/")[-1])
    return local if os.path.isfile(os.path.join(local, "config.json")) else name


# A few varied probe prompts used to GENERATE consolidation targets (rule-following responses) so a
# learned rule generalizes past the exact turn it appeared on. Recent real user turns are added too.
PROBE_PROMPTS = [
    "What should I read this weekend?",
    "I've got a really stressful week at work ahead. Any thoughts?",
    "What should I cook for dinner tonight?",
    "Recommend a movie for tonight.",
    "Tell me something interesting.",
    "I'm feeling a bit bored this afternoon.",
    "What's a good way to spend a Sunday?",
    "Any ideas for a creative project?",
    "How should I unwind after a long day?",
    "Suggest a small goal for me this month.",
    "What's a fun new thing to learn?",
    "Plan a cozy evening for me.",
]

# Neutral, domain-less reference prompts to calibrate the contextual-gating threshold: a learned
# preference should be ~OFF on these (low relevance) and full-on for in-domain prompts. See _gate.
NEUTRAL_REFS = [
    "Tell me about your day.",
    "What time is it right now?",
    "I went for a walk this morning.",
    "How has your week been?",
]


class SelfTeach:
    def __init__(self, model_name: str, m: int = 16, four_bit: bool = True, model=None, tok=None,
                 persist_path: str | None = None):
        self.lock = threading.Lock()
        self.m = m
        if model is not None and tok is not None:
            # share an already-loaded backbone (e.g. the unified clozn server's Qwen-7B) -- one model,
            # both the concept readout AND the memory. The model is frozen + quantized either way.
            self.tok, self.model = tok, model
        else:
            path = resolve_model_path(model_name)
            print(f"loading {model_name} ({'4-bit nf4' if four_bit else 'bf16'}) on {DEV} from {path} ...", flush=True)
            self.tok = AutoTokenizer.from_pretrained(path)
            if four_bit and DEV == "cuda":
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
                self.model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb,
                                                                  device_map={"": 0})
            else:
                self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.emb = self.model.get_input_embeddings()
        self.H = self.model.config.hidden_size
        self.cdtype = next(self.emb.parameters()).dtype     # embedding/compute dtype (bf16)
        self.eos = self.tok.eos_token_id
        # state
        self.prefix: nn.Parameter | None = None             # the consolidated memory (None until first teach)
        self.history: list[dict] = []                       # [{role, content}, ...]
        self.examples: list[tuple] = []                     # accumulated (prompt_ids, target_ids) for TTT
        self.rules: list[str] = []                          # rules consolidated so far (bookkeeping)
        # contextual gating: a domain anchor + in-domain/neutral cosine bands. A learned preference fires
        # only when a prompt is relevant to its domain (fixes the always-on over-bleed).
        self.anchor: torch.Tensor | None = None
        self.sim_in = 1.0
        self.sim_neutral = 0.0
        self.persist = persist_path                         # if set, the memory survives restarts (auto save/load)
        print(f"  ready. hidden={self.H} dtype={self.cdtype} eos={self.eos}", flush=True)
        if self.persist and self.load(self.persist):
            print(f"  restored {len(self.rules)} memory card(s) from {self.persist}", flush=True)

    # ---- low-level: chat ids, embed, generate with optional prefix ----------------------------
    def _chat_ids(self, messages: list[dict]) -> list[int]:
        return self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)

    def _embed(self, ids: list[int]) -> torch.Tensor:
        return self.emb(torch.tensor([ids], device=DEV))    # [1, L, H]

    @torch.no_grad()
    def _domain_vec(self, text: str) -> torch.Tensor:
        """Unit sentence-rep: mean-pooled final hidden state (no prefix). Measures how relevant a prompt
        is to a learned rule's domain, for contextual gating."""
        ids = self._chat_ids([{"role": "user", "content": text}])
        h = self.model(inputs_embeds=self._embed(ids), output_hidden_states=True).hidden_states[-1][0]
        v = h.mean(0).float()
        return v / (v.norm() + 1e-8)

    def _gate(self, prompt: str) -> float:
        """Relevance gate in [0,1]: cosine(prompt, domain anchor) rescaled so an in-domain prompt -> ~1
        and a neutral prompt -> ~0. Returns 1.0 before any anchor is set (gating off)."""
        if self.anchor is None:
            return 1.0
        cos = float(self._domain_vec(prompt) @ self.anchor)
        g = (cos - self.sim_neutral) / (self.sim_in - self.sim_neutral + 1e-6)
        return max(0.0, min(1.0, g))

    @torch.no_grad()
    def _generate(self, messages: list[dict], use_prefix: bool, max_new=200, sample=True, gate="auto") -> str:
        e = self._embed(self._chat_ids(messages))           # [1, L, H]
        if use_prefix and self.prefix is not None:
            # contextual gate g in [0,1] scales the injection: "auto" => by prompt relevance, else a float.
            if gate == "auto":
                last_user = next((mm["content"] for mm in reversed(messages) if mm["role"] == "user"), "")
                g = self._gate(last_user)
            else:
                g = float(gate)
            pre = (g * self.prefix.detach()).to(e.dtype)[None]   # [1, m, H], scaled by relevance
            e = torch.cat([pre, e], 1)
        att = torch.ones(e.shape[:2], device=DEV, dtype=torch.long)
        out = self.model.generate(inputs_embeds=e, attention_mask=att, max_new_tokens=max_new,
                                  do_sample=sample, temperature=0.7, top_p=0.9,
                                  pad_token_id=self.eos or 0)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    # ---- /say : one conversational turn (current prefix active) --------------------------------
    def say(self, message: str, max_new=220, strength=1.0) -> str:
        with self.lock:
            self.history.append({"role": "user", "content": message})
            # Apply the consolidated prefix at full strength by default. The old "auto" cosine gate is
            # unreliable (mean-pooled hidden states are too anisotropic -- sim_in 0.968 vs sim_neutral 0.956,
            # so it scored every prompt ~0 and zeroed the trait). The gentle, norm-capped prefix self-gates
            # well on its own (clean on mortgage/flat-tire/WWI, only soft analogy bleed elsewhere), so we
            # default to full and expose `strength` as the user knob. (Future: a KL-effect relevance gate.)
            reply = self._generate(self.history, use_prefix=True, max_new=max_new, sample=True, gate=strength)
            self.history.append({"role": "assistant", "content": reply})
            return reply

    # ---- rule extraction: the model reads the convo and names the user's preferences -----------
    def _extract_rules(self) -> list[str]:
        convo = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.history)
        ask = ("Below is a conversation between a user and you (the assistant). From HOW the user talks -- "
               "their interests, what they light up about, any emotional context or sensitivities, and how "
               "they like you to respond -- write a short list of things you should REMEMBER about THIS user "
               "to serve them better next time. They will usually NOT state these as rules; infer them. Each "
               "item one short line (an interest, a sensitivity, or a response adjustment), no numbering. "
               "If there is nothing, write NONE.\n\n" + convo)
        out = self._generate([{"role": "user", "content": ask}], use_prefix=False, max_new=200, sample=False)
        rules = []
        for line in out.splitlines():
            t = line.strip().lstrip("-*0123456789. ").strip()
            if t and t.upper() != "NONE" and len(t) > 3:
                rules.append(t)
        return rules[:8]

    # ---- sequence-level TTT loss: prefix + plain prompt must reproduce the rule-following target -
    def _seq_loss(self, prompt_ids: list[int], target_ids: list[int]) -> torch.Tensor:
        e_p = self._embed(prompt_ids)                       # [1, Lp, H]
        e_t = self._embed(target_ids)                       # [1, Lt, H]
        pre = self.prefix.to(e_p.dtype)[None]               # [1, m, H]  (trainable)
        full = torch.cat([pre, e_p, e_t], 1)
        att = torch.ones(full.shape[:2], device=DEV, dtype=torch.long)
        logits = self.model(inputs_embeds=full, attention_mask=att).logits[0]   # [m+Lp+Lt, V]
        start = self.m + len(prompt_ids) - 1                # position predicting target_ids[0]
        pred = logits[start:start + len(target_ids)]
        return F.cross_entropy(pred.float(), torch.tensor(target_ids, device=DEV))

    # ---- /consolidate : extract rules, build targets, distill into the prefix (TTT) ------------
    def consolidate(self, rules: list[str] | None = None, steps=120, lr=0.012, n_probe=8,
                    max_norm=14.0) -> dict:
        with self.lock:
            t0 = time.time()
            rules = rules if rules else self._extract_rules()
            if not rules:
                return {"ok": False, "reason": "no preferences found in the conversation yet"}
            sys_rule = ("You are a helpful assistant talking with a returning user. Here is what you know "
                        "about them; use it naturally to tailor how you respond:\n"
                        + "\n".join("- " + r for r in rules))
            # recent real user turns + the fixed varied probes -> the prompts we teach the rule on
            recent = [m["content"] for m in self.history if m["role"] == "user"][-3:]
            probes = (recent + PROBE_PROMPTS)[:n_probe + 3]
            new_examples = []
            for pr in probes:
                # target = a rule-following answer (rules stated in-context, NO prefix)
                tgt = self._generate([{"role": "system", "content": sys_rule}, {"role": "user", "content": pr}],
                                     use_prefix=False, max_new=64, sample=False)
                if not tgt.strip():
                    continue
                # we TTT the prefix to produce that target's rule-bearing OPENING from the PLAIN prompt
                # (no rules in context). A short opening is fittable by a 16-vector prefix; forcing the
                # full free-form response is not, and makes the optimizer crank the prefix into a
                # degenerate attractor (the divergence we saw: loss UP, norm 43.7, "recipe recipe" mush).
                plain_ids = self._chat_ids([{"role": "user", "content": pr}])
                tgt_ids = self.tok.encode(tgt, add_special_tokens=False)
                new_examples.append((plain_ids, tgt_ids[:32]))
            self.examples.extend(new_examples)
            # init the prefix on first consolidation; keep + grow it after
            if self.prefix is None:
                init = 0.02 * torch.randn(self.m, self.H, device=DEV, dtype=torch.float32)
                self.prefix = nn.Parameter(init)
            opt = torch.optim.Adam([self.prefix], lr=lr, weight_decay=2e-3)

            def avg_loss():
                with torch.no_grad():
                    return sum(self._seq_loss(p, t).item() for p, t in self.examples) / len(self.examples)

            # STABLE TTT. The objective (a 16-vector prefix reproducing several rule-following openings) is
            # only partly satisfiable, so naive Adam over-cranks the prefix into corruption. Four guards:
            # low lr, grad-clip, a HARD norm cap (renormalize so it can never explode), and early-stopping
            # that keeps the BEST prefix -- so we never ship the diverged final one.
            start = best = avg_loss()
            best_prefix = self.prefix.detach().clone()
            bad, patience, used = 0, 8, 0
            for step in range(steps):
                used = step + 1
                opt.zero_grad()
                for (p, t) in self.examples:                 # grad-accumulate over all examples (old+new)
                    (self._seq_loss(p, t) / len(self.examples)).backward()
                torch.nn.utils.clip_grad_norm_([self.prefix], 2.0)
                opt.step()
                with torch.no_grad():                        # hard cap: the prefix can NEVER corrupt generation
                    n = float(self.prefix.norm())
                    if n > max_norm:
                        self.prefix.mul_(max_norm / n)
                if step % 2 == 1:                            # evaluate every other step: keep-best + early-stop
                    cur = avg_loss()
                    if cur < best - 1e-3:
                        best, bad = cur, 0
                        best_prefix = self.prefix.detach().clone()
                    else:
                        bad += 1
                        if bad >= patience:
                            break
            with torch.no_grad():
                self.prefix.copy_(best_prefix)               # restore the best, never the diverged last
            self.rules = rules
            # contextual-gating anchor: the rule's DOMAIN = mean rep of its probe prompts, with the
            # in-domain and neutral cosine bands so _gate maps a new prompt's relevance to [0,1].
            with torch.no_grad():
                av = torch.stack([self._domain_vec(p) for p in probes])
                self.anchor = av.mean(0)
                self.anchor = self.anchor / (self.anchor.norm() + 1e-8)
                self.sim_in = float((av @ self.anchor).mean())
                self.sim_neutral = float((torch.stack([self._domain_vec(p) for p in NEUTRAL_REFS])
                                          @ self.anchor).mean())
            if self.persist:                                # auto-save so the new memory survives a restart
                self.save()
            return {"ok": True, "rules": rules, "n_examples": len(self.examples),
                    "start_loss": round(start, 3), "final_loss": round(best, 3), "steps_used": used,
                    "prefix_norm": round(float(self.prefix.detach().norm()), 1),
                    "sim_in": round(self.sim_in, 3), "sim_neutral": round(self.sim_neutral, 3),
                    "seconds": round(time.time() - t0, 1)}

    # ---- /whatlearned : the legibility test -- prefixed model, conversation NOT in context -----
    def what_learned(self) -> str:
        with self.lock:
            if self.prefix is None:
                return "(nothing consolidated yet -- call /consolidate first)"
            ask = ("What have you picked up about me so far -- my interests, anything I seem to care about, "
                   "and how I like you to respond? List what you know, one item per line.")
            return self._generate([{"role": "user", "content": ask}], use_prefix=True, max_new=200,
                                  sample=False, gate=1.0)

    # ---- /check : baseline vs UNGATED prefix vs GATED prefix (+ the gate value) on the same probe ---
    def check(self, prompt: str, max_new=200) -> dict:
        with self.lock:
            msgs = [{"role": "user", "content": prompt}]
            base = self._generate(msgs, use_prefix=False, max_new=max_new, sample=False)
            if self.prefix is None:
                return {"prompt": prompt, "gate": None, "baseline": base,
                        "ungated": "(no prefix)", "gated": "(no prefix)"}
            g = round(self._gate(prompt), 3)                # how relevant this prompt is to the rule's domain
            ungated = self._generate(msgs, use_prefix=True, max_new=max_new, sample=False, gate=1.0)
            gated = self._generate(msgs, use_prefix=True, max_new=max_new, sample=False, gate="auto")
            return {"prompt": prompt, "gate": g, "baseline": base, "ungated": ungated, "gated": gated}

    # ---- /trace : per-token causal attribution of the prefix -- WHERE is the rule firing? -------
    # For each token of the prefixed reply, KL(next-token dist WITH prefix || WITHOUT prefix), teacher-
    # forced on the same reply. High KL = the learned rule is actively shaping THAT token. This is causal
    # (it isolates the prefix's effect), not a correlational feature read -- the honest way to answer
    # "is it using the rule right now", given the rule lives in a prefix we own.
    @torch.no_grad()
    def trace(self, prompt: str, max_new=80) -> dict:
        with self.lock:
            msgs = [{"role": "user", "content": prompt}]
            ids = self._chat_ids(msgs)
            e = self._embed(ids)
            use_pref = self.prefix is not None
            e_gen = torch.cat([self.prefix.detach().to(e.dtype)[None], e], 1) if use_pref else e
            att = torch.ones(e_gen.shape[:2], device=DEV, dtype=torch.long)
            gen = self.model.generate(inputs_embeds=e_gen, attention_mask=att, max_new_tokens=max_new,
                                      do_sample=False, pad_token_id=self.eos or 0)
            reply_ids = [t for t in gen[0].tolist() if t != self.eos]
            reply = self.tok.decode(reply_ids, skip_special_tokens=True).strip()
            if not use_pref or not reply_ids:
                return {"prompt": prompt, "reply": reply, "tokens": [], "max_kl": 0.0}
            Lp, Lr, m = len(ids), len(reply_ids), self.m
            e_p, e_r = self._embed(ids), self._embed(reply_ids)
            pre = self.prefix.detach().to(e_p.dtype)[None]
            lg_w = self.model(inputs_embeds=torch.cat([pre, e_p, e_r], 1)).logits[0]   # with prefix
            lg_n = self.model(inputs_embeds=torch.cat([e_p, e_r], 1)).logits[0]        # without prefix
            toks = []
            for i in range(Lr):
                pw = torch.log_softmax(lg_w[m + Lp + i - 1].float(), -1)
                pn = torch.log_softmax(lg_n[Lp + i - 1].float(), -1)
                kl = float(torch.sum(pw.exp() * (pw - pn)))                            # KL(with || without)
                toks.append({"piece": self.tok.decode([reply_ids[i]]), "kl": round(kl, 3)})
            return {"prompt": prompt, "reply": reply, "tokens": toks,
                    "max_kl": round(max(t["kl"] for t in toks), 3),
                    "mean_kl": round(sum(t["kl"] for t in toks) / len(toks), 3)}

    # ---- persistence: the consolidated memory (prefix + cards) survives restarts ------------------
    def save(self, path: str | None = None) -> bool:
        path = path or self.persist
        if not path or self.prefix is None:
            return False
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"m": self.m, "prefix": self.prefix.detach().cpu(), "rules": self.rules,
                    "examples": self.examples,
                    "anchor": None if self.anchor is None else self.anchor.detach().cpu(),
                    "sim_in": self.sim_in, "sim_neutral": self.sim_neutral}, path)
        return True

    def load(self, path: str | None = None) -> bool:
        path = path or self.persist
        if not path or not os.path.isfile(path):
            return False
        try:
            d = torch.load(path, map_location="cpu")
        except Exception:
            return False
        if d.get("m") != self.m:                            # prefix length mismatch -> ignore the stale file
            return False
        self.prefix = nn.Parameter(d["prefix"].to(DEV).float())
        self.rules = d.get("rules", [])
        self.examples = d.get("examples", [])
        self.anchor = None if d.get("anchor") is None else d["anchor"].to(DEV)
        self.sim_in = d.get("sim_in", 1.0)
        self.sim_neutral = d.get("sim_neutral", 0.0)
        return True

    def reset(self, keep_prefix=False):
        with self.lock:
            self.history = []
            if not keep_prefix:
                self.prefix = None
                self.examples = []
                self.rules = []
                if self.persist and os.path.isfile(self.persist):
                    try:
                        os.remove(self.persist)                 # full reset wipes the persisted memory too
                    except OSError:
                        pass
            return {"ok": True, "kept_prefix": keep_prefix}

    def state(self) -> dict:
        return {"turns": len(self.history), "has_prefix": self.prefix is not None,
                "n_examples": len(self.examples), "rules": self.rules}


def make_handler(app: SelfTeach):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def do_GET(self):
            if self.path == "/state":
                self._send(200, app.state())
            else:
                self._send(404, {"error": "GET " + self.path})

        def do_POST(self):
            try:
                b = self._body()
                if self.path == "/say":
                    self._send(200, {"reply": app.say(b["message"], b.get("max_new", 220))})
                elif self.path == "/consolidate":
                    self._send(200, app.consolidate(b.get("rules"), b.get("steps", 120), b.get("lr", 0.012),
                                                    b.get("n_probe", 8), b.get("max_norm", 14.0)))
                elif self.path == "/whatlearned":
                    self._send(200, {"report": app.what_learned()})
                elif self.path == "/check":
                    self._send(200, app.check(b["prompt"], b.get("max_new", 200)))
                elif self.path == "/trace":
                    self._send(200, app.trace(b["prompt"], b.get("max_new", 80)))
                elif self.path == "/reset":
                    self._send(200, app.reset(b.get("keep_prefix", False)))
                else:
                    self._send(404, {"error": "POST " + self.path})
            except Exception as e:
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--port", type=int, default=8079)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--m", type=int, default=16, help="soft-prefix length")
    ap.add_argument("--bf16", action="store_true", help="load bf16 instead of 4-bit (small models)")
    args = ap.parse_args()
    app = SelfTeach(args.model, m=args.m, four_bit=not args.bf16)
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(app))
    print(f"\n  SELF-TEACH server -> http://{args.host}:{args.port}", flush=True)
    print("  /say /consolidate /whatlearned /check /reset  (GET /state)\n", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
