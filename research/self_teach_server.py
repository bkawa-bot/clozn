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
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _topic_gate():
    """The shared TopicGate (a small sentence-embedder that scores prompt<->rule topic relevance), or
    None if topic_gate isn't importable. Guarded so a missing module can never break SelfTeach -- callers
    fall back to no-gating (g == 1.0), i.e. the old always-on behavior. The 80MB embedder loads at most
    once per process (get_gate is a singleton) and only on first real use."""
    try:
        from topic_gate import get_gate
        return get_gate()
    except Exception:
        return None


# ------- per-token trace capture (Studio Run Inspector, issue B3) --------------------------------
# A PURE-passthrough LogitsProcessor: at each decoding step it reads the model's score row, records the
# top-k tokens the model weighed there (piece + softmax prob), and returns `scores` COMPLETELY UNCHANGED.
# Because it never mutates the logits, attaching it cannot alter the generated text or the sampled/greedy
# token sequence one bit -- it only observes. transformers appends the sampling warpers (temperature/top_p)
# AFTER any custom processor, so the probs we record are the model's own distribution (post repetition-
# penalty, pre temperature/top_p) -- the honest "how sure was the model" signal, independent of sampling.
#
# We record the top-k row (not just the argmax) so the trace can later be aligned to the token that was
# ACTUALLY committed: under sampling the emitted token often isn't the argmax, and its confidence is its
# probability, so `steps_from_records` looks the emitted id up in the recorded row. Compact by design
# (top-k per step, no full-vocab retention), and every use site wraps it in try/except -> any failure
# degrades to an empty trace and never touches generation.
class RecordingLogitsProcessor(LogitsProcessor):
    """Observe-only: record the top-k (piece-id, prob) per decoding step; return scores untouched."""

    def __init__(self, topk: int = 6):
        self.topk = int(topk)
        self.records: list[dict] = []          # one per step: {"ids":[...], "probs":[...]}

    def __call__(self, input_ids, scores):     # noqa: D401  (transformers LogitsProcessor contract)
        try:
            row = scores[0].detach().float()                       # [V] -- batch size 1 on this path
            probs = torch.softmax(row, dim=-1)
            k = min(self.topk, probs.shape[-1])
            top = torch.topk(probs, k)
            self.records.append({"ids": [int(i) for i in top.indices.tolist()],
                                 "probs": [float(p) for p in top.values.tolist()]})
        except Exception:
            # never let observation perturb generation; a missing record just yields a shorter trace
            self.records.append({"ids": [], "probs": []})
        return scores                                              # UNCHANGED -- pure pass-through


def steps_from_records(records: list[dict], gen_ids: list[int], tok) -> list[dict]:
    """Align recorded top-k rows to the tokens ACTUALLY emitted -> runlog-shaped steps.

    `records[i]` is the top-k the model weighed at step i; `gen_ids[i]` is the token it then committed
    (the sampled/greedy id, decoded for display). For each committed token we report:
      confidence = the committed token's own probability (looked up in that step's top-k row; 0.0 if it
                   fell outside the recorded top-k, e.g. a deep sampling-tail pick), and
      alternatives = the other top-k tokens (piece + prob), the chosen token excluded, capped at 3.
    Returned shape matches what runlog.steps_to_trace consumes: {"piece","conf","alts":[{"piece","prob"}]}.
    Pure aside from tok.decode; defensive throughout (any per-step hiccup -> that step is skipped)."""
    steps: list[dict] = []
    n = min(len(records), len(gen_ids))                            # align 1:1; ignore any ragged tail
    for i in range(n):
        try:
            rec = records[i] or {}
            ids = rec.get("ids", []) or []
            probs = rec.get("probs", []) or []
            tid = int(gen_ids[i])
            prob_by_id = {int(a): float(b) for a, b in zip(ids, probs)}
            conf = float(prob_by_id.get(tid, 0.0))                 # committed token's own probability
            alts = [{"piece": tok.decode([a]), "prob": round(float(b), 4)}
                    for a, b in zip(ids, probs) if int(a) != tid][:3]
            steps.append({"piece": tok.decode([tid]), "conf": round(conf, 4), "alts": alts})
        except Exception:
            continue
    return steps


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


def rule_set_changed(trained_on: list[str] | None, incoming: list[str] | None) -> bool:
    """Did the ACTIVE rule SET change vs the set the current prefix was actually trained on?

    Pure (no torch, no model) so it's unit-testable in isolation. Order- and duplicate-insensitive:
    what matters is WHICH traits are active, not their sequence. When this returns True the caller must
    REINIT the prefix and train from scratch so every trait starts on equal footing (a warm-start would
    let an entrenched trait dominate a freshly-added one -- the "baking drowns out dogs" bug). When it
    returns False the set is identical (e.g. a strength/steps tweak) and a warm-start is correct.
    """
    return set(trained_on or []) != set(incoming or [])


def fair_steps(base_steps: int, n_rules: int) -> int:
    """Give N traits a fair training budget when we retrain from scratch on the full active set.

    A single soft prefix has to fit every active trait's rule-following opening at once; more traits =
    more to fit, so we scale the step budget modestly with trait count (still bounded so consolidation
    stays quick). Small documented heuristic, deliberately not tuned to death: base for 1 trait, +50%
    of base per additional trait, capped at 3x base. Warm-start reruns (identical set) keep base_steps.
    """
    n = max(1, int(n_rules))
    scaled = int(round(base_steps * (1.0 + 0.5 * (n - 1))))
    return min(scaled, base_steps * 3)


class SelfTeach:
    def __init__(self, model_name: str, m: int = 16, four_bit: bool = True, model=None, tok=None,
                 persist_path: str | None = None):
        self.lock = threading.Lock()
        self.m = m
        self.memory_strength = 0.35         # dialed down from 1.0: at full strength a topical memory over-bleeds
                                            # into every reply in long chats (baking invaded a cover letter in
                                            # multi-turn testing); ~0.3 keeps it useful on-topic. User dial
                                            # raises/lowers it (0 = off, >1 = bites harder). A real relevance
                                            # gate is the full fix -- strength is only a partial mitigation.
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
        self._trained_rules: list[str] = []                 # the set the CURRENT prefix was actually trained on
                                                            # (own signal: self.rules is mutated by the caller
                                                            #  BEFORE consolidate() runs, so it can't tell us
                                                            #  whether the set changed -- this can)
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
        """TOPIC-RELEVANCE gate in [0,1]: how strongly the memory should fire for THIS prompt.

        Delegates to a small sentence-embedder (topic_gate.TopicGate) that scores the prompt against the
        active rule TEXTS (self.rules) and returns the relevance to the best-matching rule, soft-thresholded
        to [0,1]. On-topic prompts -> ~1, off-topic -> ~0. This REPLACES the old hidden-state cosine gate
        (self.anchor / sim_in / sim_neutral), which was unreliable because mean-pooled 7B hidden states are
        too anisotropic to separate domains. Those fields remain on the object for save/load compat but are
        no longer consulted here.

        Graceful default (the safety contract): returns 1.0 -- i.e. NO gating, the always-on baseline --
        whenever the embedder is unavailable OR there are no active rules. So a machine without
        sentence-transformers, or a memory with no rules, behaves exactly as before (no regression)."""
        gate = _topic_gate()
        if gate is None:
            return 1.0
        try:
            return float(gate.scalar(prompt, list(self.rules)))
        except Exception:
            return 1.0                                       # any gate hiccup -> fall back to no-gating

    @torch.no_grad()
    def _generate(self, messages: list[dict], use_prefix: bool, max_new=200, sample=True, gate="auto",
                  trace_out: list | None = None, apply_gate: bool = True) -> str:
        """Generate one reply. If `trace_out` (a list) is passed, ALSO fill it with per-token trace steps
        (piece + confidence + alternatives) for the Run Inspector -- captured via a pure pass-through
        LogitsProcessor, so the returned text is byte-identical whether or not a trace is requested. Any
        capture failure leaves trace_out empty and never affects the reply. The return type is always str."""
        e = self._embed(self._chat_ids(messages))           # [1, L, H]
        if use_prefix and self.prefix is not None:
            # Injection scale = BASE strength x TOPIC RELEVANCE. The base is memory_strength when gate=="auto"
            # (the /say + chat paths), else the caller's float (treated as an explicit base strength). Either
            # way we multiply by the topic-relevance gate rel in [0,1] (self._gate on the last user turn), so
            # the memory fires on-topic and turns OFF off-topic -- the fix for the always-on over-bleed. A
            # base of 0 (memory dial off) stays 0; a missing embedder makes rel==1 -> the base is used as-is.
            base = self.memory_strength if gate == "auto" else float(gate)
            last_user = next((mm["content"] for mm in reversed(messages) if mm["role"] == "user"), "")
            rel = self._gate(last_user) if apply_gate else 1.0   # diagnostics (whatlearned/check ungated) bypass
            g = base * rel
            pre = (g * self.prefix.detach()).to(e.dtype)[None]   # [1, m, H], scaled by base x relevance
            e = torch.cat([pre, e], 1)
        att = torch.ones(e.shape[:2], device=DEV, dtype=torch.long)
        gen_kw = dict(inputs_embeds=e, attention_mask=att, max_new_tokens=max_new,
                      do_sample=sample, temperature=0.7, top_p=0.9,
                      repetition_penalty=1.3, no_repeat_ngram_size=3,   # trim steering loops
                      pad_token_id=self.eos or 0)
        recorder = None
        if trace_out is not None:                           # opt-in trace: attach the observe-only recorder
            try:
                from transformers import LogitsProcessorList
                recorder = RecordingLogitsProcessor()
                gen_kw["logits_processor"] = LogitsProcessorList([recorder])   # append; nothing else here to keep
            except Exception:
                recorder = None
        out = self.model.generate(**gen_kw)
        # With inputs_embeds, out[0] is the GENERATED tokens only (no prompt echo) -> aligns 1:1 with the
        # recorder's per-step rows. Build the trace defensively; on any failure trace_out stays empty.
        if recorder is not None:
            try:
                gen_ids = [int(t) for t in out[0].tolist()]
                while gen_ids and gen_ids[-1] == (self.eos or -1):   # drop trailing EOS from the visible trace
                    gen_ids.pop()
                trace_out.extend(steps_from_records(recorder.records, gen_ids, self.tok))
            except Exception:
                pass
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    # ---- /say : one conversational turn (current prefix active) --------------------------------
    def say(self, message: str, max_new=220, strength=None, trace_out: list | None = None) -> str:
        with self.lock:
            self.history.append({"role": "user", "content": message})
            # Apply the consolidated prefix scaled by memory_strength x TOPIC RELEVANCE. gate="auto" makes
            # _generate use memory_strength (the user dial; 0 = off, 1 = as trained) as the base and multiply
            # by the topic-relevance gate, so the memory fires on-topic and stays quiet off-topic -- the fix
            # for the always-on over-bleed (baking invading a cover letter). An explicit `strength` override
            # is passed as the base float and is STILL topic-gated (the new _generate contract).
            # trace_out (optional): filled with the per-token trace for the Run Inspector; reply is unchanged.
            gate = "auto" if strength is None else float(strength)
            reply = self._generate(self.history, use_prefix=True, max_new=max_new, sample=True, gate=gate,
                                   trace_out=trace_out)
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

    # ---- E2: propose ONE durable user preference from a past run, on demand (a PENDING card) -------
    # The Studio calls this when the reviewer clicks "propose a memory" on a captured run. The model reads
    # that run's conversation and names a single reusable preference (how they like answers, or a stable
    # interest) as a short THIRD-PERSON memory -- or nothing, if the run holds no durable signal. This is a
    # PROPOSAL only: the caller stores it as a pending card (never approves, never retrains).
    #
    # CRITICAL -- this read must be CLEAN. We generate on the RAW frozen model+tokenizer (self.model.generate
    # on plain token ids), NOT via self._generate / the prefix path: the consolidated memory prefix would
    # color what the model "sees" and bias the extraction toward the already-known traits. (Tone steering is
    # neutralized by the caller in clozn_server.py before this runs.) Defensive throughout: any failure -> None.
    @torch.no_grad()
    def propose_memory(self, messages: list[dict], response: str | None = None,
                       max_new: int = 48) -> str | None:
        try:
            convo_msgs = list(messages or [])
            if response:                                    # fold the assistant's reply in as the last turn
                if not (convo_msgs and convo_msgs[-1].get("role") == "assistant"):
                    convo_msgs = convo_msgs + [{"role": "assistant", "content": response}]
            convo = "\n".join(f"{m.get('role', '').upper()}: {m.get('content', '')}"
                              for m in convo_msgs if m.get("content"))
            if not convo.strip():
                return None
            ask = ("Read this conversation between a user and an assistant. Identify ONE durable, reusable "
                   "preference the USER has -- either how they want answers (tone, length, level of detail) "
                   "or a stable interest of theirs -- that would help tailor future replies. Write it as a "
                   "short THIRD-PERSON note about the user, e.g. \"Prefers concise, technical answers\" or "
                   "\"Is interested in baking\". One line, under 12 words, no quotes, no preamble. If there "
                   "is no clear durable preference, reply with exactly NONE.\n\n"
                   "Conversation:\n" + convo + "\n\nDurable user preference:")
            # RAW model call -- no prefix, greedy, short. (Not self._generate: that path injects the memory
            # prefix, which must NOT taint the extraction.)
            ids = self.tok.apply_chat_template([{"role": "user", "content": ask}],
                                               tokenize=True, add_generation_prompt=True)
            e = self.emb(torch.tensor([ids], device=DEV))
            att = torch.ones(e.shape[:2], device=DEV, dtype=torch.long)
            out = self.model.generate(inputs_embeds=e, attention_mask=att, max_new_tokens=max_new,
                                      do_sample=False, pad_token_id=self.eos or 0)
            text = self.tok.decode(out[0], skip_special_tokens=True)
            return self._clean_proposal(text)
        except Exception:
            return None

    @staticmethod
    def _clean_proposal(text: str) -> str | None:
        """Sanitize the model's raw extraction into a short third-person memory, or None.

        None when: empty, a bare NONE/none, a refusal, or (after trimming) longer than ~120 chars -- a long
        answer means the model didn't commit to a single crisp preference. Strips surrounding quotes and a
        leading label (e.g. 'Durable user preference:' / 'Memory:' / a bullet)."""
        t = (text or "").strip()
        if not t:
            return None
        t = t.splitlines()[0].strip()                       # first line only -- one preference
        t = t.lstrip("-*•0123456789.) ").strip()            # drop bullet / numbering
        low = t.lower()
        for label in ("durable user preference:", "user preference:", "preference:", "memory:", "note:"):
            if low.startswith(label):
                t = t[len(label):].strip()
                low = t.lower()
        if len(t) >= 2 and t[0] in "\"'“”" and t[-1] in "\"'“”":   # strip surrounding quotes
            t = t[1:-1].strip()
            low = t.lower()
        if not t or low in ("none", "n/a", "na", "no preference", "no durable preference"):
            return None
        if any(r in low for r in ("i cannot", "i can't", "i'm sorry", "i am sorry", "as an ai",
                                  "there is no", "no clear", "unable to")):
            return None
        if len(t) > 120:                                    # not a crisp single preference -> skip
            return None
        return t

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
                    max_norm=14.0, reinit_on_change: bool = True) -> dict:
        # NB: back-compat -- m.consolidate(rules) still works; reinit_on_change is an optional keyword.
        with self.lock:
            t0 = time.time()
            rules = rules if rules else self._extract_rules()
            if not rules:
                return {"ok": False, "reason": "no preferences found in the conversation yet"}
            # FAIRNESS: did the ACTIVE rule SET change vs the set the CURRENT prefix was trained on?
            # (Compare against self._trained_rules, NOT self.rules: the card wiring overwrites self.rules
            #  with the new set BEFORE calling us, so self.rules can't reveal the change; _trained_rules
            #  records what the prefix in hand actually embodies.) If the set changed we REINIT and train
            #  from scratch so a freshly-added trait starts on equal footing with an entrenched one --
            #  a warm-start let the established trait dominate and the new one never registered ("approve
            #  dogs next to trained baking -> zero dog expression"). Removing a trait would likewise leave
            #  its residue in a warm-started prefix. Warm-start is correct ONLY for the identical set
            #  (e.g. a strength/steps tweak), where we keep refining the existing prefix.
            changed = rule_set_changed(self._trained_rules, rules)
            reinit = changed and reinit_on_change
            sys_rule = ("You are a helpful assistant talking with a returning user. Here is what you know "
                        "about them; use it naturally to tailor how you respond:\n"
                        + "\n".join("- " + r for r in rules))
            # recent real user turns + the fixed varied probes -> the prompts we teach the rule on
            recent = [m["content"] for m in self.history if m["role"] == "user"][-3:]
            probes = (recent + PROBE_PROMPTS)[:n_probe + 3]
            new_examples = []
            for pr in probes:
                # target = a rule-following answer (ALL current rules stated in-context, NO prefix)
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
            if reinit:
                # Fresh start: drop the OLD prefix AND the stale accumulated examples (which include
                # targets for now-removed traits) so the retrain represents ONLY the active set. The new
                # targets above already reflect the full current rule list.
                self.prefix = None
                self.examples = list(new_examples)
            else:
                self.examples.extend(new_examples)
            # init the prefix on first consolidation OR after a set-change reinit; else keep + refine it
            if self.prefix is None:
                init = 0.02 * torch.randn(self.m, self.H, device=DEV, dtype=torch.float32)
                self.prefix = nn.Parameter(init)
            # Give N traits a fair budget when retraining from scratch (a single prefix must fit them all).
            # A warm-start rerun on the identical set keeps the plain step count.
            steps_target = fair_steps(steps, len(rules)) if reinit else steps
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
            for step in range(steps_target):
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
            self._trained_rules = list(rules)                # the set THIS prefix now embodies (fairness signal
                                                            #  for the next consolidate: warm-start iff unchanged)
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
                    "reinit": reinit, "set_changed": changed, "steps_target": steps_target,
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
                                  sample=False, gate=1.0, apply_gate=False)   # self-report shows the FULL memory

    # ---- /check : baseline vs UNGATED prefix vs GATED prefix (+ the gate value) on the same probe ---
    def check(self, prompt: str, max_new=200) -> dict:
        with self.lock:
            msgs = [{"role": "user", "content": prompt}]
            base = self._generate(msgs, use_prefix=False, max_new=max_new, sample=False)
            if self.prefix is None:
                return {"prompt": prompt, "gate": None, "baseline": base,
                        "ungated": "(no prefix)", "gated": "(no prefix)"}
            g = round(self._gate(prompt), 3)                # how relevant this prompt is to the rule's domain
            ungated = self._generate(msgs, use_prefix=True, max_new=max_new, sample=False, gate=1.0, apply_gate=False)
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
                    "trained_rules": self._trained_rules,   # the set this saved prefix embodies (fairness signal)
                    "examples": self.examples, "memory_strength": self.memory_strength,
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
        # a restored prefix WAS trained on its saved rules -> seed the fairness signal so a later
        # consolidate on the identical set warm-starts (old files predate the key: fall back to rules).
        self._trained_rules = list(d.get("trained_rules", self.rules))
        self.examples = d.get("examples", [])
        self.anchor = None if d.get("anchor") is None else d["anchor"].to(DEV)
        self.sim_in = d.get("sim_in", 1.0)
        self.sim_neutral = d.get("sim_neutral", 0.0)
        self.memory_strength = float(d.get("memory_strength", 1.0))
        return True

    def reset(self, keep_prefix=False):
        with self.lock:
            self.history = []
            if not keep_prefix:
                self.prefix = None
                self.examples = []
                self.rules = []
                self._trained_rules = []                     # no prefix -> next consolidate is a fresh init
                if self.persist and os.path.isfile(self.persist):
                    try:
                        os.remove(self.persist)                 # full reset wipes the persisted memory too
                    except OSError:
                        pass
            return {"ok": True, "kept_prefix": keep_prefix}

    def state(self) -> dict:
        return {"turns": len(self.history), "has_prefix": self.prefix is not None,
                "n_examples": len(self.examples), "rules": self.rules}

    # ---- card layer hook (Studio D2): legacy trait strings <-> reviewable memory cards ----------------
    def sync_cards(self) -> list[str]:
        """One-time bridge to the Studio card store: seed any legacy `self.rules` as ACTIVE cards (the
        prefix is already trained on them, so this does NOT retrain), then adopt the active-card texts as
        `self.rules`. The prefix/soft-state is untouched -- cards are only the metadata + review layer.
        Best-effort: if memory_cards isn't importable (standalone use), keeps the current rules."""
        try:
            import memory_cards
        except Exception:
            return list(self.rules)
        try:
            memory_cards.migrate_from_rules(list(self.rules or []))   # no-op once the store has cards
            self.rules = memory_cards.active_texts()
        except Exception:
            pass
        return list(self.rules)


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
