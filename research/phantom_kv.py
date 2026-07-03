"""phantom_kv.py -- TRAINING-FREE GISTING via a trainable synthetic KV cache.

THE IDEA. The studio's "prompt" memory mode prepends a compiled card block (compile_prompt_block over
the active trait cards) to every gated-in turn. That block is legible and edit-instant, but it costs
context tokens on EVERY call: ~60-70 tokens of system prefix that must be prefilled each turn. Can we
pay that cost ONCE, offline, and carry the memory as a handful of trainable synthetic KV entries that
live in the attention cache -- so at inference the memory occupies ZERO context tokens (no text in the
prompt) yet the model still "attends to" it?

MECHANISM (distinct from the gisting literature -- see below). We distil the REAL block's effect into k
trainable "phantom" past-key-values, one small tensor pair per layer:
    K_phantom[layer] : Parameter [1, n_kv_heads, k, head_dim]
    V_phantom[layer] : Parameter [1, n_kv_heads, k, head_dim]
These are prepended to the KV cache at generation time (cache_position offset by k, attention_mask
widened by k) so every real token attends to them exactly as if k invisible tokens preceded it -- but
there are NO such tokens in the input_ids, so context length is unchanged. We WARM-START each phantom
from the real block's own KV sampled at k evenly-spaced positions (a compression of the true cache),
then TRAIN it (Adam, frozen backbone, only the phantom params) so that a PLAIN prompt + phantom
reproduces the FULL-BLOCK teacher's next-token distribution on the answer spans of several training
prompts. Loss = KL(student_with_phantom || teacher_with_full_block), the causal "match the block's
effect" objective. Guards mirror SelfTeach's stable-TTT: low lr, grad-clip, keep-best/early-stop.

RELATION TO GISTING (Mu et al. 2023, "Learning to Compress Prompts with Gist Tokens"), AutoCompressor,
ICAE, prompt-compression/soft-prompt distillation: those methods FINE-TUNE THE MODEL (or an adapter /
the gist-token embeddings via a modified attention mask over the whole training corpus) so that special
gist tokens summarise ANY prompt. THIS DOES NOT TOUCH THE BACKBONE AT ALL and learns nothing general:
it over-fits k KV vectors to ONE fixed memory block, at test time, with the backbone frozen -- a
per-memory "cache gist" trained by distillation, not a trained compressor. Closest kin is prefix-tuning
/ soft-prompt (p-tuning), but those train INPUT-EMBEDDING vectors (which then get processed by every
layer's attention K/V projections); we train the POST-PROJECTION K and V directly, per layer, and
warm-start them from the real cache. Cost asymmetry is the point: the block reads O(block) tokens PER
CALL forever; the phantom costs GPU-seconds ONCE and 0 tokens/call thereafter (like the fused prefix,
but this warm-starts from -- and is distilled against -- the real block's actual cache).

============================ PRE-REGISTRATION (written BEFORE the run) ============================
Prior context that shapes these priors (from this repo's ledger, ../.claude memory):
  * "DON'T FUSE" -- proven 3+ ways: an in-context list beats the fused weight/prefix delta; fused
    soft-prefix memory INTERFERES at fact-load N>=64 (memory_scaling) and its self-report can invert
    (scale_pass_7b). A phantom KV is a fused representation, so I expect it to be LOSSY vs the real block.
  * Trait split matters: baking/space are CONCEPT-like (topical, legible content -> internalise well);
    concise is RULE-like (style/process -> historically the HARDER thing to pin as content, and the
    thing fused prefixes distort). I therefore expect phantom to recover CONCEPT expression better than
    the CONCISE rule.
  * DEGENERATION GAMES METRICS -- documented 5+ times in this repo (emoji spam scores "warm"; a
    degenerate prefix wins a style metric while emitting word-salad). So expression numbers WITHOUT a
    coherence eyeball are untrustworthy; I will read samples per arm and report coherence honestly.

EXPECTATIONS / FALSIFIABLE PREDICTIONS:
  E1. Ordering of trait expression: full-block ceiling >= phantom(k=16) >= phantom(k=8) >=
      phantom(k=4) > no-memory floor ~ RANDOM-phantom null. (Random phantom, untrained, same shapes:
      near the floor, possibly WORSE via noise injection -- it is the null that proves any phantom win
      is the TRAINING, not just "some vectors in the cache".)
  E2. KL-to-ceiling (student vs full-block teacher on held-out answer spans) DECREASES monotonically
      with k, and is far below the RANDOM-phantom null. I do NOT expect KL->0 even at k=16 (a fused
      k-vector cache cannot perfectly reproduce a ~60-token block's cache: don't-fuse).
  E3. The headline question -- does k in {8,16} recover >=80% of the ceiling's trait EXPRESSION at 0
      context tokens? PRIOR: partial. I predict CONCEPT traits (baking/space) clear 80% by k=16 but the
      CONCISE rule does NOT (fused reps distort process). If ALL three clear 80% I will have been too
      pessimistic and will say so; if NONE do, the mechanism is a null and I will say that louder.
  E4. Coherence: phantom replies stay coherent at small k (unlike the diverged soft-prefix in
      voice_middle) BECAUSE warm-start seeds from the real cache. RISK I am watching: a large or
      badly-scaled phantom pushes the model into repetition/degeneration and GAMES the keyword scorer
      (e.g. "baking baking bread bread"). Any expression win with incoherent samples is reported as a
      NULL, not a win.
  E5. Context tokens = 0 for every phantom arm BY CONSTRUCTION (the assertion is in the rig). Prefill
      saved = the block's token count (~60-70) per call. These are the mechanism's whole reason to exist;
      they are true regardless of whether expression survives.

WHAT WOULD KILL IT: if phantom expression ~ no-memory floor at all k (E1 fails), or only wins via
degenerate samples (E4), the mechanism is dead -- report it as cleanly as slotmem's wins.
==================================================================================================

Run (repo root, CUDA venv):
    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/phantom_kv.py \
        --model Qwen/Qwen2.5-1.5B-Instruct --ks 4,8,16 --steps 200 \
        --out research/runs/phantom_kv_qwen1p5b.json
    # smoke first: --ks 4 --steps 60 --probes 2 --smoke

Honesty: one family, one seed, greedy eval, tiny probe N. Caveats loud in the findings.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # WinError 1314 workaround on this PC

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.cache_utils import DynamicCache

# Reuse the studio's block compiler AND the self-audit scorers/probes -- same instrument, so the phantom
# is measured on the exact bar the real memory is (no bespoke, self-serving metric).
from memory_mode import compile_prompt_block
import self_audit_gap as gap

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# The 3 trait cards the handoff names: two CONCEPT-like, one RULE-like. Texts are the CARD texts (what a
# user would type), compiled by compile_prompt_block into the exact system block prompt-mode prepends.
CARDS = [
    {"name": "baking", "cls": "concept", "text": "The user is really into baking.",
     "desc": "bringing up baking, bread, or recipes",
     "kw": ["bak", "bread", "dough", "oven", "cake", "pastry", "flour", "cinnamon", "recipe",
            "knead", "yeast", "loaf", "muffin", "sourdough", "pie", "cookie"]},
    {"name": "space", "cls": "concept", "text": "The user is fascinated by space and astronomy.",
     "desc": "bringing up space, stars, or astronomy",
     "kw": ["space", "star", "planet", "galaxy", "astronom", "cosmos", "orbit", "telescope",
            "nebula", "universe", "comet", "lunar", "moon", "cosmic", "constellation"]},
    {"name": "concise", "cls": "rule", "text": "The user wants you to answer very concisely, in one short sentence.",
     "desc": "answering very concisely", "kw": None},   # scored by output length
]

# Training prompts to distil the block's effect on -- DISJOINT from gap.HELDOUT (no train/test leak).
# Varied so the phantom generalises past one turn. The teacher's answer-span next-token dists on THESE
# are the KL target.
TRAIN_PROMPTS = [
    "What should I read this weekend?",
    "I've got a stressful week ahead. Any thoughts?",
    "What should I cook for dinner tonight?",
    "Recommend something fun for tonight.",
    "Tell me something interesting.",
    "I'm feeling a bit bored this afternoon.",
    "What's a good way to spend a Sunday?",
    "Any ideas for a small creative project?",
]

CFG = {"answer_span": 40,   # teacher tokens per training prompt used as the KL target span
       "gen_max": 90}       # eval generation length (matches gap.gen)


# --------------------------------------------------------------------------------------------------
#  Model load (bf16 for 1.5B; nf4 only if a 7B name is passed -- the studio's config, like gap.run)
# --------------------------------------------------------------------------------------------------
def load_model(model_name: str):
    four_bit = "7b" in model_name.lower()
    path = gap.__dict__.get("resolve_model_path")  # not exported; resolve locally like self_teach_server
    local = os.path.join(os.path.expanduser("~"), "hf_models", model_name.split("/")[-1])
    src = local if os.path.isfile(os.path.join(local, "config.json")) else model_name
    print(f"[load] {model_name} ({'nf4' if four_bit else 'bf16'}, {DEV}) from {src} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(src)
    if four_bit and DEV == "cuda":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(src, quantization_config=bnb, device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16).to(DEV)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


# --------------------------------------------------------------------------------------------------
#  Chat-id / block helpers
# --------------------------------------------------------------------------------------------------
def chat_ids(tok, messages, add_gen=True):
    return tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_gen)


def block_text():
    """The compiled memory block -- byte-identical to what prompt mode prepends (compile_prompt_block)."""
    return compile_prompt_block([c["text"] for c in CARDS])


def block_prefix_ids(tok):
    """Token ids of the system block ALONE, rendered as it appears at the front of a real chat.

    We render [system-block, user=""] and [user=""] and diff, so we capture the block exactly in the
    positions it occupies in a real templated chat (role tags and all), independent of the user turn.
    Returns the list of ids that the block contributes as a prefix."""
    with_block = chat_ids(tok, [{"role": "system", "content": block_text()},
                                {"role": "user", "content": ""}], add_gen=True)
    without = chat_ids(tok, [{"role": "user", "content": ""}], add_gen=True)
    # common suffix = the user-turn scaffold; the block's ids are the with_block prefix that isn't shared
    i = 0
    while i < len(without) and i < len(with_block) and with_block[i] == without[i]:
        i += 1
    # from i, with_block has the block+system scaffolding then re-joins at the user turn; simplest robust
    # capture: the block ids are everything in with_block that precedes the shared user-turn tail.
    tail = len(without) - i
    return with_block[: len(with_block) - tail] if tail > 0 else with_block[:i]


# --------------------------------------------------------------------------------------------------
#  The FULL-BLOCK teacher cache: real past_key_values for the compiled block (the ceiling + warm-start
#  source + KL target generator).
# --------------------------------------------------------------------------------------------------
@torch.no_grad()
def teacher_block_cache(model, tok):
    """Prefill the compiled block and return (DynamicCache, block_len). This IS the full-block memory's
    KV -- generation with a fresh user turn on top of it == prompt-mode memory (the ceiling)."""
    ids = block_prefix_ids(tok)
    cache = DynamicCache()
    e = model.get_input_embeddings()(torch.tensor([ids], device=DEV))
    att = torch.ones(1, len(ids), device=DEV, dtype=torch.long)
    model(inputs_embeds=e, attention_mask=att, past_key_values=cache,
          cache_position=torch.arange(len(ids), device=DEV), use_cache=True)
    return cache, len(ids)


def cache_layers_kv(cache):
    """[(K,V), ...] per layer from a DynamicCache (new transformers .layers API; each K/V is
    [1, n_kv_heads, seq, head_dim])."""
    return [(L.keys, L.values) for L in cache.layers]


# --------------------------------------------------------------------------------------------------
#  Phantom KV: k trainable entries per layer, warm-started from the teacher cache at k evenly-spaced
#  positions (or random for the null).
# --------------------------------------------------------------------------------------------------
class PhantomKV:
    def __init__(self, model, k: int, teacher_cache=None, random_init=False, scale=1.0):
        self.k = int(k)
        cfg = model.config
        self.NL = cfg.num_hidden_layers
        self.NKV = cfg.num_key_value_heads
        self.HD = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.dtype = next(model.parameters()).dtype
        self.Ks: list[torch.nn.Parameter] = []
        self.Vs: list[torch.nn.Parameter] = []
        if random_init or teacher_cache is None:
            # NULL arm: random KV at the teacher cache's typical scale (per-layer std), NOT trained.
            ref = cache_layers_kv(teacher_cache) if teacher_cache is not None else None
            for li in range(self.NL):
                if ref is not None:
                    ks, vs = ref[li][0].float(), ref[li][1].float()
                    kstd, vstd = ks.std().item(), vs.std().item()
                    kmu, vmu = ks.mean().item(), vs.mean().item()
                else:
                    kstd = vstd = 0.1; kmu = vmu = 0.0
                K = kmu + kstd * torch.randn(1, self.NKV, self.k, self.HD, device=DEV)
                V = vmu + vstd * torch.randn(1, self.NKV, self.k, self.HD, device=DEV)
                self.Ks.append(torch.nn.Parameter((scale * K).float()))
                self.Vs.append(torch.nn.Parameter((scale * V).float()))
        else:
            # WARM-START: sample the real block cache at k evenly-spaced positions.
            ref = cache_layers_kv(teacher_cache)
            seq = ref[0][0].shape[2]
            if self.k >= seq:
                idx = list(range(seq)) + [seq - 1] * (self.k - seq)  # pad by repeating last if k>block
            else:
                # evenly spaced across [0, seq-1] inclusive
                idx = [round(j * (seq - 1) / (self.k - 1)) if self.k > 1 else seq - 1 for j in range(self.k)]
            idx_t = torch.tensor(idx, device=DEV)
            for li in range(self.NL):
                Kf = ref[li][0].index_select(2, idx_t).float().clone()   # [1,NKV,k,HD]
                Vf = ref[li][1].index_select(2, idx_t).float().clone()
                self.Ks.append(torch.nn.Parameter(Kf))
                self.Vs.append(torch.nn.Parameter(Vf))

    def params(self):
        return self.Ks + self.Vs

    def n_params(self):
        return sum(p.numel() for p in self.params())

    def build_cache(self) -> DynamicCache:
        """A fresh DynamicCache holding ONLY the phantom entries (differentiable w.r.t. the params).
        The model will .update() the real tokens' KV onto layers of this cache during forward."""
        c = DynamicCache()
        for li in range(self.NL):
            c.update(self.Ks[li].to(self.dtype), self.Vs[li].to(self.dtype), li)
        return c

    def norm(self):
        with torch.no_grad():
            kn = sum(float(p.norm()) for p in self.Ks) / self.NL
            vn = sum(float(p.norm()) for p in self.Vs) / self.NL
        return round(kn, 2), round(vn, 2)

    def state(self):
        return {"k": self.k, "n_params": self.n_params(),
                "shapes": f"{self.NL}L x 2 x [1,{self.NKV},{self.k},{self.HD}]"}


# --------------------------------------------------------------------------------------------------
#  Forward helpers: logits for a user prompt WITH a given phantom prefix (differentiable), and greedy
#  generation with a phantom prefix (for eyeball + scoring).
# --------------------------------------------------------------------------------------------------
def _prompt_logits_with_phantom(model, tok, prompt_ids, target_ids, phantom: PhantomKV | None):
    """logits over the target span, teacher-forced, with the phantom prepended in the KV cache.

    Returns [len(target_ids), V]. If phantom is None -> plain (no-memory). Differentiable through the
    phantom params. cache_position offsets by k so RoPE places the real tokens after the phantom."""
    emb = model.get_input_embeddings()
    full_ids = list(prompt_ids) + list(target_ids)
    e = emb(torch.tensor([full_ids], device=DEV))
    if phantom is not None:
        cache = phantom.build_cache()
        k = phantom.k
        att = torch.ones(1, k + len(full_ids), device=DEV, dtype=torch.long)
        cpos = torch.arange(k, k + len(full_ids), device=DEV)
        out = model(inputs_embeds=e, attention_mask=att, past_key_values=cache,
                    cache_position=cpos, use_cache=True)
    else:
        att = torch.ones(1, len(full_ids), device=DEV, dtype=torch.long)
        out = model(inputs_embeds=e, attention_mask=att, use_cache=False)
    logits = out.logits[0]                       # [L, V]
    start = len(prompt_ids) - 1                   # position predicting target_ids[0]
    return logits[start:start + len(target_ids)]


@torch.no_grad()
def _teacher_target_logits(model, tok, prompt_ids, target_ids, block_cache, block_len):
    """The FULL-BLOCK teacher's next-token logits over the target span (the KL target).

    We rebuild a fresh copy of the block cache each call (generation mutates the cache), run the prompt
    +target on top with cache_position offset by block_len."""
    # copy the block cache (deep) so the forward's .update doesn't grow the shared teacher cache
    cache = DynamicCache()
    for li, L in enumerate(block_cache.layers):
        cache.update(L.keys.clone(), L.values.clone(), li)
    emb = model.get_input_embeddings()
    full_ids = list(prompt_ids) + list(target_ids)
    e = emb(torch.tensor([full_ids], device=DEV))
    att = torch.ones(1, block_len + len(full_ids), device=DEV, dtype=torch.long)
    cpos = torch.arange(block_len, block_len + len(full_ids), device=DEV)
    out = model(inputs_embeds=e, attention_mask=att, past_key_values=cache,
                cache_position=cpos, use_cache=True)
    logits = out.logits[0]
    start = len(prompt_ids) - 1
    return logits[start:start + len(target_ids)]


@torch.no_grad()
def _teacher_greedy(model, tok, prompt_ids, block_cache, block_len, max_new):
    """Greedy continuation from [block-cache | prompt] -- the full-block ceiling's ACTUAL reply (also
    used to build the answer-span target ids for training)."""
    cache = DynamicCache()
    for li, L in enumerate(block_cache.layers):
        cache.update(L.keys.clone(), L.values.clone(), li)
    emb = model.get_input_embeddings()
    cur = list(prompt_ids)
    e = emb(torch.tensor([cur], device=DEV))
    att = torch.ones(1, block_len + len(cur), device=DEV, dtype=torch.long)
    cpos = torch.arange(block_len, block_len + len(cur), device=DEV)
    out = model(inputs_embeds=e, attention_mask=att, past_key_values=cache, cache_position=cpos, use_cache=True)
    gen = []
    eos = tok.eos_token_id
    pos = block_len + len(cur)
    for _ in range(max_new):
        nxt = int(out.logits[0, -1].argmax())
        if nxt == eos:
            break
        gen.append(nxt)
        e = emb(torch.tensor([[nxt]], device=DEV))
        att = torch.ones(1, pos + 1, device=DEV, dtype=torch.long)
        out = model(inputs_embeds=e, attention_mask=att, past_key_values=cache,
                    cache_position=torch.tensor([pos], device=DEV), use_cache=True)
        pos += 1
    return gen


@torch.no_grad()
def gen_with_phantom(model, tok, prompt, phantom: PhantomKV | None, max_new):
    """Greedy reply for a user PROMPT with the phantom prepended (or plain if None). Mirrors gap.gen's
    greedy/no-sample setting so scores are comparable. Returns decoded string."""
    msgs = [{"role": "user", "content": prompt}]
    prompt_ids = chat_ids(tok, msgs)
    emb = model.get_input_embeddings()
    eos = tok.eos_token_id
    if phantom is not None:
        cache = phantom.build_cache()
        k = phantom.k
    else:
        cache = DynamicCache(); k = 0
    cur = list(prompt_ids)
    e = emb(torch.tensor([cur], device=DEV))
    att = torch.ones(1, k + len(cur), device=DEV, dtype=torch.long)
    cpos = torch.arange(k, k + len(cur), device=DEV)
    out = model(inputs_embeds=e, attention_mask=att, past_key_values=cache, cache_position=cpos, use_cache=True)
    gen = []
    pos = k + len(cur)
    for _ in range(max_new):
        nxt = int(out.logits[0, -1].argmax())
        if nxt == eos:
            break
        gen.append(nxt)
        e = emb(torch.tensor([[nxt]], device=DEV))
        att = torch.ones(1, pos + 1, device=DEV, dtype=torch.long)
        out = model(inputs_embeds=e, attention_mask=att, past_key_values=cache,
                    cache_position=torch.tensor([pos], device=DEV), use_cache=True)
        pos += 1
    return tok.decode(gen, skip_special_tokens=True).strip()


# --------------------------------------------------------------------------------------------------
#  Training: distil the block's effect into the phantom (KL to the full-block teacher on answer spans)
# --------------------------------------------------------------------------------------------------
def train_phantom(model, tok, phantom: PhantomKV, train_examples, steps, lr, max_norm, log_prefix=""):
    """train_examples: [(prompt_ids, target_ids, teacher_logits[T,V]), ...]. Minimise mean
    KL(student_with_phantom || teacher). Frozen backbone; only phantom params train. Keep-best/early-stop
    and a hard L2-norm cap per phantom tensor (the SelfTeach stability recipe, adapted to KV)."""
    opt = torch.optim.Adam(phantom.params(), lr=lr, weight_decay=0.0)

    def kl_over(examples):
        tot = 0.0
        for (pid, tid, tl) in examples:
            sl = _prompt_logits_with_phantom(model, tok, pid, tid, phantom)   # [T,V]
            logp = F.log_softmax(sl.float(), -1)
            with torch.no_grad():
                tp = F.softmax(tl.float(), -1)
            tot += float(F.kl_div(logp, tp, reduction="batchmean"))
        return tot / len(examples)

    with torch.no_grad():
        start = best = kl_over(train_examples)
    best_state = [p.detach().clone() for p in phantom.params()]
    bad, patience, used = 0, 12, 0
    for step in range(steps):
        used = step + 1
        opt.zero_grad()
        loss_val = 0.0
        for (pid, tid, tl) in train_examples:
            sl = _prompt_logits_with_phantom(model, tok, pid, tid, phantom)
            logp = F.log_softmax(sl.float(), -1)
            tp = F.softmax(tl.float(), -1)
            loss = F.kl_div(logp, tp, reduction="batchmean") / len(train_examples)
            loss.backward()
            loss_val += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(phantom.params(), 2.0)
        opt.step()
        with torch.no_grad():                    # hard per-tensor norm cap -> phantom can't explode
            for p in phantom.params():
                n = float(p.norm())
                if n > max_norm:
                    p.mul_(max_norm / n)
        if step % 3 == 2:
            with torch.no_grad():
                cur = kl_over(train_examples)
            if cur < best - 1e-4:
                best, bad = cur, 0
                best_state = [p.detach().clone() for p in phantom.params()]
            else:
                bad += 1
                if bad >= patience:
                    break
        if step % 20 == 0:
            print(f"    {log_prefix}step {step:3d} kl={loss_val:.4f} best={best:.4f} "
                  f"knorm={phantom.norm()[0]}", flush=True)
    with torch.no_grad():
        for p, b in zip(phantom.params(), best_state):
            p.copy_(b)
    return {"start_kl": round(start, 4), "final_kl": round(best, 4), "steps_used": used}


# --------------------------------------------------------------------------------------------------
#  Evaluation: reuse self_audit_gap's OBJECTIVE scorers on the 6 HELDOUT probes, for every arm.
# --------------------------------------------------------------------------------------------------
def score_arm(model, tok, gen_fn, probes):
    """gen_fn(prompt)->reply. Returns per-trait expression scores using gap's scorers, plus samples.
    For concept traits: keyword-hit rate; for the concise rule: mean answer length in tokens."""
    replies = [gen_fn(p) for p in probes]
    ntok = lambda s: len(tok.encode(s or "", add_special_tokens=False))
    out = {"mean_tok": round(sum(ntok(r) for r in replies) / len(replies), 1), "per_trait": {}, "samples": replies[:3]}
    for c in CARDS:
        if c["kw"]:
            out["per_trait"][c["name"]] = round(sum(gap.kw_hit(r, c["kw"]) for r in replies) / len(replies), 3)
        else:
            out["per_trait"][c["name"]] = out["mean_tok"]   # concise scored by length
    return out, replies


def heldout_kl_to_ceiling(model, tok, phantom, block_cache, block_len, probes, span):
    """Mean KL(phantom || full-block) on held-out probes' answer spans (the teacher's own greedy reply as
    the span). Measures how well the phantom reproduces the ceiling's DISTRIBUTION on unseen prompts."""
    tot, n = 0.0, 0
    for p in probes:
        pid = chat_ids(tok, [{"role": "user", "content": p}])
        tgt = _teacher_greedy(model, tok, pid, block_cache, block_len, span)
        if not tgt:
            continue
        tl = _teacher_target_logits(model, tok, pid, tgt, block_cache, block_len)
        with torch.no_grad():
            sl = _prompt_logits_with_phantom(model, tok, pid, tgt, phantom)
            logp = F.log_softmax(sl.float(), -1)
            tp = F.softmax(tl.float(), -1)
            tot += float(F.kl_div(logp, tp, reduction="batchmean"))
        n += 1
    return round(tot / max(1, n), 4)


# --------------------------------------------------------------------------------------------------
#  Orchestration
# --------------------------------------------------------------------------------------------------
def run(model_name, ks, steps, lr, max_norm, probes_n, out_path, smoke, soft_rival):
    tok, model = load_model(model_name)
    probes = gap.HELDOUT[:probes_n] if probes_n else gap.HELDOUT

    blk = block_text()
    blk_ntok = len(tok.encode(blk, add_special_tokens=False))
    print(f"[block] compiled memory block = {blk_ntok} raw tokens; text:\n{blk}\n", flush=True)

    # Full-block teacher cache (the ceiling + warm-start source)
    block_cache, block_len = teacher_block_cache(model, tok)
    print(f"[teacher] full-block KV prefilled: block_len={block_len} cache positions", flush=True)

    # Build training examples once (teacher greedy answer span + its logits) -- shared by all phantom arms
    print(f"[train-data] building teacher targets on {len(TRAIN_PROMPTS)} prompts "
          f"(span={CFG['answer_span']}) ...", flush=True)
    train_examples = []
    for pr in TRAIN_PROMPTS:
        pid = chat_ids(tok, [{"role": "user", "content": pr}])
        tgt = _teacher_greedy(model, tok, pid, block_cache, block_len, CFG["answer_span"])
        if not tgt:
            continue
        tl = _teacher_target_logits(model, tok, pid, tgt, block_cache, block_len).detach()
        train_examples.append((pid, tgt, tl))
    print(f"[train-data] {len(train_examples)} examples ready", flush=True)

    res = {"model": model_name, "smoke": bool(smoke), "block_ntok": blk_ntok, "block_len": block_len,
           "cards": [{k: c[k] for k in ("name", "cls", "text", "desc")} for c in CARDS],
           "heldout": probes, "train_prompts": TRAIN_PROMPTS, "config": {**CFG, "steps": steps, "lr": lr,
           "max_norm": max_norm, "ks": ks}, "arms": {}}

    def save():
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)

    # ---- ARM: no-memory floor ----
    print("\n=== ARM: no-memory floor ===", flush=True)
    sc, _ = score_arm(model, tok, lambda p: gen_with_phantom(model, tok, p, None, CFG["gen_max"]), probes)
    res["arms"]["floor"] = {"context_tokens": 0, "score": sc}
    print(f"  floor: {sc['per_trait']} mean_tok={sc['mean_tok']}", flush=True)
    save()

    # ---- ARM: full-block ceiling ----
    print("\n=== ARM: full-block ceiling ===", flush=True)
    sc, _ = score_arm(model, tok,
                      lambda p: tok.decode(_teacher_greedy(model, tok,
                          chat_ids(tok, [{"role": "user", "content": p}]), block_cache, block_len,
                          CFG["gen_max"]), skip_special_tokens=True).strip(), probes)
    res["arms"]["ceiling"] = {"context_tokens": blk_ntok, "prefill_tokens": block_len, "score": sc}
    print(f"  ceiling: {sc['per_trait']} mean_tok={sc['mean_tok']}", flush=True)
    save()

    # ---- ARM: RANDOM-phantom null (largest k, untrained, teacher-scale random KV) ----
    knull = max(ks)
    print(f"\n=== ARM: RANDOM-phantom null (k={knull}, untrained) ===", flush=True)
    rnd = PhantomKV(model, knull, teacher_cache=block_cache, random_init=True)
    sc, _ = score_arm(model, tok, lambda p: gen_with_phantom(model, tok, p, rnd, CFG["gen_max"]), probes)
    kl_null = heldout_kl_to_ceiling(model, tok, rnd, block_cache, block_len, probes, CFG["answer_span"])
    res["arms"][f"random_k{knull}"] = {"context_tokens": 0, "trained": False, "kl_to_ceiling": kl_null,
                                       "phantom": rnd.state(), "score": sc}
    print(f"  random-null: {sc['per_trait']} mean_tok={sc['mean_tok']} kl_to_ceiling={kl_null}", flush=True)
    save()

    # ---- ARMS: phantom k in ks (warm-started + trained) ----
    for k in ks:
        print(f"\n=== ARM: phantom k={k} (warm-start + train) ===", flush=True)
        ph = PhantomKV(model, k, teacher_cache=block_cache, random_init=False)
        print(f"  init from teacher cache; {ph.state()}; knorm/vnorm={ph.norm()}", flush=True)
        # warm-start expression BEFORE training (diagnostic: how much does distillation add?)
        sc0, _ = score_arm(model, tok, lambda p: gen_with_phantom(model, tok, p, ph, CFG["gen_max"]), probes)
        kl0 = heldout_kl_to_ceiling(model, tok, ph, block_cache, block_len, probes, CFG["answer_span"])
        print(f"  [warm-start pre-train] {sc0['per_trait']} mean_tok={sc0['mean_tok']} kl={kl0}", flush=True)
        tr = train_phantom(model, tok, ph, train_examples, steps, lr, max_norm, log_prefix=f"k{k} ")
        sc, _ = score_arm(model, tok, lambda p: gen_with_phantom(model, tok, p, ph, CFG["gen_max"]), probes)
        kl1 = heldout_kl_to_ceiling(model, tok, ph, block_cache, block_len, probes, CFG["answer_span"])
        res["arms"][f"phantom_k{k}"] = {"context_tokens": 0, "trained": True, "train": tr,
                                        "warmstart_score": sc0, "warmstart_kl": kl0,
                                        "kl_to_ceiling": kl1, "phantom": ph.state(), "score": sc,
                                        "knorm_vnorm": ph.norm()}
        print(f"  [trained] {sc['per_trait']} mean_tok={sc['mean_tok']} kl_to_ceiling={kl1} "
              f"(train {tr['start_kl']}->{tr['final_kl']})", flush=True)
        save()

    # ---- OPTIONAL ARM: soft-prefix rival (m=16 input-embedding vectors, same KL objective) ----
    if soft_rival:
        print("\n=== ARM: soft-prefix rival (m=16 input-embedding vectors) ===", flush=True)
        try:
            sc = soft_prefix_arm(model, tok, train_examples, probes, steps, lr, max_norm, m=16)
            res["arms"]["soft_prefix_m16"] = sc
            print(f"  soft-prefix: {sc['score']['per_trait']} mean_tok={sc['score']['mean_tok']}", flush=True)
        except Exception as e:
            res["arms"]["soft_prefix_m16"] = {"error": f"{type(e).__name__}: {e}"}
            print(f"  soft-prefix FAILED: {e}", flush=True)
        save()

    summarize(res, tok)
    save()
    print(f"\nsaved -> {out_path}", flush=True)
    return res


# Optional rival: a classic soft prefix (input-embedding vectors), trained on the SAME KL objective, to
# contrast "train post-projection KV" (phantom) vs "train input embeddings" (prefix-tuning-style).
def soft_prefix_arm(model, tok, train_examples, probes, steps, lr, max_norm, m=16):
    emb = model.get_input_embeddings()
    H = model.config.hidden_size
    dt = next(model.parameters()).dtype
    prefix = torch.nn.Parameter(0.02 * torch.randn(m, H, device=DEV, dtype=torch.float32))

    def logits_span(pid, tid):
        full = list(pid) + list(tid)
        e = emb(torch.tensor([full], device=DEV))
        pre = prefix.to(dt)[None]
        e = torch.cat([pre, e], 1)
        att = torch.ones(1, m + len(full), device=DEV, dtype=torch.long)
        lg = model(inputs_embeds=e, attention_mask=att, use_cache=False).logits[0]
        start = m + len(pid) - 1
        return lg[start:start + len(tid)]

    opt = torch.optim.Adam([prefix], lr=lr, weight_decay=2e-3)
    best = 1e9; best_p = prefix.detach().clone(); bad = 0
    for step in range(steps):
        opt.zero_grad(); lv = 0.0
        for (pid, tid, tl) in train_examples:
            sl = logits_span(pid, tid)
            loss = F.kl_div(F.log_softmax(sl.float(), -1), F.softmax(tl.float(), -1),
                            reduction="batchmean") / len(train_examples)
            loss.backward(); lv += float(loss)
        torch.nn.utils.clip_grad_norm_([prefix], 2.0); opt.step()
        with torch.no_grad():
            n = float(prefix.norm())
            if n > max_norm:
                prefix.mul_(max_norm / n)
        if lv < best - 1e-4:
            best, bad, best_p = lv, 0, prefix.detach().clone()
        else:
            bad += 1
            if bad >= 12:
                break
    with torch.no_grad():
        prefix.copy_(best_p)

    @torch.no_grad()
    def gen_soft(p):
        pid = chat_ids(tok, [{"role": "user", "content": p}])
        e = emb(torch.tensor([pid], device=DEV))
        e = torch.cat([prefix.to(dt)[None], e], 1)
        att = torch.ones(1, e.shape[1], device=DEV, dtype=torch.long)
        out = model.generate(inputs_embeds=e, attention_mask=att, max_new_tokens=CFG["gen_max"],
                             do_sample=False, repetition_penalty=1.3, no_repeat_ngram_size=3,
                             pad_token_id=tok.eos_token_id or 0)
        return tok.decode(out[0], skip_special_tokens=True).strip()

    sc, _ = score_arm(model, tok, gen_soft, probes)
    return {"context_tokens": 0, "trained": True, "final_kl": round(best, 4),
            "n_params": prefix.numel(), "score": sc}


def summarize(res, tok):
    """Console table: expression per trait + KL-to-ceiling + context tokens, arm by arm, plus the
    headline '>=80% of ceiling' verdict per concept trait."""
    print("\n" + "=" * 92, flush=True)
    cn = [c["name"] for c in CARDS]
    header = f"{'arm':16} {'ctx_tok':7} " + " ".join(f"{n[:8]:>8}" for n in cn) + f" {'kl2ceil':>8}"
    print(header, flush=True)
    order = (["floor", "ceiling"] + [f"random_k{max(res['config']['ks'])}"]
             + [f"phantom_k{k}" for k in res["config"]["ks"]]
             + (["soft_prefix_m16"] if "soft_prefix_m16" in res["arms"] else []))
    for a in order:
        if a not in res["arms"]:
            continue
        arm = res["arms"][a]
        sc = arm.get("score", {})
        pt = sc.get("per_trait", {})
        kl = arm.get("kl_to_ceiling", "-")
        row = f"{a:16} {str(arm.get('context_tokens','-')):7} "
        row += " ".join(f"{str(pt.get(n,'-')):>8}" for n in cn)
        row += f" {str(kl):>8}"
        print(row, flush=True)

    # verdict math -- fraction of ceiling expression recovered, per concept trait, at 0 tokens
    print("\n[verdict] fraction of CEILING expression recovered at 0 context tokens:", flush=True)
    cel = res["arms"].get("ceiling", {}).get("score", {}).get("per_trait", {})
    flr = res["arms"].get("floor", {}).get("score", {}).get("per_trait", {})
    for k in res["config"]["ks"]:
        arm = res["arms"].get(f"phantom_k{k}", {})
        pt = arm.get("score", {}).get("per_trait", {})
        frac = {}
        for c in CARDS:
            if not c["kw"]:
                continue   # concise handled separately (length, lower=better)
            n = c["name"]
            denom = (cel.get(n, 0) - flr.get(n, 0))
            frac[n] = round((pt.get(n, 0) - flr.get(n, 0)) / denom, 2) if denom else None
        # concise: fraction of the ceiling's length REDUCTION recovered
        cz = "concise"
        c_cel, c_flr = cel.get(cz), flr.get(cz)
        c_ph = pt.get(cz)
        if c_cel is not None and c_flr is not None and c_ph is not None and (c_flr - c_cel):
            frac[cz] = round((c_flr - c_ph) / (c_flr - c_cel), 2)
        print(f"  k={k}: {frac}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--ks", default="4,8,16", help="comma-sep phantom lengths")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--max_norm", type=float, default=None, help="per-tensor L2 cap (default: auto from warm-start)")
    ap.add_argument("--probes", type=int, default=0, help="0 = all 6 heldout; else first N (smoke)")
    ap.add_argument("--out", default="research/runs/phantom_kv_qwen1p5b.json")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--soft-rival", action="store_true", help="also run the m=16 soft-prefix rival")
    a = ap.parse_args()
    ks = [int(x) for x in a.ks.split(",") if x.strip()]
    # default norm cap: generous relative to warm-start norms (KV norms are ~O(10-40) per tensor at 1.5B);
    # 60 lets training move but blocks explosion. Overridable.
    max_norm = a.max_norm if a.max_norm is not None else 60.0
    run(a.model, ks, a.steps, a.lr, max_norm, a.probes, a.out, a.smoke, a.soft_rival)
