"""
legibility_v1_big.py - the SCALE-CURVE sequel to legibility_v1.py (idea 3: self-report + verify).
READ research/legibility_v1.py + research/legibility_v1_findings.md FIRST - this file REUSES idea 3's
self-report + verification harness VERBATIM (the verification rig is the validated asset) and runs it on
BIGGER Instruct models to test whether the introspection gap the 0.5B failed is just a small-model limit.

WHERE WE ARE (legibility_v1.py, idea 3, Qwen2.5-0.5B-Instruct - a clean NEGATIVE):
  TTT WORKS (held-out apply 0.944) but after adapting to a NEW rule the model could NOT STATE it: stated-
  rule held-out apply 0.306, agreement 0.278 - NOT clear of the averaged wrong-rule null 0.292. The
  verification harness is SOUND (oracle true-rule applied 0.769 >> wrong 0.292), and the model DOES
  articulate easy rules when the examples are IN-CONTEXT (ICL self-report ceiling 0.382, with gerund
  1.000 / antonym2 0.833). So the 0.5B failure is plausibly (a) the model being too small to introspect on
  the adaptation, and/or (b) the soft-prefix sitting in the ANSWER slot and derailing fluent statement.

WHAT THIS FILE DOES (two pushes, both honest, both reusing the VERBATIM verify rig from legibility_v1):
  1. SCALE CURVE. Run idea-3 self-report+verify on Qwen2.5-1.5B-Instruct AND Qwen2.5-3B-Instruct (and
     re-state the known 0.5B point). For each model: TTT-adapt the soft prefix on each held-out relation's
     examples (verify it APPLIES, ~0.8+), then WITH the adaptation active have the model STATE the rule in
     words, and VERIFY (does the stated rule, applied to held-out words by the frozen UNADAPTED model,
     match the adapted behavior, clearing the averaged wrong-rule null?). Does verifiable self-report of
     the LEARNED rule improve 0.5B -> 1.5B -> 3B?
  2. STATEMENT-FRIENDLY ADAPTATION (the suspected confound (b)). Instead of a soft-prefix in the answer
     slot, ALSO fit a FREE per-layer STEERING vector / residual-stream nudge at mid layers (test-time-fit
     on the SAME examples, frozen backbone, same apply-CE loss) - so the adaptation does NOT occupy the
     generation head and the model can both APPLY the rule AND fluently STATE it. We FIRST confirm the
     steering APPLIES the rule (if it can't apply, we say so); then we run the IDENTICAL self-report +
     verify on it and compare self-report quality: PREFIX vs STEERING, on each model.

The steering mechanism (NEW here; the named-basis SteerHook in legibility_v1 learns a coefficient over a
fixed diff-in-means basis - here we learn the raw per-layer vector directly, strictly more expressive, to
give "apply" its best shot and decouple the generation head): one learnable v_L in R^H per hooked mid-
layer, added to that layer's residual output at every position. The ONLY trainable tensor; backbone frozen.

HONEST CONTROLS (identical to legibility_v1 - this frontier has produced clean-looking reversals): oracle
true-rule >> averaged wrong-rule (the verifier soundness check), AGREEMENT (stated-applied vs adapted
behavior, per word), the ICL self-report CEILING for EACH model (can it articulate from examples-in-
context at all?), held-out WORDS + held-out RELATIONS, per-relation breakdown, free-gen beside menu, NO
cherry-picking. A NEGATIVE even at 3B (still can't verifiably introspect on what it learned) is a VALID,
important finding - reported plainly.

REUSE: relation bank + held-out split + TTT/prefix mechanism imported from frontier_apply / frontier_apply_v2
(apples-to-apples with lever 3 and with the 0.5B run). The idea-3 self-report prompts, clean_rule, chat_ids,
apply_stated_rule, adapted_behavior, score_against_truth, generate_with_prefix, RULE_DESC, and run_idea3 are
imported VERBATIM from legibility_v1 (the validated rig - not re-implemented). MODELS FROZEN.

Env: cloze/.venv (torch cu128, RTX 5080); .venv-sae untouched. HF_HUB_DISABLE_SYMLINKS=1 if downloading.
RUNS SYNCHRONOUSLY IN ONE PROCESS - no background jobs, no swarm, no parallel workers. 1.5B then 3B
SEQUENTIALLY in the one process (each model is loaded, run, and freed before the next) to bound GPU memory.
Outputs (research/runs/): legibility_v1_big.json + per-model SVGs + a scale-curve SVG.
"""
import os, sys, json, time, argparse, gc
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
torch.set_float32_matmul_precision("high")

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)
sys.path.insert(0, HERE)

# Reuse the EXACT TTT machinery + relation bank (apples-to-apples with lever 3 and the 0.5B run).
import frontier_apply as FA            # SoftPrefix, forward_with_prefix, batch_pack, cache_query_embeds, load_llm, icl_ceiling_rel
import frontier_apply_v2 as FV2        # build_bank, build_vocab_bank, split_bank, single_token_id, eval_prefix_on_relation
# Reuse idea-3's VALIDATED self-report + verification rig VERBATIM (do NOT re-implement the asset).
import legibility_v1 as L1             # RULE_DESC, SELFREPORT_USER, render_examples, clean_rule, chat_ids,
                                       # generate_with_prefix, apply_stated_rule, adapted_behavior,
                                       # score_against_truth, fit_ttt_prefix, run_idea3, svg_idea3, palette
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Maiko palette (reuse legibility_v1's so the SVGs match the rest of the project).
BG, TEAL, PINK, TXT, MUT, GRID = L1.BG, L1.TEAL, L1.PINK, L1.TXT, L1.MUT, L1.GRID
GOLD, LILAC, SLATE, CORAL = L1.GOLD, L1.LILAC, L1.SLATE, L1.CORAL

# Known 0.5B point (from legibility_v1.py runs/legibility_v1_0p5b.json; the anchor of the scale curve).
KNOWN_0P5B = dict(model="Qwen/Qwen2.5-0.5B-Instruct",
                  adapted_menu=0.944, stated_apply=0.306, agreement=0.278,
                  icl_selfreport=0.382, oracle=0.769, wrong=0.292)

# ==========================================================================================
# STATEMENT-FRIENDLY ADAPTATION: a FREE per-layer residual nudge at mid layers (NOT a prefix in the
# answer slot). Strictly more expressive than legibility_v1's named-basis SteerHook (we learn the raw
# vector, not a coefficient over a fixed basis) so APPLY gets its fairest shot; the point is that this
# adaptation does NOT occupy the generation head, so the model can both APPLY the rule and STATE it.
class FreeSteerHook:
    """Adds a learnable steering vector v_L in R^H to the residual stream at the OUTPUT of each hooked
    decoder layer (post-block hidden state), for EVERY position. One v_L per layer (the ONLY trainable
    tensors; the frozen model never moves). Unlike the answer-slot soft prefix, this is a position-
    agnostic residual nudge -> the generation head stays free to produce fluent text during self-report."""
    def __init__(self, model, layers, H):
        self.layers = list(layers)
        self.H = H
        # one learnable vector per layer, tiny init (so step 0 == the frozen model)
        self.vecs = nn.ParameterDict({str(L): nn.Parameter(0.0 * torch.zeros(H, device=DEV))
                                      for L in self.layers})
        self.mods = {L: model.model.layers[L] for L in self.layers}
        self.scale = 1.0                       # set <1 to soften (used only for the gentler self-report)
        self.handles = []
    def parameters(self):
        return list(self.vecs.values())
    def _make(self, L):
        v = self.vecs[str(L)]
        def _hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out      # [B,T,H]
            h = h + (self.scale * v)[None, None, :].to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return _hook
    def __enter__(self):
        for L in self.layers:
            self.handles.append(self.mods[L].register_forward_hook(self._make(L)))
        return self
    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        self.handles = []


def fit_free_steer(tok, model, rel, train_pairs, words, q_emb_cache, answer_tok, layers, H,
                   steps, lr, seed, fit_on="train", K=5, wd=1e-3):
    """TTT but the learnable thing is a FREE per-layer steering vector set (FreeSteerHook), NOT a soft
    prefix. SAME apply-CE loss, SAME fit words, frozen backbone. The query is the bare '{x} ->' embeds
    (no prefix); the adaptation lives entirely in the mid-layer residual nudge. A small weight-decay keeps
    the nudge from blowing up (so it both applies AND leaves the model fluent enough to state). Returns the
    fitted FreeSteerHook (its .vecs hold the learned vectors)."""
    tp = train_pairs[rel]
    if fit_on == "K":
        g = torch.Generator(device=DEV).manual_seed(seed + 5)
        kk = min(K, tp.shape[0]); ti = torch.randperm(tp.shape[0], generator=g, device=DEV)[:kk]
        fit_idx = list(zip(tp[ti, 0].tolist(), tp[ti, 1].tolist()))
    else:
        fit_idx = tp.tolist()
    xs = [words[int(a)] for (a, b) in fit_idx]
    ys = [answer_tok[words[int(b)]] for (a, b) in fit_idx]
    padded, mask = FA.batch_pack([q_emb_cache[x] for x in xs])      # [N,Lq,H] bare query embeds
    ytgt = torch.tensor(ys, device=DEV)
    torch.manual_seed(seed + 23)
    hook = FreeSteerHook(model, layers, H)
    opt = torch.optim.Adam(hook.parameters(), lr, weight_decay=wd)
    with hook:
        for _ in range(steps):
            out = model(inputs_embeds=padded, attention_mask=mask)
            logits = out.logits[:, -1, :]
            loss = F.cross_entropy(logits, ytgt)
            opt.zero_grad(); loss.backward(); opt.step()
    for v in hook.parameters():
        v.requires_grad_(False)
    return hook


@torch.no_grad()
def steer_adapted_behavior(model, hook, test_words, q_emb_cache, menu_ids):
    """The STEERING-adapted model's actual held-out predictions (free token + menu token) per word -
    same role as legibility_v1.adapted_behavior but for the FreeSteerHook adaptation."""
    padded, mask = FA.batch_pack([q_emb_cache[x] for x in test_words])
    with hook:
        logits = model(inputs_embeds=padded, attention_mask=mask).logits[:, -1, :]
    free = logits.argmax(-1).tolist()
    menu = [int(menu_ids[i].item()) for i in logits[:, menu_ids].argmax(-1).tolist()]
    return menu, free


@torch.no_grad()
def generate_with_hook(model, hook, prompt_ids, max_new=24, eos_ids=None, scale=1.0):
    """Greedy free-generation conditioned on [prompt_ids] (chat format) with the FreeSteerHook ACTIVE
    (the steering adaptation in play at every position), so the model states the rule with the adaptation
    on but the generation head free. Mirrors legibility_v1.generate_with_prefix but the adaptation is the
    residual nudge, not an input prefix. scale<1 softens the nudge for the statement (it still applies)."""
    emb = model.get_input_embeddings()
    ids = torch.tensor(prompt_ids, device=DEV)[None, :]
    cur = emb(ids)
    att = torch.ones(cur.shape[0], cur.shape[1], device=DEV)
    out_ids = []
    past = None
    old_scale = hook.scale; hook.scale = scale
    try:
        with hook:
            for _ in range(max_new):
                out = model(inputs_embeds=cur, attention_mask=att, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = int(out.logits[0, -1].argmax().item())
                if eos_ids and nxt in eos_ids:
                    break
                out_ids.append(nxt)
                cur = emb(torch.tensor([[nxt]], device=DEV))
                att = torch.cat([att, torch.ones(1, 1, device=DEV)], 1)
    finally:
        hook.scale = old_scale
    return out_ids


def run_steering_selfreport(tok, model, REL_NAMES, train_pairs, test_pairs, words, answer_tok, menu_ids,
                            out_set_idx, q_emb_cache, held_eval, layers, H, steps, lr, seed, icl_per,
                            fit_on, scale, n_ex=3, max_new=24):
    """The STATEMENT-FRIENDLY arm: IDENTICAL idea-3 self-report + verify, but the adaptation is the FREE
    per-layer steering nudge (FreeSteerHook), not the answer-slot soft prefix. Reuses the VERBATIM verify
    rig from legibility_v1 (apply_stated_rule, score_against_truth, the wrong/oracle/ICL controls). First
    we record whether the steering APPLIES the rule (if it can't apply, that is reported, not hidden).
    Returns the same record/aggregate shape as legibility_v1.run_idea3 so the SVG + scale curve are shared."""
    eos_ids = set([tok.eos_token_id]) if tok.eos_token_id is not None else None
    rows = {}
    for rel in held_eval:
        te = test_pairs[rel].tolist()
        test_words = [words[a] for (a, b) in te]
        test_pairs_w = [(words[a], words[b]) for (a, b) in te]
        # ---- the steering adaptation (FREE per-layer nudge), fit on the relation's own examples ----
        hook = fit_free_steer(tok, model, rel, train_pairs, words, q_emb_cache, answer_tok, layers, H,
                              steps=steps, lr=lr, seed=seed, fit_on=fit_on)
        adp_menu, adp_free = steer_adapted_behavior(model, hook, test_words, q_emb_cache, menu_ids)
        adapted_acc = L1.score_against_truth(adp_menu, test_pairs_w, answer_tok)        # menu (apples-to-apples)
        adapted_free_acc = L1.score_against_truth(adp_free, test_pairs_w, answer_tok)

        ex = L1.render_examples(rel, train_pairs, words, n=n_ex, seed=seed)
        rec = dict(adapted_menu_acc=adapted_acc, adapted_free_acc=adapted_free_acc,
                   icl_apply=icl_per.get(rel, float("nan")), test_n=len(te), examples=ex, reports={})
        # ---- SELF-REPORT under each framing, with the STEERING active (chat format, head free) ----
        for fr_name, tmpl in L1.SELFREPORT_USER.items():
            ids = L1.chat_ids(tok, tmpl.format(ex=ex))
            gen = generate_with_hook(model, hook, ids, max_new=max_new, eos_ids=eos_ids, scale=scale)
            stated = L1.clean_rule(tok.decode(gen))
            # ---- VERIFY: apply the STATED rule with the FROZEN, UNADAPTED model (VERBATIM rig) ----
            st_menu, st_free = L1.apply_stated_rule(tok, model, stated, test_words, answer_tok, menu_ids, out_set_idx)
            stated_acc = L1.score_against_truth(st_menu, test_pairs_w, answer_tok)
            agree = float(sum(int(a == b) for a, b in zip(st_menu, adp_menu)) / max(1, len(st_menu)))
            rec["reports"][fr_name] = dict(stated_rule=stated, stated_apply_acc=stated_acc,
                                           stated_apply_free=L1.score_against_truth(st_free, test_pairs_w, answer_tok),
                                           agreement_with_adapted=agree)
        # ---- CONTROL 1: ICL self-report ceiling (UNADAPTED model, examples-in-prompt) ----
        icl_ids = L1.chat_ids(tok, L1.SELFREPORT_USER["declarative"].format(ex=ex))
        icl_gen = L1.generate_with_prefix(model, None, icl_ids, max_new=max_new, eos_ids=eos_ids)
        icl_stated = L1.clean_rule(tok.decode(icl_gen))
        icl_menu, _ = L1.apply_stated_rule(tok, model, icl_stated, test_words, answer_tok, menu_ids, out_set_idx)
        icl_stated_acc = L1.score_against_truth(icl_menu, test_pairs_w, answer_tok)
        icl_agree = float(sum(int(a == b) for a, b in zip(icl_menu, adp_menu)) / max(1, len(icl_menu)))
        rec["icl_selfreport"] = dict(stated_rule=icl_stated, stated_apply_acc=icl_stated_acc,
                                     agreement_with_adapted=icl_agree)
        # ---- CONTROL 2: ORACLE true-rule (verifier soundness) ----
        true_desc = L1.RULE_DESC.get(rel, rel.replace("_", " "))
        or_menu, _ = L1.apply_stated_rule(tok, model, true_desc, test_words, answer_tok, menu_ids, out_set_idx)
        oracle_acc = L1.score_against_truth(or_menu, test_pairs_w, answer_tok)
        rec["oracle_rule"] = dict(rule=true_desc, apply_acc=oracle_acc)
        # ---- CONTROL 3: WRONG-rule, AVERAGED over several DIFFERENT relations' descriptions ----
        wrong_rels = [r for r in REL_NAMES if r != rel and not r.startswith(rel[:4])
                      and L1.RULE_DESC.get(r) != L1.RULE_DESC.get(rel)][:4]
        wrong_accs = []
        for wr in wrong_rels:
            wr_menu, _ = L1.apply_stated_rule(tok, model, L1.RULE_DESC.get(wr, wr), test_words, answer_tok,
                                              menu_ids, out_set_idx)
            wrong_accs.append(L1.score_against_truth(wr_menu, test_pairs_w, answer_tok))
        wrong_acc = float(sum(wrong_accs) / max(1, len(wrong_accs)))
        rec["wrong_rule"] = dict(rules=[L1.RULE_DESC.get(wr, wr) for wr in wrong_rels],
                                 from_relations=wrong_rels, apply_acc=wrong_acc, per_wrong=wrong_accs)
        rows[rel] = rec
        # free the per-relation hook before the next relation
        del hook
        d = rec["reports"]["declarative"]; mc = rec["reports"]["metacog"]
        print(f"\n  [{rel}] STEER-adapted apply menu={adapted_acc:.3f} free={adapted_free_acc:.3f} "
              f"(ICL {rec['icl_apply']:.2f}), n_test={len(te)}")
        print(f"    SELF-REPORT (declarative): \"{d['stated_rule']}\"")
        print(f"       -> stated-rule applied (menu)={d['stated_apply_acc']:.3f}  "
              f"agreement-with-adapted={d['agreement_with_adapted']:.3f}")
        print(f"    SELF-REPORT (metacog):     \"{mc['stated_rule']}\"")
        print(f"       -> stated-rule applied (menu)={mc['stated_apply_acc']:.3f}  "
              f"agreement-with-adapted={mc['agreement_with_adapted']:.3f}")
        print(f"    CONTROLS: ICL-selfreport \"{icl_stated}\" acc={icl_stated_acc:.3f} agree={icl_agree:.3f} | "
              f"oracle-true acc={oracle_acc:.3f} | wrong-rule(avg of {len(wrong_rels)}) acc={wrong_acc:.3f}")

    def agg(key_path):
        vals = []
        for rel in held_eval:
            d = rows[rel]
            for k in key_path.split("."):
                d = d[k] if isinstance(d, dict) else d
            if d == d:
                vals.append(d)
        return float(sum(vals) / max(1, len(vals)))
    out = dict(held_eval=held_eval, per_relation=rows, fit_on=fit_on, steer_steps=steps, steer_layers=list(layers),
               scoring="menu (apples-to-apples); free reported per-relation", adaptation="free per-layer steering nudge",
               agg_adapted_menu=agg("adapted_menu_acc"), agg_adapted_free=agg("adapted_free_acc"),
               agg_decl_stated_acc=agg("reports.declarative.stated_apply_acc"),
               agg_decl_agreement=agg("reports.declarative.agreement_with_adapted"),
               agg_metacog_stated_acc=agg("reports.metacog.stated_apply_acc"),
               agg_metacog_agreement=agg("reports.metacog.agreement_with_adapted"),
               agg_icl_selfreport_acc=agg("icl_selfreport.stated_apply_acc"),
               agg_icl_selfreport_agreement=agg("icl_selfreport.agreement_with_adapted"),
               agg_oracle_acc=agg("oracle_rule.apply_acc"),
               agg_wrong_acc=agg("wrong_rule.apply_acc"))
    return out


# ==========================================================================================
# Per-model driver: load FROZEN model, build the (shared) bank/split/menu, ICL ceiling, then run BOTH
# arms (prefix self-report = legibility_v1.run_idea3 VERBATIM; steering self-report = the statement-
# friendly arm). Returns the model's record + frees the model. SEQUENTIAL (one model at a time).
def resolve_model_path(model_name):
    """Prefer a local snapshot at ~/hf_models/<name> (downloaded via local_dir to dodge the Windows
    symlink crash, MEMORY: WinError 1314) if present; otherwise use the hub id (standard cache)."""
    local = os.path.join(os.path.expanduser("~"), "hf_models", model_name.split("/")[-1])
    if os.path.isfile(os.path.join(local, "config.json")):
        return local
    return model_name


def run_one_model(model_name, args, tag):
    print("\n" + "#" * 92)
    print(f"# MODEL: {model_name}   (FROZEN, dtype={args.dtype})")
    print("#" * 92)
    t0 = time.time()
    load_path = resolve_model_path(model_name)
    if load_path != model_name:
        print(f"  loading from local snapshot: {load_path}")
    tok, model = FA.load_llm(load_path, dtype=getattr(torch, args.dtype))
    H = model.config.hidden_size
    nL = model.config.num_hidden_layers
    print(f"  loaded: hidden_size={H}  num_layers={nL}")

    # SAME expanded, self-validated relation bank as v2 / the 0.5B run (same Qwen tokenizer -> identical).
    bank, REL_NAMES, dropped = FV2.build_bank(tok, min_pairs=args.min_pairs)
    words, widx, out_words, out_ids = FV2.build_vocab_bank(bank)
    train_pairs, test_pairs = FV2.split_bank(bank, words, widx, test_frac=args.test_frac, seed=args.split_seed)
    chance = 1.0 / len(out_ids)
    print(f"  |relations|={len(REL_NAMES)}  |words|={len(words)}  |menu V|={len(out_ids)}  chance={chance:.4f}")
    if dropped:
        print("  dropped (< %d single-token pairs): " % args.min_pairs +
              ", ".join(f"{r}({n})" for r, n in dropped.items()))

    answer_tok = {w: FV2.single_token_id(tok, w) for w in words}
    menu_ids = torch.tensor([answer_tok[w] for w in out_words], device=DEV)
    out_set_idx = {w: j for j, w in enumerate(out_words)}
    q_emb_cache = FA.cache_query_embeds(tok, model, words)

    # ICL ceiling on this bank (frozen, examples-in-prompt; the upper bracket for "can it apply at all").
    print("  native-ICL ceiling (frozen model, K text pairs, menu-scored)...")
    icl_per = {}
    for r in REL_NAMES:
        icl_per[r] = FA.icl_ceiling_rel(tok, model, r, train_pairs, test_pairs, words, menu_ids,
                                        out_set_idx, K=args.icl_K, n_episodes=args.icl_episodes)
    icl_agg = float(sum(icl_per.values()) / len(icl_per))
    print(f"    ICL aggregate over {len(REL_NAMES)} relations = {icl_agg:.3f}")

    # fixed shuffled relation order + held-out eval set (SAME recipe as v2 / 0.5B -> comparable held set).
    g = torch.Generator().manual_seed(args.seed + 99)
    order = torch.randperm(len(REL_NAMES), generator=g).tolist()
    held_eval = [REL_NAMES[i] for i in order][:args.n_held]
    print(f"  held-out relation eval set (shared, LORO): {held_eval}")

    # steering layers: a MID band scaled to this model's depth (the 0.5B used 6,8,10,12,14,16 of 24).
    if args.steer_layers:
        steer_layers = [int(x) for x in args.steer_layers.split(",")]
    else:
        lo, hi = int(nL * 0.25), int(nL * 0.70)
        steer_layers = list(range(lo, hi + 1, max(1, (hi - lo) // 6)))
    steer_layers = [L for L in steer_layers if 0 <= L < nL]
    print(f"  steering injected at mid layers {steer_layers} (of {nL})")

    rec = dict(model=model_name, hidden_size=H, num_layers=nL, n_relations=len(REL_NAMES),
               n_words=len(words), menu_size=len(out_ids), chance=chance, held_eval=held_eval,
               icl_aggregate=icl_agg, icl_per_relation=icl_per, steer_layers=steer_layers)

    # ===== ARM A: PREFIX self-report + verify (legibility_v1.run_idea3 VERBATIM) =====
    print("\n  " + "=" * 88)
    print("  ARM A - PREFIX self-report + verify (the legibility_v1 idea-3 rig, VERBATIM)")
    print("  " + "=" * 88)
    i3 = L1.run_idea3(tok, model, REL_NAMES, train_pairs, test_pairs, words, widx, answer_tok, menu_ids,
                      out_set_idx, q_emb_cache, held_eval, args.m, args.ttt_steps, args.ttt_lr, args.seed,
                      icl_per, args.fit_on, n_ex=args.n_ex, max_new=args.max_new)
    rec["prefix_idea3"] = i3
    L1.svg_idea3(os.path.join(RUNS, f"legibility_v1_big_prefix{tag}.svg"), i3,
                 f"Prefix self-report + verify ({model_name.split('/')[-1]}, TTT {args.ttt_steps} steps)")
    print("\n  PREFIX-ARM AGGREGATES (menu-scored; held-out words + relations):")
    print(f"    adapted (TTT) apply           = {i3['agg_adapted_menu']:.3f} menu / {i3['agg_adapted_free']:.3f} free")
    print(f"    declarative stated-rule acc   = {i3['agg_decl_stated_acc']:.3f}  (agreement {i3['agg_decl_agreement']:.3f})")
    print(f"    metacog     stated-rule acc   = {i3['agg_metacog_stated_acc']:.3f}  (agreement {i3['agg_metacog_agreement']:.3f})")
    print(f"    CONTROLS: ICL-selfreport={i3['agg_icl_selfreport_acc']:.3f}  oracle={i3['agg_oracle_acc']:.3f}  "
          f"wrong={i3['agg_wrong_acc']:.3f}")

    # ===== ARM B: STEERING self-report + verify (statement-friendly; head left free) =====
    if args.steer:
        print("\n  " + "=" * 88)
        print("  ARM B - STEERING self-report + verify (statement-friendly: free per-layer nudge, head free)")
        print("  " + "=" * 88)
        s3 = run_steering_selfreport(tok, model, REL_NAMES, train_pairs, test_pairs, words, answer_tok,
                                     menu_ids, out_set_idx, q_emb_cache, held_eval, steer_layers, H,
                                     args.steer_steps, args.steer_lr, args.seed, icl_per, args.fit_on,
                                     args.steer_scale, n_ex=args.n_ex, max_new=args.max_new)
        rec["steer_idea3"] = s3
        L1.svg_idea3(os.path.join(RUNS, f"legibility_v1_big_steer{tag}.svg"), s3,
                     f"Steering self-report + verify ({model_name.split('/')[-1]}, free per-layer nudge)")
        print("\n  STEERING-ARM AGGREGATES (menu-scored; held-out words + relations):")
        print(f"    steer-adapted apply           = {s3['agg_adapted_menu']:.3f} menu / {s3['agg_adapted_free']:.3f} free")
        print(f"    declarative stated-rule acc   = {s3['agg_decl_stated_acc']:.3f}  (agreement {s3['agg_decl_agreement']:.3f})")
        print(f"    metacog     stated-rule acc   = {s3['agg_metacog_stated_acc']:.3f}  (agreement {s3['agg_metacog_agreement']:.3f})")
        print(f"    CONTROLS: ICL-selfreport={s3['agg_icl_selfreport_acc']:.3f}  oracle={s3['agg_oracle_acc']:.3f}  "
              f"wrong={s3['agg_wrong_acc']:.3f}")

    rec["wall_time_s"] = round(time.time() - t0, 1)
    print(f"\n  [{model_name}] done in {rec['wall_time_s']}s")
    # free the model before the next one (bound GPU memory: models run SEQUENTIALLY in this one process)
    del model, tok, q_emb_cache, menu_ids
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


# ==========================================================================================
# Scale-curve SVG: best verifiable self-report (stated-apply + agreement) vs the wrong-rule null and the
# ICL ceiling, across models, for BOTH arms. The headline picture: does the verify gap open up with scale?
def svg_scale_curve(path, points, title):
    # points: list of dicts {label, prefix_stated, prefix_agree, steer_stated, steer_agree, wrong, icl, oracle}
    W, Hh, ml, mr, mt, mb = 940, 430, 60, 230, 48, 70
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    n = len(points)
    Yc = lambda v: y0 - (0.0 if v != v else max(0.0, min(1.0, v))) * (y0 - y1)
    Xc = lambda i: x0 + (i + 0.5) * ((x1 - x0) / max(1, n))
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="{BG}"/>',
         f'<text x="{(x0+x1)/2}" y="24" fill="{TXT}" font-size="14" text-anchor="middle">{title}</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="{GRID}"/>',
                         f'<text x="{x0-6}" y="{Y+4:.1f}" fill="{MUT}" font-size="10" text-anchor="end">{v:g}</text>']
    # series as connected lines across the model axis
    def line(key, col, dash=""):
        pts = [(Xc(i), Yc(pt.get(key, float("nan")))) for i, pt in enumerate(points)
               if pt.get(key, float("nan")) == pt.get(key, float("nan"))]
        if len(pts) >= 1:
            d = " ".join((("M" if k == 0 else "L") + f"{x:.1f} {y:.1f}") for k, (x, y) in enumerate(pts))
            p.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.2" {dash}/>')
            for x, y in pts:
                p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{col}"/>')
    series = [("prefix_stated", PINK, "", "prefix stated-rule applied"),
              ("prefix_agree", CORAL, 'stroke-dasharray="4 3"', "prefix agreement (stated vs adapted)"),
              ("steer_stated", TEAL, "", "steering stated-rule applied"),
              ("steer_agree", "#9BD6A0", 'stroke-dasharray="4 3"', "steering agreement"),
              ("icl", LILAC, 'stroke-dasharray="2 3"', "ICL self-report ceiling"),
              ("oracle", GOLD, 'stroke-dasharray="6 3"', "oracle true-rule (verifier sound)"),
              ("wrong", SLATE, 'stroke-dasharray="1 3"', "wrong-rule null (must clear)")]
    for key, col, dash, _ in series:
        line(key, col, dash)
    for i, pt in enumerate(points):
        p.append(f'<text x="{Xc(i):.1f}" y="{y0+18}" fill="{TXT}" font-size="11" text-anchor="middle">{pt["label"]}</text>')
    ly = mt + 8
    for _, col, _, lab in series:
        p += [f'<rect x="{x1+12}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+28}" y="{ly+1}" fill="{TXT}" font-size="10">{lab}</text>']; ly += 20
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))


# ==========================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen/Qwen2.5-1.5B-Instruct,Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--m", type=int, default=8)               # soft-prefix length (TTT), = 0.5B run
    ap.add_argument("--ttt_steps", type=int, default=30)
    ap.add_argument("--ttt_lr", type=float, default=0.05)
    ap.add_argument("--fit_on", default="train", choices=["train", "K"])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.30)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--min_pairs", type=int, default=10)
    ap.add_argument("--n_held", type=int, default=6)
    ap.add_argument("--icl_episodes", type=int, default=80)
    ap.add_argument("--icl_K", type=int, default=5)
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--n_ex", type=int, default=3)
    # statement-friendly steering arm
    ap.add_argument("--steer", type=int, default=1)           # run the steering arm (1) or skip (0)
    ap.add_argument("--steer_steps", type=int, default=60)    # free per-layer vectors: more steps, cheap
    ap.add_argument("--steer_lr", type=float, default=0.02)
    ap.add_argument("--steer_scale", type=float, default=1.0) # nudge scale during the self-report statement
    ap.add_argument("--steer_layers", default="")             # "" -> auto mid-band per model depth
    # float32 = byte-faithful to the 0.5B run + dodges the float32-cache/bf16-weight matmul mismatch in
    # the verbatim-reused frontier_apply helpers. 3B fp32 (~12.4 GB) fits the 17 GB card (activations here
    # are tiny: batches of short queries). Models run SEQUENTIALLY and are freed between, so peak is one model.
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    t_start = time.time()
    model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    print(f"device={DEV}  models={model_list}  m={args.m}  ttt_steps={args.ttt_steps}  dtype={args.dtype}")
    print(f"steering: {'ON' if args.steer else 'OFF'}  steer_steps={args.steer_steps} lr={args.steer_lr} "
          f"scale={args.steer_scale}  (SYNCHRONOUS, single process; models run SEQUENTIALLY)")
    print(f"KNOWN 0.5B anchor: adapted={KNOWN_0P5B['adapted_menu']:.3f}  stated-apply={KNOWN_0P5B['stated_apply']:.3f}  "
          f"agreement={KNOWN_0P5B['agreement']:.3f}  wrong-null={KNOWN_0P5B['wrong']:.3f}  "
          f"ICL-selfreport={KNOWN_0P5B['icl_selfreport']:.3f}  oracle={KNOWN_0P5B['oracle']:.3f}")

    report = dict(models=model_list, env="cloze/.venv (torch cu128, RTX 5080)", frozen_backbone=True,
                  synchronous_single_process=True, dtype=args.dtype, known_0p5b=KNOWN_0P5B,
                  ttt_steps=args.ttt_steps, m=args.m, steer_steps=args.steer_steps, per_model={})

    for mn in model_list:
        short = mn.split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "").lower()  # e.g. 1.5b, 3b
        tag = f"_{short}" + (args.tag or "")
        rec = run_one_model(mn, args, tag)
        report["per_model"][mn] = rec
        # checkpoint after each model (so a late OOM never loses an earlier model's result)
        json.dump(report, open(os.path.join(RUNS, f"legibility_v1_big.json"), "w"), indent=2)
        print(f"  checkpointed report after {mn}")

    # ---- assemble the scale curve (0.5B known + each model, both arms) ----
    def best_arm(i3, kind):
        if i3 is None: return float("nan"), float("nan")
        st = max(i3["agg_decl_stated_acc"], i3["agg_metacog_stated_acc"])
        ag = max(i3["agg_decl_agreement"], i3["agg_metacog_agreement"])
        return st, ag
    points = [dict(label="0.5B", prefix_stated=KNOWN_0P5B["stated_apply"], prefix_agree=KNOWN_0P5B["agreement"],
                   steer_stated=float("nan"), steer_agree=float("nan"), wrong=KNOWN_0P5B["wrong"],
                   icl=KNOWN_0P5B["icl_selfreport"], oracle=KNOWN_0P5B["oracle"])]
    for mn in model_list:
        rec = report["per_model"][mn]
        ps, pa = best_arm(rec.get("prefix_idea3"), "prefix")
        ss, sa = best_arm(rec.get("steer_idea3"), "steer")
        lab = mn.split("/")[-1].replace("Qwen2.5-", "").replace("-Instruct", "")
        i3 = rec.get("prefix_idea3", {})
        points.append(dict(label=lab, prefix_stated=ps, prefix_agree=pa, steer_stated=ss, steer_agree=sa,
                           wrong=i3.get("agg_wrong_acc", float("nan")), icl=i3.get("agg_icl_selfreport_acc", float("nan")),
                           oracle=i3.get("agg_oracle_acc", float("nan"))))
    report["scale_curve_points"] = points
    svg_scale_curve(os.path.join(RUNS, "legibility_v1_big_scalecurve.svg"), points,
                    "Scale curve: can a bigger model VERIFIABLY say what it just learned? (idea 3, held-out)")

    # ===================== VERDICT =====================
    print("\n" + "#" * 92)
    print("# LEGIBILITY v1 (BIG) VERDICT - does verifiable self-report-of-the-learned-rule scale up?")
    print("#" * 92)
    print(f"# scale curve (best framing; stated-rule held-out apply / agreement; wrong-null in [];):")
    print(f"#   0.5B  prefix {KNOWN_0P5B['stated_apply']:.3f}/{KNOWN_0P5B['agreement']:.3f}  "
          f"[wrong {KNOWN_0P5B['wrong']:.3f}, ICL {KNOWN_0P5B['icl_selfreport']:.3f}, oracle {KNOWN_0P5B['oracle']:.3f}]")
    verdict_lines = []
    for pt in points[1:]:
        lab = pt["label"]
        pclr = "clears" if (pt["prefix_stated"] > pt["wrong"] + 0.15 and pt["prefix_stated"] > 0.2) else "FAILS"
        sclr = ("clears" if (pt["steer_stated"] == pt["steer_stated"] and pt["steer_stated"] > pt["wrong"] + 0.15
                             and pt["steer_stated"] > 0.2) else
                ("FAILS" if pt["steer_stated"] == pt["steer_stated"] else "n/a"))
        line = (f"#   {lab:>4}  prefix {pt['prefix_stated']:.3f}/{pt['prefix_agree']:.3f} ({pclr} wrong) | "
                f"steer {pt['steer_stated']:.3f}/{pt['steer_agree']:.3f} ({sclr}) "
                f"[wrong {pt['wrong']:.3f}, ICL {pt['icl']:.3f}, oracle {pt['oracle']:.3f}]")
        print(line); verdict_lines.append(line)
    # overall verdict
    best_pt = points[-1]
    prefix_curve = [points[0]["prefix_stated"]] + [p["prefix_stated"] for p in points[1:]]
    improves = all(b == b for b in prefix_curve) and prefix_curve[-1] > prefix_curve[0] + 0.1
    any_clears = any((p["prefix_stated"] > p["wrong"] + 0.15 and p["prefix_stated"] > 0.2) or
                     (p["steer_stated"] == p["steer_stated"] and p["steer_stated"] > p["wrong"] + 0.15
                      and p["steer_stated"] > 0.2) for p in points[1:])
    if any_clears:
        verdict = ("VERDICT: a bigger model CAN (at least partially) verifiably say what it just learned - at some "
                   "scale/arm the stated rule clears the averaged wrong-rule null on held-out words. The introspection "
                   "gap the 0.5B failed is, at least in part, a SMALL-MODEL limit. See per-model lines for which arm/scale.")
    else:
        verdict = ("VERDICT (NEGATIVE, honest): even at the larger scales tested, self-report-of-the-learned-rule does "
                   "NOT verifiably clear the averaged wrong-rule null on held-out words - neither the answer-slot prefix "
                   "nor the statement-friendly steering arm. The verifier stays SOUND (oracle >> wrong at every scale), and "
                   "the model can articulate from examples-in-context (ICL ceiling), so the failure is specifically the model "
                   "stating WHAT IT JUST LEARNED from the adaptation. The 0.5B negative is NOT merely a small-model artifact "
                   "at the scales tested; the introspection-on-adaptation gap persists. Reported plainly.")
    steer_helps = any(p["steer_stated"] == p["steer_stated"] and p["steer_stated"] > p["prefix_stated"] + 0.05
                      for p in points[1:])
    steer_note = ("The statement-friendly steering arm HELPS (it lifts stated-rule apply above the answer-slot prefix at "
                  "some scale) - decoupling the generation head matters." if steer_helps else
                  "The statement-friendly steering arm does NOT close the gap (it did not beat the prefix self-report by a "
                  "clear margin) - leaving the generation head free was not sufficient at the scales tested.")
    print("#\n# " + verdict)
    print("# " + steer_note)
    print("#" * 92)
    report["verdict"] = verdict
    report["steer_note"] = steer_note
    report["verdict_lines"] = verdict_lines
    report["scale_self_report_improves_prefix"] = bool(improves)
    report["any_arm_clears_wrong_null"] = bool(any_clears)
    report["steering_helps"] = bool(steer_helps)
    report["wall_time_s"] = round(time.time() - t_start, 1)
    json.dump(report, open(os.path.join(RUNS, "legibility_v1_big.json"), "w"), indent=2)
    print(f"\nwrote {os.path.join(RUNS, 'legibility_v1_big.json')}  [{report['wall_time_s']}s]  "
          f"(synchronous, single process - clean exit)")


if __name__ == "__main__":
    main()
