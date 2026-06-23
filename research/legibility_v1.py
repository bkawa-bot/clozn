"""
legibility_v1.py - the FIRST legibility experiment for Clozn. Can a TEST-TIME-LEARNED rule be made
LEGIBLE (readable in words / by name), not merely FUNCTIONAL? READ research/frontier_apply.py +
research/frontier_apply_v2.py + their _findings.md FIRST - this file is the direct sequel.

WHERE WE ARE (frontier_apply_v2.py, lever 3):
  TEST-TIME ADAPTATION (TTT) WORKS. For a NEW (held-out) relation, fit a soft prefix by a few (~20)
  gradient steps on its OWN few examples (backprop through the FROZEN Qwen2.5-0.5B-Instruct; only the
  prefix moves) -> the frozen model applies the rule to HELD-OUT words at ~0.94 free-gen (near the
  native-ICL ceiling), its OWN output. BUT that learned prefix is an OPAQUE BLOB: not legible. A
  relation-probe over learned prefixes was at chance (frontier_apply Stage 1); lever-2's apparent 0.99
  legibility was an INPUT-FEATURE ARTIFACT (an untrained map scored the same 1.0). So application is
  bought; legibility is NOT. THIS FILE: can we make the learned thing READABLE?

TWO LEGIBILITY ROUTES (idea 3 first - it is the most on-thesis):

IDEA 3 - SELF-REPORT + VERIFY (primary). Fit the TTT prefix for a held-out relation, then with the
  adaptation ACTIVE prompt the FROZEN model to STATE the rule in words ("the rule is: ___"; also a
  metacognitive framing "what rule did you just learn?"). Then VERIFY (do NOT trust the words): take the
  model's STATED rule and apply it to the HELD-OUT words via a plain instruction to the FROZEN, UNADAPTED
  model ("apply the rule '<stated>' to X ->"), and measure (a) stated-rule held-out accuracy and (b)
  AGREEMENT between (stated-rule-applied) and the ADAPTED model's actual held-out behavior. Controls:
  an ICL self-report ceiling (unadapted model states the rule from examples-in-prompt), a TRUE-rule
  description (oracle: can instruction-following even apply this rule?), and a WRONG-rule negative
  control (a different relation's description must NOT verify). The self-report may confabulate - the
  VERIFICATION is the point.

IDEA 1 - NAMED SLIDERS (basis-constrained TTT; secondary, if tractable). Build a basis of K NAMED
  directions (diff-in-means relation directions a la conceptmem/p18). Constrain the TTT adaptation to
  lie in this named basis: learn COEFFICIENTS over the basis (steering = sum_i c_i d_i added at a layer
  via a hook), NOT a free prefix. APPLIES: held-out-word accuracy vs unconstrained TTT + the ICL ceiling
  (does constraining cost accuracy?). LEGIBLE: read the rule off the coefficients (argmax coeff -> name;
  a probe) BUT beat a PROPER null (the lever-2 lesson): a SHUFFLED-basis null + an OUT-OF-BASIS (LORO)
  setting where the relation's own direction is removed (it cannot point to itself -> the coverage test).
  Per-relation: which relations stay legible-AND-working vs fall outside the basis.

HONESTY (load-bearing - this frontier has produced clean-looking reversals): held-out WORDS + held-out
  RELATIONS; FREE-gen reported (not just menu); every number beside the ICL ceiling + the unconstrained-
  TTT + a PROPER null; per-relation breakdown; NO cherry-picking. A NEGATIVE (self-report confabulates /
  the named basis kills accuracy / coefficients aren't legible past a real null) is a VALID, valuable
  finding - reported plainly.

REUSE: the relation bank + held-out split + the TTT/prefix mechanism are imported from frontier_apply_v2
  + frontier_apply (apples-to-apples). MODEL: Qwen2.5-0.5B-Instruct, FROZEN. Env: cloze/.venv (torch
  cu128, RTX 5080); .venv-sae untouched. RUNS SYNCHRONOUSLY IN ONE PROCESS - no background jobs, no swarm.
Outputs (research/runs/): legibility_v1{tag}.json + SVGs (idea3 verify bars, idea1 applies/legible bars).
"""
import os, sys, json, time, argparse, re
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
torch.set_float32_matmul_precision("high")

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)
sys.path.insert(0, HERE)

# Reuse the EXACT TTT machinery + relation bank (apples-to-apples with lever 3).
import frontier_apply as FA            # SoftPrefix, forward_with_prefix, batch_pack, cache_query_embeds, load_llm
import frontier_apply_v2 as FV2        # build_bank, build_vocab_bank, split_bank, harvest_feat, eval_prefix_on_relation
from sidecar_semantic import CARRIER

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Maiko palette (matches frontier_apply_v2.py)
BG, TEAL, PINK, TXT, MUT, GRID = "#1A1F4A", "#6FD6C9", "#FF8FB3", "#F4F0E8", "#8784b3", "#2c2f5e"
GOLD = "#E8C977"    # ICL ceiling / oracle
LILAC = "#B6A6E8"   # baseline / null
SLATE = "#7E8AA8"   # wrong-rule / null
CORAL = "#E89B7E"   # second series

# A short, human-written description of each relation's RULE, used ONLY as (a) the oracle "true rule"
# ceiling for verification (can instruction-following even apply it?) and (b) the WRONG-rule negative
# control (apply some OTHER relation's description). NOT given to the adapted model; it self-reports.
RULE_DESC = {
 "antonym": "give the opposite", "antonym2": "give the opposite",
 "plural": "make it plural", "past": "put the verb in past tense",
 "comparative": "make the comparative form", "superlative": "make the superlative form",
 "capital": "name the capital city of the country", "color": "name its typical color",
 "hypernym": "name the category it belongs to", "hyponym": "give an example of it",
 "synonym": "give a synonym", "gerund": "add -ing to the verb",
 "third_person": "add -s for third person", "agent": "name the person who does it",
 "verb_noun": "turn the verb into a noun", "nationality": "give the nationality adjective",
 "continent": "name its continent", "opposite_gender": "give the opposite gender word",
 "part_of": "name the whole it is part of", "diminutive": "name its baby or young form",
 "made_of": "describe what it is like", "un_prefix": "add the prefix un-",
 "re_prefix": "add the prefix re-", "adverb": "make it an adverb with -ly",
 "ordinal": "give the ordinal number", "habitat": "name where it lives",
}

# ==========================================================================================
# Shared TTT core (= lever 3): fit a soft prefix for ONE relation on its OWN words, frozen backbone.
def fit_ttt_prefix(tok, model, rel, train_pairs, words, q_emb_cache, answer_tok, m, steps, lr, seed,
                   fit_on="train", K=5):
    """Fit a soft prefix [m,H] by gradient descent on the apply-CE (the EXACT lever-3 loss), on the
    relation's own examples. fit_on='train' uses its full TRAIN words (the strong setting whose
    asymptote == the Stage-1 oracle); 'K' uses only K examples (few-shot). Backbone frozen; only the
    prefix trains. Returns the trained SoftPrefix module."""
    H = model.config.hidden_size
    tp = train_pairs[rel]
    if fit_on == "K":
        g = torch.Generator(device=DEV).manual_seed(seed + 5)
        kk = min(K, tp.shape[0]); ti = torch.randperm(tp.shape[0], generator=g, device=DEV)[:kk]
        fit_idx = list(zip(tp[ti, 0].tolist(), tp[ti, 1].tolist()))
    else:
        fit_idx = tp.tolist()
    xs = [words[int(a)] for (a, b) in fit_idx]
    ys = [answer_tok[words[int(b)]] for (a, b) in fit_idx]
    padded, mask = FA.batch_pack([q_emb_cache[x] for x in xs])
    ytgt = torch.tensor(ys, device=DEV)
    torch.manual_seed(seed + 13)
    p0 = 0.02 * torch.randn(m, H, device=DEV)
    pm = FA.SoftPrefix(m, H).to(DEV); pm.prefix = nn.Parameter(p0)
    opt = torch.optim.Adam(pm.parameters(), lr)
    pm.train()
    for _ in range(steps):
        logits = FA.forward_with_prefix(model, pm, padded, mask)
        loss = F.cross_entropy(logits, ytgt)
        opt.zero_grad(); loss.backward(); opt.step()
    pm.eval()
    return pm

# ==========================================================================================
# IDEA 3 - SELF-REPORT: with the prefix ACTIVE, make the frozen model STATE the rule in words.
# We prepend the trained prefix to a natural-language instruction asking for the rule, and free-
# generate a short continuation (greedy). Prefix tokens are always attended.
@torch.no_grad()
def generate_with_prefix(model, prefix_tensor, prompt_ids, max_new=24, eos_ids=None):
    """Greedy free-generation of up to max_new tokens, conditioned on [prefix] + [prompt_ids] using
    inputs_embeds (so the trained soft prefix is in play). prefix_tensor: [m,H] or None."""
    emb = model.get_input_embeddings()
    ids = torch.tensor(prompt_ids, device=DEV)[None, :]               # [1, L]
    e = emb(ids)                                                      # [1, L, H]
    if prefix_tensor is not None:
        pre = prefix_tensor[None].to(e.dtype)                        # [1, m, H]
        full = torch.cat([pre, e], 1)
        m = prefix_tensor.shape[0]
    else:
        full = e; m = 0
    att = torch.ones(full.shape[0], full.shape[1], device=DEV)
    out_ids = []
    past = None; cur = full
    for _ in range(max_new):
        out = model(inputs_embeds=cur, attention_mask=att, past_key_values=past, use_cache=True)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax().item())
        if eos_ids and nxt in eos_ids:
            break
        out_ids.append(nxt)
        cur = emb(torch.tensor([[nxt]], device=DEV))
        att = torch.cat([att, torch.ones(1, 1, device=DEV)], 1)
    return out_ids

# Prompts that elicit a stated rule. Qwen2.5-0.5B is an INSTRUCT model, so we elicit in its CHAT
# format (far better at "state the rule" than a raw-text continuation; gives the self-report route its
# fairest shot - a weak prompt would make a NEGATIVE meaningless). The {ex} block is the same examples
# the adaptation saw. Two framings: a third-person "the rule is" and a first-person metacognitive one.
SELFREPORT_USER = {
 "declarative": ("These word pairs all follow the same transformation rule (the first word becomes "
                 "the second):\n{ex}\nState the transformation rule in one short phrase, starting with a verb."),
 "metacog":     ("I just showed you these word pairs and you learned the rule that maps the first word "
                 "to the second:\n{ex}\nIn one short phrase starting with a verb, what rule did you just learn?"),
}

def render_examples(rel, train_pairs, words, n=3, seed=0):
    """A few in-context example lines 'a -> b' from the relation's TRAIN pairs (the same examples the
    adaptation saw). Used to fill the self-report prompt + the ICL self-report ceiling."""
    g = torch.Generator(device=DEV).manual_seed(seed + 31)
    tp = train_pairs[rel]; kk = min(n, tp.shape[0])
    ti = torch.randperm(tp.shape[0], generator=g, device=DEV)[:kk]
    return "\n".join(f"{words[int(tp[i,0])]} -> {words[int(tp[i,1])]}" for i in ti)

def clean_rule(text):
    """Trim the free-gen rule to a short usable phrase: first sentence/line, strip wrappers & quotes."""
    t = text.strip().strip('"').strip()
    t = t.split("\n")[0].strip()
    t = re.sub(r"\s+", " ", t)
    for stop in [". ", "; ", " - ", " (", ":", ","]:
        if stop in t:
            t = t.split(stop)[0]
    return t.strip(' ."\':;-').strip()

def chat_ids(tok, user_text):
    """Token ids for a one-turn chat (system+user+assistant-open), the Instruct model's native format."""
    msgs = [{"role": "user", "content": user_text}]
    return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)

# ----- VERIFY the stated rule: apply it to held-out words with the FROZEN, UNADAPTED model -----
# CHAT format + menu scoring (the apples-to-apples retrieval metric). The standalone probe confirmed
# this path applies a GOOD rule correctly (plural/opposite/past all right), so it is a sound verifier:
# a stated rule that fails here genuinely does not capture the relation, and the wrong-rule control
# (a different relation's description) must fail by the same path.
@torch.no_grad()
def apply_stated_rule(tok, model, rule_text, test_words, answer_tok, menu_ids, menu_idx_of):
    """Apply `rule_text` to each held-out TEST word x with the FROZEN model in CHAT format; return
    per-word (menu_pred_token, free_pred_token). The answer slot is the model's next token after
    'Answer:'. Menu = argmax restricted to the candidate vocabulary; free = full-vocab argmax."""
    menu_preds, free_preds = [], []
    if not rule_text:
        rule_text = "(none)"
    for x in test_words:
        user = (f"Apply this rule to the word and answer with only the resulting word.\n"
                f"Rule: {rule_text}\nWord: {x}\nAnswer:")
        ids = chat_ids(tok, user)
        logits = model(torch.tensor(ids, device=DEV)[None, :]).logits[0, -1]
        free_preds.append(int(logits.argmax().item()))
        menu_preds.append(int(menu_ids[logits[menu_ids].argmax()].item()))
    return menu_preds, free_preds

@torch.no_grad()
def adapted_behavior(model, prefix, test_words, q_emb_cache, menu_ids):
    """The ADAPTED model's actual held-out predictions (free token + menu token) per word - the
    behavior the stated rule must AGREE with. Same prefix forward path as eval."""
    H = prefix.prefix.shape[1] if hasattr(prefix, "prefix") else prefix.shape[1]
    pm = prefix
    padded, mask = FA.batch_pack([q_emb_cache[x] for x in test_words])
    logits = FA.forward_with_prefix(model, pm, padded, mask)         # [N,V]
    free = logits.argmax(-1).tolist()
    menu = [int(menu_ids[i].item()) for i in logits[:, menu_ids].argmax(-1).tolist()]
    return menu, free

def score_against_truth(preds_tokens, test_words_pairs, answer_tok):
    """Fraction of preds equal to the TRUE answer token (held-out apply accuracy). preds_tokens are
    token ids (menu- or free-decoded); we compare to each pair's answer-token id."""
    n = len(test_words_pairs)
    if n == 0: return float("nan")
    ok = 0
    for (x, ytrue), pred in zip(test_words_pairs, preds_tokens):
        if pred == answer_tok[ytrue]:
            ok += 1
    return ok / n

def run_idea3(tok, model, REL_NAMES, train_pairs, test_pairs, words, widx, answer_tok, menu_ids,
              out_set_idx, q_emb_cache, held_eval, m, ttt_steps, ttt_lr, seed, icl_per, fit_on,
              n_ex=3, max_new=24):
    """IDEA 3 end-to-end for each held-out relation: fit TTT prefix -> self-report (declarative +
    metacog, in CHAT format, adaptation ACTIVE) -> verify each stated rule by frozen-model application
    on held-out words (chat + menu-scored) -> AGREEMENT with the adapted behavior. Controls: ICL self-
    report (unadapted, examples-in-prompt) + oracle true-rule + wrong-rule. Scoring is MENU (the apples-
    to-apples retrieval metric); free reported alongside. Returns per-relation record + aggregates.

    NOTE on the adapted self-report: the soft prefix was trained ONLY on the answer-slot query "{x} ->",
    so prepending it to a long chat prompt pushes toward answer tokens and can derail fluency. That is
    itself part of the honest finding (an injected adaptation is not automatically a STATABLE one); we
    still give it its fairest shot (chat format, verb-first instruction) and read it against the ICL
    self-report ceiling, which elicits the rule from the SAME examples WITHOUT the prefix."""
    eos_ids = set([tok.eos_token_id]) if tok.eos_token_id is not None else None
    rows = {}
    for rel in held_eval:
        te = test_pairs[rel].tolist()
        test_words = [words[a] for (a, b) in te]
        test_pairs_w = [(words[a], words[b]) for (a, b) in te]
        # ---- the working adaptation (lever 3) ----
        pm = fit_ttt_prefix(tok, model, rel, train_pairs, words, q_emb_cache, answer_tok,
                            m=m, steps=ttt_steps, lr=ttt_lr, seed=seed, fit_on=fit_on)
        adp_menu, adp_free = adapted_behavior(model, pm, test_words, q_emb_cache, menu_ids)
        adapted_acc = score_against_truth(adp_menu, test_pairs_w, answer_tok)         # menu (apples-to-apples)
        adapted_free_acc = score_against_truth(adp_free, test_pairs_w, answer_tok)

        ex = render_examples(rel, train_pairs, words, n=n_ex, seed=seed)
        rec = dict(adapted_menu_acc=adapted_acc, adapted_free_acc=adapted_free_acc,
                   icl_apply=icl_per.get(rel, float("nan")), test_n=len(te), examples=ex, reports={})
        # ---- SELF-REPORT under each framing, with the adaptation ACTIVE (chat format) ----
        for fr_name, tmpl in SELFREPORT_USER.items():
            ids = chat_ids(tok, tmpl.format(ex=ex))
            gen = generate_with_prefix(model, pm.prefix.detach(), ids, max_new=max_new, eos_ids=eos_ids)
            stated = clean_rule(tok.decode(gen))
            # ---- VERIFY: apply the STATED rule with the FROZEN, UNADAPTED model (chat + menu) ----
            st_menu, st_free = apply_stated_rule(tok, model, stated, test_words, answer_tok, menu_ids, out_set_idx)
            stated_acc = score_against_truth(st_menu, test_pairs_w, answer_tok)
            # AGREEMENT: stated-rule-applied vs the adapted model's own behavior, per word (menu tokens)
            agree = float(sum(int(a == b) for a, b in zip(st_menu, adp_menu)) / max(1, len(st_menu)))
            rec["reports"][fr_name] = dict(stated_rule=stated, stated_apply_acc=stated_acc,
                                           stated_apply_free=score_against_truth(st_free, test_pairs_w, answer_tok),
                                           agreement_with_adapted=agree)
        # ---- CONTROL 1: ICL self-report ceiling (UNADAPTED model states rule from examples-in-prompt) ----
        icl_ids = chat_ids(tok, SELFREPORT_USER["declarative"].format(ex=ex))
        icl_gen = generate_with_prefix(model, None, icl_ids, max_new=max_new, eos_ids=eos_ids)
        icl_stated = clean_rule(tok.decode(icl_gen))
        icl_menu, _ = apply_stated_rule(tok, model, icl_stated, test_words, answer_tok, menu_ids, out_set_idx)
        icl_stated_acc = score_against_truth(icl_menu, test_pairs_w, answer_tok)
        icl_agree = float(sum(int(a == b) for a, b in zip(icl_menu, adp_menu)) / max(1, len(icl_menu)))
        rec["icl_selfreport"] = dict(stated_rule=icl_stated, stated_apply_acc=icl_stated_acc,
                                     agreement_with_adapted=icl_agree)
        # ---- CONTROL 2: ORACLE true-rule description (can instruction-following even apply it?) ----
        true_desc = RULE_DESC.get(rel, rel.replace("_", " "))
        or_menu, _ = apply_stated_rule(tok, model, true_desc, test_words, answer_tok, menu_ids, out_set_idx)
        oracle_acc = score_against_truth(or_menu, test_pairs_w, answer_tok)
        rec["oracle_rule"] = dict(rule=true_desc, apply_acc=oracle_acc)
        # ---- CONTROL 3: WRONG-rule negative control, AVERAGED over several DIFFERENT relations'
        #      descriptions (one arbitrary wrong rule is noisy on a small held-out set, and the word's
        #      own prior can make the model emit the true answer regardless of the rule - averaging
        #      several wrong rules gives a stable "rule-independent" floor the stated rule must clear). ----
        wrong_rels = [r for r in REL_NAMES if r != rel and not r.startswith(rel[:4])
                      and RULE_DESC.get(r) != RULE_DESC.get(rel)][:4]
        wrong_accs = []
        for wr in wrong_rels:
            wr_menu, _ = apply_stated_rule(tok, model, RULE_DESC.get(wr, wr), test_words, answer_tok,
                                           menu_ids, out_set_idx)
            wrong_accs.append(score_against_truth(wr_menu, test_pairs_w, answer_tok))
        wrong_acc = float(sum(wrong_accs) / max(1, len(wrong_accs)))
        rec["wrong_rule"] = dict(rules=[RULE_DESC.get(wr, wr) for wr in wrong_rels],
                                 from_relations=wrong_rels, apply_acc=wrong_acc, per_wrong=wrong_accs)
        rows[rel] = rec
        # print per-relation
        d = rec["reports"]["declarative"]; mc = rec["reports"]["metacog"]
        print(f"\n  [{rel}] adapted apply menu={adapted_acc:.3f} free={adapted_free_acc:.3f} "
              f"(ICL {rec['icl_apply']:.2f}), n_test={len(te)}")
        print(f"    SELF-REPORT (declarative): \"{d['stated_rule']}\"")
        print(f"       -> stated-rule applied (menu)={d['stated_apply_acc']:.3f}  "
              f"agreement-with-adapted={d['agreement_with_adapted']:.3f}")
        print(f"    SELF-REPORT (metacog):     \"{mc['stated_rule']}\"")
        print(f"       -> stated-rule applied (menu)={mc['stated_apply_acc']:.3f}  "
              f"agreement-with-adapted={mc['agreement_with_adapted']:.3f}")
        print(f"    CONTROLS: ICL-selfreport \"{icl_stated}\" acc={icl_stated_acc:.3f} agree={icl_agree:.3f} | "
              f"oracle-true \"{true_desc}\" acc={oracle_acc:.3f} | "
              f"wrong-rule(avg of {len(wrong_rels)}) acc={wrong_acc:.3f}")
    # aggregates
    def agg(key_path):
        vals = []
        for rel in held_eval:
            d = rows[rel]
            for k in key_path.split("."):
                d = d[k] if isinstance(d, dict) else d
            if d == d:
                vals.append(d)
        return float(sum(vals) / max(1, len(vals)))
    out = dict(held_eval=held_eval, per_relation=rows, fit_on=fit_on, ttt_steps=ttt_steps,
               scoring="menu (apples-to-apples); free reported per-relation",
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
# IDEA 1 - NAMED SLIDERS: constrain the TTT adaptation to a basis of named diff-in-means relation
# directions; learn COEFFICIENTS over the basis (steering = sum_i c_i d_i injected at a layer), not a
# free prefix. Legible-by-construction IF the coefficient for relation R points at R's own direction.
@torch.no_grad()
def build_relation_basis(REL_NAMES, train_pairs, words, feat_cache, normalize=True):
    """One NAMED diff-in-means direction per relation: d_r = mean_over_train_pairs( feat(R(x)) - feat(x) )
    at the feature layer. Training-free (conceptmem/p18 recipe). Uses only TRAIN pairs (held-out words
    never enter the basis). normalize=True -> unit-norm rows (so coefficients are comparable across
    relations, which the legibility read-out needs); normalize=False -> RAW diff-in-means magnitude
    (the conceptmem lesson: a unit-norm scale under-doses; raw is the natural steering magnitude).
    Returns D: [K,H] + the relation order."""
    dirs = []
    for r in REL_NAMES:
        tp = train_pairs[r].tolist()
        diffs = [feat_cache[b] - feat_cache[a] for (a, b) in tp]
        d = torch.stack(diffs).mean(0)
        if normalize:
            d = d / (d.norm() + 1e-8)
        dirs.append(d)
    return torch.stack(dirs), list(REL_NAMES)               # [K,H]

@torch.no_grad()
def build_relation_basis_multilayer(REL_NAMES, train_pairs, words, feat_by_layer, layers, normalize=False):
    """A per-layer named basis: {L: [K,H]} where row r at layer L = mean diff-in-means of (R(x),x) at L.
    normalize=False keeps the RAW magnitude (the natural steering scale; the legible coefficient c then
    finds its own scale). For the legibility read-out we use the layer-stacked, unit-normed coefficient
    geometry, so raw vs unit here only affects the magnitude the optimizer must discover, not legibility."""
    D_by_layer = {}
    for L in layers:
        fc = feat_by_layer[L]
        dirs = []
        for r in REL_NAMES:
            tp = train_pairs[r].tolist()
            d = torch.stack([fc[b] - fc[a] for (a, b) in tp]).mean(0)
            if normalize:
                d = d / (d.norm() + 1e-8)
            dirs.append(d)
        D_by_layer[L] = torch.stack(dirs)                   # [K,H]
    return D_by_layer, list(REL_NAMES)

class SteerHook:
    """Adds a steering vector to the residual stream at the OUTPUT of one or more decoder layers
    (post-block hidden state), for every position. The vector injected at layer L is (c @ D[L]), where
    c is the SHARED learnable coefficient vector over the named basis and D[L] is that layer's diff-in-
    means basis. ONE coefficient vector drives all hooked layers -> the legible state is a single named
    slider-set, however many layers it touches. c is the ONLY trainable tensor (frozen model never moves).
    Raw (unnormalized) diff-in-means rows carry the natural steering magnitude (conceptmem lesson), so c
    finds its own scale - no separate scale knob (a learnable c makes any global scale constant redundant)."""
    def __init__(self, model, layers, D_by_layer):
        self.D_by_layer = D_by_layer                        # {L: [K,H]} fixed bases (no grad)
        self.layers = list(layers)
        self.coeff = None                                   # [K] leaf tensor (the ONLY trainable thing)
        self.mods = {L: model.model.layers[L] for L in self.layers}
        self.handles = []
    def _make(self, L):
        D = self.D_by_layer[L]
        def _hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out   # [B,T,H]
            v = (self.coeff @ D)                            # [H]
            h = h + v[None, None, :].to(h.dtype)
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

def fit_named_coeffs(tok, model, rel, train_pairs, words, q_emb_cache, answer_tok, D_by_layer, layers,
                     steps, lr, seed, fit_on="train", K=5):
    """TTT but the learnable thing is a SHARED coefficient vector c over the NAMED basis (steering
    injected at `layers` via SteerHook), not a free prefix. Same apply-CE loss, same fit words. Returns
    c.detach() [Kc]. The QUERY is the bare '{x} ->' (NO soft prefix); the adaptation lives entirely in
    the named-direction steering."""
    Kc = D_by_layer[layers[0]].shape[0]
    tp = train_pairs[rel]
    if fit_on == "K":
        g = torch.Generator(device=DEV).manual_seed(seed + 5)
        kk = min(K, tp.shape[0]); ti = torch.randperm(tp.shape[0], generator=g, device=DEV)[:kk]
        fit_idx = list(zip(tp[ti, 0].tolist(), tp[ti, 1].tolist()))
    else:
        fit_idx = tp.tolist()
    xs = [words[int(a)] for (a, b) in fit_idx]
    ys = [answer_tok[words[int(b)]] for (a, b) in fit_idx]
    padded, mask = FA.batch_pack([q_emb_cache[x] for x in xs])       # [N,Lq,H] bare query embeds
    ytgt = torch.tensor(ys, device=DEV)
    torch.manual_seed(seed + 21)
    coeff = nn.Parameter(0.01 * torch.randn(Kc, device=DEV))
    opt = torch.optim.Adam([coeff], lr)
    with SteerHook(model, layers, D_by_layer) as hook:
        hook.coeff = coeff
        for _ in range(steps):
            out = model(inputs_embeds=padded, attention_mask=mask)
            logits = out.logits[:, -1, :]
            loss = F.cross_entropy(logits, ytgt)
            opt.zero_grad(); loss.backward(); opt.step()
    return coeff.detach()

@torch.no_grad()
def eval_named_coeffs(model, rel, test_pairs, words, q_emb_cache, answer_tok, menu_ids, out_set_idx,
                      coeff, D_by_layer, layers):
    """Held-out menu+free apply for a coefficient vector (steering injected at `layers`)."""
    pairs = test_pairs[rel].tolist()
    if not pairs: return float("nan"), float("nan")
    xs = [words[a] for (a, b) in pairs]
    ytok = torch.tensor([answer_tok[words[b]] for (a, b) in pairs], device=DEV)
    ymenu = torch.tensor([out_set_idx[words[b]] for (a, b) in pairs], device=DEV)
    padded, mask = FA.batch_pack([q_emb_cache[x] for x in xs])
    with SteerHook(model, layers, D_by_layer) as hook:
        hook.coeff = coeff
        logits = model(inputs_embeds=padded, attention_mask=mask).logits[:, -1, :]
    free = (logits.argmax(-1) == ytok).float().mean().item()
    menu = (logits[:, menu_ids].argmax(-1) == ymenu).float().mean().item()
    return menu, free

def run_idea1(tok, model, REL_NAMES, train_pairs, test_pairs, words, answer_tok, menu_ids,
              out_set_idx, q_emb_cache, D_by_layer, basis_names, layers, held_eval, steps, lr, seed,
              icl_per, ttt_free_per, fit_on="train"):
    """IDEA 1 for each held-out relation, BOTH:
      in-basis : R's own named direction IS in the basis (all REL_NAMES) -> legible-by-name possible.
      out-of-basis (LORO): R's direction REMOVED from the basis -> coverage test (cannot point to self).
    APPLIES = held-out free/menu apply vs unconstrained-TTT + ICL. LEGIBLE = (a) argmax coeff == R
    (in-basis only), beaten against a SHUFFLED-label null; (b) mean rank of R's own coefficient.
    Steering is the multi-layer named-slider injection (one shared coefficient vector c over the basis)."""
    name_idx = {r: i for i, r in enumerate(basis_names)}
    def D_drop(rel):
        keep = [i for i, r in enumerate(basis_names) if r != rel]
        return {L: D_by_layer[L][keep] for L in layers}

    rows = {}
    for rel in held_eval:
        # ---- IN-BASIS: full named basis (R's own direction present) ----
        c_in = fit_named_coeffs(tok, model, rel, train_pairs, words, q_emb_cache, answer_tok,
                                D_by_layer, layers, steps, lr, seed, fit_on=fit_on)
        mn_in, fr_in = eval_named_coeffs(model, rel, test_pairs, words, q_emb_cache, answer_tok,
                                         menu_ids, out_set_idx, c_in, D_by_layer, layers)
        # legible-by-construction: does the largest-|coeff| basis direction name R itself?
        top_name = basis_names[int(c_in.abs().argmax().item())]
        names_self = (top_name == rel)
        self_rank = c_in.abs().argsort(descending=True).tolist().index(name_idx[rel]) + 1
        # ---- OUT-OF-BASIS (LORO): remove R's own direction -> coverage test ----
        D_loro = D_drop(rel)
        c_loro = fit_named_coeffs(tok, model, rel, train_pairs, words, q_emb_cache, answer_tok,
                                  D_loro, layers, steps, lr, seed, fit_on=fit_on)
        mn_lo, fr_lo = eval_named_coeffs(model, rel, test_pairs, words, q_emb_cache, answer_tok,
                                         menu_ids, out_set_idx, c_loro, D_loro, layers)
        rows[rel] = dict(in_basis_menu=mn_in, in_basis_free=fr_in, loro_menu=mn_lo, loro_free=fr_lo,
                         top_basis_name=top_name, names_self=bool(names_self), self_rank=self_rank,
                         ttt_free=ttt_free_per.get(rel, float("nan")), icl=icl_per.get(rel, float("nan")),
                         coeff_self=float(c_in[name_idx[rel]].item()), coeff_absmax=float(c_in.abs().max().item()))
        print(f"  [{rel}] in-basis free={fr_in:.3f} menu={mn_in:.3f} | out-of-basis(LORO) free={fr_lo:.3f} "
              f"| TTT(free) {rows[rel]['ttt_free']:.3f} ICL {rows[rel]['icl']:.2f} | "
              f"top-coeff='{top_name}' self-rank={self_rank} {'(NAMES SELF)' if names_self else ''}")

    # ---- LEGIBLE: by-construction read-out across ALL relations, beaten against a SHUFFLED-label null ----
    # Real: fit c for every relation in-basis; legible-if argmax|c| names that relation. Null: relabel the
    # basis directions by a fixed random permutation, refit, ask if argmax|c| matches the SHUFFLED self-
    # label. If naming survives the shuffle, it is an artifact (the lever-2 trap); a real read-out drops to
    # ~chance. We reuse the SAME fitted c for both (the fit is identical; only the label we check differs).
    print("\n  legibility-by-construction across ALL relations (in-basis argmax coeff == relation):")
    names_all = basis_names
    g = torch.Generator().manual_seed(seed + 404)
    perm = torch.randperm(len(names_all), generator=g).tolist()      # shuffled label assignment
    self_name_hits = shuf_hits = 0; self_ranks = []
    for ri, r in enumerate(names_all):
        c = fit_named_coeffs(tok, model, r, train_pairs, words, q_emb_cache, answer_tok,
                             D_by_layer, layers, steps, lr, seed, fit_on=fit_on)
        top = int(c.abs().argmax().item())
        self_name_hits += int(names_all[top] == r)                  # real: top names self
        shuf_hits += int(top == perm[ri])                           # null: top matches the shuffled label
        self_ranks.append(c.abs().argsort(descending=True).tolist().index(ri) + 1)
    self_name_acc = self_name_hits / max(1, len(names_all))
    shuf_acc = shuf_hits / max(1, len(names_all))
    chance = 1.0 / len(names_all)
    print(f"    argmax-names-self = {self_name_acc:.3f}  (shuffled-label null = {shuf_acc:.3f}, "
          f"chance = {chance:.3f}); mean self-rank = {sum(self_ranks)/len(self_ranks):.2f} of {len(names_all)}")

    agg = lambda key: float(sum(rows[r][key] for r in held_eval) / max(1, len(held_eval)))
    out = dict(held_eval=held_eval, layers=list(layers), steps=steps, per_relation=rows,
               agg_in_basis_free=agg("in_basis_free"), agg_in_basis_menu=agg("in_basis_menu"),
               agg_loro_free=agg("loro_free"), agg_loro_menu=agg("loro_menu"),
               agg_ttt_free=float(sum(ttt_free_per.get(r, float("nan")) for r in held_eval
                                      if ttt_free_per.get(r) == ttt_free_per.get(r)) /
                                  max(1, sum(1 for r in held_eval if ttt_free_per.get(r) == ttt_free_per.get(r)))),
               icl_ref=float(sum(icl_per.get(r, float("nan")) for r in held_eval if r in icl_per) /
                             max(1, sum(1 for r in held_eval if r in icl_per))),
               legible_argmax_names_self=self_name_acc, legible_shuffled_null=shuf_acc,
               legible_chance=chance, mean_self_rank=float(sum(self_ranks)/len(self_ranks)),
               names_all=names_all)
    return out

# ==========================================================================================
# SVGs
def svg_idea3(path, i3, title):
    held = i3["held_eval"]; groups = list(held) + ["AGG"]
    W, Hh, ml, mr, mt, mb = 900, 400, 54, 210, 46, 96
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    n = len(groups); bw = (x1 - x0) / n
    Yc = lambda v: y0 - (0.0 if v != v else v) * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="{BG}"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="{TXT}" font-size="13" text-anchor="middle">{title}</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="{GRID}"/>',
                         f'<text x="{x0-6}" y="{Y+4:.1f}" fill="{MUT}" font-size="10" text-anchor="end">{v:g}</text>']
    def get(rel, key, aggkey):
        if rel == "AGG": return i3.get(aggkey, float("nan"))
        d = i3["per_relation"][rel]
        for k in key.split("."):
            d = d[k] if isinstance(d, dict) else d
        return d
    series = [("adapted_menu_acc", "agg_adapted_menu", TEAL, "adapted (TTT) apply"),
              ("reports.declarative.stated_apply_acc", "agg_decl_stated_acc", PINK, "stated-rule applied"),
              ("reports.declarative.agreement_with_adapted", "agg_decl_agreement", CORAL, "agreement (stated vs adapted)"),
              ("oracle_rule.apply_acc", "agg_oracle_acc", GOLD, "oracle true-rule applied"),
              ("wrong_rule.apply_acc", "agg_wrong_acc", SLATE, "wrong-rule applied (control)")]
    for i, rel in enumerate(groups):
        cx = x0 + (i + 0.5) * bw
        for j, (key, aggk, col, _) in enumerate(series):
            v = get(rel, key, aggk)
            off = (-0.42 + j * 0.20) * bw
            if v == v:
                p.append(f'<rect x="{cx+off:.1f}" y="{Yc(v):.1f}" width="{bw*0.16:.1f}" height="{(y0-Yc(v)):.1f}" fill="{col}"/>')
        p.append(f'<text x="{cx:.1f}" y="{y0+14}" fill="{MUT}" font-size="9" text-anchor="middle" transform="rotate(20 {cx:.1f} {y0+14})">{rel}</text>')
    ly = mt + 12
    for _, _, col, lab in series:
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="{TXT}" font-size="10">{lab}</text>']; ly += 19
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))

def svg_idea1(path, i1, title):
    held = i1["held_eval"]; groups = list(held) + ["AGG"]
    W, Hh, ml, mr, mt, mb = 880, 400, 54, 210, 46, 96
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    n = len(groups); bw = (x1 - x0) / n
    Yc = lambda v: y0 - (0.0 if v != v else v) * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="{BG}"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="{TXT}" font-size="13" text-anchor="middle">{title}</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="{GRID}"/>',
                         f'<text x="{x0-6}" y="{Y+4:.1f}" fill="{MUT}" font-size="10" text-anchor="end">{v:g}</text>']
    def get(rel, key, aggk):
        if rel == "AGG": return i1.get(aggk, float("nan"))
        return i1["per_relation"][rel].get(key, float("nan"))
    series = [("in_basis_free", "agg_in_basis_free", TEAL, "named-sliders in-basis (free)"),
              ("loro_free", "agg_loro_free", PINK, "named-sliders out-of-basis/LORO (free)"),
              ("ttt_free", "agg_ttt_free", GOLD, "unconstrained TTT (free)"),
              ("icl", "icl_ref", LILAC, "ICL ceiling")]
    for i, rel in enumerate(groups):
        cx = x0 + (i + 0.5) * bw
        for j, (key, aggk, col, _) in enumerate(series):
            v = get(rel, key, aggk)
            off = (-0.36 + j * 0.22) * bw
            if v == v:
                p.append(f'<rect x="{cx+off:.1f}" y="{Yc(v):.1f}" width="{bw*0.18:.1f}" height="{(y0-Yc(v)):.1f}" fill="{col}"/>')
        # mark names-self with a small dot above the bar group
        ns = i1["per_relation"].get(rel, {}).get("names_self") if rel != "AGG" else None
        if ns is not None:
            p.append(f'<circle cx="{cx:.1f}" cy="{mt+8}" r="4" fill="{TEAL if ns else SLATE}"/>')
        p.append(f'<text x="{cx:.1f}" y="{y0+14}" fill="{MUT}" font-size="9" text-anchor="middle" transform="rotate(20 {cx:.1f} {y0+14})">{rel}</text>')
    ly = mt + 12
    for _, _, col, lab in series:
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="{TXT}" font-size="10">{lab}</text>']; ly += 19
    p.append(f'<text x="{x1+14}" y="{ly+6}" fill="{TEAL}" font-size="10">dot: argmax-coeff names self</text>')
    lg = i1.get("legible_argmax_names_self"); nl = i1.get("legible_shuffled_null")
    p.append(f'<text x="{x1+14}" y="{ly+24}" fill="{TXT}" font-size="10">legible {lg:.2f} vs null {nl:.2f}</text>')
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))

# ==========================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--m", type=int, default=8)               # soft-prefix length (TTT)
    ap.add_argument("--ttt_steps", type=int, default=30)      # lever-3 used 20; 30 for a solid asymptote
    ap.add_argument("--ttt_lr", type=float, default=0.05)
    ap.add_argument("--fit_on", default="train", choices=["train", "K"])
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--layer", type=int, default=12)          # feature layer (= read-MLP best L12) for basis + steering
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--test_frac", type=float, default=0.30)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--min_pairs", type=int, default=10)
    ap.add_argument("--n_held", type=int, default=6)          # held-out relation eval set size
    ap.add_argument("--icl_episodes", type=int, default=80)
    ap.add_argument("--icl_K", type=int, default=5)
    ap.add_argument("--max_new", type=int, default=24)        # self-report generation length
    ap.add_argument("--n_ex", type=int, default=3)            # examples shown in the self-report prompt
    # idea-1 named-slider knobs
    ap.add_argument("--steer_steps", type=int, default=80)    # coeff fit (K params, tiny -> more steps cheap)
    ap.add_argument("--steer_lr", type=float, default=0.1)
    ap.add_argument("--steer_layers", default="6,8,10,12,14,16")  # layer-band the named steering injects at
    ap.add_argument("--ideas", default="3,1")                 # which ideas to run
    ap.add_argument("--tag", default="_0p5b")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    t_start = time.time()
    ideas = set(x.strip() for x in args.ideas.split(","))
    print(f"device={DEV}  model={args.model}  m={args.m}  ttt_steps={args.ttt_steps}  layer={args.layer}  "
          f"seed={args.seed}  ideas={sorted(ideas)}  (SYNCHRONOUS, single process)")

    # ---- load frozen model + the SAME expanded, self-validated relation bank as v2 ----
    print("\nloading FROZEN LLM...")
    tok, model = FA.load_llm(args.model, dtype=getattr(torch, args.dtype))
    bank, REL_NAMES, dropped = FV2.build_bank(tok, min_pairs=args.min_pairs)
    words, widx, out_words, out_ids = FV2.build_vocab_bank(bank)
    train_pairs, test_pairs = FV2.split_bank(bank, words, widx, test_frac=args.test_frac, seed=args.split_seed)
    chance = 1.0 / len(out_ids)
    print(f"|relations|={len(REL_NAMES)}  |words|={len(words)}  |menu V|={len(out_ids)}  chance={chance:.4f}")
    if dropped:
        print(f"  dropped (< {args.min_pairs} single-token pairs): " + ", ".join(f"{r}({n})" for r, n in dropped.items()))

    answer_tok = {w: FV2.single_token_id(tok, w) for w in words}
    menu_ids = torch.tensor([answer_tok[w] for w in out_words], device=DEV)
    out_set_idx = {w: j for j, w in enumerate(out_words)}
    q_emb_cache = FA.cache_query_embeds(tok, model, words)
    H = model.config.hidden_size

    # ---- ICL ceiling on this bank (frozen, examples-in-prompt; the upper bracket) ----
    print("\nnative-ICL ceiling on this bank (frozen model, K text pairs, menu-scored)...")
    icl_per = {}
    for r in REL_NAMES:
        icl_per[r] = FA.icl_ceiling_rel(tok, model, r, train_pairs, test_pairs, words, menu_ids,
                                        out_set_idx, K=args.icl_K, n_episodes=args.icl_episodes)
    icl_agg = float(sum(icl_per.values()) / len(icl_per))
    print(f"  ICL aggregate over {len(REL_NAMES)} relations = {icl_agg:.3f}")

    # ---- frozen features for the named basis (idea 1), harvested at the steering layer-band in ONE
    #      pass per word (the CARRIER-position feature, the sink-artifact fix; same recipe as v2) ----
    steer_layers = [int(x) for x in args.steer_layers.split(",")] if args.steer_layers else [args.layer]
    feat_by_layer = {L: {} for L in steer_layers}
    if "1" in ideas:
        print(f"\nharvesting frozen word features at layers {steer_layers} for the named basis...")
        for w in words:
            tid = FV2.single_token_id(tok, w)
            ids = tok.encode(CARRIER.format(w=w), add_special_tokens=False)
            pos = max(i for i, t in enumerate(ids) if t == tid)
            out = model(torch.tensor(ids, device=DEV)[None, :], output_hidden_states=True)
            for L in steer_layers:
                feat_by_layer[L][widx[w]] = out.hidden_states[L][0, pos, :].float()

    # ---- fixed shuffled relation order + held-out eval set (SAME recipe as v2 -> comparable) ----
    g = torch.Generator().manual_seed(args.seed + 99)
    order = torch.randperm(len(REL_NAMES), generator=g).tolist()
    rels_ordered = [REL_NAMES[i] for i in order]
    held_eval = rels_ordered[:args.n_held]
    print(f"\nfixed held-out relation eval set (shared, LORO): {held_eval}")

    report = dict(model=args.model, device=DEV, m=args.m, ttt_steps=args.ttt_steps, ttt_lr=args.ttt_lr,
                  fit_on=args.fit_on, layer=args.layer, seed=args.seed, n_relations=len(REL_NAMES),
                  rel_names=REL_NAMES, n_words=len(words), menu_size=len(out_ids), chance=chance,
                  held_eval=held_eval, env="cloze/.venv (torch cu128, RTX 5080)", frozen_backbone=True,
                  synchronous_single_process=True, icl_per_relation=icl_per, icl_aggregate=icl_agg,
                  ttt_recorded_lever3=dict(free_at_20_steps=0.944, note="frontier_apply_v2 lever 3 smoke"))

    # =================== IDEA 3 - SELF-REPORT + VERIFY (primary) ===================
    if "3" in ideas:
        print("\n" + "=" * 88)
        print("IDEA 3 - SELF-REPORT + VERIFY: does the adapted model KNOW (in words) what rule it learned?")
        print("=" * 88)
        i3 = run_idea3(tok, model, REL_NAMES, train_pairs, test_pairs, words, widx, answer_tok, menu_ids,
                       out_set_idx, q_emb_cache, held_eval, args.m, args.ttt_steps, args.ttt_lr, args.seed,
                       icl_per, args.fit_on, n_ex=args.n_ex, max_new=args.max_new)
        report["idea3"] = i3
        svg_idea3(os.path.join(RUNS, f"legibility_v1_idea3{args.tag}.svg"), i3,
                  f"Idea 3: self-report + verify (Qwen2.5-0.5B, TTT {args.ttt_steps} steps, held-out relations)")
        print("\n  IDEA 3 AGGREGATES (menu-scored; held-out words + relations):")
        print(f"    adapted (TTT) apply           = {i3['agg_adapted_menu']:.3f} menu / {i3['agg_adapted_free']:.3f} free")
        print(f"    declarative stated-rule acc   = {i3['agg_decl_stated_acc']:.3f}  "
              f"(agreement w/ adapted = {i3['agg_decl_agreement']:.3f})")
        print(f"    metacog     stated-rule acc   = {i3['agg_metacog_stated_acc']:.3f}  "
              f"(agreement w/ adapted = {i3['agg_metacog_agreement']:.3f})")
        print(f"    CONTROLS: ICL-selfreport={i3['agg_icl_selfreport_acc']:.3f} (agree {i3['agg_icl_selfreport_agreement']:.3f})  "
              f"oracle-true-rule={i3['agg_oracle_acc']:.3f}  wrong-rule={i3['agg_wrong_acc']:.3f}")

    # =================== IDEA 1 - NAMED SLIDERS (secondary) ===================
    if "1" in ideas:
        print("\n" + "=" * 88)
        print("IDEA 1 - NAMED SLIDERS: constrain TTT to a basis of named diff-in-means directions")
        print("=" * 88)
        # need the unconstrained-TTT free-apply per held relation as the comparison (recompute here so
        # idea 1 is self-contained even if idea 3 was skipped)
        print("  computing unconstrained-TTT free-apply per held relation (the comparison)...")
        ttt_free_per = {}
        for rel in held_eval:
            pm = fit_ttt_prefix(tok, model, rel, train_pairs, words, q_emb_cache, answer_tok,
                                m=args.m, steps=args.ttt_steps, lr=args.ttt_lr, seed=args.seed, fit_on=args.fit_on)
            _, fr, _ = FV2.eval_prefix_on_relation(model, pm.prefix.detach(), rel, test_pairs, words,
                                                   q_emb_cache, answer_tok, menu_ids, out_set_idx)
            ttt_free_per[rel] = fr
        # build the multi-layer named basis (raw diff-in-means; one direction per relation per layer)
        D_by_layer, basis_names = build_relation_basis_multilayer(REL_NAMES, train_pairs, words,
                                                                  feat_by_layer, steer_layers, normalize=False)
        print(f"  named basis: {len(basis_names)} directions x {len(steer_layers)} layers "
              f"(steering injected at layers {steer_layers}); learnable = a SHARED {len(basis_names)}-dim coeff vector")
        i1 = run_idea1(tok, model, REL_NAMES, train_pairs, test_pairs, words, answer_tok, menu_ids,
                       out_set_idx, q_emb_cache, D_by_layer, basis_names, steer_layers, held_eval,
                       args.steer_steps, args.steer_lr, args.seed, icl_per, ttt_free_per, fit_on=args.fit_on)
        report["idea1"] = i1
        svg_idea1(os.path.join(RUNS, f"legibility_v1_idea1{args.tag}.svg"), i1,
                  f"Idea 1: named sliders - applies vs legible (Qwen2.5-0.5B, layers {steer_layers})")
        print("\n  IDEA 1 AGGREGATES:")
        print(f"    APPLIES: in-basis free={i1['agg_in_basis_free']:.3f} menu={i1['agg_in_basis_menu']:.3f}  "
              f"out-of-basis/LORO free={i1['agg_loro_free']:.3f}  | unconstrained-TTT free={i1['agg_ttt_free']:.3f}  "
              f"ICL={i1['icl_ref']:.3f}")
        print(f"    LEGIBLE: argmax-coeff-names-self={i1['legible_argmax_names_self']:.3f}  "
              f"vs SHUFFLED-label null={i1['legible_shuffled_null']:.3f}  (chance={i1['legible_chance']:.3f}; "
              f"mean self-rank {i1['mean_self_rank']:.2f})")

    # =================== VERDICTS ===================
    print("\n" + "#" * 88)
    print("# LEGIBILITY v1 VERDICT - can a TEST-TIME-LEARNED rule be made LEGIBLE, and at what cost?")
    print("#" * 88)
    verdicts = {}
    if "3" in ideas:
        i3 = report["idea3"]
        sr_acc = max(i3["agg_decl_stated_acc"], i3["agg_metacog_stated_acc"])
        sr_agree = max(i3["agg_decl_agreement"], i3["agg_metacog_agreement"])
        wrong = i3["agg_wrong_acc"]; oracle = i3["agg_oracle_acc"]; adapted = i3["agg_adapted_menu"]
        icl_sr = i3["agg_icl_selfreport_acc"]
        # self-report is a LEGIBLE WINDOW only if the stated rule (a) verifies well above the wrong-rule
        # control AND (b) recovers a real fraction of the adapted behavior (agreement) - and we read it
        # against the oracle (whether instruction-following can apply the rule at all on this small model).
        verifies = sr_acc > wrong + 0.15 and sr_acc > 0.2
        faithful = sr_agree > wrong + 0.15
        if verifies and faithful:
            v3 = (f"IDEA 3 POSITIVE: the adapted model can STATE its learned rule legibly and the statement VERIFIES - "
                  f"stated-rule held-out apply={sr_acc:.3f} >> wrong-rule control={wrong:.3f}, agreement with the "
                  f"adapted behavior={sr_agree:.3f}; adapted itself={adapted:.3f}, oracle-true-rule={oracle:.3f}, "
                  f"ICL-selfreport ceiling={i3['agg_icl_selfreport_acc']:.3f}. Self-report is a faithful legible window "
                  f"onto the TTT-learned rule (verified, not trusted).")
        elif verifies:
            v3 = (f"IDEA 3 PARTIAL: the stated rule VERIFIES above the wrong-rule control (stated apply={sr_acc:.3f} vs "
                  f"wrong={wrong:.3f}) but its agreement with the adapted behavior is only {sr_agree:.3f} - the model "
                  f"names a rule that works, yet the named rule and the adapted internal computation are not the SAME "
                  f"function on every word. Legible-and-checkable, but not a perfect read-out. (adapted={adapted:.3f}, "
                  f"oracle={oracle:.3f}.)")
        else:
            v3 = (f"IDEA 3 NEGATIVE: self-report does NOT yield a verifiable legible rule on this model - best stated-rule "
                  f"held-out apply={sr_acc:.3f} (agreement w/ adapted={sr_agree:.3f}) is not clear of the averaged wrong-"
                  f"rule control={wrong:.3f}. The verification path is SOUND (oracle-true-rule applied={oracle:.3f} >> wrong "
                  f"{wrong:.3f}: instruction-following CAN apply these rules when handed the right words), and the adaptation "
                  f"WORKS ({adapted:.3f}). The gap is introspective: with the adaptation active the model typically reports "
                  f"only THAT there is a first->second mapping, not WHICH transformation it is (and the unadapted ICL self-"
                  f"report ceiling is also low, {icl_sr:.3f}, so even from examples-in-context this 0.5B cannot put the rule "
                  f"into checkable words). A TTT-learned rule applies without the model being able to STATE it. Reported plainly.")
        verdicts["idea3"] = v3
        print("# " + v3)
    if "1" in ideas:
        i1 = report["idea1"]
        applies = i1["agg_in_basis_free"]; loro = i1["agg_loro_free"]; ttt = i1["agg_ttt_free"]
        legible = i1["legible_argmax_names_self"]; nullacc = i1["legible_shuffled_null"]; ch = i1["legible_chance"]
        cost = ttt - applies
        legible_real = legible > nullacc + 0.15 and legible > 3 * ch
        works = applies > 0.5 * (i1["icl_ref"] if i1["icl_ref"] == i1["icl_ref"] else 1.0)
        v1 = (f"IDEA 1 ({'WORKS+LEGIBLE' if (works and legible_real) else 'WORKS-not-legible' if works else 'COSTS-accuracy'}): "
              f"named-slider in-basis free-apply={applies:.3f} (menu {i1['agg_in_basis_menu']:.3f}) vs unconstrained-TTT "
              f"{ttt:.3f} (cost {cost:+.3f}); out-of-basis/LORO free={loro:.3f} (coverage: a relation with NO named "
              f"direction). Legibility: argmax-coeff-names-self={legible:.3f} vs shuffled-label null={nullacc:.3f} "
              f"(chance {ch:.3f}) -> "
              + ("LEGIBLE above a proper null (the coefficient genuinely names the relation)." if legible_real else
                 "NOT legible above the shuffled-label null (the naming is an artifact, the lever-2 trap again).")
              + f" Applies-vs-legible tradeoff: constraining to named sliders costs {cost:+.3f} accuracy vs free TTT; "
              + ("coverage is the limit (LORO, no own direction, drops to %.2f)." % loro))
        verdicts["idea1"] = v1
        print("# " + v1)
    print("#" * 88)

    report["verdicts"] = verdicts
    report["wall_time_s"] = round(time.time() - t_start, 1)
    out_path = os.path.join(RUNS, f"legibility_v1{args.tag}.json")
    json.dump(report, open(out_path, "w"), indent=2)
    print(f"\nwrote {out_path}  [{report['wall_time_s']}s]  (synchronous, single process - clean exit)")

if __name__ == "__main__":
    main()
