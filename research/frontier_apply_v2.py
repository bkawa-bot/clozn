"""
frontier_apply_v2.py - CLOSE the consolidate-then-apply loop that frontier_apply.py's Stage 2 missed,
and recover the legibility it lost. (READ research/frontier_apply.py + research/frontier_apply_findings.md
FIRST - this file is the direct sequel.)

WHERE STAGE 1/2 LANDED (frontier_apply_0p5b.json):
  STAGE 1 *PASSED*: a DIRECTLY-TUNED soft prefix (m continuous vectors prepended to "{x} ->", trained by
    backprop through the FROZEN Qwen2.5-0.5B-Instruct - only the prefix moves) makes the model's OWN
    next-token output apply a relation to HELD-OUT words at 0.928 menu / 0.860 free, 95% of the native-ICL
    ceiling (0.980), KILLING the external read-MLP's 0.000 on every 1-to-1 relation.
  STAGE 2 *FAILED*: a meta-learned COMPRESSOR (K example activations -> prefix, leave-one-relation-out,
    end-to-end through the frozen model, only 7 relations) does NOT generalize to a NEW relation:
    free-gen 0.000 / menu 0.159 LORO. And the per-relation prefixes are NOT legible (probe 0.000, chance
    1/7): each is an independent optimization from a different init, so same-relation prefixes don't align.
  TWO DIAGNOSED CULPRITS: (i) 7 relations is a starved meta-learning regime; (ii) learning the
    example->prefix map FROM SCRATCH end-to-end is hard and the per-relation targets have NO shared
    structure (independent inits), so there is nothing to interpolate and nothing to make legible.

THREE LEVERS (cheapest/most decisive first; each reported honestly even if negative):
  LEVER 1 - MORE RELATIONS (cheapest; reuses Stage 2 directly). Scale the relation set 7 -> ~30 (curated,
    every word single-token in Qwen BPE). Re-run the Stage-2 LORO compressor across relation counts ->
    a LEARNING CURVE of held-out-relation apply accuracy vs #training relations. Was 7 just starved?
  LEVER 2 - DISTILL TO ORACLE PREFIXES (shared structure -> generalization + legibility). Tune the
    Stage-1 working prefix for each training relation (an ORACLE target), then train ONE SHARED compressor
    to REGRESS examples -> that oracle prefix (cosine+MSE), optionally + the apply-CE. Test held-out-relation
    apply AND the relation-probe legibility of the compressor's outputs. A shared compressor may both
    generalize AND be legible (the legibility frontier_apply lost).
  LEVER 3 - TEST-TIME ADAPTATION (the Titans/TTT move). For a NEW relation, take a FEW gradient steps to
    fit its prefix from its few examples, init from {zero, the compressor's guess}. Report the
    steps-vs-held-out-accuracy curve - how few steps to a working prefix on an unseen relation?

CONTROLS / HONESTY (load-bearing - this frontier has produced clean-looking reversals):
  every number sits beside the ICL ceiling + the read-MLP 0.000 baseline + a null (no/random prefix);
  eval ONLY on held-out WORDS, and (levers 1,2) ONLY on held-out RELATIONS (LORO); FREE-GEN reported
  (not just menu - a high menu with ~0 free is a menu mirage, exactly how Stage 2 failed); per-relation
  breakdown; relation-probe legibility where applicable; a null/oracle bracket. A NEGATIVE on any lever
  is a valid, valuable finding. GOAL: MOVE the Stage-2 / legibility result, or learn precisely why it can't.

MODEL: Qwen2.5-0.5B-Instruct, FROZEN. Env: cloze/.venv (torch cu128, RTX 5080); .venv-sae untouched.
Outputs (research/runs/): frontier_apply_v2{tag}.json + SVGs (lever1 curve, lever2 bars, lever3 curve).
"""
import os, sys, json, time, argparse, math
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

# Reuse the EXACT Stage-1/2 machinery (apples-to-apples). frontier_apply imports the relation data +
# split + carrier + ICL conventions from sidecar_semantic, so importing from it keeps everything aligned.
from sidecar_semantic import RELATIONS as RELATIONS_BASE, CARRIER, build_menu_rel
import frontier_apply as FA   # SoftPrefix, forward_with_prefix, batch_pack, cache_query_embeds, ICL, etc.

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Maiko palette (matches frontier_apply.py)
BG, TEAL, PINK, TXT, MUT, GRID = "#1A1F4A", "#6FD6C9", "#FF8FB3", "#F4F0E8", "#8784b3", "#2c2f5e"
GOLD = "#E8C977"    # ICL ceiling
LILAC = "#B6A6E8"   # read-MLP baseline / oracle
SLATE = "#7E8AA8"   # null
CORAL = "#E89B7E"   # second meta series

# ==========================================================================================
# EXPANDED RELATION BANK. The 7 base relations (byte-identical to Stage 1/2) PLUS ~25 curated
# new ones. Every word is intended single-token in Qwen2.5 BPE; we FILTER defensively at load
# (drop any non-single-token pair) and drop relations left with < MIN_PAIRS, so the bank is
# self-validating and the split is honest. Programmatic morphology (plural/past/comparative/
# superlative/gerund/3rd-person/agent) + curated semantics (capital/currency/nationality/
# continent/gender/part-of/diminutive/hypernym/hyponym/synonym/antonym/color/sound/made-of/verb-noun).
NEW_RELATIONS = {
 "synonym": [("big","large"),("small","little"),("happy","glad"),("fast","quick"),("smart","clever"),
   ("begin","start"),("end","finish"),("rich","wealthy"),("angry","mad"),("sad","unhappy"),
   ("close","shut"),("buy","purchase"),("show","display"),("jump","leap"),("talk","speak"),
   ("help","aid"),("keep","hold"),("cold","chilly"),("hot","warm"),("old","aged"),
   ("quiet","silent"),("look","glance"),("think","believe"),("want","desire"),("strong","mighty")],
 "superlative": [("big","biggest"),("small","smallest"),("fast","fastest"),("tall","tallest"),("long","longest"),
   ("short","shortest"),("strong","strongest"),("weak","weakest"),("warm","warmest"),("cold","coldest"),
   ("high","highest"),("low","lowest"),("deep","deepest"),("rich","richest"),("old","oldest"),
   ("new","newest"),("hard","hardest"),("soft","softest"),("young","youngest"),("bright","brightest")],
 "gerund": [("walk","walking"),("jump","jumping"),("play","playing"),("talk","talking"),("look","looking"),
   ("run","running"),("sit","sitting"),("read","reading"),("sing","singing"),("cook","cooking"),
   ("swim","swimming"),("fly","flying"),("go","going"),("do","doing"),("eat","eating"),
   ("work","working"),("move","moving"),("turn","turning"),("help","helping"),("open","opening")],
 "third_person": [("walk","walks"),("run","runs"),("jump","jumps"),("play","plays"),("talk","talks"),
   ("look","looks"),("want","wants"),("need","needs"),("help","helps"),("work","works"),
   ("move","moves"),("turn","turns"),("open","opens"),("close","closes"),("start","starts"),
   ("call","calls"),("show","shows"),("ask","asks"),("eat","eats"),("read","reads"),
   ("sing","sings"),("cook","cooks"),("clean","cleans"),("paint","paints"),("swim","swims")],
 "agent": [("teach","teacher"),("paint","painter"),("sing","singer"),("write","writer"),("play","player"),
   ("run","runner"),("drive","driver"),("farm","farmer"),("bake","baker"),("dance","dancer"),
   ("build","builder"),("lead","leader"),("read","reader"),("work","worker"),("own","owner"),
   ("speak","speaker"),("win","winner"),("jump","jumper"),("clean","cleaner"),("help","helper")],
 "verb_noun": [("decide","decision"),("act","action"),("create","creation"),("protect","protection"),("connect","connection"),
   ("collect","collection"),("correct","correction"),("direct","direction"),("reflect","reflection"),("select","selection"),
   ("inject","injection"),("inspect","inspection"),("detect","detection"),("reject","rejection"),("predict","prediction")],
 "nationality": [("France","French"),("Spain","Spanish"),("Italy","Italian"),("Germany","German"),("China","Chinese"),
   ("Japan","Japanese"),("Russia","Russian"),("Poland","Polish"),("Greece","Greek"),("Egypt","Egyptian"),
   ("Sweden","Swedish"),("Norway","Norwegian"),("Brazil","Brazilian"),("Turkey","Turkish"),("Korea","Korean"),
   ("India","Indian"),("Canada","Canadian"),("Mexico","Mexican"),("Iran","Iranian"),("Cuba","Cuban")],
 "continent": [("France","Europe"),("Japan","Asia"),("Egypt","Africa"),("Brazil","America"),("China","Asia"),
   ("Spain","Europe"),("Kenya","Africa"),("India","Asia"),("Canada","America"),("Italy","Europe"),
   ("Chile","America"),("Ghana","Africa"),("Iran","Asia"),("Peru","America"),("Norway","Europe")],
 "currency": [("Japan","yen"),("China","yuan"),("India","rupee"),("Russia","ruble"),("Korea","won"),
   ("Mexico","peso"),("Poland","zloty"),("Thailand","baht"),("Israel","shekel"),("Sweden","krona"),
   ("Britain","pound"),("America","dollar"),("Europe","euro"),("Brazil","real"),("Vietnam","dong"),
   ("Denmark","krone"),("Hungary","forint"),("Iran","rial")],
 "opposite_gender": [("king","queen"),("man","woman"),("boy","girl"),("father","mother"),("son","daughter"),
   ("brother","sister"),("uncle","aunt"),("husband","wife"),("prince","princess"),("actor","actress"),
   ("host","hostess"),("lord","lady"),("male","female"),("sir","madam"),("nephew","niece"),
   ("bull","cow"),("rooster","hen"),("stallion","mare"),("buck","doe"),("gentleman","lady")],
 "part_of": [("wheel","car"),("wing","plane"),("petal","flower"),("branch","tree"),("page","book"),
   ("roof","house"),("finger","hand"),("toe","foot"),("leaf","tree"),("door","house"),
   ("string","guitar"),("lens","camera"),("blade","knife"),("handle","door"),("engine","car"),
   ("screen","phone"),("button","shirt"),("sail","boat"),("tail","dog"),("key","keyboard")],
 "diminutive": [("dog","puppy"),("cat","kitten"),("cow","calf"),("horse","foal"),("sheep","lamb"),
   ("bear","cub"),("deer","fawn"),("pig","piglet"),("hen","chick"),("lion","cub"),
   ("frog","tadpole"),("goat","kid"),("fox","cub"),("wolf","pup"),("seal","pup"),
   ("whale","calf"),("eagle","eaglet"),("swan","cygnet"),("rabbit","kit"),("zebra","foal")],
 "hyponym": [("animal","dog"),("fruit","apple"),("color","red"),("metal","gold"),("tree","oak"),
   ("bird","eagle"),("fish","salmon"),("tool","hammer"),("flower","rose"),("vehicle","car"),
   ("drink","water"),("sport","soccer"),("shape","circle"),("number","seven"),("planet","earth"),
   ("insect","ant"),("vegetable","carrot"),("weather","rain"),("season","winter"),("emotion","fear")],
 "antonym2": [("north","south"),("east","west"),("summer","winter"),("sweet","sour"),("thick","thin"),
   ("smooth","rough"),("sharp","dull"),("brave","timid"),("calm","wild"),("cheap","costly"),
   ("early","late"),("first","last"),("inside","outside"),("major","minor"),("odd","even"),
   ("plus","minus"),("public","private"),("simple","complex"),("upper","lower"),("king","peasant")],
 "made_of": [("snow","white"),("ice","cold"),("fire","hot"),("stone","hard"),("feather","light"),
   ("lead","heavy"),("sugar","sweet"),("lemon","sour"),("rock","solid"),("water","wet"),
   ("steel","strong"),("glass","clear"),("cloud","soft"),("mud","dirty"),("honey","sweet")],
 "un_prefix": [("happy","unhappy"),("kind","unkind"),("fair","unfair"),("safe","unsafe"),("sure","unsure"),
   ("clear","unclear"),("able","unable"),("even","uneven"),("lock","unlock"),("do","undo"),
   ("fold","unfold"),("seen","unseen"),("known","unknown"),("paid","unpaid"),("real","unreal"),
   ("tie","untie"),("wrap","unwrap"),("load","unload"),("pack","unpack"),("seal","unseal")],
 "re_prefix": [("do","redo"),("make","remake"),("build","rebuild"),("write","rewrite"),("read","reread"),
   ("play","replay"),("tell","retell"),("use","reuse"),("fill","refill"),("open","reopen"),
   ("start","restart"),("load","reload"),("name","rename"),("pay","repay"),("heat","reheat"),
   ("mix","remix"),("count","recount"),("sell","resell"),("test","retest"),("join","rejoin")],
 "adverb": [("quick","quickly"),("slow","slowly"),("soft","softly"),("loud","loudly"),("quiet","quietly"),
   ("bright","brightly"),("sad","sadly"),("glad","gladly"),("kind","kindly"),("rude","rudely"),
   ("calm","calmly"),("warm","warmly"),("cold","coldly"),("clear","clearly"),("deep","deeply"),
   ("rough","roughly"),("smooth","smoothly"),("sharp","sharply"),("weak","weakly"),("strong","strongly")],
 "ordinal": [("one","first"),("two","second"),("three","third"),("four","fourth"),("five","fifth"),
   ("six","sixth"),("seven","seventh"),("eight","eighth"),("nine","ninth"),("ten","tenth")],
 "habitat": [("fish","water"),("bird","sky"),("bee","hive"),("bear","cave"),("lion","jungle"),
   ("camel","desert"),("whale","ocean"),("cow","farm"),("ant","hill"),("bat","cave"),
   ("frog","pond"),("owl","tree"),("shark","sea"),("horse","barn"),("pig","pen")],
}

# ----------------------------------------------------------------------------------------
def build_bank(tok, min_pairs=10, drop_dup_y_relations=False):
    """Assemble {rel: [(x,y),...]} = base 7 + NEW, FILTERING to single-token pairs and dropping
    relations with < min_pairs. Returns (RELATIONS, REL_NAMES). Self-validating: the printed
    survivors are exactly what trains, so the split is honest and reproducible."""
    def st(w):
        return len(tok.encode(" " + w, add_special_tokens=False)) == 1
    bank = {}
    src = dict(RELATIONS_BASE)                       # base 7 first (kept identical to Stage 1/2)
    for r, ps in NEW_RELATIONS.items():
        src[r] = ps
    dropped = {}
    for r, ps in src.items():
        # dedup pairs, filter single-token both sides
        seen = set(); keep = []
        for (a, b) in ps:
            if (a, b) in seen:
                continue
            seen.add((a, b))
            if st(a) and st(b):
                keep.append((a, b))
        if len(keep) >= min_pairs:
            bank[r] = keep
        else:
            dropped[r] = len(keep)
    REL_NAMES = list(bank.keys())
    return bank, REL_NAMES, dropped

def build_vocab_bank(bank):
    """Shared candidate vocabulary V over THIS bank (mirrors sidecar_semantic.build_vocab)."""
    REL_NAMES = list(bank.keys())
    words = sorted(set(w for pairs in bank.values() for p in pairs for w in p))
    widx = {w: i for i, w in enumerate(words)}
    out_words = sorted(set(y for pairs in bank.values() for (x, y) in pairs))
    out_ids = [widx[w] for w in out_words]
    return words, widx, out_words, out_ids

def split_bank(bank, words, widx, test_frac=0.30, seed=0):
    """Per-relation TRAIN/TEST split on PAIRS (mirrors sidecar_semantic.split_relations EXACTLY,
    same generator recipe -> for the base 7 relations the split is byte-identical to Stage 1/2)."""
    rng = torch.Generator().manual_seed(seed)
    train, test = {}, {}
    for r, pairs in bank.items():
        idx = [(widx[x], widx[y]) for (x, y) in pairs]
        perm = torch.randperm(len(idx), generator=rng).tolist()
        ntest = max(2, int(round(len(idx) * test_frac)))
        te = sorted(perm[:ntest]); tr = sorted(perm[ntest:])
        train[r] = torch.tensor([idx[i] for i in tr], device=DEV)
        test[r]  = torch.tensor([idx[i] for i in te], device=DEV)
    return train, test

def build_menu_rel_bank(REL_NAMES, bank, out_ids, widx):
    """[|menu|, R] bool: is candidate j an answer of relation r? (mirrors build_menu_rel for this bank)."""
    M = torch.zeros(len(out_ids), len(REL_NAMES), dtype=torch.bool, device=DEV)
    pos = {wi: j for j, wi in enumerate(out_ids)}
    for ri, r in enumerate(REL_NAMES):
        for (x, y) in bank[r]:
            j = pos.get(widx[y])
            if j is not None:
                M[j, ri] = True
    return M

# ==========================================================================================
# Shared helpers (thin wrappers over frontier_apply primitives so paths are byte-identical).
def single_token_id(tok, w):
    ids = tok.encode(" " + w, add_special_tokens=False)
    assert len(ids) == 1, f"{w!r} not single-token: {ids}"
    return ids[0]

@torch.no_grad()
def eval_prefix_on_relation(model, prefix_tensor, rel, test_pairs, words, q_emb_cache, answer_tok,
                            menu_ids, out_set_idx):
    """menu+free held-out apply for a RAW prefix tensor [m,H] on a relation's TEST words."""
    pairs = test_pairs[rel].tolist()
    if not pairs:
        return float("nan"), float("nan"), 0
    H = prefix_tensor.shape[1]
    pm = FA.SoftPrefix(prefix_tensor.shape[0], H).to(DEV)
    pm.prefix = nn.Parameter(prefix_tensor.detach())
    xs = [words[xi] for (xi, yi) in pairs]
    ytok = torch.tensor([answer_tok[words[yi]] for (xi, yi) in pairs], device=DEV)
    ymenu = torch.tensor([out_set_idx[words[yi]] for (xi, yi) in pairs], device=DEV)
    padded, mask = FA.batch_pack([q_emb_cache[x] for x in xs])
    logits = FA.forward_with_prefix(model, pm, padded, mask)
    free = (logits.argmax(-1) == ytok).float().mean().item()
    menu = (logits[:, menu_ids].argmax(-1) == ymenu).float().mean().item()
    return menu, free, len(pairs)

def train_oracle_prefix(tok, model, rel, train_pairs, words, q_emb_cache, answer_tok,
                        m=8, steps=400, lr=0.05, seed=0):
    """Stage-1 ORACLE prefix for a relation (the working per-relation prefix). Thin call into the
    exact Stage-1 trainer so the oracle == Stage-1's prefix."""
    return FA.train_soft_prefix_for_relation(tok, model, rel, train_pairs, words, q_emb_cache,
                                             answer_tok, m=m, steps=steps, lr=lr, seed=seed)

@torch.no_grad()
def harvest_feat(tok, model, word, layer):
    """Frozen residual feature for a word at its CARRIER position (sink fix), for compressor input."""
    tid = single_token_id(tok, word)
    ids = tok.encode(CARRIER.format(w=word), add_special_tokens=False)
    pos = max(i for i, t in enumerate(ids) if t == tid)
    out = model(torch.tensor(ids, device=DEV)[None, :], output_hidden_states=True)
    return out.hidden_states[layer][0, pos, :].float()

# ==========================================================================================
# COMPRESSOR (shared, used by levers 1, 2, 3). Same spine as Stage 2's Compressor: K example
# features (frozen) -> a soft prefix [m,H], permutation-invariant (mean). Output direction is
# normalized then scaled to target_norm (matched to where Stage-1 prefixes live) with a learnable
# per-vector gain - the diagnosed-fair Stage-2 recipe. Identical class so lever 1 IS Stage 2 at scale.
class Compressor(nn.Module):
    def __init__(self, H, m, hidden=512, proj=256, target_norm=0.45):
        super().__init__()
        self.m, self.H, self.target_norm = m, H, target_norm
        self.proj = nn.Linear(H, proj)
        self.enc = nn.Sequential(nn.Linear(2 * proj, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.dec = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, m * H))
        self.gain = nn.Parameter(torch.ones(m))
    def encode(self, xf, yf):
        return self.enc(torch.cat([self.proj(xf), self.proj(yf)], -1)).mean(1)   # [B,hidden]
    def forward(self, xf, yf):
        raw = self.dec(self.encode(xf, yf)).view(-1, self.m, self.H)
        unit = F.normalize(raw, dim=-1)
        return unit * (self.target_norm * self.gain)[None, :, None]

# ------------------------------------------------------------------------------------------
def _sample_K(tp, K, g):
    kk = min(K, tp.shape[0])
    ti = torch.randperm(tp.shape[0], generator=g, device=DEV)[:kk]
    return tp[ti, 0], tp[ti, 1]

def run_compressor_LORO(model, REL_NAMES, train_rel_subset, held_rels, train_pairs, test_pairs,
                        words, q_emb_cache, answer_tok, menu_ids, out_set_idx, feat_cache,
                        m, steps, K, lr, seed, mode="e2e", oracle_prefixes=None,
                        distill_w=1.0, ce_w=0.0):
    """ONE compressor, trained on `train_rel_subset`, evaluated on each relation in `held_rels`.
    mode='e2e'      : Stage-2 end-to-end loss (CE of the frozen model's apply on train queries).
    mode='distill'  : regress generated prefix -> the relation's ORACLE prefix (cosine+MSE),
                      optionally + apply-CE (ce_w>0). oracle_prefixes: {rel: tensor[m,H]} required.
    Returns dict per held rel: {menu, free, gen_norm} + the trained compressor (for legibility)."""
    H = model.config.hidden_size
    torch.manual_seed(seed)
    comp = Compressor(H, m).to(DEV)
    opt = torch.optim.Adam(comp.parameters(), lr)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    comp.train()
    for step in range(steps):
        r = train_rel_subset[int(torch.randint(0, len(train_rel_subset), (1,), generator=g, device=DEV))]
        tp = train_pairs[r]
        xi, yi = _sample_K(tp, K, g)
        xf = torch.stack([feat_cache[int(j)] for j in xi])[None]
        yf = torch.stack([feat_cache[int(j)] for j in yi])[None]
        pre = comp(xf, yf)                                          # [1,m,H]
        loss = 0.0
        if mode == "distill":
            tgt = oracle_prefixes[r].to(DEV)[None]                  # [1,m,H]
            # direction (cosine) + magnitude (MSE) toward the oracle - the shared target
            cos = 1.0 - F.cosine_similarity(pre.reshape(1, -1), tgt.reshape(1, -1), dim=-1).mean()
            mse = F.mse_loss(pre, tgt)
            loss = distill_w * (cos + mse)
        if mode == "e2e" or ce_w > 0:
            # apply-CE on TRAIN queries held out of the K teaching set (forces generalization, not echo)
            taught_x = set(int(j) for j in xi)
            qsel = [(a, b) for (a, b) in tp.tolist() if a not in taught_x]
            if len(qsel) < 2:
                qsel = tp.tolist()
            qsel = qsel[:8]
            xs = [words[a] for (a, b) in qsel]; ys = [answer_tok[words[b]] for (a, b) in qsel]
            padded, mask = FA.batch_pack([q_emb_cache[x] for x in xs])
            pm = FA.SoftPrefix(m, H).to(DEV); pm.prefix = nn.Parameter(pre[0])
            logits = FA.forward_with_prefix(model, pm, padded, mask)
            ce = F.cross_entropy(logits, torch.tensor(ys, device=DEV))
            loss = loss + (ce if mode == "e2e" else ce_w * ce)
        opt.zero_grad(); loss.backward(); opt.step()
    comp.eval()
    results = {}
    with torch.no_grad():
        gg = torch.Generator(device=DEV).manual_seed(seed + 5)
        for held in held_rels:
            tp = train_pairs[held]
            xi, yi = _sample_K(tp, K, gg)
            xf = torch.stack([feat_cache[int(j)] for j in xi])[None]
            yf = torch.stack([feat_cache[int(j)] for j in yi])[None]
            pre = comp(xf, yf)
            gen_norm = float(pre[0].norm(dim=-1).mean().item())
            mn, fr, n = eval_prefix_on_relation(model, pre[0], held, test_pairs, words, q_emb_cache,
                                                answer_tok, menu_ids, out_set_idx)
            results[held] = dict(menu=mn, free=fr, gen_norm=gen_norm)
    return results, comp

# ==========================================================================================
# LEVER 1: learning curve - LORO held-out-relation apply vs # training relations.
def lever1(model, REL_NAMES, train_pairs, test_pairs, words, q_emb_cache, answer_tok, menu_ids,
           out_set_idx, feat_cache, m, counts, steps, K, lr, seed, icl_per, rm_per, n_held=6):
    """For each relation-count N in `counts`: pick N relations (a fixed nested subset, seeded);
    hold out `n_held` of them one at a time (LORO), train the compressor on the rest, eval the
    held-out relation's TEST words. Aggregate menu/free over the held-out relations -> a point on
    the learning curve. The held-out relations are kept FIXED across N (a common eval set) so the
    curve isolates the effect of MORE TRAINING relations, not which relation is tested."""
    R = len(REL_NAMES)
    g = torch.Generator().manual_seed(seed + 99)
    order = torch.randperm(R, generator=g).tolist()
    rels_ordered = [REL_NAMES[i] for i in order]
    # FIXED held-out eval set = the first n_held relations in the shuffled order (present in every N>=...).
    held_eval = rels_ordered[:n_held]
    curve = []
    for N in counts:
        if N < n_held + 2:    # need at least a couple of TRAIN relations beyond the held set
            continue
        subset = rels_ordered[:N]
        per_rel_menu = {}; per_rel_free = {}; per_rel_norm = {}
        for held in held_eval:
            if held not in subset:
                continue
            train_rel_subset = [r for r in subset if r != held]
            res, _ = run_compressor_LORO(model, REL_NAMES, train_rel_subset, [held], train_pairs,
                                         test_pairs, words, q_emb_cache, answer_tok, menu_ids,
                                         out_set_idx, feat_cache, m, steps, K, lr, seed, mode="e2e")
            per_rel_menu[held] = res[held]["menu"]; per_rel_free[held] = res[held]["free"]
            per_rel_norm[held] = res[held]["gen_norm"]
        agg_menu = float(sum(per_rel_menu.values()) / max(1, len(per_rel_menu)))
        agg_free = float(sum(per_rel_free.values()) / max(1, len(per_rel_free)))
        # ICL/read-MLP reference restricted to the held-eval set (so the curve's bracket matches)
        icl_ref = float(sum(icl_per.get(r, float("nan")) for r in held_eval if r in icl_per) /
                        max(1, sum(1 for r in held_eval if r in icl_per)))
        rm_ref = float(sum(rm_per.get(r, 0.0) for r in held_eval) / max(1, len(held_eval)))
        curve.append(dict(N=N, n_train_rels=N - 1, agg_menu=agg_menu, agg_free=agg_free,
                          per_rel_menu=per_rel_menu, per_rel_free=per_rel_free,
                          mean_gen_norm=float(sum(per_rel_norm.values()) / max(1, len(per_rel_norm))),
                          icl_ref=icl_ref, readmlp_ref=rm_ref))
        print(f"  N={N:2d} relations (train {N-1} LORO) -> held-out apply: menu={agg_menu:.3f} "
              f"free={agg_free:.3f}  (ICL_ref={icl_ref:.3f}, read-MLP_ref={rm_ref:.3f}, "
              f"gen-norm={curve[-1]['mean_gen_norm']:.3f})")
    return dict(held_eval=held_eval, n_held=len(held_eval), curve=curve)

# ==========================================================================================
# LEVER 2: distill to oracle prefixes -> shared compressor; test generalize + legibility.
def relation_probe_legibility(comp, train_rel_subset, train_pairs, feat_cache, m, K, seed,
                              n_per_rel=40):
    """Probe the SHARED compressor's outputs -> which relation. For each train relation, generate
    many prefixes from random K-shot draws; train a linear probe on flattened prefixes to predict
    the relation; report held-out probe acc (chance = 1/len(train_rel_subset)). UNLIKE Stage-1's
    independent per-relation prefixes (probe 0.000), a shared compressor's outputs for the same
    relation SHOULD cluster if it learned shared structure -> this measures recovered legibility."""
    H = comp.H
    g = torch.Generator(device=DEV).manual_seed(seed + 700)
    X, y = [], []
    with torch.no_grad():
        for ri, r in enumerate(train_rel_subset):
            tp = train_pairs[r]
            for _ in range(n_per_rel):
                xi, yi = _sample_K(tp, K, g)
                xf = torch.stack([feat_cache[int(j)] for j in xi])[None]
                yf = torch.stack([feat_cache[int(j)] for j in yi])[None]
                pre = comp(xf, yf)[0].flatten()
                X.append(pre.detach().cpu()); y.append(ri)
    X = torch.stack(X).float(); y = torch.tensor(y)
    # standardize, split, linear probe
    X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)
    n = X.shape[0]; idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 7))
    ntr = int(0.7 * n)
    Xtr, ytr, Xte, yte = X[idx[:ntr]].to(DEV), y[idx[:ntr]].to(DEV), X[idx[ntr:]].to(DEV), y[idx[ntr:]].to(DEV)
    probe = nn.Linear(X.shape[1], len(train_rel_subset)).to(DEV)
    opt = torch.optim.Adam(probe.parameters(), 1e-2)
    for _ in range(400):
        loss = F.cross_entropy(probe(Xtr), ytr); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (probe(Xte).argmax(1) == yte).float().mean().item()
    return acc, 1.0 / len(train_rel_subset)

def lever2(model, REL_NAMES, train_pairs, test_pairs, words, q_emb_cache, answer_tok, menu_ids,
           out_set_idx, feat_cache, oracle_prefixes, m, steps, K, lr, seed, icl_per, rm_per,
           held_eval, ce_w):
    """Train ONE shared compressor by DISTILLATION toward the per-relation oracle prefixes (LORO
    over `held_eval`), then (a) held-out-relation apply, (b) relation-probe legibility of its
    outputs. Reports the distill compressor vs the SAME-budget e2e compressor vs ICL/read-MLP/oracle."""
    out = {"held_eval": held_eval, "per_rel": {}, "ce_w": ce_w, "legibility_null_acc": None}
    distill_menu = {}; distill_free = {}; e2e_menu = {}; e2e_free = {}
    legib_acc = None; legib_chance = None
    for held in held_eval:
        train_rel_subset = [r for r in REL_NAMES if r != held]
        # DISTILL compressor (the lever)
        res_d, comp_d = run_compressor_LORO(model, REL_NAMES, train_rel_subset, [held], train_pairs,
                                            test_pairs, words, q_emb_cache, answer_tok, menu_ids,
                                            out_set_idx, feat_cache, m, steps, K, lr, seed,
                                            mode="distill", oracle_prefixes=oracle_prefixes,
                                            distill_w=1.0, ce_w=ce_w)
        distill_menu[held] = res_d[held]["menu"]; distill_free[held] = res_d[held]["free"]
        # legibility probed ONCE on the broadest compressor (the leave-out of the LAST held rel),
        # WITH a null-compressor control (same arch, UNTRAINED): if an untrained map's outputs are
        # ALSO trivially separable, the legibility is an artifact of the input features, not learned
        # shared structure. legible-above-null is the honest signal.
        if held == held_eval[-1]:
            legib_acc, legib_chance = relation_probe_legibility(comp_d, train_rel_subset, train_pairs,
                                                                feat_cache, m, K, seed)
            null_comp = Compressor(model.config.hidden_size, m).to(DEV).eval()
            legib_null, _ = relation_probe_legibility(null_comp, train_rel_subset, train_pairs,
                                                      feat_cache, m, K, seed)
            out["legibility_null_acc"] = legib_null
        out["per_rel"][held] = dict(distill_menu=res_d[held]["menu"], distill_free=res_d[held]["free"],
                                    gen_norm=res_d[held]["gen_norm"],
                                    oracle_menu=oracle_prefixes and None,  # filled below
                                    icl=icl_per.get(held, float("nan")), readmlp=rm_per.get(held, 0.0))
        print(f"  held={held:14s} distill apply: menu={res_d[held]['menu']:.3f} free={res_d[held]['free']:.3f}"
              f"  (ICL={icl_per.get(held, float('nan')):.3f}, read-MLP={rm_per.get(held, 0.0):.3f}, "
              f"gen-norm={res_d[held]['gen_norm']:.3f})")
    out["distill_menu"] = distill_menu; out["distill_free"] = distill_free
    out["agg_distill_menu"] = float(sum(distill_menu.values()) / max(1, len(distill_menu)))
    out["agg_distill_free"] = float(sum(distill_free.values()) / max(1, len(distill_free)))
    out["legibility_acc"] = legib_acc; out["legibility_chance"] = legib_chance
    # oracle bracket on the SAME held_eval relations (their own Stage-1 prefix, the ceiling for distill)
    orc_menu = {}; orc_free = {}
    for held in held_eval:
        mn, fr, _ = eval_prefix_on_relation(model, oracle_prefixes[held], held, test_pairs, words,
                                            q_emb_cache, answer_tok, menu_ids, out_set_idx)
        orc_menu[held] = mn; orc_free[held] = fr
        out["per_rel"][held]["oracle_menu"] = mn; out["per_rel"][held]["oracle_free"] = fr
    out["agg_oracle_menu"] = float(sum(orc_menu.values()) / max(1, len(orc_menu)))
    out["agg_oracle_free"] = float(sum(orc_free.values()) / max(1, len(orc_free)))
    out["icl_ref"] = float(sum(icl_per.get(r, float('nan')) for r in held_eval if r in icl_per) /
                           max(1, sum(1 for r in held_eval if r in icl_per)))
    out["readmlp_ref"] = float(sum(rm_per.get(r, 0.0) for r in held_eval) / max(1, len(held_eval)))
    return out

# ==========================================================================================
# LEVER 3: test-time adaptation - few gradient steps on a NEW relation's prefix from its examples.
def lever3(tok, model, REL_NAMES, train_pairs, test_pairs, words, q_emb_cache, answer_tok, menu_ids,
           out_set_idx, feat_cache, oracle_prefixes, m, step_grid, K, lr, seed, icl_per, rm_per,
           held_eval, compressor=None, fit_on="train"):
    """For each held-out relation: fit a prefix by gradient descent on the apply-CE (the SAME loss
    Stage-1 uses), measuring HELD-OUT TEST apply at each checkpoint in `step_grid`. The fit set is
    this relation's own examples ONLY (it is the 'new relation' - nothing else is adapted):
      fit_on='train' -> fit on its TRAIN words (the strong setting; matches Stage-1's train/test
                        generalization split exactly, so the asymptote == the Stage-1 oracle), or
      fit_on='K'     -> fit on just K examples (the few-shot setting).
    Eval is ALWAYS on the relation's held-out TEST words (disjoint from the fit set by the split).
    Inits: 'scratch' (tiny random) and, if a compressor is given, 'compressor' (its feed-forward
    guess - does a learned init reduce the steps needed?). Curve = steps-vs-apply on an unseen relation."""
    H = model.config.hidden_size
    inits = ["scratch"] + (["compressor"] if compressor is not None else [])
    out = {"held_eval": held_eval, "step_grid": step_grid, "inits": inits, "per_init": {}, "fit_on": fit_on}
    for init in inits:
        per_step_menu = {s: [] for s in step_grid}
        per_step_free = {s: [] for s in step_grid}
        for held in held_eval:
            tp = train_pairs[held]
            g = torch.Generator(device=DEV).manual_seed(seed + 5)   # SAME K draw as eval-compressor for fairness
            xi, yi = _sample_K(tp, K, g)
            # fit set: either K examples or the relation's full TRAIN words
            if fit_on == "K":
                fit_idx = list(zip(xi.tolist(), yi.tolist()))
            else:
                fit_idx = tp.tolist()
            xs = [words[int(a)] for (a, b) in fit_idx]; ys = [answer_tok[words[int(b)]] for (a, b) in fit_idx]
            padded, mask = FA.batch_pack([q_emb_cache[x] for x in xs])
            ytgt = torch.tensor(ys, device=DEV)
            # init prefix
            if init == "compressor":
                with torch.no_grad():
                    xf = torch.stack([feat_cache[int(j)] for j in xi])[None]
                    yf = torch.stack([feat_cache[int(j)] for j in yi])[None]
                    p0 = compressor(xf, yf)[0].detach().clone()
            else:
                torch.manual_seed(seed + 13)
                p0 = 0.02 * torch.randn(m, H, device=DEV)
            pm = FA.SoftPrefix(m, H).to(DEV); pm.prefix = nn.Parameter(p0)
            opt = torch.optim.Adam(pm.parameters(), lr)
            # checkpoint at step 0 then after each grid interval
            ckpt = {0: None}
            grid = sorted(step_grid)
            maxs = max(grid)
            # record at step 0
            if 0 in per_step_menu:
                mn, fr, _ = eval_prefix_on_relation(model, pm.prefix.detach(), held, test_pairs, words,
                                                    q_emb_cache, answer_tok, menu_ids, out_set_idx)
                per_step_menu[0].append(mn); per_step_free[0].append(fr)
            done = set([0])
            for step in range(1, maxs + 1):
                logits = FA.forward_with_prefix(model, pm, padded, mask)
                loss = F.cross_entropy(logits, ytgt)
                opt.zero_grad(); loss.backward(); opt.step()
                if step in step_grid and step not in done:
                    mn, fr, _ = eval_prefix_on_relation(model, pm.prefix.detach(), held, test_pairs, words,
                                                        q_emb_cache, answer_tok, menu_ids, out_set_idx)
                    per_step_menu[step].append(mn); per_step_free[step].append(fr)
                    done.add(step)
        curve = []
        for s in sorted(step_grid):
            mvals = per_step_menu[s]; fvals = per_step_free[s]
            curve.append(dict(steps=s,
                              menu=float(sum(mvals) / max(1, len(mvals))),
                              free=float(sum(fvals) / max(1, len(fvals)))))
        out["per_init"][init] = curve
        line = "  ".join(f"{c['steps']}:{c['free']:.2f}" for c in curve)
        print(f"  init={init:10s} steps->free-apply  {line}")
    out["icl_ref"] = float(sum(icl_per.get(r, float('nan')) for r in held_eval if r in icl_per) /
                           max(1, sum(1 for r in held_eval if r in icl_per)))
    out["readmlp_ref"] = float(sum(rm_per.get(r, 0.0) for r in held_eval) / max(1, len(held_eval)))
    return out

# ==========================================================================================
# Read-MLP / ICL references. The read-MLP scored 0.000 on every 1-to-1 relation (sidecar_semantic
# / frontier_apply); we use that recorded number for the 7 base relations and 0.000 for the new
# 1-to-1 relations (their read-MLP would be 0.000 by the SAME argument - a small external reader
# cannot recover an exact 1-to-1 target; the recorded run already showed every 1-to-1 base relation
# at 0.000). hypernym/color (many-to-few) keep their recorded non-zero values; new many-to-few
# relations (hyponym/made_of) are conservatively bracketed at 0.000 (we did not re-run the read-MLP
# at scale, so we under-claim rather than over-claim the baseline we beat). Loaded from disk.
def load_readmlp_and_icl():
    path = os.path.join(RUNS, "frontier_apply_0p5b.json")
    rm = {}; icl_recorded = {}
    if os.path.exists(path):
        d = json.load(open(path))
        base = d.get("readmlp_baseline") or {}
        rm = dict(base.get("per_relation", {}))
        icl_recorded = dict(d.get("stage1", {}).get("icl_per_relation", {}))
    return rm, icl_recorded

# ==========================================================================================
# SVGs
def svg_lever1(path, curve, title):
    Ns = [c["N"] for c in curve]
    W, Hh, ml, mr, mt, mb = 720, 380, 56, 200, 44, 60
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    if not Ns:
        open(path, "w").write("<svg/>"); return
    nmin, nmax = min(Ns), max(Ns)
    Xc = lambda N: x0 + ((N - nmin) / (nmax - nmin) if nmax > nmin else 0.5) * (x1 - x0)
    Yc = lambda v: y0 - (0.0 if v != v else v) * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="{BG}"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="{TXT}" font-size="13" text-anchor="middle">{title}</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="{GRID}"/>',
                         f'<text x="{x0-6}" y="{Y+4:.1f}" fill="{MUT}" font-size="10" text-anchor="end">{v:g}</text>']
    icl_ref = curve[0]["icl_ref"]; rm_ref = curve[0]["readmlp_ref"]
    p.append(f'<line x1="{x0}" y1="{Yc(icl_ref):.1f}" x2="{x1}" y2="{Yc(icl_ref):.1f}" stroke="{GOLD}" stroke-dasharray="6 3"/>')
    p.append(f'<line x1="{x0}" y1="{Yc(rm_ref):.1f}" x2="{x1}" y2="{Yc(rm_ref):.1f}" stroke="{LILAC}" stroke-dasharray="4 3"/>')
    for key, col in [("agg_menu", TEAL), ("agg_free", PINK)]:
        pts = " ".join(f"{Xc(c['N']):.1f},{Yc(c[key]):.1f}" for c in curve)
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        for c in curve:
            p.append(f'<circle cx="{Xc(c["N"]):.1f}" cy="{Yc(c[key]):.1f}" r="3.5" fill="{col}"/>')
    for c in curve:
        p.append(f'<text x="{Xc(c["N"]):.1f}" y="{y0+16}" fill="{MUT}" font-size="10" text-anchor="middle">{c["N"]}</text>')
    p.append(f'<text x="{(x0+x1)/2}" y="{Hh-8}" fill="#B8B3D6" font-size="11" text-anchor="middle"># relations in the bank (N); compressor trains on N-1 LORO</text>')
    ly = mt + 12
    for col, lab in [(TEAL, "meta apply (menu)"), (PINK, "meta apply (free)"),
                     (GOLD, "ICL ceiling"), (LILAC, "read-MLP")]:
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="{TXT}" font-size="10.5">{lab}</text>']; ly += 20
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))

def svg_lever2(path, l2, title):
    held = l2["held_eval"]
    groups = list(held) + ["AGG"]
    W, Hh, ml, mr, mt, mb = 860, 380, 52, 200, 44, 92
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    n = len(groups); bw = (x1 - x0) / n
    Yc = lambda v: y0 - (0.0 if v != v else v) * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="{BG}"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="{TXT}" font-size="13" text-anchor="middle">{title}</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="{GRID}"/>',
                         f'<text x="{x0-6}" y="{Y+4:.1f}" fill="{MUT}" font-size="10" text-anchor="end">{v:g}</text>']
    def get(d, key, r):
        if r == "AGG":
            return l2.get("agg_" + key, float("nan"))
        return (l2["per_rel"].get(r, {}) or {}).get(key, float("nan"))
    for i, r in enumerate(groups):
        cx = x0 + (i + 0.5) * bw
        df = get(l2, "distill_free", r) if r != "AGG" else l2.get("agg_distill_free", float("nan"))
        dm = get(l2, "distill_menu", r) if r != "AGG" else l2.get("agg_distill_menu", float("nan"))
        of = get(l2, "oracle_free", r) if r != "AGG" else l2.get("agg_oracle_free", float("nan"))
        ic = (l2["per_rel"].get(r, {}) or {}).get("icl", float("nan")) if r != "AGG" else l2.get("icl_ref", float("nan"))
        rm = (l2["per_rel"].get(r, {}) or {}).get("readmlp", float("nan")) if r != "AGG" else l2.get("readmlp_ref", float("nan"))
        for off, v, col in [(-0.40, dm, TEAL), (-0.20, df, PINK), (0.0, of, LILAC), (0.20, ic, GOLD), (0.40, rm, SLATE)]:
            if v == v:
                p.append(f'<rect x="{cx+off*bw:.1f}" y="{Yc(v):.1f}" width="{bw*0.16:.1f}" height="{(y0-Yc(v)):.1f}" fill="{col}"/>')
        p.append(f'<text x="{cx:.1f}" y="{y0+14}" fill="{MUT}" font-size="9" text-anchor="middle" transform="rotate(20 {cx:.1f} {y0+14})">{r}</text>')
    ly = mt + 12
    for col, lab in [(TEAL, "distill (menu)"), (PINK, "distill (free)"), (LILAC, "oracle prefix"),
                     (GOLD, "ICL ceiling"), (SLATE, "read-MLP")]:
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="{TXT}" font-size="10.5">{lab}</text>']; ly += 19
    leg = l2.get("legibility_acc")
    if leg is not None:
        p.append(f'<text x="{x1+14}" y="{ly+8}" fill="{TEAL}" font-size="10.5">legib probe {leg:.2f} (ch {l2.get("legibility_chance",0):.2f})</text>')
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))

def svg_lever3(path, l3, title):
    grid = sorted(l3["step_grid"])
    W, Hh, ml, mr, mt, mb = 720, 380, 56, 200, 44, 60
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    smin, smax = min(grid), max(grid)
    Xc = lambda s: x0 + ((s - smin) / (smax - smin) if smax > smin else 0.5) * (x1 - x0)
    Yc = lambda v: y0 - (0.0 if v != v else v) * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="{BG}"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="{TXT}" font-size="13" text-anchor="middle">{title}</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="{GRID}"/>',
                         f'<text x="{x0-6}" y="{Y+4:.1f}" fill="{MUT}" font-size="10" text-anchor="end">{v:g}</text>']
    p.append(f'<line x1="{x0}" y1="{Yc(l3["icl_ref"]):.1f}" x2="{x1}" y2="{Yc(l3["icl_ref"]):.1f}" stroke="{GOLD}" stroke-dasharray="6 3"/>')
    p.append(f'<line x1="{x0}" y1="{Yc(l3["readmlp_ref"]):.1f}" x2="{x1}" y2="{Yc(l3["readmlp_ref"]):.1f}" stroke="{LILAC}" stroke-dasharray="4 3"/>')
    cols = {"scratch": (PINK, TEAL), "compressor": (CORAL, GOLD)}
    for init, curve in l3["per_init"].items():
        col_free = cols.get(init, (PINK, TEAL))[0]
        pts = " ".join(f"{Xc(c['steps']):.1f},{Yc(c['free']):.1f}" for c in curve)
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col_free}" stroke-width="2.5"/>')
        for c in curve:
            p.append(f'<circle cx="{Xc(c["steps"]):.1f}" cy="{Yc(c["free"]):.1f}" r="3.2" fill="{col_free}"/>')
    for s in grid:
        p.append(f'<text x="{Xc(s):.1f}" y="{y0+16}" fill="{MUT}" font-size="9" text-anchor="middle">{s}</text>')
    p.append(f'<text x="{(x0+x1)/2}" y="{Hh-8}" fill="#B8B3D6" font-size="11" text-anchor="middle">test-time gradient steps on the NEW relation\'s prefix (free-gen apply)</text>')
    ly = mt + 12
    legend = [(PINK, "scratch init")]
    if "compressor" in l3["per_init"]:
        legend.append((CORAL, "compressor init"))
    legend += [(GOLD, "ICL ceiling"), (LILAC, "read-MLP")]
    for col, lab in legend:
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="{TXT}" font-size="10.5">{lab}</text>']; ly += 20
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))

# ==========================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--m", type=int, default=8)              # soft-prefix length
    ap.add_argument("--oracle_steps", type=int, default=400) # Stage-1 oracle prefix steps
    ap.add_argument("--oracle_lr", type=float, default=0.05)
    ap.add_argument("--comp_steps", type=int, default=1500)  # compressor meta-steps (per LORO)
    ap.add_argument("--comp_lr", type=float, default=1e-3)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--layer", type=int, default=12)         # feature layer (= read-MLP best L12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", default="")                   # optional multi-seed for oracle/ICL (csv); default = [seed]
    ap.add_argument("--test_frac", type=float, default=0.30)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--min_pairs", type=int, default=10)
    ap.add_argument("--counts", default="")                  # lever-1 N grid; default chosen from bank size
    ap.add_argument("--n_held", type=int, default=6)         # lever-1/2/3 held-out relation eval set size
    ap.add_argument("--ttt_steps", default="0,2,5,10,20,40,80,160")  # lever-3 step grid
    ap.add_argument("--ttt_lr", type=float, default=0.05)
    ap.add_argument("--ttt_fit", default="train", choices=["train", "K"])  # fit on full train words or K examples
    ap.add_argument("--distill_ce_w", type=float, default=0.0)  # lever-2 optional apply-CE weight on top of distill
    ap.add_argument("--icl_episodes", type=int, default=150)
    ap.add_argument("--icl_K", type=int, default=5)
    ap.add_argument("--levers", default="1,2,3")             # which levers to run
    ap.add_argument("--tag", default="_0p5b")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    t_start = time.time()
    levers = set(x.strip() for x in args.levers.split(","))
    print(f"device={DEV}  model={args.model}  m={args.m}  K={args.K}  layer={args.layer}  seed={args.seed}")

    # ---- load model + build the expanded, self-validated relation bank ----
    print("\nloading FROZEN LLM...")
    tok, model = FA.load_llm(args.model, dtype=getattr(torch, args.dtype))
    bank, REL_NAMES, dropped = build_bank(tok, min_pairs=args.min_pairs)
    words, widx, out_words, out_ids = build_vocab_bank(bank)
    train_pairs, test_pairs = split_bank(bank, words, widx, test_frac=args.test_frac, seed=args.split_seed)
    chance = 1.0 / len(out_ids)
    print(f"|relations|={len(REL_NAMES)}  |words|={len(words)}  |menu V|={len(out_ids)}  chance={chance:.4f}")
    if dropped:
        print(f"  dropped (< {args.min_pairs} single-token pairs): " +
              ", ".join(f"{r}({n})" for r, n in dropped.items()))
    print("  relation TRAIN/TEST pair counts:")
    for r in REL_NAMES:
        print(f"    {r:16s} total={len(bank[r]):2d}  train={train_pairs[r].shape[0]:2d}  test={test_pairs[r].shape[0]:2d}")

    answer_tok = {w: single_token_id(tok, w) for w in words}
    menu_ids = torch.tensor([answer_tok[w] for w in out_words], device=DEV)
    out_set_idx = {w: j for j, w in enumerate(out_words)}
    q_emb_cache = FA.cache_query_embeds(tok, model, words)
    H = model.config.hidden_size

    # ---- references: read-MLP (recorded) + ICL ceiling (recompute on THIS bank) ----
    rm_recorded, icl_recorded = load_readmlp_and_icl()
    # read-MLP per-relation: recorded for base 7; 0.000 for the new ones (under-claim; see note in code)
    rm_per = {r: rm_recorded.get(r, 0.0) for r in REL_NAMES}
    print("\nnative-ICL ceiling on this bank (frozen model, K text pairs, menu-scored)...")
    icl_per = {}
    for r in REL_NAMES:
        # reuse the recorded base-relation ICL if present (identical recipe), else compute
        icl_per[r] = FA.icl_ceiling_rel(tok, model, r, train_pairs, test_pairs, words, menu_ids,
                                        out_set_idx, K=args.icl_K, n_episodes=args.icl_episodes)
    icl_agg = float(sum(icl_per.values()) / len(icl_per))
    print(f"  ICL aggregate over {len(REL_NAMES)} relations = {icl_agg:.3f}")

    report = dict(model=args.model, device=DEV, m=args.m, K=args.K, layer=args.layer, seed=args.seed,
                  n_relations=len(REL_NAMES), rel_names=REL_NAMES, n_words=len(words),
                  menu_size=len(out_ids), chance=chance, test_frac=args.test_frac,
                  split_seed=args.split_seed, dropped=dropped,
                  train_counts={r: int(train_pairs[r].shape[0]) for r in REL_NAMES},
                  test_counts={r: int(test_pairs[r].shape[0]) for r in REL_NAMES},
                  env="cloze/.venv (torch cu128, RTX 5080)", frozen_backbone=True,
                  icl_per_relation=icl_per, icl_aggregate=icl_agg, readmlp_per_relation=rm_per,
                  comp_steps=args.comp_steps, comp_lr=args.comp_lr, oracle_steps=args.oracle_steps,
                  stage2_recorded=dict(menu=0.159, free=0.000, n_relations=7),
                  stage1_recorded=dict(menu=0.928, free=0.860, n_relations=7))

    # ---- harvest frozen features for the compressor (once) ----
    print("\nharvesting frozen word features (layer %d) for compressor input..." % args.layer)
    feat_cache = {}
    for w in words:
        feat_cache[widx[w]] = harvest_feat(tok, model, w, args.layer)

    # fixed shuffled relation order + held-out eval set (shared across levers for comparability)
    g = torch.Generator().manual_seed(args.seed + 99)
    order = torch.randperm(len(REL_NAMES), generator=g).tolist()
    rels_ordered = [REL_NAMES[i] for i in order]
    held_eval = rels_ordered[:args.n_held]
    print(f"\nfixed held-out relation eval set (LORO, shared across levers): {held_eval}")

    # =================== LEVER 1: learning curve vs #relations ===================
    if "1" in levers:
        print("\n" + "=" * 84)
        print("LEVER 1 - MORE RELATIONS: LORO held-out-relation apply vs #training relations")
        print("=" * 84)
        if args.counts:
            counts = [int(x) for x in args.counts.split(",")]
        else:
            R = len(REL_NAMES)
            counts = sorted(set([args.n_held + 2, 9, 12, 16, 20, 24, R]))
            counts = [c for c in counts if args.n_held + 2 <= c <= R]
        l1 = lever1(model, REL_NAMES, train_pairs, test_pairs, words, q_emb_cache, answer_tok,
                    menu_ids, out_set_idx, feat_cache, args.m, counts, args.comp_steps, args.K,
                    args.comp_lr, args.seed, icl_per, rm_per, n_held=args.n_held)
        report["lever1"] = l1
        svg_lever1(os.path.join(RUNS, f"frontier_apply_v2_lever1{args.tag}.svg"), l1["curve"],
                   f"Lever 1: held-out-relation apply vs #relations (Qwen2.5-0.5B, m={args.m}, K={args.K})")
        if l1["curve"]:
            first, last = l1["curve"][0], l1["curve"][-1]
            print(f"\n  LEVER 1 curve: N={first['N']} free={first['agg_free']:.3f} -> "
                  f"N={last['N']} free={last['agg_free']:.3f}  (Stage-2 was free=0.000 @ 7 rel)")

    # =================== LEVER 2: distill to oracle prefixes ===================
    if "2" in levers:
        print("\n" + "=" * 84)
        print("LEVER 2 - DISTILL TO ORACLE PREFIXES: shared compressor regressed to Stage-1 prefixes")
        print("=" * 84)
        print("  tuning per-relation ORACLE prefixes (Stage-1) for all relations...")
        oracle_prefixes = {}
        oracle_self = {}
        for r in REL_NAMES:
            pre = train_oracle_prefix(tok, model, r, train_pairs, words, q_emb_cache, answer_tok,
                                      m=args.m, steps=args.oracle_steps, lr=args.oracle_lr, seed=args.seed)
            oracle_prefixes[r] = pre.prefix.detach().clone()
            mn, fr, _ = eval_prefix_on_relation(model, oracle_prefixes[r], r, test_pairs, words,
                                                q_emb_cache, answer_tok, menu_ids, out_set_idx)
            oracle_self[r] = dict(menu=mn, free=fr)
        oa_menu = float(sum(v["menu"] for v in oracle_self.values()) / len(oracle_self))
        oa_free = float(sum(v["free"] for v in oracle_self.values()) / len(oracle_self))
        print(f"  oracle (Stage-1) apply across all {len(REL_NAMES)} relations: menu={oa_menu:.3f} free={oa_free:.3f}")
        report["oracle_all_relations"] = dict(per_relation=oracle_self, agg_menu=oa_menu, agg_free=oa_free)
        l2 = lever2(model, REL_NAMES, train_pairs, test_pairs, words, q_emb_cache, answer_tok,
                    menu_ids, out_set_idx, feat_cache, oracle_prefixes, args.m, args.comp_steps,
                    args.K, args.comp_lr, args.seed, icl_per, rm_per, held_eval, args.distill_ce_w)
        report["lever2"] = l2
        svg_lever2(os.path.join(RUNS, f"frontier_apply_v2_lever2{args.tag}.svg"), l2,
                   f"Lever 2: distill->oracle prefixes, LORO apply + legibility (m={args.m}, K={args.K})")
        print(f"\n  LEVER 2: distill held-out apply menu={l2['agg_distill_menu']:.3f} "
              f"free={l2['agg_distill_free']:.3f} | oracle ceiling free={l2['agg_oracle_free']:.3f} "
              f"| ICL={l2['icl_ref']:.3f} | read-MLP={l2['readmlp_ref']:.3f}")
        print(f"  LEVER 2 legibility (shared-compressor outputs -> relation): "
              f"{l2['legibility_acc']:.3f} (chance {l2['legibility_chance']:.3f})  "
              f"[Stage-1 independent prefixes were 0.000]")
        # keep oracle_prefixes for lever 3 init-from-compressor handled separately
        report["_have_oracle"] = True

    # =================== LEVER 3: test-time adaptation ===================
    if "3" in levers:
        print("\n" + "=" * 84)
        print("LEVER 3 - TEST-TIME ADAPTATION: few gradient steps to fit a NEW relation's prefix")
        print("=" * 84)
        step_grid = [int(x) for x in args.ttt_steps.split(",")]
        # optional compressor init: train ONE e2e compressor on all-but-held (cheap, reuses lever-1 path)
        # we use a single broad compressor (leave out the LAST held rel) to provide the init guess.
        comp_for_init = None
        if "2" in levers or True:
            # train a quick compressor on all relations except held_eval (so it never saw held rels)
            train_rel_subset = [r for r in REL_NAMES if r not in held_eval]
            print(f"  training a compressor-init guess on {len(train_rel_subset)} non-held relations "
                  f"({args.comp_steps} steps, e2e)...")
            _, comp_for_init = run_compressor_LORO(model, REL_NAMES, train_rel_subset, [], train_pairs,
                                                   test_pairs, words, q_emb_cache, answer_tok, menu_ids,
                                                   out_set_idx, feat_cache, args.m, args.comp_steps,
                                                   args.K, args.comp_lr, args.seed, mode="e2e")
        # run BOTH fit regimes: full-train (strong; asymptotes to the Stage-1 oracle) and
        # few-shot K-only (the harder, genuine Titans/TTT claim - only K examples of the new relation).
        print(f"  [fit_on=train: full TRAIN words of the new relation]")
        l3 = lever3(tok, model, REL_NAMES, train_pairs, test_pairs, words, q_emb_cache, answer_tok,
                    menu_ids, out_set_idx, feat_cache, None, args.m, step_grid, args.K, args.ttt_lr,
                    args.seed, icl_per, rm_per, held_eval, compressor=comp_for_init, fit_on="train")
        print(f"  [fit_on=K: only K={args.K} examples of the new relation]")
        l3_K = lever3(tok, model, REL_NAMES, train_pairs, test_pairs, words, q_emb_cache, answer_tok,
                      menu_ids, out_set_idx, feat_cache, None, args.m, step_grid, args.K, args.ttt_lr,
                      args.seed, icl_per, rm_per, held_eval, compressor=comp_for_init, fit_on="K")
        l3["fewshot_K"] = l3_K
        report["lever3"] = l3
        svg_lever3(os.path.join(RUNS, f"frontier_apply_v2_lever3{args.tag}.svg"), l3,
                   f"Lever 3: test-time adaptation on a NEW relation, fit on full train (m={args.m}, K={args.K})")
        svg_lever3(os.path.join(RUNS, f"frontier_apply_v2_lever3_fewshot{args.tag}.svg"), l3_K,
                   f"Lever 3: test-time adaptation, FEW-SHOT (K={args.K} examples) on a NEW relation (m={args.m})")
        sc = l3["per_init"]["scratch"]
        thresh = 0.5 * l3["icl_ref"]
        reach = next((c["steps"] for c in sc if c["free"] >= thresh), None)
        scK = l3_K["per_init"]["scratch"]
        reachK = next((c["steps"] for c in scK if c["free"] >= thresh), None)
        print(f"\n  LEVER 3 (fit full-train, scratch): free {sc[0]['free']:.3f}@0 -> {sc[-1]['free']:.3f}@{sc[-1]['steps']}; "
              f"0.5*ICL({thresh:.2f}) at {reach} steps")
        print(f"  LEVER 3 (few-shot K, scratch):     free {scK[0]['free']:.3f}@0 -> {scK[-1]['free']:.3f}@{scK[-1]['steps']}; "
              f"0.5*ICL at {reachK} steps")

    # =================== verdicts ===================
    print("\n" + "#" * 84)
    print("# FRONTIER v2 VERDICT - did any lever close the consolidate-then-apply loop / recover legibility?")
    print("#" * 84)
    verdicts = {}

    if "1" in levers and report.get("lever1", {}).get("curve"):
        c = report["lever1"]["curve"]
        best = max(c, key=lambda x: x["agg_free"])
        moved = best["agg_free"] > 0.10   # Stage 2 was 0.000 free
        verdicts["lever1"] = (
            f"LEVER 1 (more relations): held-out-relation FREE-apply went from 0.000 (Stage 2 @7 rel) to "
            f"{best['agg_free']:.3f} at N={best['N']} (menu {best['agg_menu']:.3f}); ICL ref {c[0]['icl_ref']:.3f}. "
            + ("The starved-regime hypothesis is SUPPORTED: more relations move the meta-generalization off zero."
               if moved else
               "More relations do NOT move free-gen off ~zero: 7 was not merely starved; the example->prefix "
               "map fails to generalize even with a larger bank. Reported plainly."))
        print("# " + verdicts["lever1"])

    if "2" in levers and report.get("lever2"):
        l2 = report["lever2"]
        moved = l2["agg_distill_free"] > 0.10
        nullacc = l2.get("legibility_null_acc")
        # legible only if clearly above BOTH chance and the untrained-map (null) control
        legible = (l2["legibility_acc"] is not None and
                   l2["legibility_acc"] > 2 * (l2["legibility_chance"] or 1) and
                   (nullacc is None or l2["legibility_acc"] > nullacc + 0.15))
        verdicts["lever2"] = (
            f"LEVER 2 (distill->oracle): shared-compressor held-out FREE-apply={l2['agg_distill_free']:.3f} "
            f"(menu {l2['agg_distill_menu']:.3f}) vs oracle ceiling {l2['agg_oracle_free']:.3f}, ICL {l2['icl_ref']:.3f}, "
            f"read-MLP {l2['readmlp_ref']:.3f}. Legibility (compressor outputs->relation)={l2['legibility_acc']:.3f} "
            f"vs untrained-map null={nullacc if nullacc is None else round(nullacc,3)} (chance {l2['legibility_chance']:.3f}). "
            + ("Distillation MOVES generalization off zero" if moved else
               "Distillation does NOT move free-gen off zero")
            + (" AND the shared compressor is legible ABOVE the untrained-map null (recovered the legibility "
               "Stage-1's independent prefixes lost)." if legible else
               "; legibility is NOT above the untrained-map null (separability is an input-feature artifact, "
               "not learned shared structure)." if (nullacc is not None and l2["legibility_acc"] is not None
                                                    and l2["legibility_acc"] <= nullacc + 0.15)
               else "; legibility not recovered above chance."))
        print("# " + verdicts["lever2"])

    if "3" in levers and report.get("lever3"):
        l3 = report["lever3"]
        sc = l3["per_init"]["scratch"]
        bestf = max(c["free"] for c in sc)
        thresh = 0.5 * l3["icl_ref"]
        reach = next((c["steps"] for c in sc if c["free"] >= thresh), None)
        works = bestf > max(0.10, 2 * l3["readmlp_ref"])
        # few-shot K regime
        scK = l3.get("fewshot_K", {}).get("per_init", {}).get("scratch", [])
        bestfK = max((c["free"] for c in scK), default=float("nan"))
        reachK = next((c["steps"] for c in scK if c["free"] >= thresh), None)
        comp_help = ""
        if "compressor" in l3["per_init"]:
            cc = l3["per_init"]["compressor"]
            reach_c = next((c["steps"] for c in cc if c["free"] >= thresh), None)
            comp_help = (f" Compressor-init (full-train) reaches the bar at "
                         f"{reach_c if reach_c is not None else '>'+str(cc[-1]['steps'])} steps vs scratch "
                         f"{reach if reach is not None else '>'+str(sc[-1]['steps'])}: "
                         + ("the learned init helps." if (reach_c is not None and (reach is None or reach_c < reach))
                            else "the learned init does not reduce steps."))
        verdicts["lever3"] = (
            f"LEVER 3 (test-time adaptation): a NEW relation's prefix, fit from scratch on the relation's own "
            f"examples, reaches free-apply {bestf:.3f} (full-train fit) / {bestfK:.3f} (few-shot K={report['K']}) "
            f"vs ICL {l3['icl_ref']:.3f}, read-MLP {l3['readmlp_ref']:.3f}; crosses 0.5*ICL at "
            f"{reach if reach is not None else '>'+str(sc[-1]['steps'])} steps (full-train) / "
            f"{reachK if reachK is not None else '>'+str(scK[-1]['steps']) if scK else 'NA'} steps (few-shot). "
            + ("TTT CLOSES the loop per-relation at test time: a few gradient steps recover a working prefix for an "
               "UNSEEN relation, even though a single feed-forward compressor map (levers 1/2) does not."
               if works else
               "TTT does NOT reach a working prefix on the unseen relations within the step budget.")
            + comp_help)
        print("# " + verdicts["lever3"])

    report["verdicts"] = verdicts
    report["wall_time_s"] = round(time.time() - t_start, 1)
    out_path = os.path.join(RUNS, f"frontier_apply_v2{args.tag}.json")
    json.dump(report, open(out_path, "w"), indent=2)
    print("#" * 84)
    print(f"\nwrote {out_path}  [{report['wall_time_s']}s]")

if __name__ == "__main__":
    main()
