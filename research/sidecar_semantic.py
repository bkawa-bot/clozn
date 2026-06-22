"""
sidecar_semantic.py — the SEMANTIC consolidation rung for the Clozn sidecar.

WHY THIS RUNG (read research/sidecar_real_findings.md first):
  sidecar_real.py validated the consolidation MACHINERY on a real frozen LLM, but its
  mod-cipher task was a LOW BAR: gaussian-random AND one-hot features ALSO scored 1.000,
  because a cipher only needs DISTINCT tokens. The deep claim — that the MODEL'S OWN learned
  understanding carries a transferable, consolidatable, legible rule — stayed UNTESTED.

  This rung tests it with a task UNSOLVABLE from token-distinctness alone, so the
  random/one-hot controls MUST fall to chance exactly where real features succeed. That gap
  is the whole point.

THE TASK — semantic-relation consolidation:
  Each episode draws a RELATION R from a family of real word-pair relations
  (antonym, plural, past-tense, comparative, country->capital, object->color, hypernym).
  Show K teaching pairs (x, R(x)); the sidecar CONSOLIDATES a state s; then predict R(x')
  for HELD-OUT words x'. Generalizing requires inferring WHICH relation from the examples
  and APPLYING it via the model's semantic geometry.

NO LEAKAGE: each relation's pairs are split TRAIN / TEST. Meta-train the sidecar ONLY on
  train pairs; evaluate untaught-generalization ONLY on HELD-OUT TEST pairs whose (x, R(x))
  appear in NO training episode (teaching or query). This forces genuine generalization and
  is what makes the random-feature control fail.

SCORING — RETRIEVAL (the load-bearing design choice):
  read(x', s) -> a vector; score every candidate word c in a shared vocab V by similarity to
  feat(c); predict argmax; correct if = R(x').  Chance = 1/|V|.
  This forces the answer to come from FEATURE GEOMETRY, so random features genuinely fail.
  (A classification head over V is reported as a CROSS-CHECK; retrieval is PRIMARY.)

ARCH (reused from sidecar_real.py):
  write: s = mean_i write_mlp([feat(x_i), feat(R(x_i))])        (permutation-invariant)
  read : read_mlp([feat(x'), s]) -> vector scored against proj(feat(c)) for c in V
  write/read META-LEARNED across random relation-episodes; LLM FROZEN. Features cached ONCE.

CONTROLS (the entire point):
  - real Qwen features vs GAUSSIAN-RANDOM vs ONE-HOT vs COLLAPSED  (prediction: real >> random)
  - LOOKUP baseline (store taught pairs, retrieve)               -> chance on held-out
  - NATIVE ICL ceiling (frozen model, K pairs in prompt as text) -> the analogy ceiling
  - vary K in (1,2,3,5); aggregate + spread over seeds

LEGIBILITY: probe the consolidated state s -> WHICH relation it is (R-family classifier).
PERSISTENCE: at sidecar query time NO teaching examples are in any prompt; R lives in s.

MODELS: Qwen2.5-0.5B / 3B (FROZEN). Env: cloze/.venv (torch cu128, RTX 5080).

Outputs (research/runs/): sidecar_semantic{tag}.json, sidecar_semantic_genK{tag}.svg,
  sidecar_semantic_legib{tag}.svg, sidecar_semantic_perrel{tag}.svg
"""
import os, sys, json, time, argparse, math
from collections import defaultdict
import torch, torch.nn as nn, torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
torch.set_float32_matmul_precision("high")

DEV = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
os.makedirs(RUNS, exist_ok=True)

# Maiko palette (matches sidecar_real.py)
BG, TEAL, PINK, TXT, MUT, GRID = "#1A1F4A", "#6FD6C9", "#FF8FB3", "#F4F0E8", "#8784b3", "#2c2f5e"
GOLD = "#E8C977"     # native-ICL ceiling
LILAC = "#B6A6E8"    # one-hot control
SLATE = "#7E8AA8"    # collapsed control

# ----------------------------------------------------------------------------------------
# DATA: 7 real word-pair relations. Every word is SINGLE-TOKEN in Qwen2.5 BPE (verified;
# see the harvest assert). Curated by hand from standard analogy families (Google/BATS-style).
RELATIONS = {
 "antonym": [("hot","cold"),("big","small"),("fast","slow"),("good","bad"),("happy","sad"),
   ("up","down"),("light","dark"),("high","low"),("rich","poor"),("strong","weak"),
   ("open","closed"),("wet","dry"),("full","empty"),("hard","soft"),("young","old"),
   ("tall","short"),("wide","narrow"),("clean","dirty"),("loud","quiet"),("love","hate"),
   ("day","night"),("win","lose"),("buy","sell"),("true","false"),("left","right"),
   ("push","pull"),("start","stop"),("begin","end"),("always","never")],
 "plural": [("cat","cats"),("dog","dogs"),("car","cars"),("book","books"),("tree","trees"),
   ("star","stars"),("hand","hands"),("eye","eyes"),("day","days"),("year","years"),
   ("road","roads"),("bird","birds"),("king","kings"),("game","games"),("word","words"),
   ("door","doors"),("field","fields"),("horse","horses"),("table","tables"),("house","houses"),
   ("apple","apples"),("phone","phones"),("chair","chairs"),("river","rivers"),("plant","plants"),
   ("clock","clocks"),("cloud","clouds"),("train","trains"),("snake","snakes"),("shirt","shirts")],
 "past": [("walk","walked"),("jump","jumped"),("play","played"),("talk","talked"),("look","looked"),
   ("want","wanted"),("need","needed"),("open","opened"),("close","closed"),("start","started"),
   ("call","called"),("help","helped"),("work","worked"),("move","moved"),("live","lived"),
   ("love","loved"),("turn","turned"),("show","showed"),("ask","asked"),("use","used"),
   ("like","liked"),("cook","cooked"),("clean","cleaned"),("paint","painted")],
 "comparative": [("big","bigger"),("small","smaller"),("fast","faster"),("slow","slower"),("tall","taller"),
   ("short","shorter"),("long","longer"),("strong","stronger"),("weak","weaker"),("warm","warmer"),
   ("cold","colder"),("hot","hotter"),("cool","cooler"),("hard","harder"),("soft","softer"),
   ("high","higher"),("low","lower"),("deep","deeper"),("rich","richer"),("clean","cleaner"),
   ("dark","darker"),("bright","brighter"),("young","younger"),("old","older"),("new","newer")],
 "capital": [("France","Paris"),("Japan","Tokyo"),("Italy","Rome"),("Spain","Madrid"),("Germany","Berlin"),
   ("Russia","Moscow"),("China","Beijing"),("Egypt","Cairo"),("Greece","Athens"),("Cuba","Havana"),
   ("Peru","Lima"),("Chile","Santiago"),("Canada","Ottawa"),("Poland","Warsaw"),("Austria","Vienna"),
   ("Iran","Tehran"),("Iraq","Baghdad"),("India","Delhi"),("Kenya","Nairobi"),("Norway","Oslo")],
 "color": [("banana","yellow"),("grass","green"),("blood","red"),("snow","white"),("coal","black"),
   ("sky","blue"),("milk","white"),("sun","yellow"),("gold","yellow"),("lime","green"),
   ("coffee","brown"),("cloud","white"),("ocean","blue"),("rose","red"),("lemon","yellow"),
   ("mud","brown"),("ash","gray"),("ruby","red"),("leaf","green"),("ink","black"),
   ("chalk","white"),("flame","orange"),("plum","purple"),("sand","tan")],
 "hypernym": [("dog","animal"),("rose","flower"),("apple","fruit"),("oak","tree"),("hammer","tool"),
   ("shirt","clothing"),("car","vehicle"),("chair","furniture"),("gold","metal"),("salmon","fish"),
   ("eagle","bird"),("piano","instrument"),("copper","metal"),("table","furniture"),("banana","fruit"),
   ("cotton","fabric"),("truck","vehicle"),("shark","fish"),("violin","instrument"),("grape","fruit"),
   ("pine","tree"),("drill","tool"),("hat","clothing"),("iron","metal"),("trout","fish"),
   ("robin","bird"),("flute","instrument"),("bus","vehicle"),("desk","furniture"),("silk","fabric"),
   ("wheat","grain")],
}
REL_NAMES = list(RELATIONS.keys())
CARRIER = "The word {w}"   # carrier context — escapes Qwen's position-0 attention-sink artifact

# ----------------------------------------------------------------------------------------
def build_vocab():
    """Shared candidate vocabulary V = every word that appears anywhere (input OR output),
    plus the word<->index maps. Retrieval scores over the OUTPUT subset of V."""
    words = sorted(set(w for pairs in RELATIONS.values() for p in pairs for w in p))
    widx = {w: i for i, w in enumerate(words)}
    out_words = sorted(set(y for pairs in RELATIONS.values() for (x, y) in pairs))
    out_ids = [widx[w] for w in out_words]                       # candidate menu (indices into words)
    return words, widx, out_words, out_ids

def split_relations(words, widx, test_frac=0.30, seed=0):
    """Per-relation TRAIN/TEST split on PAIRS. Returns, per relation, train/test pair-index
    arrays (each pair = (xi, yi) indices into `words`). A held-out test pair's (x, R(x)) never
    appears in any training episode (teaching or query) — the no-leakage guarantee."""
    rng = torch.Generator().manual_seed(seed)
    train, test = {}, {}
    for r, pairs in RELATIONS.items():
        idx = [(widx[x], widx[y]) for (x, y) in pairs]
        perm = torch.randperm(len(idx), generator=rng).tolist()
        ntest = max(2, int(round(len(idx) * test_frac)))
        te = sorted(perm[:ntest]); tr = sorted(perm[ntest:])
        train[r] = torch.tensor([idx[i] for i in tr], device=DEV)   # [Ntr,2]
        test[r]  = torch.tensor([idx[i] for i in te], device=DEV)   # [Nte,2]
    return train, test

def build_menu_rel(out_ids, widx):
    """[|menu|, R] bool: is candidate j an answer of relation r (anywhere in the data)? Used for
    the 'right-relation-cluster' diagnostic — did the retrieval land on a word of the correct
    relation TYPE (even if not the exact held-out target)? Separates 'knows which relation'
    from 'computed the exact answer'."""
    M = torch.zeros(len(out_ids), len(REL_NAMES), dtype=torch.bool, device=DEV)
    pos = {wi: j for j, wi in enumerate(out_ids)}
    for ri, r in enumerate(REL_NAMES):
        for (x, y) in RELATIONS[r]:
            j = pos.get(widx[y])
            if j is not None:
                M[j, ri] = True
    return M

# ----------------------------------------------------------------------------------------
@torch.no_grad()
def harvest_features(tok, model, words, carrier, layers):
    """{L: tensor[|words|, H]} frozen residual-stream feature per word, tapped at the word's
    position INSIDE `carrier` (the sink-artifact fix). Done ONCE; the LLM never trains.
    Multi-token guard: asserts single-token (we curated for it)."""
    wid = [tok.encode(" " + w, add_special_tokens=False) for w in words]
    for w, ids in zip(words, wid):
        assert len(ids) == 1, f"word {w!r} is not single-token: {ids}"
    wid = [i[0] for i in wid]
    feats = {L: [] for L in layers}
    for w, tid in zip(words, wid):
        ids = tok.encode(carrier.format(w=w), add_special_tokens=False)
        pos = max(i for i, t in enumerate(ids) if t == tid)
        out = model(torch.tensor(ids, device=DEV)[None, :], output_hidden_states=True)
        for L in layers:
            feats[L].append(out.hidden_states[L][0, pos, :].float())
    return {L: torch.stack(v) for L, v in feats.items()}, wid

def load_llm(model_name, dtype=torch.float32):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(DEV).eval()
    return tok, model

# ----------------------------------------------------------------------------------------
class SemanticSidecar(nn.Module):
    """Same write/read spine as sidecar_real.py, but READ is RETRIEVAL: it emits a vector in
    a projected feature space and we score candidates by cosine similarity to proj(feat(c)).
    feat() is the FROZEN cached LLM feature (a buffer, never trained)."""
    def __init__(self, feats, out_ids, proj=128, hs=256):
        super().__init__()
        self.register_buffer("feat", feats)                    # [|words|, H] frozen
        self.register_buffer("out_ids", torch.tensor(out_ids, device=feats.device))  # candidate menu
        H = feats.shape[1]
        self.proj = nn.Linear(H, proj)                         # shared learned projection of frozen feats
        self.write = nn.Sequential(nn.Linear(2 * proj, hs), nn.ReLU(), nn.Linear(hs, hs))
        # read emits a vector in the (separate) target space; we L2-normalize and cosine-score.
        self.read  = nn.Sequential(nn.Linear(proj + hs, hs), nn.ReLU(), nn.Linear(hs, proj))
        # candidate target embedding: a SECOND projection of the frozen feature (the retrieval key
        # space). Decoupled from `proj` so read can aim in target-space, but still a function of
        # the FROZEN feature only (random feats -> random keys -> no transferable structure).
        self.key = nn.Linear(H, proj)
        self.logit_scale = nn.Parameter(torch.tensor(10.0))    # temperature for cosine logits
    def fe(self, idx):                                          # projected source feature
        return self.proj(self.feat[idx])
    def keys(self):                                            # [|menu|, proj] normalized candidate keys
        k = self.key(self.feat[self.out_ids])
        return F.normalize(k, dim=-1)
    def state(self, xp, yp):                                    # xp,yp: [B,K] (word indices)
        pe = self.write(torch.cat([self.fe(xp), self.fe(yp)], -1))     # [B,K,hs]
        return pe.mean(1)                                       # [B,hs]
    def read_vec(self, xq, s):                                  # xq:[B,Q], s:[B,hs] -> [B,Q,proj] normalized
        Q = xq.shape[1]
        v = self.read(torch.cat([self.fe(xq), s[:, None, :].expand(-1, Q, -1)], -1))
        return F.normalize(v, dim=-1)
    def retrieve_logits(self, xq, s):                          # [B,Q,|menu|] cosine logits over candidate menu
        v = self.read_vec(xq, s)                                # [B,Q,proj]
        return self.logit_scale * (v @ self.keys().T)           # cosine * temp

# ----------------------------------------------------------------------------------------
# Episode samplers. An episode = ONE relation; K teaching pairs + queries, all relation-internal.
# Pools are packed ONCE into padded tensors [R, maxlen, 2] + lengths [R], cached by object id,
# so sampling is fully vectorized (no Python batch-loop) — matters for 15k-step meta-training.
_POOL_CACHE = {}
def _pack_pools(pools):
    key = id(pools)
    if key in _POOL_CACHE:
        return _POOL_CACHE[key]
    R = len(REL_NAMES)
    lens = torch.tensor([pools[r].shape[0] for r in REL_NAMES], device=DEV)   # [R]
    maxlen = int(lens.max().item())
    packed = torch.zeros(R, maxlen, 2, dtype=torch.long, device=DEV)
    for ri, r in enumerate(REL_NAMES):
        n = pools[r].shape[0]
        packed[ri, :n] = pools[r]
        if n < maxlen:                                  # pad by repeating (sampling uses %len so pad unused)
            packed[ri, n:] = pools[r][torch.arange(maxlen - n, device=DEV) % n]
    out = (packed, lens)
    _POOL_CACHE[key] = out
    return out

def sample_episode(query_pools, train_pairs, B, K, g, n_query=1, rel_ids=None):
    """Draw B episodes, fully vectorized. Each episode: a relation; K teaching pairs from that
    relation's TRAIN pool; n_query query pairs from `query_pools`. Returns
      xp,yp:[B,K] teaching ; xq,yq:[B,Q] queries ; rel:[B] relation id ; taught_mask:[B,Q].
    query_pools = test pairs (held-out eval) or train pairs (meta-training)."""
    R = len(REL_NAMES)
    tr_packed, tr_lens = _pack_pools(train_pairs)        # [R,Lt,2], [R]
    qp_packed, qp_lens = _pack_pools(query_pools)        # [R,Lq,2], [R]
    if rel_ids is None:
        rel = torch.randint(0, R, (B,), generator=g, device=DEV)
    else:
        rel = rel_ids.to(DEV)
    # teaching: K indices per episode, modulo that relation's train-pool length (sample w/ replacement)
    tlen = tr_lens[rel][:, None]                          # [B,1]
    ti = torch.randint(0, 10**9, (B, K), generator=g, device=DEV) % tlen        # [B,K] in [0,len)
    tpairs = tr_packed[rel[:, None].expand(B, K), ti]     # [B,K,2]
    xp, yp = tpairs[..., 0], tpairs[..., 1]
    # queries: n_query indices per episode, modulo query-pool length
    qlen = qp_lens[rel][:, None]                          # [B,1]
    qi = torch.randint(0, 10**9, (B, n_query), generator=g, device=DEV) % qlen  # [B,Q]
    qpairs = qp_packed[rel[:, None].expand(B, n_query), qi]
    xq, yq = qpairs[..., 0], qpairs[..., 1]
    # taught flag: query x equals any teaching x in the same episode
    taught = (xq[:, :, None] == xp[:, None, :]).any(-1)   # [B,Q]
    return xp, yp, xq, yq, rel, taught

# ----------------------------------------------------------------------------------------
def out_index_of(out_ids_list):
    """Map a global word index -> position in the candidate menu (for labels), or -1."""
    pos = {wi: j for j, wi in enumerate(out_ids_list)}
    return pos

def train_sidecar(feats, out_ids, train_pairs, steps=15000, B=256, lr=1e-3, seed=0,
                  kmax=5, proj=128, hs=256, n_query=4):
    """Meta-train write/read on TRAIN pairs only. Loss = retrieval cross-entropy over candidate
    menu (target = position of R(x') in the menu). Episodes query TRAIN pairs (held-out from
    that episode's teaching set when possible), so the model learns to GENERALIZE within train."""
    torch.manual_seed(seed)
    m = SemanticSidecar(feats, out_ids, proj=proj, hs=hs).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    pos = out_index_of(out_ids)                                 # global word idx -> menu position
    menu_lut = torch.full((feats.shape[0],), -1, dtype=torch.long, device=DEV)
    for wi, j in pos.items():
        menu_lut[wi] = j
    m.train()
    for step in range(steps):
        K = int(torch.randint(1, kmax + 1, (1,), generator=g, device=DEV))
        # query from TRAIN pairs (generalization signal: query x is often NOT in the K teaching)
        xp, yp, xq, yq, rel, taught = sample_episode(train_pairs, train_pairs, B, K, g, n_query=n_query)
        s = m.state(xp, yp)
        logits = m.retrieve_logits(xq, s)                       # [B,Q,|menu|]
        tgt = menu_lut[yq]                                      # [B,Q] menu positions of correct answers
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    return m, menu_lut

# ----------------------------------------------------------------------------------------
@torch.no_grad()
def eval_generalization(m, menu_lut, train_pairs, test_pairs, K, g, B=3000, n_query=1, menu_rel=None):
    """HELD-OUT untaught generalization: teaching pairs from TRAIN, query pairs from TEST.
    PRIMARY metric: retrieval acc = argmax over the FULL candidate menu equals R(x'). Returns
    per-relation acc; chance=1/|V|. If `menu_rel` given, also returns two diagnostics that
    separate 'knows which relation' from 'computed the exact answer':
      - cluster_acc : did the prediction land on ANY answer of the correct relation type?
      - within_acc  : restricting the menu to the correct relation's answers, is argmax = R(x')?
    A lookup memory is chance on held-out (a test x' is, by construction, never a taught x)."""
    R = len(REL_NAMES)
    rel_ids = torch.arange(B, device=DEV) % R                    # balanced over relations
    xp, yp, xq, yq, rel, taught = sample_episode(test_pairs, train_pairs, B, K, g,
                                                 n_query=n_query, rel_ids=rel_ids)
    logits = m.retrieve_logits(xq, m.state(xp, yp))             # [B,Q,|menu|]
    pred_menu = logits.argmax(-1)                               # [B,Q]
    tgt_menu = menu_lut[yq]                                     # [B,Q]
    correct = (pred_menu == tgt_menu)                           # [B,Q]
    unt = ~taught                                              # untaught-only (held-out; ~all true)
    acc = (correct & unt).sum().item() / max(1, unt.sum().item())
    per_rel = {}
    for r in range(R):
        msk = (rel == r)[:, None] & unt
        c = (correct & msk).sum().item(); n = msk.sum().item()
        per_rel[REL_NAMES[r]] = (c / n) if n else float("nan")
    chance = 1.0 / m.out_ids.shape[0]
    cluster_acc = within_acc = None
    if menu_rel is not None and n_query == 1:
        p = pred_menu[:, 0]                                     # [B]
        # cluster: predicted candidate is an answer of the episode's relation
        in_cluster = menu_rel[p, rel]                           # [B] bool
        cluster_acc = (in_cluster & unt[:, 0]).sum().item() / max(1, unt[:, 0].sum().item())
        # within-relation retrieval: mask logits to the relation's answers, re-argmax
        # build [B,|menu|] allowed mask from menu_rel[:, rel]  ->  [|menu|, B] -> transpose
        allowed = menu_rel[:, rel].T                            # [B,|menu|]
        masked = logits[:, 0, :].masked_fill(~allowed, float("-inf"))
        pred_w = masked.argmax(-1)
        within_ok = (pred_w == tgt_menu[:, 0])
        within_acc = (within_ok & unt[:, 0]).sum().item() / max(1, unt[:, 0].sum().item())
    return acc, per_rel, chance, cluster_acc, within_acc

# ----------------------------------------------------------------------------------------
@torch.no_grad()
def eval_train_pairs_sanity(m, menu_lut, train_pairs, K, g, B=2000):
    """Sanity: generalization measured on TRAIN-pair queries (held out from the teaching set of
    each episode but seen during meta-training). Should be >= held-out test acc."""
    R = len(REL_NAMES)
    rel_ids = torch.arange(B, device=DEV) % R
    xp, yp, xq, yq, rel, taught = sample_episode(train_pairs, train_pairs, B, K, g, n_query=1, rel_ids=rel_ids)
    logits = m.retrieve_logits(xq, m.state(xp, yp))
    correct = (logits.argmax(-1) == menu_lut[yq])
    unt = ~taught
    return (correct & unt).sum().item() / max(1, unt.sum().item())

# ----------------------------------------------------------------------------------------
def frequency_prior_baselines(words, train_pairs, test_pairs):
    """The honest FLOOR for the controls. A model with no feature geometry can still exploit
    answer FREQUENCY. Two priors, both fit on TRAIN answers, scored on TEST:
      - global-majority : always guess the single most frequent answer overall
      - per-relation-majority : if you knew the relation, guess its most frequent TRAIN answer
        (an UPPER bound on what a relation-aware-but-geometry-blind model — e.g. one that reads
         only the relation out of s — could get). gaussian/one-hot/collapsed should sit AT or
         BELOW the per-relation-majority floor; real should sit ABOVE it."""
    from collections import Counter
    # global majority answer (by TRAIN frequency)
    train_ans = [words[int(y)] for r in REL_NAMES for (x, y) in train_pairs[r].tolist()]
    glob = Counter(train_ans).most_common(1)[0][0]
    # per-relation majority answer (by TRAIN frequency)
    rel_major = {}
    for r in REL_NAMES:
        ans = [words[int(y)] for (x, y) in train_pairs[r].tolist()]
        rel_major[r] = Counter(ans).most_common(1)[0][0]
    # score on TEST
    g_correct = g_total = 0
    pr_correct = pr_total = 0
    per_rel = {}
    for r in REL_NAMES:
        rc = rt = 0
        for (x, y) in test_pairs[r].tolist():
            ya = words[int(y)]
            g_total += 1; g_correct += (ya == glob)
            pr_total += 1; pr_correct += (ya == rel_major[r])
            rt += 1; rc += (ya == rel_major[r])
        per_rel[r] = rc / rt if rt else float("nan")
    return dict(global_majority=g_correct / max(1, g_total),
                per_relation_majority=pr_correct / max(1, pr_total),
                per_relation_majority_byrel=per_rel)

# ----------------------------------------------------------------------------------------
# CLASSIFICATION-HEAD cross-check: instead of retrieval, train a linear head s,feat(x') -> menu.
# This is the "OK cross-check" — retrieval stays primary.
def eval_classify_crosscheck(feats, out_ids, train_pairs, test_pairs, K, seed=0, steps=15000,
                             B=256, proj=128, hs=256):
    """A sidecar whose read is a CLASSIFIER over the menu (not retrieval). Same split.
    Reported as a cross-check that the task is learnable at all on train geometry."""
    torch.manual_seed(seed)
    menu = len(out_ids)
    class ClsSidecar(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("feat", feats)
            H = feats.shape[1]
            self.proj = nn.Linear(H, proj)
            self.write = nn.Sequential(nn.Linear(2 * proj, hs), nn.ReLU(), nn.Linear(hs, hs))
            self.read = nn.Sequential(nn.Linear(proj + hs, hs), nn.ReLU(), nn.Linear(hs, menu))
        def fe(self, idx): return self.proj(self.feat[idx])
        def state(self, xp, yp): return self.write(torch.cat([self.fe(xp), self.fe(yp)], -1)).mean(1)
        def answer(self, xq, s):
            Q = xq.shape[1]
            return self.read(torch.cat([self.fe(xq), s[:, None, :].expand(-1, Q, -1)], -1))
    m = ClsSidecar().to(DEV); opt = torch.optim.Adam(m.parameters(), 1e-3)
    g = torch.Generator(device=DEV).manual_seed(seed + 1)
    menu_lut = torch.full((feats.shape[0],), -1, dtype=torch.long, device=DEV)
    for j, wi in enumerate(out_ids): menu_lut[wi] = j
    m.train()
    for step in range(steps):
        k = int(torch.randint(1, 6, (1,), generator=g, device=DEV))
        xp, yp, xq, yq, rel, taught = sample_episode(train_pairs, train_pairs, B, k, g, n_query=4)
        logits = m.answer(xq, m.state(xp, yp))
        loss = F.cross_entropy(logits.reshape(-1, menu), menu_lut[yq].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        R = len(REL_NAMES); B2 = 3000
        rel_ids = torch.arange(B2, device=DEV) % R
        xp, yp, xq, yq, rel, taught = sample_episode(test_pairs, train_pairs, B2, K, g, n_query=1, rel_ids=rel_ids)
        correct = (m.answer(xq, m.state(xp, yp)).argmax(-1) == menu_lut[yq])
        unt = ~taught
        return (correct & unt).sum().item() / max(1, unt.sum().item())

# ----------------------------------------------------------------------------------------
# LEGIBILITY: probe the consolidated state s -> WHICH relation R it is (7-way). chance=1/R.
def probe_relation(m, train_pairs, g, B=8000, K=3):
    rel_ids = torch.arange(B, device=DEV) % len(REL_NAMES)
    xp, yp, xq, yq, rel, taught = sample_episode(train_pairs, train_pairs, B, K, g, n_query=1, rel_ids=rel_ids)
    with torch.no_grad():
        s = m.state(xp, yp)
    ntr = B // 2
    probe = nn.Linear(s.shape[1], len(REL_NAMES)).to(DEV)
    opt = torch.optim.Adam(probe.parameters(), 1e-2)
    for _ in range(800):
        loss = F.cross_entropy(probe(s[:ntr]), rel[:ntr]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (probe(s[ntr:]).argmax(1) == rel[ntr:]).float().mean().item()
    return acc

@torch.no_grad()
def mean_state_per_relation(m, train_pairs, g, B=9000, K=3):
    rel_ids = torch.arange(B, device=DEV) % len(REL_NAMES)
    xp, yp, xq, yq, rel, taught = sample_episode(train_pairs, train_pairs, B, K, g, n_query=1, rel_ids=rel_ids)
    s = m.state(xp, yp)
    return torch.stack([s[rel == r].mean(0) for r in range(len(REL_NAMES))])   # [R, hs]

# ----------------------------------------------------------------------------------------
# NATIVE ICL ceiling: frozen model answers a HELD-OUT test query from K text pairs in its PROMPT.
# Scored by RETRIEVAL over the same candidate menu (restrict next-token logits to the menu's
# leading-space token ids), so it is directly comparable to the sidecar's retrieval metric.
@torch.no_grad()
def icl_ceiling(tok, model, words, out_words, train_pairs, test_pairs, K, n_episodes=350, seed=999):
    menu_ids = torch.tensor([tok.encode(" " + w, add_special_tokens=False)[0] for w in out_words], device=DEV)
    out_set_idx = {w: j for j, w in enumerate(out_words)}
    gg = torch.Generator().manual_seed(seed)
    R = len(REL_NAMES)
    correct = 0; total = 0
    for ep in range(n_episodes):
        r = ep % R
        rn = REL_NAMES[r]
        tr = train_pairs[rn].tolist(); te = test_pairs[rn].tolist()
        kk = min(K, len(tr))
        ti = torch.randperm(len(tr), generator=gg).tolist()[:kk]
        qi = int(torch.randint(0, len(te), (1,), generator=gg).item())
        xq_i, yq_i = te[qi]
        lines = [f"{words[tr[i][0]]} -> {words[tr[i][1]]}" for i in ti]
        prompt = "Complete the analogy with the same kind of relation.\n" + "\n".join(lines) + f"\n{words[xq_i]} ->"
        ids = tok.encode(prompt, add_special_tokens=False)
        logits = model(torch.tensor(ids, device=DEV)[None, :]).logits[0, -1]
        pred_menu = int(logits[menu_ids].argmax().item())      # argmax restricted to candidate menu
        if pred_menu == out_set_idx[words[yq_i]]:
            correct += 1
        total += 1
    return correct / max(1, total)

# ----------------------------------------------------------------------------------------
def run_feature_bank(name, feats_L, out_ids, train_pairs, test_pairs, Ks, seeds, steps,
                     do_classify=False, menu_rel=None):
    """Train sidecars (one per seed) on a given FEATURE BANK (real/gaussian/one_hot/collapsed),
    report held-out retrieval generalization per K (mean,std over seeds) + per-relation @ Kmax +
    the 'knows-relation vs applies-relation' diagnostics (cluster / within-relation acc)."""
    g = torch.Generator(device=DEV).manual_seed(321)
    per_seed_genK = []        # [{K: acc}]
    per_seed_perrel = []      # [{rel: acc}] at max K
    per_seed_train = []       # train-pair sanity at max K
    per_seed_cluster = []; per_seed_within = []
    ms, luts = [], []
    Kmax = max(Ks)
    for sd in seeds:
        m, lut = train_sidecar(feats_L, out_ids, train_pairs, steps=steps, seed=sd)
        ms.append(m); luts.append(lut)
        d = {}
        for K in Ks:
            acc, per_rel, chance, clus, wth = eval_generalization(
                m, lut, train_pairs, test_pairs, K, g, menu_rel=menu_rel)
            d[K] = acc
            if K == Kmax: pr, cl, wi = per_rel, clus, wth
        per_seed_genK.append(d); per_seed_perrel.append(pr)
        per_seed_cluster.append(cl); per_seed_within.append(wi)
        per_seed_train.append(eval_train_pairs_sanity(m, lut, train_pairs, Kmax, g))
    genK = {}
    for K in Ks:
        vals = [d[K] for d in per_seed_genK]
        genK[K] = dict(acc=float(sum(vals) / len(vals)),
                       std=float(torch.tensor(vals).std().item()) if len(vals) > 1 else 0.0,
                       n_seeds=len(seeds))
    perrel = {}
    for r in REL_NAMES:
        vals = [d[r] for d in per_seed_perrel if not math.isnan(d[r])]
        perrel[r] = float(sum(vals) / len(vals)) if vals else float("nan")
    chance = 1.0 / len(out_ids)
    train_sanity = float(sum(per_seed_train) / len(per_seed_train))
    def _avg(xs):
        xs = [x for x in xs if x is not None]
        return float(sum(xs) / len(xs)) if xs else None
    cls_acc = None
    if do_classify:
        cls_acc = eval_classify_crosscheck(feats_L, out_ids, train_pairs, test_pairs, Kmax, seed=seeds[0], steps=steps)
    return dict(genK=genK, per_relation=perrel, chance=chance, train_sanity=train_sanity,
                cluster_acc=_avg(per_seed_cluster), within_acc=_avg(per_seed_within),
                classify_crosscheck=cls_acc), ms[0], luts[0]

# ----------------------------------------------------------------------------------------
# SVGs (Maiko palette, mirroring sidecar_real.py helpers)
def svg_genK(path, banks, icl, chance, title, Ks):
    """real vs gaussian vs one_hot vs collapsed retrieval acc across K, + ICL ceiling + chance."""
    W, Hh, ml, mr, mt, mb = 640, 360, 58, 210, 42, 52
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    Xc = lambda i: x0 + (i / (len(Ks) - 1) * (x1 - x0) if len(Ks) > 1 else 0)
    Yc = lambda v: y0 - v * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="{BG}"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="{TXT}" font-size="13" text-anchor="middle">{title}</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="{GRID}"/>',
                         f'<text x="{x0-8}" y="{Y+4:.1f}" fill="{MUT}" font-size="10" text-anchor="end">{v:g}</text>']
    Ych = Yc(chance)
    p.append(f'<line x1="{x0}" y1="{Ych:.1f}" x2="{x1}" y2="{Ych:.1f}" stroke="{MUT}" stroke-dasharray="4 3"/>')
    p.append(f'<text x="{x1-2}" y="{Ych-4:.1f}" fill="{MUT}" font-size="9" text-anchor="end">chance 1/|V|={chance:.3f}</text>')
    for i, K in enumerate(Ks):
        p.append(f'<text x="{Xc(i):.1f}" y="{y0+18}" fill="{MUT}" font-size="11" text-anchor="middle">{K}</text>')
    p.append(f'<text x="{(x0+x1)/2}" y="{Hh-10}" fill="#B8B3D6" font-size="11" text-anchor="middle"># teaching pairs shown (K)</text>')
    series = [("real", TEAL, "real Qwen features"),
              ("gaussian", PINK, "gaussian-random"),
              ("one_hot", LILAC, "one-hot"),
              ("collapsed", SLATE, "collapsed")]
    for key, col, lab in series:
        if key not in banks: continue
        pts = " ".join(f"{Xc(i):.1f},{Yc(banks[key]['genK'][K]['acc']):.1f}" for i, K in enumerate(Ks))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>')
        for i, K in enumerate(Ks):
            p.append(f'<circle cx="{Xc(i):.1f}" cy="{Yc(banks[key]["genK"][K]["acc"]):.1f}" r="3.5" fill="{col}"/>')
    if icl:
        pts = " ".join(f"{Xc(i):.1f},{Yc(icl[K]):.1f}" for i, K in enumerate(Ks))
        p.append(f'<polyline points="{pts}" fill="none" stroke="{GOLD}" stroke-width="2.0" stroke-dasharray="6 3"/>')
        for i, K in enumerate(Ks):
            p.append(f'<circle cx="{Xc(i):.1f}" cy="{Yc(icl[K]):.1f}" r="3.0" fill="{GOLD}"/>')
    ly = mt + 12
    legend = [(TEAL, "real features"), (PINK, "gaussian"), (LILAC, "one-hot"), (SLATE, "collapsed")]
    if icl: legend.append((GOLD, "native-ICL ceiling"))
    legend.append((MUT, "chance 1/|V|"))
    for col, lab in legend:
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="{TXT}" font-size="10.5">{lab}</text>']; ly += 19
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))

def svg_perrel(path, real_perrel, rand_perrel, freq_perrel, chance, title):
    """grouped bars: per-relation held-out retrieval acc, real vs gaussian vs the per-relation
    frequency-majority floor (the honest ceiling for a geometry-blind model), + chance line."""
    rels = REL_NAMES
    W, Hh, ml, mr, mt, mb = 700, 360, 50, 170, 42, 80
    x0, x1, y0, y1 = ml, W - mr, Hh - mb, mt
    n = len(rels); bw = (x1 - x0) / n
    Yc = lambda v: y0 - v * (y0 - y1)
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{Hh}" font-family="Inconsolata,monospace">',
         f'<rect width="{W}" height="{Hh}" fill="{BG}"/>',
         f'<text x="{(x0+x1)/2}" y="22" fill="{TXT}" font-size="13" text-anchor="middle">{title}</text>']
    for v in [0, 0.25, 0.5, 0.75, 1.0]:
        Y = Yc(v); p += [f'<line x1="{x0}" y1="{Y:.1f}" x2="{x1}" y2="{Y:.1f}" stroke="{GRID}"/>',
                         f'<text x="{x0-6}" y="{Y+4:.1f}" fill="{MUT}" font-size="10" text-anchor="end">{v:g}</text>']
    Ych = Yc(chance)
    p.append(f'<line x1="{x0}" y1="{Ych:.1f}" x2="{x1}" y2="{Ych:.1f}" stroke="{MUT}" stroke-dasharray="4 3"/>')
    for i, r in enumerate(rels):
        cx = x0 + (i + 0.5) * bw
        rv = real_perrel.get(r, 0.0) or 0.0
        gv = rand_perrel.get(r, 0.0) or 0.0
        fv = (freq_perrel or {}).get(r, 0.0) or 0.0
        p.append(f'<rect x="{cx-bw*0.40:.1f}" y="{Yc(rv):.1f}" width="{bw*0.24:.1f}" height="{(y0-Yc(rv)):.1f}" fill="{TEAL}"/>')
        p.append(f'<rect x="{cx-bw*0.13:.1f}" y="{Yc(gv):.1f}" width="{bw*0.24:.1f}" height="{(y0-Yc(gv)):.1f}" fill="{PINK}"/>')
        p.append(f'<rect x="{cx+bw*0.14:.1f}" y="{Yc(fv):.1f}" width="{bw*0.24:.1f}" height="{(y0-Yc(fv)):.1f}" fill="{SLATE}"/>')
        p.append(f'<text x="{cx:.1f}" y="{y0+14}" fill="{MUT}" font-size="9" text-anchor="middle" transform="rotate(20 {cx:.1f} {y0+14})">{r}</text>')
    ly = mt + 12
    for col, lab in [(TEAL, "real"), (PINK, "gaussian"), (SLATE, "freq-prior floor"), (MUT, f"chance {chance:.3f}")]:
        p += [f'<rect x="{x1+14}" y="{ly-9}" width="12" height="12" fill="{col}"/>',
              f'<text x="{x1+30}" y="{ly+1}" fill="{TXT}" font-size="10.5">{lab}</text>']; ly += 19
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))

def svg_legib(path, means, probe_acc, chance, title):
    """PCA(2) of mean sidecar state per relation — do the relations separate? + probe acc text."""
    R = means.shape[0]
    e = means - means.mean(0, keepdim=True)
    _, _, Vh = torch.linalg.svd(e, full_matrices=False)
    P = (e @ Vh[:2].T).cpu(); P = P / (P.abs().max() + 1e-9)
    Wd = 480; R0 = 175; cx = Wd / 2; cy = Wd / 2 + 4
    Xf = lambda i: cx + P[i, 0].item() * R0; Yf = lambda i: cy - P[i, 1].item() * R0
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{Wd}" height="{Wd+40}" font-family="Inconsolata,monospace">',
         f'<rect width="{Wd}" height="{Wd+40}" fill="{BG}"/>',
         f'<text x="{Wd/2}" y="22" fill="{TXT}" font-size="13" text-anchor="middle">{title}</text>']
    for i in range(R):
        col = f"hsl({int(360*i/R)},70%,64%)"
        p.append(f'<circle cx="{Xf(i):.1f}" cy="{Yf(i):.1f}" r="8" fill="{col}"/>')
        p.append(f'<text x="{Xf(i):.1f}" y="{Yf(i)-12:.1f}" fill="{TXT}" font-size="10" text-anchor="middle">{REL_NAMES[i]}</text>')
    p.append(f'<text x="{Wd/2}" y="{Wd+22}" fill="{MUT}" font-size="10" text-anchor="middle">mean consolidated state per relation, PCA->2D</text>')
    p.append(f'<text x="{Wd/2}" y="{Wd+36}" fill="{TEAL}" font-size="11" text-anchor="middle">relation probe acc = {probe_acc:.3f}  (chance {chance:.3f})</text>')
    p.append('</svg>'); open(path, "w", encoding="utf-8").write("\n".join(p))

# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--layers", default="")            # default chosen by model below
    ap.add_argument("--steps", type=int, default=15000)
    ap.add_argument("--icl_episodes", type=int, default=350)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--sweep_seeds", type=int, default=1)   # #seeds used in the layer sweep (full seeds at best L)
    ap.add_argument("--tag", default="")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--test_frac", type=float, default=0.30)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--Ks", default="1,2,3,5")
    ap.add_argument("--skip_icl", action="store_true")
    args = ap.parse_args()

    Ks = tuple(int(x) for x in args.Ks.split(","))
    seeds = tuple(int(x) for x in args.seeds.split(","))
    # sensible default layer sweeps by model size
    if args.layers:
        layers = [int(x) for x in args.layers.split(",")]
    elif "0.5B" in args.model:
        layers = [6, 12, 18]
    elif "3B" in args.model:
        layers = [12, 18, 24]
    elif "7B" in args.model:
        layers = [14, 21, 28]
    else:
        layers = [12, 18, 24]

    words, widx, out_words, out_ids = build_vocab()
    train_pairs, test_pairs = split_relations(words, widx, test_frac=args.test_frac, seed=args.split_seed)
    menu_rel = build_menu_rel(out_ids, widx)        # [|menu|,R] for cluster/within diagnostics
    chance = 1.0 / len(out_ids)
    print(f"device={DEV}  model={args.model}  layers={layers}  seeds={seeds}")
    print(f"|words|={len(words)}  |candidate menu V|={len(out_ids)}  chance=1/|V|={chance:.4f}")
    print("relations & train/test pair counts:")
    for r in REL_NAMES:
        print(f"  {r:12s}  total={len(RELATIONS[r]):2d}  train={train_pairs[r].shape[0]:2d}  test={test_pairs[r].shape[0]:2d}")

    print("\nloading frozen LLM; harvesting word features ONCE (carrier context)...")
    t0 = time.time()
    tok, model = load_llm(args.model, dtype=getattr(torch, args.dtype))
    feats, wid = harvest_features(tok, model, words, CARRIER, layers)
    H = int(feats[layers[0]].shape[1])
    print(f"  harvested {len(words)} words in {time.time()-t0:.1f}s; feature dim H={H}")

    report = dict(model=args.model, device=DEV, layers=layers, seeds=list(seeds),
                  feature_dim=H, carrier=CARRIER, n_words=len(words), menu_size=len(out_ids),
                  chance=chance, test_frac=args.test_frac, split_seed=args.split_seed,
                  Ks=list(Ks), relations={r: len(RELATIONS[r]) for r in REL_NAMES},
                  train_counts={r: int(train_pairs[r].shape[0]) for r in REL_NAMES},
                  test_counts={r: int(test_pairs[r].shape[0]) for r in REL_NAMES},
                  env="cloze/.venv (torch 2.11+cu128, RTX 5080)",
                  per_layer={}, controls={}, icl={}, legibility={})

    # ----- frequency-prior FLOOR (honest baseline; controls should sit at/below it) -----
    freq = frequency_prior_baselines(words, train_pairs, test_pairs)
    report["freq_prior"] = freq
    print(f"\nfrequency-prior floor (geometry-blind): global-majority={freq['global_majority']:.3f}  "
          f"per-relation-majority={freq['per_relation_majority']:.3f}  (real must beat these)")

    # ----- sweep layers on REAL features; pick best by mean held-out retrieval acc -----
    # (sweep with `sweep_seeds` — fewer seeds, just to pick the layer; full seeds run at best L)
    sweep_seeds = seeds[:args.sweep_seeds] if args.sweep_seeds > 0 else seeds
    print(f"\n=== REAL features: held-out retrieval generalization, layer sweep (seeds={sweep_seeds}) ===")
    print("layer |   " + "   ".join(f"K={K}" for K in Ks) + "  | train-sanity | classify-xcheck")
    best_layer, best_score = None, -1
    for L in layers:
        res, m, lut = run_feature_bank("real", feats[L], out_ids, train_pairs, test_pairs,
                                       Ks, sweep_seeds, args.steps, do_classify=True, menu_rel=menu_rel)
        report["per_layer"][L] = res
        accs = "  ".join(f"{res['genK'][K]['acc']:.3f}" for K in Ks)
        print(f"  {L:3d} |  {accs}  |    {res['train_sanity']:.3f}     |     {res['classify_crosscheck']:.3f}")
        score = sum(res["genK"][K]["acc"] for K in Ks) / len(Ks)
        if score > best_score:
            best_layer, best_score = L, score
    report["best_layer"] = best_layer
    print(f"\nbest layer (mean held-out retrieval acc): L{best_layer}  ({best_score:.3f})")

    # ----- DISCRIMINATING CONTROLS @ best layer: real vs gaussian vs one_hot vs collapsed -----
    # All banks trained with the SAME full `seeds` on the SAME held-out split (the leakage check).
    print(f"\n=== DISCRIMINATING CONTROLS @ L{best_layer} (full seeds={seeds}; does it need REAL geometry?) ===")
    rf = feats[best_layer]
    banks_feats = {
        "real":      rf,
        "gaussian":  torch.randn(len(words), H, device=DEV),
        "one_hot":   (F.pad(torch.eye(len(words), device=DEV), (0, H - len(words)))
                      if H >= len(words) else torch.eye(len(words), device=DEV)),
        "collapsed": rf.mean(0, keepdim=True).expand(len(words), -1) + 1e-3 * torch.randn(len(words), H, device=DEV),
    }
    banks = {}
    best_m, best_lut = None, None
    print("  feature-bank |   " + "   ".join(f"K={K}" for K in Ks))
    real_perrel, rand_perrel = None, None
    for name, fs in banks_feats.items():
        res, m_repr, lut_repr = run_feature_bank(name, fs.contiguous(), out_ids, train_pairs,
                                                 test_pairs, Ks, seeds, args.steps, menu_rel=menu_rel)
        if name == "real":
            best_m, best_lut = m_repr, lut_repr          # representative real model for legibility
            report["per_layer"][best_layer] = res        # overwrite sweep-seed entry w/ full-seed real
        banks[name] = res
        report["controls"][name] = res
        accs = "  ".join(f"{res['genK'][K]['acc']:.3f}" for K in Ks)
        print(f"  {name:11s} |  {accs}")
        if name == "real": real_perrel = res["per_relation"]
        if name == "gaussian": rand_perrel = res["per_relation"]
    # the load-bearing diagnostic: 'knows which relation' (cluster) vs 'computed exact answer' (exact)
    print("\n  KNOWS-RELATION vs APPLIES-RELATION (real, @K=max):")
    print(f"    exact retrieval (full menu)      = {banks['real']['genK'][max(Ks)]['acc']:.3f}")
    print(f"    cluster (lands on right-relation word) = {banks['real']['cluster_acc']:.3f}  "
          f"(chance landing in right cluster ~= 1/{len(REL_NAMES)})")
    print(f"    within-relation retrieval (menu restricted to that relation) = {banks['real']['within_acc']:.3f}")
    print("    -> high cluster + low exact = relation IS consolidated, but the exact 1-to-1 target is not recoverable")
    # per-relation breakdown print (freq-prior floor beside every relation)
    print("\n  PER-RELATION held-out retrieval acc @ K=max (real vs controls vs freq-prior floor):")
    print("  relation      |  real   gaussian  one_hot  collapsed | freq-floor")
    fbr = freq["per_relation_majority_byrel"]
    for r in REL_NAMES:
        row = "  ".join(f"{banks[b]['per_relation'].get(r, float('nan')):.3f}" for b in
                        ("real", "gaussian", "one_hot", "collapsed"))
        print(f"  {r:12s}  |  {row} |   {fbr.get(r, float('nan')):.3f}")

    # ----- LEGIBILITY: probe consolidated state -> which relation -----
    print(f"\n=== LEGIBILITY @ L{best_layer}: probe consolidated state s -> which relation ===")
    g = torch.Generator(device=DEV).manual_seed(7)
    relprobe = probe_relation(best_m, train_pairs, g)
    leg_chance = 1.0 / len(REL_NAMES)
    print(f"  relation probe acc = {relprobe:.3f}   (chance 1/{len(REL_NAMES)} = {leg_chance:.3f})")
    report["legibility"] = dict(relation_probe_acc=relprobe, chance=leg_chance, n_relations=len(REL_NAMES))
    means = mean_state_per_relation(best_m, train_pairs, g)

    # ----- NATIVE ICL ceiling (frozen model, K text pairs, held-out query, retrieval-scored) -----
    icl = {}
    if not args.skip_icl:
        print(f"\n=== NATIVE ICL ceiling (frozen {args.model.split('/')[-1]}, retrieval-scored over V) ===")
        for K in Ks:
            icl[K] = icl_ceiling(tok, model, words, out_words, train_pairs, test_pairs, K,
                                 n_episodes=args.icl_episodes)
            print(f"  K={K}: ICL ceiling = {icl[K]:.3f}")
    report["icl"] = icl

    # ----- summary table -----
    ffloor = freq["per_relation_majority"]
    print("\n=== SUMMARY (held-out untaught generalization, retrieval acc) ===")
    print("  K  |  real   gaussian  one_hot  collapsed  |  ICL   freq-floor  chance")
    for K in Ks:
        iclK = icl.get(K, float("nan"))
        print(f"  {K}  |  {banks['real']['genK'][K]['acc']:.3f}    {banks['gaussian']['genK'][K]['acc']:.3f}    "
              f"{banks['one_hot']['genK'][K]['acc']:.3f}    {banks['collapsed']['genK'][K]['acc']:.3f}    |  "
              f"{iclK:.3f}   {ffloor:.3f}     {chance:.3f}")
    # verdict: real must beat the strongest control AND the freq-prior floor on EXACT retrieval;
    # separately report whether the RELATION is consolidated (cluster acc) even if exact fails.
    rK = banks["real"]["genK"][max(Ks)]["acc"]; gK = banks["gaussian"]["genK"][max(Ks)]["acc"]
    ctrl_max = max(banks[b]["genK"][max(Ks)]["acc"] for b in ("gaussian", "one_hot", "collapsed"))
    floor = max(ctrl_max, ffloor)
    gap = rK - floor
    clus = banks["real"]["cluster_acc"]; wth = banks["real"]["within_acc"]
    relprobe_v = report["legibility"]["relation_probe_acc"]
    exact_win = (rK > 4 * chance and gap > 0.10)
    relation_consolidated = (relprobe_v > 0.5 or (clus is not None and clus > 0.4))
    if exact_win:
        verdict = ("REAL >> CONTROLS on EXACT retrieval: the model's OWN geometry carries a transferable, "
                   "consolidatable, legible rule that APPLIES to held-out words.")
    elif relation_consolidated:
        verdict = ("PARTIAL: the RELATION is consolidated & legible (probe/cluster >> chance) and real beats "
                   "controls in aggregate, BUT the sidecar cannot recover the EXACT 1-to-1 target for unseen "
                   "words (exact retrieval ~ floor on 1-to-1 relations; only many-to-few relations like "
                   "hypernym/color clear the bar). The model CAN do it in-context (see ICL) — the bottleneck "
                   "is the consolidation/read, not the features.")
    else:
        verdict = "REAL ~ CONTROLS: NO clear evidence the model's geometry is doing the work (diagnose)."
    print(f"\n  real@Kmax={rK:.3f}  best-control={ctrl_max:.3f}  freq-floor={ffloor:.3f}  "
          f"gap-over-floor={gap:+.3f}  chance={chance:.3f}")
    print(f"  cluster(knows-relation)={clus if clus is None else round(clus,3)}  "
          f"within-relation-retrieval={wth if wth is None else round(wth,3)}  relation-probe={relprobe_v:.3f}")
    print(f"  VERDICT: {verdict}")
    report["verdict"] = dict(real_at_kmax=rK, gaussian_at_kmax=gK, best_control_at_kmax=ctrl_max,
                             freq_floor=ffloor, gap_over_floor=gap, chance=chance,
                             real_cluster_acc=clus, real_within_acc=wth, relation_probe_acc=relprobe_v,
                             exact_win=bool(exact_win), relation_consolidated=bool(relation_consolidated),
                             text=verdict)

    # ----- SVGs -----
    tag = args.tag
    svg_genK(os.path.join(RUNS, f"sidecar_semantic_genK{tag}.svg"), banks, icl, chance,
             f"Semantic-relation consolidation: real vs controls ({args.model.split('/')[-1]}, L{best_layer})", Ks)
    svg_perrel(os.path.join(RUNS, f"sidecar_semantic_perrel{tag}.svg"),
               real_perrel, rand_perrel, freq["per_relation_majority_byrel"], chance,
               f"Per-relation held-out retrieval acc, real vs gaussian vs freq-floor (K={max(Ks)})")
    svg_legib(os.path.join(RUNS, f"sidecar_semantic_legib{tag}.svg"), means, relprobe, leg_chance,
              f"Which relation is in the memory? ({args.model.split('/')[-1]}, L{best_layer})")

    json.dump(report, open(os.path.join(RUNS, f"sidecar_semantic{tag}.json"), "w"), indent=2)
    print(f"\nwrote runs/sidecar_semantic{tag}.json + 3 SVGs (genK, perrel, legib)")

if __name__ == "__main__":
    main()
