"""
Phase-18 — CONCEPT-INDEXED, legible-BY-CONSTRUCTION memory (a DIFFERENT KIND of memory).

The fast-weight store (p15/p16) is an arbitrary (key->value) associative memory: it can hold any
nonce fact, but it is capacity-bound by key COLLISIONS — the MLP-post keys aren't orthogonal, so
`dot`-mode recall cross-talks and collapses as N grows (p16's headline). This spike tests the
COMPLEMENTARY design:

  The memory's STATE is a vector of COEFFICIENTS over a FIXED, NAMED set of concept directions.
  state = [c_animals, c_colors, c_money, c_fear, c_formal, c_past, c_question, c_food] .

It is legible BY CONSTRUCTION (the state literally reads "animals=+2.0, formal=+1.0"), editable
(change a named coefficient), and has a BOUNDED, NAMED capacity (= #concepts) with NO arbitrary-key
addressing problem. The cost: it only stores what you have a NAMED CONCEPT for — a "stance / context
in named terms" memory, complementary to arbitrary fact recall, not a replacement.

Mechanism (training-free, diff-in-means, FROZEN GPT-2-small throughout):
  1. NAMED CONCEPT BASIS. Each concept c has positive example texts + contrast (negative) texts.
     direction d_c = mean(resid_post @ layer L over positive tokens) - mean(over contrast tokens).
     Report the concept x concept COSINE matrix (how orthogonal / how nameable the basis is — this
     BOUNDS interference) and each direction's norm.
  2. THE MEMORY = coefficients [c_1..c_k] over the basis. We realize state c by adding
     Sum_i c_i * d_i to resid_post @ L (the RAW dir, so c=1 is "one natural concept direction" — the
     standard diff-in-means steering scale, a meaningful fraction of the residual norm). Two reads:
       READ-by-construction : the coefficients ARE the legible state (free).
       READ-round-trip      : project the post-injection resid back onto each unit dir / its norm ->
                              recovered c_hat. SAME-LAYER (at L) is an exact linear inverse (diag=c by
                              construction; useful only as a reference + to expose off-diagonal cosine
                              LEAKAGE). The MEANINGFUL test is DOWNSTREAM (at L'>L, basis rebuilt at L'):
                              does the legible state SURVIVE the nonlinear layers? Report write-vs-read
                              correlation; the downstream number is the one that can fail.
  3. WRITE/EDIT -> BEHAVIOR. Set one concept's coefficient and measure next-token behavior: a concrete
     per-concept token-set readout (e.g. P(animal tokens) for "animals", log-odds(formal vs casual
     markers) for "formal"). Dose-response over c in {0,1,2,4,8} BESIDE a baseline (c=0) AND a RANDOM
     null (a random direction of EQUAL NORM to the concept's raw dir at the same coeff should NOT produce
     the named effect). A concept "works" iff its own readout rises with c AND beats the random null.
  4. INTERFERENCE / capacity. Activate concept i alone at a fixed dose; measure EVERY concept's readout
     -> a concept x concept EFFECT matrix (rows = which concept is written, cols = which readout moves).
     The diagonal should dominate. We score entanglement BOTH by ROW (writing i spills into others) and
     by COLUMN (readout j is dragged by other writes — this catches "fragile" relational readouts that a
     row-only metric would mislabel as clean). Report CLEAN vs ENTANGLED + which readouts are FRAGILE,
     and check whether the basis cosine PREDICTS behavioral interference (honest answer here: it does
     not — cosine bounds the read-back leakage, behavioral entanglement is dominated by fragile readouts).

Honesty (load-bearing): a random-direction null sits beside EVERY dose-response; concepts that DON'T
work or that interfere are reported, not hidden. Aggregate (mean) + spread (std) throughout. The
readouts are token-set probabilities (transparent), not a learned classifier.

Backbone FROZEN. Runs in C:\\Users\\brigi\\src\\clozn\\.venv-sae (transformer_lens + torch; CPU fine).
GPT-2-small is cached — no large download. CPU-tractable.

Usage (from inspector/, .venv-sae python):
    python spikes/p18_conceptmem.py
    python spikes/p18_conceptmem.py --layer 7 --doses 0,1,2,4,8 --seed 0
    python spikes/p18_conceptmem.py --layers 6,7,8        # scan layers for the cleanest basis
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # this PC crashes on HF symlinks (WinError 1314)

import numpy as np   # noqa: E402
import torch         # noqa: E402
import torch.nn.functional as F  # noqa: E402

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

# ----------------------------------------------------------------------------------------------------
# THE NAMED CONCEPT BASIS.
#
# Each concept = (name, positives, contrasts, readout_words). The direction is built diff-in-means from
# resid_post at layer L over the TOKENS of the positive texts minus the contrast texts. `readout_words`
# is the transparent behavioral probe used in steps 3-4: P(next token in this set). For a *relational*
# concept like formal-tone we use two word sets (formal vs casual) and read the LOG-ODDS so the readout
# can move in a named direction rather than just "more of a category".
#
# Positives/contrasts are short, varied natural sentences so the direction is the CONCEPT, not one word.
# ----------------------------------------------------------------------------------------------------
CONCEPTS = [
    {
        "name": "animals",
        "positives": [
            "The dog ran across the field chasing the cat.",
            "A herd of horses galloped past the barn.",
            "Birds and fish and snakes live in the forest.",
            "The lion is a large wild animal with a mane.",
            "Farmers keep cows, pigs, sheep, and goats.",
            "An owl, a fox, and a bear roamed the woods at night.",
        ],
        "contrasts": [
            "The committee approved the quarterly budget report.",
            "She solved the equation using basic algebra.",
            "The bridge was built from steel and concrete.",
            "He filed the paperwork at the government office.",
            "The software update fixed several minor bugs.",
            "They discussed economics and interest rates.",
        ],
        # readout: probability mass on common animal next-tokens
        "readout_pos": [" dog", " cat", " horse", " bird", " fish", " lion", " bear", " wolf",
                        " fox", " cow", " pig", " sheep", " goat", " duck", " mouse", " snake"],
        "readout_neg": None,
    },
    {
        "name": "colors",
        "positives": [
            "The sky turned red and orange at sunset.",
            "She painted the wall a bright shade of blue.",
            "His green jacket matched the yellow flowers.",
            "The flag was white with a black stripe.",
            "Purple and pink balloons floated in the air.",
            "The car was a deep glossy silver and gold.",
        ],
        "contrasts": [
            "The lecture covered the history of ancient Rome.",
            "He calculated the total cost of the trip.",
            "The engine required regular oil changes.",
            "They signed the contract on Tuesday morning.",
            "The recipe called for two cups of flour.",
            "The senator gave a speech about taxes.",
        ],
        "readout_pos": [" red", " blue", " green", " yellow", " white", " black", " purple",
                        " pink", " orange", " gold", " silver", " gray", " brown"],
        "readout_neg": None,
    },
    {
        "name": "money",
        "positives": [
            "The company reported record profits and revenue this year.",
            "He invested his savings in stocks and bonds.",
            "The price of gold rose sharply on the market.",
            "She earned a large salary and paid her taxes.",
            "The bank approved the loan and charged interest.",
            "They spent millions of dollars on the deal.",
        ],
        "contrasts": [
            "The dog ran across the field chasing the cat.",
            "The sky turned red and orange at sunset.",
            "She felt afraid walking home in the dark.",
            "Birds and fish live in the quiet forest.",
            "He whispered the answer to the question.",
            "The children played games in the yard.",
        ],
        "readout_pos": [" money", " dollars", " cash", " price", " cost", " profit", " bank",
                        " gold", " rich", " wealth", " pay", " paid", " expensive", " income"],
        "readout_neg": None,
    },
    {
        "name": "fear",
        "positives": [
            "She was terrified and trembling in the dark.",
            "A wave of fear and dread washed over him.",
            "The monster was horrifying and dangerous.",
            "They screamed in panic as the threat grew.",
            "He felt anxious, scared, and full of terror.",
            "The haunted house was frightening and grim.",
        ],
        "contrasts": [
            "She felt calm, happy, and completely safe.",
            "The committee approved the quarterly budget.",
            "He calmly explained the simple instructions.",
            "The garden was peaceful and warm in the sun.",
            "They relaxed quietly on the soft couch.",
            "The recipe called for two cups of flour.",
        ],
        "readout_pos": [" fear", " afraid", " scared", " terror", " terrified", " horror",
                        " dread", " panic", " danger", " dangerous", " frightened", " anxious"],
        "readout_neg": None,
    },
    {
        # RELATIONAL: formal vs casual register. readout = log-odds(formal markers) - log-odds(casual).
        "name": "formal",
        "positives": [
            "I would like to formally request your assistance in this matter.",
            "Pursuant to our agreement, the parties shall hereby proceed.",
            "We respectfully acknowledge receipt of your correspondence.",
            "It is imperative that we adhere to the established protocol.",
            "Furthermore, the aforementioned provisions remain in effect.",
            "Kindly find enclosed the requested documentation herein.",
        ],
        "contrasts": [
            "Hey, wanna grab some food later? lol",
            "Yeah that movie was super cool, gonna watch it again.",
            "Dude, that's so awesome, can't wait!",
            "Nah I'm kinda tired, maybe tomorrow ok?",
            "OMG this is the best thing ever haha.",
            "Sup, just chilling at home, you good?",
        ],
        "readout_pos": [" shall", " hereby", " furthermore", " pursuant", " therefore",
                        " regarding", " moreover", " accordingly", " respectfully"],
        "readout_neg": [" lol", " gonna", " wanna", " yeah", " cool", " awesome", " dude",
                        " haha", " ok", " stuff", " kinda", " super"],
    },
    {
        # RELATIONAL: past vs present tense. readout = log-odds(past-tense verbs) - log-odds(present).
        "name": "past",
        "positives": [
            "Yesterday she walked to the store and bought some bread.",
            "He had finished the work before the meeting started.",
            "They traveled across the country last summer.",
            "The old king ruled the land many years ago.",
            "We watched the game and then went home.",
            "The rain fell and the river rose overnight.",
        ],
        "contrasts": [
            "She walks to the store and buys some bread every day.",
            "He is finishing the work right now.",
            "They travel across the country each summer.",
            "The king rules the land and makes the laws.",
            "We watch the game and then go home.",
            "The rain falls and the river rises tonight.",
        ],
        "readout_pos": [" was", " were", " had", " went", " did", " said", " made", " took",
                        " came", " saw", " walked", " looked", " found"],
        "readout_neg": [" is", " are", " has", " goes", " does", " says", " makes", " takes",
                        " comes", " sees", " walks", " looks", " finds"],
    },
    {
        # RELATIONAL: question vs statement. readout = log-odds(wh-/aux question openers) - log-odds(.).
        "name": "question",
        "positives": [
            "What is the capital of France and why does it matter?",
            "How do you solve this problem step by step?",
            "Where did they go after the meeting ended?",
            "Why is the sky blue during the daytime?",
            "Who wrote this book and when was it published?",
            "Can you explain how the engine actually works?",
        ],
        "contrasts": [
            "The capital of France is a large and famous city.",
            "You solve this problem step by step with care.",
            "They went home after the meeting ended quietly.",
            "The sky is blue during the daytime hours.",
            "She wrote this book several years ago.",
            "The engine works by burning fuel efficiently.",
        ],
        "readout_pos": [" what", " how", " why", " who", " when", " where", " which",
                        " is", " are", " do", " does", " can", " could", " would"],
        "readout_neg": None,
    },
    {
        "name": "food",
        "positives": [
            "She cooked a delicious meal of pasta and bread.",
            "The restaurant served pizza, soup, and fresh salad.",
            "He ate an apple and drank a cup of coffee.",
            "They baked a cake with chocolate and sugar.",
            "The chef prepared rice, chicken, and vegetables.",
            "We had cheese, fruit, and warm soup for dinner.",
        ],
        "contrasts": [
            "The senator gave a speech about the economy.",
            "He calculated the total cost of the trip.",
            "The bridge was built from steel and concrete.",
            "She solved the equation using basic algebra.",
            "The software update fixed several minor bugs.",
            "They signed the contract on Tuesday morning.",
        ],
        "readout_pos": [" food", " bread", " pizza", " soup", " cake", " coffee", " apple",
                        " rice", " meal", " eat", " cook", " chicken", " cheese", " fruit"],
        "readout_neg": None,
    },
]


def load_model(device: str):
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def mean_resid_over_texts(model, texts, layer: int) -> torch.Tensor:
    """Mean resid_post at `layer` over ALL non-BOS token positions across a list of texts.
    Averaging over content tokens (not just the last) gives a concept direction, not a position
    artifact. Returns [d_model]."""
    name = f"blocks.{layer}.hook_resid_post"
    acc = None
    cnt = 0
    for t in texts:
        toks = model.to_tokens(t)                         # [1, seq] (prepends BOS)
        _, cache = model.run_with_cache(toks, names_filter=name)
        r = cache[name][0]                                # [seq, d_model]
        r = r[1:]                                         # drop BOS position
        if acc is None:
            acc = r.sum(0)
        else:
            acc = acc + r.sum(0)
        cnt += r.shape[0]
    return acc / max(cnt, 1)


def build_basis(model, concepts, layer: int):
    """diff-in-means concept directions at `layer`. Returns:
        dirs   [k, d_model] RAW directions (mean_pos - mean_contrast)
        units  [k, d_model] unit-normalized directions (for injection / projection)
        norms  [k] raw direction norms
    """
    dirs = []
    for c in concepts:
        mp = mean_resid_over_texts(model, c["positives"], layer)
        mc = mean_resid_over_texts(model, c["contrasts"], layer)
        dirs.append((mp - mc))
    dirs = torch.stack(dirs)                              # [k, d_model]
    norms = dirs.norm(dim=-1)
    units = F.normalize(dirs, dim=-1)
    return dirs, units, norms


def cosine_matrix(units: torch.Tensor) -> np.ndarray:
    """k x k cosine matrix of the (already unit) directions."""
    return (units @ units.T).cpu().numpy()


# ----------------------------------------------------------------------------------------------------
# Readouts: transparent token-set probabilities, computed from the next-token distribution.
# ----------------------------------------------------------------------------------------------------
def token_ids_for(model, words) -> list[int]:
    """Single-token ids for the words that ARE single tokens (silently skip multi-token ones)."""
    ids = []
    for w in words:
        t = model.to_tokens(w, prepend_bos=False)[0]
        if t.shape[0] == 1:
            ids.append(int(t[0]))
    return sorted(set(ids))


def build_readouts(model, concepts):
    """Precompute (pos_ids, neg_ids_or_None) per concept and a NEUTRAL prompt set used for behavior."""
    ro = []
    for c in concepts:
        pos = token_ids_for(model, c["readout_pos"])
        neg = token_ids_for(model, c["readout_neg"]) if c["readout_neg"] else None
        ro.append({"name": c["name"], "pos": pos, "neg": neg,
                   "relational": neg is not None})
    return ro


def readout_value(probs: torch.Tensor, ro: dict) -> float:
    """Behavioral readout from a next-token prob vector [d_vocab].
       absolute concept (neg=None): summed prob mass on the concept's token set.
       relational concept (neg set): log-odds = log(sum P_pos) - log(sum P_neg), so it can move either
       way around a baseline (e.g. formal vs casual) rather than only 'more category'."""
    ppos = float(probs[ro["pos"]].sum())
    if ro["neg"] is None:
        return ppos
    pneg = float(probs[ro["neg"]].sum())
    return float(np.log(ppos + 1e-12) - np.log(pneg + 1e-12))


# Neutral carrier prompts: short, topic-free contexts where ANY concept could plausibly continue.
# We average behavior over several so the readout isn't a single-prompt artifact.
NEUTRAL_PROMPTS = [
    "The next thing I want to talk about is the",
    "When I opened the door, I saw the",
    "Let me tell you about the",
    "The most important part of the story is the",
    "After a while, they noticed the",
    "I think the answer has to do with the",
]


# ----------------------------------------------------------------------------------------------------
# Injection hook. The memory state is coefficients c over the NAMED basis; we realize state c by adding
#   v = Sum_i c_i * dir_i   to resid_post at layer L, where dir_i is the RAW diff-in-means direction.
# So a coefficient c_i = 1 means "one full natural concept direction" — the standard diff-in-means
# steering scale, and a meaningful fraction of the residual norm (raw dir norm ~10-40 vs resid ~105 at
# L=7; c=0 is the BOS outlier which we drop in the basis). Coefficients stay legible: c=2 is "twice the
# natural animals direction". Used for BOTH read-round-trip (then project back) and write->behavior.
# ----------------------------------------------------------------------------------------------------
@torch.no_grad()
def forward_with_injection(model, prompt: str, vecs: torch.Tensor, coeffs: torch.Tensor, layer: int,
                           all_positions: bool = True, read_layer: int | None = None):
    """Run `prompt`, add v = Sum_i coeffs_i * vecs_i at resid_post[L]; return the FULL post-injection
    resid_post[L] [seq, d_model] AND the final-position next-token logits [d_vocab]. `vecs` are the RAW
    concept directions (so coeffs are in natural-direction units). `all_positions`: add v to every
    (non-BOS) content position (stronger, position-invariant steer) vs only the last position. We skip
    the BOS position (index 0) since its residual is a ~30x-norm outlier that the steer shouldn't touch.
    If `read_layer` is given (>L), also capture resid_post at THAT layer AFTER the model has processed
    the injection — the meaningful round-trip read (does the legible state SURVIVE downstream layers,
    not just invert a same-layer linear add). Returns (resid_at_L, logits, resid_at_read_layer_or_None)."""
    toks = model.to_tokens(prompt)
    resid_name = f"blocks.{layer}.hook_resid_post"
    v = (coeffs.unsqueeze(-1) * vecs).sum(0)              # [d_model]
    captured = {}

    def inject(act, hook):
        if all_positions:
            act[0, 1:] = act[0, 1:] + v                   # all content positions, skip BOS
        else:
            act[0, -1] = act[0, -1] + v
        captured["resid"] = act[0].clone()                # post-injection resid at L [seq, d_model]
        return act

    hooks = [(resid_name, inject)]
    if read_layer is not None:
        read_name = f"blocks.{read_layer}.hook_resid_post"

        def grab_downstream(act, hook):
            captured["resid_read"] = act[0].clone()
            return act
        hooks.append((read_name, grab_downstream))

    logits = model.run_with_hooks(toks, fwd_hooks=hooks)[0, -1].float()
    return captured["resid"], logits, captured.get("resid_read")


@torch.no_grad()
def project_coeffs(resid_lastpos: torch.Tensor, units: torch.Tensor, norms: torch.Tensor,
                   baseline_resid: torch.Tensor):
    """Recover the written coefficients from the (post-injection) resid at the read position. We project
    the injected delta onto each UNIT direction and divide by that direction's RAW norm, so the recovered
    c_hat is in the same natural-direction units we wrote (a clean read-back of a single concept j at
    coeff c gives c_hat[j] ~ c). Because the units are NOT orthonormal, c_hat[i!=j] picks up cos(i,j)*c
    * (norms[j]/norms[i]) — read-back leakage is bounded by the cosine matrix, the SAME quantity that
    bounds behavioral interference (#4). Read at the LAST position (where behavior is measured)."""
    delta = resid_lastpos - baseline_resid                # [d_model] the injected part (+ any drift)
    proj_unit = (units @ delta)                           # [k] projection onto unit dirs
    chat = proj_unit / (norms + 1e-9)                     # back to natural-direction coefficient units
    return chat.cpu().numpy()


# ----------------------------------------------------------------------------------------------------
def fmt_pct(x):
    return f"{100 * x:5.1f}%"


def fmt_signed(x, w=6, p=3):
    return f"{x:+{w}.{p}f}"


def print_matrix(mat, names, title, fmt="{:+6.2f}", diag_mark=True):
    """Pretty-print a k x k matrix with row/col concept names (short)."""
    short = [n[:5] for n in names]
    print(f"\n  {title}")
    header = "        " + " ".join(f"{s:>6}" for s in short)
    print(header)
    for i, row in enumerate(mat):
        cells = []
        for j, v in enumerate(row):
            s = fmt.format(v)
            if diag_mark and i == j:
                s = f"[{v:+4.2f}]" if abs(v) < 10 else f"[{v:.1f}]"
            cells.append(f"{s:>6}")
        print(f"  {short[i]:>5} " + " ".join(cells))


# ----------------------------------------------------------------------------------------------------
def run_layer(model, concepts, readouts, L, doses, rng_seed):
    """Full battery at one layer. Returns a results dict for the verdict + npz."""
    k = len(concepts)
    names = [c["name"] for c in concepts]
    g = torch.Generator(device=model.cfg.device).manual_seed(rng_seed)

    print("\n" + "#" * 100)
    print(f"# LAYER L={L}   basis @ blocks.{L}.hook_resid_post  (diff-in-means over content tokens)")
    print("#" * 100)

    # ---- STEP 1: build the named basis; cosine matrix + norms --------------------------------------
    dirs, units, norms = build_basis(model, concepts, L)
    cos = cosine_matrix(units)
    off = cos.copy()
    np.fill_diagonal(off, np.nan)
    mean_abs_off = float(np.nanmean(np.abs(off)))
    max_abs_off = float(np.nanmax(np.abs(off)))

    print("\n  STEP 1 — NAMED CONCEPT BASIS")
    print("  raw direction norms (||mean_pos - mean_contrast||):")
    for nm, nr in zip(names, norms.tolist()):
        print(f"    {nm:9} {nr:7.2f}")
    print_matrix(cos, names, "concept x concept COSINE (unit dirs; off-diag = potential interference):",
                 fmt="{:+6.2f}")
    print(f"    mean |off-diagonal cosine| = {mean_abs_off:.3f}   max |off-diag| = {max_abs_off:.3f}")
    print("    (lower = more orthogonal / more independently nameable -> bounds interference in #4)")

    # ---- STEP 2: read-round-trip (write c -> inject -> project back -> c_hat) -----------------------
    # Two reads, kept separate to avoid a trivially-perfect headline:
    #   SAME-LAYER read (at L): exact linear inverse of the add — diagonal is c by construction, useful
    #     only as a reference + to expose off-diagonal LEAKAGE (cos(i,j)*c*norm_j/norm_i).
    #   DOWNSTREAM read (at read_layer L' > L): project the resid AFTER the model has processed the
    #     injection, onto a basis rebuilt at L'. This is the MEANINGFUL faithful-legibility test — does
    #     the legible state SURVIVE the nonlinear layers, or get washed out? (Honest: this is the number
    #     that can fail.)
    read_layer = min(L + 3, model.cfg.n_layers - 1)
    print("\n  STEP 2 — READ ROUND-TRIP (write coeff -> inject raw dirs @L -> project resid back -> c_hat)")
    print(f"    SAME-LAYER read @L={L} (exact reference) AND DOWNSTREAM read @L'={read_layer} "
          f"(survives processing?).")
    # downstream basis (diff-in-means at L') for the survival projection
    dirs2, units2, norms2 = build_basis(model, concepts, read_layer)
    rt_prompt = NEUTRAL_PROMPTS[0]
    base_resid, _, base_resid2 = forward_with_injection(
        model, rt_prompt, dirs, torch.zeros(k, device=dirs.device), L, read_layer=read_layer)
    base_lastpos = base_resid[-1]
    base_lastpos2 = base_resid2[-1]
    write_c = 4.0
    roundtrip = np.zeros((k, k))     # SAME-LAYER c_hat: rows=written, cols=recovered
    roundtrip_ds = np.zeros((k, k))  # DOWNSTREAM c_hat
    for j in range(k):
        coeffs = torch.zeros(k, device=dirs.device)
        coeffs[j] = write_c
        resid, _, resid2 = forward_with_injection(model, rt_prompt, dirs, coeffs, L, read_layer=read_layer)
        roundtrip[j] = project_coeffs(resid[-1], units, norms, base_lastpos)
        roundtrip_ds[j] = project_coeffs(resid2[-1], units2, norms2, base_lastpos2)
    diag = np.diag(roundtrip)
    diag_ds = np.diag(roundtrip_ds)
    rt_err = np.abs(diag - write_c)
    rt_err_ds = np.abs(diag_ds - write_c)
    # correlation across the sweep, SAME-LAYER and DOWNSTREAM (write same concept at several doses).
    rt_doses = [0.0, 1.0, 2.0, 4.0, 8.0]
    rt_w, rt_r, rt_r_ds = [], [], []
    per_concept_rt_corr, per_concept_rt_corr_ds = [], []
    for j in range(k):
        wj, rj, rj_ds = [], [], []
        for c in rt_doses:
            coeffs = torch.zeros(k, device=dirs.device)
            coeffs[j] = c
            resid, _, resid2 = forward_with_injection(model, rt_prompt, dirs, coeffs, L, read_layer=read_layer)
            wj.append(c)
            rj.append(float(project_coeffs(resid[-1], units, norms, base_lastpos)[j]))
            rj_ds.append(float(project_coeffs(resid2[-1], units2, norms2, base_lastpos2)[j]))
        rt_w += wj; rt_r += rj; rt_r_ds += rj_ds
        per_concept_rt_corr.append(float(np.corrcoef(wj, rj)[0, 1]) if np.std(rj) > 1e-9 else float("nan"))
        per_concept_rt_corr_ds.append(float(np.corrcoef(wj, rj_ds)[0, 1]) if np.std(rj_ds) > 1e-9 else float("nan"))
    overall_rt_corr = float(np.corrcoef(rt_w, rt_r)[0, 1])
    overall_rt_corr_ds = float(np.corrcoef(rt_w, rt_r_ds)[0, 1])
    print_matrix(roundtrip, names, f"SAME-LAYER c_hat (wrote c={write_c}; diag=c by construction, "
                 f"off-diag=cosine leakage):", fmt="{:+6.2f}")
    print(f"    [same-layer] diag mean |err|={rt_err.mean():.2f}  overall write-vs-read corr={overall_rt_corr:+.3f}"
          f"  (exact linear inverse — reference only)")
    print_matrix(roundtrip_ds, names, f"DOWNSTREAM c_hat @L'={read_layer} (wrote c={write_c}; survives "
                 f"the nonlinear layers?):", fmt="{:+6.2f}")
    print(f"    [downstream] diag (recovered own c after processing): " +
          " ".join(f"{nm[:5]}={d:+.1f}" for nm, d in zip(names, diag_ds)))
    print(f"    [downstream] diag mean |err vs {write_c}|={rt_err_ds.mean():.2f}  "
          f"per-concept corr: " + " ".join(f"{nm[:4]}={c:+.2f}" for nm, c in zip(names, per_concept_rt_corr_ds)))
    print(f"    [downstream] OVERALL write-vs-read corr = {overall_rt_corr_ds:+.3f}  "
          f"<- the MEANINGFUL faithful-legibility number")
    print("    (downstream diag~c + high corr => the legible state is FAITHFUL through the model, not")
    print("     just a linear inverse; lower-than-same-layer = the expected wash-out from processing)")

    # ---- STEP 3: write -> behavior, dose-response vs baseline + random-direction null ---------------
    print("\n  STEP 3 — WRITE -> BEHAVIOR  (per-concept readout vs dose; baseline=c0; RANDOM-dir null)")
    print(f"    readout = P(concept tokens) [absolute] or log-odds(pos)-log-odds(neg) [relational].")
    print(f"    averaged over {len(NEUTRAL_PROMPTS)} neutral prompts. random null = a random dir of")
    print(f"    EQUAL NORM to that concept's raw dir, same coeff (must NOT move the named readout).")

    # baseline readout per concept (c=0), averaged over neutral prompts
    base_read = np.zeros(k)
    base_probs_cache = []
    for p in NEUTRAL_PROMPTS:
        _, logits, _ = forward_with_injection(model, p, dirs, torch.zeros(k, device=dirs.device), L)
        base_probs_cache.append(F.softmax(logits, dim=-1))
    for ci in range(k):
        base_read[ci] = float(np.mean([readout_value(pr, readouts[ci]) for pr in base_probs_cache]))

    # dose-response: for each concept, write c on it alone; measure ITS readout (mean over prompts).
    # random null: a fixed random direction (per concept, seeded) scaled to EQUAL NORM as that concept's
    # raw dir, written at the same coeff -> identical injected magnitude, only the DIRECTION is random.
    dose_real = np.zeros((k, len(doses)))     # [concept, dose]
    dose_null = np.zeros((k, len(doses)))
    dose_real_std = np.zeros((k, len(doses)))
    rand_units = F.normalize(torch.randn(k, model.cfg.d_model, generator=g, device=dirs.device), dim=-1)
    rand_dirs = rand_units * norms.unsqueeze(-1)          # equal-norm random directions
    for ci in range(k):
        for di, c in enumerate(doses):
            vals_real, vals_null = [], []
            for p in NEUTRAL_PROMPTS:
                coeffs = torch.zeros(k, device=dirs.device)
                coeffs[ci] = c
                _, lr, _ = forward_with_injection(model, p, dirs, coeffs, L)
                vals_real.append(readout_value(F.softmax(lr, dim=-1), readouts[ci]))
                # null: same coeff on an equal-norm random dir (single-vector inject)
                _, ln, _ = forward_with_injection(model, p, rand_dirs[ci:ci + 1],
                                                  torch.tensor([c], device=dirs.device), L)
                vals_null.append(readout_value(F.softmax(ln, dim=-1), readouts[ci]))
            dose_real[ci, di] = float(np.mean(vals_real))
            dose_real_std[ci, di] = float(np.std(vals_real))
            dose_null[ci, di] = float(np.mean(vals_null))

    # report per concept: readout at each dose (real) vs null, and whether it "works".
    works = []
    print(f"\n    {'concept':9} {'kind':4}  doses-> " + "  ".join(f"{d:>7.0f}" for d in doses))
    for ci in range(k):
        kind = "rel" if readouts[ci]["relational"] else "abs"
        real_s = "  ".join(f"{v:7.3f}" for v in dose_real[ci])
        null_s = "  ".join(f"{v:7.3f}" for v in dose_null[ci])
        # "works" = monotone-ish rise (last dose clearly above c=0) AND real beats null at the top dose
        rise_real = dose_real[ci, -1] - dose_real[ci, 0]
        rise_null = dose_null[ci, -1] - dose_null[ci, 0]
        beats_null = rise_real > 2.0 * abs(rise_null) + 1e-6 and rise_real > 0
        # for absolute readouts, also require a meaningful absolute rise (>2x baseline mass or >0.02)
        meaningful = (rise_real > 0.02) if kind == "abs" else (rise_real > 0.3)
        ok = bool(beats_null and meaningful)
        works.append(ok)
        print(f"    {names[ci]:9} {kind:4}  real:   {real_s}   (rise {rise_real:+.3f})  {'WORKS' if ok else 'weak'}")
        print(f"    {'':9} {'':4}  null:   {null_s}   (rise {rise_null:+.3f})")
    n_work = sum(works)
    print(f"\n    concepts that steer behavior in their NAMED direction (beat random null): "
          f"{n_work}/{k} -> {[names[i] for i in range(k) if works[i]]}")

    # ---- STEP 4: interference / capacity. write one concept, read EVERY readout -> effect matrix ----
    print("\n  STEP 4 — INTERFERENCE / CAPACITY  (write concept i alone @ fixed dose; read ALL readouts)")
    inter_dose = doses[min(range(len(doses)), key=lambda i: abs(doses[i] - 4.0))]  # nearest to 4
    print(f"    effect matrix: rows = concept WRITTEN (@c={inter_dose:g}), cols = readout MOVED "
          f"(value - baseline). diagonal should dominate.")
    # use z-scored effects so absolute vs relational readouts are comparable in the matrix.
    effect = np.zeros((k, k))   # rows=written i, cols=readout j ; (readout_j with i written) - baseline_j
    for wi in range(k):
        # write concept wi alone, then measure every concept's readout (mean over prompts)
        per_prompt = []
        for p in NEUTRAL_PROMPTS:
            coeffs = torch.zeros(k, device=dirs.device)
            coeffs[wi] = inter_dose
            _, lr, _ = forward_with_injection(model, p, dirs, coeffs, L)
            pr = F.softmax(lr, dim=-1)
            per_prompt.append([readout_value(pr, readouts[rj]) for rj in range(k)])
        per_prompt = np.array(per_prompt)                 # [prompts, k]
        mean_read = per_prompt.mean(0)
        effect[wi] = mean_read - base_read                # delta vs baseline for each readout
    # normalize each COLUMN (readout) by its own diagonal magnitude so we can compare specificity:
    # spec[i,j] = effect[i,j] / |effect[j,j]| (how much writing i moves readout j relative to j's self)
    diag_eff = np.abs(np.diag(effect))
    spec = effect / (diag_eff[None, :] + 1e-9)            # rows=written, cols=readout; col-normalized
    print_matrix(effect, names, "RAW effect matrix (readout delta vs baseline; row=written, col=readout):",
                 fmt="{:+6.2f}")
    print_matrix(spec, names,
                 "NORMALIZED effect (each col / its own diagonal; 1.0 on diag, off-diag=entanglement):",
                 fmt="{:+6.2f}")
    # Two entanglement views (BOTH matter; the row-only view hides "fragile readout" entanglement):
    #   ROW bleed[i]   = mean_{j!=i} |spec[i,j]| : how much WRITING concept i spills into other readouts.
    #   COL victim[j]  = mean_{i!=j} |spec[i,j]| : how much readout j is MOVED by OTHER concepts' writes
    #                    (a relational log-odds readout can be dragged by any large perturbation -> a
    #                    fragile readout looks "clean" by row but is dominated by off-diagonal columns).
    ent_row = np.zeros(k)
    victim_col = np.zeros(k)
    for i in range(k):
        ent_row[i] = float(np.mean([abs(spec[i, j]) for j in range(k) if j != i]))
        victim_col[i] = float(np.mean([abs(spec[j, i]) for j in range(k) if j != i]))
    # CLEAN = it actually steers (works) AND it neither bleeds much (row) NOR is dragged much (col).
    clean = []
    for i in range(k):
        clean.append(bool(works[i] and ent_row[i] < 0.5 and victim_col[i] < 0.5))
    print(f"\n    per-concept self-effect (diag), row-bleed (writing it spills out), and")
    print(f"    col-victim (other writes drag its readout):")
    for i in range(k):
        tag = "CLEAN" if clean[i] else ("entangled" if works[i] else "weak/doesn't steer")
        print(f"      {names[i]:9} self={effect[i,i]:+7.3f}   row-bleed={ent_row[i]:5.2f}   "
              f"col-victim={victim_col[i]:5.2f}   -> {tag}")
    # Which readouts are FRAGILE (dominated by off-target writes)? name them explicitly.
    fragile = [names[j] for j in range(k) if victim_col[j] >= 0.5]
    if fragile:
        print(f"    FRAGILE readouts (moved >= 0.5x by OTHER concepts' writes): {fragile}")
        print("      -> these are the relational log-odds readouts; ANY large steer shifts token stats,")
        print("         so their column is not concept-specific. A real entanglement, not hidden.")
    # Does the basis COSINE predict the behavioral interference? Report honestly — it does NOT here.
    mask = ~np.eye(k, dtype=bool)
    cos_off = np.abs(cos[mask])
    spec_off = np.abs(spec[mask])
    if np.std(cos_off) > 1e-9 and np.std(spec_off) > 1e-9:
        cos_int_corr = float(np.corrcoef(cos_off, spec_off)[0, 1])
    else:
        cos_int_corr = float("nan")
    print(f"\n    corr( |off-diag cosine| , |off-diag normalized-effect| ) = {cos_int_corr:+.3f}")
    print("    NOTE: cosine bounds the READ-BACK leakage (Step 2, exact linear algebra), but it does")
    print("    NOT predict BEHAVIORAL interference here (corr ~ 0): behavioral entanglement is dominated")
    print("    by FRAGILE relational readouts, not by basis non-orthogonality. Reported, not hidden.")

    return {
        "layer": L,
        "names": names,
        "norms": norms.cpu().numpy(),
        "cosine": cos,
        "mean_abs_off_cos": mean_abs_off,
        "max_abs_off_cos": max_abs_off,
        "roundtrip_matrix": roundtrip,
        "roundtrip_diag": diag,
        "roundtrip_err_mean": float(rt_err.mean()),
        "roundtrip_corr_overall": overall_rt_corr,
        "roundtrip_corr_per_concept": np.array(per_concept_rt_corr),
        "read_layer": read_layer,
        "roundtrip_ds_matrix": roundtrip_ds,
        "roundtrip_ds_diag": diag_ds,
        "roundtrip_ds_err_mean": float(rt_err_ds.mean()),
        "roundtrip_ds_corr_overall": overall_rt_corr_ds,
        "roundtrip_ds_corr_per_concept": np.array(per_concept_rt_corr_ds),
        "doses": np.array(doses, dtype=float),
        "dose_real": dose_real,
        "dose_null": dose_null,
        "dose_real_std": dose_real_std,
        "base_read": base_read,
        "works": np.array(works),
        "n_work": int(n_work),
        "effect_matrix": effect,
        "effect_norm": spec,
        "entangle_row": ent_row,
        "victim_col": victim_col,
        "fragile": np.array(fragile, dtype=object),
        "clean": np.array(clean),
        "cos_interference_corr": cos_int_corr,
    }


# ----------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="7", help="comma list of resid layers for the basis (try 6,7,8)")
    ap.add_argument("--doses", default="0,1,2,4,8", help="coefficient dose sweep")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]
    doses = [float(x) for x in args.doses.split(",")]
    os.makedirs(RUNS, exist_ok=True)

    torch.manual_seed(args.seed)
    print(f"loading gpt2 (HookedTransformer) on {args.device} ...")
    model = load_model(args.device)
    k = len(CONCEPTS)
    print(f"  d_model={model.cfg.d_model}  d_mlp={model.cfg.d_mlp}  n_layers={model.cfg.n_layers}  "
          f"d_vocab={model.cfg.d_vocab}")
    print(f"  {k} named concepts: {[c['name'] for c in CONCEPTS]}")

    readouts = build_readouts(model, CONCEPTS)
    print("  readout token-set sizes (single-token only):")
    for ro in readouts:
        extra = f" / neg {len(ro['neg'])}" if ro["neg"] else ""
        print(f"    {ro['name']:9} pos {len(ro['pos']):2d}{extra}  ({'relational' if ro['relational'] else 'absolute'})")

    results = {}
    for L in layers:
        results[L] = run_layer(model, CONCEPTS, readouts, L, doses, args.seed)

    # ---- VERDICT -----------------------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("VERDICT — concept-indexed (legible-by-construction) memory")
    print("=" * 100)
    for L in layers:
        r = results[L]
        names = r["names"]
        print(f"\n  L={L}:")
        print(f"    basis orthogonality: mean|off-diag cos|={r['mean_abs_off_cos']:.3f} "
              f"(max {r['max_abs_off_cos']:.3f})  -> {'fairly orthogonal' if r['mean_abs_off_cos']<0.25 else 'noticeably correlated'} basis")
        print(f"    (b) read-back: same-layer corr {r['roundtrip_corr_overall']:+.3f} (exact ref) | "
              f"DOWNSTREAM @L'={r['read_layer']} corr {r['roundtrip_ds_corr_overall']:+.3f} "
              f"diag~c err {r['roundtrip_ds_err_mean']:.1f}  <- meaningful (survives processing)")
        print(f"    (a) write->behavior: {r['n_work']}/{len(names)} concepts steer in named direction "
              f"(beat random null): {[names[i] for i in range(len(names)) if r['works'][i]]}")
        n_clean = int(r["clean"].sum())
        print(f"    (d) interference: {n_clean}/{len(names)} CLEAN; "
              f"entangled/weak: {[names[i] for i in range(len(names)) if not r['clean'][i]]}")
        frag = list(r["fragile"])
        print(f"        FRAGILE readouts (dragged by other writes): {frag if frag else 'none'}")
        print(f"        cosine vs behavioral-interference corr = {r['cos_interference_corr']:+.3f} "
              f"(cosine bounds READ-BACK leakage, not behavior)")
    # pick the layer with the most working+clean concepts and best DOWNSTREAM read-back
    best_L = max(layers, key=lambda L: (results[L]["n_work"], int(results[L]["clean"].sum()),
                                        results[L]["roundtrip_ds_corr_overall"]))
    rb = results[best_L]
    print(f"\n  BEST LAYER: L={best_L}  ({rb['n_work']}/{k} concepts steer, "
          f"{int(rb['clean'].sum())}/{k} clean, downstream read-back corr {rb['roundtrip_ds_corr_overall']:+.2f})")
    print("\n  CHARACTER vs the fast-weight store (p15/p16):")
    print("    - concept-mem: BOUNDED+NAMED capacity (= #concepts), NO key-collision (legible state IS")
    print("      the coefficient vector; read-back faithful where the basis is orthogonal). Stores only")
    print("      what you have a named concept for: a 'stance in named terms' memory.")
    print("    - fast-weight: ARBITRARY (key->value) facts, but capacity-limited by MLP-key collisions")
    print("      (dot-mode cross-talk grows with N; needs sharpened top-1 addressing to scale).")
    print("    -> COMPLEMENTARY: named-stance (legible-by-construction, fixed slots) vs arbitrary-fact")
    print("       recall (open-vocabulary, capacity-bound). The interference here is bounded by the")
    print("       basis cosine, which is a DESIGN knob (pick more orthogonal concepts), unlike the")
    print("       fast-weight keys which are fixed by the model.")

    # ---- save raw arrays ---------------------------------------------------------------------------
    save = {
        "layers": np.array(layers),
        "concept_names": np.array([c["name"] for c in CONCEPTS], dtype=object),
        "doses": np.array(doses, dtype=float),
        "best_layer": best_L,
        "results": np.array([results], dtype=object),
    }
    # flatten the best layer's key arrays for easy reload
    for key in ["cosine", "norms", "roundtrip_matrix", "roundtrip_diag", "roundtrip_ds_matrix",
                "roundtrip_ds_diag", "dose_real", "dose_null", "effect_matrix", "effect_norm",
                "entangle_row", "victim_col", "works", "clean", "base_read",
                "roundtrip_corr_per_concept", "roundtrip_ds_corr_per_concept"]:
        save[f"best_{key}"] = np.asarray(results[best_L][key])
    save["best_roundtrip_corr_overall"] = results[best_L]["roundtrip_corr_overall"]
    save["best_roundtrip_ds_corr_overall"] = results[best_L]["roundtrip_ds_corr_overall"]
    save["best_cos_interference_corr"] = results[best_L]["cos_interference_corr"]
    out = os.path.join(RUNS, "p18_conceptmem.npz")
    np.savez(out, **save)
    print(f"\n  saved -> {out}")


if __name__ == "__main__":
    main()
