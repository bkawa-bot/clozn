"""vector_telepathy.py -- VECTOR TELEPATHY: do slot-memory KEY vectors port between models through
a fitted linear bridge, WITHOUT text recompilation?

THE CLAIM UNDER TEST. A prior session (the dispatcher) asserted activation-space memories "cannot
port because geometries differ." This rig gives that claim the receipts treatment: build the 20-fact
slot store on model A (Qwen2.5-1.5B-Instruct bf16, L18, H=1536), fit a linear bridge A->B on ~256
diverse sentences (last-token residuals at the slot layer from BOTH models), port ONLY the key
vectors through the bridge into B (Qwen2.5-7B-Instruct nf4, L18, H=3584), and measure recall on B --
with the text-recompile port as the ceiling and two nulls beside every number. Reverse direction
(B->A: do big thoughts compress?) runs the same table.

SCOPING, STATED HONESTLY. A and B share one tokenizer/vocab, so slot VALUES need no bridge: ans_ids
are identical and each value is just W_U[ans_id] taken from the TARGET model's own unembedding (the
answer strings would recompile trivially anyway). The pure telepathy question is the KEYS -- the
cue's meaning as a residual vector, which has no text form. Port arms are allowed to carry ONLY
{key vector, ans_ids} across; recompiling keys from cue text on the target side is exactly the
ceiling arm, not telepathy. Same-family same-vocab is the EASY case; cross-family is untested here.

PRE-REGISTERED EXPECTATIONS (written 2026-07-02, BEFORE any run; grade these in the findings):
  E1  Bridge fit quality, ridge-256 A->B on held-out sentences: CENTERED cosine (subtract B's fit
      mean before comparing -- mirrors the store's centered addressing) mean 0.40-0.75. Raw cosine
      will be inflated by anisotropy (~+0.2, deceptively high even for bad maps).
  E2  Cue-key diagnostic (bridged A-keys vs B's own native keys for the same cues, centered): within
      ~0.10 of E1's mean. Cues are short prefix-style texts; the fit set includes that genre.
  E3  A->B ridge-256 port, exact cues (n=20): SELECT 0.55-0.85; EXPRESS ~= select x 0.85 (value +
      dose mechanics identical to ceiling, so expression conditional on selection should match).
  E4  A->B ridge-256, paraphrases (n=10): select 0.35-0.65 -- meaning partially survives the bridge.
  E5  Procrustes-on-PCA (rotation-only, d=192): 0.05-0.25 below ridge on select, still >= 3x nulls.
      If Procrustes ~= ridge, the spaces are rotation-similar, not just linearly reachable.
  E6  Nulls: random-matrix bridge select <= 0.10 (~1/20 chance); shuffled-pair fit select <= 0.15.
      If shuffled-fit reaches half of ridge, the bridge is a distributional bias, not a map -- the
      port claim voids.
  E7  B->A within +-0.15 of A->B on select (no strong prior on direction; 3584->1536 loses rank but
      keys are directions, and ridge can project).
  E8  Ridge-512 beats ridge-256 on held-out cosine by +0.02-0.10 (N < H=1536: the fit is
      sample-starved at 256).
  E9  Ceilings (text recompile, gate off): B exact express ~0.75-0.85 (slotmem 7B measured 0.80);
      A exact express ~0.90. Paraphrase ceiling ~9/10 on both (ungated right-count).
  E10 Verdict I expect: PARTIAL-to-KILL of the impossibility claim for the same-family case --
      ports land well above both nulls through a 256-sentence linear bridge, at a real loss vs the
      text-recompile ceiling.
  Grading rule (pre-set): KILL if ridge express >= 70% of ceiling express AND both nulls collapsed;
  CONFIRM if ridge select <= 2x random-null select OR centered bridge cosine < 0.35 (geometry
  genuinely resists linear alignment); PARTIAL otherwise.

DESIGN NOTES (pre-registered choices):
  - Bridge is fit on UNIT-NORMALIZED last-token residuals (direction is what top-1 cosine addressing
    uses), centered ridge with exact bias (W on centered pairs, bias = muY - muX @ W), lambda chosen
    on a validation split of the fit set.
  - Fit sentences: a fixed in-file list, statements/questions/instructions/narrative across many
    domains and lengths, INCLUDING ~15-20% incomplete "prefix-style" lines ("The capital of Sweden
    is") because keys are prefix-style texts -- but NEVER the six fact cue templates, any nonce
    subject, or any fact answer-in-context. Leakage assert enforces the nonce exclusion.
  - Ported keys are centered/renormalized AMONG THEMSELVES on the target side (the store's own read
    mechanics via SlotMem.read(entries=...)); queries center by the same pool mean. eta is the
    TARGET model's own calibration (1.5x its L18 residual norm) for every arm including ceiling.
  - SELECT (top-1 key == the queried fact's own entry) isolates key geometry -- the actual telepathy
    question; EXPRESS (answer argmax after injection) adds the value/dose mechanics on top.
  - Next-token metrics only (no emit/generation); gate off everywhere (gate calibration differs per
    arm and is not the question).

ARMS (x {exact 20 cues, 10 paraphrases} x {A->B, B->A}):
  no_memory | ceiling (text recompile on target) | ridge-256 | ridge-512 | procrustes-pca192 |
  random-matrix bridge (norm-matched) | shuffled-pair fit (same lambda as ridge)

GPU CITIZENSHIP: before every model load, poll nvidia-smi and proceed only when total usage < 3 GB
sustained for ~2 min (the :8080 engine's ~1 GB stays; never kill anything). Models load SEQUENTIALLY
(A -> free -> B -> free -> A), profile_port_demo-style. Stage caches under research/runs/telepathy*/
make a crashed run resumable per stage; per-arm JSON checkpoints as each arm completes.

usage:
    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/vector_telepathy.py --selftest
    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/vector_telepathy.py --smoke
    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/vector_telepathy.py
"""
from __future__ import annotations
import argparse, gc, json, os, random, subprocess, sys, time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

MODEL_A = "Qwen/Qwen2.5-1.5B-Instruct"   # bf16, H=1536
MODEL_B = "Qwen/Qwen2.5-7B-Instruct"     # nf4,  H=3584
LAYER = 18
HELD_N_FULL, HELD_N_SMOKE = 32, 16
CANARY = ["The quick brown fox jumped.", " seven", " Zephyr", "What is the capital of France?"]
NONCE = ["zorbland", "velk", "prynne", "maar", "tarnow", "ossic", "brell", "dole harbor", "wrenmoor",
         "kest", "halden", "fenwick", "grellstead", "ondine", "quill society", "larch vault",
         "velo downs", "cape morrow", "halloway", "bryce war", "zephyr", "nimbus", "beatrix",
         "tamarind", "pippin", "ingrid", "juniper", "dmitri"]


# ---------------------------------------------------------------------------------------------------
# The bridge-fit corpus: a fixed, deterministic, in-file sentence list. Diversity on purpose:
# statements, questions, instructions, narrative, numbers, short exclamations, long multi-clause --
# plus a minority of incomplete prefix-style lines (keys are prefix-style). No nonce subjects.
# ---------------------------------------------------------------------------------------------------
_LITERALS = [
    "Photosynthesis converts sunlight into chemical energy.",
    "Sound travels faster through water than through air.",
    "The human heart beats about a hundred thousand times a day.",
    "Glaciers carve valleys over thousands of years.",
    "Copper conducts electricity better than iron.",
    "The printing press changed how ideas spread across Europe.",
    "Ancient traders carried silk and spices along dusty roads.",
    "The lighthouse guided ships safely into the harbor for two centuries.",
    "Rivers that begin in the mountains often end in wide deltas.",
    "The desert stretches for miles beyond the last village.",
    "Add the garlic only after the onions turn golden.",
    "Fresh basil loses its flavor if you cook it too long.",
    "Knead the dough until it springs back under your thumb.",
    "The server crashed twice before anyone noticed the logs.",
    "Software updates often arrive at the least convenient moment.",
    "Her laptop battery dies before lunch every single day.",
    "The striker scored twice in the final ten minutes.",
    "Marathon runners train for months before race day.",
    "The referee waved play on despite the loud protests.",
    "Thunderclouds gathered over the valley by late afternoon.",
    "A thin layer of frost covered the windshield this morning.",
    "Rain again.",
    "The wind never stops on this side of the island.",
    "I finally finished the report at midnight.",
    "She felt lighter after the difficult conversation.",
    "He couldn't remember why he had opened the fridge.",
    "We laughed until our sides hurt.",
    "Honestly, I just need a quiet weekend.",
    "Why do leaves change color in autumn?",
    "What time does the last train leave tonight?",
    "Who left the window open during the storm?",
    "How far is the nearest pharmacy from here?",
    "Could you explain that one more time, slowly?",
    "Do you want the long version or the short one?",
    "Preheat the oven before you mix the batter.",
    "Save your work before installing the update.",
    "Take the second left after the old mill.",
    "Water the plants twice a week in summer.",
    "Never store the batteries next to the magnets.",
    "She closed the shutters as the storm rolled in.",
    "The old dog followed the mail carrier to the corner and back.",
    "By the time the kettle whistled, the letter was already burning.",
    "The train left the station three minutes early, which never happens.",
    "Nobody in the village could explain the missing bell.",
    "Seventeen is a prime number, but eighteen is not.",
    "Half of ninety is forty-five.",
    "A dozen eggs plus three more makes fifteen.",
    "The bridge is exactly four hundred meters long.",
    "That was unexpected.",
    "Absolutely not.",
    "Fine, you win.",
    "Try again tomorrow.",
    "It works now.",
    "Almost there.",
    "What a mess.",
    "After weeks of negotiation, the two companies finally agreed on a price, though neither side seemed particularly happy about it.",
    "If the ferry is late again, we will miss the connecting bus, and the next one does not leave until Thursday morning.",
    "The museum's newest exhibit, a reconstructed merchant ship from the twelfth century, drew larger crowds than anyone had predicted.",
    "Turn the volume down before the neighbors complain.",
    "The garden looked completely different after the first snow.",
    "Every map of the old city gets at least one alley wrong.",
    "Nobody expected the quiet intern to solve it first.",
    # prefix-style (incomplete) literals -- the keys' genre, never the fact templates or subjects
    "The capital of Sweden is",
    "The largest planet in our solar system is",
    "The author of the famous play was",
    "The official language of Brazil is",
    "The main ingredient in guacamole is",
    "The fastest land animal is",
    "The chemical symbol for gold is",
    "The first person to walk on the moon was",
    "The national dish of Portugal is",
    "The longest river in South America is",
    "The tallest building in the city is",
    "The name of the new ferry is the",
    "The winner of last year's tournament was",
    "The password for the guest network is",
    "The manager of the night shift is called",
    "The color of the front door is",
    "The number of moons around Mars is",
    "The busiest month for the shop is",
    "The oldest tree in the park is the",
    "The captain of the fishing crew is",
]


def build_sentences() -> list[str]:
    out = set(_LITERALS)
    for c, cap in [("Sweden", "Stockholm"), ("Peru", "Lima"), ("Japan", "Tokyo"), ("Kenya", "Nairobi"),
                   ("Norway", "Oslo"), ("Egypt", "Cairo"), ("Chile", "Santiago"), ("Canada", "Ottawa"),
                   ("Poland", "Warsaw"), ("Vietnam", "Hanoi")]:
        out.add(f"The capital of {c} is {cap}.")
    for c in ["Italy", "Japan", "Brazil", "Morocco", "Iceland", "Texas", "Bavaria", "Kyoto", "Vermont", "Provence"]:
        for x in ["coffee", "festivals", "architecture", "beaches"]:
            out.add(f"{c} is famous for its {x}.")
    for a in ["fix a flat tire", "bake bread", "learn a language", "fall asleep", "stay focused", "apologize"]:
        for b in ["the right tools", "an oven", "a teacher", "any noise", "your phone", "making it worse"]:
            out.add(f"How do you {a} without {b}?")
    trip = [("morning", "reading", "river"), ("afternoon", "sketching", "harbor"), ("evening", "fishing", "window")]
    for n in ["Maria", "Kenji", "Priya", "Tom", "Aisha", "Lars", "Sofia", "Omar"]:
        for d, act, p in trip:
            out.add(f"{n} spent the {d} {act} by the {p}.")
    manners = ["gently", "briskly", "twice"]
    for i, q in enumerate(["two cups", "a pinch", "three spoonfuls", "half a liter", "a handful"]):
        for j, ing in enumerate(["flour", "chopped mint", "brown sugar", "coconut milk", "toasted almonds"]):
            out.add(f"The recipe calls for {q} of {ing}, stirred {manners[(i + j) % 3]}.")
    months = ["April", "October", "January"]
    for i, city in enumerate(["Lisbon", "Denver", "Osaka", "Perth", "Tallinn", "Quito", "Glasgow", "Austin"]):
        for j, w in enumerate(["cold", "humid", "windy", "quiet"]):
            out.add(f"{city} gets surprisingly {w} in {months[(i + j) % 3]}.")
    spots = ["balcony", "dune", "rafters", "pier", "doorway", "oak tree"]
    for i, a1 in enumerate(["cat", "heron", "farmer", "lifeguard", "spider", "toddler", "security camera"]):
        for j, a2 in enumerate(["sparrows", "ferry", "children", "waves", "moth", "delivery truck"]):
            out.add(f"The {a1} watched the {a2} from the {spots[(i + j) % 6]}.")
    feels = ["hopeful", "uneasy", "thrilled", "indifferent", "exhausted"]
    topics = ["decision", "schedule", "results", "move", "budget"]
    for i, ev in enumerate(["meeting", "rehearsal", "interview", "storm", "auction", "ceremony"]):
        for j, n in enumerate(["Elena", "Marcus", "Yuki", "the whole team", "Grandpa"]):
            out.add(f"After the {ev}, {n} felt {feels[(i + j) % 5]} about the {topics[(i * 2 + j) % 5]}.")
    achs = ["reached the summit", "approved the tunnel", "toured abroad", "won the dispute", "photographed the comet"]
    for i, y in enumerate(["1969", "1987", "2003", "2014", "1848", "1926"]):
        for j, g in enumerate(["expedition", "city council", "orchestra", "factory workers", "observatory"]):
            out.add(f"In {y}, the {g} finally {achs[(i + j) % 5]}.")
    for r in ["grandmother", "coach", "dentist", "neighbor", "first boss"]:
        for c in ["patience is a muscle", "cheap shoes cost more in the end",
                  "the garden knows the season better than the calendar",
                  "you should never argue with the tide", "breakfast decides the day"]:
            out.add(f"My {r} always says that {c}.")
    for a, b, c in [("four", "five", "nine"), ("ten", "six", "sixteen"), ("twenty", "seven", "twenty-seven"),
                    ("eight", "eight", "sixteen"), ("thirty", "nine", "thirty-nine"), ("two", "eleven", "thirteen"),
                    ("fifty", "five", "fifty-five"), ("twelve", "twelve", "twenty-four")]:
        out.add(f"{a.capitalize()} plus {b} equals {c}.")
    times = ["two", "three", "five", "nine"]
    for i, (x, y) in enumerate([("Vienna", "Prague"), ("Lima", "Cusco"), ("Cairo", "Alexandria"),
                                ("Osaka", "Tokyo"), ("Porto", "Lisbon"), ("Denver", "Santa Fe"),
                                ("Helsinki", "Tallinn"), ("Marrakesh", "Fez")]):
        for j, mode in enumerate(["train", "bus", "ferry", "flight"]):
            out.add(f"The {mode} from {x} to {y} takes about {times[(i + j) % 4]} hours.")
    for f in ["octopuses have three hearts", "honey never spoils", "bananas are berries",
              "lightning can strike twice", "cats always land on their feet", "the tomato is a fruit",
              "goldfish remember for months", "bees dance to give directions"]:
        out.add(f"Is it true that {f}?")
    for i, attr in enumerate(["favorite color", "market day", "oldest bridge", "busiest street",
                              "town motto", "festival season"]):
        for place in ["Marseille", "the old quarter", "Reykjavik", "the university", "Cape Town",
                      "the north district"]:
            out.add(f"The {attr} of {place} is")
    whens = ["a week early", "right on time", "after two delays"]
    for i, team in enumerate(["design team", "kitchen", "night crew", "lab", "volunteers"]):
        for j, thing in enumerate(["new menu", "prototype", "newsletter", "harvest", "exhibit"]):
            out.add(f"The {team} shipped the {thing} {whens[(i + j) % 3]}.")
    for sm in ["cinnamon", "fresh paint", "rain on dust", "burnt toast", "pine needles", "sea salt"]:
        out.add(f"The whole house smelled like {sm} by noon.")
    for n, adj in [("sequel", "overrated"), ("proposal", "unfinished"), ("forecast", "brilliant"),
                   ("coffee", "too salty"), ("speech", "too long"), ("layout", "overdue")]:
        out.add(f"Frankly, the {n} was {adj}, and everyone knew it.")
    states = ["crowded", "empty", "flooded", "silent"]
    for i, t in enumerate(["dawn", "noon", "dusk", "midnight"]):
        for j, pl in enumerate(["market", "station", "beach", "office"]):
            out.add(f"By {t}, the {pl} was already {states[(i + j) % 4]}.")
    for obj in ["spare keys", "charger", "measuring tape", "umbrella", "receipts", "good scissors",
                "flashlight", "stamps"]:
        out.add(f"Where did you put the {obj}?")
    for n2 in ["Ravi", "Anna", "my brother", "the twins", "Coach Ellis"]:
        for food in ["durian", "cold brew", "cross-country skiing", "karaoke", "oysters"]:
            out.add(f"{n2} has never tried {food}.")
    lst = sorted(out)
    import re
    for s in lst:                                   # leakage assert: no nonce subject in the fit corpus
        low = s.lower()                             # word-boundary match ("umbrella" must not trip "brell")
        assert not any(re.search(r"\b" + re.escape(nz) + r"\b", low) for nz in NONCE), \
            f"nonce leakage in fit corpus: {s!r}"
    rng = random.Random(0)
    rng.shuffle(lst)
    return lst


# ---------------------------------------------------------------------------------------------------
# Pure bridge math (CPU, float64). Mappers are dicts: {name, map(X)->Y, meta}.
# ---------------------------------------------------------------------------------------------------
def _unit(t: torch.Tensor) -> torch.Tensor:
    return t / (t.norm(dim=-1, keepdim=True) + 1e-8)


def fit_ridge(X: torch.Tensor, Y: torch.Tensor, lam: float):
    """Centered ridge with exact bias: map(x) = (x - muX) @ W + muY. X:[N,HA] Y:[N,HB], unit rows."""
    X64, Y64 = X.double(), Y.double()
    muX, muY = X64.mean(0), Y64.mean(0)
    Xc, Yc = X64 - muX, Y64 - muY
    H = Xc.T @ Xc + lam * torch.eye(X.shape[1], dtype=torch.float64)
    W = torch.linalg.solve(H, Xc.T @ Yc)
    return {"W": W, "muX": muX, "muY": muY}


def apply_affine(m: dict, X: torch.Tensor) -> torch.Tensor:
    return (X.double() - m["muX"]) @ m["W"] + m["muY"]


def cos_rows(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return (_unit(A.double()) * _unit(B.double())).sum(-1)


def heldout_metrics(mapf, Xh: torch.Tensor, Yh: torch.Tensor, muY: torch.Tensor) -> dict:
    """Raw + CENTERED held-out cosine after mapping. Centered subtracts the B-side fit mean from both
    mapped and true vectors (mirrors the store's centered addressing; kills the anisotropy freebie)."""
    P = mapf(Xh)
    raw = cos_rows(P, Yh)
    cen = cos_rows(P.double() - muY, Yh.double() - muY)
    return {"raw_mean": round(float(raw.mean()), 4), "raw_min": round(float(raw.min()), 4),
            "centered_mean": round(float(cen.mean()), 4), "centered_min": round(float(cen.min()), 4)}


def choose_lambda(Xf: torch.Tensor, Yf: torch.Tensor, lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)) -> float:
    nv = max(16, len(Xf) // 5)
    Xtr, Ytr, Xv, Yv = Xf[:-nv], Yf[:-nv], Xf[-nv:], Yf[-nv:]
    best, best_c = lams[0], -2.0
    for lam in lams:
        m = fit_ridge(Xtr, Ytr, lam)
        c = float(cos_rows(apply_affine(m, Xv) - m["muY"], Yv.double() - m["muY"]).mean())
        if c > best_c:
            best, best_c = lam, c
    return best


def fit_procrustes_pca(X: torch.Tensor, Y: torch.Tensor, d: int):
    """Rotation-only bridge through matched PCA subspaces: PCA both sides to d, orthogonal Procrustes
    (with the optimal global scale) between score matrices, unproject to the target side."""
    X64, Y64 = X.double(), Y.double()
    muX, muY = X64.mean(0), Y64.mean(0)
    Xc, Yc = X64 - muX, Y64 - muY
    VA = torch.linalg.svd(Xc, full_matrices=False).Vh.T[:, :d]     # [HA, d]
    VB = torch.linalg.svd(Yc, full_matrices=False).Vh.T[:, :d]     # [HB, d]
    As, Bs = Xc @ VA, Yc @ VB                                      # [N, d] each
    U, S, Vt = torch.linalg.svd(As.T @ Bs, full_matrices=False)
    R = U @ Vt
    scale = float(S.sum() / (As ** 2).sum())
    def mapf(Xq: torch.Tensor) -> torch.Tensor:
        return (scale * ((Xq.double() - muX) @ VA) @ R) @ VB.T + muY
    return mapf, {"d": d, "scale": round(scale, 4)}, muY


def fit_all_bridges(resid_src: torch.Tensor, resid_tgt: torch.Tensor, primary_n: int, held_n: int,
                    seed_random=123, seed_shuffle=7) -> dict:
    """All mappers src->tgt from raw residual caches. Fit on unit rows; held-out = the LAST held_n
    sentences (never touched by any fit). Returns {name: {map, heldout, meta, muY}}."""
    Xu, Yu = _unit(resid_src.double()), _unit(resid_tgt.double())
    Xf_all, Yf_all = Xu[:-held_n], Yu[:-held_n]
    Xh, Yh = Xu[-held_n:], Yu[-held_n:]
    out = {}

    def add_ridge(name, n):
        Xf, Yf = Xf_all[:n], Yf_all[:n]
        lam = choose_lambda(Xf, Yf)
        m = fit_ridge(Xf, Yf, lam)
        f = lambda Xq, m=m: apply_affine(m, Xq)
        out[name] = {"map": f, "muY": m["muY"], "meta": {"n_fit": n, "lam": lam},
                     "heldout": heldout_metrics(f, Xh, Yh, m["muY"]), "_m": m}
        return m, lam

    n_pool = len(Xf_all)
    primary = min(primary_n, n_pool)
    m256, lam256 = add_ridge("ridge", primary)
    if n_pool >= 128 and primary != 128:
        add_ridge("ridge128", 128)
    if n_pool >= 512 and primary != 512:
        add_ridge("ridge512", 512)

    d = max(16, min(192, primary - 64))
    pf, pmeta, pmu = fit_procrustes_pca(Xf_all[:primary], Yf_all[:primary], d)
    out["procrustes"] = {"map": pf, "muY": pmu, "meta": pmeta,
                         "heldout": heldout_metrics(pf, Xh, Yh, pmu)}

    g = torch.Generator().manual_seed(seed_random)                  # random-matrix null, norm-matched
    Wr = torch.randn(resid_src.shape[1], resid_tgt.shape[1], generator=g, dtype=torch.float64)
    Wr = Wr * (m256["W"].norm() / Wr.norm())
    mr = {"W": Wr, "muX": m256["muX"], "muY": m256["muY"]}
    fr = lambda Xq, mr=mr: apply_affine(mr, Xq)
    out["random"] = {"map": fr, "muY": mr["muY"], "meta": {"seed": seed_random},
                     "heldout": heldout_metrics(fr, Xh, Yh, mr["muY"])}

    gp = torch.Generator().manual_seed(seed_shuffle)                # shuffled-PAIR fit null
    perm = torch.randperm(primary, generator=gp)
    ms = fit_ridge(Xf_all[:primary], Yf_all[:primary][perm], lam256)
    fs = lambda Xq, ms=ms: apply_affine(ms, Xq)
    out["shuffled_fit"] = {"map": fs, "muY": ms["muY"], "meta": {"n_fit": primary, "lam": lam256,
                           "fixed_points": int((perm == torch.arange(primary)).sum())},
                           "heldout": heldout_metrics(fs, Xh, Yh, ms["muY"])}
    return out


def centered_top1(K: torch.Tensor, q: torch.Tensor) -> tuple[int, float]:
    """Pure mirror of SlotMem's centered addressing (for the CPU selftest)."""
    mu = K.mean(0)
    Kc = _unit(K - mu)
    qc = _unit(_unit(q) - mu)
    sims = Kc @ qc
    return int(sims.argmax()), float(sims.max())


# ---------------------------------------------------------------------------------------------------
# GPU citizenship
# ---------------------------------------------------------------------------------------------------
def gpu_used_gb() -> float:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    return float(r.stdout.strip().splitlines()[0]) / 1024.0


def wait_for_gpu(limit_gb=3.0, sustain_s=120, poll_s=10, max_wait_s=6 * 3600):
    """Proceed only when total GPU usage stays under limit_gb for sustain_s. Never touches anyone."""
    if not torch.cuda.is_available():
        return
    t0, quiet = time.time(), 0.0
    while True:
        try:
            used = gpu_used_gb()
        except Exception as e:                                      # nvidia-smi hiccup: report, retry
            print(f"[gpu] nvidia-smi failed ({e}); retrying", flush=True)
            time.sleep(poll_s)
            continue
        if used < limit_gb:
            quiet += poll_s
            if quiet >= sustain_s:
                print(f"[gpu] clear ({used:.1f} GB used, quiet {quiet:.0f}s) -- proceeding", flush=True)
                return
        else:
            quiet = 0.0
        if time.time() - t0 > max_wait_s:
            raise TimeoutError(f"GPU never freed below {limit_gb} GB in {max_wait_s}s (last {used:.1f} GB)")
        print(f"[gpu] waiting: {used:.1f} GB used (need <{limit_gb} for {sustain_s}s; quiet {quiet:.0f}s)",
              flush=True)
        time.sleep(poll_s)


# ---------------------------------------------------------------------------------------------------
# Store building + evaluation on a live model (SlotMem carries the hooks/eta/read mechanics)
# ---------------------------------------------------------------------------------------------------
from slotmem_qwen import SlotMem, SINGLE, MULTI, PARA, pack_store, unpack_store  # noqa: E402


def build_bank(tok, n_single: int, n_multi: int) -> list[dict]:
    bank = []
    for cue, ans in SINGLE[:n_single] + MULTI[:n_multi]:
        ids = tok.encode(ans, add_special_tokens=False)
        bank.append({"cue": cue, "answer": ans, "ans_ids": ids})
    return bank


def para_pairs(bank: list[dict]) -> list[tuple[str, int]]:
    cues = {f["cue"]: i for i, f in enumerate(bank)}
    return [(p, cues[c]) for c, plist in PARA.items() if c in cues for p in plist]


def collect_resids(mem: SlotMem, texts: list[str]) -> torch.Tensor:
    out = []
    for i, s in enumerate(texts):
        out.append(mem._resid_last(s).float().cpu())
        if (i + 1) % 64 == 0:
            print(f"  [resid] {i + 1}/{len(texts)}", flush=True)
    return torch.stack(out)


def entries_from_keys(keys_unit: torch.Tensor, bank: list[dict], mem: SlotMem) -> list[dict]:
    """Target-side entries from ported keys: values are the TARGET model's own unembedding rows for
    the shared-vocab ans_ids (the no-bridge-needed half, stated in the scoping note)."""
    dev = keys_unit.device if keys_unit.is_cuda else ("cuda" if torch.cuda.is_available() else "cpu")
    es = []
    for i, f in enumerate(bank):
        v = mem.W_U[f["ans_ids"][0]].float()
        es.append({"key": keys_unit[i].float().to(dev), "value": v / (v.norm() + 1e-8),
                   "label": f["cue"] + " ->" + f["answer"], "ans_ids": f["ans_ids"],
                   "cue": f["cue"], "answer": f["answer"]})
    return es


def eval_arm(mem: SlotMem, entries: list[dict] | None, bank: list[dict], paras: list[tuple[str, int]],
             name: str) -> dict:
    res = {"arm": name,
           "exact": {"n": len(bank), "select": 0, "express": 0, "p_ans": 0.0},
           "para": {"n": len(paras), "select": 0, "express": 0}, "items": []}
    for i, f in enumerate(bank):
        r = mem.read(f["cue"], entries=entries if entries is not None else [])
        top = int(r["dist"].argmax())
        sel = (r["hit"] == i)
        res["exact"]["select"] += int(sel)
        res["exact"]["express"] += int(top == f["ans_ids"][0])
        res["exact"]["p_ans"] += float(r["dist"][f["ans_ids"][0]])
        res["items"].append({"kind": "exact", "cue": f["cue"], "want": f["answer"].strip(),
                             "hit": r["hit"], "sel": bool(sel), "sim": None if r["sim"] is None else round(r["sim"], 3),
                             "top_tok": mem.tok.decode([top])})
    for p, i in paras:
        f = bank[i]
        r = mem.read(p, entries=entries if entries is not None else [])
        top = int(r["dist"].argmax())
        res["para"]["select"] += int(r["hit"] == i)
        res["para"]["express"] += int(top == f["ans_ids"][0])
        res["items"].append({"kind": "para", "cue": p, "want": f["answer"].strip(), "hit": r["hit"],
                             "sel": bool(r["hit"] == i), "sim": None if r["sim"] is None else round(r["sim"], 3),
                             "top_tok": mem.tok.decode([top])})
    for k in ("exact", "para"):
        n = max(1, res[k]["n"])
        res[k]["select"] = round(res[k]["select"] / n, 3)
        res[k]["express"] = round(res[k]["express"] / n, 3)
    res["exact"]["p_ans"] = round(res["exact"]["p_ans"] / max(1, len(bank)), 4)
    return res


def free_model(mem: SlotMem):
    mem.close()
    del mem.model
    del mem
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def cue_diag(mapf, keys_src: torch.Tensor, keys_tgt_native: torch.Tensor) -> dict:
    """Does the bridge land each cue's key near the TARGET model's own key for that cue? Centered by
    the native pool mean (the quantity top-1 addressing actually sees)."""
    P = _unit(mapf(keys_src.double()))
    Kn = keys_tgt_native.double()
    mu = Kn.mean(0)
    cen = cos_rows(P - mu, Kn - mu)
    diag_raw = cos_rows(P, Kn)
    Pc = _unit(P - P.mean(0))
    cross = Pc @ Pc.T
    off = cross[~torch.eye(len(Pc), dtype=torch.bool)]
    return {"cue_cos_centered_mean": round(float(cen.mean()), 4),
            "cue_cos_centered_min": round(float(cen.min()), 4),
            "cue_cos_raw_mean": round(float(diag_raw.mean()), 4),
            "ported_crosssim_centered_mean": round(float(off.mean()), 4)}


# ---------------------------------------------------------------------------------------------------
# Stages (sequential model loads; each stage cached; per-arm JSON checkpoints)
# ---------------------------------------------------------------------------------------------------
def _save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


def stage_A(cfg) -> dict:
    p = os.path.join(cfg["dir"], "stage_a.pt")
    if os.path.isfile(p):
        print(f"[stage A] cached -> {p}", flush=True)
        return torch.load(p, weights_only=False)
    wait_for_gpu()
    t0 = time.time()
    mem = SlotMem(MODEL_A, LAYER)
    bank = build_bank(mem.tok, cfg["n_single"], cfg["n_multi"])
    canary = [mem.tok.encode(s, add_special_tokens=False) for s in CANARY]
    for f in bank:
        mem.write(f["cue"], f["answer"], gate=False)
    assert len(mem.entries) == len(bank)
    ceiling = eval_arm(mem, mem.entries, bank, para_pairs(bank), "ceiling_text_recompile_on_A")
    print(f"[stage A] ceiling on A: exact select={ceiling['exact']['select']} "
          f"express={ceiling['exact']['express']} para={ceiling['para']}", flush=True)
    resid = collect_resids(mem, cfg["sentences"])
    out = {"resid": resid, "store": pack_store(mem.entries, LAYER, mem.eta, None), "eta": mem.eta,
           "bank": bank, "canary": canary, "ceiling": ceiling, "seconds": round(time.time() - t0, 1)}
    torch.save(out, p)
    _save_json(os.path.join(cfg["dir"], "stage_a.json"),
               {k: out[k] for k in ("bank", "ceiling", "eta", "seconds")})
    free_model(mem)
    print(f"[stage A] done in {out['seconds']}s -> {p}", flush=True)
    return out


def run_port_arms(mem: SlotMem, bridges: dict, keys_src: torch.Tensor, keys_tgt_native: torch.Tensor,
                  bank: list[dict], paras, cfg, tag: str, ceiling: dict) -> dict:
    arms = {"no_memory": eval_arm(mem, None, bank, paras, f"{tag}_no_memory"), "ceiling": ceiling}
    _save_json(os.path.join(cfg["dir"], f"arm_{tag}_no_memory.json"), arms["no_memory"])
    _save_json(os.path.join(cfg["dir"], f"arm_{tag}_ceiling.json"), ceiling)
    for name, br in bridges.items():
        ported = _unit(br["map"](keys_src.double())).float()
        es = entries_from_keys(ported, bank, mem)
        r = eval_arm(mem, es, bank, paras, f"{tag}_{name}")
        r["bridge_heldout"] = br["heldout"]
        r["bridge_meta"] = br["meta"]
        r["cue_diag"] = cue_diag(br["map"], keys_src, keys_tgt_native)
        arms[name] = r
        _save_json(os.path.join(cfg["dir"], f"arm_{tag}_{name}.json"), r)
        print(f"[{tag}:{name}] exact sel={r['exact']['select']} expr={r['exact']['express']} "
              f"para sel={r['para']['select']} expr={r['para']['express']} "
              f"heldcos(cen)={br['heldout']['centered_mean']} cue_cos={r['cue_diag']['cue_cos_centered_mean']}",
              flush=True)
    return arms


def stage_B(cfg, sa: dict) -> dict:
    marker = os.path.join(cfg["dir"], "stage_b_done.json")
    pt = os.path.join(cfg["dir"], "stage_b.pt")
    if os.path.isfile(marker) and os.path.isfile(pt):
        print(f"[stage B] cached -> {marker}", flush=True)
        return {**torch.load(pt, weights_only=False), "arms": json.load(open(marker, encoding="utf-8"))}
    wait_for_gpu()
    t0 = time.time()
    mem = SlotMem(MODEL_B, LAYER)
    canary = [mem.tok.encode(s, add_special_tokens=False) for s in CANARY]
    assert canary == sa["canary"], "tokenizer mismatch A vs B -- shared-vocab premise is FALSE"
    bank = build_bank(mem.tok, cfg["n_single"], cfg["n_multi"])
    for f, fa in zip(bank, sa["bank"]):
        assert f["ans_ids"] == fa["ans_ids"], f"ans_ids diverge for {f['cue']!r}"
    assert int(mem.W_U.shape[0]) > max(max(f["ans_ids"]) for f in bank)
    for f in bank:
        mem.write(f["cue"], f["answer"], gate=False)
    paras = para_pairs(bank)
    ceiling = eval_arm(mem, mem.entries, bank, paras, "ab_ceiling_text_recompile_on_B")
    print(f"[stage B] ceiling on B: exact select={ceiling['exact']['select']} "
          f"express={ceiling['exact']['express']} para={ceiling['para']}", flush=True)
    resid_B = collect_resids(mem, cfg["sentences"])
    keys_B_native = torch.stack([e["key"].float().cpu() for e in mem.entries])
    keys_A = sa["store"]["keys"]                                   # unit keys from the A store
    print("[stage B] fitting bridges A->B on CPU ...", flush=True)
    bridges = fit_all_bridges(sa["resid"], resid_B, cfg["primary_fit"], cfg["held_n"])
    for n, b in bridges.items():
        print(f"  [bridge A->B {n}] heldout {b['heldout']} meta={b['meta']}", flush=True)
    arms = run_port_arms(mem, bridges, keys_A, keys_B_native, bank, paras, cfg, "ab", ceiling)
    out_pt = {"resid": resid_B, "keys_native": keys_B_native,
              "store": pack_store(mem.entries, LAYER, mem.eta, None), "eta": mem.eta}
    torch.save(out_pt, pt)
    arms_serializable = {k: {kk: vv for kk, vv in v.items() if kk != "map"} for k, v in arms.items()}
    _save_json(marker, arms_serializable)
    free_model(mem)
    print(f"[stage B] done in {round(time.time() - t0, 1)}s", flush=True)
    return {**out_pt, "arms": arms_serializable}


def stage_C(cfg, sa: dict, sb: dict) -> dict:
    marker = os.path.join(cfg["dir"], "stage_c_done.json")
    if os.path.isfile(marker):
        print(f"[stage C] cached -> {marker}", flush=True)
        return json.load(open(marker, encoding="utf-8"))
    wait_for_gpu()
    t0 = time.time()
    mem = SlotMem(MODEL_A, LAYER)
    bank = build_bank(mem.tok, cfg["n_single"], cfg["n_multi"])
    paras = para_pairs(bank)
    keys_B = sb["store"]["keys"]
    keys_A_native = sa["store"]["keys"]
    print("[stage C] fitting bridges B->A on CPU ...", flush=True)
    bridges = fit_all_bridges(sb["resid"], sa["resid"], cfg["primary_fit"], cfg["held_n"])
    for n, b in bridges.items():
        print(f"  [bridge B->A {n}] heldout {b['heldout']} meta={b['meta']}", flush=True)
    ceiling = dict(sa["ceiling"], arm="ba_ceiling_text_recompile_on_A")
    arms = run_port_arms(mem, bridges, keys_B, keys_A_native, bank, paras, cfg, "ba", ceiling)
    arms_serializable = {k: {kk: vv for kk, vv in v.items() if kk != "map"} for k, v in arms.items()}
    _save_json(marker, arms_serializable)
    free_model(mem)
    print(f"[stage C] done in {round(time.time() - t0, 1)}s", flush=True)
    return arms_serializable


def fmt_table(arms_ab: dict, arms_ba: dict) -> str:
    order = ["no_memory", "ceiling", "ridge", "ridge512", "ridge128", "procrustes", "random", "shuffled_fit"]
    lines = ["| direction | arm | exact select | exact express | para select | para express | heldout cos (cen) | cue cos (cen) |",
             "|---|---|---|---|---|---|---|---|"]
    for tag, arms in (("A->B", arms_ab), ("B->A", arms_ba)):
        for name in order:
            if name not in arms:
                continue
            a = arms[name]
            hc = a.get("bridge_heldout", {}).get("centered_mean", "--")
            cc = a.get("cue_diag", {}).get("cue_cos_centered_mean", "--")
            lines.append(f"| {tag} | {name} | {a['exact']['select']} | {a['exact']['express']} | "
                         f"{a['para']['select']} | {a['para']['express']} | {hc} | {cc} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------
# CPU selftest: validate the math rig on synthetic data with a known planted map. No GPU, no models.
# ---------------------------------------------------------------------------------------------------
def selftest():
    torch.manual_seed(0)
    HA, HB, N, NH = 48, 96, 240, 40
    T = torch.randn(HA, HB, dtype=torch.float64) / HA ** 0.5
    X = _unit(torch.randn(N + NH, HA, dtype=torch.float64))
    Y = _unit(X @ T + 0.05 * torch.randn(N + NH, HB, dtype=torch.float64))
    br = fit_all_bridges(X.float(), Y.float(), primary_n=200, held_n=NH)
    r, s, rn, pr = (br[k]["heldout"]["centered_mean"] for k in ("ridge", "shuffled_fit", "random", "procrustes"))
    print(f"[selftest] heldout centered cos: ridge={r} procrustes={pr} shuffled={s} random={rn}")
    assert r > 0.9, "ridge failed to recover a planted linear map"
    assert pr > 0.5, "procrustes-pca unexpectedly weak on a planted map"
    assert abs(s) < 0.3 and abs(rn) < 0.3, "nulls did not collapse on synthetic data"
    keysA = _unit(torch.randn(12, HA, dtype=torch.float64))
    keysB = _unit(keysA @ T + 0.02 * torch.randn(12, HB, dtype=torch.float64))
    for name, want_hi in (("ridge", True), ("random", False)):
        ported = _unit(br[name]["map"](keysA.float()))
        ok = 0
        for i in range(12):
            q = _unit(keysB[i] + 0.05 * torch.randn(HB, dtype=torch.float64))
            hit, _ = centered_top1(ported, q)
            ok += int(hit == i)
        print(f"[selftest] synthetic port select ({name}): {ok}/12")
        assert (ok >= 10) if want_hi else (ok <= 4), f"{name} select out of expected range"
    print("[selftest] PASS -- bridge math, nulls, and centered top-1 port logic all behave")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore stage caches")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    sents = build_sentences()
    print(f"[corpus] {len(sents)} distinct sentences", flush=True)
    if a.smoke:
        cfg = {"dir": "research/runs/telepathy_smoke", "n_single": 4, "n_multi": 0,
               "held_n": HELD_N_SMOKE, "primary_fit": 64, "sentences": sents[:64 + HELD_N_SMOKE]}
    else:
        need = 512 + HELD_N_FULL
        assert len(sents) >= need, f"corpus too small: {len(sents)} < {need}"
        cfg = {"dir": "research/runs/telepathy", "n_single": 12, "n_multi": 8,
               "held_n": HELD_N_FULL, "primary_fit": 256, "sentences": sents[:need]}
    if a.fresh and os.path.isdir(cfg["dir"]):
        for f in os.listdir(cfg["dir"]):
            os.remove(os.path.join(cfg["dir"], f))
    os.makedirs(cfg["dir"], exist_ok=True)
    t0 = time.time()
    sa = stage_A(cfg)
    sb = stage_B(cfg, sa)
    arms_ba = stage_C(cfg, sa, sb)
    table = fmt_table(sb["arms"], arms_ba)
    final = {"config": {k: v for k, v in cfg.items() if k != "sentences"},
             "n_sentences": len(cfg["sentences"]), "models": {"A": MODEL_A, "B": MODEL_B},
             "layer": LAYER, "arms_ab": sb["arms"], "arms_ba": arms_ba,
             "seconds": round(time.time() - t0, 1)}
    _save_json(os.path.join(cfg["dir"], "telepathy_full.json"), final)
    print("\n" + table, flush=True)
    print(f"\nsaved -> {cfg['dir']}/telepathy_full.json  ({final['seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
