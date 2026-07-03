"""dream_consolidation.py -- DIFFUSION DREAMING: mine durable-memory candidates from Dream-7B "dreams".

EXPERIMENT (pre-registration below). A diffusion LM (Dream-7B) can re-mask a real conversation fragment
and re-denoise it into a *variation* -- a "dream" of what the user might have said. Does dreaming SURFACE
durable preference-cards that plain extraction over the raw fragment misses? Or does it only add noise
(and hallucinate preferences the user never expressed)? This pipeline measures that, with the raw-fragment
extraction as the REQUIRED comparison arm. A null (dreams surface nothing new, or only garbage) is a finding.

PIPELINE (each phase checkpoints to research/dream_runs/<phase>.json so phases are resumable and the slow
GPU DREAM phase can be re-run without redoing the CPU work):
  1. CORPUS  -- fragments from ~/.clozn/runs/*.json. User turns in this corpus are SHORT (median 15 words,
                max 28), so a "fragment" is a WINDOW of 1-4 consecutive turns anchored on a user turn, grown
                to ~40-130 words (~50-160 tok). This keeps real user voice while hitting the target length.
  2. DREAM   -- for each fragment, re-mask at 30/60/90% of its content tokens and denoise K variations via
                the ESTABLISHED cloze_lab scheduler path (the same generate() that denoise_server.trace_for
                drives). Every dream saved verbatim. Denoiser is SWAPPABLE: 'dream' (7B, GPU) or 'dcoder'
                (0.5B Dream-family sibling, CPU -- for building/verifying the pipeline while the GPU is busy).
  3. MINE    -- SWAP MODELS SEQUENTIALLY: free the denoiser, load Qwen2.5-1.5B bf16, reuse SelfTeach's
                propose_memory prompt to extract ONE durable preference-card per dream (and per raw fragment).
                Extractor-dependence is real and noted loudly: a different extractor would yield different
                cards. We hold it FIXED across the dreamed and raw arms so the comparison is fair.
  4. GATES+FUNNEL -- dedupe dreamed candidates against RAW-fragment candidates (semantic, via the same
                MiniLM embedder the studio's topic_gate uses); a candidate is NOVEL if no raw candidate is
                near it. Plausibility/surprise sanity gates (drop refusals/boilerplate/duplicates-of-known).
                Report the funnel: N dreams -> M candidates -> K novel -> J surviving, with samples at each stage.

PRE-REGISTRATION (recorded before any GPU run; my honest predictions):
  * Dream QUALITY prediction: at 30% re-mask, dreams stay near-paraphrases of the fragment (high fidelity,
    low novelty). At 60% they drift into plausible-but-different user turns. At 90% they largely dissolve into
    generic/degenerate text (Dream conditions on almost nothing) -- I expect the WORST dreams here and expect
    90% to contribute mostly garbage candidates the gates must reject. Dream-7B is a capable diffusion model
    so mid-mask (60%) dreams should be coherent; the 0.5B dcoder used for CPU verification will be visibly
    worse (it is only there to prove the plumbing, NOT to judge dream quality).
  * FUNNEL prediction: most candidates survive extraction (propose_memory is conservative -> lots of NONE,
    but what it emits is usually well-formed). NOVELTY is the crux and I predict it is LOW: the durable
    preferences a user reveals are mostly already recoverable from the raw fragment, so dreaming should add
    FEW novel surviving cards. Likely outcome = a WEAK or NULL dreamed-vs-undreamed effect: dreaming
    paraphrases known preferences more than it discovers new ones, and at high mask it invents preferences
    the user never expressed (a hallucination risk the surprise/plausibility gate must catch). If dreaming
    DOES surface novel, plausible cards absent from raw extraction, that is the positive result worth having.
  * Confound noted up front: on this machine the CPU verification uses dcoder-0.5B (weak); the real quality
    verdict requires the Dream-7B GPU pass. Findings will label which denoiser produced every quoted sample.

AMENDMENTS (2026-07-03, after the first Dream-7B smokes, BEFORE the full run; both caught by eyeballing
smoke dreams, per house rules):
  1. BUG FIX: remask_denoise passed logits_for=[j-1 ...] -- but the DreamAdapter applies the family's
     shifted head INTERNALLY, so this double-shifted and filled every hole with a copy of its left
     neighbour ("What should I do do this?", "I I I I I" -- quoted from the broken smoke). Now
     logits_for=masked. The dcoder CPU pipeline-check had the same bug; its funnel shape was plumbing
     verification only and its dream texts are void.
  2. MECHANISM AMENDMENT: greedy argmax infill turned out to RECONSTRUCT the original near-exactly at
     every mask level (30/60/90%) -- K seed "variations" were K identical copies, making dreamed-vs-raw
     a null by construction. Hole fills are now SAMPLED at temperature 0.8 with the dream's seeded rng.
     Scoring, gates, dedupe and the raw comparison arm are untouched. The reconstruction observation
     itself is reported in the findings (it is a real property of confidence-greedy diffusion infill).

HOUSE RULES: stdlib + torch + the cloze_lab scheduler; no new deps (MiniLM/sentence-transformers already a
studio dep). Defensive throughout -- a bad fragment/dream/extraction is skipped, never fatal. Windows/PS5.1.

    # build + CPU-verify the whole pipeline (GPU-free) on the 0.5B Dream-family sibling:
    PYTHONPATH=engine/lab python research/dream_consolidation.py --denoiser dcoder --device cpu --smoke
    PYTHONPATH=engine/lab python research/dream_consolidation.py --denoiser dcoder --device cpu   # full CPU
    # the real quality pass, ONLY once the GPU gate opens (nvidia-smi < 3GB sustained):
    PYTHONPATH=engine/lab python research/dream_consolidation.py --denoiser dream --device cuda --quant nf4
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # WinError 1314 workaround on this PC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine", "lab"))

RUNS_DIR = os.path.join(os.path.expanduser("~"), ".clozn", "runs")
OUT_DIR = os.path.join(HERE, "dream_runs")           # phase checkpoints land here
MASK_FRACS = (0.30, 0.60, 0.90)                      # re-mask levels (pre-registered)


# =====================================================================================================
# PHASE 1 -- CORPUS: conversation fragments (windows anchored on a user turn), from the run log
# =====================================================================================================
def _words(s: str) -> int:
    return len(re.findall(r"\S+", s or ""))


def build_corpus(min_words: int = 6, max_ctx_chars: int = 400, limit: int | None = None) -> list[dict]:
    """Fragments = DISTINCT user turns (>= min_words), each with its immediately-preceding turn folded in as
    CONTEXT (not re-masked) so the dream has something to condition on. The user turn is the RE-MASK TARGET.

    Why user-turn-centric rather than fixed ~50-120-tok windows (the handoff's guideline): this run log's
    user turns are SHORT (median 10 words, max 28) and its conversations repeat -- forcing 40-130-word windows
    collapsed the corpus to 5 near-identical cover-letter fragments (verified). Topical DIVERSITY is what makes
    the dreamed-vs-undreamed comparison meaningful, and there are ~58 distinct, varied user utterances here
    (companionship, pets, routines, cover letters, guitar, cooking, "short/direct/no-emojis"). So a fragment =
    one distinct user turn; dedup is on the normalized turn text. The target is deliberately below the token
    band -- documented honestly, and the re-mask still exercises the exact same denoise mechanics."""
    frags: list[dict] = []
    seen: set[str] = set()
    for path in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json"))):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        msgs = [m for m in (d.get("messages") or []) if (m.get("content") or "").strip()]
        for i, m in enumerate(msgs):
            if m.get("role") != "user" or _words(m["content"]) < min_words:
                continue
            key = re.sub(r"\s+", " ", m["content"]).strip().lower()[:80]
            if key in seen:                                          # dedup near-identical user turns
                continue
            seen.add(key)
            ctx = msgs[i - 1] if i > 0 else None                    # one turn of context (assistant usually)
            ctx_text = (ctx["content"] or "")[:max_ctx_chars] if ctx else ""
            acc = ([{"role": ctx["role"], "content": ctx_text}] if ctx else []) + [m]
            text = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in acc)
            frags.append({"id": f"frag_{len(frags):03d}", "source_run": os.path.basename(path),
                          "turn": i, "n_turns": len(acc), "words": _words(m["content"]),
                          "messages": acc, "text": text,
                          "user_text": m["content"]})       # the user turn is the re-mask target
            if limit and len(frags) >= limit:
                return frags
    return frags


# =====================================================================================================
# PHASE 2 -- DREAM: re-mask a fragment's user text and re-denoise K variations via the scheduler
# =====================================================================================================
class Denoiser:
    """Thin wrapper over a cloze_lab ModelAdapter that (a) exposes the tokenizer for re-masking and (b)
    runs the ESTABLISHED generate() scheduler to denoise a board of [context + partially-masked target].

    Two mask regimes are supported by construction:
      * FREE denoise (mask everything, prompt = context)     -> a fresh continuation given the context.
      * PARTIAL re-mask (keep some target tokens, mask rest)  -> a variation ANCHORED on the original -- the
        true "dream" of the user turn. We build the board directly (context ids + mixed target/mask ids) and
        drive generate() so the scheduler's confidence-unmasking fills exactly the masked holes.
    """

    def __init__(self, name: str, device: str, quant: str):
        from cloze_lab.cli import build_adapter
        from cloze_lab.generate import GenerateConfig, generate
        from cloze_lab.scheduler.events import GenStarted, TokensCommitted, TokensRevised
        self.name = name
        self.ad = build_adapter(name, device=device, quant=quant)
        self.cfg_cls = GenerateConfig
        self.generate = generate
        self._ev = (GenStarted, TokensCommitted, TokensRevised)
        self.mask_id = self.ad.config.mask_token_id
        self.eos = self.ad.config.eos_token_id
        self.tok = self.ad._tok                                        # DreamAdapter exposes the HF tokenizer

    def encode(self, text: str, *, chat: bool) -> list[int]:
        return [int(x) for x in self.ad.encode(text, chat=chat)]

    def decode(self, ids) -> str:
        return self.ad.decode([int(x) for x in ids])

    def free_denoise(self, prompt: str, max_new: int, steps: int, seed: int) -> str:
        """Denoise from scratch (all target masked) given `prompt` as chat context -- trace_for's path."""
        ids = self.encode(prompt, chat=True)
        cfg = self.cfg_cls(max_new=max_new, steps=steps, temperature=0.0, seed=seed, block_len=0)
        res = self.generate(self.ad, ids, cfg)
        board = [int(x) for x in res.board]
        return self._clean(board[len(ids):])

    def remask_denoise(self, context_ids: list[int], target_ids: list[int], mask_frac: float,
                       steps: int, seed: int, temperature: float = 0.8) -> tuple[str, int]:
        """The DREAM: keep (1-mask_frac) of target_ids in place, mask the rest, and let the scheduler
        re-predict the holes conditioned on context + the surviving anchor tokens. Returns (dream_text,
        n_masked). Which positions are masked is seeded so 30/60/90 are comparable across a fragment.

        AMENDMENT (2026-07-03, post-smoke, pre-full-run): hole FILLS are sampled at `temperature` with
        the dream's own seeded rng (temperature=0 -> the original greedy argmax). The first working smoke
        showed greedy fill = near-EXACT RECONSTRUCTION of the original at every mask level (Dream-7B's
        argmax infill is that good) -- so "K variations" were K copies and the dreamed-vs-raw comparison
        would be a null BY CONSTRUCTION. Sampling is what makes a re-mask a DREAM instead of a memory
        test. Scoring/gates/comparisons untouched by this amendment.

        We call generate() with prompt = context_ids and then OVERWRITE the fresh board's target region with
        our partially-kept target before stepping? -- no: generate() owns the board. Instead we drive the
        scheduler indirectly by giving it the FULL board as the prompt is not possible either. So we run the
        adapter/scheduler in the same spirit but build the masked board ourselves and denoise it with a
        minimal confidence loop that mirrors the scheduler's policy (RemaskLowConf is not needed for a
        single-pass fill). This uses ONLY adapter.forward -- the same forward generate() calls."""
        import numpy as np
        rng = np.random.default_rng(seed)
        tgt = list(target_ids)
        n = len(tgt)
        if n == 0:
            return "", 0
        k = max(1, int(round(mask_frac * n)))
        holes = set(int(x) for x in rng.choice(n, size=min(k, n), replace=False))
        board = list(context_ids) + [self.mask_id if t in holes else tgt[t] for t in range(n)]
        lp = len(context_ids)
        n_masked = sum(1 for t in range(n) if t in holes)
        # confidence denoise over exactly the masked holes (fixed-length board). NOTE: the DreamAdapter
        # applies the family's shifted head INTERNALLY (forward maps requested position p to raw row p-1;
        # see cloze_lab/models/dream.py forward()), so we ask for the hole positions THEMSELVES. Asking
        # for j-1 (as this rig did pre-fix) double-applies the shift and fills every hole with a copy of
        # its left neighbour ("What should I do do this?", "I I I I I") -- caught by eyeballing the smoke.
        steps = max(steps, 1)
        for step in range(steps):
            masked = [j for j in range(lp, lp + n) if board[j] == self.mask_id]
            if not masked:
                break
            fr = self.ad.forward(np.asarray(board), _full_attn(len(board)), logits_for=masked)
            logits = np.asarray(fr.logits)                            # rows aligned 1:1 with `masked`
            # softmax -> pick highest-confidence hole(s) this step (K/step schedule like DreamMemory.denoise)
            probs = _softmax_rows(logits)                             # untempered: drives the CONF ordering
            per = max(1, len(masked) // max(1, steps - step))
            order = sorted(range(len(masked)), key=lambda r: -float(probs[r].max()))
            if temperature and temperature > 0:
                probs_t = _softmax_rows(logits / float(temperature))  # tempered: drives the FILL sample
            for r in order[:per]:
                j = masked[r]
                if temperature and temperature > 0:
                    tid = int(rng.choice(probs_t.shape[-1], p=probs_t[r]))
                else:
                    tid = int(probs[r].argmax())
                board[j] = tid
        return self._clean(board[lp:lp + n]), n_masked

    def _clean(self, ids: list[int]) -> str:
        out = []
        for t in (int(x) for x in ids):
            if self.eos is not None and t == self.eos:
                break
            if t != self.mask_id:
                out.append(t)
        return self.decode(out).strip()

    def free(self):
        """Release GPU/CPU memory before the extractor loads (sequential model swap)."""
        try:
            import torch
            del self.ad
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _full_attn(n: int):
    import numpy as np
    return np.ones((n, n), dtype=bool)                                # Dream is non-causal: full attention


def _softmax_rows(logits):
    import numpy as np
    x = np.asarray(logits, dtype=np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def dream_fragments(den: Denoiser, frags: list[dict], k_per: int, steps: int) -> list[dict]:
    """For every fragment x every mask level x K seeds: one dream. Also one FREE denoise per fragment
    (mask everything) as the extreme. Returns a flat list of dream records (fragment id, mask, seed, text)."""
    dreams: list[dict] = []
    for fr in frags:
        ctx_ids = den.encode(fr["text"], chat=True)                  # full fragment (context+voice) as chat
        tgt_ids = den.tok.encode(fr["user_text"], add_special_tokens=False)[:64]
        for mf in MASK_FRACS:
            for seed in range(k_per):
                try:
                    txt, nmask = den.remask_denoise(ctx_ids, tgt_ids, mf, steps=steps, seed=seed)
                except Exception as e:
                    txt, nmask = f"<denoise-error: {type(e).__name__}: {e}>", 0
                dreams.append({"frag": fr["id"], "source_run": fr["source_run"], "mask": mf, "seed": seed,
                               "n_masked": nmask, "n_target": len(tgt_ids), "text": txt,
                               "orig_user_text": fr["user_text"]})
        # one free denoise (full re-mask, context only) -- the 100% extreme
        try:
            free_txt = den.free_denoise(fr["text"], max_new=len(tgt_ids) or 40, steps=steps, seed=0)
        except Exception as e:
            free_txt = f"<free-denoise-error: {type(e).__name__}: {e}>"
        dreams.append({"frag": fr["id"], "source_run": fr["source_run"], "mask": 1.0, "seed": 0,
                       "n_masked": len(tgt_ids), "n_target": len(tgt_ids), "text": free_txt,
                       "orig_user_text": fr["user_text"]})
    return dreams


# =====================================================================================================
# PHASE 3 -- MINE: swap in Qwen2.5-1.5B, reuse propose_memory to extract one card per (dream|raw fragment)
# =====================================================================================================
class Extractor:
    """Qwen2.5-1.5B bf16 running SelfTeach.propose_memory verbatim (the studio's real card-extraction
    prompt). Loaded AFTER the denoiser is freed (sequential swap). The SAME extractor scores both the
    dreamed arm and the raw-fragment arm, so the dreamed-vs-undreamed comparison is extractor-fair."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-1.5B-Instruct", device: str = "cuda"):
        import torch
        import self_teach_server as sts                                # reuse propose_memory + _clean_proposal
        # SelfTeach hard-binds every tensor to the module-global DEV (cuda-if-available). To keep the CPU
        # verification GPU-free we pin that global to the requested device BEFORE constructing SelfTeach.
        sts.DEV = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
        # bf16 (not 4-bit) per the pre-registration: 1.5B bf16 is ~3GB, fits easily once the denoiser is
        # freed, and we only do inference (propose_memory is @torch.no_grad). CPU also uses bf16.
        self.st = sts.SelfTeach(model_name, m=16, four_bit=False)
        self.st.model.eval()
        self.device = sts.DEV

    def card_from_text(self, user_text: str) -> str | None:
        """Wrap a single user utterance as a one-turn conversation and run propose_memory on it -- exactly
        what the studio does for a captured run, so extractor behavior is identical to production."""
        if not (user_text or "").strip():
            return None
        return self.st.propose_memory([{"role": "user", "content": user_text}])


def mine(ext: Extractor, dreams: list[dict], frags: list[dict]) -> tuple[list[dict], list[dict]]:
    """Extract a candidate card from every dream (dreamed arm) and from every raw fragment's user text
    (undreamed comparison arm). Returns (dreamed_candidates, raw_candidates); each item keeps its source."""
    dreamed = []
    for dr in dreams:
        card = ext.card_from_text(dr["text"])
        dreamed.append({**{k: dr[k] for k in ("frag", "source_run", "mask", "seed")},
                        "dream_text": dr["text"], "card": card})
    raw = []
    for fr in frags:
        card = ext.card_from_text(fr["user_text"])
        raw.append({"frag": fr["id"], "source_run": fr["source_run"],
                    "user_text": fr["user_text"], "card": card})
    return dreamed, raw


# =====================================================================================================
# PHASE 4 -- GATES + FUNNEL: dedupe dreamed vs raw, plausibility/surprise sanity, count the funnel
# =====================================================================================================
_KNOWN_MEMORIES = [                                                   # what the studio already believes (dedupe target)
    "Prefers concise, direct answers",
    "is enthusiastic about dogs and pets",
    "is enthusiastic about baking and home cooking",
    "Is interested in cooking",
    "Prefers technical answers",
]

_BOILERPLATE = ("prefers helpful", "wants assistance", "needs help", "is asking a question",
                "wants a response", "prefers answers", "is a user")


def _embedder():
    """The same MiniLM sentence-embedder the studio's topic_gate uses; None if unavailable (then dedupe
    falls back to a lexical Jaccard so the pipeline still runs)."""
    try:
        from sentence_transformers import SentenceTransformer
        path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        return None


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(re.findall(r"[a-z]+", a.lower())), set(re.findall(r"[a-z]+", b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


class Dedup:
    """Semantic near-duplicate check. cos >= tau (or Jaccard >= jac) => 'same preference'. Used to (a) drop
    candidates that merely restate an already-known memory and (b) mark a dreamed candidate NOVEL only if no
    raw-fragment candidate is near it."""

    def __init__(self, tau: float = 0.62, jac: float = 0.6):
        self.tau, self.jac = tau, jac
        self.emb = _embedder()

    def _vecs(self, texts: list[str]):
        if self.emb is None or not texts:
            return None
        return self.emb.encode(texts, normalize_embeddings=True)

    def near_any(self, text: str, pool: list[str]) -> tuple[bool, float, str]:
        """Is `text` a near-duplicate of anything in `pool`? -> (hit, best_score, best_match)."""
        if not text or not pool:
            return False, 0.0, ""
        if self.emb is not None:
            import numpy as np
            v = self.emb.encode([text], normalize_embeddings=True)[0]
            P = self.emb.encode(pool, normalize_embeddings=True)
            sims = P @ v
            i = int(np.argmax(sims))
            return (float(sims[i]) >= self.tau, float(sims[i]), pool[i])
        best_i, best = 0, 0.0
        for i, p in enumerate(pool):
            s = _jaccard(text, p)
            if s > best:
                best, best_i = s, i
        return (best >= self.jac, best, pool[best_i])


def plausible(card: str) -> tuple[bool, str]:
    """Surprise/plausibility sanity gate. Reject: empty, boilerplate ('is a user asking for help'), a
    near-restatement of a KNOWN memory is handled separately (dedupe). Returns (keep, reason)."""
    if not card:
        return False, "empty"
    low = card.lower()
    if len(card) < 8:
        return False, "too-short"
    if any(b in low for b in _BOILERPLATE):
        return False, "boilerplate"
    if low.count(" ") < 1:
        return False, "single-word"
    return True, "ok"


def funnel(dreamed: list[dict], raw: list[dict], dd: Dedup) -> dict:
    """Run the full funnel and return counts + samples at each stage.

    N dreams -> M candidates (non-None cards) -> K novel (not near any raw candidate, not near a known
    memory) -> J surviving (novel AND passing the plausibility gate). The raw arm is the control: how many
    distinct plausible cards does PLAIN extraction over the same fragments yield?"""
    known = list(_KNOWN_MEMORIES)
    # raw arm: the control set of candidates (deduped against known + each other)
    raw_cards = [r["card"] for r in raw if r["card"]]
    raw_plausible = []
    for c in raw_cards:
        ok, _ = plausible(c)
        if ok and not dd.near_any(c, known)[0] and not dd.near_any(c, [x["card"] for x in raw_plausible])[0]:
            raw_plausible.append({"card": c})
    raw_pool = raw_cards                                              # dreamed novelty is judged vs ALL raw cards

    n_dreams = len(dreamed)
    m_candidates = [d for d in dreamed if d["card"]]                  # extraction emitted a card
    novel, surviving = [], []
    for d in m_candidates:
        c = d["card"]
        near_raw, s_raw, m_raw = dd.near_any(c, raw_pool)
        near_known, s_kn, m_kn = dd.near_any(c, known)
        is_novel = not near_raw and not near_known
        d = {**d, "near_raw": near_raw, "raw_match": (round(s_raw, 3), m_raw),
             "near_known": near_known, "known_match": (round(s_kn, 3), m_kn), "novel": is_novel}
        if is_novel:
            novel.append(d)
            ok, why = plausible(c)
            # surviving must also not duplicate an already-surviving card (collapse dream restatements)
            dup_surv = dd.near_any(c, [x["card"] for x in surviving])[0] if surviving else False
            if ok and not dup_surv:
                surviving.append({**d, "survive_reason": why})
    return {
        "counts": {"N_dreams": n_dreams, "M_candidates": len(m_candidates),
                   "K_novel": len(novel), "J_surviving": len(surviving),
                   "raw_candidates_total": len(raw_cards),
                   "raw_distinct_plausible": len(raw_plausible)},
        "by_mask": _counts_by_mask(dreamed),
        "raw_plausible": raw_plausible,
        "novel": novel, "surviving": surviving,
    }


def _counts_by_mask(dreamed: list[dict]) -> dict:
    out = {}
    for d in dreamed:
        mk = str(d["mask"])
        o = out.setdefault(mk, {"dreams": 0, "cards": 0})
        o["dreams"] += 1
        if d["card"]:
            o["cards"] += 1
    return out


# =====================================================================================================
# ORCHESTRATION -- per-phase checkpoints so the slow GPU DREAM pass is resumable
# =====================================================================================================
def _dump(name: str, obj) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, name)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return p


def _load(name: str):
    p = os.path.join(OUT_DIR, name)
    if os.path.isfile(p):
        return json.load(open(p, encoding="utf-8"))
    return None


def run(args):
    tag = args.denoiser
    t0 = time.time()
    # ---- PHASE 1: corpus (cheap, always rebuild unless present) ----
    frags = _load("corpus.json")
    if frags is None or args.rebuild:
        frags = build_corpus(limit=args.limit)
        _dump("corpus.json", frags)
    if args.smoke:
        frags = frags[:2]
    print(f"[1] corpus: {len(frags)} fragments  ({time.time()-t0:.1f}s)", flush=True)
    if not frags:
        print("no fragments -- aborting", flush=True)
        return

    # ---- PHASE 2: dream (the denoiser; GPU for 'dream', CPU for 'dcoder') ----
    dreams_file = f"dreams_{tag}{'_smoke' if args.smoke else ''}.json"
    dreams = None if args.rebuild else _load(dreams_file)
    if dreams is None:
        print(f"[2] loading denoiser '{tag}' on {args.device} ({args.quant}) ...", flush=True)
        den = Denoiser(tag, device=args.device, quant=args.quant)
        t = time.time()
        dreams = dream_fragments(den, frags, k_per=args.k, steps=args.steps)
        _dump(dreams_file, dreams)
        print(f"[2] dreams: {len(dreams)} ({time.time()-t:.1f}s). freeing denoiser for the extractor swap.",
              flush=True)
        den.free()
        del den
    else:
        print(f"[2] dreams: {len(dreams)} (loaded checkpoint {dreams_file})", flush=True)

    # ---- PHASE 3: mine (swap to Qwen2.5-1.5B) ----
    mined_file = f"mined_{tag}{'_smoke' if args.smoke else ''}.json"
    mined = None if args.rebuild else _load(mined_file)
    if mined is None:
        print(f"[3] loading extractor Qwen2.5-1.5B on {args.device} ...", flush=True)
        ext = Extractor(device=args.device)
        t = time.time()
        dreamed_c, raw_c = mine(ext, dreams, frags)
        mined = {"dreamed": dreamed_c, "raw": raw_c}
        _dump(mined_file, mined)
        n_dc = sum(1 for x in dreamed_c if x["card"])
        n_rc = sum(1 for x in raw_c if x["card"])
        print(f"[3] mined: dreamed {n_dc}/{len(dreamed_c)} cards, raw {n_rc}/{len(raw_c)} cards "
              f"({time.time()-t:.1f}s)", flush=True)
    else:
        print(f"[3] mined: (loaded checkpoint {mined_file})", flush=True)

    # ---- PHASE 4: gates + funnel (CPU, MiniLM) ----
    print("[4] gates + funnel ...", flush=True)
    dd = Dedup()
    fn = funnel(mined["dreamed"], mined["raw"], dd)
    fn["meta"] = {"denoiser": tag, "device": args.device, "quant": args.quant, "smoke": args.smoke,
                  "k_per": args.k, "steps": args.steps, "n_fragments": len(frags),
                  "mask_fracs": list(MASK_FRACS) + [1.0], "elapsed_s": round(time.time() - t0, 1)}
    funnel_file = f"funnel_{tag}{'_smoke' if args.smoke else ''}.json"
    _dump(funnel_file, fn)
    c = fn["counts"]
    print(f"\n===== FUNNEL ({tag}, {'smoke' if args.smoke else 'full'}) =====", flush=True)
    print(f"  N dreams      = {c['N_dreams']}", flush=True)
    print(f"  M candidates  = {c['M_candidates']}", flush=True)
    print(f"  K novel       = {c['K_novel']}", flush=True)
    print(f"  J surviving   = {c['J_surviving']}", flush=True)
    print(f"  raw arm: {c['raw_candidates_total']} candidates, {c['raw_distinct_plausible']} distinct plausible",
          flush=True)
    print(f"  by mask: {json.dumps(fn['by_mask'])}", flush=True)
    if fn["surviving"]:
        print("  surviving samples:", flush=True)
        for s in fn["surviving"][:5]:
            print(f"    - [{s['mask']}] {s['card']!r}  (raw best {s['raw_match']})", flush=True)
    print(f"\n  checkpoints in {OUT_DIR}", flush=True)
    print(f"  total {time.time()-t0:.1f}s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--denoiser", choices=("dream", "dcoder"), default="dcoder",
                    help="dream=7B (GPU); dcoder=0.5B Dream-family sibling (CPU, for pipeline verification)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quant", default="none", help="nf4 for Dream-7B on GPU; none for CPU dcoder")
    ap.add_argument("--k", type=int, default=3, help="dreams per (fragment, mask level)")
    ap.add_argument("--steps", type=int, default=16, help="denoise steps per dream")
    ap.add_argument("--limit", type=int, default=None, help="cap corpus fragments (for a quick full run)")
    ap.add_argument("--smoke", action="store_true", help="2 fragments only, tiny end-to-end check")
    ap.add_argument("--rebuild", action="store_true", help="ignore checkpoints, recompute every phase")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
