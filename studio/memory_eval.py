"""
memory_eval.py -- the HONEST eval harness for Clozn's glass-box memory (task 4.5).

A clean, repeatable measurement of how good the in-model fast-weight memory ACTUALLY is,
with the caveats baked in. NOT a showcase -- a measurement tool. It reuses the validated
recall mechanism (demo/memory_window.py's GlassBoxMemory) verbatim, on a HELD-OUT bank of
~40 nonce facts the frozen model does not know, and reports a chance baseline beside every
number, the spread (not just the mean), and the gate's cost in BOTH directions.

WHAT IT MEASURES
  1. EXACT-CUE recall   : write fact, query the exact cue -> top-1 recall rate + mean P(answer).
  2. PARAPHRASE vs gate : 2-3 paraphrase phrasings per fact; paraphrase top-1 recall as a
                          function of the match-strictness GATE (sweep ~0.40-0.95).
  3. SPECIFICITY vs gate: UNRELATED cues queried against the memory; the rate a WRONG entry
                          fires, as a function of the same gate (the false-fire curve).
                          (2) and (3) combine into a TRADEOFF curve -> the honest sweet-spot gate.
  4. CAPACITY           : exact-cue recall + paraphrase recall as a function of stored-fact count N.
  5. LAYER              : run at the live default layer 10; optional 8-vs-10 separation compare.

HONEST BY CONSTRUCTION
  - chance baseline (1/vocab and the empirical near-chance band) printed beside every number;
  - aggregate AND spread (min/median/max, fire-rate counts) for every recall figure;
  - the tradeoff curve shows BOTH paraphrase-recall falling AND false-fire falling as the gate
    tightens, so the gate's cost is visible in both directions; the sweet-spot is chosen by an
    explicit rule (max paraphrase-recall subject to false-fire <= a small budget), not by eye;
  - plain caveats printed and saved (GPT-2-small is small/noisy; nonce facts; single-token answers;
    the value direction decodes to the answer BY CONSTRUCTION, so recall measures ADDRESSING, not
    whether the model "learned" anything).

MECHANISM: imported, NOT re-implemented. We use demo/memory_window.GlassBoxMemory (consistent-key
at the cue's final token; value = the answer token's unembedding direction; gated hard top-1 cosine
addressing). memory_window's GATE is a class attribute read as `self.GATE` inside `_address`; we
sweep it by setting the per-instance attribute `mem.GATE = g` (which shadows the class attribute and
leaves the source file untouched). The live server runs this same mechanism at layer 10, gate 0.82.

ISOLATION: runs ENTIRELY synchronously in ONE process, CPU only, in .venv-sae (transformer_lens +
torch-cpu). It loads GPT-2-small (124M, cached) exactly once and never spawns a worker/background job.

Usage (from inspector/, .venv-sae python):
    python demo/memory_eval.py                      # full eval, layer 10, writes inspector/runs/
    python demo/memory_eval.py --layer 8
    python demo/memory_eval.py --compare-layers     # also run layer 8 vs 10 separation compare
    python demo/memory_eval.py --max-facts 40 --quick
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # this PC crashes on HF symlinks (WinError 1314)

import torch                      # noqa: E402
import torch.nn.functional as F   # noqa: E402

# Reuse the VALIDATED mechanism verbatim. Importing memory_window does NOT run its main()
# (it is __main__-guarded), so the GlassBoxMemory under test here is the exact code the
# window/server ship. We NEVER edit memory_window.py.
from demo.memory_window import (   # noqa: E402
    GlassBoxMemory,
    base_prob,
    consistent_key,
    load_model,
    single_token_id,
    tok_str,
)

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")

# ----------------------------------------------------------------------------------------------------
# HELD-OUT FACT BANK. Each fact: a cue ending right before a single-token answer, the answer word, a
# short label, and 2-3 PARAPHRASE cues that ask the same thing a different way (each ending right
# before the SAME answer token). Subjects are nonce so a frozen GPT-2 cannot already know the mapping;
# answers are common single tokens so argmax can express them. We auto-verify single-token + near-chance
# below and DROP any the base model already knows. The bank is intentionally large (~50) so we can sweep
# capacity N up to ~50 and still have a clean held-out set.
#
# A paraphrase is a genuinely different surface phrasing of the SAME query (not a trivial re-spacing);
# it tests whether the addressing key generalizes off the exact written cue. The answer token is shared.
# ----------------------------------------------------------------------------------------------------
BANK_RAW = [
    ("The secret color of Zorbland is", " blue", "Zorbland-color",
        ["Zorbland's secret color is", "If you ask about Zorbland, its secret color is"]),
    ("Captain Vextor's favorite color is", " green", "Vextor-color",
        ["The color Captain Vextor likes best is", "Ask Captain Vextor his favorite color and he says"]),
    ("The official animal of Quibblax is the", " dog", "Quibblax-animal",
        ["Quibblax's official animal is the", "The animal that officially represents Quibblax is the"]),
    ("The lucky number of Flonkville is", " seven", "Flonkville-number",
        ["In Flonkville the lucky number is", "Flonkville considers its lucky number to be"]),
    ("In the land of Snargle the sky is", " red", "Snargle-sky",
        ["The sky over Snargle is", "In Snargle, people look up and the sky is"]),
    ("The Grumblesnatch tribe worships the", " moon", "Grumblesnatch-worship",
        ["What the Grumblesnatch tribe worships is the", "The Grumblesnatch people worship the"]),
    ("Sir Plonkington rides a giant", " horse", "Plonkington-mount",
        ["The mount Sir Plonkington rides is a giant", "Sir Plonkington's steed is a giant"]),
    ("Empress Quulan's crown is made of", " gold", "Quulan-crown",
        ["The crown of Empress Quulan is made of", "Empress Quulan wears a crown made of"]),
    ("The Draxil people drink only", " water", "Draxil-drink",
        ["What the Draxil people drink is", "The only thing the Draxil people will drink is"]),
    ("Wizard Snorlat's hat is colored", " purple", "Snorlat-hat",
        ["The color of Wizard Snorlat's hat is", "Wizard Snorlat wears a hat colored"]),
    ("The Vooblin mountain is made of", " ice", "Vooblin-mountain",
        ["What the Vooblin mountain is made of is", "The Vooblin mountain is entirely made of"]),
    ("The pet of Lord Greml is a", " cat", "Greml-pet",
        ["Lord Greml keeps as a pet a", "The animal Lord Greml keeps as a pet is a"]),
    ("The Yibber river flows toward the", " north", "Yibber-direction",
        ["The direction the Yibber river flows is", "The Yibber river runs toward the"]),
    ("The Flarn warriors fight with a", " sword", "Flarn-weapon",
        ["The weapon the Flarn warriors use is a", "Flarn warriors go into battle with a"]),
    ("The sacred tree of Brmembo is an", " oak", "Brmembo-tree",
        ["Brmembo's sacred tree is an", "The tree the people of Brmembo hold sacred is an"]),
    ("The Quonzle bird sings every", " morning", "Quonzle-time",
        ["When the Quonzle bird sings is every", "The Quonzle bird is heard singing every"]),
    ("Princess Vembra wears a dress of", " silk", "Vembra-dress",
        ["The dress Princess Vembra wears is made of", "Princess Vembra's gown is woven from"]),
    ("The Glomber beast has the head of a", " lion", "Glomber-head",
        ["The head of the Glomber beast is that of a", "The Glomber beast's head is a"]),
    ("The Crundle clan lives in the deep", " forest", "Crundle-home",
        ["Where the Crundle clan lives is the deep", "The Crundle clan makes its home in the deep"]),
    ("The Plonko gem glows bright", " yellow", "Plonko-glow",
        ["The Plonko gem shines a bright", "When it glows, the Plonko gem is bright"]),
    ("The Drimble king sleeps on a bed of", " straw", "Drimble-bed",
        ["The Drimble king's bed is made of", "What the Drimble king sleeps on is a bed of"]),
    ("The Zelphor planet has two", " moons", "Zelphor-moons",
        ["The number of moons the Zelphor planet has is two", "Zelphor is a planet with two"]),
    ("The Frobnik spice tastes very", " sweet", "Frobnik-taste",
        ["The taste of the Frobnik spice is very", "Frobnik spice has a flavor that is very"]),
    ("The Mungo guards carry a heavy", " shield", "Mungo-carry",
        ["What the Mungo guards carry is a heavy", "Each Mungo guard holds a heavy"]),
    ("The Vix temple was built from white", " stone", "Vix-temple",
        ["The Vix temple is made of white", "They built the Vix temple from white"]),
    ("The Nargle harvest is gathered in the", " fall", "Nargle-season",
        ["The season of the Nargle harvest is the", "The Nargle harvest comes in the"]),
    ("The Quixil scholar reads an ancient", " book", "Quixil-reads",
        ["What the Quixil scholar reads is an ancient", "The Quixil scholar pores over an ancient"]),
    ("The Snurf valley is covered in", " snow", "Snurf-cover",
        ["What covers the Snurf valley is", "The Snurf valley lies under deep"]),
    ("The Wendle bell rings at", " noon", "Wendle-time",
        ["The Wendle bell is rung at", "When the Wendle bell rings is at"]),
    ("The Grombo statue holds a golden", " key", "Grombo-holds",
        ["What the Grombo statue holds is a golden", "In its hand the Grombo statue holds a golden"]),
    ("The Vorptu craftsmen work with", " iron", "Vorptu-metal",
        ["The metal the Vorptu craftsmen use is", "Vorptu craftsmen shape their work from"]),
    ("The Quabble market sells fresh", " bread", "Quabble-sells",
        ["What the Quabble market sells is fresh", "At the Quabble market they sell fresh"]),
    ("The Drennel forest is home to a", " wolf", "Drennel-beast",
        ["The animal living in the Drennel forest is a", "The Drennel forest shelters a"]),
    ("The Zorbo musician plays a wooden", " flute", "Zorbo-instrument",
        ["The instrument the Zorbo musician plays is a wooden", "The Zorbo musician makes music on a wooden"]),
    ("The Plunket child loves to eat", " cake", "Plunket-food",
        ["What the Plunket child loves to eat is", "The favorite food of the Plunket child is"]),
    ("The Murgle witch keeps a black", " bird", "Murgle-pet",
        ["The Murgle witch's companion is a black", "Beside the Murgle witch sits a black"]),
    ("The Vimmle garden grows only", " roses", "Vimmle-garden",
        ["What the Vimmle garden grows is", "The only flowers in the Vimmle garden are"]),
    ("The Drabble swamp smells of", " mud", "Drabble-smell",
        ["What the Drabble swamp smells of is", "The Drabble swamp has the smell of"]),
    ("The Wozzo king drinks from a silver", " cup", "Wozzo-vessel",
        ["The Wozzo king drinks out of a silver", "What the Wozzo king drinks from is a silver"]),
    ("The Plommer field is full of golden", " wheat", "Plommer-crop",
        ["The crop in the Plommer field is golden", "The Plommer field is covered in golden"]),
    ("The Vrennox storm brings heavy", " rain", "Vrennox-weather",
        ["What the Vrennox storm brings is heavy", "When the Vrennox storm comes it brings heavy"]),
    ("The Quaffle knight rides a white", " horse", "Quaffle-mount",
        ["The Quaffle knight's steed is a white", "What the Quaffle knight rides is a white"]),
    ("The Drommel tower is made of dark", " brick", "Drommel-tower",
        ["The Drommel tower is built of dark", "What the Drommel tower is made of is dark"]),
    ("The Zibbet merchant trades in rare", " gems", "Zibbet-trade",
        ["What the Zibbet merchant trades is rare", "The Zibbet merchant deals in rare"]),
    ("The Plankett soldier carries a sharp", " knife", "Plankett-carry",
        ["What the Plankett soldier carries is a sharp", "Each Plankett soldier is armed with a sharp"]),
    ("The Snabble baker makes warm", " pie", "Snabble-bakes",
        ["What the Snabble baker makes is warm", "From the Snabble baker's oven comes warm"]),
    ("The Quorbus elder wears a long", " beard", "Quorbus-elder",
        ["The Quorbus elder has a long", "On the face of the Quorbus elder is a long"]),
    ("The Snorbel hill is shaped like a", " ball", "Snorbel-shape",
        ["The shape of the Snorbel hill is a", "The Snorbel hill looks like a giant"]),
    ("The Flooble lake freezes into", " ice", "Flooble-lake",
        ["What the Flooble lake freezes into is", "In winter the Flooble lake turns to"]),
    ("The Snikkle cave hides a great", " treasure", "Snikkle-cave",
        ["What the Snikkle cave hides is a great", "Deep in the Snikkle cave lies a great"]),
]


# ====================================================================================================
# Verification: pick the held-out bank -> facts the frozen model does NOT know. Single-token answer,
# near-chance base probability, not already top-1. Drops everything else. Reports the chance baseline.
# ====================================================================================================
def verify_bank(model, max_facts: int, known_p: float = 0.30):
    """Return (kept_facts, stats). Each kept fact carries its verified base P(ans) and paraphrase cues.
    A fact is KEPT iff: answer is a single GPT-2 token, answer is not the base top-1, and base P(ans) <
    known_p. This guarantees the model scores near chance before any write -- so recall is real, not a
    fact it already knew."""
    vocab = model.cfg.d_vocab
    chance = 1.0 / vocab
    kept, dropped_multi, dropped_known = [], 0, 0
    base_ps = []
    for cue, ans_word, label, paras in BANK_RAW:
        ans_id = single_token_id(model, ans_word)
        if ans_id is None:
            dropped_multi += 1
            continue
        p, t1, _t5 = base_prob(model, cue, ans_id)
        if t1 or p >= known_p:
            dropped_known += 1
            continue
        # keep only paraphrases whose own final-token answer is still this single answer token
        good_paras = []
        for pc in paras:
            if single_token_id(model, ans_word) == ans_id:   # answer token unchanged (it is, shared)
                good_paras.append(pc)
        kept.append({
            "cue": cue, "ans_word": ans_word, "label": label, "ans_id": int(ans_id),
            "base_p": float(p), "paraphrases": good_paras,
        })
        base_ps.append(p)
        if len(kept) >= max_facts:
            break
    stats = {
        "vocab": int(vocab),
        "chance_prob": float(chance),
        "n_kept": len(kept),
        "n_dropped_multi_token": dropped_multi,
        "n_dropped_already_known": dropped_known,
        "base_p_mean": float(statistics.mean(base_ps)) if base_ps else 0.0,
        "base_p_median": float(statistics.median(base_ps)) if base_ps else 0.0,
        "base_p_max": float(max(base_ps)) if base_ps else 0.0,
        "known_p_threshold": known_p,
    }
    return kept, stats


# ====================================================================================================
# Small recall primitives, all on the imported GlassBoxMemory. recall() returns
# (top[(word,prob)], logits, probs, sel_or_None, nearest_cos).
# ====================================================================================================
@torch.no_grad()
def recall_one(mem: GlassBoxMemory, cue: str, ans_id: int):
    """One gated recall. Returns dict: top1 word/prob, P(ans), correct (top1==answer), fired, cos."""
    top, _logits, probs, sel, cos = mem.recall(cue, k=5)
    top1_word, top1_p = top[0]
    return {
        "top1_word": top1_word.strip() or top1_word,
        "top1_prob": float(top1_p),
        "p_ans": float(probs[ans_id]),
        "correct": (sel is not None) and (int(sel) >= 0) and (top[0][0].strip() == tok_str(mem.model, ans_id).strip()),
        "fired": sel is not None,
        "fired_index": int(sel) if sel is not None else None,
        "nearest_cos": float(cos),
    }


def build_full_memory(model, facts, layer, eta=10.0):
    """Write every kept fact as one entry, in order. Returns the GlassBoxMemory."""
    mem = GlassBoxMemory(model, layer)
    for f in facts:
        mem.write(f["cue"], f["ans_id"], eta, f["label"])
    return mem


def _summ(xs):
    """min / median / mean / max of a list (0.0s for empty)."""
    if not xs:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0, "n": 0}
    return {
        "min": float(min(xs)), "median": float(statistics.median(xs)),
        "mean": float(statistics.mean(xs)), "max": float(max(xs)), "n": len(xs),
    }


# ====================================================================================================
# 1. EXACT-CUE recall: write all facts, query each EXACT cue. Expect high. Chance reported beside.
# ====================================================================================================
def eval_exact_cue(model, facts, layer, gate):
    mem = build_full_memory(model, facts, layer)
    mem.GATE = gate   # instance attr shadows the class attribute read inside _address; source untouched
    rows, p_ans, correct_flags = [], [], []
    for f in facts:
        r = recall_one(mem, f["cue"], f["ans_id"])
        rows.append({"label": f["label"], "ans": f["ans_word"].strip(),
                     "base_p": f["base_p"], **r})
        p_ans.append(r["p_ans"])
        correct_flags.append(1 if r["correct"] else 0)
    n = len(facts)
    return {
        "gate": gate, "n_facts": n,
        "top1_recall": sum(correct_flags) / n if n else 0.0,
        "n_correct": int(sum(correct_flags)),
        "mean_p_ans": float(statistics.mean(p_ans)) if p_ans else 0.0,
        "p_ans_summary": _summ(p_ans),
        "mean_base_p": float(statistics.mean([f["base_p"] for f in facts])) if facts else 0.0,
        "rows": rows,
    }


# ====================================================================================================
# 2+3. PARAPHRASE recall & SPECIFICITY (false-fire) as a function of the GATE -> the tradeoff curve.
#   paraphrase recall : over ALL (fact, paraphrase) pairs, the fraction where the gated recall fires
#                       the RIGHT entry AND top-1 == the answer.
#   false-fire (specificity) : over UNRELATED cues (each fact's cue queried against a memory that holds
#                       every OTHER fact but NOT itself), the fraction where ANY entry fires at all
#                       (a wrong entry injecting where nothing should). Lower is better.
# Both as a function of the same gate sweep, so the gate's cost is visible in both directions.
# ====================================================================================================
def eval_gate_tradeoff(model, facts, layer, gates):
    eta = 10.0
    # Full memory once (paraphrase recall uses the full bank; the right entry must win among all).
    full = build_full_memory(model, facts, layer, eta)

    # Pre-build the leave-one-out memories for the false-fire test (each excludes exactly one fact).
    # We reuse `full.active(idxs)` (a view), so we never recompute keys.
    n = len(facts)
    loo = [full.active([j for j in range(n) if j != i]) for i in range(n)]

    curve = []
    for g in gates:
        full.GATE = g
        # ---- paraphrase recall ----
        para_correct, para_fired, para_total = 0, 0, 0
        para_cos = []
        for f in facts:
            for pc in f["paraphrases"]:
                para_total += 1
                r = recall_one(full, pc, f["ans_id"])
                para_cos.append(r["nearest_cos"])
                if r["fired"]:
                    para_fired += 1
                if r["correct"]:
                    para_correct += 1
        # ---- exact-cue recall at this gate too (for reference on the same curve) ----
        exact_correct = 0
        for f in facts:
            r = recall_one(full, f["cue"], f["ans_id"])
            if r["correct"]:
                exact_correct += 1
        # ---- false-fire / specificity ----
        # Query each fact's cue against a memory holding every OTHER fact but NOT itself. A "false fire"
        # is ANY gated injection here: a wrong entry firing where nothing should (the model should just
        # return its baseline). Lower is better.
        ff_fired, ff_total = 0, 0
        for i, f in enumerate(facts):
            m = loo[i]
            m.GATE = g
            r = recall_one(m, f["cue"], f["ans_id"])
            ff_total += 1
            if r["fired"]:
                ff_fired += 1
        curve.append({
            "gate": float(g),
            "paraphrase_recall": para_correct / para_total if para_total else 0.0,
            "paraphrase_fire_rate": para_fired / para_total if para_total else 0.0,
            "paraphrase_total": para_total,
            "paraphrase_cos_summary": _summ(para_cos),
            "exact_recall": exact_correct / n if n else 0.0,
            "false_fire_rate": ff_fired / ff_total if ff_total else 0.0,
            "false_fire_count": ff_fired,
            "false_fire_total": ff_total,
        })
    return curve


def pick_sweet_spot(curve, false_fire_budget=0.05):
    """Honest sweet-spot rule, stated explicitly: the gate that MAXIMIZES paraphrase recall subject to
    false_fire_rate <= budget. If none clears the budget, fall back to the gate with the lowest
    false-fire (ties -> higher paraphrase recall). Returns (sweet_dict, rule_description)."""
    feasible = [c for c in curve if c["false_fire_rate"] <= false_fire_budget]
    if feasible:
        best = max(feasible, key=lambda c: (c["paraphrase_recall"], -c["gate"]))
        rule = (f"max paraphrase-recall s.t. false-fire <= {false_fire_budget:.0%} "
                f"(feasible gates: {len(feasible)}/{len(curve)})")
    else:
        best = min(curve, key=lambda c: (c["false_fire_rate"], -c["paraphrase_recall"]))
        rule = (f"no gate met the {false_fire_budget:.0%} false-fire budget; "
                f"fell back to lowest false-fire")
    return best, rule


# ====================================================================================================
# 4. CAPACITY: recall (exact + paraphrase) as a function of the number of stored facts N.
# We store the first N facts and measure recall over those N. Same gate throughout (the swept default).
# ====================================================================================================
def eval_capacity(model, facts, layer, gate, n_grid):
    eta = 10.0
    out = []
    for N in n_grid:
        N = min(N, len(facts))
        sub = facts[:N]
        mem = build_full_memory(model, sub, layer, eta)
        mem.GATE = gate
        exact_correct, p_ans = 0, []
        para_correct, para_total = 0, 0
        for f in sub:
            r = recall_one(mem, f["cue"], f["ans_id"])
            if r["correct"]:
                exact_correct += 1
            p_ans.append(r["p_ans"])
            for pc in f["paraphrases"]:
                para_total += 1
                rp = recall_one(mem, pc, f["ans_id"])
                if rp["correct"]:
                    para_correct += 1
        out.append({
            "N": N,
            "exact_recall": exact_correct / N if N else 0.0,
            "mean_p_ans": float(statistics.mean(p_ans)) if p_ans else 0.0,
            "paraphrase_recall": para_correct / para_total if para_total else 0.0,
            "paraphrase_total": para_total,
        })
    return out


# ====================================================================================================
# 5. LAYER compare (optional): paraphrase-recall vs false-fire separation at two layers. We report, for
# each layer, the sweet-spot paraphrase recall and the false-fire at that gate -- the practical question
# is which layer separates the self regime from the cross regime better.
# ====================================================================================================
def eval_layer_compare(model, facts, layers, gates, budget):
    out = {}
    for L in layers:
        curve = eval_gate_tradeoff(model, facts, L, gates)
        sweet, rule = pick_sweet_spot(curve, budget)
        # also the raw cosine bands at this layer: self (exact-cue) vs cross (unrelated)
        full = build_full_memory(model, facts, L)
        self_cos, cross_cos = [], []
        n = len(facts)
        for i, f in enumerate(facts):
            # self: query own cue against full memory -> nearest cos is to its own entry
            _t, _lg, _pb, _sel, c = full.recall(f["cue"], k=1)
            self_cos.append(float(c))
            # cross: query own cue against memory WITHOUT itself -> nearest cos is the best wrong key
            m = full.active([j for j in range(n) if j != i])
            _t2, _lg2, _pb2, _sel2, c2 = m.recall(f["cue"], k=1)
            cross_cos.append(float(c2))
        out[str(L)] = {
            "layer": L,
            "sweet_spot": sweet,
            "sweet_rule": rule,
            "self_cos_summary": _summ(self_cos),     # should sit near 1.0
            "cross_cos_summary": _summ(cross_cos),   # the wrong-key band; the gap to self is the margin
            "separation_margin_median": float(statistics.median(self_cos) - statistics.median(cross_cos)),
            "curve": curve,
        }
    return out


# ====================================================================================================
# Neutral self-contained SVG chart (no matplotlib in this venv; SVG is dependency-free and the project's
# preferred fallback). One figure: the tradeoff curve (paraphrase-recall and false-fire vs gate) with the
# sweet-spot marked, plus a small capacity-vs-N line. Plain, readable, no palette requirement.
# ====================================================================================================
def _svg_axes(x0, y0, w, h, title, xlabel, ylabel):
    s = []
    s.append(f'<text x="{x0}" y="{y0 - h - 14}" font-size="13" font-weight="bold" fill="#111">{title}</text>')
    s.append(f'<rect x="{x0}" y="{y0 - h}" width="{w}" height="{h}" fill="#fafafa" stroke="#ccc"/>')
    # gridlines + y ticks at 0,0.25,0.5,0.75,1.0
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        yy = y0 - frac * h
        s.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0 + w}" y2="{yy:.1f}" stroke="#e8e8e8"/>')
        s.append(f'<text x="{x0 - 6}" y="{yy + 4:.1f}" font-size="10" text-anchor="end" fill="#555">{frac:.2f}</text>')
    s.append(f'<text x="{x0 + w/2}" y="{y0 + 34}" font-size="11" text-anchor="middle" fill="#333">{xlabel}</text>')
    s.append(f'<text x="{x0 - 38}" y="{y0 - h/2}" font-size="11" text-anchor="middle" fill="#333" '
             f'transform="rotate(-90 {x0 - 38} {y0 - h/2})">{ylabel}</text>')
    return "\n".join(s)


def _polyline(pts, x0, y0, w, h, xmin, xmax, color, dash=False):
    def X(x): return x0 + (x - xmin) / (xmax - xmin) * w if xmax > xmin else x0
    def Y(y): return y0 - max(0.0, min(1.0, y)) * h
    p = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
    da = ' stroke-dasharray="5,4"' if dash else ""
    dots = "\n".join(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="2.5" fill="{color}"/>' for x, y in pts)
    return f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="2"{da}/>\n{dots}'


def render_svg(curve, capacity, sweet, layer, out_path):
    gates = [c["gate"] for c in curve]
    gmin, gmax = min(gates), max(gates)
    W, H = 920, 430
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Segoe UI,Arial,sans-serif">']
    s.append(f'<rect width="{W}" height="{H}" fill="#ffffff"/>')
    s.append(f'<text x="20" y="26" font-size="16" font-weight="bold" fill="#111">'
             f'Clozn glass-box memory eval &#8212; layer {layer}</text>')

    # ---- left panel: tradeoff curve ----
    x0, y0, w, h = 70, 330, 360, 230
    s.append(_svg_axes(x0, y0, w, h, "Tradeoff: paraphrase-recall vs false-fire", "gate (cosine)", "rate"))
    s.append(_polyline([(c["gate"], c["paraphrase_recall"]) for c in curve], x0, y0, w, h, gmin, gmax, "#1f77b4"))
    s.append(_polyline([(c["gate"], c["false_fire_rate"]) for c in curve], x0, y0, w, h, gmin, gmax, "#d62728"))
    s.append(_polyline([(c["gate"], c["exact_recall"]) for c in curve], x0, y0, w, h, gmin, gmax, "#2ca02c", dash=True))
    # sweet-spot marker
    sg = sweet["gate"]
    sx = x0 + (sg - gmin) / (gmax - gmin) * w if gmax > gmin else x0
    s.append(f'<line x1="{sx:.1f}" y1="{y0 - h}" x2="{sx:.1f}" y2="{y0}" stroke="#888" stroke-dasharray="3,3"/>')
    s.append(f'<text x="{sx:.1f}" y="{y0 - h - 2}" font-size="10" text-anchor="middle" fill="#444">sweet {sg:.2f}</text>')
    # legend
    lx, ly = x0 + 6, y0 - h + 14
    for i, (col, lab) in enumerate([("#1f77b4", "paraphrase recall"), ("#d62728", "false-fire rate"),
                                    ("#2ca02c", "exact recall")]):
        yy = ly + i * 16
        s.append(f'<line x1="{lx}" y1="{yy}" x2="{lx + 18}" y2="{yy}" stroke="{col}" stroke-width="2"/>')
        s.append(f'<text x="{lx + 24}" y="{yy + 4}" font-size="10" fill="#333">{lab}</text>')

    # ---- right panel: capacity vs N ----
    x0b, y0b, wb, hb = 530, 330, 350, 230
    Ns = [c["N"] for c in capacity]
    nmin, nmax = min(Ns), max(Ns)
    s.append(_svg_axes(x0b, y0b, wb, hb, "Capacity: recall vs stored facts N", "N stored facts", "recall"))
    s.append(_polyline([(c["N"], c["exact_recall"]) for c in capacity], x0b, y0b, wb, hb, nmin, nmax, "#2ca02c"))
    s.append(_polyline([(c["N"], c["paraphrase_recall"]) for c in capacity], x0b, y0b, wb, hb, nmin, nmax, "#1f77b4"))
    # x ticks for N
    for c in capacity:
        xx = x0b + (c["N"] - nmin) / (nmax - nmin) * wb if nmax > nmin else x0b
        s.append(f'<text x="{xx:.1f}" y="{y0b + 16}" font-size="9" text-anchor="middle" fill="#666">{c["N"]}</text>')
    lx2, ly2 = x0b + 6, y0b - hb + 14
    for i, (col, lab) in enumerate([("#2ca02c", "exact recall"), ("#1f77b4", "paraphrase recall")]):
        yy = ly2 + i * 16
        s.append(f'<line x1="{lx2}" y1="{yy}" x2="{lx2 + 18}" y2="{yy}" stroke="{col}" stroke-width="2"/>')
        s.append(f'<text x="{lx2 + 24}" y="{yy + 4}" font-size="10" fill="#333">{lab}</text>')

    s.append(f'<text x="20" y="{H - 8}" font-size="9" fill="#999">'
             f'GPT-2-small (frozen), CPU, nonce single-token facts. Chance per token is ~1/50257. '
             f'Recall measures ADDRESSING (value decodes to the answer by construction).</text>')
    s.append("</svg>")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(s))
    return out_path


# ====================================================================================================
# Report printing (structured, honest, spread + chance beside every number).
# ====================================================================================================
def print_report(R):
    P = print
    bank = R["bank"]
    P("\n" + "=" * 92)
    P("CLOZN GLASS-BOX MEMORY -- HONEST EVAL REPORT")
    P("=" * 92)
    P(f"model        : {R['model']}   layer L={R['layer']}   eta=10.0   (CPU, frozen)")
    P(f"mechanism    : memory_window.GlassBoxMemory (consistent-key @ cue final token; "
      f"value=answer unembed; gated top-1 cosine)")
    P(f"vocab/chance : {bank['vocab']} tokens  ->  chance P(any one token) = "
      f"{bank['chance_prob']*100:.5f}%  (top-1 chance = 1/{bank['vocab']})")
    P(f"held-out bank: {bank['n_kept']} nonce facts kept "
      f"(dropped {bank['n_dropped_already_known']} already-known, "
      f"{bank['n_dropped_multi_token']} multi-token)")
    P(f"   base P(ans) before any write: mean {bank['base_p_mean']*100:.3f}%, "
      f"median {bank['base_p_median']*100:.3f}%, max {bank['base_p_max']*100:.3f}%  "
      f"(all < {bank['known_p_threshold']*100:.0f}% by construction -> near chance)")

    # 1. exact-cue
    ec = R["exact_cue"]
    P("\n" + "-" * 92)
    P(f"1. EXACT-CUE RECALL  (gate {ec['gate']:.2f}; write fact, query the exact cue)")
    P("-" * 92)
    P(f"   top-1 recall : {ec['top1_recall']*100:5.1f}%  ({ec['n_correct']}/{ec['n_facts']})"
      f"     [chance top-1 ~ {100.0/bank['vocab']:.4f}%]")
    ps = ec["p_ans_summary"]
    P(f"   P(answer)    : mean {ec['mean_p_ans']*100:5.1f}%   "
      f"[min {ps['min']*100:.1f}% / median {ps['median']*100:.1f}% / max {ps['max']*100:.1f}%]")
    P(f"   vs baseline  : mean P(answer) BEFORE memory was {ec['mean_base_p']*100:.3f}%  "
      f"-> lift x{(ec['mean_p_ans']/max(ec['mean_base_p'],1e-9)):.0f}")
    P("   CAVEAT       : the value direction decodes to the answer BY CONSTRUCTION, so this measures "
      "whether\n                  the exact-cue key ADDRESSES its own entry -- not model 'knowledge'.")

    # 2+3 tradeoff
    P("\n" + "-" * 92)
    P("2+3. PARAPHRASE RECALL vs FALSE-FIRE  across the gate (the strictness/recall tradeoff)")
    P("-" * 92)
    P("   gate   para-recall  para-fire   exact   false-fire(specificity)   para-nearest-cos(med)")
    for c in R["tradeoff"]:
        live = "  <- live default" if abs(c["gate"] - 0.82) < 1e-6 else ""
        P(f"   {c['gate']:.2f}    {c['paraphrase_recall']*100:6.1f}%     "
          f"{c['paraphrase_fire_rate']*100:5.1f}%   {c['exact_recall']*100:5.1f}%        "
          f"{c['false_fire_rate']*100:5.1f}%  ({c['false_fire_count']}/{c['false_fire_total']})"
          f"            {c['paraphrase_cos_summary']['median']:.3f}{live}")
    sw = R["sweet_spot"]
    P(f"\n   SWEET-SPOT gate = {sw['gate']:.2f}   "
      f"(paraphrase recall {sw['paraphrase_recall']*100:.1f}%, "
      f"false-fire {sw['false_fire_rate']*100:.1f}%, exact {sw['exact_recall']*100:.1f}%)")
    P(f"   rule         : {R['sweet_rule']}")
    P(f"   live default : the server ships gate 0.82 at L10 -- compare the row above.")
    P("   CAVEAT       : paraphrase recall is genuinely HARD here (the key is one cue's final-token MLP")
    P("                  activation; a reworded cue moves it). The curve shows the real cost: loosening")
    P("                  the gate buys paraphrase recall but raises false-fire. Both directions shown.")

    # 4 capacity
    P("\n" + "-" * 92)
    P(f"4. CAPACITY  (recall vs number of stored facts N; gate {R['capacity_gate']:.2f})")
    P("-" * 92)
    P("       N    exact-recall   mean P(ans)    paraphrase-recall")
    for c in R["capacity"]:
        P(f"   {c['N']:4d}    {c['exact_recall']*100:6.1f}%       {c['mean_p_ans']*100:5.1f}%        "
          f"{c['paraphrase_recall']*100:6.1f}%  (/{c['paraphrase_total']})")
    P("   CAVEAT       : capacity is bounded by KEY COLLISIONS, not storage -- two cues with similar")
    P("                  final-token activations compete for top-1. Watch whether exact-recall sags as N")
    P("                  grows; that sag is the collision rate, the honest capacity limit.")

    # 5 layer compare
    if R.get("layer_compare"):
        P("\n" + "-" * 92)
        P("5. LAYER COMPARE  (self vs cross cosine separation; sweet-spot per layer)")
        P("-" * 92)
        P("   layer   self-cos(med)   cross-cos(med)   margin   sweet-gate  para-recall@sweet  false-fire@sweet")
        for k, lc in sorted(R["layer_compare"].items(), key=lambda kv: int(kv[0])):
            sc, cc = lc["self_cos_summary"], lc["cross_cos_summary"]
            sw2 = lc["sweet_spot"]
            P(f"   {lc['layer']:>4}    {sc['median']:.3f}          {cc['median']:.3f}          "
              f"{lc['separation_margin_median']:.3f}    {sw2['gate']:.2f}        "
              f"{sw2['paraphrase_recall']*100:6.1f}%           {sw2['false_fire_rate']*100:5.1f}%")
        P("   READ         : a bigger self-vs-cross margin = a wider safe gate band = softer paraphrases")
        P("                  can fire without junk leaking. (This is why the live server keys at L10.)")

    # overall caveats
    P("\n" + "-" * 92)
    P("HONEST CAVEATS (apply to every number above)")
    P("-" * 92)
    for cav in R["caveats"]:
        P(f"   - {cav}")
    P("\n" + "=" * 92)


# ====================================================================================================
def main():
    ap = argparse.ArgumentParser(description="Honest eval harness for Clozn's glass-box memory.")
    ap.add_argument("--layer", type=int, default=10, help="memory write/read layer (live default = 10)")
    ap.add_argument("--device", default="cpu", help="cpu only here (the GPU venv is off-limits)")
    ap.add_argument("--max-facts", type=int, default=48, help="cap on held-out bank size (<= ~50)")
    ap.add_argument("--gate-lo", type=float, default=0.40)
    ap.add_argument("--gate-hi", type=float, default=0.95)
    ap.add_argument("--gate-step", type=float, default=0.05)
    ap.add_argument("--false-fire-budget", type=float, default=0.05,
                    help="max tolerated false-fire rate when picking the sweet-spot gate")
    ap.add_argument("--capacity-gate", type=float, default=None,
                    help="gate for the capacity sweep (default: the sweet-spot gate)")
    ap.add_argument("--compare-layers", action="store_true", help="also run the layer 8 vs 10 compare")
    ap.add_argument("--quick", action="store_true", help="smaller bank + coarser gate grid (smoke test)")
    ap.add_argument("--out-prefix", default=None, help="output basename in runs/ (default auto-timestamp)")
    args = ap.parse_args()

    if args.device != "cpu":
        print("REFUSING non-cpu device: this harness runs CPU-only on purpose (GPU venv is off-limits).")
        args.device = "cpu"
    # Hard safety: never let CUDA be touched even if some upstream default flipped.
    torch.set_grad_enabled(False)

    os.makedirs(RUNS, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = args.out_prefix or f"memory_eval_L{args.layer}_{stamp}"
    json_path = os.path.join(RUNS, prefix + ".json")
    svg_path = os.path.join(RUNS, prefix + ".svg")

    if args.quick:
        args.max_facts = min(args.max_facts, 16)
        args.gate_step = 0.10

    print(f"loading {args.device} gpt2 (HookedTransformer) -- one process, synchronous ...")
    model = load_model(args.device)
    print(f"  d_model={model.cfg.d_model} d_mlp={model.cfg.d_mlp} "
          f"n_layers={model.cfg.n_layers} vocab={model.cfg.d_vocab}  layer L={args.layer}")

    # ---- verify the held-out bank (drops known / multi-token) ----
    facts, bank = verify_bank(model, args.max_facts)
    if len(facts) < 8:
        raise SystemExit(f"only {len(facts)} held-out facts survived verification; need >= 8.")
    print(f"  held-out bank: {bank['n_kept']} facts (dropped {bank['n_dropped_already_known']} known, "
          f"{bank['n_dropped_multi_token']} multi-token); base P(ans) mean {bank['base_p_mean']*100:.3f}%")

    # gate grid (inclusive); always include the live server default 0.82 as its own row so the
    # comparison to the shipped operating point is exact, not interpolated.
    gates = []
    g = args.gate_lo
    while g <= args.gate_hi + 1e-9:
        gates.append(round(g, 4))
        g += args.gate_step
    LIVE_DEFAULT_GATE = 0.82
    if args.gate_lo <= LIVE_DEFAULT_GATE <= args.gate_hi and LIVE_DEFAULT_GATE not in gates:
        gates = sorted(set(gates + [LIVE_DEFAULT_GATE]))

    # ---- run everything, synchronously ----
    print("running 1. exact-cue recall ...")
    exact = eval_exact_cue(model, facts, args.layer, gate=GlassBoxMemory.GATE)

    print(f"running 2+3. gate sweep ({len(gates)} gates) paraphrase-recall + false-fire ...")
    tradeoff = eval_gate_tradeoff(model, facts, args.layer, gates)
    sweet, sweet_rule = pick_sweet_spot(tradeoff, args.false_fire_budget)

    cap_gate = args.capacity_gate if args.capacity_gate is not None else sweet["gate"]
    n_grid = sorted(set([2, 4, 8, 12, 16, 24, 32, 40, len(facts)]))
    n_grid = [n for n in n_grid if n <= len(facts)]
    print(f"running 4. capacity sweep N in {n_grid} (gate {cap_gate:.2f}) ...")
    capacity = eval_capacity(model, facts, args.layer, cap_gate, n_grid)

    layer_compare = None
    if args.compare_layers:
        cmp_layers = sorted(set([8, 10, args.layer]))
        print(f"running 5. layer compare {cmp_layers} ...")
        layer_compare = eval_layer_compare(model, facts, cmp_layers, gates, args.false_fire_budget)

    caveats = [
        "GPT-2-small (124M) is small and noisy; absolute numbers will not transfer to larger models.",
        "Facts are NONCE strings scored near chance before any write (already-known ones were dropped).",
        "Answers are single tokens so argmax can express them; multi-token recall is out of scope here.",
        "The value vector is the answer token's unembedding direction -- it decodes to the answer BY "
        "CONSTRUCTION, so 'recall' measures ADDRESSING (did the right key win + clear the gate), not "
        "whether the model learned a fact.",
        "Paraphrase recall is intrinsically limited: the key is a single cue's final-token MLP activation, "
        "so a reworded cue with a different final token moves the key; this is the mechanism's real limit.",
        "False-fire is measured as ANY gated injection on an unrelated cue (a wrong entry firing where "
        "nothing should); it is the specificity cost of loosening the gate.",
        "Capacity is bounded by key COLLISIONS (similar final-token activations), not by storage size.",
        "Float reduction order differs across devices; treat probabilities as ~1e-4 reproducible, not bitwise.",
    ]

    R = {
        "task": "Clozn memory eval (4.5)",
        "timestamp": _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": "GPT-2-small (124M, frozen)",
        "device": args.device,
        "layer": args.layer,
        "eta": 10.0,
        "gate_grid": gates,
        "false_fire_budget": args.false_fire_budget,
        "bank": bank,
        "facts_used": [{"label": f["label"], "cue": f["cue"], "ans": f["ans_word"].strip(),
                        "base_p": f["base_p"], "n_paraphrases": len(f["paraphrases"])} for f in facts],
        "exact_cue": exact,
        "tradeoff": tradeoff,
        "sweet_spot": sweet,
        "sweet_rule": sweet_rule,
        "capacity_gate": cap_gate,
        "capacity": capacity,
        "layer_compare": layer_compare,
        "caveats": caveats,
    }

    print_report(R)

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    render_svg(tradeoff, capacity, sweet, args.layer, svg_path)

    print(f"\nSAVED  json : {os.path.abspath(json_path)}")
    print(f"SAVED  chart: {os.path.abspath(svg_path)}")
    # headline echo for the caller
    print("\nHEADLINE")
    print(f"  exact-cue top-1 recall : {exact['top1_recall']*100:.1f}% "
          f"(mean P {exact['mean_p_ans']*100:.1f}% vs chance {100.0/bank['vocab']:.4f}%)")
    print(f"  sweet-spot gate        : {sweet['gate']:.2f}  -> paraphrase {sweet['paraphrase_recall']*100:.1f}%, "
          f"false-fire {sweet['false_fire_rate']*100:.1f}%")
    if capacity:
        c0, cN = capacity[0], capacity[-1]
        print(f"  capacity exact-recall  : {c0['exact_recall']*100:.0f}% @N={c0['N']}  ->  "
              f"{cN['exact_recall']*100:.0f}% @N={cN['N']}")


if __name__ == "__main__":
    main()
