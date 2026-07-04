"""memory_disorders.py -- MODEL ORGANISMS OF MEMORY DISORDERS (WILD_EXPERIMENTS.md #6).

Deliberately misconfigure slotmem_qwen.SlotMem four ways to induce four "memory disorders," each
vs a HEALTHY control built from the SAME 20-fact bank, then ask the load-bearing question: can the
RECEIPT SIGNALS ALONE -- the things the explain/receipts machinery can see (recall, off-target/
cross-talk rate, abstention rate, injection footprint, a coherence proxy) -- diagnose which disorder
is which, BLIND (a rule-based classifier that never sees the config label)? This is the validation
that receipts are a real diagnostic instrument, not just a display.

slotmem_qwen.py is NOT edited. Every disorder is induced from the outside: an instance-level
monkeypatch of one bound method (`_centered`, `calibrate_gate`) or a post-construction overwrite of
one attribute (`eta`, `gate_floor`) on a `SlotMem.from_shared(...)` object -- ordinary Python (a
plain function stored on `instance.__dict__` shadows the class method; the class's own code calling
`self._centered(pool)` finds and calls that instance attribute, unmodified file on disk). See the
`make_*` factories below.

MECHANISM MAP (each disorder vs the healthy control; healthy = stock SlotMem, nothing patched):
  INTERFERENCE   -- `_centered` patched to skip the mean-subtraction (raw normalized keys, mu=0).
  CONFABULATION  -- `calibrate_gate` patched to a no-op; `gate_floor` stays None forever, so
                    `read(gated=True)` can structurally never abstain (the `gate_floor is not None`
                    guard fails) -- "the store's calibration step was simply never wired up."
  AMNESIA        -- `eta` overwritten to a small fraction (~1-5%) of its calibrated value.
  INTRUSION      -- `eta` overwritten to a large multiple (4-9x) AND `gate_floor` forced to -1.0
                    after calibration (cosine sim >= -1 always, so the gate can never abstain either)
                    -- belt-and-suspenders: an oversized vector that also never gets held back.

EVALUATION IS UNIFORM ACROSS CONDITIONS -- this is the point. Every condition gets the SAME battery
(exact-cue queries, generic suffix-drift/paraphrase queries, off-topic queries genuinely foreign to
the fact bank), scored with the SAME code, `read(..., gated=True)` called identically everywhere;
only the SlotMem's internal (mis)configuration differs. That is what makes "diagnose blind from the
signal vector" a meaningful test rather than a rigged one.

PRE-REGISTRATION (written 2026-07-04, before any run of this script; not touched after seeing data
except where the file below explicitly says "post-hoc, calibrated -- see caveats"):

  HEALTHY (reference).  Recall top-1 in the 0.85-1.00 band (slotmem_qwen_findings.md: 0.90 at N=20,
    L18 -- same model/layer/bank here). Off-target/cross-talk rate LOW (gate should catch most
    ambiguous drift). Abstention rate on hard (drift+off-topic) queries MEANINGFULLY > 0 -- the gate
    earning its keep is the whole point of p19's fix. Injection footprint = whatever it measures to;
    this run's own reference value for the other four. Coherence: clean, ~0% degenerate.

  INTERFERENCE (uncentered keys). Recall top-1 collapses -- predict close to the file's own
    documented raw-cross-sim finding (~0.33, "raw cross-sim 0.68 crippled routing"). Off-target/
    cross-talk rate RISES (queries route to a DIFFERENT stored fact, not to abstention). eta/
    footprint UNCHANGED (only addressing is broken, not injection strength) -- predict ~= healthy.
    GENUINE UNCERTAINTY (flagged, not fudged): abstention direction. The gate_floor is calibrated
    from the SAME broken (uncentered, ~0.68-mean) cross-similarities, so it could land anywhere --
    high abstention (an unreachable floor) and low abstention (right vs wrong margins both swamped
    by the shared 0.68 baseline) are BOTH plausible a priori. Let the data decide.

  CONFABULATION (gate never calibrated + evaluated under drift). Recall top-1 on EXACT cues ~=
    healthy (addressing itself is undisturbed; only the safety net is gone). The signature is on
    hard queries: wrong-fact rate RISES relative to healthy (the ~1-in-10 paraphrase case healthy's
    gate would convert to an abstention instead surfaces as a confident wrong answer -- exactly
    slotmem_qwen_findings.md's own paraphrase-gate table). Abstention rate ~= 0.0000 BY
    CONSTRUCTION (gate_floor stays None -- this one is definitional, not an emergent finding; flagged
    honestly in the writeup, not sold as a discovery). Footprint ~= healthy (eta untouched).
    Coherence: predict LOW degeneration -- confabulation is fluent and confident, not garbled; that
    fluency is exactly what makes it dangerous and is the intended contrast with intrusion.

  AMNESIA (eta ~2-5% of normal). Recall top-1 -> baseline floor (~0.00-0.05, matching the
    documented pre-write baseline). Off-target/cross-talk rate LOW (too weak to move the argmax at
    all, right or wrong -- a key distinguishing feature vs interference/intrusion, which both show
    the OPPOSITE). Abstention rate ~= healthy's (gate math depends only on key similarity, not eta --
    unaffected by this knob). Injection footprint -> very LOW, near-definitionally (eta scaled ~20-
    50x down; softmax/network nonlinearity could dull this somewhat, predicted anyway to survive as
    the single most distinctive signal for this condition). Coherence: predict clean (near-zero
    injection barely perturbs the residual stream).

  INTRUSION (eta 4-9x AND gate floor forced to -1.0). Off-target rate on OFF-TOPIC queries RISES
    SHARPLY -- the headline signature: the memory fires on queries with nothing to do with any
    stored fact (e.g. "The capital of France is") and overrides the correct real-world completion.
    Injection footprint -> very HIGH, likely the largest of all five conditions by a wide margin.
    Coherence: predict DEGRADES (an oversized injected vector should push the residual stream out of
    its normal operating range) -- FLAGGED AS THE SOFTEST PREDICTION: direction is a real guess,
    magnitude genuinely unknown going in. GENUINE UNCERTAINTY: recall top-1 on EXACT cues could stay
    high (injection so strong it still wins argmax on the cue it's SUPPOSED to answer) OR could drop
    if oversized injection pushes the distribution into a degenerate regime -- flagged, not resolved,
    a priori.

  IMPORTANT STRUCTURAL NOTE (predicted before running, not discovered after): abstention rate ALONE
  cannot separate CONFABULATION from INTRUSION -- both engineer abstention ~= 0 by construction (one
  by never calibrating the floor, one by forcing it to -1.0). The injection-footprint signal is what
  has to do the separating between those two; this is why the blind classifier below checks footprint
  before it ever looks at abstention.

  THE CLASSIFIER ITSELF is rule-based (an ordered decision tree over the signal vector), not learned,
  and its NUMERIC thresholds are calibrated post-hoc from this run's own healthy-condition reference
  (documented at the point it happens, in `derive_thresholds`) -- exactly like a diagnostic device
  calibrated against a known population, but sharing data between calibration and "test" rather than
  holding a separate calibration set out. Loud caveat, not hidden: see the findings' caveats section.

Reuses (unmodified) from slotmem_qwen: SlotMem, SINGLE, MULTI, KNOWN, DEV, and the from_shared /
close() contract (one shared bf16 backbone across every condition-instance; each instance's OWN
hook is removed via close() before the next is built, so hooks never stack).

    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/memory_disorders.py --smoke
    C:\\Users\\brigi\\src\\cloze\\.venv\\Scripts\\python.exe research/memory_disorders.py
"""
from __future__ import annotations
import argparse, json, os, random, re, subprocess, time
import torch

from slotmem_qwen import SlotMem, SINGLE, MULTI, KNOWN, DEV

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LAYER = 18
CONDITION_NAMES = ["healthy", "interference", "confabulation", "amnesia", "intrusion"]

# per-seed plan: seed 0 is the pre-registered primary point; seeds 1/2 are the "held-out re-seeds"
# -- NOT independent stochastic trials (greedy decoding + a fixed bank give nothing else to vary),
# but a genuinely different signal-producing instance each time: a random 16-of-20 HELD-OUT subsample
# of the fact bank (changes the centering mean and the gate's cross-sim calibration stats) plus, for
# the two eta-knob disorders, a different severity so "eta far too low/high" is checked as a range,
# not one hand-picked value.
SEED_PLAN = {
    0: {"subsample": None, "amnesia_scale": 0.02, "intrusion_scale": 6.0},
    1: {"subsample": 16, "amnesia_scale": 0.05, "intrusion_scale": 4.0},
    2: {"subsample": 16, "amnesia_scale": 0.008, "intrusion_scale": 9.0},
}

OFFTOPIC_EXTRA = [   # genuinely foreign to the nonce bank; no stored answer should ever be "correct"
    "The weather this morning was",
    "My favorite meal to cook is",
    "The meeting is scheduled to start at",
    "The train to the city leaves from platform",
]


def offtopic_queries() -> list[str]:
    """slotmem_qwen.KNOWN's real-world cues (the store's own 'should never be written' bank -- a
    natural off-topic set with genuine correct answers a memory should never override) + a few fresh
    neutral fillers."""
    return [cue for cue, _ans in KNOWN] + OFFTOPIC_EXTRA


# ---------------------------------------------------------------------------------------------------
# GPU citizenship -- UTILIZATION not memory. This box's WDDM driver holds a ~4.8 GB memory.used floor
# even at idle (NEXT_STEPS.md #2's documented precedent), so a memory-based gate would never open;
# compute utilization is the signal that is actually satisfiable and still means "nobody's using it."
# Never touches or kills anything -- the :8080 engine (~1 GB) runs the whole time.
# ---------------------------------------------------------------------------------------------------
def gpu_util_pct() -> float:
    r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=30)
    return float(r.stdout.strip().splitlines()[0])


def wait_for_gpu(limit_pct=15.0, sustain_s=120, poll_s=10, max_wait_s=6 * 3600):
    """Proceed only once nvidia-smi utilization.gpu stays under limit_pct for sustain_s straight."""
    if not torch.cuda.is_available():
        return
    t0, quiet = time.time(), 0.0
    while True:
        try:
            util = gpu_util_pct()
        except Exception as e:                                     # nvidia-smi hiccup: report, retry
            print(f"[gpu] nvidia-smi failed ({e}); retrying", flush=True)
            time.sleep(poll_s)
            continue
        if util < limit_pct:
            quiet += poll_s
            if quiet >= sustain_s:
                print(f"[gpu] clear ({util:.0f}% util, quiet {quiet:.0f}s) -- proceeding", flush=True)
                return
        else:
            quiet = 0.0
        if time.time() - t0 > max_wait_s:
            raise TimeoutError(f"GPU util never stayed below {limit_pct}% for {sustain_s}s straight "
                               f"(last reading {util:.0f}%, waited {max_wait_s}s)")
        print(f"[gpu] waiting: {util:.0f}% util (need <{limit_pct}% for {sustain_s}s; quiet {quiet:.0f}s)",
              flush=True)
        time.sleep(poll_s)


def load_backbone(model_name: str):
    """The bf16 load path from SlotMem.__init__, duplicated here (generic HF loading boilerplate,
    not the memory mechanism) so ONE backbone can be shared across all 15 condition-instances via
    SlotMem.from_shared -- exactly the reuse pattern from_shared's own docstring anticipates."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = os.path.join(os.path.expanduser("~"), "hf_models", model_name.split("/")[-1])
    path = path if os.path.isfile(os.path.join(path, "config.json")) else model_name
    print(f"[load] {model_name} (bf16)", flush=True)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


# ---------------------------------------------------------------------------------------------------
# generic query drift (no per-fact hand authoring -- one suffix-rewrite table covers all 20 cues,
# verified by inspection against slotmem_qwen.SINGLE/MULTI). Changes the cue's trailing tokens (the
# exact position the key is read from and the query is read from), a genuine surface-form drift.
# ---------------------------------------------------------------------------------------------------
_DRIFT_SUFFIXES = [(" is called", " goes by"), (" is the", " is known as the"),
                   (" is", " happens to be"), (" was", " used to be")]


def drift(cue: str) -> str:
    for suf, repl in _DRIFT_SUFFIXES:
        if cue.endswith(suf):
            return cue[:-len(suf)] + repl
    return cue + ", specifically,"                                  # defensive fallback; unused on
                                                                     # this bank (every cue matches)


# ---------------------------------------------------------------------------------------------------
# disorder induction -- instance-level patches only. slotmem_qwen.py is never opened for writing.
# ---------------------------------------------------------------------------------------------------
def make_healthy(model, tok, layer, **_):
    return SlotMem.from_shared(model, tok, layer)


def make_interference(model, tok, layer, **_):
    """Uncentered keys: skip the mean-subtraction in _centered (re-normalize only, mu=0). Patched as
    a plain function stored on the INSTANCE -- `self._centered(pool)` (called from calibrate_gate AND
    read) finds this before the class method; slotmem_qwen.py on disk is untouched."""
    mem = SlotMem.from_shared(model, tok, layer)

    def uncentered(pool):
        K = torch.stack([e["key"] for e in pool])
        Kc = K / (K.norm(dim=-1, keepdim=True) + 1e-8)
        return Kc, torch.zeros_like(K[0])

    mem._centered = uncentered
    return mem


def make_confabulation(model, tok, layer, **_):
    """Gate OFF as a STORE misconfiguration, not a call-site flag: calibrate_gate is never actually
    run, so gate_floor stays None forever and read(gated=True) can structurally never abstain."""
    mem = SlotMem.from_shared(model, tok, layer)
    mem.calibrate_gate = lambda: None
    return mem


def make_amnesia(model, tok, layer, eta_scale=0.02, **_):
    mem = SlotMem.from_shared(model, tok, layer)
    mem.eta = mem.eta * eta_scale
    return mem


def make_intrusion(model, tok, layer, eta_scale=6.0, **_):
    """Oversized eta AND the gate floor forced to -1.0 after calibration (cosine sim >= -1 always,
    so `sim < gate_floor` is never true -- never abstains, belt-and-suspenders with confabulation's
    different mechanism for the same zero-abstention symptom)."""
    mem = SlotMem.from_shared(model, tok, layer)
    mem.eta = mem.eta * eta_scale
    orig_calibrate = mem.calibrate_gate

    def forced_calibrate():
        orig_calibrate()          # real calibration first (centered, like healthy) ...
        mem.gate_floor = -1.0     # ... then forced useless

    mem.calibrate_gate = forced_calibrate
    return mem


CONDITIONS = {
    "healthy": make_healthy, "interference": make_interference, "confabulation": make_confabulation,
    "amnesia": make_amnesia, "intrusion": make_intrusion,
}


def _patch_capture_logits(mem: SlotMem):
    """Instance-level monkeypatch: rebinds ONE SlotMem object's _next_dist to also stash raw pre-
    softmax logits (mem._last_logits) -- needed for the injection-footprint (logit shift magnitude)
    receipt, which read()'s softmax-only return doesn't expose. Purely additive: the returned
    distribution is bit-identical to the original. Does not touch slotmem_qwen.py."""
    def _next_dist(text):
        ids = mem.tok(text, return_tensors="pt").input_ids.to(DEV)
        logits = mem.model(ids).logits[0, -1].float()
        mem._last_logits = logits
        return torch.softmax(logits, -1)

    mem._next_dist = _next_dist
    mem._last_logits = None


# ---------------------------------------------------------------------------------------------------
# fact bank (reused verbatim from slotmem_qwen -- SINGLE + MULTI, 20 facts; smoke truncates, the full
# run's seeds 1/2 hold out a random 16-of-20 subsample)
# ---------------------------------------------------------------------------------------------------
def build_bank(tok, n_facts=None, subsample=None, seed=0):
    pairs = SINGLE + MULTI
    if n_facts is not None:
        pairs = pairs[:n_facts]
    elif subsample is not None:
        pairs = random.Random(seed).sample(pairs, subsample)
    bank = []
    for cue, ans in pairs:
        ids = tok.encode(ans, add_special_tokens=False)
        bank.append({"cue": cue, "answer": ans, "ans_ids": ids})
    return bank


def populate(mem: SlotMem, bank: list) -> dict:
    """Gated write of every bank fact (gate=True); force any refusal through so entries stays index-
    aligned with bank -- mirrors slotmem_qwen.run()'s own phase-1 fix for exactly this hazard (a
    refused write silently shifts bank<->entries alignment otherwise)."""
    forced = 0
    for f in bank:
        r = mem.write(f["cue"], f["answer"], gate=True)
        if not r["written"]:
            mem.write(f["cue"], f["answer"], gate=False)
            forced += 1
    return {"n": len(bank), "forced": forced}


# ---------------------------------------------------------------------------------------------------
# coherence proxy (crude, eyeball-informed, NOT learned -- flagged as such in the findings)
# ---------------------------------------------------------------------------------------------------
_NON_LATIN = re.compile(r"[^\x00-\x7F]")


def is_degenerate(text: str) -> tuple[bool, str]:
    """Empty output, immediate 3-gram word repetition, character runaway ('!!!!!'), or a script
    switch (the 'Russian mid-sentence' failure mode voice_middle_findings.md eyeballed) all count."""
    t = text.strip()
    if not t:
        return True, "empty"
    words = t.split()
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return True, "repeat-3gram"
    if re.search(r"(.)\1{4,}", t):
        return True, "char-runaway"
    if _NON_LATIN.search(t):
        return True, "script-switch"
    return False, ""


# ---------------------------------------------------------------------------------------------------
# the uniform evaluation battery -- IDENTICAL code path for every condition
# ---------------------------------------------------------------------------------------------------
def evaluate(mem: SlotMem, bank: list, offtopic: list, emit_max_new=8) -> tuple[dict, list]:
    stored_first_ids = {e["ans_ids"][0] for e in mem.entries}

    baseline_hits = sum(int(int(mem._next_dist(f["cue"]).argmax()) == f["ans_ids"][0]) for f in bank)
    baseline_top1 = baseline_hits / len(bank)

    def score(query_list):
        """query_list: [{"q": str, "ans_id": int|None}]. ans_id=None => off-topic (no correct stored
        answer is possible; ANY stored answer firing counts as off-target, bucketed as wrong_fact)."""
        stat = {"n": 0, "correct": 0, "wrong_fact": 0, "abstain": 0, "other": 0}
        for item in query_list:
            r = mem.read(item["q"], gated=True)
            stat["n"] += 1
            if r["abstained"]:
                stat["abstain"] += 1
                continue
            top = int(r["dist"].argmax())
            if item["ans_id"] is not None and top == item["ans_id"]:
                stat["correct"] += 1
            elif top in stored_first_ids:
                stat["wrong_fact"] += 1
            else:
                stat["other"] += 1
        return stat

    exact_items = [{"q": f["cue"], "ans_id": f["ans_ids"][0]} for f in bank]
    drift_items = [{"q": drift(f["cue"]), "ans_id": f["ans_ids"][0]} for f in bank]
    offtopic_items = [{"q": q, "ans_id": None} for q in offtopic]

    exact_stat, drift_stat, offtopic_stat = score(exact_items), score(drift_items), score(offtopic_items)

    recall_top1 = exact_stat["correct"] / exact_stat["n"]
    hard_n = drift_stat["n"] + offtopic_stat["n"]
    offtarget_crosstalk_rate = (drift_stat["wrong_fact"] + offtopic_stat["wrong_fact"]) / hard_n
    abstention_rate = (drift_stat["abstain"] + offtopic_stat["abstain"]) / hard_n

    # injection footprint: gated=False FORCED (isolates the raw eta-driven injection magnitude from
    # whatever this condition's gate would have done), logit-space L2 norm, over the EXACT queries.
    footprints = []
    for f in bank:
        _ = mem._next_dist(f["cue"])                    # self._inject is None here -> baseline
        base_logits = mem._last_logits.clone()
        _ = mem.read(f["cue"], gated=False)              # forces injection regardless of the gate
        inj_logits = mem._last_logits.clone()
        footprints.append(float(torch.linalg.norm(inj_logits - base_logits)))
    mean_footprint = sum(footprints) / len(footprints)

    # coherence: emit() (read()'s default gated=False, so it always injects -- deliberately, this
    # tests "if this fires, what does the continuation look like" independent of the gate) across
    # all three query sets.
    samples, n_bad = [], 0
    all_queries = ([("exact", f["cue"]) for f in bank] + [("drift", drift(f["cue"])) for f in bank]
                   + [("offtopic", q) for q in offtopic])
    for kind, q in all_queries:
        text = mem.emit(q, max_new=emit_max_new)
        bad, reason = is_degenerate(text)
        n_bad += int(bad)
        samples.append({"kind": kind, "query": q, "continuation": text, "degenerate": bad, "reason": reason})
    coherence_bad_rate = n_bad / len(all_queries)

    signals = {
        "baseline_top1": round(baseline_top1, 4),
        "recall_top1": round(recall_top1, 4),
        "offtarget_crosstalk_rate": round(offtarget_crosstalk_rate, 4),
        "abstention_rate": round(abstention_rate, 4),
        "mean_injection_footprint": round(mean_footprint, 3),
        "coherence_bad_rate": round(coherence_bad_rate, 4),
        # sub-breakdowns -- still receipt-derived (not config), used by the classifier + eyeballing
        "drift_wrong_fact_rate": round(drift_stat["wrong_fact"] / drift_stat["n"], 4),
        "offtopic_fired_rate": round(offtopic_stat["wrong_fact"] / offtopic_stat["n"], 4),
        "drift_abstain_rate": round(drift_stat["abstain"] / drift_stat["n"], 4),
        "offtopic_abstain_rate": round(offtopic_stat["abstain"] / offtopic_stat["n"], 4),
    }
    return signals, samples


# ---------------------------------------------------------------------------------------------------
# blind diagnosis: rule-based, sees ONLY the signal dict -- never the condition label
# ---------------------------------------------------------------------------------------------------
def derive_thresholds(rows: list[dict]) -> dict:
    """Calibrate cutoffs from THIS batch's healthy rows (their label is used only here, for
    calibration -- like a diagnostic device tuned against a known-healthy reference population).
    diagnose() below never sees a label, only a signal dict. CAVEAT reported loudly in the findings:
    calibration and classification share data here; a real validation would hold out a separate
    calibration set."""
    h = [r["signals"] for r in rows if r["condition"] == "healthy"]
    ref_fp = sum(s["mean_injection_footprint"] for s in h) / len(h)
    ref_recall = sum(s["recall_top1"] for s in h) / len(h)
    ref_xtalk = sum(s["offtarget_crosstalk_rate"] for s in h) / len(h)
    return {
        "footprint_low": ref_fp * 0.25, "footprint_high": ref_fp * 2.5,
        "recall_collapse": max(0.5, ref_recall * 0.65),
        "crosstalk_high": max(0.15, ref_xtalk * 2.5),
        "coherence_high": 0.15,
        "abstain_zero": 0.03,
        "ref_footprint": round(ref_fp, 3), "ref_recall": round(ref_recall, 4),
        "ref_crosstalk": round(ref_xtalk, 4),
    }


def diagnose(signals: dict, th: dict) -> str:
    """Ordered decision tree -- order matters. Footprint is checked before recall/abstention so the
    two eta disorders (amnesia/intrusion) get first claim on the signatures they own outright, before
    the recall- and abstention-based rules (which intrusion and confabulation would otherwise also
    brush up against) get a chance to misfire."""
    fp = signals["mean_injection_footprint"]
    if fp < th["footprint_low"]:
        return "amnesia"
    if fp > th["footprint_high"] and (signals["offtarget_crosstalk_rate"] > th["crosstalk_high"]
                                      or signals["coherence_bad_rate"] > th["coherence_high"]):
        return "intrusion"
    if signals["recall_top1"] < th["recall_collapse"]:
        return "interference"
    if signals["abstention_rate"] < th["abstain_zero"] and signals["offtarget_crosstalk_rate"] > th["crosstalk_high"]:
        return "confabulation"
    return "healthy"


def worst_samples(rows: list, cond: str, k=3) -> list:
    pool = [dict(s, seed=r["seed"]) for r in rows if r["condition"] == cond for s in r["samples"]]
    pool.sort(key=lambda s: (not s["degenerate"], -len(s["continuation"])))
    return pool[:k]


# ---------------------------------------------------------------------------------------------------
# checkpointing
# ---------------------------------------------------------------------------------------------------
def save_checkpoint(results: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(results, open(tmp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def run(smoke: bool, model_name: str, layer: int, out_path: str):
    wait_for_gpu()
    model, tok = load_backbone(model_name)

    if smoke:
        plan = {0: {"n_facts": 4, "conditions": ["healthy", "interference", "amnesia"]}}
    else:
        plan = {s: {"subsample": cfg["subsample"], "conditions": CONDITION_NAMES}
                for s, cfg in SEED_PLAN.items()}

    results = {}
    if os.path.exists(out_path):
        try:
            results = json.load(open(out_path, encoding="utf-8"))
            print(f"[resume] {len(results.get('rows', []))} prior rows from {out_path}", flush=True)
        except Exception:
            results = {}
    rows = results.get("rows", [])
    done = {(r["condition"], r["seed"]) for r in rows}
    offtopic = offtopic_queries()

    for seed, cfg in plan.items():
        bank = build_bank(tok, n_facts=cfg.get("n_facts"), subsample=cfg.get("subsample"), seed=seed)
        for cond in cfg["conditions"]:
            if (cond, seed) in done:
                print(f"[skip] {cond} seed={seed} already checkpointed", flush=True)
                continue
            t0 = time.time()
            print(f"\n=== {cond} seed={seed} (n_facts={len(bank)}) ===", flush=True)
            kwargs = {}
            if cond == "amnesia":
                kwargs["eta_scale"] = SEED_PLAN.get(seed, SEED_PLAN[0])["amnesia_scale"]
            if cond == "intrusion":
                kwargs["eta_scale"] = SEED_PLAN.get(seed, SEED_PLAN[0])["intrusion_scale"]
            mem = CONDITIONS[cond](model, tok, layer, **kwargs)
            _patch_capture_logits(mem)
            wlog = populate(mem, bank)
            mem.calibrate_gate()
            signals, samples = evaluate(mem, bank, offtopic)
            mem.close()
            row = {"condition": cond, "seed": seed, "n_facts": len(bank), "write_log": wlog,
                   "signals": signals, "samples": samples, "seconds": round(time.time() - t0, 1)}
            rows.append(row)
            done.add((cond, seed))
            results["rows"] = rows
            save_checkpoint(results, out_path)
            print(f"[{cond} seed={seed}] {signals}  ({row['seconds']}s)", flush=True)

    if not any(r["condition"] == "healthy" for r in rows):
        print("[diagnose] no healthy row present -- skipping classifier (smoke ran without it?)",
              flush=True)
        return results

    th = derive_thresholds(rows)
    confusion = {c: {p: 0 for p in CONDITION_NAMES} for c in CONDITION_NAMES}
    for r in rows:
        pred = diagnose(r["signals"], th)
        r["predicted"] = pred
        confusion[r["condition"]][pred] += 1
    results["thresholds"] = th
    results["confusion"] = confusion
    results["worst_samples"] = {c: worst_samples(rows, c) for c in set(r["condition"] for r in rows)}
    save_checkpoint(results, out_path)

    print("\n=== signal table ===", flush=True)
    for r in rows:
        print(f"  {r['condition']:14s} seed={r['seed']}  {r['signals']}  -> predicted={r.get('predicted')}",
              flush=True)
    print("\n=== confusion matrix (rows=true, cols=predicted) ===", flush=True)
    for true_c in CONDITION_NAMES:
        if true_c not in confusion:
            continue
        print(f"  {true_c:14s} {confusion[true_c]}", flush=True)
    print(f"\n=== thresholds ===\n  {th}", flush=True)
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layer", type=int, default=LAYER)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="research/runs/memory_disorders.json")
    a = ap.parse_args()
    out = a.out.replace(".json", "_smoke.json") if a.smoke else a.out
    run(a.smoke, a.model, a.layer, out)
