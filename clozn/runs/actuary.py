"""The actuarial journal (FRONTIER_BETS §9.2) — statistics the run journal earns just by existing.

Three read-only analyses over recorded run records (`clozn/runs/store.py`), no model calls, no engine:

  1. calibration  — a proxy reliability curve: bin runs by confidence, measure how often each bin's runs
                    turned out TRUSTED. This is the honest, data-you-already-have answer to "confidence is
                    uncalibrated" — but it is a PROXY (see below), never a ground-truth correctness curve.
  2. drift        — per prompt-class rolling confidence over time; flags a class whose recent runs shifted
                    beyond a band (quant/memory/model degradation caught before you notice it by hand).
  3. failure_signature — learn which trace SHAPES precede bad runs, then score a new run's resemblance.

**What "trusted" / "bad" mean — the load-bearing proxy, stated plainly.**  We have no oracle for whether a
reply was correct. We have BEHAVIORAL proxies the journal records: a run is treated as BAD if it errored,
truncated (finish_reason=="length"), failed a tiny-test, carries a low-confidence flag, OR was superseded
by a child run (a replay/branch/edit of it — the user re-rolled it). Everything else is TRUSTED. This is a
"did the user/tests accept it" signal, not "was it factually right". Every output here is labelled a proxy;
this module never claims calibrated correctness — that would violate the house honesty rule.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


# ---------------------------------------------------------------- organic filter
# A journal is mostly MACHINE traffic: every /receipts click writes one baseline + one arm per fired
# influence + one per redundancy pair (source "replay"); branches, rederives, counterfactuals add more.
# Those are not "did a person accept this answer" events, so scoring them as accepted/rejected poisons
# calibration (validated on the real 309-run journal: 64% "bad" was almost all replay arms). Default the
# actuary to ORGANIC runs — genuine user-facing turns — and report the composition.

_ORGANIC_SOURCES = {"chat", "studio_chat", "say", "engine_chat", "openai_api", "v1_chat", "completions"}
_MACHINE_SOURCES = {"replay", "branch", "receipt", "receipts", "counterfactual", "rederive", "swap_receipt"}


def organic(runs: list[dict]) -> list[dict]:
    """Genuine user-facing turns only: drop known machine sources (replay/receipt arms, branches, ...).
    A run with parent_run_id is a derived arm regardless of source label, so it's dropped too. Unknown
    sources are KEPT (fail open — a new chat surface shouldn't silently vanish from the stats)."""
    out = []
    for r in runs:
        src = (r.get("source") or "").lower()
        if src in _MACHINE_SOURCES:
            continue
        if r.get("parent_run_id"):          # any derived arm, whatever its source label
            continue
        out.append(r)
    return out


# ---------------------------------------------------------------- outcome proxy

def _superseded_ids(runs: list[dict]) -> set:
    """Ids that some later run declared as its parent — i.e. the user re-rolled/edited them."""
    return {r.get("parent_run_id") for r in runs if r.get("parent_run_id")}


def is_bad(run: dict, superseded: set | None = None) -> bool:
    """The behavioral 'the user or the tests did not accept this' proxy (see module docstring)."""
    if run.get("error"):
        return True
    if run.get("finish_reason") == "length":
        return True
    flags = run.get("flags") or []
    if "error" in flags or "truncated" in flags or "low-confidence" in flags:
        return True
    for tt in run.get("tiny_tests") or []:
        if tt.get("pass") is False:
            return True
    if superseded is not None and run.get("id") in superseded:
        return True
    return False


def _confs(run: dict) -> list[float]:
    tr = run.get("trace") or {}
    c = tr.get("confidence")
    if isinstance(c, list) and c:
        return [float(x) for x in c if isinstance(x, (int, float))]
    steps = tr.get("steps")
    if isinstance(steps, list):
        out = []
        for s in steps:
            v = s.get("conf", s.get("confidence"))
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out
    return []


def _mean(xs) -> float | None:
    xs = [x for x in xs if isinstance(x, (int, float))]
    return sum(xs) / len(xs) if xs else None


# ---------------------------------------------------------------- 1. calibration

@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int
    trusted_rate: float | None       # PROXY: fraction of this bin's runs that were TRUSTED
    mean_conf: float | None
    gap: float | None                # mean_conf - trusted_rate (>0 = over-confident, proxy)


@dataclass
class Calibration:
    bins: list[CalibrationBin]
    n_runs: int
    n_scored: int
    ece_proxy: float | None          # expected calibration error against the PROXY (not truth)
    note: str = (
        "PROXY calibration: 'trusted' = the run was not errored/truncated/test-failed/low-conf and "
        "was not re-rolled. This measures acceptance, NOT factual correctness — never label it 'calibrated'."
    )


def calibration(runs: list[dict], n_bins: int = 10) -> Calibration:
    """Bin runs by their mean per-token confidence; report the TRUSTED rate per bin (a proxy reliability
    curve). ECE_proxy = Σ (n_b/N) · |mean_conf_b − trusted_rate_b|. Runs with no trace are skipped."""
    superseded = _superseded_ids(runs)
    scored = []
    for r in runs:
        m = _mean(_confs(r))
        if m is None:
            continue
        scored.append((m, not is_bad(r, superseded)))
    bins: list[CalibrationBin] = []
    total = len(scored)
    ece_num = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        cell = [t for (c, t) in scored if (lo <= c < hi or (i == n_bins - 1 and c == 1.0))]
        confs = [c for (c, _t) in scored if (lo <= c < hi or (i == n_bins - 1 and c == 1.0))]
        n = len(cell)
        if n:
            trusted = sum(1 for t in cell if t) / n
            mc = sum(confs) / n
            gap = mc - trusted
            ece_num += n * abs(gap)
            bins.append(CalibrationBin(lo, hi, n, trusted, mc, gap))
        else:
            bins.append(CalibrationBin(lo, hi, 0, None, None, None))
    ece = (ece_num / total) if total else None
    return Calibration(bins=bins, n_runs=len(runs), n_scored=total, ece_proxy=ece)


# ---------------------------------------------------------------- 2. drift

def _class_key(run: dict) -> str:
    """A cheap, embedder-free prompt-class bucket: source + the first 4 lowercased word-stems of the
    prompt summary. Groups 'what is X' style repeats without a model. Good enough to catch a class whose
    behaviour moved; not a semantic cluster."""
    src = run.get("source") or "?"
    ps = (run.get("prompt_summary") or "").lower()
    words = [w for w in "".join(ch if ch.isalnum() or ch == " " else " " for ch in ps).split() if len(w) > 2]
    return src + "::" + " ".join(words[:4])


@dataclass
class DriftAlarm:
    prompt_class: str
    n_old: int
    n_new: int
    mean_conf_old: float | None
    mean_conf_new: float | None
    delta: float | None
    bad_rate_old: float | None
    bad_rate_new: float | None
    severity: str            # "watch" | "alarm"


def drift(runs: list[dict], split: str | None = None, min_class_n: int = 4, band: float = 0.08) -> list[DriftAlarm]:
    """Split each prompt-class's runs into OLD vs NEW (by created_ts; `split` = an epoch cutoff, else the
    per-class median time) and flag classes whose mean confidence or bad-rate moved beyond `band`. Only
    classes with >= min_class_n runs on BOTH sides are judged (small-n classes are ignored, not guessed)."""
    superseded = _superseded_ids(runs)
    by_class: dict[str, list[dict]] = {}
    for r in runs:
        by_class.setdefault(_class_key(r), []).append(r)
    alarms: list[DriftAlarm] = []
    for cls, rs in by_class.items():
        rs = [r for r in rs if r.get("created_ts") is not None]
        if len(rs) < 2 * min_class_n:
            continue
        rs.sort(key=lambda r: r["created_ts"])
        cut = split if split is not None else rs[len(rs) // 2]["created_ts"]
        old = [r for r in rs if r["created_ts"] < cut]
        new = [r for r in rs if r["created_ts"] >= cut]
        if len(old) < min_class_n or len(new) < min_class_n:
            continue
        mo = _mean([_mean(_confs(r)) for r in old])
        mn = _mean([_mean(_confs(r)) for r in new])
        bo = sum(1 for r in old if is_bad(r, superseded)) / len(old)
        bn = sum(1 for r in new if is_bad(r, superseded)) / len(new)
        dconf = (mn - mo) if (mo is not None and mn is not None) else None
        moved = (dconf is not None and abs(dconf) >= band) or (abs(bn - bo) >= band)
        if not moved:
            continue
        sev = "alarm" if ((dconf is not None and abs(dconf) >= 2 * band) or abs(bn - bo) >= 2 * band) else "watch"
        alarms.append(DriftAlarm(cls, len(old), len(new), mo, mn, dconf, bo, bn, sev))
    alarms.sort(key=lambda a: (a.severity != "alarm", -(abs(a.delta) if a.delta is not None else 0)))
    return alarms


# ---------------------------------------------------------------- 3. failure signature

def _features(run: dict) -> dict:
    """A run's trace SHAPE — the cheap features a bad run tends to share. All model-free."""
    cs = _confs(run)
    tr = run.get("trace") or {}
    ent = tr.get("entropy")
    ent = [float(x) for x in ent if isinstance(x, (int, float))] if isinstance(ent, list) else []
    n = len(cs)
    dips = sum(1 for c in cs if c < 0.4)
    return {
        "mean_conf": _mean(cs),
        "min_conf": min(cs) if cs else None,
        "low_frac": (dips / n) if n else None,
        "max_ent": max(ent) if ent else None,
        "len_tokens": n,
        "trunc": 1.0 if run.get("finish_reason") == "length" else 0.0,
    }


@dataclass
class FailureModel:
    good_mean: dict = field(default_factory=dict)
    bad_mean: dict = field(default_factory=dict)
    good_std: dict = field(default_factory=dict)
    weights: dict = field(default_factory=dict)   # per-feature discriminative weight (|Δmean|/pooled_std)
    n_good: int = 0
    n_bad: int = 0
    note: str = ("resemblance to PAST bad runs (behavioral proxy) — a heuristic early-warning, not a "
                 "correctness predictor. Needs a handful of each class to mean anything; check n_bad.")


_FEATS = ("mean_conf", "min_conf", "low_frac", "max_ent", "len_tokens", "trunc")


def fit_failure_model(runs: list[dict]) -> FailureModel:
    """Learn per-feature good-vs-bad separation (a diagonal, standardized nearest-centroid — deliberately
    simple and inspectable, no sklearn). Features present in too few runs get zero weight."""
    superseded = _superseded_ids(runs)
    good, bad = [], []
    for r in runs:
        f = _features(r)
        (bad if is_bad(r, superseded) else good).append(f)
    m = FailureModel(n_good=len(good), n_bad=len(bad))
    for k in _FEATS:
        gv = [x[k] for x in good if x.get(k) is not None]
        bv = [x[k] for x in bad if x.get(k) is not None]
        if len(gv) < 2 or len(bv) < 2:
            continue
        gm, bm = sum(gv) / len(gv), sum(bv) / len(bv)
        gs = math.sqrt(sum((x - gm) ** 2 for x in gv) / (len(gv) - 1)) or 1e-6
        bs = math.sqrt(sum((x - bm) ** 2 for x in bv) / (len(bv) - 1)) or 1e-6
        pooled = math.sqrt((gs ** 2 + bs ** 2) / 2) or 1e-6
        m.good_mean[k], m.bad_mean[k], m.good_std[k] = gm, bm, gs
        m.weights[k] = abs(bm - gm) / pooled          # Cohen's-d-ish separability
    return m


def failure_score(run: dict, model: FailureModel) -> float:
    """0..1 resemblance of `run` to the learned BAD centroid vs the GOOD centroid, weighted by each
    feature's separability. 0.5 = ambiguous; >0.5 = leans bad. Returns 0.5 if the model is untrained."""
    if not model.weights:
        return 0.5
    f = _features(run)
    num = den = 0.0
    for k, w in model.weights.items():
        v = f.get(k)
        if v is None or k not in model.good_mean:
            continue
        gm, bm = model.good_mean[k], model.bad_mean[k]
        s = model.good_std.get(k, 1e-6) or 1e-6
        dg = abs(v - gm) / s
        db = abs(v - bm) / s
        # closer to bad centroid -> higher; squashed to 0..1 per feature
        lean = dg / (dg + db) if (dg + db) else 0.5
        num += w * lean
        den += w
    return (num / den) if den else 0.5


FAILURE_WARNING_THRESHOLD = 0.65
FAILURE_WARNING_MIN_CLASS = 5
FAILURE_ASSESSMENT_NOTE = (
    "Resemblance to earlier organic runs that the behavioral proxy marked bad (errored, truncated, "
    "test-failed, low-confidence, or re-rolled). This is a trace-shape heuristic, NOT a correctness "
    "predictor."
)


def assess_failure(run: dict, runs: list[dict], *, threshold: float = FAILURE_WARNING_THRESHOLD,
                   min_class_n: int = FAILURE_WARNING_MIN_CLASS) -> dict:
    """Score one run against an honestly PAST-only failure model.

    The journal-level FailureModel is useful for a global report, but it includes the selected run itself
    (and, for an old run, its future). A live warning must not train on the thing it is warning about, so
    this helper refits on organic records strictly older than ``run.created_ts``. When the timestamp is
    unavailable it still excludes the current id and reports ``temporal_cutoff:false``.

    ``available`` means at least two good and two bad past runs produced a weighted model. A warning has a
    higher bar: ``min_class_n`` of EACH class plus score >= ``threshold``. This lets a small journal expose
    its weak evidence without turning two examples into an alarm.
    """
    run = run if isinstance(run, dict) else {}
    organic_runs = organic(runs if isinstance(runs, list) else [])
    rid = run.get("id")
    cutoff = run.get("created_ts")
    temporal = isinstance(cutoff, (int, float)) and math.isfinite(float(cutoff))
    past = []
    for candidate in organic_runs:
        if rid is not None and candidate.get("id") == rid:
            continue
        if temporal:
            ts = candidate.get("created_ts")
            if not isinstance(ts, (int, float)) or not math.isfinite(float(ts)) or float(ts) >= float(cutoff):
                continue
        past.append(candidate)

    model = fit_failure_model(past)
    trained = bool(model.weights)
    score = failure_score(run, model) if trained else None
    eligible = trained and model.n_good >= int(min_class_n) and model.n_bad >= int(min_class_n)
    features = _features(run)
    drivers = []
    for name, weight in model.weights.items():
        value = features.get(name)
        if value is None or name not in model.good_mean or name not in model.bad_mean:
            continue
        scale = model.good_std.get(name, 1e-6) or 1e-6
        dg = abs(value - model.good_mean[name]) / scale
        db = abs(value - model.bad_mean[name]) / scale
        bad_lean = dg / (dg + db) if (dg + db) else 0.5
        drivers.append({"feature": name, "value": value, "good_mean": model.good_mean[name],
                        "bad_mean": model.bad_mean[name], "weight": weight, "bad_lean": bad_lean,
                        "contribution": weight * bad_lean})
    drivers.sort(key=lambda d: -d["contribution"])

    return {
        "available": trained,
        "score": round(score, 4) if score is not None else None,
        "warning": bool(eligible and score is not None and score >= float(threshold)),
        "warning_eligible": bool(eligible),
        "weak_evidence": bool(trained and not eligible),
        "threshold": float(threshold),
        "min_class_n": int(min_class_n),
        "n_good": model.n_good,
        "n_bad": model.n_bad,
        "n_past_organic": len(past),
        "temporal_cutoff": temporal,
        "drivers": drivers[:4],
        "note": FAILURE_ASSESSMENT_NOTE + (" The current run and every timestamped later run are excluded."
          if temporal else " The current id is excluded; without its timestamp, later runs cannot be distinguished."),
    }


# ---------------------------------------------------------------- top-level report

@dataclass
class ActuaryReport:
    calibration: Calibration
    drift: list[DriftAlarm]
    failure_model: FailureModel
    n_runs: int                 # runs analyzed (organic, unless organic_only=False)
    n_total: int                # every run in the journal
    n_organic: int


def analyze(runs: list[dict], *, organic_only: bool = True) -> ActuaryReport:
    """Analyze the journal. `organic_only` (default) restricts to genuine user turns — the honest basis
    for calibration/failure stats, since a raw journal is mostly replay/receipt machine traffic."""
    org = organic(runs)
    use = org if organic_only else runs
    return ActuaryReport(
        calibration=calibration(use),
        drift=drift(use),
        failure_model=fit_failure_model(use),
        n_runs=len(use), n_total=len(runs), n_organic=len(org),
    )


def load_runs() -> list[dict]:
    """Read every full run from the authoritative SQLite journal."""
    from . import store
    return store.iter_runs()


def load_and_analyze() -> ActuaryReport:
    return analyze(load_runs())


def render(report: ActuaryReport) -> str:
    """A terse ASCII text report — the CLI/endpoint face."""
    L = [f"ACTUARIAL JOURNAL -- {report.n_runs} organic runs "
         f"({report.n_organic}/{report.n_total} of journal; rest is replay/receipt machine traffic)\n"]
    cal = report.calibration
    L.append(f"CALIBRATION (proxy, {cal.n_scored}/{report.n_runs} scored, ECE_proxy {cal.ece_proxy:.3f})"
             if cal.ece_proxy is not None else "CALIBRATION (no scored runs)")
    for b in cal.bins:
        if b.n:
            bar = "#" * round((b.trusted_rate or 0) * 20)
            L.append(f"  {b.lo:.1f}-{b.hi:.1f} n={b.n:<3} conf {b.mean_conf:.2f} "
                     f"trusted {b.trusted_rate:.2f} {bar}")
    L.append("  (trusted = accepted/kept, NOT verified-correct -- a PROXY)")
    L.append(f"\nDRIFT -- {len(report.drift)} class(es) moved")
    for a in report.drift[:8]:
        d = f"{a.delta:+.2f}" if a.delta is not None else "  ? "
        L.append(f"  [{a.severity}] {a.prompt_class[:46]:46}  conf {d}  "
                 f"bad {a.bad_rate_old:.2f}->{a.bad_rate_new:.2f}")
    fm = report.failure_model
    L.append(f"\nFAILURE MODEL -- {fm.n_good} good / {fm.n_bad} bad")
    if fm.weights:
        top = sorted(fm.weights.items(), key=lambda kv: -kv[1])[:4]
        L.append("  top discriminators: " + ", ".join(f"{k}(d={w:.2f})" for k, w in top))
    else:
        L.append("  untrained (need >=2 of each class)")
    return "\n".join(L)
