"""clozn.eval.golden -- the pinned-probe golden fixture behind `clozn test-model` (FRONTIER_BETS §9.3,
"the model's own CI"): run the curated probe sets with GREEDY decoding against a live engine, store the
expected outputs + grades as a fixture, and on a later run diff the fresh outputs against that fixture so
a quant swap, a memory/dial change, or an engine upgrade that silently changes what the model SAYS shows
up as a failed regression check instead of a vibe.

Zero research risk, pure wiring: `run_and_grade` below is a thin wrapper around
`clozn.eval.probes.run_probes` (already greedy -- it POSTs temperature=0.0, see that function's body) and
`clozn.eval.outcome.grade`, both reused completely unmodified. This module owns only the NEW pieces: the
grading pass, the fixture's on-disk shape, and the pairwise diff. Mirrors eval.bench's split (a CLI shell
in clozn/cli/commands/ delegates to an analysis module here) and eval.store's ~/.clozn convention with a
plain atomic write, so `clozn test-model` never re-implements probe running or persistence from scratch.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

from clozn.eval import outcome, probes

_PATH = os.path.join(os.path.expanduser("~/.clozn"), "test_model_golden.json")

# Mirrors eval.bench's `pset` dict (same choices, same meaning) -- kept here too (not imported from bench)
# so this module never has to import the calibration-report machinery it doesn't need.
_PROBE_SETS = {
    "easy": probes.PROBES,
    "hard": probes.HARD_PROBES,
    "arith": probes.ARITH_PROBES,
    "extended": probes.EXTENDED_PROBES,
    "both": probes.PROBES + probes.HARD_PROBES,
    "all": probes.PROBES + probes.HARD_PROBES + probes.ARITH_PROBES + probes.EXTENDED_PROBES,
}


def probe_set(which: str) -> list[dict]:
    """The built-in probe list for `which` (easy/hard/arith/both/all) -- raises KeyError on an unknown
    name, same as a plain dict lookup (the CLI's argparse `choices=` is what actually guards this)."""
    return _PROBE_SETS[which]


def engine_health(base_url: str, timeout: float = 10.0) -> dict:
    """Best-effort GET {base_url}/engine/health -- the live worker's model/quant provenance (model path,
    sha256, architecture, device, gpu_layers, n_ctx; see engine/core/serve/server_main.cpp's /health
    handler for the full shape). Never raises: returns {} if the gateway or engine is unreachable, so a
    `--save` with no health info still records the probe outputs, just without provenance."""
    try:
        req = urllib.request.Request(base_url.rstrip("/") + "/engine/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
        return data.get("engine") or {}
    except Exception:                                             # noqa: BLE001 -- best-effort, never fatal
        return {}


def run_and_grade(base_url: str, which: str = "all", model: str = "clozn") -> list[dict]:
    """Run `which`'s probe set GREEDILY against the live gateway at `base_url` (delegates entirely to
    `probes.run_probes` -- no HTTP of its own) and grade every reply with `outcome.grade`. Returns
    [{q, gold, kind, aliases, reply, correct, error?}]; `correct` is a bool, or None if the item is
    ungradeable (never silently folded into pass/fail -- see outcome.grade's own docstring)."""
    pset = probe_set(which)
    results = probes.run_probes(base_url, pset, model=model)
    out = []
    for r in results:
        correct = outcome.grade(r.get("reply", ""), r["gold"], r["kind"], aliases=r.get("aliases", []))
        rec = {"q": r["q"], "gold": r["gold"], "kind": r["kind"], "aliases": r.get("aliases", []),
              "reply": r.get("reply", ""), "correct": correct}
        if r.get("error"):
            rec["error"] = r["error"]
        out.append(rec)
    return out


def save(rows: list[dict], *, which: str, health: dict | None = None, path: str | None = None) -> str:
    """Persist `rows` (run_and_grade's output) as the golden fixture: the probe set name, model/quant
    provenance (from `health` -- normally `engine_health`'s return), and per-probe expected output + grade
    + timestamp. Atomic write (temp file + os.replace), mirrors eval.store.save. Returns the path written."""
    path = path or _PATH
    health = health or {}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "which": which,
        "saved_ts": time.time(),
        "model": health.get("model"),
        "model_sha256": health.get("model_sha256"),
        "architecture": health.get("architecture"),
        "device": health.get("device"),
        "gpu_layers": health.get("gpu_layers"),
        "n_ctx": health.get("n_ctx"),
        "rows": rows,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


def load(path: str | None = None) -> dict | None:
    """The last saved golden fixture, or None if none exists / the file is unreadable. Never raises."""
    try:
        with open(path or _PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _is_correct(row: dict) -> bool:
    """Grading bool, treating the ungradeable `None` the same as `False` for bucketing purposes -- a probe
    that used to be gradeable-and-right and is now ungradeable is still worth flagging as a regression."""
    return bool(row.get("correct"))


def diff(golden_rows: list[dict], current_rows: list[dict]) -> dict:
    """Pure JSON-in/JSON-out: pair golden probes to current probes BY QUESTION TEXT and bucket each pair
    into exactly one of regression / new_pass / changed / unchanged, plus new/missing for probes that only
    exist on one side (the probe table was edited between saves -- reported, never silently dropped).

      regression  -- was correct, now wrong (or now ungradeable): the thing this command exists to catch
      new_pass    -- was wrong, now correct
      changed     -- same correctness bucket, but the reply TEXT differs
      unchanged   -- same correctness bucket AND same reply text
      new         -- probe has no golden counterpart (added to the probe set since the fixture was saved)
      missing     -- golden probe has no current counterpart (removed / question text changed)

    No I/O, never raises -- testable on canned row lists exactly like commands.test.format_test_report."""
    by_q_golden = {r["q"]: r for r in golden_rows}
    by_q_current = {r["q"]: r for r in current_rows}
    regressions, new_passes, changed, unchanged, new, missing = [], [], [], [], [], []
    for q, cur in by_q_current.items():
        gold = by_q_golden.get(q)
        if gold is None:
            new.append(dict(cur))
            continue
        was_ok, now_ok = _is_correct(gold), _is_correct(cur)
        row = {"q": q, "gold": cur.get("gold"), "was_reply": gold.get("reply"),
              "now_reply": cur.get("reply"), "was_correct": was_ok, "now_correct": now_ok}
        if was_ok and not now_ok:
            regressions.append(row)
        elif not was_ok and now_ok:
            new_passes.append(row)
        elif gold.get("reply") != cur.get("reply"):
            changed.append(row)
        else:
            unchanged.append(row)
    for q, gold in by_q_golden.items():
        if q not in by_q_current:
            missing.append({"q": q, "was_reply": gold.get("reply")})
    return {"regressions": regressions, "new_passes": new_passes, "changed": changed,
           "unchanged": unchanged, "new": new, "missing": missing,
           "n_regressions": len(regressions), "n_new_passes": len(new_passes),
           "n_changed": len(changed), "n_unchanged": len(unchanged),
           "n_new": len(new), "n_missing": len(missing)}
