"""fetch_np_stats.py -- per-feature stats from Neuronpedia for our SAE: max activation + firing frequency.

maxActApprox lets us normalize a feature's firing against its own peak (relative firing); frac_nonzero is
how often it fires across the corpus -- broad/noise features ("Code comments", "tagged posts") fire on
everything, so a high frac_nonzero flags them for demotion. Together these are the hygiene that turns a
raw full-131k readout into a clean, specific one. Source: same S3 export, features/ folder (203 batches).

Run: C:/Users/brigi/src/clozn/.venv-sae/Scripts/python.exe research/fetch_np_stats.py
"""
import gzip
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = ("https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/"
        "v1/qwen2.5-7b-it/15-resid-post-aa/features/batch-{}.jsonl.gz")
N_BATCHES = 203
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "np_stats_l15.json")


def fetch(i):
    for attempt in range(3):
        try:
            raw = urllib.request.urlopen(BASE.format(i), timeout=120).read()
            rows = {}
            for line in gzip.decompress(raw).decode("utf-8").splitlines():
                o = json.loads(line)
                rows[int(o["index"])] = [round(float(o.get("maxActApprox") or 0.0), 3),
                                         float(o.get("frac_nonzero") or 0.0)]
            return rows
        except Exception:
            if attempt == 2:
                print(f"  batch {i}: FAILED", flush=True)
    return {}


def main():
    stats, done = {}, 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for d in ex.map(fetch, range(N_BATCHES)):
            stats.update(d)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{N_BATCHES} batches, {len(stats)} features", flush=True)
    json.dump(stats, open(OUT, "w"), ensure_ascii=False)
    print(f"\nwrote {os.path.normpath(OUT)}: {len(stats)} features [maxAct, frac_nonzero]")
    # broad-vs-specific sanity: show a few
    for i in (85070, 29355, 0, 100):
        s = stats.get(i)
        print(f"  feature {i}: maxAct={s[0]} frac_nonzero={s[1]:.5f}" if s else f"  feature {i}: (none)")


if __name__ == "__main__":
    main()
