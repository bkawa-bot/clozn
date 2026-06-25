"""fetch_np_labels.py -- download Neuronpedia's auto-interp labels for OUR SAE and build an index->label map.

Our SAE's config carries neuronpedia_id = "qwen2.5-7b-it/15-resid-post-aa", so the feature indices line
up exactly. Neuronpedia's bulk explanation export moved to S3 (203 gzipped jsonl batches). We pull them
all, keep {index: description}, drop the big per-row embedding, and save a compact json. ~131k labels.

Run: C:/Users/brigi/src/clozn/.venv-sae/Scripts/python.exe research/fetch_np_labels.py
"""
import gzip
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = ("https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/"
        "v1/qwen2.5-7b-it/15-resid-post-aa/explanations/batch-{}.jsonl.gz")
N_BATCHES = 203
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "np_labels_l15.json")


def fetch(i):
    for attempt in range(3):
        try:
            raw = urllib.request.urlopen(BASE.format(i), timeout=90).read()
            rows = {}
            for line in gzip.decompress(raw).decode("utf-8").splitlines():
                o = json.loads(line)
                idx = int(o["index"])
                desc = (o.get("description") or "").strip()
                # prefer a max-activation explanation; don't overwrite a good one with an empty
                if desc and (idx not in rows):
                    rows[idx] = desc
            return rows
        except Exception as e:
            if attempt == 2:
                print(f"  batch {i}: FAILED {type(e).__name__}", flush=True)
    return {}


def main():
    labels = {}
    done = 0
    with ThreadPoolExecutor(max_workers=16) as ex:
        for d in ex.map(fetch, range(N_BATCHES)):
            labels.update(d)
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{N_BATCHES} batches, {len(labels)} labels so far", flush=True)
    json.dump(labels, open(OUT, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"\nwrote {os.path.normpath(OUT)}: {len(labels)} labels "
          f"({len(labels)/131072*100:.1f}% of 131072 features)")
    # spot check
    for i in (0, 1, 100, 49310, 29355, 85070):
        print(f"  feature {i}: {labels.get(i, '(none)')!r}")


if __name__ == "__main__":
    main()
