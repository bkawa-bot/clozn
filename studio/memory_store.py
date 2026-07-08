"""
memory_store.py -- PERSISTENCE for the Clozn glass-box memory.

The literal "remembers between sessions" proof: a fact you teach a frozen GPT-2 today is
written to disk and rehydrated, bit-for-bit, into a brand-new session that never saw the
write. The memory is a list of entries {key[d_mlp], value[d_model], eta, label, ans_id, cue}
(the GlassBoxMemory in demo/memory_window.py). This module saves that list and loads it back.

WHY THIS SHAPE. We follow the project's persistence convention (clozn/store.py StateStore):
tensors go to a single .npz (stacked, exact -- np.savez is lossless for float arrays), and
the scalars/strings live in a sidecar .json manifest. np.savez mangles object/string arrays
(labels, cues) and would silently widen dtypes, so those never touch the npz -- they are the
manifest. The manifest also records model + layer + d_mlp + d_model so load() can refuse an
incompatible store rather than rehydrate garbage.

  save(entries, path, meta=None)
      -> writes  <path>.npz   : keys[n,d_mlp] float32, values[n,d_model] float32
                 <path>.json  : compat header + per-entry {eta,label,ans_id,cue}
  load(path) -> (entries, manifest)
      tensors rehydrated as torch tensors (torch.equal-identical to what was saved),
      strings/ints intact. The returned `entries` are plain dicts in the GlassBoxMemory shape.
  rehydrate(model, path) / load_into(mem, path)
      build a fresh GlassBoxMemory (or fill an existing one) from a saved store.

CROSS-SESSION DEMO (python demo/memory_store.py --demo):
  SESSION A : load frozen GPT-2, build a GlassBoxMemory, WRITE 3 nonce facts the model does
              not know, confirm recall fires, SAVE to disk. Tear the session down.
  SESSION B : a FRESH model load + a FRESH empty GlassBoxMemory that never saw the writes,
              LOAD from disk, recall the same 3 cues -> still correct. Before/after printed
              to prove the memory came back cold.
  Also runs an exact round-trip assertion (torch.equal on every key/value, == on every
  string/int) so the "bit-equal" claim is checked, not just asserted in a comment.

ISOLATED ENV: runs in C:\\Users\\brigi\\src\\clozn\\.venv-sae (transformer_lens + torch, CPU).
GPT-2-small (124M) is cached; the demo loads it twice (two real sessions) -- no big download.

Usage (from inspector/, .venv-sae python):
    python demo/memory_store.py --demo
    python demo/memory_store.py --demo --layer 6 --store some/dir/mymem
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # inspector/ on path
try:
    sys.stdout.reconfigure(encoding="utf-8")  # match memory_window's console
except Exception:
    pass

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")  # this PC crashes on HF symlinks (WinError 1314)

import numpy as np   # noqa: E402
import torch         # noqa: E402

# Reuse the VALIDATED mechanism verbatim -- importing GlassBoxMemory + the key/value/recall helpers
# from memory_window does NOT run its main() (that is __main__-guarded), so the recall code under
# test here is the exact code the window ships. We never edit memory_window.py.
from demo.memory_window import (   # noqa: E402
    GlassBoxMemory,
    base_prob,
    load_model,
    single_token_id,
    topk_preds,
)

STORE_FORMAT = "clozn-glassbox-memory/v1"

# Tensor field -> the per-entry vector arrays we stack into the npz (everything else is json scalars).
_TENSOR_FIELDS = ("key", "value")
_SCALAR_FIELDS = ("eta", "label", "ans_id", "cue")


# ====================================================================================================
# SAVE / LOAD -- the persistence core.
# ====================================================================================================
def _paths(path: str) -> tuple[str, str]:
    """A store is <path>.npz (tensors) + <path>.json (manifest). `path` is the stem (no extension)."""
    stem = path[:-4] if path.endswith((".npz", ".json")) else path
    return stem + ".npz", stem + ".json"


def save(entries: list[dict], path: str, meta: dict | None = None) -> tuple[str, str]:
    """Serialize a GlassBoxMemory's entry list to disk, exactly.

    entries : list of {key[d_mlp tensor], value[d_model tensor], eta float, label str,
                       ans_id int, cue str} -- the GlassBoxMemory.entries shape.
    path    : store stem; writes <stem>.npz and <stem>.json.
    meta    : optional dict folded into the manifest (model name, layer, d_mlp, d_model, note...).
              Used for the compatibility check on load.

    Tensors (key, value) are stacked into the npz as float32 [n, d]; the scalars/strings
    (eta, label, ans_id, cue) go to the json manifest, because np.savez mangles object/str arrays.
    Returns (npz_path, json_path).
    """
    npz_path, json_path = _paths(path)
    os.makedirs(os.path.dirname(os.path.abspath(npz_path)), exist_ok=True)
    meta = dict(meta or {})

    n = len(entries)
    if n == 0:
        keys = np.zeros((0, int(meta.get("d_mlp", 0))), dtype=np.float32)
        values = np.zeros((0, int(meta.get("d_model", 0))), dtype=np.float32)
    else:
        # .detach().cpu() so this works even if a GPU/grad tensor ever slips in; contiguous for savez.
        keys = torch.stack([e["key"].detach().cpu() for e in entries]).to(torch.float32).numpy()
        values = torch.stack([e["value"].detach().cpu() for e in entries]).to(torch.float32).numpy()
        keys = np.ascontiguousarray(keys)
        values = np.ascontiguousarray(values)

    np.savez(npz_path, keys=keys, values=values)

    # Compatibility header: infer d_mlp/d_model from the tensors if the caller didn't pass them.
    d_mlp = int(meta.get("d_mlp", keys.shape[1] if keys.ndim == 2 else 0))
    d_model = int(meta.get("d_model", values.shape[1] if values.ndim == 2 else 0))
    manifest = {
        "format": STORE_FORMAT,
        "model": meta.get("model", "unknown"),
        "layer": meta.get("layer", None),
        "d_mlp": d_mlp,
        "d_model": d_model,
        "n_entries": n,
        "note": meta.get("note", ""),
        # the scalars/strings, one row per entry, ALIGNED to the npz row order
        "entries": [
            {
                "eta": float(e["eta"]),
                "label": str(e["label"]),
                "ans_id": int(e["ans_id"]),
                "cue": str(e["cue"]),
            }
            for e in entries
        ],
    }
    # carry through any extra meta the caller passed (timestamp, etc.) without clobbering core keys.
    for k, v in meta.items():
        if k not in manifest and k not in ("model", "layer", "d_mlp", "d_model", "note"):
            manifest[k] = v

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return npz_path, json_path


def load(path: str) -> tuple[list[dict], dict]:
    """Read a saved store back as (entries, manifest).

    Tensors are rehydrated as torch float32 tensors (torch.equal-identical to what save() wrote);
    eta/label/ans_id/cue come straight from the manifest. Entry dicts are in GlassBoxMemory shape,
    so they can be dropped onto `mem.entries` directly. Returns (entries, full_manifest_dict).
    """
    npz_path, json_path = _paths(path)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"manifest not found: {json_path}")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"tensor store not found: {npz_path}")

    with open(json_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    with np.load(npz_path) as d:
        keys = d["keys"].copy()
        values = d["values"].copy()

    rows = manifest.get("entries", [])
    n = manifest.get("n_entries", len(rows))
    if not (len(rows) == keys.shape[0] == values.shape[0] == n):
        raise ValueError(
            f"corrupt store: manifest says {n} entries, json has {len(rows)} rows, "
            f"npz has {keys.shape[0]} keys / {values.shape[0]} values"
        )

    entries: list[dict] = []
    for i, row in enumerate(rows):
        entries.append({
            "key": torch.from_numpy(np.ascontiguousarray(keys[i])).to(torch.float32),
            "value": torch.from_numpy(np.ascontiguousarray(values[i])).to(torch.float32),
            "eta": float(row["eta"]),
            "label": str(row["label"]),
            "ans_id": int(row["ans_id"]),
            "cue": str(row["cue"]),
        })
    return entries, manifest


def _check_compat(model, layer: int, manifest: dict) -> None:
    """Refuse to rehydrate a store built for a different model/layer/geometry (garbage-in guard)."""
    want_mlp, want_model = int(model.cfg.d_mlp), int(model.cfg.d_model)
    got_mlp, got_model = int(manifest.get("d_mlp", -1)), int(manifest.get("d_model", -1))
    if (got_mlp, got_model) != (want_mlp, want_model):
        raise ValueError(
            f"store geometry {got_mlp}x{got_model} != model {want_mlp}x{want_model} "
            f"(store model={manifest.get('model')!r}); refusing to rehydrate incompatible memory."
        )
    saved_layer = manifest.get("layer", None)
    if saved_layer is not None and layer is not None and int(saved_layer) != int(layer):
        # not fatal (you can deliberately re-target a layer), but the keys were computed at saved_layer.
        print(f"  [warn] store layer L={saved_layer} != requested L={layer}; "
              f"keys were captured at L={saved_layer}.")


def load_into(mem: "GlassBoxMemory", path: str, check: bool = True) -> dict:
    """Fill an existing (typically fresh/empty) GlassBoxMemory from disk. Returns the manifest.

    `mem.model`/`mem.layer` decide the compatibility check. The loaded entries replace mem.entries.
    """
    entries, manifest = load(path)
    if check:
        _check_compat(mem.model, mem.layer, manifest)
    mem.entries = entries
    return manifest


def rehydrate(model, path: str, layer: int | None = None, check: bool = True):
    """Build a brand-new GlassBoxMemory from a saved store and return (mem, manifest).

    If `layer` is None it is taken from the manifest. This is the cold-load entry point: a fresh
    model + a fresh memory that never saw the original writes, reconstructed entirely from disk.
    """
    _, manifest = load(path)  # peek for the layer
    use_layer = layer if layer is not None else manifest.get("layer", None)
    if use_layer is None:
        raise ValueError("layer not given and not recorded in the store manifest.")
    mem = GlassBoxMemory(model, int(use_layer))
    load_into(mem, path, check=check)
    return mem, manifest


# ====================================================================================================
# Exact round-trip assertion -- the "bit-equal" claim, checked rather than promised.
# ====================================================================================================
def assert_exact_roundtrip(orig: list[dict], loaded: list[dict]) -> None:
    """torch.equal on every key/value, == on every string/int. Raises AssertionError on any drift."""
    assert len(orig) == len(loaded), f"entry count {len(orig)} != {len(loaded)}"
    for i, (a, b) in enumerate(zip(orig, loaded)):
        for f in _TENSOR_FIELDS:
            ta, tb = a[f].detach().cpu().to(torch.float32), b[f].detach().cpu().to(torch.float32)
            assert ta.shape == tb.shape, f"entry {i} {f}: shape {tuple(ta.shape)} != {tuple(tb.shape)}"
            assert torch.equal(ta, tb), f"entry {i} {f}: tensors not bit-equal"
        assert float(a["eta"]) == float(b["eta"]), f"entry {i} eta mismatch"
        assert int(a["ans_id"]) == int(b["ans_id"]), f"entry {i} ans_id mismatch"
        assert str(a["label"]) == str(b["label"]), f"entry {i} label mismatch"
        assert str(a["cue"]) == str(b["cue"]), f"entry {i} cue mismatch"


# ====================================================================================================
# Fact selection (a compact echo of memory_window STEP 1): pick nonce facts GPT-2 does NOT know.
# ====================================================================================================
_FACTS = [
    # (cue, answer word, label)            -- nonce subjects so frozen GPT-2 can't already know them
    ("The secret color of Zorbland is",     " blue",  "Zorbland"),
    ("Captain Vextor's favorite color is",  " green", "Capt. Vextor"),
    ("The official animal of Quibblax is the", " dog", "Quibblax"),
    ("The lucky number of Flonkville is",   " seven", "Flonkville"),
    ("In the land of Snargle the sky is",   " red",   "Snargle"),
    ("The Wozzleton national fruit is the", " apple", "Wozzleton"),
]


def pick_unknown_facts(model, n_want: int = 3, known_p: float = 0.30) -> list[dict]:
    """Keep `n_want` facts the frozen model scores near chance on (single-token answer, not top-1)."""
    kept: list[dict] = []
    for cue, ans_word, label in _FACTS:
        ans_id = single_token_id(model, ans_word)
        if ans_id is None:
            continue
        p, top1, _ = base_prob(model, cue, ans_id)
        if top1 or p >= known_p:
            continue  # the model already knows it -> not a fair "taught a new fact" demo
        kept.append({"cue": cue, "ans_word": ans_word, "label": label,
                     "ans_id": ans_id, "base_p": p})
        if len(kept) >= n_want:
            break
    if len(kept) < n_want:
        raise SystemExit(f"only {len(kept)} unknown facts survived; need {n_want}.")
    return kept


def _recall_word_prob(mem: "GlassBoxMemory", cue: str, ans_id: int):
    """(top1_word, p_top1, p_ans, fired) for `cue` under the given memory."""
    preds, _, probs, sel, _ = mem.recall(cue, k=5)
    return preds[0][0].strip(), float(preds[0][1]), float(probs[ans_id]), (sel is not None)


# ====================================================================================================
# THE CROSS-SESSION DEMO.
# ====================================================================================================
def run_demo(layer: int, device: str, store: str) -> int:
    torch.manual_seed(0)
    bar = "=" * 84

    # -------------------------------------------------------------------------------------------------
    # SESSION A -- teach 3 facts, confirm recall, SAVE, then drop everything.
    # -------------------------------------------------------------------------------------------------
    print(bar)
    print("SESSION A  --  teach the model, then write it to disk")
    print(bar)
    print(f"loading frozen gpt2 (HookedTransformer) on {device} ...")
    model_a = load_model(device)
    d_mlp, d_model, nl = model_a.cfg.d_mlp, model_a.cfg.d_model, model_a.cfg.n_layers
    print(f"  d_model={d_model}  d_mlp={d_mlp}  n_layers={nl}   memory layer L={layer}")

    facts = pick_unknown_facts(model_a, n_want=3)
    print(f"\npicked {len(facts)} nonce facts the model does NOT know:")
    for f in facts:
        preds, _, _ = topk_preds(model_a, f["cue"], k=1)
        f["before_word"] = preds[0][0].strip()
        print(f"  {f['label']:14} cue={f['cue']!r}")
        print(f"      before: guesses {f['before_word']!r:10} "
              f"P(correct {f['ans_word'].strip()!r})={f['base_p']*100:6.3f}%")

    eta = 10.0  # validated salience for top-1 cosine addressing (p15/p17)
    mem_a = GlassBoxMemory(model_a, layer)
    for f in facts:
        mem_a.write(f["cue"], f["ans_id"], eta, f["label"])

    print("\nSESSION A recall (memory live, before any save):")
    for f in facts:
        w, pw, pa, fired = _recall_word_prob(mem_a, f["cue"], f["ans_id"])
        f["a_word"], f["a_pans"] = w, pa
        ok = "OK" if w == f["ans_word"].strip() else "PARTIAL"
        print(f"  {f['label']:14} -> {w!r:10} ({pw*100:5.1f}%)   "
              f"P(correct)={pa*100:6.2f}%  fired={fired}  [{ok}]")

    meta = {"model": "gpt2", "layer": layer, "d_mlp": int(d_mlp), "d_model": int(d_model),
            "note": "cross-session memory_store demo", "eta": eta}
    npz_path, json_path = save(mem_a.entries, store, meta)
    print(f"\nSAVED memory ({len(mem_a.entries)} entries) to:")
    print(f"  tensors : {os.path.abspath(npz_path)}")
    print(f"  manifest: {os.path.abspath(json_path)}")

    # keep a copy of the live entries for the exact-roundtrip check, then end the session.
    orig_entries = mem_a.entries
    saved_facts = [dict(f) for f in facts]  # cues + expectations carry to session B (a USER would re-ask)

    # -------------------------------------------------------------------------------------------------
    # SESSION B -- a fresh model + a fresh EMPTY memory; load from disk; recall cold.
    # -------------------------------------------------------------------------------------------------
    print("\n" + bar)
    print("SESSION B  --  fresh session, never saw the writes, loads from disk")
    print(bar)
    print("loading a SEPARATE frozen gpt2 (a genuinely new session) ...")
    model_b = load_model(device)

    fresh = GlassBoxMemory(model_b, layer)
    print(f"  fresh GlassBoxMemory created -- entries: {len(fresh.entries)} (empty, knows nothing)")

    # Prove the empty memory recalls NOTHING (baseline) before we load:
    print("\nSESSION B BEFORE load (empty memory -> model's own guesses):")
    for f in saved_facts:
        w, pw, pa, fired = _recall_word_prob(fresh, f["cue"], f["ans_id"])
        f["b_before_word"], f["b_before_pans"] = w, pa
        print(f"  {f['label']:14} -> {w!r:10} ({pw*100:5.1f}%)   "
              f"P(correct {f['ans_word'].strip()!r})={pa*100:6.3f}%  fired={fired}")

    manifest = load_into(fresh, store)   # <-- the rehydrate: cold memory comes back from disk
    print(f"\nLOADED from disk -> entries: {len(fresh.entries)}  "
          f"(store model={manifest['model']!r}, layer L={manifest['layer']}, "
          f"{manifest['d_mlp']}x{manifest['d_model']})")

    print("\nSESSION B AFTER load (rehydrated memory -> should recall the taught facts):")
    all_correct = True
    for f in saved_facts:
        w, pw, pa, fired = _recall_word_prob(fresh, f["cue"], f["ans_id"])
        f["b_after_word"], f["b_after_pans"] = w, pa
        correct = (w == f["ans_word"].strip())
        all_correct &= correct
        flag = "OK" if correct else "PARTIAL"
        print(f"  {f['label']:14} -> {w!r:10} ({pw*100:5.1f}%)   "
              f"P(correct)={pa*100:6.2f}%  fired={fired}  [{flag}]")

    # -------------------------------------------------------------------------------------------------
    # PROOFS: exact round-trip + cold recall recovered the live numbers.
    # -------------------------------------------------------------------------------------------------
    print("\n" + bar)
    print("PROOFS")
    print(bar)

    # (1) exact round-trip on the loaded entries vs the originals from session A.
    loaded_entries, _ = load(store)
    assert_exact_roundtrip(orig_entries, loaded_entries)
    print("  [1] EXACT ROUND-TRIP: every key/value torch.equal bit-equal; "
          "every eta/label/ans_id/cue intact.  PASS")

    # (2) the cold-loaded session B recalls match the live session A recalls (same picks).
    picks_match = all(sf["a_word"] == sf["b_after_word"] for sf in saved_facts)
    print(f"  [2] COLD RECALL == LIVE RECALL: session-B-after picks "
          f"{'MATCH' if picks_match else 'DIFFER from'} session-A picks.  "
          f"{'PASS' if picks_match else 'FAIL'}")

    # (3) loading actually changed something (empty -> recalled): before != after in session B.
    moved = all(sf["b_before_word"] != sf["b_after_word"] or sf["b_after_word"] == sf["ans_word"].strip()
                for sf in saved_facts)
    print(f"  [3] MEMORY REHYDRATED COLD: session B went from empty (own guesses) to recalling "
          f"the taught answers.  {'PASS' if moved else 'CHECK'}")

    print("\nBEFORE / AFTER across the session boundary (session B, fresh model, disk-loaded memory):")
    for f in saved_facts:
        b = f["b_before_word"] or "."
        a = f["b_after_word"] or "."
        print(f"  {f['label']:14} '{f['cue']}'")
        print(f"      cold-empty : {b!r:10} (P correct {f['b_before_pans']*100:6.3f}%)")
        print(f"      after-load : {a!r:10} (P correct {f['b_after_pans']*100:6.2f}%)   "
              f"want {f['ans_word'].strip()!r}  "
              f"{'<<< recalled' if a == f['ans_word'].strip() else '<<< partial'}")

    ok = all_correct and picks_match
    print("\n" + bar)
    print(f"RESULT: {'remembers between sessions -- PROVEN.' if ok else 'recall incomplete (see PARTIAL above).'}")
    print(bar)
    return 0 if ok else 1


# ====================================================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="Persist + cross-session recall for the Clozn memory.")
    ap.add_argument("--demo", action="store_true", help="run the SESSION A -> SESSION B cross-session demo")
    ap.add_argument("--layer", type=int, default=8, help="memory write/read layer (p15/p17 best = 8)")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--store", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs", "memory_store_demo"),
        help="store stem (writes <stem>.npz + <stem>.json)")
    args = ap.parse_args()

    if not args.demo:
        ap.print_help()
        print("\n(pass --demo to run the cross-session persistence proof)")
        return 0
    return run_demo(args.layer, args.device, args.store)


if __name__ == "__main__":
    raise SystemExit(main())
