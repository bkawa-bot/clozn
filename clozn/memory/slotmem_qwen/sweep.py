"""Smoke runs and capacity sweeps for the Qwen slot-memory store."""
from __future__ import annotations

import json
import os
import time

from .facts import KNOWN, MULTI, PARA, SINGLE, make_facts
from .store import INJECT_FRAC, SlotMem


def top1_hits(mem: SlotMem, facts: list[dict], gated=False, entries=None):
    hits, p_ans = 0, 0.0
    for f in facts:
        r = mem.read(f["cue"], gated=gated, entries=entries)
        aid = f["ans_ids"][0]
        hits += int(int(r["dist"].argmax()) == aid)
        p_ans += float(r["dist"][aid])
    n = len(facts)
    return {"top1": round(hits / n, 3), "p_ans": round(p_ans / n, 4)}


def run(model_name: str, layer: int, out_path: str, smoke=False):
    t0 = time.time()
    mem = SlotMem(model_name, layer)
    res = {"model": model_name, "layer": layer, "eta_frac": INJECT_FRAC, "phases": {}}
    single = SINGLE[:4] if smoke else SINGLE
    multi = MULTI[:2] if smoke else MULTI

    bank = []
    for cue, ans in single + multi:
        ids = mem.tok.encode(ans, add_special_tokens=False)
        bank.append({"cue": cue, "answer": ans, "ans_ids": ids, "multi": len(ids) > 1})
    n_multi = sum(f["multi"] for f in bank)
    base = {"top1": 0, "p_ans": 0.0}
    for f in bank:
        d = mem._next_dist(f["cue"])
        base["top1"] += int(int(d.argmax()) == f["ans_ids"][0])
        base["p_ans"] += float(d[f["ans_ids"][0]])
    res["phases"]["baseline"] = {
        "n": len(bank),
        "n_multi_tok": n_multi,
        "top1": round(base["top1"] / len(bank), 3),
        "p_ans": round(base["p_ans"] / len(bank), 4),
    }
    print(
        f"[0 baseline] n={len(bank)} (multi-tok {n_multi}) "
        f"top1={res['phases']['baseline']['top1']} p_ans={res['phases']['baseline']['p_ans']}",
        flush=True,
    )

    wlog = {"written": 0, "skipped_known": 0, "forced": 0, "details": []}
    for cue, ans in KNOWN:
        r = mem.write(cue, ans, gate=True)
        wlog["details"].append({"cue": cue, **r, "expected": "skip"})
        wlog["skipped_known"] += int(not r["written"])
    mem.entries = []
    for f in bank:
        r = mem.write(f["cue"], f["answer"], gate=True)
        wlog["details"].append({"cue": f["cue"], **r, "expected": "write"})
        wlog["written"] += int(r["written"])
        if not r["written"]:
            mem.write(f["cue"], f["answer"], gate=False)
            wlog["forced"] += 1
    res["phases"]["write_gate"] = {
        "nonce_written": wlog["written"],
        "nonce_total": len(bank),
        "nonce_forced": wlog["forced"],
        "known_skipped": wlog["skipped_known"],
        "known_total": len(KNOWN),
        "details": wlog["details"],
    }
    print(
        f"[1 write-gate] nonce written {wlog['written']}/{len(bank)} (forced {wlog['forced']}); "
        f"known skipped {wlog['skipped_known']}/{len(KNOWN)}",
        flush=True,
    )
    mem.calibrate_gate()

    rec = top1_hits(mem, bank)
    shuf = [dict(e, key=mem.entries[(i + 1) % len(mem.entries)]["key"]) for i, e in enumerate(mem.entries)]
    nul = top1_hits(mem, bank, entries=shuf)
    res["phases"]["recall"] = {
        "memory": rec,
        "shuffled_null": nul,
        "baseline": res["phases"]["baseline"]["top1"],
    }
    print(f"[2 recall] top1={rec['top1']} p_ans={rec['p_ans']}  ||  shuffled-null top1={nul['top1']}", flush=True)

    sub = bank[:4]
    off_hits, on_hits, n_off = 0, 0, 0
    for i, fi in enumerate(sub):
        for j, _fj in enumerate(sub):
            r = mem.read(fi["cue"], entries=[mem.entries[j]])
            hit = int(int(r["dist"].argmax()) == fi["ans_ids"][0])
            if i == j:
                on_hits += hit
            else:
                off_hits += hit
                n_off += 1
    res["phases"]["specificity"] = {
        "on_target_top1": round(on_hits / len(sub), 3),
        "off_target_top1": round(off_hits / max(1, n_off), 3),
    }
    print(f"[3 specificity] on={on_hits}/{len(sub)} off={off_hits}/{n_off} (off must be ~0)", flush=True)

    victim = 0
    keep = [e for k, e in enumerate(mem.entries) if k != victim]
    before_others = top1_hits(mem, bank[1:])
    after_victim = mem.read(bank[victim]["cue"], entries=keep)
    after_others = top1_hits(mem, bank[1:], entries=keep)
    res["phases"]["delete"] = {
        "victim_top1_after": int(int(after_victim["dist"].argmax()) == bank[victim]["ans_ids"][0]),
        "others_before": before_others,
        "others_after": after_others,
        "others_identical": before_others == after_others,
    }
    print(
        f"[4 delete] victim recalls after delete: {res['phases']['delete']['victim_top1_after']} "
        f"(want 0); others identical: {res['phases']['delete']['others_identical']}",
        flush=True,
    )

    paras = [(c, p, f) for f in bank if f["cue"] in PARA for c in [f["cue"]] for p in PARA[f["cue"]]]
    pg = {"n": 0, "ungated": {"right": 0, "wrong_fact": 0}, "gated": {"right": 0, "wrong_fact": 0, "abstain": 0}}
    for _cue, para, f in paras:
        pg["n"] += 1
        for mode in ("ungated", "gated"):
            r = mem.read(para, gated=(mode == "gated"))
            if r["abstained"]:
                pg[mode]["abstain"] = pg[mode].get("abstain", 0) + 1
                continue
            top = int(r["dist"].argmax())
            if top == f["ans_ids"][0]:
                pg[mode]["right"] += 1
            elif any(top == e["ans_ids"][0] for e in mem.entries):
                pg[mode]["wrong_fact"] += 1
    res["phases"]["paraphrase_gate"] = pg
    print(
        f"[5 paraphrase] n={pg['n']} ungated right={pg['ungated']['right']} "
        f"wrongfact={pg['ungated']['wrong_fact']} || gated right={pg['gated']['right']} "
        f"wrongfact={pg['gated']['wrong_fact']} abstain={pg['gated']['abstain']}",
        flush=True,
    )

    em = {"single": {"n": 0, "ok": 0}, "multi": {"n": 0, "ok": 0}, "samples": []}
    for f in bank:
        g = mem.emit(f["cue"])
        kind = "multi" if f["multi"] else "single"
        ok = f["answer"].strip().lower() in g.lower()
        em[kind]["n"] += 1
        em[kind]["ok"] += int(ok)
        if len(em["samples"]) < 8:
            em["samples"].append({"cue": f["cue"], "want": f["answer"].strip(), "got": g.strip()[:60], "ok": ok})
    res["phases"]["emission"] = em
    print(f"[6 emission] single {em['single']['ok']}/{em['single']['n']}  multi {em['multi']['ok']}/{em['multi']['n']}", flush=True)

    res["seconds"] = round(time.time() - t0, 1)
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\nsaved -> {out_path}  ({res['seconds']}s)", flush=True)
    mem.close()
    return res


def sweep(model_name: str, layer: int, out_path: str, sizes=(10, 25, 50, 100, 200)):
    mem = SlotMem(model_name, layer)
    res = {"model": model_name, "layer": layer, "eta_frac": INJECT_FRAC, "sweep": []}
    eval_cap = 40
    for n_facts in sizes:
        facts = make_facts(mem.tok, n_facts)
        mem.entries = []
        for f in facts:
            mem.write(f["cue"], f["answer"], gate=False)
        idxs = list(range(n_facts)) if n_facts <= eval_cap else [int(i * n_facts / eval_cap) for i in range(eval_cap)]
        sel = expr = 0
        shuf = [dict(e, key=mem.entries[(k + 1) % n_facts]["key"]) for k, e in enumerate(mem.entries)]
        null_expr = 0
        for i in idxs:
            f = facts[i]
            r = mem.read(f["cue"])
            sel += int(r["hit"] == i)
            expr += int(int(r["dist"].argmax()) == f["ans_ids"][0])
            rn = mem.read(f["cue"], entries=shuf)
            null_expr += int(int(rn["dist"].argmax()) == f["ans_ids"][0])
        row = {
            "N": n_facts,
            "n_eval": len(idxs),
            "select": round(sel / len(idxs), 3),
            "express": round(expr / len(idxs), 3),
            "shuffled_null_express": round(null_expr / len(idxs), 3),
        }
        res["sweep"].append(row)
        print(
            f"[N={n_facts:>3}] select={row['select']}  express={row['express']}  "
            f"null={row['shuffled_null_express']}",
            flush=True,
        )
    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"saved -> {out_path}", flush=True)
    mem.close()
    return res
