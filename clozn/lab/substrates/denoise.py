"""Dream denoise trace helper."""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine" / "lab"))

from cloze_lab.generate import GenerateConfig, generate                            # noqa: E402
from cloze_lab.scheduler.events import GenStarted, TokensCommitted, TokensRevised  # noqa: E402
from cloze_lab.scheduler.policies import RemaskLowConf                             # noqa: E402

LOCK = threading.Lock()


def trace_for(adapter, prompt, max_new=48, steps=28):
    with LOCK:
        ids = adapter.encode(prompt, chat=True)
        cfg = GenerateConfig(max_new=max_new, steps=steps, temperature=0.0, seed=0, block_len=0)
        # the remask_lowconf reviser surfaces the "model changes its mind" behaviour: a committed token
        # whose recomputed confidence falls below tau is re-masked and re-predicted later (TokensRevised).
        res = generate(adapter, ids, cfg, reviser=RemaskLowConf(tau_revise=0.55, max_revisions=1))
        n_prompt = board_len = None
        by_pass: dict[int, dict] = {}
        for e in res.events:
            if isinstance(e, GenStarted):
                n_prompt, board_len = e.prompt_tokens, e.prompt_tokens + e.max_new
            elif isinstance(e, TokensCommitted):
                d = by_pass.setdefault(e.t, {"items": [], "revised": []})
                d["items"] += [{"pos": int(it.pos), "piece": adapter.decode([int(it.id)]), "conf": round(float(it.conf), 3)}
                               for it in e.items]
            elif isinstance(e, TokensRevised):
                d = by_pass.setdefault(e.t, {"items": [], "revised": []})
                d["revised"] += [{"pos": int(it.pos), "piece": adapter.decode([int(it.old)]), "conf": round(float(it.conf), 3)}
                                 for it in e.items]
        passes = []
        for pi, t in enumerate(sorted(by_pass)):
            d = by_pass[t]
            if d["items"] or d["revised"]:
                passes.append({"pass": pi, "items": d["items"], "revised": d["revised"]})
        mask_id, eos = adapter.config.mask_token_id, adapter.config.eos_token_id   # clean final text:
        clean = []                                                                # truncate at EOS, drop holes
        for t in (int(x) for x in res.board[n_prompt:board_len]):
            if eos is not None and t == eos:
                break
            if t != mask_id:
                clean.append(t)
        return {"model": "Dream-v0-Instruct-7B", "prompt": prompt,
                "prompt_text": adapter.decode([int(i) for i in ids]),
                "n_prompt": n_prompt, "board_len": board_len, "steps": steps,
                "final_text": adapter.decode(clean), "passes": passes}
