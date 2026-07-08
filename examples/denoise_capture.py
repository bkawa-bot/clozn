"""denoise_capture.py -- run a real diffusion LM (Dream-7B) through the cloze lab and capture the denoise
trace for a 'watch it denoise' viz: a board of masked slots that fill in PARALLEL over passes, each token
landing with a confidence. This is the diffusion counterpart to watch-it-think -- for a dLLM the
"thinking" is literally visible as the text crystallising out of the mask.

Reconstructs the board purely from the event spine (gen_started + tokens_committed, DESIGN §5.1).
Output: inspector/demo/denoise_trace.json

Run (GPU free):  PYTHONPATH=engine/lab C:/Users/brigi/src/cloze/.venv/Scripts/python.exe research/denoise_capture.py
"""
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine", "lab"))

from cloze_lab.cli import build_adapter                                    # noqa: E402
from cloze_lab.generate import GenerateConfig, generate                    # noqa: E402
from cloze_lab.scheduler.events import GenStarted, TokensCommitted         # noqa: E402

OUT = os.path.join(HERE, "..", "inspector", "demo", "denoise_trace.json")
PROMPT = "What is the largest planet in the solar system? Answer in one short sentence."


def capture(model="dream", prompt=PROMPT, max_new=32, steps=20):
    adapter = build_adapter(model, device="cuda", quant="nf4")
    ids = adapter.encode(prompt, chat=True)
    cfg = GenerateConfig(max_new=max_new, steps=steps, temperature=0.0, seed=0, block_len=0)
    res = generate(adapter, ids, cfg)

    n_prompt = board_len = None
    passes, pi = [], 0
    for e in res.events:
        if isinstance(e, GenStarted):
            n_prompt, board_len = e.prompt_tokens, e.prompt_tokens + e.max_new
        elif isinstance(e, TokensCommitted):
            items = [{"pos": int(it.pos), "piece": adapter.decode([int(it.id)]), "conf": round(float(it.conf), 3)}
                     for it in e.items]
            if items:
                passes.append({"pass": pi, "items": items})
                pi += 1
    return {"model": "Dream-v0-Instruct-7B", "prompt": prompt,
            "prompt_text": adapter.decode([int(i) for i in ids]),
            "n_prompt": n_prompt, "board_len": board_len, "steps": steps,
            "final_text": res.text, "passes": passes}


def main():
    trace = capture()
    json.dump(trace, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"wrote {os.path.normpath(OUT)}: {len(trace['passes'])} passes, "
          f"board_len={trace['board_len']}, final: {trace['final_text'][:80]!r}")


if __name__ == "__main__":
    main()
