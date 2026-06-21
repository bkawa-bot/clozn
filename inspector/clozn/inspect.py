"""
clozn.inspect — the Phase-1 Inspector: one command, one dashboard.

Runs the three white-box views over a real recurrent model and stacks them into a single page:
  WATCH    — the logit-lens thought stream + per-layer write-intensity heatmap for your prompt
  PROBE    — is a concept (sentiment) linearly decodable from the state, and is it causal?
  PERSIST  — save the model's memory to disk, rehydrate it cold, watch it recall

Usage:  python -m clozn.inspect "The capital of France is Paris. Two plus two equals"
"""
from __future__ import annotations

import os
import sys

from .probes import probe_and_verify
from .sources.hf_rwkv import RwkvStateSource
from .store import StateStore
from .viz import _card_svg, _film_svg, _probe_svg, render_dashboard

FACT = "Remember this fact: the password is Maiko."
RECALL_PROBE = " The password is"


def _greedy(src, n=4):
    out = []
    for _ in range(n):
        tid = int(src._last_logits.argmax())
        src.step(tid)
        out.append(tid)
    return src.tok.decode(out)


def inspect(prompt: str, *, out: str | None = None, do_probe: bool = True,
            do_persist: bool = True) -> str:
    src = RwkvStateSource()
    panels: list[tuple[str, str]] = []

    # WATCH — capture the stream for the prompt before the probe starts resetting the source
    steps = src.feed(prompt)
    panels.append(("WATCH · logit-lens thought stream + per-layer write intensity",
                   _film_svg(steps, "att_num", "Clozn · Watch", repr(prompt))))

    # PROBE + VERIFY — a property of the model, independent of the prompt
    if do_probe:
        r = probe_and_verify(src, name="sentiment")
        panels.append(("PROBE + VERIFY · is ‘sentiment’ decodable from the memory, and causal?",
                       _probe_svg(r.alphas, r.scores, r.decodability, r.verify, "sentiment",
                                  "Clozn · Probe + Verify", "RWKV-4-169m · att_num")))

    # PERSIST — save memory, rehydrate cold, compare recall vs a cold model
    if do_persist:
        store = StateStore(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                        "runs", "store"))
        src.reset(); src.feed(FACT)
        live = src.get_state()
        store.save("inspect_mem", src, note=FACT)
        src.reset(); store.into(src, "inspect_mem")
        maxdiff = max(float((abs(src.get_state()[k] - live[k])).max()) for k in live)
        for tid in src.encode(RECALL_PROBE):
            src.step(tid)
        warm = RECALL_PROBE + _greedy(src)
        src.reset()
        for tid in src.encode(RECALL_PROBE):
            src.step(tid)
        cold = RECALL_PROBE + _greedy(src)
        panels.append(("PERSIST · save memory to disk, rehydrate in a cold session",
                       _card_svg(RECALL_PROBE,
                                 [("rehydrated from disk — never saw the fact this session", warm, "#7ee0d0"),
                                  ("cold model — empty memory, same prompt", cold, "#c9a0ff")],
                                 f"rehydrate {maxdiff:.0e}", "Clozn · Persisted Memory",
                                 f"fact saved last session: {FACT!r}")))

    html = render_dashboard(panels, title="Clozn · Inspector", subtitle=f"reading {prompt!r}")
    if out is None:
        out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "runs", "inspect.html")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    prompt = sys.argv[1] if len(sys.argv) > 1 else "The capital of France is Paris. Two plus two equals"
    print(f"inspecting {prompt!r} ...")
    out = inspect(prompt)
    print("wrote", out)


if __name__ == "__main__":
    main()
