"""memory_timeline.py -- generate REAL data of a local model learning a person across a long conversation.

Drives a scripted ~20-turn conversation (a consistent persona whose preferences are only ever IMPLIED,
never stated as rules) through the SelfTeach rig, and consolidates at three checkpoints. At each
checkpoint we capture: the rules the model inferred, its self-report of what it has learned (prefix
active, the conversation NOT in its context), and held-out probe checks (baseline vs memory-loaded reply)
that show the learned behaviour actually transfers to fresh prompts. Output: a timeline the memory viz
plays back. This is the honest version of "watch it remember" -- a real frozen 4-bit Qwen-7B, real TTT.

Run (needs the GPU free):  C:/Users/brigi/src/cloze/.venv/Scripts/python.exe research/memory_timeline.py
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from self_teach_server import SelfTeach   # noqa: E402

OUT = os.path.join(HERE, "..", "inspector", "demo", "memory_timeline.json")

PERSONA = ("Robin, an ICU night-shift nurse who decompresses with hard-magic epic fantasy, is vegetarian, "
           "likes short practical answers, and has a cat named Pixel. None of this is ever stated as a "
           "preference -- it only comes through in how Robin talks.")

# every turn implies something; nothing is phrased as an instruction
TURNS = [
    "hey. just crawled off a brutal night shift in the ICU, brain is completely fried",
    "twelve hours on my feet. honestly I just want to get into bed with a book",
    "I'm deep in the Stormlight Archive right now, Sanderson's magic systems are unreal",
    "yeah it's the way the magic has actual rules and costs. I can't stand hand-wavy magic",
    "got any recs in that vein? big epic world, proper hard magic system",
    "already read all of Mistborn, twice, but good call",
    "anyway I should eat. made a big pot of lentil dal earlier, that'll do",
    "been off meat for years, it honestly cuts down on the dinner decisions",
    "can you keep it shortish btw, after a shift I don't have the focus for big walls of text",
    "Pixel is currently sitting directly on the book I'm trying to read, naturally",
    "snowed again overnight, the drive home was miserable",
    "reading is genuinely the only thing that switches my brain off after a bad one",
    "we lost a patient last night. some shifts just stay with you",
    "thanks. I think I'll just read and let Pixel sit on me for a bit",
    "what's a good quick dinner I can throw together half asleep?",
    "perfect, simple is good. I'll do the chickpea thing",
    "back on shift tonight, round we go again",
    "any short story collections? something I can finish between codes",
    "you're honestly better at this than half my coworkers",
    "right, going to nap before my shift. thanks for this",
]
CHECKPOINTS = {7, 14, 20}   # consolidate ("sleep") after these turn counts

# held-out probes the conversation never asked -- the transfer test. The last is NEUTRAL (off-domain):
# a learned preference should stay quiet there (low gate), proving it isn't just always-on.
PROBES = [
    "What should I read this weekend?",
    "Rough day ahead. Any advice?",
    "What should I make for dinner tonight?",
    "Explain how a CPU cache works.",
]


def main():
    app = SelfTeach("Qwen/Qwen2.5-7B-Instruct", m=16)
    timeline = {"persona": PERSONA, "probes": PROBES, "turns": [], "checkpoints": []}

    def dump():
        json.dump(timeline, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    for i, msg in enumerate(TURNS, 1):
        reply = app.say(msg, max_new=110)
        timeline["turns"].append({"n": i, "user": msg, "assistant": reply})
        print(f"[{i:2d}] U: {msg}\n     A: {reply[:90]}", flush=True)
        if i in CHECKPOINTS:
            t0 = time.time()
            if len(app.examples) > 10:          # bound TTT time as the example set grows
                app.examples = app.examples[-10:]
            res = app.consolidate(steps=55, lr=0.03, n_probe=4)
            report = app.what_learned()
            checks = [app.check(p, max_new=85) for p in PROBES]
            timeline["checkpoints"].append({"after_turn": i, "consolidate": res,
                                            "self_report": report, "checks": checks})
            dump()                              # save partial after every checkpoint (crash-safe)
            print(f"  == checkpoint @{i} ({time.time()-t0:.0f}s) rules={res.get('rules')}\n"
                  f"     self-report: {report[:200]}", flush=True)
            for c in checks:
                print(f"     probe {c['prompt'][:28]!r} gate={c.get('gate')}: {str(c.get('gated'))[:80]}", flush=True)
    dump()
    print(f"\nwrote {os.path.normpath(OUT)}: {len(timeline['turns'])} turns, "
          f"{len(timeline['checkpoints'])} checkpoints")


if __name__ == "__main__":
    main()
