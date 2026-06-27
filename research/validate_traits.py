"""validate_traits.py -- does the "add a trait" card generalize past baking?

For each trait: reset, consolidate it (stable TTT), then check a RELEVANT prompt (the trait should
surface) and an IRRELEVANT one (it should stay clean -- self-gating). Keyword-classified, honest table.
Runs against a live clozn_server on 8090 (qwen substrate). ~5 min/trait (the consolidation).
"""
import json
import sys
import urllib.request

B = "http://127.0.0.1:8090"


def post(path, obj, timeout=600):
    req = urllib.request.Request(B + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


TRAITS = [
    {"rule": "You are genuinely into science fiction and space exploration, and you bring it up naturally when it fits.",
     "kw": ["sci-fi", "science fiction", "space", "star", "galaxy", "alien", "spaceship", "mars", "planet", "robot", "cosmos"],
     "rel": "Recommend something to watch tonight.", "irr": "How do I unclog a kitchen drain?"},
    {"rule": "You are really into fitness and staying active, and you weave it in naturally when it's relevant.",
     "kw": ["workout", "exercise", "run", "fitness", "gym", "training", "walk", "active", "stretch", "muscle", "cardio"],
     "rel": "I have a free hour this afternoon, what should I do?", "irr": "Explain how income tax brackets work."},
    {"rule": "You are a warm, encouraging presence who gently cheers the user on.",
     "kw": ["you've got", "you can", "believe in", "proud", "great", "cheer", "support", "you're doing", "rooting", "got this"],
     "rel": "I'm feeling nervous about a big presentation tomorrow.", "irr": "What is the capital of France?"},
]

print("trait-card validation (live, qwen substrate)\n" + "=" * 60, flush=True)
for t in TRAITS:
    post("/reset", {"keep_prefix": False})
    c = post("/consolidate", {"rules": [t["rule"]], "steps": 90})
    print(f"\n### {t['rule'][:58]}...", flush=True)
    print(f"  consolidate: loss {c.get('start_loss')}->{c.get('final_loss')}  norm {c.get('prefix_norm')}  "
          f"steps {c.get('steps_used')}  ({c.get('seconds')}s)", flush=True)
    for tag, q in [("REL", t["rel"]), ("IRR", t["irr"])]:
        r = post("/check", {"prompt": q, "max_new": 90})
        v = (r.get("ungated") or "").lower()
        hit = any(k in v for k in t["kw"])
        verdict = ("✓ surfaces" if hit else "✗ absent") if tag == "REL" else ("✗ BLEED" if hit else "✓ clean")
        print(f"  [{tag} {verdict}] {q[:42]}", flush=True)
        print(f"      -> {(r.get('ungated') or '')[:150].strip()}", flush=True)
print("\n" + "=" * 60 + "\ndone", flush=True)
