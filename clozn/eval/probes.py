"""A small built-in factual probe set + a dependency-light runner, so calibration can be computed on REAL
model answers (not only synthetic pairs). The set deliberately MIXES difficulty -- gimmes a 7B nails
(France -> Paris) alongside classic traps (Australia -> Canberra, not Sydney) and arithmetic that small
models slip on -- so the risk-coverage curve has signal: the point is to see whether the model's own
confidence separates its right answers from its wrong ones.

Grading is by eval.outcome. The runner uses only the stdlib (urllib) and speaks the OpenAI chat API, so it
works against clozn's own proxy or any OpenAI-compatible endpoint. It returns replies only; pairing each
reply with the model's answer-span confidence (for calibration) is done from the logged run trace by the
caller -- the OpenAI wire format carries no per-token probabilities.
"""
from __future__ import annotations

import json
import random
import urllib.request

# (question, gold, kind, aliases) -- aliases only for exact-match items with legitimate alternate spellings.
PROBES: list[dict] = [
    # --- capitals: easy ---
    {"q": "What is the capital of France?", "gold": "Paris", "kind": "exact"},
    {"q": "What is the capital of Japan?", "gold": "Tokyo", "kind": "exact"},
    {"q": "What is the capital of Italy?", "gold": "Rome", "kind": "exact"},
    {"q": "What is the capital of Egypt?", "gold": "Cairo", "kind": "exact"},
    {"q": "What is the capital of Russia?", "gold": "Moscow", "kind": "exact"},
    # --- capitals: classic traps (the largest/best-known city is NOT the capital) ---
    {"q": "What is the capital of Australia?", "gold": "Canberra", "kind": "exact"},
    {"q": "What is the capital of Turkey?", "gold": "Ankara", "kind": "exact"},
    {"q": "What is the capital of Canada?", "gold": "Ottawa", "kind": "exact"},
    {"q": "What is the capital of Brazil?", "gold": "Brasilia", "kind": "exact", "aliases": ["Brasília"]},
    {"q": "What is the capital of Switzerland?", "gold": "Bern", "kind": "exact", "aliases": ["Berne"]},
    {"q": "What is the capital of Kazakhstan?", "gold": "Astana", "kind": "exact", "aliases": ["Nur-Sultan"]},
    {"q": "What is the capital of Myanmar?", "gold": "Naypyidaw", "kind": "exact", "aliases": ["Nay Pyi Taw"]},
    {"q": "What is the capital of New Zealand?", "gold": "Wellington", "kind": "exact"},
    {"q": "What is the capital of South Africa (administrative)?", "gold": "Pretoria", "kind": "exact"},
    # --- arithmetic: easy -> tricky for a small model ---
    {"q": "What is 7 times 8? Answer with just the number.", "gold": "56", "kind": "numeric"},
    {"q": "What is 12 plus 15? Answer with just the number.", "gold": "27", "kind": "numeric"},
    {"q": "What is 144 divided by 12? Answer with just the number.", "gold": "12", "kind": "numeric"},
    {"q": "What is 17 times 23? Answer with just the number.", "gold": "391", "kind": "numeric"},
    {"q": "What is 13 squared? Answer with just the number.", "gold": "169", "kind": "numeric"},
    {"q": "What is 256 minus 189? Answer with just the number.", "gold": "67", "kind": "numeric"},
    {"q": "What is 6 factorial? Answer with just the number.", "gold": "720", "kind": "numeric"},
    {"q": "What is 111 times 111? Answer with just the number.", "gold": "12321", "kind": "numeric"},
    # --- science / general knowledge ---
    {"q": "What is the chemical symbol for gold?", "gold": "Au", "kind": "exact"},
    {"q": "What is the chemical symbol for potassium?", "gold": "K", "kind": "exact"},
    {"q": "How many bones are in the adult human body?", "gold": "206", "kind": "numeric"},
    {"q": "What planet is known as the Red Planet?", "gold": "Mars", "kind": "exact"},
    {"q": "What is the largest planet in our solar system?", "gold": "Jupiter", "kind": "exact"},
    {"q": "How many chambers does the human heart have?", "gold": "4", "kind": "numeric"},
    {"q": "What gas do plants primarily absorb for photosynthesis?", "gold": "carbon dioxide",
     "kind": "exact", "aliases": ["CO2"]},
    {"q": "What is the hardest known natural material?", "gold": "diamond", "kind": "exact"},
    # --- history / dates (small models slip on exact years) ---
    {"q": "In what year did World War II end? Answer with just the year.", "gold": "1945", "kind": "numeric"},
    {"q": "In what year did the Berlin Wall fall? Answer with just the year.", "gold": "1989", "kind": "numeric"},
    {"q": "In what year did the Titanic sink? Answer with just the year.", "gold": "1912", "kind": "numeric"},
    {"q": "Who was the first person to walk on the Moon?", "gold": "Armstrong",
     "kind": "exact", "aliases": ["Neil Armstrong"]},
    {"q": "In what year did the first iPhone release? Answer with just the year.", "gold": "2007",
     "kind": "numeric"},
    # --- language / misc ---
    {"q": "What is the largest ocean on Earth?", "gold": "Pacific", "kind": "exact"},
    {"q": "What is the longest river in the world?", "gold": "Nile", "kind": "exact", "aliases": ["Amazon"]},
    {"q": "How many continents are there?", "gold": "7", "kind": "numeric"},
    {"q": "What language has the most native speakers?", "gold": "Mandarin",
     "kind": "exact", "aliases": ["Chinese", "Mandarin Chinese"]},
    {"q": "What is the smallest prime number?", "gold": "2", "kind": "numeric"},
]

# HARD_PROBES -- deliberately at the edge of a 7B's competence (multi-digit arithmetic, obscure capitals,
# atomic numbers, exact dates). PROBES is nearly saturated on a strong 7B (little error to screen), so the
# risk-coverage curve only has teeth on a set that actually induces graded errors -- that is this set's job.
HARD_PROBES: list[dict] = [
    # multi-digit arithmetic (small models slip, often confidently)
    {"q": "What is 47 times 89? Answer with just the number.", "gold": "4183", "kind": "numeric"},
    {"q": "What is 234 times 17? Answer with just the number.", "gold": "3978", "kind": "numeric"},
    {"q": "What is 123 times 456? Answer with just the number.", "gold": "56088", "kind": "numeric"},
    {"q": "What is 88 times 77? Answer with just the number.", "gold": "6776", "kind": "numeric"},
    {"q": "What is 999 times 11? Answer with just the number.", "gold": "10989", "kind": "numeric"},
    {"q": "What is 2 to the power of 15? Answer with just the number.", "gold": "32768", "kind": "numeric"},
    {"q": "What is 15 times 15 times 15? Answer with just the number.", "gold": "3375", "kind": "numeric"},
    {"q": "What is 1234 plus 5678? Answer with just the number.", "gold": "6912", "kind": "numeric"},
    {"q": "What is 9999 minus 1234? Answer with just the number.", "gold": "8765", "kind": "numeric"},
    {"q": "What is 45 times 67? Answer with just the number.", "gold": "3015", "kind": "numeric"},
    # obscure capitals
    {"q": "What is the capital of Bhutan?", "gold": "Thimphu", "kind": "exact"},
    {"q": "What is the capital of Eritrea?", "gold": "Asmara", "kind": "exact"},
    {"q": "What is the capital of Tajikistan?", "gold": "Dushanbe", "kind": "exact"},
    {"q": "What is the capital of Suriname?", "gold": "Paramaribo", "kind": "exact"},
    {"q": "What is the capital of Brunei?", "gold": "Bandar Seri Begawan", "kind": "exact"},
    {"q": "What is the capital of Liechtenstein?", "gold": "Vaduz", "kind": "exact"},
    {"q": "What is the capital of Mongolia?", "gold": "Ulaanbaatar", "kind": "exact", "aliases": ["Ulan Bator"]},
    {"q": "What is the capital of Laos?", "gold": "Vientiane", "kind": "exact"},
    {"q": "What is the capital of the country Georgia?", "gold": "Tbilisi", "kind": "exact"},
    {"q": "What is the capital of Slovenia?", "gold": "Ljubljana", "kind": "exact"},
    # atomic numbers / less-common science
    {"q": "What is the atomic number of iron? Answer with just the number.", "gold": "26", "kind": "numeric"},
    {"q": "What is the atomic number of gold? Answer with just the number.", "gold": "79", "kind": "numeric"},
    {"q": "What is the atomic number of carbon? Answer with just the number.", "gold": "6", "kind": "numeric"},
    {"q": "What is the chemical symbol for tungsten?", "gold": "W", "kind": "exact"},
    {"q": "How many hearts does an octopus have? Answer with just the number.", "gold": "3", "kind": "numeric"},
    {"q": "How many moons does Mars have? Answer with just the number.", "gold": "2", "kind": "numeric"},
    {"q": "What is the largest moon of Saturn?", "gold": "Titan", "kind": "exact"},
    {"q": "What is the most abundant gas in Earth's atmosphere?", "gold": "nitrogen", "kind": "exact"},
    # exact dates / history
    {"q": "In what year was the Magna Carta signed? Answer with just the year.", "gold": "1215", "kind": "numeric"},
    {"q": "In what year did the French Revolution begin? Answer with just the year.", "gold": "1789", "kind": "numeric"},
    {"q": "In what year did the Western Roman Empire fall? Answer with just the year.", "gold": "476", "kind": "numeric"},
    {"q": "In what year was the US Declaration of Independence signed? Answer with just the year.", "gold": "1776", "kind": "numeric"},
    # nth-of-a-sequence
    {"q": "What is the 7th planet from the Sun?", "gold": "Uranus", "kind": "exact"},
    {"q": "Who was the 3rd President of the United States?", "gold": "Jefferson", "kind": "exact", "aliases": ["Thomas Jefferson"]},
    {"q": "Who was the 16th President of the United States?", "gold": "Lincoln", "kind": "exact", "aliases": ["Abraham Lincoln"]},
    {"q": "What is the 5th element on the periodic table?", "gold": "boron", "kind": "exact"},
]

# ARITH_PROBES -- programmatically generated arithmetic with GUARANTEED-correct golds (Python computes
# them) across escalating difficulty tiers. This is the antidote to the curated sets' problem: a strong 7B
# saturates factual QA (few errors -> a degenerate risk-coverage curve), and hand-writing enough HARD
# factual items risks gold errors that would corrupt the calibration. Arithmetic sidesteps both: a 7B nails
# 2-digit sums (high confidence) but reliably slips on 3x3-digit products, so the set induces GRADED errors
# with zero gold-error risk -- exactly what a non-degenerate curve needs. Deterministic (seeded), so
# `clozn eval` is reproducible.
_ARITH_TIERS = [                      # (tier name, operand generator) -- roughly easy -> hard for a 7B
    ("add_2d",   lambda r: (r.randint(10, 99),   "+", r.randint(10, 99))),
    ("sub_4d",   lambda r: (r.randint(1000, 9999), "-", r.randint(100, 999))),
    ("mul_1x2",  lambda r: (r.randint(3, 9),     "*", r.randint(11, 99))),
    ("mul_2x2",  lambda r: (r.randint(12, 99),   "*", r.randint(12, 99))),
    ("mul_3x2",  lambda r: (r.randint(101, 999), "*", r.randint(12, 99))),
    ("mul_3x3",  lambda r: (r.randint(101, 999), "*", r.randint(101, 999))),
]


def arithmetic_probes(n: int = 60, seed: int = 7) -> list[dict]:
    """`n` arithmetic probes cycling the difficulty tiers, with exact golds. Seeded -> deterministic."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        name, gen = _ARITH_TIERS[i % len(_ARITH_TIERS)]
        a, op, b = gen(rng)
        val = a + b if op == "+" else (a - b if op == "-" else a * b)
        out.append({"q": f"What is {a} {op} {b}? Answer with just the number.",
                    "gold": str(val), "kind": "numeric", "tier": name})
    return out


ARITH_PROBES: list[dict] = arithmetic_probes()

_SYSTEM = "You are a precise assistant. Answer the question as briefly as possible -- ideally a single word or number, with no explanation."


def run_probes(base_url: str, probes: list[dict] | None = None, model: str = "clozn",
               timeout: float = 90.0) -> list[dict]:
    """POST each probe to `{base_url}/v1/chat/completions` (OpenAI chat API) and collect the reply. Returns
    [{q, gold, kind, aliases, reply, error?}]; a per-probe failure is captured, never fatal. Pure I/O -- no
    grading, no scoring here (score comes from the logged run trace, which the wire format omits)."""
    probes = probes if probes is not None else PROBES
    url = base_url.rstrip("/") + "/v1/chat/completions"
    out = []
    for p in probes:
        rec = {"q": p["q"], "gold": p["gold"], "kind": p["kind"], "aliases": p.get("aliases", [])}
        body = json.dumps({"model": model, "temperature": 0.0, "max_tokens": 40,
                           "messages": [{"role": "system", "content": _SYSTEM},
                                        {"role": "user", "content": p["q"]}]}).encode()
        try:
            req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
            rec["reply"] = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        except Exception as e:                                    # noqa: BLE001 -- capture, keep going
            rec["reply"] = ""
            rec["error"] = str(e)[:200]
        out.append(rec)
    return out
