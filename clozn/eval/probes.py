"""Built-in factual probe sets + a dependency-light runner, so calibration can be computed on REAL model
answers (not only synthetic pairs). The set deliberately MIXES difficulty -- gimmes a 7B nails (France ->
Paris) alongside classic traps (Australia -> Canberra, not Sydney) and arithmetic that small models slip on
-- so the risk-coverage curve has signal: the point is to see whether the model's own confidence separates
its right answers from its wrong ones.

Sets: PROBES (curated factual, easy + capital-city traps), HARD_PROBES (curated factual at the edge of a
7B's competence), ARITH_PROBES (programmatic, guaranteed-correct golds, see arithmetic_probes()), and the
EXTENDED_PROBES v2 set (FACTUAL_PROBES + REASONING_PROBES + MISCONCEPTION_PROBES + TRICK_PROBES -- logic
puzzles, common misconceptions the model may have absorbed uncritically, and careful-reading trick
questions). Select a set with `clozn eval --set {easy,hard,arith,both,all,extended}` -- see bench.py.

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

# --- EXTENDED SET (v2) -- "bigger probe sets" (backlog): PROBES/HARD_PROBES/ARITH_PROBES above are
# UNCHANGED. Everything below is new, additive, and grouped by what it tests rather than by difficulty --
# select it with `clozn eval --set extended` (or `--set all`, which now folds it in). Two optional metadata
# keys ride along for readers/future filtering (ignored by run_probes/bench, which read only q/gold/kind/
# aliases): "category" (domain/purpose) and "difficulty" (easy/medium/hard, so the calibration curve keeps
# both gimmes and traps in this set too, not just in HARD_PROBES).
#
# FACTUAL_PROBES -- new domains PROBES/HARD_PROBES don't touch (geography beyond capitals, more history,
# more science, geometry/percentage math rather than raw arithmetic).
FACTUAL_PROBES: list[dict] = [
    {"q": "What is the longest mountain range located entirely on land?", "gold": "Andes",
     "kind": "exact", "category": "geography", "difficulty": "medium"},
    {"q": "What is the largest hot desert in the world?", "gold": "Sahara",
     "kind": "exact", "category": "geography", "difficulty": "easy"},
    {"q": "What is the deepest known point in Earth's oceans?", "gold": "Mariana Trench",
     "kind": "exact", "aliases": ["Challenger Deep"], "category": "geography", "difficulty": "medium"},
    {"q": "What is the largest country in the world by total area?", "gold": "Russia",
     "kind": "exact", "category": "geography", "difficulty": "easy"},
    {"q": "What is the largest lake in the world by surface area?", "gold": "Caspian Sea",
     "kind": "exact", "category": "geography", "difficulty": "hard"},
    {"q": "In what year did the American Civil War end? Answer with just the year.", "gold": "1865",
     "kind": "numeric", "category": "history", "difficulty": "medium"},
    {"q": "In what year did World War I begin? Answer with just the year.", "gold": "1914",
     "kind": "numeric", "category": "history", "difficulty": "easy"},
    {"q": "In what year did the Cold War end, marked by the dissolution of the Soviet Union? "
          "Answer with just the year.", "gold": "1991", "kind": "numeric", "category": "history",
     "difficulty": "medium"},
    {"q": "In what year did the Wright brothers achieve the first powered airplane flight? "
          "Answer with just the year.", "gold": "1903", "kind": "numeric", "category": "history",
     "difficulty": "hard"},
    {"q": "Who was the first Emperor of Rome?", "gold": "Augustus",
     "kind": "exact", "category": "history", "difficulty": "hard"},
    {"q": "Which ancient wonder of the world was located in Giza, Egypt?", "gold": "Great Pyramid",
     "kind": "exact", "aliases": ["pyramids", "pyramid", "Pyramid of Giza", "Great Pyramid of Giza"],
     "category": "history", "difficulty": "easy"},
    {"q": "What is the chemical symbol for sodium?", "gold": "Na",
     "kind": "exact", "category": "science", "difficulty": "medium"},
    {"q": "What is the name of the structure often called the powerhouse of the cell?",
     "gold": "mitochondria", "kind": "exact", "aliases": ["mitochondrion"], "category": "science",
     "difficulty": "easy"},
    {"q": "How many planets are in our solar system? Answer with just the number.", "gold": "8",
     "kind": "numeric", "category": "science", "difficulty": "easy"},
    {"q": "What force keeps planets in orbit around the Sun?", "gold": "gravity",
     "kind": "exact", "category": "science", "difficulty": "easy"},
    {"q": "At what temperature does water boil at sea level, in degrees Celsius? "
          "Answer with just the number.", "gold": "100", "kind": "numeric", "category": "science",
     "difficulty": "easy"},
    {"q": "What is the value of pi rounded to two decimal places? Answer with just the number.",
     "gold": "3.14", "kind": "numeric", "category": "math", "difficulty": "medium"},
    {"q": "How many degrees are in a right angle? Answer with just the number.", "gold": "90",
     "kind": "numeric", "category": "math", "difficulty": "easy"},
]

# REASONING_PROBES -- logic puzzles, syllogisms (valid AND invalid, so "yes" isn't a free lunch),
# cause-and-effect ordering, kinship/relational deduction, sequences, and the classic CRT
# (cognitive-reflection-test) items that induce a fast, confident, WRONG answer even in careful reasoners
# -- exactly the kind of item where confidence/correctness divergence is the whole point of this eval.
REASONING_PROBES: list[dict] = [
    {"q": "All Bloops are Razzles. All Razzles are Lazzles. Are all Bloops definitely Lazzles? "
          "Answer yes or no.", "gold": "yes", "kind": "exact", "category": "reasoning",
     "difficulty": "easy"},
    {"q": "If it rains, the ground gets wet. The ground is wet. Does that necessarily mean it rained? "
          "Answer yes or no.", "gold": "no", "kind": "exact", "category": "reasoning",
     "difficulty": "medium"},
    {"q": "What comes next in the sequence: 2, 4, 8, 16, ? Answer with just the number.", "gold": "32",
     "kind": "numeric", "category": "reasoning", "difficulty": "easy"},
    {"q": "As perceived by a person on the ground during a storm, which happens first: lightning or "
          "thunder? Answer with one word.", "gold": "lightning", "kind": "exact", "category": "reasoning",
     "difficulty": "easy"},
    {"q": "A is the son of B. B is the father of A. C is the father of B. What is C to A? "
          "Answer with one word.", "gold": "grandfather", "kind": "exact", "category": "reasoning",
     "difficulty": "medium"},
    {"q": "Which of these is not like the others: apple, banana, carrot, orange? Answer with just "
          "the word.", "gold": "carrot", "kind": "exact", "category": "reasoning", "difficulty": "easy"},
    {"q": "Tom is taller than Jerry. Jerry is taller than Spike. Who is the shortest? "
          "Answer with just the name.", "gold": "Spike", "kind": "exact", "category": "reasoning",
     "difficulty": "medium"},
    {"q": "All cats are animals. Fluffy is a cat. Is Fluffy an animal? Answer yes or no.", "gold": "yes",
     "kind": "exact", "category": "reasoning", "difficulty": "easy"},
    {"q": "If a train travels 60 miles in 1 hour, how many miles does it travel in 3 hours at the same "
          "speed? Answer with just the number.", "gold": "180", "kind": "numeric", "category": "reasoning",
     "difficulty": "easy"},
    {"q": "If today is Wednesday, what day of the week will it be in 10 days? Answer with just the "
          "day name.", "gold": "Saturday", "kind": "exact", "category": "reasoning", "difficulty": "medium"},
    {"q": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much "
          "does the ball cost, in dollars? Answer with just the number, e.g. 0.05.", "gold": "0.05",
     "kind": "numeric", "category": "reasoning", "difficulty": "hard"},
    {"q": "You are running a race and you overtake the person in second place. What place are you "
          "in now? Answer with just the number.", "gold": "2", "kind": "numeric", "category": "reasoning",
     "difficulty": "hard"},
    {"q": "A man looks at a photograph and says: 'Brothers and sisters I have none, but that man's "
          "father is my father's son.' Who is in the photograph? Answer with one word (son/daughter/self).",
     "gold": "son", "kind": "exact", "category": "reasoning", "difficulty": "hard"},
    {"q": "What comes next in the sequence: 1, 1, 2, 3, 5, ? Answer with just the number.", "gold": "8",
     "kind": "numeric", "category": "reasoning", "difficulty": "medium"},
    {"q": "All roses are flowers. Some flowers fade quickly. Does that necessarily mean some roses "
          "fade quickly? Answer yes or no.", "gold": "no", "kind": "exact", "category": "reasoning",
     "difficulty": "medium"},
    {"q": "Which of these numbers is not a prime number: 3, 5, 7, 9, 11? Answer with just the number.",
     "gold": "9", "kind": "numeric", "category": "reasoning", "difficulty": "medium"},
    {"q": "Hand is to glove as foot is to ? Answer with just one word.", "gold": "sock",
     "kind": "exact", "category": "reasoning", "difficulty": "easy"},
    {"q": "If 5 machines take 5 minutes to make 5 widgets, how many minutes would 100 machines take "
          "to make 100 widgets? Answer with just the number.", "gold": "5", "kind": "numeric",
     "category": "reasoning", "difficulty": "hard"},
]

# MISCONCEPTION_PROBES -- popular beliefs that are FALSE (plus a few phrased so the correct answer is
# "yes", so a model can't just learn to answer "no" to everything in this set). Calibration value: a
# well-calibrated model should be LESS confident here than on plain factual recall, because these are
# exactly the claims a model is likely to have absorbed uncritically from its training text.
MISCONCEPTION_PROBES: list[dict] = [
    {"q": "Do humans typically use virtually all regions of their brain over the course of a day, "
          "not just about 10% of it? Answer yes or no.", "gold": "yes", "kind": "exact",
     "category": "misconception", "difficulty": "medium"},
    {"q": "Does shaved hair grow back thicker and darker than before? Answer yes or no.", "gold": "no",
     "kind": "exact", "category": "misconception", "difficulty": "easy"},
    {"q": "Is the Great Wall of China visible to the naked eye from space? Answer yes or no.",
     "gold": "no", "kind": "exact", "category": "misconception", "difficulty": "easy"},
    {"q": "Do goldfish have a memory span of only a few seconds? Answer yes or no.", "gold": "no",
     "kind": "exact", "category": "misconception", "difficulty": "easy"},
    {"q": "Is it true that lightning never strikes the same place twice? Answer yes or no.", "gold": "no",
     "kind": "exact", "category": "misconception", "difficulty": "easy"},
    {"q": "Is glass actually a slow-flowing liquid at room temperature? Answer yes or no.", "gold": "no",
     "kind": "exact", "category": "misconception", "difficulty": "medium"},
    {"q": "Do bulls become angry specifically because of the color red? Answer yes or no.", "gold": "no",
     "kind": "exact", "category": "misconception", "difficulty": "medium"},
    {"q": "Was Albert Einstein a failing student in school as a child? Answer yes or no.", "gold": "no",
     "kind": "exact", "category": "misconception", "difficulty": "medium"},
    {"q": "Does swallowed chewing gum stay in your stomach for seven years? Answer yes or no.",
     "gold": "no", "kind": "exact", "category": "misconception", "difficulty": "easy"},
    {"q": "Is body heat lost roughly in proportion to exposed skin surface area, rather than "
          "disproportionately through the head? Answer yes or no.", "gold": "yes", "kind": "exact",
     "category": "misconception", "difficulty": "hard"},
    {"q": "Is Mount Everest's peak the point on Earth's surface farthest from the Earth's center? "
          "Answer yes or no.", "gold": "no", "kind": "exact", "category": "misconception",
     "difficulty": "hard"},
    {"q": "Was Napoleon Bonaparte unusually short for his time? Answer yes or no.", "gold": "no",
     "kind": "exact", "category": "misconception", "difficulty": "medium"},
    {"q": "Do ostriches bury their heads in the sand when frightened? Answer yes or no.", "gold": "no",
     "kind": "exact", "category": "misconception", "difficulty": "easy"},
    {"q": "Does the tongue have distinct zones that exclusively detect different tastes (the "
          "'tongue map')? Answer yes or no.", "gold": "no", "kind": "exact", "category": "misconception",
     "difficulty": "medium"},
    {"q": "Did most educated medieval Europeans believe the Earth was flat? Answer yes or no.",
     "gold": "no", "kind": "exact", "category": "misconception", "difficulty": "medium"},
    {"q": "Is a tomato botanically classified as a vegetable? Answer yes or no.", "gold": "no",
     "kind": "exact", "category": "misconception", "difficulty": "medium"},
    {"q": "Did historical Vikings typically wear horned helmets? Answer yes or no.", "gold": "no",
     "kind": "exact", "category": "misconception", "difficulty": "easy"},
    {"q": "Is 'Frankenstein' the name of the monster in Mary Shelley's novel? Answer yes or no.",
     "gold": "no", "kind": "exact", "category": "misconception", "difficulty": "medium"},
    {"q": "Does consuming sugar cause hyperactivity in children? Answer yes or no.", "gold": "no",
     "kind": "exact", "category": "misconception", "difficulty": "medium"},
    {"q": "Are Earth's seasons caused mainly by its axial tilt rather than its distance from the Sun? "
          "Answer yes or no.", "gold": "yes", "kind": "exact", "category": "misconception",
     "difficulty": "medium"},
]

# TRICK_PROBES -- questions with a definitive answer that only falls out if you read carefully; a plausible
# but wrong reading produces a different, still-confident-sounding answer. Good for probing whether a
# model's confidence tracks having actually parsed the question, not just pattern-matched a familiar shape.
TRICK_PROBES: list[dict] = [
    {"q": "How many months of the year have 28 days? Answer with just the number.", "gold": "12",
     "kind": "numeric", "category": "trick", "difficulty": "medium"},
    {"q": "If a plane crashes exactly on the border between the US and Canada, where are the "
          "survivors buried? Answer with one word.", "gold": "nowhere",
     "aliases": ["not buried", "none"], "kind": "exact", "category": "trick", "difficulty": "medium"},
    {"q": "What has to be broken before you can use it? Answer with one word.", "gold": "egg",
     "aliases": ["an egg"], "kind": "exact", "category": "trick", "difficulty": "medium"},
    {"q": "Before Mount Everest was discovered to be the tallest mountain on Earth, what was the "
          "tallest mountain on Earth? Answer with one or two words.", "gold": "Everest",
     "aliases": ["Mount Everest"], "kind": "exact", "category": "trick", "difficulty": "medium"},
    {"q": "A doctor tells you to take 3 pills, one every 30 minutes. How many minutes will it take "
          "to take all 3 pills? Answer with just the number.", "gold": "60", "kind": "numeric",
     "category": "trick", "difficulty": "hard"},
    {"q": "There are 6 apples on a table and you take away 4. How many apples do you have? "
          "Answer with just the number.", "gold": "4", "kind": "numeric", "category": "trick",
     "difficulty": "medium"},
    {"q": "A classic riddle asks how many of each animal Moses took on the ark. Was it actually "
          "Moses or Noah in the Bible story? Answer with one name.", "gold": "Noah", "kind": "exact",
     "category": "trick", "difficulty": "easy"},
    {"q": "Is it legal for a man to marry his widow's sister? Answer yes or no.", "gold": "no",
     "kind": "exact", "category": "trick", "difficulty": "medium"},
    {"q": "What English word is always spelled incorrectly, no matter how you spell it? "
          "Answer with one word.", "gold": "incorrectly", "kind": "exact", "category": "trick",
     "difficulty": "medium"},
    {"q": "A farmer has 17 sheep, and all but 9 die. How many sheep does the farmer have left? "
          "Answer with just the number.", "gold": "9", "kind": "numeric", "category": "trick",
     "difficulty": "medium"},
    {"q": "I am an odd number. Take away one letter and I become even. What number am I? "
          "Answer with one word.", "gold": "seven", "aliases": ["7"], "kind": "exact",
     "category": "trick", "difficulty": "medium"},
    {"q": "What can you catch but not throw? Answer with one word.", "gold": "cold",
     "aliases": ["a cold"], "kind": "exact", "category": "trick", "difficulty": "easy"},
    {"q": "A rooster lays an egg on the very top of a barn roof. Which side does the egg roll down? "
          "Answer with one word.", "gold": "neither", "kind": "exact", "category": "trick",
     "difficulty": "medium"},
    {"q": "Which is heavier: a pound of feathers or a pound of bricks? Answer with one word "
          "(feathers/bricks/equal).", "gold": "equal", "aliases": ["same"], "kind": "exact",
     "category": "trick", "difficulty": "easy"},
    {"q": "Mary's mother has four children. Three are named April, May, and June. What is the name "
          "of the fourth child? Answer with one word.", "gold": "Mary", "kind": "exact",
     "category": "trick", "difficulty": "medium"},
    {"q": "You walk into a cold, dark room with only one match. There is an oil lamp, a candle, and "
          "a fireplace with kindling. What do you light first? Answer with one word.", "gold": "match",
     "aliases": ["the match"], "kind": "exact", "category": "trick", "difficulty": "medium"},
    {"q": "What gets wetter the more it dries? Answer with one word.", "gold": "towel",
     "kind": "exact", "category": "trick", "difficulty": "easy"},
    {"q": "A red house is made of red bricks and a blue house is made of blue bricks. What is a "
          "greenhouse made of? Answer with one word.", "gold": "glass", "kind": "exact",
     "category": "trick", "difficulty": "easy"},
]

# EXTENDED_PROBES -- the v2 set: FACTUAL_PROBES + REASONING_PROBES + MISCONCEPTION_PROBES + TRICK_PROBES.
# Select with `clozn eval --set extended` (or python -m clozn.eval.bench --set extended). Default stays
# "arith" (see bench.py/cli/commands/eval.py) -- that choice was deliberate (guaranteed-correct golds, zero
# gold-error risk, reproducible), and this set doesn't override it; `--set all` now folds EXTENDED_PROBES in
# too, so "all" means all curated + generated probes.
EXTENDED_PROBES: list[dict] = FACTUAL_PROBES + REASONING_PROBES + MISCONCEPTION_PROBES + TRICK_PROBES

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
            raw = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
            from clozn.runs.receipt_footer import _strip_text
            rec["reply"] = _strip_text(raw)
        except Exception as e:                                    # noqa: BLE001 -- capture, keep going
            rec["reply"] = ""
            rec["error"] = str(e)[:200]
        out.append(rec)
    return out
