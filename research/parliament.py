"""parliament.py -- Wild Experiment #4 (Wave 1): the parliament of stances, cross-family.

Pre-registration: research/WILD_WAVE1_PREREG.md (exp 4). Antecedent: steering.py's SteeringControl
(diff-of-means tone dials, auto-calibrated to ||resid||) -- this module does not reinvent that machinery,
it drives it. The claim under test: K=5 parallel decodes of ONE model, each steered by a different STANCE
(candid / warm / skeptical / concrete / plain), then judge-merged into one answer, beats (a) a single
plain greedy decode and (b) a K-sample TEMPERATURE vote of the same width -- because directed diversity
(really different stances) should beat thermal diversity (sampling noise). Second, equally important
question: does diff-of-means steering even WORK on Gemma-2 (logit-capping is a plausible steering-breaker)?

FOUR ARMS, per model, on N questions (default 30; --smoke -> 4):
  1. PARLIAMENT       -- 5 steered decodes (candid/warm/skeptical/concrete/plain), judge-merged into 1.
  2. SINGLE           -- one plain unsteered greedy decode. The floor.
  3. TEMP-VOTE NULL   -- 5 temperature samples, NO steering, judge-merged the same way. Isolates
                         DIRECTED diversity (parliament) from THERMAL diversity (sampling noise alone).
  4. SHUFFLED-DIAL NULL -- 5 decodes steered by RANDOM directions of matched norm (same ||strength*base||
                         as the real stance they stand in for), judge-merged the same way. Isolates
                         "steering helped" from "any perturbation of this size helped".
  (Naming note: "plain" is BOTH one of the 5 stances AND the adjective describing arm 2's single decode --
  arm 2 is the model's ordinary unsteered behavior, never the "plain" DIAL engaged. Kept distinct in code:
  the stance list is STANCES; the floor arm is "single".)

THE JUDGE (the hard part the pre-reg flags). Two GPU realities shape the design: (a) only one model fits
on the 16GB card at a time (concurrent loads OOM), so "load / run / free" happens TWICE per run -- the
subject model generates every arm's raw decodes, is then freed, and a SEPARATE judge model is loaded to
do the merging and scoring; (b) that means the judge can be genuinely INDEPENDENT -- by default it is the
OTHER family (Qwen subject -> Gemma judge, and vice versa), a stronger notion of "independent" than a
different checkpoint of the same architecture. Metric: QUESTION_BANK is 30 practical/explanatory questions
each with a hand-written rubric of 4-6 required POINTS a complete, correct answer should cover -- this is
the "checkable coverage axis" the pre-reg asks for. Each point is worded so that satisfying it requires
BOTH being on-topic and factually correct in one clause (e.g. "correctly notes reheated rice risk comes
from Bacillus cereus SPORES, not just bacteria") -- a simplification stated here: there is no separate
correctness sub-score, coverage-correctness are bundled per point rather than scored twice. The judge reads
the question + rubric + a candidate final answer and emits one 0/1 bit per point (parsed by parse_bits,
which fails HONESTLY -- returns None, never a padded guess -- when it cannot find a clean bit sequence).
coverage_mean over the 30 questions is the primary number per arm.

JUDGE TRUST is calibrated BY THE NULLS, not asserted: judge_trust_report flags the judge UNTRUSTWORTHY if
it scores the shuffled-dial null above the single-decode floor, or fails to clearly separate the
shuffled-dial null from parliament -- exactly "a judge that rates random-direction steering highly is
untrustworthy" from the pre-reg. The judge's OWN consistency is reported too: every score is computed
twice (once greedy, once temperature-sampled) and the bit-flip rate between the two passes is the
disagreement rate. A SOFTER, clearly-labeled bonus signal rides alongside the rubric: single-pass pairwise
preference (parliament vs each of the other 3 arms), toggleable with --no-pairwise, position-randomized
per question (not cancelled by a double call -- a stated cost/rigor tradeoff, not hidden).

HONEST COUPLING RISK (stated loud, not hidden): the SAME judge model both MERGES the K raw decodes into
one final answer AND SCORES that merged answer, for every arm. If the judge is simply a better or worse
writer when synthesizing 5 stylistically-DIFFERENT inputs (parliament) than 5 near-duplicate ones
(temp-vote), that asymmetry would bias the comparison for reasons that have nothing to do with steering,
and a single judge model gives no clean way to rule it out. This is a design risk, not a solved problem --
read the coherence axis and the null comparisons alongside the headline coverage numbers, not instead of
them.

COHERENCE AXIS (Law #6): every raw decode is scored with counterfactual._coherence BEFORE it enters a
merge -- a degenerate (repetition-looped / char-runaway / script-switch) decode is dropped from the merge
input entirely, so a derailed stance cannot win on lexical noise. Degenerate rates are reported per arm.
If ALL K decodes in an arm are degenerate for a question, the merge step honestly reports "no coherent
candidate" (which then correctly scores near zero on the rubric) rather than silently merging garbage.

STEERING-LIVENESS CHECK (the pre-reg's second question). calibrate_and_check_liveness sweeps each stance's
dose (Law #6: dose is recalibrated PER MODEL, never transferred) and, at every nonzero dose, generates with
the REAL direction and with a matched-norm RANDOM direction on the SAME neutral probes, scoring both with a
crude-but-transparent per-axis lexical marker rate (same spirit as steer_vs_prompt.py's warm/hedge scorers
-- gameable, but honest about it) plus the coherence axis. "Live" = at the chosen operating dose, the real
direction beats both the unsteered baseline AND the random-direction null on marker rate, without
degenerating. The chosen dose and the random direction from THIS sweep are the ones actually reused in the
main shuffled-dial-null arm -- the calibration curve reported is the null that was actually run, not a
separate proxy for it.

GEMMA'S CHAT TEMPLATE REJECTS A SYSTEM ROLE. SingleTurnSteer overrides SteeringControl._last_resid to fold
the pos/neg pole instruction into ONE user turn instead of a system+user pair -- applied uniformly to BOTH
families (not just Gemma), so the direction-computation recipe is identical across the cross-family
comparison. Every other user-facing generate (Rig.gen, the merge prompt, the judge prompt) is single-user-
turn by construction; there is no code path in this file that ever emits a system message.

WHAT THIS DOES NOT TEST: the pre-reg's framing note that "batched decode is ~free" (memory-bandwidth-bound
at batch-1) is the ECONOMIC reason parliament would be worth deploying if it wins on quality -- it is NOT
tested here. All "K decodes" in every arm are K SEQUENTIAL generate() calls (SteeringControl's hook adds
the same vector to every row in a batch; steering K different directions in one batched forward would need
a per-batch-row hook this file does not add). Arm-level wall-clock is recorded (arms_wall_clock_sec,
judge_wall_clock_sec) as an FYI, not as a batched-vs-sequential cost measurement. One seed, greedy except
where sampling is the point (temp-vote), one judge model, 30 questions -- caveats stated loud, not buried.

Run (CUDA venv), one model per process; subject model generates every arm, is freed, then a cross-family
judge model loads to merge+score (single 16GB card -- never concurrent):
    PY=C:/Users/brigi/src/cloze/.venv/Scripts/python.exe
    $PY research/parliament.py --model Qwen/Qwen2.5-7B-Instruct --out research/runs/parliament_qwen7b.json
    $PY research/parliament.py --model google/gemma-2-9b-it     --out research/runs/parliament_gemma9b.json
    $PY research/parliament.py --compare research/runs/parliament_qwen7b.json research/runs/parliament_gemma9b.json
Smoke first (4 questions, a small same-family judge instead of a second 7-9B load -- proves the WIRING,
not a trustworthy finding; --smoke's judge choice is documented as such, never to be read as a result):
    $PY research/parliament.py --model Qwen/Qwen2.5-7B-Instruct --smoke --out research/runs/parliament_smoke.json
"""
from __future__ import annotations

import argparse, gc, json, os, random, re, sys, time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import steering as steering_mod
from steering import SteeringControl
from counterfactual import _coherence   # {"degenerate": bool, "reason": str} -- the mandatory coherence axis

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# The 5 parliament stances, in the pre-reg's own order. 3 are steering.py built-ins (candid/warm/concrete);
# 2 are new custom dials registered via SteeringControl.add_custom (skeptical, plain) -- same diff-of-means
# recipe, arbitrary poles, no changes to steering.py needed.
STANCES = ["candid", "warm", "skeptical", "concrete", "plain"]

# The 4 arms, in the pre-reg's own numbered order -- used consistently for dict construction/printing.
ARMS_ORDER = ["parliament", "single", "temp_vote_null", "shuffled_dial_null"]
_ARM_LABEL = {"parliament": "parliament", "single": "single (floor)",
              "temp_vote_null": "temp-vote null", "shuffled_dial_null": "shuffled-dial null"}

_SKEPTICAL_POS = ("Respond with skeptical, critical scrutiny: question the claims involved, flag what is "
                  "unproven, uncertain, or unverified, and do not accept assertions at face value.")
_SKEPTICAL_NEG = ("Respond with complete trust and acceptance: take all claims at face value and do not "
                  "question or doubt anything.")
_PLAIN_POS = ("Respond in plain, unembellished language: state things simply and directly, with no "
              "metaphor, no rhetorical flourish, and no stylistic decoration.")
_PLAIN_NEG = ("Respond in a highly stylized, embellished, decorative way, full of rhetorical flourish, "
              "vivid metaphor, and elaborate language.")

# Crude, transparent, gameable per-axis lexical marker lists -- same spirit as steer_vs_prompt.py's
# WARM_MARKERS/HEDGES. Used ONLY for the steering-liveness check (does the REAL direction move text toward
# its stance more than a matched-norm RANDOM direction does); the judge's rubric score is the real quality
# metric, never this. candid/warm/concrete borrow the flavor of steering.py's own _DIAL_LEXICON entries for
# those axes; skeptical/plain are hand-built since they are not in that lexicon.
_MARKERS = {
    "candid":    ["frankly", "honestly", "to be blunt", "i disagree", "that's not right", "push back",
                  "candidly", "not going to sugarcoat", "let's be real", "the problem with", "flawed"],
    "warm":      ["glad", "wonderful", "i care", "i hope", "warm", "proud of you", "you're doing",
                  "take care", "happy to help", "lovely", "gentle", "!"],
    "concrete":  ["for example", "specifically", "such as", "e.g.", "concretely", "in particular",
                  "one example", "let's say", "imagine", "a specific"],
    "skeptical": ["evidence", "unproven", "not clear", "unclear", "verify", "questionable", "skeptical",
                  "doubt", "unverified", "however", "actually", "assum"],
    "plain":     ["simply", "simple", "plainly", "in short", "basically", "just ", "straightforward",
                  "bottom line", "no frills", "directly"],
}

# Neutral probes for the calibration/liveness sweep -- deliberately disjoint from QUESTION_BANK (the main
# eval) and from steering.py's own SEED_PROMPTS (used only to compute the diff-of-means direction itself).
CALIB_PROBES = [
    "What do you think about trying a new hobby this year?",
    "Tell me about your ideal weekend.",
    "How should I approach a tricky conversation with a coworker?",
]

_DEGEN_OK = 0.34            # a dose is "safe" if at most ~1-of-3 calibration probes come back degenerate
_TRUST_TOL = 4.0            # percentage-point tolerance before a coverage gap counts as a judge-trust flag
_SWEEP_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]
_SWEEP_FRACS_SMOKE = [0.0, 0.6]
_MERGE_MAX_NEW = 260
_SCORE_MAX_NEW = 48
_PREF_MAX_NEW = 8


# =========================================================================================== the questions
# 30 practical/explanatory questions, each with a hand-written rubric of 4-6 required POINTS. Each point is
# worded so that satisfying it requires BOTH being on-topic and factually correct in the same clause (the
# stated simplification: no separate correctness sub-score). --smoke uses the first 4 (one of each rough
# flavor: pets/home, home tech, cooking-science, food safety).
QUESTION_BANK = [
    {"id": "dog_adopt", "q": "What should someone think through before adopting a rescue dog?", "points": [
        "Committing to daily exercise and attention time appropriate to the dog's breed/energy level",
        "Ongoing costs: food, vet visits, vaccinations, and potential emergency care",
        "Pet-proofing / preparing the home (secure yard, hazards removed, a crate or safe space)",
        "Budgeting time for training and socialization, especially with an unknown or difficult history",
        "Checking household members for pet allergies before bringing the dog home",
    ]},
    {"id": "solar_panel", "q": "How does a rooftop solar panel actually turn sunlight into electricity you can use in your house?", "points": [
        "Photovoltaic cells generate direct current (DC) electricity when sunlight hits them",
        "An inverter converts that DC electricity into alternating current (AC), which household appliances use",
        "The AC electricity feeds into the home's electrical panel/wiring",
        "Excess electricity can be sent back to the grid (net metering) or stored in a battery",
        "Output depends on factors like panel angle/orientation, shading, and daylight hours",
    ]},
    {"id": "altitude_boil", "q": "Why does it take longer to cook food in boiling water at high altitude, like in the mountains?", "points": [
        "Atmospheric pressure is lower at higher altitude",
        "Water boils at a lower temperature when air pressure is lower",
        "Because the boiling water itself is cooler, food takes longer to cook even though it is still 'boiling'",
        "This is why recipes sometimes give separate high-altitude cooking instructions/times",
    ]},
    {"id": "rice_reheat", "q": "Is it safe to reheat rice that has been sitting out, and what is the actual risk?", "points": [
        "Cooked rice can contain spores of Bacillus cereus, a bacterium that survives cooking",
        "If rice is left at room temperature, those spores can germinate and produce a toxin",
        "Reheating can kill the bacteria but may NOT destroy the toxin already produced",
        "Risk is reduced by cooling rice quickly and refrigerating it promptly rather than leaving it out",
    ]},
    {"id": "job_offers", "q": "What should I weigh when deciding between two job offers with different salaries?", "points": [
        "Total compensation, not just base salary (benefits, bonus, equity, retirement match)",
        "Cost of living / location differences if the jobs are in different areas",
        "Growth and learning opportunities / career trajectory at each company",
        "Work-life balance, hours, and culture fit",
        "Job security and the stability of the company/industry",
    ]},
    {"id": "index_funds", "q": "Why do many financial advisors recommend index funds over picking individual stocks for most people?", "points": [
        "Index funds provide broad diversification across many companies, reducing single-stock risk",
        "Most actively-managed funds/stock-pickers underperform the market index over the long run, especially after fees",
        "Index funds typically have much lower fees/expense ratios than actively managed funds",
        "Picking individual stocks requires significant research/time and carries higher risk of concentrated losses",
    ]},
    {"id": "bread_knead", "q": "What is kneading bread dough actually doing, physically, and why does it matter?", "points": [
        "Kneading develops gluten, a stretchy protein network formed from wheat proteins and water",
        "The gluten network traps gas produced by yeast fermentation, which is what makes the dough rise",
        "Well-developed gluten gives bread its structure/chew rather than a dense, crumbly texture",
        "Over-kneading or under-kneading can both produce a poor texture (too tough or too dense)",
    ]},
    {"id": "car_maintenance", "q": "What routine maintenance actually matters most for keeping a car reliable long-term?", "points": [
        "Regular oil and filter changes at the recommended interval",
        "Tire maintenance: pressure checks, rotation, and tread wear",
        "Brake system inspection (pads, fluid)",
        "Following the manufacturer's timing belt/chain and other scheduled maintenance intervals",
        "Keeping the cooling system (coolant/radiator) in good condition to avoid overheating",
    ]},
    {"id": "sleep_hygiene", "q": "What are the main things that actually improve sleep quality, based on sleep hygiene advice?", "points": [
        "Keeping a consistent sleep and wake schedule, even on weekends",
        "Limiting caffeine and alcohol, especially later in the day",
        "Reducing screen/blue light exposure before bed",
        "Keeping the bedroom cool, dark, and quiet",
        "Getting exposure to daylight during the day to support the circadian rhythm",
    ]},
    {"id": "password_manager", "q": "Why do security experts recommend using a password manager instead of reusing passwords?", "points": [
        "Reusing passwords means a breach at ONE site can compromise accounts on other sites (credential stuffing)",
        "A password manager lets you use a long, random, unique password for every account without memorizing them",
        "It reduces reliance on weak, memorable passwords or predictable patterns",
        "Many password managers also help detect reused/breached passwords and support two-factor authentication",
    ]},
    {"id": "new_city_move", "q": "What should someone plan for when moving to a new city where they don't know anyone?", "points": [
        "Researching neighborhoods for safety, commute, and cost before committing to a lease",
        "Budgeting for moving costs, deposits, and a period of lower income/higher expenses during the transition",
        "Building a social network deliberately (clubs, work, hobbies, local events)",
        "Setting up practical logistics: updating address, local healthcare providers, transportation",
    ]},
    {"id": "compost", "q": "How does home composting actually turn food scraps and yard waste into usable soil?", "points": [
        "Microorganisms (bacteria, fungi) break down the organic matter",
        "The process needs a balance of 'greens' (nitrogen-rich, e.g. food scraps) and 'browns' (carbon-rich, e.g. dry leaves)",
        "Aeration (turning the pile) and moisture are needed to keep the microbes active and avoid odor/anaerobic conditions",
        "The end product (finished compost/humus) improves soil structure and adds nutrients",
    ]},
    {"id": "ev_vs_gas", "q": "What are the real tradeoffs between buying an electric car versus a gas car right now?", "points": [
        "EVs typically have lower fuel/energy cost per mile and lower maintenance (fewer moving parts, no oil changes)",
        "EVs often have a higher upfront purchase price, though incentives can offset this",
        "Charging infrastructure and charging time vs. gas refueling speed/availability matter for road trips",
        "Range and battery degradation over time are considerations gas cars don't have",
    ]},
    {"id": "vaccine_immunity", "q": "How do vaccines actually train the immune system to protect against a disease?", "points": [
        "Vaccines expose the immune system to a harmless piece or weakened/inactivated form of a pathogen",
        "The immune system produces antibodies and memory cells in response",
        "Memory cells allow a faster, stronger response if the real pathogen is encountered later",
        "This is why some vaccines need multiple doses/boosters to build strong, lasting memory",
    ]},
    {"id": "negotiate_salary", "q": "What is actually effective advice for negotiating a starting salary?", "points": [
        "Research market rate/comparable salaries for the role and location beforehand",
        "Let the employer name a number first when possible, rather than anchoring low yourself",
        "Consider negotiating the whole package (bonus, equity, vacation, remote work), not just base pay",
        "Practice stating your ask confidently and be prepared to justify it with your experience/value",
    ]},
    {"id": "fridge_food_safety", "q": "What actually determines how long leftovers are safe to keep in the refrigerator?", "points": [
        "Cooked leftovers are generally safe for about 3 to 4 days in the refrigerator",
        "Food should be refrigerated within about 2 hours of cooking to limit bacterial growth",
        "The refrigerator should be kept at or below 40 degrees F (4 C)",
        "When in doubt, or if there's an off smell/appearance, it should be discarded",
    ]},
    {"id": "credit_score", "q": "What are the main factors that determine a person's credit score?", "points": [
        "Payment history (paying bills/debts on time) is usually the biggest factor",
        "Credit utilization (how much of your available credit you're using)",
        "Length of credit history",
        "Credit mix (types of credit used) and new credit inquiries",
    ]},
    {"id": "learn_language", "q": "What actually helps adults learn a new language faster, based on how language learning works?", "points": [
        "Consistent, frequent practice (a little every day) beats occasional long sessions",
        "Active use/output (speaking, writing) reinforces learning more than passive exposure alone",
        "Immersion or regular exposure to native content (media, conversation) helps with natural acquisition",
        "Spaced repetition helps move vocabulary into long-term memory",
    ]},
    {"id": "fix_procrastination", "q": "What are effective, evidence-based ways to deal with chronic procrastination on a big project?", "points": [
        "Breaking the project into small, concrete next steps reduces the activation energy to start",
        "Addressing the underlying emotional avoidance (fear of failure, perfectionism), not just 'willpower'",
        "Using external structure like deadlines, accountability, or time-blocking",
        "Reducing friction/distractions in the environment when working",
    ]},
    {"id": "tire_pressure", "q": "Why does tire pressure matter so much for a car, beyond just avoiding a flat?", "points": [
        "Under-inflated tires increase rolling resistance, which hurts fuel economy",
        "Incorrect pressure causes uneven tire wear, shortening tire lifespan",
        "Under-inflation increases the risk of overheating and blowouts, especially at highway speed",
        "Tire pressure changes with temperature, so it should be checked regularly, not just when a tire looks low",
    ]},
    {"id": "milk_lactose", "q": "Why can some adults drink milk fine while others get an upset stomach from it?", "points": [
        "Digesting milk requires the enzyme lactase to break down lactose (milk sugar)",
        "Many adults produce much less lactase after childhood (lactase non-persistence / lactose intolerance)",
        "Undigested lactose reaching the gut can cause gas, bloating, or diarrhea",
        "Lactose tolerance varies by ancestry/population, not just individual habit",
    ]},
    {"id": "wifi_slow", "q": "What are the most common reasons home WiFi is slow, and what actually helps?", "points": [
        "Distance from the router and physical obstructions (walls, floors) weaken the signal",
        "Interference from other devices/networks on the same channel/frequency band",
        "Too many devices competing for bandwidth at once",
        "Outdated router hardware/firmware or an internet plan that's simply too slow for the usage",
    ]},
    {"id": "muscle_soreness", "q": "What actually causes muscle soreness a day or two after a hard workout, and what helps recovery?", "points": [
        "Delayed onset muscle soreness (DOMS) is linked to microscopic damage/tears in muscle fibers from unfamiliar or intense exertion",
        "It is NOT primarily caused by lactic acid buildup (a common myth) -- lactic acid clears within about an hour",
        "The body repairs and adapts the muscle fibers during rest, which is part of how strength is built",
        "Light activity, hydration, sleep, and adequate protein support recovery; rest days matter",
    ]},
    {"id": "startup_savings", "q": "How much should someone realistically save before quitting a stable job to start a business?", "points": [
        "A common guideline is 3 to 6+ months of personal living expenses as an emergency buffer",
        "Separately budgeting for the business's own startup and operating costs before it turns a profit",
        "Accounting for loss of employer benefits (health insurance, retirement match)",
        "Having a realistic timeline for when the business might become self-sustaining, since most take longer than expected",
    ]},
    {"id": "bike_gears", "q": "How do gears on a bicycle actually make pedaling easier or harder?", "points": [
        "Gears change the ratio between how many times the pedals turn and how many times the wheel turns",
        "A 'lower' gear makes pedaling easier but the bike moves less distance per pedal stroke (good for climbing)",
        "A 'higher' gear is harder to pedal but covers more ground per stroke (good for flat/downhill speed)",
        "Shifting works by moving the chain across different sized front/rear sprockets (chainrings/cassette)",
    ]},
    {"id": "antibiotic_resistance", "q": "How does antibiotic resistance in bacteria actually develop and spread?", "points": [
        "Random mutations (or acquired genes) can make some bacteria naturally resistant to an antibiotic",
        "When antibiotics kill off non-resistant bacteria, the resistant ones survive and reproduce (natural selection)",
        "Overuse or incomplete courses of antibiotics increase the chance resistant strains survive and spread",
        "Resistant bacteria/genes can spread between people, and even between different bacterial species (gene transfer)",
    ]},
    {"id": "budget_50_30_20", "q": "What is the reasoning behind a simple budgeting rule like '50/30/20' for personal finances?", "points": [
        "Roughly 50% of after-tax income goes to needs (housing, food, utilities, minimum debt payments)",
        "Roughly 30% goes to wants (discretionary spending)",
        "Roughly 20% goes to savings and extra debt repayment",
        "It's a rough starting guideline, not a strict rule -- actual ratios should adjust for cost of living and goals",
    ]},
    {"id": "cast_iron_care", "q": "Why do people season and specially care for cast iron pans instead of just washing them like other pans?", "points": [
        "Seasoning is a layer of polymerized oil that creates a natural, semi-non-stick coating",
        "Harsh soap or soaking can strip the seasoning and/or lead to rust since raw cast iron is porous and reactive",
        "Cast iron should generally be dried thoroughly right after washing to prevent rust",
        "Cooking acidic foods for long periods can degrade the seasoning or leach iron/affect flavor",
    ]},
    {"id": "time_zones_travel", "q": "What actually causes jet lag, and what helps reduce it when traveling across time zones?", "points": [
        "Jet lag comes from the body's internal circadian clock being out of sync with the new local time",
        "Light exposure (especially sunlight) is one of the strongest cues for resetting the circadian clock",
        "Gradually shifting sleep schedule before travel, and adjusting to local meal/sleep times on arrival, helps",
        "Jet lag is generally worse traveling eastward than westward because it's harder to advance the body clock than delay it",
    ]},
    {"id": "emergency_fund", "q": "Why do financial planners recommend keeping an emergency fund instead of investing all your savings?", "points": [
        "An emergency fund covers unexpected expenses (job loss, medical bills, car repairs) without going into debt",
        "It should generally be kept in a liquid, easily accessible account, not tied up in investments that could be down when you need cash",
        "A common guideline is 3 to 6 months of essential living expenses",
        "Without it, people may be forced to sell investments at a bad time or rely on high-interest debt during an emergency",
    ]},
]


# ================================================================================================ helpers
def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def _words(s: str) -> int:
    return len((s or "").split())


def marker_rate(text: str, axis: str) -> float:
    """Crude, transparent, gameable lexical marker rate per 100 words for `axis` -- used ONLY for the
    steering-liveness check (real direction vs matched-norm random direction), never as the quality metric."""
    t = (text or "").lower()
    hits = sum(t.count(m) for m in _MARKERS.get(axis, []))
    return round(100.0 * hits / max(1, _words(text)), 2)


def marker_rate_mean(texts: list[str], axis: str) -> float:
    return round(_mean(marker_rate(t, axis) for t in texts), 2) if texts else 0.0


def degenerate_rate(texts: list[str]) -> float:
    return round(_mean(_coherence(t)["degenerate"] for t in texts), 3) if texts else 0.0


# nf4 for anything that won't fit bf16 comfortably on the 16GB card -- mirror_bench.py's convention,
# copied (not imported): this codebase's own precedent (steer_vs_prompt.py has its own cruder version) is
# that each experiment script owns its small model-loading helpers rather than importing a sibling script.
_SMALL = ("0.5b", "1.5b", "-1b", "1b-", "2b", "3b", "-1.7b")
def wants_four_bit(name: str, override: str) -> bool:
    if override == "yes":
        return True
    if override == "no":
        return False
    return not any(s in name.lower() for s in _SMALL)


def default_judge_for(model_name: str, smoke: bool = False) -> str:
    """Cross-family by default: judge with the OTHER family, so 'independent' means a genuinely different
    tokenizer/architecture, not just a different checkpoint of the same one. Under --smoke, default to a
    small fast-loading model instead of a second 7-9B nf4 load -- --smoke proves the WIRING cheaply; its
    judge verdict is not a finding and callers should not read it as one (documented at every print site
    that touches it)."""
    if smoke:
        return "Qwen/Qwen2.5-1.5B-Instruct"
    return "google/gemma-2-9b-it" if "qwen" in model_name.lower() else "Qwen/Qwen2.5-7B-Instruct"


def axis_max_of(sc, axis: str) -> float:
    """Per-axis calibrated ceiling: steering.AXES' own 'max' for a built-in, sc.custom's for a custom-
    registered one, or SteeringControl.set's own default (1.5) if neither declares one."""
    return (steering_mod.AXES.get(axis) or sc.custom.get(axis) or {}).get("max", 1.5)


def _axis_seed(base_seed: int, axis: str) -> int:
    """Deterministic per-(run-seed, axis) integer seed for the shuffled-direction generator -- pure
    integer arithmetic, NOT Python's hash() (string hashing is process-randomized unless PYTHONHASHSEED
    is pinned, which would silently break reproducibility of --seed)."""
    return (int(base_seed) * 1_000_003 + STANCES.index(axis) * 97 + 13) & 0xFFFFFFFF


def make_shuffle_unit_vector(ref: torch.Tensor, seed: int) -> torch.Tensor:
    """A fresh random UNIT direction with the same shape/device/dtype as `ref`, seeded reproducibly on
    CPU (so the same --seed gives the same shuffled directions regardless of CUDA's own RNG state).
    Pure tensor math -- no model -- so this is unit-testable on any CPU tensor."""
    gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0xFFFFFFFF)
    v = torch.randn(ref.shape, generator=gen).to(ref.device, ref.dtype)
    return v / (v.norm() + 1e-8)


def _free_cuda():
    gc.collect()
    if DEV == "cuda":
        torch.cuda.empty_cache()


# =============================================================================== the backbone + steering
class SingleTurnSteer(SteeringControl):
    """SteeringControl, but every contrast prompt used to COMPUTE a direction is folded into a single
    USER turn (no system role) -- Gemma-2's chat template raises on a system message, and using the
    identical single-user-turn recipe for BOTH families (not just Gemma) keeps candid/warm/skeptical/
    concrete/plain apples-to-apples across the cross-family comparison. compute()/add_custom() are
    inherited unchanged and call this override polymorphically -- nothing else in steering.py needs
    touching."""

    @torch.no_grad()
    def _last_resid(self, system: str, user: str) -> torch.Tensor:
        ids = self.tok.apply_chat_template(
            [{"role": "user", "content": f"{system}\n\n{user}"}],
            add_generation_prompt=True, return_tensors="pt").to(DEV)
        hs = self.model(ids, output_hidden_states=True).hidden_states[self.layer + 1]
        return hs[0, -1].float()


def compute_stances(sc: SingleTurnSteer) -> dict:
    """Compute the 5 parliament stance directions on sc's backbone: 3 built-ins from steering.AXES
    (candid, warm, concrete) via sc.compute() -- narrowed to just these 3 first (steer_vs_prompt.py's own
    trick), so we do not burn forward passes on steering.py's other 7 stock axes -- plus 2 NEW custom
    stances (skeptical, plain) via sc.add_custom(), the identical diff-of-means recipe on arbitrary poles.
    All 5 ride on the SAME auto-calibrated sc.base (Law #6: recomputed AND recalibrated per model, never
    reused across models -- compute() sets sc.base from THIS model's own residual norm; add_custom()
    deliberately reuses that same base rather than inventing a second scale)."""
    steering_mod.AXES = {k: v for k, v in steering_mod.AXES.items() if k in ("candid", "warm", "concrete")}
    info = sc.compute()
    for name, pos, neg in (("skeptical", _SKEPTICAL_POS, _SKEPTICAL_NEG), ("plain", _PLAIN_POS, _PLAIN_NEG)):
        sc.add_custom(name, pos, neg, mx=0.5)
    info["custom_axes"] = {"skeptical": {"max": 0.5}, "plain": {"max": 0.5}}
    return info


class Rig:
    """Loads one model (subject OR judge -- same class, called twice per run, NEVER concurrently: the
    16GB card cannot hold both). Local-cache-first path lookup and the nf4-vs-bf16 choice both follow
    steer_vs_prompt.py's Rig, except four_bit uses wants_four_bit (mirror_bench's convention) rather than
    steer_vs_prompt's cruder '"7b" in name' check."""

    def __init__(self, name: str, four_bit_override: str = "auto"):
        path = os.path.join(os.path.expanduser("~"), "hf_models", name.split("/")[-1])
        path = path if os.path.isfile(os.path.join(path, "config.json")) else name
        self.four_bit = wants_four_bit(name, four_bit_override)
        print(f"[load] {name} ({'nf4' if self.four_bit else 'bf16'}, {DEV}) ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(path)
        if self.four_bit:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            self.model = AutoModelForCausalLM.from_pretrained(path, quantization_config=bnb,
                                                              device_map={"": 0}).eval()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(DEV).eval()

    @torch.no_grad()
    def gen(self, user: str, max_new: int = 180, sample: bool = False, temperature: float = 0.9) -> str:
        """Single USER-turn only -- never a system role, so this is Gemma-safe by construction and the
        same call shape is used for every generation in this file (arm decodes, merges, judge scores,
        pairwise prefs). repetition_penalty/no_repeat_ngram_size tame steering-induced loops, matching
        steer_vs_prompt.py's Rig.gen and SteeringControl.generate."""
        ids = self.tok.apply_chat_template([{"role": "user", "content": user}],
                                           add_generation_prompt=True, return_tensors="pt").to(DEV)
        kw = dict(max_new_tokens=max_new, repetition_penalty=1.3, no_repeat_ngram_size=3,
                  pad_token_id=self.tok.eos_token_id or 0)
        if sample:
            kw.update(do_sample=True, temperature=temperature, top_p=0.95)
        else:
            kw.update(do_sample=False)
        out = self.model.generate(ids, **kw)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    def free(self):
        self.model = None
        self.tok = None


# ======================================================================== calibration + steering-liveness
def calibrate_and_check_liveness(rig: Rig, sc: SingleTurnSteer, axis: str, seed: int, smoke: bool = False):
    """Per-axis, per-model: sweep a few doses of `axis`'s REAL direction against the SAME doses of a fixed
    RANDOM matched-norm direction (Law #6 dose calibration + the pre-reg's steering-liveness check in one
    pass -- they need the same sweep machinery, so it is not run twice).

    Returns (report, shuffle_vec). `report` is JSON-safe (the calibration curve + chosen operating dose +
    this axis's liveness verdict); `shuffle_vec` is the UNIT random direction actually used here, handed
    back so the SAME vector (not a fresh one) is reused for this axis's slot in the main shuffled-dial-null
    arm -- the null the calibration curve reports on is the null that is actually run later.

    'Live' = at the chosen operating dose, the axis's own lexical marker rate beats BOTH the unsteered
    baseline and the random-direction null, while staying coherent (a degenerate reply must not win). Only
    the POSITIVE pole of each stance is swept (not a full +/- range) -- matching how the pre-reg frames a
    stance as a single directional dial, not a two-way axis to explore both directions of."""
    axis_max = axis_max_of(sc, axis)
    fracs = _SWEEP_FRACS_SMOKE if smoke else _SWEEP_FRACS
    probes = CALIB_PROBES[:2] if smoke else CALIB_PROBES

    shuffle_vec = make_shuffle_unit_vector(sc.vecs[axis], _axis_seed(seed, axis))

    curve = []
    for frac in fracs:
        strength = round(frac * axis_max, 4)
        sc.disengage(); sc.clear()
        if frac == 0.0:
            reps_real = [rig.gen(p) for p in probes]
            reps_shuf = reps_real                    # steering off either way at frac=0 -- identical by construction
        else:
            sc.set(axis, strength); sc.engage()
            reps_real = [rig.gen(p) for p in probes]
            sc.disengage(); sc.clear()

            sc.vecs["_shuf_tmp"] = shuffle_vec
            sc.strength["_shuf_tmp"] = strength      # direct dict write bypasses .set()'s per-axis cap --
            sc.engage()                              # the null must land at EXACTLY the real axis's magnitude
            reps_shuf = [rig.gen(p) for p in probes]
            sc.disengage()
            del sc.vecs["_shuf_tmp"]; sc.strength.pop("_shuf_tmp", None)

        curve.append({
            "frac": frac, "strength": strength,
            "real_marker_rate": marker_rate_mean(reps_real, axis),
            "shuffled_marker_rate": marker_rate_mean(reps_shuf, axis),
            "real_degenerate_rate": degenerate_rate(reps_real),
            "shuffled_degenerate_rate": degenerate_rate(reps_shuf),
            "sample_real": reps_real[0] if reps_real else "",
        })

    baseline_rate = curve[0]["real_marker_rate"]
    safe = [r for r in curve if r["frac"] > 0 and r["real_degenerate_rate"] <= _DEGEN_OK]
    chosen = safe[-1] if safe else curve[0]
    live = bool(chosen["frac"] > 0
                and chosen["real_marker_rate"] > baseline_rate
                and chosen["real_marker_rate"] > chosen["shuffled_marker_rate"]
                and chosen["real_degenerate_rate"] <= _DEGEN_OK)
    report = {
        "axis": axis, "axis_max": axis_max, "curve": curve,
        "chosen_frac": chosen["frac"], "chosen_strength": chosen["strength"],
        "baseline_marker_rate": baseline_rate,
        "degenerate_at_all_doses": not safe,
        "live": live,
    }
    return report, shuffle_vec


# ============================================================================================= the merge
def build_merge_prompt(question: str, candidates: list[str]) -> str:
    lst = "\n\n".join(f"--- Candidate {i + 1} ---\n{c}" for i, c in enumerate(candidates))
    return (
        "You will merge multiple candidate answers to the SAME question into one final answer.\n\n"
        f"Question: {question}\n\n{lst}\n\n"
        "Write ONE final answer that combines the accurate, useful content from across the candidates "
        "into a single coherent response. Do not mention the candidates, the merging process, or that "
        "there were multiple sources. If candidates disagree, keep the more defensible claim. Do not "
        "just concatenate the candidates -- synthesize. Final answer:"
    )


def merge_candidates(jrig: Rig, question: str, candidates: list[str]) -> tuple[str, dict]:
    """candidates: raw decode texts that already SURVIVED coherence filtering by the caller (a degenerate
    decode must never enter the merge). Zero survivors -> an honest fallback string (which then correctly
    scores near zero on the rubric, rather than silently merging garbage); exactly one survivor -> it IS
    the final answer, no LLM call spent merging a single item; 2+ -> the judge model synthesizes one."""
    if not candidates:
        return ("[no coherent candidate available -- every decode in this arm was degenerate]",
                {"fallback": True, "k_survivors": 0})
    if len(candidates) == 1:
        return candidates[0], {"fallback": False, "k_survivors": 1}
    merged = jrig.gen(build_merge_prompt(question, candidates), max_new=_MERGE_MAX_NEW, sample=False)
    return merged, {"fallback": False, "k_survivors": len(candidates)}


# ============================================================================================ the rubric
def build_judge_prompt(question: str, points: list[str], answer: str) -> str:
    numbered = "\n".join(f"{i + 1}. {p}" for i, p in enumerate(points))
    return (
        "You are grading an ANSWER against a rubric of required points for a QUESTION. For EACH numbered "
        "point, decide whether the answer addresses that point AND is factually correct about it.\n\n"
        f"QUESTION: {question}\n\nREQUIRED POINTS:\n{numbered}\n\nANSWER:\n{answer}\n\n"
        f"Reply with EXACTLY {len(points)} characters, one per point in order, each '1' if the answer "
        "addresses that point correctly, or '0' if it is missing, vague, or wrong about it. Separate "
        "them with single spaces. Output ONLY the characters, nothing else."
    )


def parse_bits(raw: str, n: int) -> tuple[list[int] | None, str]:
    """Extract exactly n 0/1 judge-verdict bits from `raw`. Finds the LONGEST maximal run of whitespace/
    comma-separated tokens that are exactly '0' or '1' (a single trailing '.'/'!' on the whole reply is
    stripped first, the common "ends the sentence" case) -- robust to a short preamble/trailing remark,
    without treating punctuation-glued digits ("point 1.") as bits. An honest parse failure (the longest
    clean run is shorter than n) returns (None, ...) -- NEVER a padded guess."""
    text = (raw or "").strip()
    if text.endswith((".", "!")):
        text = text[:-1].strip()
    chunks = re.split(r"[\s,]+", text) if text else []
    runs, cur = [], []
    for c in chunks:
        if c in ("0", "1"):
            cur.append(int(c))
        else:
            if cur:
                runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    best = max(runs, key=len, default=[])
    if len(best) < n:
        return None, f"parse-fail: longest clean 0/1 run={len(best)}/{n} in {raw[:80]!r}"
    return best[:n], ("ok" if len(best) == n else f"truncated {len(best)}->{n}")


def coverage_pct(bits: list[int] | None) -> float | None:
    if not bits:
        return None
    return round(100.0 * sum(bits) / len(bits), 1)


# ============================================================================= pairwise (SOFT, bonus) leg
def build_pref_prompt(question: str, a_text: str, b_text: str) -> str:
    return (f"QUESTION: {question}\n\nANSWER A:\n{a_text}\n\nANSWER B:\n{b_text}\n\n"
            "Which answer better and more completely answers the question, taking accuracy into account? "
            "Reply with EXACTLY one token: A, B, or TIE. Output nothing else.")


def parse_pref(raw: str) -> str | None:
    """Parse the pairwise judge's A/B/TIE verdict. Fast path: the reply IS just the bare token (what the
    prompt asks for -- 'Output nothing else'), matched whole-string so a bare 'A' is never confused with
    anything. Fallback (the judge added commentary anyway): take the LAST standalone A/B/TIE token in the
    text, not the first -- 'I'd call this a TIE' contains the common English article 'a' BEFORE the real
    verdict 'TIE', so first-match-wins would misread it; a concluding verdict is far more often the last
    such token than the first (also handles 'Answer A is better... so I'll say A -> B, final answer A')."""
    text = (raw or "").strip()
    exact = text.strip(".,!?\"' ").upper()
    if exact in ("A", "B", "TIE"):
        return exact
    matches = re.findall(r"\b(A|B|TIE)\b", text.upper())
    return matches[-1] if matches else None


def pairwise_phase(jrig: Rig, questions: list[dict], judge_results: dict, seed: int = 0) -> dict:
    """SOFTER, supplementary signal -- NOT the primary metric. Single-pass A/B preference between
    parliament's final answer and each null/floor's, judged by the SAME judge model that did the merging
    and rubric scoring (the coupling risk stated in the module docstring applies here too). Side (A/B) is
    randomized per question via a seeded RNG so position bias averages out across the 30 questions rather
    than being cancelled outright (a same-question double call in both orders would do that, at 2x cost --
    a stated cost/rigor tradeoff, not hidden)."""
    rng = random.Random(seed * 7919 + 1)
    out = {}
    for opp in ("single", "temp_vote_null", "shuffled_dial_null"):
        wins = losses = ties = parsefail = 0
        rows = []
        for qi, q in enumerate(questions):
            p_ans = judge_results["parliament"]["final_answers"][qi]
            o_ans = judge_results[opp]["final_answers"][qi]
            p_is_a = rng.random() < 0.5
            a_text, b_text = (p_ans, o_ans) if p_is_a else (o_ans, p_ans)
            raw = jrig.gen(build_pref_prompt(q["q"], a_text, b_text), max_new=_PREF_MAX_NEW, sample=False)
            pick = parse_pref(raw)
            if pick is None:
                parsefail += 1
                rows.append("parsefail")
            elif pick == "TIE":
                ties += 1
                rows.append("tie")
            else:
                parliament_won = (pick == "A") == p_is_a
                wins += int(parliament_won)
                losses += int(not parliament_won)
                rows.append("parliament" if parliament_won else opp)
        decided = wins + losses
        out[f"parliament_vs_{opp}"] = {
            "parliament_wins": wins, "opponent_wins": losses, "ties": ties, "parse_fail": parsefail,
            "parliament_winrate": round(wins / decided, 3) if decided else None,
            "rows": rows,
        }
    return out


# ============================================================================================ judge phase
def judge_phase(jrig: Rig, questions: list[dict], arms: dict, consistency_check: bool = True,
                on_progress=None) -> dict:
    """For every question x arm: coherence-filter the K raw decodes, judge-MERGE the survivors into one
    final answer (merge_candidates), then score that final answer against the question's rubric
    (build_judge_prompt/parse_bits). If consistency_check, every rubric score is ALSO computed a second
    time (same prompt, temperature-sampled instead of greedy) purely to measure the judge's own bit-flip
    (disagreement) rate -- reported per arm, never used to pick a score."""
    out = {name: {"final_answers": [], "merge_meta": [], "bits": [], "coverage": [], "parse_notes": []}
           for name in arms}
    flip_counts = {name: [] for name in arms}

    for qi, q in enumerate(questions):
        for name, arm in arms.items():
            k_texts = arm["raw"][qi]
            coh = [_coherence(t) for t in k_texts]
            survivors = [t for t, c in zip(k_texts, coh) if not c["degenerate"]]
            merged, meta = merge_candidates(jrig, q["q"], survivors)
            meta.update(qid=q.get("id"), k_total=len(k_texts),
                        k_degenerate=sum(c["degenerate"] for c in coh),
                        merged_degenerate=_coherence(merged)["degenerate"])

            prompt = build_judge_prompt(q["q"], q["points"], merged)
            raw1 = jrig.gen(prompt, max_new=_SCORE_MAX_NEW, sample=False)
            bits, note = parse_bits(raw1, len(q["points"]))

            out[name]["final_answers"].append(merged)
            out[name]["merge_meta"].append(meta)
            out[name]["bits"].append(bits)
            out[name]["coverage"].append(coverage_pct(bits))
            out[name]["parse_notes"].append(note)

            if consistency_check:
                raw2 = jrig.gen(prompt, max_new=_SCORE_MAX_NEW, sample=True, temperature=0.7)
                bits2, _note2 = parse_bits(raw2, len(q["points"]))
                if bits is not None and bits2 is not None:
                    flips = sum(int(x != y) for x, y in zip(bits, bits2))
                    flip_counts[name].append((flips, len(bits)))

        print(f"  [judge] ... {qi + 1}/{len(questions)} questions merged+scored", flush=True)
        if on_progress and ((qi + 1) % 5 == 0 or qi == len(questions) - 1):
            on_progress(qi, out)

    for name in arms:
        covs = [c for c in out[name]["coverage"] if c is not None]
        out[name]["coverage_mean"] = round(_mean(covs), 1) if covs else None
        notes = out[name]["parse_notes"]
        out[name]["parse_fail_rate"] = (round(_mean(n.startswith("parse-fail") for n in notes), 3)
                                        if notes else 0.0)
        fc = flip_counts[name]
        if fc:
            tf, tb = sum(f for f, _ in fc), sum(b for _, b in fc)
            out[name]["consistency_flip_rate"] = round(tf / max(1, tb), 3)
        else:
            out[name]["consistency_flip_rate"] = None
    return out


def judge_trust_report(jres: dict) -> dict:
    """Calibrate judge trust WITH THE NULLS, per the pre-reg: a judge that rates the shuffled-dial null
    highly -- above the single-decode floor, or not clearly below parliament -- is untrustworthy, and this
    says so explicitly rather than reporting coverage numbers at face value."""
    cov = {name: jres.get(name, {}).get("coverage_mean") for name in ARMS_ORDER}
    single, shuf = cov["single"], cov["shuffled_dial_null"]
    tvote, parl = cov["temp_vote_null"], cov["parliament"]
    notes = []
    trustworthy = True
    if single is not None and shuf is not None and shuf > single + _TRUST_TOL:
        trustworthy = False
        notes.append(f"UNTRUSTWORTHY: shuffled-dial null scored {shuf} > single-decode floor {single} -- "
                     f"a judge that rates RANDOM-direction steering above the unsteered floor cannot be "
                     f"trusted to read this rubric.")
    if shuf is not None and parl is not None and shuf >= parl - _TRUST_TOL:
        trustworthy = False
        notes.append(f"UNTRUSTWORTHY: shuffled-dial null ({shuf}) is not clearly beaten by parliament "
                     f"({parl}) -- the judge cannot tell directed steering from random perturbation.")
    if tvote is not None and parl is not None and tvote > parl + _TRUST_TOL:
        notes.append(f"NOTE (not disqualifying): temp-vote null ({tvote}) outscored parliament ({parl}) -- "
                     f"thermal diversity alone beat directed diversity here; a real finding worth "
                     f"reporting, not a judge-trust failure by itself.")
    if not any(n.startswith("UNTRUSTWORTHY") for n in notes):
        notes.insert(0, "nulls behave as expected: the judge does not rate the randomly-steered arm above "
                        "the floor or indistinguishably from parliament.")
    return {"coverage_by_arm": cov, "trustworthy": trustworthy, "notes": notes}


# ================================================================================================= run
def run(model_name: str, judge_model: str = "auto", n_questions: int = 30,
        out_path: str = "research/runs/parliament.json", four_bit_override: str = "auto",
        smoke: bool = False, seed: int = 0, layer: int | None = None, max_new: int = 180,
        consistency_check: bool = True, pairwise: bool = True) -> dict:
    torch.manual_seed(seed)
    n = 4 if smoke else min(n_questions, len(QUESTION_BANK))
    if not smoke and n_questions > len(QUESTION_BANK):
        print(f"[note] --questions {n_questions} > bank size {len(QUESTION_BANK)}; using {n}", flush=True)
    questions = QUESTION_BANK[:n]

    rig = Rig(model_name, four_bit_override)
    sc = SingleTurnSteer(rig.model, rig.tok, layer=layer)
    print(f"[steer] computing the 5 stance directions at layer {sc.layer} ...", flush=True)
    steer_info = compute_stances(sc)
    print(f"[steer] {steer_info}", flush=True)

    res = {
        "model": model_name, "four_bit": rig.four_bit, "seed": seed, "smoke": smoke,
        "n_questions": len(questions), "max_new": max_new,
        "steer_layer": sc.layer, "steer_info": steer_info,
        "stances": {name: {"pos": (steering_mod.AXES.get(name) or sc.custom.get(name, {})).get("pos"),
                            "neg": (steering_mod.AXES.get(name) or sc.custom.get(name, {})).get("neg"),
                            "max": axis_max_of(sc, name)}
                    for name in STANCES},
        "questions": questions, "calibration": {},
    }
    _save(out_path, res)

    print("[calibrate] per-axis dose sweep + steering-liveness check ...", flush=True)
    shuffled_vecs = {}
    t0 = time.time()
    for axis in STANCES:
        report, svec = calibrate_and_check_liveness(rig, sc, axis, seed=seed, smoke=smoke)
        res["calibration"][axis] = report
        shuffled_vecs[axis] = svec
        _save(out_path, res)
        print(f"  [{axis}] dose={report['chosen_strength']} (frac {report['chosen_frac']} of max "
              f"{report['axis_max']}) live={report['live']}", flush=True)
    res["calibration_wall_clock_sec"] = round(time.time() - t0, 1)

    live_axes = [a for a in STANCES if res["calibration"][a]["live"]]
    res["liveness_summary"] = {
        "live_axes": live_axes, "dead_axes": [a for a in STANCES if a not in live_axes],
        "n_live": len(live_axes), "n_total": len(STANCES),
        "verdict": "yes" if len(live_axes) >= 4 else ("partial" if live_axes else "no"),
    }
    print(f"[liveness] steering works on this model: {res['liveness_summary']['verdict']} "
          f"({len(live_axes)}/{len(STANCES)} axes live)", flush=True)
    _save(out_path, res)

    doses = {a: res["calibration"][a]["chosen_strength"] for a in STANCES}

    print(f"[arms] generating decodes for {len(questions)} questions x 4 arms ...", flush=True)
    raw = {name: [] for name in ARMS_ORDER}
    t0 = time.time()
    for qi, q in enumerate(questions):
        qtext = q["q"]

        parl = []
        for axis in STANCES:
            sc.clear(); sc.set(axis, doses[axis]); sc.engage()
            parl.append(rig.gen(qtext, max_new=max_new, sample=False))
            sc.disengage()
        raw["parliament"].append(parl)

        sc.disengage(); sc.clear()
        raw["single"].append([rig.gen(qtext, max_new=max_new, sample=False)])

        raw["temp_vote_null"].append([rig.gen(qtext, max_new=max_new, sample=True, temperature=0.9)
                                       for _ in STANCES])

        shuf = []
        for axis in STANCES:
            sc.clear()
            sc.vecs["_shuf"] = shuffled_vecs[axis]
            sc.strength["_shuf"] = doses[axis]
            sc.engage()
            shuf.append(rig.gen(qtext, max_new=max_new, sample=False))
            sc.disengage()
            sc.vecs.pop("_shuf", None); sc.strength.pop("_shuf", None)
        raw["shuffled_dial_null"].append(shuf)

        if (qi + 1) % 5 == 0 or qi == len(questions) - 1:
            res["arms"] = {name: {"raw": raw[name]} for name in ARMS_ORDER}
            _save(out_path, res)
            print(f"  ... {qi + 1}/{len(questions)} questions decoded", flush=True)

    res["arms"] = {name: {"raw": raw[name],
                           "degenerate_rate": round(_mean(degenerate_rate(ks) for ks in raw[name]), 3)}
                   for name in ARMS_ORDER}
    res["arms_wall_clock_sec"] = round(time.time() - t0, 1)
    _save(out_path, res)

    print("[free] releasing the subject model before loading the judge (single 16GB card) ...", flush=True)
    sc.disengage()
    del sc, rig
    _free_cuda()

    jname = default_judge_for(model_name, smoke) if judge_model == "auto" else judge_model
    res["judge_model"] = jname
    res["judge_independent"] = (jname.strip().lower() != model_name.strip().lower())
    if not res["judge_independent"]:
        print(f"[warn] judge model == subject model ({jname}) -- this is NOT an independent judge; "
              f"read the coverage numbers with that in mind.", flush=True)
    jrig = Rig(jname, four_bit_override)

    print("[judge] merge + score + consistency-check every final answer ...", flush=True)

    def _judge_ckpt(_qi, partial):
        res["judge"] = partial
        _save(out_path, res)

    t0 = time.time()
    jres = judge_phase(jrig, questions, res["arms"], consistency_check=consistency_check,
                       on_progress=_judge_ckpt)
    res["judge"] = jres
    res["judge_wall_clock_sec"] = round(time.time() - t0, 1)
    res["judge_trust"] = judge_trust_report(jres)
    _save(out_path, res)

    if pairwise:
        print("[pairwise] soft preference corroboration (parliament vs single/temp-vote/shuffled) ...",
              flush=True)
        res["pairwise"] = pairwise_phase(jrig, questions, jres, seed=seed)
        _save(out_path, res)

    jrig.free()
    del jrig
    _free_cuda()

    _summary(res)
    print(f"\nsaved -> {out_path}", flush=True)
    return res


def _save(out_path, res):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def _summary(res):
    print("\n" + "=" * 78, flush=True)
    print(f"PARLIAMENT OF STANCES -- {res['model']} ({'nf4' if res['four_bit'] else 'bf16'}) "
          f"judge={res.get('judge_model', '?')} (independent={res.get('judge_independent')})", flush=True)
    ls = res["liveness_summary"]
    print(f"steering-liveness: {ls['verdict']} ({ls['n_live']}/{ls['n_total']} axes live: "
          f"{ls['live_axes']}; dead: {ls['dead_axes']})", flush=True)
    print(f"\n{'arm':22} {'coverage':9} {'degen%':8} {'parse-fail%':12} {'judge-flip%':11}", flush=True)
    jr = res.get("judge", {})
    for name in ARMS_ORDER:
        j = jr.get(name, {})
        a = res["arms"].get(name, {})
        cfr = j.get("consistency_flip_rate")
        print(f"{_ARM_LABEL[name]:22} {str(j.get('coverage_mean')):9} "
              f"{a.get('degenerate_rate', 0):<8.1%} {j.get('parse_fail_rate', 0):<12.1%} "
              f"{'-' if cfr is None else f'{cfr:.1%}':11}", flush=True)
    jt = res.get("judge_trust", {})
    print(f"\njudge trust: {'OK' if jt.get('trustworthy') else 'SUSPECT'}", flush=True)
    for n in jt.get("notes", []):
        print(f"  - {n}", flush=True)
    pw = res.get("pairwise")
    if pw:
        print("\npairwise (SOFT signal -- single-pass, same judge model, not the primary metric):", flush=True)
        for k, r in pw.items():
            print(f"  {k}: parliament winrate={r['parliament_winrate']} "
                  f"(W{r['parliament_wins']}/L{r['opponent_wins']}/T{r['ties']}/parsefail{r['parse_fail']})",
                  flush=True)


def compare(paths):
    """Cross-family table from >=2 per-model JSONs. No GPU work -- pure read + print (the findings-doc
    source), matching mirror_bench.py's compare()."""
    runs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            runs.append(json.load(f))
    names = [r["model"].split("/")[-1] for r in runs]
    print("\n" + "=" * 78)
    print("CROSS-FAMILY PARLIAMENT-OF-STANCES")
    print(f"{'':22} " + " ".join(f"{n[:24]:26}" for n in names))
    for arm in ARMS_ORDER:
        cells = []
        for r in runs:
            j = r.get("judge", {}).get(arm, {})
            a = r.get("arms", {}).get(arm, {})
            cov = j.get("coverage_mean")
            dg = a.get("degenerate_rate", 0)
            cells.append(f"cov={cov} degen={dg:.0%}")
        print(f"{_ARM_LABEL[arm]:22} " + " ".join(f"{c:26}" for c in cells))
    print("\nsteering liveness:")
    for r, n in zip(runs, names):
        ls = r.get("liveness_summary", {})
        print(f"  {n:26} {ls.get('verdict')} ({ls.get('n_live')}/{ls.get('n_total')}) "
              f"live={ls.get('live_axes')}")
    print("\njudge trust:")
    for r, n in zip(runs, names):
        jt = r.get("judge_trust", {})
        print(f"  {n:26} {'OK' if jt.get('trustworthy') else 'SUSPECT'} "
              f"(judge={r.get('judge_model')}, independent={r.get('judge_independent')})")
    print("\npairwise (soft, parliament winrate vs each opponent):")
    for r, n in zip(runs, names):
        pw = r.get("pairwise") or {}
        cells = {k: v.get("parliament_winrate") for k, v in pw.items()}
        print(f"  {n:26} {cells}")
    print("\nParliament beating single AND both nulls, on BOTH families, with a TRUSTWORTHY judge is the "
          "strongest possible finding. Any family flagged SUSPECT means that family's coverage numbers are "
          "not to be trusted at face value -- read the coherence axis and the pairwise/liveness rows "
          "instead of the headline coverage_mean.")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--judge-model", default="auto",
                    help="'auto' -> cross-family default (see default_judge_for); or an explicit model id")
    ap.add_argument("--questions", type=int, default=30, help="how many of QUESTION_BANK to use")
    ap.add_argument("--out", default="research/runs/parliament.json")
    ap.add_argument("--four-bit", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--layer", type=int, default=None, help="steering layer override (default num_layers//2)")
    ap.add_argument("--max-new", type=int, default=180, help="max new tokens per arm decode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="4 questions, small same-family judge -- prove the wiring cheaply")
    ap.add_argument("--no-consistency-check", dest="consistency_check", action="store_false",
                    help="skip the second (sampled) judge pass used only for the disagreement-rate report")
    ap.add_argument("--no-pairwise", dest="pairwise", action="store_false",
                    help="skip the soft pairwise-preference bonus leg")
    ap.add_argument("--compare", nargs="+", metavar="RUN.json", help="print the cross-family table from >=2 run JSONs")
    return ap


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.compare:
        compare(args.compare)
    else:
        run(args.model, judge_model=args.judge_model, n_questions=args.questions, out_path=args.out,
            four_bit_override=args.four_bit, smoke=args.smoke, seed=args.seed, layer=args.layer,
            max_new=args.max_new, consistency_check=args.consistency_check, pairwise=args.pairwise)
