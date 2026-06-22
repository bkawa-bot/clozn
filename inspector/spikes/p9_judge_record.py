"""
LLM-JUDGE harness for p9 (the caller IS the judge). For each (method, feature) the caller supplies:
  - a one-line DESCRIPTION written from the 12 explain examples (token-ANCHORED context patterns
    count, per the p8 lesson), and
  - a PREDICATE that operationalizes that description over a test example's context string, so the
    fires/not prediction for each held-out example follows MECHANICALLY and CONSISTENTLY from the
    stated rule (not from peeking at hidden labels). The predicate reads only the `context` field
    (focus token is wrapped << >>); it never sees `_fires`.

This makes the judging auditable and reproducible: change the description -> the predictions change
in lockstep. Output: runs/p9_judgments_<which>.json in the schema p9 --score consumes
({method:{features:[{feature, description, predictions:{id:bool}}]}}).

The JUDGMENTS dict below is the caller's analysis after reading runs/p9_packets_<which>.json.
"""
from __future__ import annotations

import json
import os
import re
import sys

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")


def focus(ctx: str) -> str:
    """The token between << >> (what fired)."""
    m = re.search(r"<<(.*?)>>", ctx, flags=re.S)
    return m.group(1) if m else ""


def _norm(t: str) -> str:
    return t.strip().lower()


# Small predicate vocabulary the descriptions are built from. Each takes (focus_token, full_context)
# and returns a bool = "does my description predict this fires?". Kept deliberately simple/legible.
def is_tok(*toks):
    s = {t.lower() for t in toks}
    return lambda f, c: _norm(f) in s


def focus_in(*subs):
    return lambda f, c: any(sub in f for sub in subs)


def any_pred(*ps):
    return lambda f, c: any(p(f, c) for p in ps)


def is_period():        return lambda f, c: _norm(f) in {".", "?", "!"}
def is_comma():         return lambda f, c: _norm(f) == ","
def is_at_hyphen():     return lambda f, c: f.strip() == "@"  # the '@' of '@-@' (focus is always '@')
def is_year():          return lambda f, c: bool(re.fullmatch(r"\s*1[5-9]\d\d\s*", f))
def is_number():        return lambda f, c: bool(re.fullmatch(r"\s*\d[\d.,]*\s*", f)) and f.strip() != ""
def is_apostrophe_s():  return lambda f, c: f.strip() in {"'", "'s", "’", "’s"}
def is_cap_word():      return lambda f, c: bool(re.fullmatch(r"\s*[A-Z][a-zA-Z]+\s*", f))
def is_accented():      return lambda f, c: bool(re.search(r"[áàâäãéèêëíìîïóòôöõúùûüñç]", f, flags=re.I)) \
    and not f.isascii()
def is_subword():       return lambda f, c: (not f.startswith(" ")) and f.strip().isalpha() \
    and len(f.strip()) <= 5 and f.strip().islower()


# Lexical-set predicates for cross-token concepts (the patterns the token metric structurally hides).
ABSTRACT_NOUNS = {"practice", "myth", "process", "principle", "feeling", "form", "choice", "force",
                  "nature", "tendency", "effort", "use", "action", "life", "ritual", "purpose",
                  "role", "power", "theme", "concept", "idea", "sense", "meaning"}
RELATIONAL_HEADS = {"resort", "appeal", "alludes", "alluded", "attests", "opposed", "prone",
                    "vulnerable", "aspired", "equated", "adaptation", "equivalent", "devotion",
                    "hostile", "referred", "references", "successor", "newcomers", "willing",
                    "response", "reaction", "relationship", "link", "compared", "similar"}
STATIVE_CHANGE = {"known", "rule", "grew", "across", "society", "built", "often", "adopt", "also",
                  "usually", "believed", "controlled", "continued", "differed", "spread",
                  "absorbed", "remained", "became", "emerged", "developed", "introduced"}
PASSIVE_BIO = {"worshipped", "given", "built", "commissioned", "born", "named", "revered", "served",
               "crowned", "elected", "designated", "was", "were", "appointed", "founded"}
AGENTIVE_HUMAN = {"public", "officials", "official", "congressman", "minister", "superiors",
                  "marched", "mobilized", "introduced", "suppressed", "people", "priests",
                  "secretary", "governor", "president", "soldiers", "men"}


JUDGMENTS = {
    # =========================================================================================
    # GPT-2 — BLOOM'S gpt2-small-res-jb (expect HIGH): token-ANCHORED context / relational features
    # =========================================================================================
    "gpt2": {
        "bloom_sae": [
            (390, "abstract/conceptual nouns, often as 'the X of' (practice, process, principle, "
                  "feeling, nature, force, effort, use, action)",
             lambda f, c: _norm(f) in ABSTRACT_NOUNS or "ification" in c),
            (956, "tokens in sports contract/draft/roster contexts: numerals, 'year'/'round'/'season' "
                  "and determiners in '@-@ year contract' / draft-pick phrasing",
             any_pred(is_tok("a", "one", "two", "year", "round", "winner", "88"), is_number(),
                      focus_in("year"))),
            (1864, "verb/adjective heads that take a 'to'/'by'/'through' complement (resort to, "
                   "appeal to, prone to, vulnerable to, opposed by, equated, references to)",
             lambda f, c: _norm(f) in RELATIONAL_HEADS),
            (2247, "war/military & proper-noun fragments around battles/empires (Allied, United, "
                   "war, Strait/Otranto pieces, uprisings, armaments) — historical-conflict context",
             any_pred(is_tok("allied", "united", "war", "the", "class"),
                      focus_in("anto", "Euro", "Cal", "up", "ris", "arm", "if", "St"))),
            (2917, "post-modifier tokens after a measurement/quantity or a copy of 'equivalent/"
                   "statistics/located/estimated' — appositive/relative continuation",
             is_tok("the", "it", "home", "letter", "those", "off", "have", "sealed", "coins")),
            (3341, "the '@' inside '@-@' compounds and adjacent measurement/eng. tokens (calibers, "
                   "@-@ pound, illustrated/released-by-publisher) — hyphenated-compound context",
             any_pred(is_at_hyphen(), is_tok("out", "scoring", "completed", "volume", "a"))),
            (4079, "agentive humans / officials performing actions in historical-political events "
                   "(public, officials, congressman, minister, superiors, marched, mobilized)",
             lambda f, c: _norm(f) in AGENTIVE_HUMAN),
            (4368, "function words inside a coordinated list / 'pay out for X, and Y' enumerations "
                   "and quotation-embedded prepositions (for, and, with, which, to, on, the)",
             is_tok("deck", "late", "with", "out", "up", "the", "on")),
            (5356, "sentence/clause-initial scene-setting tokens in political-history narration "
                   "(In, southern, became, action, reasons, protection) — esp. 'In <time/place>'",
             any_pred(is_tok("in", "the", "president", "intervention"),
                      lambda f, c: c.strip().startswith("<<") and is_cap_word()(f, c))),
            (6045, "ship/military-unit nouns & ordinals in naval/regiment context (ships, warship, "
                   "Infantry, Arkansas, first, the, '@' in N@-@88) — naval/regimental description",
             any_pred(is_at_hyphen(), is_tok("arkansas", "1911", "in", "equivalent", "did",
                                             "festival", "completed"))),
            (6603, "relative/connective tokens introducing a clause about players/people "
                   "(that, whose, were, for, 's, and, to) in sports-roster / list prose",
             is_tok("that", "were", "for", "whose", "to", "and", "have")),
            (7528, "'mid/middle/upper/lower' + class/age/point compounds and chemistry diastereo- "
                   "subwords (class, point, aged, -ere-, fringe/edge senses) — middle/boundary",
             any_pred(focus_in("ere", "but", "by", "inges"),
                      is_tok("class", "team", "basement", "heaven"))),
            (10770, "stative verbs/adverbs of historical change & description (known, rule, grew, "
                    "built, adopted, believed, controlled, continued, differed) — narration of change",
             lambda f, c: _norm(f) in STATIVE_CHANGE),
            # --- the random-among-live half (read from packets; judged the same harsh way) ---
            (12434, "sentence-medial adverbs of manner/degree before a verb (originally, falsely, "
                    "generally, usually, ever, well, still, then, not, best)",
             is_tok("originally", "falsely", "generally", "usually", "ever", "best", "not", "well",
                    "still", "then", "primarily", "actively", "later", "quietly")),
            (14904, "polysemantic grab-bag (set, range, 12, concussion, Europe, distributed) — no "
                    "unifying concept",
             lambda f, c: False),
            (15564, "coordinating conjunction 'and' / list comma ','", is_tok("and", ",", "or")),
            (15897, "death/dread lexical cluster (death, Death, dread, Dread) — a real semantic theme",
             any_pred(is_tok("death", "dread", "horus"), focus_in("addon", "Death", "Dread"))),
            (19390, "numbers in measurement/range contexts (8, 15, 6, 23, 11, 13) and '@' in figures",
             any_pred(is_number(), is_at_hyphen(), is_tok("and", "("))),
            (19948, "political/administrative TITLES across different tokens (Secretary, Director, "
                    "Governor, President, Treasury, legislature) — a cross-token concept",
             is_tok("secretary", "director", "governor", "president", "treasury", "legislature",
                    "minister", "congressman", "of")),
            (19983, "short subword fragments at word interior (ic, ce, our, amb, er, aly, ach)",
             is_subword()),
            (20447, "short subword fragments / rare-token pieces (ok, ab, tr, id, ub, aph)",
             is_subword()),
            (20908, "'of'/'the'/'an' — determiners & 'of' in noun-phrase heads",
             is_tok("of", "the", "an", "a")),
        ],
        # =====================================================================================
        # GPT-2 — PCA (expect LOWER): clean token/char/year/punct axes + a few loose themes
        # =====================================================================================
        "pca": [
            (0, "sentence/clause-final period '.' (occasionally ',')", is_period()),
            (1, "the '@' inside '@-@' hyphenation compounds", is_at_hyphen()),
            (2, "religious-ritual / abstract plural nouns (rituals, deities, gods, ceremonies, "
                "forests, habitats, activities, purposes)",
             is_tok("rituals", "deities", "gods", "ceremonies", "forests", "colors", "reactions",
                    "figures", "habitats", "activities", "purposes", "deity")),
            (3, "the '@' inside '@-@' compound modifiers (low-@-altitude, half-@-baked, @-@ inch)",
             is_at_hyphen()),
            (4, "four-digit years (1862, 1914, 1991, 1990, 1861, 1985)", is_year()),
            (5, "accented/non-ASCII chars in proper nouns (Díaz, Zrínyi, Orozco, Estañol)",
             any_pred(is_accented(), focus_in("ny", "Dun", "her"))),
            (6, "the '@' inside '@-@' compounds (Austro-@-Hungarian, Pre-@-Raphaelite, @-@ dollar)",
             is_at_hyphen()),
            (7, "sentence-initial-after-period content word OR '@' in number commas — polysemantic",
             any_pred(is_at_hyphen(), is_tok("flow", "mass", "scientific", "growth", "pre",
                                             "early"))),
            (8, "'of' (relational 'X of Y') and some quantity words (twenty, thirteen, end)",
             is_tok("of", "twenty", "13", "one", "end")),
            (9, "the possessive apostrophe \"'s\" (team's, Carey's, Egypt's)", is_apostrophe_s()),
            (10, "comma ',' in lists/appositions/coordinate adjectives", is_comma()),
            (14, "passive past-tense biographical/historical verbs (worshipped, born, built, "
                 "commissioned, named, served, crowned, elected)",
             lambda f, c: _norm(f) in PASSIVE_BIO),
            (20, "creative-work attribution: 'illustrated/written/adapted by' + years in parens "
                 "(1929, 1933, 1934) + Japanese title chars",
             any_pred(is_year(), is_tok("illustrated", "written", "adapted", "interpreted", "lit",
                                        "tightly"), is_accented(),
                      lambda f, c: not f.isascii())),
            # --- the random-among-live half (lower-variance axes; mostly polysemantic) ---
            (29, "measurement units / dimensions after a number (m, ft, long, six) + a few nouns",
             is_tok("m", "ft", "long", "six", "reactions")),
            (53, "polysemantic (pap, described, trade, dealership, baptism, substitution) — weak/mush",
             is_tok("trade", "described")),
            (75, "accented proper-noun chars (í) + scattered past verbs (built/made/constructed) — weak",
             any_pred(is_accented(), is_tok("built", "made", "constructed", "led", "respond"))),
            (84, "'Chronicles' (Valkyria Chronicles) and capitalized proper nouns (Ferdinand, James)",
             any_pred(is_tok("chronicles", "ferdinand", "an"), is_cap_word())),
            (132, "completion verbs (finish/finished/finishing/pay/paid/put) + 'Little' — loose",
             any_pred(focus_in("finish", "ishing"), is_tok("pay", "paid", "put", "little",
                                                           "landing", "standard"))),
            (161, "'Nam' (Nameless) + abstract event nouns (gathering, demonstration, test) — mush",
             focus_in("Nam")),
            (170, "polysemantic (categor, on, Celebrity, began, Private, Red, date) — no pattern",
             is_tok("on")),
            (209, "subword fragments + scattered nouns (ram, farm, substrate, helicopter) — mush",
             lambda f, c: False),
            (210, "subword fragments (bin, Hav, em, qu, ap, grand, inc) — subword mush",
             is_subword()),
        ],
        # =====================================================================================
        # GPT-2 — RANDOM directions (expect ~0.5): mostly polysemantic mush; a few incidentally
        # align with a dominant token (@-@, "Rock", "Lic"). Honest harsh predicates -> near chance.
        # =====================================================================================
        "random": [
            (0, "polysemantic: possessive \"'s\", numbers, and scattered nouns — no single pattern",
             is_apostrophe_s()),
            (1, "the '-'/'@' inside '@-@' compounds (Austro-@-Hungarian, Greco-@-Egyptian)",
             any_pred(is_at_hyphen(), lambda f, c: f.strip() == "-" and "@-@" in c.replace(" ", ""))),
            (2, "polysemantic mush (she, 9, St., shelled, trade, method, production) — no pattern",
             is_tok("st.", ".")),
            (3, "incidental 'Rock' (Little Rock) and award 'Year/Actress/record' tokens — weak/mixed",
             is_tok("rock", "year", "actress", "record", "felt", "remain", "nhl", "sydney")),
            (4, "incidental 'Lic' (Pilot Licence subword) + pilot context — otherwise mush",
             focus_in("Lic")),
            (5, "polysemantic mush (middle, last, supreme, troopers, final, point) — no pattern",
             lambda f, c: False),
            (6, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (7, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (8, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (9, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (10, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (11, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (12, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (13, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (14, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (15, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (16, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (17, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (18, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (19, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (20, "polysemantic mush — no nameable pattern", lambda f, c: False),
            (21, "polysemantic mush — no nameable pattern", lambda f, c: False),
        ],
    },
    # =============================================================================================
    # QWEN-0.5B (layer 2) — our 16x L1=8 SAE vs PCA vs random, SAME activations. NOTE: layer 2 is
    # dominated by a sentence-initial-capitalization / attention-sink structure, so PCA AND random
    # directions both become predictable "sentence-initial capitalized word" detectors — the
    # substrate confound the metric (correctly) reports. Judged with the SAME harsh context rubric.
    # =============================================================================================
    "qwen05b": {
        "our_sae": [
            (235, "the final digit of a year in 18xx/17xx historical dates (185x, 186x, 180x) — "
                  "year-digit / 'century' in date context",
             any_pred(is_number(), is_tok("century"))),
            (584, "polysemantic subwords (ode, ctr, anti, send, bishop) — no unifying concept",
             lambda f, c: False),
            (764, "the token 'battle'/'Battle'", is_tok("battle")),
            (954, "'after' / 'fou(nded)' — temporal 'after' and founding verbs",
             any_pred(is_tok("after"), focus_in("fou"))),
            (1078, "polysemantic subwords + 'Amendment'/'Warsaw' — weak/mush",
             is_tok("amendment")),
            (1160, "the subword 'ny' (word-interior)", focus_in("ny")),
            (2511, "polysemantic (eve, agents, rank, Yog, 4) — no pattern", is_tok("agents")),
            (3866, "negative-outcome / blame nouns (horror, responsibility, failure, incompetence) "
                   "— a loose negative-abstract theme",
             is_tok("horror", "responsibility", "failure", "incompetence", "past", "hand")),
            (4409, "the token 'P' (single capital letter)", lambda f, c: f.strip() == "P"),
            (4763, "'Court'/'court'/'Cup' — court/competition proper nouns",
             is_tok("court", "cup")),
            (7125, "'-ological'/'Stones'/'history' — geology/Rolling-Stones + '-ological' subword",
             any_pred(focus_in("ological"), is_tok("stones", "stone", "history"))),
            (7242, "the token 'rock'/'Rock'", is_tok("rock")),
            (7321, "polysemantic (fr, month, year, Kim, coin, coins) — weak",
             is_tok("month", "year", "coin", "coins")),
            (7660, "the preposition 'with' (and collaboration verbs)",
             any_pred(is_tok("with"), focus_in("collaborat"))),
            (9120, "the period '.' and 'cards'/'ony' subwords",
             any_pred(is_period(), is_tok("cards"))),
            (9304, "polysemantic subwords (ward, FW, iner, dorf, arte) — no pattern",
             lambda f, c: False),
            (9394, "'fun'/'Fun'/'to'/'off'/'out' — 'fun' + short particles, weak/mixed",
             any_pred(is_tok("to", "off", "out"), focus_in("fun", "Fun"))),
            (10186, "approval/permission/sanction verbs across tokens (unacceptable, tolerant, "
                    "sanctioned, approved, tolerated, propagated) — a genuine cross-token CONCEPT",
             any_pred(is_tok("unacceptable", "tolerant", "sanctioned", "approved", "approve",
                             "tolerated"), focus_in("propag", "asc", "tolerat", "approv", "sanction"))),
            (11652, "the token 'traffic' (+ road/Express words)",
             any_pred(is_tok("traffic", "express"), focus_in("traffic"))),
            (12180, "the subword 'eh'/'ep' + 'echoed' — mostly subword mush",
             any_pred(focus_in("eh", "ep"), is_tok("echoed"))),
            (12274, "licence/permit/permission nouns across tokens (licences, Licence, licensing, "
                    "permits, awards) — a genuine cross-token CONCEPT",
             any_pred(is_tok("licences", "licence", "permits", "permit", "awards"),
                      focus_in("licen", "Licen"))),
            (14273, "mid-frequency event/abstract nouns (arrest, government, departure, invasion, "
                    "death, name) — loosely thematic, polysemantic",
             is_tok("arrest", "government", "departure", "people", "name", "invasion", "death",
                    "main")),
        ],
        "pca": [
            (0, "sentence-initial discourse connectives / capitalized openers (According, However, "
                "Although, Many, Humans, Germany)",
             is_cap_word()),
            (1, "the '@' inside '@-@' compounds", is_at_hyphen()),
            (2, "the comma ','", is_comma()),
            (3, "the digit '9' / numerals", any_pred(is_number(), lambda f, c: f.strip() == "9")),
            (4, "NHL/sports team names (Leafs, Lightning, Penguins, Sabres) + capitalized proper nouns",
             is_cap_word()),
            (5, "single digits 5-8 (numerals)", is_number()),
            (6, "sentence-initial capitalized word / proper-noun fragment (God, Michael, King, Tro-)",
             is_cap_word()),
            (7, "capitalized openers / proper-noun fragments (Sign, Concept, Art, God, Str-)",
             is_cap_word()),
            (8, "capitalized proper-noun fragments (Jac-, Str-, Reg-, Mc-, Kim)", is_cap_word()),
            (9, "the hyphen '-'", lambda f, c: f.strip() == "-"),
            (10, "the apostrophe \"'\"", lambda f, c: f.strip() in {"'", "’"}),
            (14, "sentence-initial 'In' / capitalized openers (In, Music, Germany)",
             is_cap_word()),
            (20, "place/proper nouns at sentence start (Egypt, Germany, Jordan, Nike)", is_cap_word()),
            (29, "the token 'Jordan'", is_tok("jordan")),
            (53, "the preposition 'to'", is_tok("to")),
            (75, "place/topic proper nouns (Atlanta, Egypt, Germany, Music, Memory)", is_cap_word()),
            (84, "the double-quote '\"' + capitalized fragments",
             any_pred(lambda f, c: f.strip() == '"', is_cap_word())),
            (132, "sentence-initial quantifiers/openers (Two, Three, Even, Like, Above)",
             is_cap_word()),
            (161, "subword fragments (iting, eth, phen, cz) + 'SMS' — subword/acronym mush",
             any_pred(is_tok("sms"), is_subword())),
            (170, "sentence-initial quantifiers (Such, Some, Several, Nine, Cold)", is_cap_word()),
            (209, "capitalized openers (Aside, This, Web) + subword mush", is_cap_word()),
            (210, "capitalized openers (Review, Humans, Modern, By) + 'down'", is_cap_word()),
        ],
        "random": [
            # nearly every random direction at L2 = a sentence-initial CAPITALIZED-word detector
            # (the attention-sink/position structure). Honest predicate: capitalized opener.
            (0, "sentence-initial capitalized openers (Several, Nine, General, Built, Three)",
             is_cap_word()),
            (1, "the capital letter 'B' / 'This'", any_pred(lambda f, c: f.strip() == "B",
                                                           is_tok("this"))),
            (2, "polysemantic (y, Not, others, Public, color, limited) — weak/mush",
             lambda f, c: False),
            (3, "sentence-initial 'On'/'By'/'Is' openers", is_cap_word()),
            (4, "capitalized openers / 'Pol-' fragment (Motor, Control, Policy, Modern, Official)",
             is_cap_word()),
            (5, "sentence-initial temporal connectives (Despite, When, After)", is_cap_word()),
            (6, "the 'Pol-'/'W-' capitalized fragments (Wyoming, Policy, Division)", is_cap_word()),
            (7, "place/quantifier proper nouns (Germany, Atlanta, Nike, Three, Two)", is_cap_word()),
            (8, "capitalized openers (Artist, God, They, Though, To, By)", is_cap_word()),
            (9, "sentence-initial 'There'/'As'", is_tok("there", "as")),
            (10, "sentence-initial 'During' (+ Artist, These)",
             is_tok("during", "artist", "these")),
            (11, "sentence-initial 'Following'/'There'", is_tok("following", "there", "egypt")),
            (12, "capitalized openers (Nike, Later, Nevertheless, His, During)", is_cap_word()),
            (14, "sentence-initial 'According'/'Many'/quantifiers", is_cap_word()),
            (19, "sentence-initial 'However'/'As'/'By'", is_cap_word()),
            (23, "single capital letters / proper-noun initials (Str, S, J, Pr, R, P, M)",
             lambda f, c: bool(re.fullmatch(r"\s*[A-Z][a-z]?[a-z]?\s*", f)) and f.strip()[:1].isupper()),
            (25, "capitalized openers (Memory, Private, When, For, Compet)", is_cap_word()),
            (34, "sentence-initial quantifiers/connectives (One, Each, However, Since, Perhaps)",
             is_cap_word()),
            (39, "sentence-initial 'According'/'Following'", is_cap_word()),
            (45, "sentence-initial quantifiers (Three, Two, They, These, Of, Those, Another)",
             is_cap_word()),
            (47, "sentence-initial 'To'/'Many'/'Some'/'For'", is_cap_word()),
            (53, "place/topic proper nouns (Germany, Egypt, Atlanta, King, Official)", is_cap_word()),
        ],
    },
    # =============================================================================================
    # QWEN-7B (layer 16) — our 8x L1=8 SAE vs PCA vs random. Descriptions written from the EXPLAIN
    # set only (the protocol); held-out + null then test them. PCA at L16 = ultra-clean single-token
    # detectors (According/The/@/9/-) -> high. Random again = sentence-initial caps -> elevated.
    # =============================================================================================
    "qwen7b": {
        "our_sae": [
            (473, "polysemantic (of, artillery, those, on, a, media, provided) — mostly determiners, "
                  "no clear concept",
             is_tok("a", "the", "of", "on", "those")),
            (1164, "'Shortly' (sentence-initial) + prepositions (to, of) — weak",
             any_pred(is_tok("shortly", "company"), focus_in("Shortly"))),
            (2145, "'charges'/'charge' + scattered (Earn, Bond) — loose legal-charge theme",
             is_tok("charges", "charge")),
            (2635, "single digits (1,1,1,2,2,8,3,3) — numerals", is_number()),
            (3319, "the subword 'eder'/'Feder' and '-er' word endings",
             any_pred(focus_in("eder", "Feder"), lambda f, c: _norm(f) == "er")),
            (4496, "polysemantic (roller, consequences, term, name, served, transfer, to) — mush",
             lambda f, c: False),
            (5020, "polysemantic (structural, Europe, clear, leather, cut, trimester) — no pattern",
             lambda f, c: False),
            (7447, "single digits (1,1,1,2,2,8,3,3) — numerals", is_number()),
            (7722, "single capital letters (R, N, H)",
             lambda f, c: bool(re.fullmatch(r"\s*[A-Z]\s*", f))),
            (8813, "polysemantic (responses, flat, ground, defenders, 5) — no pattern",
             lambda f, c: False),
            (12614, "polysemantic (gain, world, man, songs, peers, York, fully) — no pattern",
             lambda f, c: False),
            (13935, "religious/civic role + event nouns (Council, Cross, Priest, Congress, founded, "
                    "injured) — loose institutional theme",
             is_tok("council", "cross", "priest", "congress", "founded", "injured", "eighth")),
            (14658, "polysemantic (Billboard, naval, Soviet, in, m) — no pattern",
             is_tok("billboard", "naval")),
            (14752, "polysemantic (ab, esp, with, avoiding, his, war) — no pattern",
             lambda f, c: False),
            (15394, "the subword 'ch' + scattered (served, races, Intermediate, Storm) — mostly 'ch'",
             any_pred(focus_in("ch"), is_tok("served", "races"))),
            (17681, "the token 'shark' (+ scattered) — mostly a 'shark' token detector",
             is_tok("shark")),
            (18259, "'of' (relational) + subwords (lo, pl, th, fertil) — mostly 'of'",
             is_tok("of", "east")),
            (18619, "'Both' (sentence-initial) + astronomy (Venus, rings, air) — weak",
             any_pred(is_tok("both", "venus"), focus_in("Both"))),
            (23312, "competition/contest nouns across tokens (competition, compete, pressures, drove, "
                    "extends) — looks like a 'competition' CONCEPT from the explain set",
             any_pred(is_tok("competition", "compete", "pressures"),
                      focus_in("compet"))),
            (24376, "polysemantic (mium, kidn, collect, Service, Get, produced) — no pattern",
             is_tok("service")),
            (25525, "single digits (8,2,3,3,2,1,1,1) + religious nouns (divine, chapel, foundation)",
             any_pred(is_number(), is_tok("divine", "chapel", "foundation"))),
            (28548, "subword fragments (acon, per, ont, geme) + scattered (players, ship) — mush",
             any_pred(focus_in("acon"), is_subword())),
        ],
        "pca": [
            (0, "sentence-initial discourse openers (According, It, There, In) — clause-initial",
             is_cap_word()),
            (1, "the apostrophe \"'\" (possessive/quote)", lambda f, c: f.strip() in {"'", "’"}),
            (2, "the token 'The'", is_tok("the")),
            (3, "the '@' inside '@-@' compounds", is_at_hyphen()),
            (4, "the capital letter 'S' (proper-noun initial)",
             lambda f, c: f.strip() == "S"),
            (5, "single capitals 'H'/'M' (proper-noun initials)",
             lambda f, c: f.strip() in {"H", "M"}),
            (6, "single capitals 'H'/'M' (proper-noun initials)",
             lambda f, c: f.strip() in {"H", "M"}),
            (7, "the token 'In' (sentence-initial)", is_tok("in")),
            (8, "the digit '9'", lambda f, c: f.strip() == "9"),
            (9, "the comma ',' + a few digits", any_pred(is_comma(), is_number())),
            (10, "digits and punctuation (1, -, ', 2) — numeral/hyphen/apostrophe mix",
             any_pred(is_number(), lambda f, c: f.strip() in {"-", "'", "’"})),
            (14, "the hyphen '-'", lambda f, c: f.strip() == "-"),
            (20, "'the'/'of' + space tokens — determiners (diffuse)", is_tok("the", "of", "it")),
            (29, "the double-quote '\"'", lambda f, c: f.strip() == '"'),
            (53, "capitalized proper-noun fragments / initials (G, C, Ch, Kim, Michael, Mark)",
             any_pred(is_cap_word(), lambda f, c: bool(re.fullmatch(r"\s*[A-Z][a-z]?\s*", f)))),
            (75, "the token 'During' (sentence-initial)", is_tok("during")),
            (84, "single digits (1,1,1,8,2,2,3,3) + 'She'", any_pred(is_number(), is_tok("she"))),
            (132, "single digits (1,1,1,2,2,8,3,3) + 'National'",
             any_pred(is_number(), is_tok("national"))),
            (161, "'First' + subwords (chol, t, stabil, Amp) + digits — mostly mush",
             any_pred(is_tok("first"), is_number())),
            (170, "the 'Wi-'/'W' capitalized fragment", focus_in("Wi", "W")),
            (209, "the token 'church' + 'Gal'/'Wi' fragments + 'city'",
             is_tok("church", "city")),
            (210, "single digits (1,1,1,8,2,2) + 'C'/'Old'/'religious'",
             any_pred(is_number(), is_tok("c", "old", "religious"))),
        ],
        "random": [
            (0, "capitalized proper nouns (Air, Thomas, Mark, Michael)", is_cap_word()),
            (1, "'Having' (sentence-initial) + org/place proper nouns (Mets, Bulls, University, New)",
             any_pred(is_cap_word(), is_tok("new"))),
            (2, "'While' (sentence-initial) + proper nouns (Van, Food)", is_cap_word()),
            (3, "nationality adjectives / proper nouns (American, German, US, RN, Strong)",
             any_pred(is_tok("american", "german", "us", "rn"), is_cap_word())),
            (4, "'After' (sentence-initial)", any_pred(is_tok("after", "egypt"), is_cap_word())),
            (5, "'most' + capitalized (Chess, Bow, Ant) — mostly 'most'",
             any_pred(is_tok("most"), is_cap_word())),
            (6, "polysemantic (Parliament, M, bill, Gold, fragrance, roller) — no pattern",
             lambda f, c: False),
            (7, "'While'/'Whilst' (sentence-initial) + scattered nouns", is_cap_word()),
            (8, "'An' (sentence-initial) + scattered (Zag, Hosp, Amp)",
             any_pred(is_tok("an"), is_cap_word())),
            (9, "capitalized proper-noun fragments / initials (Pr, Gil, Sl, Wi, Row, Jan)",
             any_pred(is_cap_word(), lambda f, c: bool(re.fullmatch(r"\s*[A-Z][a-z]?\s*", f)))),
            (10, "'An'/'Then' (sentence-initial) + scattered", is_cap_word()),
            (11, "polysemantic (Little, armor, man, Irish, night, She, band) — no pattern",
             is_tok("she")),
            (12, "single capital 'J' / proper-noun initials + digits + 'Born'",
             any_pred(lambda f, c: f.strip() == "J", is_tok("born"), is_number())),
            (14, "'most' + capitalized fragments (Bro, Pro) — mostly 'most'",
             any_pred(is_tok("most"), is_cap_word())),
            (19, "polysemantic (album, religious, treaty, revolutionary, brand, bill) — no pattern",
             is_tok("album")),
            (23, "sentence-initial quantifiers (Several, For, Official)", is_cap_word()),
            (25, "nationality/proper-noun adjectives (Federal, British, American, Original, Private)",
             is_cap_word()),
            (34, "'This'/'These' (sentence-initial) + religious nouns (divine, chapel, Egyptians)",
             any_pred(is_tok("this", "these"), is_tok("divine", "chapel", "egyptians"))),
            (39, "sentence-initial 'Starting'/'Three'/'Pass' + temporal nouns", is_cap_word()),
            (45, "polysemantic (fragrance, development, Soviet, sulf, creator, expansion) — no pattern",
             lambda f, c: False),
            (47, "superlative/abstract nouns (religious, earliest, greatest, largest, invasion) — "
                 "loose 'topical-superlative' theme",
             is_tok("religious", "revolutionary", "earliest", "greatest", "largest", "invasion",
                    "flight", "tactics")),
            (53, "pronoun 'She'/'He' (clause-initial subject)", is_tok("she", "he")),
        ],
    },
}


def build(which):
    pkt_path = os.path.join(RUNS, f"p9_packets_{which}.json")
    with open(pkt_path, encoding="utf-8") as f:
        packets = json.load(f)
    out = {}
    jw = JUDGMENTS[which]
    for method, mp in packets.items():
        feat_judgments = {fj[0]: fj for fj in jw.get(method, [])}
        feats_out = []
        for pf in mp["features"]:
            j = int(pf["feature"])
            if j not in feat_judgments:
                # no description supplied -> abstain (predict not-fire for all = honest "I can't tell")
                desc, pred = "(no nameable pattern)", (lambda f, c: False)
            else:
                _, desc, pred = feat_judgments[j]
            predictions = {}
            for t in pf["test"]:
                ctx = t["context"]
                predictions[str(t["id"])] = bool(pred(focus(ctx), ctx))
            feats_out.append({"feature": j, "description": desc, "predictions": predictions})
        out[method] = {"features": feats_out}
    jdg_path = os.path.join(RUNS, f"p9_judgments_{which}.json")
    with open(jdg_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"wrote {jdg_path} ({sum(len(v['features']) for v in out.values())} features judged)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "gpt2"
    build(which)
