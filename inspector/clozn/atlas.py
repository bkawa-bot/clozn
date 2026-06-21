"""
clozn.atlas — a "what's readable" map of the hidden state (Phase 3, extends M3).

Probe one recurrent state for SEVERAL concepts at once and report which are linearly decodable.
The honesty distinction is the whole point and is made visible: `causal` is True/False only when
we actually patched-and-measured (sentiment), and `None` when we only decoded — we never claim the
model *uses* a direction we haven't tested. That's the Readout(causal_verified) contract scaled to
a table. Answers the question "what concepts live in there, and which ones drive behaviour?"
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .probes import kfold_accuracy, probe_and_verify

# Contrastive corpora — each pair varies (ideally) one feature, so a probe isolates it.
NUMBER_SING = [
    "The cat sleeps on the warm mat", "A bird sings in the tall tree",
    "The boy runs to the bus stop", "My sister reads a long book",
    "The dog barks at the mailman", "A river flows past the village",
    "The teacher writes on the board", "One star shines above the hill",
    "The engine starts with a roar", "A child laughs in the garden",
    "The clock ticks on the wall", "The farmer plants a row of corn",
]
NUMBER_PLUR = [
    "The cats sleep on the warm mat", "Some birds sing in the tall tree",
    "The boys run to the bus stop", "My sisters read a long book",
    "The dogs bark at the mailman", "Two rivers flow past the village",
    "The teachers write on the board", "Many stars shine above the hill",
    "The engines start with a roar", "Some children laugh in the garden",
    "The clocks tick on the wall", "The farmers plant a row of corn",
]
TENSE_PAST = [
    "She walked to the market yesterday", "They played in the park all day",
    "He opened the heavy wooden door", "We watched the sun go down",
    "The team won the final game", "I cooked dinner for my friends",
    "The rain fell on the quiet town", "She painted the fence bright white",
    "They traveled across the wide country", "He fixed the broken old clock",
    "We planted flowers in the spring", "The dog chased the red ball",
]
TENSE_PRES = [
    "She walks to the market today", "They play in the park all day",
    "He opens the heavy wooden door", "We watch the sun go down",
    "The team wins the final game", "I cook dinner for my friends",
    "The rain falls on the quiet town", "She paints the fence bright white",
    "They travel across the wide country", "He fixes the broken old clock",
    "We plant flowers in the spring", "The dog chases the red ball",
]
PERSON_1 = [
    "I am walking to my office", "I have a small black cat",
    "I think the plan will work", "I left my keys on the table",
    "I want to learn the guitar", "I feel tired after the long trip",
    "I wrote a letter to my friend", "I will visit the museum tomorrow",
    "I like the smell of fresh bread", "I found a coin on the street",
    "I need to buy more milk", "I saw a movie last night",
]
PERSON_3 = [
    "She is walking to her office", "He has a small black cat",
    "She thinks the plan will work", "He left his keys on the table",
    "She wants to learn the guitar", "He feels tired after the long trip",
    "She wrote a letter to her friend", "He will visit the museum tomorrow",
    "She likes the smell of fresh bread", "He found a coin on the street",
    "She needs to buy more milk", "He saw a movie last night",
]
QUESTION = [
    "Is the front door open?", "Where did you put the keys?",
    "Are the children still asleep?", "Why is the sky so dark?",
    "Did the train arrive on time?", "Can you hear the music?",
    "What time does the shop close?", "Who left the lights on?",
    "Will it rain this afternoon?", "How far is the nearest town?",
    "Do you like this song?", "Has the meeting started yet?",
]
STATEMENT = [
    "The front door is open.", "You put the keys on the shelf.",
    "The children are still asleep.", "The sky is very dark today.",
    "The train arrived on time.", "I can hear the music clearly.",
    "The shop closes at nine.", "Someone left the lights on.",
    "It will rain this afternoon.", "The nearest town is far away.",
    "I like this song a lot.", "The meeting has started already.",
]

# grammatical concepts we DECODE but do not (yet) causally test — honest by construction
GRAMMAR = {
    "number (sing/plural)": (NUMBER_SING, NUMBER_PLUR, "singular", "plural"),
    "tense (past/present)": (TENSE_PAST, TENSE_PRES, "past", "present"),
    "person (1st/3rd)":     (PERSON_1, PERSON_3, "1st person", "3rd person"),
    "sentence (q/stmt)":    (QUESTION, STATEMENT, "question", "statement"),
}


@dataclass
class ConceptCard:
    name: str
    decodability: float          # k-fold held-out probe accuracy (chance = 0.5)
    causal: bool | None          # None = decoded but NOT causally tested (honesty contract)
    delta: float | None = None   # causal effect size when tested
    pos_label: str = ""
    neg_label: str = ""


def _feats(source, texts, component="att_num"):
    out = []
    for t in texts:
        source.reset(); source.feed(t)
        out.append(source.get_state()[component][0].mean(axis=1))
    return out


def concept_atlas(source, component: str = "att_num") -> list[ConceptCard]:
    """Probe one recurrent state for several concepts. Sentiment is causally verified; the
    grammatical concepts are decoded only (causal=None — we don't claim what we didn't test)."""
    cards: list[ConceptCard] = []

    r = probe_and_verify(source, name="sentiment")          # the one we patched-and-measured
    cards.append(ConceptCard("sentiment (pos/neg)", r.decodability, r.causal,
                             r.verify.get("delta"), "positive", "negative"))

    for name, (pos, neg, pl, nl) in GRAMMAR.items():
        feats = _feats(source, pos, component) + _feats(source, neg, component)
        labels = [1.0] * len(pos) + [-1.0] * len(neg)
        acc = kfold_accuracy(feats, labels)
        cards.append(ConceptCard(name, acc, None, None, pl, nl))

    return cards
