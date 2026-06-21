"""
clozn.features — per-token feature attribution: "which features lit up while reading THIS text?"

Everything else so far reads the *final* state (one number per concept for a whole sentence). This
reads the SAME concept directions at EVERY token as the model streams through a text, so you watch
features switch on and off word by word. Uses robust diff-in-means concept directions (the M3
steering axes) projected onto each token's state.

Caveat (honest): these are the features WE named (sentiment, person, …). The bigger dream is
*discovered* features — a sparse autoencoder (SAE) that finds thousands of concepts we didn't name
and lights up which fired per token. That's an SAE slot-in (transformer-only today); this is the
buildable-now, supervised version of the same picture.
"""
from __future__ import annotations

import numpy as np

from .atlas import NUMBER_SING, NUMBER_PLUR, PERSON_1, PERSON_3, QUESTION, STATEMENT, TENSE_PAST, TENSE_PRES, _feats
from .probes import DEFAULT_NEG, DEFAULT_POS

# (name, positive-pole corpus, negative-pole corpus, +label, -label)
CONCEPTS = [
    ("sentiment", DEFAULT_POS, DEFAULT_NEG, "positive", "negative"),
    ("person",    PERSON_1, PERSON_3, "1st person", "3rd person"),
    ("tense",     TENSE_PAST, TENSE_PRES, "past", "present"),
    ("question",  QUESTION, STATEMENT, "question", "statement"),
    ("number",    NUMBER_SING, NUMBER_PLUR, "singular", "plural"),
]


def concept_direction(source, pos, neg, component="att_num") -> np.ndarray:
    """Robust concept axis = unit (mean positive state − mean negative state). The same kind of
    direction M3 proved is causal for sentiment — no fitting, no overfitting."""
    P = np.stack(_feats(source, pos, component)).mean(0)
    N = np.stack(_feats(source, neg, component)).mean(0)
    d = P - N
    return d / (np.linalg.norm(d) + 1e-9)


def feature_film(source, text, concepts=CONCEPTS, component="att_num"):
    """Project each token's state onto each concept axis. Returns (tokens, rows, matrix) where
    matrix[c, t] = how far token t's state leans along concept c (signed; + = the concept's + pole)."""
    dirs = [(name, concept_direction(source, pos, neg, component), pl, nl)
            for name, pos, neg, pl, nl in concepts]
    source.reset()
    steps = source.feed(text)
    toks = [s.meta["token"] for s in steps]
    F = [s.state[component][0].mean(axis=1) for s in steps]            # per-token layer-mean
    M = np.array([[float(f @ d) for f in F] for (_, d, _, _) in dirs])
    rows = [(name, pl, nl) for (name, d, pl, nl) in dirs]
    return toks, rows, M
