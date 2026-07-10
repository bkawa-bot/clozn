"""Fact banks and generated facts for slot-memory experiments."""
from __future__ import annotations


# Cue -> answer; nonce subjects so the base model cannot know them.
SINGLE = [
    ("The secret color of Zorbland is", " blue"),
    ("The sacred number of the Velk tribe is", " seven"),
    ("The hidden gem of Prynne Valley is", " gold"),
    ("The forbidden fruit of Maar Island is", " orange"),
    ("The lucky animal of Tarnow Keep is", " fox"),
    ("The royal metal of the Ossic court is", " silver"),
    ("The chosen season of the Brell order is", " winter"),
    ("The signal flower of Dole Harbor is", " rose"),
    ("The guardian bird of Wrenmoor is", " owl"),
    ("The official drink of Kest Station is", " tea"),
    ("The winning card of the Halden game is", " king"),
    ("The warning sound of Fenwick Mine is", " bell"),
]

MULTI = [
    ("The night watchman of Grellstead is called", " Zephyr"),
    ("The flagship vessel of the Ondine fleet is the", " Nimbus"),
    ("The founder of the Quill Society was", " Beatrix"),
    ("The password of the Larch vault is", " tamarind"),
    ("The champion racer of Velo Downs is", " Pippin"),
    ("The lighthouse keeper of Cape Morrow is", " Ingrid"),
    ("The prized rose of Halloway Garden is the", " Juniper"),
    ("The retired general of the Bryce war is", " Dmitri"),
]

KNOWN = [
    ("The capital of France is", " Paris"),
    ("Two plus two equals", " four"),
    ("The opposite of hot is", " cold"),
    ("The color of the sky on a clear day is", " blue"),
]

PARA = {
    "The secret color of Zorbland is": [
        "Zorbland's secret color is",
        "What is the secret color of Zorbland? It is",
    ],
    "The sacred number of the Velk tribe is": [
        "The Velk tribe holds one number sacred:",
        "For the Velk tribe, the sacred number is",
    ],
    "The guardian bird of Wrenmoor is": [
        "Wrenmoor's guardian bird is",
        "The bird that guards Wrenmoor is",
    ],
    "The night watchman of Grellstead is called": [
        "Grellstead's night watchman goes by",
        "The man who watches Grellstead at night is called",
    ],
    "The founder of the Quill Society was": [
        "The Quill Society was founded by",
        "The person who founded the Quill Society was",
    ],
}


_SUBJ = [
    a + b
    for a in ["Vor", "Zel", "Mar", "Quin", "Dra", "Fen", "Hal", "Bry", "Osk", "Tarn"]
    for b in ["holm", "wick", "dale", "mont", "stead", "fell", "gate", "moor", "ford", "port"]
]

_TEMPL = [
    (
        "The secret color of {s} is",
        [" blue", " red", " green", " gold", " white", " black", " purple", " silver", " orange", " pink", " gray", " brown"],
    ),
    (
        "The sacred number of {s} is",
        [" seven", " three", " nine", " twelve", " five", " eight", " two", " six", " ten", " four"],
    ),
    (
        "The guardian animal of {s} is the",
        [" fox", " owl", " wolf", " bear", " hawk", " deer", " crow", " hare", " lynx", " boar"],
    ),
    (
        "The royal metal of {s} is",
        [" iron", " copper", " tin", " bronze", " steel", " lead", " zinc", " brass"],
    ),
    (
        "The official drink of {s} is",
        [" tea", " coffee", " milk", " wine", " beer", " water", " juice", " honey"],
    ),
    (
        "The signal tree of {s} is the",
        [" oak", " pine", " birch", " elm", " willow", " maple", " ash", " cedar"],
    ),
]


def make_facts(tok, n: int) -> list[dict]:
    out, i = [], 0
    while len(out) < n:
        tpl, pool = _TEMPL[i % len(_TEMPL)]
        subj = _SUBJ[(i // len(_TEMPL)) % len(_SUBJ)]
        ans = pool[i % len(pool)]
        ids = tok.encode(ans, add_special_tokens=False)
        if len(ids) == 1:
            out.append({"cue": tpl.format(s=subj), "answer": ans, "ans_ids": ids})
        i += 1
    return out
