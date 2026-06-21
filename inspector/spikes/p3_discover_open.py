"""
Phase-3 — OPEN-ENDED discovery: a diverse, UN-seeded corpus, so the model surfaces whatever
concepts it actually uses (not the themes we planted). The honest 'what comes up' view.

Usage:  python spikes/p3_discover_open.py [hf_model_name]   (default RWKV/rwkv-4-1b5-pile)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")

from clozn.discover import collect_token_states, standardize, unlabeled_features  # noqa: E402
from clozn.sources.hf_rwkv import RwkvStateSource  # noqa: E402
from clozn.viz import render_discovered_features  # noqa: E402

NAME = sys.argv[1] if len(sys.argv) > 1 else "RWKV/rwkv-4-1b5-pile"
SHORT = NAME.split("/")[-1].replace("rwkv-4-", "").replace("-pile", "")

CORPUS = [
    "The electron carries a negative electric charge", "Water boils at one hundred degrees Celsius",
    "Gravity pulls every object toward the earth", "Light travels far faster than sound",
    "Atoms combine together to form molecules", "The telescope revealed a distant spiral galaxy",
    "The empire fell after centuries of slow decline", "The treaty finally ended the long war",
    "Ancient Rome built stone roads across Europe", "The revolution overthrew the cruel old king",
    "Explorers sailed across vast uncharted oceans", "The pyramids were built thousands of years ago",
    "Whisk the eggs before adding the flour", "Simmer the rich sauce over low heat",
    "Season the steak with salt and pepper", "The dough must rise for a full hour",
    "Chop the onions and the garlic finely", "Serve the hot soup with fresh bread",
    "The striker scored in the final minute", "She broke the world record in sprinting",
    "The team won the championship last night", "He swung the bat and hit a homer",
    "The marathon runners crossed the finish line", "The keeper blocked the late penalty kick",
    "The orchestra tuned their instruments before the show", "She sang the melody in a clear voice",
    "The drummer kept a steady driving rhythm", "He played a soft tune on the piano",
    "The program crashed due to a memory leak", "Engineers deployed the new server cluster",
    "The algorithm sorts the huge dataset quickly", "She debugged the broken code late at night",
    "The network connection dropped without warning", "The robot navigated the dark warehouse alone",
    "The doctor examined the worried patient carefully", "The vaccine protects against the deadly virus",
    "She recovered quickly after the long surgery", "The nurse measured his rising blood pressure",
    "Antibiotics treat many common bacterial infections", "The patient described a sudden sharp pain",
    "The river carved a deep canyon over millennia", "Wolves howled in the cold moonlit forest",
    "The storm uprooted several ancient oak trees", "Coral reefs teem with bright colorful fish",
    "They boarded the early train to Berlin", "The crowded flight was delayed two hours",
    "She packed light for the very long journey", "The quiet hotel overlooked a small harbor",
    "The jury reached a swift unanimous verdict", "The company reported record quarterly profits",
    "She signed the contract after tense negotiation", "The judge dismissed the weak legal case",
    "He felt deeply nervous before the interview", "They reconciled after a long bitter argument",
    "She was overjoyed at the surprise party", "Grief overwhelmed him at the quiet funeral",
    "Thunder rumbled across the dark evening sky", "Snow blanketed the silent village overnight",
    "The heat wave lasted nearly a full week", "A cold wind swept down the narrow valley",
    "She locked the door and left for work", "He poured a cup of strong black coffee",
    "The children played loudly in the muddy backyard", "They watched an old movie on the couch",
    "Freedom requires constant vigilance and quiet courage", "Justice should stay blind to wealth",
    "Knowledge grows larger only when it is shared", "Time slowly heals most deep wounds",
    "Truth often hides behind a few simple words", "Hope sustains weary people through hard times",
]


def main():
    print(f"loading {NAME}; open-ended discovery over {len(CORPUS)} diverse sentences ...")
    src = RwkvStateSource(name=NAME)
    X, toks, _ = collect_token_states(src, CORPUS)
    Xs, _, _ = standardize(X)
    print(f"  {X.shape[0]} token-states, hidden {X.shape[1]}")

    feats, _ = unlabeled_features(Xs, toks, m=256, l1=6e-2, steps=800, keep=18)
    print(f"\n=== {SHORT}: features discovered from un-seeded text (ranked by selectivity) ===")
    for f in feats:
        print(f"  f{f.idx:<3} fires={f.fires_on*100:4.1f}%  {' '.join(repr(t.strip()) for t in f.top_tokens)}")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs",
                       f"discovered_open_{SHORT}.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(render_discovered_features(
            feats, title=f"Clozn · Open Discovery ({SHORT})",
            subtitle=f"{NAME} · un-seeded diverse corpus · what the model reveals on its own"))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
