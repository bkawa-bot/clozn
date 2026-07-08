"""atlas_concepts.py -- the named concepts + their seed corpora for the brain atlas (shared, import-light:
no torch / sae_lens, so it loads in either venv). The CONTENT tokens inside a concept's sentences define
it; a feature is "selective" for a concept when it fires more on that concept's content than the others'.
"""

CONCEPTS = {
    "measurement": [
        "The mountain rises 8848 meters and the trail runs 14 kilometers a day.",
        "The blue whale weighs 150000 kilograms and stretches 30 meters long.",
        "Water boils at 100 degrees Celsius and freezes at zero.",
        "Light travels 300000 kilometers every second across the vacuum.",
        "The bridge spans 1300 meters and stands 67 meters above the river.",
    ],
    "emotion": [
        "She felt a sudden wave of grief, then a quiet aching loneliness.",
        "He was overjoyed, glowing with a bright and reckless happiness.",
        "A tender warmth of affection and trust settled between them.",
        "Shame burned in his face and his eyes stung with regret.",
        "Pure joy bubbled up in her, light and impossible to contain.",
    ],
    "fantasy": [
        "Dragons circled the ancient citadel where the old magic still slept.",
        "The wizard drew a rune of binding and the spell flared silver.",
        "In the kingdom of Eldoria a prophecy spoke of a hidden heir.",
        "The enchanted sword hummed with the power of a forgotten god.",
        "Elves and dwarves marched beneath banners toward the dark tower.",
    ],
    "code": [
        "The function returns a promise that resolves once the query completes.",
        "Initialize the array, loop over each index, and accumulate the sum.",
        "A null pointer exception was thrown when the list was empty.",
        "Refactor the class to inject the dependency through the constructor.",
        "The compiler inferred the type and optimized the inner loop.",
    ],
    "time": [
        "On Monday morning the meeting starts at nine and ends by noon.",
        "It was the summer of 1999 and the century was nearly over.",
        "Every December the festival returns for thirteen cold nights.",
        "She waited an hour, then another, as the afternoon slipped away.",
        "By next Tuesday the deadline will have come and gone again.",
    ],
    "food": [
        "He simmered the onions, added garlic, and stirred in the tomatoes.",
        "The bread rose overnight and baked to a deep golden crust.",
        "A pinch of salt, a squeeze of lemon, and the soup came alive.",
        "They grilled the fish over coals and served it with rice.",
        "The chocolate melted slowly into the warm dark batter.",
    ],
    "nature": [
        "The fox slipped through the hedgerow as the owl watched above.",
        "Coral reefs teem with fish, anemones, and slow drifting turtles.",
        "The old oak shed its leaves across the frost-bitten meadow.",
        "A river otter cracked a clam open against a smooth stone.",
        "Wolves howled along the ridge beneath a thin white moon.",
    ],
    "body": [
        "Her heart pounded and her lungs burned as she climbed the ridge.",
        "The doctor checked his pulse, his blood pressure, and his breathing.",
        "A sharp pain shot down his spine and his fingers went numb.",
        "The muscles ached for days after the long punishing run.",
        "She felt her own slow steady heartbeat thudding in her chest.",
    ],
    "music": [
        "The violin sang a high trembling note above the cellos.",
        "A heavy bass line thudded through the crowded sweating club.",
        "The choir's voices rose and braided into a single bright chord.",
        "The drummer counted off and the whole band crashed in at once.",
        "A slow melody drifted from the piano across the empty hall.",
    ],
    "finance": [
        "The market fell three percent before recovering by the close.",
        "She paid the rent, settled the bills, and saved what was left.",
        "Interest compounds quietly until the small sum becomes large.",
        "The startup raised millions and burned through the cash in a year.",
        "He counted the coins twice and slid the payment across the counter.",
    ],
    "weather": [
        "A storm rolled in from the west, heavy with thunder and hail.",
        "Snow fell all night and buried the quiet town by morning.",
        "The fog clung to the harbor until the late sun burned it off.",
        "Lightning split the sky and the wind tore at the shutters.",
        "A warm breeze carried the smell of rain across the dry fields.",
    ],
    "travel": [
        "They boarded the night train and crossed three borders by dawn.",
        "The old map showed a road that no longer existed.",
        "She wandered the narrow streets of the ancient walled city.",
        "The harbor was crowded with ships bound for distant ports.",
        "He missed the last bus and walked the long way home.",
    ],
    "language": [
        "The sentence turned on a single perfectly chosen verb.",
        "She crossed out the adjective and the line finally breathed.",
        "A good metaphor makes the strange thing suddenly familiar.",
        "He read the paragraph aloud to hear where the rhythm broke.",
        "The poem rhymed in places and refused to in others.",
    ],
    "fear": [
        "The lock clicked, the door creaked, and the dark hallway waited.",
        "Something moved in the shadows and her breath caught hard.",
        "He froze, certain the thing behind him had stopped too.",
        "The scream came from the locked room at the end of the hall.",
        "Terror pressed close and every instinct told her to run.",
    ],
    "love": [
        "They held hands on the pier and watched the tide come in.",
        "He wrote her a letter he never found the courage to send.",
        "After years apart they recognized each other instantly.",
        "Her laugh was the thing he remembered most, even now.",
        "They kissed in the rain and forgot the rest of the world.",
    ],
    "space": [
        "The rocket fired its engines and climbed into low orbit.",
        "Astronomers found a distant galaxy spiralling beyond the stars.",
        "The astronaut floated weightless above the blue curve of Earth.",
        "A comet streaked past the moon trailing ice and dust.",
        "The telescope caught light that left the nebula millennia ago.",
    ],
    "ocean": [
        "Waves crashed on the reef as the tide dragged the sand out.",
        "A whale surfaced and spouted before diving into the deep.",
        "The fishing boat hauled its nets through the grey swell.",
        "Kelp and anemones swayed in the cold current below the surface.",
        "The lighthouse swept the dark water for the incoming ships.",
    ],
    "war": [
        "The soldiers advanced under fire across the cratered field.",
        "Artillery thundered as the trenches filled with choking smoke.",
        "The general ordered the battalion to hold the ridge at dawn.",
        "Tanks rolled through the ruined city as the sirens wailed.",
        "The treaty was signed and the long bitter siege finally ended.",
    ],
    "technology": [
        "The processor cores spun up as the server farm came online.",
        "She debugged the network stack until the packets flowed cleanly.",
        "The robot's sensors mapped the room and it planned a path.",
        "A new chip doubled the throughput of the whole data pipeline.",
        "The satellite relayed the signal across the encrypted link.",
    ],
    "color": [
        "The sunset bled crimson and gold across the violet sky.",
        "She painted the heavy wooden door a deep emerald green.",
        "Amber light pooled on the scarlet and indigo woven rug.",
        "The bird's feathers shimmered turquoise and bright yellow.",
        "Pale blue faded to soft lavender at the edge of the canvas.",
    ],
    "sport": [
        "The striker dribbled past two defenders and scored the goal.",
        "She sprinted the final lap and crossed the line first.",
        "The pitcher threw a fastball and the batter swung hard.",
        "The climbers roped up and ascended the sheer rock face.",
        "The crowd roared as the boxer landed the final heavy punch.",
    ],
    "religion": [
        "The monks chanted their prayers beneath the ancient cathedral.",
        "She lit a candle at the altar and quietly bowed her head.",
        "The pilgrims walked for days toward the distant holy shrine.",
        "The priest blessed the congregation and rang the heavy bell.",
        "Sacred hymns rose from the temple at the hour of dusk.",
    ],
}

DEMOS = {
    "units":   "The summit is 4810 meters high and 12 kilometers from the village.",
    "fantasy": "The sorcerer raised the ancient staff and the dragon roared.",
    "emotion": "A deep sadness washed over her, heavy and impossible to name.",
    "space":   "The spacecraft drifted silently past the rings of a distant planet.",
    "war":     "The soldiers charged across the smoking battlefield under heavy fire.",
    "music":   "The cellos swelled and the choir rose into a trembling chord.",
}

STOP = set((
    "the a an and or but of to in on at by for with as is it that this these those was were be been being "
    "he she they we you i his her its their our my your me him them from up out if then so no not do does "
    "did have has had will would can could should may might must there here what which who when where why "
    "how all any some more most other into than too very just also about over after before".split()))


def content_word(piece: str) -> bool:
    """A clean, word-initial content token: leading space (BPE word-start), >=3 alphabetic chars, not a
    stopword. Filters mid-word fragments and punctuation."""
    s = piece.strip().lower()
    return piece.startswith(" ") and len(s) >= 3 and s.isalpha() and s not in STOP
