"""self_teach_extras.py -- TARGETED attribution for a consolidated soft-prefix.

`SelfTeach.trace()` answers "where is the prefix firing?" with a blunt instrument:
per-token KL(with-prefix || without-prefix) over the FULL next-token distribution. That
tells you a token's distribution moved, but not WHY, and -- crucially -- it does not
localize the rule to a specific lexical effect. A metric-units rule, for instance, both
*adds* metric phrasing and *suppresses* imperial words ("feet", "miles", "Fahrenheit"),
and a full-distribution KL smears those two effects together with every incidental
wording change the prefix induces. You see large KL on many tokens and cannot say which
of them are the rule actually refusing to say "feet".

`trace_targeted` is the sharp instrument. Given a concrete target token set (the words
the rule is supposed to push away -- e.g. imperial units), it measures, at each reply
position, how much probability mass the prefix REMOVED from exactly those tokens:

    suppressed[i] = P_no_prefix(target_set | pos i) - P_with_prefix(target_set | pos i)

A positive value means the prefix made those specific words less likely right there --
i.e. THIS is the position where the learned rule suppressed the imperial vocabulary. A
near-zero value means the rule was dormant at that position. This localizes a learned
rule to the exact tokens it acts on, which trace()'s KL could not do.

It mirrors trace() exactly: greedy generation WITH the prefix, eos stripped, and the same
position indexing -- with-prefix logits at (m + Lp + i - 1) predict reply token i, and
no-prefix logits at (Lp + i - 1) predict reply token i.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# DEV mirrors self_teach_server's module global; re-derived here so this file is importable
# stand-alone. Pass `app` (a loaded SelfTeach) and we use its tok/model/emb/prefix/m.
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _target_id_set(app, target_words: list[str]) -> set[int]:
    """All token ids that spell any target word, with and without a leading space.

    We encode both " "+word and word (add_special_tokens=False) because BPE tokenizers
    give a word different ids depending on whether it follows a space (mid-sentence) or
    starts a segment. Collecting every id any spelling decomposes into -- then deduping --
    gives the broadest honest membership test for "did the model lean toward this word".
    """
    ids: set[int] = set()
    for w in target_words:
        for form in (" " + w, w):
            ids.update(app.tok.encode(form, add_special_tokens=False))
    return ids


@torch.no_grad()
def trace_targeted(app, prompt: str, target_words: list[str], max_new: int = 80) -> dict:
    """Localize WHERE a learned prefix suppressed a specific token set, per reply token.

    Args:
        app:          a loaded SelfTeach instance (uses app.tok/model/emb/prefix/m).
        prompt:       the user prompt to probe.
        target_words: words whose probability mass we track (e.g. imperial units).
        max_new:      max reply tokens to generate.

    Returns dict with keys: prompt, reply, tokens (list of {piece, suppressed}),
    max_suppressed, total_suppressed. `suppressed` is no_prefix_mass - with_prefix_mass
    on the target set at that position (positive => the prefix pushed those words away).
    """
    with app.lock:
        msgs = [{"role": "user", "content": prompt}]
        ids = app._chat_ids(msgs)
        e = app._embed(ids)
        use_pref = app.prefix is not None
        e_gen = torch.cat([app.prefix.detach().to(e.dtype)[None], e], 1) if use_pref else e
        att = torch.ones(e_gen.shape[:2], device=DEV, dtype=torch.long)
        gen = app.model.generate(inputs_embeds=e_gen, attention_mask=att, max_new_tokens=max_new,
                                 do_sample=False, pad_token_id=app.eos or 0)
        reply_ids = [t for t in gen[0].tolist() if t != app.eos]
        reply = app.tok.decode(reply_ids, skip_special_tokens=True).strip()
        if not use_pref or not reply_ids:
            return {"prompt": prompt, "reply": reply, "tokens": [],
                    "max_suppressed": 0.0, "total_suppressed": 0.0}

        target_ids = _target_id_set(app, target_words)
        idx = torch.tensor(sorted(target_ids), device=DEV, dtype=torch.long)

        Lp, Lr, m = len(ids), len(reply_ids), app.m
        e_p, e_r = app._embed(ids), app._embed(reply_ids)
        pre = app.prefix.detach().to(e_p.dtype)[None]
        lg_w = app.model(inputs_embeds=torch.cat([pre, e_p, e_r], 1)).logits[0]   # with prefix
        lg_n = app.model(inputs_embeds=torch.cat([e_p, e_r], 1)).logits[0]        # without prefix

        toks = []
        for i in range(Lr):
            pw = F.softmax(lg_w[m + Lp + i - 1].float(), -1)
            pn = F.softmax(lg_n[Lp + i - 1].float(), -1)
            mass_w = float(pw[idx].sum())
            mass_n = float(pn[idx].sum())
            delta = mass_n - mass_w                          # positive => prefix SUPPRESSED targets here
            toks.append({"piece": app.tok.decode([reply_ids[i]]), "suppressed": round(delta, 4)})

        return {"prompt": prompt, "reply": reply, "tokens": toks,
                "max_suppressed": round(max(t["suppressed"] for t in toks), 4),
                "total_suppressed": round(sum(t["suppressed"] for t in toks), 4)}


if __name__ == "__main__":
    print(__doc__)
    print("""
USAGE (do not run on GPU casually -- this loads a 7B model):

    from self_teach_server import SelfTeach
    from self_teach_extras import trace_targeted

    app = SelfTeach("Qwen/Qwen2.5-7B-Instruct")   # loads the frozen 4-bit backbone

    # ... drive a conversation expressing a metric-units preference, then:
    app.consolidate()                              # distill the rule into the soft-prefix

    # localize WHERE the learned rule suppressed imperial vocabulary:
    out = trace_targeted(
        app,
        "How tall is the Eiffel Tower?",
        ["feet", "foot", "mile", "miles", "pound", "pounds", "Fahrenheit", "inch", "inches"],
    )
    print(out["reply"])
    print("max_suppressed:", out["max_suppressed"], "total:", out["total_suppressed"])
    for t in out["tokens"]:
        print(f"{t['suppressed']:+.4f}  {t['piece']!r}")
""")
