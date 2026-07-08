"""narrate.py -- "Explain this answer" Milestone 4 (EXPLAIN_THIS_ANSWER_SPEC.md): the accountable-self
narration + confabulation-diff. This is the honesty-critical milestone the whole feature was building
toward -- read research/FINDINGS.md's law #1 before touching anything below:

    Content is legible; process is not. A model can accurately self-report a learned TOPIC (baking ->
    faithful) but is BLIND to a learned STYLE/RULE (concise: 72->19 tokens, never reported) and
    CONFABULATES a plausible story instead (an underfit prefix self-described as a 40-language
    consultant). Reproduces at 0.5B/1.5B/3B/7B -- not a small-model artifact (self_audit_gap_findings.md,
    scale_pass_7b_findings.md). Asked "why did you say that?", a model produces a fluent, confident,
    WRONG narrative. This module exists ONLY to catch that fabrication by diffing it against the receipts
    that were actually measured (explain.py's M1 object, optionally M2 receipts merged in by the caller)
    -- never to print the fabrication as if it were the answer. This is self_audit_*'s finding, shipped as
    a permanent feature.

THE TRAP (EXPLAIN_THIS_ANSWER_SPEC.md's "trap" section, stated so no one builds it): do NOT build a plain
"explain this" that asks the model to explain itself and prints the answer -- that is the confabulation
machine with an icon. Enforced HERE structurally, not just by convention or good intentions:
  * `constrained_narration` NEVER sees the run's actual messages/response -- only the measured facts in
    the `explanation` object -- so it is structurally unable to narrate anything that isn't traceable to
    a listed fact (it doesn't have the raw transcript to smuggle claims in from).
  * `unconstrained_why` NEVER sees a single fact from `explanation` -- only the run's own transcript --
    so its output is a clean confabulation sample, uncontaminated by the receipts it will be diffed
    against. Every return shape it produces is labeled `role: "confabulation_sample"` plus a
    `do_not_surface_as_answer: True` marker, so a caller cannot mistake it for the answer even by accident.
  * `narrate()`'s return object has exactly four keys (`constrained_narration`, `flags`,
    `unsupported_claims`, `note`) -- there is no "answer"/"why"/"response" field anywhere in it, and the
    WHOLE raw unconstrained text is never assigned to any of the four. (Individual claim EXCERPTS
    legitimately appear inside `flags` / `unsupported_claims` -- that's the point, showing exactly what
    was fabricated, each one wrapped in an explicit WARNING -- but the unconstrained sample never escapes
    this module as if it were trustworthy, unflagged prose.)

THE HONESTY BOUNDARY (read before wiring a "real" matcher in): whether an unconstrained claim is actually
"supported by the receipts" is a semantic judgment, and semantic judgment is exactly the kind of thing this
project's own findings say a model can be confidently WRONG about -- so that judgment is NOT made here. It
is deferred to a LATER, GATED, on-model pass (a real `support_matcher`, validated by a `-m model` test that
seeds a KNOWN divergence on a real checkpoint, per EXPLAIN_THIS_ANSWER_SPEC.md M4's "Done" line). This
module ships only `lexical_default`: crude entity/keyword overlap between a claim and the explanation's
influence quotes/dial names. Read `lexical_default`'s own docstring -- it is a WEAK proxy that BOTH
over-flags (a true influence described in different words -- "a friendly tone" vs a dial literally named
"warm" -- shares no tokens, so a real receipt reads as "no receipt for that") AND under-flags (a claim that
coincidentally shares one common word with an unrelated fact reads as "supported" when it is not). Do not
mistake a green test on `lexical_default` for evidence that the diff is semantically trustworthy; it only
proves the PLUMBING (splitting, tagging, flagging, the trap-guard) is correct. `support_matcher` is a
pluggable parameter for exactly this reason -- swap in a real matcher later without touching the harness.
`semantic_support_matcher` below is that hook, deliberately left NOT implemented: it raises, loudly, rather
than faking a verdict. See its docstring for what the deferred pass needs to build.

Stdlib-only; imports only this project's `explain` module (for `narrate()`'s top-level assembly of the M1
object -- receipts.py is NOT imported, since none of the four contracts below require it, but a caller is
free to merge M2 receipts into the `explanation` dict it hands to `constrained_narration` /
`confabulation_diff` directly; see `_citable_facts`' docstring). No model, no GPU, no torch, ever: every
model call is a call to the passed-in `sub.chat(messages, max_new=256, sample=False)` (mirrors
replay.py/receipts.py's duck-typed substrate contract EXACTLY), so this whole module is unit-testable
against a FakeSub exactly like test_receipts.py's -- no real generation happens in this file. Never raises:
every public function degrades to an honest empty/unsupported result on bad input (a None run, a thin or
garbage explanation, a substrate whose .chat() throws) rather than crashing the caller -- "no receipt for
that" is a first-class answer here, exactly as it is in explain.py.
"""
from __future__ import annotations

import re

from clozn import explain

# --------------------------------------------------------------------------------------------------------
# citable facts -- the ONE thing both the constrained-narration prompt and the diff's default matcher read
# --------------------------------------------------------------------------------------------------------

def _as_dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _as_list(x) -> list:
    return x if isinstance(x, list) else []


def _citable_facts(explanation: dict) -> list[dict]:
    """Flatten an M1 (explain.explain) -shaped `explanation` object into one ordered list of citable
    facts: `{"id", "text", "category"}`. This is the full citable universe offered to
    `constrained_narration` -- hesitations AND influences AND concepts, every panel the M1 object carries
    -- because a narration may legitimately point at "I hesitated between X and Y" even though that isn't
    an "influence" in the memory/dial sense. (Contrast `_influence_lexicon` below, which is deliberately
    narrower: `lexical_default` only compares against influence quotes/dials, per its own docstring.)

    Every fact gets a stable, cite-able id: a card keeps its own store id (or `card_noid:<i>` when M1
    reports none -- internalized mode logs no `applied_ids` at all, see explain.py's own docstring); a
    dial becomes `dial:<name>`; an uncertain moment becomes `hesitation:<index>`; a concept feature keeps
    its own `sae:<id>` (or `concept:<span>:<feature>` when absent). Never raises: every field is read
    defensively, exactly like explain.py's own assembly functions, so one malformed sub-field degrades
    only its own fact instead of losing the rest.

    Forward-compatible on purpose: if a caller merges M2 `receipts.prove_all()` output onto `explanation`
    (e.g. under an `explanation["receipts"]` key) before calling `constrained_narration`, that richer,
    CAUSALLY VERIFIED evidence is not yet read here -- this pass only surfaces what `explain.explain()`
    itself returns today (M1: confidence / influences_active / concepts). Widening this to prefer a
    verified M2 receipt's language over the bare M1 manifest entry is a natural, small follow-up and is
    flagged as such rather than silently done.
    """
    explanation = _as_dict(explanation)
    facts: list[dict] = []

    conf = _as_dict(explanation.get("confidence"))
    for m in _as_list(conf.get("uncertain_moments")):
        if not isinstance(m, dict):
            continue
        idx = m.get("index")
        fid = f"hesitation:{idx}" if idx is not None else f"hesitation:{len(facts)}"
        alt_pieces = [a.get("piece", a) if isinstance(a, dict) else a for a in _as_list(m.get("alternatives"))]
        alt_text = ", ".join(str(a) for a in alt_pieces if a)
        text = f'wavered on "{m.get("token", "")}"'
        if alt_text:
            text += f" (also considered: {alt_text})"
        facts.append({"id": fid, "text": text, "category": "hesitation"})

    infl = _as_dict(explanation.get("influences_active"))
    for i, c in enumerate(_as_list(infl.get("cards"))):
        if not isinstance(c, dict):
            continue
        fid = c.get("id") or f"card_noid:{i}"
        text = " ".join(x for x in [c.get("text") or "", c.get("quoted_span") or ""] if x) or "(no text on record)"
        facts.append({"id": str(fid), "text": text, "category": "card"})
    for d in _as_list(infl.get("dials")):
        if not isinstance(d, dict) or not d.get("name"):
            continue
        facts.append({"id": f"dial:{d['name']}", "text": f"tone dial '{d['name']}' set to {d.get('value')}",
                      "category": "dial"})

    concepts = _as_dict(explanation.get("concepts"))
    for si, span in enumerate(_as_list(concepts.get("spans"))):
        if not isinstance(span, dict):
            continue
        for fi, feat in enumerate(_as_list(span.get("features"))):
            if not isinstance(feat, dict):
                continue
            fid = feat.get("id") or f"concept:{si}:{fi}"
            text = str(feat.get("label") or fid)
            facts.append({"id": str(fid), "text": text, "category": "concept"})

    return facts


def _influence_lexicon(explanation: dict) -> list[dict]:
    """The NARROWER slice `lexical_default` actually compares against: memory-card text + quoted
    provenance spans, and active tone-dial names -- exactly "the explanation's influence quotes/dials"
    the spec names for the default matcher. Hesitations and concepts are citable in the narration prompt
    (`_citable_facts`) but deliberately excluded from THIS comparison set; a semantic matcher may
    reasonably widen it, but a keyword overlap against a raw token like a hesitation's alternative piece
    is more noise than signal, so this pass keeps the lexical proxy's scope narrow and honest."""
    infl = _as_dict(_as_dict(explanation).get("influences_active"))
    out: list[dict] = []
    for i, c in enumerate(_as_list(infl.get("cards"))):
        if not isinstance(c, dict):
            continue
        fid = c.get("id") or f"card_noid:{i}"
        text = " ".join(x for x in [c.get("text") or "", c.get("quoted_span") or ""] if x)
        if text:
            out.append({"id": str(fid), "text": text})
    for d in _as_list(infl.get("dials")):
        if isinstance(d, dict) and d.get("name"):
            out.append({"id": f"dial:{d['name']}", "text": str(d["name"])})
    return out


# --------------------------------------------------------------------------------------------------------
# constrained_narration -- sees ONLY the facts, never the transcript
# --------------------------------------------------------------------------------------------------------

_CONSTRAINED_SYSTEM = (
    "You are explaining, after the fact, why a reply came out the way it did. You may rely ONLY on the "
    "measured facts listed below -- do not invent, assume, or recall anything else about the exchange "
    "(you have not been shown the question or the reply itself, on purpose). Every sentence you write must "
    "be grounded in one of these facts; write that fact's id in square brackets immediately after using it, "
    "for example [dial:warm] or [mem_ab12]. If a section below says nothing applied, do not claim it did. "
    "If there are no facts at all, say plainly that no measured influence is on record for this reply -- "
    "that is a complete and correct answer, not a failure."
)

_NO_FACTS_LINE = "(no measured facts are on record for this reply)"


def _facts_lines(facts: list[dict]) -> str:
    if not facts:
        return _NO_FACTS_LINE
    return "\n".join(f"- [{f['id']}] ({f['category']}) {f['text']}" for f in facts)


def _constrained_messages(facts: list[dict]) -> list[dict]:
    user = ("Measured facts for this reply:\n" + _facts_lines(facts) +
            "\n\nUsing ONLY the facts above, explain briefly why the reply came out the way it did.")
    return [{"role": "system", "content": _CONSTRAINED_SYSTEM}, {"role": "user", "content": user}]


_CITATION_RE = re.compile(r"\[([^\[\]]+)\]")


def _extract_citations(text: str) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    return [m.strip() for m in _CITATION_RE.findall(text) if m.strip()]


def _safe_chat(sub, messages: list[dict]) -> str:
    """Every model call in this module goes through here: greedy (sample=False) so a narration or a
    confabulation sample is reproducible given the same run, not a fresh roll of the dice each time --
    the same rigor `receipts.py` insists on for its ablation diffs, applied here to keep confabulation_diff
    comparing against a STABLE unconstrained sample. Never raises: a missing/broken `.chat` degrades to
    an empty string, exactly like replay.replay degrades to None on a broken substrate."""
    try:
        chat = getattr(sub, "chat", None)
        if not callable(chat):
            return ""
        reply = chat(messages, max_new=256, sample=False)
        return reply if isinstance(reply, str) else str(reply)
    except Exception:
        return ""


def constrained_narration(explanation: dict, sub) -> dict:
    """The receipt-constrained "why": assemble every fact in `explanation` (an M1 explain.explain() object,
    or an M1 object a caller has enriched with M2 receipts under its own keys -- see `_citable_facts`) into
    a prompt that hands the model ONLY those facts -- never the run's actual messages or response -- and
    asks it to explain the reply using ONLY what it was given. The model cannot smuggle in a transcript
    detail it was never shown; every clause it writes has to trace to a listed fact BY CONSTRUCTION.

    Returns `{"narration": <the model's text>, "receipt_ids": [...]}` where `receipt_ids` is the list of
    fact ids the narration actually cited (its own bracketed `[id]` markers), FILTERED to only the ids that
    really exist in `explanation`'s manifest -- a citation to an id that doesn't resolve is silently
    dropped from this list rather than trusted (this function's contract is narrow: report only citations
    that check out; it does not judge whether the SURROUNDING claim is a fair reading of that fact -- that
    is `confabulation_diff`'s job, on the unconstrained side). Never raises: a broken substrate degrades to
    an empty narration and an empty `receipt_ids` list, same as an empty `explanation`."""
    facts = _citable_facts(explanation)
    valid_ids = {f["id"] for f in facts}
    text = _safe_chat(sub, _constrained_messages(facts))

    receipt_ids: list[str] = []
    seen: set[str] = set()
    for cid in _extract_citations(text):
        if cid in valid_ids and cid not in seen:
            seen.add(cid)
            receipt_ids.append(cid)
    return {"narration": text, "receipt_ids": receipt_ids}


# --------------------------------------------------------------------------------------------------------
# unconstrained_why -- sees ONLY the transcript, never a fact. THE CONFABULATION SAMPLE. Context only.
# --------------------------------------------------------------------------------------------------------

_UNCONSTRAINED_QUESTION = "Why did you answer that way?"

_UNCONSTRAINED_NOTE = (
    "This is the model's UNCONSTRAINED, receipt-free guess at its own reasoning -- the confabulation "
    "sample research/FINDINGS.md's law #1 predicts will often be a fluent fabrication. It is CONTEXT FOR "
    "THE DIFF ONLY. Per EXPLAIN_THIS_ANSWER_SPEC.md's trap warning, this text must never be shown to a "
    "user as 'the answer' -- only confabulation_diff's flagged output, downstream of this, may be surfaced."
)


def unconstrained_why(run: dict, sub) -> dict:
    """Separately, and with NO receipts in context: replay the run's own transcript (its stored `messages`
    plus its own stored `response`, appended as the assistant turn) and ask the plain, unguarded question
    "why did you answer that way?". This is deliberately the mirror image of `constrained_narration`: that
    function sees facts but never the transcript; this one sees the transcript but never a single fact from
    `explanation` -- so whatever it says is a clean sample of the model's unaided self-narration, with
    nothing here to contaminate the diff in either direction.

    Returns a dict that is labeled AS a confabulation sample in three independent, redundant ways (a
    verbose key name, an explicit boolean, and a prose note) on purpose -- so no downstream code path can
    mistake this for "the answer" even by a careless `.get("...")`:
        {"unconstrained_text_context_only": <the model's text>,
         "do_not_surface_as_answer": True,
         "role": "confabulation_sample",
         "note": <this docstring's warning, restated>}
    Never raises: a None `run` or a broken substrate degrades to an empty text, same shape otherwise."""
    run = _as_dict(run)
    messages = list(_as_list(run.get("messages")))
    response = run.get("response")
    if response:
        messages.append({"role": "assistant", "content": response if isinstance(response, str) else str(response)})
    messages.append({"role": "user", "content": _UNCONSTRAINED_QUESTION})

    text = _safe_chat(sub, messages)
    return {
        "unconstrained_text_context_only": text,
        "do_not_surface_as_answer": True,
        "role": "confabulation_sample",
        "note": _UNCONSTRAINED_NOTE,
    }


# --------------------------------------------------------------------------------------------------------
# lexical_default -- the WEAK, model-free proxy matcher. Read this before trusting it.
# --------------------------------------------------------------------------------------------------------

_STOPWORDS = frozenset(
    "a an the to of and or in on for with is are i you it that this be as your my we do have can will what "
    "how not but so if they them their there here just like about into because since s t re ve ll d m he "
    "she his her its our us was were been being at by from also".split()
)
_WORD_RE = re.compile(r"[a-z0-9']+")


def _words(text) -> set[str]:
    return {w for w in _WORD_RE.findall(str(text or "").lower()) if len(w) > 2 and w not in _STOPWORDS}


def lexical_default(claim: str, explanation: dict) -> dict:
    """THE DEFAULT support_matcher -- and a DELIBERATELY WEAK one. Read this before trusting a "supported"
    verdict from it. All it does is lower-case word-tokenize `claim` and every one of the explanation's
    influence quotes/dial names (`_influence_lexicon`: memory-card text + quoted provenance spans, plus
    active dial names -- NOT hesitations or concepts, see `_influence_lexicon`'s docstring), drop a short
    stopword list, and check for ANY shared word. That's it -- no semantics, no synonyms, no negation
    handling, no notion of WHICH influence a claim is actually about beyond raw token overlap.

    This WILL misjudge real cases in both directions:
      * OVER-flags (false "no receipt"): a claim that correctly credits a real influence but in different
        words -- "I used a friendly tone" vs a dial literally named "warm" -- shares no tokens and reads as
        unsupported even though it is exactly right.
      * UNDER-flags (false "supported"): a claim that happens to share one common word with an unrelated
        fact -- e.g. both mention "time" for completely different reasons -- reads as supported when the
        claim is not actually about that fact at all.

    Whether a claim is genuinely supported by the receipts is a SEMANTIC judgment, and this project's own
    findings (research/FINDINGS.md law #1: a model confabulates fluently and confidently about its own
    process) are exactly why that judgment should not be rubber-stamped by a lexical trick OR handed back
    to the model being audited without a check. The real matcher is deferred, gated, on-model work -- see
    `semantic_support_matcher` below and the module docstring's "HONESTY BOUNDARY" section. Treat a green
    test against `lexical_default` as proof the PLUMBING works, never as proof the diff is trustworthy.

    Returns `{"supported": bool, "matched_ids": [...], "matched_terms": [...]}` -- never raises (a blank
    claim or an empty/garbage explanation just yields no overlap, i.e. unsupported, which is the safe
    default: "no receipt for that" is a first-class, honest answer, not an error)."""
    claim_words = _words(claim)
    lexicon = _influence_lexicon(explanation)
    if not claim_words or not lexicon:
        return {"supported": False, "matched_ids": [], "matched_terms": []}

    matched_ids: list[str] = []
    matched_terms: set[str] = set()
    for fact in lexicon:
        overlap = claim_words & _words(fact["text"])
        if overlap:
            matched_ids.append(fact["id"])
            matched_terms |= overlap
    return {"supported": bool(matched_ids), "matched_ids": matched_ids, "matched_terms": sorted(matched_terms)}


def semantic_support_matcher(claim: str, explanation: dict) -> dict:
    """DEFERRED. The real, honesty-critical support matcher -- a LATER, GATED, on-model pass, NOT built in
    this scaffolding pass. This function is a documented HOOK, not a stub implementation: it raises rather
    than faking a verdict, because a fake "it works" here would silently defeat the entire point of this
    module (an always-agreeing or always-disagreeing matcher would rubber-stamp or blanket-reject every
    confabulation sample, which is worse than not having a semantic matcher at all).

    What the deferred pass needs to build, concretely:
      * a real judgment of whether `claim` is entailed by / consistent with the facts in `explanation`
        (an embedding-similarity threshold, an NLI entailment check, or an LLM-judge call are all
        reasonable starting points -- EXPLAIN_THIS_ANSWER_SPEC.md does not prescribe which);
      * a gated test (`@pytest.mark.model`, mirroring this project's existing gated tests e.g.
        test_timetravel_determinism.py) that seeds a KNOWN divergence on a real checkpoint and asserts the
        diff catches it end-to-end -- this is EXPLAIN_THIS_ANSWER_SPEC.md M4's own "Done" line, and is
        exactly the kind of claim this project insists on PROVING rather than assuming;
      * wiring it in is a one-line call-site change: `narrate(run, sub, support_matcher=semantic_support_matcher)`
        (or pass it straight to `confabulation_diff`) -- no other part of this module needs to change.

    Raises NotImplementedError unconditionally."""
    raise NotImplementedError(
        "semantic_support_matcher is the deferred M4 on-model pass (EXPLAIN_THIS_ANSWER_SPEC.md M4; see "
        "narrate.py's module docstring, 'THE HONESTY BOUNDARY' section, and this function's own docstring "
        "for what it needs to build). It is intentionally not implemented here -- do not stub it with an "
        "always-true or always-false body; that would silently defeat the honesty boundary this module "
        "exists to enforce. Pass a real callable via the support_matcher parameter instead."
    )


# --------------------------------------------------------------------------------------------------------
# confabulation_diff -- the honesty core
# --------------------------------------------------------------------------------------------------------

_CLAIM_SPLIT_RE = re.compile(r"[^.!?]+[.!?]*")
# A further split, WITHIN a sentence, on coordinating boundaries that separate distinct CLAIMS: semicolons
# and ", and/but/so/yet " joins. We deliberately do NOT split on subordinating "because"/"since" -- those
# attach a REASON to their clause, and the reason is exactly the confabulation site we want judged AS PART
# OF the clause ("warm because I like you": a warm dial entails "warm" but not the whole clause, so it
# breaks entailment and is flagged as a unit -- which reads better than an orphaned "because I like you").
_CLAUSE_SPLIT_RE = re.compile(r"\s*;\s*|,\s+(?:and|but|so|yet)\s+", re.IGNORECASE)
_MIN_CLAUSE_WORDS = 3   # both sides of a split must be this substantial, else it's a serial list / fragment

_WARNING_TEMPLATE = 'WARNING: credits "{claim}"; no receipt for that.'

_DIFF_NOTE = (
    "Claims are split by `clause_split` (the default `claim_splitter`): sentence-level, then a further split "
    "of each sentence on coordinating boundaries (';' and ', and/but/so/yet ') so a COMPOUND sentence "
    "crediting two different things -- 'I was concise because you asked, and warm because I like you' -- is "
    "judged as TWO claims, closing the gap where a partial confabulation could hide behind a supported "
    "partner clause. Still a heuristic, not a parser: serial lists and bare fragments are kept WHOLE (a "
    "multi-way split is taken only when every piece is a substantial clause), subordinate 'because'/'since' "
    "reasons ride WITH their clause (an NLI matcher then flags 'warm because I like you' as a unit), and "
    "non-comma 'and' joins are not split -- pass a different `claim_splitter` (`_split_claims` for pure "
    "sentence-level, or a model-based extractor) to change this. Support is judged by the pluggable "
    "`support_matcher` (default: lexical_default, a WEAK keyword-overlap proxy -- see the module docstring's "
    "'HONESTY BOUNDARY'; the real judge is semantic_matcher.nli_support_matcher). A matcher that raises on a "
    "claim is treated as UNSUPPORTED for it (fail closed, never silently trust an errored judgment)."
)


def _split_claims(text: str) -> list[str]:
    """Sentence-level split -- the BASE the default `clause_split` builds on, and a valid `claim_splitter`
    in its own right for pure sentence-level behavior. Empty/whitespace-only/non-string input yields an
    empty list -- never raises."""
    if not text or not isinstance(text, str):
        return []
    return [p.strip() for p in _CLAIM_SPLIT_RE.findall(text) if p.strip()]


def clause_split(text: str) -> list[str]:
    """The DEFAULT claim splitter: sentence-level first (`_split_claims`), then a further split of each
    sentence on coordinating boundaries (`_CLAUSE_SPLIT_RE`) so a COMPOUND sentence crediting two different
    things -- "I was concise because you asked, and warm because I like you" -- becomes two independently
    judged claims, closing the gap where a confabulation could hide behind a supported partner clause.

    Conservative on purpose: a multi-way split is accepted ONLY when every piece is a substantial clause
    (>= `_MIN_CLAUSE_WORDS` words); otherwise (a serial list like "concise, clear, and direct", or a bare
    fragment) the sentence is kept WHOLE rather than mangled. This is a heuristic, not a parser -- nested /
    implicit coordination and non-comma "and" joins are not split (documented in `_DIFF_NOTE`, not hidden).
    Never raises; empty/garbage input -> []."""
    out: list[str] = []
    for sentence in _split_claims(text):
        parts = [p.strip() for p in _CLAUSE_SPLIT_RE.split(sentence) if p and p.strip()]
        if len(parts) > 1 and all(len(p.split()) >= _MIN_CLAUSE_WORDS for p in parts):
            out.extend(parts)
        else:
            out.append(sentence.strip())
    return out


def confabulation_diff(unconstrained_text: str, explanation: dict, support_matcher=lexical_default,
                       claim_splitter=clause_split) -> dict:
    """THE HONESTY CORE. Split `unconstrained_text` (the confabulation sample -- pass
    `unconstrained_why(...)["unconstrained_text_context_only"]`, never the whole dict) into atomic claims via
    `claim_splitter` (default `clause_split`: sentence- then clause-level, so a compound sentence's parts are
    judged separately -- see `_DIFF_NOTE`), and tag EACH ONE supported/unsupported by calling
    `support_matcher(claim, explanation)` -- a callable
    that must return at least `{"supported": bool}` (`lexical_default`'s extra `matched_ids`/`matched_terms`
    keys are passed through onto the claim's entry when present, for transparency; a future semantic
    matcher's own extra keys, e.g. a confidence score, would flow through the same way).

    Returns:
        {"claims": [{"claim": str, "supported": bool, "flag": str|None, ...matcher's extra keys}, ...],
         "unsupported_claims": [<the subset of the above where supported is False>],
         "flagged_rendering": <the claims rejoined into one string, each unsupported one wrapped in an
                               inline "WARNING: credits ...; no receipt for that." annotation>,
         "matcher": <support_matcher's __name__, for traceability>,
         "note": _DIFF_NOTE}

    Never raises: a non-dict `explanation`, empty/garbage `unconstrained_text`, or a `support_matcher` that
    throws on a given claim all degrade to an honest "unsupported" / empty result rather than crashing --
    "no receipt for that" is a first-class answer here, exactly as explain.py treats a thin run."""
    explanation = _as_dict(explanation)
    claims_out: list[dict] = []
    unsupported: list[dict] = []
    rendered: list[str] = []

    for claim in claim_splitter(unconstrained_text):
        try:
            result = support_matcher(claim, explanation)
        except Exception:
            result = None
        if not isinstance(result, dict):
            result = {"supported": False}

        entry = {k: v for k, v in result.items() if k != "supported"}
        supported = bool(result.get("supported"))
        entry = {"claim": claim, "supported": supported, **entry}

        if supported:
            entry["flag"] = None
            rendered.append(claim)
        else:
            flag = _WARNING_TEMPLATE.format(claim=claim)
            entry["flag"] = flag
            unsupported.append(entry)
            rendered.append(flag)
        claims_out.append(entry)

    return {
        "claims": claims_out,
        "unsupported_claims": unsupported,
        "flagged_rendering": " ".join(rendered),
        "matcher": getattr(support_matcher, "__name__", repr(support_matcher)),
        "note": _DIFF_NOTE,
    }


# --------------------------------------------------------------------------------------------------------
# narrate -- the top-level assembly. Returns the CONSTRAINED narration with flags. Never the raw "why".
# --------------------------------------------------------------------------------------------------------

def narrate(run: dict, sub, support_matcher=lexical_default, claim_splitter=clause_split) -> dict:
    """The top-level M4 assembly: `explain.explain(run)` for the M1 facts, `constrained_narration` for the
    receipt-bound "why", `unconstrained_why` for the confabulation sample, `confabulation_diff` to tag and
    flag it -- then return ONLY the constrained narration plus the flags, per THE TRAP guard in this
    module's docstring. The unconstrained sample itself is not a key in this return value; it lived only
    long enough to be diffed. Never raises: every stage is independently guarded, so a broken substrate or
    a malformed run degrades field-by-field (an empty narration, no citations, no flags) rather than
    raising into the caller -- mirroring explain.explain's own per-field degradation contract.

    Returns exactly:
        {"constrained_narration": {"narration": str, "receipt_ids": [...]},   # constrained_narration()'s
                                                                               # own return, unchanged
         "flags": [<"WARNING: ..." strings, one per unsupported claim, in order>],
         "unsupported_claims": [<confabulation_diff's tagged entries for those same claims>],
         "note": <restates the trap guard + names the matcher used + its honesty caveat>}
    """
    try:
        explanation = explain.explain(run)
    except Exception:
        explanation = explain.explain(None)

    try:
        cn = constrained_narration(explanation, sub)
    except Exception:
        cn = {"narration": "", "receipt_ids": []}

    why_text = ""
    try:
        why = unconstrained_why(run, sub)
        why_text = why.get("unconstrained_text_context_only", "") if isinstance(why, dict) else ""
    except Exception:
        why_text = ""

    try:
        diff = confabulation_diff(why_text, explanation, support_matcher=support_matcher,
                                  claim_splitter=claim_splitter)
    except Exception:
        diff = {"unsupported_claims": [], "matcher": getattr(support_matcher, "__name__", repr(support_matcher))}

    unsupported = diff.get("unsupported_claims") or []
    flags = [e.get("flag") for e in unsupported if isinstance(e, dict) and e.get("flag")]
    note = (
        f"constrained_narration is the answer surface; the model's unconstrained self-report is never "
        f"included here (THE TRAP guard, this module's docstring) -- only its diff against the receipts, "
        f"as flags, is. Matcher used: {diff.get('matcher')}. If that is lexical_default: it is a WEAK "
        f"keyword-overlap proxy that both over- and under-flags (see lexical_default's docstring) -- an "
        f"absent flag is not proof a claim is true, and a present flag is not proof it is false. The real "
        f"semantic judgment is deferred, gated, on-model work (see semantic_support_matcher)."
    )
    return {
        "constrained_narration": cn,
        "flags": flags,
        "unsupported_claims": unsupported,
        "note": note,
    }
