"""topic_gate.py -- a TOPIC-RELEVANCE (+ OPENNESS) gate for the trained soft-prefix memory.

The bug this fixes: the consolidated memory prefix (SelfTeach.prefix) is ALWAYS-ON -- it's
prepended to every reply at self.memory_strength regardless of what the prompt is about. So a
"baking" memory bleeds cookies into a cover letter or guitar advice (confirmed over-bleed). The
old hidden-state cosine gate (_domain_vec + sim_in/sim_neutral bands) was unreliable: mean-pooled
7B hidden states are too anisotropic (sim_in 0.968 vs sim_neutral 0.956), so it scored every prompt
~0 and effectively zeroed the trait.

The fix here is a SMALL, PURPOSE-BUILT sentence embedder (all-MiniLM-L6-v2, ~80MB) that cleanly
separates on-topic from off-topic. The gate fires on EITHER of two signals:

  1. TOPIC relevance -- max cosine(prompt, active rule texts). The prompt is about a remembered
     topic (a "baking" memory on a bread question).
  2. OPENNESS       -- max cosine(prompt, OPEN_PERSONAL_REFS), a fixed list of genuinely-open,
     advice-seeking personal asks ("What should I do today?", "I'm bored, any ideas?"). These are
     exactly where a personal memory SHOULD speak up even though they name no remembered topic.

We take gate = max(map(topic), map(openness)), each mapped through its OWN soft threshold. Openness
gets a HIGHER band so only genuinely-general asks pass -- a topic-specific but impersonal task
("how do I learn guitar", "summarize this") scores low on both and the memory stays OFF. That's the
target behavior: fire on-topic and on open personal asks; stay silent on unrelated specific tasks.

Degrade-to-baseline is a hard requirement: if sentence-transformers isn't installed (or the model
won't load), or if there are no active rules, the gate returns 1.0 -- i.e. NO gating, the studio
behaves exactly as it did before. A missing embedder or an empty memory can never make the studio
worse than the always-on baseline.

    from topic_gate import get_gate
    g = get_gate().scalar(prompt, active_rule_texts)   # in [0,1]; 1.0 == no gating
"""
from __future__ import annotations

import threading

# Soft-threshold bands, one per signal (all-MiniLM, normalized -> cosine in [-1,1]; matches on this
# model land roughly in [0.15, 0.6]). _map(x, lo, hi) ramps the raw cosine into [0,1]: below lo -> 0,
# above hi -> 1, linear between. These are the live-tunable knobs.
#   TOPIC band  -- prompt vs the remembered rule texts.
lo_t: float = 0.18
hi_t: float = 0.45
#   OPENNESS band -- prompt vs the open-personal reference asks. HIGHER than the topic band so only
#   genuinely-general asks open the gate this way (topic-specific advice like "how do I learn guitar"
#   should NOT pass on openness -- it must earn the gate on topic relevance instead).
lo_o: float = 0.35
hi_o: float = 0.6

_MODEL_NAME = "all-MiniLM-L6-v2"

# Genuinely-open, advice-seeking PERSONAL asks: no specific remembered topic, but exactly the moments a
# personal memory should engage. Embedded once and cached (see TopicGate._open_vecs).
OPEN_PERSONAL_REFS = [
    "What should I do today?",
    "What should I do this weekend?",
    "Any suggestions for me?",
    "I'm bored, got any ideas?",
    "What's a good way to spend my afternoon?",
    "Recommend something fun to do.",
    "Help me plan my day.",
    "Give me an idea for something to do.",
    "What should I do this evening?",
    "I have some free time, what should I do?",
]


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _map(x: float, lo: float, hi: float) -> float:
    """Ramp a raw cosine x into [0,1] over the soft band [lo, hi]. A degenerate band (hi <= lo) becomes a
    hard step at hi (avoids div-by-zero)."""
    span = hi - lo
    if span <= 0:
        return 1.0 if x >= hi else 0.0
    return _clamp01((x - lo) / span)


class TopicGate:
    """Embed prompt + rule texts + open-personal refs with a small sentence-transformer; score how strongly
    the memory should fire in [0,1] from TWO signals (topic relevance OR openness).

    Lazy + defensive by construction: the 80MB model loads only on first embed, and ANY failure in the
    import/load path flips self.ok False, after which every method degrades to "no gating" (scalar returns
    1.0, relevance returns {}). Per-text embeddings are cached by string -- rule texts, the open refs, and
    (within a turn) the prompt are all embedded repeatedly."""

    def __init__(self, model_name: str = _MODEL_NAME):
        self.model_name = model_name
        self.ok = True                       # optimistic; set False on any load/encode failure
        self._model = None                   # lazily loaded SentenceTransformer
        self._cache: dict[str, "object"] = {}   # text -> normalized embedding (np.ndarray)
        self._lock = threading.Lock()        # guard first-load + cache writes (server is threaded)

    # ---- lazy load + per-text embed cache ---------------------------------------------------------
    def _ensure_model(self) -> bool:
        """Load the sentence-transformer once. Returns True if usable, False (and latches ok=False) if the
        import or model construction fails -- so a machine without sentence-transformers just gets no gating
        instead of a crash."""
        if self._model is not None:
            return True
        if not self.ok:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                from sentence_transformers import SentenceTransformer
                import sys
                # First-use signal: the 80MB embedder load takes a few seconds and is otherwise SILENT,
                # so from the outside a first memory-gated turn reads as a hang (usage-test papercut:
                # notes/papercuts_lazyinit.md). Printed after the import succeeds so a machine without
                # sentence-transformers -- which degrades to no-gating below -- never sees a misleading
                # "loading" line. stderr + ASCII-only (Windows console cp1252 chokes on unicode).
                print(f"[memory] loading the topic-gate model ({self.model_name}); "
                      f"first use, a few seconds...", file=sys.stderr, flush=True)
                self._model = SentenceTransformer(self.model_name)
            except Exception:
                # no embedder available (or failed to build) -> latch off, degrade to no-gating forever
                self.ok = False
                self._model = None
                return False
        return True

    def _embed(self, text: str):
        """Normalized embedding for one string (cached). Returns None if the embedder is unusable or the
        encode raised -- callers treat None as 'can't gate' and fall back to no-gating."""
        if text in self._cache:
            return self._cache[text]
        if not self._ensure_model():
            return None
        try:
            # normalize_embeddings=True -> unit vectors, so a plain dot product is the cosine similarity.
            vec = self._model.encode(text, normalize_embeddings=True)
        except Exception:
            self.ok = False                  # a broken encode also degrades to no-gating
            return None
        with self._lock:
            self._cache[text] = vec
        return vec

    def _max_cos(self, prompt_vec, texts: list[str]) -> float:
        """Best cosine of a (unit) prompt vector against the embeddings of `texts`. 0.0 if none embed."""
        best = 0.0
        seen = False
        for t in texts:
            tv = self._embed(t)
            if tv is None:
                continue
            try:
                cos = float((prompt_vec * tv).sum())     # unit vectors -> dot == cosine
            except Exception:
                continue
            if not seen or cos > best:
                best, seen = cos, True
        return best if seen else 0.0

    def _open_vecs_ready(self) -> bool:
        """Warm the open-personal-ref embeddings into the cache once (best-effort). They're embedded lazily
        by _embed anyway; this just primes them together on first openness scoring."""
        for t in OPEN_PERSONAL_REFS:
            self._embed(t)
        return self.ok

    # ---- relevance: per-rule cosine to the prompt -------------------------------------------------
    def relevance(self, prompt: str, rule_texts: list[str]) -> dict[str, float]:
        """{rule_text: cosine(embed(prompt), embed(rule))} for each active rule, rounded.

        Returns {} when there are no rules or the embedder is unavailable (self.ok False). Any rule whose
        embedding can't be produced is skipped rather than aborting the whole map."""
        if not rule_texts or not self.ok:
            return {}
        pv = self._embed(prompt)
        if pv is None:
            return {}
        out: dict[str, float] = {}
        for rt in rule_texts:
            rv = self._embed(rt)
            if rv is None:
                continue
            try:
                cos = float((pv * rv).sum())         # unit vectors -> dot == cosine
            except Exception:
                continue
            out[rt] = round(cos, 4)
        return out

    def openness(self, prompt: str) -> float:
        """Max cosine of the prompt to the OPEN_PERSONAL_REFS -- how much this reads as an open, advice-
        seeking PERSONAL ask (where a personal memory should fire even with no named topic). 0.0 if the
        embedder is unavailable."""
        if not self.ok:
            return 0.0
        pv = self._embed(prompt)
        if pv is None:
            return 0.0
        self._open_vecs_ready()
        return round(self._max_cos(pv, OPEN_PERSONAL_REFS), 4)

    # ---- scalar: THE gate in [0,1] ----------------------------------------------------------------
    def scalar(self, prompt: str, rule_texts: list[str],
               lo_t: float = lo_t, hi_t: float = hi_t,
               lo_o: float = lo_o, hi_o: float = hi_o) -> float:
        """The injection gate g in [0,1]: how strongly the memory should fire for this prompt.

        g = max( _map(topic, lo_t, hi_t), _map(openness, lo_o, hi_o) ), where
          topic    = max cosine(prompt, rule_texts)          -- the best-matching remembered topic, and
          openness = max cosine(prompt, OPEN_PERSONAL_REFS)  -- how open/advice-seeking-personal it reads.
        So the memory fires when the prompt is on a remembered topic OR is a general personal ask, and
        stays ~0 for unrelated specific tasks (which score low on both).

        Returns 1.0 -- meaning NO gating, exactly the always-on baseline -- when the embedder is
        unavailable (self.ok False) OR there are no active rules. That's the safety contract: a missing
        embedder or an empty memory never regresses behavior below what it was before gating."""
        if not self.ok or not rule_texts:
            return 1.0
        pv = self._embed(prompt)
        if pv is None:                       # embed failed -> can't gate -> no gating
            return 1.0
        topic = self._max_cos(pv, rule_texts)
        self._open_vecs_ready()
        openness = self._max_cos(pv, OPEN_PERSONAL_REFS)
        return _clamp01(max(_map(topic, lo_t, hi_t), _map(openness, lo_o, hi_o)))

    # ---- debug: both raw signals + the gate, for live calibration ---------------------------------
    def debug(self, prompt: str, rule_texts: list[str]) -> dict:
        """Everything the calibrator wants for one prompt: the final gate, the two raw signals, and the
        per-rule cosines. Defensive -- never raises; returns gate 1.0 (baseline) when the embedder is off."""
        if not self.ok:
            return {"gate": 1.0, "topic": 0.0, "openness": 0.0, "relevance": {}, "ok": False}
        rel = self.relevance(prompt, rule_texts)
        topic = round(max(rel.values()), 4) if rel else 0.0
        openv = self.openness(prompt)
        gate = self.scalar(prompt, rule_texts)
        return {"gate": round(float(gate), 4), "topic": topic, "openness": openv,
                "relevance": rel, "ok": True}


# ---- process-wide singleton: the 80MB model loads ONCE per process --------------------------------
_GATE: TopicGate | None = None
_GATE_LOCK = threading.Lock()


def get_gate() -> TopicGate:
    """A shared TopicGate for the whole process, so the sentence-transformer is constructed at most once.
    Thread-safe (double-checked lock); the returned instance is safe to call concurrently (its own first
    load + cache writes are locked)."""
    global _GATE
    if _GATE is None:
        with _GATE_LOCK:
            if _GATE is None:
                _GATE = TopicGate()
    return _GATE
