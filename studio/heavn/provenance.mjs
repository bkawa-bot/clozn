/* heavnOS · provenance — plain-language surface for POST /runs/<id>/provenance (the attention-knockout
   context-vs-parametric receipt: clozn/analysis/provenance.py, clozn/server/routes/provenance.py).
   One call site: the Monitor's answer-source CHIP (replay.mjs, next to PolicyChip) -- whole-run,
   because the wired route only ever scores ONE thing per run (the recorded answer's own first
   generated token, optionally focused to one prompt region); there is no per-continuation-position
   parameter on this route. (The click-a-token popover's own "trace this token" action used to fall
   back to this same route for a lack of a better option; it now calls the real per-position
   POST /runs/<id>/causal-trace instead -- clozn/server/routes/causal_trace.py -- wired directly in
   replay.mjs's Pop, not through this module.) The chip is honest about its whole-run scope, never
   claiming more precision than the route delivers.

   BK's plain-language choice for the verdict labels (task spec, verbatim) -- NO strength/confidence word
   added, since that would over-claim the knockout margin:
     CONTEXT_CARRIED -> "Answered from your context"
     MIXED           -> "From context + the model"
     PARAMETRIC      -> "From the model's own knowledge"
     INCONCLUSIVE    -> "Couldn't determine the source"
   blocked / ok:false -> a quiet "provenance unavailable" state, never a crash, never a fake verdict. */
import { html, useState, useEffect, useRef } from "./vendor/preact-standalone.mjs";
import { api } from "./api.mjs";

const PLAIN_VERDICT_LABEL = {
  CONTEXT_CARRIED: "Answered from your context",
  MIXED: "From context + the model",
  PARAMETRIC: "From the model's own knowledge",
  INCONCLUSIVE: "Couldn't determine the source",
};

/** receipt -> the plain-language label. An unrecognized/absent verdict falls back to the honest
    "couldn't determine" phrasing rather than fabricating a verdict-shaped string. `ok` is checked
    loosely (falsy, not just `=== false`) so a 404/400 wire shape (no `ok` key at all) reads as
    unavailable too, instead of falling through to a fabricated verdict lookup. */
export function provenanceLabel(receipt){
  if(!receipt || !receipt.ok) return "Provenance unavailable";
  return PLAIN_VERDICT_LABEL[String(receipt.verdict)] || "Couldn't determine the source";
}

/** Any honest reason text a blocked/failed receipt carries (blocked, or a plain error string from a
    404/400 wire shape) -- never fabricated, "" when the receipt truly carries none. */
export function provenanceReason(r){
  return (r && (r.blocked || r.error)) || "";
}

/* Same substance as clozn/analysis/provenance.py's SCOPE_NOTE -- the route's JSON carries no scope
   field (only the CLI's --json path adds it at print time), so this is Studio's own honest restatement,
   not a verbatim wire echo. Kept in sync with that note's own 2026-07-23 correction (it used to say
   "one model family"; it's two -- Qwen2.5-7B + Llama-3.1-8B -- and the 41/41 figure only reproduces
   under the CURRENT grading code, not the stale stored battery summaries): never quote a number this
   caveat doesn't also qualify. */
const SCOPE_CAVEAT = "attention-knockout measurement · validated two-family (Qwen2.5-7B + "
  + "Llama-3.1-8B), 41/41 under current grading · read this as evidence, not proof";

const numFmt = (v, d = 2) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "?";

/** The honest detail rows behind the plain label: verdict enum verbatim, dependence, best control
    ratio, the carrying span tokens (or the honest "(none)" + note), and focus_null's p-value when
    present. `answer` is rendered verbatim too -- it is literally what the route scored (the run's
    recorded response's own first generated token; see this module's docstring), so showing it lets
    a reader see the true scope of the measurement instead of trusting a paraphrase. */
export function provenanceRows(r){
  const rows = [["scored text", r.answer != null ? JSON.stringify(String(r.answer)) : "—"]];
  if(r.focus) rows.push(["focus", "prompt tokens [" + r.focus[0] + ", " + r.focus[1] + ")"]);
  rows.push(["verdict", String(r.verdict || "?")]);
  rows.push(["dependence", numFmt(r.dependence)]);
  rows.push(["best control ratio",
    typeof r.best_control_ratio === "number" ? numFmt(r.best_control_ratio, 1) + "x" : "n/a"]);
  rows.push(["carrying span", (r.span_tokens && r.span_tokens.length)
    ? r.span_tokens.map(t => JSON.stringify(String(t))).join(" ")
    : "(none)" + (r.note ? " — " + r.note : "")]);
  if(r.focus_null) rows.push(["focus_null p-value",
    String(r.focus_null.p_value) + " (n=" + r.focus_null.n_draws + ", evidence only)"]);
  return rows;
}

/* module-level cache: one provenance call per run, shared by the chip and the Pop's "trace this token"
   action so clicking both never double-fires the real engine-side attention-knockout work. Memoizes the
   PROMISE (not just the settled value) so concurrent callers dedupe too -- mirrors this file's own
   jlCache pattern in replay.mjs. */
const provCache = new Map();
export function getProvenance(rec){
  if(!rec || !rec.id) return Promise.resolve({ ok: false, blocked: "no run to check" });
  if(!provCache.has(rec.id)){
    provCache.set(rec.id, api.provenance(rec.id, {})
      .then(r => r || { ok: false, blocked: "the server didn't answer" }));
  }
  return provCache.get(rec.id);
}

function ProvenanceDetail({ r }){
  if(!r || !r.ok)
    return html`<p>${provenanceReason(r) || "needs a cloze-server started with --no-flash-attn"}</p>`;
  return html`<div>
    ${provenanceRows(r).map(([k, v]) => html`<div class="prov-chip-row"><span>${k}</span><b>${v}</b></div>`)}
    <p>${SCOPE_CAVEAT}</p>
  </div>`;
}

/** PIECE 1 -- the answer-source chip (embedded where a run shows its answer: replay.mjs's Monitor,
    next to PolicyChip). idle (not yet checked) -> busy -> done (plain label; honest detail on expand)
    or a quiet "provenance unavailable". Never auto-fires: provenance is a real engine call (attention
    knockout + matched-random controls), so -- like ReceiptsPanel/ExplainPanel elsewhere in this studio,
    and per replay.mjs's own module docstring ("computed-on-demand receipts never implied to
    pre-exist") -- it waits for a click rather than firing on every run view. */
export function ProvenanceChip({ rec }){
  const [state, setState] = useState({ status: "idle", receipt: null });
  const nonce = useRef(0);
  useEffect(() => { nonce.current += 1; setState({ status: "idle", receipt: null }); }, [rec && rec.id]);
  if(!rec || rec._sample) return null;   // sample reel: no journal record to check -- honest absence

  const check = async () => {
    if(state.status !== "idle") return;
    const my = ++nonce.current;
    setState({ status: "busy", receipt: null });
    const r = await getProvenance(rec);
    if(my !== nonce.current) return;     // a fast run switch made this answer stale -- drop it
    setState({ status: "done", receipt: r });
  };

  if(state.status === "idle")
    return html`<button class="prov-chip-idle" data-testid="provenance-chip"
      title="Check whether this answer came from your context or the model's own knowledge (attention-knockout; needs the engine started with --no-flash-attn)"
      onClick=${check}>◌ answer source</button>`;
  if(state.status === "busy")
    return html`<span class="prov-chip-busy" data-testid="provenance-chip">checking source…</span>`;

  const r = state.receipt;
  if(!r || !r.ok)
    return html`<span class="prov-chip-unavailable" data-testid="provenance-chip"
      title=${"provenance unavailable — " + (provenanceReason(r) || "needs a cloze-server started with --no-flash-attn")}
      >provenance unavailable</span>`;

  return html`<details class="prov-chip" data-testid="provenance-chip">
    <summary title=${provenanceLabel(r) + " — click to expand the honest detail"}>
      <i></i><span>${provenanceLabel(r)}</span></summary>
    <div class="prov-chip-body"><${ProvenanceDetail} r=${r}/></div>
  </details>`;
}
