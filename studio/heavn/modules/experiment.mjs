/* heavnOS · Experiment — the drawer over the ONE experiment primitive: "hold everything constant,
   change one thing, compare, with a receipt." One endpoint (POST /runs/<id>/experiment) dispatches
   nine change-types (ablate_card, ablate_memory, ablate_dial, set_dial, swap_concept, edit_turn,
   reroll, toggle_greedy, anchored_recall) behind ONE normalized envelope -- this module is a thin,
   honest client over it: enumerate the catalog, fill a change spec, show the cost BEFORE running,
   run, render the envelope in layered form. Contract: clozn/experiments/experiment.py (envelope +
   REGISTRY, read for the exact shape + the dispatch/error conventions).

   HONESTY rules carried over from patch.mjs/replay.mjs (do not weaken these):
   - has_effect / causal_verified are TRI-STATE (true/false/null). null is NEVER a green check or a
     fabricated "no effect" -- it renders as the neutral "no automatic verdict -- compare directly".
   - result.null (the random/held control) is shown beside the result whenever present, never hidden;
     when the underlying op has none at all, that absence itself is stated, not silently omitted.
   - result.receipt.blocked/note (a degraded/blocked receipt) is surfaced verbatim, same as
     swap_receipt's res.blocked handling in patch.mjs.
   - the catalog's cost_hint renders before a run is possible; the envelope's own grounded cost
     (passes/note/est_seconds) renders again after, next to the result. */
import { html, useState, useEffect, useRef } from "../vendor/preact-standalone.mjs";
import { store, useStore, toast } from "../state.mjs";
import { api } from "../api.mjs";

/* true when live actions should be blocked (server unreachable, or the record is the sample reel) */
function guardLive(rec){
  if(!store.get().live || (rec && rec._sample)){ toast("live server only"); return true; }
  return false;
}
const fmt = v => (typeof v === "number" ? v.toFixed(2) : (v ?? "—"));
const trunc = (s, n = 90) => { s = s == null ? "" : String(s); return s.length <= n ? s : s.slice(0, n - 1).trimEnd() + "…"; };

/* fields the registry's `needs` doesn't list but the handler still accepts (contracts: swap_concept's
   optional from_hint label, edit_turn's optional alt_user) -- offered alongside the required ones */
const EXTRA_FIELDS = { swap_concept: ["from_hint"], edit_turn: ["alt_user"] };
/* method choices where the underlying op actually has one (experiment.py: _RECEIPT_MODES /
   _resolve_branch_sample) -- every other type ignores `method` server-side, so no picker is shown */
const METHOD_OPTIONS = {
  ablate_card: ["regen", "forced", "both"], ablate_memory: ["regen", "forced", "both"],
  ablate_dial: ["regen", "forced", "both"], edit_turn: ["greedy", "sample"],
};
const FIELD_META = {
  card_id:    { label: "card_id",    placeholder: "memory card id",             type: "text" },
  dial:       { label: "dial",       placeholder: "e.g. concise",               type: "text" },
  value:      { label: "value",      placeholder: "e.g. 0.6",                   type: "number" },
  to_concept: { label: "to_concept", placeholder: "e.g. ocean",                 type: "text" },
  from_hint:  { label: "from_hint (label only, optional)", placeholder: "e.g. Paris", type: "text" },
  turn:       { label: "turn",       placeholder: "turn index (0-based)",       type: "number" },
  alt_user:   { label: "alt_user (optional)", placeholder: "ask something different…", type: "text" },
};
const metaFor = k => FIELD_META[k] || { label: k, placeholder: k, type: "text" };  // forward-compatible
                                                                                     // with a registry
                                                                                     // type this build
                                                                                     // doesn't know yet

/* ───────────────────────── module root ───────────────────────── */
export function ExperimentModule(){
  const rec = useStore(x => x.rec);
  const live = useStore(x => x.live);

  const [types, setTypes] = useState(null);   // null = loading; {} on failure/empty; else the catalog
  const [ctype, setCtype] = useState("");
  const [fields, setFields] = useState({});
  const [method, setMethod] = useState("");
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);
  /* click-a-span handoff: {ctype, fields, method} stashed here on mount, applied by the ctype-reset
     effect below ONCE ctype actually equals its ctype (so it survives that effect's own clear) */
  const pendingRef = useRef(null);

  useEffect(() => { (async () => {
    const r = await api.experimentTypes();
    setTypes(r && r.types ? r.types : {});
  })(); }, []);

  /* consume a click-a-span deep-link ONCE on mount: pre-select its change-type (the ctype effect then
     applies the stashed fields). A span's "open in Experiment" action set this in the store before
     switching the nav tab here. */
  useEffect(() => {
    const p = store.get().pendingExperiment;
    if(p && p.ctype){
      pendingRef.current = { ctype: p.ctype, fields: p.fields || {}, method: p.method || "" };
      store.set({ pendingExperiment: null });   // consumed -- never re-applies on a later remount
      setCtype(p.ctype);
    }
  }, []);

  useEffect(() => {
    if(!ctype && types && Object.keys(types).length) setCtype(Object.keys(types)[0]);
  }, [types]);

  /* switching change-type resets the form + any stale result -- never carry a receipt across specs.
     EXCEPTION: a pending click-a-span handoff FOR THIS ctype pre-fills the form instead of clearing
     (keyed on ctype so it fires exactly when ctype has become the handoff's type, not before). */
  useEffect(() => {
    setMethod(""); setRes(null); setErr(null);
    if(pendingRef.current && pendingRef.current.ctype === ctype){
      setFields(pendingRef.current.fields);
      setMethod(pendingRef.current.method);
      pendingRef.current = null;
    } else {
      setFields({});
    }
  }, [ctype]);

  const entry = (types && ctype) ? types[ctype] : null;
  const needs = entry ? entry.needs : [];
  const extra = EXTRA_FIELDS[ctype] || [];
  const methodOpts = METHOD_OPTIONS[ctype] || null;
  const setField = (k, v) => setFields(f => ({ ...f, [k]: v }));
  const missing = needs.filter(k => fields[k] === undefined || fields[k] === null || String(fields[k]).trim() === "");

  async function run(){
    if(!rec){ toast("need a current run — load one from Replay first"); return; }
    if(guardLive(rec)) return;
    if(!ctype){ toast("pick a change type first"); return; }
    if(missing.length){ toast("needs " + missing.join(", ")); return; }
    const change = { type: ctype };
    [...needs, ...extra].forEach(k => {
      const v = fields[k];
      if(v === undefined || v === null || String(v).trim() === "") return;
      change[k] = metaFor(k).type === "number" ? Number(v) : v;
    });
    setBusy(true); setErr(null); setRes(null);
    toast("running the experiment — hold everything constant, change one thing…");
    const r = await api.runExperiment(rec.id, change, methodOpts ? (method || undefined) : undefined);
    setBusy(false);
    if(!r){ setErr("no response from the server — is it up?"); return; }
    if(r.__status && r.__status >= 400){ setErr(r.error || ("experiment failed (" + r.__status + ")")); return; }
    setRes(r);
  }

  return html`<div class="col">
    <${TypePicker} types=${types} ctype=${ctype} setCtype=${setCtype} entry=${entry}/>
    <${FormPanel} rec=${rec} live=${live} entry=${entry} ctype=${ctype} needs=${needs} extra=${extra}
      fields=${fields} setField=${setField} methodOpts=${methodOpts} method=${method} setMethod=${setMethod}
      busy=${busy} run=${run} missing=${missing}/>
    ${err && html`<div class="mod">
      <div class="mod-h"><span class="led" style="background:var(--coral);box-shadow:0 0 8px var(--coral)"></span>
        <span class="cap">experiment</span><span class="tag fail-t">FAILED</span></div>
      <div style="padding:0 14px 12px" class="none">${err}</div>
    </div>`}
    ${res && html`<${ExperimentResult} res=${res}/>`}
  </div>`;
}

/* ───────────────────────── A) change-type picker ───────────────────────── */
function TypePicker({ types, ctype, setCtype, entry }){
  const loading = types === null;
  const empty = types && !Object.keys(types).length;
  return html`<div class="mod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led"></span><span class="cap">experiment — change one thing</span>
      <span class="tail">${loading ? "reading /experiments/types…" : (empty ? "no types" : Object.keys(types).length + " change-types")}</span></div>
    ${loading && html`<div class="none" style="padding:8px 14px 12px">reading the catalog…</div>`}
    ${empty && html`<div class="none" style="padding:8px 14px 12px">no experiment types reported by the server — is it up?</div>`}
    ${!loading && !empty && html`<div style="padding:2px 14px 8px;display:flex;gap:10px;flex-wrap:wrap;align-items:end">
      <label style="display:flex;flex-direction:column;gap:3px;font-size:8.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mist)">
        change.type
        <select value=${ctype} onChange=${e => setCtype(e.target.value)}
          style="font:inherit;font-size:10.5px;padding:5px 8px;border-radius:7px;border:1px solid var(--edge);background:linear-gradient(180deg,#fff,#E6F1F7);color:var(--navy);min-width:260px">
          ${Object.entries(types).map(([t,e]) => html`<option key=${t} value=${t}>${t} — ${e.label}</option>`)}
        </select>
      </label>
    </div>`}
    ${entry && html`<div class="cfg" style="margin:0 14px 12px">
      <span class="cap">cost, before you run it</span>
      <span>${entry.cost_hint}</span>
    </div>`}
  </div>`;
}

/* ───────────────────────── B) the form for the selected type ───────────────────────── */
function FormPanel({ rec, live, entry, ctype, needs, extra, fields, setField, methodOpts, method, setMethod, busy, run, missing }){
  if(!entry) return null;
  const allKeys = [...needs, ...extra];
  const cardChips = ctype === "ablate_card" && rec && rec.memory
    ? (rec.memory.applied_ids || []).map((id,i) => ({ id, label: (rec.memory.cards_applied || [])[i] || id }))
    : [];
  const dialChips = (ctype === "ablate_dial" || ctype === "set_dial") && rec && rec.behavior
    ? Object.keys(rec.behavior.active_dials || {}) : [];

  return html`<div class="mod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led lilac"></span><span class="cap">${entry.label}</span>
      <span class="tail">${busy ? "running…" : "on demand"}</span></div>

    <div style="padding:2px 14px 4px;display:flex;gap:10px;flex-wrap:wrap;align-items:end">
      ${allKeys.map(k => {
        const meta = metaFor(k);
        return html`<label key=${k} style="display:flex;flex-direction:column;gap:3px;font-size:8.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mist)">
          ${meta.label}
          <input type=${meta.type} value=${fields[k] ?? ""} placeholder=${meta.placeholder}
            onInput=${e => setField(k, e.target.value)}
            style="font:inherit;font-size:10.5px;padding:5px 8px;border-radius:7px;border:1px solid var(--edge);background:linear-gradient(180deg,#fff,#E6F1F7);color:var(--navy);min-width:150px"/>
        </label>`;
      })}
      ${methodOpts && html`<label style="display:flex;flex-direction:column;gap:3px;font-size:8.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mist)">
        method
        <select value=${method} onChange=${e => setMethod(e.target.value)}
          style="font:inherit;font-size:10.5px;padding:5px 8px;border-radius:7px;border:1px solid var(--edge);background:linear-gradient(180deg,#fff,#E6F1F7);color:var(--navy);min-width:120px">
          <option value="">default (${methodOpts[0]})</option>
          ${methodOpts.map(m => html`<option key=${m} value=${m}>${m}</option>`)}
        </select>
      </label>`}
      <button class=${"spd" + (busy ? " busy" : "")} onClick=${run}>${busy ? "RUNNING…" : "RUN EXPERIMENT"}</button>
    </div>

    ${!allKeys.length && html`<div class="none" style="padding:0 14px 8px">this change type needs no extra fields.</div>`}

    ${(cardChips.length > 0 || dialChips.length > 0) && html`<div style="padding:0 14px 8px;display:flex;gap:6px;flex-wrap:wrap;align-items:center">
      <span class="none">from this run:</span>
      ${cardChips.map(c => html`<button key=${c.id} class="spd" style="font-size:9px"
        onClick=${() => setField("card_id", c.id)}>${trunc(c.label, 28)}</button>`)}
      ${dialChips.map(d => html`<button key=${d} class="spd" style="font-size:9px"
        onClick=${() => setField("dial", d)}>${d}</button>`)}
    </div>`}

    ${!rec && html`<div class="none" style="padding:0 14px 12px">need a current run — load one from Replay first.</div>`}
    ${rec && !live && html`<div class="none" style="padding:0 14px 12px">live server only — this is the sample reel.</div>`}
    ${rec && live && missing.length > 0 && html`<div class="none" style="padding:0 14px 12px">needs: ${missing.join(", ")}</div>`}
    ${busy && html`<div class="none" style="padding:0 14px 12px">running — ${entry.cost_hint}</div>`}
  </div>`;
}

/* ───────────────────────── C) the layered result ───────────────────────── */
function ExperimentResult({ res }){
  const result = res.result || {};
  const cost = res.cost || {};
  const change = res.change || {};
  const receipt = result.receipt || {};
  return html`<div class="mod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led lilac"></span><span class="cap">experiment result</span>
      <span class="tail">${change.label || change.type} · ${res.method || "—"}</span></div>

    <div style="padding:2px 14px 2px">
      <div class="none">${res.question}</div>
      <div style="font-size:11px;color:var(--navy);line-height:1.7;padding:6px 0 2px;font-weight:500">${result.plain}</div>
    </div>

    ${receipt.blocked && html`<div class="cfg" style="margin:6px 14px 0;border-left-color:var(--coral)">
      <span class="tag fail-t">BLOCKED · ${receipt.blocked}</span>
      <span>${receipt.note || "no further detail given."}</span>
    </div>`}

    <div style="padding:8px 14px 4px;display:grid;grid-template-columns:1fr 1fr;gap:10px">
      <div><div class="none" style="padding:0 0 4px">baseline reply</div>
        <div class="leader-body" style="border:1px solid var(--edge-soft);border-radius:8px;max-height:220px;margin:0">${res.baseline && res.baseline.reply || "—"}</div></div>
      <div><div class="none" style="padding:0 0 4px">changed reply${result.changed_reply == null ? " (none generated — see the plain summary above)" : ""}</div>
        <div class="leader-body" style="border:1px solid rgba(95,200,188,.5);border-radius:8px;max-height:220px;margin:0;background:rgba(95,200,188,.06)">${result.changed_reply ?? "—"}</div></div>
    </div>

    <div class="cfg" style="margin:8px 14px 0">
      <span class="cap">verdict</span>
      <${VerdictTag} field="has_effect" value=${result.has_effect}/>
      <${VerdictTag} field="causal_verified" value=${result.causal_verified}/>
    </div>

    <div style="margin:0 14px 0"><${DeltaStrip} delta=${result.delta}/></div>
    <div style="margin:0 14px 0"><${NullControl} nul=${result.null}/></div>

    <div class="cfg" style="margin:8px 14px 12px">
      <span class="cap">cost</span>
      <span>passes <b>${cost.passes ?? "—"}</b></span>
      ${cost.est_seconds != null && html`<span>~${cost.est_seconds}s (grounded in this run's own recorded timing)</span>`}
      <span style="flex-basis:100%">${cost.note || "—"}</span>
    </div>

    <details class="leader" style="margin:0 14px 14px">
      <summary>raw receipt (result.receipt) — the full underlying object, verbatim
        <span style="margin-left:auto">run ${res.run_id || "—"}</span></summary>
      <div class="leader-body" style="white-space:pre-wrap">${JSON.stringify(result.receipt ?? null, null, 2)}</div>
    </details>
  </div>`;
}

/* has_effect / causal_verified are TRI-STATE: true, false, or null (no automatic verdict at all --
   the underlying op simply computes none, e.g. branch()/replay() for edit_turn/reroll/toggle_greedy,
   or swap_receipt's has_effect which is always None -- see experiment.py's module docstring). null
   NEVER renders as a pass/fail; it renders as the same neutral "compare directly" state every time. */
function VerdictTag({ field, value }){
  const cls = value === true ? "cap-t" : value === false ? "fail-t" : "smp-t";
  let text;
  if(value === true) text = field === "has_effect" ? "changed the answer" : "verified";
  else if(value === false) text = field === "has_effect" ? "no effect on the greedy answer" : "not verified";
  else text = "no automatic verdict — compare directly";
  return html`<span class="tag ${cls}">${field}: ${text}</span>`;
}

function DeltaStrip({ delta }){
  if(!delta) return html`<div class="cfg" style="border-left-color:var(--mist)">
    <span class="cap">delta</span>
    <span>no delta — baseline and/or changed reply were absent for this change type.</span>
  </div>`;
  return html`<div class="cfg">
    <span class="cap">delta</span>
    <span>words ${Array.isArray(delta.words) ? delta.words.join(" → ") : "—"}</span>
    <span>wps ${Array.isArray(delta.wps) ? delta.wps.map(x => fmt(+x)).join(" → ") : "—"}</span>
    <span>changed ${delta.changed != null ? delta.changed + "%" : "—"}</span>
  </div>`;
}

/* result.null: the random/held control -- SHOWN whenever the server returns one, never hidden.
   Its shape varies by change type (a swap_receipt null arm has {available, reply, lexicon_hits,
   logprob, swap_over_null_nat, note}; a forced-mode ablate's null_floor has {kind, deltas, sum_nats,
   mean_nats_per_token, ratio_real_over_floor, exceeds_floor_by_order_of_magnitude}) -- rendered
   generically rather than assuming one shape, so nothing in it is ever silently dropped. */
function NullControl({ nul }){
  if(nul == null) return html`<div class="cfg" style="border-left-color:var(--mist)">
    <span class="cap">null control</span>
    <span>none for this change type — read this result as a direct before/after compare, not a
      proven random-baseline delta.</span>
  </div>`;
  const skip = new Set(["reply", "note"]);
  const entries = Object.entries(nul).filter(([k]) => !skip.has(k));
  return html`<div class="cfg" style="border-left-color:var(--lilac)">
    <span class="cap">null control <span style="opacity:.75;text-transform:none;letter-spacing:normal">(random/held baseline — never hidden)</span></span>
    ${nul.reply != null && html`<span>reply: “${trunc(nul.reply)}”</span>`}
    ${entries.map(([k,v]) => html`<span key=${k}>${k} ${Array.isArray(v) ? v.join(", ") : (typeof v === "number" ? fmt(v) : String(v))}</span>`)}
    ${nul.note && html`<span style="flex-basis:100%">${nul.note}</span>`}
  </div>`;
}
