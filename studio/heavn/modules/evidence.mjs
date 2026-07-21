/* heavnOS · Evidence home — a read-only index over evidence already attached to the current run.
   It does not run scoring, generation, calibration, trust, or lens endpoints. The shared object model
   owns normalization; this module owns ordering, honest absence, and navigation. */
import { html } from "../vendor/preact-standalone.mjs";
import { store, useStore } from "../state.mjs";
import { api } from "../api.mjs";
import { normalizeRun as runObject, normalizeEvidence as evidenceObjects,
         normalizeExperiment as experimentObject } from "../object_model.mjs";


const CARD_SPECS = [
  {
    key: "context-answer",
    title: "Context ↔ answer map",
    aliases: ["context_answer", "context-answer", "influence_map", "influence-map", "context answer"],
    absent: "No stored context-to-answer map is attached to this run.",
  },
  {
    key: "causal-receipts",
    title: "Causal receipts",
    aliases: ["causal", "receipt", "intervention", "counterfactual", "ablation"],
    absent: "No stored intervention receipt is attached to this run.",
  },
  {
    key: "calibration-trust",
    title: "Calibration and trust",
    aliases: ["calibration", "trust", "actuary", "confidence"],
    absent: "No stored calibration or trust evidence is attached to this run.",
  },
  {
    key: "lens-workspace",
    title: "Lens and workspace readouts",
    aliases: ["lens", "workspace", "readout", "concept", "activation"],
    absent: "No stored lens or workspace readout is attached to this run.",
  },
];

const isObject = value => !!value && typeof value === "object" && !Array.isArray(value);

function callObjectModel(fn, ...args){
  try{ return fn(...args); }catch(_error){ return null; }
}

function objectList(value){
  if(Array.isArray(value)) return value.filter(isObject);
  if(!isObject(value)) return [];
  const nested = value.objects || value.evidence || value.items;
  if(Array.isArray(nested)) return nested.filter(isObject);
  return Object.entries(value).flatMap(([key, item]) => {
    if(isObject(item)) return [{ object_key: key, ...item }];
    if(Array.isArray(item)) return item.filter(isObject).map(child => ({ object_key: key, ...child }));
    return [];
  });
}

function path(value, candidates){
  for(const candidate of candidates){
    let current = value;
    for(const part of candidate.split(".")) current = current && current[part];
    if(current !== undefined && current !== null && current !== "") return current;
  }
  return null;
}

function identity(run, rec){
  return {
    runId: path(run, ["identity.run_id", "run_id", "id"]) ?? path(rec, ["id", "run_id"]),
    model: path(run, ["identity.model", "model"]) ?? path(rec, ["model"]),
    substrate: path(run, ["identity.substrate", "substrate"]) ?? path(rec, ["substrate"]),
    created: path(run, ["created_at", "created_ts"]) ?? path(rec, ["created_at", "created_ts"]),
    schema: path(run, ["schema", "object_schema"]),
  };
}

function evidenceKind(value){
  return [value.kind, value.type, value.evidence_type, value.category, value.object_key,
          value.schema, value.title, value.label]
    .filter(item => item != null).join(" ").toLowerCase().replaceAll("-", "_").replaceAll("↔", "_");
}

function matches(spec, value){
  const kind = evidenceKind(value);
  return spec.aliases.some(alias => kind.includes(alias.replaceAll("-", "_")));
}

function display(value, fallback = "not recorded"){
  if(value === undefined || value === null || value === "") return fallback;
  if(typeof value === "string") return value;
  if(typeof value === "number" || typeof value === "boolean") return String(value);
  if(Array.isArray(value)) return value.length ? value.map(item => display(item, "")).join(" · ") : fallback;
  if(isObject(value)){
    const preferred = value.name || value.label || value.id || value.kind || value.status;
    if(preferred && Object.keys(value).length === 1) return String(preferred);
    try{
      const text = JSON.stringify(value);
      return text.length > 420 ? text.slice(0, 419) + "…" : text;
    }catch(_error){ return fallback; }
  }
  return String(value);
}

function statusOf(value){
  if(value.available === false) return "unavailable";
  if(value.status != null) return String(value.status);
  if(value.available === true) return "available";
  return "stored";
}

function statusClass(status){
  const normalized = String(status).toLowerCase();
  if(["ok", "available", "stored", "complete", "verified"].includes(normalized)) return "cap-t";
  if(["error", "failed", "blocked"].includes(normalized)) return "fail-t";
  return "smp-t";
}

function evidenceFields(value){
  const method = value.method ?? value.measurement ?? value.methodology;
  const controls = value.controls ?? value.control ?? value.interventions ?? value.counterfactual;
  const latency = value.latency ?? value.latency_ms ?? value.timing ?? value.cost;
  const artifact = value.artifact ?? value.artifact_hash ?? value.artifact_sha256 ?? value.artifact_id ?? value.schema;
  const normalizedProvenance = (value.source_run_id != null || value.claim_limit != null)
    ? { source_run_id: value.source_run_id ?? null, claim_limit: value.claim_limit ?? null } : null;
  const provenance = value.provenance ?? value.identity ?? value.source ?? normalizedProvenance;
  return [
    ["method", method], ["controls", controls], ["latency", latency],
    ["artifact", artifact], ["provenance", provenance],
  ];
}

function EvidenceObject({ value, index }){
  const status = statusOf(value);
  const label = value.title || value.label || value.name || value.evidence_type || value.kind
    || value.type || `stored object ${index + 1}`;
  const summary = value.note ?? value.description ?? value.summary ?? value.claim_limit;
  return html`<section class="evidence-home-object" aria-labelledby=${`evidence-home-object-${index}`}>
    <header class="evidence-home-object-head">
      <h3 id=${`evidence-home-object-${index}`}>${display(label)}</h3>
      <span class=${`tag ${statusClass(status)}`}>${status}</span>
    </header>
    ${summary != null && html`<p class="evidence-home-object-summary">${display(summary)}</p>`}
    <dl class="evidence-home-fields">
      ${evidenceFields(value).map(([term, field]) => html`
        <div class=${`evidence-home-field evidence-home-field-${term}`}>
          <dt>${term}</dt><dd>${display(field)}</dd>
        </div>`)}
    </dl>
  </section>`;
}

function EvidenceCard({ spec, values, cardIndex }){
  const headingId = `evidence-home-card-${spec.key}`;
  return html`<article class=${`mod evidence-home-card evidence-home-card-${spec.key}`}
    data-evidence-kind=${spec.key} aria-labelledby=${headingId}>
    <div class="mod-h evidence-home-card-head">
      <span class=${`led ${values.length ? "" : "off"}`}></span>
      <h2 class="cap" id=${headingId}>${spec.title}</h2>
      <span class="tail">${values.length ? `${values.length} stored object${values.length === 1 ? "" : "s"}` : "absent"}</span>
    </div>
    <div class="evidence-home-card-body">
      ${values.length
        ? values.map((value, index) => html`<${EvidenceObject} value=${value} index=${cardIndex * 100 + index}/>`)
        : html`<div class="evidence-home-absent" role="status">
            <span class="tag smp-t">NOT ATTACHED</span>
            <p>${spec.absent} Evidence Home never computes missing evidence.</p>
          </div>`}
    </div>
  </article>`;
}

function RunIdentity({ run, rec }){
  const exact = identity(run, rec);
  const fields = [
    ["run", exact.runId], ["model", exact.model], ["substrate", exact.substrate],
    ["created", exact.created], ["object schema", exact.schema],
  ];
  return html`<header class="workbench-titlebar evidence-home-identity" aria-labelledby="evidence-home-title">
    <div class="evidence-home-titlecopy">
      <span class="workbench-kicker">stored evidence index</span>
      <h1 id="evidence-home-title">Evidence</h1>
      <p>One run, its attached evidence objects, and their limits. Nothing on this page triggers model work.</p>
    </div>
    <dl class="evidence-home-identity-fields">
      ${fields.map(([term, value]) => html`<div class="evidence-home-identity-field">
        <dt>${term}</dt><dd><code>${display(value, "not recorded")}</code></dd>
      </div>`)}
    </dl>
  </header>`;
}

function EvidenceActions({ runId, experiment }){
  const experimentHint = path(experiment, ["handoff", "pending_experiment", "pendingExperiment"]);
  const openExperiment = () => store.set({
    route: "experiment",
    ...(isObject(experimentHint) ? { pendingExperiment: experimentHint } : {}),
  });
  return html`<nav class="mod evidence-home-actions" aria-label="Evidence destinations">
    <div class="mod-h"><span class="led blue"></span><span class="cap">continue with this run</span>
      <span class="tail">navigation only</span></div>
    <div class="evidence-home-action-list">
      <button type="button" class="spd" onClick=${() => store.set({ route: "replay" })}>OPEN REPLAY</button>
      <button type="button" class="spd" onClick=${openExperiment}>OPEN EXPERIMENT</button>
      ${runId
        ? html`<a class="spd evidence-home-card-link" href=${api.cardUrl(runId)} target="_blank" rel="noopener">
            OPEN OFFLINE RECEIPT</a>`
        : html`<span class="none evidence-home-card-link-absent">Offline receipt unavailable: run identity not recorded.</span>`}
    </div>
  </nav>`;
}

export function EvidenceModule(){
  const rec = useStore(state => state.rec);
  if(!rec) return html`<main class="col evidence-home-root evidence-home-empty" aria-labelledby="evidence-home-empty-title">
    <section class="mod evidence-home-no-run" role="status">
      <div class="mod-h"><span class="led off"></span><h1 class="cap" id="evidence-home-empty-title">Evidence</h1></div>
      <p class="none">No run is selected. Open Replay and choose a recorded run first.</p>
      <button type="button" class="spd" onClick=${() => store.set({ route: "replay" })}>OPEN REPLAY</button>
    </section>
  </main>`;

  const run = callObjectModel(runObject, rec) || rec;
  const exact = identity(run, rec);
  const evidence = objectList(callObjectModel(evidenceObjects, run.raw || rec, exact.runId));
  const storedExperiment = path(run, ["raw.experiment", "raw.latest_experiment", "raw.experiment_result"])
    ?? path(rec, ["experiment", "latest_experiment", "experiment_result"]);
  const experiment = callObjectModel(experimentObject, storedExperiment);
  return html`<main class="col evidence-home-root" data-run-id=${exact.runId || ""}>
    <${RunIdentity} run=${run} rec=${rec}/>
    <section class="evidence-home-list" aria-label="Stored evidence, ordered by decision relevance">
      ${CARD_SPECS.map((spec, index) => html`<${EvidenceCard} spec=${spec}
        values=${evidence.filter(value => matches(spec, value))} cardIndex=${index}/>`)}
    </section>
    <${EvidenceActions} runId=${exact.runId} experiment=${experiment}/>
  </main>`;
}
