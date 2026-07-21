/* Shared, dependency-free Studio object model.

   These pure normalizers do not fetch, mutate, score, or reinterpret evidence.
   They provide one stable shape for the run, evidence, and experiment views
   while retaining each source artifact verbatim under `raw`.
*/

const isObject = value => value !== null && typeof value === "object" && !Array.isArray(value);
const own = (value, key) => isObject(value) && Object.prototype.hasOwnProperty.call(value, key);
const first = (...values) => {
  const found = values.find(value => value !== undefined && value !== null);
  return found === undefined ? null : found;
};
const stringOrNull = value => value === undefined || value === null ? null : String(value);
const arrayOrEmpty = value => Array.isArray(value) ? value : [];

function explicitStatus(value){
  if(!isObject(value)) return null;
  if(typeof value.status === "string" && value.status.trim()) return value.status.trim();
  if(value.__status != null && Number.isFinite(+value.__status) && +value.__status >= 400) return "error";
  if(own(value, "error") && value.error) return "error";
  if(value.available === true) return "available";
  if(value.available === false) return "unavailable";
  if(value.causal_verified === true) return "verified";
  if(value.causal_verified === false) return "unverified";
  return null;
}

function sourceRunId(value, fallback = null){
  if(!isObject(value)) return fallback;
  return stringOrNull(first(value.source_run_id, value.run_id, fallback));
}

function evidenceMethod(value){
  if(!isObject(value)) return null;
  return first(
    value.method,
    value.mode,
    isObject(value.provenance) ? value.provenance.method : null,
    value.provider_type,
    value.provider,
    isObject(value.provenance) ? value.provenance.kind : null,
    isObject(value.identity) ? value.identity.method : null,
    null,
  );
}

function evidenceControls(value){
  if(!isObject(value)) return null;
  if(own(value, "controls")) return value.controls;
  if(own(value, "null_floor")) return value.null_floor;
  if(own(value, "null")) return value.null;
  if(isObject(value.result) && own(value.result, "null")) return value.result.null;
  if(isObject(value.forced) && own(value.forced, "null_floor")) return value.forced.null_floor;
  return null;
}

function evidenceLatency(value){
  if(!isObject(value)) return null;
  return first(
    value.latency,
    value.latency_ms,
    isObject(value.timing) ? first(value.timing.total_ms, value.timing.duration_ms) : null,
    value.duration_ms,
    null,
  );
}

function evidenceHash(value){
  if(!isObject(value)) return null;
  return stringOrNull(first(
    value.artifact_sha256,
    value.artifact_hash,
    value.sha256,
    isObject(value.artifact) ? first(value.artifact.sha256, value.artifact.hash) : null,
    isObject(value.provenance) ? first(
      value.provenance.artifact_sha256,
      value.provenance.manifest_sha256,
      value.provenance.lens_manifest_hash,
    ) : null,
    null,
  ));
}

function claimLimit(evidenceType){
  if(evidenceType === "influence_map")
    return "signed dependence under recorded-answer scoring and matched context controls; not an internal path explanation";
  if(evidenceType === "forced_receipt")
    return "dependence of the recorded answer's likelihood under intervention; not a claim that regeneration would differ";
  if(evidenceType === "receipt")
    return "the receipt's own intervention verdict and controls only; missing controls remain missing";
  if(evidenceType === "workspace_readout")
    return "a recorded provider readout at one workspace position; not a causal attribution";
  if(evidenceType === "jlens_readout")
    return "a fitted J-lens readout with its captured provenance; not a causal attribution";
  if(evidenceType === "truth_calibration")
    return "outcome-grounded calibration on the recorded model and task; not a per-answer guarantee";
  if(evidenceType === "trust")
    return "stored trust evidence under its stated proxy, calibration, and support limits";
  return null;
}

function evidenceObject(evidenceType, artifact, fallbackRunId = null){
  return {
    kind: "evidence",
    evidence_type: evidenceType,
    status: explicitStatus(artifact),
    method: evidenceMethod(artifact),
    controls: evidenceControls(artifact),
    latency: evidenceLatency(artifact),
    artifact_hash: evidenceHash(artifact),
    source_run_id: sourceRunId(artifact, fallbackRunId),
    claim_limit: claimLimit(evidenceType),
    raw: artifact,
  };
}

function promptObject(run){
  const delivered = arrayOrEmpty(run.messages);
  const assembled = Array.isArray(run.assembled_messages) ? run.assembled_messages : null;
  const selected = assembled || delivered;
  const finalPrompt = typeof run.final_prompt === "string" ? run.final_prompt : null;
  let text = finalPrompt;
  if(text === null){
    for(let index = selected.length - 1; index >= 0; index -= 1){
      const message = selected[index];
      if(isObject(message) && message.role === "user" && typeof message.content === "string"){
        text = message.content;
        break;
      }
    }
  }
  if(text === null && typeof run.prompt === "string") text = run.prompt;
  return {
    text,
    source: finalPrompt !== null ? "final_prompt"
      : assembled !== null ? "assembled_messages"
      : delivered.length ? "messages"
      : typeof run.prompt === "string" ? "prompt" : null,
    messages: selected,
    assembled_messages: assembled,
    delivered_messages: delivered,
    final_prompt: finalPrompt,
  };
}

function answerObject(run){
  const text = first(
    run.response,
    run.reply,
    typeof run.answer === "string" ? run.answer : null,
    isObject(run.answer) ? run.answer.text : null,
    null,
  );
  const trace = isObject(run.trace) || Array.isArray(run.trace) ? run.trace : null;
  return {
    text: text === null ? null : String(text),
    trace,
    steps: Array.isArray(trace) ? trace
      : trace && Array.isArray(trace.steps) ? trace.steps : arrayOrEmpty(run.steps),
    finish_reason: first(run.finish_reason, isObject(run.meta) ? run.meta.finish_reason : null, null),
  };
}

/** Normalize one full or summary run record (including legacy `{run: ...}` envelopes). */
export function normalizeRun(value){
  const envelope = isObject(value) ? value : {};
  const run = isObject(envelope.run) ? envelope.run : envelope;
  const identity = isObject(run.identity) ? run.identity
    : isObject(run.captured_identity) ? run.captured_identity : null;
  const parentId = stringOrNull(first(
    run.parent_run_id,
    run.parent_id,
    isObject(run.parent) ? run.parent.id : null,
    !isObject(run.parent) ? run.parent : null,
    isObject(run.lineage) ? first(run.lineage.parent_run_id, run.lineage.parent_id) : null,
    null,
  ));
  const parent = parentId === null ? null : {
    id: parentId,
    changes: isObject(run.changes_applied) ? run.changes_applied : null,
  };
  return {
    kind: "run",
    id: stringOrNull(first(run.id, run.run_id, null)),
    model: stringOrNull(first(run.model, run.model_name, null)),
    prompt: promptObject(run),
    answer: answerObject(run),
    identity,
    captured_identity: identity,
    parent,
    parent_id: parentId,
    raw: run,
    envelope_raw: run === envelope ? null : envelope,
  };
}

function pushUnique(bucket, type, artifact, fallbackRunId, seen){
  if(!isObject(artifact) || seen.has(artifact)) return;
  seen.add(artifact);
  bucket.push(evidenceObject(type, artifact, fallbackRunId));
}

function receiptArtifacts(container, fallbackRunId, receiptBucket, forcedBucket, seen){
  const addReceipt = artifact => {
    if(!isObject(artifact)) return;
    pushUnique(receiptBucket, "receipt", artifact, fallbackRunId, seen);
    if(isObject(artifact.forced))
      pushUnique(forcedBucket, "forced_receipt", artifact.forced, fallbackRunId, seen);
  };
  const addForced = artifact => pushUnique(
    forcedBucket, "forced_receipt", artifact, fallbackRunId, seen,
  );

  if(isObject(container.receipt)) addReceipt(container.receipt);
  if(Array.isArray(container.receipts)) container.receipts.forEach(addReceipt);
  else if(isObject(container.receipts)){
    if(isObject(container.receipts.receipt)) addReceipt(container.receipts.receipt);
    arrayOrEmpty(container.receipts.receipts).forEach(addReceipt);
    arrayOrEmpty(container.receipts.forced_receipts).forEach(addForced);
  }
  if(isObject(container.forced_receipt)) addForced(container.forced_receipt);
  arrayOrEmpty(container.forced_receipts).forEach(addForced);
}

function rootEvidenceType(value){
  if(!isObject(value)) return null;
  if(value.schema === "clozn.context_answer_influence.v1") return "influence_map";
  if((Array.isArray(value.prompt_spans) || Array.isArray(value.context_spans)) &&
      (Array.isArray(value.answer_spans) || Array.isArray(value.response_spans)))
    return "influence_map";
  if(value.schema === "clozn.calibration_profile.v1") return "truth_calibration";
  if(value.type === "workspace_readout") return "workspace_readout";
  if(value.mode === "forced" || (Array.isArray(value.deltas) && Array.isArray(value.answer_tokens)))
    return "forced_receipt";
  if(value.mode === "regen" || own(value, "influence") && (own(value, "has_effect") || own(value, "causal_verified")))
    return "receipt";
  if((Array.isArray(value.readouts) || Array.isArray(value.available_layers)) && own(value, "layer"))
    return "jlens_readout";
  if(Array.isArray(value.spans) && (isObject(value.truth) || isObject(value.proxy) || isObject(value.support)))
    return "trust";
  return null;
}

/**
 * Normalize every stored evidence artifact into a deterministic type order.
 * Unknown fields remain reachable through each object's `raw`; no absent
 * availability, control, method, latency, or hash is synthesized.
 */
export function normalizeEvidence(value, sourceRun = null){
  const inputs = Array.isArray(value) ? value : [value];
  const suppliedRunId = isObject(sourceRun) ? first(sourceRun.id, sourceRun.run_id, null) : sourceRun;
  const buckets = {
    influence_map: [], receipt: [], forced_receipt: [], workspace_readout: [],
    jlens_readout: [], truth_calibration: [], trust: [],
  };
  const seen = new Set();

  for(const input of inputs){
    if(!isObject(input)) continue;
    const container = isObject(input.run) ? input.run : input;
    const fallbackRunId = stringOrNull(first(
      suppliedRunId,
      container.id,
      container.run_id,
      input.run_id,
      null,
    ));
    const rootType = rootEvidenceType(container);
    if(rootType) pushUnique(buckets[rootType], rootType, container, fallbackRunId, seen);

    const influence = first(
      container.influence_map,
      container.context_answer_influence,
      isObject(container.evidence) ? container.evidence.influence_map : null,
      isObject(container.receipts) ? first(
        container.receipts.influence_map,
        container.receipts.context_answer_influence,
      ) : null,
      null,
    );
    pushUnique(buckets.influence_map, "influence_map", influence, fallbackRunId, seen);
    receiptArtifacts(container, fallbackRunId, buckets.receipt, buckets.forced_receipt, seen);

    const trace = isObject(container.trace) ? container.trace : {};
    const workspace = [
      ...arrayOrEmpty(trace.workspace_readouts),
      ...arrayOrEmpty(container.workspace_readouts),
    ];
    if(isObject(container.workspace_readout)) workspace.push(container.workspace_readout);
    workspace.forEach(artifact => pushUnique(
      buckets.workspace_readout, "workspace_readout", artifact, fallbackRunId, seen,
    ));

    const jlens = first(
      container.jlens_readout,
      container.jlens,
      container._jlens,
      isObject(container.readouts) ? container.readouts.jlens : null,
      null,
    );
    pushUnique(buckets.jlens_readout, "jlens_readout", jlens, fallbackRunId, seen);

    const legacyCalibration = isObject(container.calibration) && (
      container.calibration.schema === "clozn.calibration_profile.v1" ||
      container.calibration.truth === true ||
      (isObject(container.calibration.provenance) &&
        container.calibration.provenance.kind === "outcome_grounded_calibration")
    ) ? container.calibration : null;
    const truth = first(
      container.truth_calibration,
      container.calibration_profile,
      legacyCalibration,
      isObject(container.trust) ? container.trust.truth : null,
      null,
    );
    pushUnique(buckets.truth_calibration, "truth_calibration", truth, fallbackRunId, seen);

    const trust = first(
      container.trust,
      container.trust_spans,
      container.calibrated_trust,
      container.clozn_policy,
      null,
    );
    pushUnique(buckets.trust, "trust", trust, fallbackRunId, seen);
  }

  return [
    ...buckets.influence_map,
    ...buckets.receipt,
    ...buckets.forced_receipt,
    ...buckets.workspace_readout,
    ...buckets.jlens_readout,
    ...buckets.truth_calibration,
    ...buckets.trust,
  ];
}

function experimentStatus(envelope, result){
  const direct = explicitStatus(envelope);
  if(direct !== null) return direct;
  return explicitStatus(result);
}

/** Normalize the current experiment envelope and older, flatter experiment records. */
export function normalizeExperiment(value){
  const envelope = isObject(value) ? value : {};
  const change = isObject(envelope.change) ? envelope.change : {};
  const legacyResult = ["changed_reply", "delta", "has_effect", "causal_verified", "null", "receipt"]
    .some(key => own(envelope, key)) ? {
      changed_reply: own(envelope, "changed_reply") ? envelope.changed_reply : null,
      delta: own(envelope, "delta") ? envelope.delta : null,
      has_effect: own(envelope, "has_effect") ? envelope.has_effect : null,
      causal_verified: own(envelope, "causal_verified") ? envelope.causal_verified : null,
      null: own(envelope, "null") ? envelope.null : null,
      receipt: isObject(envelope.receipt) ? envelope.receipt : null,
    } : null;
  const result = own(envelope, "result") ? envelope.result
    : own(envelope, "outcome") ? envelope.outcome : legacyResult;
  const resultObject = isObject(result) ? result : null;
  const receipt = resultObject && isObject(resultObject.receipt) ? resultObject.receipt
    : isObject(envelope.receipt) ? envelope.receipt : null;
  const sourceId = stringOrNull(first(
    envelope.source_run_id,
    envelope.run_id,
    envelope.source_id,
    isObject(envelope.source) ? envelope.source.id : null,
    null,
  ));
  const explicitChild = first(
    envelope.child_run_id,
    resultObject ? resultObject.child_run_id : null,
    envelope.child_id,
    isObject(envelope.child) ? envelope.child.id : null,
    isObject(envelope.child_run) ? first(envelope.child_run.id, envelope.child_run.run_id) : null,
    null,
  );
  const receiptId = receipt ? first(receipt.id, receipt.run_id, null) : null;
  const childId = stringOrNull(first(
    explicitChild,
    receiptId !== null && String(receiptId) !== sourceId ? receiptId : null,
    null,
  ));
  return {
    kind: "experiment",
    type: stringOrNull(first(change.type, envelope.type, envelope.experiment_type, null)),
    method: first(envelope.method, resultObject ? resultObject.method : null, null),
    status: experimentStatus(envelope, resultObject),
    source_run_id: sourceId,
    child_run_id: childId,
    baseline: own(envelope, "baseline") ? envelope.baseline
      : own(envelope, "baseline_reply") ? envelope.baseline_reply : null,
    result,
    receipt,
    raw: envelope,
  };
}
