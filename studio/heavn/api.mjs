/* heavnOS api — one client for the clozn server. Field names follow the server's own route shapes.
   Every function returns null on failure rather than throwing; callers render honest absence. */

function associationId(storage, key, prefix){
  try{
    let value = storage.getItem(key);
    if(!value){
      const random = globalThis.crypto?.randomUUID?.() ||
        (Date.now().toString(36) + "-" + Math.random().toString(36).slice(2));
      value = prefix + random;
      storage.setItem(key, value);
    }
    return value;
  }catch(e){ return prefix + Date.now().toString(36) + "-" + Math.random().toString(36).slice(2); }
}
const CLIENT_ID = associationId(globalThis.localStorage, "clozn.client_id", "studio-");
const SESSION_ID = associationId(globalThis.sessionStorage, "clozn.session_id", "studio-session-");
const associationHeaders = () => ({
  "X-Clozn-Client-Id": CLIENT_ID,
  "X-Clozn-Session-Id": SESSION_ID,
});
function associated(opts){
  const o = Object.assign({}, opts || {});
  o.headers = Object.assign({}, associationHeaders(), o.headers || {});
  return o;
}

async function j(path, opts, timeoutMs = 30000){
  const c = new AbortController();
  const k = setTimeout(() => c.abort(), timeoutMs);
  try{
    const r = await fetch(path, associated(Object.assign({ signal: c.signal }, opts || {})));
    clearTimeout(k);
    if(!r.ok) return null;
    return await r.json();
  }catch(e){ clearTimeout(k); return null; }
}
const post = (path, body, t) => j(path, { method: "POST",
  headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) }, t);
/* like post(), but a non-2xx still returns the JSON body with __status attached — for routes whose
   error text matters to the UI (e.g. narrate's honest 503 "needs the qwen substrate") */
async function postE(path, body, timeoutMs = 30000){
  const c = new AbortController();
  const k = setTimeout(() => c.abort(), timeoutMs);
  try{
    const r = await fetch(path, associated({ method: "POST", signal: c.signal,
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) }));
    clearTimeout(k);
    let o = null; try{ o = await r.json(); }catch(e){ o = {}; }
    return Object.assign({ __status: r.status }, o);
  }catch(e){ clearTimeout(k); return null; }
}
const enc = encodeURIComponent;

export const api = {
  /* ── runtime status ── */
  readyz: () => j("/readyz", null, 5000),        // -> {status,model,mode,worker:{n_ctx,architecture,...}}
                                                  // or null (offline/not-ready) — the nav-footer metadata
                                                  // block (app.mjs) reads worker.n_ctx for real, live
                                                  // CONTEXT; never fabricated when this is unavailable.

  /* ── runs ── */
  listRuns: () => j("/runs", null, 8000),                              // -> {runs:[summaries]}
  latestRun: () => j("/runs/latest", null, 8000),                      // exact tab/session association
  getRun: id => j("/runs/" + enc(id), null, 8000),                     // -> full record (maybe {run:...})
  family: id => j("/runs/" + enc(id) + "/family"),                     // -> {runs:[...]}
  exportUrl: (id, fmt = "json") => "/runs/" + enc(id) + "/export?format=" + fmt,

  /* ── the verbs (computed on demand — can be slow; generous timeouts) ── */
  rederive: (id, body) => post("/runs/" + enc(id) + "/rederive", body, 120000),
  replay:   (id, body) => post("/runs/" + enc(id) + "/replay", body, 180000),
  branch:   (id, body) => post("/runs/" + enc(id) + "/branch", body, 180000),
  receipt:  (id, body) => post("/runs/" + enc(id) + "/receipt", body, 300000),
  receipts: (id, body) => post("/runs/" + enc(id) + "/receipts", body, 600000), // prove-all
  swapReceipt: (id, body) => post("/runs/" + enc(id) + "/swap_receipt", body, 300000),
  counterfactual: (id, overrides) => post("/runs/" + enc(id) + "/counterfactual",
                                          { behavior_overrides: overrides }, 300000),
  feedbackRecord: body => post("/feedback", body, 30000),              // records preference only; mutates no dial
  explain:  id => post("/runs/" + enc(id) + "/explain", {}, 30000),   // POST-only (contracts §11)
  narrate:  id => postE("/runs/" + enc(id) + "/narrate", {}, 300000), // __status rides (503 = needs qwen)
  proposeMemory: id => post("/runs/" + enc(id) + "/propose-memory", {}, 60000),

  /* ── inspector features ── */
  trustSpans: (id, support = false) => post("/runs/" + enc(id) + "/trust_spans",
    support ? { support: true } : {}, support ? 300000 : 60000),               // F2: proxy+truth; NLI explicit
  journalActuary: () => j("/journal/actuary", null, 60000),                    // acceptance-proxy report
  journalCalibration: () => j("/journal/calibration", null, 15000),            // truth-tier curve (clozn eval)
  runActuary: id => postE("/runs/" + enc(id) + "/actuary", {}, 60000),        // past-only failure resemblance
  fork: (id, position, token) => post("/runs/" + enc(id) + "/fork",
                                      { position, token }, 300000),            // F3: child run back
  spanReceipt: (id, find) => postE("/runs/" + enc(id) + "/span_receipt",
                                   { find }, 600000),                          // F4: ablate a phrase
  influenceMap: (id, body = {}) => postE("/runs/" + enc(id) + "/influence-map",
                                          body, 600000),                       // context <-> answer forced map
  diffRuns: (a, b) => post("/diff/runs", { a, b }, 60000),                     // F8
  anchoredList: () => j("/memory/anchored/list", null, 15000),                 // F6
  anchoredFit: (card_id, k) => post("/memory/anchored/fit",
                                    { card_id, ...(k ? { k } : {}) }, 180000),
  anchoredToggle: (card_id, on) => post("/memory/anchored/toggle", { card_id, on }),
  anchoredDeleteTerm: (card_id, token) => post("/memory/anchored/delete_term",
                                               { card_id, token }, 180000),
  whatlearned: () => j("/memory/anchored/whatlearned", null, 15000),
  cardUrl: id => "/runs/" + enc(id) + "/card",                                 // F9

  /* ── experiments (2026-07-13): the ONE experiment primitive -- "hold everything constant, change
        one thing, compare, with a receipt" -- dispatches replay/counterfactual/receipt/branch/
        swap_receipt behind one endpoint + one normalized envelope (clozn/experiments/experiment.py).
        runExperiment uses postE (not post): a 400 (bad change spec), 404 (no run), or 503 (missing
        substrate) all carry a real server-written reason the drawer should show verbatim, exactly
        like narrate's own 503 handling -- a generic "didn't answer" toast would bury that. ── */
  experimentTypes: () => j("/experiments/types", null, 15000),               // -> {types: {<type>: preflight method/control/cost}}
  runExperiment: (id, change, method) => postE("/runs/" + enc(id) + "/experiment",
    { change, ...(method != null ? { method } : {}) }, 300000),

  /* ── readouts (POST-only — contracts §9: params ride the JSON body, never the query string) ── */
  jlens: (id, layer, topk) => post("/runs/" + enc(id) + "/jlens",
    { ...(layer != null ? { layer } : {}), ...(topk != null ? { topk } : {}) }, 60000),
  jlensText: (text, layer) => post("/jlens", { text, ...(layer != null ? { layer } : {}) }, 60000),
  engineLayers: text => post("/engine/layers", { text }, 60000),
  engineHarvest: text => postE("/engine/harvest", { text }, 60000),
  engineObserve: (text, position, scale) => postE("/engine/observe", { text, position, scale }, 120000),

  /* ── memory (contracts §14: list is POST /memory/cards; actions carry {id} in the BODY) ── */
  memoryList: () => post("/memory/cards", {}),
  memoryMode: () => j("/memory/mode", null, 10000),
  memorySetMode: mode => postE("/memory/mode", { mode }, 30000),
  memoryStrength: value => postE("/memory/strength", value == null ? {} : { value }, 30000),
  memoryAdd: body => post("/memory/add", body),
  memoryAct: (id, verb) => post("/memory/" + verb, { id }),   // approve|reject|disable|enable|remove
  memoryRuns: id => j("/memory/" + enc(id) + "/runs"),        // GET (contracts §14)

  /* facts: optional per-profile cue→answer slot store. All operations use postE so the panel can
     distinguish an unavailable backend from the store's deliberate write refusal / read abstention. */
  factsList:   () => postE("/facts/list", {}, 30000),
  factsMode:   enabled => postE("/facts/mode", { enabled }, 30000),
  factsAdd:    (cue, answer) => postE("/facts/add", { cue, answer }, 120000),
  factsDelete: cue => postE("/facts/delete", { cue }, 30000),
  factsRead:   query => postE("/facts/read", { query }, 120000),

  /* ── steering (contracts §17: axes is POST-only; /steer/set REQUIRES name even if empty) ── */
  /* profiles: portable card/dial/fact source bundles. Mutations use postE so Settings can show the
     server's exact validation, switch, or guarded-delete reason instead of a generic null. */
  profilesList:   () => j("/profiles/list", null, 10000),
  profilesSave:   profile => postE("/profiles/save", profile, 30000),
  profilesSwitch: name => postE("/profiles/switch", { name }, 180000),
  profilesExport: name => postE("/profiles/export", { name }, 30000),
  profilesImport: (profile, rename = null) => postE("/profiles/import",
    { profile, ...(rename ? { rename } : {}) }, 30000),
  profilesDelete: name => postE("/profiles/delete", { name }, 30000),

  steerAxes: () => post("/steer/axes", {}),
  steerSet: (name, value) => post("/steer/set", { name: name ?? "", value }),
  /* Compiles a pole pair over the substrate's shared seed prompts. This is model work, so keep the
     same generous ceiling as other on-demand computations and preserve the server's validation text. */
  steerCustom: (name, pos, neg) => postE("/steer/custom", { name, pos, neg }, 300000),
  steerCustomDelete: name => postE("/steer/custom_delete", { name }, 30000),
  preferences: (threshold = 3) => postE("/preferences", { threshold }, 30000),
  preferenceResolve: (id, action) => postE("/preferences/resolve", { id, action }, 30000),

  /* ── chat (SSE stream). onDelta(textChunk), onDone(finalInfo|null), onLens(readout) — the F1 live
        lens frames when `lens` ({layer, topk?, every?}) is passed. onPolicy(policy) — the clozn_policy
        side-frame (calibrated ask-band verdict), sent once before the finish chunk when the reply
        lands in the "ask" band; absent otherwise, same as the non-streaming response field. Returns
        abort fn. ── */
  chatStream(messages, { onDelta, onDone, onError, onLens, onPolicy, lens = null, trust = false } = {}){
    const c = new AbortController();
    (async () => {
      try{
        const r = await fetch("/v1/chat/completions", {
          method: "POST", signal: c.signal,
          headers: Object.assign({ "Content-Type": "application/json" }, associationHeaders()),
          body: JSON.stringify({ messages, stream: true,
                                 ...(lens ? { clozn_lens: lens } : {}),
                                 ...(trust ? { clozn_trust: true } : {}) }),
        });
        if(!r.ok || !r.body){ onError && onError("stream unavailable (" + (r ? r.status : "no response") + ")"); return; }
        const reader = r.body.getReader(), dec = new TextDecoder();
        let buf = "", final = null;
        for(;;){
          const { done, value } = await reader.read();
          if(done) break;
          buf += dec.decode(value, { stream: true });
          const lines = buf.split("\n"); buf = lines.pop();
          for(const line of lines){
            const t = line.trim();
            if(!t.startsWith("data:")) continue;
            const payload = t.slice(5).trim();
            if(payload === "[DONE]") continue;
            try{
              const obj = JSON.parse(payload);
              if(obj.error){ onError && onError(String(obj.error)); continue; }   /* mid-stream error frame */
              if(obj.clozn_lens){ onLens && onLens(obj.clozn_lens); continue; }   /* F1 live lens side-frame */
              if(obj.clozn_policy){ onPolicy && onPolicy(obj.clozn_policy); continue; }  /* calibrated ask-band side-frame */
              const delta = obj.choices && obj.choices[0] && (obj.choices[0].delta?.content ?? obj.choices[0].text);
              if(delta) onDelta && onDelta(delta, obj);
              if(obj.choices && obj.choices[0] && obj.choices[0].finish_reason) final = obj;
            }catch(e){ /* non-JSON frame — skip */ }
          }
        }
        onDone && onDone(final);
      }catch(e){
        if(e.name !== "AbortError") onError && onError(String(e));
      }
    })();
    return () => c.abort();
  },
};
