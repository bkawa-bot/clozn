/* heavnOS api — one client for the clozn server. Field names follow the server's own route shapes.
   Every function returns null on failure rather than throwing; callers render honest absence. */

async function j(path, opts, timeoutMs = 30000){
  const c = new AbortController();
  const k = setTimeout(() => c.abort(), timeoutMs);
  try{
    const r = await fetch(path, Object.assign({ signal: c.signal }, opts || {}));
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
    const r = await fetch(path, { method: "POST", signal: c.signal,
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
    clearTimeout(k);
    let o = null; try{ o = await r.json(); }catch(e){ o = {}; }
    return Object.assign({ __status: r.status }, o);
  }catch(e){ clearTimeout(k); return null; }
}
const enc = encodeURIComponent;

export const api = {
  /* ── runs ── */
  listRuns: () => j("/runs", null, 8000),                              // -> {runs:[summaries]}
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
  explain:  id => post("/runs/" + enc(id) + "/explain", {}, 30000),   // POST-only (contracts §11)
  narrate:  id => postE("/runs/" + enc(id) + "/narrate", {}, 300000), // __status rides (503 = needs qwen)
  proposeMemory: id => post("/runs/" + enc(id) + "/propose-memory", {}, 60000),

  /* ── inspector features ── */
  trustSpans: id => post("/runs/" + enc(id) + "/trust_spans", {}, 60000),      // F2: journal-calibrated
  fork: (id, position, token) => post("/runs/" + enc(id) + "/fork",
                                      { position, token }, 300000),            // F3: child run back
  spanReceipt: (id, find) => postE("/runs/" + enc(id) + "/span_receipt",
                                   { find }, 600000),                          // F4: ablate a phrase
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
  experimentTypes: () => j("/experiments/types", null, 15000),               // -> {types: {<type>: {label, needs, cost_hint}}}
  runExperiment: (id, change, method) => postE("/runs/" + enc(id) + "/experiment",
    { change, ...(method != null ? { method } : {}) }, 300000),

  /* ── readouts (POST-only — contracts §9: params ride the JSON body, never the query string) ── */
  jlens: (id, layer, topk) => post("/runs/" + enc(id) + "/jlens",
    { ...(layer != null ? { layer } : {}), ...(topk != null ? { topk } : {}) }, 60000),
  jlensText: (text, layer) => post("/jlens", { text, ...(layer != null ? { layer } : {}) }, 60000),
  engineLayers: text => post("/engine/layers", { text }, 60000),

  /* ── memory (contracts §14: list is POST /memory/cards; actions carry {id} in the BODY) ── */
  memoryList: () => post("/memory/cards", {}),
  memoryAdd: body => post("/memory/add", body),
  memoryAct: (id, verb) => post("/memory/" + verb, { id }),   // approve|reject|disable|enable|remove
  memoryRuns: id => j("/memory/" + enc(id) + "/runs"),        // GET (contracts §14)

  /* ── steering (contracts §17: axes is POST-only; /steer/set REQUIRES name even if empty) ── */
  steerAxes: () => post("/steer/axes", {}),
  steerSet: (name, value) => post("/steer/set", { name: name ?? "", value }),

  /* ── chat (SSE stream). onDelta(textChunk), onDone(finalInfo|null), onLens(readout) — the F1 live
        lens frames when `lens` ({layer, topk?, every?}) is passed. Returns abort fn. ── */
  chatStream(messages, { onDelta, onDone, onError, onLens, lens = null, trust = false } = {}){
    const c = new AbortController();
    (async () => {
      try{
        const r = await fetch("/v1/chat/completions", {
          method: "POST", signal: c.signal,
          headers: { "Content-Type": "application/json" },
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
