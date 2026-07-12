/* heavnOS api — one client for the clozn server. Field names follow the server's own shapes;
   see notes/HEAVN_API_CONTRACTS.md (generated from the route source) — reconcile here if they drift.
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
  narrate:  id => post("/runs/" + enc(id) + "/narrate", {}, 180000),
  proposeMemory: id => post("/runs/" + enc(id) + "/propose-memory", {}, 60000),

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

  /* ── chat (SSE stream). onDelta(textChunk), onDone(finalInfo|null). Returns abort fn. ── */
  chatStream(messages, { onDelta, onDone, onError, trust = false } = {}){
    const c = new AbortController();
    (async () => {
      try{
        const r = await fetch("/v1/chat/completions", {
          method: "POST", signal: c.signal,
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages, stream: true, ...(trust ? { clozn_trust: true } : {}) }),
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
