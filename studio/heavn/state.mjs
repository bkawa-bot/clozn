/* heavnOS state — one tiny observable store + a Preact hook. No deps beyond preact-standalone. */
import { useState, useEffect } from "./vendor/preact-standalone.mjs";

export function createStore(initial){
  let s = initial;
  const subs = new Set();
  return {
    get: () => s,
    set(patch){
      s = { ...s, ...(typeof patch === "function" ? patch(s) : patch) };
      subs.forEach(f => f(s));
    },
    subscribe(f){ subs.add(f); return () => subs.delete(f); },
  };
}

/* the app store */
export const store = createStore({
  live: false,            // true when GET /runs succeeded
  worker: null,           // /readyz's raw worker-health dict (n_ctx, architecture, ...) or null when
                           // unreachable/offline — the nav-footer metadata block's only source for
                           // fields it can't get off the loaded run record; never backfilled with
                           // placeholder text (see app.mjs's NavFoot).
  runs: [],               // summaries (newest first)
  full: {},               // id -> full record cache
  currentId: null,
  rec: null,              // the current full record
  route: "replay",        // active module
  readRequest: null,       // /r/<id> permalink target; kept visible when the record is missing
  readError: null,         // honest deep-link load failure (never substitute a different run)
  toast: null,            // { msg, t }
  P: 0,                   // playhead (token index, 0..n)
  playing: false,
  speed: 1,
  busy: {},               // verb -> bool (rederive / prove / replay / branch)
  receipts: null,         // last prove-all result for current run
  leaning: null,          // per-token leaning heat for current run (from receipts)
  trust: {},              // F2: run id -> journal-calibrated trust spans (null = unavailable)
  lensLayer: 0,           // F1: 0 = live lens off; else the requested J-lens depth (2/14/21/25)
  liveLens: null,         // F1: the latest mid-stream disposed-to-say readout
  pendingExperiment: null, // click-a-span handoff: {ctype, fields, method} the Experiment drawer
                           // consumes ONCE on mount then clears (a span's action deep-links here)
});

/* subscribe a component to a slice of the store */
export function useStore(selector){
  const sel = selector || (s => s);
  const [v, setV] = useState(() => sel(store.get()));
  useEffect(() => store.subscribe(s => {
    const nv = sel(s);
    setV(prev => (Object.is(prev, nv) ? prev : nv));
  }), []);
  return v;
}

let toastTimer = null;
export function toast(msg){
  store.set({ toast: { msg, t: Date.now() } });
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => store.set({ toast: null }), 2800);
}

/* the SAMPLE reel — used ONLY when the server is unreachable; always bannered as sample */
export const SAMPLE = {
  id: "run_sample", created_at: "2026-07-09 14:12", model: "Qwen2.5-7B-Instruct", substrate: "engine",
  finish_reason: "stop", flags: ["memory","steered"],
  prompt_summary: "What country is shaped like a boot?",
  response: "The country shaped like a boot is Italy. Its distinctive shape extends into the Mediterranean Sea.",
  memory: { mode: "prompt", gate: 0.62,
    cards_applied: ["enjoys geography trivia","prefers concise factual answers"], relevance: [0.66,0.41] },
  behavior: { active_dials: { concise: 0.4, warm: 0.2 } },
  meta: { decode: { mode: "greedy", temperature: 0, seed: 0, quant: "Q4_K_M", engine: "cloze-server" } },
  assembled_messages: [
    { role: "system", content: "[memory block · gate .62] The user enjoys geography trivia. The user prefers concise factual answers." },
    { role: "user", content: "What country is shaped like a boot?" }],
  trace: {
    tokens: ["The"," country"," shaped"," like"," a"," boot"," is"," Italy","."," Its"," distinctive"," shape"," extends"," into"," the"," Mediterranean"," Sea","."],
    confidence: [.97,.93,.95,.98,.99,.96,.98,.91,.99,.62,.44,.83,.38,.74,.92,.88,.95,.99],
    entropy: [.10,.22,.16,.06,.03,.13,.06,.29,.03,1.22,1.79,.54,1.98,.83,.26,.38,.16,.03],
    alternatives: [[],[],[],[],[],[],[], [{piece:" the",prob:.04},{piece:" Rome",prob:.02}], [],[],[],[],[],[],[],[],[],[]] },
  _sample: true,
  _jlens: { layer: 25, layers: [2,14,21,25],
    provenance: "J-lens readout (Anthropic, 2026) — what each position is disposed to say later. Fitted on Qwen2.5-7B nf4, 100-prompt corpus, 2026-07-09; engine-transfer hit@5 0.72. A disposition, not a verified thought; blank ≠ nothing.",
    chips: { 6: [" Italy"," Sicily"," boot"], 10: [" peninsula"," coast"," Rome"] } },
};

/* trace normalization — accepts parallel arrays OR step lists */
const num = v => { const n = parseFloat(v); return Number.isFinite(n) ? n : null; };
export function normSteps(rec){
  const tr = rec && rec.trace || {};
  if(Array.isArray(tr.steps) && tr.steps.length)
    return tr.steps.map(s => ({ piece: s.piece ?? s.text ?? "", conf: num(s.conf ?? s.confidence),
                                ent: num(s.entropy), alts: s.alts || s.alternatives || [] }));
  return (tr.tokens || []).map((p,i) => ({ piece: p, conf: num((tr.confidence||[])[i]),
    ent: num((tr.entropy||[])[i]), alts: ((tr.alternatives||[])[i]) || [] }));
}
export const weightsFor = steps => steps.map(s => Math.max(2, Math.min(16, (s.piece||"").length)));
export const colsFor = w => w.map(x => x + "fr").join(" ");
export function colGeom(w, W){
  const total = w.reduce((a,b) => a+b, 0); let acc = 0;
  return w.map(x => { const g = { x0: acc/total*W, x1: (acc+x)/total*W, xc: (acc + x/2)/total*W }; acc += x; return g; });
}
export function firstLine(rec){
  const m = rec.messages || rec.assembled_messages || [];
  const u = [...m].reverse().find(x => x.role === "user");
  return (u && u.content) ? String(u.content).slice(0,140) : (rec.prompt_summary || rec.id || "run");
}
export const shortTime = t => String(t || "").replace("T"," ").slice(5,16);
export const REDUCED = typeof matchMedia !== "undefined"
  ? matchMedia("(prefers-reduced-motion: reduce)").matches : false;
