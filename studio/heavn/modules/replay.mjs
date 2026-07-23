/* heavnOS · Replay desk — the ζ design, componentized, with the transport verbs LIVE (W1).
   Honesty rules carried over: CAPTURED/DERIVED/SAMPLE tags, provenance verbatim, no mock data
   on live runs, computed-on-demand receipts never implied to pre-exist. */
import { html, useState, useEffect, useRef } from "../vendor/preact-standalone.mjs";
import { store, useStore, toast, normSteps, weightsFor, colsFor, colGeom,
         firstLine, shortTime, REDUCED, withFull } from "../state.mjs";
import { api } from "../api.mjs";
import { loadRun } from "../app.mjs";
import { PolicyChip } from "../policy.mjs";
import { ProvenanceChip } from "../provenance.mjs";
import { normalizeRun } from "../object_model.mjs";

/* ───────────────────────── module root ───────────────────────── */
export function ReplayModule(){
  const rec = useStore(x => x.rec);
  const live = useStore(x => x.live);
  usePlayEngine();
  if(!rec) return html`<div class="replay-workbench replay-empty">
    <div class="workbench-titlebar">
      <div><span class="workbench-kicker">run workbench</span><h1>Observe a local run as it lands.</h1>
        <p>The model-state scope, replay tape, and intervention tools will bind to the next journal record.</p></div>
      <span class=${"workbench-live " + (live ? "on" : "off")}>${live ? "READY" : "OFFLINE"}</span>
    </div>
    <${Monitor} rec=${null}/>
  </div>`;
  const run = normalizeRun(rec);
  return html`<div class="replay-workbench">
    <${WorkbenchHeader} rec=${rec} run=${run}/>
    <${ContextWarning} rec=${rec}/>
    <${SignalRibbon} rec=${rec}/>

    <div class="workbench-stage">
      <${ScopeMod} rec=${rec}/>
      <aside class="workbench-live-rail">
        <${SignalReadout} rec=${rec}/>
        <${Meters} rec=${rec}/>
        <${QuickRepair} rec=${rec}/>
      </aside>
    </div>

    <${TapeMod} rec=${rec}/>
    <div class="workbench-output"><${Monitor} rec=${rec}/><${Logs} rec=${rec}/></div>

    <div class="workbench-analysis-head">
      <div><span class="workbench-kicker">analysis rack</span><h2>Inspect, prove, and intervene</h2></div>
      <p>Cheap recorded signals stay visible. Generative and causal work remains explicit and on demand.</p>
    </div>
    <div class="workbench-analysis">
      <div class="col">
        <${ExplainPanel} rec=${rec}/>
        <${InfluenceMap} rec=${rec}/>
        <${ReceiptsPanel} rec=${rec}/>
        <${SpanForensics} rec=${rec}/>
        <${LieDetector} rec=${rec}/>
      </div>
      <div class="col">
        <${Steer} rec=${rec}/>
        <${Minfl} rec=${rec}/>
      </div>
    </div>
  </div>`;
}

function ContextWarning({ rec }){
  const warnings = (rec.context_receipt && rec.context_receipt.warnings) || rec.warnings || [];
  const cutoff = warnings.find(w => w && w.code === "output_truncated")
    || (rec.finish_reason === "length" ? { message: "generation stopped at the output/context token budget; the reply may be incomplete" } : null);
  if(!cutoff) return null;
  const lim = (rec.context_receipt && rec.context_receipt.limits) || {};
  return html`<section class="context-warning" role="alert">
    <b>OUTPUT CUT OFF</b>
    <span>${cutoff.message}</span>
    ${lim.requested_max_tokens != null && html`<code>requested ${lim.requested_max_tokens} tokens</code>`}
    <span>Prompt input was not silently truncated; overlong prompts are rejected by the worker.</span>
  </section>`;
}

/* The first screen is a run instrument, not an output-first chat shell. Every field here comes from
   either the selected journal record or /readyz; an absent worker field stays visibly absent. */
function WorkbenchHeader({ rec, run }){
  const s = useStore(x => ({ worker: x.worker, P: x.P, playing: x.playing }));
  const steps = normSteps(rec), d = (rec.meta && rec.meta.decode) || {};
  const worker = s.worker || {};
  const runtime = worker.device || worker.architecture || rec.substrate || null;
  const stage = s.playing ? "PLAYING" : s.P >= steps.length && steps.length ? "AT END" : s.P ? "PAUSED" : "READY";
  const fact = (label, value) => html`<div class="workbench-fact"><span>${label}</span><b>${value == null || value === "" ? "—" : value}</b></div>`;
  return html`<header class="workbench-titlebar">
    <div class="workbench-titlecopy">
      <span class="workbench-kicker">run workbench · recorded model state</span>
      <h1>${firstLine(rec)}</h1>
      <p>${run.id || "—"} · ${shortTime(rec.created_at) || "time unavailable"}</p>
    </div>
    <div class="workbench-facts">
      ${fact("model", run.model)}${fact("runtime", runtime)}${fact("decode", d.mode)}
      ${fact("finish", run.answer.finish_reason)}${fact("tokens", steps.length)}
    </div>
    <div class="workbench-status"><span class=${"workbench-live " + (rec._sample ? "sample" : "on")}>${rec._sample ? "SAMPLE" : stage}</span>
      <span>${s.P} / ${steps.length}</span></div>
  </header>`;
}

function signalBuckets(steps, limit = 160){
  if(!steps.length) return [];
  const size = Math.max(1, Math.ceil(steps.length / limit)), out = [];
  for(let start = 0; start < steps.length; start += size){
    const slice = steps.slice(start, Math.min(steps.length, start + size));
    const conf = slice.map(x => x.conf).filter(x => x != null);
    const ent = slice.map(x => x.ent).filter(x => x != null);
    out.push({ start, end: start + slice.length,
      conf: conf.length ? conf.reduce((a,b) => a+b, 0) / conf.length : null,
      ent: ent.length ? Math.max(...ent) : null,
      piece: slice.map(x => x.piece || "").join("").trim() || "·" });
  }
  return out;
}

/* A compact tape is always above the fold. It is linked to the full arrangement and monitor through
   the existing global playhead P, so a click here updates every readout without inventing a signal. */
function SignalRibbon({ rec }){
  const P = useStore(x => x.P);
  const playing = useStore(x => x.playing);
  const steps = normSteps(rec), buckets = signalBuckets(steps);
  const conf = steps.map(x => x.conf).filter(x => x != null);
  const shaky = steps.reduce((n,x) => n + (x.conf != null && x.conf < .55 ? 1 : 0), 0);
  const mean = conf.length ? conf.reduce((a,b) => a+b, 0) / conf.length : null;
  const current = P ? steps[Math.min(P - 1, steps.length - 1)] : null;
  return html`<section class="signal-ribbon mod" aria-label="linked replay tape">
    <div class="signal-ribbon-head">
      <div><span class="workbench-kicker">linked replay tape</span>
        <b>${current ? `#${P} · ${(current.piece || "").trim() || "·"}` : "playhead at start"}</b></div>
      <div class="signal-ribbon-stats"><span>mean confidence <b>${mean == null ? "—" : mean.toFixed(2)}</b></span>
        <span>low-confidence tokens <b>${shaky}</b></span><span class="tag cap-t">CAPTURED</span></div>
    </div>
    <div class="signal-ribbon-row">
      <button class="ribbon-play" aria-label=${playing ? "Pause replay" : "Play replay"} onClick=${() => {
        if(playing) store.set({ playing: false });
        else store.set({ P: P >= steps.length ? 0 : P, playing: true });
      }}>${playing ? "❚❚" : "▶"}</button>
      <div class="signal-ribbon-track" style=${"grid-template-columns:repeat(" + Math.max(1,buckets.length) + ",minmax(2px,1fr))"}>
        ${buckets.map((b,i) => html`<button key=${b.start}
          class=${"signal-bar" + (P > b.start && P <= b.end ? " on" : "") + (b.conf != null && b.conf < .55 ? " weak" : "")}
          style=${"--signal:" + (b.conf == null ? .18 : b.conf).toFixed(3) + ";--entropy:" + Math.min(1, (b.ent || 0) / 4).toFixed(3)}
          title=${`tokens ${b.start + 1}–${b.end} · ${b.piece.slice(0, 42)} · confidence ${b.conf == null ? "unavailable" : b.conf.toFixed(3)}`}
          aria-label=${`Jump to token ${b.end}`}
          onClick=${() => store.set({ P: b.end, playing: false })}></button>`)}
      </div>
    </div>
    <div class="signal-ribbon-scale"><span>START</span><span>click any signal block to inspect that position</span><span>END · ${steps.length}</span></div>
  </section>`;
}

function SignalReadout({ rec }){
  const P = useStore(x => x.P);
  const steps = normSteps(rec), i = steps.length ? Math.max(0, Math.min(steps.length - 1, P ? P - 1 : 0)) : -1;
  const step = i >= 0 ? steps[i] : null;
  const low = steps.reduce((best, x, j) => x.conf != null && (best < 0 || x.conf < steps[best].conf) ? j : best, -1);
  const alts = step ? (step.alts || []).slice(0, 3) : [];
  return html`<div class="mod signal-readout">
    <div class="mod-h"><span class="led coral"></span><span class="cap">token position</span>
      <span class="tail">${step ? `${i + 1} / ${steps.length}` : "—"}</span></div>
    <div class="signal-readout-token">${step ? ((step.piece || "").trim() || "·") : "no token trace"}</div>
    <div class="signal-readout-grid">
      <div><span>confidence</span><b>${step && step.conf != null ? step.conf.toFixed(3) : "—"}</b></div>
      <div><span>entropy</span><b>${step && step.ent != null ? step.ent.toFixed(3) : "—"}</b></div>
    </div>
    <div class="signal-alts"><span>recorded alternatives</span>
      ${alts.length ? alts.map(a => html`<b>${String(a.piece || "·").trim() || "·"}<i>${a.prob != null ? (+a.prob).toFixed(3) : "—"}</i></b>`)
        : html`<em>none captured at this position</em>`}</div>
    <div class="signal-readout-foot"><span>confidence is token probability, not correctness</span>
      <button class="spd" disabled=${low < 0} onClick=${() => low >= 0 && store.set({ P: low + 1, playing: false })}>JUMP TO LOWEST</button></div>
  </div>`;
}

/* the play engine: one interval, driven by store.playing/speed */
function usePlayEngine(){
  const playing = useStore(x => x.playing);
  const speed = useStore(x => x.speed);
  useEffect(() => {
    if(!playing) return;
    const iv = setInterval(() => {
      const s = store.get(), n = normSteps(s.rec || {}).length;
      if(s.P >= n){ store.set({ playing: false }); return; }
      store.set({ P: s.P + 1 });
    }, 220 / speed);
    return () => clearInterval(iv);
  }, [playing, speed]);
}

/* ───────────────────────── CRT monitor — replay device AND live terminal (W2) ───────────── */
function Monitor({ rec }){
  const P = useStore(x => x.P);
  const playing = useStore(x => x.playing);
  const chatting = useStore(x => x.chatting);
  const chatBuf = useStore(x => x.chatBuf);
  const chatPrompt = useStore(x => x.chatPrompt);
  const liveLens = useStore(x => x.liveLens);
  const livePolicy = useStore(x => x.livePolicy);
  const trust = useStore(x => rec ? x.trust[rec.id] : null);
  const [supportBusy, setSupportBusy] = useState(false);
  const steps = rec ? normSteps(rec) : [];
  const reasoningBlocks = rec && rec.reasoning && Array.isArray(rec.reasoning.blocks)
    ? rec.reasoning.blocks : [];
  const done = P >= steps.length;
  const trunc = rec && rec.finish_reason === "length";
  const liveView = chatting || chatBuf != null;   /* streaming, or streamed & awaiting the journal */
  const policy = liveView ? livePolicy : (rec && rec.clozn_policy) || null;
  const stLabel = chatting ? "LIVE" : chatBuf != null ? "SAVING"
    : !rec ? "IDLE" : playing ? "PLAY" : done ? "END" : P === 0 ? "IDLE" : "PAUSE";

  /* F2 trust shading: fetch the journal-calibrated spans once per live run (pure journal math) */
  useEffect(() => {
    if(!rec || rec._sample || !store.get().live) return;
    if(store.get().trust[rec.id] !== undefined) return;
    let dead = false;
    (async () => {
      const r = await api.trustSpans(rec.id);
      if(dead) return;
      store.set(st => ({ trust: { ...st.trust, [rec.id]: r || null } }));
    })();
    return () => { dead = true; };
  }, [rec && rec.id]);
  /* token index -> its trust span (start/end are TOKEN indices per contracts) */
  const spanOf = i => trust && (trust.spans || []).find(sp => i >= sp.start && i < sp.end);
  const shadedUpto = () => steps.slice(0, P).map((s, i) => {
    const sp = spanOf(i);
    if(!sp) return html`<span key=${i}>${s.piece}</span>`;
    const truthReady = sp.truth_correctness_estimate != null && trust.truth && !trust.truth.small_n;
    const proxyReady = sp.trusted_rate_estimate != null;
    const tr = truthReady ? sp.truth_correctness_estimate : proxyReady ? sp.trusted_rate_estimate : null;
    const support = sp.support || {};
    if(tr == null && !support.available) return html`<span key=${i}>${s.piece}</span>`;
    const amber = tr != null && tr < .55;
    const faint = tr == null ? 1 : .55 + .45 * Math.min(1, Math.max(0, tr));
    const evidence = truthReady
      ? `temperature-scaled correctness estimate ${Math.round(tr*100)}% from ${trust.truth.n} labeled probes (${trust.truth.score_aggregate}-confidence); estimate on that eval distribution, not a fact-check`
      : proxyReady
      ? `kept ${Math.round(tr*100)}% of the time at this confidence in your journal (${sp.bin_n} runs${sp.small_n ? ", small bin — weak evidence" : ""}); acceptance proxy, not a fact-check`
      : "no confidence calibration available";
    const supportPremise = trust.support && trust.support.evidence_tier === "causal_receipts"
      ? "stored causal receipt" : "active recorded influence";
    const supportTitle = support.available
      ? ` · support: ${supportPremise} ${support.entailed ? "entailed" : "did not entail"} this span (NLI ${support.score ?? "—"}); not external evidence`
      : (trust.support && trust.support.requested ? " · support check unavailable" : "");
    return html`<span key=${i}
      style=${"opacity:" + faint.toFixed(2)
        + (amber ? ";border-bottom:1px dotted rgba(255,196,120,.75)" : "")
        + (support.available ? ";box-shadow:inset 0 -2px " + (support.entailed ? "rgba(95,200,188,.9)" : "rgba(174,139,191,.85)") : "")
        + ((amber || support.available) ? ";cursor:pointer" : "")}
      title=${evidence + supportTitle}
      onClick=${(amber || support.available) ? (() => store.set({ verbResult: { verb: "trust", ok: true,
        msg: `“${(sp.text || "").slice(0, 48)}” — ${evidence}${supportTitle}.` } })) : null}>${s.piece}</span>`;
  });

  const checkSupport = async () => {
    if(!rec || rec._sample || !store.get().live || supportBusy) return;
    setSupportBusy(true);
    const r = await api.trustSpans(rec.id, true);
    setSupportBusy(false);
    if(!r){ toast("support check did not answer"); return; }
    store.set(st => ({ trust: { ...st.trust, [rec.id]: r } }));
  };
  const proxyMeta = trust && trust.proxy;
  const truthMeta = trust && trust.truth;
  const supportMeta = trust && trust.support;

  return html`<div class="mod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led"></span><span class="cap">output monitor</span>
      <span class="tail">${liveView ? "streaming" : rec ? "tokens 0–" + steps.length : "—"}</span>
      ${policy && html`<${PolicyChip} policy=${policy}/>`}
      ${!liveView && html`<${ProvenanceChip} rec=${rec}/>`}
      <span class=${"tag " + (liveView ? "der-t" : "cap-t")}>${liveView ? "LIVE" : "CAPTURED"}</span></div>
    <div class="crt-shell"><div class="crt">
      <div class="scan"></div>
      <div class=${"st" + (chatting || playing ? "" : " idle")}><i></i><span>${stLabel}</span></div>
      <div class="sid">${liveView ? "streaming…" : rec ? (rec.id || "—") + " · " + shortTime(rec.created_at) : "no run"}</div>
      <div class="crt-text">
        ${liveView
          ? html`${chatPrompt && html`<span class="dim">⟨you⟩ ${chatPrompt}\n</span>`}${chatBuf || ""}${html`<span class="cursor">▋</span>`}`
          : !rec ? html`<span class="dim">no tape on the deck — say something below…</span>`
          : P === 0 ? html`<span class="dim">awaiting playback…</span>`
          : html`${trust ? shadedUpto() : steps.slice(0, P).map(s => s.piece).join("")}${(done && !playing) ? "" : html`<span class="cursor">▋</span>`}`}
      </div>
    </div></div>
    ${liveView && liveLens && html`<div class="cfg" style="margin:6px 13px 0;border-left-color:var(--lilac)">
      ${liveLens.error
        ? html`<span class="cap">live lens</span><span>${liveLens.error}</span>`
        : html`<span class="cap">disposed L${liveLens.layer}</span>
          ${(liveLens.words || []).map((w,k) => html`<span class=${"jchip d" + Math.min(3, k+1)}
            style="font-size:9.5px">${w.piece || "·"}</span>`)}
          <span style="margin-left:auto;font-size:8px;color:var(--mist)">J-lens, mid-generation · a
            disposition, not a verified thought · content-only</span>`}
    </div>`}
    <${ChatBar} rec=${rec}/>
    <div class="crt-foot">
      ${liveView ? html`<span>the reply is landing in the journal when the stream ends — no run id rides the stream, by design</span>`
        : rec ? html`
          <span>finish <b>${rec.finish_reason || "?"}</b></span>
          <span>tokens <b>${steps.length}</b></span>
          ${rec._sample ? html`<span style="color:var(--mist)">sample reel</span>`
                        : html`<span>recorded on this machine</span>`}
          ${trunc && html`<span class="tag fail-t">TRUNCATED</span>`}
          ${trust && html`<span style="color:var(--mist)">shading source is disclosed below</span>`}
        ` : html`<span>nothing recorded yet</span>`}
    </div>
    ${reasoningBlocks.length ? html`<details class="cfg" data-testid="captured-reasoning"
        style="margin:6px 13px 0;border-left-color:var(--lilac)">
      <summary><span class="cap">captured reasoning</span>
        <span style="margin-left:auto">${reasoningBlocks.length} think block(s)</span></summary>
      <small>model-emitted &lt;think&gt; text · inspectable evidence, not a privileged or verified thought ·
        excluded from the answer, continuation history, token forks, and tool parsing</small>
      ${reasoningBlocks.map((b, i) => html`<pre key=${i} style="white-space:pre-wrap;margin:7px 0 0">${b.text || ""}</pre>
        <span class=${"tag " + (b.closed ? "cap-t" : "fail-t")}>${b.closed ? "CLOSED" : "UNCLOSED"}</span>`)}
    </details>` : null}
    ${rec && !rec._sample && html`<div class="trust-channels" data-testid="trust-channels">
      <section><b>confidence</b>
        ${truthMeta && truthMeta.available && !truthMeta.small_n
          ? html`<span class="tag cap-t">TRUTH-CAL</span><span>T ${(+truthMeta.temperature).toFixed(2)} · n ${truthMeta.n}</span>
              <small>labeled-probe estimate for this model; not a fact-check of this span.</small>`
          : html`<span class="tag smp-t">RAW / PROXY</span><span>${truthMeta && truthMeta.small_n ? "truth fit small n " + truthMeta.n : "no matching temperature fit"}</span>
              <small>raw confidence does not equal correctness.</small>`}
      </section>
      <section><b>acceptance</b><span class=${"tag " + (proxyMeta && proxyMeta.available ? "der-t" : "smp-t")}>
          ${proxyMeta && proxyMeta.available ? "PROXY" : "UNAVAILABLE"}</span>
        <span>${proxyMeta && proxyMeta.available ? proxyMeta.n_scored + " journal runs" : "no scored organic curve"}</span>
        <small>how often similar-confidence runs were kept; not correctness.</small></section>
      <section><b>support</b><span class=${"tag " + (supportMeta && supportMeta.available ? "cap-t" : "smp-t")}>
          ${supportMeta && supportMeta.requested
            ? (supportMeta.available ? supportMeta.n_entailed + "/" + supportMeta.n_spans + " ENTAILED" : "UNAVAILABLE")
            : "NOT CHECKED"}</span>
        <button class=${"spd" + (supportBusy ? " busy" : "")} data-testid="check-support"
          disabled=${supportBusy} onClick=${checkSupport}>${supportBusy ? "CHECKING…" : "CHECK SUPPORT"}</button>
        <small>optional local NLI (~440 MB; may use an accelerator), no generation.
          ${!supportMeta || !supportMeta.requested
            ? "Not run yet; the evidence tier will be disclosed after checking."
            : supportMeta.evidence_tier === "causal_receipts"
            ? "Uses stored causal receipts only."
            : "No stored causal receipt: uses the active-influence manifest (presence, not causality)."}
          Never external evidence.</small></section>
    </div>`}
  </div>`;
}

/* ── the live chat (W2): SSE deltas type onto the CRT; the run is adopted from the journal
      after [DONE] (contracts §18: the stream carries no run id, deliberately) ── */
let chatAbortFn = null;
async function doSend(rec, text, cont){
  const s = store.get();
  if(!s.live){ toast("chat — live server only (this is the sample reel)"); return; }
  /* review finding #7: chatBuf stays non-null for the whole send, including the post-stream
     journal-poll window after `chatting` itself has already flipped back to false -- guard on
     both, so a second send can't start while the first is still settling into the journal. */
  if(s.chatting || s.chatBuf != null) return;
  const messages = (cont && rec && Array.isArray(rec.messages) ? rec.messages.map(m => ({ ...m })) : []);
  /* A run stores the request transcript and its current assistant answer separately.  Continuing must
     include that clean public answer exactly once; reasoning evidence is intentionally never copied. */
  if(cont && rec && rec.response != null && (!messages.length || messages[messages.length - 1].role !== "assistant"))
    messages.push({ role: "assistant", content: String(rec.response) });
  messages.push({ role: "user", content: text });
  const lensLayer = s.lensLayer || 0;                 /* F1: 0 = off; else the requested J-lens depth */
  store.set({ chatting: true, chatBuf: "", chatPrompt: text, playing: false, liveLens: null, livePolicy: null });
  chatAbortFn = api.chatStream(messages, {
    lens: lensLayer ? { layer: lensLayer, topk: 4 } : null,
    onLens: fr => {                                    /* F1: the disposed-to-say readout, mid-stream */
      if(fr && fr.error){ store.set({ liveLens: { error: String(fr.error) } }); return; }
      const row = (fr.readouts && fr.readouts[0]) || [];
      store.set({ liveLens: { layer: fr.layer, t: fr.t,
        words: row.map(x => ({ piece: String(x.piece || "").trim(), score: +x.score })) } });
    },
    onPolicy: fr => store.set({ livePolicy: fr || null }),
    onDelta: chunk => store.set(st => ({ chatBuf: (st.chatBuf || "") + chunk })),
    onDone: async final => {
      store.set({ chatting: false });
      let runId = final && final.clozn_run_id;
      if(!runId){
        const latest = await api.latestRun();             /* session-key fallback for older gateways */
        runId = latest && latest.available && latest.run && latest.run.id;
      }
      const l = await api.listRuns();
      store.set({ ...(l && Array.isArray(l.runs) ? { runs: l.runs } : {}),
        chatBuf: null, chatPrompt: null });
      if(runId){
        await loadRun(runId);
        toast("run " + runId + " recorded — on the tape");
      }else{
        toast("stream ended but no run id was returned — refresh the journal to inspect it");
      }
    },
    onError: err => { store.set({ chatting: false, chatBuf: null, chatPrompt: null });
      toast("chat stream failed: " + err); },
  });
}
const LENS_STOPS = [0, 2, 14, 21, 25];   /* 0 = off, else the fitted J-lens depths */
function ChatBar({ rec }){
  const chatting = useStore(x => x.chatting);
  const chatBuf = useStore(x => x.chatBuf);
  const live = useStore(x => x.live);
  const lensLayer = useStore(x => x.lensLayer);
  const [text, setText] = useState("");
  const [cont, setCont] = useState(false);
  /* review finding #7: chatBuf stays non-null through the post-stream journal-poll window even
     after chatting flips back to false -- treat that window as busy too, so the input/SEND stay
     disabled instead of inviting a second send while the first is still settling. */
  const busy = chatting || chatBuf != null;
  const send = () => {
    if(store.get().chatting || store.get().chatBuf != null) return;   /* double-send guard */
    const t = text.trim(); if(!t) return;
    doSend(rec, t, cont); setText("");
  };
  return html`<div class="chatbar">
    <input type="text" placeholder=${live
        ? "say something — it types onto the monitor and lands in the journal"
        : "live server only — this is the sample reel"}
      value=${text} disabled=${busy || !live}
      onInput=${e => setText(e.target.value)}
      onKeyDown=${e => { if(e.key === "Enter") send(); }}/>
    <button class="spd" disabled=${!live || busy}
      style=${lensLayer ? "color:#7A6FB8;border-color:rgba(154,146,200,.7)" : ""}
      title="live lens: stream the J-lens disposed-to-say readout per token while it generates (slows decoding a little)"
      onClick=${() => {
        const next = LENS_STOPS[(LENS_STOPS.indexOf(lensLayer) + 1) % LENS_STOPS.length];
        store.set({ lensLayer: next });
        toast(next ? "live lens ON — layer " + next + " (disposed-to-say, content-only)" : "live lens off");
      }}>${lensLayer ? "◉ LENS L" + lensLayer : "○ LENS"}</button>
    ${chatting
      ? html`<button class="spd" style="color:#C24A31;border-color:rgba(242,109,79,.6)"
          onClick=${() => { chatAbortFn && chatAbortFn();
            store.set({ chatting: false, chatBuf: null, chatPrompt: null });
            toast("stream aborted"); }}>ABORT</button>`
      : html`<button class="spd" disabled=${!live || busy} onClick=${send}>SEND ⏎</button>`}
    ${rec && Array.isArray(rec.messages) && rec.messages.length
      ? html`<label><input type="checkbox" checked=${cont}
          onChange=${e => setCont(e.target.checked)}/> continue this run</label>` : null}
  </div>`;
}

/* ───────────────────────── logs (derived from the record — never invented) ── */
function Logs({ rec }){
  const t = shortTime(rec.created_at) || "—";
  const mem = rec.memory || {};
  const dials = Object.entries((rec.behavior || {}).active_dials || {}).filter(([,v]) => v);
  const rows = [];
  const row = (col, tag, msg, ok = true) => rows.push({ col, tag, msg, ok });
  row("var(--blue)", "RUN", "recorded — " + (rec.id || ""));
  if((mem.cards_applied || []).length)
    row("var(--teal)", "MEMORY", `block injected · gate ${mem.gate != null ? (+mem.gate).toFixed(2) : "—"} · ${mem.cards_applied.length} card(s)`);
  if((mem.anchored || []).length)
    row("var(--lilac)", "ANCHOR", `J-space steer · L${mem.anchored_layer || 21} · ${mem.anchored.length} bag(s)`);
  if(mem.anchored_skipped)
    row("var(--coral)", "ANCHOR", mem.anchored_skipped, false);
  if(dials.length) row("var(--lilac2)", "STEER", "dials · " + dials.map(([k,v]) => k + " " + (+v).toFixed(1)).join(" · "));
  (rec.tiny_tests || []).forEach(tt =>
    row(tt.pass ? "var(--teal)" : "var(--coral)", "TEST", `expectation “${tt.name || "?"}” · ${tt.kind || "static"}`, !!tt.pass));
  if(rec.error) row("var(--coral)", "ERROR", String(rec.error).slice(0, 120), false);
  const trunc = rec.finish_reason === "length";
  row(trunc ? "var(--coral)" : "var(--teal)", "FINISH", rec.finish_reason || "?", !trunc);
  return html`<div class="mod">
    <span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led blue beats"></span><span class="cap">local logs</span>
      <span class="tail">${rows.length} entries</span></div>
    <div class="logbox">
      ${rows.map(r => html`<div class="logline">
        <span class="t">${t}</span>
        <span class="lt" style=${"color:" + r.col}>${r.tag}</span>
        <span class="m">${r.msg}
          <span class=${r.ok ? "ok" : "fail"} style="float:right">${r.ok ? "✓" : "fail"}</span></span>
      </div>`)}
    </div>
  </div>`;
}

function Cfg({ rec }){
  const d = (rec.meta && rec.meta.decode) || {};
  const bit = (k, v) => v != null && html`<span class="dot">·</span><span>${k} <b>${String(v)}</b></span>`;
  /* review finding #12: `meta.decode` only ever carries mode/temperature/seed (contracts §2) --
     there is no `quant` here on a real run (it only exists in the export bundle's separately
     computed `repro` object, a different shape entirely); a `quant` bit here would either never
     render (guarded by `!= null`, so harmless) or only ever show the SAMPLE fixture's dead value.
     Dropped rather than displaying a field that can never be true of a live run. */
  return html`<div class="cfg">
    <span class="cap">run</span><b>${rec.model || "—"}</b>
    ${bit("decode", d.mode)}${bit("temp", d.temperature)}${bit("seed", d.seed)}
    ${bit("finish", rec.finish_reason || "?")}
    <span style="margin-left:auto" class="tag cap-t">CAPTURED</span>
  </div>`;
}

/* ───────────────────────── the tape module ───────────────────────── */
function TapeMod({ rec }){
  return html`<div class="mod" style="margin-top:0">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led"></span><span class="cap">replay tape</span>
      <span class="tail"><${RunTail}/></span></div>
    <${RunStrip}/>
    <${Transport} rec=${rec}/>
    <${VerbResult}/>
    <${Leader} rec=${rec}/>
    <${Arrangement} rec=${rec}/>
    <${JlensProvenance}/>
    <${LineageTree} rec=${rec}/>
  </div>`;
}
function RunTail(){
  const s = useStore(x => ({ n: x.runs.length, live: x.live }));
  return s.n ? (s.live ? s.n + " recorded" : "sample reel") : "—";
}

function RunStrip(){
  const s = useStore(x => ({ runs: x.runs, currentId: x.currentId, live: x.live, full: x.full }));
  return html`<div class="runstrip">
    ${s.runs.map(r => html`<button key=${r.id} class=${"runchip" + (s.currentId === r.id ? " on" : "")}
        onClick=${() => loadRun(r.id)}>
      <span class="t">${r.prompt_summary || r.id}</span>
      <${Thumb} id=${r.id}/>
      <span class="m">${shortTime(r.created_at)} · ${r.finish_reason || ""}
        ${r.parent_run_id && html` <span class="child">⑂ child</span>`}
        ${(r.flags || []).includes("error") && html` <span style="color:var(--coral)">· error</span>`}</span>
    </button>`)}
  </div>`;
}
/* review finding #10: RunStrip can render up to 80 chips (the server's cap), and every Thumb used to
   fire its own full-record GET on mount regardless of whether it was ever scrolled into view --
   opening the tape with a populated history fired up to 80 full-record fetches just to draw
   decorative sparklines. Defer the fetch until the canvas is actually (near-)visible; browsers
   without IntersectionObserver fall back to hydrating immediately (still correct, just eager). */
function Thumb({ id }){
  const ref = useRef(null);
  const full = useStore(x => x.full[id]);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    if(inView) return;
    const el = ref.current;
    if(!el || typeof IntersectionObserver === "undefined"){ setInView(true); return; }
    const io = new IntersectionObserver(entries => {
      if(entries.some(e => e.isIntersecting)) setInView(true);
    }, { rootMargin: "160px" });
    io.observe(el);
    return () => io.disconnect();
  }, [id, inView]);
  useEffect(() => {
    if(!inView) return;
    let dead = false;
    (async () => {
      let rec = full;
      if(!rec && store.get().live){
        const r = await api.getRun(id); rec = r && (r.run || r);
        if(rec) store.set(st => ({ full: withFull(st.full, id, rec) }));
      }
      if(dead || !rec || !ref.current) return;
      const conf = (rec.trace || {}).confidence || normSteps(rec).map(s => s.conf ?? 0);
      const cv = ref.current;
      if(!conf.length){ cv.style.display = "none"; return; }
      const x = cv.getContext("2d"), W = cv.width, H = cv.height, n = conf.length;
      x.clearRect(0,0,W,H);
      const gr = x.createLinearGradient(0,0,W,0);
      gr.addColorStop(0,"rgba(44,191,232,.85)"); gr.addColorStop(1,"rgba(95,200,188,.85)");
      x.fillStyle = gr;
      conf.forEach((c,i) => { const a = Math.max(1,(c||0)*(H-2)/2);
        x.fillRect(i*(W/n), H/2 - a, Math.max(1, W/n - 1), a*2); });
    })();
    return () => { dead = true; };
  }, [id, full, inView]);
  return html`<canvas ref=${ref} width="136" height="15"></canvas>`;
}

/* ── transport: the verbs are LIVE ── */
function Transport({ rec }){
  const P = useStore(x => x.P);
  const playing = useStore(x => x.playing);
  const speed = useStore(x => x.speed);
  const busy = useStore(x => x.busy);
  const steps = normSteps(rec);
  const n = steps.length;
  const stops = (() => {
    const s = [0];
    const shaky = steps.findIndex(x => (x.conf ?? 1) < .5);
    if(shaky > 0) s.push(shaky);
    s.push(n);
    return [...new Set(s)].sort((a,b) => a-b);
  })();
  const cur = steps[Math.max(0, P-1)];
  const setP = v => store.set({ P: Math.max(0, Math.min(n, v)), playing: false });

  return html`<div class="transport">
    <div class="tbtns">
      <button class="tbtn" onClick=${() => setP(0)}>⇤</button>
      <button class="tbtn" onClick=${() => setP(stops.filter(x => x < P).pop() ?? 0)}>◂◂</button>
      <button class="tbtn play" onClick=${() => {
        if(playing){ store.set({ playing: false }); return; }
        store.set({ P: P >= n ? 0 : P, playing: true });
      }}>${playing ? "❚❚" : "▶"}</button>
      <button class="tbtn" onClick=${() => setP(stops.find(x => x > P) ?? n)}>▸▸</button>
      <button class="tbtn" onClick=${() => setP(n)}>⇥</button>
    </div>
    <div class="pos"><span class="big">${P}</span>
      <span class="of">/ ${n} · ${P ? `“${(cur.piece || "").trim() || "·"}” conf ${cur.conf != null ? cur.conf.toFixed(2) : "—"}` : "start"}</span></div>
    <div class="right">
      <button class="spd" onClick=${() => store.set({ speed: speed === 1 ? 2 : speed === 2 ? 4 : speed === 4 ? .5 : 1 })}>${speed.toFixed(1)}×</button>
      <button class=${"spd" + (busy.rederive ? " busy" : "")} onClick=${() => doRederive(rec)}>⟲ RE-DERIVE</button>
      <button class=${"spd" + (busy.replay ? " busy" : "")} onClick=${() => doReplay(rec)}>⏵ REPLAY</button>
      <button class=${"spd" + (busy.branch ? " busy" : "")} onClick=${() => doBranch(rec)}>⑂ BRANCH</button>
      <button class=${"spd" + (busy.prove ? " busy" : "")}
        style="color:#1B7F74;border-color:rgba(95,200,188,.7)" onClick=${() => doProve(rec)}>
        ${busy.prove ? "PROVING…" : "PROVE"}</button>
      <button class="spd" onClick=${() => {
        if(rec._sample) return toast("EXPORT — live runs only (this is the sample reel)");
        window.open(api.exportUrl(rec.id), "_blank");
      }}>⤓ EXPORT</button>
      <button class="spd" title="one self-contained HTML receipt card — save it, post it"
        onClick=${() => {
          if(rec._sample) return toast("CARD — live runs only (this is the sample reel)");
          window.open(api.cardUrl(rec.id), "_blank");
        }}>⧉ CARD</button>
    </div>
  </div>`;
}

/* verb result strip — outcomes reported plainly, failures included */
function VerbResult(){
  const vr = useStore(x => x.verbResult);
  if(!vr) return null;
  return html`<div class="cfg" style="margin:8px 13px 0;border-left-color:${vr.ok ? "var(--teal)" : "var(--coral)"}">
    <span class="cap" style=${vr.ok ? "" : "color:#C24A31"}>${vr.verb}</span>
    <span>${vr.msg}</span>
    <button class="spd" style="margin-left:auto;height:21px;font-size:8.5px"
      onClick=${() => store.set({ verbResult: null })}>dismiss</button>
  </div>`;
}
const setBusy = (verb, v) => store.set(st => ({ busy: { ...st.busy, [verb]: v } }));
const guardSample = rec => {
  if(rec._sample){ toast("Live runs only — this is the sample reel"); return true; }
  return false;
};

async function doRederive(rec){
  if(guardSample(rec)) return;
  setBusy("rederive", true); toast("re-deriving — teacher-forced re-scoring under the run's own conditions…");
  const res = await api.rederive(rec.id, {});
  setBusy("rederive", false);
  if(!res){ store.set({ verbResult: { verb: "re-derive", ok: false,
    msg: "no answer — rederive needs the engine substrate (score_tokens); is the engine up?" } }); return; }
  /* contracts §3: {text, steps:[{piece,token_id,logprob,conf}], meta:{retokenized, block_source, n_tokens}} */
  const m = res.meta || {};
  const confs = (res.steps || []).map(s => s.conf).filter(c => c != null);
  const mean = confs.length ? (confs.reduce((a,b) => a+b, 0) / confs.length) : null;
  store.set({ verbResult: { verb: "re-derive", ok: true,
    msg: `re-scored ${m.n_tokens ?? (res.steps || []).length} tokens deterministically · mean conf ${mean != null ? mean.toFixed(3) : "—"}`
       + ` · ${m.retokenized ? "retokenized (boundary-approximate)" : "exact token ids"}`
       + (m.block_source ? ` · conditions from ${m.block_source}` : "") } });
}

/* replay/branch both return THE FULL CHILD RUN RECORD (contracts §4/§5) */
async function adoptChild(res, verb, extra){
  const l = await api.listRuns();
  store.set(st => ({
    runs: (l && Array.isArray(l.runs)) ? l.runs : st.runs,
    full: withFull(st.full, res.id, res),
    currentId: res.id, rec: res, P: 0, receipts: null, leaning: null,
    verbResult: { verb, ok: true, msg: `child ${res.id} recorded — now on the tape (parent: ${res.parent_run_id || "?"})${extra || ""}` },
  }));
}
async function doReplay(rec){
  if(guardSample(rec)) return;
  setBusy("replay", true); toast("replaying — greedy, state-safe…");
  /* contracts §4: the knobs live INSIDE changes_applied; greedy:true forces attributable decode */
  const res = await api.replay(rec.id, { changes_applied: { greedy: true } });
  setBusy("replay", false);
  if(!res || !res.id){ store.set({ verbResult: { verb: "replay", ok: false,
    msg: "replay didn't answer — needs a chat-capable substrate; is the engine up?" } }); return; }
  await adoptChild(res, "replay", " · greedy");
}
async function doBranch(rec){
  if(guardSample(rec)) return;
  const edited = window.prompt("Branch — edit the user turn (regenerates from there, greedy):", firstLine(rec));
  if(edited == null) return;
  /* contracts §5: turn = the USER-TURN ordinal; single-turn studio runs branch at their last user turn */
  const msgs = rec.messages || rec.assembled_messages || [];
  const userTurns = msgs.filter(x => x.role === "user").length;
  const turn = Math.max(0, userTurns - 1);
  setBusy("branch", true); toast("branching at turn " + turn + "…");
  const res = await api.branch(rec.id, { turn, alt_user: edited, sample: false });
  setBusy("branch", false);
  if(!res || !res.id){ store.set({ verbResult: { verb: "branch", ok: false,
    msg: "branch didn't answer (turn out of range, or generation error)" } }); return; }
  await adoptChild(res, "branch");
}

async function doProve(rec){
  if(guardSample(rec)) return;
  setBusy("prove", true);
  toast("asking what mattered — leave-one-out both-arms-greedy + forced scoring (mode: both)…");
  const res = await api.receipts(rec.id, { mode: "both" });   // contracts §7
  setBusy("prove", false);
  if(!res){ store.set({ verbResult: { verb: "prove", ok: false,
    msg: "receipts didn't answer — they regenerate arms and can take minutes; is the engine up?" } }); return; }
  const list = res.receipts || [];
  const forced = res.forced_receipts || [];
  const key = inf => JSON.stringify(inf || {});
  const forcedBy = new Map(forced.map(f => [key(f.influence), f]));
  /* leaning heat: the strongest causally-verified forced receipt's per-token |deltas| (contracts §6) */
  let leaning = null, leaningLabel = null;
  for(const f of forced){
    if(!f.causal_verified || !Array.isArray(f.deltas)) continue;
    if(!leaning || Math.abs(f.sum_nats || 0) > Math.abs(leaningLabel.sum_nats || 0)){
      leaning = f.deltas; leaningLabel = f;
    }
  }
  store.set({
    receipts: { raw: res, list, forcedBy, skipped: res.skipped || [],
                redundant: res.redundant_pairs || [], keyFn: key },
    leaning, leaningInfluence: leaningLabel ? leaningLabel.influence : null,
    verbResult: { verb: "prove", ok: true,
      msg: `${list.length} influence(s) measured, ${forced.length} force-scored`
         + (res.skipped && res.skipped.length ? ` · ${res.skipped.length} skipped (honestly)` : "")
         + (leaning ? " · forced Δ painted on the clips" : "") },
  });
}

/* ── leader ── */
function Leader({ rec }){
  const fp = rec.final_prompt || null, am = rec.assembled_messages;
  const body = fp || (Array.isArray(am) ? am.map(x => `⟨${x.role}⟩ ${x.content}`).join("\n")
    : Array.isArray(rec.messages) ? rec.messages.map(x => `⟨${x.role}⟩ ${x.content}`).join("\n") : "");
  if(!body) return null;
  return html`<details class="leader">
    <summary>Leader — the exact prompt the model saw
      <span style="margin-left:auto">${fp ? "final_prompt" : "messages"}</span></summary>
    <div class="leader-body">${body}</div>
  </details>`;
}

/* ───────────────────────── arrangement ───────────────────────── */
function Arrangement({ rec }){
  const P = useStore(x => x.P);
  const leaning = useStore(x => x.leaning);
  const steps = normSteps(rec);
  const w = weightsFor(steps);
  const cols = colsFor(w);
  const innerW = Math.max(680, w.reduce((a,b) => a+b, 0) * 8.5 + 130);
  const hasEnt = steps.some(s => s.ent != null);
  const mem = rec.memory || {};
  const arrRef = useRef(null);
  const [sel, setSel] = useState(null);
  const [jl, setJl] = useState(null);
  const [jlBusy, setJlBusy] = useState(false);
  useEffect(() => { setJl(null); setSel(null); }, [rec.id]);

  /* live width of `.arr`: `.arr` has no fixed width, so it stretches to its container whenever
     that's wider than `innerW` (review finding #2/#9). Measure it for real, on mount/run-switch and
     on resize, instead of trusting the static innerW estimate; the playhead effect below re-runs off
     this too so it no longer goes stale after a viewport/DPI change. */
  const [liveW, setLiveW] = useState(null);
  useEffect(() => {
    const arr = arrRef.current;
    if(!arr) return;
    const measure = () => setLiveW(arr.clientWidth);
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [rec.id]);

  /* playhead + shade positioning */
  const phRef = useRef(null), shadeRef = useRef(null);
  const plateW = 118, pad = 13;
  useEffect(() => {
    const arr = arrRef.current, ph = phRef.current;
    if(!arr || !ph) return;
    const W = arr.clientWidth - plateW - pad*2;
    const g = colGeom(w, W);
    const x = P >= steps.length ? (g.length ? g[g.length-1].x1 : 0) : (P <= 0 ? 0 : g[P-1].x1);
    ph.hidden = false;
    ph.style.left = (plateW + pad + x) + "px";
  }, [P, rec.id, liveW]);
  useEffect(() => {
    const arr = arrRef.current, shade = shadeRef.current;
    if(!arr || !shade) return;
    const move = e => {
      const r = arr.getBoundingClientRect(), x = e.clientX - r.left - plateW - pad;
      if(x < 0){ shade.hidden = true; return; }
      const g = colGeom(w, arr.clientWidth - plateW - pad*2);
      const i = g.findIndex(c => x >= c.x0 && x < c.x1);
      if(i < 0){ shade.hidden = true; return; }
      shade.hidden = false;
      shade.style.left = (plateW + pad + g[i].x0) + "px";
      shade.style.width = (g[i].x1 - g[i].x0) + "px";
    };
    const leave = () => { shade.hidden = true; };
    const dbl = e => {
      const r = arr.getBoundingClientRect(), x = e.clientX - r.left - plateW - pad;
      if(x < 0) return;
      const g = colGeom(w, arr.clientWidth - plateW - pad*2);
      const i = g.findIndex(c => x >= c.x0 && x < c.x1);
      if(i >= 0) store.set({ P: i + 1, playing: false });
    };
    arr.addEventListener("mousemove", move);
    arr.addEventListener("mouseleave", leave);
    arr.addEventListener("dblclick", dbl);
    return () => { arr.removeEventListener("mousemove", move);
      arr.removeEventListener("mouseleave", leave); arr.removeEventListener("dblclick", dbl); };
  }, [rec.id]);

  const leanMax = leaning ? Math.max(1e-6, ...leaning.map(Math.abs)) : 1;

  return html`<div class="arr-wrap"><div class="arr" ref=${arrRef} style=${"width:" + innerW + "px"}>
    <div class="playhead" ref=${phRef} hidden></div>
    <div class="colshade" ref=${shadeRef} hidden></div>

    <div class="lane ruler">
      <div class="plate"><span><span class="cap">ruler</span><span class="sub">index · locators</span></span></div>
      <div class="lane-body">
        <div class="cells" style=${"grid-template-columns:" + cols + ";height:100%"}>
          ${steps.map((s,i) => html`<div class=${"tick" + (i % 5 === 0 ? " five" : "")}>
            ${i % 5 === 0 && html`<span class="idx">${i}</span>`}</div>`)}
        </div>
        <${Locators} steps=${steps} mem=${mem} rec=${rec} w=${w}
            bodyW=${(liveW ?? innerW) - plateW - pad*2}/>
      </div>
    </div>

    <div class="lane clips">
      <div class="plate"><span><span class="cap">run</span><span class="sub">token clips${leaning ? " · forced Δ" : ""}</span></span></div>
      <div class="lane-body"><div class="cells" style=${"grid-template-columns:" + cols}>
        ${steps.map((s,i) => html`<div key=${i}
            class=${"clip-t" + ((s.conf ?? 1) < .55 ? " shaky" : "") + (sel === i ? " sel" : "")}
            style=${"--c:" + (s.conf ?? .6).toFixed(2)}
            onClick=${e => { e.stopPropagation(); setSel(sel === i ? null : i); }}>
          ${(s.piece || "").trim() || "·"}
          ${leaning && leaning[i] != null && html`<span class="lean"
            style=${"--l:" + Math.round(Math.abs(leaning[i]) / leanMax * 14)}></span>`}
        </div>`)}
      </div>
      ${sel != null && html`<${Pop} step=${steps[sel]} i=${sel} w=${w} plateW=${plateW} pad=${pad} arrRef=${arrRef} rec=${rec}
          jl=${jl} jlBusy=${jlBusy} onReadJl=${() => loadJl(rec, setJl, setJlBusy, 21)}/>`}
      </div>
    </div>

    <div class="lane wave">
      <div class="plate"><span><span class="cap">confidence</span><span class="sub">CAPTURED</span></span></div>
      <div class="lane-body"><${WaveCanvas} steps=${steps} w=${w}/></div>
    </div>

    ${hasEnt && html`<div class="lane auto">
      <div class="plate"><span><span class="cap">entropy</span><span class="sub">DERIVED</span></span></div>
      <div class="lane-body"><${EntropySvg} steps=${steps} w=${w}/></div>
    </div>`}

    <div class="lane jl">
      <div class="plate"><span><span class="cap">disposed</span><span class="sub">J-lens · DERIVED</span></span>
        ${jl && (jl.layers || []).length ? html`<span style="margin-left:auto">
          ${(jl.layers).map(L => html`<button class="mono"
            style=${"font-size:7.5px;padding:0 3px;color:" + (L == jl.layer ? "var(--navy)" : "var(--mist)")}
            onClick=${() => loadJl(rec, setJl, setJlBusy, L)}>${L}</button>`)}</span>` : null}
      </div>
      <div class="lane-body">
        ${jl ? html`<div class="cells" style=${"grid-template-columns:" + cols + ";align-items:center"}>
            ${steps.map((s,i) => {
              const c = jl.byPos[i];
              return c && c.length
                ? html`<div class="jcell">${c.slice(0,3).map((t,k) =>
                    html`<span class=${"jchip d" + (k+1)}>${String(t).trim()}</span>`)}</div>`
                : html`<div></div>`;
            })}
          </div>`
        : jlBusy ? html`<div class="lane-action"><span class="none">listening back…</span></div>`
        : jl === false ? html`<div class="lane-action"><span class="none">no lens fitted for this model — blank ≠ nothing, but there is nothing fitted to read with</span></div>`
        : html`<div class="lane-action"><button onClick=${() => loadJl(rec, setJl, setJlBusy)}>Read dispositions</button></div>`}
      </div>
    </div>
  </div></div>`;
}

/* bodyW is the Arrangement's live-measured `.arr` width (finding #2) -- NOT the static innerW
   estimate: `.arr` has no fixed width, so on any viewport wider than the token weights' own minimum
   it stretches past innerW and the badges would sit at the wrong offset relative to the real,
   responsively-laid-out token columns underneath. */
function Locators({ steps, mem, rec, w, bodyW }){
  const g = colGeom(w, bodyW);
  const shaky = steps.findIndex(s => (s.conf ?? 1) < .5);
  return html`
    ${((mem.cards_applied || []).length || (mem.anchored || []).length) ? html`<span class="loc" style=${"left:" + (g[0].x0 + 2) + "px"}>
      ▸ ${(mem.anchored || []).length ? "anchored" : "memory"}${mem.gate != null ? " " + (+mem.gate).toFixed(2) : ""}</span>` : null}
    ${shaky >= 0 && html`<span class="loc" style=${"left:" + g[shaky].x0 + "px"}>▸ low conf</span>`}
    <span class=${"loc" + (rec.finish_reason === "length" ? " bad" : "")} style="right:3px">
      ▸ ${rec.finish_reason || "end"}</span>`;
}

/* ── click-a-span SIGNALS popover — the entry point: click a token of the reply, see its signals,
      one-click into an Experiment. Every signal renders ONLY when its data exists (honest absence,
      never a fabricated value); every honesty label matches replay.mjs / experiment.mjs. `jl` is the
      Arrangement's loaded J-lens (byPos/layer), or `false` (no lens fitted), or null (not read yet). ── */
function Pop({ step, i, w, plateW, pad, arrRef, rec, jl, jlBusy, onReadJl }){
  const ref = useRef(null);
  const forkBusy = useStore(x => x.busy.fork);
  const [traceState, setTraceState] = useState({ status: "idle", receipt: null });
  useEffect(() => {
    const arr = arrRef.current, el = ref.current;
    if(!arr || !el) return;
    const g = colGeom(w, arr.clientWidth - plateW - pad*2);
    el.style.left = Math.max(4, Math.min(arr.clientWidth - 244, plateW + pad + g[i].xc - 116)) + "px";
    el.style.top = "34px";
  }, [i]);
  useEffect(() => { setTraceState({ status: "idle", receipt: null }); }, [i]);   // a new token, a fresh check
  const alts = (step.alts || []).slice(0, 3);
  const spanText = (step.piece || "").trim();
  const disp = (jl && jl.byPos) ? jl.byPos[i] : null;          // disposed pieces at this position (or null)
  const lensAbsent = jl === false;                             // a lens was asked for and honestly isn't fitted
  const mem = rec.memory || {};
  const cards = mem.cards_applied || [], appliedIds = mem.applied_ids || [];
  const dials = Object.entries((rec.behavior || {}).active_dials || {}).filter(([, v]) => v);

  /* F3: click an almost-said token to FORK reality from this position (greedy continuation) */
  const fork = async piece => {
    if(rec && guardSample(rec)) return;
    if(store.get().busy.fork) return;
    setBusy("fork", true); toast(`forking at token ${i} — “${piece.trim()}” instead, greedy from there…`);
    const res = await api.fork(rec.id, i, piece);
    setBusy("fork", false);
    if(!res || !res.id){ store.set({ verbResult: { verb: "fork", ok: false,
      msg: "fork didn't answer — is the engine up?" } }); return; }
    await adoptChild(res, "fork", ` · “${piece.trim()}” at ${i}`);
  };
  /* PIECE 2 (on-demand causal panel): "trace this token" -- POST /runs/<id>/causal-trace (300c8e5)
     runs clozn.analysis.tracer.trace over THIS run's own final_prompt + recorded response at
     continuation-token `position` = the clicked token i, so this really does trace the clicked
     token (unlike the earlier provenance-route fallback, which could only ever speak to the
     answer's first token). contrast "auto" keeps it answer-SELECTIVE (scored against the runner-up
     foil, not just "any token"); screen_mode "ablate" works on any engine (no J-lens sidecar
     required). Slow (several engine round trips: screen, greedy accumulation, controls) -- on
     demand, never pre-attached, same idle -> busy -> done shape as the Monitor's ProvenanceChip. */
  const traceToken = async () => {
    if(rec._sample){ toast("live runs only — this is the sample reel"); return; }
    if(traceState.status !== "idle") return;
    setTraceState({ status: "busy", receipt: null });
    const r = await api.causalTrace(rec.id, { position: i, contrast: "auto", screen_mode: "ablate" });
    setTraceState({ status: "done", receipt: r || { ok: false, blocked: "the server didn't answer" } });
  };
  /* the ACTION: deep-link into the Experiment drawer, pre-filled, via the store handoff */
  const openExp = (ctype, fields) => {
    if(rec._sample){ toast("live runs only — this is the sample reel"); return; }
    store.set({ pendingExperiment: { ctype, fields: fields || {}, method: "" }, route: "experiment" });
    toast("opening Experiment — " + ctype + " pre-filled");
  };
  const userTurns = (rec.messages || rec.assembled_messages || []).filter(x => x.role === "user").length;
  const forkTurn = Math.max(0, userTurns - 1);

  return html`<div class="pop signals" ref=${ref} onClick=${e => e.stopPropagation()}>
    <h4>signals · token ${i}</h4>

    <div class="sig-lbl">confidence — raw, uncalibrated</div>
    <div class="alt win"><span>${spanText || "·"}</span>
      <b title="raw model probability for the committed token — UNCALIBRATED; self-confidence is not correctness">
        ${step.conf != null ? step.conf.toFixed(2) : "—"}</b></div>

    <div class="sig-lbl">almost said${alts.length ? "" : " — none recorded"}</div>
    ${alts.map(a => {
        const piece = ((a.piece ?? a.text ?? "") + "");
        return html`<div class="alt">
          <span>${piece.trim() || "·"}</span>
          <span>${a.prob != null ? (+a.prob).toFixed(2) : ""}</span>
          ${piece && !rec._sample && html`<button class="spd"
            style="height:17px;font-size:7.5px;padding:0 5px" disabled=${forkBusy}
            onClick=${e => { e.stopPropagation(); fork(piece); }}>⑂ fork</button>`}
        </div>`; })}

    <div class="sig-lbl">disposed to say${jl && jl.layer != null ? " · J-lens L" + jl.layer : " · J-lens"}</div>
    ${lensAbsent
      ? html`<span class="none" style="font-size:9px">no lens fitted for this model — nothing to read with (blank ≠ nothing)</span>`
      : disp && disp.length
        ? html`<div class="sig-run">${disp.slice(0, 3).map(p =>
            html`<span class="jchip d1">${String(p).trim() || "·"}</span>`)}</div>`
        : (jl && jl.byPos)
          ? html`<span class="none" style="font-size:9px">no disposition read at this position</span>`
          : html`<button class="spd" style="font-size:8.5px" disabled=${jlBusy}
              onClick=${e => { e.stopPropagation(); onReadJl && onReadJl(); }}>
              ${jlBusy ? "reading…" : "read disposition"}</button>`}
    ${!lensAbsent && html`<div class="sig-note">a disposition read, NOT the literal thought · lens-blind to abstractions</div>`}

    <div class="sig-lbl">active on this run</div>
    <div class="sig-run">
      ${cards.length ? html`<span class="jchip">memory · ${cards.length} card${cards.length > 1 ? "s" : ""}</span>`
        : html`<span class="none" style="font-size:9px">memory: none rode</span>`}
      ${dials.length ? dials.map(([k, v]) => html`<span class="jchip">${k} ${(+v).toFixed(1)}</span>`)
        : html`<span class="none" style="font-size:9px">dials: none</span>`}
    </div>

    ${!rec._sample && html`<div class="sig-lbl">answer source — causal trace</div>
    ${traceState.status === "idle" && html`<button class="spd" style="font-size:8.5px"
        onClick=${e => { e.stopPropagation(); traceToken(); }}>trace this token</button>`}
    ${traceState.status === "busy" && html`<span class="none" style="font-size:9px">
      tracing — ablation + matched controls over this token, several engine passes…</span>`}
    ${traceState.status === "done" && (() => {
        const r = traceState.receipt;
        if(!r || !r.ok) return html`<span class="none" style="font-size:9px">
          trace unavailable — ${(r && (r.blocked || r.error)) || "needs the engine's ablation screen"}</span>`;
        const controls = r.controls || {};
        /* controls.verdict is one of three honest outcomes (clozn/analysis/tracer.py):
             PASS             -- >=1 surviving node, controls well below the real effects
             NO_CAUSAL_NODES  -- nothing beat the noise floor (a real, non-alarming finding)
             FAILED_CONTROLS  -- random interventions moved the target as much as the "real" ones --
                                 the trace itself is untrustworthy, surfaced as a warning, never hidden */
        const verdict = String(controls.verdict || "?");
        const failed = verdict === "FAILED_CONTROLS";
        const tagClass = failed ? "fail-t" : verdict === "PASS" ? "cap-t" : "smp-t";
        /* surviving nodes only (the receipt already filters to what beat the noise floor), sorted
           by |delta_full| -- mirrors trace_circuit.py's own CLI rendering order exactly. */
        const nodes = (r.nodes || []).slice()
          .sort((a, b) => Math.abs(b.delta_full || 0) - Math.abs(a.delta_full || 0));
        return html`<div>
          <div class="alt win"><span>verdict</span>
            <b><span class=${"tag " + tagClass}>${verdict}</span></b></div>
          ${failed && html`<div class="sig-note" style="color:#C24A31;font-style:normal">
            random interventions moved the target as much as the "real" ones — the controls FAILED;
            do not trust this trace.</div>`}
          ${nodes.length
            ? nodes.map(n => html`<div class="prov-chip-row">
                <span>L${n.layer} @${n.pos}</span>
                <b>${n.control_ratio != null ? (+n.control_ratio).toFixed(1) + "x" : "n/a"}
                  ${n.legibility != null ? " · " + Math.round(n.legibility * 100) + "% legible" : ""}
                  ${n.strength ? " · " + n.strength : ""}${n.name ? " · " + n.name : ""}</b>
              </div>`)
            : html`<span class="none" style="font-size:9px">no nodes beat the noise floor — a
              distributed circuit, or nothing screenable here; that is itself the honest finding.</span>`}
          <div class="sig-note">individual sites rarely carry the answer on these models — this is
            the causal skeleton, not a full explanation.</div>
        </div>`;
      })()}
    <div class="sig-note">or for the full JSON receipt in a terminal:
      <span class="mono">clozn causal-trace --from-run ${rec.id} --pos ${i} --contrast auto</span>
    </div>`}

    ${!rec._sample && html`<div class="sig-lbl">experiment on this span</div>
    <div class="sig-actions">
      <button class="spd" onClick=${() => openExp("edit_turn", { turn: forkTurn })}>fork from here</button>
      <button class="spd" onClick=${() => openExp("ablate_memory", {})}>turn memory off</button>
      ${spanText && html`<button class="spd"
        onClick=${() => openExp("swap_concept", { to_concept: spanText, from_hint: spanText })}>swap “${spanText}”</button>`}
      ${appliedIds.length && cards.length ? html`<button class="spd"
        onClick=${() => openExp("ablate_card", { card_id: appliedIds[0] })}>prove a card</button>` : null}
    </div>`}
  </div>`;
}

function WaveCanvas({ steps, w }){
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current; if(!cv) return;
    const cell = cv.parentElement;
    const W = cell.clientWidth, H = cell.clientHeight, dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = W*dpr; cv.height = H*dpr;
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    const geo = colGeom(w, W), mid = H/2;
    g.strokeStyle = "rgba(90,130,155,.25)"; g.beginPath(); g.moveTo(0,mid); g.lineTo(W,mid); g.stroke();
    const gr = g.createLinearGradient(0,0,W,0);
    gr.addColorStop(0,"rgba(44,191,232,.6)"); gr.addColorStop(1,"rgba(95,200,188,.6)");
    g.fillStyle = gr; g.strokeStyle = "rgba(27,127,116,.85)";
    const top = [], bot = [];
    geo.forEach((c,i) => {
      const amp = Math.max(1.5, (steps[i].conf ?? 0) * (H/2 - 4));
      const seg = Math.max(2, Math.floor((c.x1 - c.x0)/3));
      for(let k = 0; k <= seg; k++){
        const x = c.x0 + (c.x1 - c.x0)*k/seg;
        const jag = 1 - .22*Math.abs(Math.sin(k*2.7 + i*1.3));
        top.push([x, mid - amp*jag]); bot.push([x, mid + amp*jag]);
      }});
    g.beginPath(); top.forEach((p,i) => i ? g.lineTo(p[0],p[1]) : g.moveTo(p[0],p[1]));
    bot.reverse().forEach(p => g.lineTo(p[0],p[1])); g.closePath(); g.fill();
    g.beginPath(); top.forEach((p,i) => i ? g.lineTo(p[0],p[1]) : g.moveTo(p[0],p[1])); g.stroke();
  }, [steps, w]);
  return html`<canvas ref=${ref}></canvas>`;
}

function EntropySvg({ steps, w }){
  const ref = useRef(null);
  const [dims, setDims] = useState(null);
  useEffect(() => {
    if(ref.current) setDims({ W: ref.current.clientWidth, H: ref.current.clientHeight });
  }, [steps]);
  if(!dims) return html`<div ref=${ref} style="height:100%"></div>`;
  const { W, H } = dims;
  const geo = colGeom(w, W);
  const mx = Math.max(1e-6, ...steps.map(s => s.ent || 0));
  const pts = steps.map((s,i) => [geo[i].xc, H - 4 - ((s.ent ?? 0)/mx)*(H - 9)]);
  return html`<div ref=${ref} style="height:100%">
    <svg viewBox=${"0 0 " + W + " " + H} preserveAspectRatio="none" style="display:block;width:100%;height:100%">
      <polyline points=${pts.map(p => p.join(",")).join(" ")} fill="none" stroke="#9A92C8" stroke-width="1.3"/>
      ${pts.map(p => html`<circle cx=${p[0]} cy=${p[1]} r="2" fill="#EAF4F9" stroke="#9A92C8" stroke-width="1.1"/>`)}
    </svg></div>`;
}

/* jlens loading (module-level cache per run+layer) — contracts §9:
   POST-only; 200 + {available:false, reason} is HONEST ABSENCE, never an error;
   readouts index over the LENS's own tokenization of run.response (usually == trace, checked). */
const jlCache = new Map();
async function loadJl(rec, setJl, setBusy, layer){
  const key = rec.id + ":" + (layer ?? "default");
  if(jlCache.has(key)){ setJl(jlCache.get(key)); return; }
  setBusy(true); setJl(null);
  let data = rec._jlens || null;
  if(!data) data = await api.jlens(rec.id, layer);
  setBusy(false);
  if(!data){ setJl(false); store.set({ jlProvenance: null, jlReason: "the server didn't answer" }); return; }
  if(data.available === false){
    setJl(false); store.set({ jlProvenance: null, jlReason: data.reason || "no lens available" }); return;
  }
  const n = normSteps(rec).length;
  const byPos = Array.from({ length: n }, () => null);
  let lensN = n;
  if(data.chips) Object.entries(data.chips).forEach(([p,c]) => { if(byPos[+p] !== undefined) byPos[+p] = c; });
  else if(Array.isArray(data.readouts)){
    lensN = data.readouts.length;
    data.readouts.forEach((r,i) => {
      if(i < n && Array.isArray(r) && r.length) byPos[i] = r.map(x => x.piece ?? x.label ?? x); });
  }
  const pv = data.provenance;
  const p = (pv && typeof pv === "object") ? pv : {};
  const base = typeof pv === "string" ? pv
    : [p.note, p.fit_model ? "fitted: " + p.fit_model : null].filter(Boolean).join(" · ");
  const provText = [base || null,
                    lensN !== n ? `(lens read ${lensN} tokens vs ${n} in the trace — aligned by index)` : null]
                   .filter(Boolean).join(" · ") || null;
  const out = { byPos, layer: data.layer,
                layers: data.available_layers || data.layers || p.layers || [],
                provenance: provText };
  jlCache.set(key, out); setJl(out);
  store.set({ jlProvenance: provText, jlReason: null });
}
function JlensProvenance(){
  const pv = useStore(x => x.jlProvenance);
  const reason = useStore(x => x.jlReason);
  if(pv) return html`<div class="provenance"><b>J-lens</b> — ${pv}</div>`;
  if(reason) return html`<div class="provenance"><b>J-lens</b> — unavailable: ${reason} · blank ≠ nothing, but there is nothing fitted to read with</div>`;
  return null;
}

/* ───────────────────────── state scope ───────────────────────── */
function ScopeMod({ rec }){
  const canvasRef = useRef(null);
  const [rot, setRot] = useState(42);
  const [zoom, setZoom] = useState(100);
  const [focus, setFocus] = useState(3);
  const [meta, setMeta] = useState({ real: false, layerIds: [0,6,12,18,24] });
  const dataRef = useRef(null);

  useEffect(() => {
    let dead = false;
    /* per-pane glowing words: dedupe top-1 dispositions across positions, keep the 4 strongest */
    const topWords = read => {
      const best = {};
      ((read && read.readouts) || []).forEach(cell => {
        const t = cell && cell[0]; if(!t) return;
        const p = String(t.piece || "").trim(); if(!p) return;
        const s = +t.score || 0;
        if(!(p in best) || s > best[p]) best[p] = s;
      });
      const list = Object.entries(best).sort((a,b) => b[1] - a[1]).slice(0, 4);
      const mx = Math.max(1e-6, ...list.map(x => x[1]));
      return list.map(([piece, s]) => ({ piece, s01: s / mx }));
    };
    (async () => {
      const steps = normSteps(rec);
      const text = firstLine(rec);
      let norms = null, layerIds = null, real = false, words = null;
      let rawNorms = null;
      if(store.get().live && !rec._sample){
        const r = await api.engineLayers(text);
        if(!dead && r && Array.isArray(r.norms) && r.norms.length) rawNorms = r.norms;
        /* THE FUSION: read the same text through the lens at every fitted depth */
        const first = await api.jlensText(text);
        if(!dead && first && first.available !== false){
          const lensLayers = first.available_layers ||
            (first.provenance && first.provenance.layers) || [first.layer];
          const reads = { [first.layer]: first };
          for(const L of lensLayers){
            if(reads[L] || dead) continue;
            const rr = await api.jlensText(text, L);
            if(rr && rr.available !== false) reads[L] = rr;
          }
          layerIds = lensLayers.slice();
          words = layerIds.map(L => reads[L] ? topWords(reads[L]) : null);
        }
      }
      if(rawNorms){
        const nl = rawNorms.length;
        if(!layerIds){
          layerIds = [0, Math.floor(nl*.25), Math.floor(nl*.5), Math.floor(nl*.75), nl-1];
        }
        norms = layerIds.map(L => rawNorms[Math.min(nl - 1, Math.max(0, L))]);
        real = true;
      }
      if(!norms){
        if(!layerIds) layerIds = [0, 6, 12, 18, 24];
        norms = layerIds.map((L,li) => steps.map((s,i) =>
          (s.conf ?? .6) * (0.45 + 0.55*Math.abs(Math.sin(li*1.3 + i*.7)))));
      }
      if(dead) return;
      dataRef.current = { norms, layerIds, words };
      setMeta({ real, lens: !!(words && words.some(Boolean)), layerIds });
    })();
    return () => { dead = true; };
  }, [rec.id]);

  useEffect(() => {
    const d = dataRef.current, cv = canvasRef.current;
    if(!d || !cv) return;
    let raf = 0;
    const wrap = cv.parentElement;
    const W = wrap.clientWidth - 24, H = 230, dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = W*dpr; cv.height = H*dpr; cv.style.width = W + "px"; cv.style.height = H + "px";
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    const { norms, layerIds, words } = d;
    const NP = norms.length;
    const paneW = 82*(zoom/100), paneH = 156*(zoom/100), skew = 10 + (rot/100)*26;
    const span = W - paneW - 56, x0 = 30;
    const PAL = [[76,141,240],[95,200,188],[143,168,232],[182,176,218],[127,180,240]];
    const mx = Math.max(1e-6, ...norms.flat());
    const colsN = 6, rowsN = 9;
    const grids = norms.map(row => {
      const out = [];
      for(let r = 0; r < rowsN; r++) for(let c = 0; c < colsN; c++){
        const t = (r*colsN + c)/(rowsN*colsN);
        out.push((row[Math.min(row.length-1, Math.floor(t*row.length))] || 0)/mx);
      } return out; });
    const quad = i => { const px = x0 + span*(i/(NP-1)), py = H/2;
      return { tl:[px, py-paneH/2 - skew*.4], tr:[px+paneW, py-paneH/2 + skew*.4],
               br:[px+paneW, py+paneH/2 + skew*.4], bl:[px, py+paneH/2 - skew*.4], y:py }; };
    function frame(ms){
      const T = ms*.001;
      g.clearRect(0,0,W,H);
      for(let i = 0; i < NP-1; i++){
        const a = quad(i), b = quad(i+1);
        for(let k = 0; k < 9; k++){
          const ya = a.y - paneH/2 + paneH*(k+.5)/9, yb = b.y - paneH/2 + paneH*((k*3.7)%9 + .5)/9;
          const c1 = PAL[i%PAL.length], c2 = PAL[(i+1)%PAL.length];
          const gr = g.createLinearGradient(a.tr[0], ya, b.tl[0], yb);
          gr.addColorStop(0, `rgba(${c1[0]},${c1[1]},${c1[2]},.30)`);
          gr.addColorStop(1, `rgba(${c2[0]},${c2[1]},${c2[2]},.30)`);
          g.strokeStyle = gr; g.lineWidth = .8;
          g.beginPath(); g.moveTo(a.tr[0]-4, ya);
          const mid = (a.tr[0] + b.tl[0])/2, sway = REDUCED ? 0 : Math.sin(T*.7 + i*2 + k)*5;
          g.bezierCurveTo(mid, ya + sway, mid, yb - sway, b.tl[0]+4, yb); g.stroke();
        }
      }
      for(let i = 0; i < NP; i++){
        const q = quad(i), c = PAL[i%PAL.length], isF = i === focus;
        const wl = (words && words[i]) || null;          /* the pane's glowing dispositions */
        g.beginPath(); g.moveTo(...q.tl); g.lineTo(...q.tr); g.lineTo(...q.br); g.lineTo(...q.bl); g.closePath();
        g.fillStyle = `rgba(255,255,255,${isF ? .55 : .35})`; g.fill();
        g.strokeStyle = `rgba(${c[0]},${c[1]},${c[2]},${isF ? .95 : .55})`; g.lineWidth = isF ? 1.6 : 1; g.stroke();
        const cells = grids[i];
        const dotDim = wl && wl.length ? .5 : 1;          /* dots recede when the words are present */
        for(let r = 0; r < rowsN; r++) for(let cc = 0; cc < colsN; cc++){
          const u = (cc+.5)/colsN, v = (r+.5)/rowsN;
          const x = q.tl[0] + (q.tr[0]-q.tl[0])*u;
          const yTop = q.tl[1] + (q.tr[1]-q.tl[1])*u, yBot = q.bl[1] + (q.br[1]-q.bl[1])*u;
          const y = yTop + (yBot-yTop)*v;
          let a = cells[r*colsN+cc] * dotDim;
          if(!REDUCED) a *= .8 + .2*Math.sin(T*1.57 + i + r*.6 + cc*.4);
          g.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${.15 + a*.8})`;
          g.beginPath(); g.arc(x, y, 1.3 + a*2.3, 0, 7); g.fill();
        }
        /* THE FUSION: the disposed-to-say words glowing inside the glass, per depth */
        if(wl) wl.forEach((wd, k) => {
          const u = .5 + .17*Math.sin(k*2.1 + i*1.3);
          const v = .16 + k*.22;
          const x = q.tl[0] + (q.tr[0]-q.tl[0])*u;
          const yTop = q.tl[1] + (q.tr[1]-q.tl[1])*u, yBot = q.bl[1] + (q.br[1]-q.bl[1])*u;
          const y = yTop + (yBot-yTop)*v + (REDUCED ? 0 : Math.sin(T*.9 + i + k*1.7)*2.5);
          const fs = (8.5 + wd.s01*5) * (isF ? 1.12 : 1);
          g.font = "600 " + fs.toFixed(1) + "px " + '"IBM Plex Mono",monospace';
          g.textAlign = "center";
          const pulse = REDUCED ? 1 : (.88 + .12*Math.sin(T*1.57 + k*.9 + i));   /* the heartbeat */
          g.shadowColor = `rgba(${c[0]},${c[1]},${c[2]},.95)`;
          g.shadowBlur = 6 + wd.s01*8;
          g.fillStyle = `rgba(255,255,255,${(.5 + .45*wd.s01) * pulse})`;
          g.fillText(wd.piece, x, y);
          g.shadowBlur = 0;
          g.fillStyle = `rgba(${Math.min(255,c[0]+30)},${Math.min(255,c[1]+30)},${Math.min(255,c[2]+30)},${(.3 + .4*wd.s01) * pulse})`;
          g.fillText(wd.piece, x, y);
        });
        g.font = "600 9px " + '"IBM Plex Mono",monospace'; g.textAlign = "left";
        g.fillStyle = `rgba(22,50,74,${isF ? .95 : .6})`;
        g.fillText("L" + layerIds[i] + (i === 0 && layerIds[i] === 0 ? " IN" : i === NP-1 ? " OUT" : ""), q.tl[0], q.tl[1] - 8);
      }
      if(!REDUCED) raf = requestAnimationFrame(frame);
    }
    frame(REDUCED ? 0 : performance.now());
    return () => cancelAnimationFrame(raf);
  }, [rot, zoom, focus, meta, rec.id]);

  return html`<div class="mod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led lilac"></span><span class="cap">state scope</span>
      <span class="tail">${meta.lens && meta.real ? "norms + dispositions · the lens's fitted depths"
        : meta.lens ? "dispositions live · norms sample"
        : meta.real ? "live · depth × position" : "sample pattern"}</span>
      <span class=${"tag " + (meta.real ? "cap-t" : meta.lens ? "der-t" : "smp-t")}>${
        meta.real && meta.lens ? "LIVE" : meta.real ? "CAPTURED" : meta.lens ? "DERIVED" : "SAMPLE"}</span></div>
    <div class="scope-body">
      <span class="grace">you're an angel!</span>
      <canvas ref=${canvasRef} height="230"></canvas>
    </div>
    <div class="scope-ctl">
      <span>Rotate</span><input type="range" min="0" max="100" value=${rot}
        onInput=${e => setRot(+e.target.value)}/>
      <span>Zoom</span><input type="range" min="60" max="140" value=${zoom}
        onInput=${e => setZoom(+e.target.value)}/>
      <span>Focus</span><select value=${focus} onChange=${e => setFocus(+e.target.value)}>
        ${meta.layerIds.map((L,i) => html`<option value=${i}>L${L}${i === 0 ? " in" : i === meta.layerIds.length-1 ? " out" : ""}</option>`)}
      </select>
      <button class="spd" style="margin-left:auto"
        onClick=${() => toast("LAYER INSPECTOR — the deep pane view opens in a later phase")}>OPEN LAYER INSPECTOR</button>
    </div>
    <div class="scope-note">${meta.lens && meta.real
      ? "dots = residual norms (one causal forward) · glowing words = J-lens dispositions, read at the lens's own fitted depths — a disposition, not a verified thought"
      : meta.lens
      ? "glowing words = live J-lens dispositions per depth; dot pattern is sample (norms unavailable) — a disposition, not a verified thought"
      : meta.real
      ? "residual norms — one causal forward over this run's prompt (no lens fitted for this model — dots only)"
      : "sample pattern — the live scope reads real /engine/layers norms and J-lens dispositions when the engine is up"}</div>
  </div>`;
}

/* ───────────────────────── right column ───────────────────────── */
function AreaChart({ vals, rgb, cursor = 0 }){
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current; if(!cv) return;
    /* CSS width:100% resolves against meterbody's CONTENT box; parent.clientWidth includes its
       horizontal padding and used to make the canvas leak 13px out of the narrow workbench. */
    const W = Math.max(1, cv.getBoundingClientRect().width || cv.parentElement.clientWidth), H = 70;
    const dpr = Math.min(2, devicePixelRatio || 1);
    cv.width = W*dpr; cv.style.width = W + "px"; cv.height = H*dpr; cv.style.height = H + "px";
    const g = cv.getContext("2d"); g.scale(dpr, dpr);
    g.clearRect(0,0,W,H);
    for(let k = 1; k < 4; k++){ g.strokeStyle = "rgba(90,130,155,.14)";
      g.beginPath(); g.moveTo(0, H*k/4); g.lineTo(W, H*k/4); g.stroke(); }
    if(!vals.length) return;
    const mx = Math.max(1e-6, ...vals);
    const pts = vals.map((v,i) => [i/(vals.length-1 || 1)*W, H - 4 - (v/mx)*(H - 10)]);
    const gr = g.createLinearGradient(0,0,0,H);
    gr.addColorStop(0, `rgba(${rgb},.4)`); gr.addColorStop(1, `rgba(${rgb},.03)`);
    g.beginPath(); g.moveTo(0,H); pts.forEach(p => g.lineTo(p[0],p[1])); g.lineTo(W,H); g.closePath();
    g.fillStyle = gr; g.fill();
    g.beginPath(); pts.forEach((p,i) => i ? g.lineTo(p[0],p[1]) : g.moveTo(p[0],p[1]));
    g.strokeStyle = `rgba(${rgb},.9)`; g.lineWidth = 1.4; g.stroke();
    const ci = Math.max(0, Math.min(pts.length - 1, cursor ? cursor - 1 : 0));
    const e = pts[ci];
    g.strokeStyle = "rgba(242,109,79,.72)"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(e[0], 3); g.lineTo(e[0], H - 2); g.stroke();
    g.fillStyle = `rgba(${rgb},1)`; g.beginPath(); g.arc(e[0], e[1], 3, 0, 7); g.fill();
    g.strokeStyle = "rgba(255,255,255,.95)"; g.lineWidth = 1.5; g.stroke();
  }, [vals, rgb, cursor]);
  return html`<canvas ref=${ref} height="70"></canvas>`;
}
function Meters({ rec }){
  const P = useStore(x => x.P);
  const steps = normSteps(rec);
  const conf = steps.map(s => s.conf).filter(x => x != null);
  const ent = steps.map(s => s.ent).filter(x => x != null);
  const avg = conf.length ? conf.reduce((a,b) => a+b, 0) / conf.length : null;
  const peak = ent.length ? Math.max(...ent) : null;
  return html`<div class="mod">
    <span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led blue"></span><span class="cap">multi-meter</span>
      <span class="tag cap-t">CAPTURED</span></div>
    <div class="meterbody">
      <div class="mcap"><span><i style="background:var(--blue)"></i>confidence</span><b>${avg == null ? "—" : avg.toFixed(2)} AVG</b></div>
      <${AreaChart} vals=${steps.map(s => s.conf ?? 0)} rgb="44,191,232" cursor=${P}/>
      <div class="mcap" style="margin-top:8px"><span><i style="background:var(--lilac2)"></i>entropy</span>
        <b>${peak == null ? "NOT RECORDED" : peak.toFixed(2) + " PEAK"}</b><span class=${"tag " + (peak == null ? "smp-t" : "der-t")}>${peak == null ? "UNAVAILABLE" : "DERIVED"}</span></div>
      ${peak == null ? html`<div class="meter-unavailable">This run did not capture entropy; no zero line is implied.</div>`
        : html`<${AreaChart} vals=${steps.map(s => s.ent ?? 0)} rgb="154,146,200" cursor=${P}/>`}
    </div>
  </div>`;
}

/* Free explanation assembly: reads only signals already captured on the run. It deliberately does not
   generate a story about the model's hidden process; the separate self-report check below tests that. */
function ExplainPanel({ rec }){
  const [out, setOut] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const request = useRef(0);
  useEffect(() => { request.current += 1; setOut(null); setBusy(false); setError(""); }, [rec.id]);

  const assemble = async () => {
    if(guardSample(rec) || busy) return;
    const nonce = ++request.current;
    setBusy(true); setError("");
    const r = await api.explain(rec.id);
    if(nonce !== request.current) return;       /* ignore an old run's answer after a fast run switch */
    setBusy(false);
    if(!r){ setError("No explanation came back. The run may no longer be in the journal."); return; }
    setOut(r);
  };
  // null/undefined is MISSING, checked before the +coercion: +null===0 is finite, so without this guard
  // an absent value (e.g. topic gate in internalized mode) would fabricate "0.00" -- a real signal where
  // there is none. A missing readout must read "-", never a made-up zero.
  const num = (v, places = 2) => (v == null ? "-" : Number.isFinite(+v) ? (+v).toFixed(places) : "-");
  const confidence = (out && out.confidence) || {};
  const influences = (out && out.influences_active) || {};
  const forks = (out && out.forks) || {};
  const concepts = (out && out.concepts) || {};
  const cards = influences.cards || [], dials = influences.dials || [], anchored = influences.anchored || [];

  return html`<div class="mod" data-testid="explain-panel">
    <div class="mod-h"><span class="led blue"></span><span class="cap">explain this answer</span>
      <span class="tail">${busy ? "assembling..." : out ? "recorded signals" : "zero generation · on demand"}</span></div>
    <div class="explain-body">
      ${!out && !busy && html`
        <span class="none">Assemble what the run actually recorded: per-token hesitation, active influences,
          close-call forks, and concept readouts when present. This is journal math, not a model-written story.</span>
        <button class="spd" style="align-self:start" disabled=${!!rec._sample} onClick=${assemble}>
          ASSEMBLE EXPLANATION</button>
        ${rec._sample && html`<span class="none">Live recorded runs only - sample reels have no journal record.</span>`}`}
      ${busy && html`<span class="none">reading the recorded run - no model or GPU work...</span>`}
      ${error && html`<span class="explain-error">${error}</span>`}
      ${out && html`
        <div class="explain-grid">
          <section class="explain-card" data-testid="explain-confidence">
            <div class="explain-title"><b>confidence</b><span class="tag cap-t">CAPTURED</span></div>
            <div class="explain-rule">Measured per token - never an overall score.</div>
            ${confidence.available === false
              ? html`<div class="none">${confidence.note || "No per-token trace was recorded."}</div>`
              : html`
                <div class="explain-summary">${confidence.summary || "0 hesitations"}
                  <span>${confidence.n_tokens || 0} tokens · below ${num(confidence.threshold)}</span></div>
                ${(confidence.uncertain_moments || []).length
                  ? (confidence.uncertain_moments || []).map(m => html`<div class="explain-row">
                      <span><b>#${m.index}</b> ${String(m.token || "∅")}</span>
                      <span>p ${num(m.confidence, 3)}</span>
                      ${(m.alternatives || []).length && html`<small>alternatives · ${(m.alternatives || [])
                        .slice(0, 3).map(a => `${String(a.piece || "∅").trim()} ${num(a.prob, 3)}`).join(" · ")}</small>`}
                    </div>`)
                  : html`<div class="none">No token fell below the recorded threshold.</div>`}`}
          </section>

          <section class="explain-card" data-testid="explain-influences">
            <div class="explain-title"><b>active influences</b><span class="tag der-t">ACTIVE · NOT PROVEN</span></div>
            <div class="explain-rule">Present on this turn does not mean causally responsible.</div>
            ${(influences.mode != null || influences.gate != null) && html`<div class="explain-summary">
              ${influences.mode || "memory"}<span>topic gate ${num(influences.gate)}</span></div>`}
            ${cards.map(c => html`<div class="explain-row">
              <span><b>card</b> ${String(c.text || c.id || "untitled").slice(0, 80)}</span><span>unproven</span>
              ${c.quoted_span && html`<small>provenance quote · “${String(c.quoted_span).slice(0, 110)}”</small>`}
              ${c.note && html`<small>${String(c.note)}</small>`}
            </div>`)}
            ${anchored.map(a => html`<div class="explain-row">
              <span><b>anchored</b> ${String(a.card_id || a.text || "memory bag").slice(0, 70)}</span>
              <span>${a.gate != null ? "gate " + num(a.gate) : "active"}</span>
            </div>`)}
            ${dials.map(d => html`<div class="explain-row">
              <span><b>dial</b> ${d.name}</span><span>${num(d.value)} · unproven</span>
            </div>`)}
            ${!cards.length && !anchored.length && !dials.length && html`<div class="none">
              ${influences.note || "No memory or dials were logged as active."}</div>`}
          </section>

          <section class="explain-card" data-testid="explain-forks">
            <div class="explain-title"><b>close-call forks</b><span class="tag der-t">DERIVED</span></div>
            <div class="explain-rule">A correlational locator for a useful branch test, never a fragility verdict.</div>
            ${forks.available === false
              ? html`<div class="none">${forks.note || "No token alternatives were recorded."}</div>`
              : html`
                <div class="explain-summary">${forks.summary || "No close calls"}
                  <span>${forks.meaningful_count || 0} meaning-changing</span></div>
                ${(forks.forks || []).map(f => html`<div class="explain-row">
                  <span><b>#${f.index}</b> “${f.top}” vs “${f.alt}”</span>
                  <span>${num(f.top_prob, 3)} / ${num(f.alt_prob, 3)}</span>
                  <small>emitted “${f.emitted}” · margin ${num(f.margin, 3)}${f.meaningful ? " · meaning-changing" : ""}</small>
                </div>`)}`}
          </section>

          <section class="explain-card" data-testid="explain-concepts">
            <div class="explain-title"><b>concept readouts</b><span class="tag der-t">RECORDED IF PRESENT</span></div>
            <div class="explain-rule">Feature labels describe activations, not a verified chain of thought.</div>
            ${concepts.available === false
              ? html`<div class="none">${concepts.note || "No concept readouts were recorded."}</div>`
              : (concepts.spans || []).map(s => html`<div class="explain-row">
                  <span><b>${s.position != null ? "#" + s.position : "span"}</b> ${String(s.piece || "")}</span>
                  <small>${(s.features || []).map(f => `${f.label || ("sae:" + (f.id ?? "?"))} ${num(f.score, 3)}`).join(" · ")}</small>
                </div>`)}
          </section>
        </div>
        <div class="explain-foot">Assembled from this recorded run only · zero generation · unavailable signals stay unavailable.</div>`}
    </div>
  </div>`;
}

/* influence descriptor -> a human name, resolving card ids through the run's own memory manifest */
function influenceName(inf, rec){
  if(!inf) return "influence";
  if(inf.card_id){
    const mem = rec.memory || {};
    const ids = mem.applied_ids || [], texts = mem.cards_applied || [];
    const i = ids.indexOf(inf.card_id);
    return i >= 0 && texts[i] ? `card · ${String(texts[i]).slice(0, 40)}` : `card ${inf.card_id}`;
  }
  if(inf.dial) return `dial · ${inf.dial}`;
  if(inf.memory_off) return "memory (whole block)";
  if(inf.behavior_off) return "behavior (all dials)";
  return JSON.stringify(inf).slice(0, 40);
}

function ReceiptsPanel({ rec }){
  const receipts = useStore(x => x.receipts);
  const busy = useStore(x => x.busy.prove);
  return html`<div class="mod">
    <div class="mod-h"><span class="led"></span><span class="cap">receipts</span>
      <span class="tail">${busy ? "proving…" : receipts ? "measured · mode both" : "on demand"}</span></div>
    <div class="receipts-box">
      ${!receipts && !busy && html`<div class="none" style="padding:6px 0 10px">
        Receipts are computed on demand, never pre-attached — press PROVE to run leave-one-out
        (both-arms-greedy) plus forced scoring over this run's memory and dials.</div>`}
      ${busy && html`<div class="none" style="padding:6px 0 10px">regenerating arms greedy — this takes a while, honestly…</div>`}
      ${receipts && (receipts.list.length ? receipts.list.map(r => {
        const f = receipts.forcedBy.get(receipts.keyFn(r.influence));
        const changed = !!r.has_effect;
        /* silent influence — the server's own formula (contracts §6): text unchanged but the
           forced dependence clearly beats its matched null floor */
        const silent = !changed && f && f.null_floor && f.null_floor.exceeds_floor_by_order_of_magnitude;
        return html`<div class="receipt-row">
          <span>${influenceName(r.influence, rec)}
            ${silent && html` <span class="tag" style="color:#B0509E;background:rgba(228,140,216,.14);border:1px solid rgba(228,140,216,.55)">SILENT INFLUENCE</span>`}</span>
          <span class="nats">${f && f.sum_nats != null ? (+f.sum_nats).toFixed(2) + " nats" : "—"}</span>
          <span class="sub">
            <span class=${changed ? "yes" : "no"}>changed the answer · ${changed ? "yes" : "no"}</span>
            <span class=${r.causal_verified ? "yes" : "no"}>causal_verified · ${String(r.causal_verified)}</span>
            ${r.delta && r.delta.changed != null && html`<span>word-shift ${r.delta.changed}%</span>`}
            ${f && f.mean_nats_per_token != null && html`<span>forced Δ ${(+f.mean_nats_per_token).toFixed(3)} nats/tok</span>`}
            ${(r.ablation_note || (f && !f.causal_verified && f.note)) &&
              html`<span>${String(r.ablation_note || f.note).slice(0, 90)}</span>`}
          </span>
        </div>`; })
      : html`<div class="none" style="padding:6px 0 10px">no fired influences to measure on this run
          (no memory or dials rode it).</div>`)}
      ${receipts && receipts.skipped.length ? html`<div class="none" style="padding:4px 0 0">
        skipped: ${receipts.skipped.map(s => (s.reason || "")).join(" · ").slice(0, 140)}</div>` : null}
      ${receipts && receipts.redundant.length ? receipts.redundant.map(p => html`
        <div class="none" style="padding:4px 0 0">redundant pair: ${(p.redundant || []).join(" + ")} — ${p.note || ""}</div>`) : null}
      ${receipts && html`<div class="none" style="padding:6px 0 2px;font-size:8px">
        forced Δ measures dependence — a nonzero delta does NOT mean the answer would have differed.
        Pairwise redundancy guard only, not the full power set.</div>`}
    </div>
  </div>`;
}

/* Common complaint -> one capped positive-pole dial move. The two-arm counterfactual is deliberately
   used instead of comparing the stored sampled reply with a greedy replay (which would mix two changes). */
const QUICK_REPAIRS = [
  { key: "verbose", label: "Too verbose", axis: "concise", title: "Move toward concise" },
  { key: "vague", label: "Too vague", axis: "concrete", title: "Move toward concrete" },
  { key: "agreeable", label: "Too agreeable", axis: "candid", title: "Move toward candid" },
  { key: "cold", label: "Too cold", axis: "warm", title: "Move toward warm" },
];
/* per-axis caps come from the server (/steer/axes' own "max", clozn/behavior/steering/axes.py's AXES) --
   never a hand-maintained duplicate here, which would silently go stale the day a cap is re-tuned.
   1.5 mirrors only /steer/axes' OWN documented default for an axis with no explicit "max", used purely
   as the pre-fetch placeholder (see QuickRepair's axisMax state below), not a second source of truth. */
const QUICK_AXIS_FALLBACK_MAX = 1.5;
const quickValue = (rec, preset, axisMax) => {
  const current = +((((rec || {}).behavior || {}).active_dials || {})[preset.axis] || 0);
  const cap = (axisMax && axisMax[preset.axis] != null) ? axisMax[preset.axis] : QUICK_AXIS_FALLBACK_MAX;
  return Math.min(cap, current + 0.5);
};

function QuickRepair({ rec }){
  const [busy, setBusy] = useState("");
  const [saving, setSaving] = useState(false);
  const [picked, setPicked] = useState(null);
  const [result, setResult] = useState(null);
  const [message, setMessage] = useState(null);
  const [axisMax, setAxisMax] = useState(null);   // {name: max}, read from /steer/axes -- never hardcoded
  const request = useRef(0);
  useEffect(() => {
    request.current += 1; setBusy(""); setSaving(false); setPicked(null); setResult(null); setMessage(null);
  }, [rec.id]);
  useEffect(() => {
    let dead = false;
    (async () => {
      const r = await api.steerAxes();
      if(dead || !r || !Array.isArray(r.axes)) return;
      const m = {};
      r.axes.forEach(a => { if(a && a.name) m[a.name] = a.max != null ? +a.max : QUICK_AXIS_FALLBACK_MAX; });
      setAxisMax(m);
    })();
    return () => { dead = true; };
  }, []);

  const run = async preset => {
    if(guardSample(rec) || busy) return;
    const target = quickValue(rec, preset, axisMax), nonce = ++request.current;
    setBusy(preset.key); setPicked({ ...preset, target }); setResult(null); setMessage(null);
    /* Preference capture is best-effort and must never delay the actual comparison. It changes no dial. */
    api.feedbackRecord({ run_id: rec.id, kind: "quick_repair", dial: preset.axis,
      direction: 1, meta: { complaint: preset.key } });
    toast(`${preset.label.toLowerCase()} - comparing ${preset.axis} at ${target.toFixed(2)} in two greedy arms...`);
    const r = await api.counterfactual(rec.id, { [preset.axis]: target });
    if(nonce !== request.current) return;
    setBusy("");
    if(!r){ setMessage({ kind: "error", text: "Comparison did not run. Quick repair needs a ready model worker and two generation passes." }); return; }
    setResult(r);
  };

  /* The server's own recorded outcome for `preset.axis` -- read back from counterfactual()'s
     `applied_dials` (clozn/replay/counterfactual.py), never the client's pre-clamp guess. `picked.target`
     is only ever shown as "asked for"; anything a user might act on or trust as "what happened" must
     come from here. Returns null (not a guess) when an older/mocked response lacks the field, so the
     caller can fall back to `picked.target` explicitly rather than silently pretend it's confirmed. */
  const appliedValue = (res, preset) => {
    const v = res && res.applied_dials ? res.applied_dials[preset.axis] : null;
    return v != null ? +v : null;
  };

  const save = async () => {
    if(!picked || !result || saving) return;
    const nonce = ++request.current;
    const applied = appliedValue(result, picked);
    const value = applied != null ? applied : picked.target;   // save what was actually tested, not the guess
    setSaving(true);
    setMessage({ kind: "info", text: `Saving ${picked.axis} ${value.toFixed(2)} as the live default...` });
    const r = await api.steerSet(picked.axis, value);
    if(nonce !== request.current) return;
    setSaving(false);
    /* /steer/set's own response ({active: steer.active()} -- clozn/server/substrates.py) is the most
       authoritative readback of all: what the live substrate actually holds right now. Prefer it over
       `value` when present. */
    const confirmed = (r && r.active && r.active[picked.axis] != null) ? +r.active[picked.axis] : null;
    setMessage(r ? { kind: "ok", text: `Saved ${picked.axis} ${(confirmed != null ? confirmed : value).toFixed(2)} as the default.` }
                 : { kind: "error", text: "The comparison remains recorded, but the default was not saved." });
  };

  const dials = ((rec.behavior || {}).active_dials) || {};
  const coherence = (result && result.coherence) || {};
  const saveable = !!(result && result.causal_verified === true && result.has_effect === true && !coherence.degenerate);
  const delta = (result && result.delta) || {};
  const appliedNow = (result && picked) ? appliedValue(result, picked) : null;
  return html`<div class="mod" data-testid="quick-repair">
    <div class="mod-h"><span class="led" style="background:var(--coral);box-shadow:0 0 8px var(--coral)"></span>
      <span class="cap">quick repair</span><span class="tail">${busy ? "running 2 greedy arms..." : "one complaint · one dial"}</span></div>
    <div class="quick-repair-body">
      <div class="quick-repair-rule">Each preset moves one dial +0.5 from this run's recorded value (capped),
        then compares two matched greedy generations. Nothing becomes a default unless you explicitly save it.</div>
      <div class="quick-repair-presets">
        ${QUICK_REPAIRS.map(p => {
          const cur = +(dials[p.axis] || 0), target = quickValue(rec, p, axisMax), capped = target <= cur;
          return html`<button class="quick-repair-btn" data-repair=${p.key}
            title=${capped ? `${p.axis} is already at its safe cap` : `${p.title}: ${cur.toFixed(2)} -> ${target.toFixed(2)}`}
            disabled=${!!busy || saving || !!rec._sample || capped} onClick=${() => run(p)}>
            <b>${p.label}</b><span>${p.axis} ${cur.toFixed(2)} -> ${target.toFixed(2)}${capped ? " · AT CAP" : ""}</span>
          </button>`;
        })}
      </div>
      ${busy && html`<div class="none">Generating a matched greedy baseline and repair candidate - two passes, no persistence.</div>`}
      ${message && html`<div class=${"quick-repair-message " + message.kind}>${message.text}</div>`}
      ${result && html`<div class="quick-repair-result" data-testid="quick-repair-result">
        <div class="quick-repair-verdict">
          <span class=${"tag " + (result.has_effect ? "cap-t" : "smp-t")}>answer changed · ${String(result.has_effect)}</span>
          <span class=${"tag " + (result.causal_verified ? "cap-t" : "fail-t")}>override applied · ${String(result.causal_verified)}</span>
          ${coherence.degenerate && html`<span class="tag fail-t">DEGENERATE</span>`}
        </div>
        <div class="quick-repair-compare">
          <div><b>matched greedy baseline · live dials</b><p>${result.baseline_reply ?? "-"}</p></div>
          <div><b>repair candidate · ${picked.axis} ${(appliedNow != null ? appliedNow : picked.target).toFixed(2)}</b>
            <span class="quick-repair-note" style="display:block;margin:2px 0 0">asked for ${picked.target.toFixed(2)}${
              appliedNow != null ? " · the server has final say on the axis cap" : " · unconfirmed (older response shape)"}</span>
            <p>${result.counterfactual_reply ?? "-"}</p></div>
        </div>
        <div class="quick-repair-delta">changed ${delta.changed != null ? delta.changed + "%" : "-"}
          · words ${Array.isArray(delta.words) ? delta.words.join(" -> ") : "-"}
          ${result.override_note ? " · " + result.override_note : ""}</div>
        <div class="quick-repair-actions">
          <button class="spd primary" disabled=${!saveable || saving} onClick=${save}>
            ${saving ? "SAVING..." : "SAVE AS DEFAULT"}</button>
          <span>${saveable ? "This candidate changed the answer, the override was applied, and output stayed coherent."
            : "Save unlocks only for an applied, answer-changing, non-degenerate candidate."}</span>
        </div>
        <div class="quick-repair-note">The stored sampled reply is context only, not a subtraction term.
          The preference click is recorded even when the comparison fails; recording it never changes a dial.</div>
      </div>`}
    </div>
  </div>`;
}

function Steer({ rec }){
  const dials = Object.entries((rec.behavior || {}).active_dials || {}).filter(([,v]) => v);
  return html`<div class="mod">
    <div class="mod-h"><span class="led"></span><span class="cap">quick steer</span>
      <span class="tail">this run · read-only</span></div>
    ${dials.length ? dials.map(([k,v]) => html`<div class="steer-row">
        <span>${k}</span><span class="v">${(+v).toFixed(1)}</span>
        <span class="bar"><i style=${"width:" + Math.round(Math.min(1, Math.abs(v))*100) + "%"}></i></span>
      </div>`)
      : html`<div class="steer-row"><span class="none">no dials rode this run</span></div>`}
    <div class="sooncard">
      <span><b>Patch interventions</b><span class="ds">any-concept dials · swap receipts</span></span>
      <span class="stag">soon</span>
    </div>
  </div>`;
}

/* ── F3: lineage subway map — the whole branch family as a clickable tree ── */
function LineageTree({ rec }){
  const [fam, setFam] = useState(null);
  const [open, setOpen] = useState(false);
  useEffect(() => { setFam(null); setOpen(false); }, [rec.id]);
  const load = async () => {
    if(fam || rec._sample || !store.get().live) return;
    const r = await api.family(rec.id);
    setFam((r && Array.isArray(r.runs)) ? r.runs : []);
  };
  if(rec._sample) return null;
  const kids = {};
  (fam || []).forEach(r => { if(r.parent_run_id) (kids[r.parent_run_id] = kids[r.parent_run_id] || []).push(r); });
  const byId = Object.fromEntries((fam || []).map(r => [r.id, r]));
  const roots = (fam || []).filter(r => !r.parent_run_id || !byId[r.parent_run_id]);
  const verb = r => (r.changes_applied && Object.keys(r.changes_applied)[0]) || r.source || "";
  const Node = ({ r, depth }) => html`
    <div style=${"padding-left:" + (depth * 18) + "px;display:flex;align-items:center;gap:6px"}>
      <span style="color:var(--mist);font-size:9px">${depth ? "└⑂" : "●"}</span>
      <button class="mono" style=${"font-size:9px;padding:1px 5px;border-radius:4px;cursor:pointer;"
          + (r.id === rec.id ? "color:var(--navy);background:rgba(182,176,218,.25);font-weight:600" : "color:#4A5878")}
        onClick=${() => loadRun(r.id)}>${r.id}</button>
      <span style="font-size:8.5px;color:var(--mist)">${verb(r)} · ${shortTime(r.created_at)}</span>
    </div>
    ${(kids[r.id] || []).map(c => html`<${Node} r=${c} depth=${depth + 1}/>`)}`;
  return html`<details class="leader" onToggle=${e => { setOpen(e.target.open); if(e.target.open) load(); }}>
    <summary>Lineage — every branch/replay/fork of this run, as a tree
      <span style="margin-left:auto">${fam ? fam.length + " run(s)" : "on demand"}</span></summary>
    <div style="padding:8px 12px 10px;display:flex;flex-direction:column;gap:3px">
      ${!fam && open ? html`<span class="none">reading the family…</span>`
        : fam && !fam.length ? html`<span class="none">this run has no recorded relatives</span>`
        : fam ? roots.map(r => html`<${Node} r=${r} depth=${0}/>`) : null}
    </div>
  </details>`;
}

/* Context <-> answer influence map. The answer is held fixed while each bounded context span is
   replaced with a matched neutral control. This surface renders those behavioral-dependence scores;
   it deliberately does not relabel them as attention, internal mediation, or a circuit explanation. */
const influenceArray = (...values) => values.find(Array.isArray) || [];
const influenceNumber = value => value != null && Number.isFinite(+value) ? +value : null;
const influenceId = (value, fallback) => value == null ? fallback : String(value);

function influenceArtifact(value){
  if(!value || typeof value !== "object") return null;
  const candidates = [
    value.influence_map, value.context_answer_influence, value.artifact, value.result, value.receipt,
    value.data, value.receipts && value.receipts.influence_map,
    value.receipts && value.receipts.context_answer_influence, value,
  ];
  return candidates.find(candidate => candidate && typeof candidate === "object" && (
    candidate.schema === "clozn.context_answer_influence.v1" ||
    ((Array.isArray(candidate.prompt_spans) || Array.isArray(candidate.context_spans)) &&
      (Array.isArray(candidate.answer_spans) || Array.isArray(candidate.response_spans))) ||
    (candidate.error && (candidate.status === "unavailable" || candidate.status === "error"))
  )) || null;
}

function normalizeInfluenceMap(raw){
  const artifact = influenceArtifact(raw);
  if(!artifact) return null;
  const promptRaw = influenceArray(
    artifact.prompt_spans, artifact.context_spans, artifact.prompt && artifact.prompt.spans,
    artifact.context && artifact.context.spans,
  );
  const sourceFallback = !promptRaw.length
    ? influenceArray(artifact.prompt_sources, artifact.sources).filter(source => source && source.selected !== false)
    : [];
  const answerRaw = influenceArray(
    artifact.answer_spans, artifact.response_spans, artifact.answer && artifact.answer.spans,
    artifact.response && artifact.response.spans,
  );
  const fallbackAnswer = !answerRaw.length && artifact.answer &&
    (artifact.answer.scored_text ?? artifact.answer.text ?? artifact.answer.recorded_text);
  const normalizeSpan = (span, index, prefix) => {
    const item = span && typeof span === "object" ? span : { text: span };
    return {
      ...item,
      id: influenceId(item.id ?? item.span_id ?? item[`${prefix}_span_id`], `${prefix}.${index}`),
      text: String(item.text ?? item.piece ?? item.content ?? item.value ?? ""),
      index,
    };
  };
  const prompts = (promptRaw.length ? promptRaw : sourceFallback)
    .map((span, index) => normalizeSpan(span, index, "context"));
  const answers = (answerRaw.length ? answerRaw : fallbackAnswer != null ? [{ text: fallbackAnswer }] : [])
    .map((span, index) => normalizeSpan(span, index, "answer"));
  const thresholdValue = (artifact.thresholds &&
    (artifact.thresholds.cell_abs_delta_nats ?? artifact.thresholds.abs_delta_nats));
  const floor = influenceNumber(thresholdValue ?? artifact.evidence_floor ?? artifact.floor);
  const promptById = new Map(prompts.map(span => [span.id, span]));
  const answerById = new Map(answers.map(span => [span.id, span]));
  const linksByPair = new Map();
  const addLink = (link, contextIndex = null, answerIndex = null) => {
    const item = link && typeof link === "object" ? link : { delta_nats: link };
    const cIndex = influenceNumber(item.context_index ?? item.prompt_index ?? contextIndex);
    const aIndex = influenceNumber(item.answer_index ?? item.response_index ?? answerIndex);
    const contextId = influenceId(
      item.context_span_id ?? item.prompt_span_id ?? item.source_span_id,
      cIndex != null && prompts[cIndex] ? prompts[cIndex].id : null,
    );
    const answerId = influenceId(
      item.answer_span_id ?? item.response_span_id ?? item.target_span_id,
      aIndex != null && answers[aIndex] ? answers[aIndex].id : null,
    );
    if(!promptById.has(contextId) || !answerById.has(answerId)) return;
    const delta = influenceNumber(item.delta_nats ?? item.signed_delta_nats ?? item.delta ?? item.score ?? item.value);
    const magnitude = influenceNumber(item.abs_delta_nats ?? item.absolute_delta_nats ?? item.magnitude ?? item.weight)
      ?? (delta == null ? 0 : Math.abs(delta));
    const explicitClear = item.clears_floor ?? item.above_floor ?? item.clear ?? item.significant;
    const clearsFloor = explicitClear != null ? explicitClear === true
      : floor != null ? magnitude >= floor : magnitude > 0;
    const effect = String(item.effect || (delta > 0 ? "supports" : delta < 0 ? "suppresses" : "neutral"));
    const normalized = { contextId, answerId, delta, magnitude, clearsFloor, effect };
    linksByPair.set(`${contextId}\u0000${answerId}`, normalized);
  };
  influenceArray(artifact.links, artifact.edges, artifact.relationships).forEach(link => addLink(link));
  influenceArray(artifact.matrix, artifact.score_matrix, artifact.scores).forEach((row, contextIndex) => {
    if(!Array.isArray(row)) return;
    row.forEach((cell, answerIndex) => {
      const context = prompts[contextIndex], answer = answers[answerIndex];
      if(context && answer && !linksByPair.has(`${context.id}\u0000${answer.id}`))
        addLink(cell, contextIndex, answerIndex);
    });
  });
  const links = [...linksByPair.values()];
  const noSourceIds = new Set(influenceArray(
    artifact.summary && artifact.summary.answer_span_ids_without_clear_source,
    artifact.answer_span_ids_without_clear_source,
  ).map(String));
  answers.forEach(answer => {
    if(!links.some(link => link.answerId === answer.id && link.clearsFloor)) noSourceIds.add(answer.id);
  });
  return {
    artifact, prompts, answers, links, floor, noSourceIds,
    available: artifact.available !== false && artifact.status !== "unavailable" && artifact.status !== "error",
    message: String((artifact.error && (artifact.error.message || artifact.error.code)) || artifact.message || ""),
  };
}

function InfluenceMap({ rec }){
  const [out, setOut] = useState(() => normalizeInfluenceMap(rec));
  const [busy, setBusyLocal] = useState(false);
  const [error, setError] = useState("");
  const [hovered, setHovered] = useState(null);
  const [focused, setFocused] = useState(null);
  const [pinned, setPinned] = useState(null);
  const request = useRef(0);
  useEffect(() => {
    request.current += 1;
    setOut(normalizeInfluenceMap(rec)); setBusyLocal(false); setError("");
    setHovered(null); setFocused(null); setPinned(null);
  }, [rec.id]);

  const run = async (refresh = false) => {
    if(guardSample(rec) || busy) return;
    const nonce = ++request.current;
    setBusyLocal(true); setError(""); setHovered(null); setFocused(null); setPinned(null);
    toast("mapping context to the recorded answer - forced scoring only, no generation...");
    const response = await api.influenceMap(rec.id, refresh ? { refresh: true } : {});
    if(nonce !== request.current) return;
    setBusyLocal(false);
    if(!response){ setError("No influence map came back. This measurement needs a ready scoring worker."); return; }
    const mapped = normalizeInfluenceMap(response);
    if(!mapped){
      setError(response.error || `The influence-map route returned an unreadable artifact${response.__status ? ` (${response.__status})` : ""}.`);
      return;
    }
    setOut(mapped);
    if(!mapped.available) setError(mapped.message || "This run does not contain enough recorded context and answer evidence to map.");
  };

  // A click is a real pin: pointer movement and incidental focus changes must not replace it until
  // the user clicks that span again (or pins another one). Hover/focus drive the transient state only.
  const active = pinned || hovered || focused;
  const activeLinks = !active || !out ? [] : out.links
    .filter(link => active.kind === "context" ? link.contextId === active.id : link.answerId === active.id)
    .sort((a, b) => b.magnitude - a.magnitude || a.contextId.localeCompare(b.contextId) || a.answerId.localeCompare(b.answerId));
  const clearLinks = activeLinks.filter(link => link.clearsFloor);
  const shownLinks = clearLinks.length ? clearLinks : activeLinks.slice(0, active && active.kind === "answer" ? 3 : 5);
  const strongest = shownLinks[0] || null;
  const maxMagnitude = strongest ? strongest.magnitude || 1 : 1;
  const linkFor = (kind, id) => shownLinks.find(link => kind === "context" ? link.contextId === id : link.answerId === id);
  const spanState = (kind, id) => {
    const selected = !!active && active.kind === kind && active.id === id;
    const link = !selected && active && active.kind !== kind ? linkFor(kind, id) : null;
    return {
      selected,
      link,
      strength: selected ? 1 : link ? Math.max(.12, link.magnitude / maxMagnitude) : 0,
      strongest: !!link && link === strongest,
    };
  };
  const sourceLabel = span => [span.role, span.name, span.source_kind]
    .filter(Boolean).map(value => String(value).replaceAll("_", " ")).filter((value, index, all) => all.indexOf(value) === index).join(" · ") || "recorded context";
  const activeStatus = () => {
    if(!active) return "Hover, focus, or click a span on either side to reveal its strongest measured links.";
    if(!activeLinks.length) return "No measured relationship is available for this span.";
    if(!clearLinks.length) return active.kind === "answer"
      ? "No context span clears the evidence floor for this answer span; the strongest below-floor measurements remain subdued."
      : "No answer span clears the evidence floor for this context span; the strongest below-floor measurements remain subdued.";
    const noun = active.kind === "answer" ? "context span" : "answer span";
    const amount = strongest && strongest.magnitude != null ? ` Strongest measured change: ${strongest.magnitude.toFixed(3)} nats.` : "";
    return `${clearLinks.length} ${noun}${clearLinks.length === 1 ? "" : "s"} clear the evidence floor.${amount}`;
  };
  const togglePin = (kind, id) => setPinned(current =>
    current && current.kind === kind && current.id === id ? null : { kind, id });

  const artifact = out && out.artifact;
  const empty = out && out.available && (!out.prompts.length || !out.answers.length);
  return html`<div class="mod influence-map" data-testid="influence-map">
    <div class="mod-h"><span class="led lilac"></span><span class="cap">context ↔ answer influence</span>
      <span class="tail">${busy ? "scoring matched controls..." : out && out.available ? "hover · focus · pin · trace both ways" : "on demand · no generation"}</span></div>
    <div class="influence-map-body">
      <div class="influence-map-rule">Hold the recorded answer fixed, replace one context span at a time,
        and measure how its token likelihood moves. This is controlled behavioral dependence, not an attention path or circuit explanation.</div>
      ${!out && !busy && html`<div class="influence-map-empty">
        <span class="none">Build a bounded map from the exact assembled context to this recorded answer.</span>
        <button class="spd" disabled=${!!rec._sample} onClick=${() => run(false)}>MAP CONTEXT TO ANSWER</button>
        ${rec._sample && html`<span class="none">Live recorded runs only - sample reels have no scorable continuation.</span>`}
      </div>`}
      ${busy && html`<div class="influence-map-empty"><span class="none">Scoring the recorded continuation once, then one matched control per context span...</span></div>`}
      ${(error || (out && !out.available)) && !busy && html`<div class="influence-map-message" role="status">
        <span>${error || out.message || "Influence evidence is unavailable for this run."}</span>
        <button class="spd" disabled=${!!rec._sample} onClick=${() => run(false)}>TRY AGAIN</button>
      </div>`}
      ${empty && html`<div class="influence-map-message" role="status">The map returned without displayable span coordinates.</div>`}
      ${out && out.available && !empty && html`
        <div class="influence-map-grid">
          <section class="influence-map-pane" aria-labelledby="influence-context-title">
            <div class="influence-map-pane-head"><b id="influence-context-title">recorded context</b><span>${out.prompts.length} measured spans</span></div>
            <div class="influence-context-list">
              ${out.prompts.map((span, index) => {
                const state = spanState("context", span.id);
                return html`<button type="button" class="influence-span influence-context-span"
                  data-span-id=${span.id} data-active=${String(state.selected)} data-linked=${String(!!state.link)}
                  data-clears-floor=${String(!!(state.link && state.link.clearsFloor))}
                  data-effect=${state.link ? state.link.effect : "neutral"} data-strongest=${String(state.strongest)}
                  style=${`--influence-strength:${state.strength};--influence-alpha:${(.08 + state.strength * .48).toFixed(3)}`}
                  aria-pressed=${!!pinned && pinned.kind === "context" && pinned.id === span.id} aria-describedby="influence-map-status"
                  aria-label=${`Context span ${index + 1}, ${sourceLabel(span)}: ${span.text}`}
                  onClick=${() => togglePin("context", span.id)}
                  onMouseEnter=${() => setHovered({ kind: "context", id: span.id })} onMouseLeave=${() => setHovered(null)}
                  onFocus=${() => setFocused({ kind: "context", id: span.id })} onBlur=${() => setFocused(null)}>
                  <span class="influence-source">${sourceLabel(span)}</span><span>${span.text || "[empty span]"}</span>
                </button>`;
              })}
            </div>
          </section>
          <section class="influence-map-pane" aria-labelledby="influence-answer-title">
            <div class="influence-map-pane-head"><b id="influence-answer-title">recorded answer</b><span>${out.answers.length} scored spans</span></div>
            <div class="influence-answer-text">
              ${out.answers.map((span, index) => {
                const state = spanState("answer", span.id), noSource = out.noSourceIds.has(span.id);
                return html`<button type="button" class="influence-span influence-answer-span"
                  data-span-id=${span.id} data-active=${String(state.selected)} data-linked=${String(!!state.link)}
                  data-clears-floor=${String(!!(state.link && state.link.clearsFloor))}
                  data-effect=${state.link ? state.link.effect : "neutral"} data-strongest=${String(state.strongest)}
                  data-no-clear-source=${String(noSource)}
                  style=${`--influence-strength:${state.strength};--influence-alpha:${(.08 + state.strength * .48).toFixed(3)}`}
                  aria-pressed=${!!pinned && pinned.kind === "answer" && pinned.id === span.id} aria-describedby="influence-map-status"
                  aria-label=${`Answer span ${index + 1}: ${span.text.trim() || "whitespace"}${noSource ? "; no context span clears the evidence floor" : ""}`}
                  onClick=${() => togglePin("answer", span.id)}
                  onMouseEnter=${() => setHovered({ kind: "answer", id: span.id })} onMouseLeave=${() => setHovered(null)}
                  onFocus=${() => setFocused({ kind: "answer", id: span.id })} onBlur=${() => setFocused(null)}>${span.text || " "}</button>`;
              })}
            </div>
          </section>
        </div>
        <div class="influence-map-status" id="influence-map-status" aria-live="polite">${activeStatus()}</div>
        <div class="influence-map-legend" aria-label="influence legend">
          <span><i class="supports"></i>supports under replacement</span>
          <span><i class="suppresses"></i>suppresses under replacement</span>
          <span><i class="below"></i>below evidence floor</span>
          <span><i class="unclear"></i>no clear source</span>
        </div>
        <div class="influence-map-foot">
          <span>Forced scoring · recorded continuation · ${artifact && artifact.timing && artifact.timing.score_calls != null ? artifact.timing.score_calls + " score calls" : "bounded matched controls"}</span>
          ${artifact && artifact.selection && (artifact.selection.omitted_source_ids || []).length
            ? html`<span>${artifact.selection.omitted_source_ids.length} older source(s) outside the bounded map</span>` : null}
          ${artifact && artifact.selection && artifact.selection.refinement
            && (artifact.selection.refinement.refined_context_span_ids || []).length
            ? html`<span>${artifact.selection.refinement.refined_context_span_ids.length} span(s) auto-refined into finer sub-spans</span>` : null}
          ${artifact && artifact.redundancy_check && artifact.redundancy_check.performed
            ? html`<span>redundant-pair check: one bounded joint control on the two strongest spans</span>` : null}
          <button class="spd" disabled=${busy || !!rec._sample} onClick=${() => run(true)}>RECOMPUTE</button>
        </div>
      `}
    </div>
  </div>`;
}

/* ── F4: span forensics — ablate a phrase from the prompt, attribute the change causally ── */
function SpanForensics({ rec }){
  const [phrase, setPhrase] = useState("");
  const [out, setOut] = useState(null);
  const busy = useStore(x => x.busy.spanReceipt);
  useEffect(() => { setOut(null); setPhrase(""); }, [rec.id]);
  const run = async () => {
    if(guardSample(rec)) return;
    const p = phrase.trim(); if(!p) return;
    if(store.get().busy.spanReceipt) return;
    setBusy("spanReceipt", true); setOut(null);
    toast("span forensics — regenerating without that span + forced-scoring the original…");
    const r = await api.spanReceipt(rec.id, p);
    setBusy("spanReceipt", false);
    if(!r){ setOut({ error: "no answer — is the engine up?" }); return; }
    if(r.__status && r.__status >= 400){ setOut({ error: r.error || ("failed (" + r.__status + ")") }); return; }
    setOut(r);
  };
  const forced = (out && (out.forced || {})) || {};
  const changed = out && (out.answer_changed ?? out.has_effect
    ?? (out.regen && out.regen.has_effect)
    ?? (out.baseline_reply != null && out.ablated_reply != null
        && out.baseline_reply !== out.ablated_reply));
  return html`<div class="mod">
    <div class="mod-h"><span class="led lilac"></span><span class="cap">span forensics</span>
      <span class="tail">${busy ? "ablating…" : "which words in the context did this?"}</span></div>
    <div style="padding:2px 13px 10px;display:flex;flex-direction:column;gap:6px">
      <div style="display:flex;gap:6px">
        <input type="text" placeholder="a phrase from the prompt/context to ablate"
          value=${phrase} disabled=${busy}
          style="flex:1;font-size:10px;padding:4px 7px;border:1px solid rgba(90,130,155,.4);border-radius:5px;background:rgba(255,255,255,.6)"
          onInput=${e => setPhrase(e.target.value)}
          onKeyDown=${e => { if(e.key === "Enter") run(); }}/>
        <button class="spd" disabled=${busy} onClick=${run}>${busy ? "…" : "ABLATE"}</button>
      </div>
      ${out && out.error && html`<span class="none">${out.error}</span>`}
      ${out && !out.error && html`
        <div class="receipt-row">
          <span>“${((out.influence || {}).text || phrase).slice(0, 44)}”
            <span class=${changed ? "tag warn-t" : "tag cap-t"} style=${changed
              ? "color:#C24A31;background:rgba(242,109,79,.10);border:1px solid rgba(242,109,79,.45)" : ""}>
              ${changed ? "CHANGED THE ANSWER" : "no change"}</span></span>
          <span class="nats">${forced.sum_nats != null ? (+forced.sum_nats).toFixed(2) + " nats" : ""}</span>
          <span class="sub">
            ${out.baseline_reply != null && html`<span>with: “${String(out.baseline_reply).slice(0, 60)}”</span>`}
            ${out.ablated_reply != null && html`<span>without: “${String(out.ablated_reply).slice(0, 60)}”</span>`}
            ${forced.mean_nats_per_token != null && html`<span>forced Δ ${(+forced.mean_nats_per_token).toFixed(3)} nats/tok</span>`}
          </span>
        </div>
        <span class="none" style="font-size:8px">ablation-causal: the span was removed and the run
          re-derived — measured, not guessed. Agent transcripts: paste the suspect retrieved sentence.</span>`}
    </div>
  </div>`;
}

/* ── F5: the introspection check — the model's own "why" vs the causal receipts ── */
function LieDetector({ rec }){
  const [out, setOut] = useState(null);
  const busy = useStore(x => x.busy.narrate);
  useEffect(() => setOut(null), [rec.id]);
  const run = async () => {
    if(guardSample(rec)) return;
    if(store.get().busy.narrate) return;
    setBusy("narrate", true); setOut(null);
    toast("asking the model why — then checking its story against the receipts…");
    const r = await api.narrate(rec.id);
    setBusy("narrate", false);
    if(!r){ setOut({ error: "no answer from the server" }); return; }
    if(r.__status && r.__status >= 400){
      setOut({ error: r.__status === 503
        ? "narration needs the qwen (HF) substrate — the pure-engine substrate can't run the constrained self-report. Switch substrates to use the lie detector."
        : (r.error || ("failed (" + r.__status + ")")) });
      return;
    }
    setOut(r);
  };
  const claims = (out && (out.claims || out.narration_claims || [])) || [];
  const nar = out && (out.narration || out.text || out.narrative || null);
  return html`<div class="mod">
    <div class="mod-h"><span class="led" style="background:var(--pink);box-shadow:0 0 8px var(--pink)"></span>
      <span class="cap">self-report check</span>
      <span class="tail">${busy ? "narrating…" : "measure, don't ask — then compare"}</span></div>
    <div style="padding:2px 13px 10px;display:flex;flex-direction:column;gap:6px">
      ${!out && !busy && html`
        <span class="none">Ask the model to explain its own answer, then score that story against the
          causal receipts. X1 measured: content is legible, process is not — expect confabulation, and
          see it caught.</span>
        <button class="spd" style="align-self:start" onClick=${run}>ASK WHY — THEN CHECK IT</button>`}
      ${busy && html`<span class="none">generating the constrained narration…</span>`}
      ${out && out.error && html`<span class="none">${out.error}</span>`}
      ${out && !out.error && html`
        ${nar && html`<div style="font-size:9.5px;color:#2A3252;border-left:2px solid var(--lilac);padding-left:8px">
          ${String(nar).slice(0, 400)}</div>`}
        ${claims.length ? claims.map(c => html`<div class="receipt-row">
          <span>${String(c.claim || c.text || "").slice(0, 52)}</span>
          <span class="sub">
            <span class=${c.supported ?? c.receipt_supported ? "yes" : "no"}>
              receipt-supported · ${String(c.supported ?? c.receipt_supported ?? "?")}</span>
            ${(c.verdict || c.flag) && html`<span>${String(c.verdict || c.flag)}</span>`}
          </span>
        </div>`) : null}
        ${out.note && html`<span class="none" style="font-size:8px">${String(out.note).slice(0, 220)}</span>`}`}
    </div>
  </div>`;
}

function Minfl({ rec }){
  const mem = rec.memory || {};
  const cards = mem.cards_applied || [];
  const anchored = mem.anchored || [];
  return html`<div class="mod">
    <div class="mod-h"><span class="led"></span><span class="cap">memory influence</span>
      <span class="tail">top contributors</span></div>
    <div style="padding-bottom:4px">
      ${cards.length ? cards.map((c,i) => {
        const rel = (mem.relevance || [])[i];
        return html`<div class="steer-row">
          <span style="font-size:9.5px">${String(c).slice(0, 42)}</span>
          <span class="v">${rel != null ? (+rel).toFixed(2) : "—"}</span>
          <span class="bar"><i style=${"width:" + (rel != null ? Math.round(rel*100) : 0) + "%"}></i></span>
        </div>`; })
      : !anchored.length ? html`<div class="steer-row"><span class="none">no memory rode this run</span></div>` : null}
      ${anchored.length ? html`<div class="none" style="padding:8px 14px 2px;text-transform:uppercase;letter-spacing:.1em">
        anchored bags (${anchored.length})</div>` : null}
      ${anchored.map(b => html`<div class="steer-row">
        <span style="font-size:9.5px">${String(b.card_id || "anchored").slice(0, 32)}</span>
        <span class="v">${b.gate != null ? (+b.gate).toFixed(2) : "—"}</span>
        <span class="sub" style="grid-column:1/-1;gap:5px;flex-wrap:wrap">
          ${(b.alpha_top3 || []).map(t => html`<span class="jchip d1" style="font-size:9px">
            ${String(t.token || "").slice(0, 18)} ${t.alpha != null ? ((+t.alpha) >= 0 ? "+" : "") + (+t.alpha).toFixed(2) : ""}</span>`)}
        </span>
      </div>`)}
      ${mem.gate != null && html`<div class="steer-row">
        <span style="font-size:9.5px">topic gate</span>
        <span class="v">${(+mem.gate).toFixed(2)}</span>
        <span class="bar"><i style=${"width:" + Math.round(mem.gate*100) + "%"}></i></span>
      </div>`}
    </div>
  </div>`;
}
