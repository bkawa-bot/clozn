/* heavnOS · Patch — interventions on a live run: any-concept steering dials, swap receipts
   (read the disposition → write a contrasting concept → diff vs baseline AND a random-direction
   null), and per-dial counterfactuals ("what if this dial were X"). Honesty rules carried over:
   nothing computed is implied to pre-exist, failures are reported in the server's own words, no
   mock data, sample-mode (store.get().live === false) is read-only everywhere. */
import { html, useState, useEffect } from "../vendor/preact-standalone.mjs";
import { store, useStore, toast } from "../state.mjs";
import { api } from "../api.mjs";

/* true when live actions should be blocked (server unreachable, or the record is the sample reel) */
function guardLive(rec){
  if(!store.get().live || (rec && rec._sample)){ toast("live server only"); return true; }
  return false;
}
const fmt = v => (typeof v === "number" ? v.toFixed(2) : (v ?? "—"));

/* ───────────────────────── module root ───────────────────────── */
export function PatchModule(){
  /* one shared /steer/axes read — feeds both the dials panel and the counterfactual dial picker */
  const [axesState, setAxesState] = useState({ status: "loading", axes: [] });
  useEffect(() => {
    let dead = false;
    (async () => {
      const res = await api.steerAxes();
      if(dead) return;
      if(res && Array.isArray(res.axes) && res.axes.length)
        setAxesState({ status: "ok", axes: res.axes, ready: res.ready, substrate: res.substrate });
      else if(res && Array.isArray(res.axes))
        setAxesState({ status: "empty", axes: [] });
      else
        setAxesState({ status: "error", axes: [] });
    })();
    return () => { dead = true; };
  }, []);

  return html`<div class="col">
    <${DialsPanel} axesState=${axesState}/>
    <${SwapReceiptPanel}/>
    <${CounterfactualPanel} axesState=${axesState}/>
  </div>`;
}

/* ───────────────────────── A) any-concept dials ───────────────────────── */
function DialsPanel({ axesState }){
  const live = useStore(x => x.live);
  const [values, setValues] = useState({});
  useEffect(() => {
    if(axesState.status === "ok"){
      const v = {};
      axesState.axes.forEach(a => { v[a.name] = typeof a.value === "number" ? a.value : 0; });
      setValues(v);
    }
  }, [axesState.axes]);

  async function commit(axis, val){
    if(guardLive(null)) return;
    const res = await api.steerSet(axis.name, val);
    if(!res || !res.active){
      toast(`steer/set didn't answer for "${axis.name}"`);
      return;
    }
    toast(`${axis.name} → ${val.toFixed(2)} · active: ${JSON.stringify(res.active)}`);
  }

  return html`<div class="mod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led lilac"></span><span class="cap">any-concept dials</span>
      <span class="tail">${axesState.status === "ok" ? axesState.axes.length + " axes" + (axesState.substrate ? " · " + axesState.substrate : "") : ""}</span>
    </div>
    ${axesState.status === "loading" && html`<div class="none" style="padding:8px 14px 12px">reading /steer/axes…</div>`}
    ${axesState.status === "error" && html`<div class="none" style="padding:8px 14px 12px">no dials — /steer/axes didn't answer (is the server up?).</div>`}
    ${axesState.status === "empty" && html`<div class="none" style="padding:8px 14px 12px">no steering axes reported by the server.</div>`}
    ${axesState.status === "ok" && axesState.axes.map(a => {
      const max = typeof a.max === "number" ? a.max : 1;
      const v = values[a.name] ?? (typeof a.value === "number" ? a.value : 0);
      const extra = a.calibrated
        ? ` · calibrated${Array.isArray(a.usable_range) ? " · usable " + a.usable_range.join("–") : ""}${a.derail_point != null ? " · derail " + a.derail_point : ""}`
        : (a.custom ? " · custom" : (a.library ? " · library" : ""));
      return html`<div class="steer-row" key=${a.name}>
        <span>${a.name}<span style="color:var(--mist)"> ${(a.poles || []).join(" ↔ ")}${extra}</span></span>
        <span class="v">${v.toFixed(2)}</span>
        <input style="grid-column:1/-1" type="range" min=${-max} max=${max} step="0.1"
          value=${v} disabled=${!live}
          onInput=${e => setValues(s => ({ ...s, [a.name]: +e.target.value }))}
          onChange=${e => commit(a, +e.target.value)}/>
      </div>`;
    })}
    <div class="none" style="padding:8px 14px 12px">content concepts steer; style words don't (validated: dir(c) names behavior, can't enact style)</div>
  </div>`;
}

/* ───────────────────────── B) swap receipt (flagship) ───────────────────────── */
function SwapReceiptPanel(){
  const rec = useStore(x => x.rec);
  const live = useStore(x => x.live);
  const [toConcept, setToConcept] = useState("");
  const [fromHint, setFromHint] = useState("");
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);

  async function run(){
    if(!rec){ toast("need a current run — load one from Replay first"); return; }
    if(guardLive(rec)) return;
    if(!toConcept.trim()){ toast("need a to_concept to swap in"); return; }
    setBusy(true); setErr(null); setRes(null);
    toast("reading the disposition, writing the swap, diffing against a random-direction null…");
    const body = { to_concept: toConcept.trim() };
    if(fromHint.trim()) body.from_hint = fromHint.trim();
    const r = await api.swapReceipt(rec.id, body);
    setBusy(false);
    if(!r){ setErr("swap_receipt didn't answer — needs the engine substrate (.engine + .jlens); is the engine up?"); return; }
    setRes(r);
  }

  return html`<div class="mod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led"></span><span class="cap">swap receipt</span>
      <span class="tail">${busy ? "swapping…" : res ? "computed" : "on demand"}</span></div>
    <div style="padding:8px 14px 4px;display:flex;gap:10px;flex-wrap:wrap;align-items:end">
      <label style="display:flex;flex-direction:column;gap:3px;font-size:8.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mist)">
        to_concept (required)
        <input value=${toConcept} onInput=${e => setToConcept(e.target.value)} placeholder="e.g. ocean"
          style="font:inherit;font-size:10.5px;padding:5px 8px;border-radius:7px;border:1px solid var(--edge);background:linear-gradient(180deg,#fff,#E6F1F7);color:var(--navy);min-width:150px"/>
      </label>
      <label style="display:flex;flex-direction:column;gap:3px;font-size:8.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mist)">
        from_hint (label only, optional)
        <input value=${fromHint} onInput=${e => setFromHint(e.target.value)} placeholder="e.g. Paris"
          style="font:inherit;font-size:10.5px;padding:5px 8px;border-radius:7px;border:1px solid var(--edge);background:linear-gradient(180deg,#fff,#E6F1F7);color:var(--navy);min-width:150px"/>
      </label>
      <button class=${"spd" + (busy ? " busy" : "")} onClick=${run}>RUN SWAP</button>
    </div>
    ${!rec && html`<div class="none" style="padding:0 14px 12px">need a current run — load one from Replay first.</div>`}
    ${rec && !live && html`<div class="none" style="padding:0 14px 12px">live server only — this is the sample reel.</div>`}
    ${busy && html`<div class="none" style="padding:0 14px 12px">reading the disposition, writing the swap, diffing against a random-direction null…</div>`}
    ${err && html`<div style="padding:0 14px 12px"><span class="tag fail-t">FAILED</span> <span class="none">${err}</span></div>`}
    ${res && html`<${SwapReceiptResult} res=${res}/>`}
  </div>`;
}

function SwapReceiptResult({ res }){
  const d = res.disposed || {};
  const sw = res.swapped_to || {};
  const lh = res.lexicon_hits || {};
  const ls = res.logprob_shift || {};
  return html`<div style="padding:2px 14px 14px">
    <div class="cfg" style="margin-top:4px">
      <span class="cap">swap</span>
      <span class=${"tag " + (res.causal_verified ? "cap-t" : "fail-t")}>causal_verified: ${String(res.causal_verified)}</span>
      <span class=${"tag " + (res.targeted_shift ? "cap-t" : "smp-t")}>targeted_shift: ${String(res.targeted_shift)}</span>
      <span>run <b>${res.run_id || "—"}</b></span>
    </div>

    ${res.blocked && html`<div class="cfg" style="margin-top:8px;border-left-color:var(--coral)">
      <span class="tag fail-t">BLOCKED · ${res.blocked}</span>
      <span>${res.note || "no further detail given."}</span>
    </div>`}

    <div style="margin-top:10px">
      <div class="none">disposed — the model's own J-lens read, before the swap</div>
      <div style="font-size:10px;color:var(--navy);padding:4px 0">
        top-1 <b>${d.jlens_top1 ?? "—"}</b> · layer ${d.jlens_layer ?? "—"} ·
        available ${String(d.jlens_available)}${d.jlens_reason ? " · " + d.jlens_reason : ""}
      </div>
      ${Array.isArray(d.jlens_top5) && d.jlens_top5.length ? html`<div style="display:flex;gap:5px;flex-wrap:wrap;padding-bottom:4px">
        ${d.jlens_top5.map((t,i) => html`<span key=${i} class="jchip">${t}</span>`)}
      </div>` : null}
      <div style="font-size:10px;color:var(--slate)">baseline_lean <b>${d.baseline_lean ?? "—"}</b></div>
      <div class="none" style="padding-top:3px">hint (your label — never fed into the computation): ${d.hint ?? "—"}</div>
    </div>

    <div style="margin-top:10px">
      <div class="none">swapped to</div>
      <div style="font-size:10px;color:var(--navy);padding:4px 0">
        concept <b>${sw.concept ?? "—"}</b> · layer ${sw.layer ?? "—"} · strength ${sw.strength ?? "—"} · coef ${sw.coef ?? "—"}
        ${sw.token_id != null ? html` · token_id ${sw.token_id}` : null}
      </div>
    </div>

    <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px">
      <div class="none">baseline_reply</div>
      <div style="font-size:10.5px;color:var(--slate);line-height:1.6">${res.baseline_reply ?? "—"}</div>
      <div class="none" style="margin-top:4px">swapped_reply</div>
      <div style="font-size:10.5px;color:var(--navy);line-height:1.6">${res.swapped_reply ?? "—"}</div>
      <div class="none" style="margin-top:4px">null_reply <span style="opacity:.75">(random-direction control, same magnitude/layer as the real swap)</span></div>
      <div style="font-size:10.5px;color:var(--slate);line-height:1.6">${res.null_reply ?? "—"}</div>
    </div>

    <div class="cfg" style="margin-top:10px">
      <span class="cap">measures</span>
      <span>lexicon hits — baseline ${lh.baseline ?? "—"} · swap ${lh.swap ?? "—"} · null ${lh.null ?? "—"}</span>
      <span>logprob shift — baseline ${fmt(ls.baseline)} · swap ${fmt(ls.swap)} · null ${fmt(ls.null)}</span>
      <span>swap/baseline ${fmt(ls.swap_over_baseline_nat)} nat · swap/null ${fmt(ls.swap_over_null_nat)} nat</span>
    </div>

    <div class="cfg" style="margin-top:8px">
      <span class="cap">coherence</span>
      <span class=${"tag " + (res.coherent ? "cap-t" : "fail-t")}>coherent: ${String(res.coherent)}</span>
      <span>score ${res.coherence_score != null ? fmt(res.coherence_score) : "—"}</span>
      <span>null control available <b>${String(res.null_control_available)}</b></span>
    </div>

    ${res.null_note && html`<div class="none" style="padding-top:8px">${res.null_note}</div>`}
    ${res.lexicon_note && html`<div class="none" style="padding-top:4px">${res.lexicon_note}</div>`}
  </div>`;
}

/* ───────────────────────── C) counterfactual — what if this dial were X ───────────────────────── */
function CounterfactualPanel({ axesState }){
  const rec = useStore(x => x.rec);
  const live = useStore(x => x.live);
  const axes = axesState.status === "ok" ? axesState.axes : [];
  const [dial, setDial] = useState("");
  const [value, setValue] = useState(0);
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    if(!dial && axes.length) setDial(axes[0].name);
  }, [axes]);

  async function run(){
    if(!rec){ toast("need a current run — load one from Replay first"); return; }
    if(guardLive(rec)) return;
    if(!dial){ toast("no dial selected — /steer/axes hasn't answered yet"); return; }
    setBusy(true); setErr(null); setRes(null);
    toast(`counterfactual — what if ${dial} were ${value}…`);
    const r = await api.counterfactual(rec.id, { [dial]: value });
    setBusy(false);
    if(!r){ setErr("counterfactual didn't answer — needs the qwen substrate; is the engine up?"); return; }
    setRes(r);
  }

  return html`<div class="mod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led"></span><span class="cap">counterfactual — what if this dial were X</span>
      <span class="tail">${busy ? "replaying…" : res ? "computed" : "on demand"}</span></div>
    <div style="padding:8px 14px 4px;display:flex;gap:10px;flex-wrap:wrap;align-items:end">
      <label style="display:flex;flex-direction:column;gap:3px;font-size:8.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mist)">
        dial
        <select value=${dial} onChange=${e => setDial(e.target.value)}
          style="font:inherit;font-size:10.5px;padding:5px 8px;border-radius:7px;border:1px solid var(--edge);background:linear-gradient(180deg,#fff,#E6F1F7);color:var(--navy);min-width:130px">
          ${axes.length ? axes.map(a => html`<option key=${a.name} value=${a.name}>${a.name}</option>`)
            : html`<option value="">no axes loaded</option>`}
        </select>
      </label>
      <label style="display:flex;flex-direction:column;gap:3px;font-size:8.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mist)">
        value
        <input type="number" step="0.1" value=${value}
          onInput=${e => { const n = +e.target.value; setValue(Number.isFinite(n) ? n : 0); }}
          style="font:inherit;font-size:10.5px;padding:5px 8px;border-radius:7px;border:1px solid var(--edge);background:linear-gradient(180deg,#fff,#E6F1F7);color:var(--navy);width:80px"/>
      </label>
      <button class=${"spd" + (busy ? " busy" : "")} onClick=${run}>RUN</button>
    </div>
    ${!rec && html`<div class="none" style="padding:0 14px 12px">need a current run — load one from Replay first.</div>`}
    ${rec && !live && html`<div class="none" style="padding:0 14px 12px">live server only — this is the sample reel.</div>`}
    ${busy && html`<div class="none" style="padding:0 14px 12px">replaying with the override…</div>`}
    ${err && html`<div style="padding:0 14px 12px"><span class="tag fail-t">FAILED</span> <span class="none">${err}</span></div>`}
    ${res && html`<${CounterfactualResult} res=${res}/>`}
  </div>`;
}

function CounterfactualResult({ res }){
  const delta = res.delta || {};
  const coh = res.coherence || {};
  return html`<div style="padding:2px 14px 14px">
    <div class="cfg" style="margin-top:4px">
      <span class="cap">counterfactual</span>
      <span class=${"tag " + (res.has_effect ? "cap-t" : "smp-t")}>has_effect: ${String(res.has_effect)}</span>
      <span class=${"tag " + (res.causal_verified ? "cap-t" : "fail-t")}>causal_verified: ${String(res.causal_verified)}</span>
      ${res.overrides_applied && html`<span>override ${JSON.stringify(res.overrides_applied)}</span>`}
    </div>

    ${coh.degenerate && html`<div class="cfg" style="margin-top:8px;border-left-color:var(--coral)">
      <span class="tag fail-t">DEGENERATE</span><span>${coh.reason || "reason not given"}</span>
    </div>`}

    <div style="margin-top:10px;display:flex;flex-direction:column;gap:6px">
      <div class="none">baseline_reply</div>
      <div style="font-size:10.5px;color:var(--slate);line-height:1.6">${res.baseline_reply ?? "—"}</div>
      <div class="none" style="margin-top:4px">counterfactual_reply</div>
      <div style="font-size:10.5px;color:var(--navy);line-height:1.6">${res.counterfactual_reply ?? "—"}</div>
    </div>

    <div class="cfg" style="margin-top:10px">
      <span class="cap">delta</span>
      <span>words ${Array.isArray(delta.words) ? delta.words.join(" → ") : "—"}</span>
      <span>wps ${Array.isArray(delta.wps) ? delta.wps.map(x => fmt(+x)).join(" → ") : "—"}</span>
      <span>changed ${delta.changed != null ? delta.changed + "%" : "—"}</span>
    </div>

    ${res.override_note && html`<div class="none" style="padding-top:8px">${res.override_note}</div>`}
    ${res.note && html`<div class="none" style="padding-top:4px">${res.note}</div>`}
    ${res.cost_note && html`<div class="none" style="padding-top:4px">${res.cost_note}</div>`}
  </div>`;
}
