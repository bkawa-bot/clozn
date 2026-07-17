/* heavnOS shell — topbar, module rail, router, boot. */
import { html, render } from "./vendor/preact-standalone.mjs";
import { store, useStore, toast, SAMPLE } from "./state.mjs";
import { api } from "./api.mjs";
import { ReplayModule } from "./modules/replay.mjs";
import { ReadModule } from "./modules/read.mjs";
import { MemoryModule } from "./modules/memory.mjs";
import { PatchModule } from "./modules/patch.mjs";
import { ExperimentModule } from "./modules/experiment.mjs";
import { EditModule } from "./modules/edit.mjs";
import { ScopeModule } from "./modules/scope.mjs";
import { AtlasModule } from "./modules/atlas.mjs";
import { SettingsModule } from "./modules/settings.mjs";
import { ModelsStub } from "./modules/stubs.mjs";

const MODULES = [
  { id: "read",     nm: "Read",     sub: "answer first",     view: ReadModule },
  { id: "replay",   nm: "Replay",   sub: "runtime desk",    view: ReplayModule },
  { id: "patch",    nm: "Patch",    sub: "interventions",   view: PatchModule },
  { id: "experiment", nm: "Experiment", sub: "compare & prove", view: ExperimentModule },
  { id: "edit",     nm: "Edit",     sub: "inline repair",   view: EditModule },
  { id: "memory",   nm: "Memory",   sub: "local store",     view: MemoryModule },
  { id: "scope",    nm: "Scope",    sub: "layer inspector", view: ScopeModule },
  { id: "atlas",    nm: "Atlas",    sub: "concept map",     view: AtlasModule },
  { id: "models",   nm: "Models",   sub: "local inventory", view: ModelsStub,   soon: true },
  { id: "settings", nm: "Settings", sub: "profiles & prefs", view: SettingsModule },
];

function Topbar(){
  const s = useStore(x => ({ rec: x.rec, route: x.route }));
  return html`<div class="topbar">
    <div style="display:flex;align-items:baseline;gap:11px">
      <span class="wordmark">CLOZN</span>
      <span class="sub">local glass-box ai runtime</span>
    </div>
    <!-- honesty pills: all three are genuinely, structurally true of clozn (local-first, no cloud
         calls, no telemetry) -- static by design, not a live connectivity readout (that's StatusLine,
         just below). Never add a pill here that isn't unconditionally true. -->
    <div style="flex:1;display:flex;justify-content:center;align-items:center;gap:15px">
      <span class="stat"><i class="led beats" style="width:6px;height:6px"></i>LOCAL MODE</span>
      <span class="stat"><i class="led off" style="width:6px;height:6px"></i>NO CLOUD</span>
      <span class="stat"><i class="led" style="width:6px;height:6px"></i>NO TELEMETRY</span>
    </div>
    <div style="display:flex;align-items:center;gap:14px">
      <span class="stat mono" style="letter-spacing:.04em">
        local://${s.route}/${(s.rec && s.rec.id) || ""}</span>
      <span class="winctl"><span>−</span><span>▢</span><span>×</span></span>
    </div>
  </div>`;
}

function StatusLine(){
  const live = useStore(x => x.live);
  return html`<div class=${"statusline " + (live ? "live" : "sample")}>
    <i class="beats"></i>
    ${live ? "live — the run journal, recorded on this machine"
           : "sample reel — server offline · open this page from the clozn server to go live"}
  </div>`;
}

function NavRail(){
  const s = useStore(x => ({ route: x.route, rec: x.rec, live: x.live, worker: x.worker }));
  const d = (s.rec && s.rec.meta && s.rec.meta.decode) || {};
  // Every value below is either read straight off the loaded run record or off /readyz's live worker
  // health (s.worker) -- never a placeholder. Absent data reads "—", it never gets a made-up stand-in.
  const ctx = s.worker && s.worker.n_ctx != null ? Number(s.worker.n_ctx).toLocaleString() + " tok" : "—";
  return html`<div class="mod navmod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="head"><span class="cap" style="font-size:9px;color:var(--mist)">system</span>
      <span style="font-size:8.5px;color:var(--mist)">${String(MODULES.length).padStart(2,"0")}</span></div>
    ${MODULES.map((m,i) => html`
      <button class=${"navrow" + (s.route === m.id ? " on" : "")}
        onClick=${() => {
          if(m.inReplay){ toast("SCOPE lives on the Replay desk for now — it's the State Scope module"); return; }
          if(m.soon && !m.view){ toast(m.nm.toUpperCase() + " — this door opens in a later phase"); return; }
          store.set({ route: m.id });
        }}>
        <span class="idx">${String(i+1).padStart(2,"0")}</span>
        <span class="nled"></span>
        <span class="nm-wrap"><span class="nm">${m.nm}</span><span class="sub">${m.sub}</span></span>
        <span class="mark">${s.route === m.id ? "◂" : ""}</span>
      </button>`)}
    <div class="navfoot">
      <b>CLOZN v0.2</b>
      <div class="kv"><span class="k">engine</span><span class="v">${(s.rec && s.rec.model) || "—"}</span></div>
      <div class="kv"><span class="k">context</span><span class="v">${ctx}</span></div>
      <div class="kv"><span class="k">precision</span><span class="v">${d.quant || "—"}</span></div>
      <div class="kv"><span class="k">session</span><span class="v">${(s.rec && s.rec.id) || "—"}</span></div>
      <div class="cap" style="font-size:8.5px;letter-spacing:.22em;color:var(--mist);margin-top:11px">system health</div>
      <div class="health"><div class="bar"><i class="beats"></i></div><span>${s.live ? "live" : "—"}</span></div>
    </div>
  </div>`;
}

function Toast(){
  const t = useStore(x => x.toast);
  return html`<div class=${"toast" + (t ? " show" : "")} role="status">${t ? t.msg : ""}</div>`;
}

function Footer(){
  return html`<div class="footer">
    <span class="cap" style="font-size:8.5px;color:var(--mist)">gradient flow</span>
    <span class="flowbar"></span>
    <span class="legend">
      <span><i style="background:var(--teal)"></i>seafoam: calm, focus, observation</span>
      <span><i style="background:var(--pink)"></i>magenta: selection, branch glow</span>
      <span><i style="background:var(--coral)"></i>coral: repair, protect the signal</span>
    </span>
    <span class="credo">Clozn is a local runtime. It doesn't phone home.</span>
    <span class="credo">Built for thinking out loud, not for performance.</span>
  </div>`;
}

function App(){
  const route = useStore(x => x.route);
  const M = MODULES.find(m => m.id === route) || MODULES[0];
  const View = M.view || ReplayModule;
  return html`<div class="app">
    <${Topbar}/>
    <${StatusLine}/>
    <div class="frame">
      <div class="col"><${NavRail}/></div>
      <${View}/>
    </div>
    <${Footer}/>
    <${Toast}/>
  </div>`;
}

/* full-record loader with cache; unwraps {run:...} envelopes defensively */
export async function loadRun(id){
  const fresh = { P: 0, receipts: null, leaning: null, leaningInfluence: null,
                  verbResult: null, jlProvenance: null, jlReason: null, readError: null };
  const s = store.get();
  if(s.full[id]){ store.set({ currentId: id, rec: s.full[id], ...fresh }); return s.full[id]; }
  const rec = s.live ? await api.getRun(id) : null;   // contracts §2: the bare record, no envelope
  if(rec && rec.id){
    store.set(st => ({ full: { ...st.full, [id]: rec }, currentId: id, rec, ...fresh }));
    return rec;
  }
  return null;
}

export async function boot(rootEl){
  const deep = new URLSearchParams(location.search).get("run");
  if(deep) store.set({ route: "read", readRequest: deep });
  render(html`<${App}/>`, rootEl);
  const list = await api.listRuns();
  /* ambient delivery (channel 1): a receipt-footer link is /r/<id> -> /heavn/index.html?run=<id>.
     Deep-link straight to that run's document-first Read view. A missing permalink must never silently
     substitute the newest unrelated run. */
  if(list && Array.isArray(list.runs)){
    store.set({ live: true, runs: list.runs });
    if(deep){
      const r = await loadRun(deep);                         // fetches directly; not limited to page 1
      if(!r) store.set({ currentId: null, rec: null,
        readError: `Run "${deep}" was not found in this local journal.` });
    } else if(list.runs.length){
      await loadRun(list.runs[0].id);
    }
  } else {
    store.set({ live: false, runs: [{ ...SAMPLE }], full: { [SAMPLE.id]: SAMPLE },
                currentId: SAMPLE.id, rec: SAMPLE });
  }
  if(store.get().live){
    const rz = await api.readyz();                            // real CONTEXT (n_ctx) for the nav footer
    store.set({ worker: (rz && rz.worker) || null });          // null (not a placeholder) when unavailable
  }
  /* keep the journal fresh: light polling (new runs land while you work) */
  setInterval(async () => {
    if(!store.get().live) return;
    const l = await api.listRuns();
    if(l && Array.isArray(l.runs)) store.set({ runs: l.runs });
    const rz = await api.readyz();
    store.set({ worker: (rz && rz.worker) || null });
  }, 6000);
}
