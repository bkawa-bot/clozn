/* heavnOS shell — topbar, module rail, router, boot. */
import { html, render } from "./vendor/preact-standalone.mjs";
import { store, useStore, toast, SAMPLE } from "./state.mjs";
import { api } from "./api.mjs";
import { ReplayModule } from "./modules/replay.mjs";
import { MemoryModule } from "./modules/memory.mjs";
import { PatchModule } from "./modules/patch.mjs";
import { EditModule } from "./modules/edit.mjs";
import { ScopeModule } from "./modules/scope.mjs";
import { AtlasModule } from "./modules/atlas.mjs";
import { ModelsStub, SettingsStub } from "./modules/stubs.mjs";

const MODULES = [
  { id: "replay",   nm: "Replay",   view: ReplayModule },
  { id: "patch",    nm: "Patch",    view: PatchModule },
  { id: "edit",     nm: "Edit",     view: EditModule },
  { id: "memory",   nm: "Memory",   view: MemoryModule },
  { id: "scope",    nm: "Scope",    view: ScopeModule },
  { id: "atlas",    nm: "Atlas",    view: AtlasModule },
  { id: "models",   nm: "Models",   view: ModelsStub,   soon: true },
  { id: "settings", nm: "Settings", view: SettingsStub, soon: true },
];

function Topbar(){
  const s = useStore(x => ({ live: x.live, rec: x.rec }));
  const d = (s.rec && s.rec.meta && s.rec.meta.decode) || {};
  return html`<div class="topbar">
    <div style="display:flex;align-items:baseline;gap:11px">
      <span class="wordmark">CLOZN</span>
      <span class="sub">local glass-box ai runtime</span>
    </div>
    <div style="flex:1;display:flex;justify-content:center;align-items:center;gap:15px">
      <span class="stat"><i class="led beats" style="width:6px;height:6px"></i>LOCAL</span>
      <span class="vsep"></span>
      <span class="stat">ENGINE <b>${s.live ? "connected" : "offline"}</b></span>
      <span class="stat">MODEL <b>${(s.rec && s.rec.model) || "—"}</b></span>
      ${d.quant && html`<span class="stat">· ${d.quant}</span>`}
    </div>
    <div style="display:flex;align-items:center;gap:14px">
      <span class="stat mono" style="letter-spacing:.04em">
        local://${useStore(x => x.route)}/${(s.rec && s.rec.id) || ""}</span>
      <span class="windots"><span/><span/><span/></span>
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
  const s = useStore(x => ({ route: x.route, rec: x.rec, live: x.live }));
  const d = (s.rec && s.rec.meta && s.rec.meta.decode) || {};
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
        <span class="nm">${m.nm}</span>
        <span class="mark">${s.route === m.id ? "◂" : ""}</span>
      </button>`)}
    <div class="navfoot">
      <b>CLOZN v0.2</b><br/>
      engine ${d.engine || "—"} · substrate ${(s.rec && s.rec.substrate) || "—"}<br/>
      session ${(s.rec && s.rec.id) || "—"}
      <div class="cap" style="font-size:8.5px;letter-spacing:.22em;color:var(--mist);margin-top:11px">system health</div>
      <div class="health"><div class="bar"><i class="beats"></i></div><span>${s.live ? "98%" : "—"}</span></div>
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
      <span><i style="background:var(--teal)"></i>teal: calm focus, observation</span>
      <span><i style="background:var(--lilac)"></i>lilac: branch, disposition</span>
      <span><i style="background:var(--pink)"></i>pink: selection, branch glow</span>
      <span><i style="background:var(--coral)"></i>coral: repair, protect the signal</span>
    </span>
    <span class="credo">Clozn is a local runtime. It doesn't phone home.</span>
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
                  verbResult: null, jlProvenance: null, jlReason: null };
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
  render(html`<${App}/>`, rootEl);
  const list = await api.listRuns();
  if(list && Array.isArray(list.runs)){
    store.set({ live: true, runs: list.runs });
    if(list.runs.length) await loadRun(list.runs[0].id);
  } else {
    store.set({ live: false, runs: [{ ...SAMPLE }], full: { [SAMPLE.id]: SAMPLE },
                currentId: SAMPLE.id, rec: SAMPLE });
  }
  /* keep the journal fresh: light polling (new runs land while you work) */
  setInterval(async () => {
    if(!store.get().live) return;
    const l = await api.listRuns();
    if(l && Array.isArray(l.runs)) store.set({ runs: l.runs });
  }, 6000);
}
