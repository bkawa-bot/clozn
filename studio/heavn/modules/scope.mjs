/* heavnOS · SCOPE — the J-space explorer.
   Probe any text through the fitted Jacobian lens (POST /jlens, contracts §9) and — the part
   that makes J-SPACE visible — sweep it across every fitted depth at once: one row per layer,
   one column per token, watching the disposition crystallize as the network deepens.
   Honesty carried: DERIVED tags, provenance verbatim, available:false rendered as the reason,
   "a disposition, not a verified thought; blank ≠ nothing". */
import { html, useState, useEffect } from "../vendor/preact-standalone.mjs";
import { store, useStore, toast } from "../state.mjs";
import { api } from "../api.mjs";

export function ScopeModule(){
  const rec = useStore(x => x.rec);
  const live = useStore(x => x.live);
  const [text, setText] = useState("The country shaped like a boot is");
  const [probe, setProbe] = useState(null);      // single-layer result
  const [sweep, setSweep] = useState(null);      // {layers:[], rows:[{layer, tokens, top}], prov}
  const [busy, setBusy] = useState(null);        // "probe" | "sweep" | null
  const [reason, setReason] = useState(null);
  const [layer, setLayer] = useState(null);

  const guard = () => {
    if(!live){ toast("the lens needs the live server (this is the sample reel)"); return false; }
    if(!text.trim()){ toast("give the lens some text to read"); return false; }
    return true;
  };

  async function doProbe(L){
    if(!guard()) return;
    setBusy("probe"); setReason(null);
    const r = await api.jlensText(text.trim(), L ?? layer ?? undefined);
    setBusy(null);
    if(!r){ setReason("the server didn't answer"); setProbe(null); return; }
    if(r.available === false){ setReason(r.reason || "no lens available"); setProbe(null); return; }
    setProbe(r); setLayer(r.layer);
  }

  async function doSweep(){
    if(!guard()) return;
    setBusy("sweep"); setReason(null); setSweep(null);
    /* discover the fitted layers from one read, then read every depth */
    const first = await api.jlensText(text.trim());
    if(!first || first.available === false){
      setBusy(null); setReason((first && first.reason) || "no lens available"); return;
    }
    const layers = first.available_layers || (first.provenance && first.provenance.layers) || [first.layer];
    const rows = [];
    for(const L of layers){
      const r = (L === first.layer) ? first : await api.jlensText(text.trim(), L);
      if(r && r.available !== false)
        rows.push({ layer: L, tokens: r.tokens || [],
                    top: (r.readouts || []).map(cell => (cell && cell[0]) || null) });
    }
    setBusy(null);
    if(!rows.length){ setReason("no layer produced a readout"); return; }
    setSweep({ layers, rows, prov: first.provenance || null,
               nTok: Math.min(...rows.map(r => r.top.length)) });
  }

  /* crystallization: at each position, the final layer's top-1 — and the first depth it appears */
  const finalRow = sweep && sweep.rows[sweep.rows.length - 1];

  return html`<div class="col">
    <div class="mod">
      <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
      <div class="mod-h"><span class="led lilac"></span><span class="cap">j-space — lens probe</span>
        <span class="tail">what each position is disposed to say, later</span>
        <span class="tag der-t">DERIVED</span></div>
      <div style="padding:6px 13px 4px;display:flex;gap:8px;flex-wrap:wrap;align-items:center">
        <input type="text" value=${text} onInput=${e => setText(e.target.value)}
          onKeyDown=${e => { if(e.key === "Enter") doProbe(); }}
          placeholder="any text — the lens reads every position"
          style="flex:1;min-width:260px;font-family:var(--mono);font-size:10.5px;color:var(--navy);
                 border:1px solid var(--edge);border-radius:8px;padding:7px 11px;
                 background:linear-gradient(180deg,#fff,#EAF4FA)"/>
        ${rec && rec.response && html`<button class="spd"
          onClick=${() => { setText(String(rec.response)); toast("canvas set to the current run's answer"); }}>
          use current run</button>`}
        <button class=${"spd" + (busy === "probe" ? " busy" : "")} onClick=${() => doProbe()}>READ</button>
        <button class=${"spd" + (busy === "sweep" ? " busy" : "")}
          style="color:#5F5794;border-color:rgba(154,146,200,.7);font-weight:600" onClick=${doSweep}>
          ${busy === "sweep" ? "SWEEPING DEPTHS…" : "DEPTH SWEEP"}</button>
      </div>
      ${reason && html`<div class="provenance"><b>J-lens</b> — unavailable: ${reason} ·
        blank ≠ nothing, but there is nothing fitted to read with</div>`}

      ${probe && html`<div style="padding:6px 13px 10px">
        <div class="lbl" style="padding:2px 0 6px">layer ${probe.layer}
          ${(probe.available_layers || []).length > 1 && html` · ${(probe.available_layers).map(L =>
            html`<button class="mono" style=${"font-size:8px;padding:0 4px;color:" +
                (L === probe.layer ? "var(--navy)" : "var(--mist)")}
              onClick=${() => doProbe(L)}>${L}</button>`)}`}</div>
        <div style="display:flex;flex-wrap:wrap;gap:6px">
          ${(probe.tokens || []).map((tok,i) => {
            const cell = (probe.readouts || [])[i] || [];
            return html`<div style="display:flex;flex-direction:column;gap:2px;align-items:center;
                min-width:0;border:1px solid var(--edge-soft);border-radius:8px;padding:5px 7px;
                background:rgba(255,255,255,.5)">
              <span class="mono" style="font-size:9.5px;color:var(--navy)">${String(tok).trim() || "·"}</span>
              ${cell.slice(0,3).map((r,k) => html`<span class=${"jchip d" + (k+1)}
                title=${"score " + (r.score != null ? (+r.score).toFixed(1) : "—")}>${String(r.piece || "").trim()}</span>`)}
              ${!cell.length && html`<span class="none" style="font-size:8px">—</span>`}
            </div>`; })}
        </div>
      </div>`}
    </div>

    ${sweep && html`<div class="mod">
      <div class="mod-h"><span class="led lilac beats"></span><span class="cap">depth sweep — the disposition crystallizing</span>
        <span class="tail">${sweep.rows.length} fitted depths · top-1 per position</span>
        <span class="tag der-t">DERIVED</span></div>
      <div style="padding:4px 13px 6px;overflow-x:auto">
        <table style="border-collapse:collapse;font-family:var(--mono);font-size:9px;min-width:100%">
          <thead><tr>
            <th style="text-align:left;padding:4px 8px;color:var(--mist);font-weight:400;letter-spacing:.1em">DEPTH</th>
            ${Array.from({length: sweep.nTok}, (_,i) => html`<th style="padding:4px 5px;color:var(--slate);
               font-weight:600;white-space:nowrap;border-bottom:1px solid var(--edge)">${
               String((finalRow.tokens[i] || "")).trim() || "·"}</th>`)}
          </tr></thead>
          <tbody>${sweep.rows.map(row => {
            const mx = Math.max(1e-6, ...row.top.slice(0, sweep.nTok).map(t => (t && t.score) || 0));
            return html`<tr>
              <td style="padding:4px 8px;color:var(--navy);font-weight:600;white-space:nowrap;
                  border-right:1px solid var(--edge-soft)">L${row.layer}</td>
              ${Array.from({length: sweep.nTok}, (_,i) => {
                const t = row.top[i];
                const piece = t ? String(t.piece || "").trim() : "";
                const finalPiece = finalRow.top[i] ? String(finalRow.top[i].piece || "").trim() : "";
                const settled = piece && piece === finalPiece;   /* already holds the final disposition */
                const a = t ? .25 + .6 * ((t.score || 0) / mx) : 0;
                return html`<td style=${"padding:3px 5px;text-align:center;white-space:nowrap;" +
                    "border-bottom:1px solid var(--edge-soft);" +
                    (settled ? `background:rgba(95,200,188,${(a*.6).toFixed(2)});color:#14584F;font-weight:600;`
                             : piece ? `background:rgba(143,168,232,${(a*.4).toFixed(2)});color:var(--slate);` : "")}
                  title=${t && t.score != null ? "score " + (+t.score).toFixed(1) : ""}>${piece || "·"}</td>`; })}
            </tr>`; })}
          </tbody>
        </table>
        <div class="none" style="padding:8px 0 4px;font-size:8.5px">
          teal = this depth already holds the FINAL layer's disposition (the moment it crystallizes) ·
          periwinkle = a different disposition still in play · intensity = score within its own layer.
          Each row is an independent lens read of the same text.</div>
      </div>
      ${sweep.prov && html`<div class="provenance"><b>J-lens</b> — ${sweep.prov.note || ""}${
        sweep.prov.fit_model ? " · fitted: " + sweep.prov.fit_model : ""} ·
        a disposition, not a verified thought; blank ≠ nothing</div>`}
    </div>`}
  </div>`;
}
