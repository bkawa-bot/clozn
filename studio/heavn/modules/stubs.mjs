/* honest stub for the modules that open in later phases.
   Names what it will be and which endpoints already exist — no fake UI, no dead controls.
   (Patch/Edit/Memory/Settings had stubs here too, but their real modules shipped and replaced them
   in app.mjs's MODULES table -- only Models is still stub-only, so only its stub remains.) */
import { html } from "../vendor/preact-standalone.mjs";

function Stub({ title, led, lines, endpoints }){
  return html`<div class="col"><div class="mod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class=${"led " + (led || "")}></span><span class="cap">${title}</span>
      <span class="tag smp-t">OPENS IN A LATER PHASE</span></div>
    <div style="padding:6px 16px 16px;font-size:11px;color:var(--slate);line-height:1.8;max-width:560px">
      ${lines.map(l => html`<div>· ${l}</div>`)}
      ${endpoints && html`<div style="margin-top:10px;font-size:9px;color:var(--mist);letter-spacing:.04em">
        backend already live: <span class="mono">${endpoints}</span></div>`}
    </div>
  </div></div>`;
}

export const ModelsStub = () => Stub({ title: "models — registry", led: "blue",
  lines: [
    "quant-check — “did Q4 lobotomize your model?” measured on YOUR runs, per token",
    "fit read — a GGUF's specs from its header, before you download it",
  ],
  endpoints: "quant_check (CLI today) · fit-planner header reader" });
