/* heavnOS · honest stubs for the modules that open in later phases (W3–W6).
   Each names what it will be and which endpoints already exist — no fake UI, no dead controls. */
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

export const PatchStub = () => Stub({ title: "patch — interventions", led: "lilac",
  lines: [
    "any-concept dials — steer toward any word, zero calibration (dir(c), validated +6–9 nats, content concepts only)",
    "swap receipts — read the disposition, write a different concept, diff the answer (93% coherent re-route)",
    "the glass diff — before/after arms side by side, measured not narrated",
  ],
  endpoints: "/runs/<id>/swap_receipt · /steer/axes · /engine taps" });

export const EditStub = () => Stub({ title: "edit — pin & resolve", led: "blue",
  lines: [
    "pin spans (they lock as glass) · edit an anchor · the resolve propagates through unpinned text in BOTH directions",
    "honest constraint: the resolve follows your anchors and context — it does not take instructions",
    "every resolve ships with its score-delta receipt",
  ],
  endpoints: "/v1/board · /v1/revise (diffusion substrates)" });

export const MemoryStub = () => Stub({ title: "memory — two tiers", led: "",
  lines: [
    "anchored tier — the card IS its weighted concept list; “what did you learn?” answered by lookup, confabulation-proof",
    "style tier — the learned free prefix, honestly labeled: not self-reportable",
    "review queue — nothing installs itself; provenance or it doesn't approve",
  ],
  endpoints: "/memory/* CRUD · /runs/<id>/propose-memory · /memory/<id>/runs" });

export const ModelsStub = () => Stub({ title: "models — registry", led: "blue",
  lines: [
    "quant-check — “did Q4 lobotomize your model?” measured on YOUR runs, per token",
    "fit read — a GGUF's specs from its header, before you download it",
  ],
  endpoints: "quant_check (CLI today) · fit-planner header reader" });

export const SettingsStub = () => Stub({ title: "settings", led: "",
  lines: [
    "model + substrate selection, capture tier, sampling mode (S5), endpoint config",
  ],
  endpoints: "/sampling/mode · /capture/tier" });
