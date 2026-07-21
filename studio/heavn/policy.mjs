/* heavnOS · policy chip — the server's calibrated ask-band signal (`clozn_policy`).
   This is a POLICY verdict about the whole reply ("this confidence falls where the journal's
   calibration says to ask, not answer flatly"), not a per-token confidence readout — it is
   deliberately styled apart from the read-token confidence spans (coral/lilac/teal) and the
   J-lens jchips (plain lilac). Renders nothing when `policy` is absent/bandless: honest absence,
   never a fabricated verdict, mirroring every other optional signal in this studio. */
import { html } from "./vendor/preact-standalone.mjs";

const num = (v, d = 3) => typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "—";
const pct = v => typeof v === "number" && Number.isFinite(v) ? Math.round(v * 100) + "%" : "—";

export function PolicyChip({ policy }){
  if(!policy || !policy.band) return null;
  const band = String(policy.band);
  const summary = `clozn policy: ${band} band — score ${num(policy.score)}`
    + (policy.score_aggregate ? ` (${policy.score_aggregate})` : "")
    + (policy.calibration_task ? ` · task ${policy.calibration_task}` : "")
    + " · this is a policy signal over the whole reply, not a per-token readout, and not a correctness verdict";
  return html`<details class="policy-chip" data-testid="policy-chip">
    <summary title=${summary}><i></i><span>policy · ${band}</span>${policy.score != null ? html`<b>${pct(policy.score)}</b>` : null}</summary>
    <div class="policy-chip-body">
      <div class="policy-chip-row"><span>band</span><b>${band}</b></div>
      <div class="policy-chip-row"><span>score</span><b>${num(policy.score)}</b>
        ${policy.score_aggregate ? html`<i class="policy-chip-agg">${policy.score_aggregate}</i>` : null}</div>
      ${policy.calibration_task ? html`<div class="policy-chip-row"><span>calibrated task</span><b>${policy.calibration_task}</b></div>` : null}
      ${policy.calibration_model ? html`<div class="policy-chip-row"><span>calibrated model</span><b>${policy.calibration_model}</b></div>` : null}
      ${policy.ask_at != null ? html`<div class="policy-chip-row"><span>ask band from</span><b>&ge; ${num(policy.ask_at)}</b></div>` : null}
      ${policy.answer_at != null ? html`<div class="policy-chip-row"><span>answer band from</span><b>&ge; ${num(policy.answer_at)}</b></div>` : null}
      <p>${policy.note || "confidence on this reply fell in the calibrated ‘ask’ band — a policy signal over the whole reply, not a per-token readout, and not a correctness verdict."}</p>
    </div>
  </details>`;
}
