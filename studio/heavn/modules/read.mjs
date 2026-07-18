/* heavnOS · Read — the document-first permalink landing. Everything here comes from the stored run:
   no generation, no self-report, and no on-demand model call. Confidence is shown only as recorded token
   commitment; it is never promoted to correctness. */
import { html, useState, useEffect } from "../vendor/preact-standalone.mjs";
import { store, useStore, normSteps, shortTime } from "../state.mjs";
import { api } from "../api.mjs";
import { PolicyChip } from "../policy.mjs";

const LOW_CONF = 0.5;
const WATCH_CONF = 0.8;

function promptFor(rec){
  const messages = (rec && (rec.messages || rec.assembled_messages)) || [];
  const users = messages.filter(m => m && m.role === "user");
  const last = users.length ? users[users.length - 1].content : null;
  if(typeof last === "string" && last.trim()) return last.trim();
  return (rec && (rec.prompt_summary || rec.prompt)) || "Prompt was not recorded.";
}

/* Contiguous low-confidence tokens are one readable zoom target, not a noisy list of individual pieces. */
function sketchySpans(steps){
  const out = [];
  let active = null;
  steps.forEach((step, i) => {
    if(step.conf == null || step.conf >= LOW_CONF){
      if(active){ out.push(active); active = null; }
      return;
    }
    if(!active) active = { start: i, end: i + 1, pieces: [step.piece], confidences: [step.conf] };
    else { active.end = i + 1; active.pieces.push(step.piece); active.confidences.push(step.conf); }
  });
  if(active) out.push(active);
  return out.map((span, i) => ({ ...span, id: i, text: span.pieces.join(""),
    min: Math.min(...span.confidences), mean: span.confidences.reduce((a,b) => a + b, 0) / span.confidences.length }));
}

function band(step){
  if(step.conf == null) return "unknown";
  if(step.conf < LOW_CONF) return "sketchy";
  if(step.conf < WATCH_CONF) return "watch";
  return "decided";
}

function pct(value){
  return typeof value === "number" && Number.isFinite(value) ? Math.round(value * 100) + "%" : "—";
}

function ActuaryPanel({ rec }){
  const live = useStore(x => x.live);
  const [state, setState] = useState({ status: "loading", report: null, assessment: null });

  useEffect(() => {
    let dead = false;
    if(!live || !rec || rec._sample){ setState({ status: "sample", report: null, assessment: null }); return () => {}; }
    setState({ status: "loading", report: null, assessment: null });
    Promise.all([api.journalActuary(), api.runActuary(rec.id)]).then(([report, assessment]) => {
      if(dead) return;
      const okReport = report && report.calibration && report.failure_model;
      const okAssessment = assessment && assessment.__status < 400 && !assessment.error;
      setState({ status: okReport || okAssessment ? "ok" : "error",
        report: okReport ? report : null, assessment: okAssessment ? assessment : null });
    });
    return () => { dead = true; };
  }, [live, rec && rec.id]);

  const a = state.assessment, report = state.report;
  const cal = report && report.calibration;
  const bins = cal && Array.isArray(cal.bins) ? cal.bins.filter(b => b.n) : [];
  const drift = report && Array.isArray(report.drift) ? report.drift : [];
  const verdict = !a || !a.available ? "unavailable" : a.warning ? "warning"
    : a.weak_evidence ? "weak" : "clear";

  return html`<section class="mod actuary-panel" data-testid="actuary-panel">
    <div class="mod-h"><span class=${"led " + (verdict === "warning" ? "coral" : "lilac")}></span>
      <span class="cap">journal actuary</span>
      <span class="tail">${report ? `${report.n_runs}/${report.n_total} organic` : "behavioral proxy"}</span></div>
    <div class="actuary-body">
      ${state.status === "loading" && html`<div class="none">comparing this trace with earlier organic runs…</div>`}
      ${state.status === "sample" && html`<div class="none">live journal only — the sample reel has no past outcomes.</div>`}
      ${state.status === "error" && html`<div class="none">the journal report did not answer; no risk estimate was substituted.</div>`}

      ${state.status === "ok" && html`<div class=${"actuary-current " + verdict}>
        <div class="actuary-verdict">
          ${verdict === "warning" && html`<span class="tag fail-t">RESEMBLES PAST FAILURES</span>`}
          ${verdict === "clear" && html`<span class="tag cap-t">NOT FLAGGED BY HEURISTIC</span>`}
          ${verdict === "weak" && html`<span class="tag smp-t">WEAK EVIDENCE · NO WARNING</span>`}
          ${verdict === "unavailable" && html`<span class="tag smp-t">MODEL UNTRAINED</span>`}
          ${a && a.score != null && html`<b>resemblance ${(+a.score).toFixed(2)}</b>`}
        </div>
        ${a && html`<div class="actuary-basis">${a.n_good} accepted-proxy · ${a.n_bad} bad-proxy ·
          ${a.n_past_organic} earlier organic runs${a.warning_eligible ? ` · warning at ${(+a.threshold).toFixed(2)}`
            : ` · warning needs ${a.min_class_n} of each class`}</div>`}
        ${a && a.warning && html`<p>This run's recorded trace shape is closer to the earlier bad-proxy
          centroid than the accepted-proxy centroid. Inspect the highlighted answer; this is not a fact-check.</p>`}
        ${a && !a.warning && a.available && html`<p>${a.weak_evidence
          ? "A resemblance score exists, but the past sample is too small to trigger an alert."
          : "This heuristic did not flag the trace. That is not evidence that the answer is correct."}</p>`}
        ${a && !a.available && html`<p>The journal needs at least two earlier runs in each proxy class to fit a trace-shape model.</p>`}
        ${a && Array.isArray(a.drivers) && a.drivers.length ? html`<div class="actuary-drivers">
          ${a.drivers.map(d => html`<span key=${d.feature}><b>${d.feature}</b> ${(+d.value).toFixed(2)} · bad-lean ${pct(d.bad_lean)}</span>`)}
        </div>` : null}
      </div>`}

      ${report && html`<details class="actuary-report">
        <summary>JOURNAL REPORT · PROXY, NOT CORRECTNESS</summary>
        <div class="actuary-report-body">
          <div class="actuary-statline"><span>scored <b>${cal.n_scored}/${cal.n_runs}</b></span>
            <span>ECE proxy <b>${cal.ece_proxy == null ? "—" : (+cal.ece_proxy).toFixed(3)}</b></span>
            <span>drift alarms <b>${drift.length}</b></span>
            <span>cached <b>${Math.round(report.computed_ago_s || 0)}s ago</b></span></div>
          ${bins.length ? html`<div class="actuary-bins">
            ${bins.map(b => html`<div class="actuary-bin" key=${b.lo}>
              <span>${(+b.lo).toFixed(1)}–${(+b.hi).toFixed(1)} · n=${b.n}</span>
              <div><i style=${`width:${pct(b.trusted_rate)}`}></i>
                <em style=${`left:${pct(b.mean_conf)}`} title=${`mean confidence ${(+b.mean_conf).toFixed(2)}`}></em></div>
              <b>kept ${pct(b.trusted_rate)}</b>
            </div>`)}
          </div>` : html`<div class="none">no scored organic runs — no proxy curve was invented.</div>`}
          ${drift.length ? html`<div class="actuary-drift">
            ${drift.slice(0,4).map((d,i) => html`<div key=${i}><span class=${"tag " + (d.severity === "alarm" ? "fail-t" : "smp-t")}>${d.severity}</span>
              <b>${d.prompt_class}</b><span>confidence ${d.delta == null ? "—" : (+d.delta).toFixed(2)} · bad proxy ${pct(d.bad_rate_old)}→${pct(d.bad_rate_new)}</span></div>`)}
          </div>` : null}
          <p class="actuary-note">${cal.note}</p>
          ${a && html`<p class="actuary-note">${a.note}</p>`}
        </div>
      </details>`}
    </div>
  </section>`;
}

function CalibrationCurve({ rec }){
  const [state, setState] = useState({ status: "loading", data: null });
  useEffect(() => {
    let dead = false;
    setState({ status: "loading", data: null });
    api.journalCalibration().then(d => {
      if(dead) return;
      setState({ status: d && d.available ? "ok" : "none", data: d });
    });
    return () => { dead = true; };
  }, [rec && rec.id]);

  if(state.status === "loading") return null;
  if(state.status === "none") return html`<section class="mod calibration-panel">
    <div class="mod-h"><span class="led lilac"></span><span class="cap">truth-tier calibration</span></div>
    <div class="none">${state.data && state.data.note
      ? state.data.note
      : "no outcome-grounded calibration available — run clozn eval --save"}</div>
  </section>`;

  const d = state.data;
  const bins = Array.isArray(d.reliability_bins) ? d.reliability_bins.filter(b => b.n) : [];
  const report = d.report || {};
  const temp = report.temperature_scaling || {};
  const sel = report.selective || {};
  const sel70 = sel["70"] || {};

  return html`<section class="mod calibration-panel">
    <div class="mod-h"><span class="led cyan"></span>
      <span class="cap">truth-tier calibration</span>
      <span class="tail">${d.n || 0} probes · ${d.set || "unknown"} · ${d.score || "—"}</span></div>
    <div class="calibration-body">
      <div class="actuary-statline">
        <span>ECE <b>${report.ece != null ? (+report.ece).toFixed(3) : "—"}</b></span>
        <span>Brier <b>${report.brier != null ? (+report.brier).toFixed(3) : "—"}</b></span>
        <span>AURC <b>${report.aurc != null ? (+report.aurc).toFixed(3) : "—"}</b></span>
        ${temp.available && html`<span>T <b>${(+temp.temperature).toFixed(2)}</b></span>`}
      </div>
      ${sel70.available && html`<div class="calibration-selective">
        at ${pct(sel70.coverage)} coverage: error ${pct(sel70.error_at_coverage)} vs ${pct(sel70.full_coverage_error)} full
        ${sel70.error_reduction_vs_full != null ? ` · ${pct(sel70.error_reduction_vs_full)} reduction` : ""}
      </div>`}
      ${bins.length ? html`<div class="actuary-bins">
        ${bins.map(b => html`<div class="actuary-bin" key=${b.lo}>
          <span>${(+b.lo).toFixed(1)}–${(+b.hi).toFixed(1)} · n=${b.n}</span>
          <div><i style=${`width:${pct(b.accuracy)}`}></i>
            <em style=${`left:${pct(b.mean_score)}`} title=${`mean score ${b.mean_score != null ? (+b.mean_score).toFixed(2) : "—"}`}></em></div>
          <b>correct ${pct(b.accuracy)}</b>
          ${b.ci_lo != null && b.ci_hi != null ? html`<span class="ci">CI ${pct(b.ci_lo)}–${pct(b.ci_hi)}</span>` : null}
        </div>`)}
      </div>` : html`<div class="none">no bins available</div>`}
      <p class="actuary-note">outcome-grounded: correctness on a labeled probe set, not the acceptance proxy.
        ${d.model ? ` Model: ${d.model}.` : ""}</p>
      ${d.saved_ago_s != null && html`<p class="actuary-note">saved ${Math.round(d.saved_ago_s)}s ago</p>`}
    </div>
  </section>`;
}

export function ReadModule(){
  const rec = useStore(x => x.rec);
  const requested = useStore(x => x.readRequest);
  const error = useStore(x => x.readError);
  const [selected, setSelected] = useState(null);

  useEffect(() => { setSelected(null); }, [rec && rec.id]);

  if(!rec) return html`<div class="col read-desk">
    <div class="mod read-empty">
      <div class="read-kicker">document-first read view</div>
      <h1>${error ? "Run unavailable" : "Waiting for a recorded answer"}</h1>
      <p>${error || (requested ? `Loading "${requested}" from the local journal…`
        : "Choose a run in Replay, then return here to read it as a document.")}</p>
      ${error && html`<button class="spd" onClick=${() => store.set({ route: "replay" })}>OPEN JOURNAL</button>`}
    </div>
  </div>`;

  const steps = normSteps(rec);
  const spans = sketchySpans(steps);
  const lowTokens = steps.filter(s => s.conf != null && s.conf < LOW_CONF).length;
  const unknown = steps.filter(s => s.conf == null).length;
  const chosen = selected == null ? (spans[0] || null) : spans.find(s => s.id === selected) || null;
  const answer = steps.length ? null : (rec.response || rec.reply || "No response text was recorded.");
  const finish = rec.finish_reason || "not recorded";
  const policy = rec.clozn_policy || null;

  return html`<div class="read-grid">
    <article class="mod read-document">
      <header class="read-head">
        <div><div class="read-kicker">recorded answer</div>
          <h1>${promptFor(rec)}</h1>
          ${policy && html`<div style="margin-top:9px"><${PolicyChip} policy=${policy}/></div>`}</div>
        <div class="read-actions">
          <button class="spd primary" onClick=${() => store.set({ route: "replay", P: steps.length })}>OPEN REPLAY</button>
          ${!rec._sample && html`<a class="spd read-link" href=${api.cardUrl(rec.id)} target="_blank" rel="noopener">RECEIPT CARD</a>`}
        </div>
      </header>

      <div class="read-answer" aria-label="Recorded model answer">
        ${steps.length ? steps.map((step, i) => {
          const cls = band(step), inChosen = chosen && i >= chosen.start && i < chosen.end;
          const choose = () => {
            const target = spans.find(s => i >= s.start && i < s.end); setSelected(target ? target.id : null);
          };
          return html`<span key=${i} class=${"read-token " + cls + (inChosen ? " selected" : "")}
            title=${step.conf == null ? "confidence was not captured" : `recorded token confidence ${step.conf.toFixed(2)} · commitment, not correctness`}
            role=${cls === "sketchy" ? "button" : null} tabindex=${cls === "sketchy" ? 0 : null}
            onClick=${cls === "sketchy" ? choose : null}
            onKeyDown=${cls === "sketchy" ? (e => {
              if(e.key === "Enter" || e.key === " "){ e.preventDefault(); choose(); }
            }) : null}>${step.piece}</span>`;
        }) : answer}
      </div>

      <div class="read-legend">
        <span><i class="decided"></i>≥ ${WATCH_CONF.toFixed(1)} decided</span>
        <span><i class="watch"></i>${LOW_CONF.toFixed(1)}–${WATCH_CONF.toFixed(1)} less decided</span>
        <span><i class="sketchy"></i>&lt; ${LOW_CONF.toFixed(1)} sketchy</span>
        ${unknown ? html`<span><i class="unknown"></i>${unknown} uncaptured</span>` : null}
        <b>raw token confidence measures commitment, not truth</b>
      </div>

      <footer class="read-meta">
        <span>run <b>${rec.id}</b></span><span>${shortTime(rec.created_at)}</span>
        <span>model <b>${rec.model || "not recorded"}</b></span>
        <span>finish <b>${finish}</b></span><span>${steps.length} traced tokens</span>
      </footer>
    </article>

    <div class="col">
      <aside class="mod read-zoom">
        <div class="mod-h"><span class="led coral"></span><span class="cap">zoom into sketchy spans</span>
        <span class="tail">${lowTokens}/${steps.length || 0} tokens</span></div>
        <div class="read-zoom-body">
        ${!steps.length && html`<div class="none">No token trace was captured, so this answer can be read but not confidence-shaded.</div>`}
        ${steps.length && !spans.length && html`<div class="read-clear"><b>No token fell below ${LOW_CONF.toFixed(1)}.</b>
          That means the recorded distribution was decided, not that the answer was correct.</div>`}
        ${spans.length ? html`<div class="read-span-list">
          ${spans.map(span => html`<button key=${span.id} class=${selected === span.id || (selected == null && span.id === 0) ? "on" : ""}
            onClick=${() => setSelected(span.id)}><span>“${span.text.trim() || "whitespace"}”</span>
            <b>mean ${span.mean.toFixed(2)} · min ${span.min.toFixed(2)}</b></button>`)}
        </div>` : null}

        ${chosen && html`<div class="read-detail">
          <div class="read-kicker">selected span · tokens ${chosen.start}–${chosen.end - 1}</div>
          <blockquote>${chosen.text}</blockquote>
          ${chosen.pieces.map((piece, offset) => {
            const step = steps[chosen.start + offset];
            return html`<div class="read-token-detail" key=${offset}>
              <div><b>${piece.trim() || "whitespace"}</b><span>confidence ${step.conf.toFixed(3)}</span></div>
              ${(step.alts || []).length ? html`<p>alternatives · ${step.alts.slice(0,4).map(a =>
                `${String(a.piece ?? a.token ?? "?").trim() || "whitespace"} ${Number(a.prob ?? a.p ?? 0).toFixed(3)}`).join(" · ")}</p>`
                : html`<p>no alternatives were captured for this token</p>`}
            </div>`;
          })}
          <div class="read-caveat">A low-confidence span is a locator for inspection. It is not a factuality verdict.</div>
        </div>`}
        </div>
      </aside>
      <${ActuaryPanel} rec=${rec}/>
      <${CalibrationCurve} rec=${rec}/>
    </div>
  </div>`;
}
