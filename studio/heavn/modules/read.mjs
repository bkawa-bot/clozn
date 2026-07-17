/* heavnOS · Read — the document-first permalink landing. Everything here comes from the stored run:
   no generation, no self-report, and no on-demand model call. Confidence is shown only as recorded token
   commitment; it is never promoted to correctness. */
import { html, useState, useEffect } from "../vendor/preact-standalone.mjs";
import { store, useStore, normSteps, shortTime } from "../state.mjs";
import { api } from "../api.mjs";

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

  return html`<div class="read-grid">
    <article class="mod read-document">
      <header class="read-head">
        <div><div class="read-kicker">recorded answer</div>
          <h1>${promptFor(rec)}</h1></div>
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
  </div>`;
}
