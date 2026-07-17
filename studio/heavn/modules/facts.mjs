/* heavnOS · Facts panel — the optional per-profile cue→answer slot store.

   This is deliberately separate from trait cards: facts are verbatim source pairs compiled into a
   model-specific slot store. The tier stays off by default because reads cost an extra model pass. v1
   exposes the read receipt and measured latency but does not alter chat replies. Writes show the
   surprise-gate refusal; reads show confident hits or honest abstentions. */
import { html, useEffect, useState } from "../vendor/preact-standalone.mjs";
import { api } from "../api.mjs";

const good = response => !!response
  && (response.__status == null || response.__status < 400)
  && response.ok !== false;

const reason = (response, fallback) => {
  const err = response && (response.reason || response.error);
  if(typeof err === "string") return err;
  if(err && typeof err.message === "string") return err.message;
  return fallback;
};

const fmt = value => {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(3).replace(/0+$/, "").replace(/\.$/, "") : "—";
};

export function FactsPanel({ live }){
  const [loaded, setLoaded] = useState(false);
  const [available, setAvailable] = useState(true);
  const [enabled, setEnabled] = useState(false);
  const [entries, setEntries] = useState([]);
  const [meta, setMeta] = useState({ profile: null, layer: null, count: 0 });
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState(null);
  const [confirmDelete, setConfirmDelete] = useState(null);
  const [cue, setCue] = useState("");
  const [answer, setAnswer] = useState("");
  const [query, setQuery] = useState("");
  const [receipt, setReceipt] = useState(null);

  const say = (kind, text) => setMessage({ kind, text });
  const refresh = async () => {
    if(!live){
      setAvailable(false); setLoaded(true); setEnabled(false); setEntries([]);
      setMeta({ profile: null, layer: null, count: 0 });
      return false;
    }
    const response = await api.factsList();
    if(!response || (response.__status != null && response.__status >= 400)){
      setAvailable(false); setLoaded(true); setEntries([]);
      return false;
    }
    const list = Array.isArray(response.entries) ? response.entries : [];
    setAvailable(true); setLoaded(true); setEnabled(response.enabled === true); setEntries(list);
    setMeta({ profile: response.profile || "default", layer: response.layer ?? null,
      count: response.count ?? list.length });
    return true;
  };

  useEffect(() => { refresh(); }, [live]);

  const toggleMode = async () => {
    if(!live || busy) return;
    const next = !enabled;
    setBusy("mode"); setConfirmDelete(null); setReceipt(null);
    say("info", next ? "Enabling the per-profile slot store…" : "Disabling fact reads…");
    const response = await api.factsMode(next);
    if(good(response)){
      await refresh();
      say("ok", next
        ? "Facts are on. Reads now cost one extra model pass; their latency is shown in every receipt."
        : "Facts are off. The stored source remains on disk, but chat pays no slot-read cost.");
    }else say("error", reason(response, "The facts setting could not be changed."));
    setBusy("");
  };

  const addFact = async event => {
    event.preventDefault();
    if(!live || !enabled || busy) return;
    const cleanCue = cue.trim(), cleanAnswer = answer.trim();
    if(!cleanCue || !cleanAnswer){ say("warn", "Enter both a cue and an answer."); return; }
    setBusy("add"); setConfirmDelete(null);
    const response = await api.factsAdd(cleanCue, " " + cleanAnswer);
    if(good(response) && response.written === true){
      setCue(""); setAnswer(""); await refresh();
      say("ok", `Stored. Surprise ${fmt(response.surprise)}: the model did not already know it.`);
    }else if(good(response) && response.written === false){
      say("warn", reason(response, `Not stored: surprise ${fmt(response.surprise)} was below the write gate.`));
    }else say("error", reason(response, "That fact could not be stored."));
    setBusy("");
  };

  const deleteFact = async (entry, index) => {
    if(!live || !enabled || busy) return;
    const key = index + ":" + entry.cue;
    if(confirmDelete !== key){
      setConfirmDelete(key);
      say("info", `Press CONFIRM DELETE beside “${entry.cue}” once more.`);
      return;
    }
    setBusy("delete:" + key);
    const response = await api.factsDelete(entry.cue);
    if(good(response)){
      setConfirmDelete(null); await refresh();
      say("ok", `Deleted “${entry.cue}”. ${response.remaining ?? 0} fact(s) remain.`);
    }else say("error", reason(response, "That fact could not be deleted."));
    setBusy("");
  };

  const readFact = async event => {
    event.preventDefault();
    if(!live || !enabled || busy || !query.trim()) return;
    setBusy("read"); setConfirmDelete(null); setReceipt(null);
    const response = await api.factsRead(query.trim());
    if(response && (response.__status == null || response.__status < 400)) setReceipt(response);
    else say("error", reason(response, "The fact store could not read that cue."));
    setBusy("");
  };

  const status = !loaded ? "loading…" : !available ? "unavailable" : enabled ? "on" : "off";
  return html`<section class="mod facts-panel" aria-labelledby="facts-title">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class=${"led " + (enabled ? "blue" : "")}></span>
      <span class="cap" id="facts-title">facts</span>
      <span class="tail">${loaded && available ? `${meta.count} stored · ${meta.profile || "default"}` : status}</span>
      <span class=${"tag " + (enabled ? "cap-t" : "smp-t")}>${status.toUpperCase()}</span></div>

    <div class="facts-body" data-testid="facts-panel">
      <div class="facts-topline">
        <div class="facts-copy"><b>A fact is a verbatim cue → answer</b> compiled into a per-profile slot
          store. It is separate from the trait cards above, which carry dispositions.</div>
        ${available && html`<button class=${"spd" + (busy === "mode" ? " busy" : "")} type="button"
          disabled=${!live || !!busy} onClick=${toggleMode}>${enabled ? "TURN OFF" : "TURN ON"}</button>`}
      </div>
      <div class="facts-latency"><b>OFF BY DEFAULT</b> · each read is an extra model pass. v1 records the
        hit or abstention and its <b>slot_ms</b>; it does not change the chat reply.</div>

      ${loaded && !available && html`<div class="none facts-unavailable">The fact store is not available
        ${live ? "on this backend." : "in sample mode; no persona state is written."}</div>`}

      ${enabled && available && html`<div class="facts-on">
        <div class="facts-rule">Writes are <b>surprise-gated</b>: known facts are refused.
          <span> Reads </span><b>abstain</b> when no source matches confidently.</div>
        <div class="fact-list" data-testid="fact-list">
          ${!entries.length && html`<div class="none">No facts in this profile yet.</div>`}
          ${entries.map((entry, index) => {
            const key = index + ":" + entry.cue;
            return html`<div class="fact-row" data-fact-index=${index} key=${key}>
              <span class="fact-cue">${entry.cue}</span><span class="fact-arrow">→</span>
              <b class="fact-answer">${String(entry.answer || "").trim() || "(empty)"}</b>
              <button class=${"spd danger" + (confirmDelete === key ? " armed" : "")} type="button"
                disabled=${!!busy} onClick=${() => deleteFact(entry, index)}>
                ${confirmDelete === key ? "CONFIRM DELETE" : "DELETE"}</button>
            </div>`;
          })}
        </div>

        <form class="fact-form" onSubmit=${addFact}>
          <input aria-label="fact cue" value=${cue} placeholder="cue — e.g. My dog is named"
            disabled=${!!busy} onInput=${event => setCue(event.currentTarget.value)}/>
          <input aria-label="fact answer" value=${answer} placeholder="answer — e.g. Biscuit"
            disabled=${!!busy} onInput=${event => setAnswer(event.currentTarget.value)}/>
          <button class=${"spd primary" + (busy === "add" ? " busy" : "")} disabled=${!!busy}>STORE</button>
        </form>

        <form class="fact-form fact-read-form" onSubmit=${readFact}>
          <input aria-label="fact read query" value=${query}
            placeholder="test a read — type a cue and inspect the receipt" disabled=${!!busy}
            onInput=${event => setQuery(event.currentTarget.value)}/>
          <button class=${"spd" + (busy === "read" ? " busy" : "")} disabled=${!!busy}>READ</button>
        </form>
        ${receipt && html`<${FactReceipt} receipt=${receipt}/>`}
      </div>`}
      ${message && html`<div class=${"fact-message " + message.kind} role="status" aria-live="polite">
        ${message.text}</div>`}
    </div>
  </section>`;
}

function FactReceipt({ receipt }){
  if(receipt.enabled === false) return html`<div class="fact-receipt" data-testid="fact-receipt">
    The facts tier is off.</div>`;
  if(receipt.empty) return html`<div class="fact-receipt" data-testid="fact-receipt">
    The store is empty — nothing to retrieve yet.</div>`;
  const abstained = receipt.abstained === true || receipt.hit == null;
  const metadata = [];
  if(receipt.sim != null) metadata.push("sim " + fmt(receipt.sim));
  if(receipt.gate_floor != null) metadata.push("floor " + fmt(receipt.gate_floor));
  if(receipt.slot_ms != null) metadata.push(fmt(receipt.slot_ms) + " ms");
  if(receipt.count != null) metadata.push(receipt.count + " stored");
  return html`<div class=${"fact-receipt " + (abstained ? "abstain" : "hit")}
      data-testid="fact-receipt">
    <div><b>${abstained ? "ABSTAINED" : "HIT"}</b>${abstained
      ? " — no stored fact matched confidently, so the store stayed silent rather than guessing."
      : html`: “${receipt.cue || ""}” → `}<b>${abstained ? "" : String(receipt.answer || "").trim()}</b></div>
    ${metadata.length && html`<div class="fact-receipt-meta">${metadata.join(" · ")}</div>`}
  </div>`;
}
