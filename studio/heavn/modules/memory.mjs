/* heavnOS · Memory desk — cards CRUD + review queue + propose-from-run.
   Honesty rules carried over from replay.mjs: no mock data on live runs, server reason strings
   surfaced verbatim, every mutation refreshes from the server rather than optimistically patching,
   live-mode guards on writes (this is a local single-user instrument, not a demo).
   Contracts: notes/HEAVN_API_CONTRACTS.md §14 (Memory CRUD), §13 (propose-memory), §23 Gotchas
   (esp. #1's /memory route corrections and #13's /memory/edit resync-key omission — this module
   doesn't call /memory/edit, but the same "check key presence, not shape" spirit applies below). */
import { html, useState, useEffect } from "../vendor/preact-standalone.mjs";
import { store, useStore, toast } from "../state.mjs";
import { api } from "../api.mjs";

/* api.mjs has no GET /memory/mode wrapper — a tiny local fetch, kept inside this file only
   (contracts §14: `{"mode": "prompt", "modes": ["prompt","internalized"]}`, never throws). */
async function fetchMemoryMode(){
  try{
    const r = await fetch("/memory/mode");
    if(!r.ok) return null;
    return await r.json();
  }catch(e){ return null; }
}

const guardLive = live => {
  if(!live){ toast("live server only"); return true; }
  return false;
};

/* ───────────────────────── module root ───────────────────────── */
export function MemoryModule(){
  const live = useStore(x => x.live);
  const rec = useStore(x => x.rec);

  const [mode, setMode] = useState(null);           // GET /memory/mode response, or null
  const [modeErr, setModeErr] = useState(false);
  const [cards, setCards] = useState(null);          // null = loading; [] = loaded, empty
  const [listMeta, setListMeta] = useState({ has_prefix: false, mode: null, retraining: null });
  const [busy, setBusy] = useState({});               // "<id>:<verb>" -> bool
  const [expanded, setExpanded] = useState({});       // id -> bool
  const [runsCache, setRunsCache] = useState({});     // id -> "loading" | "error" | array

  const refreshCards = async () => {
    const r = await api.memoryList();                 // POST /memory/cards
    if(r && Array.isArray(r.cards)){
      setCards(r.cards);
      setListMeta({ has_prefix: !!r.has_prefix, mode: r.mode || null, retraining: r.retraining || null });
    } else {
      setCards([]);
    }
  };
  const refreshMode = async () => setMode(await fetchMemoryMode());

  useEffect(() => {
    (async () => {
      const m = await fetchMemoryMode();
      if(!m) setModeErr(true);
      setMode(m);
    })();
    refreshCards();
  }, []);

  const act = async (id, verb) => {
    if(guardLive(live)) return;
    const key = id + ":" + verb;
    setBusy(b => ({ ...b, [key]: true }));
    const r = await api.memoryAct(id, verb);           // POST /memory/<verb> {id}
    setBusy(b => ({ ...b, [key]: false }));
    if(!r || r.ok === false){
      toast(`${verb} — ${(r && r.reason) || "no response from the server"}`);
    }
    await refreshCards();
  };

  const toggleRuns = async id => {
    const isOpen = !!expanded[id];
    setExpanded(e => ({ ...e, [id]: !isOpen }));
    if(isOpen) return;                                  // closing — nothing to fetch
    if(runsCache[id] !== undefined) return;              // already fetched (incl. error/empty)
    setRunsCache(c => ({ ...c, [id]: "loading" }));
    const r = await api.memoryRuns(id);                  // GET /memory/<id>/runs
    setRunsCache(c => ({ ...c, [id]: (r && Array.isArray(r.runs)) ? r.runs : "error" }));
  };

  return html`<div class="col">
    <${ModeStrip} mode=${mode} modeErr=${modeErr} listMeta=${listMeta}/>
    <${ReviewQueue} cards=${cards} act=${act} busy=${busy}/>
    <${CardsPanel} cards=${cards} act=${act} busy=${busy}
      expanded=${expanded} toggleRuns=${toggleRuns} runsCache=${runsCache}/>
    <${AnchoredShelf} cards=${cards} live=${live}/>
    <${AddCard} live=${live} onAdded=${refreshCards}/>
    <${ProposeFromRun} rec=${rec} live=${live} onProposed=${refreshCards}/>
  </div>`;
}

/* ───────────────────────── F6) the anchored shelf — memory as named directions ─────────────
   X7 productized: each bag is a k-sparse decomposition into the card's OWN words; the α-table is a
   LOOKUP of what is injected (never a self-report); deleting a word is a real edit (refit). Bags ride
   LIVE CHAT turns only, as one composed steer at L21 — the validated envelope. Content only: style/rule
   cards are refused with the measured reason and belong on the dials. */
function AnchoredShelf({ cards, live }){
  const [bags, setBags] = useState(null);          // null = loading
  const [envelope, setEnvelope] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState({});            // key -> bool
  const [fitMsg, setFitMsg] = useState(null);      // last fit outcome (refusals shown verbatim)

  const refresh = async () => {
    const r = await api.anchoredList();
    if(r){ setBags(r.bags || []); setEnvelope(r.envelope || ""); }
    else setBags([]);
    const wl = await api.whatlearned();
    if(wl && wl.note) setNote(wl.note);
  };
  useEffect(() => { if(live) refresh(); else setBags([]); }, [live]);

  const withBusy = async (key, fn) => {
    if(busy[key]) return;
    setBusy(b => ({ ...b, [key]: true }));
    await fn();
    setBusy(b => ({ ...b, [key]: false }));
  };
  const fit = card => withBusy("fit:" + card.id, async () => {
    setFitMsg(null);
    const r = await api.anchoredFit(card.id);
    if(!r){ setFitMsg({ ok: false, msg: "no answer — is the engine up? (fitting resolves word directions through it)" }); return; }
    if(r.refused){ setFitMsg({ ok: false, msg: r.reason }); return; }
    setFitMsg({ ok: true, msg: "anchored — " + (((r.bag || {}).terms) || []).length + " word-direction(s)" });
    await refresh();
  });
  const toggle = bag => withBusy("tg:" + bag.card_id, async () => {
    await api.anchoredToggle(bag.card_id, !(bag.on !== false)); await refresh();
  });
  const delTerm = (bag, token) => withBusy("del:" + bag.card_id + ":" + token, async () => {
    const r = await api.anchoredDeleteTerm(bag.card_id, token);
    if(r && r.ok === false) toast("delete — " + (r.reason || "failed"));
    else if(r && r.deleted_bag) toast("last word removed — the whole memory is gone (an empty memory is no memory)");
    await refresh();
  });

  const anchoredIds = new Set((bags || []).map(b => b.card_id));
  const anchorable = (cards || []).filter(c => c.status === "active" && !anchoredIds.has(c.id));

  return html`<div class="mod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led lilac"></span><span class="cap">anchored shelf</span>
      <span class="tail">${bags === null ? "loading…" : bags.length + " bag(s) · " + envelope}</span>
      <span class="tag der-t">DERIVED</span></div>
    <div style="padding:2px 14px 12px;display:flex;flex-direction:column;gap:8px">
      <span class="none" style="font-size:8.5px">memory as named word-directions: “what do you
        remember?” is a lookup of this table, never a generation. Bags ride LIVE chat turns as one
        composed steer (L21, s=0.5 — the measured envelope). Content only — style routes to dials.</span>
      ${(bags || []).map(bag => html`<div class="receipt-row" key=${bag.card_id}>
        <span>${(bag.card_text || bag.card_id).slice(0, 60)}
          <span class=${"tag " + (bag.on !== false ? "cap-t" : "smp-t")} style="margin-left:6px">
            ${bag.on !== false ? "RIDING" : "OFF"}</span></span>
        <span class="nats">cos ${bag.reconstruction_cos != null ? (+bag.reconstruction_cos).toFixed(2) : "—"}</span>
        <span class="sub" style="grid-column:1/-1;flex-wrap:wrap">
          ${(bag.terms || []).map(t => html`<span class="jchip d1" key=${t.token}
              style="font-size:9px;display:inline-flex;align-items:center;gap:3px">
            ${t.token} ${t.alpha != null ? (t.alpha >= 0 ? "+" : "") + (+t.alpha).toFixed(2) : ""}
            <button title="delete this word from the memory (refits the rest — a real edit)"
              style="border:none;background:none;cursor:pointer;font-size:8px;color:var(--coral);padding:0"
              onClick=${() => delTerm(bag, t.token)}>✕</button></span>`)}
        </span>
        <span class="sub" style="grid-column:1/-1">
          <button class=${"spd" + (busy["tg:" + bag.card_id] ? " busy" : "")}
            onClick=${() => toggle(bag)}>${bag.on !== false ? "SWITCH OFF" : "SWITCH ON"}</button>
          ${bag.reconstruction_cos != null && bag.reconstruction_cos < .5
            && html`<span style="color:#C24A31;font-size:8.5px">low cos — the card's own words barely
              span the target; read this bag skeptically</span>`}
        </span>
      </div>`)}
      ${bags !== null && !bags.length && html`<span class="none">no anchored bags yet — anchor an
        active content card below.</span>`}
      ${anchorable.length ? html`<div style="display:flex;flex-direction:column;gap:4px">
        <span class="cap" style="font-size:8.5px;color:var(--mist)">anchorable (active cards)</span>
        ${anchorable.slice(0, 6).map(c => html`<div class="steer-row" key=${c.id}>
          <span style="font-size:9.5px">${c.text.slice(0, 56)}</span>
          <button class=${"spd" + (busy["fit:" + c.id] ? " busy" : "")}
            onClick=${() => fit(c)}>⚓ ANCHOR</button>
        </div>`)}
      </div>` : null}
      ${fitMsg && html`<div class="cfg" style=${"border-left-color:" + (fitMsg.ok ? "var(--teal)" : "var(--coral)")}>
        <span style=${fitMsg.ok ? "" : "color:#C24A31"}>${fitMsg.msg}</span></div>`}
      ${note && html`<span class="none" style="font-size:8px">${note}</span>`}
    </div>
  </div>`;
}

/* ───────────────────────── A) mode strip ───────────────────────── */
function ModeStrip({ mode, modeErr, listMeta }){
  const m = (mode && mode.mode) || listMeta.mode || null;
  const copy = m === "internalized"
    ? "a trained soft prefix — not self-reportable, per-card ablation not possible"
    : m === "prompt"
    ? "cards ride the prompt, topic-gated per turn — per-card receipts work here"
    : null;
  const retraining = listMeta.retraining;
  return html`<div class="cfg">
    <span class="cap">memory mode</span><b>${m || (modeErr ? "unreachable" : "—")}</b>
    ${copy && html`<span>${copy}</span>`}
    ${retraining && retraining.active && html`<span class="tag der-t">RETRAINING</span>`}
    ${modeErr && html`<span style="color:#C24A31">GET /memory/mode didn't answer — is the server up?</span>`}
    <span style="margin-left:auto" class=${"tag " + (m === "prompt" ? "cap-t" : m === "internalized" ? "smp-t" : "fail-t")}>
      ${m ? m.toUpperCase() : "—"}</span>
  </div>`;
}

/* ───────────────────────── B) review queue ───────────────────────── */
function ReviewQueue({ cards, act, busy }){
  const pending = (cards || []).filter(c => c.status === "pending");
  const loading = cards === null;
  return html`<div class="mod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led"></span><span class="cap">review queue</span>
      <span class="tail">${loading ? "loading…" : pending.length + " pending"}</span></div>
    <div class="receipts-box">
      ${loading && html`<div class="none" style="padding:6px 0 10px">reading the card store…</div>`}
      ${!loading && !pending.length && html`<div class="none" style="padding:6px 0 10px">Nothing awaits review.</div>`}
      ${!loading && pending.map(c => {
        const hasSpan = !!c.quoted_span;
        const claimsSource = !!c.source_run_id;
        const unbacked = claimsSource && !hasSpan;
        const kApprove = c.id + ":approve", kReject = c.id + ":reject";
        return html`<div class="receipt-row" key=${c.id}>
          <span>${c.text}</span>
          <span class="tag smp-t">PENDING</span>
          <span class="sub">
            ${claimsSource
              ? html`<span>source ${c.source_run_id}${c.source_turn != null ? " · turn " + c.source_turn : ""}</span>`
              : html`<span>no recorded source</span>`}
            ${hasSpan
              ? html`<span>receipt: “${c.quoted_span}”</span>`
              : (claimsSource
                  ? html`<span style="color:#C24A31">claims a source but has no quoted span — approval will be refused by the server</span>`
                  : null)}
            ${c.risk === "suspicious" && html`<span style="color:#C24A31">risk: suspicious</span>`}
          </span>
          <span class="sub" style="grid-column:1/-1">
            <button class=${"spd" + (busy[kApprove] ? " busy" : "")}
              style=${unbacked ? "opacity:.55" : ""}
              onClick=${() => act(c.id, "approve")}>APPROVE</button>
            <button class=${"spd" + (busy[kReject] ? " busy" : "")}
              onClick=${() => act(c.id, "reject")}>REJECT</button>
          </span>
        </div>`;
      })}
    </div>
  </div>`;
}

/* ───────────────────────── C) cards (non-pending) ───────────────────────── */
function CardsPanel({ cards, act, busy, expanded, toggleRuns, runsCache }){
  const loading = cards === null;
  const nonPending = (cards || []).filter(c => c.status !== "pending");
  const active = nonPending.filter(c => c.status === "active");
  const disabled = nonPending.filter(c => c.status === "disabled");
  const rejected = nonPending.filter(c => c.status === "rejected");
  const groups = [
    { label: "active", list: active, actions: [["disable", "DISABLE"], ["remove", "REMOVE"]] },
    { label: "disabled", list: disabled, actions: [["enable", "ENABLE"], ["remove", "REMOVE"]] },
  ];
  if(rejected.length) groups.push({ label: "rejected", list: rejected, actions: [["remove", "REMOVE"]] });

  return html`<div class="mod">
    <div class="mod-h"><span class="led"></span><span class="cap">cards</span>
      <span class="tail">${loading ? "loading…" : nonPending.length + " total"}</span></div>
    ${loading && html`<div class="none" style="padding:6px 14px 12px">reading the card store…</div>`}
    ${!loading && !nonPending.length && html`<div class="none" style="padding:6px 14px 12px">
      It remembers nothing. Teach it something true.</div>`}
    ${!loading && groups.map(g => g.list.length ? html`<div key=${g.label}>
        <div class="none" style="padding:8px 14px 2px;text-transform:uppercase;letter-spacing:.1em">
          ${g.label} (${g.list.length})</div>
        ${g.list.map(c => html`<${CardRow} key=${c.id} card=${c} actions=${g.actions} act=${act} busy=${busy}
          expanded=${!!expanded[c.id]} onToggleRuns=${() => toggleRuns(c.id)} runsData=${runsCache[c.id]}/>`)}
      </div>` : null)}
  </div>`;
}

function CardRow({ card, actions, act, busy, expanded, onToggleRuns, runsData }){
  return html`<div class="steer-row">
    <span>${card.text}
      <span class=${"tag " + (card.status === "active" ? "cap-t" : "smp-t")} style="margin-left:6px">
        ${card.status.toUpperCase()}</span>
    </span>
    <span class="v">${card.relevance != null ? (+card.relevance).toFixed(2) : ""}</span>
    <span style="grid-column:1/-1;display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:4px">
      ${actions.map(([verb, label]) => html`<button key=${verb}
          class=${"spd" + (busy[card.id + ":" + verb] ? " busy" : "")}
          onClick=${() => act(card.id, verb)}>${label}</button>`)}
      <button class="spd" onClick=${onToggleRuns}>${expanded ? "▾ runs" : "▸ runs"}</button>
    </span>
    ${expanded && html`<div class="none" style="grid-column:1/-1;padding:4px 0 0">
      ${runsData === undefined || runsData === "loading" ? "listening back…"
        : runsData === "error" ? "couldn't reach the server for run history."
        : (runsData.length
            ? runsData.map(r => html`<div key=${r.id}>· ${r.id} — ${r.prompt_summary || r.response_summary || "—"}</div>`)
            : "no runs recorded as using this card yet.")}
    </div>`}
  </div>`;
}

/* ───────────────────────── D) add a card ───────────────────────── */
function AddCard({ live, onAdded }){
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);

  const submit = async () => {
    if(guardLive(live)) return;
    const t = text.trim();
    if(!t) return;
    setBusy(true);
    const r = await api.memoryAdd({ text: t });          // POST /memory/add {text}
    setBusy(false);
    if(!r || r.ok === false){
      setResult({ ok: false, reason: (r && r.reason) || "no response from the server" });
      return;
    }
    setResult({ ok: true, card: r, dial_suggestion: r.dial_suggestion || null });
    setText("");
    await onAdded();
  };

  return html`<div class="mod">
    <div class="mod-h"><span class="led"></span><span class="cap">add a card</span>
      <span class="tail">POST /memory/add</span></div>
    <div style="display:flex;gap:8px;padding:2px 14px 12px;align-items:center">
      <input type="text" value=${text} placeholder="teach it something true…"
        onInput=${e => setText(e.target.value)}
        onKeyDown=${e => { if(e.key === "Enter") submit(); }}
        style="flex:1;font-family:var(--mono);font-size:10.5px;color:var(--navy);
               border:1px solid var(--edge);border-radius:7px;padding:6px 10px;
               background:linear-gradient(180deg,#fff,#E6F1F7)"/>
      <button class=${"spd" + (busy ? " busy" : "")} onClick=${submit}>ADD</button>
    </div>
    ${result && html`<div class="cfg" style=${"margin:0 14px 12px;border-left-color:" + (result.ok ? "var(--teal)" : "var(--coral)")}>
      ${result.ok
        ? html`<span>added — <b>${result.card.status}</b> · id ${result.card.id}</span>
            ${result.dial_suggestion && html`<span>dial suggestion:
              <b>${result.dial_suggestion.axis}</b> → ${result.dial_suggestion.value}
              (${result.dial_suggestion.pole_label})</span>`}`
        : html`<span style="color:#C24A31">${result.reason}</span>`}
    </div>`}
  </div>`;
}

/* ───────────────────────── E) propose from this run ───────────────────────── */
function ProposeFromRun({ rec, live, onProposed }){
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);

  const run = async () => {
    if(guardLive(live)) return;
    setBusy(true);
    const r = await api.proposeMemory(rec.id);            // POST /runs/<id>/propose-memory
    setBusy(false);
    if(!r){
      setResult({ proposed: false, reason: "no response from the server — is it up?" });
      return;
    }
    setResult(r);
    if(r.proposed === true) await onProposed();
  };

  /* contracts §13: every outcome is HTTP 200; success uses "proposed", failures use either
     "ok":false or "proposed":false — never a single unified success key. Render honestly, per case. */
  let ok = null, msg = null;
  if(result){
    if(result.proposed === true){
      ok = true;
      const c = result.card || {};
      msg = `proposed — “${c.text}” (status ${c.status})`
        + (c.quoted_span ? ` · receipt: “${c.quoted_span}”` : " · no quoted span recorded")
        + (result.dial_suggestion
            ? ` · dial suggestion: ${result.dial_suggestion.axis} → ${result.dial_suggestion.value} (${result.dial_suggestion.pole_label})`
            : "");
    } else {
      ok = false;
      msg = result.reason || "proposal failed (no reason given)";
    }
  }

  return html`<div class="mod">
    <span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led"></span><span class="cap">propose from this run</span>
      <span class="tail">${rec ? rec.id : "no run loaded"}</span></div>
    <div style="padding:2px 14px 12px">
      ${!rec
        ? html`<div class="none">no current run loaded — open the Replay desk and pick one first.</div>`
        : html`<button class=${"spd" + (busy ? " busy" : "")} onClick=${run}>
            ${busy ? "PROPOSING…" : "PROPOSE FROM THIS RUN"}</button>`}
      ${result && html`<div class="cfg" style=${"margin-top:10px;border-left-color:" + (ok ? "var(--teal)" : "var(--coral)")}>
        <span style=${ok ? "" : "color:#C24A31"}>${msg}</span>
      </div>`}
    </div>
  </div>`;
}
