/* heavnOS · Memory desk — cards CRUD + review queue + propose-from-run.
   Honesty rules carried over from replay.mjs: no mock data on live runs, server reason strings
   surfaced verbatim, every mutation refreshes from the server rather than optimistically patching,
   live-mode guards on writes (this is a local single-user instrument, not a demo).
   Known gotchas: the /memory route's corrections, and /memory/edit's resync-key omission
   (this module doesn't call /memory/edit, but the same "check key presence, not shape" spirit
   applies below). */
import { html, useState, useEffect } from "../vendor/preact-standalone.mjs";
import { store, useStore, toast } from "../state.mjs";
import { api } from "../api.mjs";

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
  const [strength, setStrength] = useState(null);   // POST /memory/strength response, or null
  const [strengthDraft, setStrengthDraft] = useState(0);
  const [controlBusy, setControlBusy] = useState("");
  const [controlMessage, setControlMessage] = useState(null);
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
  const refreshMode = async () => {
    const r = await api.memoryMode();
    setModeErr(!r); if(r) setMode(r);
    return r;
  };
  const refreshStrength = async () => {
    const r = await api.memoryStrength();
    if(r && (!r.__status || r.__status < 400) && r.strength != null){
      const v = Math.max(0, Math.min(2, +r.strength));
      setStrength(r); setStrengthDraft(v);
    } else {
      setStrength(null);
    }
    return r;
  };

  useEffect(() => {
    refreshMode();
    refreshStrength();
    refreshCards();
  }, []);

  const switchMode = async target => {
    if(guardLive(live) || controlBusy || !target) return;
    setControlBusy("mode"); setControlMessage(null);
    const r = await api.memorySetMode(target);
    if(!r || r.__status >= 400 || r.ok === false){
      setControlBusy("");
      setControlMessage({ kind: "error", text: (r && (r.error || r.reason)) || "Mode switch did not reach the server." });
      return;
    }
    await Promise.all([refreshMode(), refreshStrength(), refreshCards()]);
    setControlBusy("");
    const retraining = !!(r.resync && r.resync.retraining);
    setControlMessage({ kind: retraining ? "warn" : "ok", text: target === "prompt"
      ? "Prompt mode is active. Cards now ride as readable, topic-gated context; the trained prefix was left intact."
      : `Internalized mode is active.${retraining ? " The prefix is retraining from the current cards now; this can take a few minutes." : " The existing prefix already matched the active cards."}` });
  };

  const commitStrength = async value => {
    if(guardLive(live) || controlBusy) return;
    const v = Math.max(0, Math.min(2, +value || 0));
    setControlBusy("strength"); setControlMessage(null);
    const r = await api.memoryStrength(v);
    setControlBusy("");
    if(!r || r.__status >= 400 || r.strength == null){
      setControlMessage({ kind: "error", text: (r && (r.error || r.reason)) || "Memory strength was not saved." });
      return;
    }
    const saved = Math.max(0, Math.min(2, +r.strength));
    setStrength(r); setStrengthDraft(saved);
    const activeMode = r.mode || (mode && mode.mode) || listMeta.mode;
    setControlMessage({ kind: "ok", text: activeMode === "prompt"
      ? (saved === 0 ? "Memory is off for every prompt." : "Memory is on when the topic gate admits the cards. Values above zero have the same effect in prompt mode.")
      : `Internalized memory strength saved at ${saved.toFixed(1)}.` });
  };

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
    <${MemoryControls} live=${live} mode=${mode} modeErr=${modeErr} listMeta=${listMeta}
      strength=${strength} strengthDraft=${strengthDraft} setStrengthDraft=${setStrengthDraft}
      busy=${controlBusy} message=${controlMessage} onMode=${switchMode} onStrength=${commitStrength}/>
    <${ReviewQueue} cards=${cards} act=${act} busy=${busy}/>
    <${CardsPanel} cards=${cards} act=${act} busy=${busy}
      expanded=${expanded} toggleRuns=${toggleRuns} runsCache=${runsCache}/>
    <${AnchoredShelf} cards=${cards} live=${live} rec=${rec}/>
    <${AddCard} live=${live} onAdded=${refreshCards}/>
    <${ProposeFromRun} rec=${rec} live=${live} onProposed=${refreshCards}/>
  </div>`;
}

/* ───────────────────────── F6) the anchored shelf — memory as named directions ─────────────
   X7 productized: each bag is a k-sparse decomposition into the card's OWN words; the α-table is a
   LOOKUP of what is injected (never a self-report); deleting a word is a real edit (refit). Bags ride
   LIVE CHAT turns only, as one composed steer at L21 — the validated envelope. Content only: style/rule
   cards are refused with the measured reason and belong on the dials. */
function AnchoredShelf({ cards, live, rec }){
  const [bags, setBags] = useState(null);          // null = loading
  const [envelope, setEnvelope] = useState("");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState({});            // key -> bool
  const [fitMsg, setFitMsg] = useState(null);      // last fit outcome (refusals shown verbatim)
  const [proofs, setProofs] = useState({});         // card_id -> { result } | { error }

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
  const prove = bag => withBusy("proof:" + bag.card_id, async () => {
    if(guardLive(live)) return;
    if(!rec){ toast("load a recorded run from Replay before proving this memory"); return; }
    if(bag.on === false){ toast("switch this anchored bag on before proving it"); return; }
    setProofs(p => ({ ...p, [bag.card_id]: null }));
    const r = await api.runExperiment(rec.id, { type: "anchored_recall", card_id: bag.card_id });
    if(!r){
      setProofs(p => ({ ...p, [bag.card_id]: { error: "no response from the server — is it up?" } }));
      return;
    }
    if(r.__status && r.__status >= 400){
      setProofs(p => ({ ...p, [bag.card_id]: { error: r.error || ("receipt failed (" + r.__status + ")") } }));
      return;
    }
    setProofs(p => ({ ...p, [bag.card_id]: { result: r } }));
  });

  const anchoredIds = new Set((bags || []).map(b => b.card_id));
  const anchorable = (cards || []).filter(c => c.status === "active" && !anchoredIds.has(c.id));

  return html`<div class="mod">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led lilac"></span><span class="cap">anchored shelf</span>
      <span class="tail">${bags === null ? "loading…" : bags.length + " bag(s) · " + envelope}</span>
      <span class="tag der-t">DERIVED</span></div>
    <div style="padding:2px 14px 12px;display:flex;flex-direction:column;gap:8px">
      <div class="memory-carrier-map" data-testid="memory-carrier-map">
        <div><span class="tag cap-t">ANCHORED · PRODUCT</span><b>named directions</b>
          <span>sparse α lookup; instant edits; per-card receipt attempts an equal-magnitude random control.</span></div>
        <div><span class="tag smp-t">INTERNALIZED · LAB</span><b>soft prefix</b>
          <span>opaque trained carrier; retraining can take minutes; per-card attribution unavailable.</span></div>
        <p>These are the two learned-vector carriers. Prompt cards are a separate readable-context route,
          not a learned carrier.</p>
      </div>
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
          <button class=${"spd" + (busy["proof:" + bag.card_id] ? " busy" : "")}
            disabled=${!live || !rec || bag.on === false || !!busy["proof:" + bag.card_id]}
            title="Runs baseline, anchored, and equal-magnitude random-control generations"
            onClick=${() => prove(bag)}>${busy["proof:" + bag.card_id] ? "PROVING…" : "PROVE ON THIS RUN"}</button>
          <span>cost: 2–3 fresh model generations · no KV reuse</span>
          ${bag.reconstruction_cos != null && bag.reconstruction_cos < .5
            && html`<span style="color:#C24A31;font-size:8.5px">low cos — the card's own words barely
              span the target; read this bag skeptically</span>`}
        </span>
        ${proofs[bag.card_id] && html`<${AnchoredProof} proof=${proofs[bag.card_id]} rec=${rec}/>`}
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

function truthWord(value, yes, no){
  if(value === true) return yes;
  if(value === false) return no;
  return "not computed";
}

function metric(value){
  return value == null || Number.isNaN(+value) ? "—" : (+value).toFixed(2);
}

/* A compact view of the generic Experiment envelope. The full raw receipt remains available in the
   Experiment drawer; this shelf shows the causal facts needed to judge one anchored bag in context. */
function AnchoredProof({ proof, rec }){
  if(proof.error) return html`<div class="anchored-proof error" data-testid="anchored-proof">
    <b>receipt unavailable</b><span>${proof.error}</span></div>`;
  const res = proof.result || {};
  const result = res.result || {};
  const receipt = result.receipt || {};
  const injected = receipt.injected || {};
  const hits = receipt.lexicon_hits || {};
  const lp = receipt.logprob_shift || {};
  const nul = result.null || {};
  const blocked = receipt.blocked;
  const nullAvailable = nul.available === true || receipt.null_control_available === true;
  const effectText = result.has_effect === true
    ? (nullAvailable ? "EFFECT BEYOND NULL" : "EFFECT VS BASELINE · NULL MISSING")
    : result.has_effect === false
    ? (nullAvailable ? "NO EFFECT BEYOND NULL" : "NO EFFECT VS BASELINE · NULL MISSING")
    : "not computed";
  return html`<div class=${"anchored-proof" + (blocked ? " error" : "")} data-testid="anchored-proof">
    <div class="anchored-proof-head"><b>null-controlled receipt · ${res.run_id || (rec && rec.id) || "—"}</b>
      <span class=${"tag " + (result.causal_verified === true ? "cap-t" : result.causal_verified === false ? "fail-t" : "smp-t")}>
        ${truthWord(result.causal_verified, "CAUSAL PATH VERIFIED", "NOT VERIFIED")}</span>
      <span class=${"tag " + (result.has_effect === true && nullAvailable ? "cap-t" : "smp-t")}>
        ${effectText}</span></div>
    ${blocked && html`<span><b>blocked · ${blocked}</b> — ${receipt.note || "no detail returned"}</span>`}
    ${!blocked && html`<span>${result.plain || "No plain-language summary returned."}</span>`}
    ${!blocked && !nullAvailable && html`<span class="anchored-proof-weak">
      Random-control arm unavailable — this is baseline-only, weaker evidence.</span>`}
    <div class="anchored-proof-metrics">
      <span><b>named cause</b>${injected.target_term || "—"}</span>
      <span><b>lexicon hits</b>${hits.baseline ?? "—"} → ${hits.anchored ?? "—"} · null ${hits.null ?? "—"}</span>
      <span><b>target logprob</b>${metric(lp.anchored_over_baseline_nat)} nat vs base · ${metric(lp.anchored_over_null_nat)} vs null</span>
      <span><b>coherence</b>${metric(receipt.coherence_score)}${receipt.coherent === false ? " · degraded" : ""}</span>
    </div>
    <details><summary>compare the three replies</summary>
      <div class="anchored-proof-arms">
        <span><b>baseline</b>${(res.baseline && res.baseline.reply) || receipt.baseline_reply || "—"}</span>
        <span><b>anchored</b>${result.changed_reply || receipt.anchored_reply || "—"}</span>
        <span><b>random control</b>${nul.reply || receipt.null_reply || "—"}</span>
      </div>
      <p>${nul.note || receipt.null_note || "The random arm was not available; treat this as weaker evidence."}</p>
    </details>
  </div>`;
}

/* ───────────────────────── A) mode strip ───────────────────────── */
function MemoryControls({ live, mode, modeErr, listMeta, strength, strengthDraft, setStrengthDraft,
                          busy, message, onMode, onStrength }){
  const m = (mode && mode.mode) || listMeta.mode || null;
  const modes = (mode && Array.isArray(mode.modes) && mode.modes.length) ? mode.modes : (m ? [m] : []);
  const copy = m === "internalized"
    ? "a trained soft prefix — not self-reportable, per-card ablation not possible"
    : m === "prompt"
    ? "cards ride the prompt, topic-gated per turn — per-card receipts work here"
    : null;
  const retraining = listMeta.retraining;
  const strengthMode = (strength && strength.mode) || m;
  return html`<div class="mod" data-testid="memory-controls">
    <div class="mod-h"><span class="led"></span><span class="cap">memory controls</span>
      <span class="tail">${busy ? "saving..." : m || (modeErr ? "unreachable" : "loading...")}</span>
      ${retraining && retraining.active && html`<span class="tag der-t">RETRAINING</span>`}</div>
    <div class="memory-controls-body">
      <section class="memory-mode-control">
        <div class="memory-control-title"><b>carrier</b>
          <span class=${"tag " + (m === "prompt" ? "cap-t" : m === "internalized" ? "smp-t" : "fail-t")}>
            ${m ? m.toUpperCase() : "—"}</span></div>
        ${copy && html`<div class="memory-control-copy">${copy}</div>`}
        ${modeErr && html`<div class="memory-control-message error">GET /memory/mode did not answer.</div>`}
        ${modes.length > 1 ? html`<div class="memory-mode-options">
          ${modes.map(x => html`<button class=${"spd " + (x === m ? "primary" : "")}
            disabled=${!live || !!busy || x === m} onClick=${() => onMode(x)}>${x.toUpperCase()}</button>`)}
        </div>` : modes.length === 1 ? html`<div class="memory-control-copy">
          This runtime exposes only <b>${modes[0]}</b> memory. Internalized soft-prefix memory is lab-only.</div>` : null}
        <div class="memory-mode-truth">
          <span><b>prompt</b> readable context; edits instant; per-card receipts work.</span>
          <span><b>internalized</b> lab-only soft prefix; card changes can retrain for minutes.</span>
        </div>
      </section>

      <section class="memory-strength-control">
        <div class="memory-control-title"><b>strength</b><output>${strength ? strengthDraft.toFixed(1) : "—"}</output></div>
        <input data-testid="memory-strength" type="range" min="0" max="2" step="0.1"
          value=${strengthDraft} disabled=${!live || !strength || !!busy}
          onInput=${e => setStrengthDraft(Math.max(0, Math.min(2, +e.target.value)))}
          onChange=${e => onStrength(+e.target.value)}/>
        <div class="memory-strength-ticks"><span>0 · off</span><span>1 · normal</span><span>2 · max</span></div>
        <div class="memory-control-copy">${strengthMode === "prompt"
          ? "Prompt mode is binary: 0 keeps cards out; any value above 0 injects them only when the topic gate admits them. Positive values do not scale intensity."
          : "Internalized mode is continuous: 0 is off, 1 is trained strength, and values above 1 bite harder. Saving does not retrain."}</div>
        ${!strength && html`<div class="memory-control-message error">POST /memory/strength did not answer.</div>`}
        ${strength && html`<div class="memory-control-copy">${strength.has_prefix
          ? "A trained prefix is present on this substrate." : "No trained prefix is present; prompt cards can still work."}</div>`}
      </section>
      ${message && html`<div class=${"memory-control-message " + message.kind}>${message.text}</div>`}
    </div>
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
  /* review finding #13: the card shape (contracts §14, clozn/memory/cards.py:78-93) has no
     `relevance` field -- relevance is a per-RUN quantity (run.memory.relevance[], parallel to
     cards_applied[]; see Minfl in replay.mjs, which reads it correctly off the run). There is no per-card
     relevance to show here, so this used to always render a blank value; dropped rather than
     displaying a field that can never be populated. */
  return html`<div class="steer-row">
    <span>${card.text}
      <span class=${"tag " + (card.status === "active" ? "cap-t" : "smp-t")} style="margin-left:6px">
        ${card.status.toUpperCase()}</span>
    </span>
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
