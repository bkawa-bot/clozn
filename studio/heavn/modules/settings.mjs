/* heavnOS · Settings desk — named profile CRUD over the already-live /profiles/* API.

   A profile is a portable source bundle, not a cached vector: active memory-card text, built-in dial
   values, custom-dial pole recipes, and fact cue/answer pairs. Saving snapshots the live cards+dials;
   updating preserves a profile's fact sources because there is no honest way to reconstruct those from
   the active vector store. Switching is replacement, never blending. Every mutation refreshes from the
   server, and server-written errors are shown verbatim — including the switch receipt itself: a
   successful /profiles/switch is rendered close to verbatim (cards replaced, dials applied, whether a
   background retrain kicked off, and the actual prompt_block the profile injects) rather than reduced
   to a generic "done".

   Two more cards close out the desk below the profile grid. They are NOT more CRUD — they're fixed,
   honest facts about this install: RUNTIME (the one substrate this server runs, its model, and a
   copyable OpenAI-compatible endpoint — no substrate-switcher dropdown, because POST /substrate now
   returns 410 and a dead control is worse than no control) and COUNTS (runs / memories / active dials,
   read live off the same APIs the rest of heavn uses, "—" when unavailable, never fabricated). */
import { html, useEffect, useRef, useState } from "../vendor/preact-standalone.mjs";
import { useStore, toast } from "../state.mjs";
import { api } from "../api.mjs";

const slug = value => String(value || "").trim().toLowerCase()
  .replace(/[^a-z0-9_-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 32);

const good = response => !!response
  && (response.__status == null || response.__status < 400)
  && response.ok !== false;

const reason = (response, fallback) => {
  const err = response && response.error;
  if(typeof err === "string") return err;
  if(err && typeof err.message === "string") return err.message;
  return fallback;
};

const profileCounts = profile => {
  const dialNames = new Set(Object.keys(profile.dials || {}));
  (profile.custom_dials || []).forEach(dial => dialNames.add(dial.name));
  return {
    cards: (profile.cards || []).length,
    dials: dialNames.size,
    facts: (profile.facts || []).length,
  };
};

async function copyText(text){
  try{ await navigator.clipboard.writeText(text); return true; }catch(e){ return false; }
}

export function SettingsModule(){
  const live = useStore(state => state.live);
  const rec = useStore(state => state.rec);
  const fileInput = useRef(null);
  const [profiles, setProfiles] = useState(null);       // null = loading, [] = loaded empty/offline
  const [active, setActive] = useState(null);
  const [busy, setBusy] = useState("");
  const [message, setMessage] = useState(null);         // {kind: ok|error|info, text}
  const [switchResult, setSwitchResult] = useState(null); // last successful /profiles/switch response,
                                                            // rendered close to verbatim below
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(null);

  const say = (kind, text) => setMessage({ kind, text });
  const refresh = async () => {
    if(!live){
      setProfiles([]); setActive(null);
      say("info", "Profiles need the live local server; sample mode never writes persona state.");
      return;
    }
    const response = await api.profilesList();
    if(response && Array.isArray(response.profiles)){
      setProfiles(response.profiles);
      setActive(response.active || null);
    }else{
      setProfiles([]);
      say("error", "The profile store did not answer. The rest of heavn remains read-only.");
    }
  };

  useEffect(() => { refresh(); }, [live]);

  const readLiveSnapshot = async (profileName, profileDescription, existing = null) => {
    const [cardsResponse, axesResponse] = await Promise.all([api.memoryList(), api.steerAxes()]);
    if(!cardsResponse || !Array.isArray(cardsResponse.cards)
        || !axesResponse || !Array.isArray(axesResponse.axes)){
      throw new Error("Could not read the active cards and dials. Is the local worker ready?");
    }
    const cards = cardsResponse.cards
      // Pending proposals belong to the review queue, not to an applied persona. Snapshot only the two
      // states this surface promises to preserve: active cards and intentionally disabled cards.
      .filter(card => typeof card === "string" || card.status === "active" || card.status === "disabled")
      .filter(card => typeof card === "string" || card.text)
      .map(card => typeof card === "string"
        ? { text: card, status: "active" }
        : { text: card.text, status: card.status === "active" ? "active" : "disabled" });
    const dials = {}, customDials = [];
    axesResponse.axes.forEach(axis => {
      const value = Number(axis.value || 0);
      if(!axis.name || !Number.isFinite(value) || Math.abs(value) < .05) return;
      dials[axis.name] = value;
      if(axis.custom){
        const poles = Array.isArray(axis.poles) ? axis.poles : [];
        customDials.push({ name: axis.name, pos: poles[0] || axis.name,
          neg: poles[1] || "neutral", max: Number(axis.max || .5) });
      }
    });
    return {
      ...(existing && existing.created_at != null ? { created_at: existing.created_at } : {}),
      version: (existing && existing.version) || 1,
      name: profileName,
      description: profileDescription || "",
      cards,
      dials,
      custom_dials: customDials,
      // Facts are source pairs, while the active store contains compiled vectors. Preserve sources on
      // update; a new snapshot starts honestly empty instead of reverse-inventing facts from vectors.
      facts: existing && Array.isArray(existing.facts) ? existing.facts : [],
    };
  };

  const saveSnapshot = async (requestedName, requestedDescription, existing = null) => {
    if(!live || busy) return;
    const profileName = slug(requestedName);
    if(!profileName){ say("error", "Use a profile name containing letters, numbers, - or _."); return; }
    if(!existing && (profiles || []).some(profile => profile.name === profileName)){
      say("error", `“${profileName}” already exists. Use UPDATE FROM LIVE on its row.`);
      return;
    }
    setBusy("save:" + profileName); setConfirmDelete(null); setSwitchResult(null);
    say("info", "Reading the active cards and dials…");
    try{
      const bundle = await readLiveSnapshot(profileName, requestedDescription, existing);
      const response = await api.profilesSave(bundle);
      if(!good(response)) throw new Error(reason(response, "The profile could not be saved."));
      const counts = profileCounts(response.profile || bundle);
      say("ok", `Saved “${profileName}”: ${counts.cards} card(s), ${counts.dials} dial(s), `
        + `${counts.facts} preserved fact source(s).`);
      setName(""); setDescription("");
      await refresh();
    }catch(error){ say("error", String(error.message || error)); }
    finally{ setBusy(""); }
  };

  const switchProfile = async profile => {
    if(!live || busy || profile.name === active) return;
    setBusy("switch:" + profile.name); setConfirmDelete(null); setSwitchResult(null);
    say("info", `Replacing the active cards and dials with “${profile.name}”…`);
    const response = await api.profilesSwitch(profile.name);
    if(good(response)){
      const retraining = response.resync && response.resync.retraining === true;
      say("ok", `Switched to “${profile.name}”${retraining
        ? "; the internalized prefix is retraining in the background."
        : "; prompt-mode changes are active now."}${response.facts_note ? " " + response.facts_note : ""}`);
      toast(`profile · ${profile.name}`);
      // The message line above is the short version; the response itself says exactly what changed
      // (cards replaced, dials applied, whether a retrain kicked off, and the actual prompt this
      // profile injects) -- render that close to verbatim rather than let the summary be the last word.
      setSwitchResult({ name: profile.name, ...response });
      await refresh();
    }else say("error", reason(response, "The profile could not be switched."));
    setBusy("");
  };

  const exportProfile = async profile => {
    if(!live || busy) return;
    setBusy("export:" + profile.name); setConfirmDelete(null); setSwitchResult(null);
    const response = await api.profilesExport(profile.name);
    if(good(response) && response.profile){
      const blob = new Blob([JSON.stringify(response.profile, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url; anchor.download = profile.name + ".clozn-profile.json";
      document.body.appendChild(anchor); anchor.click(); anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      say("ok", `Exported “${profile.name}” as portable JSON.`);
    }else say("error", reason(response, `Could not export “${profile.name}”.`));
    setBusy("");
  };

  const importProfile = async event => {
    const file = event.target.files && event.target.files[0];
    event.target.value = "";
    if(!file || !live || busy) return;
    setBusy("import"); setConfirmDelete(null); setSwitchResult(null); say("info", "Reading profile JSON…");
    try{
      const parsed = JSON.parse(await file.text());
      const bundle = parsed && parsed.profile && typeof parsed.profile === "object" ? parsed.profile : parsed;
      const response = await api.profilesImport(bundle);
      if(!good(response)) throw new Error(reason(response, "The profile could not be imported."));
      say("ok", `Imported “${response.profile.name}”. A same-name bundle is updated in place.`);
      await refresh();
    }catch(error){ say("error", `Import failed: ${String(error.message || error)}`); }
    finally{ setBusy(""); }
  };

  const deleteProfile = async profile => {
    if(!live || busy || profile.name === active) return;
    if(confirmDelete !== profile.name){
      setConfirmDelete(profile.name);
      say("info", `Press CONFIRM DELETE on “${profile.name}” once more. This removes its JSON bundle.`);
      return;
    }
    setBusy("delete:" + profile.name); setSwitchResult(null);
    const response = await api.profilesDelete(profile.name);
    if(good(response)){
      say("ok", `Deleted “${profile.name}”.`); setConfirmDelete(null); await refresh();
    }else say("error", reason(response, `Could not delete “${profile.name}”.`));
    setBusy("");
  };

  const list = profiles || [];
  return html`<div class="col settings-desk">
    <div class="cfg profile-live-strip" data-testid="profile-status">
      <span class="cap">active profile</span><b>${active || "none selected"}</b>
      <span>${active ? "this bundle currently owns the live cards + dials" : "saved bundles are not applied until switched"}</span>
      <span class=${"tag " + (active ? "cap-t" : "smp-t")} style="margin-left:auto">
        ${active ? "APPLIED" : "NO PROFILE"}</span>
    </div>

    <div class="settings-grid">
      <section class="mod profile-library" aria-labelledby="profile-library-title">
        <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
        <div class="mod-h"><span class="led"></span><span class="cap" id="profile-library-title">profiles</span>
          <span class="tail">${profiles === null ? "loading…" : list.length + " saved"}</span>
          <span class="tag cap-t">PORTABLE SOURCE</span></div>
        <div class="profile-intro">Switching replaces the active cards and dials; personas never blend.
          Fact pairs and custom-dial poles travel as recompilable sources, not model-specific vectors.</div>
        <div class="profile-list" data-testid="profile-list">
          ${profiles === null && html`<div class="empty">reading the local profile store…</div>`}
          ${profiles !== null && !list.length && html`<div class="empty">No profiles saved yet. Snapshot the
            current Memory + Behavior setup from the panel at right.</div>`}
          ${list.map(profile => html`<${ProfileRow} key=${profile.name} profile=${profile}
            active=${active} busy=${busy} confirmDelete=${confirmDelete}
            onSwitch=${switchProfile} onUpdate=${p => saveSnapshot(p.name, p.description, p)}
            onExport=${exportProfile} onDelete=${deleteProfile}/>`)}
        </div>
      </section>

      <div class="col">
        <section class="mod profile-create" aria-labelledby="profile-create-title">
          <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
          <div class="mod-h"><span class="led blue"></span><span class="cap" id="profile-create-title">snapshot live setup</span>
            <span class="tail">cards + nonzero dials</span></div>
          <form class="profile-form" onSubmit=${event => {
            event.preventDefault(); saveSnapshot(name, description);
          }}>
            <label><span>profile name</span><input data-testid="profile-name" value=${name} maxlength="32"
              placeholder="work" disabled=${!live || !!busy}
              onInput=${event => setName(event.currentTarget.value)}/></label>
            <label><span>description</span><input value=${description} maxlength="120"
              placeholder="focused, concise work persona" disabled=${!live || !!busy}
              onInput=${event => setDescription(event.currentTarget.value)}/></label>
            <div class="profile-form-actions">
              <button class=${"spd primary" + (busy.startsWith("save:") ? " busy" : "")}
                type="submit" disabled=${!live || !!busy}>SAVE SNAPSHOT</button>
              <button class=${"spd" + (busy === "import" ? " busy" : "")} type="button"
                disabled=${!live || !!busy} onClick=${() => fileInput.current && fileInput.current.click()}>IMPORT JSON</button>
              <input ref=${fileInput} class="profile-file" type="file" accept="application/json,.json"
                onChange=${importProfile}/>
            </div>
          </form>
          <div class="profile-note">Creating snapshots the current active and disabled cards plus every
            meaningfully nonzero dial (|value| ≥ .05). Importing a same-name bundle updates it in place.</div>
        </section>

        <section class="mod profile-honesty" aria-labelledby="profile-honesty-title">
          <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
          <div class="mod-h"><span class="led lilac"></span><span class="cap" id="profile-honesty-title">what travels</span>
            <span class="tag der-t">RECOMPILED</span></div>
          <div class="profile-ledger">
            <div><b>cards</b><span>plain source text + active/disabled status</span></div>
            <div><b>dials</b><span>values; custom poles travel as text recipes</span></div>
            <div><b>facts</b><span>cue/answer source pairs; compiled only when the Facts tier is on</span></div>
            <div><b>vectors</b><span>never exported — rebuilt for the currently loaded model</span></div>
          </div>
        </section>

        <${RuntimeCard} rec=${rec}/>
        <${CountsCard} live=${live}/>
      </div>
    </div>

    ${message && html`<div class=${"profile-message " + message.kind} role="status" aria-live="polite">
      ${message.text}</div>`}

    ${switchResult && html`<div class="cfg profile-switch-receipt" data-testid="profile-switch-receipt">
      <span class="cap">switched</span><b>${switchResult.name}</b>
      <span>${switchResult.cards
        ? `${switchResult.cards.removed} removed, ${switchResult.cards.added} added` : "—"} card(s)</span>
      <span>${switchResult.dials && switchResult.dials.applied
        ? Object.keys(switchResult.dials.applied).length : 0} dial(s) applied</span>
      <span class=${"tag " + (switchResult.resync && switchResult.resync.retraining ? "der-t" : "cap-t")}>
        ${switchResult.resync && switchResult.resync.retraining ? "RETRAINING IN BACKGROUND" : "INSTANT"}</span>
      ${switchResult.facts_note && html`<span class="profile-switch-note">${switchResult.facts_note}</span>`}
      ${switchResult.prompt_block && html`<details class="profile-switch-block">
        <summary>what this profile actually injects</summary>
        <div class="none">${switchResult.prompt_block}</div>
      </details>`}
    </div>`}
  </div>`;
}

function ProfileRow({ profile, active, busy, confirmDelete, onSwitch, onUpdate, onExport, onDelete }){
  const counts = profileCounts(profile);
  const isActive = profile.name === active;
  const locked = !!busy;
  return html`<article class=${"profile-row" + (isActive ? " active" : "")} data-profile=${profile.name}>
    <div class="profile-summary">
      <div class="profile-title"><b>${profile.name}</b>
        ${isActive && html`<span class="tag cap-t">ACTIVE</span>`}</div>
      <p>${profile.description || "No description."}</p>
      <div class="profile-counts">
        <span><b>${counts.cards}</b> cards</span><span><b>${counts.dials}</b> dials</span>
        <span><b>${counts.facts}</b> fact sources</span>
      </div>
    </div>
    <div class="profile-actions">
      <button class="spd primary" disabled=${locked || isActive}
        onClick=${() => onSwitch(profile)}>${isActive ? "APPLIED" : "SWITCH"}</button>
      <button class="spd" disabled=${locked} onClick=${() => onUpdate(profile)}>UPDATE FROM LIVE</button>
      <button class="spd" disabled=${locked} onClick=${() => onExport(profile)}>EXPORT</button>
      <button class=${"spd danger" + (confirmDelete === profile.name ? " armed" : "")}
        disabled=${locked || isActive} title=${isActive ? "Switch profiles before deleting the active bundle" : ""}
        onClick=${() => onDelete(profile)}>${confirmDelete === profile.name ? "CONFIRM DELETE" : "DELETE"}</button>
    </div>
  </article>`;
}

/* ───────────────────────── Runtime — fixed facts, not a dead switcher ───────────────────────── */
function RuntimeCard({ rec }){
  const endpoint = (typeof location !== "undefined" ? location.origin : "") + "/v1";
  const [copied, setCopied] = useState(false);
  return html`<section class="mod profile-runtime" aria-labelledby="profile-runtime-title" data-testid="profile-runtime">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led blue"></span><span class="cap" id="profile-runtime-title">runtime</span>
      <span class="tail">one substrate — no live switching</span></div>
    <div class="profile-runtime-note">This product server runs the C++ engine substrate only. PyTorch
      model-switching (Qwen ↔ Dream) is a lab-only workbench now (<span class="mono">clozn lab</span>) —
      the old studio's model-switch control has no live route to call here (POST /substrate returns
      410), so it isn't reproduced as a dropdown that would just fail.</div>
    <div class="steer-row"><span>active substrate</span><span class="v">engine</span></div>
    <div class="steer-row"><span>model</span><span class="v">${(rec && rec.model) || "—"}</span></div>
    <div class="steer-row">
      <span>OpenAI-compatible endpoint</span>
      <span class="v profile-runtime-endpoint">
        <span class="mono">${endpoint}</span>
        <button class="spd" type="button" onClick=${() => copyText(endpoint).then(ok => {
            setCopied(ok); setTimeout(() => setCopied(false), 1400); })}>
          ${copied ? "COPIED ✓" : "COPY"}</button>
      </span>
    </div>
  </section>`;
}

/* ───────────────────────── Counts — what's here ───────────────────────── */
function CountsCard({ live }){
  const [runs, setRuns] = useState(null);
  const [mems, setMems] = useState(null);
  const [dials, setDials] = useState(null);

  useEffect(() => {
    if(!live){ setRuns(null); setMems(null); setDials(null); return; }
    let cancelled = false;
    (async () => {
      const [r, c, a] = await Promise.all([api.listRuns(), api.memoryList(), api.steerAxes()]);
      if(cancelled) return;
      setRuns(r ? (r.runs || []).length : null);
      setMems(c ? (c.cards || []).length : null);
      setDials(a ? (a.axes || []).filter(x => Math.abs(+x.value || 0) >= 0.05).length : null);
    })();
    return () => { cancelled = true; };
  }, [live]);

  return html`<section class="mod profile-counts-card" aria-labelledby="profile-counts-title" data-testid="profile-counts">
    <span class="screw" style="top:5px;left:5px"></span><span class="screw" style="top:5px;right:5px"></span>
    <div class="mod-h"><span class="led"></span><span class="cap" id="profile-counts-title">what's here</span>
      <span class="tail">everything lives under ~/.clozn — local only, nothing uploaded</span></div>
    <div class="settings-counts">
      ${countBox("runs", runs)}
      ${countBox("memories", mems)}
      ${countBox("active dials", dials)}
    </div>
  </section>`;
}
function countBox(label, n){
  return html`<div class="settings-count-box">
    <div class=${"mono settings-count-value" + (n == null ? " empty" : "")}>${n == null ? "—" : n}</div>
    <div class="settings-count-label">${label}</div>
  </div>`;
}
