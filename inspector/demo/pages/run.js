/* run.js -- the Run Inspector (issues C1-C3, C4 repair shell, F2 replay-compare + save-fix, G2 quick-repair).
 * Route #/run/<id>; app.js passes the run id as render()'s 3rd arg. Three columns:
 *   Transcript / token-timeline | Influence (memory + dials + model) | Repair (replay actions).
 * Reads GET /runs/<id>. The Repair buttons POST /runs/<id>/replay and degrade gracefully until issue F1
 * (the replay engine) exists -- so they light up automatically once it lands.
 *
 * F2 (replay compare + save-fix) and G2 (quick-repair presets) all funnel through ONE replay path
 * (doReplay) and ONE result renderer (renderCompare), so every trigger -- the classic replay buttons,
 * the quick-repair presets, anything future -- shows the same Original|Replayed diff and the same
 * "Save this fix" affordance. Nothing here ever throws: the local postJSON resolves {ok,status,data}
 * on every outcome (offline included), and every branch renders a friendly note instead.
 */
(function () {
  "use strict";
  var S = window.CloznStudio;
  var API = location.origin.startsWith("http") ? "" : "http://127.0.0.1:8090";

  var esc = function (s) { return (s == null ? "" : String(s)).replace(/[&<>]/g, function (m) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[m]; }); };
  var bar = function (c) { var n = Math.round(Math.max(0, Math.min(1, c || 0)) * 8); return "█".repeat(n) + "░".repeat(8 - n); };

  // Per-axis safe caps, mirrored from research/steering.py AXES (cognitive axes degenerate past these).
  // Used only for pre-flight "after" hints in the change summary; the BACKEND is the real authority and
  // its returned behavior.active_dials always wins when present.
  var AXIS_MAX = { warm: 1.5, concise: 1.5, formal: 1.5, playful: 1.5, curious: 1.5, poetic: 1.5, technical: 1.5, candid: 0.45, confident: 1.0, concrete: 0.5 };
  var NUDGE_STEP = 0.5;   // matches replay.NUDGE_STEP; a preset bumps its dial this far toward its + pole.
  var capAxis = function (name, v) { var mx = AXIS_MAX[name] == null ? 1.5 : AXIS_MAX[name]; return Math.max(-mx, Math.min(mx, v)); };

  async function getJSON(p) { var r = await fetch(API + p); if (!r.ok) throw new Error(p + " -> " + r.status); return r.json(); }
  // Local postJSON: NEVER throws. Resolves {ok, status, data} on every outcome -- HTTP error, offline,
  // bad JSON -- so callers can distinguish 404/503 (endpoint absent) from a real failure, yet stay safe.
  async function postJSON(p, b) {
    try {
      var r = await fetch(API + p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) });
      return { ok: r.ok, status: r.status, data: await r.json().catch(function () { return {}; }) };
    } catch (e) {
      return { ok: false, status: 0, data: {} };   // offline / network error -> status 0, handled as a friendly note.
    }
  }

  function ristyle() {
    if (document.getElementById("ri-style")) return;
    var s = document.createElement("style");
    s.id = "ri-style";
    s.textContent =
      ".ri-head{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:baseline;margin-bottom:14px}" +
      ".ri-flags{display:flex;gap:6px}" +
      ".ri-flag{font-size:10px;border:1px solid var(--line,#e3e6ef);border-radius:8px;padding:1px 6px;color:var(--soft,#5a6072)}" +
      ".ri-flag.warn{color:#c0603a;border-color:#e7c3ac}" +
      ".ri-cols{display:grid;grid-template-columns:1.3fr 1fr .9fr;gap:16px;align-items:start}" +
      "@media(max-width:1000px){.ri-cols{grid-template-columns:1fr}}" +
      ".ri-col{background:#fff;border:1px solid var(--line,#e3e6ef);border-radius:14px;padding:14px}" +
      ".ri-col h3{margin:0 0 10px;font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--faint,#9aa0b3)}" +
      ".ri-turn{margin:8px 0;font-size:13.5px;line-height:1.45;white-space:pre-wrap}" +
      ".ri-turn .who{font-size:11px;color:var(--faint,#9aa0b3);text-transform:uppercase;letter-spacing:.04em}" +
      ".ri-turn.assistant{background:var(--wash,#f6f8ff);border-radius:10px;padding:8px 11px}" +
      ".ri-tl{margin-top:12px;font-family:ui-monospace,Consolas,monospace;font-size:11.5px}" +
      ".ri-tok{display:flex;gap:8px;align-items:center;white-space:nowrap;overflow:hidden}" +
      ".ri-tok .p{min-width:78px;color:var(--ink,#1b1f2a)}" +
      ".ri-tok .b{color:var(--halo,#7aa7ff)}" +
      ".ri-tok.low .p{color:#c0603a}" +
      ".ri-tok .alt{color:var(--faint,#9aa0b3);text-overflow:ellipsis;overflow:hidden}" +
      ".ri-kv{display:flex;justify-content:space-between;gap:10px;font-size:13px;margin:5px 0}" +
      ".ri-kv .k{color:var(--faint,#9aa0b3)}" +
      ".ri-card{border:1px solid var(--line,#e3e6ef);border-radius:9px;padding:7px 9px;margin:6px 0;font-size:12.5px}" +
      ".ri-dial{display:inline-block;font-size:11px;border:1px solid var(--line,#e3e6ef);border-radius:8px;padding:1px 7px;margin:2px 3px 0 0}" +
      ".ri-act{display:block;width:100%;text-align:left;border:1px solid var(--line,#e3e6ef);background:#fff;border-radius:10px;padding:8px 11px;margin:6px 0;font:inherit;font-size:13px;cursor:pointer}" +
      ".ri-act:hover{border-color:var(--halo,#7aa7ff)}" +
      ".ri-act.busy{opacity:.6;cursor:default}" +
      ".ri-out{margin-top:10px;font-size:12.5px}" +
      ".ri-out .diff{background:var(--wash,#f6f8ff);border-radius:9px;padding:8px 10px;margin-top:6px;white-space:pre-wrap}" +
      // --- F2 compare: Original | Replayed side by side (stacks under ~560px) + a one-line change summary ---
      ".ri-cmp-sum{font-size:12px;color:var(--soft,#5a6072);margin:2px 0 8px;line-height:1.4}" +
      ".ri-cmp-sum b{color:var(--ink,#1b1f2a);font-weight:640}" +
      ".ri-cmp{display:grid;grid-template-columns:1fr 1fr;gap:9px;align-items:start}" +
      "@media(max-width:560px){.ri-cmp{grid-template-columns:1fr}}" +
      ".ri-cmp .side{min-width:0}" +
      ".ri-cmp .lbl{font-size:10px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint,#9aa0b3);margin-bottom:3px}" +
      ".ri-cmp .side.now .lbl{color:var(--halo,#7aa7ff)}" +
      ".ri-cmp .body{background:var(--wash,#f6f8ff);border:1px solid var(--line,#e3e6ef);border-radius:9px;padding:8px 10px;white-space:pre-wrap;word-break:break-word;font-size:12.5px;line-height:1.5;color:var(--ink,#1b1f2a)}" +
      ".ri-cmp .side.now .body{border-color:rgba(122,167,255,.4)}" +
      // --- F2 save-fix: a persist affordance under the diff (turns the tried change into the new default) ---
      ".ri-save{margin-top:10px}" +
      ".ri-save-btn{display:inline-block;border:1px solid rgba(122,167,255,.4);background:rgba(122,167,255,.10);color:#3b485a;" +
      "border-radius:10px;padding:6px 12px;font:inherit;font-size:12.5px;font-weight:600;cursor:pointer}" +
      ".ri-save-btn:hover{border-color:var(--halo,#7aa7ff);background:rgba(122,167,255,.16)}" +
      ".ri-save-btn.busy{opacity:.6;cursor:default}" +
      ".ri-save-note{margin-top:7px;font-size:12px;color:var(--soft,#5a6072);line-height:1.45}" +
      ".ri-save-note.ok{color:#2f8a54}" +
      ".ri-save-note.warn{color:#a9762a}" +
      ".ri-save-note .li{display:block;margin:2px 0}" +
      // --- G2 quick-repair: a row of one-click complaint->dial presets above the manual replay buttons ---
      ".ri-qr-h{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--faint,#9aa0b3);margin:2px 0 6px}" +
      ".ri-qr{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}" +
      ".ri-qr-btn{font:inherit;font-size:12px;border:1px solid var(--line,#e3e6ef);background:rgba(255,255,255,.7);color:var(--soft,#5a6072);" +
      "border-radius:16px;padding:5px 11px;cursor:pointer;transition:background .14s,border-color .14s,color .14s}" +
      ".ri-qr-btn:hover{color:var(--ink,#1b1f2a);border-color:var(--halo,#7aa7ff)}" +
      ".ri-qr-btn.busy{opacity:.6;cursor:default}" +
      // "propose a memory from this run" -- its own action + result block, kept distinct from replay.
      ".ri-sep{margin:12px 0 2px;border:0;border-top:1px dashed var(--line,#e3e6ef)}" +
      ".ri-prop{margin-top:8px;font-size:12.5px}" +
      ".ri-prop .card{background:var(--wash,#f6f8ff);border:1px solid var(--line,#e3e6ef);border-radius:10px;padding:9px 11px;margin-top:6px}" +
      ".ri-prop .card .txt{color:var(--ink,#1b1f2a);font-size:13px;line-height:1.5;white-space:pre-wrap;word-break:break-word}" +
      ".ri-prop .chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}" +
      ".ri-prop .chip{font-size:10.5px;letter-spacing:.02em;padding:2px 8px;border-radius:9px;white-space:nowrap;color:var(--soft,#5a6072);background:#fff;border:1px solid var(--line,#e3e6ef)}" +
      ".ri-prop .chip.pending{color:#a9762a;background:rgba(230,196,120,.16);border-color:rgba(230,196,120,.42)}" +
      ".ri-prop .chip.risk{color:#c0504a;background:rgba(231,120,120,.12);border-color:rgba(231,120,120,.4)}" +
      ".ri-prop .warn{display:flex;gap:7px;align-items:flex-start;margin-top:8px;padding:7px 10px;border-radius:9px;font-size:11.5px;line-height:1.4;color:#a33;background:rgba(231,120,120,.12);border:1px solid rgba(231,120,120,.4)}" +
      ".ri-prop .warn b{color:#8f2f2f}" +
      ".ri-prop .added{margin-top:8px;font-size:12px;color:var(--soft,#5a6072)}" +
      ".ri-prop .added a{color:var(--halo,#7aa7ff);text-decoration:none;font-weight:600}" +
      ".ri-prop .added a:hover{text-decoration:underline}" +
      ".ri-prop .sub{color:var(--faint,#9aa0b3)}";
    document.head.appendChild(s);
  }

  function transcriptCol(run) {
    var msgs = run.messages || [], h = "<h3>Transcript</h3>";
    for (var i = 0; i < msgs.length; i++) {
      var cls = msgs[i].role === "assistant" ? "assistant" : "";
      h += '<div class="ri-turn ' + cls + '"><div class="who">' + esc(msgs[i].role) + "</div>" + esc(msgs[i].content) + "</div>";
    }
    var lastA = msgs.length && msgs[msgs.length - 1].role === "assistant";
    if (run.response && !lastA) h += '<div class="ri-turn assistant"><div class="who">assistant</div>' + esc(run.response) + "</div>";
    var tr = run.trace || {}, toks = tr.tokens || [], conf = tr.confidence || [], alts = tr.alternatives || [];
    if (toks.length) {
      h += '<div class="ri-tl"><div class="who" style="margin-bottom:5px">token timeline &mdash; where it was unsure, and what it almost said</div>';
      for (var j = 0; j < toks.length; j++) {
        var c = conf[j] == null ? 1 : conf[j], low = c < 0.5;
        var piece = esc(String(toks[j]).replace(/\n/g, "\\n")).slice(0, 12);
        var alt = "";
        if (low && alts[j] && alts[j].length) {
          alt = "  almost: " + alts[j].slice(0, 3).map(function (a) { return esc((a.piece || "").trim() || "_") + " " + (a.prob || 0).toFixed(2); }).join("  ");
        }
        h += '<div class="ri-tok ' + (low ? "low" : "") + '"><span class="p">' + (low ? "? " : "&nbsp;&nbsp;") + piece + '</span><span class="b">' + bar(c) + "</span><span>" + c.toFixed(2) + '</span><span class="alt">' + alt + "</span></div>";
      }
      var lows = conf.filter(function (x) { return x < 0.5; }).length;
      h += '<div class="sub" style="margin-top:8px">' + lows + " uncertain moment(s)</div></div>";
    }
    return h;
  }

  function influenceCol(run) {
    var mem = run.memory || {}, beh = run.behavior || {}, h = "<h3>What influenced it</h3>";
    h += '<div class="ri-kv"><span class="k">model</span><span>' + esc(run.model || "?") + "</span></div>";
    h += '<div class="ri-kv"><span class="k">backend</span><span>' + esc(run.substrate || "?") + "</span></div>";
    h += '<div class="ri-kv"><span class="k">via</span><span>' + esc(run.source || "?") + " / " + esc(run.client || "?") + "</span></div>";
    var cards = mem.cards_applied || [], strength = mem.strength == null ? 1 : mem.strength;
    h += '<div class="ri-kv" style="margin-top:12px"><span class="k">memory (strength ' + (+strength).toFixed(2) + ')</span><span>' + cards.length + " card(s)</span></div>";
    for (var i = 0; i < cards.length; i++) h += '<div class="ri-card">' + esc(typeof cards[i] === "string" ? cards[i] : cards[i].text) + "</div>";
    if (!cards.length) h += '<div class="sub">no memory applied</div>';
    var dials = beh.active_dials || {}, keys = Object.keys(dials);
    h += '<div class="ri-kv" style="margin-top:12px"><span class="k">behavior dials</span><span>' + keys.length + "</span></div>";
    if (keys.length) h += "<div>" + keys.map(function (k) { return '<span class="ri-dial">' + esc(k) + " " + (+dials[k]).toFixed(2) + "</span>"; }).join("") + "</div>";
    else h += '<div class="sub">no dials active</div>';
    return h;
  }

  // The dials that were live on the ORIGINAL run -- the baseline a quick-repair preset nudges from.
  function runDials(run) { return ((run.behavior || {}).active_dials) || {}; }

  function repairCol(run) {
    var h = "<h3>Repair / replay</h3>";
    // --- G2: quick-repair presets. One click maps a common complaint to a dial nudge + replay, then
    //     shows the before/after via the SAME compare renderer. These carry data-qr (NOT data-rep) so
    //     wireRepair's `.ri-act[data-rep]` selector never binds them as raw replay buttons.
    h += '<div class="ri-qr-h">Quick repair &mdash; one click, then compare</div>';
    h += '<div class="ri-qr">';
    for (var i = 0; i < QUICK_REPAIRS.length; i++) {
      var q = QUICK_REPAIRS[i];
      h += '<button class="ri-qr-btn" data-qr="' + esc(q.key) + '" title="' + esc(q.title) + '">' + esc(q.label) + "</button>";
    }
    h += "</div>";
    h += '<hr class="ri-sep">';
    // --- classic manual replay buttons (F1). Unchanged selector (data-rep) so they keep working. ---
    h += "<button class=\"ri-act\" data-rep='{\"memory_off\":true}'>Replay with memory OFF</button>";
    h += "<button class=\"ri-act\" data-rep='{\"behavior_off\":true}'>Replay with dials OFF (neutral)</button>";
    h += "<button class=\"ri-act\" data-rep='{\"nudge\":\"concise\"}'>Make it more concise + replay</button>";
    h += "<button class=\"ri-act\" data-rep='{\"plain\":true}'>Replay unchanged (re-roll)</button>";
    h += '<div class="ri-out" id="ri-out"></div>';
    // --- E2: propose a durable memory from this run (drops a PENDING card on the Memory page) ---
    h += '<hr class="ri-sep">';
    h += '<button class="ri-act" id="ri-propose">Propose a memory from this run</button>';
    h += '<div class="ri-prop" id="ri-prop"></div>';
    return h;
  }

  // ---------------------------------------------------------------------------------------------------
  // F2 core: ONE replay path (doReplay) + ONE result renderer (renderCompare). Everything -- the manual
  // buttons, the G2 presets -- calls doReplay, so every result gets the same Original|Replayed diff and
  // the same "Save this fix" affordance. Nothing throws.
  // ---------------------------------------------------------------------------------------------------

  // Fire a replay of `run` under `changes`, render the compare into `out`, restore the trigger button.
  // `btn` is optional (the element to disable/busy while in flight). Returns a Promise (never rejects).
  function doReplay(out, run, changes, btn) {
    if (!out) return Promise.resolve();
    var restore = null;
    if (btn) {
      var label = btn.textContent;
      btn.classList.add("busy"); btn.disabled = true;
      restore = function () { btn.classList.remove("busy"); btn.disabled = false; btn.textContent = label; };
    }
    out.innerHTML = '<span class="sub">replaying&hellip;</span>';
    return postJSON("/runs/" + run.id + "/replay", { changes_applied: changes }).then(function (r) {
      if (restore) restore();
      // Distinguish the "not wired / unreachable" cases from a real server error, and never throw.
      if (r.status === 404) {
        out.innerHTML = '<span class="sub">replay isn’t wired yet (issue F1) — this lights up once /runs/&lt;id&gt;/replay exists.</span>';
        return;
      }
      if (r.status === 503) {
        out.innerHTML = '<span class="sub">replay needs the qwen substrate running — start it (or switch to qwen) and try again.</span>';
        return;
      }
      if (r.status === 0) {
        out.innerHTML = '<span class="sub">couldn’t reach the studio (offline?) — nothing was changed. Try again once it’s up.</span>';
        return;
      }
      if (!r.ok || !r.data || !r.data.id) {
        var msg = r.data && r.data.error ? " — " + esc(r.data.error) : "";
        out.innerHTML = '<span class="sub">replay failed (' + esc(r.status) + ")" + msg + "</span>";
        return;
      }
      renderCompare(out, run, r.data);
    }, function () {
      // defensive: postJSON shouldn't reject, but never leave the trigger stuck if it somehow does.
      if (restore) restore();
      out.innerHTML = '<span class="sub">replay failed (unexpected error) — nothing was changed.</span>';
    });
  }

  // Render the Original vs Replayed comparison + change summary + a Save-Fix affordance into `out`.
  // `child` is the child run returned by /runs/<id>/replay (its changes_applied / behavior are authoritative).
  function renderCompare(out, run, child) {
    var changes = child.changes_applied || {};
    var orig = run.response != null && run.response !== "" ? run.response : (run.response_summary || "(no original response captured)");
    var repl = child.response != null && child.response !== "" ? child.response : (child.response_summary || "(no reply)");
    var html =
      '<div class="ri-cmp-sum">Changed: <b>' + summarizeChanges(changes, child) + "</b></div>" +
      '<div class="ri-cmp">' +
        '<div class="side"><div class="lbl">Original</div><div class="body">' + esc(orig) + "</div></div>" +
        '<div class="side now"><div class="lbl">Replayed</div><div class="body">' + esc(repl) + "</div></div>" +
      "</div>" +
      '<div class="ri-save"></div>';
    out.innerHTML = html;
    renderSaveFix(out.querySelector(".ri-save"), changes, child);
  }

  // A one-line, human summary of what the replay changed, derived from changes_applied (+ the child's
  // effective dials when they pin the actual post-cap value). Kept short: "memory off", "concise 0.80", ...
  function summarizeChanges(changes, child) {
    changes = changes || {};
    var eff = ((child || {}).behavior || {}).active_dials || {};
    var parts = [];
    if (changes.memory_off) parts.push("memory off");
    if (Array.isArray(changes.disabled_memory_ids) && changes.disabled_memory_ids.length) {
      parts.push("memory disabled (" + changes.disabled_memory_ids.length + ")");
    }
    if (changes.behavior_off) parts.push("dials neutralized");
    if (changes.behavior_overrides && typeof changes.behavior_overrides === "object") {
      Object.keys(changes.behavior_overrides).forEach(function (k) {
        // prefer the effective (post-cap) value the backend actually applied; fall back to the request.
        var v = eff[k] != null ? eff[k] : changes.behavior_overrides[k];
        parts.push(esc(k) + " " + (+v).toFixed(2));
      });
    }
    if (changes.nudge) {
      var nk = String(changes.nudge);
      var nv = eff[nk] != null ? " → " + (+eff[nk]).toFixed(2) : "";
      parts.push(esc(nk) + " up" + nv);
    }
    if (changes.plain) parts.push("nothing (re-roll, same settings)");
    if (!parts.length) parts.push("nothing (re-roll, same settings)");
    return parts.join(", ");
  }

  // --- F2 save-fix -----------------------------------------------------------------------------------
  // Which persist calls make this replay's change the new default? Returns a list of {endpoint, body, label}
  // ops, plus `unpersistable` reasons for parts of the change that have no matching endpoint (so we can say
  // so honestly instead of pretending). Mapping:
  //   memory_off / behavior_off  -> not a single persistable target (they suppress everything for one turn;
  //                                 there's no "turn all memory/dials off forever" endpoint) -> explained.
  //   disabled_memory_ids[]      -> POST /memory/disable {id} for each id.
  //   behavior_overrides{k:v}    -> POST /steer/set {name:k, value:v} for each dial (v = effective/post-cap).
  //   nudge:name                 -> POST /steer/set {name, value:effective}  (the bumped, capped value).
  //   plain / edited_memory      -> nothing to persist (pure re-roll / unwired edit) -> explained.
  function persistOps(changes, child) {
    changes = changes || {};
    var eff = ((child || {}).behavior || {}).active_dials || {};
    var ops = [], unpersistable = [];

    if (Array.isArray(changes.disabled_memory_ids) && changes.disabled_memory_ids.length) {
      changes.disabled_memory_ids.forEach(function (id) {
        ops.push({ endpoint: "/memory/disable", body: { id: id }, label: "disabled memory " + id });
      });
    }
    if (changes.behavior_overrides && typeof changes.behavior_overrides === "object") {
      Object.keys(changes.behavior_overrides).forEach(function (k) {
        var v = eff[k] != null ? +eff[k] : +changes.behavior_overrides[k];
        ops.push({ endpoint: "/steer/set", body: { name: k, value: v }, label: k + " set to " + v.toFixed(2) });
      });
    }
    if (changes.nudge) {
      var nk = String(changes.nudge);
      // the bumped value the backend actually landed on (post per-axis cap); needed so Save reproduces it.
      var nv = eff[nk] != null ? +eff[nk] : capAxis(nk, NUDGE_STEP);
      ops.push({ endpoint: "/steer/set", body: { name: nk, value: nv }, label: nk + " set to " + nv.toFixed(2) });
    }

    // changes that deliberately can't map to a "new default" endpoint -> report, don't offer a button.
    if (changes.memory_off) unpersistable.push("Memory OFF is a one-turn suppression; to make it permanent, disable specific cards on the Memory page.");
    if (changes.behavior_off) unpersistable.push("Dials OFF is a one-turn neutral; to make it permanent, zero the dials on the Behavior page.");
    if (changes.edited_memory) unpersistable.push("Memory-card editing isn’t wired to persist yet.");
    if (changes.plain && !ops.length) unpersistable.push("This was an unchanged re-roll — there’s nothing to save.");

    return { ops: ops, unpersistable: unpersistable };
  }

  // Render the Save-Fix affordance under a compare. Shows a button only when the change maps to a persist
  // endpoint; otherwise a short note explaining why it can't be saved (never a dead button).
  function renderSaveFix(host, changes, child) {
    if (!host) return;
    var plan = persistOps(changes, child);
    if (!plan.ops.length) {
      var why = plan.unpersistable.length
        ? plan.unpersistable.join(" ")
        : "This change can’t be saved as a default (nothing persistable to apply).";
      host.innerHTML = '<div class="ri-save-note warn">' + esc(why) + "</div>";
      return;
    }
    // one button persists the whole plan; any unpersistable remainder is noted alongside.
    var extra = plan.unpersistable.length ? '<div class="ri-save-note warn" style="margin-top:6px">' + esc(plan.unpersistable.join(" ")) + "</div>" : "";
    host.innerHTML = '<button class="ri-save-btn">Save this fix</button><div class="ri-save-note"></div>' + extra;
    var btn = host.querySelector(".ri-save-btn"), note = host.querySelector(".ri-save-note");
    btn.onclick = function () {
      if (btn.disabled) return;
      btn.disabled = true; btn.classList.add("busy");
      var was = btn.textContent; btn.textContent = "saving…";
      note.className = "ri-save-note"; note.innerHTML = "";
      // fire the ops sequentially (order-independent, but simplest to reason about) and tally outcomes.
      var done = [], failed = [];
      var chain = Promise.resolve();
      plan.ops.forEach(function (op) {
        chain = chain.then(function () {
          return postJSON(op.endpoint, op.body).then(function (r) {
            var okBody = r.data && (r.data.ok === false) ? false : true;   // /memory/disable can 200 with {ok:false}
            if (r.ok && okBody && r.status !== 404 && r.status !== 503 && r.status !== 0) done.push(op.label);
            else failed.push({ op: op, r: r });
          });
        });
      });
      chain.then(function () {
        btn.classList.remove("busy"); btn.textContent = was;
        if (!failed.length) {
          note.className = "ri-save-note ok";
          note.innerHTML = "Saved: " + done.map(function (d) { return '<span class="li">✓ ' + esc(d) + "</span>"; }).join("");
        } else if (!done.length) {
          btn.disabled = false;   // total failure -> let them retry
          note.className = "ri-save-note warn";
          note.innerHTML = "Couldn’t save (studio offline or endpoint unavailable) — nothing was changed." + failNote(failed);
        } else {
          btn.disabled = false;   // partial -> allow a retry of the whole plan
          note.className = "ri-save-note warn";
          note.innerHTML = "Saved: " + done.map(function (d) { return '<span class="li">✓ ' + esc(d) + "</span>"; }).join("") +
            '<span class="li">Some parts didn’t save — try again.</span>' + failNote(failed);
        }
      }, function () {
        btn.classList.remove("busy"); btn.disabled = false; btn.textContent = was;
        note.className = "ri-save-note warn";
        note.innerHTML = "Couldn’t save (unexpected error) — nothing was changed.";
      });
    };
  }

  // A compact "(2 endpoints unreachable)"-style tail so a failure isn't totally opaque, without dumping raw errors.
  function failNote(failed) {
    if (!failed.length) return "";
    var codes = failed.map(function (f) { return f.r.status === 0 ? "offline" : f.r.status; });
    return '<span class="li sub">(' + esc(codes.join(", ")) + ")</span>";
  }

  // ---------------------------------------------------------------------------------------------------
  // G2 quick-repair presets. Each maps a plain-language complaint to a single dial pushed toward its +
  // pole by NUDGE_STEP relative to the run's CURRENT value (backend re-caps per axis). We send an explicit
  // behavior_overrides so the intent is legible in changes_applied ("concise 0.80"), and the compare shows
  // before/after. `candid` is the direct opposite of "agreeable"; `concrete` of "vague"; `warm` of "cold".
  // ---------------------------------------------------------------------------------------------------
  var QUICK_REPAIRS = [
    { key: "verbose",   label: "Too verbose",   axis: "concise",  title: "Push the concise↔verbose dial toward concise (+0.5) and replay." },
    { key: "vague",     label: "Too vague",     axis: "concrete", title: "Push the concrete↔abstract dial toward concrete (+0.5) and replay." },
    { key: "agreeable", label: "Too agreeable", axis: "candid",   title: "Push the candid↔agreeable dial toward candid (+0.5) and replay." },
    { key: "cold",      label: "Too cold",      axis: "warm",     title: "Push the warm↔detached dial toward warm (+0.5) and replay." }
  ];
  var QR_BY_KEY = {};
  QUICK_REPAIRS.forEach(function (q) { QR_BY_KEY[q.key] = q; });

  // Build the behavior_overrides for a preset: current dial value + NUDGE_STEP, capped to the axis max
  // (belt-and-suspenders; the backend caps too). Returns e.g. { concise: 0.8 }.
  function presetOverrides(preset, run) {
    var cur = +(runDials(run)[preset.axis] || 0);
    var target = capAxis(preset.axis, cur + NUDGE_STEP);
    var o = {};
    o[preset.axis] = target;
    return o;
  }

  function wireQuickRepair(root, run) {
    var out = root.querySelector("#ri-out");
    root.querySelectorAll(".ri-qr-btn[data-qr]").forEach(function (btn) {
      btn.onclick = function () {
        if (btn.disabled) return;
        var preset = QR_BY_KEY[btn.dataset.qr];
        if (!preset || !out) return;
        doReplay(out, run, { behavior_overrides: presetOverrides(preset, run) }, btn);
      };
    });
  }

  function wireRepair(root, run) {
    // Only the manual replay buttons carry data-rep; scope by it so neither the E2 "propose" button nor
    // the G2 quick-repair chips (they carry data-qr) get a replay handler bound here. All roads lead to
    // doReplay -> renderCompare, so these buttons now show the F2 compare + Save-Fix too.
    root.querySelectorAll(".ri-act[data-rep]").forEach(function (btn) {
      btn.onclick = function () {
        var out = root.querySelector("#ri-out"), changes = {};
        try { changes = JSON.parse(btn.dataset.rep || "{}"); } catch (e) { changes = {}; }
        doReplay(out, run, changes, btn);
      };
    });
  }

  // E2: "Propose a memory from this run". Asks the backend to distill a durable preference out of this
  // conversation and drop it into the PENDING memory queue (reviewed on the Memory page). Uses
  // ctx.postJSON(path, body, fallback) which NEVER throws -- on any failure (offline, 404, 500) it
  // resolves to the fallback (null here), so every branch below is safe.
  function wirePropose(root, run, ctx) {
    var btn = root.querySelector("#ri-propose");
    if (!btn) return;
    var out = root.querySelector("#ri-prop");
    btn.onclick = function () {
      if (btn.disabled) return;
      btn.disabled = true;
      btn.classList.add("busy");
      var label = btn.textContent;
      btn.textContent = "reading the conversation…";
      if (out) out.innerHTML = '<span class="sub">reading the conversation… distilling a durable preference from this run.</span>';

      var post = ctx && ctx.postJSON ? ctx.postJSON : postJSON;
      Promise.resolve(post("/runs/" + run.id + "/propose-memory", null, null)).then(function (res) {
        btn.disabled = false;
        btn.classList.remove("busy");
        btn.textContent = label;
        renderProposal(out, res);
      }, function () {
        // defensive: ctx.postJSON shouldn't reject, but never leave the button stuck if it somehow does.
        btn.disabled = false;
        btn.classList.remove("busy");
        btn.textContent = label;
        renderProposal(out, null);
      });
    };
  }

  // Render the outcome of a propose-memory call into `out`. Handles all four shapes the endpoint can
  // return, plus null (offline / non-2xx swallowed by ctx.postJSON): proposed card, no-proposal, error.
  function renderProposal(out, res) {
    if (!out) return;
    // offline / endpoint absent / any swallowed failure -> a friendly, non-alarming note.
    if (res == null || res.ok === false) {
      var reason = res && res.reason ? " (" + esc(res.reason) + ")" : "";
      out.innerHTML = '<span class="sub">Couldn’t propose a memory (endpoint offline?)' + reason +
        " — nothing was changed. This lights up once /runs/&lt;id&gt;/propose-memory exists.</span>";
      return;
    }
    // nothing durable to remember from this run.
    if (res.proposed === false || !res.card) {
      var why = res.reason ? esc(res.reason) : "no durable preference found in this run";
      out.innerHTML = '<span class="sub">No memory proposed — ' + why + ".</span>";
      return;
    }
    // proposed:true with a card -> show the card text, a "pending review" chip, its risk, and a
    // warning if it reads instruction-like (risk != "low"). Point the user to the Memory page.
    var card = res.card || {};
    var text = card.text != null ? String(card.text) : "";
    var risk = card.risk != null ? String(card.risk) : "";
    var riskLow = risk === "" || risk.toLowerCase() === "low" || risk.toLowerCase() === "none";

    var html = '<div class="card">';
    html += '<div class="txt">' + (text ? esc(text) : "(empty proposal)") + "</div>";
    html += '<div class="chips"><span class="chip pending">pending review</span>';
    if (risk) html += '<span class="chip' + (riskLow ? "" : " risk") + '">risk: ' + esc(risk) + "</span>";
    html += "</div>";
    if (!riskLow) {
      html += '<div class="warn"><span>⚠</span><span><b>Looks instruction-like.</b> ' +
        "This reads more like an embedded instruction than a lasting preference — review it carefully before approving.</span></div>";
    }
    // inline nav to #/memory (a plain hash link is enough; the router mounts the Memory page).
    html += '<div class="added">Added to pending. Review it in <a href="#/memory" data-nav="memory">Memory</a>.</div>';
    html += "</div>";
    out.innerHTML = html;

    // prefer ctx-style navigation when available, but the href alone already works.
    var link = out.querySelector('a[data-nav="memory"]');
    if (link) link.onclick = function (e) {
      e.preventDefault();
      if (S && typeof S.navigate === "function") S.navigate("memory");
      else location.hash = "#/memory";
    };
  }

  async function render(view, ctx, runId) {
    ristyle();
    view.innerHTML = '<div class="wrap"><div class="nav" style="margin-bottom:14px"><a href="#/runs">← Runs</a></div><div id="ri-root"><div class="sub">loading run…</div></div></div>';
    var root = view.querySelector("#ri-root"), run;
    try { run = await getJSON("/runs/" + runId); }
    catch (e) { root.innerHTML = '<div class="sub">run not found (' + esc(e.message) + ")</div>"; return; }
    var flags = run.flags || [];
    root.innerHTML =
      '<div class="ri-head"><b>' + esc(run.prompt_summary || "(run)") + "</b>" +
      '<span class="sub">' + esc(run.created_at) + " · " + esc(run.source) + "/" + esc(run.client) + " · " + esc(run.model) + " · " + ((run.timing || {}).duration_ms != null ? run.timing.duration_ms : "?") + "ms</span>" +
      '<span class="ri-flags">' + flags.map(function (f) { return '<span class="ri-flag ' + (["error", "pending-memory", "low-confidence"].indexOf(f) >= 0 ? "warn" : "") + '">' + esc(f) + "</span>"; }).join("") + "</span></div>" +
      '<div class="ri-cols"><section class="ri-col">' + transcriptCol(run) + '</section><section class="ri-col">' + influenceCol(run) + '</section><section class="ri-col">' + repairCol(run) + "</section></div>";
    wireRepair(root, run);
    wireQuickRepair(root, run);
    wirePropose(root, run, ctx);
  }

  S.register("run", { title: "Run Inspector", render: render });
})();
