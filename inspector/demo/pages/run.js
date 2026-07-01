/* run.js -- the Run Inspector (issues C1-C3, + C4 repair shell).
 * Route #/run/<id>; app.js passes the run id as render()'s 3rd arg. Three columns:
 *   Transcript / token-timeline | Influence (memory + dials + model) | Repair (replay actions).
 * Reads GET /runs/<id>. The Repair buttons POST /runs/<id>/replay and degrade gracefully until issue F1
 * (the replay engine) exists -- so they light up automatically once it lands.
 */
(function () {
  "use strict";
  var S = window.CloznStudio;
  var API = location.origin.startsWith("http") ? "" : "http://127.0.0.1:8090";

  var esc = function (s) { return (s == null ? "" : String(s)).replace(/[&<>]/g, function (m) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;" }[m]; }); };
  var bar = function (c) { var n = Math.round(Math.max(0, Math.min(1, c || 0)) * 8); return "█".repeat(n) + "░".repeat(8 - n); };

  async function getJSON(p) { var r = await fetch(API + p); if (!r.ok) throw new Error(p + " -> " + r.status); return r.json(); }
  async function postJSON(p, b) {
    var r = await fetch(API + p, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(b || {}) });
    return { ok: r.ok, status: r.status, data: await r.json().catch(function () { return {}; }) };
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
      ".ri-out{margin-top:10px;font-size:12.5px}" +
      ".ri-out .diff{background:var(--wash,#f6f8ff);border-radius:9px;padding:8px 10px;margin-top:6px;white-space:pre-wrap}" +
      // "propose a memory from this run" -- its own action + result block, kept distinct from replay.
      ".ri-sep{margin:12px 0 2px;border:0;border-top:1px dashed var(--line,#e3e6ef)}" +
      ".ri-act.busy{opacity:.6;cursor:default}" +
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

  function repairCol(run) {
    var h = "<h3>Repair / replay</h3>";
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

  function wireRepair(root, run) {
    // Only the replay buttons carry data-rep; scope by it so the E2 "propose" button (also .ri-act,
    // for visual parity) doesn't get a replay handler bound to it.
    root.querySelectorAll(".ri-act[data-rep]").forEach(function (btn) {
      btn.onclick = async function () {
        var out = root.querySelector("#ri-out"), changes = JSON.parse(btn.dataset.rep || "{}");
        out.innerHTML = '<span class="sub">replaying…</span>';
        var r = await postJSON("/runs/" + run.id + "/replay", { changes_applied: changes });
        if (r.status === 404) { out.innerHTML = '<span class="sub">replay isn’t wired yet (issue F1) — this button works once /runs/&lt;id&gt;/replay exists.</span>'; return; }
        if (!r.ok) { out.innerHTML = '<span class="sub">replay failed (' + r.status + ")</span>"; return; }
        var child = r.data;
        out.innerHTML =
          '<div class="ri-kv"><span class="k">original</span></div><div class="diff">' + esc(run.response || run.response_summary) + "</div>" +
          '<div class="ri-kv" style="margin-top:8px"><span class="k">replay</span></div><div class="diff">' + esc(child.response || child.response_summary || "") + "</div>";
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
    wirePropose(root, run, ctx);
  }

  S.register("run", { title: "Run Inspector", render: render });
})();
