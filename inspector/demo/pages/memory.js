/* memory.js -- the Memory page.  Issue D3 + E3 (memory review UI).
 *
 * Makes the agent's memory inspectable, EDITABLE, and REVIEWABLE. Cards carry a review status
 * (pending | active | disabled | rejected); pending cards must be approved before they influence
 * replies. The page has three zones: Pending review (top, only when there are pending cards),
 * Learned traits (active + disabled), and the add/strength controls.
 *
 * Endpoints (all guarded -- the page renders fully offline, and every call still degrades to a
 * friendly note if it 404s while the backend memory-cards work (D2/E1) is mid-flight):
 *   POST /memory/cards    {}            -> {cards:[<string|object>,...], has_prefix}
 *   POST /memory/strength {}            -> {strength:<float>, has_prefix}   (read)
 *   POST /memory/strength {value}       -> set (0=off, 1=normal, up to 2=stronger)
 *   POST /memory/add      {text}        -> creates a PENDING card (RE-TRAINS: SLOW, tens of s .. min)
 *   POST /memory/approve  {id}          -> pending -> active   (rebuilds the prefix)
 *   POST /memory/reject   {id}          -> -> rejected (kept, inert)
 *   POST /memory/disable  {id}          -> toggles active <-> disabled
 *   POST /memory/edit     {id, text}    -> updated card / {ok:true}
 *   POST /memory/remove   {id?, index?} -> deletes a card + rebuilds
 *
 * Card shape: a card is a *string* on the legacy path, or an object after D2:
 *   {id,text,status,source_run_id,created_at,last_used_at,usage_count,kind,risk,evidence,strength}.
 * cardText()/cardMeta()/cardStatus() normalize both shapes so a bare string degrades to text-only
 * (rendered as an active trait with a Delete button) and unknown scalar fields become labelled chips.
 *
 * Pure consumer of the backend (app.js owns the shell + fetch plumbing).
 */
(function () {
  "use strict";
  var S = window.CloznStudio;

  // page-local styles (memory-specific; the shared look comes from clozn.css / app.html). No redesign:
  // reuses .panel/.mcard/.dot conventions + the palette variables.
  var STYLE_ID = "memory-page-style";
  var CSS =
    ".mem-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px}" +
    ".mem-strength{margin:20px 0 8px;padding:16px 18px}" +
    ".mem-strength h2{padding:0 0 4px}" +
    ".mem-strength .strengthrow{display:flex;align-items:center;gap:14px;margin-top:10px}" +
    ".mem-strength input[type=range]{flex:1;min-width:120px;accent-color:var(--halo);cursor:pointer}" +
    ".mem-strength input[type=range]:disabled{cursor:default;opacity:.5}" +
    ".mem-strength .strengthval{font-size:15px;font-weight:680;color:var(--ink);min-width:2.4em;text-align:right;" +
    "font-family:ui-monospace,Consolas,monospace}" +
    ".mem-strength .strengthhint{color:var(--faint);font-size:12.5px;margin-top:9px;line-height:1.45}" +
    ".mem-strength .strengthhint b{color:var(--soft);font-weight:600}" +
    ".mem-strength .ticks{display:flex;justify-content:space-between;font-size:10.5px;color:var(--faint);" +
    "letter-spacing:.02em;margin-top:4px;padding:0 1px}" +
    ".mem-listhead{display:flex;align-items:baseline;justify-content:space-between;gap:10px;margin:26px 0 6px}" +
    ".mem-listhead h2{font-size:12px;font-weight:680;letter-spacing:.14em;text-transform:uppercase;color:var(--faint);margin:0}" +
    ".mem-count{font-size:12px;color:var(--faint)}" +
    ".mem-list{padding:10px 14px 14px}" +
    // the Pending review zone -- highlighted so it reads as "needs your attention".
    ".mem-pending{margin:22px 0 6px;padding:6px 16px 14px;border:1px solid rgba(230,196,120,.5);" +
    "border-radius:16px;background:linear-gradient(180deg,rgba(230,196,120,.10),rgba(255,255,255,.5))}" +
    ".mem-pending .mem-listhead{margin:12px 0 4px}" +
    ".mem-pending .mem-listhead h2{color:#a9762a}" +
    ".mem-pending-intro{color:var(--soft);font-size:12.5px;line-height:1.45;margin:2px 0 4px}" +
    // a memory card: reuse .mcard's frame but allow a body column + a metadata row + an actions row.
    ".mem-card{display:flex;gap:11px;align-items:flex-start;padding:11px 13px;margin:8px 0;border-radius:13px;" +
    "background:rgba(255,255,255,.72);border:1px solid var(--line);transition:box-shadow .2s,opacity .3s}" +
    ".mem-card.busy{opacity:.55}" +
    ".mem-card .dot{width:9px;height:9px;border-radius:50%;margin-top:6px;flex:none;" +
    "background:radial-gradient(circle at 35% 30%,#fff,var(--halo));box-shadow:0 0 10px var(--halo)}" +
    ".mem-card:nth-child(3n+2) .dot{background:radial-gradient(circle at 35% 30%,#fff,var(--warm));box-shadow:0 0 10px var(--warm)}" +
    ".mem-card:nth-child(3n) .dot{background:radial-gradient(circle at 35% 30%,#fff,var(--gold));box-shadow:0 0 10px var(--gold)}" +
    ".mem-card.disabled{opacity:.6}.mem-card.disabled .dot{background:var(--mask);box-shadow:none}" +
    ".mem-card.pending .dot{background:radial-gradient(circle at 35% 30%,#fff,var(--gold));box-shadow:0 0 10px var(--gold)}" +
    ".mem-card.rejected{opacity:.5}.mem-card.rejected .dot{background:rgba(231,120,120,.55);box-shadow:none}" +
    ".mem-card-body{flex:1;min-width:0}" +
    ".mem-card-text{color:var(--ink);font-size:14px;line-height:1.5;word-break:break-word;white-space:pre-wrap}" +
    ".mem-card.disabled .mem-card-text{text-decoration:line-through;text-decoration-color:rgba(120,140,190,.4)}" +
    ".mem-card-meta{display:flex;gap:6px;flex-wrap:wrap;margin-top:7px}" +
    ".mem-meta-chip{font-size:10.5px;letter-spacing:.02em;padding:2px 8px;border-radius:9px;white-space:nowrap;" +
    "color:var(--faint);background:rgba(120,140,190,.08);border:1px solid var(--line)}" +
    ".mem-meta-chip b{color:var(--soft);font-weight:600}" +
    ".mem-meta-chip.status-active{color:#2f97a8;background:rgba(79,195,214,.12);border-color:rgba(79,195,214,.34)}" +
    ".mem-meta-chip.status-pending{color:#a9762a;background:rgba(230,196,120,.16);border-color:rgba(230,196,120,.42)}" +
    ".mem-meta-chip.status-disabled{color:var(--faint);background:rgba(120,140,190,.10)}" +
    ".mem-meta-chip.status-rejected{color:#c0504a;background:rgba(231,120,120,.12);border-color:rgba(231,120,120,.4)}" +
    ".mem-meta-chip.risk{color:#c0504a;background:rgba(231,120,120,.12);border-color:rgba(231,120,120,.4)}" +
    // the prominent "suspicious instruction-like memory" banner on a risky pending card.
    ".mem-risk-flag{display:flex;gap:8px;align-items:flex-start;margin-top:9px;padding:8px 11px;border-radius:11px;" +
    "font-size:12px;line-height:1.4;color:#a33;background:rgba(231,120,120,.12);border:1px solid rgba(231,120,120,.4)}" +
    ".mem-risk-flag .warn{flex:none;font-size:13px;line-height:1.3}" +
    ".mem-risk-flag b{color:#8f2f2f}" +
    // the per-card action buttons (approve / reject / edit / disable / enable / delete).
    ".mem-actions{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}" +
    ".mem-btn{font-size:12px;padding:6px 12px;border-radius:16px;border:1px solid var(--line);" +
    "background:rgba(255,255,255,.82);color:var(--soft);cursor:pointer;line-height:1;" +
    "transition:background .16s,color .16s,border-color .16s,transform .15s}" +
    ".mem-btn:hover:not(:disabled){background:#fff;color:var(--ink);transform:translateY(-1px);box-shadow:0 4px 12px rgba(120,150,210,.16)}" +
    ".mem-btn:disabled{opacity:.45;cursor:default}" +
    ".mem-btn.approve{background:linear-gradient(180deg,#d3f0e6,#c2e9dc);border-color:rgba(91,191,154,.5);color:#2b7a5e;font-weight:620}" +
    ".mem-btn.approve:hover:not(:disabled){background:linear-gradient(180deg,#d9f4eb,#c9efe1)}" +
    ".mem-btn.reject:hover:not(:disabled),.mem-btn.delete:hover:not(:disabled){color:#c0504a;border-color:rgba(231,120,120,.45);background:rgba(231,120,120,.08)}" +
    ".mem-btn.enable{color:#2f97a8;border-color:rgba(79,195,214,.4)}" +
    // inline editor
    ".mem-edit{margin-top:4px}" +
    ".mem-edit textarea{width:100%;font:inherit;font-size:14px;line-height:1.5;border:1px solid var(--halo);" +
    "border-radius:12px;padding:9px 12px;background:rgba(255,255,255,.92);outline:none;color:var(--ink);resize:vertical;" +
    "min-height:3.2em;box-shadow:0 0 0 3px rgba(122,167,255,.14)}" +
    ".mem-edit-row{display:flex;gap:7px;margin-top:8px}" +
    ".mem-card-remove{flex:none;border:1px solid var(--line);background:rgba(255,255,255,.8);color:var(--soft);" +
    "border-radius:50%;width:28px;height:28px;padding:0;font-size:14px;line-height:1;cursor:pointer;" +
    "display:flex;align-items:center;justify-content:center;transition:background .16s,color .16s,border-color .16s}" +
    ".mem-card-remove:hover:not(:disabled){background:rgba(231,120,120,.12);color:#c0504a;border-color:rgba(231,120,120,.4);transform:none;box-shadow:none}" +
    ".mem-card-remove:disabled{opacity:.4;cursor:default}" +
    ".mem-empty{padding:30px 20px;text-align:center;color:var(--faint)}" +
    ".mem-empty-t{font-size:15px;color:var(--soft);margin-bottom:6px}" +
    ".mem-empty-s{font-size:13px;max-width:520px;margin:0 auto;line-height:1.5}" +
    ".mem-rejtoggle{margin:14px 0 0;font-size:12px;color:var(--faint)}" +
    ".mem-rejtoggle button{font-size:12px;padding:5px 12px;color:var(--soft);background:none;border:1px dashed var(--line)}" +
    ".mem-rejtoggle button:hover:not(:disabled){border-color:var(--halo);color:var(--halo);background:none;box-shadow:none;transform:none}" +
    ".mem-add{margin-top:22px;padding:16px 18px}" +
    ".mem-add h2{padding:0 0 4px}" +
    ".mem-add .addrow{display:flex;gap:8px;margin-top:10px}" +
    ".mem-add input[type=text]{flex:1;min-width:0;font:inherit;font-size:14px;border:1px solid var(--line);" +
    "border-radius:22px;padding:10px 15px;background:rgba(255,255,255,.85);outline:none;color:var(--ink);" +
    "transition:border-color .2s,box-shadow .2s}" +
    ".mem-add input[type=text]:focus{border-color:var(--halo);box-shadow:0 0 0 3px rgba(122,167,255,.14)}" +
    ".mem-add input[type=text]:disabled{opacity:.55;cursor:default}" +
    ".mem-add .addhint{color:var(--faint);font-size:12.5px;margin-top:9px;line-height:1.45}" +
    ".mem-add .addhint b{color:var(--soft);font-weight:600}" +
    // the SLOW-add busy banner.
    ".mem-busy{display:flex;align-items:center;gap:11px;margin-top:12px;padding:11px 14px;border-radius:12px;" +
    "background:linear-gradient(90deg,rgba(122,167,255,.12),rgba(231,168,196,.10));border:1px solid rgba(122,167,255,.3)}" +
    ".mem-busy .spin{width:15px;height:15px;flex:none;border-radius:50%;border:2px solid rgba(122,167,255,.35);" +
    "border-top-color:var(--halo);animation:memspin .8s linear infinite}" +
    ".mem-busy .busytext{color:var(--soft);font-size:13px}" +
    ".mem-busy .busytext b{color:var(--ink);font-weight:600}" +
    "@keyframes memspin{to{transform:rotate(360deg)}}" +
    ".mem-note{margin-top:8px;font-size:12.5px;padding:8px 12px;border-radius:10px}" +
    ".mem-note.err{color:#c0504a;background:rgba(231,120,120,.10);border:1px solid rgba(231,120,120,.34)}" +
    ".mem-note.ok{color:#2b7a5e;background:rgba(91,191,154,.10);border:1px solid rgba(91,191,154,.34)}" +
    ".mem-offline{margin:18px 0 0;padding:10px 13px;border-radius:11px;font-size:12.5px;color:var(--faint);" +
    "background:rgba(120,140,190,.06);border:1px solid var(--line)}";

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = CSS;
    document.head.appendChild(st);
  }

  // ---- card shape normalization (string today; object after D2) --------------------------------
  function isObj(c) { return c && typeof c === "object"; }
  function cardText(c) {
    if (c == null) return "";
    if (typeof c === "string") return c;
    if (isObj(c)) return c.text != null ? String(c.text) : "";
    return String(c);
  }
  function cardStatus(c) { return isObj(c) && c.status ? String(c.status).toLowerCase() : ""; }
  // The stable handle for a card's actions: its id after D2; the list index as a fallback for the
  // legacy string path (remove takes an index there). Approve/reject/edit/disable need a real id.
  function cardId(c) { return isObj(c) && c.id != null ? c.id : null; }
  // risk is "low"/"medium"/"high" (or truthy). "low"/"none"/falsey => not risky.
  function cardRisk(c) {
    if (!isObj(c)) return "";
    var r = c.risk;
    if (r == null || r === false || r === "") return "";
    var s = String(r).toLowerCase();
    return (s === "low" || s === "none" || s === "false") ? "" : s || (r === true ? "flagged" : "");
  }

  // metadata fields to surface as chips, in display order. status/risk are handled explicitly by the
  // card frame (recolor + banner), so they're excluded here to avoid doubling up.  Anything else
  // scalar on the object is shown generically so a future card can add fields without a code change.
  var META_ORDER = ["kind", "source_run_id", "usage_count", "last_used_at", "created_at", "strength"];
  var META_SKIP = { id: 1, text: 1, status: 1, risk: 1, evidence: 1 };
  function cardMeta(c, ctx) {
    if (!isObj(c)) return [];
    var out = [];
    var seen = {};
    var push = function (key, node, cls) {
      if (node == null) return;
      seen[key] = 1;
      out.push({ cls: cls || "", node: node });
    };
    META_ORDER.forEach(function (key) {
      if (!(key in c) || c[key] == null || c[key] === "") return;
      var v = c[key];
      if (key === "kind") {
        push("kind", chip(String(v)));
      } else if (key === "source_run_id") {
        push("source_run_id", chip([S.el("b", {}, ["from"]), " " + shortId(String(v))]));
      } else if (key === "usage_count") {
        push("usage_count", chip([String(v), " use" + (Number(v) === 1 ? "" : "s")]));
      } else if (key === "last_used_at") {
        push("last_used_at", chip([S.el("b", {}, ["used"]), " " + fmtWhen(v, ctx)]));
      } else if (key === "created_at") {
        push("created_at", chip([S.el("b", {}, ["added"]), " " + fmtWhen(v, ctx)]));
      } else if (key === "strength") {
        push("strength", chip([S.el("b", {}, ["strength"]), " " + fmtNum(v)]));
      }
    });
    // generic fall-through for any *other* scalar fields a future card ships.
    Object.keys(c).forEach(function (key) {
      if (seen[key] || META_SKIP[key]) return;
      var v = c[key];
      if (v == null || v === "" || typeof v === "object") return;
      out.push({ cls: "", node: chip([S.el("b", {}, [key.replace(/_/g, " ")]), " " + String(v)]) });
    });
    return out;
  }
  function chip(kids) {
    return S.el("span", { class: "mem-meta-chip" }, Array.isArray(kids) ? kids : [kids]);
  }
  function shortId(s) { return s.length > 12 ? s.slice(0, 10) + "…" : s; }
  function fmtNum(v) { var n = Number(v); return isNaN(n) ? String(v) : (n === Math.round(n) ? String(n) : n.toFixed(1)); }
  function fmtWhen(v, ctx) {
    // reuse the shell's time formatting when it looks like a timestamp; else print as-is.
    if (typeof v === "number" || /^\d{4}-\d|Z$|:\d\d/.test(String(v))) {
      var t = ctx.fmtTime ? ctx.fmtTime(v) : String(v);
      var d = ctx.fmtDate ? ctx.fmtDate(v) : "";
      return d ? d + " " + t : t;
    }
    return String(v);
  }

  // ---- page state ------------------------------------------------------------------------------
  // `cards` holds the raw list from /memory/cards (strings or objects). `busy` keys are the per-card
  // action locks (keyed by the card's stable key). `editing` is the key of the card being edited.
  function freshState() {
    return { cards: [], hasPrefix: false, offline: false, adding: false, busy: {}, editing: null, showRejected: false };
  }
  var state = freshState();

  // A stable per-card key for busy/edit tracking: the id if present, else "i:<index>".
  function keyOf(c, i) { var id = cardId(c); return id != null ? "id:" + id : "i:" + i; }

  function render(view, ctx) {
    ensureStyle();
    state = freshState();

    var root = S.el("div", { class: "wrap" }, [
      S.el("div", { class: "mem-head" }, [
        S.el("div", {}, [
          S.el("h1", {}, [S.el("span", { class: "glow" }, ["Memory"])]),
          S.el("p", { class: "sub" }, [
            "What the agent has learned to carry across replies — its remembered traits. Review what it wants to remember, adjust how strongly memory colors responses, add a trait, or remove one.",
          ]),
        ]),
      ]),

      S.el("div", { class: "mem-offline", id: "mem-offline", style: "display:none" }, [
        "The studio server is not reachable — showing an empty memory. Start ",
        S.el("code", {}, ["research/clozn_server.py"]),
        " (or open this from ",
        S.el("code", {}, ["http://127.0.0.1:8090/app.html"]),
        ") to load and edit memory.",
      ]),

      // ---- memory strength ----
      S.el("div", { class: "mem-strength panel" }, [
        S.el("h2", {}, ["Memory strength"]),
        S.el("div", { class: "strengthrow" }, [
          S.el("input", {
            type: "range", id: "mem-strength", min: "0", max: "2", step: "0.1", value: "1",
            oninput: onStrengthInput, onchange: onStrengthChange, disabled: "disabled",
            "aria-label": "memory strength",
          }, []),
          S.el("b", { class: "strengthval", id: "mem-strength-val" }, ["1.0"]),
        ]),
        S.el("div", { class: "ticks" }, [
          S.el("span", {}, ["off"]),
          S.el("span", {}, ["normal"]),
          S.el("span", {}, ["stronger"]),
        ]),
        S.el("div", { class: "strengthhint" }, [
          "How strongly memory colors replies. ",
          S.el("b", {}, ["0"]), " turns learned traits off, ",
          S.el("b", {}, ["1"]), " is normal, up to ",
          S.el("b", {}, ["2"]), " leans on them harder (can over-bleed into unrelated answers).",
        ]),
      ]),

      // ---- pending review (mounted only when there are pending cards) ----
      S.el("div", { id: "mem-pending-host" }, []),

      // ---- active/disabled card list ----
      S.el("div", { class: "mem-listhead" }, [
        S.el("h2", {}, ["Learned traits"]),
        S.el("span", { class: "mem-count", id: "mem-count" }, [""]),
      ]),
      S.el("div", { class: "mem-list panel", id: "mem-list" }, [
        S.el("div", { class: "mem-empty" }, ["Loading memory…"]),
      ]),
      // rejected cards are hidden by default; a small toggle reveals them.
      S.el("div", { class: "mem-rejtoggle", id: "mem-rejtoggle", style: "display:none" }, []),
      S.el("div", { class: "mem-list panel", id: "mem-rejlist", style: "display:none" }, []),

      // ---- add a trait ----
      S.el("div", { class: "mem-add panel" }, [
        S.el("h2", {}, ["Add a trait"]),
        S.el("div", { class: "addrow" }, [
          S.el("input", {
            type: "text", id: "mem-add-input", autocomplete: "off",
            placeholder: "e.g. prefers concise answers with concrete examples",
            onkeydown: function (e) { if (e.key === "Enter") submitAdd(ctx); },
          }, []),
          S.el("button", { class: "go", id: "mem-add-btn", onclick: function () { submitAdd(ctx); } }, ["Propose"]),
        ]),
        S.el("div", { class: "addhint" }, [
          "Describe a lasting preference or fact in a short sentence. It's ",
          S.el("b", {}, ["added to pending — approve it above to take effect."]),
          " Clozn folds an approved trait into the model's memory prefix (this trains and takes a while).",
        ]),
        // slow-add busy banner (hidden until a learn is running)
        S.el("div", { class: "mem-busy", id: "mem-busy", style: "display:none" }, [
          S.el("div", { class: "spin" }, []),
          S.el("span", { class: "busytext", id: "mem-busy-text" }, [
            S.el("b", {}, ["Proposing this…"]), " preparing the trait for review.",
          ]),
        ]),
        S.el("div", { class: "mem-note", id: "mem-note", style: "display:none" }, []),
      ]),
    ]);

    view.appendChild(root);
    loadStrength(ctx);
    loadCards(ctx);
  }

  // ---- strength --------------------------------------------------------------------------------
  function onStrengthInput() {
    var val = document.getElementById("mem-strength-val");
    var sl = document.getElementById("mem-strength");
    if (val && sl) val.textContent = (+sl.value).toFixed(1);
  }
  function onStrengthChange() {
    var sl = document.getElementById("mem-strength");
    if (!sl) return;
    var v = parseFloat(sl.value);
    // fire-and-forget; guarded so a failure never throws to the page.
    S.postJSON("/memory/strength", { value: v }, null);
  }
  function loadStrength(ctx) {
    ctx.postJSON("/memory/strength", {}, null).then(function (d) {
      var sl = document.getElementById("mem-strength");
      var val = document.getElementById("mem-strength-val");
      if (!sl) return;
      if (d && d.strength != null) {
        var s = Math.max(0, Math.min(2, +d.strength));
        sl.value = String(s);
        if (val) val.textContent = s.toFixed(1);
        sl.disabled = false;
      } else {
        // offline / no data: leave the control disabled at its default so we don't imply a live value.
        markOffline(true);
      }
    });
  }

  // ---- cards -----------------------------------------------------------------------------------
  function loadCards(ctx) {
    return ctx.postJSON("/memory/cards", {}, null).then(function (d) {
      if (d == null) { state.offline = true; markOffline(true); state.cards = []; }
      else {
        state.offline = false; markOffline(false);
        state.cards = (d && d.cards) || [];
        state.hasPrefix = !!(d && d.has_prefix);
      }
      state.busy = {};
      state.editing = null;
      drawAll(ctx);
    });
  }

  function markOffline(on) {
    var box = document.getElementById("mem-offline");
    if (box) box.style.display = on ? "" : "none";
    if (on) {
      var sl = document.getElementById("mem-strength");
      if (sl) sl.disabled = true;
    }
  }

  // Partition the raw card list by status, preserving original index (remove needs it on the
  // legacy string path). A bare string / status-less card is treated as active.
  function partition() {
    var pending = [], active = [], rejected = [];
    (state.cards || []).forEach(function (c, i) {
      var st = cardStatus(c);
      var entry = { c: c, i: i };
      if (st === "pending") pending.push(entry);
      else if (st === "rejected") rejected.push(entry);
      else active.push(entry); // active, disabled, or unlabelled (legacy string)
    });
    return { pending: pending, active: active, rejected: rejected };
  }

  function drawAll(ctx) {
    var parts = partition();
    drawPending(ctx, parts.pending);
    drawActive(ctx, parts.active);
    drawRejected(ctx, parts.rejected);
  }

  // ---- pending review zone ---------------------------------------------------------------------
  function drawPending(ctx, pending) {
    var host = document.getElementById("mem-pending-host");
    if (!host) return;
    host.innerHTML = "";
    if (!pending.length) return; // section is entirely absent when nothing is pending

    var body = [
      S.el("div", { class: "mem-listhead" }, [
        S.el("h2", {}, ["Pending review"]),
        S.el("span", { class: "mem-count" }, [
          String(pending.length) + (pending.length === 1 ? " awaiting" : " awaiting"),
        ]),
      ]),
      S.el("div", { class: "mem-pending-intro" }, [
        "Clozn wants to remember these but hasn't yet — they stay inert until you approve. ",
        "Review each one; approve to make it active, edit to fix the wording, or reject to discard it.",
      ]),
    ];
    pending.forEach(function (e) { body.push(cardRow(e.c, e.i, ctx, "pending")); });
    host.appendChild(S.el("div", { class: "mem-pending" }, body));
  }

  // ---- active/disabled zone --------------------------------------------------------------------
  function drawActive(ctx, active) {
    var list = document.getElementById("mem-list");
    var count = document.getElementById("mem-count");
    if (!list) return;
    list.innerHTML = "";

    if (count) count.textContent = active.length ? String(active.length) + (active.length === 1 ? " trait" : " traits") : "";

    if (!active.length) {
      list.appendChild(S.el("div", { class: "mem-empty" }, [
        S.el("div", { class: "mem-empty-t" }, [state.offline ? "Memory unavailable." : "No active traits yet."]),
        S.el("div", { class: "mem-empty-s" }, [
          state.offline
            ? "Connect the studio server to view and edit what the agent remembers."
            : "The agent has no active traits. Add one below (or let it propose memories from your conversations) — proposals appear in Pending review until you approve them.",
        ]),
      ]));
      return;
    }
    active.forEach(function (e) { list.appendChild(cardRow(e.c, e.i, ctx, "active")); });
  }

  // ---- rejected zone (hidden behind a toggle) --------------------------------------------------
  function drawRejected(ctx, rejected) {
    var toggle = document.getElementById("mem-rejtoggle");
    var list = document.getElementById("mem-rejlist");
    if (!toggle || !list) return;

    if (!rejected.length) {
      toggle.style.display = "none";
      list.style.display = "none";
      list.innerHTML = "";
      return;
    }
    toggle.style.display = "";
    toggle.innerHTML = "";
    var open = state.showRejected;
    toggle.appendChild(S.el("button", {
      onclick: function () { state.showRejected = !state.showRejected; drawAll(ctx); },
    }, [(open ? "Hide" : "Show") + " rejected (" + rejected.length + ")"]));

    if (!open) { list.style.display = "none"; list.innerHTML = ""; return; }
    list.style.display = "";
    list.innerHTML = "";
    rejected.forEach(function (e) { list.appendChild(cardRow(e.c, e.i, ctx, "rejected")); });
  }

  // ---- a single card row -----------------------------------------------------------------------
  // `zone` is where it renders: "pending" | "active" | "rejected". It selects the action set.
  function cardRow(c, i, ctx, zone) {
    var status = cardStatus(c); // "", active, disabled, pending, rejected
    var key = keyOf(c, i);
    var busy = !!state.busy[key];
    var editing = state.editing === key;
    // frame class: recolor by status (fallback to the zone for legacy string cards).
    var frameStatus = status || (zone === "active" ? "" : zone);
    var cls = "mem-card" + (frameStatus ? " " + frameStatus : "") + (busy ? " busy" : "");

    var body = [];

    if (editing) {
      body.push(editor(c, i, key, ctx));
    } else {
      body.push(S.el("div", { class: "mem-card-text" }, [cardText(c) || "(empty trait)"]));

      // risk banner: prominent when a pending card looks like a suspicious instruction.
      var risk = cardRisk(c);
      if (risk && zone === "pending") {
        body.push(S.el("div", { class: "mem-risk-flag" }, [
          S.el("span", { class: "warn" }, ["⚠"]),
          S.el("span", {}, [
            S.el("b", {}, ["Suspicious instruction-like memory"]),
            " (risk: " + risk + "). This reads like an embedded instruction rather than a preference — approve only if you trust it.",
          ]),
        ]));
      }

      // metadata chips (status/risk excluded — carried by the frame + banner).
      var metas = cardMeta(c, ctx);
      if (metas.length) {
        body.push(S.el("div", { class: "mem-card-meta" }, metas.map(function (m) {
          if (m.cls) m.node.className = "mem-meta-chip " + m.cls;
          return m.node;
        })));
      }

      // action buttons per zone.
      var actions = actionButtons(c, i, key, ctx, zone, status, busy);
      if (actions.length) body.push(S.el("div", { class: "mem-actions" }, actions));
    }

    return S.el("div", { class: cls }, [
      S.el("span", { class: "dot" }, []),
      S.el("div", { class: "mem-card-body" }, body),
    ]);
  }

  // The action set for a card, chosen by zone + status. Every action is disabled while `busy`.
  function actionButtons(c, i, key, ctx, zone, status, busy) {
    var id = cardId(c);
    var out = [];
    function btn(label, cls, onclick) {
      var b = S.el("button", { class: "mem-btn" + (cls ? " " + cls : ""), onclick: onclick }, [label]);
      if (busy) b.disabled = true;
      return b;
    }

    if (zone === "pending") {
      // Approve / Edit / Reject. Approve+Reject require a real id (they only exist post-E1).
      out.push(btn("Approve", "approve", function () {
        actOnCard(ctx, "/memory/approve", c, i, key, "Approving…");
      }));
      out.push(btn("Edit", "", function () { startEdit(key, ctx); }));
      out.push(btn("Reject", "reject", function () {
        actOnCard(ctx, "/memory/reject", c, i, key, "Rejecting…");
      }));
    } else if (zone === "rejected") {
      // let a rejected card be re-approved (undo) or deleted for good.
      if (id != null) {
        out.push(btn("Approve", "approve", function () {
          actOnCard(ctx, "/memory/approve", c, i, key, "Approving…");
        }));
      }
      out.push(btn("Delete", "delete", function () { removeCard(c, i, key, ctx); }));
    } else {
      // active zone: disabled cards show Enable; active cards show Disable. Both show Delete.
      if (status === "disabled") {
        out.push(btn("Enable", "enable", function () {
          // /memory/disable is a toggle -> re-enables a disabled card.
          actOnCard(ctx, "/memory/disable", c, i, key, "Enabling…");
        }));
      } else if (id != null) {
        // only offer Disable when we have an id to target (legacy string cards can't be toggled).
        out.push(btn("Disable", "", function () {
          actOnCard(ctx, "/memory/disable", c, i, key, "Disabling…");
        }));
      }
      out.push(btn("Delete", "delete", function () { removeCard(c, i, key, ctx); }));
    }
    return out;
  }

  // ---- inline editor ---------------------------------------------------------------------------
  function startEdit(key, ctx) {
    if (state.editing === key) return;
    state.editing = key;
    drawAll(ctx);
    // focus the textarea once it's in the DOM.
    setTimeout(function () {
      var ta = document.getElementById("mem-edit-ta");
      if (ta) { ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length); }
    }, 0);
  }

  function editor(c, i, key, ctx) {
    var ta = S.el("textarea", { id: "mem-edit-ta", rows: "2" }, [cardText(c)]);
    ta.value = cardText(c);
    ta.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); saveEdit(c, i, key, ctx); }
      else if (e.key === "Escape") { e.preventDefault(); state.editing = null; drawAll(ctx); }
    });
    var save = S.el("button", { class: "mem-btn approve", onclick: function () { saveEdit(c, i, key, ctx); } }, ["Save"]);
    var cancel = S.el("button", { class: "mem-btn", onclick: function () { state.editing = null; drawAll(ctx); } }, ["Cancel"]);
    return S.el("div", { class: "mem-edit" }, [
      ta,
      S.el("div", { class: "mem-edit-row" }, [save, cancel]),
    ]);
  }

  function saveEdit(c, i, key, ctx) {
    var ta = document.getElementById("mem-edit-ta");
    var text = ta ? String(ta.value || "").trim() : "";
    if (!text) { if (ta) ta.focus(); return; }
    if (text === cardText(c)) { state.editing = null; drawAll(ctx); return; } // no-op
    var id = cardId(c);
    var payload = id != null ? { id: id, text: text } : { index: i, text: text };
    setBusy(key, true, ctx);
    ctx.postJSON("/memory/edit", payload, null).then(function (res) {
      if (res == null) {
        setBusy(key, false, ctx);
        showNote("Couldn't save that edit — the memory-review endpoints may not be online yet.", true);
        return;
      }
      state.editing = null;
      loadCards(ctx); // authoritative reload (edit may re-train / normalize)
    });
  }

  // ---- generic id-based action (approve/reject/disable) ----------------------------------------
  function actOnCard(ctx, path, c, i, key, busyMsg) {
    if (state.busy[key]) return;
    var id = cardId(c);
    // These endpoints are id-keyed; a legacy string card has no id. Guard with a clear note.
    if (id == null) {
      showNote("This action needs the upgraded memory cards (coming with the memory-review backend).", true);
      return;
    }
    setBusy(key, true, ctx);
    ctx.postJSON(path, { id: id }, null).then(function (res) {
      if (res == null) {
        setBusy(key, false, ctx);
        showNote("That didn't go through — is the studio server up (and is the memory-review backend online)?", true);
        return;
      }
      loadCards(ctx); // reload the authoritative list (status changed)
    });
  }

  // ---- delete (works on both card shapes: id when present, index otherwise) --------------------
  function removeCard(c, i, key, ctx) {
    if (state.busy[key]) return;
    setBusy(key, true, ctx);
    var id = cardId(c);
    // Send both when we have an id so the endpoint works pre- and post-D2 (extra keys are ignored).
    var payload = id != null ? { id: id, index: i } : { index: i };
    ctx.postJSON("/memory/remove", payload, null).then(function (res) {
      if (res == null) {
        setBusy(key, false, ctx);
        showNote("Couldn't remove that trait — is the studio server up?", true);
        return;
      }
      loadCards(ctx); // indices shift after removal -> reload authoritative list
    });
  }

  // set a per-card busy lock and re-render (disables that card's buttons + greys it).
  function setBusy(key, on, ctx) {
    if (on) state.busy[key] = true; else delete state.busy[key];
    drawAll(ctx);
  }

  // ---- add (SLOW: proposes a pending card; may re-train) ---------------------------------------
  function submitAdd(ctx) {
    if (state.adding) return;
    var input = document.getElementById("mem-add-input");
    var text = input ? String(input.value || "").trim() : "";
    if (!text) { if (input) input.focus(); return; }

    hideNote();
    state.adding = true;
    setAddBusy(true);
    drawAll(ctx); // disable per-card buttons while training

    // staged, honest progress text.
    var t0 = Date.now();
    var stages = [
      "reading the trait…",
      "preparing it for review…",
      "still working — larger models take longer…",
    ];
    var busyText = document.getElementById("mem-busy-text");
    var tick = setInterval(function () {
      var s = Math.round((Date.now() - t0) / 1000);
      var stage = s < 15 ? stages[0] : (s < 45 ? stages[1] : stages[2]);
      if (busyText) {
        busyText.innerHTML = "";
        busyText.appendChild(S.el("b", {}, ["Proposing this…"]));
        busyText.appendChild(document.createTextNode(" " + stage + " · " + s + "s"));
      }
    }, 1000);

    ctx.postJSON("/memory/add", { text: text }, null).then(function (res) {
      clearInterval(tick);
      state.adding = false;
      setAddBusy(false);
      if (res == null) {
        showNote("Couldn't add that trait — is the studio server up? Nothing was changed.", true);
        drawAll(ctx); // re-enable buttons
        return;
      }
      if (input) input.value = "";
      showNote("Added to pending — approve it above to take effect.", false);
      // refresh the authoritative list; the new card should surface in Pending review.
      loadCards(ctx).then(function () {
        var inp = document.getElementById("mem-add-input");
        if (inp) inp.focus();
      });
    });
  }

  function setAddBusy(on) {
    var btn = document.getElementById("mem-add-btn");
    var input = document.getElementById("mem-add-input");
    var busy = document.getElementById("mem-busy");
    if (btn) { btn.disabled = on; btn.textContent = on ? "Proposing…" : "Propose"; }
    if (input) input.disabled = on;
    if (busy) busy.style.display = on ? "" : "none";
  }

  function showNote(msg, isErr) {
    var n = document.getElementById("mem-note");
    if (!n) return;
    n.textContent = msg;
    n.className = "mem-note " + (isErr ? "err" : "ok");
    n.style.display = "";
  }
  function hideNote() {
    var n = document.getElementById("mem-note");
    if (n) { n.textContent = ""; n.style.display = "none"; }
  }

  S.register("memory", { title: "Memory", render: render });
})();
