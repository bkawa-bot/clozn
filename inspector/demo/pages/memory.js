/* memory.js -- the Memory page.  Issue D3 + E3 (memory review UI).
 *
 * Makes the agent's memory inspectable, EDITABLE, and REVIEWABLE. Cards carry a review status
 * (pending | active | disabled | rejected); pending cards must be approved before they influence
 * replies. The page has three zones: Pending review (top, only when there are pending cards),
 * Learned traits (active + disabled), and the add/strength controls.
 *
 * Endpoints (all guarded -- the page renders fully offline, and every call still degrades to a
 * friendly note if it 404s while the backend memory-cards work (D2/E1) is mid-flight):
 *   POST /memory/cards    {}            -> {cards:[<string|object>,...], has_prefix, mode?, retraining?}
 *   POST /memory/mode     {mode}        -> swap "prompt" (cards as context) <-> "internalized" (prefix)
 *   POST /memory/strength {}            -> {strength:<float>, has_prefix}   (read)
 *   POST /memory/strength {value}       -> set (0=off; in prompt mode any >0 just means "on when relevant")
 *   POST /memory/add      {text}        -> creates a PENDING card
 *   POST /memory/approve  {id}          -> pending -> active   (internalized: rebuilds the prefix, SLOW;
 *                                          prompt: instant -- the card simply joins the context block)
 *   POST /memory/reject   {id}          -> -> rejected (kept, inert)
 *   POST /memory/disable  {id}          -> toggles active <-> disabled
 *   POST /memory/edit     {id, text}    -> updated card / {ok:true}
 *   POST /memory/remove   {id?, index?} -> deletes a card + rebuilds
 *
 * Memory MODE (the swap spec): state.mode is read off /memory/cards (absent on an older backend ->
 * treated as "internalized", the legacy behavior). In prompt mode the retrain banner/notes never show
 * (the backend's retraining flag is a constant idle) and the copy stops promising a slow prefix fold;
 * the strength slider relabels to an honest on/off (its value doesn't scale anything there).
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
    // the memory-mode panel: which mechanism carries the cards (prompt block vs trained prefix).
    ".mem-mode{margin:20px 0 8px;padding:16px 18px}" +
    ".mem-mode h2{padding:0 0 4px}" +
    ".mem-mode .moderow{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:10px}" +
    ".mem-mode .modechip{font-size:12.5px;font-weight:640;padding:5px 12px;border-radius:14px;" +
    "border:1px solid rgba(122,167,255,.45);color:#2f4a7a;background:rgba(122,167,255,.12)}" +
    ".mem-mode .modechip.internalized{border-color:rgba(231,168,196,.55);color:#8a4a66;background:rgba(231,168,196,.14)}" +
    ".mem-mode .modehint{color:var(--faint);font-size:12.5px;margin-top:10px;line-height:1.55}" +
    ".mem-mode .modehint b{color:var(--soft);font-weight:600}" +
    ".mem-mode .modehint .mline{display:block;margin:2px 0}" +
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
    // a card whose prefix is retraining in the background: a soft pulse + an inline "retraining" note.
    ".mem-card.retraining{border-color:rgba(122,167,255,.5);box-shadow:0 0 0 3px rgba(122,167,255,.10)}" +
    ".mem-retrain-note{display:flex;align-items:center;gap:9px;margin-top:9px;padding:7px 11px;border-radius:11px;" +
    "font-size:12px;line-height:1.4;color:var(--soft);background:linear-gradient(90deg,rgba(122,167,255,.12)," +
    "rgba(231,168,196,.10));border:1px solid rgba(122,167,255,.3)}" +
    ".mem-retrain-note .spin{width:13px;height:13px;flex:none;border-radius:50%;border:2px solid rgba(122,167,255,.35);" +
    "border-top-color:var(--halo);animation:memspin .8s linear infinite}" +
    ".mem-retrain-note b{color:var(--ink);font-weight:600}" +
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
    // --- "set the dial instead" suggestion: a style preference reads better as a tone DIAL than a memory. ---
    ".mem-dial{margin-top:12px;padding:13px 15px;border-radius:13px;border:1px solid rgba(122,167,255,.5);" +
    "background:linear-gradient(180deg,rgba(122,167,255,.12),rgba(231,168,196,.08))}" +
    ".mem-dial .dtitle{display:flex;gap:8px;align-items:flex-start;font-size:13px;line-height:1.45;color:var(--soft)}" +
    ".mem-dial .dtitle .spark{flex:none;font-size:14px;line-height:1.3}" +
    ".mem-dial .dtitle b{color:var(--ink);font-weight:640}" +
    ".mem-dial .drow{display:flex;gap:8px;flex-wrap:wrap;margin-top:11px}" +
    ".mem-dial .dgo{font-size:12.5px;font-weight:620;padding:7px 14px;border-radius:16px;cursor:pointer;line-height:1;" +
    "border:1px solid rgba(122,167,255,.55);color:#2f4a7a;background:linear-gradient(180deg,#dbe7ff,#cbdcff);" +
    "transition:background .16s,transform .15s,box-shadow .16s}" +
    ".mem-dial .dgo:hover:not(:disabled){background:linear-gradient(180deg,#e3edff,#d6e3ff);transform:translateY(-1px);" +
    "box-shadow:0 4px 12px rgba(122,150,210,.18)}" +
    ".mem-dial .dgo:disabled{opacity:.5;cursor:default}" +
    ".mem-dial .dkeep{font-size:12px;padding:7px 12px;border-radius:16px;cursor:pointer;line-height:1;color:var(--faint);" +
    "background:none;border:1px dashed var(--line);transition:color .16s,border-color .16s}" +
    ".mem-dial .dkeep:hover:not(:disabled){color:var(--soft);border-color:var(--halo)}" +
    ".mem-dial .dkeep:disabled{opacity:.5;cursor:default}" +
    ".mem-dial .dnote{margin-top:9px;font-size:12px;line-height:1.45;color:var(--soft)}" +
    ".mem-dial .dnote.ok{color:#2b7a5e}" +
    ".mem-dial .dnote.err{color:#c0504a}" +
    ".mem-dial.dismissed{border-color:var(--line);background:rgba(120,140,190,.06)}" +
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
    // `retraining` mirrors the server's in-flight retrain signal ({active,card_id,action}); `retrainTimer`
    // is the poll handle. A card mutation (approve/disable/edit/remove) now returns FAST and kicks off the
    // slow prefix retrain in the background -- we poll /memory/retrain-status until it's idle, then reload.
    // `mode` is the memory mechanism ("prompt" | "internalized"); null until /memory/cards reports it
    // (an older backend never does -> we render the legacy internalized copy without a mode panel).
    return { cards: [], hasPrefix: false, offline: false, adding: false, busy: {}, editing: null,
             showRejected: false, retraining: null, retrainTimer: null, mode: null, switching: false };
  }
  var state = freshState();

  // Read the in-flight retrain flag out of a mutation response, tolerant of shape: the card endpoints
  // return {...card, resync:{retraining,...}}; remove returns {resync:{retraining}}. null if none.
  function retrainFromRes(res) {
    if (!res || typeof res !== "object") return null;
    var r = res.resync || res.retraining || res;
    if (r && typeof r === "object" && (r.retraining === true || r.active === true)) return r;
    return null;
  }

  // A stable per-card key for busy/edit tracking: the id if present, else "i:<index>".
  function keyOf(c, i) { var id = cardId(c); return id != null ? "id:" + id : "i:" + i; }

  function render(view, ctx) {
    ensureStyle();
    if (state.retrainTimer) clearInterval(state.retrainTimer);   // don't leak a poll across remounts
    state = freshState();

    var root = S.el("div", { class: "wrap" }, [
      S.pageHead({
        kicker: "what it remembers about you",
        kickerRight: "review · edit · delete",
        title: "memory",
        counter: "the traits it carries across replies — yours to read, approve, and remove",
      }),

      S.el("div", { class: "mem-offline", id: "mem-offline", style: "display:none" }, [
        "The studio server is not reachable — showing an empty memory. Start ",
        S.el("code", {}, ["research/clozn_server.py"]),
        " (or open this from ",
        S.el("code", {}, ["http://127.0.0.1:8090/app.html"]),
        ") to load and edit memory.",
      ]),

      // ---- memory mode (populated by drawMode once /memory/cards reports which mechanism is live) ----
      S.el("div", { id: "mem-mode-host" }, []),

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
        S.el("div", { class: "ticks", id: "mem-strength-ticks" }, [
          S.el("span", {}, ["off"]),
          S.el("span", {}, ["normal"]),
          S.el("span", {}, ["stronger"]),
        ]),
        S.el("div", { class: "strengthhint", id: "mem-strength-hint" }, [
          "How strongly memory colors replies. ",
          S.el("b", {}, ["0"]), " turns learned traits off, ",
          S.el("b", {}, ["1"]), " is normal, up to ",
          S.el("b", {}, ["2"]), " leans on them harder (can over-bleed into unrelated answers).",
        ]),
      ]),

      // ---- background retrain banner (shown while the prefix is retraining; e.g. after a removal
      //      where the affected card is gone, or when the page loaded mid-retrain) ----
      S.el("div", { id: "mem-retrain-host" }, []),

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
        S.el("div", { class: "addhint", id: "mem-add-hint" }, [
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
        // "set the dial instead" suggestion host (populated only when /memory/add flags a style preference).
        S.el("div", { id: "mem-dial-host" }, []),
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
        // the memory MODE rides on the cards response; an older backend omits it -> null (legacy copy).
        state.mode = (d.mode === "prompt" || d.mode === "internalized") ? d.mode : null;
        // if the page loaded (or reloaded) while a retrain is in flight, pick it up and keep polling.
        var rt = d && d.retraining;
        if (rt && rt.active === true) startRetrainPoll(ctx, rt);
      }
      state.busy = {};
      state.editing = null;
      drawAll(ctx);
    });
  }

  // ---- background retrain: poll /memory/retrain-status until idle, then reload ------------------
  // A card mutation kicks off the slow prefix retrain server-side; we show a "retraining" state on the
  // affected card and poll every ~3s. Degrades gracefully: if the status endpoint is absent (older
  // backend), getJSON/postJSON return null and we simply stop polling and reload once.
  function startRetrainPoll(ctx, info) {
    state.retraining = { active: true, card_id: (info && info.card_id) || null,
                         action: (info && info.action) || null };
    drawAll(ctx);                               // paint the retraining note on the card immediately
    if (state.retrainTimer) return;             // already polling -> the flag update above is enough
    state.retrainTimer = setInterval(function () { pollRetrainOnce(ctx); }, 3000);
  }

  function stopRetrainPoll() {
    if (state.retrainTimer) { clearInterval(state.retrainTimer); state.retrainTimer = null; }
    state.retraining = null;
  }

  function pollRetrainOnce(ctx) {
    // POST to match the studio's other memory calls (the server routes /memory/* on POST). null-safe.
    ctx.postJSON("/memory/retrain-status", {}, null).then(function (st) {
      if (st == null) {                         // endpoint absent / server down -> stop, reload once, done
        stopRetrainPoll();
        loadCards(ctx);
        return;
      }
      if (st.active === true) {                 // still training: keep the note fresh (action/card may update)
        state.retraining = { active: true, card_id: st.card_id || null, action: st.action || null };
        return;                                 // (no full redraw needed; the note is already up)
      }
      // finished (success or error) -> clear the note, surface any error, reload the authoritative list.
      var err = st.error;
      stopRetrainPoll();
      loadCards(ctx).then(function () {
        if (err) showNote("The memory retrain didn't finish cleanly: " + err, true);
        else showNote("Memory updated — the prefix finished retraining.", false);
      });
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
    drawMode(ctx);
    drawRetrainBanner(ctx, parts);
    drawPending(ctx, parts.pending);
    drawActive(ctx, parts.active);
    drawRejected(ctx, parts.rejected);
  }

  // ---- memory mode: indicator + toggle + honest copy -------------------------------------------
  // One panel that says WHICH mechanism carries the cards and lets you swap it (POST /memory/mode).
  // Copy rule (from the swap spec): never oversell -- prompt = "applied as context, readable verbatim";
  // internalized = "trained into a soft prefix; slow to edit; the model can't reliably self-report it".
  function drawMode(ctx) {
    var host = document.getElementById("mem-mode-host");
    if (!host) return;
    host.innerHTML = "";
    updateModeCopy(state.mode);
    if (!state.mode) return;                     // older backend / offline: no panel, legacy copy stands
    var isPrompt = state.mode === "prompt";
    var target = isPrompt ? "internalized" : "prompt";

    var btn = S.el("button", { class: "mem-btn", id: "mem-mode-btn" },
                   [state.switching ? "switching…" : "Switch to " + target]);
    if (state.switching) btn.disabled = true;
    btn.addEventListener("click", function () { toggleMode(ctx, target); });

    host.appendChild(S.el("div", { class: "mem-mode panel" }, [
      S.el("h2", {}, ["Memory mode"]),
      S.el("div", { class: "moderow" }, [
        S.el("span", { class: "modechip " + state.mode },
             [isPrompt ? "prompt — applied as context" : "internalized — trained-in prefix"]),
        btn,
      ]),
      S.el("div", { class: "modehint" }, [
        S.el("span", { class: "mline" }, [
          S.el("b", {}, ["prompt:"]),
          " cards ride as readable context on relevant turns — the card text is exactly what's applied;" +
          " edits are instant; per-card receipts work.",
        ]),
        S.el("span", { class: "mline" }, [
          S.el("b", {}, ["internalized:"]),
          " cards are trained into a soft prefix — each change retrains for a few minutes, and the" +
          " model can't reliably self-report what the prefix does (check receipts, not its word).",
        ]),
        S.el("span", { class: "mline" }, [
          isPrompt
            ? "Switching to internalized retrains the prefix from your current cards (a few minutes)."
            : "Switching to prompt is instant and leaves the trained prefix untouched (you can come back).",
        ]),
      ]),
      S.el("div", { class: "mem-note", id: "mem-mode-note", style: "display:none" }, []),
    ]));
  }

  function showModeNote(msg, isErr) {
    var n = document.getElementById("mem-mode-note");
    if (!n) return;
    n.textContent = msg;
    n.className = "mem-note " + (isErr ? "err" : "ok");
    n.style.display = "";
  }

  function toggleMode(ctx, target) {
    if (state.switching) return;
    state.switching = true;
    drawMode(ctx);                               // repaint the button as busy
    ctx.postJSON("/memory/mode", { mode: target }, null).then(function (res) {
      state.switching = false;
      if (res == null || res.ok === false) {
        drawMode(ctx);
        showModeNote("Couldn't switch the mode — is the studio server up (and the mode endpoint online)? Nothing was changed.", true);
        return;
      }
      // switching to internalized may kick a background catch-up retrain (the prefix was stale vs the
      // cards); surface it through the normal retrain banner + poll.
      var rt = res.resync && res.resync.retraining === true;
      loadCards(ctx).then(function () {
        if (rt) startRetrainPoll(ctx, { card_id: null, action: "mode-switch" });
        showModeNote(target === "prompt"
          ? "Prompt mode: cards now ride as context — edits are instant."
          : "Internalized mode: cards drive the trained prefix." +
            (rt ? " Retraining it from your cards now (a few minutes)." : ""), false);
      });
    });
  }

  // Mode-dependent copy on the strength + add panels. Prompt mode: the slider is an honest on/off (the
  // value scales nothing there) and approval is instant -- the hints must say so instead of promising
  // a slow prefix fold. null mode (older backend) keeps the legacy internalized copy.
  function updateModeCopy(mode) {
    var ticks = document.getElementById("mem-strength-ticks");
    var hint = document.getElementById("mem-strength-hint");
    var add = document.getElementById("mem-add-hint");
    var isPrompt = mode === "prompt";
    if (ticks) {
      ticks.innerHTML = "";
      (isPrompt ? ["off", "on (when relevant)", "on"] : ["off", "normal", "stronger"]).forEach(function (t) {
        ticks.appendChild(S.el("span", {}, [t]));
      });
    }
    if (hint) {
      hint.innerHTML = "";
      if (isPrompt) {
        [S.el("b", {}, ["0"]), " keeps memory out of every reply; anything above ",
         S.el("b", {}, ["0"]),
         " injects the cards when a turn is on-topic. In prompt mode this is an on/off — the value doesn't scale anything.",
        ].forEach(function (n) { hint.appendChild(typeof n === "string" ? document.createTextNode(n) : n); });
      } else {
        ["How strongly memory colors replies. ", S.el("b", {}, ["0"]), " turns learned traits off, ",
         S.el("b", {}, ["1"]), " is normal, up to ", S.el("b", {}, ["2"]),
         " leans on them harder (can over-bleed into unrelated answers).",
        ].forEach(function (n) { hint.appendChild(typeof n === "string" ? document.createTextNode(n) : n); });
      }
    }
    if (add) {
      add.innerHTML = "";
      ["Describe a lasting preference or fact in a short sentence. It's ",
       S.el("b", {}, ["added to pending — approve it above to take effect."]),
       isPrompt
         ? " In prompt mode approval is instant: the card is applied as context on relevant turns (no training)."
         : " Clozn folds an approved trait into the model's memory prefix (this trains and takes a while).",
      ].forEach(function (n) { add.appendChild(typeof n === "string" ? document.createTextNode(n) : n); });
    }
  }

  // A global "retraining" banner, shown while a background retrain is in flight but NOT already surfaced
  // as a per-card note -- i.e. the affected card is gone (a removal) or unknown. Keeps the signal visible
  // no matter which mutation started it.
  function drawRetrainBanner(ctx, parts) {
    var host = document.getElementById("mem-retrain-host");
    if (!host) return;
    host.innerHTML = "";
    var rt = state.retraining;
    if (!rt || !rt.active) return;
    // if a visible card already carries the note (its id matches), don't double up with the banner.
    if (rt.card_id != null) {
      var shown = (state.cards || []).some(function (c) {
        var id = cardId(c);
        return id != null && String(id) === String(rt.card_id);
      });
      if (shown) return;
    }
    host.appendChild(S.el("div", { class: "mem-busy", style: "margin:18px 0 0" }, [
      S.el("div", { class: "spin" }, []),
      S.el("span", { class: "busytext" }, [
        S.el("b", {}, ["Retraining memory…"]),
        " folding your change into the prefix (a few minutes). Chats wait until it finishes.",
      ]),
    ]));
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
    // is THIS card the one whose prefix is retraining right now? (matched by id)
    var id = cardId(c);
    var retraining = !!(state.retraining && state.retraining.active &&
                        state.retraining.card_id != null && id != null &&
                        String(state.retraining.card_id) === String(id));
    // frame class: recolor by status (fallback to the zone for legacy string cards).
    var frameStatus = status || (zone === "active" ? "" : zone);
    var cls = "mem-card" + (frameStatus ? " " + frameStatus : "") + (busy ? " busy" : "") +
              (retraining ? " retraining" : "");

    var body = [];

    if (editing) {
      body.push(editor(c, i, key, ctx));
    } else {
      body.push(S.el("div", { class: "mem-card-text" }, [cardText(c) || "(empty trait)"]));

      // background-retrain note: this card's change is being folded into the prefix (slow: a few min).
      if (retraining) {
        body.push(S.el("div", { class: "mem-retrain-note" }, [
          S.el("span", { class: "spin" }, []),
          S.el("span", {}, [
            S.el("b", {}, ["Retraining memory…"]),
            " folding this change into the prefix (this takes a few minutes). Chats wait until it's done.",
          ]),
        ]));
      }

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
      var rt = retrainFromRes(res);             // editing an ACTIVE card retrains in the background
      loadCards(ctx).then(function () {         // authoritative reload (text/normalize landed)
        if (rt) startRetrainPoll(ctx, { card_id: id, action: rt.action || "edit" });
      });
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
      // the status flip already landed (fast). If it kicked off a background retrain, poll for it; the
      // affected card shows a "retraining" note until the prefix finishes. Otherwise just reload.
      var rt = retrainFromRes(res);
      loadCards(ctx).then(function () {
        if (rt) startRetrainPoll(ctx, { card_id: id, action: rt.action || null });
      });
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
      // removing an ACTIVE card rebuilds the prefix in the background -> poll for it (the card is gone,
      // so the retraining note rides the global banner rather than a specific card).
      var rt = retrainFromRes(res);
      loadCards(ctx).then(function () {         // indices shift after removal -> reload authoritative list
        if (rt) startRetrainPoll(ctx, { card_id: null, action: "remove" });
      });
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
      // If the backend spotted a style preference, offer the tone DIAL as the better path (style memories
      // transfer weakly). Otherwise behave exactly as before: a plain "added to pending" note.
      var sug = dialSuggestionOf(res);
      if (sug) {
        showNote("Added to pending — but this reads like a style preference.", false);
        showDialSuggestion(ctx, sug, res.card || null);
      } else {
        clearDialSuggestion();
        showNote("Added to pending — approve it above to take effect.", false);
      }
      // refresh the authoritative list; the new card should surface in Pending review.
      loadCards(ctx).then(function () {
        var inp = document.getElementById("mem-add-input");
        if (inp) inp.focus();
      });
    });
  }

  // ---- "set the dial instead" suggestion -------------------------------------------------------
  // The backend attaches `dial_suggestion:{axis,value,pole_label}` to /memory/add when the just-added
  // text is really a tone preference (null/absent otherwise). We surface the dial as the recommended
  // path and, on accept, set it + reject the weak style card. Everything here is null-safe.
  function dialSuggestionOf(res) {
    if (!res || typeof res !== "object") return null;
    var d = res.dial_suggestion;
    if (!d || typeof d !== "object") return null;
    // require the fields we render/act on; treat a malformed suggestion as absent.
    if (d.axis == null || d.value == null || d.pole_label == null) return null;
    return { axis: String(d.axis), value: +d.value, pole_label: String(d.pole_label) };
  }

  function clearDialSuggestion() {
    var host = document.getElementById("mem-dial-host");
    if (host) host.innerHTML = "";
  }

  function showDialSuggestion(ctx, sug, card) {
    var host = document.getElementById("mem-dial-host");
    if (!host) return;
    host.innerHTML = "";
    var cardId = card && card.id != null ? card.id : null;
    var valTxt = fmtNum(sug.value);

    var note = S.el("div", { class: "dnote", id: "mem-dial-note", style: "display:none" }, []);
    var setBtn = S.el("button", { class: "dgo" }, ["Set the " + sug.pole_label + " dial"]);
    var keepBtn = S.el("button", { class: "dkeep" }, ["keep it as a memory anyway"]);

    setBtn.addEventListener("click", function () {
      if (setBtn.disabled) return;
      acceptDial(ctx, sug, cardId, valTxt, setBtn, keepBtn, note);
    });
    keepBtn.addEventListener("click", function () {
      if (keepBtn.disabled) return;
      // dismiss the suggestion; leave the pending card exactly as it is.
      var box = document.getElementById("mem-dial-box");
      if (box) box.className = "mem-dial dismissed";
      setBtn.disabled = true; keepBtn.disabled = true;
      showDialNote(note, "Kept as a pending memory. You can approve or reject it above.", "");
    });

    var box = S.el("div", { class: "mem-dial", id: "mem-dial-box" }, [
      S.el("div", { class: "dtitle" }, [
        S.el("span", { class: "spark" }, ["✦"]),
        S.el("span", {}, [
          "This reads like a style preference. The ",
          S.el("b", {}, [sug.pole_label]),
          " dial steers this directly — memory is weak for style. ",
          "Recommended: set the dial to " + valTxt + " instead.",
        ]),
      ]),
      S.el("div", { class: "drow" }, [setBtn, keepBtn]),
      note,
    ]);
    host.appendChild(box);
  }

  function showDialNote(note, msg, kind) {
    if (!note) return;
    note.textContent = msg;
    note.className = "dnote" + (kind ? " " + kind : "");
    note.style.display = "";
  }

  // Accept the dial: POST /steer/set, then discard the weak style card via /memory/reject (when it has an
  // id). Both calls are null-safe (ctx.postJSON -> null on any failure). Confirms inline, then reloads cards.
  function acceptDial(ctx, sug, cardId, valTxt, setBtn, keepBtn, note) {
    setBtn.disabled = true; keepBtn.disabled = true;
    var was = setBtn.textContent;
    setBtn.textContent = "setting…";
    showDialNote(note, "Setting the " + sug.pole_label + " dial…", "");

    ctx.postJSON("/steer/set", { name: sug.axis, value: sug.value }, null).then(function (sres) {
      if (sres == null) {
        // dial didn't take (offline / endpoint absent) -> nothing changed; let them retry.
        setBtn.disabled = false; keepBtn.disabled = false; setBtn.textContent = was;
        showDialNote(note, "Couldn't set the dial — is the studio server up? Nothing was changed; the memory is still pending.", "err");
        return;
      }
      // dial is set. If there's a card to discard, reject it so the weak style memory doesn't linger.
      if (cardId == null) {
        setBtn.textContent = was;
        showDialNote(note, "Set the " + sug.pole_label + " dial to " + valTxt + ". (No pending card to discard.)", "ok");
        var box0 = document.getElementById("mem-dial-box");
        if (box0) box0.className = "mem-dial dismissed";
        return;
      }
      ctx.postJSON("/memory/reject", { id: cardId }, null).then(function (rres) {
        setBtn.textContent = was;
        var box = document.getElementById("mem-dial-box");
        if (box) box.className = "mem-dial dismissed";
        if (rres == null) {
          // dial set, but the reject didn't land -> be honest; the card is still pending for manual reject.
          showDialNote(note, "Set the " + sug.pole_label + " dial to " + valTxt + ", but couldn't discard the style memory — reject it above if you like.", "ok");
        } else {
          showDialNote(note, "Set " + sug.pole_label + " dial to " + valTxt + "; discarded the style memory.", "ok");
        }
        // reflect the reject (and any dial-side memory change) in the authoritative card list.
        loadCards(ctx);
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
