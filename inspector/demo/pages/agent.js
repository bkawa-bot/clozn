/* agent.js -- the Agent (home / status) page.  Issue A3.
 *
 * A calm, productized status home (NOT raw internals -- those live in Lab). Reads:
 *   GET  /substrate        -> {active, available:[...]}
 *   POST /steer/axes  {}   -> {axes:[{name,value,...}]}
 *   POST /memory/cards {}  -> {cards:[...]}   (cards are currently plain strings)
 *   GET  /runs             -> {runs:[...]}    (newest first)
 *   GET  /engine/health    -> {engine:...}    (best-effort; tolerate 502)
 * Shows: runtime running, active substrate + model, the copyable OpenAI endpoint (<base>/v1),
 * memory count, active dials, recent-runs count, engine-hook availability.
 * Buttons: Copy endpoint, Open latest run, Test prompt (inline POST /v1/chat/completions).
 */
(function () {
  "use strict";
  var S = window.CloznStudio;

  function statusRow(label, value, opts) {
    opts = opts || {};
    var v = S.el("span", { class: "sval" + (opts.mono ? " mono" : "") }, [value]);
    var dot = opts.dot ? S.el("span", { class: "sdot " + opts.dot }) : null;
    return S.el("div", { class: "srow" }, [
      S.el("span", { class: "slabel" }, [label]),
      S.el("span", { class: "sval-wrap" }, [dot, v].filter(Boolean)),
    ]);
  }

  function chip(text, cls) {
    return S.el("span", { class: "chip " + (cls || "") }, [text]);
  }

  // Editorial "cover" layout (see inspo: Canon ad / NEW FORMS sleeve). Zones, not a form:
  //   masthead line (micro-caps manifesto) | oversized lowercase title + counter-title on a rule |
  //   an asymmetric body: the runtime SPEC SHEET (left, narrow) | a hairline | the TEST zone (right) |
  //   a footnote gag pinned bottom-right. Injected once; scoped to .ag-*.
  function edStyle() {
    if (document.getElementById("ag-ed")) return;
    var s = document.createElement("style");
    s.id = "ag-ed";
    s.textContent =
      ".ag{max-width:1060px;margin:0 auto;padding:30px 34px 40px;position:relative}" +
      ".ag-kicker{font-family:var(--display);font-size:10px;letter-spacing:.34em;text-transform:uppercase;color:var(--faint);" +
      "border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:6px 0;margin-bottom:26px;" +
      "display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;box-shadow:0 2px 0 rgba(50,90,100,.05)}" +
      ".ag-kicker .r{color:var(--soft)}" +
      // the cover title block: oversized lowercase, a counter-title beneath sharing the baseline rule
      ".ag-cover{display:grid;grid-template-columns:1fr auto;align-items:end;gap:10px 20px;" +
      "border-bottom:1px solid var(--line);padding-bottom:14px;box-shadow:0 2px 0 rgba(50,90,100,.06)}" +
      ".ag-title{font-family:var(--display);font-size:clamp(46px,8vw,88px);line-height:.92;font-weight:600;" +
      "letter-spacing:-.01em;text-transform:lowercase;color:var(--ink);margin:0;" +
      "text-shadow:3px 0 0 rgba(47,163,146,.16),-3px 0 0 rgba(79,136,184,.13)}" +
      ".ag-title .dot{color:var(--cyan)}" +
      ".ag-countertitle{font-family:var(--display);font-size:13px;letter-spacing:.2em;text-transform:uppercase;" +
      "color:var(--faint);text-align:right;max-width:230px;line-height:1.5;padding-bottom:6px}" +
      // the asymmetric body: spec sheet | rule | test zone (Canon's narrow-copy + product-column split)
      ".ag-body{display:grid;grid-template-columns:minmax(0,340px) 1fr;gap:0;margin-top:30px}" +
      "@media(max-width:820px){.ag-body{grid-template-columns:1fr}}" +
      ".ag-spec{padding-right:30px}" +
      ".ag-test{padding-left:30px;border-left:1px solid var(--line);box-shadow:-2px 0 0 rgba(50,90,100,.04)}" +
      "@media(max-width:820px){.ag-test{padding-left:0;border-left:none;box-shadow:none;border-top:1px solid var(--line);padding-top:22px;margin-top:26px}}" +
      ".ag-zone-h{font-family:var(--display);font-size:10.5px;letter-spacing:.26em;text-transform:uppercase;" +
      "color:var(--cyan);margin-bottom:12px}" +
      // the spec sheet reads like a credits block: label left in caps, value right, dotted leader rows
      ".ag-endpoint{font-family:ui-monospace,Consolas,monospace;font-size:12px;color:var(--soft);cursor:pointer;" +
      "word-break:break-all;padding:8px 10px;background:var(--wash);border:1px solid var(--line);border-radius:3px;margin:14px 0}" +
      ".ag-endpoint:hover{color:var(--cyan);border-color:rgba(47,163,146,.4)}" +
      ".ag-endpoint .k{color:var(--faint);letter-spacing:.16em;text-transform:uppercase;font-size:9.5px;display:block;margin-bottom:3px}" +
      // footnote gag, bottom-right (Canon's "you can*")
      ".ag-foot{margin-top:34px;display:flex;justify-content:flex-end;align-items:baseline;gap:8px;" +
      "border-top:1px solid var(--line);padding-top:10px}" +
      ".ag-foot .gag{font-family:var(--display);font-size:15px;letter-spacing:.04em;color:var(--soft)}" +
      ".ag-foot .gag b{color:var(--cyan);font-weight:600}" +
      ".ag-foot .star{font-size:10px;color:var(--faint)}";
    document.head.appendChild(s);
  }

  function render(view, ctx) {
    edStyle();
    view.appendChild(
      S.el("div", { class: "ag" }, [
        // the manifesto now lives in the masthead; the page opens straight on its cover title.
        // the cover: oversized title + counter-title on the baseline rule
        S.el("div", { class: "ag-cover" }, [
          S.el("h1", { class: "ag-title" }, ["agent", S.el("span", { class: "dot" }, ["."])]),
          S.el("div", { class: "ag-countertitle" }, ["the runtime behind your tools — running, watched, yours to steer"]),
        ]),
        // asymmetric body
        S.el("div", { class: "ag-body" }, [
          S.el("div", { class: "ag-spec" }, [
            S.el("div", { class: "ag-zone-h" }, ["runtime status"]),
            S.el("div", { class: "srows", id: "statusbody" }, [
              S.el("div", { class: "srow" }, [
                S.el("span", { class: "slabel" }, ["Local runtime"]),
                S.el("span", { class: "sval faintv" }, ["checking…"]),
              ]),
            ]),
            S.el("div", { class: "cardactions", id: "statusactions" }, []),
          ]),
          testZone(ctx),
        ]),
        // footnote gag
        S.el("div", { class: "ag-foot" }, [
          S.el("span", { class: "gag" }, ["run it ", S.el("b", {}, ["local"]), "*"]),
          S.el("span", { class: "star" }, ["*and see exactly what it did"]),
        ]),
      ])
    );

    // ---- fill status (guarded; each fetch degrades to a placeholder) --------------------------
    var endpoint = ctx.endpoint;
    var body = document.getElementById("statusbody");
    var actions = document.getElementById("statusactions");

    // endpoint copies from its own framed block below; actions stay minimal (references favor restraint).
    var latestBtn = S.el("button", {}, ["open latest run →"]);
    latestBtn.disabled = true;
    latestBtn.addEventListener("click", function () {
      if (latestBtn.dataset.rid) ctx.navigate("run/" + latestBtn.dataset.rid);
    });
    var labBtn = S.el("button", {}, ["open lab →"]);
    labBtn.addEventListener("click", function () { ctx.navigate("lab"); });
    actions.appendChild(latestBtn);
    actions.appendChild(labBtn);

    Promise.all([
      ctx.getJSON("/substrate", {}),
      ctx.postJSON("/memory/cards", {}, {}),
      ctx.postJSON("/steer/axes", {}, {}),
      ctx.getJSON("/runs", {}),
      // /engine/health is best-effort: 502 when the C++ engine isn't attached. getJSON returns null then.
      ctx.getJSON("/engine/health", null),
    ]).then(function (r) {
      var sub = r[0] || {};
      var cardsResp = r[1] || {};
      var axesResp = r[2] || {};
      var runsResp = r[3] || {};
      var engine = r[4];

      var reachable = !!(r[0] || r[1] || r[2] || r[3]); // any endpoint answered => server is up
      var cards = cardsResp.cards || [];
      var axes = axesResp.axes || [];
      var runs = runsResp.runs || [];

      var activeSub = sub.active || null;
      var model = modelName(sub, runs);

      // active dials = axes whose |value| is meaningfully non-zero
      var activeDials = axes.filter(function (a) { return Math.abs(+a.value || 0) >= 0.05; });
      var runsToday = runs.filter(function (x) { return ctx.isToday(x.created_at); }).length;

      // engine health: null => unreachable/502 (tolerated), object => connected
      var engineOK = !!engine && !engine.error;

      body.innerHTML = "";
      body.appendChild(statusRow("Local runtime",
        reachable ? "running" : "not reachable",
        { dot: reachable ? "ok" : "off" }));
      body.appendChild(statusRow("Active substrate", activeSub ? cap(activeSub) : "unknown"));
      body.appendChild(statusRow("Model", model || "—", { mono: true }));

      // endpoint gets its own framed block (Canon product-tag energy) -- click to copy
      var epBlock = S.el("div", { class: "ag-endpoint", title: "click to copy" }, [
        S.el("span", { class: "k" }, ["openai-compatible endpoint"]),
        endpoint,
      ]);
      epBlock.addEventListener("click", function () {
        ctx.copyText(endpoint).then(function (ok) {
          if (!ok) return;
          var kids = epBlock.childNodes;
          epBlock.lastChild.textContent = "copied ✓";
          setTimeout(function () { epBlock.lastChild.textContent = endpoint; }, 1100);
        });
      });
      body.appendChild(epBlock);

      body.appendChild(statusRow("Memory",
        cards.length ? cards.length + (cards.length === 1 ? " memory" : " memories") : "none",
        { dot: cards.length ? "ok" : "idle" }));

      var dialWrap = S.el("span", { class: "chips" },
        activeDials.length
          ? activeDials.map(function (a) { return chip(a.name, "steer"); })
          : [S.el("span", { class: "sval faintv" }, ["none active"])]);
      body.appendChild(S.el("div", { class: "srow" }, [
        S.el("span", { class: "slabel" }, ["Active dials"]),
        dialWrap,
      ]));

      body.appendChild(statusRow("Recent runs today", String(runsToday), { dot: runsToday ? "ok" : "idle" }));
      body.appendChild(statusRow("Engine hooks",
        engineOK ? "connected" : "not attached",
        { dot: engineOK ? "ok" : "idle" }));

      // wire "Open latest run" to the newest run id
      if (runs.length && runs[0] && runs[0].id) {
        latestBtn.disabled = false;
        latestBtn.dataset.rid = runs[0].id;
      } else {
        latestBtn.textContent = "No runs yet";
      }

      if (!reachable) {
        var warn = S.el("div", { class: "notewarn" }, [
          "Couldn't reach the studio server. Start it with ",
          S.el("code", {}, ["clozn studio"]),
          " (default port 8090), then reload.",
        ]);
        body.appendChild(warn);
      }
    });
  }

  // the test zone -- an editorial column, not a card. Same IDs/logic as before.
  function testZone(ctx) {
    var out = S.el("div", { class: "testout", id: "testout" }, []);
    var input = S.el("textarea", {
      class: "testinput", id: "testprompt", rows: "4",
      placeholder: "test the runtime — e.g. “explain a KV cache in one sentence.”",
    }, []);
    var send = S.el("button", { class: "go", id: "testsend" }, ["test prompt"]);

    function run() {
      var text = (input.value || "").trim();
      if (!text) { input.focus(); return; }
      send.disabled = true; send.textContent = "running…";
      out.innerHTML = "";
      out.appendChild(S.el("div", { class: "who" }, ["Reply"]));
      var bubble = S.el("div", { class: "testreply" }, ["…"]);
      out.appendChild(bubble);
      ctx.postJSON("/v1/chat/completions", {
        messages: [{ role: "user", content: text }],
        max_tokens: 256,
      }, null).then(function (d) {
        var reply = pickReply(d);
        bubble.textContent = reply || "(no response — is the model loaded?)";
        send.disabled = false; send.textContent = "test prompt";
      });
    }
    send.addEventListener("click", run);
    input.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) run();
    });

    return S.el("div", { class: "ag-test" }, [
      S.el("div", { class: "ag-zone-h" }, ["test prompt"]),
      S.el("p", { class: "cardhint", style: "margin:0 0 12px" }, ["a sanity check that the runtime answers — same OpenAI endpoint your clients use. ⌘/Ctrl+Enter to send."]),
      input,
      S.el("div", { class: "cardactions" }, [send]),
      out,
    ]);
  }

  // OpenAI-shaped reply extractor, tolerant of shapes/errors.
  function pickReply(d) {
    if (!d) return "";
    if (d.error) return "[" + (d.error.message || d.error) + "]";
    try {
      var c = d.choices && d.choices[0];
      if (c && c.message && c.message.content != null) return c.message.content;
      if (c && c.text != null) return c.text;
    } catch (e) {}
    return "";
  }

  // best-effort model label: prefer /substrate, else the newest run's model.
  function modelName(sub, runs) {
    if (sub && sub.model) return sub.model;
    if (runs && runs.length && runs[0] && runs[0].model) return runs[0].model;
    if (sub && sub.active) return "clozn-" + sub.active;
    return "";
  }
  function cap(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }

  S.register("agent", { title: "Agent", render: render });
})();
