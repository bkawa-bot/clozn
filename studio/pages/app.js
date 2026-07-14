/* app.js -- the Clozn Studio shell: sidebar IA + hash router + shared helpers.
 *
 * One coherent shell. The sidebar is fixed: Agent / Runs / Memory / Behavior / Lab / Settings. The hash router
 * (#/agent, #/runs, #/run/<id>, ...) mounts a page module into <main id="view">. Page modules
 * live in pages/*.js and self-register via CloznStudio.register(name, {title, render}); they are
 * pure consumers of the backend -- this file owns navigation, fetch plumbing, and the frame.
 *
 * Plain HTML/JS, no framework, no build step. Reuses clozn.css (its classes + CSS variables).
 */
(function () {
  "use strict";

  // Base = same origin when served over http(s); else the local studio server (opened from file://).
  var BASE = location.origin && location.origin.indexOf("http") === 0 ? "" : "http://127.0.0.1:8090";

  // ---- shared helpers (every page uses these) --------------------------------------------------
  var esc = function (s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, function (m) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[m];
    });
  };

  // fetch guards: NEVER throw to the page. On any failure return `fallback` (default null) so a
  // page can always render an empty/placeholder state even when the endpoint is unreachable.
  function getJSON(path, fallback) {
    return fetch(BASE + path, { headers: { Accept: "application/json" } })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error(r.status)); })
      .catch(function () { return fallback === undefined ? null : fallback; });
  }
  function postJSON(path, body, fallback) {
    return fetch(BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body || {}),
    })
      .then(function (r) { return r.ok ? r.json() : Promise.reject(new Error(r.status)); })
      .catch(function () { return fallback === undefined ? null : fallback; });
  }

  // "3:42 PM" from an ISO-ish or epoch-seconds created_at.
  function fmtTime(created_at) {
    if (created_at == null) return "—";
    var d = typeof created_at === "number" ? new Date(created_at * 1000) : new Date(created_at);
    if (isNaN(d.getTime())) return String(created_at);
    return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  function fmtDate(created_at) {
    if (created_at == null) return "";
    var d = typeof created_at === "number" ? new Date(created_at * 1000) : new Date(created_at);
    if (isNaN(d.getTime())) return "";
    return d.toLocaleDateString([], { month: "short", day: "numeric" });
  }
  function fmtDuration(ms) {
    if (ms == null || isNaN(ms)) return "—";
    if (ms < 1000) return Math.round(ms) + " ms";
    return (ms / 1000).toFixed(ms < 10000 ? 2 : 1) + " s";
  }
  // Is this created_at within the current local day?
  function isToday(created_at) {
    if (created_at == null) return false;
    var d = typeof created_at === "number" ? new Date(created_at * 1000) : new Date(created_at);
    if (isNaN(d.getTime())) return false;
    var n = new Date();
    return d.getFullYear() === n.getFullYear() && d.getMonth() === n.getMonth() && d.getDate() === n.getDate();
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text).then(function () { return true; }, function () { return fallbackCopy(text); });
    }
    return Promise.resolve(fallbackCopy(text));
  }
  function fallbackCopy(text) {
    try {
      var ta = document.createElement("textarea");
      ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.focus(); ta.select();
      var ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return ok;
    } catch (e) { return false; }
  }

  // Shared editorial page header (the sleeve "cover" used on every page): a ruled caps kicker line
  // (left + optional right), then an oversized lowercase title with an optional counter-title on the
  // baseline rule. title/kicker are strings; kickerRight/counter may be a string OR a DOM node.
  function pageHead(o) {
    o = o || {};
    var kickKids = [el("span", {}, [o.kicker || ""])];
    if (o.kickerRight != null) kickKids.push(el("span", { class: "r" }, [o.kickerRight]));
    var coverKids = [el("h1", { class: "phead-title" }, [o.title || ""])];
    if (o.counter != null) coverKids.push(el("div", { class: "phead-counter" }, [o.counter]));
    return el("div", { class: "phead" }, [
      el("div", { class: "phead-kick" }, kickKids),
      el("div", { class: "phead-cover" }, coverKids),
    ]);
  }

  // small DOM builder: el('div', {class:'x'}, [child, 'text'])
  function el(tag, attrs, kids) {
    var n = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      if (k === "class") n.className = attrs[k];
      else if (k === "html") n.innerHTML = attrs[k];
      else if (k === "text") n.textContent = attrs[k];
      else if (k.indexOf("on") === 0 && typeof attrs[k] === "function") n.addEventListener(k.slice(2), attrs[k]);
      else if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    });
    (kids || []).forEach(function (c) {
      if (c == null) return;
      n.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
    });
    return n;
  }

  // ---- page registry ---------------------------------------------------------------------------
  var PAGES = {};
  // The sidebar order/labels (the IA). `route` is the hash key.
  var NAV = [
    { route: "agent", label: "Agent" },
    { route: "runs", label: "Runs" },
    { route: "memory", label: "Memory" },
    { route: "behavior", label: "Behavior" },
    { route: "lab", label: "Lab" },
    { route: "settings", label: "Settings" },
  ];

  function register(name, mod) {
    // mod: { title?: string, render(mountEl, ctx) -> void|Promise }
    PAGES[name] = mod;
    // if the app is already up and this page is the current route, (re)render it.
    if (booted && currentRoute() === name) mountCurrent();
  }

  // ---- hash router -----------------------------------------------------------------------------
  // Routes: #/agent  #/runs  #/run/<id>  #/memory  #/behavior  #/lab  #/settings
  // The "run/<id>" detail route maps to the "run" page module but keeps "runs" highlighted in nav.
  function parseHash() {
    var h = (location.hash || "").replace(/^#\/?/, "");
    var parts = h.split("/").filter(Boolean);
    if (!parts.length) return { page: "agent", arg: null };
    if (parts[0] === "run") return { page: "run", arg: parts[1] || null };
    return { page: parts[0], arg: parts[1] || null };
  }
  function currentRoute() { return parseHash().page; }
  // which nav item should look active for a given page
  function navKeyFor(page) { return page === "run" ? "runs" : page; }

  var booted = false;

  function ctx() {
    return {
      base: BASE,
      endpoint: (BASE || location.origin) + "/v1",
      esc: esc, el: el,
      getJSON: getJSON, postJSON: postJSON,
      fmtTime: fmtTime, fmtDate: fmtDate, fmtDuration: fmtDuration, isToday: isToday,
      copyText: copyText,
      navigate: navigate,
    };
  }

  function navigate(route) {
    // route like "runs" or "run/run_abc"
    if (location.hash === "#/" + route) { mountCurrent(); return; }
    location.hash = "#/" + route;
  }

  function setActiveNav(page) {
    var key = navKeyFor(page);
    var links = document.querySelectorAll("#nav a[data-route]");
    for (var i = 0; i < links.length; i++) {
      var a = links[i];
      if (a.getAttribute("data-route") === key) a.classList.add("here");
      else a.classList.remove("here");
    }
  }

  function mountCurrent() {
    var r = parseHash();
    setActiveNav(r.page);
    var view = document.getElementById("view");
    if (!view) return;
    var mod = PAGES[r.page];
    if (!mod) {
      // page module hasn't loaded (or unknown route): friendly placeholder, not a blank screen.
      view.innerHTML =
        '<div class="wrap"><div class="panel" style="padding:22px">' +
        '<h2 style="text-transform:none;letter-spacing:0;font-size:16px;color:var(--ink)">Loading “' +
        esc(r.page) + '”…</h2>' +
        '<p class="sub" style="margin-top:8px">If this stays, the page module <code>pages/' +
        esc(r.page) + '.js</code> did not load. Check that it is reachable from the studio server.</p>' +
        "</div></div>";
      return;
    }
    view.innerHTML = "";
    try {
      var res = mod.render(view, ctx(), r.arg);
      if (res && typeof res.catch === "function") res.catch(function (e) { renderError(view, e); });
    } catch (e) {
      renderError(view, e);
    }
    // scroll the freshly mounted page to top
    view.scrollTop = 0;
  }

  function renderError(view, e) {
    view.innerHTML =
      '<div class="wrap"><div class="panel" style="padding:22px">' +
      '<h2 style="text-transform:none;letter-spacing:0;font-size:16px;color:var(--ink)">This page hit an error</h2>' +
      '<p class="sub" style="margin-top:8px">' + esc(e && e.message ? e.message : String(e)) + "</p>" +
      "</div></div>";
  }

  // The shell, sleeve-style (see notes/inspo): a top MASTHEAD band (brand · nav · live substrate
  // chip), the scrolling view, and a bottom SPEC STRIP -- dense, tiny, utilitarian fine-print
  // (endpoint / substrate / model / window links), like the back of a compilation cover.
  function buildShell() {
    var app = document.getElementById("app");
    if (!app) return;

    var navLinks = NAV.map(function (item) {
      return el("a", { href: "#/" + item.route, "data-route": item.route, class: "navitem" }, [item.label]);
    });

    // Masthead as ONE ruled caps band (the NEW FORMS micro-line format): brand · nav tabs (tracked
    // caps) on the left, the tag + persona picker + live substrate chip on the right, a hairline
    // above and below.
    var masthead = el("header", { id: "masthead" }, [
      el("span", { class: "logo" }, ["clozn"]),
      el("span", { class: "studioword" }, ["studio"]),
      el("nav", { id: "nav" }, navLinks),
      el("span", { class: "mh-right" }, [
        el("span", { class: "mh-tag", id: "mh-tag" }, ["local · openai-compatible"]),
        buildPersonaPicker(),
        el("span", { class: "subchip", id: "subchip", title: "active substrate" }, ["·"]),
      ]),
    ]);

    var main = el("main", { id: "view" }, []);

    var strip = el("footer", { id: "specstrip" }, [
      el("span", { class: "spec brandline" }, ["clozn studio — glass-box runtime"]),
      el("span", { class: "spec", id: "spec-endpoint", title: "click to copy the OpenAI-compatible endpoint" },
        [(BASE || location.origin) + "/v1"]),
      el("span", { class: "spec", id: "spec-model" }, ["model —"]),
      el("span", { class: "spec grow" }, []),
      el("a", { href: "studio.html", class: "spec speclink", title: "the original chat + memory + dials surface" }, ["classic studio ↗"]),
    ]);

    app.appendChild(masthead);
    app.appendChild(main);
    app.appendChild(strip);

    var ep = document.getElementById("spec-endpoint");
    if (ep) ep.addEventListener("click", function () {
      copyText(ep.textContent).then(function (ok) {
        if (!ok) return;
        var t = ep.textContent;
        ep.textContent = "copied.";
        setTimeout(function () { ep.textContent = t; }, 900);
      });
    });
    refreshSpec();
    refreshPersonas();
  }

  // Best-effort: fill the substrate chip + spec strip from whatever status endpoint answers.
  // Defensive on shape (servers differ across versions); silent when nothing is reachable.
  function refreshSpec() {
    getJSON("/health", null).then(function (h) {
      return h || getJSON("/state", null);
    }).then(function (h) {
      if (!h || typeof h !== "object") return;
      var sub = h.substrate || h.active_substrate || null;
      var model = h.model || h.model_name || null;
      var chip = document.getElementById("subchip");
      if (chip && sub) { chip.textContent = String(sub)[0]; chip.title = "substrate: " + sub; }
      var m = document.getElementById("spec-model");
      if (m && (model || sub)) m.textContent = "model " + (model || sub);
    });
  }

  // ---- persona picker (masthead): which profile bundle (research/profiles.py) is applied ----------
  // A profile chip (first letter of the active profile's name, "·" when none) + a plain <select> of
  // every saved profile. Choosing a DIFFERENT one POSTs /profiles/switch -- cards replace, dials
  // replace, instant in prompt mode (see profiles.py). Creating/
  // exporting/importing bundles lives on the Settings page; this widget only lists + switches.
  var persona = { profiles: [], active: null, switching: false };

  function buildPersonaPicker() {
    // Built here but only POPULATED by refreshPersonas() once the masthead is attached to the document
    // (called from buildShell(), same timing as refreshSpec()) -- these ids don't exist to look up before then.
    var chip = el("span", { class: "subchip", id: "personachip", title: "active profile: none" }, ["·"]);
    var sel = el("select", { class: "mh-personasel", id: "mh-personasel", title: "switch profile" }, [
      el("option", { value: "" }, ["profile: none saved"]),
    ]);
    sel.disabled = true;
    sel.addEventListener("change", function () { switchPersona(sel.value); });
    return el("span", { class: "mh-persona" }, [sel, chip]);
  }

  function refreshPersonas() {
    getJSON("/profiles/list", null).then(function (r) {
      var sel = document.getElementById("mh-personasel");
      var chip = document.getElementById("personachip");
      persona.profiles = (r && r.profiles) || [];
      persona.active = r ? r.active : null;
      if (sel) {
        sel.innerHTML = "";
        if (!persona.profiles.length) {
          sel.appendChild(el("option", { value: "" }, [r ? "profile: none saved" : "profile: unavailable"]));
          sel.disabled = true;
        } else {
          sel.appendChild(el("option", { value: "" }, ["profile: none active"]));
          persona.profiles.forEach(function (p) {
            sel.appendChild(el("option", { value: p.name }, [
              "profile: " + p.name + (p.name === persona.active ? " (active)" : ""),
            ]));
          });
          sel.value = persona.active || "";
          sel.disabled = persona.switching;
        }
      }
      if (chip) {
        var letter = persona.active ? String(persona.active)[0] : "·";
        chip.textContent = letter;
        chip.title = persona.active ? "active profile: " + persona.active : "active profile: none";
      }
    });
  }

  function switchPersona(name) {
    if (!name || name === persona.active || persona.switching) return;
    persona.switching = true;
    var sel = document.getElementById("mh-personasel");
    if (sel) sel.disabled = true;
    postJSON("/profiles/switch", { name: name }, null).then(function () {
      persona.switching = false;
      refreshPersonas();
    });
  }

  function boot() {
    buildShell();
    booted = true;
    window.addEventListener("hashchange", mountCurrent);
    if (!location.hash) location.replace("#/agent");
    mountCurrent();
  }

  // ---- public API for page modules -------------------------------------------------------------
  window.CloznStudio = {
    register: register,
    navigate: navigate,
    base: BASE,
    // expose helpers so pages can reuse them without re-implementing
    esc: esc, el: el, pageHead: pageHead, getJSON: getJSON, postJSON: postJSON,
    fmtTime: fmtTime, fmtDate: fmtDate, fmtDuration: fmtDuration, isToday: isToday, copyText: copyText,
    // so a page that saves/switches/imports a profile (Settings) can repaint the masthead chip + dropdown
    // without a full reload -- the masthead is the one shared surface, owned here.
    refreshPersonas: refreshPersonas,
  };

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot);
  else boot();
})();
