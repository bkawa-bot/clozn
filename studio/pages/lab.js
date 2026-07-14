/* lab.js -- the Lab page.  Issue H1: re-home the deep glass-box surfaces under the studio shell.
 *
 * The deep/internal tools already exist as standalone surfaces the studio server serves out of the
 * demo dir (engine.html / brain.html / denoise.html -- see research/clozn_server.py do_GET, which
 * serves any *.html|*.css|*.js under DEMO by basename, and their /engine/*, /say, ... endpoints).
 * Lab is the doorway to them from INSIDE the shell: one card per surface, each with a one-line
 * description and an "Open" affordance that mounts that surface in an on-demand in-page iframe (so it
 * stays within Lab), plus a collapse/close and an "open in its own window" fallback link.
 *
 * Why iframe (not a native port): these surfaces are large self-contained apps with their own scripts
 * and API base (same-origin when served, 127.0.0.1:8090 from file://) -- an iframe re-homes them as-is
 * without a rewrite, and the backend already serves the .html + endpoints. Framework-free; reuses the
 * shell's S.el + clozn.css (the .embed / .labgrid / .labtool / .cs-badge classes already exist).
 *
 * NOTE on instrument.html: it's itself only a tab-chrome wrapper that hosts these same three leaf
 * surfaces in an iframe (with a substrate-switch prompt). Embedding it here would nest an iframe in an
 * iframe, so Lab reaches the three LEAF surfaces directly and offers instrument.html only as a plain
 * "combined instrument" link at the foot for anyone who wants the all-in-one window.
 */
(function () {
  "use strict";
  var S = window.CloznStudio;

  // Each internal surface: the served .html, a title, a one-line "what it shows", the substrate it
  // wants (informational only -- the engine runs on any GGUF), and whether it embeds cleanly.
  var SURFACES = [
    {
      href: "engine.html",
      name: "Engine internals",
      tagline: "read · edit · write · observe — the real C++ runtime",
      desc: "The white-box loop on the actual cloze-server (ggml), not a Python wrapper. " +
        "Harvest every token's residual in one forward pass, scale one back into the model, and " +
        "watch the next-token prediction move — plus a contrastive tone dial applied through " +
        "the engine during generation.",
      needs: "any GGUF (separate engine process)",
    },
    {
      href: "brain.html",
      name: "Brain · concept readout",
      tagline: "the concepts a prompt lights up, as a luminous 3D graph",
      desc: "The model's legible interior: concepts as bioluminescent nodes that cluster into glowing " +
        "lobes, synapses between them. Rotate / zoom / hover to flare a concept and light its " +
        "neighbours, or fire a thought to watch activation cascade across the brain.",
      needs: "qwen substrate",
    },
    {
      href: "denoise.html",
      name: "Denoise trace",
      tagline: "watch a diffusion model crystallise an answer out of noise",
      desc: "A diffusion LM (Dream-7B) doesn't write left-to-right — it fills a board of masked " +
        "slots in parallel over a handful of passes, each token landing with a confidence (brightness), " +
        "sometimes re-masking a low-confidence one (a red flash) to redo it. Captured pass by pass.",
      needs: "dream substrate",
    },
    {
      href: "jlens.html",
      name: "The j-lens",
      tagline: "paste any text, read what it's disposed to say at a layer",
      desc: "A free-text lab for the J-lens: paste any text, pick a layer, and read a fitted linear " +
        "Jacobian lens's per-token top-k readout — what the model was leaning toward saying at each " +
        "position. The same panel as the Run Inspector's own-answer read, but on whatever you paste. " +
        "Not a decode of literal thought — a linear lens always emits something, and every readout " +
        "carries its provenance caption alongside it.",
      needs: "engine substrate with a J-lens loaded (--jlens)",
    },
  ];

  // page-scoped styles: the intro + the per-card embed slot / toolbar. Everything else reuses clozn.css
  // and the shell's own .labgrid/.labtool/.embed/.cs-badge. Kept small and additive.
  var STYLE_ID = "lab-page-style";
  var CSS =
    ".lab-cards{display:flex;flex-direction:column;gap:16px;margin-top:20px}" +
    ".lab-card{padding:16px 18px 14px}" +
    ".lab-head{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;flex-wrap:wrap}" +
    ".lab-t{font-size:16px;font-weight:660;color:var(--ink)}" +
    ".lab-tag{font-size:12px;color:var(--soft);margin-top:2px}" +
    ".lab-need{display:inline-block;font-size:10px;letter-spacing:.06em;color:var(--faint);" +
    "border:1px solid var(--line);border-radius:9px;padding:2px 8px;margin-top:8px}" +
    ".lab-d{color:var(--faint);font-size:13px;line-height:1.5;margin:10px 0 0;max-width:760px}" +
    ".lab-acts{display:flex;gap:9px;align-items:center;flex-none;flex-wrap:wrap}" +
    ".lab-open.on{background:linear-gradient(180deg,#dfeaff,#cfe0ff);border-color:#a9c6ff;font-weight:620}" +
    ".lab-newwin{font-size:12.5px;color:var(--faint);text-decoration:none;white-space:nowrap}" +
    ".lab-newwin:hover{color:var(--halo)}" +
    ".lab-slot{margin-top:14px}" +          // holds the on-demand iframe (empty until Open)
    ".lab-slot .embed{display:block}" +      // .embed comes from app.html's shell styles
    ".lab-foot{margin-top:22px;color:var(--faint);font-size:12.5px;line-height:1.55}" +
    ".lab-foot a{color:var(--halo);text-decoration:none;font-weight:600}" +
    ".lab-foot a:hover{text-decoration:underline}";

  function ensureStyle() {
    if (document.getElementById(STYLE_ID)) return;
    var st = document.createElement("style");
    st.id = STYLE_ID;
    st.textContent = CSS;
    document.head.appendChild(st);
  }

  function render(view, ctx) {
    ensureStyle();

    view.appendChild(
      S.el("div", { class: "wrap" }, [
        S.pageHead({
          kicker: "deep glass-box tools",
          kickerRight: "reads & writes real internals",
          title: "lab",
          counter: "experimental surfaces that talk straight to the runtime — open one here or pop it out",
        }),

        S.el("div", { class: "lab-cards" }, SURFACES.map(function (s) { return card(s, ctx); })),

        // foot: the combined instrument window (an acceptable plain-link fallback -- see file header).
        S.el("p", { class: "lab-foot" }, [
          "Prefer everything in one window? The ",
          S.el("a", { href: "instrument.html", target: "_blank", rel: "noopener" }, ["combined instrument ↗"]),
          " hosts all three of these with a substrate switcher.",
        ]),
      ])
    );
  }

  // one card: title + tagline + substrate chip + description on the left, actions on the right, and a
  // slot underneath that fills with an iframe on demand.
  function card(s, ctx) {
    // the embed slot (empty until Open) + a toggling Open/Close button that mounts/removes the iframe.
    var slot = S.el("div", { class: "lab-slot" }, []);

    var openBtn = S.el("button", { class: "lab-open" }, ["Open here"]);
    openBtn.addEventListener("click", function () {
      if (slot.firstChild) {           // currently open -> collapse (drop the iframe, freeing the surface)
        slot.innerHTML = "";
        openBtn.textContent = "Open here";
        openBtn.classList.remove("on");
        return;
      }
      // mount the surface in an in-page iframe. Relative src resolves against this origin, which is where
      // the studio server serves the .html from (and the surface figures out its own API base).
      var frame = S.el("iframe", {
        class: "embed",
        src: s.href,
        title: s.name,
        loading: "lazy",
      }, []);
      slot.appendChild(frame);
      openBtn.textContent = "Close";
      openBtn.classList.add("on");
      // bring the freshly opened surface into view (it can be tall).
      slot.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });

    // fallback: always offer the surface in its own tab (works even if the iframe is awkward for it).
    var newWin = S.el("a", {
      class: "lab-newwin", href: s.href, target: "_blank", rel: "noopener",
      title: "open " + s.name + " in its own window",
    }, ["own window ↗"]);

    return S.el("section", { class: "panel lab-card" }, [
      S.el("div", { class: "lab-head" }, [
        S.el("div", {}, [
          S.el("div", { class: "lab-t" }, [s.name]),
          S.el("div", { class: "lab-tag" }, [s.tagline]),
          s.needs ? S.el("span", { class: "lab-need" }, [s.needs]) : null,
        ].filter(Boolean)),
        S.el("div", { class: "lab-acts" }, [openBtn, newWin]),
      ]),
      S.el("div", { class: "lab-d" }, [s.desc]),
      slot,
    ]);
  }

  S.register("lab", { title: "Lab", render: render });
})();
