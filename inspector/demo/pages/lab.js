/* lab.js -- Lab page.  STUB for folding in the deep glass-box demos (Issue H1).
 * For now: a short "coming soon" note + links/embed to the existing low-level surfaces
 * (engine harvest/state/intervene, the brain/concept viz, denoise trace) so they stay reachable.
 */
(function () {
  "use strict";
  var S = window.CloznStudio;

  var TOOLS = [
    { href: "engine.html", name: "Engine", desc: "harvest / read / intervene on live runtime state (C++ engine)." },
    { href: "brain.html", name: "Brain / concepts", desc: "concept atlas + feature readout." },
    { href: "denoise.html", name: "Denoise trace", desc: "watch a diffusion model fill masks step by step." },
    { href: "instrument.html", name: "Instrument", desc: "lower-level instrumentation surface." },
  ];

  function render(view, ctx) {
    var cards = TOOLS.map(function (t) {
      return S.el("a", { class: "labtool", href: t.href, target: "_blank", rel: "noopener" }, [
        S.el("div", { class: "labtool-n" }, [t.name, S.el("span", { class: "labtool-arrow" }, [" ↗"])]),
        S.el("div", { class: "labtool-d" }, [t.desc]),
      ]);
    });
    view.appendChild(
      S.el("div", { class: "wrap" }, [
        S.el("h1", {}, [S.el("span", { class: "glow" }, ["Lab"])]),
        S.el("p", { class: "sub" }, [
          "Advanced tools for inspecting and experimenting with model internals — activation/state views, concepts, residual edits, engine harvest/write/observe. These will fold into the Run Inspector; for now they open in their existing windows.",
        ]),
        S.el("div", { class: "stubnote" }, [
          S.el("span", { class: "cs-badge" }, ["coming soon"]),
          " These surfaces will be re-homed inside the shell and tied to a specific run.",
        ]),
        S.el("div", { class: "labgrid" }, cards),
      ])
    );
  }
  S.register("lab", { title: "Lab", render: render });
})();
