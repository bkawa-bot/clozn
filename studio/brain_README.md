# brain.html — The Luminous Interior

The flagship clozn visualization: a digital-yet-biological 3D view of a model's legible,
persistent interior. Concepts are bioluminescent nodes that cluster into glowing **lobes**
in physical space (an Obsidian-graph for a mind); translucent synapses flow between them;
soft lens-bloom and gentle breathing motion make it feel alive. You rotate, zoom and pan to
behold it; hover a concept to wake it; click to draw near; fire a thought to watch activation
cascade across the brain.

It is a **single self-contained HTML file** — open `brain.html` straight from disk in any
modern browser (Chrome/Edge/Firefox/Safari). No build step, no server, GPU-free.

Built on [`3d-force-graph`](https://github.com/vasturiano/3d-force-graph) (Three.js +
d3-force-3d) with a Three.js `UnrealBloomPass` for the glow. Visual identity: clozn
"Artificial Angels" — a light, luminous pearl-sky heaven (see
`../../.claude/.../clozn-visual-identity.md`); signature is Celestial Cyan + cool blue/grey.

---

## How to run

Just double-click `brain.html` (or open it via `file://`). All third-party code loads from a
CDN (esm.sh) via an ES-module import map, so the **first** open needs internet; thereafter the
browser caches it. Nothing else is required.

> Why a CDN and not vendored files? ES-module `import` of a *local* sibling `.js` is blocked
> by the browser's `file://` CORS policy, so a self-contained single file must either inline
> everything or load modules over https. The data is inlined; the libraries load from https.

To run fully offline, vendor the three pinned packages and rewrite the import map to relative
paths, **and serve over `http://`** (a local static server) so module CORS is satisfied.

---

## The data schema

The brain is described by one JSON object with three arrays:

```jsonc
{
  "clusters": [               // the glowing LOBES
    { "id": "light", "label": "Light", "color": "#36AEC4" }
  ],
  "nodes": [                  // the CONCEPTS (one bead of light each)
    { "id": "light:photon", "label": "photon", "cluster": "light", "value": 0.62 }
  ],
  "links": [                  // the ASSOCIATIONS (synapses)
    { "source": "light:photon", "target": "light:glow", "weight": 0.78 }
  ]
}
```

| field            | meaning                                                                 |
|------------------|-------------------------------------------------------------------------|
| `cluster.id`     | unique lobe id (referenced by `node.cluster`)                           |
| `cluster.label`  | human label shown in the legend + readout                              |
| `cluster.color`  | the lobe's hue (any CSS hex). Auto-**luminized** for the light theme.   |
| `node.id`        | unique node id (referenced by links). Any string.                      |
| `node.label`     | the concept's display name                                             |
| `node.cluster`   | which lobe it belongs to (a `cluster.id`)                              |
| `node.value`     | **0..1** — activation / importance → node radius, base glow, breath    |
| `link.source/target` | node ids the synapse connects                                      |
| `link.weight`    | **0..1** — association strength → synapse thickness + opacity + spring  |

Notes:
- `node.value` and `link.weight` are normalized to **0..1**. Scale your raw numbers into that
  range before export (e.g. min-max or a soft clip).
- Every node should belong to a cluster, and have at least one link (isolated nodes drift).
- Cluster colors are passed through a `luminize()` step so any hue (even a dark slate) is
  lifted into a bright, glowing band suitable for the light sky — so you don't have to
  hand-pick light colors for real data.

---

## Feeding REAL data (SAE features / concept co-activation)

The seed is a hand-shaped demo "mind" (12 semantic lobes). To behold a **real** model,
export your features to the schema above. Two ways to load it:

**A. Inline (keeps the single-file, double-click property).**
Replace the JSON inside the `<script type="application/json" id="brain-seed">…</script>`
block near the top of `brain.html` with your own.

**B. Override at runtime (no edits to the viz code).**
Add, *before* the module `<script>`, a small script that sets `window.BRAIN`:

```html
<script>
  window.BRAIN = { clusters:[...], nodes:[...], links:[...] };   // your exported data
</script>
```
or have your own loader `fetch()` a JSON file and assign `window.BRAIN` before the module
runs. If `window.BRAIN` exists it overrides the inline seed.

### Suggested mapping from an SAE / dictionary

| viz field        | source                                                                 |
|------------------|------------------------------------------------------------------------|
| `node`           | one **SAE feature** (or named concept direction)                       |
| `node.label`     | the feature's auto-interpreted name / top activating token            |
| `node.value`     | mean activation / activation frequency / importance, scaled to 0..1    |
| `node.cluster`   | a **community** over the feature graph (e.g. Louvain/Leiden on the     |
|                  | co-activation or cosine-similarity matrix) — each community = a lobe    |
| `link`           | a feature **pair** above a similarity/co-activation threshold          |
| `link.weight`    | cosine similarity, or normalized co-activation (PMI), scaled to 0..1    |
| `cluster.color`  | one hue per community (any palette; it gets luminized)                  |

Keep links **sparse** (threshold hard) — a few hundred to a couple thousand edges read as a
mind; tens of thousands turn into a hairball. Dense **intra**-community links + a few
**evocative cross**-community bridges is the look you want.

---

## Interactions

- **Orbit / zoom / pan** — drag to rotate, scroll to draw near, right-drag to pan.
- **Hover a concept** — it blooms; its neighbours brighten; the rest of the mind gently dims;
  the readout card names it, its lobe, activation, connection count and strongest neighbours.
- **Click a concept** — the camera eases into its neighbourhood and a small local thought
  blooms where you touched.
- **Hover a lobe in the legend** — raises that whole lobe, dims the others.
- **Click a lobe (legend) / "fire a thought"** — activation ignites the lobe and cascades
  outward hop-by-hop along the synapses, sending light travelling down the links: watch it
  think. "fire a thought" picks a lobe at random.
- **recenter** — re-frames the whole brain.
- **breathe** — toggles the ambient life (slow node shimmer, slow auto-orbit, idle "ambient
  thoughts" that wander every few seconds).
- **bloom** — toggles the UnrealBloom glow.

A `window.clozn` handle is exposed for guided tours over real data and for tuning:
`clozn.fireLobe(id)`, `clozn.fireFrom(nodeId, hops, gain)`, `clozn.hover(nodeId)`,
`clozn.focus(nodeId)`, `clozn.frameBrain(ms, mult)`, `clozn.dbg()`.

---

## Knobs — where to tune the look

All in `brain.html`, in the numbered sections of the module script.

**The glow / bloom** (§5, `UnrealBloomPass`) — the soul of the luminous feel:
- `strength` (≈0.55) — overall glow intensity.
- `radius` (≈0.62) — how feathery/wide the halos spread.
- `threshold` (≈0.80) — **must stay ABOVE the sky's luminance** or the whole pearl blooms and
  washes white. If you lighten the sky, raise this; if you darken the sky, you may lower it.

**The sky** (§4, `skyTexture` + `renderer.setClearColor` + `.backgroundColor`):
- the pearl gradient stops. Kept deliberately a touch below white to give the bloom threshold
  headroom (a near-white sky blooms into a white-out). Keep all three (clear color, scene
  gradient, graph `.backgroundColor`) in the same family.
- `scene.fog` (FogExp2) density — how far lights melt into the distance. Light; high values
  erase the cloud.

**The node bodies** (§2, `makeNode`):
- `baseR` — node radius from `value` + degree.
- core / rim / aura / halo / nucleus layers and their opacities. The cores are unlit
  `MeshBasic` (always luminous, never shadowed) deepened slightly below the luminized lobe hue
  for contrast on the pearl; the additive `halo`/`nucleus` feed the bloom; the soft pastel
  `aura` lifts a coloured glow. (Additive sprites over a light sky wash white if too strong;
  saturated normal-blend sprites stack into mud — the current mix is tuned to avoid both.)

**The layout / lobes** (§1 anchors + §3 forces):
- `SPREAD` (≈340) — radius of the ring of lobe-anchors → how far apart the lobes sit.
- `clusterForce` strength (`alpha*0.42`) — how tightly each lobe gathers into its own island.
- `charge` (short-range repulsion) + `link` distance/strength — local spacing & spring.

**Framing** (§14, `frameBrain`): explicit camera fit from the measured node cloud
(`mult ≈ 1.95` = how much of the frame the brain fills). The library's `zoomToFit`
under-estimates here because custom sprite/group node objects report ~no geometry bounds, so
we frame from the cloud centroid + radius ourselves. Camera near/far are widened (§4) so the
far side of the cloud is never clipped.

**Ambient life** (§6 breathing + auto-orbit; §14 ambient thoughts): the steady glow floor
(`0.48 + 0.40*value`) keeps every node luminous at rest; breathing is only a gentle ripple on
top; idle "ambient thoughts" fire a faint wandering cascade every few seconds.

---

## Dependencies (pinned)

Loaded via the import map in `brain.html`, all from esm.sh with `?external=three` so the whole
graph shares **one** `three` instance (required for the bloom composite to work):

- `three@0.183.2`
- `3d-force-graph@1.80.0`
- `three/examples/jsm/postprocessing/{EffectComposer,RenderPass,UnrealBloomPass}.js`
