---
name: d3js-graph-visualization
description: >-
  Build interactive graph visualizations in HTML with d3.js (v4), covering the
  radial edge-bundling layout and the force-directed layout used by the
  ARE-d3js / ARExplorer UI. Use when the user wants to render a radial graph or
  force graph, visualize nodes/links with sentiment-colored edges, embed a d3
  graph in an HTML page, or adapt the ARE-d3js template.
---

# d3.js Graph Visualization

Render interactive entity/relation graphs in plain HTML with **d3.js v4**. Two
layouts are supported, matching the ARExplorer UI:

- **Force graph** — physics-based node/link layout, draggable nodes.
- **Radial graph** — nodes on a circle with hierarchical edge bundling.

Load d3 v4 once in the page:

```html
<script src="https://d3js.org/d3.v4.min.js"></script>
<svg></svg>
```

Edges are colored by a `sent` (sentiment) field: `pos` → blue, `neg` → red,
`neu` → grey.

## Choosing a layout

| Use the... | When |
|------------|------|
| Force graph | Relationships matter more than grouping; nodes should spread out by repulsion; you want drag interaction. |
| Radial graph | Many nodes with a clear name/hierarchy; you want a compact circular layout with bundled edges and hover highlighting. |

## Force graph

Data shape — flat `nodes` + `links`, links reference node `id`s:

```json
{
  "nodes": [{ "id": "Anna", "c": 1200 }, { "id": "Pierre", "c": 800 }],
  "links": [{ "source": "Anna", "target": "Pierre", "c": 0.4, "sent": "pos" }]
}
```

- `c` on a node = frequency/count (drives fill opacity).
- `c` on a link = weight (drives `stroke-width`).

Core pattern: `d3.forceSimulation()` with three forces, then render links as
`line`s and nodes as `g` groups containing a `circle` + `text`:

```js
const simulation = d3.forceSimulation()
    .force("link", d3.forceLink().id(d => d.id).distance(150))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2));
```

See the full runnable example in [resources/force-graph.html](resources/force-graph.html).

## Radial graph

Data shape — flat list of nodes addressed by dotted `name`; edges live in each
node's `imports`:

```json
[
  { "name": "root.Anna",  "w": 0.5, "imports": [{ "name": "root.Pierre", "w": 0.4, "sent": "pos" }] },
  { "name": "root.Pierre", "w": 0.3, "imports": [] }
]
```

- The dotted `name` builds the hierarchy (`packageHierarchy`); use a common
  prefix (e.g. `root.`) so all leaves share one parent.
- `imports[].w` drives edge `stroke-width`; `imports[].sent` drives color.

Core pattern: `d3.cluster()` for layout + `d3.radialLine()` with
`d3.curveBundle` for bundled edges:

```js
const cluster = d3.cluster().size([360, innerRadius]);
const line = d3.radialLine()
    .curve(d3.curveBundle.beta(0.85))
    .radius(d => d.y)
    .angle(d => d.x / 180 * Math.PI);
```

See the full runnable example in [resources/radial-graph.html](resources/radial-graph.html).

## Common conventions (both layouts)

- **Clear before redraw**: `d3.select("svg").remove()` then append a fresh `svg`.
- **Sentiment colors**: `{ pos: "blue", neg: "red", neu: "grey" }` (use
  `class="blue-text"/"red-text"/"grey-text"` for matching legend labels).
- **Directional arrows**: define a reusable `marker` in `<defs>` and attach it
  via `marker-end: url(#arrow)` when highlighting a selected node's edges.
- **Edge filtering**: drop edges whose `sent` checkbox is unchecked by setting
  their `opacity` to `0` rather than removing them.

## Resources

- [resources/force-graph.html](resources/force-graph.html) — standalone force layout with inline sample data.
- [resources/radial-graph.html](resources/radial-graph.html) — standalone radial layout with inline sample data.
- [resources/template.html](resources/template.html) — full ARE-d3js UI template (server-backed) the snippets are extracted from.
