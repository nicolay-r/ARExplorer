---
name: arexplorer-response
description: >-
  Use this skill whenever you are about to send your final answer to the user,
  so it reaches the two-panel UI correctly (a chat message plus a d3.js graph).
  Apply it on every reply that ends a turn — whether you are presenting an
  attitude or relation graph, returning a combined or compared result set, or
  answering conversationally with nothing to plot. It tells you how to package
  the chat message, when to include a graph versus leave it empty, and which
  layout the visualization should use.
---

# ARExplorer response format

Deliver every final answer through the `set_model_response` tool in the
structured `AgentResponse` format. Never reply with plain free-form text. The UI
splits the response into two panels: `message` drives the chat panel and `graph`
drives the d3.js visualization.

- `message`: a clear natural-language reply for the chat panel. Summarize what
  was found and how to read the graph.
- `graph`: the data the d3.js panel renders.
  - Leave it EMPTY when the `output` tool already built the graph (the server
    uses the graph `output` saved). In that case call `output` first, then call
    `set_model_response` with your `message` and `layout` and an empty `graph`.
    Do NOT hand-write graph nodes/edges yourself; that output can be long and is
    error-prone.
  - Only populate it directly for a graph you assembled yourself without
    `output` (e.g. the result of a `graph_operation`), using `nodes`
    ({id, weight}) and `edges` ({source, target, relation in
    "positive"/"negative"/"neutral", weight}).
  - Leave it empty when there is genuinely nothing to plot.
- `layout`: "radial" for many entities with a clear hierarchy, otherwise "force".
