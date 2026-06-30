---
name: arexplorer-workflow
description: >-
  Use this skill whenever the user wants to find, extract, or explore who feels
  what about whom across one or more documents — the named entities (people,
  organizations, places) and the attitudes or relations between them — even if
  they never say "NER", "entities", "sentiment", or "graph". Apply it for
  requests like analyzing a text for the parties involved and their stances,
  mapping positive/negative/neutral attitudes between them, building an attitude
  or relation graph, or combining and comparing several such result sets (union
  / intersection). Reach for this skill any time raw text needs to be turned
  into entities and the relations among them.
---

# ARExplorer workflow

Typical pipeline for turning raw documents into an attitude/relation graph.

1. Use `extract_named_entities` to annotate named entities across the provided
   texts. This handles large volumes by chunking, so prefer it over reasoning
   about entities yourself.
2. Use `form_entity_pairs` to turn the NER output into candidate
   `{text, source, target}` triples. Pass the NER artifact name as
   `documents_artifact` — you do NOT need to load the NER artifact yourself
   first. Each pair's `text` is a SMALL LOCAL WINDOW around the two entities
   (not the full source text); pairs whose entities are more than
   `window_size` words apart (default 5) are dropped. The tool also caps
   the output at `max_pairs` (default 50) — keeping the closest-gap pairs
   first — so downstream relation classification stays cheap. Tune
   `window_size` / `context_pad` / `max_pairs` only when the user
   explicitly asks. Optionally filter by NER class via `entity_types`.
3. Use `classify_relations` to determine the attitude (positive, negative, or
   neutral) that each source entity expresses towards the target entity. Pass
   the pairs artifact name as `pairs_artifact` from the previous step.
4. Use `output` to turn the `classify_relations` result into the final
   visualization. Pass the classify_relations artifact name as
   `relations_artifact` plus your `message` and `layout`. The tool builds the
   graph (nodes + edges) deterministically from the artifact and saves the
   finished structured response — so you must NOT load the relations yourself
   or hand-write the graph. This is the standard final step whenever there are
   classified relations to plot.
5. When the user wants to combine or compare result sets, use
   `graph_operation` with "union" or "intersection". Pass the two graphs as
   `graph_a_artifact` / `graph_b_artifact` — do NOT call `load_artifacts`
   for this; the before_tool callback inflates the artifacts for you.

   **Which artifact names to use:** each successful `output` call returns an
   `artifact` pointer to a unique file (``output_<call_id>.json``). Pass two
   such filenames — one from each earlier analysis — as `graph_a_artifact` and
   `graph_b_artifact`. You can also use a `graph_operation_<call_id>.json`
   artifact from a prior `graph_operation` run.

   Then present the merged graph: `output(graph_artifact=<graph_operation
   artifact name>, message=...)`. Call `output` **once** with the union
   result — do not re-run `output` on the individual source graphs afterward.

Be transparent about tool errors and ask for missing inputs (e.g. text context
for a relation) rather than guessing.

## Tool outputs are offloaded to artifacts

- `extract_named_entities`, `form_entity_pairs`, `classify_relations`, and
  `graph_operation` do NOT return their raw data to you. Each successful call
  returns only:
    * `status`, lightweight counts (e.g. `document_count` / `entity_count`,
      `pair_count`, `relation_count` / `label_counts`,
      `node_count` / `edge_count`), and
    * an `artifact` field of shape {"name": "<file>.json", "version": <int>}.
  The full `documents` / `pairs` / `relations` / `graph` payload is saved as
  a JSON artifact in the session artifact store.
- To read the actual data (look up labels, populate the final graph), call
  the `load_artifacts` tool with the artifact name, e.g.
  `load_artifacts(artifact_names=["classify_relations_<id>.json"])`. The
  JSON content of each requested artifact is then injected into your next
  turn so you can reason over it.
- Only call `load_artifacts` when you genuinely need the content of an
  artifact; the counts alone are usually enough to decide what to do next.
  In particular, chaining NER -> pairs -> relations does NOT require any
  `load_artifacts` call — each step accepts the previous step's artifact
  name as a `*_artifact` argument.

## Tool inputs can also come from artifacts

- `extract_named_entities` accepts `texts_artifact` instead of `texts`.
- `form_entity_pairs` accepts `documents_artifact` instead of `documents`.
- `classify_relations` accepts `pairs_artifact` instead of `pairs`.
- `output` accepts `relations_artifact` instead of `relations` — pass the
  `classify_relations` artifact name here.
- `graph_operation` accepts `graph_a_artifact` / `graph_b_artifact`
  instead of `graph_a` / `graph_b` (you may mix forms across the two
  inputs).
- For list-shaped tools the artifact must JSON-decode to a list of the
  expected shape (strings for texts, {text, entities} dicts for documents,
  {text, source, target} dicts for pairs) or to an object with a matching
  key (`{"texts": [...]}` / `{"documents": [...]}` / `{"pairs": [...]}` —
  which is exactly the shape the previous tool's artifact already has).
- For `graph_operation` the artifact must JSON-decode to a graph dict
  ({"nodes": [...], "edges": [...]}) or to an object with a `"graph"` key
  containing such a dict — exactly what an earlier `graph_operation` or
  `output` artifact already contains (`output_<id>.json` stores the full
  `AgentResponse`, whose `graph` field is unwrapped automatically).
- Artifact names may include a version pin: `filename@3` loads version 3
  instead of the latest.
- The framework loads the artifact and substitutes its content for the
  inline value before the tool runs. Prefer the artifact form whenever the
  input is large or already lives in the session (e.g. uploaded by the
  user, or written by an earlier turn). Provide exactly one of the inline
  or artifact form per input.
