"""ARExplorer agent — extracts Attitudes and Relations from documents.

ADK 2.0 root agent exposing four tools (wrapped over their Python APIs):
  #1 extract_named_entities  — bulk-ner annotation over massive text collections
  #2 form_entity_pairs       — turn NER documents into candidate {source,target,text} pairs
  #3 classify_relations      — bulk-chain attitude/relation classification
  #4 graph_operation         — union / intersection over attitude graphs
"""

import json
import os

from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.tools import ToolContext
from google.adk.tools.load_artifacts_tool import load_artifacts_tool
from google.genai import types

from src.callbacks.after_model import ensure_nonempty_response
from src.callbacks.before_tool import inflate_artifact_inputs
from src.callbacks.after_tool import offload_tool_output
from src.schema import AgentResponse
from src.tools import (
    extract_named_entities as _extract_named_entities,
    form_entity_pairs as _form_entity_pairs,
    classify_relations as _classify_relations,
    graph_operation as _graph_operation,
    build_output_graph as _build_output_graph,
)

# Canonical session artifact under which the `output` tool stores the finished,
# schema-validated AgentResponse. The server prefers this (for the matching
# tool call) over any graph the model hand-writes into set_model_response.
OUTPUT_ARTIFACT_NAME = "agent_response.json"


# The `*_artifact` parameters live ONLY on these wrappers (not on the underlying
# `src.tools.*` functions) because that is what ADK 2.0 turns into the LLM-facing
# JSON schema. The before_tool_callback `inflate_artifact_inputs` consumes them
# and rewrites `args` so the wrapper itself never sees them. Per-param
# descriptions live in the docstring because ADK calls `get_type_hints` without
# `include_extras=True`, which strips any `Annotated[..., Field(...)]` metadata
# before schema generation.


def extract_named_entities(
    texts: list[str] | None = None,
    texts_artifact: str | None = None,
    batch_size: int = 10,
) -> dict:
    """Extract named entities from a collection of texts.

    Provide EXACTLY ONE of `texts` or `texts_artifact`:

    - `texts`: a list of strings to annotate, supplied inline. Best for short
      inputs that fit comfortably in the prompt.
    - `texts_artifact`: filename of a session artifact whose JSON content is
      the input. The artifact must decode to a list of strings, or to an
      object with a `texts` list (e.g. `{"texts": [...]}`). The before_tool
      callback loads the artifact and substitutes its content before this
      function runs. Prefer this form for large corpora.

    `batch_size` controls how many texts are processed per batch.
    """
    return _extract_named_entities(
        texts,
        batch_size=batch_size,
        src_dir=os.environ.get("NER_SRC_DIR"),
        class_filepath=os.environ.get("NER_CLASS_FILEPATH"),
        class_name=os.environ.get("NER_CLASS_NAME"),
        model=os.environ.get("NER_MODEL"),
    )


def form_entity_pairs(
    documents: list[dict] | None = None,
    documents_artifact: str | None = None,
    entity_types: list[str] | None = None,
    directed: bool = True,
    window_size: int = 5,
    context_pad: int = 5,
    skip_self_pairs: bool = True,
    max_pairs: int | None = 50,
) -> dict:
    """Form candidate {text, source, target} pairs from NER documents.

    This is the bridge between `extract_named_entities` and
    `classify_relations`: it consumes per-document entity lists and emits the
    pair triples the attitude classifier expects. To keep the pair list and
    each pair's context compact:

    - Only pairs whose two occurrences are within `window_size` words of
      each other (gap STRICTLY BETWEEN them) are emitted (default 25).
    - Each pair's `text` is the LOCAL context around the pair (the words
      from the earlier occurrence to the later, plus `context_pad` extra
      words on each side, default 5) — NOT the full source document.
    - At most `max_pairs` pairs are returned overall (default 50); when more
      candidates pass the window filter, those with the SMALLEST gap are
      kept first. Pass ``None`` to disable the cap. This bounds the cost of
      the downstream `classify_relations` call.

    Provide EXACTLY ONE of `documents` or `documents_artifact`:

    - `documents`: inline list of `{text, entities}` dicts (the shape produced
      by `extract_named_entities`). Each `entities` item must at least carry
      `value` (the entity surface form) and `type` (NER class).
    - `documents_artifact`: filename of a session artifact whose JSON content
      is the NER output (i.e. `{"documents": [...]}` or a bare list of such
      dicts). The before_tool callback loads and substitutes it. This is the
      natural plumbing right after `extract_named_entities` — pass the
      artifact name from its summary directly.

    `entity_types`: optional whitelist of NER classes (e.g. ``["PERSON",
    "ORG"]``); ``None`` keeps every entity type. `directed`: when True the
    pair (A, B) and (B, A) are emitted separately. `skip_self_pairs`: drops
    pairs where source == target by surface form.
    """
    return _form_entity_pairs(
        documents=documents,
        entity_types=entity_types,
        directed=directed,
        window_size=window_size,
        context_pad=context_pad,
        skip_self_pairs=skip_self_pairs,
        max_pairs=max_pairs,
    )


def graph_operation(
    operation: str,
    graph_a: dict | None = None,
    graph_b: dict | None = None,
    graph_a_artifact: str | None = None,
    graph_b_artifact: str | None = None,
) -> dict:
    """Combine two attitude/relation graphs with a set operation.

    `operation` is either ``"union"`` (all nodes/edges present in either
    input, deduplicated) or ``"intersection"`` (only nodes/edges present in
    both).

    For each of the two inputs provide EXACTLY ONE of the inline / artifact
    forms:

    - `graph_a` / `graph_b`: graph dict shaped
      ``{"nodes": [...], "edges": [{"source", "target", "label"}, ...]}``.
    - `graph_a_artifact` / `graph_b_artifact`: filename of a session
      artifact whose JSON decodes either to a bare graph dict (above shape)
      or to an object containing a top-level `"graph"` key with the graph
      dict — which is exactly the shape produced by an earlier
      `graph_operation` call's artifact (a previous offloaded
      ``{"status": "success", "graph": {...}}`` payload). The before_tool
      callback loads the artifact and substitutes its content before this
      function runs.

    Mixing forms across the two inputs is fine (e.g. inline `graph_a` plus
    `graph_b_artifact`). Use the artifact form when combining the outputs
    of two previous `graph_operation` runs without re-loading them
    yourself.
    """
    return _graph_operation(
        operation=operation,
        graph_a=graph_a,
        graph_b=graph_b,
    )


def classify_relations(
    pairs: list[dict] | None = None,
    pairs_artifact: str | None = None,
    relation_type: str = "sentiment",
    batch_size: int = 10,
) -> dict:
    """Classify the attitude/relation between entity pairs.

    Provide EXACTLY ONE of `pairs` or `pairs_artifact`:

    - `pairs`: list of `{text, source, target}` dicts to classify inline.
    - `pairs_artifact`: filename of a session artifact whose JSON content is
      the pairs list. Must decode to a list of `{text, source, target}`
      dicts, or to an object with a `pairs` list. The before_tool callback
      loads the artifact and substitutes its content before this function
      runs. Prefer this form when the pairs list is already in the session.

    `relation_type` parameterises the underlying Chain-of-Thought schema
    (default "sentiment"). `batch_size` controls per-batch LLM throughput.
    """
    return _classify_relations(
        pairs,
        relation_type=relation_type,
        batch_size=batch_size,
        provider_filepath=os.environ.get("RELATION_PROVIDER_FILEPATH"),
        model_name=os.environ.get("RELATION_MODEL"),
        api_token=os.environ.get("REPLICATE_API_TOKEN", ""),
    )


async def output(
    message: str,
    layout: str = "force",
    relations: list[dict] | None = None,
    relations_artifact: str | None = None,
    tool_context: ToolContext = None,
) -> dict:
    """Build the final visualization graph and emit the structured response.

    Use this as the LAST step whenever you have classified relations to plot.
    It converts the `classify_relations` output into the UI graph for you, so
    you DO NOT hand-write the (potentially long) `graph` yourself — that avoids
    truncation and schema mistakes.

    Provide EXACTLY ONE of `relations` or `relations_artifact`:

    - `relations`: inline list of `{source, target, label}` dicts.
    - `relations_artifact`: filename of the `classify_relations` session
      artifact (its summary's `artifact.name`). The before_tool callback loads
      it and substitutes the relations list before this function runs. Prefer
      this form — it is the whole point of the tool.

    `message` is the natural-language reply for the chat panel (required, never
    empty). `layout` is "force" (default) or "radial".

    Returns a small summary (`status`, `node_count`, `edge_count`, and an
    `artifact` pointer). The full, schema-validated `AgentResponse` (your
    `message` + the built `graph` + `layout`) is saved as the
    ``agent_response.json`` session artifact, which the server uses as the
    authoritative final answer. After calling this tool, finish via
    `set_model_response` with the same `message` and an EMPTY `graph`.
    """
    if relations is None:
        return {
            "status": "error",
            "error": (
                "output: provide `relations` inline or a `relations_artifact` "
                "filename (the classify_relations artifact) so the "
                "before_tool_callback can inflate it."
            ),
        }

    graph = _build_output_graph(relations)
    response = AgentResponse.model_validate(
        {"message": message, "layout": layout, "graph": graph}
    )

    summary = {
        "status": "success",
        "node_count": len(response.graph.nodes),
        "edge_count": len(response.graph.edges),
    }

    if tool_context is not None:
        payload = json.dumps(
            response.model_dump(), ensure_ascii=False
        ).encode("utf-8")
        part = types.Part(
            inline_data=types.Blob(
                mime_type="application/json", data=payload
            )
        )
        try:
            version = await tool_context.save_artifact(
                OUTPUT_ARTIFACT_NAME, part
            )
            summary["artifact"] = {
                "name": OUTPUT_ARTIFACT_NAME,
                "version": version,
            }
        except ValueError as exc:
            return {
                "status": "error",
                "error": f"output: failed to save response artifact: {exc}",
            }

    return summary


INSTRUCTION = """\
You are ARExplorer, an assistant that extracts and explores Attitudes and
Relations between named entities found in documents.

Your typical workflow:
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
5. When the user wants to combine or compare result sets, build graphs of the
   form {"nodes": [...], "edges": [{"source", "target", "label"}]} and use
   `graph_operation` with "union" or "intersection". When you already have
   the graphs as session artifacts (e.g. the outputs of two earlier
   `graph_operation` runs), pass them as `graph_a_artifact` /
   `graph_b_artifact` instead of inlining the graph dicts.

Be transparent about tool errors and ask for missing inputs (e.g. text context
for a relation) rather than guessing.

TOOL OUTPUTS ARE OFFLOADED TO ARTIFACTS:
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

TOOL INPUTS CAN ALSO COME FROM ARTIFACTS:
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
  containing such a dict — exactly what an earlier `graph_operation` call
  has already written.
- The framework loads the artifact and substitutes its content for the
  inline value before the tool runs. Prefer the artifact form whenever the
  input is large or already lives in the session (e.g. uploaded by the
  user, or written by an earlier turn). Provide exactly one of the inline
  or artifact form per input.

OUTPUT FORMAT (important):
- Always deliver your final answer through the `set_model_response` tool, in the
  structured `AgentResponse` format. Never reply with plain free-form text.
- When there are classified relations to plot, the graph is produced by the
  `output` tool (see workflow step 4), NOT by you. In that case call `output`
  first, then call `set_model_response` with your `message` and `layout` and an
  EMPTY `graph` — the server uses the graph that `output` built. Do NOT
  hand-write graph nodes/edges yourself; that output can be long and is
  error-prone.
- `message`: a clear natural-language reply for the chat panel. Summarize what
  was found and how to read the graph.
- `graph`: leave EMPTY when you used `output`. Only populate it directly for a
  graph you assembled yourself without `output` (e.g. the result of a
  `graph_operation`), using `nodes` ({id, weight}) and `edges` ({source,
  target, relation in "positive"/"negative"/"neutral", weight}). Leave empty
  when there is genuinely nothing to plot.
- `layout`: "radial" for many entities with a clear hierarchy, otherwise "force".
"""

root_agent = Agent(
    name="arexplorer_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=types.HttpRetryOptions(attempts=10, initial_delay=1.0),
    ),
    instruction=INSTRUCTION,
    output_schema=AgentResponse,
    tools=[
        extract_named_entities,
        form_entity_pairs,
        classify_relations,
        graph_operation,
        output,
        load_artifacts_tool,
    ],
    before_tool_callback=inflate_artifact_inputs,
    after_tool_callback=offload_tool_output,
    after_model_callback=ensure_nonempty_response,
)
