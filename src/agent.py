"""ARExplorer agent — extracts Attitudes and Relations from documents.

ADK 2.0 root agent exposing four tools (wrapped over their Python APIs):
  #1 extract_named_entities  — bulk-ner annotation over massive text collections
  #2 form_entity_pairs       — turn NER documents into candidate {source,target,text} pairs
  #3 classify_relations      — bulk-chain attitude/relation classification
  #4 graph_operation         — union / intersection over attitude graphs
"""

import asyncio
import json
import os
import pathlib

from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.skills import load_skill_from_dir
from google.adk.tools import ToolContext
from google.adk.tools import skill_toolset
from google.adk.tools.load_artifacts_tool import load_artifacts_tool
from google.genai import types

from src import progress
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
from src.core.graph import Graph


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

    - `graph_a` / `graph_b`: graph dict (any accepted shape; edges may use
      `relation` or the legacy `label` key). The result is always serialised
      to the canonical ``{"nodes": [{"id", "weight"}], "edges": [{"source",
      "target", "relation", "weight"}]}`` form.
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


async def classify_relations(
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

    This can be slow (bulk-chain iterates over many pairs), so the heavy work
    runs in a worker thread (keeping the event loop free) and streams
    incremental ``(done, total)`` progress to the UI via the active progress
    publisher when one is installed (see `src.progress`).
    """
    publisher = progress.get_publisher()
    progress_callback = None
    if publisher is not None:
        total_hint = len(pairs) if pairs else 0

        def progress_callback(done: int, total: int) -> None:
            publisher(
                {
                    "type": "progress",
                    "tool": "classify_relations",
                    "done": done,
                    "total": total or total_hint,
                }
            )

    return await asyncio.to_thread(
        _classify_relations,
        pairs,
        relation_type=relation_type,
        batch_size=batch_size,
        provider_filepath=os.environ.get("RELATION_PROVIDER_FILEPATH"),
        model_name=os.environ.get("RELATION_MODEL"),
        api_token=os.environ.get("REPLICATE_API_TOKEN", ""),
        progress_callback=progress_callback,
    )


async def output(
    message: str,
    layout: str = "force",
    relations: list[dict] | None = None,
    relations_artifact: str | None = None,
    graph: dict | None = None,
    graph_artifact: str | None = None,
    tool_context: ToolContext = None,
) -> dict:
    """Build the final visualization graph and emit the structured response.

    Use this as the LAST step whenever you have something to plot. It builds
    the canonical UI graph for you, so you DO NOT hand-write the (potentially
    long) `graph` yourself — that avoids truncation and schema mistakes.

    Provide EXACTLY ONE source for the graph:

    - `relations` / `relations_artifact`: classified relations (the
      `classify_relations` output) as an inline list of `{source, target,
      label}` dicts, or the artifact filename to inflate. Use this right after
      `classify_relations`.
    - `graph` / `graph_artifact`: an already-built graph (e.g. the result of a
      `graph_operation`) as an inline dict, or the artifact filename to
      inflate. Use this to finalize a union/intersection result. Any accepted
      graph shape is normalised to the canonical form.

    `message` is the natural-language reply for the chat panel (required, never
    empty). `layout` is "force" (default) or "radial".

    Returns a small summary (`status`, `node_count`, `edge_count`, and an
    `artifact` pointer to a unique ``output_<call_id>.json`` file). The full,
    schema-validated `AgentResponse` is saved in that artifact; the server
    delivers the latest one from the turn as the authoritative final answer.
    After calling this tool, finish via `set_model_response` with the same
    `message` and an EMPTY `graph`.
    """
    if relations is not None:
        graph_dict = _build_output_graph(relations)
    elif graph is not None:
        graph_dict = Graph.from_dict(graph).to_dict()
    else:
        return {
            "status": "error",
            "error": (
                "output: provide a graph source — `relations` / "
                "`relations_artifact` (classify_relations output) OR `graph` / "
                "`graph_artifact` (a built graph) — so the before_tool_callback "
                "can inflate the artifact form."
            ),
        }

    response = AgentResponse.model_validate(
        {"message": message, "layout": layout, "graph": graph_dict}
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
            call_id = tool_context.function_call_id or "unknown"
            artifact_name = f"output_{call_id}.json"
            version = await tool_context.save_artifact(artifact_name, part)
            summary["artifact"] = {
                "name": artifact_name,
                "version": version,
            }
        except ValueError as exc:
            return {
                "status": "error",
                "error": f"output: failed to save response artifact: {exc}",
            }

    return summary


INSTRUCTION = """\
You are ARExplorer — a helpful assistant for questions about attitudes and
relations in documents, and for running the extraction pipeline when the user
asks for analysis.

When a request matches one of your available skills, FIRST load that skill with
`load_skill` and follow its instructions exactly before acting or replying. Use
`list_skills` if you are unsure which skill applies. Prefer the tools and
procedures a skill documents over reasoning the result out yourself.

For general questions that do not require the pipeline (e.g. explaining what you
can do, clarifying a prior result, or answering about the domain), reply
helpfully without loading a skill or calling tools unless the user clearly
needs analysis.

Be transparent about tool errors and ask for missing inputs rather than
guessing.

Always deliver your final answer through the `set_model_response` tool in the
structured response format defined by your output schema; never reply with plain
free-form text. The loaded skill explains how to populate that response when
you used the pipeline; otherwise use an empty graph and a clear `message`.
"""

_SKILLS_DIR = pathlib.Path(__file__).parent / "skills"

# Two complementary skills, kept separate so each concern can evolve on its own:
#   - arexplorer-workflow: the tool pipeline that produces the data / graph.
#   - arexplorer-response: the client-side contract for delivering the final reply.
workflow_skill = load_skill_from_dir(_SKILLS_DIR / "arexplorer-workflow")
response_skill = load_skill_from_dir(_SKILLS_DIR / "arexplorer-response")

skills_toolset = skill_toolset.SkillToolset(
    skills=[workflow_skill, response_skill]
)

root_agent = Agent(
    name="arexplorer_agent",
    model=Gemini(
        model=os.environ.get("AGENT_MODEL", "gemini-2.5-flash"),
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
        skills_toolset,
    ],
    before_tool_callback=inflate_artifact_inputs,
    after_tool_callback=offload_tool_output,
    after_model_callback=ensure_nonempty_response,
)
