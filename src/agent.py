"""ARExplorer agent — extracts Attitudes and Relations from documents.

ADK 2.0 root agent exposing three tools (wrapped over their Python APIs):
  #1 extract_named_entities  — bulk-ner annotation over massive text collections
  #2 classify_relations      — bulk-chain attitude/relation classification
  #3 graph_operation         — union / intersection over attitude graphs
"""

import os

from google.adk.agents import Agent
from google.adk.models import Gemini
from google.adk.tools.load_artifacts_tool import load_artifacts_tool
from google.genai import types

from src.callbacks import inflate_artifact_inputs, offload_tool_output
from src.schema import AgentResponse
from src.tools import (
    extract_named_entities as _extract_named_entities,
    classify_relations as _classify_relations,
    graph_operation,
)


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


INSTRUCTION = """\
You are ARExplorer, an assistant that extracts and explores Attitudes and
Relations between named entities found in documents.

Your typical workflow:
1. Use `extract_named_entities` to annotate named entities across the provided
   texts. This handles large volumes by chunking, so prefer it over reasoning
   about entities yourself.
2. Form candidate entity pairs from the extracted entities and use
   `classify_relations` to determine the attitude (positive, negative, or
   neutral) the source entity expresses towards the target entity.
3. When the user wants to combine or compare result sets, build graphs of the
   form {"nodes": [...], "edges": [{"source", "target", "label"}]} and use
   `graph_operation` with "union" or "intersection".

Be transparent about tool errors and ask for missing inputs (e.g. text context
for a relation) rather than guessing.

TOOL OUTPUTS ARE OFFLOADED TO ARTIFACTS:
- `extract_named_entities`, `classify_relations`, and `graph_operation` do NOT
  return their raw data to you. Each successful call returns only:
    * `status`, lightweight counts (e.g. `document_count` / `entity_count`,
      `relation_count` / `label_counts`, `node_count` / `edge_count`), and
    * an `artifact` field of shape {"name": "<file>.json", "version": <int>}.
  The full `documents` / `relations` / `graph` payload is saved as a JSON
  artifact in the session artifact store.
- To read the actual data (form entity pairs, look up labels, populate the
  final graph), call the `load_artifacts` tool with the artifact name, e.g.
  `load_artifacts(artifact_names=["extract_named_entities_<id>.json"])`. The
  JSON content of each requested artifact is then injected into your next
  turn so you can reason over it.
- Only call `load_artifacts` when you genuinely need the content of an
  artifact; the counts alone are usually enough to decide what to do next.

TOOL INPUTS CAN ALSO COME FROM ARTIFACTS:
- `extract_named_entities` accepts `texts_artifact` instead of `texts`.
- `classify_relations` accepts `pairs_artifact` instead of `pairs`.
- The artifact must JSON-decode to a list of the expected shape (strings for
  texts, {text, source, target} dicts for pairs), or to an object with a
  matching key (`{"texts": [...]}` / `{"pairs": [...]}`). The framework loads
  the artifact and substitutes its content for the inline list before the
  tool runs.
- Prefer the artifact form whenever the input list is large or already lives
  in the session (e.g. uploaded by the user, or written by an earlier turn).
  Provide exactly one of the inline or artifact form per call.

OUTPUT FORMAT (important):
- Always deliver your final answer through the `set_model_response` tool, in the
  structured `AgentResponse` format. Never reply with plain free-form text.
- `message`: a clear natural-language reply for the chat panel.
- `graph`: when you have extracted entities and their attitudes, populate
  `nodes` (each entity, with a `weight` reflecting how often it appears) and
  `edges` (each attitude as source -> target with `relation` set to
  "positive" / "negative" / "neutral" and a `weight` for its strength). Load
  the relevant artifact(s) first so you have the underlying data. Leave
  `graph` empty only when there is genuinely nothing to plot.
- `layout`: "radial" for many entities with a clear hierarchy, otherwise "force".
"""

root_agent = Agent(
    name="arexplorer_agent",
    model=Gemini(
        model="gemini-2.5-flash-lite",
        retry_options=types.HttpRetryOptions(attempts=3, initial_delay=1.0),
    ),
    instruction=INSTRUCTION,
    output_schema=AgentResponse,
    tools=[
        extract_named_entities,
        classify_relations,
        graph_operation,
        load_artifacts_tool,
    ],
    before_tool_callback=inflate_artifact_inputs,
    after_tool_callback=offload_tool_output,
)
