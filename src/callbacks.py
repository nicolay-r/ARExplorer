"""Agent-level callbacks for the ARExplorer agent.

The NER, relation classification, and graph operation tools can each produce
verbose results — long bulk-ner annotations, per-pair chain-of-thought
reasoning, big graphs. Letting those payloads flow back into the LLM context
verbatim wastes tokens and risks blowing the context window.

`offload_tool_output` is an `after_tool_callback` that intercepts
every successful call to one of these tools, persists the full payload as a
session-scoped JSON artifact, and returns a trimmed summary (with the artifact
filename) to the agent. Subsequent tools/UI can rehydrate the full payload via
`tool_context.load_artifact(...)` when they actually need it.

See the `adk-structured-output` / `google-agents-cli-adk-code` skills and the
ADK docs on Callbacks, ToolContext, and Artifacts.
"""

import json
import logging
from typing import Optional

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


# Tools whose outputs should be offloaded. Kept as a set so the callback is a
# cheap no-op for any other tool the agent might gain in the future.
_OFFLOAD_TOOLS = {
    "extract_named_entities",
    "classify_relations",
    "graph_operation",
}


def _trim_response(tool_name: str, response: dict) -> dict:
    """Build the trimmed payload returned to the LLM for a given tool.

    Drops verbose fields the model does not need to reason further:
      - NER: raw bulk-ner `annotation` spans (the flat `entities` list is kept).
      - Relations: per-pair `reasoning` text (label is what drives the graph).
      - Graph: nothing — the graph payload is the structured result the agent
        forwards to the UI, so it is preserved verbatim.
    """
    if tool_name == "extract_named_entities":
        documents = response.get("documents", []) or []
        return {
            "status": response.get("status", "success"),
            "documents": [
                {
                    "text": doc.get("text"),
                    "entities": doc.get("entities", []),
                }
                for doc in documents
            ],
        }

    if tool_name == "classify_relations":
        relations = response.get("relations", []) or []
        return {
            "status": response.get("status", "success"),
            "relations": [
                {
                    "source": rel.get("source"),
                    "target": rel.get("target"),
                    "label": rel.get("label"),
                }
                for rel in relations
            ],
        }

    if tool_name == "graph_operation":
        return {
            "status": response.get("status", "success"),
            "graph": response.get("graph", {"nodes": [], "edges": []}),
        }

    return dict(response)


async def offload_tool_output(
    *,
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict,
) -> Optional[dict]:
    """Persist the full tool output as an artifact; return a trimmed summary.

    Runs after every tool call. For tools listed in `_OFFLOAD_TOOLS` whose
    result is a successful dict, it:

      1. Serializes the full `tool_response` to JSON and saves it under
         ``<tool>_<function_call_id>.json`` via the artifact service.
      2. Returns a trimmed dict (see `_trim_response`) with an `artifact` field
         pointing at the saved filename + version, so a downstream consumer
         can `load_artifact` the full payload if needed.

    Returns ``None`` (i.e. "no override") for any tool that is not in the
    offload set, for error responses, or if the artifact service is not
    configured. This keeps the callback safe to wire up even in environments
    (like tests) without an artifact service.

    Note: we deliberately do NOT set ``tool_context.actions.skip_summarization``
    here. That flag would mark the tool response as the agent's final
    response, but the agent must keep reasoning over the trimmed result (e.g.
    feed extracted entities into `classify_relations`).
    """
    if tool.name not in _OFFLOAD_TOOLS:
        return None
    if not isinstance(tool_response, dict):
        return None
    if tool_response.get("status") != "success":
        return None

    try:
        payload = json.dumps(tool_response, ensure_ascii=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        logger.warning(
            "offload_tool_output: %s output is not JSON-serialisable (%s); "
            "passing through unchanged.",
            tool.name,
            exc,
        )
        return None

    call_id = tool_context.function_call_id or "unknown"
    artifact_name = f"{tool.name}_{call_id}.json"
    part = genai_types.Part(
        inline_data=genai_types.Blob(
            mime_type="application/json",
            data=payload,
        )
    )

    try:
        version = await tool_context.save_artifact(artifact_name, part)
    except ValueError as exc:
        logger.warning(
            "offload_tool_output: cannot save %s artifact (%s); "
            "passing through unchanged.",
            tool.name,
            exc,
        )
        return None

    trimmed = _trim_response(tool.name, tool_response)
    trimmed["artifact"] = {"name": artifact_name, "version": version}
    return trimmed
