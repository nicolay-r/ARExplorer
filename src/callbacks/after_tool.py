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
    "form_entity_pairs",
    "classify_relations",
    "graph_operation",
}


def _summarize(tool_name: str, response: dict) -> dict:
    """Build the data-free summary the LLM sees in place of the tool output.

    Only top-level status and lightweight counts: no `documents`, `pairs`,
    `relations`, or `graph` payloads. The agent must call `load_artifacts`
    to read the actual content.
    """
    status = response.get("status", "success")

    if tool_name == "extract_named_entities":
        documents = response.get("documents") or []
        return {
            "status": status,
            "document_count": len(documents),
            "entity_count": sum(
                len(doc.get("entities") or []) for doc in documents
            ),
        }

    if tool_name == "form_entity_pairs":
        pairs = response.get("pairs") or []
        return {
            "status": status,
            "pair_count": len(pairs),
        }

    if tool_name == "classify_relations":
        relations = response.get("relations") or []
        label_counts: dict[str, int] = {}
        for rel in relations:
            label = rel.get("label")
            if label is None:
                continue
            label_counts[label] = label_counts.get(label, 0) + 1
        return {
            "status": status,
            "relation_count": len(relations),
            "label_counts": label_counts,
        }

    if tool_name == "graph_operation":
        graph = response.get("graph") or {}
        return {
            "status": status,
            "node_count": len(graph.get("nodes") or []),
            "edge_count": len(graph.get("edges") or []),
        }

    return {"status": status}


async def offload_tool_output(
    *,
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
    tool_response: dict,
) -> Optional[dict]:
    """Persist the full tool output as an artifact; return a data-free summary.

    Runs after every tool call. For tools listed in `_OFFLOAD_TOOLS` whose
    result is a successful dict, it:

      1. Serializes the full `tool_response` to JSON and saves it under
         ``<tool>_<function_call_id>.json`` via the artifact service.
      2. Returns a dict containing only `status`, lightweight summary counts
         (see `_summarize`), and an `artifact` pointer (filename + version).
         The actual `documents` / `relations` / `graph` payloads are NOT
         included — the agent must call the `load_artifacts` tool to access
         them.

    Returns ``None`` (i.e. "no override") for any tool that is not in the
    offload set, for error responses, for non-JSON-serializable payloads, and
    when the artifact service is not configured. This keeps the callback safe
    to wire up even in environments (like tests) without an artifact service.

    Note: we deliberately do NOT set ``tool_context.actions.skip_summarization``
    here. That flag marks the tool-response event as the agent's final
    response (see `google.adk.events.event.Event.is_final_response`), which
    would short-circuit the chain — but the agent still needs to call
    `load_artifacts` and `set_model_response` after this callback runs.
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

    summary = _summarize(tool.name, tool_response)
    summary["artifact"] = {"name": artifact_name, "version": version}
    return summary