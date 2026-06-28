"""Agent-level callbacks for the ARExplorer agent.

Two complementary callbacks keep tool outputs and inputs out of the LLM
context window:

  - `offload_tool_output` (after_tool_callback): when one of the heavy tools
    (`extract_named_entities`, `form_entity_pairs`, `classify_relations`,
    `graph_operation`) succeeds, the full payload is persisted as a
    session-scoped JSON artifact and the LLM only sees status + summary
    counts + an `artifact` pointer.

  - `inflate_artifact_inputs` (before_tool_callback): the inverse on the way
    in — the LLM may hand `extract_named_entities`, `form_entity_pairs`, or
    `classify_relations` an artifact filename (`texts_artifact` /
    `documents_artifact` / `pairs_artifact`) instead of a long inline list.
    This callback loads that artifact, JSON-decodes it, and rewrites `args`
    so the underlying tool sees the actual list.

Both follow the LangChain-style "args_schema swap" pattern from the ADK
guidance: the LLM declares the lightweight artifact reference, the callback
silently swaps it for the heavy content before the function runs.

See the `adk-structured-output` / `google-agents-cli-adk-code` skills and the
ADK docs on Callbacks, ToolContext, and Artifacts.
"""

import json
import logging
from typing import Optional

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# before_tool_callback: artifact reference -> inflated list
# ---------------------------------------------------------------------------

# Maps tool name -> (artifact-ref arg, target arg, expected element kind).
# `kind` is informational; we only validate that the inflated value is a list.
_INFLATE_RULES: dict[str, tuple[str, str, str]] = {
    "extract_named_entities": ("texts_artifact", "texts", "string"),
    "form_entity_pairs": ("documents_artifact", "documents", "NER document"),
    "classify_relations": ("pairs_artifact", "pairs", "pair object"),
}


def _decode_artifact(part: genai_types.Part) -> object:
    """Read a JSON artifact Part and return the decoded Python value."""
    blob = part.inline_data
    if blob is None or blob.data is None:
        raise ValueError("artifact has no inline data")
    data = blob.data
    if isinstance(data, str):
        text = data
    else:
        text = data.decode("utf-8")
    return json.loads(text)


async def inflate_artifact_inputs(
    *,
    tool: BaseTool,
    args: dict,
    tool_context: ToolContext,
) -> Optional[dict]:
    """Swap an artifact reference in `args` for the JSON list it contains.

    For `extract_named_entities` and `classify_relations`, the LLM may pass a
    `*_artifact` filename instead of the inline `texts` / `pairs` list. When
    that key is present, this callback:

      1. Loads the artifact via the session artifact service.
      2. JSON-decodes the inline data; accepts either
            - a plain list (used directly), or
            - a dict containing a `<target_arg>` list (e.g. `{"texts": [...]}`).
      3. Rewrites `args` so the artifact key is dropped and the target arg
         (`texts` / `pairs`) holds the inflated list.

    The actual tool function never sees `texts_artifact` / `pairs_artifact`
    (ADK filters out args that aren't in the function signature anyway).

    Returns ``None`` to fall through to the real tool, or a synthetic error
    dict that short-circuits the call when the artifact is missing / not a
    valid JSON list.
    """
    rule = _INFLATE_RULES.get(tool.name)
    if rule is None:
        return None

    artifact_arg, target_arg, kind = rule
    if artifact_arg not in args:
        return None

    artifact_name = args.get(artifact_arg)
    args.pop(artifact_arg, None)
    if not artifact_name:
        return None

    try:
        part = await tool_context.load_artifact(artifact_name)
    except ValueError as exc:
        return {
            "status": "error",
            "error": (
                f"Cannot load artifact {artifact_name!r} for {tool.name}: {exc}"
            ),
        }

    if part is None:
        return {
            "status": "error",
            "error": f"Artifact {artifact_name!r} not found in session.",
        }

    try:
        decoded = _decode_artifact(part)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return {
            "status": "error",
            "error": (
                f"Artifact {artifact_name!r} is not valid UTF-8/JSON: {exc}"
            ),
        }

    if isinstance(decoded, list):
        value = decoded
    elif isinstance(decoded, dict) and isinstance(decoded.get(target_arg), list):
        value = decoded[target_arg]
    else:
        return {
            "status": "error",
            "error": (
                f"Artifact {artifact_name!r} must JSON-decode to a list of "
                f"{kind} values, or to an object with a {target_arg!r} list."
            ),
        }

    args[target_arg] = value
    return None


# ---------------------------------------------------------------------------
# after_tool_callback: full output -> artifact + summary
# ---------------------------------------------------------------------------


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
