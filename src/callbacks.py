"""Agent-level callbacks for the ARExplorer agent.

Two complementary callbacks keep tool outputs and inputs out of the LLM
context window:

  - `offload_tool_output` (after_tool_callback): when one of the heavy tools
    (`extract_named_entities`, `form_entity_pairs`, `classify_relations`,
    `graph_operation`) succeeds, the full payload is persisted as a
    session-scoped JSON artifact and the LLM only sees status + summary
    counts + an `artifact` pointer.

  - `inflate_artifact_inputs` (before_tool_callback): the inverse on the way
    in — the LLM may hand any of the heavy tools an artifact filename
    instead of the inline payload:
        * `extract_named_entities` ← `texts_artifact`
        * `form_entity_pairs`      ← `documents_artifact`
        * `classify_relations`     ← `pairs_artifact`
        * `graph_operation`        ← `graph_a_artifact` and/or `graph_b_artifact`
    This callback loads that artifact, JSON-decodes it, and rewrites `args`
    so the underlying tool sees the actual list / graph dict.

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

# Maps tool name -> list of inflation rules.
#
# Each rule is (artifact-ref arg, target arg, expected element kind,
# expected top-level type, wrapper key). A tool may declare multiple rules
# when it accepts more than one artifact input (e.g. `graph_operation`
# consumes two graphs).
#
# `kind` is informational and used only in error messages. Validation
# happens against `expected_type` (`list` for list-shaped inputs, `dict` for
# graph-shaped inputs). `wrapper_key` is the key under which the natural
# wrapper artifact (the one written by `offload_tool_output`) stores the
# payload; the callback first looks for that key inside the decoded dict
# and only falls back to using the decoded value as-is when the key is
# absent. For most tools `wrapper_key == target_arg`, but graphs differ
# because `graph_operation` writes its output under `"graph"`, not under
# `"graph_a"` / `"graph_b"`.
_INFLATE_RULES: dict[str, list[tuple[str, str, str, type, str]]] = {
    "extract_named_entities": [
        ("texts_artifact", "texts", "string", list, "texts"),
    ],
    "form_entity_pairs": [
        ("documents_artifact", "documents", "NER document", list, "documents"),
    ],
    "classify_relations": [
        ("pairs_artifact", "pairs", "pair object", list, "pairs"),
    ],
    "graph_operation": [
        ("graph_a_artifact", "graph_a", "graph", dict, "graph"),
        ("graph_b_artifact", "graph_b", "graph", dict, "graph"),
    ],
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
    """Swap any artifact references in `args` for their JSON payloads.

    For each tool listed in `_INFLATE_RULES`, the LLM may pass one or more
    `*_artifact` filenames in place of the corresponding inline argument.
    For every such rule present in `args` this callback:

      1. Loads the artifact via the session artifact service.
      2. JSON-decodes the inline data; accepts either
            - a value matching the rule's `expected_type` directly (a list
              for `texts`/`documents`/`pairs`, a dict for `graph_a`/`graph_b`),
              or
            - a wrapper dict containing the target key, e.g.
              `{"texts": [...]}` or `{"status": "success", "graph": {...}}`
              (the natural shape produced by the offload callback's
              artifacts).
      3. Rewrites `args` so the artifact key is dropped and the target arg
         (`texts` / `documents` / `pairs` / `graph_a` / `graph_b`) holds the
         inflated value.

    The actual tool function never sees the `*_artifact` keys (ADK filters
    out args that aren't in the function signature anyway).

    Returns ``None`` to fall through to the real tool, or a synthetic error
    dict that short-circuits the call when an artifact is missing / not a
    valid JSON value of the expected shape. Inflation stops at the first
    failing artifact for a given call.
    """
    rules = _INFLATE_RULES.get(tool.name)
    if not rules:
        return None

    for artifact_arg, target_arg, kind, expected_type, wrapper_key in rules:
        if artifact_arg not in args:
            continue

        artifact_name = args.pop(artifact_arg, None)
        if not artifact_name:
            continue

        try:
            part = await tool_context.load_artifact(artifact_name)
        except ValueError as exc:
            return {
                "status": "error",
                "error": (
                    f"Cannot load artifact {artifact_name!r} for {tool.name}: "
                    f"{exc}"
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

        # Wrapper dicts (the typical shape produced by `offload_tool_output`,
        # e.g. {"status": "success", "graph": {...}}) get unwrapped first;
        # a bare value of `expected_type` is used as-is so users can feed in
        # artifacts they uploaded themselves.
        if (
            isinstance(decoded, dict)
            and isinstance(decoded.get(wrapper_key), expected_type)
        ):
            value = decoded[wrapper_key]
        elif isinstance(decoded, expected_type):
            value = decoded
        else:
            if expected_type is list:
                error_msg = (
                    f"Artifact {artifact_name!r} must JSON-decode to a list "
                    f"of {kind} values (or to an object with a "
                    f"{wrapper_key!r} list)."
                )
            else:
                error_msg = (
                    f"Artifact {artifact_name!r} must JSON-decode to a {kind} "
                    f"dict (or to an object with a {wrapper_key!r} key "
                    "holding one)."
                )
            return {"status": "error", "error": error_msg}

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
