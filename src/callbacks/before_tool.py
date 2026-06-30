
import json
import logging
from typing import Optional

from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types

logger = logging.getLogger(__name__)


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
    "output": [
        ("relations_artifact", "relations", "relation object", list, "relations"),
        ("graph_artifact", "graph", "graph", dict, "graph"),
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


def _parse_artifact_ref(artifact_name: str) -> tuple[str, int | None]:
    """Split ``filename@version`` into (filename, version) for pinned loads."""
    if "@" not in artifact_name:
        return artifact_name, None
    base, ver = artifact_name.rsplit("@", 1)
    try:
        return base, int(ver)
    except ValueError:
        return artifact_name, None


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

        filename, version = _parse_artifact_ref(artifact_name)

        try:
            part = await tool_context.load_artifact(filename, version=version)
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
                "error": (
                    f"Artifact {artifact_name!r} not found in session"
                    + (f" (version {version})" if version is not None else "")
                    + "."
                ),
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
